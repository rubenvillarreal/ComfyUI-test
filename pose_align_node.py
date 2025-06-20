"""
pose_align_node_optimized.py – ComfyUI custom nodes with data export for canvas sync
"""

from __future__ import annotations
import cv2, numpy as np, torch, os, json, folder_paths, math, time
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
from nodes import PreviewImage
from aiohttp import web

# ─────────────────────────── Constants and mappings ────────────────────────────
NOSE, NECK, R_SH, L_SH, MIDHIP = 0, 1, 2, 5, 8
NUM_BODY_JOINTS = 18
TORSO = np.asarray([NOSE, NECK, R_SH, L_SH, MIDHIP])
HEAD_SEG = [(15, 16), (0, 15), (0, 16)]
TORSO_SEG = [(R_SH, L_SH), (NECK, MIDHIP)]
STABLE_SEGMENTS = HEAD_SEG + TORSO_SEG
ROBUST_JOINTS = np.asarray([NOSE, NECK, R_SH, L_SH, MIDHIP, 15, 16])

OPENPOSE_COLOUR_MAP = {i:(255-i*12,85+i*8,i*15) for i in range(18)}  # Simplified color map
BODY_25_PAIRS=[(1,8),(1,2),(1,5),(2,3),(3,4),(5,6),(6,7),(8,9),(9,10),(10,11),
               (8,12),(12,13),(13,14),(1,0),(0,15),(15,17),(0,16),(16,18),
               (14,19),(19,20),(14,21),(11,22),(22,23),(11,24)]

# ───────────────── Utility functions ─────────────────────────────
def _to_chw(img: torch.Tensor) -> torch.Tensor:
    if img.dim() == 4:  # (B,H,W,C) from ComfyUI
        img = img[0]
    if img.dim() == 3 and img.shape[0] != 3:  # (H,W,C) -> (C,H,W)
        img = img.permute(2,0,1)
    return img

def torch_to_u8(t: torch.Tensor) -> np.ndarray:
    # Handle ComfyUI tensor format: (B,H,W,C) -> BGR uint8
    if t.dim() == 4:
        t = t[0]  # Take first batch item: (H,W,C)
    if t.dim() == 3:
        if t.shape[0] == 3:  # (C,H,W) -> (H,W,C)
            t = t.permute(1,2,0)
        # Now t should be (H,W,C)
        rgb = t.cpu().numpy()
        rgb_uint8 = (rgb * 255).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    else:
        raise ValueError(f"Unexpected tensor shape: {t.shape}")

def u8_to_torch(bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    return torch.from_numpy(rgb[None])  # Add batch dimension: (1,H,W,C)

Keypoints = np.ndarray

def _coords_to_xy(arr: np.ndarray, thr: float = 0.15) -> Keypoints:
    kps = np.full((NUM_BODY_JOINTS,2), np.nan, np.float32)
    for j in range(min(NUM_BODY_JOINTS, arr.shape[0])):
        x, y, c = arr[j]
        if c >= thr: kps[j] = (x, y)
    return kps

def dict_to_kps_single(p: Dict[str,Any], w: int, h: int) -> Keypoints:
    raw = np.asarray(p.get("pose_keypoints_2d", p.get("keypoints", [])), np.float32).reshape(-1, 3)
    if raw[:, :2].max() <= 1.01:
        raw[:, 0] *= w; raw[:, 1] *= h
    return _coords_to_xy(raw)

def kps_from_pose_json(js: List[Dict[str,Any]]) -> List[Keypoints]:
    if not js: return []
    frame = js[0]
    w, h = frame.get("canvas_width",1), frame.get("canvas_height",1)
    if "people" in frame:
        return [dict_to_kps_single(p,w,h) for p in frame["people"]]
    if "animals" in frame:
        return [dict_to_kps_single({"pose_keypoints_2d":a[:NUM_BODY_JOINTS*3]}, w, h) for a in frame["animals"]]
    return []

def estimate_translation(kps_json: Keypoints, kps_img: Keypoints, focus: np.ndarray = TORSO) -> np.ndarray:
    vis = (~np.isnan(kps_json[:,0]) & ~np.isnan(kps_img[:,0]) & np.isin(np.arange(NUM_BODY_JOINTS), focus))
    return np.nanmedian(kps_img[vis] - kps_json[vis], 0).astype(np.float32) if vis.sum() >= 2 else np.zeros(2, np.float32)

def correct_json_offset(kps_json: Keypoints, kps_img: Keypoints) -> Keypoints:
    return kps_json + estimate_translation(kps_json, kps_img)

def robust_scale(src: Keypoints, dst: Keypoints) -> Optional[float]:
    ratios = []
    for i,j in STABLE_SEGMENTS:
        if not any(np.isnan([src[i], src[j], dst[i], dst[j]]).flat):
            d_src, d_dst = np.linalg.norm(src[i]-src[j]), np.linalg.norm(dst[i]-dst[j])
            if d_src > 1e-3: ratios.append(d_dst / d_src)
    return np.median(ratios) if ratios else None

def robust_translation(src: Keypoints, dst: Keypoints, s: float, R: np.ndarray) -> np.ndarray:
    deltas = [dst[j] - s*(R @ src[j]) for j in ROBUST_JOINTS 
              if j < NUM_BODY_JOINTS and not any(np.isnan([src[j], dst[j]]).flat)]
    return np.median(deltas,0).astype(np.float32) if deltas else np.zeros(2,np.float32)

def refine_translation(src: Keypoints, dst: Keypoints, s: float, R: np.ndarray, t0: np.ndarray,
                       max_iter: int = 3, tol: float = 0.05) -> np.ndarray:
    """Iteratively refine translation to correct for offset mismatches"""
    vis = (~np.isnan(src[:,0]) & ~np.isnan(dst[:,0]) & np.isin(np.arange(NUM_BODY_JOINTS), TORSO))
    if vis.sum() < 2:
        return t0
    t = t0.copy()
    for _ in range(max_iter):
        warped = (s*(R @ src[vis].T)).T + t
        delta = np.nanmedian(dst[vis]-warped,0).astype(np.float32)
        if np.linalg.norm(delta) < tol:
            break
        t += delta
    return t

def extract_kps_from_mask(img: np.ndarray, mask: Optional[np.ndarray]=None, tol: int = 10) -> Keypoints:
    if mask is None: mask = np.ones(img.shape[:2], bool)
    kps = np.full((NUM_BODY_JOINTS,2), np.nan, np.float32)
    for j,(r,g,b) in OPENPOSE_COLOUR_MAP.items():
        m = ((abs(img[...,2]-r)<tol)&(abs(img[...,1]-g)<tol)&(abs(img[...,0]-b)<tol)&mask)
        if m.any():
            ys,xs = np.nonzero(m)
            kps[j]=(xs.mean(),ys.mean())
    return kps

def fit_pair(src: Keypoints, dst: Keypoints) -> Tuple[float,np.ndarray,np.ndarray,float]:
    vis = ~np.isnan(src[:,0]) & ~np.isnan(dst[:,0])
    if vis.sum() < 2: return 1., np.eye(2,dtype=np.float32), np.zeros(2,np.float32), float("inf")
    
    # 1) Scale estimation
    s = robust_scale(src, dst) or 1.0
    
    # 2) Rotation from Procrustes analysis at that scale
    muX, muY = src[vis].mean(0), dst[vis].mean(0)
    U,_,Vt = np.linalg.svd((src[vis]-muX).T @ (dst[vis]-muY))
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0: Vt[1] *= -1; R = Vt.T @ U.T
    
    # 3) Translation (robust, then iteratively refined for offset correction)
    t = robust_translation(src, dst, s, R)
    t = refine_translation(src, dst, s, R, t)  # Critical: refine for offset correction
    
    # 4) Calculate residual error
    recon = (s*(R @ src[vis].T)).T + t
    mse = float(np.mean(np.linalg.norm(recon-dst[vis],axis=1)**2))
    return s, R.astype(np.float32), t.astype(np.float32), mse

def two_largest(mask: np.ndarray)->Tuple[np.ndarray,np.ndarray]:
    n,lab = cv2.connectedComponents(mask.astype(np.uint8), 8)
    if n<=2: return [mask], [np.zeros_like(mask,bool)]
    areas=[(lab==i).sum() for i in range(1,n)]
    a,b = np.argsort(areas)[-2:]
    return [(lab==a+1),(lab==b+1)]

def normalize_angle(angle_deg: float) -> float:
    return ((angle_deg % 360) + 360) % 360

def _build_affine(scale: float, angle_deg: float, tx: float, ty: float, cx: float, cy: float) -> np.ndarray:
    th = math.radians(normalize_angle(angle_deg))
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]], np.float32) * scale
    t = np.array([tx, ty], np.float32) + np.array([cx, cy]) - R @ np.array([cx, cy])
    return np.hstack([R, t[:, None]])

def decompose_affine_matrix(matrix: np.ndarray, cx: float, cy: float) -> Tuple[float, float, float, float]:
    R, t = matrix[:2, :2], matrix[:2, 2]
    scale = np.sqrt(np.linalg.det(R))
    if np.linalg.det(R) < 0: scale = -scale
    R_normalized = R / scale
    angle_deg = normalize_angle(math.degrees(math.atan2(R_normalized[1, 0], R_normalized[0, 0])))
    center = np.array([cx, cy])
    tx, ty = t - center + R @ center
    return float(scale), float(angle_deg), float(tx), float(ty)

# Global storage for transformation data
_transform_data_cache = {}

def store_transform_data(node_id: str, matrices: Dict[str, np.ndarray], offset_corrections: Dict[str, Dict[str, float]]):
    global _transform_data_cache
    _transform_data_cache[node_id] = {
        'timestamp': time.time(),
        'matrices': {k: v.tolist() if v is not None else None for k, v in matrices.items()},
        'offsetCorrections': offset_corrections
    }

def get_transform_data(node_id: str) -> Optional[Dict]:
    return _transform_data_cache.get(node_id)

# ──────────────────────────── Main alignment node ─────────────────────────
class PoseAlignTwoToOne(PreviewImage):
    CATEGORY = "AInseven"

    def __init__(self):
        super().__init__()
        self.prefix_append, self.compress_level = "", 4
        self._MA = self._MB = self._out_size = None
        self._offset_corrections = {'A': {'x': 0, 'y': 0}, 'B': {'x': 0, 'y': 0}}

    @classmethod
    def INPUT_TYPES(cls):
        params = {}
        for pose in ["A", "B"]:
            params.update({
                f"angle_deg_{pose}": ("FLOAT", {"default": 0, "min": -720, "max": 720, "step": 0.1}),
                f"scale_{pose}": ("FLOAT", {"default": 1.0, "min": 0.20, "max": 3.0, "step": 0.01}),
                f"tx_{pose}": ("INT", {"default": 0, "min": -2048, "max": 2048}),
                f"ty_{pose}": ("INT", {"default": 0, "min": -2048, "max": 2048})
            })
        
        return {
            "required": {
                "ref_pose_img": ("IMAGE",), 
                "ref_pose_json": ("POSE_KEYPOINT",),
                "poseA_img": ("IMAGE",), 
                "poseA_json": ("POSE_KEYPOINT",),
                "poseB_img": ("IMAGE",), 
                "poseB_json": ("POSE_KEYPOINT",),
                "assignment": (["auto", "A_to_first", "A_to_second"],),
                "manual": ("BOOLEAN", {"default": False}), 
                "reset": ("BOOLEAN", {"default": False}),
                **params
            },
            "optional": {"debug": ("BOOLEAN", {"default": False})},
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("aligned_poseA", "aligned_poseB", "combined_AB", "combine_all")
    FUNCTION = "align"

    def _get_kps(self, img: np.ndarray, js: List[Dict[str, Any]], idx: int) -> Keypoints:
        people = kps_from_pose_json(js)
        return people[idx] if people and idx < len(people) else extract_kps_from_mask(img)

    def _save_preview_images(self, ref_pose_img, poseA_img, poseB_img, prompt=None, extra_pnginfo=None):
        import torch.nn.functional as F
        tensors = [img[0:1] for img in [ref_pose_img, poseA_img, poseB_img]]
        max_h, max_w = max(t.shape[1] for t in tensors), max(t.shape[2] for t in tensors)
        
        def resize_tensor(tensor, target_h, target_w):
            tensor = tensor.permute(0, 3, 1, 2)
            return F.interpolate(tensor, size=(target_h, target_w), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        
        resized = [resize_tensor(t, max_h, max_w) for t in tensors]
        combined = torch.cat(resized, dim=2)
        return self.save_images(combined, filename_prefix="pose_align_preview", prompt=prompt, extra_pnginfo=extra_pnginfo)

    def _update_widget_values(self, cx: float, cy: float, debug: bool = False):
        for pose, matrix in [("A", self._MA), ("B", self._MB)]:
            if matrix is not None:
                scale, angle_deg, tx, ty = decompose_affine_matrix(matrix, cx, cy)
                self.properties = getattr(self, 'properties', {})
                self.properties.update({f'scale_{pose}': scale, f'angle_deg_{pose}': angle_deg, 
                                      f'tx_{pose}': int(round(tx)), f'ty_{pose}': int(round(ty))})
                if debug: print(f"[PoseAlign] Updated widget {pose}: scale={scale:.3f}, angle={angle_deg:.1f}°, tx={tx:.1f}, ty={ty:.1f}")

    def _store_transform_data_for_canvas(self, debug: bool = False):
        store_transform_data(str(id(self)), {'A': self._MA, 'B': self._MB}, self._offset_corrections)
        if debug: print(f"[PoseAlign] Stored transform data for node {id(self)}")

    def align(self, ref_pose_img: torch.Tensor, ref_pose_json: List[Dict[str, Any]],
              poseA_img: torch.Tensor, poseA_json: List[Dict[str, Any]],
              poseB_img: torch.Tensor, poseB_json: List[Dict[str, Any]],
              assignment="auto", manual=False, reset=False, debug=False,
              angle_deg_A=0.0, scale_A=1.0, tx_A=0, ty_A=0,
              angle_deg_B=0.0, scale_B=1.0, tx_B=0, ty_B=0,
              prompt=None, extra_pnginfo=None):

        ui_result = self._save_preview_images(ref_pose_img, poseA_img, poseB_img, prompt, extra_pnginfo)
        N = poseA_img.shape[0]
        ref_np = torch_to_u8(ref_pose_img[0:1])
        h, w = ref_np.shape[:2]
        cx, cy = w / 2.0, h / 2.0

        if manual:
            MA = _build_affine(scale_A, normalize_angle(angle_deg_A), tx_A, ty_A, cx, cy).astype(np.float32)
            MB = _build_affine(scale_B, normalize_angle(angle_deg_B), tx_B, ty_B, cx, cy).astype(np.float32)
            self.properties = getattr(self, 'properties', {})
            self.properties.update({
                'scale_A': scale_A, 'angle_deg_A': angle_deg_A, 'tx_A': tx_A, 'ty_A': ty_A,
                'scale_B': scale_B, 'angle_deg_B': angle_deg_B, 'tx_B': tx_B, 'ty_B': ty_B
            })
            self._MA, self._MB = MA, MB
        else:
            if reset or self._MA is None or self._MB is None:
                A_np, B_np = torch_to_u8(poseA_img[0:1]), torch_to_u8(poseB_img[0:1])
                
                # Get reference keypoints with offset correction
                ref_people_raw = kps_from_pose_json(ref_pose_json)
                if len(ref_people_raw) >= 2:
                    m1, m2 = two_largest(cv2.cvtColor(ref_np, cv2.COLOR_BGR2GRAY) > 2)
                    img_people = [extract_kps_from_mask(ref_np, m1), extract_kps_from_mask(ref_np, m2)]
                    kR1, kR2 = correct_json_offset(ref_people_raw[0], img_people[0]), correct_json_offset(ref_people_raw[1], img_people[1])
                else:
                    m1, m2 = two_largest(cv2.cvtColor(ref_np, cv2.COLOR_BGR2GRAY) > 2)
                    kR1, kR2 = extract_kps_from_mask(ref_np, m1), extract_kps_from_mask(ref_np, m2)

                # Process poses A and B
                poses_data = []
                for img_np, json_data in [(A_np, poseA_json), (B_np, poseB_json)]:
                    kps_json = self._get_kps(img_np, json_data, 0)
                    kps_img = extract_kps_from_mask(img_np)
                    offset = estimate_translation(kps_json, kps_img)
                    poses_data.append((correct_json_offset(kps_json, kps_img), offset))
                
                (kA, offset_A), (kB, offset_B) = poses_data
                self._offset_corrections = {'A': {'x': float(offset_A[0]), 'y': float(offset_A[1])},
                                          'B': {'x': float(offset_B[0]), 'y': float(offset_B[1])}}

                # Fit transformations
                fits = [(fit_pair(kA, kR1), fit_pair(kB, kR2)), (fit_pair(kA, kR2), fit_pair(kB, kR1))]
                pick = 0 if assignment == "A_to_first" else 1 if assignment == "A_to_second" else (0 if sum(fits[0][i][3] for i in [0,1]) <= sum(fits[1][i][3] for i in [0,1]) else 1)
                
                (sA, RA, tA, _), (sB, RB, tB, _) = fits[pick]
                MA = np.hstack([sA * RA, tA[:, None]]).astype(np.float32)
                MB = np.hstack([sB * RB, tB[:, None]]).astype(np.float32)
                self._MA, self._MB, self._out_size = MA, MB, (w, h)
                
                self._update_widget_values(cx, cy, debug)
            else:
                MA, MB = self._MA, self._MB
                self._update_widget_values(cx, cy, debug)

        self._store_transform_data_for_canvas(debug)

        # Apply transformations
        outA, outB, outC, outAll = [], [], [], []
        for i in range(N):
            A_np, B_np = torch_to_u8(poseA_img[i:i+1]), torch_to_u8(poseB_img[i:i+1])
            A_w = cv2.warpAffine(A_np, MA, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            B_w = cv2.warpAffine(B_np, MB, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            combined = np.where(A_w > 0, A_w, B_w)
            combo_all = ref_np.copy()
            for img in [A_w, B_w]:
                m = img > 0; combo_all[m] = img[m]

            outA.append(u8_to_torch(A_w)); outB.append(u8_to_torch(B_w))
            outC.append(u8_to_torch(combined)); outAll.append(u8_to_torch(combo_all))

        result = {"result": tuple(torch.cat(out, 0) for out in [outA, outB, outC, outAll])}
        if "ui" in ui_result: result["ui"] = ui_result["ui"]
        return result

# ───────────────────── Compact pose viewer ──────────────────
class PoseViewer:
    CATEGORY="AInseven/Debug"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"image":("IMAGE",), "pose_json":("POSE_KEYPOINT",),
                           "point_radius":("INT",{"default":8,"min":1,"max":50}),
                           "line_thickness":("INT",{"default":4,"min":1,"max":20}),
                           "draw_limbs":("BOOLEAN",{"default":True})}}
    RETURN_TYPES=("IMAGE",); FUNCTION="view"

    def view(self,image:torch.Tensor, pose_json:List[Dict[str,Any]], point_radius:int, line_thickness:int, draw_limbs:bool):
        img_np=torch_to_u8(image)
        for person_kps in kps_from_pose_json(pose_json):
            color=tuple(map(int,np.random.randint(100,256,3)))
            if draw_limbs:
                for p1,p2 in BODY_25_PAIRS:
                    if p1<len(person_kps) and p2<len(person_kps) and not any(np.isnan([person_kps[p1], person_kps[p2]]).flat):
                        cv2.line(img_np,tuple(np.int32(person_kps[p1])),tuple(np.int32(person_kps[p2])),color,line_thickness)
            for pt in person_kps:
                if not np.isnan(pt).any():
                    cv2.circle(img_np,tuple(np.int32(pt)),point_radius,color,-1)
                    cv2.circle(img_np,tuple(np.int32(pt)),point_radius,(0,0,0),2)
        return (u8_to_torch(img_np),)

# ──────────────────────────── API & Registration ─────────────────────────
async def get_pose_align_data(request):
    try:
        data = get_transform_data(request.match_info.get('node_id'))
        return web.json_response(data if data else {'error': 'No data found'}, status=200 if data else 404)
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)

# Node registration - CRITICAL: These must be defined for ComfyUI to find the nodes
NODE_CLASS_MAPPINGS = {
    "PoseAlignTwoToOne": PoseAlignTwoToOne,
    "PoseViewer": PoseViewer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PoseAlignTwoToOne": "Pose Align Two To One (Fixed)",
    "PoseViewer": "Pose Viewer (Debug)"
}

# API registration
try:
    from main import app
    app.router.add_get('/AInseven/pose_align_data/{node_id}', get_pose_align_data)
    print("[PoseAlign] API routes registered successfully")
except:
    try:
        import server
        if hasattr(server, 'app'):
            server.app.router.add_get('/AInseven/pose_align_data/{node_id}', get_pose_align_data)
            print("[PoseAlign] API routes registered via server module")
    except:
        print("[PoseAlign] API routes not registered - canvas sync may not work")

# Export for ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
