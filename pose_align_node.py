"""
pose_align_node_fixed.py – ComfyUI custom nodes with data export for canvas sync
"""

from __future__ import annotations
import cv2, numpy as np, torch, os, json, folder_paths
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
import uuid
from nodes import PreviewImage
import math
import time
from aiohttp import web
import aiohttp

# [Keep all the existing helper functions from your original code]
# ─────────────────────────── body-joint indices ────────────────────────────
NOSE, NECK, R_SH, L_SH, MIDHIP = 0, 1, 2, 5, 8
NUM_BODY_JOINTS   = 18
FULL_TORSO        = [NOSE, NECK, R_SH, L_SH, MIDHIP]
UPPER_TORSO       = [NOSE, NECK, R_SH, L_SH]
TORSO             = np.asarray(FULL_TORSO)

# extra segments and joints we trust for robust estimates
HEAD_SEG  = [(15, 16), (0, 15), (0, 16)]      # ear-to-ear, nose-ear
TORSO_SEG = [(R_SH, L_SH), (NECK, MIDHIP)]    # shoulder width, neck-hip
STABLE_SEGMENTS = HEAD_SEG + TORSO_SEG
ROBUST_JOINTS   = np.asarray([NOSE, NECK, R_SH, L_SH, MIDHIP, 15, 16])

# OpenPose colour map (RGB 0-255) used by the mask extractor
OPENPOSE_COLOUR_MAP = {
    0:(255,0,0),1:(255,85,0),2:(255,170,0),3:(255,255,0),4:(170,255,0),5:(85,255,0),6:(0,255,0),
    7:(0,255,85),8:(0,255,170),9:(0,255,255),10:(0,170,255),11:(0,85,255),12:(0,0,255),13:(85,0,255),
    14:(170,0,255),15:(255,0,255),16:(255,0,170),17:(255,0,85)}

# BODY-25 limb pairs for the viewer debug node
BODY_25_PAIRS=[(1,8),(1,2),(1,5),(2,3),(3,4),(5,6),(6,7),(8,9),(9,10),(10,11),
               (8,12),(12,13),(13,14),(1,0),(0,15),(15,17),(0,16),(16,18),
               (14,19),(19,20),(14,21),(11,22),(22,23),(11,24)]

# ───────────────── tensor ↔ OpenCV convenience ─────────────────────────────
def _to_chw(img: torch.Tensor) -> torch.Tensor:
    if img.dim() == 4:             # (1,3,H,W) from ComfyUI
        img = img[0]
    return img if img.shape[0] == 3 else img.permute(2,0,1)

def torch_to_u8(t: torch.Tensor) -> np.ndarray:
    chw = _to_chw(t)                              # (3,H,W)
    rgb = chw.permute(1,2,0).cpu().numpy()
    return cv2.cvtColor((rgb*255).clip(0,255).astype(np.uint8), cv2.COLOR_RGB2BGR)

def u8_to_torch(bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    return torch.from_numpy(rgb[None])            # (1,H,W,3) for ComfyUI

def torch_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert torch tensor to PIL Image for saving"""
    chw = _to_chw(tensor)
    rgb = chw.permute(1,2,0).cpu().numpy()
    rgb_uint8 = (rgb * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(rgb_uint8)

# [Include all the existing helper functions...]
Keypoints = np.ndarray

def _reshape(flat: List[float]) -> np.ndarray:
    return np.asarray(flat, np.float32).reshape(-1, 3)[:, :3]

def _coords_to_xy(arr: np.ndarray, thr: float = 0.15) -> Keypoints:
    kps = np.full((NUM_BODY_JOINTS,2), np.nan, np.float32)
    J = min(NUM_BODY_JOINTS, arr.shape[0])
    for j in range(J):
        x, y, c = arr[j]
        if c >= thr:
            kps[j] = (x, y)
    return kps

def dict_to_kps_single(p: Dict[str,Any], w: int, h: int) -> Keypoints:
    raw = _reshape(p["pose_keypoints_2d"] if "pose_keypoints_2d" in p else p["keypoints"])
    if raw[:, :2].max() <= 1.01:                   # 0-1 normalised coords
        raw[:, 0] *= w
        raw[:, 1] *= h
    return _coords_to_xy(raw)

def kps_from_pose_json(js: List[Dict[str,Any]]) -> List[Keypoints]:
    if not (js and isinstance(js, list)):
        return []
    frame = js[0]
    w, h = frame.get("canvas_width",1), frame.get("canvas_height",1)
    if "people" in frame:      # OpenPose
        return [dict_to_kps_single(p,w,h) for p in frame["people"]]
    if "animals" in frame:     # AP-10K
        return [dict_to_kps_single({"pose_keypoints_2d":a[:NUM_BODY_JOINTS*3]}, w, h)
                for a in frame["animals"]]
    return []

def estimate_translation(kps_json: Keypoints, kps_img: Keypoints,
                         focus: np.ndarray = TORSO, min_pairs: int = 2) -> np.ndarray:
    vis = (~np.isnan(kps_json[:,0]) & ~np.isnan(kps_img[:,0]) &
           np.isin(np.arange(NUM_BODY_JOINTS), focus))
    if vis.sum() < min_pairs:
        return np.zeros(2, np.float32)
    return np.nanmedian(kps_img[vis] - kps_json[vis], 0).astype(np.float32)

def correct_json_offset(kps_json: Keypoints, kps_img: Keypoints) -> Keypoints:
    return kps_json + estimate_translation(kps_json, kps_img)

def robust_scale(src: Keypoints, dst: Keypoints) -> Optional[float]:
    ratios = []
    for i,j in STABLE_SEGMENTS:
        if not (np.isnan(src[i]).any() or np.isnan(src[j]).any() or
                np.isnan(dst[i]).any() or np.isnan(dst[j]).any()):
            d_src = np.linalg.norm(src[i]-src[j])
            d_dst = np.linalg.norm(dst[i]-dst[j])
            if d_src > 1e-3:
                ratios.append(d_dst / d_src)
    return np.median(ratios) if ratios else None

def robust_translation(src: Keypoints, dst: Keypoints,
                       s: float, R: np.ndarray,
                       joints: np.ndarray = ROBUST_JOINTS) -> np.ndarray:
    deltas = []
    for j in joints:
        if j < NUM_BODY_JOINTS and not (np.isnan(src[j]).any() or np.isnan(dst[j]).any()):
            deltas.append(dst[j] - s*(R @ src[j]))
    return (np.median(deltas,0).astype(np.float32)
            if deltas else np.zeros(2,np.float32))

def refine_translation(src: Keypoints, dst: Keypoints,
                       s: float, R: np.ndarray, t0: np.ndarray,
                       max_iter: int = 3, tol: float = 0.05) -> np.ndarray:
    vis = (~np.isnan(src[:,0]) & ~np.isnan(dst[:,0]) &
           np.isin(np.arange(NUM_BODY_JOINTS), TORSO))
    if vis.sum() < 2:
        return t0
    t = t0.copy()
    for _ in range(max_iter):
        warped = (s*(R @ src[vis].T)).T + t
        delta  = np.nanmedian(dst[vis]-warped,0).astype(np.float32)
        if np.linalg.norm(delta) < tol:
            break
        t += delta
    return t

def extract_kps_from_mask(img: np.ndarray,
                          mask: Optional[np.ndarray]=None, tol: int = 10) -> Keypoints:
    if mask is None:
        mask = np.ones(img.shape[:2], bool)
    kps = np.full((NUM_BODY_JOINTS,2), np.nan, np.float32)
    for j,(r,g,b) in OPENPOSE_COLOUR_MAP.items():
        m = ((abs(img[...,2]-r)<tol)&(abs(img[...,1]-g)<tol)&
             (abs(img[...,0]-b)<tol)&mask)
        if m.any():
            ys,xs = np.nonzero(m)
            kps[j]=(xs.mean(),ys.mean())
    return kps

def fit_pair(src: Keypoints, dst: Keypoints
            ) -> Tuple[float,np.ndarray,np.ndarray,float]:
    vis = ~np.isnan(src[:,0]) & ~np.isnan(dst[:,0])
    if vis.sum() < 2:
        return 1., np.eye(2,dtype=np.float32), np.zeros(2,np.float32), float("inf")

    # 1) scale
    s = robust_scale(src, dst) or 1.0

    # 2) rotation from Procrustes at that scale
    muX, muY = src[vis].mean(0), dst[vis].mean(0)
    Xc, Yc   = src[vis]-muX,  dst[vis]-muY
    U,_,Vt   = np.linalg.svd(Xc.T @ Yc)
    R        = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1] *= -1
        R = Vt.T @ U.T

    # 3) translation (robust, then iteratively polished)
    t  = robust_translation(src, dst, s, R)
    t  = refine_translation(src, dst, s, R, t)

    # 4) residual
    recon = (s*(R @ src[vis].T)).T + t
    mse = float(np.mean(np.linalg.norm(recon-dst[vis],axis=1)**2))
    return s, R.astype(np.float32), t.astype(np.float32), mse

def two_largest(mask: np.ndarray)->Tuple[np.ndarray,np.ndarray]:
    n,lab = cv2.connectedComponents(mask.astype(np.uint8), 8)
    if n<=2:
        return [mask], [np.zeros_like(mask,bool)]
    areas=[(lab==i).sum() for i in range(1,n)]
    a,b  = np.argsort(areas)[-2:]
    return [(lab==a+1),(lab==b+1)]

def normalize_angle(angle_deg: float) -> float:
    """Normalize angle to 0-360 degrees range"""
    return ((angle_deg % 360) + 360) % 360

def _build_affine(scale: float, angle_deg: float,
                  tx: float, ty: float,
                  cx: float, cy: float) -> np.ndarray:
    # Normalize angle to prevent overflow issues
    angle_deg = normalize_angle(angle_deg)
    
    th = math.radians(angle_deg)
    R  = np.array([[math.cos(th), -math.sin(th)],
                   [math.sin(th),  math.cos(th)]], np.float32) * scale
    t  = np.array([tx, ty], np.float32) + np.array([cx, cy]) - R @ np.array([cx, cy])
    return np.hstack([R, t[:, None]])

def decompose_affine_matrix(matrix: np.ndarray, cx: float, cy: float) -> Tuple[float, float, float, float]:
    """
    Decompose an affine transformation matrix back into scale, rotation, and translation components.
    
    Args:
        matrix: 2x3 affine transformation matrix [R|t] where R is 2x2 rotation+scale, t is 2x1 translation
        cx, cy: Center of rotation used in the original transformation
        
    Returns:
        (scale, angle_deg, tx, ty) - Individual transformation components
    """
    # Extract rotation+scale matrix and translation vector
    R = matrix[:2, :2]  # 2x2 rotation+scale matrix
    t = matrix[:2, 2]   # 2x1 translation vector
    
    # Decompose scale and rotation from the 2x2 matrix
    # For a rotation+scale matrix: R = scale * [[cos(θ), -sin(θ)], [sin(θ), cos(θ)]]
    scale = np.sqrt(np.linalg.det(R))  # Determinant gives scale^2 for rotation matrix
    
    # Handle potential negative scales (reflections)
    if np.linalg.det(R) < 0:
        scale = -scale
    
    # Extract rotation matrix by normalizing out the scale
    R_normalized = R / scale
    
    # Extract angle from normalized rotation matrix
    # cos(θ) = R_normalized[0,0], sin(θ) = R_normalized[1,0]
    angle_rad = math.atan2(R_normalized[1, 0], R_normalized[0, 0])
    angle_deg = math.degrees(angle_rad)
    
    # Recover the original translation (tx, ty) by reversing the center compensation
    # In _build_affine: final_t = t + center - R @ center
    # So: original_t = final_t - center + R @ center
    center = np.array([cx, cy])
    original_t = t - center + R @ center
    
    tx, ty = original_t[0], original_t[1]
    
    # Normalize angle to 0-360 range
    angle_deg = normalize_angle(angle_deg)
    
    return float(scale), float(angle_deg), float(tx), float(ty)

# Global storage for transformation data (for API access)
_transform_data_cache = {}

def store_transform_data(node_id: str, matrices: Dict[str, np.ndarray], offset_corrections: Dict[str, Dict[str, float]]):
    """Store transformation data for API access"""
    global _transform_data_cache
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_matrices = {}
    for key, matrix in matrices.items():
        if matrix is not None:
            serializable_matrices[key] = matrix.tolist()
        else:
            serializable_matrices[key] = None
    
    _transform_data_cache[node_id] = {
        'timestamp': time.time(),
        'matrices': serializable_matrices,
        'offsetCorrections': offset_corrections
    }

def get_transform_data(node_id: str) -> Optional[Dict]:
    """Retrieve transformation data for API access"""
    global _transform_data_cache
    return _transform_data_cache.get(node_id)

# ──────────────────────────── Enhanced Main alignment node ─────────────────────────
class PoseAlignTwoToOne(PreviewImage):
    CATEGORY = "AInseven"

    def __init__(self):
        # Initialize parent class
        super().__init__()

        # Add the required attribute for PreviewImage
        self.prefix_append = ""
        self.compress_level = 4

        # Your custom attributes
        self._MA: Optional[np.ndarray] = None
        self._MB: Optional[np.ndarray] = None
        self._out_size: Optional[Tuple[int, int]] = None
        self._offset_corrections: Dict[str, Dict[str, float]] = {'A': {'x': 0, 'y': 0}, 'B': {'x': 0, 'y': 0}}

    # ─────────────────────────── node inputs ──────────────────────────
    @classmethod
    def INPUT_TYPES(cls):
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
                # Manual-mode sliders for pose A
                "angle_deg_A": ("FLOAT", {"default": 0, "min": -720, "max": 720, "step": 0.1}),
                "scale_A": ("FLOAT", {"default": 1.0, "min": 0.20, "max": 3.0, "step": 0.01}),
                "tx_A": ("INT", {"default": 0, "min": -2048, "max": 2048}),
                "ty_A": ("INT", {"default": 0, "min": -2048, "max": 2048}),
                # Manual-mode sliders for pose B
                "angle_deg_B": ("FLOAT", {"default": 0, "min": -720, "max": 720, "step": 0.1}),
                "scale_B": ("FLOAT", {"default": 1.0, "min": 0.20, "max": 3.0, "step": 0.01}),
                "tx_B": ("INT", {"default": 0, "min": -2048, "max": 2048}),
                "ty_B": ("INT", {"default": 0, "min": -2048, "max": 2048}),
            },
            "optional": {
                "debug": ("BOOLEAN", {"default": False})
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("aligned_poseA", "aligned_poseB", "combined_AB", "combine_all")
    FUNCTION = "align"

    # ────────────────── helper to fetch key-points ────────────────────
    def _get_kps(self, img: np.ndarray, js: List[Dict[str, Any]], idx: int) -> Keypoints:
        people = kps_from_pose_json(js)
        return people[idx] if people and idx < len(people) else extract_kps_from_mask(img)

    # ─────────────────── Store images using ComfyUI's PreviewImage system ─────────────────
    def _save_preview_images(self, ref_pose_img, poseA_img, poseB_img, prompt=None, extra_pnginfo=None):
        """Save input images using ComfyUI's PreviewImage mechanism"""

        # Get tensors from the first batch
        ref_tensor = ref_pose_img[0:1]
        A_tensor = poseA_img[0:1]
        B_tensor = poseB_img[0:1]

        # Find the maximum height and width
        max_height = max(ref_tensor.shape[1], A_tensor.shape[1], B_tensor.shape[1])
        max_width = max(ref_tensor.shape[2], A_tensor.shape[2], B_tensor.shape[2])

        # Resize all images to the same dimensions
        import torch.nn.functional as F

        def resize_tensor(tensor, target_h, target_w):
            # tensor shape: (B, H, W, C) in ComfyUI
            # Need to permute to (B, C, H, W) for F.interpolate
            tensor = tensor.permute(0, 3, 1, 2)
            resized = F.interpolate(tensor, size=(target_h, target_w), mode='bilinear', align_corners=False)
            # Permute back to (B, H, W, C)
            return resized.permute(0, 2, 3, 1)

        # Resize all images to the maximum dimensions
        ref_resized = resize_tensor(ref_tensor, max_height, max_width)
        A_resized = resize_tensor(A_tensor, max_height, max_width)
        B_resized = resize_tensor(B_tensor, max_height, max_width)

        # Stack images horizontally (along width dimension)
        combined = torch.cat([ref_resized, A_resized, B_resized], dim=2)  # dim=2 is width in (B,H,W,C)

        # Use PreviewImage's save_images method
        return self.save_images(combined, filename_prefix="pose_align_preview", prompt=prompt, extra_pnginfo=extra_pnginfo)

    def _update_widget_values(self, cx: float, cy: float, debug: bool = False):
        """
        Decompose cached transformation matrices and update widget values.
        This ensures the canvas widget can read the current transformation parameters.
        """
        if self._MA is not None:
            scale_A, angle_deg_A, tx_A, ty_A = decompose_affine_matrix(self._MA, cx, cy)
            
            # Update node properties (which the canvas reads)
            self.properties = getattr(self, 'properties', {})
            self.properties.update({
                'scale_A': scale_A,
                'angle_deg_A': angle_deg_A,
                'tx_A': int(round(tx_A)),
                'ty_A': int(round(ty_A))
            })
            
            if debug:
                print(f"[PoseAlign] Updated widget A: scale={scale_A:.3f}, angle={angle_deg_A:.1f}°, tx={tx_A:.1f}, ty={ty_A:.1f}")
        
        if self._MB is not None:
            scale_B, angle_deg_B, tx_B, ty_B = decompose_affine_matrix(self._MB, cx, cy)
            
            # Update node properties (which the canvas reads)
            self.properties = getattr(self, 'properties', {})
            self.properties.update({
                'scale_B': scale_B,
                'angle_deg_B': angle_deg_B,
                'tx_B': int(round(tx_B)),
                'ty_B': int(round(ty_B))
            })
            
            if debug:
                print(f"[PoseAlign] Updated widget B: scale={scale_B:.3f}, angle={angle_deg_B:.1f}°, tx={tx_B:.1f}, ty={ty_B:.1f}")

    def _store_transform_data_for_canvas(self, debug: bool = False):
        """Store transformation data for canvas access via API"""
        node_id = str(id(self))  # Use object id as unique identifier
        
        matrices = {
            'A': self._MA,
            'B': self._MB
        }
        
        # Store the transformation data
        store_transform_data(node_id, matrices, self._offset_corrections)
        
        if debug:
            print(f"[PoseAlign] Stored transform data for node {node_id}")
            print(f"[PoseAlign] Offset corrections: {self._offset_corrections}")

    # ───────────────────────── main function ──────────────────────────
    def align(self,
              ref_pose_img: torch.Tensor, ref_pose_json: List[Dict[str, Any]],
              poseA_img: torch.Tensor, poseA_json: List[Dict[str, Any]],
              poseB_img: torch.Tensor, poseB_json: List[Dict[str, Any]],
              assignment="auto", manual=False, reset=False, debug=False,
              angle_deg_A=0.0, scale_A=1.0, tx_A=0, ty_A=0,
              angle_deg_B=0.0, scale_B=1.0, tx_B=0, ty_B=0,
              prompt=None, extra_pnginfo=None):

        # Save preview images - this will populate the UI metadata
        ui_result = self._save_preview_images(ref_pose_img, poseA_img, poseB_img, prompt, extra_pnginfo)

        N = poseA_img.shape[0]                       # batch size
        ref_np = torch_to_u8(ref_pose_img[0:1])
        h, w = ref_np.shape[:2]
        cx, cy = w / 2.0, h / 2.0

        # Normalize angles to prevent issues
        angle_deg_A = normalize_angle(angle_deg_A)
        angle_deg_B = normalize_angle(angle_deg_B)

        # ───────────── compute / fetch transformation matrices ───────
        if manual:
            # build from sliders (no caching needed)
            MA = _build_affine(scale_A, angle_deg_A, tx_A, ty_A, cx, cy).astype(np.float32)
            MB = _build_affine(scale_B, angle_deg_B, tx_B, ty_B, cx, cy).astype(np.float32)
            
            # Update properties even in manual mode to keep canvas in sync
            self.properties = getattr(self, 'properties', {})
            self.properties.update({
                'scale_A': scale_A, 'angle_deg_A': angle_deg_A, 'tx_A': tx_A, 'ty_A': ty_A,
                'scale_B': scale_B, 'angle_deg_B': angle_deg_B, 'tx_B': tx_B, 'ty_B': ty_B
            })
            
            # Store current matrices and offset corrections for canvas
            self._MA, self._MB = MA, MB
            
            if debug:
                print(f"[PoseAlign] Manual mode - A: scale={scale_A:.3f}, angle={angle_deg_A:.1f}°, tx={tx_A}, ty={ty_A}")
                print(f"[PoseAlign] Manual mode - B: scale={scale_B:.3f}, angle={angle_deg_B:.1f}°, tx={tx_B}, ty={ty_B}")
        else:
            # automatic mode – identical to original implementation
            need_fit = reset or self._MA is None or self._MB is None
            if need_fit:
                A_np = torch_to_u8(poseA_img[0:1])
                B_np = torch_to_u8(poseB_img[0:1])

                # reference key-points (with JSON↔image offset correction)
                ref_people_raw = kps_from_pose_json(ref_pose_json)
                if len(ref_people_raw) >= 2:
                    m1, m2 = two_largest(cv2.cvtColor(ref_np, cv2.COLOR_BGR2GRAY) > 2)
                    img_people = [extract_kps_from_mask(ref_np, m1),
                                  extract_kps_from_mask(ref_np, m2)]
                    kR1 = correct_json_offset(ref_people_raw[0], img_people[0])
                    kR2 = correct_json_offset(ref_people_raw[1], img_people[1])
                else:
                    m1, m2 = two_largest(cv2.cvtColor(ref_np, cv2.COLOR_BGR2GRAY) > 2)
                    kR1, kR2 = extract_kps_from_mask(ref_np, m1), extract_kps_from_mask(ref_np, m2)

                # pose A & B key-points with offset correction tracking
                kA_json = self._get_kps(A_np, poseA_json, 0)
                kB_json = self._get_kps(B_np, poseB_json, 0)
                kA_img = extract_kps_from_mask(A_np)
                kB_img = extract_kps_from_mask(B_np)
                
                # Calculate and store offset corrections
                offset_A = estimate_translation(kA_json, kA_img)
                offset_B = estimate_translation(kB_json, kB_img)
                self._offset_corrections = {
                    'A': {'x': float(offset_A[0]), 'y': float(offset_A[1])},
                    'B': {'x': float(offset_B[0]), 'y': float(offset_B[1])}
                }
                
                kA = correct_json_offset(kA_json, kA_img)
                kB = correct_json_offset(kB_json, kB_img)

                # similarity fits
                sA1, RA1, tA1, eA1 = fit_pair(kA, kR1)
                sB2, RB2, tB2, eB2 = fit_pair(kB, kR2)
                sA2, RA2, tA2, eA2 = fit_pair(kA, kR2)
                sB1, RB1, tB1, eB1 = fit_pair(kB, kR1)

                pick = 0 if assignment == "A_to_first" else \
                       1 if assignment == "A_to_second" else \
                       (0 if eA1 + eB2 <= eA2 + eB1 else 1)

                if pick == 0:
                    sA, RA, tA = sA1, RA1, tA1
                    sB, RB, tB = sB2, RB2, tB2
                else:
                    sA, RA, tA = sA2, RA2, tA2
                    sB, RB, tB = sB1, RB1, tB1

                MA = np.hstack([sA * RA, tA[:, None]]).astype(np.float32)
                MB = np.hstack([sB * RB, tB[:, None]]).astype(np.float32)
                self._MA, self._MB, self._out_size = MA, MB, (w, h)
                
                # **KEY FIX**: Decompose matrices and update widget values for canvas
                self._update_widget_values(cx, cy, debug)
                
                if debug:
                    print(f"[PoseAlign] cached new transforms | size: {(w, h)}")
                    print(f"[PoseAlign] Matrix A:\n{MA}")
                    print(f"[PoseAlign] Matrix B:\n{MB}")
                    print(f"[PoseAlign] Offset corrections: {self._offset_corrections}")
            else:
                MA, MB = self._MA, self._MB
                # Update widget values even when using cached matrices
                self._update_widget_values(cx, cy, debug)

        # Store transformation data for canvas access
        self._store_transform_data_for_canvas(debug)

        # ─────────────────── apply transforms to batch ─────────────────
        outA, outB, outC, outAll = [], [], [], []
        for i in range(N):
            A_np = torch_to_u8(poseA_img[i:i+1])
            B_np = torch_to_u8(poseB_img[i:i+1])

            A_w = cv2.warpAffine(A_np, MA, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            B_w = cv2.warpAffine(B_np, MB, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            combined = np.where(A_w > 0, A_w, B_w)
            combo_all = ref_np.copy()
            m = A_w > 0; combo_all[m] = A_w[m]
            m = B_w > 0; combo_all[m] = B_w[m]

            outA.append(u8_to_torch(A_w))
            outB.append(u8_to_torch(B_w))
            outC.append(u8_to_torch(combined))
            outAll.append(u8_to_torch(combo_all))

        result = {"result": (torch.cat(outA, 0),
                            torch.cat(outB, 0),
                            torch.cat(outC, 0),
                            torch.cat(outAll, 0))}
        
        # Merge UI result with output result
        if "ui" in ui_result:
            result["ui"] = ui_result["ui"]
            
        return result

# ───────────────────── pose viewer debug node (unchanged) ──────────────────
class PoseViewer:
    CATEGORY="AInseven/Debug"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
                    "image":("IMAGE",),
                    "pose_json":("POSE_KEYPOINT",),
                    "point_radius":("INT",{"default":8,"min":1,"max":50}),
                    "line_thickness":("INT",{"default":4,"min":1,"max":20}),
                    "draw_limbs":("BOOLEAN",{"default":True}),
                }}

    RETURN_TYPES=("IMAGE",)
    FUNCTION="view"

    def view(self,image:torch.Tensor, pose_json:List[Dict[str,Any]],
             point_radius:int, line_thickness:int, draw_limbs:bool):
        img_np=torch_to_u8(image)
        all_kps=kps_from_pose_json(pose_json)
        for person_kps in all_kps:
            color=tuple(map(int,np.random.randint(100,256,3)))
            if draw_limbs:
                for p1,p2 in BODY_25_PAIRS:
                    if p1<len(person_kps) and p2<len(person_kps):
                        pt1,pt2=person_kps[p1],person_kps[p2]
                        if not np.isnan(pt1).any() and not np.isnan(pt2).any():
                            cv2.line(img_np,tuple(np.int32(pt1)),tuple(np.int32(pt2)),
                                     color,line_thickness)
            for pt in person_kps:
                if not np.isnan(pt).any():
                    cv2.circle(img_np,tuple(np.int32(pt)),point_radius,color,-1)
                    cv2.circle(img_np,tuple(np.int32(pt)),point_radius,(0,0,0),2)
        return (u8_to_torch(img_np),)
        

# ──────────────────────────── API Route for Canvas Data Access ─────────────────────────
async def get_pose_align_data(request):
    """API endpoint to get transformation data for canvas"""
    try:
        node_id = request.match_info.get('node_id')
        data = get_transform_data(node_id)
        
        if data is None:
            return web.json_response({'error': 'No data found for node'}, status=404)
        
        return web.json_response(data)
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)

# Register the API route (this should be called when ComfyUI loads the extension)
def register_api_routes(app):
    """Register API routes for pose align data"""
    app.router.add_get('/AInseven/pose_align_data/{node_id}', get_pose_align_data)

# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "PoseAlignTwoToOne": PoseAlignTwoToOne
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PoseAlignTwoToOne": "Pose Align Two To One (Fixed)"
}

# Try to register API routes if ComfyUI app is available
try:
    from main import app
    register_api_routes(app)
    print("[PoseAlign] API routes registered successfully")
except:
    print("[PoseAlign] Could not register API routes - will try later")
    # Alternative registration method
    try:
        import server
        if hasattr(server, 'app'):
            register_api_routes(server.app)
            print("[PoseAlign] API routes registered via server module")
    except:
        print("[PoseAlign] API routes not registered - canvas sync may not work")
