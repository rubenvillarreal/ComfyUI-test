"""
pose_align_node.py  –  ComfyUI custom nodes
────────────────────────────────────────────
Align two single‑person pose layers to a two‑person reference pose *and*
supply an easy viewer.  Once the reference scene is calibrated the node
remembers the 2 × 3 affine matrices and applies them to every subsequent
batch coming from two live video streams.

©2025 — MIT licence.
"""

from __future__ import annotations
import cv2, numpy as np, torch
from typing import List, Dict, Any, Tuple, Optional

# ─────────────────────────── body‑joint indices ────────────────────────────
NOSE, NECK, R_SH, L_SH, MIDHIP = 0, 1, 2, 5, 8
FULL_TORSO        = [NOSE, NECK, R_SH, L_SH, MIDHIP]
UPPER_TORSO       = [NOSE, NECK, R_SH, L_SH]
NUM_BODY_JOINTS   = 18
TORSO             = np.asarray([NOSE, NECK, R_SH, L_SH, MIDHIP])

# OpenPose colour map (RGB 0‑255) used by the mask extractor
OPENPOSE_COLOUR_MAP = {
    0:(255,0,0),1:(255,85,0),2:(255,170,0),3:(255,255,0),4:(170,255,0),5:(85,255,0),6:(0,255,0),
    7:(0,255,85),8:(0,255,170),9:(0,255,255),10:(0,170,255),11:(0,85,255),12:(0,0,255),13:(85,0,255),
    14:(170,0,255),15:(255,0,255),16:(255,0,170),17:(255,0,85)}

# BODY‑25 limb pairs for the viewer debug node
BODY_25_PAIRS=[(1,8),(1,2),(1,5),(2,3),(3,4),(5,6),(6,7),(8,9),(9,10),(10,11),
               (8,12),(12,13),(13,14),(1,0),(0,15),(15,17),(0,16),(16,18),
               (14,19),(19,20),(14,21),(11,22),(22,23),(11,24)]

# ───────────────── tensor ↔ OpenCV convenience ─────────────────────────────

def _to_chw(img: torch.Tensor) -> torch.Tensor:
    """Force (C,H,W) layout regardless of what ComfyUI gives us."""
    if img.dim() == 4:
        img = img[0]
    if img.shape[0] == 3:        # CHW already
        return img
    # NHWC ➔ CHW
    return img.permute(2, 0, 1)

def torch_to_u8(t: torch.Tensor) -> np.ndarray:
    """(…,H,W,3) or (…,3,H,W) → uint8 BGR OpenCV array."""
    chw = _to_chw(t)             # (3,H,W)
    rgb = chw.permute(1,2,0).cpu().numpy()
    return cv2.cvtColor((rgb*255).clip(0,255).astype(np.uint8), cv2.COLOR_RGB2BGR)

def u8_to_torch(bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb[None])    # (1,H,W,3) for ComfyUI

# ───────────────────── JSON → key‑points utilities ─────────────────────────
Keypoints = np.ndarray

def _reshape(flat: List[float]) -> np.ndarray:
    return np.asarray(flat, np.float32).reshape(-1, 3)[:, :3]

def _coords_to_xy(arr: np.ndarray, thr: float = 0.15) -> Keypoints:
    kps = np.full((NUM_BODY_JOINTS, 2), np.nan, np.float32)
    J = min(NUM_BODY_JOINTS, arr.shape[0])
    for j in range(J):
        x, y, c = arr[j]
        if c >= thr:
            kps[j] = (x, y)
    return kps

def dict_to_kps_single(p: Dict[str, Any], w: int, h: int) -> Keypoints:
    raw = _reshape(p["pose_keypoints_2d"] if "pose_keypoints_2d" in p else p["keypoints"])
    if raw[:, :2].max() <= 1.01:  # normalised 0‑1 coords
        raw[:, 0] *= w
        raw[:, 1] *= h
    return _coords_to_xy(raw)

def kps_from_pose_json(js: List[Dict[str, Any]]) -> List[Keypoints]:
    if not (js and isinstance(js, list)):
        return []
    frame = js[0]
    w, h = frame.get("canvas_width", 1), frame.get("canvas_height", 1)
    if "people" in frame:
        return [dict_to_kps_single(p, w, h) for p in frame["people"]]
    if "animals" in frame:
        return [dict_to_kps_single({"pose_keypoints_2d": a[:NUM_BODY_JOINTS*3]}, w, h)
                for a in frame["animals"]]
    return []

# ─────────────────────── Translation‑offset helpers ───────────────────────

def estimate_translation(kps_json: Keypoints, kps_img: Keypoints,
                         focus: np.ndarray = TORSO, min_pairs: int = 2) -> np.ndarray:
    """Median (dx, dy) between matching visible joints."""
    vis = (~np.isnan(kps_json[:, 0]) & ~np.isnan(kps_img[:, 0]) &
           np.isin(np.arange(NUM_BODY_JOINTS), focus))
    if vis.sum() < min_pairs:
        return np.zeros(2, np.float32)
    return np.nanmedian(kps_img[vis] - kps_json[vis], 0).astype(np.float32)

def correct_json_offset(kps_json: Keypoints, kps_img: Keypoints) -> Keypoints:
    return kps_json + estimate_translation(kps_json, kps_img)

# ───────────────────── similarity & refinement helpers ────────────────────

def procrustes(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    muX, muY = src.mean(0), dst.mean(0)
    Xc, Yc = src - muX, dst - muY
    U, _, Vt = np.linalg.svd(Xc.T @ Yc)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1] *= -1
        R = Vt.T @ U.T
    s = np.sum(Yc * (R @ Xc.T).T) / np.sum(Xc**2)
    t = muY - s * (R @ muX)
    return float(s), R.astype(np.float32), t.astype(np.float32)

def refine_translation(src: Keypoints, dst: Keypoints, s: float, R: np.ndarray, t0: np.ndarray,
                       max_iter: int = 3, tol: float = 0.05) -> np.ndarray:
    vis = (~np.isnan(src[:, 0]) & ~np.isnan(dst[:, 0]) &
           np.isin(np.arange(NUM_BODY_JOINTS), TORSO))
    if vis.sum() < 2:
        return t0
    t = t0.copy().astype(np.float32)
    for _ in range(max_iter):
        warped = (s * (R @ src[vis].T)).T + t
        delta = np.nanmedian(dst[vis] - warped, 0).astype(np.float32)
        if np.linalg.norm(delta) < tol:
            break
        t += delta
    return t

def two_largest(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), 8)
    if n <= 2:
        return [mask], [np.zeros_like(mask, bool)]
    areas = [(lab == i).sum() for i in range(1, n)]
    a, b = np.argsort(areas)[-2:]
    return [(lab == a + 1), (lab == b + 1)]

def fit_pair(src: Keypoints, dst: Keypoints) -> Tuple[float, np.ndarray, np.ndarray, float]:
    vis = ~np.isnan(src[:, 0]) & ~np.isnan(dst[:, 0])
    if vis.sum() < 2:
        return 1.0, np.eye(2, dtype=np.float32), np.zeros(2, np.float32), float("inf")
    s, R = 1.0, np.eye(2, dtype=np.float32)
    tiers = [FULL_TORSO, UPPER_TORSO, [NECK, MIDHIP], np.where(vis)[0]]
    fit_mask = None
    for idxs in tiers:
        mask = vis & np.isin(np.arange(NUM_BODY_JOINTS), np.asarray(idxs))
        if mask.sum() >= 2:
            s, R, _ = procrustes(src[mask], dst[mask])
            fit_mask = mask
            break
    if fit_mask is None:
        return 1.0, np.eye(2, dtype=np.float32), np.zeros(2, np.float32), float("inf")
    t = refine_translation(src, dst, s, R, np.zeros(2, np.float32))
    recon = (s * (R @ src[fit_mask].T)).T + t
    mse = float(np.mean(np.linalg.norm(recon - dst[fit_mask], axis=1) ** 2))
    return s, R, t, mse

def extract_kps_from_mask(img: np.ndarray, mask: Optional[np.ndarray] = None, tol: int = 10) -> Keypoints:
    if mask is None:
        mask = np.ones(img.shape[:2], bool)
    kps = np.full((NUM_BODY_JOINTS, 2), np.nan, np.float32)
    for j, (r, g, b) in OPENPOSE_COLOUR_MAP.items():
        m = ((abs(img[..., 2] - r) < tol) & (abs(img[..., 1] - g) < tol) &
             (abs(img[..., 0] - b) < tol) & mask)
        if m.any():
            ys, xs = np.nonzero(m)
            kps[j] = (xs.mean(), ys.mean())
    return kps

# ──────────────────────────── Main alignment node ─────────────────────────
class PoseAlignTwoToOne:
    CATEGORY = "AInseven"

    def __init__(self):
        # cached similarity transforms and output size (w,h)
        self._MA: Optional[np.ndarray] = None
        self._MB: Optional[np.ndarray] = None
        self._out_size: Optional[Tuple[int, int]] = None

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
                "reset": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "debug": ("BOOLEAN", {"default": False})
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("aligned_poseA", "aligned_poseB", "combined_AB", "combine_all")
    FUNCTION = "align"

    # helper to pull key‑points from json OR mask
    def _get_kps(self, img: np.ndarray, js: List[Dict[str, Any]], idx: int) -> Keypoints:
        people = kps_from_pose_json(js)
        return people[idx] if people and idx < len(people) else extract_kps_from_mask(img)

    # ------------------------------------------------------------------
    def align(
        self,
        ref_pose_img: torch.Tensor,
        ref_pose_json: List[Dict[str, Any]],
        poseA_img: torch.Tensor,
        poseA_json: List[Dict[str, Any]],
        poseB_img: torch.Tensor,
        poseB_json: List[Dict[str, Any]],
        assignment: str = "auto",
        reset: bool = False,
        debug: bool = False,
    ):
        """Align two single‑person pose frames (A,B) to a two‑person reference.

        * If *reset* is False and cached matrices exist, they are reused.
        * Otherwise the first frame of each input is used to recalibrate.
        """

        # number of frames in current batch
        N = poseA_img.shape[0]

        # ---------------------------------------------------------
        # (Re‑)fit similarity transforms if requested or missing
        # ---------------------------------------------------------
        need_fit = reset or self._MA is None or self._MB is None

        if need_fit:
            # Pull numpy images from the first frame only
            ref_np = torch_to_u8(ref_pose_img[0:1])
            A_np   = torch_to_u8(poseA_img[0:1])
            B_np   = torch_to_u8(poseB_img[0:1])
            h, w   = ref_np.shape[:2]

            # ───── reference people (apply offset correction too) ───────────────
            ref_people_raw = kps_from_pose_json(ref_pose_json)
            if len(ref_people_raw) >= 2:
                m1, m2 = two_largest(cv2.cvtColor(ref_np, cv2.COLOR_BGR2GRAY) > 2)
                img_people = [extract_kps_from_mask(ref_np, m1), extract_kps_from_mask(ref_np, m2)]
                kR1 = correct_json_offset(ref_people_raw[0], img_people[0])
                kR2 = correct_json_offset(ref_people_raw[1], img_people[1])
            else:
                m1, m2 = two_largest(cv2.cvtColor(ref_np, cv2.COLOR_BGR2GRAY) > 2)
                kR1, kR2 = extract_kps_from_mask(ref_np, m1), extract_kps_from_mask(ref_np, m2)

            # ───── pose A & B: json → corrected → used -------------------------
            kA_json = self._get_kps(A_np, poseA_json, 0)
            kB_json = self._get_kps(B_np, poseB_json, 0)
            kA_img  = extract_kps_from_mask(A_np)
            kB_img  = extract_kps_from_mask(B_np)
            kA      = correct_json_offset(kA_json, kA_img)
            kB      = correct_json_offset(kB_json, kB_img)

            # ───── similarity fits ---------------------------------------------
            sA1, RA1, tA1, eA1 = fit_pair(kA, kR1)
            sB2, RB2, tB2, eB2 = fit_pair(kB, kR2)
            sA2, RA2, tA2, eA2 = fit_pair(kA, kR2)
            sB1, RB1, tB1, eB1 = fit_pair(kB, kR1)

            pick = (
                0 if assignment == "A_to_first" else
                1 if assignment == "A_to_second" else
                0 if eA1 + eB2 <= eA2 + eB1 else 1
            )
            if pick == 0:
                sA, RA, tA = sA1, RA1, tA1
                sB, RB, tB = sB2, RB2, tB2
            else:
                sA, RA, tA = sA2, RA2, tA2
                sB, RB, tB = sB1, RB1, tB1

            MA = np.hstack([sA * RA, tA[:, None]]).astype(np.float32)
            MB = np.hstack([sB * RB, tB[:, None]]).astype(np.float32)

            # cache results
            self._MA, self._MB, self._out_size = MA, MB, (w, h)

            if debug:
                print("\n[PoseAlign] cached new transforms | size:", (w, h))
        else:
            MA, MB = self._MA, self._MB
            w, h   = self._out_size

        # ---------------------------------------------------------
        # Apply the two affine matrices to every frame in the batch
        # ---------------------------------------------------------
        ref_np = torch_to_u8(ref_pose_img[0:1])   # constant reference

        outA, outB, outC, outAll = [], [], [], []
        for i in range(N):
            A_np = torch_to_u8(poseA_img[i:i+1])
            B_np = torch_to_u8(poseB_img[i:i+1])

            A_w = cv2.warpAffine(
                A_np, MA, (w, h), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
            B_w = cv2.warpAffine(
                B_np, MB, (w, h), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )

            combined = np.where(A_w > 0, A_w, B_w)
            combo_all = ref_np.copy()
            m = A_w > 0
            combo_all[m] = A_w[m]
            m = B_w > 0
            combo_all[m] = B_w[m]

            outA.append(u8_to_torch(A_w))
            outB.append(u8_to_torch(B_w))
            outC.append(u8_to_torch(combined))
            outAll.append(u8_to_torch(combo_all))

        return (
            torch.cat(outA, 0),
            torch.cat(outB, 0),
            torch.cat(outC, 0),
            torch.cat(outAll, 0),
        )

# ───────────────────── pose viewer debug node ─────────────────────────────
class PoseViewer:
    CATEGORY = "AInseven/Debug"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "pose_json": ("POSE_KEYPOINT",),
                "point_radius": ("INT", {"default": 8, "min": 1, "max": 50}),
                "line_thickness": ("INT", {"default": 4, "min": 1, "max": 20}),
                "draw_limbs": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "view"

    def view(
        self,
        image: torch.Tensor,
        pose_json: List[Dict[str, Any]],
        point_radius: int,
        line_thickness: int,
        draw_limbs: bool,
    ):
        img_np = torch_to_u8(image)
        all_kps = kps_from_pose_json(pose_json)
        for person_kps in all_kps:
            color = tuple(map(int, np.random.randint(100, 256, 3)))
            if draw_limbs:
                for p1_idx, p2_idx in BODY_25_PAIRS:
                    if p1_idx < len(person_kps) and p2_idx < len(person_kps):
                        pt1, pt2 = person_kps[p1_idx], person_kps[p2_idx]
                        if not np.isnan(pt1).any() and not np.isnan(pt2).any():
                            cv2.line(img_np, tuple(np.int32(pt1)), tuple(np.int32(pt2)), color, line_thickness)
            for pt in person_kps:
                if not np.isnan(pt).any():
                    cv2.circle(img_np, tuple(np.int32(pt)), point_radius, color, -1)
                    cv2.circle(img_np, tuple(np.int32(pt)), point_radius, (0, 0, 0), 2)
        return (u8_to_torch(img_np),)

# ─────────────────────── node registration ───────────────────────────────
NODE_CLASS_MAPPINGS = {
    "PoseAlignTwoToOne": PoseAlignTwoToOne,
    "PoseViewer": PoseViewer,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PoseAlignTwoToOne": "Pose Align (2→1)",
    "PoseViewer": "Pose Viewer (Debug)",
}
