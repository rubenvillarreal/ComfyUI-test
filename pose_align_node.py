"""
pose_align_nodes.py – ComfyUI custom nodes for pose alignment
"""

from __future__ import annotations
import cv2
import numpy as np
import torch
import json
import time
from typing import List, Dict, Any, Tuple, Optional
from nodes import PreviewImage
from aiohttp import web

# Import all utilities from our utils module
try:
    from .pose_align_utils import (
        # Constants
        NUM_BODY_JOINTS, TORSO, ROBUST_JOINTS, STABLE_SEGMENTS, BODY_25_PAIRS, OPENPOSE_COLOUR_MAP,
        
        # Type aliases
        Keypoints,
        
        # Tensor conversion functions
        torch_to_u8, u8_to_torch, torch_to_pil,
        
        # Keypoint processing functions
        kps_from_pose_json, extract_kps_from_mask, estimate_translation, correct_json_offset,
        
        # Transformation functions
        robust_scale, robust_translation, refine_translation, fit_pair,
        
        # Image processing utilities
        two_largest,
        
        # Affine transformation utilities
        normalize_angle, _build_affine, decompose_affine_matrix,
        
        # Data storage functions
        store_transform_data, get_transform_data
    )
except ImportError as e:
    print(f"[PoseAlign] Import error: {e}")
    print("[PoseAlign] Falling back to direct imports...")
    # If relative import fails, try absolute import
    from pose_align_utils import (
        NUM_BODY_JOINTS, TORSO, ROBUST_JOINTS, STABLE_SEGMENTS, BODY_25_PAIRS, OPENPOSE_COLOUR_MAP,
        Keypoints, torch_to_u8, u8_to_torch, torch_to_pil, kps_from_pose_json, extract_kps_from_mask,
        estimate_translation, correct_json_offset, robust_scale, robust_translation, refine_translation,
        fit_pair, two_largest, normalize_angle, _build_affine, decompose_affine_matrix,
        store_transform_data, get_transform_data
    )

# ──────────────────────────── Enhanced Main Alignment Node ─────────────────────────
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

    # ─────────────────────────── Node Inputs ──────────────────────────
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

    # ────────────────── Helper Methods ────────────────────
    def _get_kps(self, img: np.ndarray, js: List[Dict[str, Any]], idx: int) -> Keypoints:
        """Helper to fetch key-points from JSON or extract from image mask"""
        people = kps_from_pose_json(js)
        return people[idx] if people and idx < len(people) else extract_kps_from_mask(img)

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

    # ───────────────────────── Main Alignment Function ──────────────────────────
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

        # ───────────── Compute / Fetch Transformation Matrices ───────
        if manual:
            # Build from sliders (no caching needed)
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
            # Automatic mode – compute optimal transformations
            need_fit = reset or self._MA is None or self._MB is None
            if need_fit:
                A_np = torch_to_u8(poseA_img[0:1])
                B_np = torch_to_u8(poseB_img[0:1])

                # Reference key-points (with JSON↔image offset correction)
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

                # Pose A & B key-points with offset correction tracking
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

                # Similarity fits
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
                
                # Update widget values for canvas
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

        # ─────────────────── Apply Transforms to Batch ─────────────────
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


# ───────────────────── Pose Viewer Debug Node ──────────────────
class PoseViewer:
    CATEGORY = "AInseven/Debug"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "image": ("IMAGE",),
                    "pose_json": ("POSE_KEYPOINT",),
                    "point_radius": ("INT", {"default": 8, "min": 1, "max": 50}),
                    "line_thickness": ("INT", {"default": 4, "min": 1, "max": 20}),
                    "draw_limbs": ("BOOLEAN", {"default": True}),
                }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "view"

    def view(self, image: torch.Tensor, pose_json: List[Dict[str, Any]],
             point_radius: int, line_thickness: int, draw_limbs: bool):
        """Visualize pose keypoints and limbs on image"""
        img_np = torch_to_u8(image)
        all_kps = kps_from_pose_json(pose_json)
        
        for person_kps in all_kps:
            color = tuple(map(int, np.random.randint(100, 256, 3)))
            
            if draw_limbs:
                for p1, p2 in BODY_25_PAIRS:
                    if p1 < len(person_kps) and p2 < len(person_kps):
                        pt1, pt2 = person_kps[p1], person_kps[p2]
                        if not np.isnan(pt1).any() and not np.isnan(pt2).any():
                            cv2.line(img_np, tuple(np.int32(pt1)), tuple(np.int32(pt2)),
                                     color, line_thickness)
            
            for pt in person_kps:
                if not np.isnan(pt).any():
                    cv2.circle(img_np, tuple(np.int32(pt)), point_radius, color, -1)
                    cv2.circle(img_np, tuple(np.int32(pt)), point_radius, (0, 0, 0), 2)
        
        return (u8_to_torch(img_np),)


# ──────────────────────────── Node Registration ─────────────────────────
NODE_CLASS_MAPPINGS = {
    "PoseAlignTwoToOne": PoseAlignTwoToOne,
    "PoseViewer": PoseViewer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PoseAlignTwoToOne": "Pose Align Two To One (Fixed)",
    "PoseViewer": "Pose Viewer (Debug)"
}
