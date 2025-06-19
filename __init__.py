"""
Register PoseAlignTwoToOne and the new PoseViewer with ComfyUI
"""

# 1. Import both classes from your node file
from .pose_align_node import PoseAlignTwoToOne, PoseViewer

# 2. Add both classes to the mapping
NODE_CLASS_MAPPINGS = {
    "PoseAlignTwoToOne": PoseAlignTwoToOne,
    "PoseViewer": PoseViewer # <-- Add this line
}

# 3. Give each class a user-friendly name for the menu
NODE_DISPLAY_NAME_MAPPINGS = {
    "PoseAlignTwoToOne": "Pose Align (Full Body)",
    "PoseViewer": "Pose Viewer (Debug)" # <-- Add this line
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

