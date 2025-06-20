# ©2025 – MIT licence
"""
__init__.py for the PoseAlignTwoToOne custom node package
"""

# Import the node classes from your python file
from .pose_align_node import PoseAlignTwoToOne, PoseViewer

# Tell ComfyUI about the nodes in this package
NODE_CLASS_MAPPINGS = {
    "PoseAlignTwoToOne": PoseAlignTwoToOne,
    "PoseViewer": PoseViewer,
}

# Give the nodes a friendly name for the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "PoseAlignTwoToOne": "Pose Align (2→1)",
    "PoseViewer": "Pose Viewer (Debug)",
}

# Tell ComfyUI that this node has a web directory to serve
# This is the crucial line that will load your javascript
WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
