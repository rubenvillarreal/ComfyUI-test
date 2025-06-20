"""
__init__.py for the PoseAlignTwoToOne custom node package
"""
from aiohttp import web
import json
import time

# Import the node classes from your python file
try:
    from .pose_align_node import PoseAlignTwoToOne, PoseViewer, get_transform_data
    print("[PoseAlign] Successfully imported node classes")
except ImportError as e:
    print(f"[PoseAlign] Failed to import node classes: {e}")
    raise

# Tell ComfyUI about the nodes in this package
NODE_CLASS_MAPPINGS = {
    "PoseAlignTwoToOne": PoseAlignTwoToOne,
    "PoseViewer": PoseViewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PoseAlignTwoToOne": "Pose Align (2→1) Fixed",
    "PoseViewer": "Pose Viewer (Debug)",
}

# Tell ComfyUI that this node has a web directory to serve
WEB_DIRECTORY = "./web"

# ──────────────────────────── API ROUTES FOR CANVAS SYNC ─────────────────────────

try:
    from server import PromptServer
    
    @PromptServer.instance.routes.get("/AInseven/pose_align_data/{node_id}")
    async def get_pose_align_data(request):
        """API endpoint to get transformation data for canvas"""
        try:
            node_id = request.match_info.get('node_id')
            data = get_transform_data(node_id)
            
            if data is None:
                return web.json_response({
                    'error': 'No data found for node',
                    'timestamp': time.time(),
                    'matrices': {'A': None, 'B': None},
                    'offsetCorrections': {'A': {'x': 0, 'y': 0}, 'B': {'x': 0, 'y': 0}}
                }, status=200)
            
            return web.json_response(data)
        except Exception as e:
            print(f"[PoseAlign API] Error: {e}")
            return web.json_response({
                'error': str(e),
                'timestamp': time.time(),
                'matrices': {'A': None, 'B': None},
                'offsetCorrections': {'A': {'x': 0, 'y': 0}, 'B': {'x': 0, 'y': 0}}
            }, status=200)
    
    print("[PoseAlign] API routes registered successfully using PromptServer.instance.routes")

except ImportError as e:
    print(f"[PoseAlign] Could not import PromptServer: {e}")
    print("[PoseAlign] Canvas will fall back to widget-only mode")
except Exception as e:
    print(f"[PoseAlign] Error registering API routes: {e}")
    print("[PoseAlign] Canvas will fall back to widget-only mode")

# Critical: Export these for ComfyUI to find
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

print(f"[PoseAlign] Registered nodes: {list(NODE_CLASS_MAPPINGS.keys())}")
