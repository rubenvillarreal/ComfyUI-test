"""
debug_imports.py - Helper script to test imports and debug issues
Place this in the same directory as your pose align files to test imports.
"""

import sys
import traceback

def test_imports():
    print("Testing pose align imports...")
    print("=" * 50)
    
    # Test 1: Import utilities
    try:
        print("1. Testing pose_align_utils import...")
        import pose_align_utils
        print("   ✓ pose_align_utils imported successfully")
        
        # Test specific functions
        funcs_to_test = [
            'torch_to_u8', 'u8_to_torch', 'kps_from_pose_json',
            'fit_pair', '_build_affine', 'store_transform_data'
        ]
        
        for func_name in funcs_to_test:
            if hasattr(pose_align_utils, func_name):
                print(f"   ✓ {func_name} found")
            else:
                print(f"   ✗ {func_name} missing")
                
    except Exception as e:
        print(f"   ✗ Failed to import pose_align_utils: {e}")
        traceback.print_exc()
    
    print()
    
    # Test 2: Import nodes
    try:
        print("2. Testing pose_align_nodes import...")
        import pose_align_nodes
        print("   ✓ pose_align_nodes imported successfully")
        
        # Test node classes
        if hasattr(pose_align_nodes, 'PoseAlignTwoToOne'):
            print("   ✓ PoseAlignTwoToOne found")
            node_class = pose_align_nodes.PoseAlignTwoToOne
            print(f"   ✓ Node class: {node_class}")
            
            # Test node methods
            if hasattr(node_class, 'INPUT_TYPES'):
                print("   ✓ INPUT_TYPES method found")
            if hasattr(node_class, 'align'):
                print("   ✓ align method found")
        else:
            print("   ✗ PoseAlignTwoToOne not found")
            
        if hasattr(pose_align_nodes, 'PoseViewer'):
            print("   ✓ PoseViewer found")
        else:
            print("   ✗ PoseViewer not found")
            
    except Exception as e:
        print(f"   ✗ Failed to import pose_align_nodes: {e}")
        traceback.print_exc()
    
    print()
    
    # Test 3: Test the __init__.py import logic
    try:
        print("3. Testing __init__.py import logic...")
        
        # Simulate the import logic from __init__.py
        try:
            from pose_align_nodes import PoseAlignTwoToOne, PoseViewer
            from pose_align_utils import get_transform_data
            print("   ✓ Direct imports successful")
            
            # Test node mappings
            NODE_CLASS_MAPPINGS = {
                "PoseAlignTwoToOne": PoseAlignTwoToOne,
                "PoseViewer": PoseViewer
            }
            print("   ✓ Node mappings created successfully")
            print(f"   ✓ Available nodes: {list(NODE_CLASS_MAPPINGS.keys())}")
            
        except ImportError as e:
            print(f"   ✗ Direct imports failed: {e}")
            
    except Exception as e:
        print(f"   ✗ __init__.py logic test failed: {e}")
        traceback.print_exc()
    
    print()
    print("=" * 50)
    print("Import testing complete!")

if __name__ == "__main__":
    test_imports()
