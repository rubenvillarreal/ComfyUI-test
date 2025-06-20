"""
debug_imports_fixed.py - Helper script to test imports and debug issues
Place this in the same directory as your pose align files to test imports.
"""

import sys
import os
import traceback

def test_imports():
    print("Testing pose align imports...")
    print("=" * 50)
    
    # Print current working directory and Python path
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    print(f"Files in current directory: {os.listdir('.')}")
    print()
    
    # Test 1: Import utilities
    try:
        print("1. Testing pose_align_utils import...")
        import pose_align_utils
        print("   ✓ pose_align_utils imported successfully")
        print(f"   ✓ Module file: {pose_align_utils.__file__}")
        
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
    
    # Test 2: Import nodes - try different variations
    print("2. Testing pose_align_node import...")
    try:
        # Try the actual filename first
        import pose_align_node
        print("   ✓ pose_align_node imported successfully")
        print(f"   ✓ Module file: {pose_align_node.__file__}")
        
        # Test node classes
        if hasattr(pose_align_node, 'PoseAlignTwoToOne'):
            print("   ✓ PoseAlignTwoToOne found")
            node_class = pose_align_node.PoseAlignTwoToOne
            print(f"   ✓ Node class: {node_class}")
            
            # Test node methods
            if hasattr(node_class, 'INPUT_TYPES'):
                print("   ✓ INPUT_TYPES method found")
                try:
                    input_types = node_class.INPUT_TYPES()
                    print(f"   ✓ INPUT_TYPES returns: {list(input_types.keys())}")
                except Exception as e:
                    print(f"   ⚠ INPUT_TYPES call failed: {e}")
            if hasattr(node_class, 'align'):
                print("   ✓ align method found")
        else:
            print("   ✗ PoseAlignTwoToOne not found")
            print(f"   Available attributes: {[attr for attr in dir(pose_align_node) if not attr.startswith('_')]}")
            
        if hasattr(pose_align_node, 'PoseViewer'):
            print("   ✓ PoseViewer found")
        else:
            print("   ✗ PoseViewer not found")
            
    except Exception as e:
        print(f"   ✗ Failed to import pose_align_node: {e}")
        traceback.print_exc()
    
    print()
    
    # Test 3: Test the __init__.py import logic
    try:
        print("3. Testing __init__.py import logic...")
        
        # Check if __init__.py exists
        if os.path.exists('__init__.py'):
            print("   ✓ __init__.py file exists")
            
            # Try to read the content
            with open('__init__.py', 'r') as f:
                content = f.read()
                if 'PoseAlignTwoToOne' in content:
                    print("   ✓ PoseAlignTwoToOne mentioned in __init__.py")
                else:
                    print("   ✗ PoseAlignTwoToOne not found in __init__.py")
        else:
            print("   ✗ __init__.py file not found")
        
        # Simulate the import logic from __init__.py
        try:
            from pose_align_node import PoseAlignTwoToOne
            print("   ✓ Direct import from pose_align_node successful")
            
            try:
                from pose_align_node import PoseViewer
                print("   ✓ PoseViewer import successful")
            except ImportError:
                print("   ✗ PoseViewer import failed")
            
            from pose_align_utils import get_transform_data
            print("   ✓ get_transform_data import successful")
            
            # Test node mappings
            NODE_CLASS_MAPPINGS = {
                "PoseAlignTwoToOne": PoseAlignTwoToOne,
            }
            print("   ✓ Node mappings created successfully")
            print(f"   ✓ Available nodes: {list(NODE_CLASS_MAPPINGS.keys())}")
            
        except ImportError as e:
            print(f"   ✗ Direct imports failed: {e}")
            traceback.print_exc()
            
    except Exception as e:
        print(f"   ✗ __init__.py logic test failed: {e}")
        traceback.print_exc()
    
    print()
    
    # Test 4: Check ComfyUI compatibility
    try:
        print("4. Testing ComfyUI compatibility...")
        
        # Try to import ComfyUI nodes module
        try:
            from nodes import PreviewImage
            print("   ✓ ComfyUI nodes module accessible")
            print("   ✓ PreviewImage class available")
        except ImportError as e:
            print(f"   ✗ ComfyUI nodes not accessible: {e}")
        
        # Check if our node inherits correctly
        try:
            from pose_align_node import PoseAlignTwoToOne
            from nodes import PreviewImage
            
            if issubclass(PoseAlignTwoToOne, PreviewImage):
                print("   ✓ PoseAlignTwoToOne correctly inherits from PreviewImage")
            else:
                print("   ✗ PoseAlignTwoToOne does not inherit from PreviewImage")
                print(f"   MRO: {PoseAlignTwoToOne.__mro__}")
                
        except Exception as e:
            print(f"   ✗ Inheritance check failed: {e}")
    
    except Exception as e:
        print(f"   ✗ ComfyUI compatibility test failed: {e}")
        traceback.print_exc()
    
    print()
    print("=" * 50)
    print("Import testing complete!")
    
    # Final summary
    print("\nSUMMARY:")
    try:
        import pose_align_utils
        import pose_align_node
        from pose_align_node import PoseAlignTwoToOne
        print("✓ All critical imports successful")
        print("✓ Node should be available in ComfyUI")
        
        # Test if we can create an instance
        try:
            node_instance = PoseAlignTwoToOne()
            print("✓ Node instance creation successful")
        except Exception as e:
            print(f"⚠ Node instance creation failed: {e}")
            
    except Exception as e:
        print(f"✗ Critical import failures detected: {e}")
        print("✗ Node will NOT be available in ComfyUI")

if __name__ == "__main__":
    test_imports()
