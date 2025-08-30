"""Simple import test for steganalysis"""
import torch
print("✅ PyTorch imported successfully")

try:
    import cv2
    print("✅ OpenCV imported successfully")
except ImportError as e:
    print(f"❌ OpenCV import failed: {e}")

try:
    import seaborn
    print("✅ Seaborn imported successfully")
except ImportError as e:
    print(f"❌ Seaborn import failed: {e}")

try:
    from sklearn.metrics import roc_curve
    print("✅ Scikit-learn imported successfully")
except ImportError as e:
    print(f"❌ Scikit-learn import failed: {e}")

# Test basic steganalysis import
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from steganalysis_system import SteganalysisSystem
    print("✅ SteganalysisSystem imported successfully")
    
    # Test basic functionality
    system = SteganalysisSystem()
    print("✅ SteganalysisSystem initialized successfully")
    
except Exception as e:
    print(f"❌ SteganalysisSystem error: {e}")
    import traceback
    traceback.print_exc()

print("Import test completed!")
