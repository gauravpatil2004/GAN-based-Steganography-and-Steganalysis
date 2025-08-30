"""
Quick import test for GAN components
"""

import sys
sys.path.append('src')

print("🔧 Testing imports...")

try:
    import torch
    print("✅ PyTorch imported")
    
    from gan_architecture import SteganoGenerator, SteganoDiscriminator, SecretExtractor
    print("✅ GAN architecture imported")
    
    from gan_losses import SteganographyLoss, MetricsCalculator
    print("✅ GAN losses imported")
    
    from data_loader import create_data_loader
    print("✅ Data loader imported")
    
    from gan_training import SteganographyTrainer
    print("✅ Training framework imported")
    
    print("\n🎉 All imports successful!")
    print("🚀 Ready to train!")
    
except Exception as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()
