"""
Pre-Training Checklist for GAN Steganography

Verify all components are ready before starting the long training process.
"""

import os
import torch
import sys

def run_checklist():
    print("🔍 PRE-TRAINING CHECKLIST")
    print("=" * 40)
    
    checks_passed = 0
    total_checks = 8
    
    # Check 1: Python environment
    print("1. Python Environment:")
    try:
        print(f"   ✅ Python version: {sys.version.split()[0]}")
        checks_passed += 1
    except:
        print("   ❌ Python environment issue")
    
    # Check 2: PyTorch installation
    print("2. PyTorch Installation:")
    try:
        print(f"   ✅ PyTorch version: {torch.__version__}")
        print(f"   ✅ CUDA available: {torch.cuda.is_available()}")
        checks_passed += 1
    except:
        print("   ❌ PyTorch not available")
    
    # Check 3: Source files
    print("3. Source Files:")
    required_files = [
        'src/gan_architecture.py',
        'src/gan_losses.py', 
        'src/gan_training.py',
        'src/data_loader.py'
    ]
    
    all_files_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} missing")
            all_files_exist = False
    
    if all_files_exist:
        checks_passed += 1
    
    # Check 4: Dataset
    print("4. Dataset:")
    if os.path.exists('data'):
        if os.path.exists('data/cifar-10-batches-py') or os.path.exists('data/cifar-10-python.tar.gz'):
            print("   ✅ CIFAR-10 dataset available")
            checks_passed += 1
        else:
            print("   ⚠️  Dataset will be downloaded during training")
            checks_passed += 1
    else:
        print("   ⚠️  Data directory will be created")
        checks_passed += 1
    
    # Check 5: Output directories
    print("5. Output Directories:")
    dirs_to_check = ['models', 'results']
    for dir_name in dirs_to_check:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
            print(f"   ✅ Created {dir_name}/ directory")
        else:
            print(f"   ✅ {dir_name}/ directory exists")
    checks_passed += 1
    
    # Check 6: Imports test
    print("6. Import Test:")
    try:
        sys.path.append('src')
        from gan_architecture import SteganoGenerator
        from gan_losses import SteganographyLoss
        from gan_training import SteganographyTrainer
        print("   ✅ All modules import successfully")
        checks_passed += 1
    except Exception as e:
        print(f"   ❌ Import error: {e}")
    
    # Check 7: Memory estimate
    print("7. Memory Check:")
    try:
        # Estimate memory usage
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   ✅ GPU memory: {gpu_memory:.1f} GB")
            if gpu_memory > 4:
                print("   ✅ Sufficient GPU memory for training")
            else:
                print("   ⚠️  Limited GPU memory - reduce batch size if needed")
        else:
            print("   ⚠️  Using CPU - training will be slower")
        checks_passed += 1
    except:
        print("   ⚠️  Could not check memory")
        checks_passed += 1
    
    # Check 8: Training script
    print("8. Training Script:")
    if os.path.exists('run_gan_training.py'):
        print("   ✅ Training script ready")
        checks_passed += 1
    else:
        print("   ❌ Training script missing")
    
    # Summary
    print("\n" + "=" * 40)
    print(f"📊 CHECKLIST RESULT: {checks_passed}/{total_checks} checks passed")
    
    if checks_passed == total_checks:
        print("🎉 ALL CHECKS PASSED!")
        print("🚀 Ready to start training!")
        print("\n📋 To begin training, run:")
        print("   python run_gan_training.py")
        print("\n⏱️  Estimated training time: 2-4 hours")
        print("🎯 Target: >30 dB PSNR with >50% capacity")
        return True
    elif checks_passed >= total_checks - 2:
        print("⚠️  MOSTLY READY - Minor issues detected")
        print("🚀 Can proceed with training")
        print("📋 Monitor for any issues during training")
        return True
    else:
        print("❌ SEVERAL ISSUES DETECTED")
        print("🔧 Please resolve issues before training")
        return False

if __name__ == "__main__":
    ready = run_checklist()
    if ready:
        print("\n🏁 READY TO TRAIN!")
    else:
        print("\n🛠️  FIX ISSUES FIRST")
