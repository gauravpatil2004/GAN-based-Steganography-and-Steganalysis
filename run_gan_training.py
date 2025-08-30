"""
Execute GAN Steganography Training

This script runs the complete GAN training for steganography with proper
configuration and monitoring.
"""

import torch
import torch.nn as nn
import sys
import os

# Add src to path
sys.path.append('src')

def main():
    print("🚀 STARTING GAN STEGANOGRAPHY TRAINING")
    print("=" * 50)
    
    # Check environment
    print("🔧 Environment Check:")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Training device: {device}")
    
    # Import components
    try:
        from gan_training import SteganographyTrainer
        print("✅ Training framework imported")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return
    
    # Training configuration
    config = {
        'batch_size': 16,
        'num_epochs': 50,
        'lr_g': 0.0002,      # Generator learning rate
        'lr_d': 0.0002,      # Discriminator learning rate  
        'lr_e': 0.0002,      # Extractor learning rate
        'device': device,
        'save_every': 10,
        'log_every': 100,
        'data_path': './data'
    }
    
    print(f"\n📋 Training Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Initialize trainer
    try:
        trainer = SteganographyTrainer(config)
        print("✅ Trainer initialized")
        
        # Create data loader
        from data_loader import create_data_loader
        dataloader = create_data_loader(
            batch_size=config['batch_size'], 
            data_path=config['data_path']
        )
        print("✅ Data loader created")
        
    except Exception as e:
        print(f"❌ Trainer initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Start training
    print(f"\n🏃‍♂️ Starting training for {config['num_epochs']} epochs...")
    print("   This will take some time. Monitor the progress!")
    
    try:
        history = trainer.train(dataloader, config['num_epochs'])
        
        print("\n🎉 Training completed successfully!")
        
        # Print final results
        final_epoch = len(history['cover_psnr']) - 1
        print(f"\n📊 Final Results (Epoch {final_epoch + 1}):")
        print(f"   Cover PSNR: {history['cover_psnr'][-1]:.2f} dB")
        print(f"   Secret PSNR: {history['secret_psnr'][-1]:.2f} dB")
        print(f"   Generator Loss: {history['gen_loss'][-1]:.4f}")
        print(f"   Discriminator Loss: {history['disc_loss'][-1]:.4f}")
        print(f"   Extractor Loss: {history['ext_loss'][-1]:.4f}")
        
        # Performance evaluation
        cover_psnr = history['cover_psnr'][-1]
        secret_psnr = history['secret_psnr'][-1]
        
        print(f"\n🎯 Performance Evaluation:")
        if cover_psnr > 30:
            print(f"   ✅ Cover Quality: Excellent ({cover_psnr:.1f} dB > 30 dB target)")
        elif cover_psnr > 25:
            print(f"   📈 Cover Quality: Good ({cover_psnr:.1f} dB, approaching target)")
        else:
            print(f"   📊 Cover Quality: Needs improvement ({cover_psnr:.1f} dB < 25 dB)")
        
        if secret_psnr > 20:
            print(f"   ✅ Secret Recovery: Good ({secret_psnr:.1f} dB)")
        else:
            print(f"   📈 Secret Recovery: Needs improvement ({secret_psnr:.1f} dB)")
        
        print(f"\n💾 Results saved to ./models/ directory")
        print(f"📊 Training plots saved to ./results/ directory")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Training interrupted by user")
        print("   Partial results may be saved in ./models/")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
