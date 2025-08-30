"""
Run Text-in-Image Steganography Training

Main script to train GAN for hiding text in images.
Much faster and more practical than image-to-image steganography.
"""

import torch
import torch.nn as nn
import sys
import os
import time

# Add src to path
sys.path.append('src')

def main():
    print("🚀 STARTING TEXT-IN-IMAGE STEGANOGRAPHY TRAINING")
    print("=" * 60)
    
    # Check environment
    print("🔧 Environment Check:")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Training device: {device}")
    
    # Import training components
    try:
        from text_gan_training import TextSteganoTrainer
        print("✅ Text training framework imported")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Training configuration optimized for text
    config = {
        # Data parameters
        'batch_size': 32,           # Larger batch size for stable text training
        'max_text_length': 128,     # Support up to 128 characters
        'text_embed_dim': 128,      # Text embedding dimension
        'data_path': './data',
        
        # Training parameters
        'num_epochs': 30,           # Much fewer epochs needed for text
        'lr_g': 0.0001,            # Generator learning rate
        'lr_d': 0.0002,            # Discriminator learning rate  
        'lr_e': 0.0002,            # Extractor learning rate
        
        # Loss weights optimized for text steganography
        'loss_weights': {
            'adversarial': 1.0,      # Keep stego images realistic
            'reconstruction': 10.0,   # Preserve image quality
            'text_recovery': 100.0,   # MOST IMPORTANT: accurate text extraction
            'perceptual': 2.0,       # Maintain visual appeal
            'capacity': 0.5          # Efficient text embedding
        }
    }
    
    print(f"\n📋 Training Configuration:")
    for key, value in config.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for k, v in value.items():
                print(f"     {k}: {v}")
        else:
            print(f"   {key}: {value}")
    
    # Expected training time
    estimated_time = config['num_epochs'] * 2  # ~2 minutes per epoch
    print(f"\n⏰ Estimated training time: {estimated_time} minutes")
    print(f"   Much faster than image-to-image training! ⚡")
    
    # Initialize trainer
    try:
        print(f"\n🎯 Initializing Text Steganography Trainer...")
        trainer = TextSteganoTrainer(config)
        print("✅ Trainer initialized successfully")
        
    except Exception as e:
        print(f"❌ Trainer initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Start training
    print(f"\n🏃‍♂️ Starting training for {config['num_epochs']} epochs...")
    print("   Text steganography training in progress...")
    print("   Monitoring character accuracy and image quality!")
    
    start_time = time.time()
    
    try:
        history = trainer.train(config['num_epochs'])
        
        training_time = time.time() - start_time
        print(f"\n🎉 Training completed in {training_time/60:.1f} minutes!")
        
        # Print final results
        final_epoch = len(history['character_accuracy']) - 1
        print(f"\n📊 Final Results (Epoch {final_epoch + 1}):")
        print(f"   Character Accuracy: {history['character_accuracy'][-1]:.3f}")
        print(f"   Word Accuracy: {history['word_accuracy'][-1]:.3f}")
        print(f"   Cover PSNR: {history['cover_psnr'][-1]:.2f} dB")
        print(f"   Cover SSIM: {history['cover_ssim'][-1]:.4f}")
        print(f"   Generator Loss: {history['generator_loss'][-1]:.4f}")
        
        # Performance evaluation
        char_acc = history['character_accuracy'][-1]
        cover_psnr = history['cover_psnr'][-1]
        
        print(f"\n🎯 Performance Evaluation:")
        if char_acc > 0.99:
            print(f"   ✅ Character Accuracy: Excellent ({char_acc:.3f} > 0.99)")
        elif char_acc > 0.95:
            print(f"   📈 Character Accuracy: Very Good ({char_acc:.3f} > 0.95)")
        elif char_acc > 0.90:
            print(f"   📊 Character Accuracy: Good ({char_acc:.3f} > 0.90)")
        else:
            print(f"   📉 Character Accuracy: Needs improvement ({char_acc:.3f})")
        
        if cover_psnr > 35:
            print(f"   ✅ Image Quality: Excellent ({cover_psnr:.1f} dB > 35 dB)")
        elif cover_psnr > 30:
            print(f"   📈 Image Quality: Very Good ({cover_psnr:.1f} dB > 30 dB)")
        elif cover_psnr > 25:
            print(f"   📊 Image Quality: Good ({cover_psnr:.1f} dB > 25 dB)")
        else:
            print(f"   📉 Image Quality: Needs improvement ({cover_psnr:.1f} dB)")
        
        # Calculate text capacity
        capacity_chars = config['max_text_length']
        capacity_bits = capacity_chars * 8
        image_bits = 32 * 32 * 3 * 8  # CIFAR-10 total bits
        efficiency = (capacity_bits / image_bits) * 100
        
        print(f"\n📈 Text Hiding Capacity:")
        print(f"   Maximum text length: {capacity_chars} characters")
        print(f"   Bits per image: {capacity_bits} bits")
        print(f"   Embedding efficiency: {efficiency:.2f}% of image bits")
        
        print(f"\n💾 Results saved:")
        print(f"   📁 Models: ./models/best_text_model.pth")
        print(f"   📊 Training curves: ./results/training_curves.png")
        print(f"   🖼️ Sample results: ./results/text_samples_epoch_*.png")
        print(f"   📄 History: ./results/training_history.json")
        
        # Comparison with previous approach
        print(f"\n⚡ Advantages over Image-to-Image Steganography:")
        print(f"   🚄 Training Time: {training_time/60:.1f} min vs 2-4 hours")
        print(f"   📊 Clear Metrics: Character accuracy vs visual comparison")
        print(f"   🎯 Practical Use: Hide passwords, URLs, messages")
        print(f"   🔍 Easy Validation: Text extraction success is binary")
        print(f"   💼 Real Applications: Secure communication, watermarking")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Training interrupted by user")
        print("   Partial results may be saved in ./models/")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
