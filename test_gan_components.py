"""
Simple test to verify the GAN components work
"""

print("🔧 Testing GAN components...")

try:
    import torch
    print(f"✅ PyTorch {torch.__version__} loaded")
    
    import torch.nn as nn
    print("✅ Neural network modules loaded")
    
    import sys
    import os
    sys.path.append('src')
    
    from gan_architecture import SteganoGenerator, SteganoDiscriminator, SecretExtractor
    print("✅ GAN networks imported successfully")
    
    from gan_losses import SteganographyLoss, MetricsCalculator
    print("✅ Loss functions imported successfully")
    
    # Test network creation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Device: {device}")
    
    generator = SteganoGenerator().to(device)
    discriminator = SteganoDiscriminator().to(device)
    extractor = SecretExtractor().to(device)
    print("✅ Networks created successfully")
    
    # Test forward pass with dummy data
    batch_size = 2
    cover = torch.randn(batch_size, 3, 32, 32).to(device)
    secret = torch.randn(batch_size, 3, 32, 32).to(device)
    
    print("🔄 Testing forward passes...")
    
    with torch.no_grad():
        stego = generator(cover, secret)
        print(f"✅ Generator output shape: {stego.shape}")
        
        disc_out = discriminator(stego)
        print(f"✅ Discriminator output shape: {disc_out.shape}")
        
        extracted = extractor(stego)
        print(f"✅ Extractor output shape: {extracted.shape}")
    
    # Test loss calculation
    loss_fn = SteganographyLoss(device=device)
    
    disc_real = discriminator(cover)
    disc_fake = discriminator(stego.detach())
    
    disc_losses = loss_fn.discriminator_loss(disc_real, disc_fake)
    print(f"✅ Discriminator loss: {disc_losses['total']:.4f}")
    
    gen_losses = loss_fn.generator_loss(cover, secret, stego, extracted, disc_fake)
    print(f"✅ Generator loss: {gen_losses['total']:.4f}")
    
    ext_loss = loss_fn.extractor_loss(secret, extracted)
    print(f"✅ Extractor loss: {ext_loss:.4f}")
    
    # Test metrics
    metrics = MetricsCalculator.calculate_metrics(cover, stego, secret, extracted)
    print(f"✅ Cover PSNR: {metrics['cover_psnr']:.2f} dB")
    print(f"✅ Secret PSNR: {metrics['secret_psnr']:.2f} dB")
    
    print("\n🎉 All GAN components working correctly!")
    print("🚀 Ready for training!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
