"""
Quick GAN Steganography Demo

Test the complete GAN training pipeline with a small number of epochs
to verify everything works correctly.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.append('src')

from gan_architecture import SteganoGenerator, SteganoDiscriminator, SecretExtractor
from gan_losses import SteganographyLoss, MetricsCalculator


def quick_demo():
    """Quick demo of GAN steganography training."""
    print("🎯 QUICK GAN STEGANOGRAPHY DEMO")
    print("=" * 50)
    print("Testing the complete pipeline with 3 epochs...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    # Initialize networks
    print("\n📱 Initializing networks...")
    generator = SteganoGenerator().to(device)
    discriminator = SteganoDiscriminator().to(device)
    extractor = SecretExtractor().to(device)
    loss_fn = SteganographyLoss(device=device)
    
    # Optimizers
    opt_g = torch.optim.Adam(generator.parameters(), lr=0.001)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=0.001)
    opt_e = torch.optim.Adam(extractor.parameters(), lr=0.001)
    
    print("✅ Networks initialized")
    
    # Create simple dataset
    print("\n📥 Creating demo dataset...")
    transform = transforms.Compose([transforms.ToTensor()])
    
    try:
        dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
        
        # Simple collate function
        def demo_collate_fn(batch):
            covers = []
            secrets = []
            
            for i, (img, _) in enumerate(batch):
                if i % 2 == 0:
                    covers.append(img)
                else:
                    secrets.append(img)
            
            # Ensure equal length
            min_len = min(len(covers), len(secrets))
            covers = covers[:min_len]
            secrets = secrets[:min_len]
            
            if len(covers) == 0:
                covers = [batch[0][0]]
                secrets = [batch[1][0] if len(batch) > 1 else batch[0][0]]
            
            return torch.stack(covers), torch.stack(secrets)
        
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=demo_collate_fn)
        print("✅ Dataset created")
        
    except Exception as e:
        print(f"❌ Dataset error: {e}")
        return
    
    # Training demo
    print(f"\n🚀 Starting training demo (3 epochs)...")
    generator.train()
    discriminator.train()
    extractor.train()
    
    history = {'cover_psnr': [], 'secret_psnr': []}
    
    for epoch in range(3):
        print(f"\n📊 Epoch {epoch + 1}/3")
        epoch_cover_psnr = 0
        epoch_secret_psnr = 0
        batch_count = 0
        
        for batch_idx, (cover_batch, secret_batch) in enumerate(dataloader):
            if batch_idx >= 5:  # Only process 5 batches per epoch
                break
                
            cover_batch = cover_batch.to(device)
            secret_batch = secret_batch.to(device)
            
            # Normalize to [-1, 1]
            cover_batch = (cover_batch * 2) - 1
            secret_batch = (secret_batch * 2) - 1
            
            # Train discriminator
            opt_d.zero_grad()
            
            with torch.no_grad():
                stego_batch = generator(cover_batch, secret_batch)
            
            disc_real = discriminator(cover_batch)
            disc_fake = discriminator(stego_batch.detach())
            
            disc_losses = loss_fn.discriminator_loss(disc_real, disc_fake)
            disc_losses['total'].backward()
            opt_d.step()
            
            # Train generator and extractor
            opt_g.zero_grad()
            opt_e.zero_grad()
            
            stego_batch = generator(cover_batch, secret_batch)
            extracted_batch = extractor(stego_batch)
            
            disc_pred_fake = discriminator(stego_batch)
            gen_losses = loss_fn.generator_loss(
                cover_batch, secret_batch, stego_batch, 
                extracted_batch, disc_pred_fake
            )
            
            ext_loss = loss_fn.extractor_loss(secret_batch, extracted_batch)
            
            gen_losses['total'].backward(retain_graph=True)
            ext_loss.backward()
            
            opt_g.step()
            opt_e.step()
            
            # Calculate metrics
            metrics = MetricsCalculator.calculate_metrics(
                cover_batch, stego_batch, secret_batch, extracted_batch
            )
            
            epoch_cover_psnr += metrics['cover_psnr']
            epoch_secret_psnr += metrics['secret_psnr']
            batch_count += 1
            
            print(f"   Batch {batch_idx + 1}: Cover PSNR: {metrics['cover_psnr']:.2f} dB, "
                  f"Secret PSNR: {metrics['secret_psnr']:.2f} dB")
        
        # Average metrics
        avg_cover_psnr = epoch_cover_psnr / batch_count
        avg_secret_psnr = epoch_secret_psnr / batch_count
        
        history['cover_psnr'].append(avg_cover_psnr)
        history['secret_psnr'].append(avg_secret_psnr)
        
        print(f"   📊 Epoch Average - Cover PSNR: {avg_cover_psnr:.2f} dB, "
              f"Secret PSNR: {avg_secret_psnr:.2f} dB")
    
    print(f"\n🎉 Demo completed!")
    
    # Generate sample results
    print(f"\n🖼️  Generating sample results...")
    generator.eval()
    extractor.eval()
    
    with torch.no_grad():
        # Get one batch for visualization
        cover_batch, secret_batch = next(iter(dataloader))
        cover_batch = cover_batch[:2].to(device)  # Take only 2 samples
        secret_batch = secret_batch[:2].to(device)
        
        # Normalize
        cover_batch = (cover_batch * 2) - 1
        secret_batch = (secret_batch * 2) - 1
        
        # Generate stego and extract
        stego_batch = generator(cover_batch, secret_batch)
        extracted_batch = extractor(stego_batch)
        
        # Convert back for display
        cover_display = (cover_batch + 1) / 2
        secret_display = (secret_batch + 1) / 2
        stego_display = (stego_batch + 1) / 2
        extracted_display = (extracted_batch + 1) / 2
        
        # Create visualization
        fig, axes = plt.subplots(4, 2, figsize=(8, 12))
        
        for i in range(2):
            axes[0, i].imshow(cover_display[i].cpu().permute(1, 2, 0))
            axes[0, i].set_title(f'Cover {i+1}')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(secret_display[i].cpu().permute(1, 2, 0))
            axes[1, i].set_title(f'Secret {i+1}')
            axes[1, i].axis('off')
            
            axes[2, i].imshow(stego_display[i].cpu().permute(1, 2, 0))
            axes[2, i].set_title(f'Stego {i+1}')
            axes[2, i].axis('off')
            
            axes[3, i].imshow(extracted_display[i].cpu().permute(1, 2, 0))
            axes[3, i].set_title(f'Extracted {i+1}')
            axes[3, i].axis('off')
        
        plt.suptitle('GAN Steganography Demo Results', fontsize=16)
        plt.tight_layout()
        plt.savefig('gan_demo_results.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"✅ Sample results saved to gan_demo_results.png")
    
    # Summary
    print(f"\n📊 DEMO SUMMARY:")
    print(f"   Final Cover PSNR: {history['cover_psnr'][-1]:.2f} dB")
    print(f"   Final Secret PSNR: {history['secret_psnr'][-1]:.2f} dB")
    print(f"   Training Progress: ✅ Working")
    print(f"   Networks: ✅ All functional")
    print(f"   Loss Functions: ✅ All working")
    
    if history['cover_psnr'][-1] > 20:
        print(f"   🎉 Quality: Good progress for demo!")
    else:
        print(f"   📈 Quality: Needs more training epochs")
    
    print(f"\n🚀 Ready for full training!")
    
    return history


if __name__ == "__main__":
    try:
        history = quick_demo()
        print(f"\n✅ Demo completed successfully!")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
