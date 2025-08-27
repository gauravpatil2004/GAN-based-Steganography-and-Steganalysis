"""
GAN Steganography Training Loop

Complete training implementation for GAN-based steganography.
Trains Generator, Discriminator, and Extractor networks simultaneously.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json

from gan_architecture import SteganoGenerator, SteganoDiscriminator, SecretExtractor
from gan_losses import SteganographyLoss, MetricsCalculator
from data_loader import create_data_loader


class SteganographyTrainer:
    """Complete training class for GAN-based steganography."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        
        # Initialize networks
        self.generator = SteganoGenerator().to(self.device)
        self.discriminator = SteganoDiscriminator().to(self.device)
        self.extractor = SecretExtractor().to(self.device)
        
        # Initialize loss function
        self.loss_fn = SteganographyLoss(device=self.device)
        
        # Initialize optimizers
        self.opt_g = optim.Adam(self.generator.parameters(), 
                               lr=config['lr_g'], betas=(0.5, 0.999))
        self.opt_d = optim.Adam(self.discriminator.parameters(), 
                               lr=config['lr_d'], betas=(0.5, 0.999))
        self.opt_e = optim.Adam(self.extractor.parameters(), 
                               lr=config['lr_e'], betas=(0.5, 0.999))
        
        # Training history
        self.history = {
            'generator_loss': [],
            'discriminator_loss': [],
            'extractor_loss': [],
            'cover_psnr': [],
            'secret_psnr': [],
            'cover_ssim': [],
            'secret_ssim': []
        }
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch."""
        self.generator.train()
        self.discriminator.train()
        self.extractor.train()
        
        epoch_metrics = {
            'gen_loss': 0, 'disc_loss': 0, 'ext_loss': 0,
            'cover_psnr': 0, 'secret_psnr': 0,
            'cover_ssim': 0, 'secret_ssim': 0
        }
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, (cover_batch, secret_batch) in enumerate(progress_bar):
            cover_batch = cover_batch.to(self.device)
            secret_batch = secret_batch.to(self.device)
            
            batch_size = cover_batch.size(0)
            
            # Normalize to [-1, 1] range
            cover_batch = (cover_batch * 2) - 1
            secret_batch = (secret_batch * 2) - 1
            
            # ===================
            # Train Discriminator
            # ===================
            self.opt_d.zero_grad()
            
            # Generate stego images
            with torch.no_grad():
                stego_batch = self.generator(cover_batch, secret_batch)
            
            # Discriminator predictions
            disc_real = self.discriminator(cover_batch)
            disc_fake = self.discriminator(stego_batch.detach())
            
            # Discriminator loss
            disc_losses = self.loss_fn.discriminator_loss(disc_real, disc_fake)
            disc_losses['total'].backward()
            self.opt_d.step()
            
            # =================
            # Train Generator & Extractor
            # =================
            self.opt_g.zero_grad()
            self.opt_e.zero_grad()
            
            # Generate stego images
            stego_batch = self.generator(cover_batch, secret_batch)
            
            # Extract secrets
            extracted_batch = self.extractor(stego_batch)
            
            # Generator loss (including extractor feedback)
            disc_pred_fake = self.discriminator(stego_batch)
            gen_losses = self.loss_fn.generator_loss(
                cover_batch, secret_batch, stego_batch, 
                extracted_batch, disc_pred_fake
            )
            
            # Extractor loss
            ext_loss = self.loss_fn.extractor_loss(secret_batch, extracted_batch)
            
            # Backward pass
            gen_losses['total'].backward(retain_graph=True)
            ext_loss.backward()
            
            self.opt_g.step()
            self.opt_e.step()
            
            # Calculate metrics
            metrics = MetricsCalculator.calculate_metrics(
                cover_batch, stego_batch, secret_batch, extracted_batch
            )
            
            # Update epoch metrics
            epoch_metrics['gen_loss'] += gen_losses['total'].item()
            epoch_metrics['disc_loss'] += disc_losses['total'].item()
            epoch_metrics['ext_loss'] += ext_loss.item()
            epoch_metrics['cover_psnr'] += metrics['cover_psnr']
            epoch_metrics['secret_psnr'] += metrics['secret_psnr']
            epoch_metrics['cover_ssim'] += metrics['cover_ssim']
            epoch_metrics['secret_ssim'] += metrics['secret_ssim']
            
            # Update progress bar
            progress_bar.set_postfix({
                'G_Loss': f"{gen_losses['total'].item():.3f}",
                'D_Loss': f"{disc_losses['total'].item():.3f}",
                'Cover_PSNR': f"{metrics['cover_psnr']:.1f}",
                'Secret_PSNR': f"{metrics['secret_psnr']:.1f}"
            })
        
        # Average metrics over epoch
        num_batches = len(dataloader)
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    def train(self, dataloader, num_epochs):
        """Complete training loop."""
        print(f"🚀 Starting GAN Steganography Training")
        print(f"📊 Training for {num_epochs} epochs")
        print(f"🎯 Target: >30 dB PSNR with >50% capacity")
        print("=" * 60)
        
        best_cover_psnr = 0
        
        for epoch in range(num_epochs):
            epoch_metrics = self.train_epoch(dataloader, epoch)
            
            # Save metrics
            self.history['generator_loss'].append(epoch_metrics['gen_loss'])
            self.history['discriminator_loss'].append(epoch_metrics['disc_loss'])
            self.history['extractor_loss'].append(epoch_metrics['ext_loss'])
            self.history['cover_psnr'].append(epoch_metrics['cover_psnr'])
            self.history['secret_psnr'].append(epoch_metrics['secret_psnr'])
            self.history['cover_ssim'].append(epoch_metrics['cover_ssim'])
            self.history['secret_ssim'].append(epoch_metrics['secret_ssim'])
            
            # Print epoch summary
            print(f"\n📊 Epoch {epoch+1}/{num_epochs} Summary:")
            print(f"   Generator Loss: {epoch_metrics['gen_loss']:.4f}")
            print(f"   Discriminator Loss: {epoch_metrics['disc_loss']:.4f}")
            print(f"   Extractor Loss: {epoch_metrics['ext_loss']:.4f}")
            print(f"   Cover PSNR: {epoch_metrics['cover_psnr']:.2f} dB")
            print(f"   Secret PSNR: {epoch_metrics['secret_psnr']:.2f} dB")
            print(f"   Cover SSIM: {epoch_metrics['cover_ssim']:.4f}")
            print(f"   Secret SSIM: {epoch_metrics['secret_ssim']:.4f}")
            
            # Save best model
            if epoch_metrics['cover_psnr'] > best_cover_psnr:
                best_cover_psnr = epoch_metrics['cover_psnr']
                self.save_models(f"best_model_epoch_{epoch+1}")
                print(f"   🏆 New best cover PSNR: {best_cover_psnr:.2f} dB")
            
            # Generate sample every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.generate_samples(f"samples_epoch_{epoch+1}.png")
        
        print(f"\n🎉 Training completed!")
        print(f"🏆 Best cover PSNR achieved: {best_cover_psnr:.2f} dB")
        
        return self.history
    
    def save_models(self, filename_prefix):
        """Save trained models."""
        os.makedirs("models", exist_ok=True)
        
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'extractor': self.extractor.state_dict(),
            'history': self.history
        }, f"models/{filename_prefix}.pth")
    
    def generate_samples(self, filename):
        """Generate sample results."""
        self.generator.eval()
        self.extractor.eval()
        
        with torch.no_grad():
            # Load sample data
            from src.lsb_stego import load_cifar10_sample
            
            images = load_cifar10_sample(4)
            covers = []
            secrets = []
            
            for i in range(2):
                cover = transforms.ToTensor()(images[i]).unsqueeze(0)
                secret = transforms.ToTensor()(images[i+2]).unsqueeze(0)
                covers.append((cover * 2) - 1)  # Normalize to [-1, 1]
                secrets.append((secret * 2) - 1)
            
            cover_batch = torch.cat(covers).to(self.device)
            secret_batch = torch.cat(secrets).to(self.device)
            
            # Generate stego and extract secrets
            stego_batch = self.generator(cover_batch, secret_batch)
            extracted_batch = self.extractor(stego_batch)
            
            # Convert back to [0, 1] for display
            cover_batch = (cover_batch + 1) / 2
            secret_batch = (secret_batch + 1) / 2
            stego_batch = (stego_batch + 1) / 2
            extracted_batch = (extracted_batch + 1) / 2
            
            # Create visualization
            fig, axes = plt.subplots(4, 2, figsize=(8, 16))
            
            for i in range(2):
                # Cover
                axes[0, i].imshow(cover_batch[i].cpu().permute(1, 2, 0))
                axes[0, i].set_title(f'Cover {i+1}')
                axes[0, i].axis('off')
                
                # Secret
                axes[1, i].imshow(secret_batch[i].cpu().permute(1, 2, 0))
                axes[1, i].set_title(f'Secret {i+1}')
                axes[1, i].axis('off')
                
                # Stego
                axes[2, i].imshow(stego_batch[i].cpu().permute(1, 2, 0))
                axes[2, i].set_title(f'Stego {i+1}')
                axes[2, i].axis('off')
                
                # Extracted
                axes[3, i].imshow(extracted_batch[i].cpu().permute(1, 2, 0))
                axes[3, i].set_title(f'Extracted {i+1}')
                axes[3, i].axis('off')
            
            plt.tight_layout()
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
        
        self.generator.train()
        self.extractor.train()


def create_training_config():
    """Create default training configuration."""
    return {
        'batch_size': 32,
        'num_epochs': 50,
        'lr_g': 0.0002,       # Generator learning rate
        'lr_d': 0.0002,       # Discriminator learning rate  
        'lr_e': 0.0002,       # Extractor learning rate
        'image_size': 32,
        'cover_dir': 'data/cifar-10-batches-py',
        'secret_dir': 'data/cifar-10-batches-py'
    }


def main():
    """Main training function."""
    print("🎯 GAN STEGANOGRAPHY TRAINING")
    print("=" * 50)
    
    # Configuration
    config = create_training_config()
    print("📋 Training Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Create data loader
    print(f"\n📥 Loading CIFAR-10 dataset...")
    try:
        # Use CIFAR-10 directly
        import torchvision.datasets as datasets
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        dataset = datasets.CIFAR10(
            root='./data', 
            train=True, 
            download=False,
            transform=transform
        )
        
        # Custom collate function for steganography
        def stego_collate_fn(batch):
            # Split batch into covers and secrets randomly
            covers = []
            secrets = []
            
            for i, (img, _) in enumerate(batch):
                if i < len(batch) // 2:
                    covers.append(img)
                else:
                    secrets.append(img)
            
            # If odd number, duplicate last secret
            while len(secrets) < len(covers):
                secrets.append(secrets[-1])
            
            return torch.stack(covers), torch.stack(secrets)
        
        dataloader = DataLoader(
            dataset, 
            batch_size=config['batch_size'],
            shuffle=True,
            collate_fn=stego_collate_fn
        )
        
        print(f"✅ Dataset loaded: {len(dataset)} images")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Initialize trainer
    trainer = SteganographyTrainer(config)
    
    # Start training
    history = trainer.train(dataloader, config['num_epochs'])
    
    # Plot training history
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(history['generator_loss'], label='Generator')
    plt.plot(history['discriminator_loss'], label='Discriminator')
    plt.plot(history['extractor_loss'], label='Extractor')
    plt.title('Training Losses')
    plt.legend()
    
    plt.subplot(2, 3, 2)
    plt.plot(history['cover_psnr'], label='Cover PSNR')
    plt.plot(history['secret_psnr'], label='Secret PSNR')
    plt.title('PSNR Over Time')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    
    plt.subplot(2, 3, 3)
    plt.plot(history['cover_ssim'], label='Cover SSIM')
    plt.plot(history['secret_ssim'], label='Secret SSIM')
    plt.title('SSIM Over Time')
    plt.ylabel('SSIM')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n🎉 Training completed successfully!")
    print(f"📁 Generated files:")
    print(f"   - models/best_model_*.pth")
    print(f"   - training_history.png")
    print(f"   - samples_epoch_*.png")


if __name__ == "__main__":
    main()
