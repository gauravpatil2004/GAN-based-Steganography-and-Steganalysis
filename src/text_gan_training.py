"""
Training Framework for Text-in-Image Steganography

Complete training pipeline for GAN-based text hiding in images.
Much faster than image-to-image training with clear text metrics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
import time
from typing import Dict, List, Tuple

from text_gan_architecture import TextSteganoGenerator, TextSteganoDiscriminator, TextExtractor
from text_gan_losses import TextSteganoLoss, TextMetricsCalculator
from text_data_loader import create_text_data_loader
from text_processor import TextProcessor


class TextSteganoTrainer:
    """Complete training framework for text steganography."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🚀 TextSteganoTrainer initialized")
        print(f"   Device: {self.device}")
        print(f"   Config: {config}")
        
        # Create data loader
        self.dataloader, self.text_processor, self.text_embedding = create_text_data_loader(
            batch_size=config['batch_size'],
            max_text_length=config['max_text_length'],
            data_path=config['data_path']
        )
        
        # Initialize networks
        self.generator = TextSteganoGenerator(
            text_embed_dim=config['text_embed_dim']
        ).to(self.device)
        
        self.discriminator = TextSteganoDiscriminator().to(self.device)
        
        self.extractor = TextExtractor(
            vocab_size=self.text_processor.vocab_size,
            max_text_length=config['max_text_length']
        ).to(self.device)
        
        self.text_embedding = self.text_embedding.to(self.device)
        
        # Initialize loss function
        self.loss_fn = TextSteganoLoss(
            device=self.device,
            weights=config.get('loss_weights', None)
        )
        
        # Initialize optimizers
        self.opt_g = optim.Adam(
            list(self.generator.parameters()) + list(self.text_embedding.parameters()),
            lr=config['lr_g'], 
            betas=(0.5, 0.999)
        )
        self.opt_d = optim.Adam(
            self.discriminator.parameters(),
            lr=config['lr_d'], 
            betas=(0.5, 0.999)
        )
        self.opt_e = optim.Adam(
            self.extractor.parameters(),
            lr=config['lr_e'], 
            betas=(0.5, 0.999)
        )
        
        # Training history
        self.history = {
            'gen_loss': [],
            'disc_loss': [],
            'ext_loss': [],
            'cover_psnr': [],
            'character_accuracy': [],
            'word_accuracy': [],
            'cover_ssim': []
        }
        
        # Create output directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
        print(f"✅ All components initialized successfully!")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.generator.train()
        self.discriminator.train()
        self.extractor.train()
        self.text_embedding.train()
        
        epoch_metrics = {
            'gen_loss': 0, 'disc_loss': 0, 'ext_loss': 0,
            'cover_psnr': 0, 'character_accuracy': 0,
            'word_accuracy': 0, 'cover_ssim': 0
        }
        
        progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}")
        batch_count = 0
        
        for batch_idx, (cover_batch, text_batch) in enumerate(progress_bar):
            cover_batch = cover_batch.to(self.device)
            text_batch = text_batch.to(self.device)
            
            batch_size = cover_batch.size(0)
            
            # Normalize images to [-1, 1]
            cover_batch = (cover_batch * 2) - 1
            
            # Generate text embeddings
            text_embeddings = self.text_embedding(text_batch)
            
            # ===================
            # Train Discriminator
            # ===================
            self.opt_d.zero_grad()
            
            # Generate stego images
            with torch.no_grad():
                stego_batch = self.generator(cover_batch, text_embeddings)
            
            # Discriminator predictions
            disc_real = self.discriminator(cover_batch)
            disc_fake = self.discriminator(stego_batch.detach())
            
            # Discriminator loss
            disc_losses = self.loss_fn.discriminator_loss(disc_real, disc_fake)
            disc_losses['total'].backward()
            self.opt_d.step()
            
            # =================
            # Train Generator & Text Embedding
            # =================
            self.opt_g.zero_grad()
            
            # Generate stego images
            stego_batch = self.generator(cover_batch, text_embeddings)
            
            # Discriminator prediction on fake
            disc_pred_fake = self.discriminator(stego_batch)
            
            # Extract text from stego
            extracted_logits = self.extractor(stego_batch)
            
            # Generator loss
            gen_losses = self.loss_fn.generator_loss(
                cover_batch, stego_batch, text_batch,
                extracted_logits, disc_pred_fake, text_embeddings
            )
            
            gen_losses['total'].backward(retain_graph=True)
            self.opt_g.step()
            
            # =================
            # Train Extractor
            # =================
            self.opt_e.zero_grad()
            
            # Extract text from stego (detached)
            extracted_logits = self.extractor(stego_batch.detach())
            
            # Extractor loss
            ext_loss = self.loss_fn.extractor_loss(text_batch, extracted_logits)
            ext_loss.backward()
            self.opt_e.step()
            
            # =================
            # Calculate Metrics
            # =================
            with torch.no_grad():
                metrics = TextMetricsCalculator.calculate_text_metrics(
                    cover_batch, stego_batch, text_batch, extracted_logits
                )
                
                # Accumulate metrics
                epoch_metrics['gen_loss'] += gen_losses['total'].item()
                epoch_metrics['disc_loss'] += disc_losses['total'].item()
                epoch_metrics['ext_loss'] += ext_loss.item()
                epoch_metrics['cover_psnr'] += metrics['cover_psnr']
                epoch_metrics['character_accuracy'] += metrics['character_accuracy']
                epoch_metrics['word_accuracy'] += metrics['word_accuracy']
                epoch_metrics['cover_ssim'] += metrics['cover_ssim']
                
                batch_count += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'G_Loss': f"{gen_losses['total'].item():.3f}",
                    'D_Loss': f"{disc_losses['total'].item():.3f}",
                    'Char_Acc': f"{metrics['character_accuracy']:.3f}",
                    'PSNR': f"{metrics['cover_psnr']:.1f}"
                })
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= batch_count
        
        return epoch_metrics
    
    def train(self, num_epochs: int) -> Dict[str, List[float]]:
        """Complete training loop."""
        print(f"🚀 Starting Text Steganography Training")
        print(f"📊 Training for {num_epochs} epochs")
        print(f"🎯 Target: >99% character accuracy with >35 dB PSNR")
        print("=" * 60)
        
        best_char_accuracy = 0
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Train one epoch
            epoch_metrics = self.train_epoch(epoch)
            
            # Save metrics
            for key, value in epoch_metrics.items():
                self.history[key].append(value)
            
            # Calculate time
            epoch_time = time.time() - epoch_start
            total_time = time.time() - start_time
            
            # Print epoch summary
            print(f"\n📊 Epoch {epoch+1}/{num_epochs} Summary ({epoch_time:.1f}s):")
            print(f"   Generator Loss: {epoch_metrics['gen_loss']:.4f}")
            print(f"   Discriminator Loss: {epoch_metrics['disc_loss']:.4f}")
            print(f"   Extractor Loss: {epoch_metrics['ext_loss']:.4f}")
            print(f"   Cover PSNR: {epoch_metrics['cover_psnr']:.2f} dB")
            print(f"   Character Accuracy: {epoch_metrics['character_accuracy']:.3f}")
            print(f"   Word Accuracy: {epoch_metrics['word_accuracy']:.3f}")
            print(f"   Cover SSIM: {epoch_metrics['cover_ssim']:.4f}")
            
            # Save best model
            if epoch_metrics['character_accuracy'] > best_char_accuracy:
                best_char_accuracy = epoch_metrics['character_accuracy']
                self.save_models(f"best_text_model")
                print(f"   🏆 New best character accuracy: {best_char_accuracy:.3f}")
            
            # Generate samples every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.generate_samples(f"text_samples_epoch_{epoch+1}")
            
            # Early stopping if accuracy is very high
            if epoch_metrics['character_accuracy'] > 0.99 and epoch_metrics['cover_psnr'] > 30:
                print(f"🎉 Early stopping: Excellent performance achieved!")
                break
        
        total_training_time = time.time() - start_time
        print(f"\n🎉 Training completed in {total_training_time/60:.1f} minutes!")
        
        # Save final training history
        self.save_training_history()
        
        return self.history
    
    def save_models(self, filename_prefix: str):
        """Save all model checkpoints."""
        checkpoint = {
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'extractor': self.extractor.state_dict(),
            'text_embedding': self.text_embedding.state_dict(),
            'opt_g': self.opt_g.state_dict(),
            'opt_d': self.opt_d.state_dict(),
            'opt_e': self.opt_e.state_dict(),
            'config': self.config,
            'history': self.history
        }
        
        torch.save(checkpoint, f"models/{filename_prefix}.pth")
    
    def load_models(self, filepath: str):
        """Load model checkpoints."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.extractor.load_state_dict(checkpoint['extractor'])
        self.text_embedding.load_state_dict(checkpoint['text_embedding'])
        
        print(f"✅ Models loaded from {filepath}")
    
    def generate_samples(self, filename_prefix: str):
        """Generate sample results for visualization."""
        self.generator.eval()
        self.extractor.eval()
        self.text_embedding.eval()
        
        with torch.no_grad():
            # Get one batch for visualization
            cover_batch, text_batch = next(iter(self.dataloader))
            cover_batch = cover_batch[:4].to(self.device)  # Take 4 samples
            text_batch = text_batch[:4].to(self.device)
            
            # Normalize images
            cover_batch = (cover_batch * 2) - 1
            
            # Generate embeddings and stego images
            text_embeddings = self.text_embedding(text_batch)
            stego_batch = self.generator(cover_batch, text_embeddings)
            extracted_logits = self.extractor(stego_batch)
            
            # Convert back to [0, 1] for display
            cover_display = (cover_batch + 1) / 2
            stego_display = (stego_batch + 1) / 2
            
            # Extract text
            extracted_text = []
            original_text = []
            
            for i in range(4):
                orig = self.text_processor.decode_text(text_batch[i])
                pred_indices = torch.argmax(extracted_logits[i], dim=-1)
                pred = self.text_processor.sequence_to_text(pred_indices.cpu().tolist())
                
                original_text.append(orig)
                extracted_text.append(pred)
            
            # Create visualization
            fig, axes = plt.subplots(3, 4, figsize=(16, 12))
            
            for i in range(4):
                # Cover image
                axes[0, i].imshow(cover_display[i].cpu().permute(1, 2, 0))
                axes[0, i].set_title(f'Cover {i+1}')
                axes[0, i].axis('off')
                
                # Stego image
                axes[1, i].imshow(stego_display[i].cpu().permute(1, 2, 0))
                axes[1, i].set_title(f'Stego {i+1}')
                axes[1, i].axis('off')
                
                # Text comparison
                axes[2, i].text(0.1, 0.7, f"Original:\n{original_text[i][:50]}...", 
                               fontsize=8, verticalalignment='top', wrap=True)
                axes[2, i].text(0.1, 0.3, f"Extracted:\n{extracted_text[i][:50]}...", 
                               fontsize=8, verticalalignment='top', wrap=True, color='red')
                axes[2, i].set_xlim(0, 1)
                axes[2, i].set_ylim(0, 1)
                axes[2, i].axis('off')
                axes[2, i].set_title(f'Text {i+1}')
            
            plt.suptitle('Text Steganography Results', fontsize=16)
            plt.tight_layout()
            plt.savefig(f'results/{filename_prefix}.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    def save_training_history(self):
        """Save training history as JSON and plots."""
        # Convert numpy/torch values to Python floats for JSON serialization
        json_history = {}
        for key, values in self.history.items():
            json_history[key] = [float(val) for val in values]
        
        # Save as JSON
        with open('results/training_history.json', 'w') as f:
            json.dump(json_history, f, indent=2)
        
        # Create plots
        epochs = range(1, len(self.history['gen_loss']) + 1)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Loss plots
        axes[0, 0].plot(epochs, self.history['gen_loss'], label='Generator', color='blue')
        axes[0, 0].plot(epochs, self.history['disc_loss'], label='Discriminator', color='red')
        axes[0, 0].plot(epochs, self.history['ext_loss'], label='Extractor', color='green')
        axes[0, 0].set_title('Training Losses')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # PSNR
        axes[0, 1].plot(epochs, self.history['cover_psnr'], color='purple')
        axes[0, 1].set_title('Cover Image PSNR')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('PSNR (dB)')
        axes[0, 1].grid(True)
        
        # Character accuracy
        axes[0, 2].plot(epochs, self.history['character_accuracy'], color='orange')
        axes[0, 2].set_title('Character Accuracy')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Accuracy')
        axes[0, 2].grid(True)
        
        # Word accuracy
        axes[1, 0].plot(epochs, self.history['word_accuracy'], color='brown')
        axes[1, 0].set_title('Word Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].grid(True)
        
        # SSIM
        axes[1, 1].plot(epochs, self.history['cover_ssim'], color='pink')
        axes[1, 1].set_title('Cover Image SSIM')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('SSIM')
        axes[1, 1].grid(True)
        
        # Combined quality plot
        axes[1, 2].plot(epochs, self.history['character_accuracy'], label='Char Accuracy', color='green')
        axes[1, 2].plot(epochs, np.array(self.history['cover_psnr'])/50, label='PSNR/50', color='blue')
        axes[1, 2].set_title('Quality Metrics')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Normalized Value')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        plt.suptitle('Text Steganography Training Progress', fontsize=16)
        plt.tight_layout()
        plt.savefig('results/training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()


def test_text_trainer():
    """Test the text steganography trainer."""
    print("🧪 Testing Text Steganography Trainer")
    print("=" * 50)
    
    # Quick test configuration
    config = {
        'batch_size': 8,
        'max_text_length': 64,
        'text_embed_dim': 128,
        'lr_g': 0.0002,
        'lr_d': 0.0002,
        'lr_e': 0.0002,
        'data_path': './data',
        'loss_weights': {
            'adversarial': 1.0,
            'reconstruction': 5.0,
            'text_recovery': 50.0,
            'perceptual': 1.0,
            'capacity': 0.1
        }
    }
    
    try:
        trainer = TextSteganoTrainer(config)
        print("✅ Trainer initialized successfully!")
        
        # Test one epoch
        print("\n🏃‍♂️ Testing one training epoch...")
        metrics = trainer.train_epoch(0)
        
        print(f"\n📊 Test Results:")
        for key, value in metrics.items():
            print(f"   {key}: {value:.4f}")
        
        print(f"\n✅ Text steganography trainer working correctly!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_text_trainer()
