"""
Loss Functions for Text-in-Image Steganography

Specialized loss functions for training GAN to hide text in images.
Includes text reconstruction, adversarial, and quality preservation losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Tuple
import numpy as np
from skimage.metrics import structural_similarity as ssim


class TextSteganoLoss:
    """Complete loss function suite for text steganography."""
    
    def __init__(self, device: torch.device = None, weights: Dict[str, float] = None):
        self.device = device or torch.device('cpu')
        
        # Default loss weights
        self.weights = weights or {
            'adversarial': 1.0,      # Discriminator fooling
            'reconstruction': 10.0,   # Image quality preservation  
            'text_recovery': 100.0,   # Text extraction accuracy
            'perceptual': 1.0,       # Perceptual quality
            'capacity': 0.1          # Embedding efficiency
        }
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        
        # VGG for perceptual loss
        self.vgg = models.vgg16(pretrained=True).features[:16].to(device).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        print("🎯 TextSteganoLoss initialized")
        print(f"   Loss weights: {self.weights}")
    
    def adversarial_loss(self, disc_fake_output: torch.Tensor, target_real: bool = True) -> torch.Tensor:
        """Adversarial loss for generator."""
        target = torch.ones_like(disc_fake_output) if target_real else torch.zeros_like(disc_fake_output)
        return self.bce_loss(disc_fake_output, target)
    
    def discriminator_loss(self, disc_real: torch.Tensor, disc_fake: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Discriminator loss components."""
        real_target = torch.ones_like(disc_real)
        fake_target = torch.zeros_like(disc_fake)
        
        real_loss = self.bce_loss(disc_real, real_target)
        fake_loss = self.bce_loss(disc_fake, fake_target)
        
        total_loss = (real_loss + fake_loss) / 2
        
        return {
            'real': real_loss,
            'fake': fake_loss,
            'total': total_loss
        }
    
    def reconstruction_loss(self, cover_image: torch.Tensor, stego_image: torch.Tensor) -> torch.Tensor:
        """Image reconstruction quality loss."""
        return self.mse_loss(cover_image, stego_image)
    
    def text_recovery_loss(self, text_target: torch.Tensor, text_logits: torch.Tensor) -> torch.Tensor:
        """Text extraction accuracy loss."""
        # text_target: [B, L] with token indices
        # text_logits: [B, L, vocab_size]
        
        # Flatten for cross-entropy
        batch_size, seq_len, vocab_size = text_logits.shape
        text_logits_flat = text_logits.view(-1, vocab_size)
        text_target_flat = text_target.view(-1)
        
        return self.ce_loss(text_logits_flat, text_target_flat)
    
    def perceptual_loss(self, cover_image: torch.Tensor, stego_image: torch.Tensor) -> torch.Tensor:
        """VGG-based perceptual loss."""
        # Normalize to [0, 1] for VGG
        cover_norm = (cover_image + 1) / 2
        stego_norm = (stego_image + 1) / 2
        
        # Repeat single channel to 3 channels if needed
        if cover_norm.size(1) == 1:
            cover_norm = cover_norm.repeat(1, 3, 1, 1)
            stego_norm = stego_norm.repeat(1, 3, 1, 1)
        
        # Extract VGG features
        cover_features = self.vgg(cover_norm)
        stego_features = self.vgg(stego_norm)
        
        return self.mse_loss(cover_features, stego_features)
    
    def capacity_loss(self, text_embedding: torch.Tensor) -> torch.Tensor:
        """Encourage efficient use of embedding space."""
        # L2 norm of text embedding (encourage compact representations)
        return torch.mean(torch.norm(text_embedding, p=2, dim=1))
    
    def generator_loss(self, 
                      cover_image: torch.Tensor,
                      stego_image: torch.Tensor, 
                      text_target: torch.Tensor,
                      text_logits: torch.Tensor,
                      disc_fake_output: torch.Tensor,
                      text_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Complete generator loss."""
        
        # Individual loss components
        adv_loss = self.adversarial_loss(disc_fake_output, target_real=True)
        recon_loss = self.reconstruction_loss(cover_image, stego_image)
        text_loss = self.text_recovery_loss(text_target, text_logits)
        percep_loss = self.perceptual_loss(cover_image, stego_image)
        cap_loss = self.capacity_loss(text_embedding)
        
        # Weighted combination
        total_loss = (
            self.weights['adversarial'] * adv_loss +
            self.weights['reconstruction'] * recon_loss +
            self.weights['text_recovery'] * text_loss +
            self.weights['perceptual'] * percep_loss +
            self.weights['capacity'] * cap_loss
        )
        
        return {
            'adversarial': adv_loss,
            'reconstruction': recon_loss,
            'text_recovery': text_loss,
            'perceptual': percep_loss,
            'capacity': cap_loss,
            'total': total_loss
        }
    
    def extractor_loss(self, text_target: torch.Tensor, text_logits: torch.Tensor) -> torch.Tensor:
        """Standalone text extractor loss."""
        return self.text_recovery_loss(text_target, text_logits)


class TextMetricsCalculator:
    """Calculate text steganography metrics."""
    
    @staticmethod
    def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Calculate PSNR between two images."""
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * torch.log10(2.0 / torch.sqrt(mse)).item()  # Range [-1,1]
    
    @staticmethod
    def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Calculate SSIM between two images."""
        # Convert to numpy and adjust range to [0, 1]
        img1_np = ((img1.cpu().numpy() + 1) / 2).transpose(1, 2, 0)
        img2_np = ((img2.cpu().numpy() + 1) / 2).transpose(1, 2, 0)
        
        # Get image dimensions
        height, width = img1_np.shape[:2]
        
        # Use smaller window size for small images (CIFAR-10 is 32x32)
        win_size = min(7, min(height, width))
        if win_size % 2 == 0:  # Ensure odd window size
            win_size -= 1
        win_size = max(3, win_size)  # Minimum window size of 3
        
        if img1_np.shape[2] == 1:
            img1_np = img1_np.squeeze(2)
            img2_np = img2_np.squeeze(2)
            return ssim(img1_np, img2_np, data_range=1.0, win_size=win_size)
        else:
            return ssim(img1_np, img2_np, channel_axis=2, data_range=1.0, win_size=win_size)
    
    @staticmethod
    def calculate_character_accuracy(text_target: torch.Tensor, text_logits: torch.Tensor) -> float:
        """Calculate character-level accuracy."""
        text_pred = torch.argmax(text_logits, dim=-1)
        
        # Mask out padding tokens (0)
        mask = text_target != 0
        
        if mask.sum() == 0:
            return 1.0  # All padding
        
        correct = (text_pred == text_target) & mask
        accuracy = correct.sum().float() / mask.sum().float()
        return accuracy.item()
    
    @staticmethod
    def calculate_text_metrics(cover_batch: torch.Tensor,
                              stego_batch: torch.Tensor,
                              text_target: torch.Tensor,
                              text_logits: torch.Tensor) -> Dict[str, float]:
        """Calculate comprehensive text steganography metrics."""
        
        batch_size = cover_batch.size(0)
        
        # Image quality metrics
        psnr_values = []
        ssim_values = []
        
        for i in range(batch_size):
            psnr = TextMetricsCalculator.calculate_psnr(cover_batch[i], stego_batch[i])
            ssim_val = TextMetricsCalculator.calculate_ssim(cover_batch[i], stego_batch[i])
            
            psnr_values.append(psnr)
            ssim_values.append(ssim_val)
        
        # Text accuracy metrics
        char_accuracy = TextMetricsCalculator.calculate_character_accuracy(text_target, text_logits)
        
        # Word-level accuracy (approximate)
        text_pred = torch.argmax(text_logits, dim=-1)
        word_accuracy = 0.0
        for i in range(batch_size):
            # Find end token positions
            target_seq = text_target[i]
            pred_seq = text_pred[i]
            
            # Convert to words (split by space token if available)
            # For simplicity, use sequence-level accuracy
            seq_match = torch.equal(target_seq, pred_seq)
            word_accuracy += float(seq_match)
        
        word_accuracy /= batch_size
        
        return {
            'cover_psnr': np.mean(psnr_values),
            'cover_ssim': np.mean(ssim_values),
            'character_accuracy': char_accuracy,
            'word_accuracy': word_accuracy,
            'text_recovery_rate': char_accuracy  # Alias for compatibility
        }


def test_text_stegano_loss():
    """Test text steganography loss functions."""
    print("🧪 Testing Text Steganography Loss Functions")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create test data
    batch_size = 4
    vocab_size = 95
    seq_len = 32
    
    cover_images = torch.randn(batch_size, 3, 32, 32).to(device)
    stego_images = torch.randn(batch_size, 3, 32, 32).to(device)
    text_target = torch.randint(1, vocab_size, (batch_size, seq_len)).to(device)
    text_logits = torch.randn(batch_size, seq_len, vocab_size + 3).to(device)
    text_embedding = torch.randn(batch_size, 128).to(device)
    disc_output = torch.rand(batch_size, 1).to(device)
    
    print(f"📊 Test data shapes:")
    print(f"   Cover/Stego images: {cover_images.shape}")
    print(f"   Text target: {text_target.shape}")
    print(f"   Text logits: {text_logits.shape}")
    
    # Initialize loss function
    loss_fn = TextSteganoLoss(device=device)
    
    # Test generator loss
    print(f"\n🎨 Testing Generator Loss...")
    gen_losses = loss_fn.generator_loss(
        cover_images, stego_images, text_target, 
        text_logits, disc_output, text_embedding
    )
    
    for name, loss in gen_losses.items():
        print(f"   {name}: {loss.item():.4f}")
    
    # Test discriminator loss
    print(f"\n🔍 Testing Discriminator Loss...")
    disc_real = torch.rand(batch_size, 1).to(device)
    disc_fake = torch.rand(batch_size, 1).to(device)
    
    disc_losses = loss_fn.discriminator_loss(disc_real, disc_fake)
    for name, loss in disc_losses.items():
        print(f"   {name}: {loss.item():.4f}")
    
    # Test metrics
    print(f"\n📊 Testing Metrics...")
    metrics = TextMetricsCalculator.calculate_text_metrics(
        cover_images, stego_images, text_target, text_logits
    )
    
    for name, value in metrics.items():
        print(f"   {name}: {value:.4f}")
    
    print(f"\n✅ All loss functions working correctly!")


if __name__ == "__main__":
    test_text_stegano_loss()
