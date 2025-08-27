"""
GAN Steganography Loss Functions

Multiple loss functions to train the GAN for high-quality steganography:
1. Adversarial Loss - Make stego images undetectable
2. Reconstruction Loss - Preserve cover image quality  
3. Secret Recovery Loss - Ensure secret can be extracted
4. Perceptual Loss - Maintain visual similarity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SteganographyLoss(nn.Module):
    """
    Combined loss function for GAN-based steganography training.
    """
    
    def __init__(self, device='cpu'):
        super(SteganographyLoss, self).__init__()
        self.device = device
        
        # Loss components
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCELoss()
        
        # Perceptual loss using pre-trained VGG
        self.vgg = self._load_vgg().to(device)
        
        # Loss weights
        self.lambda_adv = 1.0      # Adversarial loss weight
        self.lambda_cover = 10.0   # Cover reconstruction weight
        self.lambda_secret = 10.0  # Secret recovery weight
        self.lambda_percep = 1.0   # Perceptual loss weight
    
    def _load_vgg(self):
        """Load pre-trained VGG for perceptual loss."""
        vgg = models.vgg16(pretrained=True).features[:16]  # Up to conv3_3
        vgg.eval()
        for param in vgg.parameters():
            param.requires_grad = False
        return vgg
    
    def perceptual_loss(self, pred, target):
        """Calculate perceptual loss using VGG features."""
        # Resize to 224x224 for VGG (CIFAR-10 is 32x32)
        pred_resized = F.interpolate(pred, size=(224, 224), mode='bilinear', align_corners=False)
        target_resized = F.interpolate(target, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Get VGG features
        pred_features = self.vgg(pred_resized)
        target_features = self.vgg(target_resized)
        
        return self.mse_loss(pred_features, target_features)
    
    def generator_loss(self, cover, secret, stego, extracted_secret, disc_pred_fake):
        """
        Calculate generator loss.
        
        Args:
            cover: Original cover images
            secret: Original secret images
            stego: Generated stego images
            extracted_secret: Extracted secret from stego
            disc_pred_fake: Discriminator prediction on stego images
        
        Returns:
            dict: Loss components and total loss
        """
        # Adversarial loss (fool discriminator)
        adv_loss = self.bce_loss(disc_pred_fake, torch.ones_like(disc_pred_fake))
        
        # Cover reconstruction loss (stego should look like cover)
        cover_loss = self.l1_loss(stego, cover)
        
        # Secret recovery loss (extracted secret should match original)
        secret_loss = self.l1_loss(extracted_secret, secret)
        
        # Perceptual loss (maintain visual similarity)
        percep_loss = self.perceptual_loss(stego, cover)
        
        # Total generator loss
        total_loss = (self.lambda_adv * adv_loss + 
                     self.lambda_cover * cover_loss + 
                     self.lambda_secret * secret_loss + 
                     self.lambda_percep * percep_loss)
        
        return {
            'total': total_loss,
            'adversarial': adv_loss,
            'cover_reconstruction': cover_loss,
            'secret_recovery': secret_loss,
            'perceptual': percep_loss
        }
    
    def discriminator_loss(self, disc_pred_real, disc_pred_fake):
        """
        Calculate discriminator loss.
        
        Args:
            disc_pred_real: Discriminator predictions on real images
            disc_pred_fake: Discriminator predictions on stego images
        
        Returns:
            dict: Loss components and total loss
        """
        # Real images should be classified as real (1)
        real_loss = self.bce_loss(disc_pred_real, torch.ones_like(disc_pred_real))
        
        # Fake images should be classified as fake (0)
        fake_loss = self.bce_loss(disc_pred_fake, torch.zeros_like(disc_pred_fake))
        
        # Total discriminator loss
        total_loss = (real_loss + fake_loss) / 2
        
        return {
            'total': total_loss,
            'real_loss': real_loss,
            'fake_loss': fake_loss
        }
    
    def extractor_loss(self, secret, extracted_secret):
        """
        Calculate secret extractor loss.
        
        Args:
            secret: Original secret images
            extracted_secret: Extracted secret from stego
        
        Returns:
            torch.Tensor: Extractor loss
        """
        return self.l1_loss(extracted_secret, secret)


def calculate_psnr_batch(img1, img2):
    """Calculate PSNR for a batch of images."""
    mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])
    psnr = 20 * torch.log10(2.0 / torch.sqrt(mse))  # Assuming images in [-1, 1]
    return psnr.mean()


def calculate_ssim_batch(img1, img2):
    """Simplified SSIM calculation for batches."""
    # Convert to [0, 1] range
    img1 = (img1 + 1) / 2
    img2 = (img2 + 1) / 2
    
    mu1 = torch.mean(img1, dim=[2, 3], keepdim=True)
    mu2 = torch.mean(img2, dim=[2, 3], keepdim=True)
    
    sigma1_sq = torch.var(img1, dim=[2, 3], keepdim=True)
    sigma2_sq = torch.var(img2, dim=[2, 3], keepdim=True)
    sigma12 = torch.mean((img1 - mu1) * (img2 - mu2), dim=[2, 3], keepdim=True)
    
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim.mean()


class MetricsCalculator:
    """Helper class to calculate quality metrics during training."""
    
    @staticmethod
    def calculate_metrics(cover, stego, secret, extracted_secret):
        """
        Calculate quality metrics for steganography.
        
        Returns:
            dict: Quality metrics
        """
        with torch.no_grad():
            # Cover quality (how well stego preserves cover)
            cover_psnr = calculate_psnr_batch(cover, stego)
            cover_ssim = calculate_ssim_batch(cover, stego)
            
            # Secret recovery quality
            secret_psnr = calculate_psnr_batch(secret, extracted_secret)
            secret_ssim = calculate_ssim_batch(secret, extracted_secret)
            
            return {
                'cover_psnr': cover_psnr.item(),
                'cover_ssim': cover_ssim.item(),
                'secret_psnr': secret_psnr.item(),
                'secret_ssim': secret_ssim.item()
            }


def test_loss_functions():
    """Test the loss functions."""
    print("🧪 Testing Loss Functions...")
    print("=" * 40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create loss function
    loss_fn = SteganographyLoss(device=device)
    
    # Create dummy data
    batch_size = 4
    cover = torch.randn(batch_size, 3, 32, 32).to(device)
    secret = torch.randn(batch_size, 3, 32, 32).to(device)
    stego = torch.randn(batch_size, 3, 32, 32).to(device)
    extracted_secret = torch.randn(batch_size, 3, 32, 32).to(device)
    
    disc_pred_real = torch.rand(batch_size, 1).to(device)
    disc_pred_fake = torch.rand(batch_size, 1).to(device)
    
    # Test generator loss
    print("📱 Testing Generator Loss...")
    gen_losses = loss_fn.generator_loss(cover, secret, stego, extracted_secret, disc_pred_fake)
    print(f"  Total loss: {gen_losses['total'].item():.4f}")
    print(f"  Adversarial: {gen_losses['adversarial'].item():.4f}")
    print(f"  Cover reconstruction: {gen_losses['cover_reconstruction'].item():.4f}")
    print(f"  Secret recovery: {gen_losses['secret_recovery'].item():.4f}")
    print(f"  Perceptual: {gen_losses['perceptual'].item():.4f}")
    print("  ✅ Generator loss working!")
    
    # Test discriminator loss
    print("\n🔍 Testing Discriminator Loss...")
    disc_losses = loss_fn.discriminator_loss(disc_pred_real, disc_pred_fake)
    print(f"  Total loss: {disc_losses['total'].item():.4f}")
    print(f"  Real loss: {disc_losses['real_loss'].item():.4f}")
    print(f"  Fake loss: {disc_losses['fake_loss'].item():.4f}")
    print("  ✅ Discriminator loss working!")
    
    # Test extractor loss
    print("\n🔓 Testing Extractor Loss...")
    ext_loss = loss_fn.extractor_loss(secret, extracted_secret)
    print(f"  Extractor loss: {ext_loss.item():.4f}")
    print("  ✅ Extractor loss working!")
    
    # Test metrics
    print("\n📊 Testing Quality Metrics...")
    metrics = MetricsCalculator.calculate_metrics(cover, stego, secret, extracted_secret)
    print(f"  Cover PSNR: {metrics['cover_psnr']:.2f} dB")
    print(f"  Cover SSIM: {metrics['cover_ssim']:.4f}")
    print(f"  Secret PSNR: {metrics['secret_psnr']:.2f} dB")
    print(f"  Secret SSIM: {metrics['secret_ssim']:.4f}")
    print("  ✅ Metrics calculation working!")
    
    print(f"\n🎉 All loss functions tested successfully!")
    
    return loss_fn


if __name__ == "__main__":
    print("🎯 STEGANOGRAPHY LOSS FUNCTIONS")
    print("=" * 50)
    print("Testing multi-objective loss for GAN training")
    print("=" * 50)
    
    loss_fn = test_loss_functions()
    
    print(f"\n💡 Loss Function Summary:")
    print(f"1. 🎭 Adversarial Loss: Makes stego undetectable")
    print(f"2. 🖼️  Cover Loss: Preserves cover image quality")
    print(f"3. 🔐 Secret Loss: Ensures secret recovery")
    print(f"4. 👁️  Perceptual Loss: Maintains visual similarity")
    
    print(f"\n🎯 Training Strategy:")
    print(f"  • Generator: Minimize all 4 losses")
    print(f"  • Discriminator: Distinguish real vs stego")
    print(f"  • Extractor: Maximize secret recovery")
    
    print(f"\n🚀 Ready for training implementation!")
