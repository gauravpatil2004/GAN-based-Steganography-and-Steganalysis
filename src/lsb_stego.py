import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os


def encode_lsb(cover_image, secret_image):
    """
    Encode a secret image into a cover image using LSB steganography.
    
    Args:
        cover_image (numpy.ndarray or PIL.Image): Cover image (H, W, 3)
        secret_image (numpy.ndarray or PIL.Image): Secret image to hide (H, W, 3)
    
    Returns:
        numpy.ndarray: Stego image with hidden secret
    """
    # Convert PIL Images to numpy arrays if needed
    if isinstance(cover_image, Image.Image):
        cover_image = np.array(cover_image)
    if isinstance(secret_image, Image.Image):
        secret_image = np.array(secret_image)
    
    # Ensure images are the same size
    if cover_image.shape != secret_image.shape:
        raise ValueError(f"Cover and secret images must have the same shape. "
                        f"Got {cover_image.shape} and {secret_image.shape}")
    
    # Convert to uint8 if not already
    cover_image = cover_image.astype(np.uint8)
    secret_image = secret_image.astype(np.uint8)
    
    # Create a copy of the cover image for the stego image
    stego_image = cover_image.copy()
    
    # Get the most significant 4 bits of the secret image
    secret_msb = (secret_image >> 4) & 0x0F
    
    # Clear the least significant 4 bits of the cover image
    stego_image = stego_image & 0xF0
    
    # Embed the secret's MSB into the cover's LSB
    stego_image = stego_image | secret_msb
    
    return stego_image


def decode_lsb(stego_image):
    """
    Decode a secret image from a stego image using LSB steganography.
    
    Args:
        stego_image (numpy.ndarray or PIL.Image): Stego image containing hidden secret
    
    Returns:
        numpy.ndarray: Extracted secret image
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(stego_image, Image.Image):
        stego_image = np.array(stego_image)
    
    # Convert to uint8 if not already
    stego_image = stego_image.astype(np.uint8)
    
    # Extract the least significant 4 bits (which contain the secret)
    secret_lsb = stego_image & 0x0F
    
    # Shift them to the most significant 4 bits position
    extracted_secret = secret_lsb << 4
    
    # Fill the remaining 4 bits with the same pattern for better visibility
    extracted_secret = extracted_secret | secret_lsb
    
    return extracted_secret


def load_cifar10_sample(num_images=5, image_size=32):
    """
    Load sample CIFAR-10 images for testing.
    
    Args:
        num_images (int): Number of images to load
        image_size (int): Size of images (CIFAR-10 is 32x32)
    
    Returns:
        list: List of PIL Images
    """
    # Define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage()
    ])
    
    # Load CIFAR-10 dataset
    try:
        dataset = torchvision.datasets.CIFAR10(
            root='./data', 
            train=False, 
            download=True, 
            transform=transform
        )
        
        # Get sample images
        images = []
        for i in range(min(num_images, len(dataset))):
            image, _ = dataset[i]
            images.append(image)
        
        return images
        
    except Exception as e:
        print(f"Error loading CIFAR-10: {e}")
        print("Creating dummy images instead...")
        
        # Create dummy images if CIFAR-10 fails to load
        images = []
        for i in range(num_images):
            # Create random RGB image
            dummy_array = np.random.randint(0, 256, (image_size, image_size, 3), dtype=np.uint8)
            dummy_image = Image.fromarray(dummy_array)
            images.append(dummy_image)
        
        return images


def calculate_psnr(original, compressed):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Args:
        original (numpy.ndarray): Original image
        compressed (numpy.ndarray): Compressed/modified image
    
    Returns:
        float: PSNR value in dB
    """
    # Convert to float for calculations
    original = original.astype(np.float64)
    compressed = compressed.astype(np.float64)
    
    # Calculate Mean Squared Error
    mse = np.mean((original - compressed) ** 2)
    
    if mse == 0:
        return float('inf')  # Perfect match
    
    # Calculate PSNR
    max_pixel_value = 255.0
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    
    return psnr


def calculate_ssim(img1, img2):
    """
    Calculate Structural Similarity Index (SSIM) between two images.
    Simplified version - for full SSIM, use skimage.metrics.structural_similarity
    
    Args:
        img1 (numpy.ndarray): First image
        img2 (numpy.ndarray): Second image
    
    Returns:
        float: SSIM value between -1 and 1
    """
    # Convert to float
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # Calculate means
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    
    # Calculate variances and covariance
    var1 = np.var(img1)
    var2 = np.var(img2)
    cov = np.mean((img1 - mu1) * (img2 - mu2))
    
    # SSIM constants
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    
    # Calculate SSIM
    ssim = ((2 * mu1 * mu2 + c1) * (2 * cov + c2)) / ((mu1**2 + mu2**2 + c1) * (var1 + var2 + c2))
    
    return ssim


def test_lsb_steganography():
    """
    Test LSB steganography with CIFAR-10 images.
    """
    print("🔬 Testing LSB Steganography on CIFAR-10 Images")
    print("=" * 60)
    
    # Load sample CIFAR-10 images
    print("📥 Loading CIFAR-10 sample images...")
    images = load_cifar10_sample(num_images=5)
    
    if len(images) < 2:
        print("❌ Need at least 2 images for testing")
        return False
    
    print(f"✅ Loaded {len(images)} images")
    
    # Test encoding and decoding
    results = []
    
    for i in range(min(3, len(images) - 1)):  # Test with first 3 pairs
        print(f"\n🧪 Test {i+1}: Encoding image {i+1} into image {i+2}")
        
        try:
            # Get cover and secret images
            cover_img = images[i]
            secret_img = images[i + 1]
            
            print(f"   Cover image size: {cover_img.size}")
            print(f"   Secret image size: {secret_img.size}")
            
            # Convert to numpy arrays
            cover_array = np.array(cover_img)
            secret_array = np.array(secret_img)
            
            # Encode
            stego_array = encode_lsb(cover_array, secret_array)
            print("   ✅ Encoding successful")
            
            # Decode
            extracted_array = decode_lsb(stego_array)
            print("   ✅ Decoding successful")
            
            # Calculate quality metrics
            psnr_cover = calculate_psnr(cover_array, stego_array)
            psnr_secret = calculate_psnr(secret_array, extracted_array)
            ssim_cover = calculate_ssim(cover_array, stego_array)
            ssim_secret = calculate_ssim(secret_array, extracted_array)
            
            print(f"   📊 Cover vs Stego - PSNR: {psnr_cover:.2f} dB, SSIM: {ssim_cover:.4f}")
            print(f"   📊 Secret vs Extracted - PSNR: {psnr_secret:.2f} dB, SSIM: {ssim_secret:.4f}")
            
            # Store results
            results.append({
                'test': i+1,
                'cover_psnr': psnr_cover,
                'secret_psnr': psnr_secret,
                'cover_ssim': ssim_cover,
                'secret_ssim': ssim_secret,
                'success': True
            })
            
        except Exception as e:
            print(f"   ❌ Test {i+1} failed: {e}")
            results.append({
                'test': i+1,
                'success': False,
                'error': str(e)
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("📈 TEST SUMMARY")
    print("=" * 60)
    
    successful_tests = [r for r in results if r['success']]
    
    if successful_tests:
        avg_cover_psnr = np.mean([r['cover_psnr'] for r in successful_tests])
        avg_secret_psnr = np.mean([r['secret_psnr'] for r in successful_tests])
        avg_cover_ssim = np.mean([r['cover_ssim'] for r in successful_tests])
        avg_secret_ssim = np.mean([r['secret_ssim'] for r in successful_tests])
        
        print(f"✅ Successful tests: {len(successful_tests)}/{len(results)}")
        print(f"📊 Average Cover PSNR: {avg_cover_psnr:.2f} dB")
        print(f"📊 Average Secret Recovery PSNR: {avg_secret_psnr:.2f} dB")
        print(f"📊 Average Cover SSIM: {avg_cover_ssim:.4f}")
        print(f"📊 Average Secret Recovery SSIM: {avg_secret_ssim:.4f}")
        
        # Quality assessment
        if avg_cover_psnr > 30:
            print("🎉 Excellent cover image quality preservation!")
        elif avg_cover_psnr > 25:
            print("👍 Good cover image quality preservation")
        else:
            print("⚠️  Cover image quality could be improved")
            
        if avg_secret_psnr > 20:
            print("🎉 Excellent secret recovery quality!")
        elif avg_secret_psnr > 15:
            print("👍 Good secret recovery quality")
        else:
            print("⚠️  Secret recovery quality could be improved")
            
    else:
        print("❌ All tests failed")
        return False
    
    return len(successful_tests) == len(results)


def visualize_steganography_results(save_path="lsb_test_results.png"):
    """
    Create a visualization of steganography results.
    """
    print(f"\n🖼️  Creating visualization...")
    
    try:
        # Load sample images
        images = load_cifar10_sample(num_images=2)
        
        if len(images) < 2:
            print("❌ Need at least 2 images for visualization")
            return False
        
        cover_img = np.array(images[0])
        secret_img = np.array(images[1])
        
        # Perform steganography
        stego_img = encode_lsb(cover_img, secret_img)
        extracted_img = decode_lsb(stego_img)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle('LSB Steganography Test Results', fontsize=16, fontweight='bold')
        
        # Plot images
        axes[0, 0].imshow(cover_img)
        axes[0, 0].set_title('Cover Image', fontsize=14)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(secret_img)
        axes[0, 1].set_title('Secret Image', fontsize=14)
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(stego_img)
        axes[1, 0].set_title('Stego Image (with hidden secret)', fontsize=14)
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(extracted_img)
        axes[1, 1].set_title('Extracted Secret', fontsize=14)
        axes[1, 1].axis('off')
        
        # Add quality metrics as text
        psnr_cover = calculate_psnr(cover_img, stego_img)
        psnr_secret = calculate_psnr(secret_img, extracted_img)
        
        fig.text(0.5, 0.02, 
                f'Cover PSNR: {psnr_cover:.2f} dB | Secret Recovery PSNR: {psnr_secret:.2f} dB',
                ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualization saved to {save_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        return False


if __name__ == "__main__":
    print("🎯 LSB STEGANOGRAPHY TESTING MODULE")
    print("=" * 60)
    
    # Run tests
    success = test_lsb_steganography()
    
    # Create visualization
    visualize_steganography_results()
    
    # Final status
    if success:
        print("\n🎉 All LSB steganography tests completed successfully!")
        print("💡 Day 3: Basic Steganography Implementation - COMPLETED ✅")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
    
    print("\n📚 Next steps:")
    print("   - Experiment with different bit allocation (2-bit, 3-bit LSB)")
    print("   - Try different image datasets")
    print("   - Implement LSB steganography for text messages")
    print("   - Start working on GAN-based steganography models")
