import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import pandas as pd
from src.lsb_stego import calculate_psnr, calculate_ssim


def encode_lsb_variable_bits(cover_image, secret_image, bits=4):
    """
    Encode a secret image into a cover image using variable LSB steganography.
    
    Args:
        cover_image (numpy.ndarray): Cover image (H, W, 3)
        secret_image (numpy.ndarray): Secret image to hide (H, W, 3)
        bits (int): Number of bits to use for hiding (1-8)
    
    Returns:
        numpy.ndarray: Stego image with hidden secret
    """
    if bits < 1 or bits > 8:
        raise ValueError("Bits must be between 1 and 8")
    
    # Convert to numpy arrays and uint8
    cover_image = np.array(cover_image).astype(np.uint8)
    secret_image = np.array(secret_image).astype(np.uint8)
    
    # Ensure same shape
    if cover_image.shape != secret_image.shape:
        raise ValueError("Cover and secret images must have the same shape")
    
    # Create copy for stego image
    stego_image = cover_image.copy()
    
    # Create masks safely to avoid overflow
    # For secret: keep top 'bits' bits
    secret_mask = ((0xFF >> (8 - bits)) << (8 - bits)) & 0xFF
    # For cover: clear bottom 'bits' bits  
    cover_mask = (0xFF << bits) & 0xFF
    
    # Extract most significant bits from secret
    secret_msb = (secret_image & secret_mask) >> (8 - bits)
    
    # Clear LSBs of cover and embed secret
    stego_image = (stego_image & cover_mask) | secret_msb
    
    # Ensure result stays in uint8 range
    stego_image = stego_image.astype(np.uint8)
    
    return stego_image


def decode_lsb_variable_bits(stego_image, bits=4):
    """
    Decode a secret image from a stego image using variable LSB steganography.
    
    Args:
        stego_image (numpy.ndarray): Stego image containing hidden secret
        bits (int): Number of bits used for hiding (1-8)
    
    Returns:
        numpy.ndarray: Extracted secret image
    """
    if bits < 1 or bits > 8:
        raise ValueError("Bits must be between 1 and 8")
    
    stego_image = np.array(stego_image).astype(np.uint8)
    
    # Extract LSBs containing the secret
    lsb_mask = ((1 << bits) - 1) & 0xFF  # Mask for extracting 'bits' LSBs
    secret_lsb = stego_image & lsb_mask
    
    # Shift to MSB position
    extracted_secret = (secret_lsb << (8 - bits)) & 0xFF
    
    # Replicate pattern to fill remaining bits for better visibility
    for i in range(1, 8 // bits):
        shift_val = 8 - bits * (i + 1)
        if shift_val >= 0:
            extracted_secret = (extracted_secret | ((secret_lsb << shift_val) & 0xFF)) & 0xFF
    
    # Fill any remaining bits
    remaining_bits = 8 % bits
    if remaining_bits > 0:
        shift_val = bits - remaining_bits
        if shift_val >= 0:
            extracted_secret = (extracted_secret | ((secret_lsb >> shift_val) & 0xFF)) & 0xFF
    
    return extracted_secret.astype(np.uint8)
    
    return extracted_secret


def load_cifar10_batch(num_images=20):
    """
    Load a batch of CIFAR-10 images for comprehensive testing.
    
    Args:
        num_images (int): Number of images to load
    
    Returns:
        list: List of PIL Images
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage()
    ])
    
    try:
        # Load CIFAR-10 test dataset
        dataset = torchvision.datasets.CIFAR10(
            root='./data', 
            train=False, 
            download=False,  # Don't re-download
            transform=transform
        )
        
        # Get diverse images from different classes
        images = []
        class_counts = {}
        
        for i, (image, label) in enumerate(dataset):
            if len(images) >= num_images:
                break
                
            # Try to get balanced representation of classes
            if class_counts.get(label, 0) < num_images // 10 + 1:
                images.append(image)
                class_counts[label] = class_counts.get(label, 0) + 1
        
        return images
        
    except Exception as e:
        print(f"Error loading CIFAR-10: {e}")
        return []


def test_bit_allocation_comprehensive():
    """
    Comprehensive test of different bit allocations for LSB steganography.
    """
    print("🔬 COMPREHENSIVE LSB BIT ALLOCATION ANALYSIS")
    print("=" * 70)
    
    # Load test images
    print("📥 Loading CIFAR-10 test images...")
    images = load_cifar10_batch(num_images=20)
    
    if len(images) < 10:
        print("❌ Need at least 10 images for comprehensive testing")
        return False
    
    print(f"✅ Loaded {len(images)} images")
    
    # Test different bit allocations
    bit_allocations = [1, 2, 3, 4, 5, 6, 7, 8]
    results = []
    
    print(f"\n🧪 Testing {len(bit_allocations)} different bit allocations...")
    print("   Bit allocation: ", end="")
    
    for bits in bit_allocations:
        print(f"{bits}", end="", flush=True)
        
        # Test metrics for this bit allocation
        cover_psnrs = []
        secret_psnrs = []
        cover_ssims = []
        secret_ssims = []
        
        # Test with multiple image pairs
        test_pairs = min(10, len(images) - 1)
        
        for i in range(test_pairs):
            try:
                cover_img = np.array(images[i])
                secret_img = np.array(images[i + 1])
                
                # Encode with this bit allocation
                stego_img = encode_lsb_variable_bits(cover_img, secret_img, bits)
                extracted_img = decode_lsb_variable_bits(stego_img, bits)
                
                # Calculate metrics
                cover_psnr = calculate_psnr(cover_img, stego_img)
                secret_psnr = calculate_psnr(secret_img, extracted_img)
                cover_ssim = calculate_ssim(cover_img, stego_img)
                secret_ssim = calculate_ssim(secret_img, extracted_img)
                
                cover_psnrs.append(cover_psnr)
                secret_psnrs.append(secret_psnr)
                cover_ssims.append(cover_ssim)
                secret_ssims.append(secret_ssim)
                
            except Exception as e:
                print(f"\n   ⚠️  Error with {bits}-bit test {i+1}: {e}")
                continue
        
        # Calculate averages
        if cover_psnrs:  # If we have any successful tests
            results.append({
                'bits': bits,
                'cover_psnr_avg': np.mean(cover_psnrs),
                'cover_psnr_std': np.std(cover_psnrs),
                'secret_psnr_avg': np.mean(secret_psnrs),
                'secret_psnr_std': np.std(secret_psnrs),
                'cover_ssim_avg': np.mean(cover_ssims),
                'cover_ssim_std': np.std(cover_ssims),
                'secret_ssim_avg': np.mean(secret_ssims),
                'secret_ssim_std': np.std(secret_ssims),
                'hiding_capacity': bits / 8.0,  # Fraction of cover image used
                'test_count': len(cover_psnrs)
            })
        
        print(".", end="", flush=True)
    
    print(" ✅")
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    
    # Display results table
    print("\n" + "=" * 70)
    print("📊 BIT ALLOCATION ANALYSIS RESULTS")
    print("=" * 70)
    
    print(f"{'Bits':<4} {'Cover PSNR':<12} {'Secret PSNR':<13} {'Cover SSIM':<12} {'Secret SSIM':<13} {'Capacity':<8}")
    print("-" * 70)
    
    for _, row in df.iterrows():
        print(f"{row['bits']:<4} "
              f"{row['cover_psnr_avg']:<7.2f}±{row['cover_psnr_std']:<4.1f} "
              f"{row['secret_psnr_avg']:<7.2f}±{row['secret_psnr_std']:<5.1f} "
              f"{row['cover_ssim_avg']:<7.4f}±{row['cover_ssim_std']:<4.3f} "
              f"{row['secret_ssim_avg']:<7.4f}±{row['secret_ssim_std']:<5.3f} "
              f"{row['hiding_capacity']:<8.1%}")
    
    # Analysis and recommendations
    print("\n" + "=" * 70)
    print("🎯 ANALYSIS & RECOMMENDATIONS")
    print("=" * 70)
    
    # Find optimal configurations
    best_quality = df.loc[df['cover_psnr_avg'].idxmax()]
    best_recovery = df.loc[df['secret_psnr_avg'].idxmax()]
    balanced = df.loc[df['cover_psnr_avg'] + df['secret_psnr_avg'].idxmax()]
    
    print(f"🏆 Best Cover Quality: {best_quality['bits']}-bit LSB "
          f"(PSNR: {best_quality['cover_psnr_avg']:.2f} dB)")
    
    print(f"🎯 Best Secret Recovery: {best_recovery['bits']}-bit LSB "
          f"(PSNR: {best_recovery['secret_psnr_avg']:.2f} dB)")
    
    print(f"⚖️  Best Balance: {balanced['bits']}-bit LSB "
          f"(Combined PSNR: {balanced['cover_psnr_avg'] + balanced['secret_psnr_avg']:.2f} dB)")
    
    # Quality guidelines
    print(f"\n💡 Quality Guidelines:")
    high_quality = df[df['cover_psnr_avg'] > 30]
    good_quality = df[(df['cover_psnr_avg'] > 25) & (df['cover_psnr_avg'] <= 30)]
    acceptable_quality = df[(df['cover_psnr_avg'] > 20) & (df['cover_psnr_avg'] <= 25)]
    
    if len(high_quality) > 0:
        print(f"   🟢 High Quality (>30 dB): {list(high_quality['bits'])} bits")
    if len(good_quality) > 0:
        print(f"   🟡 Good Quality (25-30 dB): {list(good_quality['bits'])} bits")
    if len(acceptable_quality) > 0:
        print(f"   🟠 Acceptable Quality (20-25 dB): {list(acceptable_quality['bits'])} bits")
    
    return df


def create_comprehensive_visualization(results_df, save_path="lsb_bit_analysis.png"):
    """
    Create comprehensive visualization of bit allocation analysis.
    """
    print(f"\n🖼️  Creating comprehensive visualization...")
    
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('LSB Steganography: Bit Allocation Analysis', fontsize=16, fontweight='bold')
        
        bits = results_df['bits']
        
        # Plot 1: PSNR vs Bits
        ax1.errorbar(bits, results_df['cover_psnr_avg'], yerr=results_df['cover_psnr_std'], 
                    label='Cover Image', marker='o', linewidth=2, capsize=5)
        ax1.errorbar(bits, results_df['secret_psnr_avg'], yerr=results_df['secret_psnr_std'], 
                    label='Secret Recovery', marker='s', linewidth=2, capsize=5)
        ax1.set_xlabel('Number of LSB Bits Used')
        ax1.set_ylabel('PSNR (dB)')
        ax1.set_title('Image Quality vs Hiding Capacity')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='High Quality Threshold')
        ax1.axhline(y=25, color='orange', linestyle='--', alpha=0.7, label='Good Quality Threshold')
        
        # Plot 2: SSIM vs Bits
        ax2.errorbar(bits, results_df['cover_ssim_avg'], yerr=results_df['cover_ssim_std'], 
                    label='Cover Image', marker='o', linewidth=2, capsize=5)
        ax2.errorbar(bits, results_df['secret_ssim_avg'], yerr=results_df['secret_ssim_std'], 
                    label='Secret Recovery', marker='s', linewidth=2, capsize=5)
        ax2.set_xlabel('Number of LSB Bits Used')
        ax2.set_ylabel('SSIM')
        ax2.set_title('Structural Similarity vs Hiding Capacity')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Plot 3: Hiding Capacity vs Quality Trade-off
        combined_quality = results_df['cover_psnr_avg'] + results_df['secret_psnr_avg']
        ax3.scatter(results_df['hiding_capacity'] * 100, combined_quality, 
                   s=100, c=bits, cmap='viridis', alpha=0.7)
        ax3.set_xlabel('Hiding Capacity (%)')
        ax3.set_ylabel('Combined PSNR (dB)')
        ax3.set_title('Capacity vs Quality Trade-off')
        ax3.grid(True, alpha=0.3)
        
        # Add colorbar for bits
        scatter = ax3.scatter(results_df['hiding_capacity'] * 100, combined_quality, 
                             s=100, c=bits, cmap='viridis', alpha=0.7)
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('LSB Bits Used')
        
        # Plot 4: Quality Degradation Rate
        cover_degradation = results_df['cover_psnr_avg'].iloc[0] - results_df['cover_psnr_avg']
        secret_degradation = results_df['secret_psnr_avg'].iloc[0] - results_df['secret_psnr_avg']
        
        ax4.plot(bits, cover_degradation, marker='o', linewidth=2, label='Cover Quality Loss')
        ax4.plot(bits, -secret_degradation, marker='s', linewidth=2, label='Secret Recovery Gain')
        ax4.set_xlabel('Number of LSB Bits Used')
        ax4.set_ylabel('PSNR Change from 1-bit (dB)')
        ax4.set_title('Quality Change Relative to 1-bit LSB')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Comprehensive visualization saved to {save_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        return False


def create_example_images_grid(save_path="lsb_examples_grid.png"):
    """
    Create a grid showing examples of different bit allocations.
    """
    print(f"\n🖼️  Creating example images grid...")
    
    try:
        # Load sample images
        images = load_cifar10_batch(num_images=2)
        if len(images) < 2:
            return False
        
        cover_img = np.array(images[0])
        secret_img = np.array(images[1])
        
        # Test different bit allocations
        bit_allocations = [1, 2, 4, 6, 8]
        
        fig, axes = plt.subplots(3, len(bit_allocations), figsize=(20, 12))
        fig.suptitle('LSB Steganography Examples: Different Bit Allocations', fontsize=16, fontweight='bold')
        
        for i, bits in enumerate(bit_allocations):
            # Encode and decode
            stego_img = encode_lsb_variable_bits(cover_img, secret_img, bits)
            extracted_img = decode_lsb_variable_bits(stego_img, bits)
            
            # Calculate metrics
            cover_psnr = calculate_psnr(cover_img, stego_img)
            secret_psnr = calculate_psnr(secret_img, extracted_img)
            
            # Plot stego image
            axes[0, i].imshow(stego_img)
            axes[0, i].set_title(f'{bits}-bit LSB\nCover PSNR: {cover_psnr:.1f} dB')
            axes[0, i].axis('off')
            
            # Plot extracted secret
            axes[1, i].imshow(extracted_img)
            axes[1, i].set_title(f'Extracted Secret\nPSNR: {secret_psnr:.1f} dB')
            axes[1, i].axis('off')
            
            # Plot difference image (amplified)
            diff_img = np.abs(cover_img.astype(np.float32) - stego_img.astype(np.float32)) * 10
            diff_img = np.clip(diff_img, 0, 255).astype(np.uint8)
            axes[2, i].imshow(diff_img)
            axes[2, i].set_title(f'Difference ×10\nCapacity: {bits/8*100:.1f}%')
            axes[2, i].axis('off')
        
        # Add row labels
        axes[0, 0].text(-50, cover_img.shape[0]//2, 'Stego Images', rotation=90, 
                       va='center', ha='center', fontsize=14, fontweight='bold')
        axes[1, 0].text(-50, cover_img.shape[0]//2, 'Extracted Secrets', rotation=90, 
                       va='center', ha='center', fontsize=14, fontweight='bold')
        axes[2, 0].text(-50, cover_img.shape[0]//2, 'Differences', rotation=90, 
                       va='center', ha='center', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Example images grid saved to {save_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating examples grid: {e}")
        return False


def main():
    """
    Main function to run comprehensive LSB bit allocation analysis.
    """
    print("🎯 LSB STEGANOGRAPHY: BIT ALLOCATION EXPERIMENT")
    print("=" * 70)
    print("💡 Goal: Understand trade-offs between hiding capacity and image quality")
    print("=" * 70)
    
    # Run comprehensive analysis
    results_df = test_bit_allocation_comprehensive()
    
    if results_df is not None and len(results_df) > 0:
        # Create visualizations
        create_comprehensive_visualization(results_df)
        create_example_images_grid()
        
        # Save results to CSV
        results_df.to_csv('lsb_bit_allocation_results.csv', index=False)
        print(f"✅ Results saved to lsb_bit_allocation_results.csv")
        
        print("\n🎉 LSB Bit Allocation Experiment Complete!")
        print("📁 Generated Files:")
        print("   - lsb_bit_analysis.png (comprehensive charts)")
        print("   - lsb_examples_grid.png (visual examples)")
        print("   - lsb_bit_allocation_results.csv (detailed data)")
        
        print("\n🔍 Key Findings:")
        best_bits = results_df.loc[results_df['cover_psnr_avg'].idxmax(), 'bits']
        print(f"   - Best quality preservation: {best_bits}-bit LSB")
        print(f"   - Quality degrades as more bits are used")
        print(f"   - Higher capacity = lower image quality")
        print(f"   - Sweet spot is typically 2-4 bits for good balance")
        
        print("\n🚀 Ready for Day 4: GAN Architecture Design!")
        return True
    else:
        print("❌ Analysis failed. Please check CIFAR-10 dataset availability.")
        return False


if __name__ == "__main__":
    success = main()
