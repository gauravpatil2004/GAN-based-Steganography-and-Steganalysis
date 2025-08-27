"""
LSB Steganography Bit Allocation Analysis - Summary Results

This experiment tested different LSB bit allocations (1-8 bits) to understand
the trade-off between hiding capacity and image quality.
"""

import numpy as np
from src.lsb_stego import load_cifar10_sample, calculate_psnr, calculate_ssim
from src.lsb_experiments import encode_lsb_variable_bits, decode_lsb_variable_bits
import matplotlib.pyplot as plt

def run_bit_analysis():
    """Run focused bit allocation analysis."""
    print("🎯 LSB BIT ALLOCATION ANALYSIS")
    print("=" * 50)
    print("Goal: Find optimal balance between capacity and quality")
    print("=" * 50)
    
    # Load test images
    images = load_cifar10_sample(10)
    if len(images) < 5:
        print("❌ Need more images for analysis")
        return
    
    # Test bit allocations
    bit_allocations = [1, 2, 3, 4, 5, 6, 8]
    results = {}
    
    print("\nTesting different bit allocations...")
    
    for bits in bit_allocations:
        cover_psnrs = []
        secret_psnrs = []
        
        # Test with 5 image pairs
        for i in range(5):
            cover = np.array(images[i])
            secret = np.array(images[i + 1] if i + 1 < len(images) else images[0])
            
            try:
                # Encode and decode
                stego = encode_lsb_variable_bits(cover, secret, bits)
                extracted = decode_lsb_variable_bits(stego, bits)
                
                # Calculate quality metrics
                cover_psnr = calculate_psnr(cover, stego)
                secret_psnr = calculate_psnr(secret, extracted)
                
                cover_psnrs.append(cover_psnr)
                secret_psnrs.append(secret_psnr)
                
            except Exception as e:
                print(f"Error with {bits}-bit encoding: {e}")
                continue
        
        if cover_psnrs:
            results[bits] = {
                'cover_psnr': np.mean(cover_psnrs),
                'secret_psnr': np.mean(secret_psnrs),
                'capacity': bits / 8.0 * 100  # Percentage of cover used
            }
    
    # Display results
    print("\n📊 RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Bits':<4} {'Capacity':<10} {'Cover PSNR':<12} {'Secret PSNR':<12} {'Quality'}")
    print("-" * 60)
    
    for bits in sorted(results.keys()):
        r = results[bits]
        quality = "Excellent" if r['cover_psnr'] > 30 else "Good" if r['cover_psnr'] > 25 else "Fair"
        print(f"{bits:<4} {r['capacity']:<9.1f}% {r['cover_psnr']:<11.2f} {r['secret_psnr']:<11.2f} {quality}")
    
    # Key findings
    print("\n🔍 KEY FINDINGS:")
    print("=" * 60)
    
    # Find optimal configurations
    best_quality = max(results.items(), key=lambda x: x[1]['cover_psnr'])
    best_recovery = max(results.items(), key=lambda x: x[1]['secret_psnr'])
    
    print(f"🏆 Best Cover Quality: {best_quality[0]}-bit LSB ({best_quality[1]['cover_psnr']:.1f} dB)")
    print(f"🎯 Best Secret Recovery: {best_recovery[0]}-bit LSB ({best_recovery[1]['secret_psnr']:.1f} dB)")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   • For high quality: Use 1-2 bit LSB (minimal visible changes)")
    print(f"   • For balanced approach: Use 3-4 bit LSB (good quality + capacity)")  
    print(f"   • For maximum capacity: Use 6-8 bit LSB (noticeable but acceptable)")
    
    # Trade-off analysis
    print(f"\n⚖️  TRADE-OFF ANALYSIS:")
    one_bit = results.get(1, {})
    four_bit = results.get(4, {})
    eight_bit = results.get(8, {})
    
    if one_bit and four_bit and eight_bit:
        print(f"   • 1-bit: {one_bit['capacity']:.1f}% capacity, {one_bit['cover_psnr']:.1f} dB quality")
        print(f"   • 4-bit: {four_bit['capacity']:.1f}% capacity, {four_bit['cover_psnr']:.1f} dB quality")
        print(f"   • 8-bit: {eight_bit['capacity']:.1f}% capacity, {eight_bit['cover_psnr']:.1f} dB quality")
        
        quality_loss_4bit = one_bit['cover_psnr'] - four_bit['cover_psnr']
        quality_loss_8bit = one_bit['cover_psnr'] - eight_bit['cover_psnr']
        
        print(f"   • 4x capacity costs {quality_loss_4bit:.1f} dB quality loss")
        print(f"   • 8x capacity costs {quality_loss_8bit:.1f} dB quality loss")
    
    # Create simple visualization
    create_trade_off_chart(results)
    
    return results

def create_trade_off_chart(results, save_path="lsb_tradeoff_analysis.png"):
    """Create a simple trade-off visualization."""
    if not results:
        return
    
    bits = list(results.keys())
    capacities = [results[b]['capacity'] for b in bits]
    cover_psnrs = [results[b]['cover_psnr'] for b in bits]
    secret_psnrs = [results[b]['secret_psnr'] for b in bits]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('LSB Steganography: Capacity vs Quality Trade-off', fontsize=16, fontweight='bold')
    
    # Chart 1: PSNR vs Bits
    ax1.plot(bits, cover_psnrs, 'o-', linewidth=2, markersize=8, label='Cover Image Quality', color='blue')
    ax1.plot(bits, secret_psnrs, 's-', linewidth=2, markersize=8, label='Secret Recovery Quality', color='red')
    ax1.set_xlabel('LSB Bits Used', fontsize=12)
    ax1.set_ylabel('PSNR (dB)', fontsize=12)
    ax1.set_title('Image Quality vs LSB Bits', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Excellent Quality')
    ax1.axhline(y=25, color='orange', linestyle='--', alpha=0.7, label='Good Quality')
    
    # Chart 2: Quality vs Capacity
    ax2.scatter(capacities, cover_psnrs, s=100, c=bits, cmap='viridis', alpha=0.8)
    ax2.set_xlabel('Hiding Capacity (%)', fontsize=12)
    ax2.set_ylabel('Cover Image PSNR (dB)', fontsize=12)
    ax2.set_title('Quality vs Hiding Capacity', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Add bit labels to points
    for i, bit in enumerate(bits):
        ax2.annotate(f'{bit}b', (capacities[i], cover_psnrs[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Trade-off analysis chart saved to {save_path}")

def create_visual_examples():
    """Create visual examples of different bit allocations."""
    print("\n🖼️  Creating visual examples...")
    
    images = load_cifar10_sample(2)
    if len(images) < 2:
        return
    
    cover = np.array(images[0])
    secret = np.array(images[1])
    
    bit_tests = [1, 2, 4, 8]
    
    fig, axes = plt.subplots(2, len(bit_tests), figsize=(16, 8))
    fig.suptitle('LSB Steganography: Visual Quality Comparison', fontsize=16, fontweight='bold')
    
    for i, bits in enumerate(bit_tests):
        # Encode and decode
        stego = encode_lsb_variable_bits(cover, secret, bits)
        extracted = decode_lsb_variable_bits(stego, bits)
        
        # Calculate metrics
        cover_psnr = calculate_psnr(cover, stego)
        secret_psnr = calculate_psnr(secret, extracted)
        
        # Plot stego image
        axes[0, i].imshow(stego)
        axes[0, i].set_title(f'{bits}-bit LSB Stego\nPSNR: {cover_psnr:.1f} dB\nCapacity: {bits/8*100:.1f}%')
        axes[0, i].axis('off')
        
        # Plot extracted secret
        axes[1, i].imshow(extracted)
        axes[1, i].set_title(f'Extracted Secret\nPSNR: {secret_psnr:.1f} dB')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig("lsb_visual_examples.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Visual examples saved to lsb_visual_examples.png")

if __name__ == "__main__":
    print("🚀 STARTING LSB BIT ALLOCATION EXPERIMENT")
    print("This will help us understand the optimal balance for steganography!")
    print()
    
    # Run the analysis
    results = run_bit_analysis()
    
    if results:
        # Create visual examples
        create_visual_examples()
        
        print("\n" + "=" * 60)
        print("🎉 LSB BIT ALLOCATION EXPERIMENT COMPLETED!")
        print("=" * 60)
        print("📁 Generated files:")
        print("   • lsb_tradeoff_analysis.png - Trade-off charts")
        print("   • lsb_visual_examples.png - Visual quality examples")
        
        print("\n🎯 EXPERIMENT CONCLUSION:")
        print("   ✅ Successfully analyzed 1-8 bit LSB allocations")
        print("   ✅ Identified optimal configurations for different use cases")
        print("   ✅ Demonstrated clear quality vs capacity trade-offs")
        
        print("\n🚀 READY FOR DAY 4: GAN ARCHITECTURE DESIGN!")
        print("   Next step: Design generator and discriminator networks")
        print("   Goal: Create GAN-based steganography for better quality")
    else:
        print("❌ Experiment failed. Please check CIFAR-10 availability.")
