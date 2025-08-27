#!/usr/bin/env python3

import sys
import os
sys.path.append('src')

from lsb_experiments import (
    encode_lsb_variable_bits, 
    decode_lsb_variable_bits, 
    load_cifar10_batch,
    calculate_psnr,
    calculate_ssim
)
import numpy as np

def quick_test():
    """Quick test of LSB bit allocation functionality."""
    print("🔬 Quick LSB Bit Allocation Test")
    print("=" * 40)
    
    # Load 2 CIFAR-10 images
    print("Loading CIFAR-10 images...")
    images = load_cifar10_batch(2)
    
    if len(images) < 2:
        print("❌ Failed to load CIFAR-10 images")
        return False
    
    print(f"✅ Loaded {len(images)} images")
    
    # Convert to numpy
    cover = np.array(images[0])
    secret = np.array(images[1])
    
    print(f"Image shapes: {cover.shape}")
    
    # Test different bit allocations
    bit_tests = [1, 2, 4, 6, 8]
    
    print("\nTesting bit allocations:")
    print("Bits | Cover PSNR | Secret PSNR")
    print("-" * 35)
    
    for bits in bit_tests:
        try:
            # Encode and decode
            stego = encode_lsb_variable_bits(cover, secret, bits)
            extracted = decode_lsb_variable_bits(stego, bits)
            
            # Calculate quality
            cover_psnr = calculate_psnr(cover, stego)
            secret_psnr = calculate_psnr(secret, extracted)
            
            print(f" {bits:2d}  |  {cover_psnr:8.2f}  |  {secret_psnr:9.2f}")
            
        except Exception as e:
            print(f" {bits:2d}  |   ERROR: {e}")
    
    print("\n✅ Quick test completed!")
    return True

if __name__ == "__main__":
    quick_test()
