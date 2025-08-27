from src.lsb_experiments import encode_lsb_variable_bits, decode_lsb_variable_bits
from src.lsb_stego import calculate_psnr, load_cifar10_sample
import numpy as np

print("🔬 Testing LSB Variable Bit Allocations")
print("=" * 45)

# Load test images
images = load_cifar10_sample(2)
cover = np.array(images[0])
secret = np.array(images[1])

print(f"Cover image shape: {cover.shape}")
print(f"Secret image shape: {secret.shape}")
print()

print("Testing different bit allocations:")
print("Bits | Cover PSNR | Secret PSNR | Capacity")
print("-" * 42)

for bits in [1, 2, 3, 4, 5, 6, 7, 8]:
    try:
        # Encode
        stego = encode_lsb_variable_bits(cover, secret, bits)
        # Decode  
        extracted = decode_lsb_variable_bits(stego, bits)
        
        # Calculate quality
        cover_psnr = calculate_psnr(cover, stego)
        secret_psnr = calculate_psnr(secret, extracted)
        capacity = bits / 8.0 * 100
        
        print(f" {bits:2d}  |  {cover_psnr:8.2f}  |  {secret_psnr:9.2f}  | {capacity:6.1f}%")
        
    except Exception as e:
        print(f" {bits:2d}  |   ERROR: {str(e)[:30]}...")

print("\n✅ LSB Variable Bits Test Complete!")
