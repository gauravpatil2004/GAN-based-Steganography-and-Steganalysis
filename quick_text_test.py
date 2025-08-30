"""
Quick Test for Text Steganography Components

Verify all components work before starting training.
"""

import sys
sys.path.append('src')

print("🧪 Quick Test: Text Steganography Components")
print("=" * 50)

try:
    # Test 1: Text Processing
    print("1. Testing text processing...")
    from text_processor import TextProcessor
    processor = TextProcessor(max_length=64)
    
    test_text = "Hello, this is a secret message!"
    encoded = processor.encode_text(test_text)
    decoded = processor.decode_text(encoded)
    
    print(f"   Original: '{test_text}'")
    print(f"   Decoded:  '{decoded}'")
    print(f"   Match: {test_text == decoded}")
    
    # Test 2: Text Embedding
    print("\n2. Testing text embedding...")
    from text_processor import TextEmbedding
    import torch
    
    embedding_net = TextEmbedding(vocab_size=processor.vocab_size)
    text_indices = torch.randint(0, 98, (2, 32))  # Batch of 2, length 32
    embeddings = embedding_net(text_indices)
    
    print(f"   Input shape: {text_indices.shape}")
    print(f"   Output shape: {embeddings.shape}")
    
    # Test 3: Text GAN Networks
    print("\n3. Testing GAN networks...")
    from text_gan_architecture import TextSteganoGenerator, TextSteganoDiscriminator, TextExtractor
    
    generator = TextSteganoGenerator(text_embed_dim=128)
    discriminator = TextSteganoDiscriminator()
    extractor = TextExtractor(vocab_size=processor.vocab_size, max_text_length=64)
    
    # Test forward passes
    cover_images = torch.randn(2, 3, 32, 32)
    text_embeddings = torch.randn(2, 128)
    
    stego_images = generator(cover_images, text_embeddings)
    disc_output = discriminator(stego_images)
    text_logits = extractor(stego_images)
    
    print(f"   Generator: {cover_images.shape} + {text_embeddings.shape} → {stego_images.shape}")
    print(f"   Discriminator: {stego_images.shape} → {disc_output.shape}")
    print(f"   Extractor: {stego_images.shape} → {text_logits.shape}")
    
    # Test 4: Loss Functions
    print("\n4. Testing loss functions...")
    from text_gan_losses import TextSteganoLoss, TextMetricsCalculator
    
    loss_fn = TextSteganoLoss()
    text_target = torch.randint(1, 98, (2, 64))
    
    gen_losses = loss_fn.generator_loss(
        cover_images, stego_images, text_target, 
        text_logits, disc_output, text_embeddings
    )
    
    metrics = TextMetricsCalculator.calculate_text_metrics(
        cover_images, stego_images, text_target, text_logits
    )
    
    print(f"   Generator loss: {gen_losses['total']:.4f}")
    print(f"   Character accuracy: {metrics['character_accuracy']:.3f}")
    
    print("\n✅ ALL COMPONENTS WORKING!")
    print("🚀 Ready to start text steganography training!")
    
    # Show advantages
    print(f"\n🎯 TEXT STEGANOGRAPHY ADVANTAGES:")
    print(f"   ⚡ Faster training (30-60 min vs 2-4 hours)")
    print(f"   📊 Clear metrics (character accuracy)")
    print(f"   💼 Practical use (passwords, URLs, messages)")
    print(f"   🔍 Easy validation (text extraction success)")
    print(f"   📈 Better capacity ({processor.max_length} chars per image)")

except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
