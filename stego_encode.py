import numpy as np
from PIL import Image
import sys

# Simple LSB steganography encoder for PNG images
def encode_image(input_image_path, output_image_path, secret_message):
    image = Image.open(input_image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    data = np.array(image)
    flat_data = data.flatten()

    # Convert message to binary
    message_bytes = secret_message.encode('utf-8')
    message_bits = ''.join([format(byte, '08b') for byte in message_bytes])
    message_bits += '00000000'  # Null terminator

    if len(message_bits) > len(flat_data):
        raise ValueError('Message too long to encode in image.')

    # Encode message bits into image
    for i, bit in enumerate(message_bits):
        flat_data[i] = (flat_data[i] & ~1) | int(bit)

    # Reshape and save
    encoded_data = flat_data.reshape(data.shape)
    encoded_image = Image.fromarray(encoded_data.astype(np.uint8))
    encoded_image.save(output_image_path)
    print(f'Encoded message into {output_image_path}')

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: python stego_encode.py <input_image> <output_image> <secret_message>')
        sys.exit(1)
    encode_image(sys.argv[1], sys.argv[2], sys.argv[3])
