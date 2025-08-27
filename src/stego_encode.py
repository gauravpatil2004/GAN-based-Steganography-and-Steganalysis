from PIL import Image

def encode_image(input_path, output_path, secret_message):
    # Open the image
    image = Image.open(input_path)
    encoded = image.copy()
    width, height = image.size
    message = secret_message + "####"   # delimiter to know when message ends
    binary_message = ''.join([format(ord(i), "08b") for i in message])

    data_index = 0
    for y in range(height):
        for x in range(width):
            pixel = list(image.getpixel((x, y)))
            for n in range(3):  # for R, G, B
                if data_index < len(binary_message):
                    pixel[n] = pixel[n] & ~1 | int(binary_message[data_index])
                    data_index += 1
            encoded.putpixel((x, y), tuple(pixel))
            if data_index >= len(binary_message):
                break
        if data_index >= len(binary_message):
            break

    encoded.save(output_path)
    print("Message encoded and saved as", output_path)


if __name__ == "__main__":
    encode_image("data/input/sample.png", "data/output/stego.png", "Secret Key: 456")
