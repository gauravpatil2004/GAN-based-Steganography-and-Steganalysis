from PIL import Image

def decode_image(stego_path):
    image = Image.open(stego_path)
    binary_message = ""
    for y in range(image.height):
        for x in range(image.width):
            pixel = list(image.getpixel((x, y)))
            for n in range(3):
                binary_message += str(pixel[n] & 1)

    # convert binary to text until "####"
    message = ""
    for i in range(0, len(binary_message), 8):
        byte = binary_message[i:i+8]
        char = chr(int(byte, 2))
        message += char
        if message.endswith("####"):
            return message[:-4]

    return message


if __name__ == "__main__":
    hidden_message = decode_image("data/output/stego.png")
    print("Hidden message:", hidden_message)
