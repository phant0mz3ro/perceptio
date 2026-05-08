from PIL import Image

# ASCII characters from dark → light
ASCII_CHARS = "@%#*+=-:. "

def resize_image(image, new_width=100):
    width, height = image.size
    ratio = height / width
    new_height = int(new_width * ratio * 0.55)  # adjust for font shape
    return image.resize((new_width, new_height))

def grayscale(image):
    return image.convert("L")

def pixels_to_ascii(image):
    pixels = image.getdata()
    ascii_str = ""
    for pixel in pixels:
        ascii_str += ASCII_CHARS[pixel // 25]
    return ascii_str

def image_to_ascii(path):
    image = Image.open(path)

    image = resize_image(image)
    image = grayscale(image)

    ascii_str = pixels_to_ascii(image)

    width = image.width
    ascii_img = "\n".join(
        ascii_str[i:(i+width)] for i in range(0, len(ascii_str), width)
    )

    return ascii_img

# Run it
ascii_art = image_to_ascii("c:\\Users\\ASUS\\Documents\\perceptio\\pillow\\your_image.jpeg")
print(ascii_art)

# Optional: save to file
with open("ascii_art.txt", "w") as f:
    f.write(ascii_art)