import numpy as np
from PIL import Image, ImageDraw

def create_watermark(image):
    # open logo, scale, change opacity
    logo = Image.open('Assets/logo.png').convert("RGBA")
    scale_factor = min(image.width, image.height) / 8
    new_size = (int(logo.width * scale_factor / logo.width), int(logo.height * scale_factor / logo.height))
    logo = logo.resize(new_size, Image.LANCZOS)
    logo_opacity = 220  # 0-255
    r, g, b, a = logo.split()
    a = a.point(lambda i: i * logo_opacity // 255)
    logo = Image.merge('RGBA', (r, g, b, a))

    # create logo PIL image
    watermark_image = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(watermark_image)
    x_logo = image.width - logo.width - 10
    y_logo = image.height - logo.height - 10
    watermark_image.paste(logo, (x_logo, y_logo), logo)

    # combine
    combined = Image.alpha_composite(image.convert('RGBA'), watermark_image)

    return combined
