from main import setup, generate_imgs, generate_from_imgs, generate_prompt, get_manipulations
from PIL import Image
from filter_images import cartoonify, pencil, oil_painting, watercolor, black_and_white, sepia, blue_tone, xray_effect
from watermark import create_watermark
from transparent import segment_head
import torch

def apply_filters(images, filter_checkbox):
    filtered = []
    for image in images:

        for filter in filter_checkbox:

            if filter == "Cartoon":
                filtered.append(cartoonify(image))

            if filter == "Pencil Sketch":
                filtered.append(pencil(image))

            if filter == "Oil Painting":
                filtered.append(oil_painting(image))

            if filter == "Watercolor":
                filtered.append(watercolor(image))
                    
            if filter == "Xray":
                filtered.append(xray_effect(image))

            if filter == "Black and White":
                filtered.append(black_and_white(image))

            if filter == "Sepia":
                filtered.append(sepia(image))

            if filter == "Blue Tone":
                filtered.append(blue_tone(image))
    return filtered

def load_generator_from_checkpoint(checkpoint_path):
    return setup(checkpoint_path)

def convert_np_image_to_tensor(image_np):
    return torch.from_numpy(image_np).permute(2, 0, 1).float().div(127.5).sub(1)

def perform_manipulations(images, resolution_dropdown,filter_checkbox, head_checkbox):
    heads = []
    if head_checkbox:
        for img in images:
            heads.append(segment_head(img))
    
    filtered = []
    if len(filter_checkbox) > 0:
        filtered = apply_filters(images, filter_checkbox)
    
    images = images + heads + filtered


    if resolution_dropdown == '256x256':
        watermarked = []
        for img in images:
            watermarked.append(create_watermark(img.resize((256, 256), Image.LANCZOS)))
        return watermarked
    
    if resolution_dropdown == 'all':
        all_res = []
        for img in images:
            all_res.append(create_watermark(img.resize((256, 256), Image.LANCZOS)))
            all_res.append(create_watermark(img.resize((512, 512), Image.LANCZOS)))
            all_res.append(create_watermark(img.resize((1024, 1024), Image.LANCZOS)))
        return all_res
    
    resolution = 256
    if resolution_dropdown == '512x512':
        resolution = 512
    elif resolution_dropdown == '1024x1024':
        resolution = 1024
    
    resized_images = []
    for img in images:
        resized_images.append(create_watermark(img.resize((resolution, resolution), Image.LANCZOS)))

    return resized_images
    
def image_to_image(image, num_images,resolution_dropdown,filter_checkbox, head_checkbox, gen, mapping_network):
    images = generate_from_imgs(image,num_images, gen, mapping_network)
    return perform_manipulations(images, resolution_dropdown,filter_checkbox, head_checkbox)

def generate_random_images(num_images,resolution_dropdown,filter_checkbox, head_checkbox, gen, mapping_network):
    images = generate_imgs(num_images, gen, mapping_network)
    return perform_manipulations(images, resolution_dropdown,filter_checkbox, head_checkbox)

def generate_from_prompt(prompt, num_images_dropdown, resolution_dropdown, filter_checkbox, head_checkbox, gen, mapping_network):
    images = generate_prompt(prompt, num_images_dropdown, gen, mapping_network)
    return perform_manipulations(images, resolution_dropdown,filter_checkbox, head_checkbox)

def manipulate_imgs(scale, num_images_dropdown, resolution_dropdown, filter_checkbox, head_checkbox, gen):
    images = get_manipulations(scale, num_images_dropdown, gen)
    return perform_manipulations(images, resolution_dropdown,filter_checkbox, head_checkbox)

    

if __name__ == '__main__':
    num_images=1
    gen, mapping_network = load_generator_from_checkpoint('/app/facecraft_256x256_24.pth')
    images = generate_random_images(num_images, gen, mapping_network)
    
