import gradio as gr
from css import css
import generator
import torch
import cv2

gr.set_static_paths(['.'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
gen, mapping_network = generator.load_generator_from_checkpoint('/app/App/Checkpoint/facecraft_256x256_80.pth')

def generate_random(num_images_dropdown, resolution_dropdown, filter_checkbox, head_checkbox):
    return generator.generate_random_images(num_images_dropdown, resolution_dropdown, filter_checkbox, head_checkbox,gen, mapping_network)

def generate_from_image(image, num_images_dropdown, resolution_dropdown, filter_checkbox, head_checkbox):
    resized_image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    return generator.image_to_image(resized_image, num_images_dropdown,resolution_dropdown,filter_checkbox, head_checkbox, gen, mapping_network)

with gr.Blocks(title="FaceCraft",css=css) as demo:
    gr.HTML("""
                <header>
                    <img class="logo" src="file/Assets/logo.png" />
                    <h1>FaceCraft</h1>
                </header>
            """)

    with gr.Row():
        gallery = gr.Gallery(interactive=False)

    label = gr.Label("Click \"Generate\" to generate a realistic face.", scale=0,show_label=False)
    with gr.Accordion("Settings"):
        with gr.Row():
            num_images_dropdown = gr.Dropdown(choices=[1, 2, 3, 4], label="Number of Images to Generate", value=1, interactive=True)
            resolution_dropdown = gr.Dropdown(choices=['256x256','512x512', '1024x1024', 'all'], label="Image Resolution to Generate", value='256x256', interactive=True)
            filter_checkbox = gr.CheckboxGroup(["Cartoon", "Pencil Sketch", "Watercolor", "Xray", "Black and White", "Sepia", "Blue Tone"], label="Apply Filters")
            head_checkbox = gr.Checkbox(label="Transparent Background", value=False)

    with gr.Row():
        btn_generate = gr.Button("Generate")
        btn_generate.click(fn=generate_random, inputs=[num_images_dropdown, resolution_dropdown, filter_checkbox, head_checkbox], outputs=[gallery])

    image_label = gr.Label("Upload an image to generate a realistic face.", scale=0,show_label=False)
    input_image = gr.Image(show_download_button=False, scale=0,show_label=False)
    input_image.upload(fn=generate_from_image, inputs=[input_image,num_images_dropdown, resolution_dropdown, filter_checkbox, head_checkbox], outputs=[gallery])

    gr.HTML("""
                <footer class="custom-footer">
                    <p>Developed By</p>
                    <p>Will Hoover | Temitayo Shorunke | Ethan Stanks</p>
                </footer>
            """)

demo.launch()
