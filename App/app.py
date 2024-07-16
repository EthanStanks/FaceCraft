import gradio as gr
from css import css
import generator
import torch
import cv2

gr.set_static_paths(['.'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
gen_1, mapping_network_1 = generator.load_generator_from_checkpoint('/app/App/Checkpoint/facecraft_256x256_1.pth')
gen_40, mapping_network_40 = generator.load_generator_from_checkpoint('/app/App/Checkpoint/facecraft_256x256_40.pth')
gen_86, mapping_network_86 = generator.load_generator_from_checkpoint('/app/App/Checkpoint/facecraft_256x256_86.pth')
gen_120, mapping_network_120 = generator.load_generator_from_checkpoint('/app/App/Checkpoint/facecraft_256x256_120.pth')
gen_140, mapping_network_140 = generator.load_generator_from_checkpoint('/app/App/Checkpoint/facecraft_256x256_140.pth')

def get_checkpoint(checkpoint_dropdown):
    if checkpoint_dropdown == "epoch 1": return gen_1, mapping_network_1
    elif checkpoint_dropdown == "epoch 40": return gen_40, mapping_network_40
    elif checkpoint_dropdown == "epoch 86": return gen_86, mapping_network_86
    elif checkpoint_dropdown == "epoch 120": return gen_120, mapping_network_120
    else: return gen_140, mapping_network_140

def generate_prompt(prompt, num_images_dropdown, resolution_dropdown, filter_checkbox, head_checkbox, checkpoint_dropdown):
    gen, mapping_network = get_checkpoint(checkpoint_dropdown)
    return generator.generate_from_prompt(prompt, num_images_dropdown, resolution_dropdown, filter_checkbox, head_checkbox, gen, mapping_network)

def generate_random(num_images_dropdown, resolution_dropdown, filter_checkbox, head_checkbox, checkpoint_dropdown):
    gen, mapping_network = get_checkpoint(checkpoint_dropdown)
    return generator.generate_random_images(num_images_dropdown, resolution_dropdown, filter_checkbox, head_checkbox,gen, mapping_network)

def generate_from_image(image, num_images_dropdown, resolution_dropdown, filter_checkbox, head_checkbox, checkpoint_dropdown):
    gen, mapping_network = get_checkpoint(checkpoint_dropdown)
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
            checkpoint_dropdown = gr.Dropdown(choices=["epoch 1", "epoch 40", "epoch 86","epoch 120","epoch 140"], label="Change Training Checkpoint", value="epoch 140", interactive=True)
            resolution_dropdown = gr.Dropdown(choices=['256x256','512x512', '1024x1024', 'all'], label="Image Resolution to Generate", value='256x256', interactive=True)
            filter_checkbox = gr.CheckboxGroup(["Cartoon", "Pencil Sketch", "Watercolor", "Xray", "Black and White", "Sepia", "Blue Tone"], label="Apply Filters")
            head_checkbox = gr.Checkbox(label="Transparent Background", value=False)

    prompt_input = gr.Textbox(lines=1, placeholder="Your prompt...", visible=True, show_label=False)
    with gr.Row():
        btn_prompt = gr.Button("Prompt Generate")
        btn_random = gr.Button("Random Generate")
        btn_prompt.click(fn=generate_prompt, inputs=[prompt_input,num_images_dropdown, resolution_dropdown, filter_checkbox, head_checkbox, checkpoint_dropdown], outputs=[gallery])
        btn_random.click(fn=generate_random, inputs=[num_images_dropdown, resolution_dropdown, filter_checkbox, head_checkbox, checkpoint_dropdown], outputs=[gallery])

    image_label = gr.Label("Upload an image to generate a realistic face.", scale=0,show_label=False)
    input_image = gr.Image(show_download_button=False, scale=0,show_label=False)
    input_image.upload(fn=generate_from_image, inputs=[input_image,num_images_dropdown, resolution_dropdown, filter_checkbox, head_checkbox, checkpoint_dropdown], outputs=[gallery])

    gr.HTML("""
                <footer class="custom-footer">
                    <p>Developed By</p>
                    <p>Will Hoover | Temitayo Shorunke | Ethan Stanks</p>
                </footer>
            """)

demo.launch(server_port=7861)
