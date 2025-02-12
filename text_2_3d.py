import os
import torch
import gradio as gr
import tempfile
import numpy as np
import rembg
from PIL import Image
from diffusers import FluxPipeline
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True, max_split_size_mb:64'
device = "cuda" if torch.cuda.is_available() else "cpu"

def free_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()

flux_pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16, 
)
flux_pipe.enable_model_cpu_offload()
flux_pipe.enable_sequential_cpu_offload()  
flux_pipe.enable_attention_slicing()

tsr_model = TSR.from_pretrained(
    "stabilityai/TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt",
)
tsr_model.renderer.set_chunk_size(8192)
tsr_model.to(device)

rembg_session = rembg.new_session()

def fill_background(image):
    image_np = np.array(image).astype(np.float32) / 255.0
    if image_np.shape[2] == 4:
        image_np = image_np[:, :, :3] * image_np[:, :, 3:4] + (1 - image_np[:, :, 3:4]) * 0.5
    filled = Image.fromarray((image_np * 255.0).astype(np.uint8))
    return filled

def preprocess_image(input_image, do_remove_bg, foreground_ratio):

    if do_remove_bg:
        image = input_image.convert("RGB")
        image = remove_background(image, rembg_session)
        image = resize_foreground(image, foreground_ratio)
        image = fill_background(image)
    else:
        image = input_image
        if image.mode == "RGBA":
            image = fill_background(image)
    return image

def export_mesh_files(mesh, formats=["obj", "glb"]):
    exported_files = {}
    for fmt in formats:
        temp_file = tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False)
        mesh.export(temp_file.name)
        exported_files[fmt] = temp_file.name
    return exported_files

def text_to_3d(text_prompt, flux_inference_steps, remove_bg, foreground_ratio, mc_resolution):

    free_gpu_memory()

    if not text_prompt.strip():
        text_prompt = "digital illustration"
    optimized_prompt = (
        f"{text_prompt}, detailed digital illustration, minimalistic style, "
        "smooth gray background, plain colors, high quality, vector art"
    )
    with torch.no_grad():
        flux_result = flux_pipe(
            prompt=optimized_prompt,
            height=512,
            width=512,
            num_inference_steps=flux_inference_steps,
        )
    gen_image = flux_result.images[0]
    free_gpu_memory()  

    preprocessed_image = preprocess_image(gen_image, remove_bg, foreground_ratio)
    
    free_gpu_memory()  

    with torch.no_grad():
        scene_codes = tsr_model([preprocessed_image], device=device)
        mesh = tsr_model.extract_mesh(scene_codes, True, resolution=mc_resolution)[0]
    mesh = to_gradio_3d_orientation(mesh)
    free_gpu_memory()  

    exported_files = export_mesh_files(mesh, formats=["obj", "glb"])
    free_gpu_memory()

    return gen_image, exported_files["obj"], exported_files["glb"]

with gr.Blocks(title="Text-to-3D Generator") as demo:
    gr.Markdown(
        """
        # Text-to-3D Generator
        Enter a text prompt below to generate a 2D illustration using the FLUX.1 model,
        and then convert it into a 3D model using TripoSR.
        """
    )
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(label="Text Prompt", placeholder="Enter your prompt here...", lines=2)
            flux_steps_slider = gr.Slider(label="FLUX Inference Steps", minimum=2, maximum=20, step=1, value=2)
            remove_bg_checkbox = gr.Checkbox(label="Remove Background", value=True)
            foreground_ratio_slider = gr.Slider(label="Foreground Ratio", minimum=0.5, maximum=1.0, step=0.05, value=0.85)
            mc_resolution_slider = gr.Slider(label="Marching Cubes Resolution", minimum=32, maximum=320, step=32, value=256)
            generate_button = gr.Button("Generate 3D Model")
        with gr.Column():
            gen_image_output = gr.Image(label="Generated 2D Image", interactive=False)
            output_obj = gr.Model3D(label="3D Model (OBJ Format)", interactive=False)
            output_glb = gr.Model3D(label="3D Model (GLB Format)", interactive=False)
    
    generate_button.click(
        fn=text_to_3d,
        inputs=[text_input, flux_steps_slider, remove_bg_checkbox, foreground_ratio_slider, mc_resolution_slider],
        outputs=[gen_image_output, output_obj, output_glb],
    )

if __name__ == "__main__":
    demo.launch()
