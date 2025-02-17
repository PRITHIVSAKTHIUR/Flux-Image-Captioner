import spaces
import gradio as gr
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, pipeline
from diffusers import DiffusionPipeline
import random
import numpy as np
import os
from qwen_vl_utils import process_vision_info

# Initialize models
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16

huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

# FLUX.1-dev model
pipe = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=dtype, token=huggingface_token
).to(device)

# Initialize Qwen2VL model
qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "prithivMLmods/JSONify-Flux", trust_remote_code=True, torch_dtype=torch.float16
).to(device).eval()
qwen_processor = AutoProcessor.from_pretrained("prithivMLmods/JSONify-Flux", trust_remote_code=True)

# Prompt Enhancer
enhancer_long = pipeline("summarization", model="prithivMLmods/t5-Flan-Prompt-Enhance", device=device)

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048

# Qwen2VL caption function – updated to request plain text caption instead of JSON
@spaces.GPU
def qwen_caption(image):
    # Convert image to PIL if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Generate a detailed and optimized caption for the given image."},
            ],
        }
    ]

    text = qwen_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = qwen_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    generated_ids = qwen_model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = qwen_processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    
    return output_text

# Prompt Enhancer function (unchanged)
def enhance_prompt(input_prompt):
    result = enhancer_long("Enhance the description: " + input_prompt)
    enhanced_text = result[0]['summary_text']
    return enhanced_text

@spaces.GPU
def process_workflow(image, text_prompt, use_enhancer, seed, randomize_seed, width, height, guidance_scale, num_inference_steps, progress=gr.Progress(track_tqdm=True)):
    if image is not None:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        prompt = qwen_caption(image)
        print(prompt)
    else:
        prompt = text_prompt
    
    if use_enhancer:
        prompt = enhance_prompt(prompt)
    
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    
    generator = torch.Generator(device=device).manual_seed(seed)
    
    torch.cuda.empty_cache()
    
    try:
        image = pipe(
            prompt=prompt,
            generator=generator,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            guidance_scale=guidance_scale
        ).images[0]
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            raise RuntimeError("CUDA out of memory. Try reducing image size or inference steps.")
        else:
            raise e
    
    return image, prompt, seed

custom_css = """
.input-group, .output-group {

}
.submit-btn {
    background: linear-gradient(90deg, #4B79A1 0%, #283E51 100%) !important;
    border: none !important;
    color: white !important;
}
.submit-btn:hover {
    background-color: #3498db !important;
}
"""

title = """<h1 align="center">FLUX.1-dev with Qwen2VL Captioner and Prompt Enhancer</h1>
<p><center>
<a href="https://huggingface.co/black-forest-labs/FLUX.1-dev" target="_blank">[FLUX.1-dev Model]</a>
<a href="https://huggingface.co/prithivMLmods/JSONify-Flux" target="_blank">[JSONify Flux Model]</a>
<a href="https://huggingface.co/prithivMLmods/t5-Flan-Prompt-Enhance" target="_blank">[Prompt Enhancer t5]</a>
<p align="center">Create long prompts from images or enhance your short prompts with prompt enhancer</p>
</center></p>
"""

with gr.Blocks(css=custom_css) as demo:
    gr.HTML(title)
    
    with gr.Sidebar(label="Parameters", open=True):
        gr.Markdown(
            """
            ### About
            
            #### Flux.1-Dev
            FLUX.1 [dev] is a 12 billion parameter rectified flow transformer capable of generating images from text descriptions. FLUX.1 [dev] is an open-weight, guidance-distilled model for non-commercial applications. Directly distilled from FLUX.1 [pro], FLUX.1 [dev] obtains similar quality and prompt adherence capabilities, while being more efficient than a standard model of the same size.  
            [FLUX.1-dev Model](https://huggingface.co/black-forest-labs/FLUX.1-dev)
            
            #### JSONify-Flux
            JSONify-Flux is a multimodal image-text-text model trained on a dataset of FLUX-generated images with context-rich captions based on the Qwen2VL architecture. The JSON-based instruction has been manually removed to avoid JSON format captions.  
            [JSONify-Flux Model](https://huggingface.co/prithivMLmods/JSONify-Flux)
            
            #### t5-Flan-Prompt-Enhance
            t5-Flan-Prompt-Enhance is a prompt summarization model that enriches synthetic FLUX prompts with more detailed descriptions.  
            [t5-Flan-Prompt-Enhance Model](https://huggingface.co/prithivMLmods/t5-Flan-Prompt-Enhance)
            """
        )
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group(elem_classes="input-group"):
                input_image = gr.Image(label="Input Image (Qwen2VL Captioner)")
            
            with gr.Accordion("Advanced Settings", open=False):
                text_prompt = gr.Textbox(label="Text Prompt (optional, used if no image is uploaded)")
                use_enhancer = gr.Checkbox(label="Use Prompt Enhancer", value=False)
                seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
                randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
                width = gr.Slider(label="Width", minimum=256, maximum=MAX_IMAGE_SIZE, step=32, value=512)
                height = gr.Slider(label="Height", minimum=256, maximum=MAX_IMAGE_SIZE, step=32, value=512)
                guidance_scale = gr.Slider(label="Guidance Scale", minimum=1, maximum=15, step=0.1, value=3.5)
                num_inference_steps = gr.Slider(label="Inference Steps", minimum=1, maximum=50, step=1, value=32)
            
            generate_btn = gr.Button("Generate Image & Prompt", elem_classes="submit-btn")
        
        with gr.Column(scale=1):
            with gr.Group(elem_classes="output-group"):
                output_image = gr.Image(label="result", elem_id="gallery", show_label=False)
                final_prompt = gr.Textbox(label="prompt")
                used_seed = gr.Number(label="seed")
    
    generate_btn.click(
        fn=process_workflow,
        inputs=[
            input_image, text_prompt, use_enhancer, seed, randomize_seed,
            width, height, guidance_scale, num_inference_steps
        ],
        outputs=[output_image, final_prompt, used_seed]
    )

demo.launch(debug=True)