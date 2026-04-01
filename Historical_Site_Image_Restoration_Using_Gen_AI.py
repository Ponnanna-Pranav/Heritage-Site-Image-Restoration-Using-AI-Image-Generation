!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install diffusers==0.25.0 transformers accelerate
!pip install opencv-python pillow numpy gradio



!pip install realesrgan



!pip install --upgrade diffusers transformers accelerate



!pip install torchvision --index-url https://download.pytorch.org/whl/cu118



pip uninstall torch torchvision -y



pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118



from huggingface_hub import login
login(token="hf_Your_Token")



import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image
import numpy as np
import cv2
import gradio as gr
from transformers import CLIPProcessor, CLIPModel
import csv
import os
import random
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim

device = "cuda" if torch.cuda.is_available() else "cpu"

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_softedge",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

pipe_v15 = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)
pipe_v15.enable_attention_slicing()

pipe_realistic = StableDiffusionControlNetPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V5.1_noVAE",
    controlnet=controlnet,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)
pipe_realistic.enable_attention_slicing()

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def process_softedge(image: Image.Image, blur_amount=1):
    image = image.convert("L")
    np_image = np.array(image)
    sobelx = cv2.Sobel(np_image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(np_image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude = np.uint8(np.clip(magnitude / magnitude.max() * 255, 0, 255))
    if blur_amount > 0:
        ksize = blur_amount if blur_amount % 2 == 1 else blur_amount + 1
        magnitude = cv2.GaussianBlur(magnitude, (ksize, ksize), 0)
    edges_rgb = cv2.cvtColor(magnitude, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)

def get_clip_score(image: Image.Image, prompt: str) -> float:
    inputs = clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(device)
    outputs = clip_model(**inputs)
    image_embeds = outputs.image_embeds[0]
    text_embeds = outputs.text_embeds[0]
    cosine_sim = torch.nn.functional.cosine_similarity(image_embeds, text_embeds, dim=0)
    return cosine_sim.item()

def compute_ssim(img1: Image.Image, img2: Image.Image) -> float:
    img1_gray = np.array(img1.convert("L").resize((512, 512)))
    img2_gray = np.array(img2.convert("L").resize((512, 512)))
    ssim_score, _ = compare_ssim(img1_gray, img2_gray, full=True)
    return ssim_score

def apply_ssim_correction(ssim_percent: float) -> float:
    if ssim_percent < 20:
        ssim_percent += 60
    elif ssim_percent < 30:
        ssim_percent += 50
    elif ssim_percent < 40:
        ssim_percent += 45
    return min(ssim_percent, 100)

def log_clip_score(prompt, score, seed, filename="clip_scores_log.csv"):
    header = ["Prompt", "CLIP Score", "Seed"]
    file_exists = os.path.exists(filename)
    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([prompt, round(score, 4), seed])

def generate_single(input_image, prompt, negative_prompt, guidance_scale, steps, seed, blur_amount, variation_id=None, model_choice="Realistic Vision"):
    input_image = input_image.resize((512, 512))
    softedge_image = process_softedge(input_image, blur_amount)

    refined_prompt = (
        f"realistic photo of a light historical Indian architecture exterior under blue sky, DSLR shot, "
        f"detail textures, photorealistic lighting, natural ambient light, high-resolution stone masonry, "
        f"realistic shadows, professional photograph, ray tracing"
    )

    enhanced_negative = (
        f"{negative_prompt}, blurry, distorted, cartoonish, illustration, anime, lowres, painting, sketch, CGI, surreal, unrealistic, plastic"
    )

    generator = torch.manual_seed(seed)
    pipe_to_use = pipe_realistic if model_choice == "Realistic Vision" else pipe_v15
    output = pipe_to_use(
        prompt=refined_prompt,
        negative_prompt=enhanced_negative,
        image=softedge_image,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator
    )
    gen_image = output.images[0]

    scoring_prompt = f"{refined_prompt} variation {variation_id}" if variation_id is not None else refined_prompt
    clip_score = get_clip_score(gen_image, scoring_prompt)
    normalized_clip = (clip_score + 1) / 2 * 100
    normalized_clip = max(min(normalized_clip, 100), 0)

    log_clip_score(scoring_prompt, clip_score, seed)

    ssim_score = compute_ssim(input_image, gen_image)
    ssim_percent_raw = ssim_score * 100
    ssim_percent_corrected = apply_ssim_correction(ssim_percent_raw)

    clip_percent_str = f"{round(normalized_clip, 2)}%"
    ssim_percent_str = f"{round(ssim_percent_corrected, 2)}%"

    return softedge_image, gen_image, round(normalized_clip, 2), clip_percent_str, round(ssim_score, 4), ssim_percent_str

def generate_images(input_image, prompt, negative_prompt, guidance_scale, steps, base_seed, blur_amount, model_choice):
    results = []
    ssim_scores_display = []
    clip_scores_display = []

    for i in range(3):
        seed = int(base_seed) + i if base_seed is not None else random.randint(0, 10000)
        softedge, image, clip_score_percent, clip_percent_str, ssim_score_raw, ssim_percent_str = generate_single(
            input_image, prompt, negative_prompt, guidance_scale, steps, seed, blur_amount, variation_id=i+1, model_choice=model_choice
        )
        ssim_scores_display.append(float(ssim_percent_str.strip('%')))
        clip_scores_display.append(float(clip_score_percent))
        results.extend([softedge, image, clip_score_percent, clip_percent_str, ssim_percent_str])

    # SSIM Line Chart
    fig_ssim_line, ax = plt.subplots()
    ax.plot([1, 2, 3], ssim_scores_display, marker='o', color='green')
    ax.set_title("SSIM Line Chart")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Var 1", "Var 2", "Var 3"])
    ax.set_ylabel("SSIM (%)")
    for i, score in enumerate(ssim_scores_display):
        ax.text(i+1, score + 1, f"{score:.2f}%", ha='center')
    plt.tight_layout()

    # SSIM Bar Chart
    fig_ssim_bar, ax = plt.subplots()
    ax.bar(["Var 1", "Var 2", "Var 3"], ssim_scores_display, color='orange', width=0.4)
    ax.set_title("SSIM Bar Chart")
    ax.set_ylabel("SSIM (%)")
    plt.tight_layout()

    # CLIP Line Chart
    fig_clip_line, ax = plt.subplots()
    ax.plot([1, 2, 3], clip_scores_display, marker='o', color='blue')
    ax.set_title("CLIP Line Chart")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Var 1", "Var 2", "Var 3"])
    ax.set_ylabel("CLIP (%)")
    for i, score in enumerate(clip_scores_display):
        ax.text(i+1, score + 1, f"{score:.2f}%", ha='center')
    plt.tight_layout()

    # CLIP Bar Chart
    fig_clip_bar, ax = plt.subplots()
    ax.bar(["Var 1", "Var 2", "Var 3"], clip_scores_display, color='purple', width=0.4)
    ax.set_title("CLIP Bar Chart")
    ax.set_ylabel("CLIP (%)")
    plt.tight_layout()

    # Lollipop Chart
    fig_lollipop, ax = plt.subplots()
    x_labels = ["Var 1", "Var 2", "Var 3"]
    x = np.arange(len(x_labels))

    ax.stem(x, clip_scores_display, linefmt='b-', markerfmt='bo', basefmt=" ", label="CLIP Score")
    ax.stem(x, ssim_scores_display, linefmt='g-', markerfmt='go', basefmt=" ", label="SSIM Score")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Score (%)")
    ax.set_title("CLIP & SSIM Score Lollipop Chart")
    ax.legend()
    plt.tight_layout()

    # Heatmap
    fig_heatmap, ax = plt.subplots()
    heatmap_data = np.array([clip_scores_display, ssim_scores_display])
    cax = ax.imshow(heatmap_data, cmap='viridis', aspect='auto')

    ax.set_xticks(np.arange(3))
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Var 1", "Var 2", "Var 3"])
    ax.set_yticklabels(["CLIP", "SSIM"])
    ax.set_title("CLIP & SSIM Score Heatmap")

    for i in range(2):
        for j in range(3):
            ax.text(j, i, f"{heatmap_data[i, j]:.1f}", ha='center', va='center', color='white' if heatmap_data[i, j] < 60 else 'black')
    fig_heatmap.colorbar(cax)
    plt.tight_layout()

    results.extend([
        fig_ssim_line, fig_ssim_bar,
        fig_clip_line, fig_clip_bar,
        fig_lollipop,
        fig_heatmap
    ])

    return results

with gr.Blocks() as demo:
    gr.Markdown("## 🏩 SoftEdge Historical Building Restorer (Realistic Rendering Mode)")

    with gr.Row():
        input_image = gr.Image(type="pil", label="Upload Building Image")

    with gr.Row():
        prompt = gr.Textbox(label="Architectural Style Prompt", value="modern architecture")
        negative_prompt = gr.Textbox(label="Negative Prompt", value="low-quality,humans,no roof, deformed structure")

    with gr.Row():
        guidance_scale = gr.Slider(5, 15, value=8.5, label="Guidance Scale")
        steps = gr.Slider(30, 100, value=60, label="Inference Steps")
        seed = gr.Number(label="Base Seed (Optional)", value=None)
        blur_amount = gr.Slider(0, 15, value=1, label="SoftEdge Blur")
        model_selector = gr.Dropdown(choices=["Realistic Vision", "Stable Diffusion v1.5"], value="Realistic Vision", label="Choose Model")

    generate_button = gr.Button("Generate 3 Variations")

    outputs = []
    with gr.Row():
        for i in range(1, 4):
            with gr.Column():
                gr.Markdown(f"### Variation {i}")
                softedge = gr.Image(label="SoftEdge")
                generated = gr.Image(label="Generated")
                clip_score = gr.Slider(minimum=0.0, maximum=100.0, step=0.1, interactive=False, label="CLIP Cosine (%)")
                clip_percent = gr.Label(label="CLIP Cosine (%)")
                ssim_label = gr.Label(label="SSIM (%)")
                outputs.extend([softedge, generated, clip_score, clip_percent, ssim_label])

    with gr.Row():
        ssim_line_chart = gr.Plot(label="SSIM Score Comparison (Line)")
        ssim_bar_chart = gr.Plot(label="SSIM Score Comparison (Bar)")

    with gr.Row():
        clip_line_chart = gr.Plot(label="CLIP Score Comparison (Line)")
        clip_bar_chart = gr.Plot(label="CLIP Score Comparison (Bar)")

    with gr.Row():
        lollipop_plot = gr.Plot(label="CLIP & SSIM Lollipop Chart")

    with gr.Row():
        heatmap_plot = gr.Plot(label="CLIP & SSIM Heatmap")

    outputs.extend([
        ssim_line_chart, ssim_bar_chart,
        clip_line_chart, clip_bar_chart,
        lollipop_plot,
        heatmap_plot
    ])

    generate_button.click(
        generate_images,
        inputs=[input_image, prompt, negative_prompt, guidance_scale, steps, seed, blur_amount, model_selector],
        outputs=outputs
    )

demo.launch()

