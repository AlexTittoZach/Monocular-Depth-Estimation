import sys
import os
import time
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import gradio as gr
import spaces
from PIL import Image

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.models.depth_anything import load_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOADED_MODELS = {}

MODEL_CHECKPOINTS = {
    "Baseline (Pretrained Zero-Shot)": (
        "baseline",
        "pretrained/depth_anything_v2_vitl.pth",
    ),
    "Decoder-Only Fine-Tuned": (
        "decoder",
        "finetuned/depth_anything_v2_vitl_decoder.pth",
    ),
    "LoRA Adapter": (
        "lora",
        "finetuned/depth_anything_v2_vitl_lora.pth",
    ),
    "Full Fine-Tuned": (
        "full",
        "finetuned/depth_anything_v2_vitl_full.pth",
    )
}


def get_model(strategy_name: str):
    if strategy_name in LOADED_MODELS:
        return LOADED_MODELS[strategy_name]

    mode, ckpt_path = MODEL_CHECKPOINTS[strategy_name]
    model, _ = load_model(
        mode=mode,
        encoder_type="vitl",
        checkpoint_path=ckpt_path,
    )
    model.to(DEVICE)
    model.eval()
    LOADED_MODELS[strategy_name] = model
    return model

def colorize_depth(depth: np.ndarray, colormap: str = "inferno") -> np.ndarray:
    d_min, d_max = depth.min(), depth.max()
    norm_depth = (depth - d_min) / (d_max - d_min + 1e-8)
    norm_depth_uint8 = (norm_depth * 255).astype(np.uint8)

    cmap_dict = {
        "inferno": cv2.COLORMAP_INFERNO,
        "magma": cv2.COLORMAP_MAGMA,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "plasma": cv2.COLORMAP_PLASMA,
        "gray": None,
    }

    cv_cmap = cmap_dict.get(colormap, cv2.COLORMAP_INFERNO)

    if cv_cmap is None:
        return np.stack([norm_depth_uint8] * 3, axis=-1)

    colorized = cv2.applyColorMap(norm_depth_uint8, cv_cmap)
    return cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)


@spaces.GPU
def predict_depth(input_image: Image.Image, strategy_name: str, colormap: str):
    if input_image is None:
        return None, "Please upload an input image."

    orig_w, orig_h = input_image.size
    img_np = np.array(input_image.convert("RGB"))

    resized = cv2.resize(img_np, (504, 378))

    img_tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    img_tensor = ((img_tensor - mean) / std).unsqueeze(0).to(DEVICE)

    start = time.time()

    model = get_model(strategy_name)

    with torch.no_grad():
        prediction = model(img_tensor)
        prediction = F.interpolate(
            prediction[:, None],
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=True,
        )[0, 0]

    latency = (time.time() - start) * 1000

    depth = prediction.cpu().numpy()
    print("\n========================")
    print(strategy_name)
    print("min :", depth.min())
    print("max :", depth.max())
    print("mean:", depth.mean())
    print("std :", depth.std())
    print("========================")
    color_depth = colorize_depth(depth, colormap)


    return color_depth, stats


with gr.Blocks(title="Depth Anything V2 Benchmark Suite") as demo:

    gr.Markdown("""
# 🚀 Depth Anything V2 Monocular Depth Estimation

Compare zero-shot, decoder-only, LoRA, and full fine-tuning models.
""")

    with gr.Row():

        with gr.Column():
            input_img = gr.Image(type="pil", label="Upload RGB Image")

            strategy = gr.Radio(
                list(MODEL_CHECKPOINTS.keys()),
                value="LoRA Adapter",
                label="Fine-Tuning Strategy",
            )

            colormap = gr.Dropdown(
                ["inferno", "magma", "viridis", "plasma", "gray"],
                value="inferno",
                label="Depth Colormap",
            )

            btn = gr.Button("Estimate Depth", variant="primary")

        with gr.Column():
            output_img = gr.Image(type="numpy", label="Depth Map")
            metrics = gr.Markdown()

    btn.click(
        fn=predict_depth,
        inputs=[input_img, strategy, colormap],
        outputs=[output_img, metrics],
    )


if __name__ == "__main__":
    demo.launch()