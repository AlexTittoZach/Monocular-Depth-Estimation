import sys
import os
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any

from huggingface_hub import hf_hub_download

# Add external Depth-Anything-V2 to python path
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../external/Depth-Anything-V2")
    )
)

from depth_anything_v2.dpt import DepthAnythingV2

MODEL_CONFIGS = {
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    },
    "vitb": {
        "encoder": "vitb",
        "features": 128,
        "out_channels": [96, 192, 384, 768],
    },
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
}

MODEL_REPO = "alextittozach/depth-anything-v2-kitti-models"

DEFAULT_CHECKPOINTS = {
    "vitl": "pretrained/depth_anything_v2_vitl.pth",
}


def resolve_checkpoint(checkpoint_path: str):
    """
    Returns a local checkpoint path.

    If checkpoint_path is already a local file,
    it is returned directly.

    Otherwise it is downloaded from Hugging Face Hub.
    """

    if checkpoint_path is None:
        return None

    if os.path.isfile(checkpoint_path):
        print(f"[Checkpoint] Using local checkpoint: {checkpoint_path}")
        return checkpoint_path

    print(f"[Checkpoint] Downloading '{checkpoint_path}' from HF Hub...")

    local_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=checkpoint_path,
        repo_type="model",
    )

    print(f"[Checkpoint] Cached at: {local_path}")

    return local_path


def load_checkpoint(model, ckpt_path, title):

    print(f"\n==============================")
    print(f"Loading {title}")
    print(f"{ckpt_path}")
    print(f"==============================")

    state_dict = torch.load(ckpt_path, map_location="cpu")

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
    )

    print(f"Missing keys    : {len(missing)}")
    print(f"Unexpected keys : {len(unexpected)}")

    if missing:
        print("First Missing:")
        print(missing[:10])

    if unexpected:
        print("First Unexpected:")
        print(unexpected[:10])

    return missing, unexpected


def load_model(
    mode: str = "baseline",
    encoder_type: str = "vitl",
    checkpoint_path: str = None,
    lora_r: int = 16,
    lora_alpha: int = 32,
) -> Tuple[nn.Module, Dict[str, Any]]:

    if encoder_type not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown encoder_type '{encoder_type}'. "
            f"Available: {list(MODEL_CONFIGS.keys())}"
        )

    config = MODEL_CONFIGS[encoder_type]

    model = DepthAnythingV2(**config)

    if encoder_type not in DEFAULT_CHECKPOINTS:
        raise ValueError(
            f"No default checkpoint configured for {encoder_type}"
        )

    stock_checkpoint = resolve_checkpoint(
        DEFAULT_CHECKPOINTS[encoder_type]
    )

    if checkpoint_path is not None:
        checkpoint_path = resolve_checkpoint(checkpoint_path)

    # Always load official pretrained weights first
    load_checkpoint(
        model,
        stock_checkpoint,
        "Official Pretrained Checkpoint",
    )

    if mode == "baseline":

        for p in model.parameters():
            p.requires_grad = False

        print("[Experiment Config] Baseline Mode")

    elif mode == "decoder":

        for p in model.pretrained.parameters():
            p.requires_grad = False

        for p in model.depth_head.parameters():
            p.requires_grad = True

        print("[Experiment Config] Decoder Only")

        if checkpoint_path is not None and checkpoint_path != stock_checkpoint:

            load_checkpoint(
                model,
                checkpoint_path,
                "Decoder Fine-tuned Checkpoint",
            )

    elif mode == "lora":

        for p in model.parameters():
            p.requires_grad = False

        from peft import (
            LoraConfig,
            get_peft_model,
        )

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=[
                "qkv",
                "proj",
                "q",
                "v",
            ],
            lora_dropout=0.05,
            bias="none",
        )

        model.pretrained = get_peft_model(
            model.pretrained,
            lora_config,
        )

        print(
            f"[Experiment Config] LoRA (r={lora_r}, alpha={lora_alpha})"
        )

        if checkpoint_path is not None and checkpoint_path != stock_checkpoint:

            load_checkpoint(
                model,
                checkpoint_path,
                "LoRA Fine-tuned Checkpoint",
            )

    elif mode == "full":

        for p in model.parameters():
            p.requires_grad = True

        print("[Experiment Config] Full Fine-Tuning")

        if checkpoint_path is not None and checkpoint_path != stock_checkpoint:

            load_checkpoint(
                model,
                checkpoint_path,
                "Full Fine-tuned Checkpoint",
            )

    else:

        raise ValueError(
            f"Invalid mode '{mode}'. "
            "Choose from ['baseline','decoder','lora','full']"
        )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    info = {

        "encoder": encoder_type,

        "mode": mode,

        "checkpoint_path": checkpoint_path
        if checkpoint_path is not None
        else stock_checkpoint,

        "total_params": total_params,

        "trainable_params": trainable_params,

        "trainable_ratio":
            trainable_params
            / total_params
            * 100.0,
    }

    return model, info


# Backward compatibility
build_depth_anything_v2 = load_model