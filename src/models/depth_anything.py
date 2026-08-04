import sys
import os
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any

# Add external Depth-Anything-V2 to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../external/Depth-Anything-V2")))
from depth_anything_v2.dpt import DepthAnythingV2

MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}

import pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINTS = {
    'vits': str(PROJECT_ROOT / 'checkpoints' / 'pretrained' / 'depth_anything_v2_vits.pth'),
    'vitb': str(PROJECT_ROOT / 'checkpoints' / 'pretrained' / 'depth_anything_v2_vitb.pth'),
    'vitl': str(PROJECT_ROOT / 'checkpoints' / 'pretrained' / 'depth_anything_v2_vitl.pth'),
}

def load_official_checkpoint(model: nn.Module, checkpoint_path: str):
    """
    Loads official pretrained checkpoint and verifies state dict key alignment.
    """
    if not os.path.exists(checkpoint_path):
        print(f"[Warning] Official checkpoint not found at '{checkpoint_path}'. Skipping loading.")
        return
        
    print(f"[Model Loader] Loading official checkpoint from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[Model Verification] Missing keys: {len(missing)} | {missing}")
    print(f"[Model Verification] Unexpected keys: {len(unexpected)} | {unexpected}")

def load_model(
    mode: str = "baseline",
    encoder_type: str = "vitl",
    checkpoint_path: str = None,
    lora_r: int = 16,
    lora_alpha: int = 32
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Single unified function for loading model and applying experiment strategy.
    Supports baseline, decoder-only, lora, and full fine-tuning.
    """
    if encoder_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown encoder_type '{encoder_type}'. Options: {list(MODEL_CONFIGS.keys())}")
        
    config = MODEL_CONFIGS[encoder_type]
    model = DepthAnythingV2(**config)
    
    stock_checkpoint = DEFAULT_CHECKPOINTS.get(encoder_type, 'checkpoints/pretrained/depth_anything_v2_vitl.pth')

    if checkpoint_path and os.path.exists(checkpoint_path): 
        stock_checkpoint = checkpoint_path

    # 1. Load official stock weights first into base architecture, if it exists
    if os.path.exists(stock_checkpoint):
        load_official_checkpoint(model, stock_checkpoint)
    else:
        print(f"[Warning] Official checkpoint not found at '{stock_checkpoint}'. Skipping loading.")
    
    # 2. Configure experiment mode
    if mode == "baseline":
        for p in model.parameters():
            p.requires_grad = False
        print("[Experiment Config] Baseline Mode: 100% of parameters frozen.")
        
    elif mode == "decoder":
        for p in model.pretrained.parameters():
            p.requires_grad = False
        for p in model.depth_head.parameters():
            p.requires_grad = True
        print("[Experiment Config] Decoder-Only Mode: Encoder frozen, DPT Depth Head unfrozen.")
        
        # If fine-tuned checkpoint provided, load fine-tuned weights
        if checkpoint_path and checkpoint_path != stock_checkpoint and os.path.exists(checkpoint_path):
            print(f"[Model Loader] Loading fine-tuned decoder checkpoint from: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            
    elif mode == "lora":
        for p in model.parameters():
            p.requires_grad = False
        try:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["qkv", "proj", "q", "v"],
                lora_dropout=0.05,
                bias="none",
            )
            model.pretrained = get_peft_model(model.pretrained, lora_config)
            print(f"[Experiment Config] LoRA Mode: Injected adapters (r={lora_r}, alpha={lora_alpha}). Base model frozen.")

            # If fine-tuned checkpoint provided, load fine-tuned LoRA weights
            if checkpoint_path and checkpoint_path != stock_checkpoint and os.path.exists(checkpoint_path):
                print(f"[Model Loader] Loading fine-tuned LoRA checkpoint from: {checkpoint_path}")
                state_dict = torch.load(checkpoint_path, map_location='cpu')
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                print(f"[LoRA Verification] Loaded fine-tuned LoRA weights! (Missing: {len(missing)}, Unexpected: {len(unexpected)})")

        except Exception as e:
            print(f"[Warning] Failed to initialize PEFT LoRA: {e}. Falling back to decoder-only mode.")
            for p in model.depth_head.parameters():
                p.requires_grad = True
                
    elif mode == "full":
        for p in model.parameters():
            p.requires_grad = True
        print("[Experiment Config] Full Fine-Tuning Mode: 100% of parameters unfrozen.")
        if checkpoint_path and checkpoint_path != stock_checkpoint and os.path.exists(checkpoint_path):
            print(f"[Model Loader] Loading fine-tuned full model checkpoint from: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            
    else:
        raise ValueError(f"Invalid mode '{mode}'. Choose from ['baseline', 'decoder', 'lora', 'full']")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        "encoder": encoder_type,
        "mode": mode,
        "checkpoint_path": checkpoint_path or stock_checkpoint,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": (trainable_params / total_params * 100.0) if total_params > 0 else 0.0
    }
    
    return model, info

# Backward compatibility alias
build_depth_anything_v2 = load_model
