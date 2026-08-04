import os
import sys
import time
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any

# Ensure src is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.depth_anything import load_model
from src.models.losses import CombinedDepthLoss
from src.utils.metrics import compute_depth_metrics, count_parameters

# Enable TF32 & Cap GPU VRAM Memory to 74% (35.0 GB max out of 48 GB)
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    try:
        torch.cuda.set_per_process_memory_fraction(0.74, 0)
        print("[GPU Config] Capped PyTorch VRAM memory fraction to 74% (35.0 GB max).")
    except Exception as e:
        print(f"[GPU Config Warning] Could not set memory fraction: {e}")

class DepthTrainer:
    """
    Trainer Engine for Depth Anything V2 Fine-Tuning Experiments.
    Supports Decoder-Only, LoRA, and Full Fine-Tuning strategies.
    """
    def __init__(
        self,
        mode: str = "decoder",
        encoder_type: str = "vitl",
        checkpoint_path: str = "checkpoints/pretrained/depth_anything_v2_vitl.pth",
        output_dir: str = "checkpoints/finetuned",
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        device: torch.device = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode
        self.encoder_type = encoder_type
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"\n=======================================================")
        print(f"  INITIALIZING TRAINER: Mode='{mode}' | Encoder='{encoder_type}'")
        print(f"  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        print(f"=======================================================\n")

        # 1. Load Model with unified function
        self.model, self.info = load_model(mode=mode, encoder_type=encoder_type, checkpoint_path=checkpoint_path)
        self.model.to(self.device)

        # 2. Count parameters
        self.total_params, self.trainable_params, self.trainable_percent = count_parameters(self.model)
        print(f"[Trainer Setup] Trainable Params: {self.trainable_params:,} ({self.trainable_percent:.2f}% of {self.total_params:,})")

        # 3. Optimizer on trainable parameters ONLY
        trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_parameters, lr=lr, weight_decay=weight_decay)
        
        # 4. Loss Function
        self.criterion = CombinedDepthLoss(grad_weight=0.5)

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        
        # Re-apply mode freezing to ensure frozen backbone stays locked during train()
        if self.mode == "decoder":
            self.model.pretrained.eval()
        elif self.mode == "baseline":
            self.model.eval()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        for imgs, gts, _ in pbar:
            imgs = imgs.to(self.device)
            gts = gts.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            raw_preds = self.model(imgs) # (B, H_model, W_model)
            
            # Upsample prediction back to ground truth resolution
            target_h, target_w = gts.shape[-2], gts.shape[-1]
            preds_upsampled = F.interpolate(raw_preds[:, None], size=(target_h, target_w), mode="bilinear", align_corners=True)[:, 0]
            
            loss = self.criterion(preds_upsampled, gts)

            # Backward pass & Gradient step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        return avg_loss

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        metrics_accum = {"abs_rel": [], "rmse": [], "rmse_log": [], "delta1": [], "delta2": [], "delta3": []}

        for imgs, gts, _ in tqdm(val_loader, desc="[Validation]"):
            imgs = imgs.to(self.device)
            raw_preds = self.model(imgs)
            
            target_h, target_w = gts.shape[-2], gts.shape[-1]
            preds_upsampled = F.interpolate(raw_preds[:, None], size=(target_h, target_w), mode="bilinear", align_corners=True)[:, 0]

            preds_np = preds_upsampled.cpu().numpy()
            gts_np = gts.numpy()

            for b in range(imgs.shape[0]):
                m = compute_depth_metrics(gts_np[b], preds_np[b])
                for k, v in m.items():
                    metrics_accum[k].append(v)

        return {k: float(torch.tensor(v).mean()) for k, v in metrics_accum.items()}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 5) -> Dict[str, Any]:
        best_rmse = float("inf")
        save_name = f"depth_anything_v2_{self.encoder_type}_{self.mode}.pth"
        best_ckpt_path = os.path.join(self.output_dir, save_name)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t_start = time.time()
        print(f"\n🚀 Starting Fine-Tuning Training for max {epochs} Epochs...\n")

        actual_epochs_run = 0
        for epoch in range(1, epochs + 1):
            actual_epochs_run += 1
            t0 = time.time()
            train_loss = self.train_epoch(train_loader, epoch)
            val_metrics = self.evaluate(val_loader)
            t1 = time.time()

            epoch_time_s = t1 - t0
            val_rmse = val_metrics["rmse"]
            val_abs_rel = val_metrics["abs_rel"]
            val_delta1 = val_metrics["delta1"]

            print(f"  Epoch {epoch}/{epochs} ({epoch_time_s:.1f}s) | Train Loss: {train_loss:.4f} | Val RMSE: {val_rmse:.4f}m | Abs Rel: {val_abs_rel:.4f} | Delta1: {val_delta1:.4f}")

            # Save best checkpoint if RMSE improves
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                torch.save(self.model.state_dict(), best_ckpt_path)
                print(f"  💾 Saved Best Model Checkpoint to: {best_ckpt_path}")

            # Dynamic config check: stop cleanly if target epochs changed in config
            try:
                config_file = f"configs/exp4_full.yaml" if self.mode == "full" else f"configs/exp2_decoder.yaml"
                if os.path.exists(config_file):
                    with open(config_file, 'r') as f:
                        disk_cfg = yaml.safe_load(f)
                    max_target = disk_cfg.get('epochs', epochs)
                    if epoch >= max_target:
                        print(f"\n⏹️ Reached target of {max_target} epochs requested in config. Stopping training cleanly!")
                        break
            except Exception:
                pass

        t_total = time.time() - t_start
        total_time_min = t_total / 60.0
        peak_vram_gb = (torch.cuda.max_memory_allocated() / (1024 ** 3)) if torch.cuda.is_available() else 0.0

        print(f"\n✅ Training Complete in {total_time_min:.2f} mins! Peak VRAM: {peak_vram_gb:.3f} GB")
        print(f"   Best Val RMSE: {best_rmse:.4f} meters")

        return {
            "best_checkpoint_path": best_ckpt_path,
            "best_rmse": best_rmse,
            "total_time_min": round(total_time_min, 2),
            "peak_vram_gb": round(peak_vram_gb, 3),
            "time_per_epoch_s": round(t_total / max(1, actual_epochs_run), 2)
        }
