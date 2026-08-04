import os
import sys
import time
import torch
import cv2
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure src is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.depth_anything import load_model
from src.datasets.kitti_dataset import KITTIDepthDataset
from src.utils.metrics import compute_depth_metrics, count_parameters, measure_inference_latency
from src.utils.helpers import save_experiment_results, generate_markdown_report

def evaluate_experiment(
    experiment_id: str = "EXP_01_BASELINE",
    encoder_type: str = "vitl",
    mode: str = "baseline",
    checkpoint_path: str = "checkpoints/pretrained/depth_anything_v2_vitl.pth",
    data_dir: str = "datasets/kitti",
    batch_size: int = 1
) -> dict:
    """
    Run evaluation benchmark for a specified experiment.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"  RUNNING BENCHMARK: {experiment_id} (Mode: {mode})")
    print(f"  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"=======================================================\n")

    # Reset CUDA memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # 1. Load Model with Unified Function & Verify Checkpoint
    model, info = load_model(mode=mode, encoder_type=encoder_type, checkpoint_path=checkpoint_path)
    model.to(device)
    model.eval()

    total_params, trainable_params, trainable_percent = count_parameters(model)
    
    # Estimate Checkpoint Size (MB)
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    ckpt_size_mb = (param_size + buffer_size) / (1024 ** 2)

    # 2. Load Dataset
    val_dataset = KITTIDepthDataset(data_dir=data_dir, split="val", target_size=(378, 504))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 3. Measure Latency & FPS
    sample_img, _, _ = val_dataset[0]
    sample_img = sample_img.unsqueeze(0).to(device)
    latency_ms, fps = measure_inference_latency(model, sample_img, num_runs=50, warmup=10)

    # 4. Accuracy Evaluation Loop
    metrics_accum = {"abs_rel": [], "rmse": [], "rmse_log": [], "delta1": [], "delta2": [], "delta3": []}
    printed_stats = False

    with torch.no_grad():
        for imgs, gts, _ in tqdm(val_loader, desc=f"Evaluating {experiment_id}"):
            imgs = imgs.to(device)
            raw_preds = model(imgs) # (B, H, W) in model resolution
            
            # Upsample prediction back to ground truth resolution
            target_h, target_w = gts.shape[-2], gts.shape[-1]
            preds_upsampled = F.interpolate(raw_preds[:, None], size=(target_h, target_w), mode="bilinear", align_corners=True)[:, 0]
            
            preds_np = preds_upsampled.cpu().numpy()
            gts_np = gts.numpy()
            
            # Print Output Statistics immediately after inference on first batch
            if not printed_stats:
                print(f"\n[Inference Output Stats]")
                print(f"  Shape: {preds_np.shape}")
                print(f"  Min  : {preds_np.min():.4f}")
                print(f"  Max  : {preds_np.max():.4f}")
                print(f"  Mean : {preds_np.mean():.4f}\n")
                printed_stats = True
            
            for b in range(imgs.shape[0]):
                # Raw prediction remains untouched for metrics evaluation
                m = compute_depth_metrics(gts_np[b], preds_np[b])
                for k, v in m.items():
                    metrics_accum[k].append(v)

    # Compute Average Metrics
    avg_metrics = {k: float(np.mean(v)) for k, v in metrics_accum.items()}

    # Measure Peak GPU VRAM (GB)
    peak_vram_gb = (torch.cuda.max_memory_allocated() / (1024 ** 3)) if torch.cuda.is_available() else 0.0

    # 5. Compile Complete Experiment Card
    results = {
        "experiment_id": experiment_id,
        "mode": mode,
        "encoder": encoder_type,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_percent": round(trainable_percent, 2),
        "checkpoint_size_mb": round(ckpt_size_mb, 2),
        "peak_vram_gb": round(peak_vram_gb, 3),
        "training_time_per_epoch_s": 0.0 if mode == "baseline" else "N/A",
        "total_training_time_min": 0.0 if mode == "baseline" else "N/A",
        "latency_ms": round(latency_ms, 2),
        "throughput_fps": round(fps, 1),
        "abs_rel": round(avg_metrics["abs_rel"], 4),
        "rmse": round(avg_metrics["rmse"], 4),
        "rmse_log": round(avg_metrics["rmse_log"], 4),
        "delta1": round(avg_metrics["delta1"], 4),
        "delta2": round(avg_metrics["delta2"], 4),
        "delta3": round(avg_metrics["delta3"], 4)
    }

    # Save Results & Update Markdown Report
    save_experiment_results(experiment_id, results)
    report = generate_markdown_report()

    print(f"\n✅ Finished {experiment_id} Benchmark!")
    print(f"   Trainable Params: {trainable_params:,} ({trainable_percent:.2f}%)")
    print(f"   Latency: {latency_ms:.2f} ms ({fps:.1f} FPS)")
    print(f"   Abs Rel: {avg_metrics['abs_rel']:.4f} | RMSE: {avg_metrics['rmse']:.4f} | Delta1: {avg_metrics['delta1']:.4f}")
    
    return results

if __name__ == "__main__":
    evaluate_experiment(
        experiment_id="EXP_01_BASELINE",
        mode="baseline",
        encoder_type="vitl",
        checkpoint_path="checkpoints/pretrained/depth_anything_v2_vitl.pth"
    )
