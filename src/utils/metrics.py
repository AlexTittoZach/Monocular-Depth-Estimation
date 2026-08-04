import torch
import numpy as np
import time
from typing import Dict, Tuple

def compute_depth_metrics(gt: np.ndarray, pred: np.ndarray, min_depth: float = 0.001, max_depth: float = 80.0) -> Dict[str, float]:
    """
    Compute standard KITTI monocular depth estimation metrics.
    
    Args:
        gt (np.ndarray): Ground truth depth array (in meters).
        pred (np.ndarray): Predicted depth array (in meters).
        min_depth (float): Minimum valid depth threshold.
        max_depth (float): Maximum valid depth threshold.
        
    Returns:
        Dict[str, float]: Dictionary containing Abs Rel, RMSE, RMSE log, delta1, delta2, delta3.
    """
    # Create valid mask
    mask = (gt > min_depth) & (gt < max_depth) & ~np.isnan(gt) & ~np.isinf(gt)
    if not np.any(mask):
        return {"abs_rel": 0.0, "rmse": 0.0, "rmse_log": 0.0, "delta1": 0.0, "delta2": 0.0, "delta3": 0.0}
    
    gt_valid = gt[mask]
    pred_valid = pred[mask]
    
    # Clip predictions to prevent negative or zero values
    pred_valid = np.clip(pred_valid, min_depth, max_depth)
    
    # Median scaling alignment for relative depth predictions
    scale = np.median(gt_valid) / (np.median(pred_valid) + 1e-8)
    pred_valid = pred_valid * scale
    pred_valid = np.clip(pred_valid, min_depth, max_depth)
    
    # Threshold accuracy
    thresh = np.maximum((gt_valid / pred_valid), (pred_valid / gt_valid))
    delta1 = (thresh < 1.25).mean()
    delta2 = (thresh < 1.25 ** 2).mean()
    delta3 = (thresh < 1.25 ** 3).mean()
    
    # Error metrics
    abs_rel = np.mean(np.abs(gt_valid - pred_valid) / gt_valid)
    rmse = np.sqrt(np.mean((gt_valid - pred_valid) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(gt_valid) - np.log(pred_valid)) ** 2))
    
    return {
        "abs_rel": float(abs_rel),
        "rmse": float(rmse),
        "rmse_log": float(rmse_log),
        "delta1": float(delta1),
        "delta2": float(delta2),
        "delta3": float(delta3)
    }

def count_parameters(model: torch.nn.Module) -> Tuple[int, int, float]:
    """
    Count total and trainable parameters in PyTorch model.
    
    Returns:
        total_params (int)
        trainable_params (int)
        trainable_percentage (float)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_percent = (trainable_params / total_params * 100.0) if total_params > 0 else 0.0
    return total_params, trainable_params, trainable_percent

def measure_inference_latency(model: torch.nn.Module, sample_input: torch.Tensor, num_runs: int = 50, warmup: int = 10) -> Tuple[float, float]:
    """
    Measure inference latency (ms per image) and Throughput (FPS) on current device.
    """
    device = next(model.parameters()).device
    model.eval()
    sample_input = sample_input.to(device)
    
    # GPU Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(sample_input)
            
    if device.type == 'cuda':
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        timings = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_event.record()
                _ = model(sample_input)
                end_event.record()
                torch.cuda.synchronize()
                timings.append(start_event.elapsed_time(end_event)) # in ms
        avg_latency_ms = np.mean(timings)
    else:
        timings = []
        with torch.no_grad():
            for _ in range(num_runs):
                t0 = time.perf_counter()
                _ = model(sample_input)
                t1 = time.perf_counter()
                timings.append((t1 - t0) * 1000.0) # in ms
        avg_latency_ms = np.mean(timings)
        
    fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0
    return float(avg_latency_ms), float(fps)
