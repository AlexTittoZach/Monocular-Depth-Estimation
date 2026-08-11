import os
import sys
import json
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# ============================================================
# Project path
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(0, PROJECT_ROOT)

# ============================================================
# Project imports
# ============================================================

from src.datasets.kitti_dataset import KITTIDepthDataset
from src.models.depth_anything import load_model
from src.utils.metrics import compute_depth_metrics
from src.utils.helpers import (
    save_experiment_results,
    generate_markdown_report,
)


# ============================================================
# Configuration
# ============================================================

EXPERIMENT_ID = "BASELINE_DEPTH_ANYTHING_V2"

DATA_DIR = "/home/inicai/depth-anything-v2-vitl/datasets/kitti"

CHECKPOINT_PATH = "pretrained/depth_anything_v2_vitl.pth"

ENCODER_TYPE = "vitl"

TARGET_SIZE = (378, 504)

BATCH_SIZE = 8

NUM_WORKERS = 4

OUTPUT_DIR = "outputs/logs"


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("  BASELINE EVALUATION: DEPTH ANYTHING V2 ViT-L")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"[Device] {device}")

    if torch.cuda.is_available():
        print(
            f"[GPU] {torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print("\n[Dataset] Loading KITTI validation dataset...")

    val_dataset = KITTIDepthDataset(
        data_dir=DATA_DIR,
        split="val",
        target_size=TARGET_SIZE,
    )

    print(
        f"[Dataset] Validation samples: "
        f"{len(val_dataset)}"
    )

    # IMPORTANT:
    # Prevent accidentally evaluating synthetic data.
    if len(val_dataset) <= 20:
        raise RuntimeError(
            "\nERROR: Validation dataset contains "
            f"{len(val_dataset)} samples.\n\n"
            "The KITTIDepthDataset loader may have "
            "fallen back to synthetic samples.\n"
            "Check your dataset directory:\n"
            f"{DATA_DIR}/val/images\n"
            f"{DATA_DIR}/val/depths\n"
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    print(
        f"[DataLoader] Batches: {len(val_loader)}"
    )

    # --------------------------------------------------------
    # Load official baseline model
    # --------------------------------------------------------

    print("\n[Model] Loading official pretrained model...")

    result = load_model(
        mode="baseline",
        encoder_type=ENCODER_TYPE,
        checkpoint_path=CHECKPOINT_PATH,
    )

    # Your load_model() returns:
    # (model, metadata)
    if isinstance(result, tuple):
        model = result[0]
    else:
        model = result

    model = model.to(device)
    model.eval()

    print("[Model] Official pretrained weights loaded.")
    print("[Model] Baseline mode: all parameters frozen.")

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("  STARTING BASELINE EVALUATION")
    print("=" * 70)

    metric_values = []

    start_time = time.time()

    with torch.no_grad():

        for imgs, depths, names in tqdm(
            val_loader,
            desc="Baseline Evaluation",
        ):

            imgs = imgs.to(
                device,
                non_blocking=True,
            )

            depths = depths.to(
                device,
                non_blocking=True,
            )

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            preds = model(imgs)

            # ------------------------------------------------
            # Make prediction and GT dimensions match
            # ------------------------------------------------

            if preds.ndim == 3:
                preds = preds.unsqueeze(1)

            if preds.shape[-2:] != depths.shape[-2:]:

                preds = torch.nn.functional.interpolate(
                    preds,
                    size=depths.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            preds = preds.squeeze(1)

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            for i in range(preds.shape[0]):

                pred = preds[i].detach().cpu().numpy()
                gt = depths[i].detach().cpu().numpy()

                metrics = compute_depth_metrics(
                    pred,
                    gt,
    )

                metric_values.append(metrics)

    elapsed = time.time() - start_time

    # ========================================================
    # Aggregate metrics
    # ========================================================

    if not metric_values:
        raise RuntimeError(
            "No metrics were generated."
        )

    metric_names = metric_values[0].keys()

    final_metrics = {}

    for metric_name in metric_names:

        values = []

        for metrics in metric_values:

            value = metrics.get(metric_name)

            if value is None:
                continue

            value = float(value)

            if np.isfinite(value):
                values.append(value)

        if values:
            final_metrics[metric_name] = float(
                np.mean(values)
            )

    # ========================================================
    # Add experiment information
    # ========================================================

    results = {
        "experiment_id": EXPERIMENT_ID,
        "model": "Depth Anything V2 ViT-L",
        "mode": "baseline",
        "encoder_type": ENCODER_TYPE,
        "checkpoint": CHECKPOINT_PATH,
        "dataset": DATA_DIR,
        "split": "val",
        "num_samples": len(val_dataset),
        "batch_size": BATCH_SIZE,
        "target_size": list(TARGET_SIZE),
        "evaluation_time_seconds": elapsed,
        **final_metrics,
    }

    # ========================================================
    # Print final results
    # ========================================================

    print("\n")
    print("=" * 70)
    print("  BASELINE RESULTS")
    print("=" * 70)

    for key, value in final_metrics.items():

        print(
            f"{key:20s}: {value:.6f}"
        )

    print(
        f"{'Samples':20s}: {len(val_dataset)}"
    )

    print(
        f"{'Evaluation Time':20s}: "
        f"{elapsed:.2f} seconds"
    )

    print("=" * 70)

    # ========================================================
    # Save results
    # ========================================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    save_experiment_results(
        experiment_id=EXPERIMENT_ID,
        results=results,
        output_dir=OUTPUT_DIR,
    )

    # ========================================================
    # Generate Markdown benchmark report
    # ========================================================

    report = generate_markdown_report(
        output_dir=OUTPUT_DIR
    )

    print("\n[Logger] Benchmark report generated.")

    print("\n" + "=" * 70)
    print("  BASELINE EVALUATION COMPLETE")
    print("=" * 70)

    print("\nSaved files:")

    print(
        f"  {OUTPUT_DIR}/"
        f"{EXPERIMENT_ID}_results.json"
    )

    print(
        f"  {OUTPUT_DIR}/benchmark_summary.csv"
    )

    print(
        f"  {OUTPUT_DIR}/benchmark_report.md"
    )


if __name__ == "__main__":
    main()