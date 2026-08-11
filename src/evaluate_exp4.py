import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# ============================================================
# Project Path
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# Project Imports
# ============================================================

from src.datasets.kitti_dataset import KITTIDepthDataset
from src.models.depth_anything import load_model
from src.utils.metrics import compute_depth_metrics


# ============================================================
# Configuration
# ============================================================

EXPERIMENT_ID = "EXP_04_FULL_FT"

MODE = "full"

ENCODER_TYPE = "vitl"

CHECKPOINT_PATH = (
    "checkpoints/finetuned/"
    "depth_anything_v2_vitl_full.pth"
)

DATA_DIR = (
    "/home/inicai/depth-anything-v2-vitl/"
    "datasets/kitti"
)

TARGET_SIZE = (378, 504)

BATCH_SIZE = 8

NUM_WORKERS = 4

OUTPUT_DIR = "outputs/logs"


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("  EVALUATING EXP_04_FULL_FT")
    print("  Full Fine-Tuned Depth Anything V2 ViT-L")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\n[Device] {device}")

    if torch.cuda.is_available():
        print(
            f"[GPU] "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Check checkpoint
    # --------------------------------------------------------

    if not os.path.isfile(CHECKPOINT_PATH):

        raise FileNotFoundError(
            f"\nCheckpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    checkpoint_size_gb = (
        os.path.getsize(CHECKPOINT_PATH)
        / (1024 ** 3)
    )

    print(
        f"[Checkpoint] {CHECKPOINT_PATH}"
    )

    print(
        f"[Checkpoint] Size: "
        f"{checkpoint_size_gb:.2f} GB"
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

    # Prevent synthetic fallback
    if len(val_dataset) <= 20:

        raise RuntimeError(
            "\nERROR: Only "
            f"{len(val_dataset)} validation samples found.\n\n"
            "The dataset loader may have fallen back "
            "to synthetic KITTI samples.\n\n"
            "Expected real dataset at:\n"
            f"{DATA_DIR}/val/images\n"
            f"{DATA_DIR}/val/depths"
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    print(
        f"[DataLoader] Batches: "
        f"{len(val_loader)}"
    )

    # --------------------------------------------------------
    # Load Full Fine-Tuned Model
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("  Loading Full Fine-Tuned Checkpoint")
    print("=" * 70)

    result = load_model(
        mode=MODE,
        encoder_type=ENCODER_TYPE,
        checkpoint_path=CHECKPOINT_PATH,
    )

    # load_model() returns (model, metadata)
    if isinstance(result, tuple):
        model = result[0]
    else:
        model = result

    model = model.to(device)

    model.eval()

    print(
        "[Model] Full Fine-Tuned model loaded successfully."
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("  STARTING EXP_04 EVALUATION")
    print("=" * 70)

    all_metrics = []

    start_time = time.time()

    with torch.no_grad():

        for imgs, depths, names in tqdm(
            val_loader,
            desc="EXP-04 Evaluation",
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
            # Model inference
            # ------------------------------------------------

            raw_preds = model(imgs)

            # ------------------------------------------------
            # Match prediction dimensions to GT
            # ------------------------------------------------

            if raw_preds.ndim == 3:
                raw_preds = raw_preds.unsqueeze(1)

            preds = F.interpolate(
                raw_preds,
                size=depths.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )

            preds = preds.squeeze(1)

            # ------------------------------------------------
            # Calculate metrics
            # ------------------------------------------------

            for i in range(preds.shape[0]):

                pred = (
                    preds[i]
                    .detach()
                    .cpu()
                    .numpy()
                )

                gt = (
                    depths[i]
                    .detach()
                    .cpu()
                    .numpy()
                )

                metrics = compute_depth_metrics(
                    gt,
                    pred,
                )

                all_metrics.append(metrics)

    elapsed = time.time() - start_time

    # ========================================================
    # Aggregate Metrics
    # ========================================================

    if not all_metrics:

        raise RuntimeError(
            "No metrics were generated."
        )

    metric_names = all_metrics[0].keys()

    final_metrics = {}

    for metric_name in metric_names:

        values = []

        for metrics in all_metrics:

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
    # Print Results
    # ========================================================

    print("\n")
    print("=" * 70)
    print("  EXP_04 FULL FINE-TUNED RESULTS")
    print("=" * 70)

    for name, value in final_metrics.items():

        print(
            f"{name:20s}: {value:.6f}"
        )

    print(
        f"{'Samples':20s}: "
        f"{len(val_dataset)}"
    )

    print(
        f"{'Batches':20s}: "
        f"{len(val_loader)}"
    )

    print(
        f"{'Evaluation Time':20s}: "
        f"{elapsed:.2f} seconds"
    )

    print("=" * 70)

    # ========================================================
    # Build Results Dictionary
    # ========================================================

    results = {
        "experiment_id": EXPERIMENT_ID,
        "model": "Depth Anything V2 ViT-L",
        "mode": MODE,
        "encoder_type": ENCODER_TYPE,
        "checkpoint": CHECKPOINT_PATH,
        "dataset": DATA_DIR,
        "split": "val",
        "num_samples": len(val_dataset),
        "batch_size": BATCH_SIZE,
        "target_size": list(TARGET_SIZE),
        "evaluation_time_seconds": elapsed,
        "metrics": final_metrics,
    }

    # ========================================================
    # Save JSON
    # ========================================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    json_path = os.path.join(
        OUTPUT_DIR,
        "EXP_04_FULL_FT_results.json",
    )

    with open(json_path, "w") as f:

        json.dump(
            results,
            f,
            indent=4,
        )

    print(
        f"\n[Logger] Saved experiment results to:"
    )

    print(json_path)

    # ========================================================
    # Update benchmark_summary.csv
    # ========================================================

    csv_path = os.path.join(
        OUTPUT_DIR,
        "benchmark_summary.csv",
    )

    import csv

    # Keep existing benchmark rows if the file exists
    existing_rows = []

    if os.path.isfile(csv_path):

        with open(
            csv_path,
            "r",
            newline="",
        ) as f:

            reader = csv.DictReader(f)

            for row in reader:

                # Replace an existing EXP_04 row
                if row.get("experiment_id") != EXPERIMENT_ID:
                    existing_rows.append(row)

    # Create new row
    new_row = {
        "experiment_id": EXPERIMENT_ID,
        **{
            key: str(value)
            for key, value in final_metrics.items()
        },
    }

    existing_rows.append(new_row)

    # Determine all columns
    fieldnames = []

    for row in existing_rows:

        for key in row.keys():

            if key not in fieldnames:
                fieldnames.append(key)

    with open(
        csv_path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(existing_rows)

    print(
        "[Logger] Updated benchmark summary:"
    )

    print(csv_path)

    # ========================================================
    # Final
    # ========================================================

    print("\n" + "=" * 70)
    print("  EXP_04 EVALUATION COMPLETE")
    print("=" * 70)

    print("\nOutput files:")

    print(
        f"  ✓ {json_path}"
    )

    print(
        f"  ✓ {csv_path}"
    )


if __name__ == "__main__":
    main()