from huggingface_hub import HfApi
from kaggle_secrets import UserSecretsClient
import os
import sys
import argparse
import yaml
import torch
from torch.utils.data import DataLoader

# Add src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.datasets.kitti_dataset import KITTIDepthDataset
from src.train.trainer import DepthTrainer
from src.train.evaluate import evaluate_experiment

def main():
    parser = argparse.ArgumentParser(description="Fine-Tune Depth Anything V2 Model")
    parser.add_argument("--config", type=str, default="configs/exp2_decoder.yaml", help="Path to config file")
    args = parser.parse_args()

    # Load Config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    print(f"=======================================================")
    print(f"  LAUNCHING TRAINING: {cfg['experiment_id']} (Mode: {cfg['mode']})")
    print(f"=======================================================\n")

    # 1. Prepare Datasets & DataLoaders
    train_dataset = KITTIDepthDataset(data_dir=cfg['data_dir'], split="train", target_size=(378, 504))
    val_dataset = KITTIDepthDataset(data_dir=cfg['data_dir'], split="val", target_size=(378, 504))

    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.get('batch_size', 4), 
        shuffle=True, 
        num_workers=cfg.get('num_workers', 2),
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.get('batch_size', 4), 
        shuffle=False, 
        num_workers=cfg.get('num_workers', 2)
    )

    # 2. Instantiate Trainer Engine
    trainer = DepthTrainer(
        mode=cfg['mode'],
        encoder_type=cfg.get('encoder_type', 'vitl'),
        checkpoint_path=cfg.get('checkpoint_path', 'checkpoints/pretrained/depth_anything_v2_vitl.pth'),
        output_dir=cfg.get('output_dir', 'checkpoints/finetuned'),
        lr=cfg.get('learning_rate', 1e-4),
        weight_decay=cfg.get('weight_decay', 0.01)
    )

    # 3. Fit Model & Fine-Tune
    train_summary = trainer.fit(train_loader, val_loader, epochs=cfg.get('epochs', 5))

    # 4. Evaluate Fine-Tuned Checkpoint on 1,000 Real KITTI Validation Set
    print("\n=======================================================")
    print(f"  EVALUATING FINE-TUNED CHECKPOINT: {cfg['experiment_id']}")
    print("=======================================================\n")

    eval_results = evaluate_experiment(
        experiment_id=cfg['experiment_id'],
        encoder_type=cfg.get('encoder_type', 'vitl'),
        mode=cfg['mode'],
        checkpoint_path=train_summary['best_checkpoint_path'],
        data_dir=cfg['data_dir'],
        batch_size=1
    )

    # Update training time & peak VRAM in final experiment card
    eval_results['training_time_per_epoch_s'] = train_summary['time_per_epoch_s']
    eval_results['total_training_time_min'] = train_summary['total_time_min']
    eval_results['peak_vram_gb'] = max(eval_results['peak_vram_gb'], train_summary['peak_vram_gb'])

    from src.utils.helpers import save_experiment_results, generate_markdown_report
    save_experiment_results(cfg['experiment_id'], eval_results)
    generate_markdown_report()

    try:
        user_secrets = UserSecretsClient()
        HF_TOKEN = user_secrets.get_secret("HF_TOKEN")

        api = HfApi()

        output_dir = "outputs/logs"

        files_to_upload = [
            f"{cfg['experiment_id']}_results.json",
            "benchmark_summary.csv",
            "benchmark_report.md",
        ]

        for file in files_to_upload:
            local_path = os.path.join(output_dir, file)

            if os.path.exists(local_path):
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=f"logs/{file}",
                    repo_id="alextittozach/depth-anything-v2-kitti-models",
                    repo_type="model",
                    token=HF_TOKEN,
                )
                print(f"✅ Uploaded {file}")

    except Exception as e:
        print(f"⚠️ Failed to upload logs: {e}")



















    print("\nFine-Tuning & Comparative Evaluation Completed Successfully!")

if __name__ == "__main__":
    main()
