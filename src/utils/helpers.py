import os
import json
import yaml
import torch
import pandas as pd
from typing import Dict, Any

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_experiment_results(experiment_id: str, results: Dict[str, Any], output_dir: str = "outputs/logs"):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{experiment_id}_results.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    # Append/Update summary CSV
    summary_csv = os.path.join(output_dir, "benchmark_summary.csv")
    df_row = pd.DataFrame([results])
    
    if os.path.exists(summary_csv):
        df_existing = pd.read_csv(summary_csv)
        # Drop row if experiment_id already exists to overwrite
        df_existing = df_existing[df_existing.get("experiment_id") != experiment_id]
        df_combined = pd.concat([df_existing, df_row], ignore_index=True)
        df_combined.to_csv(summary_csv, index=False)
    else:
        df_row.to_csv(summary_csv, index=False)
        
    print(f"[Logger] Saved experiment results to {json_path} and updated {summary_csv}")

def generate_markdown_report(output_dir: str = "outputs/logs") -> str:
    summary_csv = os.path.join(output_dir, "benchmark_summary.csv")
    if not os.path.exists(summary_csv):
        return "No benchmark summary available yet."
        
    df = pd.read_csv(summary_csv)
    report_md = "# 📊 Fine-Tuning Monocular Depth Estimation Strategy Benchmark\n\n"
    report_md += df.to_markdown(index=False)
    report_md += "\n"
    
    report_path = os.path.join(output_dir, "benchmark_report.md")
    with open(report_path, 'w') as f:
        f.write(report_md)
        
    return report_md
