"""YOLO model evaluation utilities."""

import os


def parse_yolo_results(results_dir):
    """Parse YOLO training results from CSV."""
    import pandas as pd

    results_csv = os.path.join(results_dir, "results.csv")
    if not os.path.exists(results_csv):
        print(f"Results file not found: {results_csv}")
        return None

    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]

    return {
        "final_mAP50": df["metrics/mAP50(B)"].iloc[-1],
        "final_mAP50_95": df["metrics/mAP50-95(B)"].iloc[-1],
        "best_mAP50": df["metrics/mAP50(B)"].max(),
        "total_epochs": len(df)
    }


if __name__ == "__main__":
    print("YOLO evaluation utility - run with results directory path")
