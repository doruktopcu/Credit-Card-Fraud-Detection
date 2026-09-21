"""End-to-end script to train and evaluate competitive fraud detection models."""
import sys
import time
import logging
from pathlib import Path
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TABLES_DIR
from src.data import load_raw_data, get_stratified_split
from src.models import get_model_portfolio
from src.evaluation import (
    evaluate_model_full,
    plot_precision_recall_curves,
    plot_roc_curves,
    plot_confusion_matrices,
    plot_feature_importance,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TrainAndEvaluate")


def main():
    start_time = time.time()
    logger.info("=== Starting Credit Card Fraud Detection Benchmark Pipeline ===")

    # 1. Load Data
    df = load_raw_data()

    # 2. Stratified Train/Test Split (Preserves pristine test distribution)
    X_train, X_test, y_train, y_test = get_stratified_split(df)

    # 3. Model Portfolio
    models = get_model_portfolio()
    results = []
    summary_rows = []

    logger.info("Training and evaluating %d candidate models...", len(models))

    for model_name, pipeline in models.items():
        logger.info("-" * 60)
        logger.info(">>> Training Model: %s", model_name)
        t0 = time.time()
        pipeline.fit(X_train, y_train)
        fit_time = time.time() - t0
        logger.info("Fitted %s in %.2f seconds.", model_name, fit_time)

        # Evaluate on Held-Out Test Set
        logger.info("Evaluating %s on unseen test set (N=%d)...", model_name, len(X_test))
        res = evaluate_model_full(model_name, pipeline, X_test, y_test)
        results.append(res)

        opt = res["optimal"]
        dflt = res["default"]
        logger.info(
            "%s -> PR-AUC: %.4f | ROC-AUC: %.4f | Opt-Thresh: %.3f | Opt-F1: %.4f (Prec: %.4f, Rec: %.4f) | Cost: $%d",
            model_name,
            res["pr_auc"],
            res["roc_auc"],
            opt["threshold"],
            opt["f1"],
            opt["precision"],
            opt["recall"],
            opt["cost"],
        )

        summary_rows.append({
            "Model": model_name,
            "PR-AUC (AP)": round(res["pr_auc"], 4),
            "ROC-AUC": round(res["roc_auc"], 4),
            "Opt Threshold": round(opt["threshold"], 4),
            "Opt F1-Score": round(opt["f1"], 4),
            "Opt Recall": round(opt["recall"], 4),
            "Opt Precision": round(opt["precision"], 4),
            "Opt Accuracy": round(opt["accuracy"], 5),
            "Default F1 (0.50)": round(dflt["f1"], 4),
            "Default Recall (0.50)": round(dflt["recall"], 4),
            "Default Precision (0.50)": round(dflt["precision"], 4),
            "Estimated Cost ($)": int(opt["cost"]),
            "Train Time (s)": round(fit_time, 2),
        })

        # Feature importance if applicable
        preprocessor = pipeline.named_steps.get("preprocessor")
        feature_names = preprocessor.get_feature_names_out() if preprocessor else list(X_train.columns)
        plot_feature_importance(pipeline, feature_names, model_name)

    # 4. Save Tabular Summaries
    results_df = pd.DataFrame(summary_rows).sort_values(by="PR-AUC (AP)", ascending=False)
    csv_path = TABLES_DIR / "benchmark_results.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info("Saved benchmark results to CSV: %s", csv_path)

    md_path = TABLES_DIR / "benchmark_results.md"
    results_df.to_markdown(md_path, index=False)
    logger.info("Saved benchmark results to Markdown: %s", md_path)

    # 5. Generate Publication-Ready Comparison Plots
    logger.info("Generating comparative visualization figures...")
    plot_precision_recall_curves(results)
    plot_roc_curves(results)
    plot_confusion_matrices(results)

    total_time = time.time() - start_time
    logger.info("=== Pipeline Completed Successfully in %.2f seconds ===", total_time)
    print("\n" + "=" * 80)
    print("FINAL BENCHMARK COMPARISON TABLE (Ranked by PR-AUC):")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
