"""Comprehensive evaluation, threshold optimization, and visualization module."""
import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)
from src.config import FIGURES_DIR, COST_FN, COST_FP

logger = logging.getLogger(__name__)

# Styling for figures
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 300


def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[float, float]:
    """
    Search for the decision threshold that maximizes F1-Score on the Precision-Recall curve.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # Avoid zero division
    f1_scores = np.where(
        (precisions + recalls) > 0,
        (2 * precisions * recalls) / (precisions + recalls),
        0,
    )
    # thresholds has len = len(precisions) - 1
    best_idx = np.argmax(f1_scores[:-1])
    best_thresh = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])
    return best_thresh, best_f1


def evaluate_predictions(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute binary classification metrics at a given probability threshold.
    """
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    cost = (fn * COST_FN) + (fp * COST_FP)

    return {
        "threshold": threshold,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "accuracy": acc,
        "specificity": specificity,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "cost": cost,
    }


def evaluate_model_full(
    model_name: str,
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """
    Full evaluation of a trained model on the unseen test set,
    including ROC-AUC, PR-AUC, default threshold (0.50), and optimal threshold.
    """
    y_true = y_test.values
    y_proba = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)

    # 1. Default threshold 0.5
    default_metrics = evaluate_predictions(y_true, y_proba, threshold=0.5)

    # 2. Optimal F1 threshold
    opt_thresh, opt_f1 = find_optimal_threshold(y_true, y_proba)
    opt_metrics = evaluate_predictions(y_true, y_proba, threshold=opt_thresh)

    return {
        "model_name": model_name,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "y_true": y_true,
        "y_proba": y_proba,
        "default": default_metrics,
        "optimal": opt_metrics,
        "optimal_threshold": opt_thresh,
    }


def plot_precision_recall_curves(results: List[Dict[str, Any]], filename: str = "precision_recall_curves.png"):
    """Plot overlaid Precision-Recall curves for all evaluated models."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for res in results:
        prec, rec, _ = precision_recall_curve(res["y_true"], res["y_proba"])
        label = f"{res['model_name']} (PR-AUC = {res['pr_auc']:.4f})"
        ax.plot(rec, prec, lw=2, label=label)

    # Baseline: random guessing precision = positive class prevalence
    prevalence = np.mean(results[0]["y_true"])
    ax.axhline(prevalence, color="navy", linestyle="--", alpha=0.7, label=f"Random Chance (AP={prevalence:.4f})")

    ax.set_title("Precision-Recall Curves (Key Metric for Imbalanced Data)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Recall (Fraud Detection Rate)", fontsize=12)
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=12)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower left", fontsize=10, frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = FIGURES_DIR / filename
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info("Saved PR curves figure to %s", out_path)


def plot_roc_curves(results: List[Dict[str, Any]], filename: str = "roc_curves.png"):
    """Plot overlaid ROC curves for all evaluated models."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for res in results:
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_proba"])
        label = f"{res['model_name']} (ROC-AUC = {res['roc_auc']:.4f})"
        ax.plot(fpr, tpr, lw=2, label=label)

    ax.plot([0, 1], [0, 1], color="navy", linestyle="--", lw=1.5, label="Random Guess (AUC = 0.5000)")
    ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=12)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower right", fontsize=10, frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = FIGURES_DIR / filename
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info("Saved ROC curves figure to %s", out_path)


def plot_confusion_matrices(results: List[Dict[str, Any]], filename: str = "confusion_matrices_optimal.png"):
    """Plot side-by-side confusion matrices at optimal threshold."""
    n_models = len(results)
    cols = 3
    rows = int(np.ceil(n_models / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = np.array(axes).flatten()

    for idx, res in enumerate(results):
        ax = axes[idx]
        opt = res["optimal"]
        cm = np.array([[opt["tn"], opt["fp"]], [opt["fn"], opt["tp"]]])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            annot_kws={"size": 13, "weight": "bold"},
        )
        title = f"{res['model_name']}\nThresh={opt['threshold']:.3f} | F1={opt['f1']:.3f}\nCost=${opt['cost']:,.0f}"
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Class", fontsize=10)
        ax.set_ylabel("Actual Class", fontsize=10)
        ax.set_xticklabels(["Legit (0)", "Fraud (1)"])
        ax.set_yticklabels(["Legit (0)", "Fraud (1)"])

    # Hide extra unused subplots
    for j in range(n_models, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    out_path = FIGURES_DIR / filename
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info("Saved Confusion Matrices figure to %s", out_path)


def plot_feature_importance(model, feature_names: List[str], model_name: str, top_n: int = 15):
    """Plot top N feature importances for tree-based estimators."""
    # Extract classifier from pipeline if wrapped
    clf = model.named_steps.get("classifier", model)
    importances = getattr(clf, "feature_importances_", None)
    if importances is None:
        return

    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_importances, y=top_features, hue=top_features, palette="viridis", legend=False, ax=ax)
    ax.set_title(f"Top {top_n} Feature Importances - {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Relative Importance", fontsize=11)
    fig.tight_layout()

    filename = f"feature_importance_{model_name.lower().replace(' ', '_').replace('+', '_')}.png"
    out_path = FIGURES_DIR / filename
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info("Saved feature importance figure to %s", out_path)
