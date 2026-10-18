# =============================================================================
# WhichAI — Benchmark Visualization and Performance Analysis
# =============================================================================
#
# Copyright (c) 2026 Naeim Mousavi
# All rights reserved.
#
# Description:
#     Generates publication-quality visualizations for comparing machine
#     learning and deep learning classification models.
#
# Visualizations:
#     1. Classification Performance — Accuracy, Precision, Recall, and F1-score.
#     2. Time Performance — Training Time and Inference Latency.
#     3. Model Efficiency — Balanced Performance and Speed Ranking.
#
# Output:
#     plots/1_classification_performance.png
#     plots/2_time_performance.png
#     plots/3_efficiency_score.png
#
# Requirements:
#     matplotlib, numpy, scikit-learn
#
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt


def create_plots(
    results,
    output_dir="plots",
    best_name=None,
    y_true=None,
    y_pred=None,
    labels=None,
):
    os.makedirs(output_dir, exist_ok=True)
    results = results.copy()

    # 1. Classification Performance
    metrics = [
        ("accuracy", "Accuracy"),
        ("precision_macro", "Precision"),
        ("recall_macro", "Recall"),
        ("f1_macro", "F1-score"),
    ]
    metrics = [
        (col, label) for col, label in metrics
        if col in results.columns
    ]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(results))
    width = 0.8 / len(metrics)

    for i, (col, label) in enumerate(metrics):
        ax.bar(
            x - 0.4 + width / 2 + i * width,
            results[col],
            width,
            label=label,
        )

    ax.set_title("1. Classification Performance")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(results["model"], rotation=25, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "1_classification_performance.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 2. Time Performance
    time_metrics = [
        ("train_time_sec", "Training Time (seconds)"),
        ("inference_ms_per_row", "Inference Latency (ms/sample)"),
    ]
    available = [
        (col, title) for col, title in time_metrics
        if col in results.columns
    ]

    if available:
        fig, axes = plt.subplots(
            1, len(available),
            figsize=(7 * len(available), 5),
        )
        axes = np.atleast_1d(axes)

        for ax, (col, title) in zip(axes, available):
            data = results.sort_values(col)
            bars = ax.barh(data["model"], data[col])
            ax.set_title(title)
            ax.set_xlabel(title)
            ax.grid(axis="x", alpha=0.25)
            ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)

        fig.suptitle("2. Time Performance")
        fig.tight_layout()
        fig.savefig(
            os.path.join(output_dir, "2_time_performance.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    # 3. Model Efficiency
    score_col = (
        "balanced_score"
        if "balanced_score" in results.columns
        else "f1_macro"
    )
    data = results.sort_values(score_col)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(data["model"], data[score_col])
    ax.set_title("3. Model Efficiency Ranking")
    ax.set_xlabel(score_col.replace("_", " ").title())
    ax.grid(axis="x", alpha=0.25)
    ax.bar_label(bars, fmt="%.4f", padding=3)
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "3_efficiency_score.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Three plots saved to: {output_dir}")
