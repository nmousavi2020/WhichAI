# =============================================================================
# WhichAI — ML/DL Model Benchmarking and Selection
# Classification + Performance Evaluation + Efficiency Analysis
# =============================================================================
#
# Copyright (c) 2026 Naeim Mousavi
# All rights reserved.
#
# Description:
#     A machine-learning benchmarking framework for comparing ML and DL
#     classification models on tabular datasets and selecting the best
#     model based on predictive performance, computational efficiency,
#     training time, and inference latency.
#
# Methods:
#     Logistic Regression, SVM, Random Forest, Extra Trees,
#     Gradient Boosting, and Multi-Layer Perceptron (MLP).
#
# Evaluation Metrics:
#     Accuracy, Precision, Recall, F1-score, ROC-AUC,
#     Training Time, Inference Latency, and Model Size.
#
# Model Selection:
#     Multi-criteria ranking balancing predictive quality and speed.
#
# Input:
#     CSV dataset with a user-defined target column.
#
# Outputs:
#     results/benchmark_results.csv
#     results/best_model.joblib
#     plots/quality_comparison.png
#     plots/training_time.png
#     plots/inference_latency.png
#     plots/balanced_score.png
#     plots/confusion_matrix.png
#
# =============================================================================

print()
print("╭────────────────────────────────────────────────────────────────────╮")
print("│                         WhichAI                                    │")
print("│             ML/DL Benchmarking and Model Selection                 │")
print("│          Classification • Speed • Efficiency • Accuracy            │")
print("│                                                                    │")
print("│                 Copyright © 2026 Naeim Mousavi                      │")
print("│       For methodology and usage, please see the README.            │")
print("╰────────────────────────────────────────────────────────────────────╯")

# benchmark.py
import os
import time
import json
import argparse
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")


# ---------------- Deep Learning ----------------

class DeepNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.network(x)


class TorchClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        random_state=42,
        epochs=100,
        learning_rate=0.001,
        batch_size=64,
    ):
        self.random_state = random_state
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        self.classes_ = np.unique(y)
        self.model_ = DeepNet(X.shape[1], len(self.classes_))

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
        )
        criterion = nn.CrossEntropyLoss()

        self.model_.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = self.model_(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.model_.eval()

        with torch.no_grad():
            logits = self.model_(torch.tensor(X))
            probabilities = torch.softmax(logits, dim=1)

        return probabilities.cpu().numpy()

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


# ---------------- Benchmark ----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data/credit_risk_1500.csv",
        help="Path to CSV dataset",
    )
    parser.add_argument("--target", default="default")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Dataset not found: {args.data}")

    df = pd.read_csv(args.data)

    if args.target not in df.columns:
        raise ValueError(
            f"Target '{args.target}' not found. "
            f"Columns: {list(df.columns)}"
        )

    df = df.dropna(subset=[args.target]).copy()
    if df.empty:
        raise ValueError("No rows remain after removing missing targets.")

    X = df.drop(columns=[args.target])
    encoder = LabelEncoder()
    y = encoder.fit_transform(df[args.target].astype(str))

    if len(encoder.classes_) < 2:
        raise ValueError("Classification requires at least two classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    numeric_cols = X_train.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()
    categorical_cols = [
        c for c in X_train.columns if c not in numeric_cols
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocess = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])

    X_train_processed = preprocess.fit_transform(X_train)
    X_test_processed = preprocess.transform(X_test)

    # PyTorch expects dense numeric arrays.
    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()
        X_test_processed = X_test_processed.toarray()

    X_train_processed = np.asarray(X_train_processed, dtype=np.float32)
    X_test_processed = np.asarray(X_test_processed, dtype=np.float32)

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=args.seed
        ),
        "SVM_RBF": SVC(
            kernel="rbf", probability=True, random_state=args.seed
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=args.seed, n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=200, random_state=args.seed, n_jobs=-1
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            random_state=args.seed
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            early_stopping=True,
            random_state=args.seed,
        ),
        "PyTorch_DeepLearning": TorchClassifier(
            random_state=args.seed,
            epochs=args.epochs,
        ),
    }

    os.makedirs("results", exist_ok=True)
    rows = []
    fitted_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")

        # HistGradientBoosting requires dense input.
        train_X = X_train_processed
        test_X = X_test_processed

        start = time.perf_counter()
        model.fit(train_X, y_train)
        train_seconds = time.perf_counter() - start

        start = time.perf_counter()
        predictions = model.predict(test_X)
        inference_seconds = time.perf_counter() - start

        accuracy = accuracy_score(y_test, predictions)
        f1_macro = f1_score(
            y_test, predictions, average="macro", zero_division=0
        )
        precision = precision_score(
            y_test, predictions, average="macro", zero_division=0
        )
        recall = recall_score(
            y_test, predictions, average="macro", zero_division=0
        )

        rows.append({
            "model": name,
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "precision_macro": precision,
            "recall_macro": recall,
            "train_time_sec": train_seconds,
            "inference_ms_per_row": (
                inference_seconds / len(test_X) * 1000
            ),
        })

        fitted_models[name] = model
        print(
            f"Accuracy={accuracy:.4f} | "
            f"F1-macro={f1_macro:.4f} | "
            f"Train={train_seconds:.2f}s"
        )

    results = pd.DataFrame(rows)

    # Score rewards predictive quality while considering speed.
    max_train_time = max(results["train_time_sec"].max(), 1e-9)
    results["speed_score"] = (
        1 - results["train_time_sec"] / max_train_time
    )
    results["balanced_score"] = (
        0.7 * results["f1_macro"]
        + 0.3 * results["speed_score"]
    )

    results = results.sort_values(
        "balanced_score", ascending=False
    ).reset_index(drop=True)

    results.to_csv("results/benchmark_results.csv", index=False)

    best_name = results.iloc[0]["model"]
    best_model = fitted_models[best_name]

    # Save preprocessing and model together for inference.
    artifact = {
        "preprocessor": preprocess,
        "model": best_model,
        "target_encoder": encoder,
        "feature_columns": list(X.columns),
    }
    joblib.dump(artifact, "results/best_model.joblib")

    with open("results/label_classes.json", "w", encoding="utf-8") as f:
        json.dump(
            encoder.classes_.tolist(),
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n===== BENCHMARK RESULTS =====")
    print(results.to_string(index=False))
    print(f"\nBest model: {best_name}")
    print("Saved: results/benchmark_results.csv")
    print("Saved: results/best_model.joblib")

    try:
        from plotting import create_plots

        create_plots(
            results,
            "results",
            best_name=best_name,
            y_true=y_test,
            y_pred=best_model.predict(X_test_processed),
            labels=encoder.classes_.tolist(),
        )
        print("Plots saved in results/")
    except ImportError:
        print("plotting.py not found; skipping plots.")
    except Exception as exc:
        print(f"Plot generation skipped: {exc}")


if __name__ == "__main__":
    main()

    import plotting
    print("Plotting module:", plotting.__file__)