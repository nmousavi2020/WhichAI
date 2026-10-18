# WhichAI — ML/DL Benchmarking and Model Selection

**Classification • Performance • Speed • Efficiency**

WhichAI is a Python benchmarking framework for comparing machine learning and deep learning classification models on tabular datasets. It evaluates predictive performance, measures execution time, and ranks models using a balanced performance and speed score.

## Features

- Compare machine learning and deep learning classifiers.
- Evaluate Accuracy, Precision, Recall, and F1-score.
- Measure training time and inference latency.
- Rank models using a balanced efficiency score.
- Save trained models and benchmark results.
- Generate three performance visualization plots.

## Supported Models

1. Logistic Regression
2. Support Vector Machine (RBF)
3. Random Forest
4. Extra Trees
5. Histogram-based Gradient Boosting
6. Multi-Layer Perceptron (MLP)
7. PyTorch Deep Neural Network

## Project Structure

```text
WhichAI/
├── benchmark.py
├── plotting.py
├── synthetic.py
├── README.md
├── LICENSE
├── requirements.txt
├── data/
│   └── credit_risk_1500.csv
└── results/
    ├── 1_classification_performance.png
    ├── 2_time_performance.png
    ├── 3_efficiency_score.png
    ├── benchmark_results.csv
    ├── best_model.joblib
    └── label_classes.json
```

## Requirements

- Python 3.10+
- pandas
- NumPy
- scikit-learn
- Matplotlib
- joblib
- PyTorch

Install dependencies:

```bash
pip install -r requirements.txt
```

Alternatively:

```bash
pip install pandas numpy scikit-learn matplotlib joblib torch
```

## Dataset

The default dataset is:

`data/credit_risk_1500.csv`

The default target column is `default`.

The input CSV should contain feature columns and one target column representing the class labels. Missing feature values are handled through preprocessing, and rows with missing target values are removed.

## Usage

Run the benchmark from the project root:

```bash
python benchmark.py
```

Use a custom dataset:

```bash
python benchmark.py --data data/your_dataset.csv --target target
```

Configure the test-set proportion and random seed:

```bash
python benchmark.py --test-size 0.2 --seed 42
```

Set the number of deep-learning training epochs:

```bash
python benchmark.py --epochs 100
```

## Evaluation Metrics

| Metric | Description |
|---|---|
| Accuracy | Overall proportion of correct predictions |
| Precision | Reliability of positive predictions |
| Recall | Ability to identify positive cases |
| F1-score | Harmonic mean of precision and recall |
| Training Time | Time required to train a model |
| Inference Latency | Average prediction time per sample |
| Balanced Score | Combined classification quality and relative training speed |

### Model Selection

The benchmark calculates its balanced score using:

```python
balanced_score = (
    0.7 * f1_macro
    + 0.3 * speed_score
)
```

The speed score rewards models that train faster relative to the slowest model in the benchmark. The selected model is the one with the highest balanced score, which may differ from the model with the highest accuracy.

## Visualization

WhichAI generates three plots in the `results/` directory.

### 1. Classification Performance

`results/1_classification_performance.png`

Compares Accuracy, Precision, Recall, and F1-score across evaluated models.

### 2. Time Performance

`results/2_time_performance.png`

Compares model training time and inference latency.

### 3. Model Efficiency

`results/3_efficiency_score.png`

Ranks models by their balanced score, combining F1-score and relative training speed.

## Output Files

| File | Description |
|---|---|
| `results/benchmark_results.csv` | Performance metrics and model rankings |
| `results/best_model.joblib` | Saved preprocessor and selected model |
| `results/label_classes.json` | Original target-class labels |
| `results/1_classification_performance.png` | Classification metrics chart |
| `results/2_time_performance.png` | Training and inference timing chart |
| `results/3_efficiency_score.png` | Efficiency ranking chart |

## Loading the Saved Model

The saved artifact contains the fitted preprocessor, trained model, target encoder, and expected feature columns.

```python
import joblib
import pandas as pd

artifact = joblib.load("results/best_model.joblib")

preprocessor = artifact["preprocessor"]
model = artifact["model"]
encoder = artifact["target_encoder"]
feature_columns = artifact["feature_columns"]

# Replace these example values with actual feature values.
new_data = pd.DataFrame(
    [[/* your feature values */]],
    columns=feature_columns,
)

X = preprocessor.transform(new_data)

if hasattr(X, "toarray"):
    X = X.toarray()

predictions = model.predict(X)
predicted_labels = encoder.inverse_transform(predictions)

print(predicted_labels)
```

Replace `/* your feature values */` with actual values in the same order as the dataset's feature columns.

## Notes and Limitations

- Results depend on the dataset, hardware, and model configuration.
- Training time and inference latency are machine-dependent.
- Inference latency measures prediction time on already-preprocessed data; preprocessing time is excluded.
- The balanced score is a project-specific comparison metric, not a universal measure of model quality.
- The current implementation does not include cross-validation or direct model-size benchmarking.

## License

Copyright (c) 2026 Naeim Mousavi. All rights reserved.

See `LICENSE` for licensing terms.
