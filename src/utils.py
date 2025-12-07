# src/utils.py
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

MODEL_FILENAME = "best_model_gradient_boosting.pkl"

def load_model(path=None):
    """Load model and handle when pickle wraps a dict {'model': ...}."""
    if path is None:
        path = os.path.join(os.getcwd(), MODEL_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    obj = joblib.load(path)
    if isinstance(obj, dict) and 'model' in obj:
        return obj['model']
    return obj

def ensure_features(df, features):
    return set(features).issubset(df.columns)

def sample_input_csv(features, path="sample_input.csv"):
    df = pd.DataFrame(columns=features)
    df.to_csv(path, index=False)
    return path

def plot_confusion(y_true, y_pred, labels=[0,1]):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax,
                xticklabels=['Normal','Fraud'], yticklabels=['Normal','Fraud'])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return buf

def compute_metrics(y_true, y_pred, y_prob=None):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    auc = None
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = None
    return {"report": report, "roc_auc": auc}
