import os
import json
import pickle
import random
import numpy as np
import pandas as pd
from datasets import load_dataset


def load_model(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def load_label_mapping(label_map_path):
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_to_id = json.load(f)
    id_to_label = {v: k for k, v in label_to_id.items()}
    return label_to_id, id_to_label


def load_data(dataset_name, hf_token):
    dataset = load_dataset(dataset_name, token=hf_token)
    df = dataset["train"].to_pandas()
    return df


def get_model_features(model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    else:
        raise ValueError("Model does not expose feature_names_in_")


def select_random_row(df):
    idx = random.randint(0, len(df) - 1)
    row = df.iloc[idx].copy()
    return idx, row


def predict_sample(model, row, model_features, id_to_label):
    X = row[model_features].values.reshape(1, -1)

    pred_id = int(model.predict(X)[0])

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        confidence = float(np.max(proba))
    else:
        proba = None
        confidence = None

    pred_label = id_to_label.get(pred_id, f"UNKNOWN_{pred_id}")

    return pred_id, pred_label, confidence


def extract_true_label(row, df):
    candidates = ["label", "Label", "attack", "attack_name", "y", "y_true"]

    for c in candidates:
        if c in df.columns:
            return row[c]

    return None