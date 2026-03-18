from matplotlib import lines
import pandas as pd


def percentile_bin(series):
    q = series.quantile([0.2, 0.4, 0.6, 0.8]).values

    def label(v):
        if v == 0:
            return "zero"
        elif v <= q[0]:
            return "very_low"
        elif v <= q[1]:
            return "low"
        elif v <= q[2]:
            return "medium"
        elif v <= q[3]:
            return "high"
        else:
            return "very_high"

    return series.apply(label)


def compute_feature_levels(df, model_features):
    feature_levels = {}

    for feat in model_features:
        if feat in df.columns:
            feature_levels[feat] = percentile_bin(df[feat])

    return feature_levels


def describe_row(df, feature_levels, model_features, row_index):
    descriptions = []

    for feat in model_features:
        value = df.loc[row_index, feat]
        level = feature_levels[feat].loc[row_index]

        name = feat.replace("_", " ").lower()
        sentence = f"{name} is {level} ({value})."

        descriptions.append(sentence)

    return descriptions


def build_incident_packet_descriptive(
    df,
    feature_levels,
    model_features,
    row_index,
    predicted_label,
    confidence,
    true_label=None
):
    lines = []

    lines.append("Network intrusion alert detected.")
    lines.append("")
    lines.append("📊 Model Output:")
    lines.append(f"- Predicted attack: {predicted_label}")
    lines.append(f"- Confidence: {confidence:.4f}")

    if true_label is not None:
        lines.append(f"- Ground truth: {true_label}")
    else:
        lines.append("- Ground truth: Not available")

    lines.append("")
    lines.append("Observed traffic behavior:")

    feature_sentences = describe_row(
        df,
        feature_levels,
        model_features,
        row_index
    )

    for s in feature_sentences:
        lines.append(f"- {s}")

    return "\n".join(lines)


def build_incident_data(
    row_index,
    pred_id,
    pred_label,
    confidence,
    true_label,
    incident_packet,
    model_features
):
    return {
        "row_index": int(row_index),
        "predicted_id": pred_id,
        "predicted_label": pred_label,
        "confidence": confidence,
        "true_label": true_label,
        "incident_packet": incident_packet,
        "incident_packet_type": "descriptive",
        "model_features": model_features
    }