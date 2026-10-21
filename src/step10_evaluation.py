from __future__ import annotations

import csv
import math
import random
from pathlib import Path


def evaluate_experimental_data(summary_csv: Path, experimental_csv: Path, output_csv: Path) -> dict:
    with summary_csv.open(newline="", encoding="utf-8") as handle:
        predictions = {row["Ligand"]: row for row in csv.DictReader(handle)}
    with experimental_csv.open(newline="", encoding="utf-8") as handle:
        measured = list(csv.DictReader(handle))
    rows = []
    for row in measured:
        ligand = row["Ligand"]
        if ligand not in predictions:
            continue
        prediction = predictions[ligand]
        rows.append({
            "Ligand": ligand,
            "Experimental_Value": row.get("Experimental_Value", ""),
            "Predicted_Affinity_kcal_mol": prediction.get("Binding_Affinity_kcal_mol", ""),
            "Predicted_Competent": prediction.get("Catalytically_Competent", ""),
        })
    metrics = {"matched": len(rows)}
    numeric = [(float(row["Experimental_Value"]), float(row["Predicted_Affinity_kcal_mol"])) for row in rows if row["Experimental_Value"] not in ("", None) and row["Predicted_Affinity_kcal_mol"] not in ("", None)]
    if len(numeric) >= 2:
        x, y = zip(*numeric)
        metrics["pearson_r"] = _pearson(x, y)
        metrics["spearman_r"] = _spearman(x, y)
    labels = [int(row["Experimental_Value"]) for row in rows if row["Experimental_Value"] in ("0", "1", 0, 1)]
    scores = [-float(row["Predicted_Affinity_kcal_mol"]) for row in rows if row["Experimental_Value"] in ("0", "1", 0, 1) and row["Predicted_Affinity_kcal_mol"] not in ("", None)]
    if len(set(labels)) == 2 and len(labels) == len(scores):
        metrics["roc_auc"] = _roc_auc(labels, scores)
        metrics["pr_auc"] = _pr_auc(labels, scores)
        metrics["enrichment_factor_top_25pct"] = _enrichment_factor(labels, scores, 0.25)
        metrics["roc_auc_bootstrap_95ci"] = _bootstrap_auc_ci(labels, scores)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Ligand", "Experimental_Value", "Predicted_Affinity_kcal_mol", "Predicted_Competent"])
        writer.writeheader()
        writer.writerows(rows)
    return metrics


def _pearson(x, y) -> float:
    x_mean, y_mean = sum(x) / len(x), sum(y) / len(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    denominator = math.sqrt(sum((a - x_mean) ** 2 for a in x) * sum((b - y_mean) ** 2 for b in y))
    return round(numerator / denominator, 6) if denominator else 0.0


def _spearman(x, y) -> float:
    return _pearson(_ranks(x), _ranks(y))


def _ranks(values):
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    for rank, index in enumerate(order, 1):
        ranks[index] = float(rank)
    return ranks


def _roc_auc(labels, scores) -> float:
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    wins = sum(1.0 if positive > negative else 0.5 if positive == negative else 0.0 for positive in positives for negative in negatives)
    return round(wins / (len(positives) * len(negatives)), 6)


def _pr_auc(labels, scores) -> float:
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    positives = sum(labels)
    if positives == 0:
        return 0.0
    tp = 0
    previous_recall = 0.0
    area = 0.0
    for rank, index in enumerate(order, 1):
        tp += labels[index]
        recall = tp / positives
        precision = tp / rank
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return round(area, 6)


def _enrichment_factor(labels, scores, fraction: float) -> float:
    cutoff = max(1, math.ceil(len(labels) * fraction))
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    top_rate = sum(labels[index] for index in order[:cutoff]) / cutoff
    baseline = sum(labels) / len(labels) if labels else 0.0
    return round(top_rate / baseline, 6) if baseline else 0.0


def _bootstrap_auc_ci(labels, scores, iterations: int = 200, seed: int = 17) -> list[float]:
    if len(labels) < 4 or len(set(labels)) < 2:
        return [0.0, 0.0]
    rng = random.Random(seed)
    values = []
    for _ in range(iterations):
        indices = [rng.randrange(len(labels)) for _ in labels]
        sampled_labels = [labels[index] for index in indices]
        if len(set(sampled_labels)) < 2:
            continue
        values.append(_roc_auc(sampled_labels, [scores[index] for index in indices]))
    if not values:
        return [0.0, 0.0]
    values.sort()
    return [round(values[int(len(values) * 0.025)], 6), round(values[int(len(values) * 0.975) - 1], 6)]
