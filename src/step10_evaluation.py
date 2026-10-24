from __future__ import annotations

import csv
import math
import random
from pathlib import Path


def _exp_to_delta_g_kcal(val: float, unit_hint: str = "") -> float | None:
    """Convert experimental Ki/Kd/IC50 (in M, uM, nM, pIC50/pKd) or direct DeltaG into kcal/mol at 298.15 K."""
    rt = 0.59248  # kcal/mol at 298.15 K
    u = unit_hint.lower()
    if "kcal" in u or val < 0.0:
        return float(val)
    if "pic50" in u or "pkd" in u or "pki" in u:
        return round(-rt * math.log(10.0) * float(val), 4)
    if val <= 0.0:
        return None
    if "nm" in u:
        molar = val * 1e-9
    elif "um" in u:
        molar = val * 1e-6
    elif "mm" in u:
        molar = val * 1e-3
    elif val > 1.0:
        # Typical IC50/Ki reported in uM
        molar = val * 1e-6
    else:
        molar = val
    return round(rt * math.log(max(1e-15, molar)), 4)


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
            "Experimental_Unit": row.get("Unit", ""),
            "Predicted_Affinity_kcal_mol": prediction.get("Binding_Affinity_kcal_mol", ""),
            "PhysChem_Score_kcal_mol": prediction.get("PhysChem_Score_kcal_mol", ""),
            "Catalytic_Geometry_Score": prediction.get("Catalytic_Geometry_Score", ""),
            "Universal_Consensus_Rank": prediction.get("Universal_Consensus_Rank", ""),
            "Predicted_Competent": prediction.get("Catalytically_Competent", ""),
        })

    metrics: dict = {"matched": len(rows), "by_scorer": {}}
    numeric = [
        (float(row["Experimental_Value"]), float(row["Predicted_Affinity_kcal_mol"]), str(row.get("Experimental_Unit", "")))
        for row in rows
        if row["Experimental_Value"] not in ("", None) and row["Predicted_Affinity_kcal_mol"] not in ("", None)
    ]
    if len(numeric) >= 2:
        x = [item[0] for item in numeric]
        y = [item[1] for item in numeric]
        metrics["pearson_r"] = _pearson(x, y)
        metrics["spearman_r"] = _spearman(x, y)

        # Compute thermodynamic DeltaG RMSE and MAE when non-binary experimental affinities/IC50s are provided
        if not set(x).issubset({0.0, 1.0}):
            dg_pairs = []
            for exp_v, pred_v, unit_str in numeric:
                dg_exp = _exp_to_delta_g_kcal(exp_v, unit_str)
                if dg_exp is not None:
                    dg_pairs.append((dg_exp, pred_v))
            if dg_pairs:
                sq_err = [(de - dp) ** 2 for de, dp in dg_pairs]
                abs_err = [abs(de - dp) for de, dp in dg_pairs]
                metrics["dg_rmse_kcal_mol"] = round(math.sqrt(sum(sq_err) / len(sq_err)), 4)
                metrics["dg_mae_kcal_mol"] = round(sum(abs_err) / len(abs_err), 4)

    labels = [int(float(row["Experimental_Value"])) for row in rows if str(row["Experimental_Value"]).strip() in ("0", "1", "0.0", "1.0")]
    scores = [
        -float(row["Predicted_Affinity_kcal_mol"])
        for row in rows
        if str(row["Experimental_Value"]).strip() in ("0", "1", "0.0", "1.0") and row["Predicted_Affinity_kcal_mol"] not in ("", None)
    ]
    if len(set(labels)) == 2 and len(labels) == len(scores):
        metrics["roc_auc"] = _roc_auc(labels, scores)
        metrics["pr_auc"] = _pr_auc(labels, scores)
        metrics["bedroc_alpha20"] = _bedroc(labels, scores, alpha=20.0)
        metrics["enrichment_factor_top_25pct"] = _enrichment_factor(labels, scores, 0.25)
        metrics["roc_auc_bootstrap_95ci"] = _bootstrap_auc_ci(labels, scores)

    # Benchmark all 4 scoring functions side by side
    scorer_specs = [
        ("vina_affinity", "Predicted_Affinity_kcal_mol", -1.0),
        ("physchem_rescore", "PhysChem_Score_kcal_mol", -1.0),
        ("catalytic_geometry", "Catalytic_Geometry_Score", +1.0),
        ("universal_consensus_rank", "Universal_Consensus_Rank", -1.0),
    ]
    for s_name, col, sign in scorer_specs:
        valid_pairs = [
            (float(r["Experimental_Value"]), sign * float(r[col]))
            for r in rows
            if r.get("Experimental_Value") not in ("", None) and r.get(col) not in ("", None)
        ]
        if len(valid_pairs) >= 2:
            vx, vy = zip(*valid_pairs)
            s_entry: dict[str, float] = {
                "pearson_r": _pearson(vx, vy),
                "spearman_r": _spearman(vx, vy),
            }
            if set(vx).issubset({0.0, 1.0}) and len(set(vx)) == 2:
                b_labels = [int(v) for v in vx]
                b_scores = list(vy)
                s_entry["roc_auc"] = _roc_auc(b_labels, b_scores)
                s_entry["pr_auc"] = _pr_auc(b_labels, b_scores)
                s_entry["bedroc_alpha20"] = _bedroc(b_labels, b_scores, alpha=20.0)
            metrics["by_scorer"][s_name] = s_entry

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Ligand", "Experimental_Value", "Predicted_Affinity_kcal_mol",
        "PhysChem_Score_kcal_mol", "Catalytic_Geometry_Score",
        "Universal_Consensus_Rank", "Predicted_Competent",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
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
    """Average ranks for ties (required for a correct Spearman correlation)."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        average_rank = (position + end + 2) / 2.0  # 1-based average of position+1..end+1
        for pointer in range(position, end + 1):
            ranks[order[pointer]] = average_rank
        position = end + 1
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


def _bedroc(labels, scores, alpha: float = 20.0) -> float:
    """Compute Truchon & Bayly (2007) Boltzmann-Enhanced Discrimination of ROC (BEDROC) for early virtual screening recognition."""
    n = len(labels)
    n_act = sum(1 for l in labels if l == 1)
    if n == 0 or n_act == 0 or n_act == n:
        return 0.0
    order = sorted(range(n), key=lambda idx: scores[idx], reverse=True)
    ra = n_act / n
    s_sum = sum(math.exp(-alpha * (rank / n)) for rank, idx in enumerate(order, 1) if labels[idx] == 1)
    rie_num = s_sum / n_act
    rie_den = (1.0 - math.exp(-alpha)) / (math.exp(alpha / n) - 1.0) / n
    if abs(rie_den) < 1e-12:
        return 0.0
    rie = rie_num / rie_den
    rie_min = (1.0 - math.exp(alpha * ra)) / (ra * (1.0 - math.exp(alpha)))
    rie_max = (1.0 - math.exp(-alpha * ra)) / (ra * (1.0 - math.exp(-alpha)))
    if abs(rie_max - rie_min) < 1e-12:
        return 0.0
    return round(max(0.0, min(1.0, (rie - rie_min) / (rie_max - rie_min))), 6)


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
