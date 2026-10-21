from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np

from .common import atom_dict, distance

LOGGER = logging.getLogger(__name__)


def parse_models(pdbqt_path: Path) -> list[list[dict]]:
    models, current = [], []
    for line in pdbqt_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("MODEL"):
            current = []
        elif line.startswith("ATOM") or line.startswith("HETATM"):
            try:
                current.append(atom_dict(line))
            except (ValueError, IndexError):
                continue
        elif line.startswith("ENDMDL"):
            if current:
                models.append(current)
    if not models and current:
        models.append(current)
    elif current and models and current != models[-1]:
        models.append(current)
    return models


def parse_vina_scores(pdbqt_path: Path) -> list[float | None]:
    scores: list[float | None] = []
    for line in pdbqt_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("REMARK VINA RESULT:"):
            fields = line.split()
            try:
                scores.append(float(fields[3]))
            except (IndexError, ValueError):
                scores.append(None)
    return scores


def find_carbonyl(model: list[dict], min_bond: float = 1.15, max_bond: float = 1.35) -> tuple[dict, dict]:
    oxygens = [a for a in model if a["element"].upper() == "O" or a["name"] == "OA"]
    carbons = [a for a in model if a["element"].upper() == "C" or a["name"] in {"C", "A"}]
    if not oxygens or not carbons:
        raise ValueError("Ligand model does not contain recognizable carbonyl O/C atoms.")
    candidates = [
        (oxygen, carbon, distance(oxygen["coord"], carbon["coord"]))
        for oxygen in oxygens
        for carbon in carbons
        if min_bond <= distance(oxygen["coord"], carbon["coord"]) <= max_bond
    ]
    if not candidates:
        raise ValueError("No carbonyl C-O pair was found within the configured bond range.")
    oxygen, carbonyl, _ = min(candidates, key=lambda item: item[2])
    return oxygen, carbonyl


def read_receptor_atoms(pdb_path: Path) -> list[dict]:
    atoms = []
    for line in pdb_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(("ATOM", "HETATM")):
            try:
                atoms.append(atom_dict(line))
            except (ValueError, IndexError):
                continue
    return atoms


def clean_ligand_pdbqt(model: list[dict], start_serial: int = 9000) -> list[str]:
    lines = []
    for index, atom in enumerate(model, start=start_serial):
        element = {"OA": "O", "NA": "N", "SA": "S", "HD": "H", "A": "C"}.get(atom["name"], atom["element"][:1].upper())
        x, y, z = atom["coord"]
        lines.append(f"HETATM{index:5d} {atom['name'][:4]:>4s} LIG L{999:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}")
    return lines


def score_docking(output_pdbqt: Path, receptor_pdb: Path, ligand_name: str, config: dict, reports_dir: Path) -> dict:
    models = parse_models(output_pdbqt)
    scores = parse_vina_scores(output_pdbqt)
    receptor_atoms = read_receptor_atoms(receptor_pdb)
    catalytic = config["receptor"]["catalytic_residue"]
    cys = next((a for a in receptor_atoms if a["resname"] == "CYS" and a["chain"] == catalytic["chain"] and a["resseq"] == catalytic["number"] and a["name"] == "SG"), None)
    if cys is None:
        raise ValueError("Catalytic Cys SG was not found in receptor.")
    nad = next((a for a in receptor_atoms if a["resname"] in {"NAD", "NAP", "NDP"} and a["name"] in {"C4N", "C4"}), None)
    oxyanion = config["receptor"].get("oxyanion_residue")
    oxyanion_nd2 = next(
        (
            a for a in receptor_atoms
            if oxyanion
            and a["resname"] == oxyanion["name"]
            and a["resseq"] == oxyanion["number"]
            and a["name"] == oxyanion["atom"]
        ),
        None,
    )
    cys_backbone_n = next(
        (
            a for a in receptor_atoms
            if a["resname"] == "CYS"
            and a["chain"] == catalytic["chain"]
            and a["resseq"] == catalytic["number"]
            and a["name"] == "N"
        ),
        None,
    )
    rows = []
    bond_min = config["thresholds"].get("carbonyl_bond_min", 1.15)
    bond_max = config["thresholds"].get("carbonyl_bond_max", 1.35)
    for mode_index, model in enumerate(models, start=1):
        oxygen, carbonyl = find_carbonyl(model, bond_min, bond_max)
        row = {
            "Ligand": ligand_name, "Mode": mode_index,
            "Binding_Affinity_kcal_mol": scores[mode_index - 1] if mode_index <= len(scores) else "",
            "d_nuc_A": round(distance(carbonyl["coord"], cys["coord"]), 3),
            "d_hyd_A": round(distance(carbonyl["coord"], nad["coord"]), 3) if nad else "",
            "d_oxy_asn_A": round(distance(oxygen["coord"], oxyanion_nd2["coord"]), 3) if oxyanion_nd2 else "",
            "d_oxy_cysN_A": round(distance(oxygen["coord"], cys_backbone_n["coord"]), 3) if cys_backbone_n else "",
            "Catalytically_Competent": False,
        }
        nuc_ok = config["thresholds"]["d_nuc_min"] <= row["d_nuc_A"] <= config["thresholds"]["d_nuc_max"]
        hyd_ok = nad is None or row["d_hyd_A"] <= config["thresholds"].get("d_hyd_max", 4.5)
        row["Catalytically_Competent"] = nuc_ok and hyd_ok
        rows.append((row, model))
    if not rows:
        raise ValueError(f"No docking models found in {output_pdbqt}")
    scored_rows = [item for item in rows if item[0]["Binding_Affinity_kcal_mol"] != ""]
    affinity_order = lambda item: item[0]["Binding_Affinity_kcal_mol"]
    top_row, _ = min(scored_rows, key=affinity_order) if scored_rows else rows[0]
    competent = [item for item in rows if item[0]["Catalytically_Competent"]]
    best_row, best_model = min(competent or rows, key=lambda item: (
        item[0]["Binding_Affinity_kcal_mol"] if item[0]["Binding_Affinity_kcal_mol"] != "" else float("inf"),
        item[0]["d_nuc_A"],
    ))
    best_row["Top_Affinity_Mode"] = top_row["Mode"]
    best_row["Top_Affinity_kcal_mol"] = top_row["Binding_Affinity_kcal_mol"]
    best_row["Parsed_Mode_Count"] = len(rows)
    best_row["Catalytically_Competent_Mode_Count"] = len(competent)
    best_row["Selection_Status"] = (
        "Catalytically_Competent"
        if competent
        else "Affinity_Fallback_No_Catalytic_Pose"
    )
    best_row["Best_Catalytic_Mode"] = best_row["Mode"] if competent else ""
    best_row["Best_Catalytic_kcal_mol"] = (
        best_row["Binding_Affinity_kcal_mol"] if competent else ""
    )
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "docking_summary.csv"
    existing: list[dict[str, str]] = []
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            existing = [
                row for row in csv.DictReader(handle)
                if row.get("Ligand") != ligand_name
            ]
    fieldnames = list(dict.fromkeys(
        [key for row in existing for key in row.keys()] + list(best_row.keys())
    ))
    existing.append({key: best_row.get(key, "") for key in fieldnames})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing)

    active_path = reports_dir / "active_poses" / f"{ligand_name}_active.pdb"
    active_path.parent.mkdir(parents=True, exist_ok=True)
    protein_lines = [a["line"] for a in receptor_atoms]
    active_path.write_text("\n".join(protein_lines + clean_ligand_pdbqt(best_model)) + "\nEND\n", encoding="utf-8")
    best_row["active_pose"] = str(active_path)
    return best_row
