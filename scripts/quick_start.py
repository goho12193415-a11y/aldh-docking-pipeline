from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import PDBParser
from src.step1_ligand_prep import discover_raw_ligands, fetch_smiles
from src.step7_advanced import auto_configure_receptor, detect_catalytic_cys


def _pick_chain(pdb_path: Path) -> tuple[str, dict]:
    structure = PDBParser(QUIET=True).get_structure("quick", str(pdb_path))
    candidates = []
    for chain in structure[0]:
        protein_residues = [residue for residue in chain if residue.id[0] == " " and "CA" in residue]
        cofactors = [residue for residue in chain if residue.id[0] != " " and residue.get_resname().strip() in {"NAD", "NAP", "NDP", "FAD", "FMN"}]
        try:
            catalytic = detect_catalytic_cys(pdb_path, chain.id)
        except (KeyError, ValueError):
            catalytic = None
        candidates.append({
            "chain": chain.id,
            "protein_residue_count": len(protein_residues),
            "cofactor_count": len(cofactors),
            "cofactor_names": sorted({residue.get_resname().strip() for residue in cofactors}),
            "catalytic_cys": catalytic,
        })
    if not candidates:
        raise ValueError("No protein chain found in receptor PDB.")
    selected = sorted(
        candidates,
        key=lambda item: (
            -int(item["cofactor_count"] > 0),
            -int(item["catalytic_cys"] is not None),
            -item["cofactor_count"],
            item["chain"],
            -item["protein_residue_count"],
        ),
    )[0]
    rationale = {
        "selected_chain": selected["chain"],
        "selection_rule": "cofactor_then_catalytic_cys_then_cofactor_count_then_deterministic_chain_order",
        "candidates": candidates,
    }
    return selected["chain"], rationale


def _parse_ligands(values: list[str] | None, raw_dir: Path | None = None) -> list[dict]:
    ligands = []
    for value in values or []:
        text = value.strip()
        if not text:
            continue
        if "=" in text:
            name, rhs = text.split("=", 1)
            name, rhs = name.strip(), rhs.strip()
            if not name or not rhs:
                raise ValueError(f"Ligand must have both name and value: {value}")
            if rhs.lower().endswith((".mol2", ".sdf", ".mol", ".pdb", ".pdbqt")):
                ligands.append({"name": name, "file": rhs})
            else:
                ligands.append({"name": name, "smiles": rhs})
        elif text.lower().endswith((".mol2", ".sdf", ".mol", ".pdb", ".pdbqt")):
            p = Path(text)
            ligands.append({"name": p.stem, "file": str(p)})
        else:
            name = text
            smiles = fetch_smiles(name)
            ligands.append({"name": name, "smiles": smiles})
    if raw_dir is not None and raw_dir.is_dir():
        ligands = discover_raw_ligands(raw_dir, ligands)
    if not ligands:
        raise ValueError("At least one ligand must be provided via --ligand or data/02_raw_ligands/.")
    return ligands


def create_quick_config(project_root: Path, receptor: Path | str, ligand_values: list[str], output: Path) -> Path:
    template_path = project_root / "config" / "pipeline_config.yaml"
    with template_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    raw_rec_parts = [p.strip() for p in str(receptor).split(";") if p.strip()]
    resolved_receptors = []
    for rp in raw_rec_parts:
        cand = Path(rp)
        if not cand.is_absolute():
            cand = (project_root / cand).resolve()
        else:
            cand = cand.resolve()
        if not cand.is_file():
            raise FileNotFoundError(cand)
        resolved_receptors.append(cand)
    if not resolved_receptors:
        raise FileNotFoundError(receptor)
    receptor = resolved_receptors[0]
    chain, selection = _pick_chain(receptor)
    config = copy.deepcopy(config)
    config["receptor"]["input_pdb"] = str(receptor)
    config["receptor"]["input_pdbs"] = [str(r) for r in resolved_receptors]
    if len(resolved_receptors) > 1:
        config.setdefault("advanced", {}).setdefault("receptor_ensemble", {})["enabled"] = True
        config["advanced"]["receptor_ensemble"]["inputs"] = [str(r) for r in resolved_receptors[1:]]
    config["receptor"]["chain_id"] = chain
    config["receptor"]["cofactor_chain"] = chain
    config["receptor"]["template_pdb"] = None
    config["ligands"] = _parse_ligands(ligand_values, project_root / "data" / "02_raw_ligands")
    try:
        cys = detect_catalytic_cys(receptor, chain)
        config["receptor"]["catalytic_residue"] = cys
        config["advanced"]["auto_detect_catalytic_cys"] = True
    except (KeyError, ValueError):
        config["advanced"]["auto_detect_catalytic_cys"] = False
    auto_configure_receptor(receptor, config)
    config["project"]["name"] = f"Quick_{receptor.stem}"
    config["project"]["auto_selection"] = selection
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Create an auto-configured docking experiment.")
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--receptor", type=Path, required=True)
    parser.add_argument("--ligand", action="append", default=[], help="Name=SMILES or Name; repeat for multiple ligands")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.project / "config" / "quick_config.yaml"
    path = create_quick_config(args.project.resolve(), args.receptor, args.ligand, output)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
