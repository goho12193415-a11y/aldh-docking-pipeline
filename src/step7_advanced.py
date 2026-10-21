from __future__ import annotations

import csv
import json
import logging
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import numpy as np
from Bio.PDB import PDBParser

from .common import require_executable

LOGGER = logging.getLogger(__name__)

METAL_SYMBOLS = {
    "LI", "BE", "NA", "MG", "AL", "K", "CA", "SC", "TI", "V", "CR",
    "MN", "FE", "CO", "NI", "CU", "ZN", "GA", "SR", "Y", "ZR", "NB",
    "MO", "TC", "RU", "RH", "PD", "AG", "CD", "IN", "SN", "SB", "BA",
    "LA", "CE", "PR", "ND", "SM", "EU", "GD", "TB", "DY", "HO", "ER",
    "TM", "YB", "LU", "HF", "TA", "W", "RE", "OS", "IR", "PT", "AU",
    "HG", "TL", "PB", "BI",
}


def assess_ligand_compatibility(name: str, smiles: str) -> dict:
    """Classify limitations before docking; this is a warning gate, not a validity claim."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES for {name}: {smiles}")
    elements = {atom.GetSymbol().upper() for atom in molecule.GetAtoms()}
    metal_atoms = sorted(elements & METAL_SYMBOLS)
    formal_charge = sum(atom.GetFormalCharge() for atom in molecule.GetAtoms())
    heavy_atoms = molecule.GetNumHeavyAtoms()
    ring_count = Lipinski.RingCount(molecule)
    oxygen_count = sum(atom.GetSymbol() == "O" for atom in molecule.GetAtoms())
    stereocenter_count = len(Chem.FindMolChiralCenters(molecule, includeUnassigned=True))
    flags = []
    if metal_atoms:
        flags.append("metal_or_inorganic_species_requires_specialized_parameters")
    if any(atom.GetIsotope() for atom in molecule.GetAtoms()):
        flags.append("isotopically_labeled_species_needs_manual_review")
    if any(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms()):
        flags.append("radical_species_not_reliably_supported_by_standard_vina_preparation")
    if heavy_atoms > 80:
        flags.append("large_ligand_may_exceed_standard_vina_sampling_assumptions")
    if Lipinski.NumRotatableBonds(molecule) > 20:
        flags.append("high_flexibility_requires_conformer_and_sampling_validation")
    if abs(formal_charge) >= 2:
        flags.append("multiply_charged_species_requires_protonation_and_electrostatics_review")
    sugar_like = _looks_like_polysaccharide(molecule, heavy_atoms, oxygen_count, ring_count)
    if sugar_like:
        flags.append("polysaccharide_or_oligosaccharide_requires_sugar_specific_conformers_and_sampling")
    likely_covalent = _has_likely_covalent_warhead(molecule)
    if likely_covalent:
        flags.append("possible_covalent_reactivity_requires_covalent_docking_or_qm_mm")
    if metal_atoms or sugar_like or any(flag.endswith("requires_covalent_docking_or_qm_mm") for flag in flags):
        status = "specialized_method_required"
    elif flags:
        status = "manual_review_required"
    else:
        status = "standard_small_molecule"
    return {
        "name": name,
        "status": status,
        "heavy_atom_count": heavy_atoms,
        "rotatable_bond_count": Lipinski.NumRotatableBonds(molecule),
        "formal_charge": formal_charge,
        "molecular_weight": round(Descriptors.MolWt(molecule), 3),
        "ring_count": ring_count,
        "oxygen_count": oxygen_count,
        "stereocenter_count": stereocenter_count,
        "sugar_like": sugar_like,
        "elements": sorted(elements),
        "metal_atoms": metal_atoms,
        "flags": flags,
        "standard_vina_interpretation": not flags,
    }


def assess_ligand_chemical_state(name: str, smiles: str, ph: float = 7.4) -> dict:
    """Audit ionization and salt ambiguity without pretending to predict pKa."""
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES for {name}: {smiles}")
    fragments = Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=True)
    acidic_patterns = {
        "carboxylic_acid_or_carboxylate": "[CX3](=O)[O;H1,-1]",
        "phenol_or_enol": "[c,C][O;H1,-1]",
        "thiol_or_thiolate": "[S;H1,-1]",
    }
    basic_patterns = {
        "aliphatic_or_aromatic_amine": "[N;H1,H2,H3;+0;!$(N-C=O)]",
        "heteroaromatic_nitrogen": "[nH0;+0]",
    }
    acidic_sites = [
        label for label, pattern in acidic_patterns.items()
        if molecule.HasSubstructMatch(Chem.MolFromSmarts(pattern))
    ]
    basic_sites = [
        label for label, pattern in basic_patterns.items()
        if molecule.HasSubstructMatch(Chem.MolFromSmarts(pattern))
    ]
    formal_charge = sum(atom.GetFormalCharge() for atom in molecule.GetAtoms())
    flags = []
    if not 0.0 <= float(ph) <= 14.0:
        raise ValueError(f"pH must be between 0 and 14, got {ph}")
    if len(fragments) > 1:
        flags.append("multiple_fragments_or_salt_form_requires_standardized_counterion_handling")
    if acidic_sites or basic_sites:
        flags.append("ionizable_groups_detected_pka_and_protonation_state_not_predicted")
    if formal_charge:
        flags.append("explicit_formal_charge_requires_ph_and_electrostatics_review")
    status = "manual_review_required" if flags else "single_neutral_state_candidate"
    return {
        "name": name,
        "input_smiles": smiles,
        "experimental_ph": float(ph),
        "formal_charge": formal_charge,
        "fragment_count": len(fragments),
        "acidic_site_types": acidic_sites,
        "basic_site_types": basic_sites,
        "flags": flags,
        "status": status,
        "interpretation": "Heuristic chemical-state audit only; pKa, protonation enumeration, and force-field charges are not generated here.",
    }


def write_chemical_state_report(rows: list[dict], output_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _looks_like_polysaccharide(molecule, heavy_atoms: int, oxygen_count: int, ring_count: int) -> bool:
    """Conservative heuristic for sugar-rich oligosaccharides and polysaccharides."""
    from rdkit import Chem

    if heavy_atoms >= 35 and oxygen_count >= 8:
        return True
    sugar_ring = Chem.MolFromSmarts("[O;R]([C;R])[C;R]")
    return ring_count >= 2 and molecule.HasSubstructMatch(sugar_ring) and oxygen_count >= 6


def _has_likely_covalent_warhead(molecule) -> bool:
    """Conservative motifs only; ordinary thioethers must not trigger this warning."""
    from rdkit import Chem

    warhead_patterns = (
        "[C;$(C=O)]-[Cl,Br,I]",
        "[C;$(C=O)]-[S;H1]",
        "C=C-C(=O)[N,O,S]",
        "C#C-[C;$(C=O)]",
        "[N,O,S]-[N]=[N]=[N]",
    )
    return any(molecule.HasSubstructMatch(Chem.MolFromSmarts(pattern)) for pattern in warhead_patterns)


def write_ligand_compatibility_report(rows: list[dict], output_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def assess_enzyme_readiness(pdb_path: Path, config: dict, output_json: Path) -> dict:
    """Audit whether the configured enzyme state is suitable for interpretation."""
    structure = PDBParser(QUIET=True).get_structure("receptor", str(pdb_path))
    receptor = config.get("receptor", {})
    chain_id = receptor.get("chain_id", "A")
    chain = structure[0][chain_id]
    flags = []
    residues = []
    for role in ("catalytic_residue", "acid_base_residue", "stabilizing_residue"):
        item = receptor.get(role)
        if item:
            residues.append((role, item))
    residue_checks = []
    for role, item in residues:
        number = int(item["number"])
        residue = chain[(" ", number, " ")] if (" ", number, " ") in chain else None
        expected = str(item.get("name", "")).upper()
        actual = residue.get_resname().strip().upper() if residue else None
        atoms = sorted(atom.get_name().strip() for atom in residue) if residue else []
        ok = residue is not None and (not expected or actual == expected)
        residue_checks.append({"role": role, "chain": chain_id, "number": number, "expected": expected, "actual": actual, "atoms": atoms, "ok": ok})
        if not ok:
            flags.append(f"{role}_missing_or_residue_mismatch")
    cofactors = set(receptor.get("keep_cofactors", []))
    found_cofactors = sorted({
        residue.get_resname().strip()
        for residue in chain
        if residue.id[0] != " " and residue.get_resname().strip() in cofactors
    })
    missing_cofactors = sorted(cofactors - set(found_cofactors))
    if cofactors and missing_cofactors and receptor.get("cofactor_chain", chain_id) == chain_id:
        flags.append("configured_cofactor_missing_from_target_chain")
    water_count = sum(1 for model in structure for chain_item in model for residue in chain_item if residue.get_resname().strip() in {"HOH", "WAT"})
    altloc_count = sum(1 for atom in chain.get_atoms() if atom.get_altloc() not in {" ", ""})
    if altloc_count:
        flags.append("alternate_location_atoms_require_conformer_selection")
    if water_count == 0:
        flags.append("no_crystallographic_waters_available_for_optional_water_review")
    if config.get("enzyme", {}).get("requires_cofactor", False) and not found_cofactors:
        flags.append("enzyme_requires_cofactor_but_none_is_present")
    status = "ready_with_review" if not flags else "manual_review_required"
    result = {
        "pdb": str(pdb_path),
        "chain": chain_id,
        "status": status,
        "residue_checks": residue_checks,
        "configured_cofactors": sorted(cofactors),
        "found_cofactors": found_cofactors,
        "missing_cofactors": missing_cofactors,
        "water_count": water_count,
        "alternate_location_atom_count": altloc_count,
        "flags": flags,
        "interpretation": "Readiness audit only; it does not validate protonation, catalytic mechanism, or force-field parameters.",
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def audit_receptor_state(pdb_path: Path, config: dict, output_json: Path) -> dict:
    """Create a conservative receptor-state audit before any docking interpretation."""
    structure = PDBParser(QUIET=True).get_structure("receptor_state", str(pdb_path))
    receptor = config.get("receptor", {})
    chain_id = receptor.get("chain_id", "A")
    chain = structure[0][chain_id]
    altloc_atoms = [
        {"chain": chain_id, "residue": residue.id[1], "atom": atom.get_name(), "altloc": atom.get_altloc()}
        for residue in chain
        for atom in residue
        if atom.get_altloc() not in {" ", ""}
    ]
    hetero_residues = sorted({
        residue.get_resname().strip()
        for residue in chain
        if residue.id[0] != " "
    })
    configured_cofactors = set(receptor.get("keep_cofactors", []))
    present_cofactors = sorted(set(hetero_residues) & configured_cofactors)
    metal_residues = sorted(set(hetero_residues) & METAL_SYMBOLS)
    flags = []
    if altloc_atoms:
        flags.append("alternate_locations_require_deterministic_selection")
    if configured_cofactors and not present_cofactors:
        flags.append("configured_cofactor_not_found_on_target_chain")
    if metal_residues:
        flags.append("metal_center_requires_coordination_review")
    water_count = sum(
        1
        for model in structure
        for model_chain in model
        for residue in model_chain
        if residue.get_resname().strip() in {"HOH", "WAT"}
    )
    if water_count:
        flags.append("conserved_water_candidates_require_manual_or_md_review")
    status = "manual_review_required" if flags else "ready_for_preparation"
    result = {
        "pdb": str(pdb_path),
        "chain": chain_id,
        "status": status,
        "alternate_location_atom_count": len(altloc_atoms),
        "alternate_locations": altloc_atoms[:100],
        "hetero_residues_on_target_chain": hetero_residues,
        "configured_cofactors": sorted(configured_cofactors),
        "present_cofactors": present_cofactors,
        "metal_residues": metal_residues,
        "water_count": water_count,
        "flags": flags,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": "State audit only; it does not repair missing atoms, protonate residues, or validate force-field parameters.",
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def audit_docking_box(pdb_path: Path, box: dict, config: dict, output_json: Path) -> dict:
    """Check that a configured box covers the intended catalytic landmarks."""
    structure = PDBParser(QUIET=True).get_structure("box_audit", str(pdb_path))
    receptor = config.get("receptor", {})
    chain_id = receptor.get("chain_id", "A")
    catalytic = receptor.get("catalytic_residue", {})
    points = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                if chain.id == chain_id and residue.get_resname().strip() == "CYS":
                    if residue.id[1] == int(catalytic.get("number", -1)) and "SG" in residue:
                        points["catalytic_cys_sg"] = residue["SG"].coord.astype(float)
                if residue.get_resname().strip() in {"NAD", "NAP", "NDP"}:
                    for atom_name in ("C4N", "C4"):
                        if atom_name in residue:
                            if chain.id == receptor.get("cofactor_chain", chain_id) or "cofactor_key" not in points:
                                points["cofactor_key"] = residue[atom_name].coord.astype(float)
                            break
    center = np.array([box["center_x"], box["center_y"], box["center_z"]], dtype=float)
    size = np.array([box["size_x"], box["size_y"], box["size_z"]], dtype=float)
    half = size / 2.0
    flags = []
    if np.any(size < 8.0):
        flags.append("docking_box_dimension_too_small_for_standard_ligand_sampling")
    if np.any(size > 40.0):
        flags.append("docking_box_dimension_too_large_for_focused_active_site_sampling")
    containment = {}
    for label, point in points.items():
        distance_from_center = np.abs(point - center)
        contained = bool(np.all(distance_from_center <= half))
        containment[label] = {
            "contained": contained,
            "distance_from_center_A": [round(float(value), 3) for value in distance_from_center],
        }
        if not contained:
            flags.append(f"{label}_outside_docking_box")
    if "catalytic_cys_sg" not in points:
        flags.append("catalytic_cys_sg_not_found_for_box_audit")
    if receptor.get("keep_cofactors") and "cofactor_key" not in points:
        flags.append("configured_cofactor_key_atom_not_found_for_box_audit")
    status = "manual_review_required" if flags else "box_passed"
    result = {
        "status": status,
        "box": box,
        "chain": chain_id,
        "landmark_containment": containment,
        "flags": flags,
        "interpretation": "Geometric coverage audit only; it does not prove that the box samples the biologically correct binding site.",
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def build_execution_plan(project_root: Path, config: dict, output_json: Path) -> dict:
    """Expand configured receptors, ligands, and seeds into an auditable run matrix."""
    receptor_cfg = config.get("receptor", {})
    ensemble_cfg = config.get("advanced", {}).get("receptor_ensemble", {})
    primary = Path(receptor_cfg.get("input_pdb", ""))
    receptor_inputs = [primary]
    if ensemble_cfg.get("enabled"):
        receptor_inputs.extend(Path(item) for item in ensemble_cfg.get("inputs", []))
    receptor_inputs = list(dict.fromkeys(receptor_inputs))
    replicate_cfg = config.get("advanced", {}).get("replicates", {})
    replicate_count = max(1, int(replicate_cfg.get("count", 1)))
    configured_seeds = [int(seed) for seed in replicate_cfg.get("seeds", [])]
    seeds = [
        configured_seeds[index % len(configured_seeds)]
        if configured_seeds
        else None
        for index in range(replicate_count)
    ]
    ligands = [
        {"name": str(item["name"]), "smiles": item.get("smiles")}
        for item in config.get("ligands", [])
        if item.get("name")
    ]
    runs = []
    for receptor_index, receptor_path in enumerate(receptor_inputs, start=1):
        for ligand in ligands:
            for replicate_index, seed in enumerate(seeds, start=1):
                runs.append({
                    "run_id": f"receptor{receptor_index}__{ligand['name']}__rep{replicate_index}",
                    "receptor_index": receptor_index,
                    "receptor": str(receptor_path),
                    "ligand": ligand["name"],
                    "seed": seed,
                    "replicate": replicate_index,
                    "comparison_group": ligand["name"],
                })
    flags = []
    missing_inputs = []
    for path in receptor_inputs:
        resolved = path if path.is_absolute() else project_root / path
        if not resolved.is_file():
            missing_inputs.append(str(resolved))
    if ensemble_cfg.get("enabled") and len(receptor_inputs) < 2:
        flags.append("receptor_ensemble_enabled_but_no_additional_receptor_inputs_configured")
    if missing_inputs:
        flags.append("execution_plan_contains_missing_receptor_inputs")
    result = {
        "status": "manual_review_required" if flags else "plan_ready",
        "receptor_count": len(receptor_inputs),
        "ligand_count": len(ligands),
        "replicate_count": replicate_count,
        "planned_run_count": len(runs),
        "receptors": [str(path) for path in receptor_inputs],
        "seeds": seeds,
        "runs": runs,
        "missing_inputs": missing_inputs,
        "flags": flags,
        "interpretation": "Execution plan only; scores from different receptors are not merged into one physical quantity.",
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def build_capability_report(config: dict, compatibility: list[dict], enzyme_readiness: dict | None = None) -> dict:
    """Summarize what this configured run can and cannot claim."""
    specialized = [row["name"] for row in compatibility if row["status"] == "specialized_method_required"]
    review = [row["name"] for row in compatibility if row["status"] == "manual_review_required"]
    enabled = config.get("advanced", {})
    claims = ["standard_small_molecule_initial_screening"]
    limitations = [
        "vina_affinity_is_not_experimental_binding_free_energy",
        "catalytic_geometry_is_not_proof_of_reactivity",
        "md_and_qmmm_files_are_preparation_artifacts_unless_external_runs_succeed",
    ]
    if enabled.get("replicates", {}).get("count", 1) > 1:
        claims.append("seed_replicate_stability_summary")
    if enabled.get("plip", {}).get("enabled"):
        claims.append("optional_plip_interaction_cross_check")
    if enabled.get("receptor_ensemble", {}).get("enabled"):
        claims.append("configured_receptor_ensemble_requires_per_structure_comparison")
    if enzyme_readiness and enzyme_readiness.get("status") == "ready_with_review":
        claims.append("enzyme_active_site_and_cofactor_readiness_audit")
    enzyme_flags = enzyme_readiness.get("flags", []) if enzyme_readiness else []
    if enzyme_flags:
        limitations.append("enzyme_structure_requires_manual_review_before_mechanistic_interpretation")
    return {
        "pipeline_scope": "mechanism_guided_protein_small_molecule_docking",
        "ready_for_standard_screening": not specialized,
        "requires_manual_review": review,
        "requires_specialized_method": specialized,
        "supported_claims": claims,
        "non_supported_claims": limitations,
        "strict_gate_recommendation": "block_interpretation" if specialized else "allow_with_review" if review or enzyme_flags else "allow_initial_screening",
        "enzyme_readiness": enzyme_readiness,
    }


def enforce_capability_gate(
    config: dict,
    compatibility: list[dict],
    enzyme_readiness: dict | None = None,
    allow_specialized: bool = False,
    mechanism: dict | None = None,
    box_audit: dict | None = None,
    receptor_state: dict | None = None,
) -> dict:
    """Prevent unsupported systems from silently producing standard claims."""
    specialized = [row["name"] for row in compatibility if row["status"] == "specialized_method_required"]
    review = [row["name"] for row in compatibility if row["status"] == "manual_review_required"]
    enzyme_flags = (enzyme_readiness or {}).get("flags", [])
    mechanism = mechanism or {"family": "aldh", "implemented_scoring": True}
    unsupported_mechanism = not bool(mechanism.get("implemented_scoring", False))
    box_flags = (box_audit or {}).get("flags", [])
    receptor_state_flags = (receptor_state or {}).get("flags", [])
    policy = config.get("safety", {})
    strict = bool(policy.get("strict_capability_gate", True))
    if (specialized or unsupported_mechanism or box_flags) and strict and not allow_specialized:
        reasons = list(specialized)
        if unsupported_mechanism:
            reasons.append(f"mechanism:{mechanism.get('family', 'unknown')}")
        reasons.extend(f"box:{flag}" for flag in box_flags)
        raise RuntimeError(
            "Capability gate blocked standard docking for: "
            + ", ".join(reasons)
            + ". Review reports/capability_report.json, or explicitly use --allow-specialized "
            "for a non-standard exploratory run."
        )
    return {
        "strict": strict,
        "overridden": bool((specialized or unsupported_mechanism or box_flags) and allow_specialized),
        "specialized_ligands": specialized,
        "manual_review_ligands": review,
        "enzyme_review_flags": enzyme_flags,
        "box_review_flags": box_flags,
        "receptor_state_review_flags": receptor_state_flags,
        "mechanism": mechanism,
        "unsupported_mechanism": unsupported_mechanism,
        "interpretation_status": (
            "exploratory_override"
            if (specialized or unsupported_mechanism or box_flags) and allow_specialized
            else "standard_or_reviewed"
        ),
    }


def write_conformer_summary(rows: list[dict], output_csv: Path) -> None:
    """Persist per-conformer results without collapsing them into one score."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["Ligand", "Conformer", "Binding_Affinity_kcal_mol", "Catalytically_Competent", "Selection_Status"]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def detect_catalytic_cys(pdb_path: Path, chain_id: str, patterns: list[str] | None = None) -> dict:
    """Detect a likely ALDH catalytic Cys using sequence motifs, with a clear fallback."""
    patterns = patterns or [r"[FY].{1,3}[GAQ].{0,2}C", r"[LIVMF].{2,5}C"]
    structure = PDBParser(QUIET=True).get_structure("receptor", str(pdb_path))
    chain = structure[0][chain_id]
    residues = [residue for residue in chain if residue.id[0] == " " and "CA" in residue]
    sequence = "".join(_one_letter(residue.get_resname()) for residue in residues)
    for pattern in patterns:
        match = re.search(pattern, sequence)
        if match:
            cys_index = match.start() + match.group().rfind("C")
            residue = residues[cys_index]
            return {"name": "CYS", "number": int(residue.id[1]), "chain": chain_id, "source": pattern}
    candidates = [residue for residue in residues if residue.get_resname().strip() == "CYS"]
    if not candidates:
        raise ValueError(f"No CYS residue found in chain {chain_id}")
    residue = min(candidates, key=lambda item: abs(int(item.id[1]) - 302))
    return {"name": "CYS", "number": int(residue.id[1]), "chain": chain_id, "source": "nearest_number_fallback"}


def _one_letter(name: str) -> str:
    return {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }.get(name.strip(), "X")


def assess_structure_quality(pdb_path: Path, output_json: Path, chain_id: str) -> dict:
    structure = PDBParser(QUIET=True).get_structure("receptor", str(pdb_path))
    atoms, b_factors = [], []
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            for residue in chain:
                if residue.id[0] == " ":
                    atoms.extend(residue.get_atoms())
                    b_factors.extend(float(atom.get_bfactor()) for atom in residue.get_atoms())
    result = {
        "pdb": str(pdb_path),
        "chain": chain_id,
        "atom_count": len(atoms),
        "mean_b_factor": round(mean(b_factors), 3) if b_factors else None,
        "missing_b_factor_fraction": round(sum(value == 0 for value in b_factors) / len(b_factors), 4) if b_factors else None,
        "interpretation": "B-factor proxy; for AlphaFold inputs, use pLDDT stored in B-factors.",
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def aggregate_replicates(rows: list[dict], output_csv: Path) -> list[dict]:
    if output_csv.is_file():
        with output_csv.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle)) + rows
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["Ligand"]), []).append(row)
    result = []
    for ligand, group in grouped.items():
        affinities = [float(row["Binding_Affinity_kcal_mol"]) for row in group if row.get("Binding_Affinity_kcal_mol") not in ("", None)]
        competent = [row for row in group if str(row.get("Catalytically_Competent")) == "True"]
        nuc_distances = [float(row["d_nuc_A"]) for row in group if row.get("d_nuc_A") not in ("", None)]
        hyd_distances = [float(row["d_hyd_A"]) for row in group if row.get("d_hyd_A") not in ("", None)]
        affinity_sd = float(np.std(affinities, ddof=1)) if len(affinities) > 1 else 0.0 if affinities else None
        nuc_sd = float(np.std(nuc_distances, ddof=1)) if len(nuc_distances) > 1 else 0.0 if nuc_distances else None
        hyd_sd = float(np.std(hyd_distances, ddof=1)) if len(hyd_distances) > 1 else 0.0 if hyd_distances else None
        competence_fraction = len(competent) / len(group)
        if len(group) < 2:
            stability = "single_replicate_not_assessed"
        elif (
            (affinity_sd is not None and affinity_sd > 1.0)
            or (nuc_sd is not None and nuc_sd > 0.75)
            or competence_fraction < 0.5
        ):
            stability = "unstable_manual_review_required"
        else:
            stability = "stable_by_configured_heuristic"
        result.append({
            "Ligand": ligand,
            "Replicate_Count": len(group),
            "Affinity_Mean_kcal_mol": round(mean(affinities), 4) if affinities else "",
            "Affinity_SD_kcal_mol": round(affinity_sd, 4) if affinity_sd is not None else "",
            "Competent_Replicate_Count": len(competent),
            "Competence_Fraction": round(competence_fraction, 4),
            "Stability_Status": stability,
            "d_nuc_SD_A": round(nuc_sd, 4) if nuc_sd is not None else "",
            "d_hyd_SD_A": round(hyd_sd, 4) if hyd_sd is not None else "",
            "Consensus_Score": round((-mean(affinities) if affinities else 0.0) + 2.0 * competence_fraction, 4),
            "d_nuc_Mean_A": round(mean(float(row["d_nuc_A"]) for row in group), 4),
            "d_hyd_Mean_A": round(mean(float(row["d_hyd_A"]) for row in group if row.get("d_hyd_A") not in ("", None)), 4) if any(row.get("d_hyd_A") not in ("", None) for row in group) else "",
        })
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result[0]) if result else ["Ligand"])
        writer.writeheader()
        writer.writerows(result)
    return result


def run_optional_plip(complex_pdb: Path, output_dir: Path, executable: str = "plip") -> Path | None:
    try:
        resolved = require_executable(executable)
    except FileNotFoundError:
        LOGGER.info("PLIP not installed; geometry interaction report remains authoritative.")
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([resolved, "-f", str(complex_pdb), "-o", str(output_dir), "--name", complex_pdb.stem], check=False, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"PLIP failed for {complex_pdb}: {result.stderr}")
    return output_dir
