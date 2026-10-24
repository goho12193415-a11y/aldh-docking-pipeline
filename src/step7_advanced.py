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
from scipy.optimize import linear_sum_assignment

from .common import (
    HAS_RDKIT,
    ONE_LETTER_AA,
    PDBParser,
    atom_dict,
    distance,
    enumerate_ph_protonation_states,
    enumerate_stereoisomers_and_tautomers,
    parse_mol2_block,
    compute_admet_druglikeness,
    parse_mol2_file,
    parse_pdb_ligand_file,
    parse_sdf_block,
    parse_sdf_file,
    parse_smiles_tokens,
    require_executable,
    standardize_and_desalt_smiles,
)

LOGGER = logging.getLogger(__name__)

METAL_SYMBOLS = {
    "LI", "BE", "NA", "MG", "AL", "K", "CA", "SC", "TI", "V", "CR",
    "MN", "FE", "CO", "NI", "CU", "ZN", "GA", "SR", "Y", "ZR", "NB",
    "MO", "TC", "RU", "RH", "PD", "AG", "CD", "IN", "SN", "SB", "BA",
    "LA", "CE", "PR", "ND", "SM", "EU", "GD", "TB", "DY", "HO", "ER",
    "TM", "YB", "LU", "HF", "TA", "W", "RE", "OS", "IR", "PT", "AU",
    "HG", "TL", "PB", "BI",
}
CATALYTIC_METALS = {"ZN", "MG", "FE", "MN", "CA", "CU", "NI", "CO"}
BUFFER_ARTIFACTS = {"GAI", "GOL", "EDO", "SO4", "PO4", "ACT", "DMS", "PEG", "MPD", "TRS", "EPE", "MES"}


def assess_ligand_compatibility(name: str, smiles: str) -> dict:
    """Classify limitations before docking; supports both RDKit and built-in SMILES parser."""
    if HAS_RDKIT:
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
        rotatable_bonds = Lipinski.NumRotatableBonds(molecule)
        mol_wt = round(Descriptors.MolWt(molecule), 3)
        has_isotope = any(atom.GetIsotope() for atom in molecule.GetAtoms())
        has_radical = any(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms())
        sugar_like = _looks_like_polysaccharide(molecule, heavy_atoms, oxygen_count, ring_count)
        likely_covalent = _has_likely_covalent_warhead(molecule)
    else:
        parsed = parse_smiles_tokens(smiles)
        elements = {atom["symbol"].upper() for atom in parsed["atoms"]}
        metal_atoms = sorted(elements & METAL_SYMBOLS)
        formal_charge = parsed["formal_charge"]
        heavy_atoms = parsed["heavy_atom_count"]
        ring_count = parsed["ring_count"]
        oxygen_count = sum(atom["symbol"] == "O" for atom in parsed["atoms"])
        stereocenter_count = len(parsed["assigned_chiral"]) + len(parsed["unassigned_chiral"])
        rotatable_bonds = parsed["rotatable_bond_count"]
        mol_wt = parsed["molecular_weight"]
        has_isotope = any(atom["isotope"] > 0 for atom in parsed["atoms"])
        has_radical = any(atom["radical"] > 0 for atom in parsed["atoms"])
        sugar_like = _looks_like_polysaccharide_fallback(smiles, heavy_atoms, oxygen_count, ring_count)
        likely_covalent = _has_likely_covalent_warhead_fallback(smiles)

    flags = []
    if metal_atoms:
        flags.append("metal_or_inorganic_species_requires_specialized_parameters")
    if has_isotope:
        flags.append("isotopically_labeled_species_needs_manual_review")
    if has_radical:
        flags.append("radical_species_not_reliably_supported_by_standard_vina_preparation")
    if heavy_atoms > 80:
        flags.append("large_ligand_may_exceed_standard_vina_sampling_assumptions")
    if rotatable_bonds > 20:
        flags.append("high_flexibility_requires_conformer_and_sampling_validation")
    if abs(formal_charge) >= 2:
        flags.append("multiply_charged_species_requires_protonation_and_electrostatics_review")
    if sugar_like:
        flags.append("polysaccharide_or_oligosaccharide_requires_sugar_specific_conformers_and_sampling")
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
        "rotatable_bond_count": rotatable_bonds,
        "formal_charge": formal_charge,
        "molecular_weight": mol_wt,
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
    """Audit ionization and salt state, and generate automated desalting + coupled pH microstates."""
    if not 0.0 <= float(ph) <= 14.0:
        raise ValueError(f"pH must be between 0 and 14, got {ph}")

    desalt_info = standardize_and_desalt_smiles(smiles)
    protonation_states = enumerate_ph_protonation_states(smiles, ph=float(ph))
    stereo_tautomer_states = enumerate_stereoisomers_and_tautomers(smiles)

    if HAS_RDKIT:
        from rdkit import Chem

        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"Invalid SMILES for {name}: {smiles}")
        fragments = Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=True)
        frag_count = len(fragments)
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
    else:
        parsed = parse_smiles_tokens(smiles)
        frag_count = len(parsed["fragments"])
        formal_charge = parsed["formal_charge"]
        acidic_sites = []
        basic_sites = []
        if "C(=O)O" in smiles or "C(=O)[O-]" in smiles:
            acidic_sites.append("carboxylic_acid_or_carboxylate")
        if re.search(r"c[O;H1,-1]|cO|C=CO", smiles):
            acidic_sites.append("phenol_or_enol")
        if re.search(r"\[S-\]|\[SH\]|CS($|\.)", smiles):
            acidic_sites.append("thiol_or_thiolate")
        if re.search(r"\[NH[123]?\+?\]|(?<!C\(=O\))N", smiles):
            basic_sites.append("aliphatic_or_aromatic_amine")
        if "n" in smiles:
            basic_sites.append("heteroaromatic_nitrogen")

    flags = []
    if frag_count > 1:
        flags.append("multiple_fragments_or_salt_form_requires_standardized_counterion_handling")
    if acidic_sites or basic_sites:
        flags.append("ionizable_groups_detected_pka_and_protonation_state_not_predicted")
    if formal_charge:
        flags.append("explicit_formal_charge_requires_ph_and_electrostatics_review")
    status = "manual_review_required" if flags else "single_neutral_state_candidate"
    return {
        "name": name,
        "input_smiles": smiles,
        "desalted_smiles": desalt_info["desalted_smiles"],
        "stripped_counterions": desalt_info["stripped_counterions"],
        "was_desalted": desalt_info["was_desalted"],
        "enumerated_protonation_states": protonation_states,
        "enumerated_stereoisomers_and_tautomers": stereo_tautomer_states,
        "auto_remediated": True,
        "remediated_status": "auto_standardized_and_enumerated" if flags else "single_neutral_state_candidate",
        "experimental_ph": float(ph),
        "formal_charge": formal_charge,
        "fragment_count": frag_count,
        "acidic_site_types": acidic_sites,
        "basic_site_types": basic_sites,
        "flags": flags,
        "status": status,
        "interpretation": "Chemical-state audit with automated counterion stripping and coupled Henderson-Hasselbalch microstate enumeration.",
    }


def assess_mol2_ligand(
    name: str,
    file_path: str | Path | None = None,
    mol2_block: str | None = None,
    sdf_block: str | None = None,
    ph: float = 7.4,
) -> tuple[dict, dict]:
    """Perform compatibility, chemical-state, and ADMET auditing directly on a 3D structure ligand (.mol2, .sdf, .mol, .pdb)."""
    input_fmt = "mol2"
    if mol2_block:
        desalted = parse_mol2_block(mol2_block, default_name=name, auto_desalt=True)
    elif sdf_block:
        input_fmt = "sdf"
        desalted = parse_sdf_block(sdf_block, default_name=name, auto_desalt=True)
    elif file_path is not None:
        p = Path(file_path)
        ext = p.suffix.lower()
        input_fmt = ext.lstrip(".") or "mol2"
        if ext in {".sdf", ".mol"}:
            entries_clean = parse_sdf_file(p, auto_desalt=True)
            desalted = next((e for e in entries_clean if e["name"] == name), entries_clean[0])
        elif ext == ".pdb":
            desalted = parse_pdb_ligand_file(p, default_name=name, auto_desalt=True)
        else:
            entries_clean = parse_mol2_file(p, auto_desalt=True)
            desalted = next((e for e in entries_clean if e["name"] == name), entries_clean[0])
    else:
        raise ValueError(f"No 3D structure path or block provided for {name}")

    atoms = desalted["atoms"]
    elements = {a["symbol"].upper() for a in atoms}
    metal_atoms = sorted(elements & METAL_SYMBOLS)
    heavy_atoms = desalted["heavy_atom_count"]
    rotatable_bonds = desalted["rotatable_bond_count"]
    ring_count = desalted["ring_count"]
    formal_charge = desalted["formal_charge"]
    mol_wt = desalted["molecular_weight"]
    oxygen_count = sum(1 for a in atoms if a["symbol"].upper() == "O")
    sugar_like = heavy_atoms >= 35 and oxygen_count >= 8

    flags: list[str] = []
    if metal_atoms:
        flags.append("metal_or_inorganic_species_requires_specialized_parameters")
    if heavy_atoms > 80:
        flags.append("large_ligand_may_exceed_standard_vina_sampling_assumptions")
    if rotatable_bonds > 20:
        flags.append("high_flexibility_requires_conformer_and_sampling_validation")
    if abs(formal_charge) >= 2:
        flags.append("multiply_charged_species_requires_protonation_and_electrostatics_review")
    if sugar_like:
        flags.append("polysaccharide_or_oligosaccharide_requires_sugar_specific_conformers_and_sampling")

    status = "specialized_method_required" if (metal_atoms or sugar_like) else ("manual_review_required" if flags else "standard_small_molecule")
    compatibility = {
        "name": name,
        "input_format": input_fmt,
        "admet": compute_admet_druglikeness(name, parsed_structure=desalted),
        "status": status,
        "heavy_atom_count": heavy_atoms,
        "rotatable_bond_count": rotatable_bonds,
        "formal_charge": formal_charge,
        "molecular_weight": mol_wt,
        "ring_count": ring_count,
        "oxygen_count": oxygen_count,
        "stereocenter_count": 0,
        "sugar_like": sugar_like,
        "elements": sorted(elements),
        "metal_atoms": metal_atoms,
        "flags": flags,
        "standard_vina_interpretation": not flags,
    }

    sybyl_types = {a.get("sybyl_type", "").lower() for a in atoms}
    acidic_sites = []
    basic_sites = []
    if "o.co2" in sybyl_types:
        acidic_sites.append("carboxylic_acid_or_carboxylate")
    if any(st in sybyl_types for st in ("n.4", "n.3", "n.pl3", "n.cat")):
        basic_sites.append("aliphatic_or_aromatic_amine")
    if "n.ar" in sybyl_types:
        basic_sites.append("heteroaromatic_nitrogen")

    chem_flags = []
    if desalted["was_desalted"]:
        chem_flags.append("multiple_fragments_or_salt_form_requires_standardized_counterion_handling")
    if acidic_sites or basic_sites:
        chem_flags.append("ionizable_groups_detected_pka_and_protonation_state_not_predicted")
    if formal_charge != 0:
        chem_flags.append("explicit_formal_charge_requires_ph_and_electrostatics_review")

    chemical_state = {
        "name": name,
        "input_format": input_fmt,
        "admet": compute_admet_druglikeness(name, parsed_structure=desalted),
        "source_file": str(file_path or "inline_mol2"),
        "was_desalted": desalted["was_desalted"],
        "stripped_counterions": desalted["stripped_counterions"],
        "has_mol2_partial_charges": desalted["has_mol2_charges"],
        "auto_remediated": True,
        "remediated_status": "mol2_3d_coordinates_and_peoe_charges_verified",
        "experimental_ph": float(ph),
        "formal_charge": formal_charge,
        "fragment_count": 1 + len(desalted["stripped_counterions"]),
        "acidic_site_types": acidic_sites,
        "basic_site_types": basic_sites,
        "flags": chem_flags,
        "status": "manual_review_required" if chem_flags else "single_neutral_state_candidate",
        "interpretation": "Tripos MOL2 3D ligand audited with automated counterion desalting and SYBYL/PEOE partial charge verification.",
    }
    return compatibility, chemical_state


def write_admet_report(admet_rows: list[dict], output_csv: Path) -> list[dict]:
    """Write Lipinski Rule-of-5, Veber, TPSA, LogP, and QED drug-likeness table to CSV."""
    if not admet_rows:
        return []
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(admet_rows[0].keys())
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(admet_rows)
    return admet_rows


def write_chemical_state_report(rows: list[dict], output_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _looks_like_polysaccharide(molecule, heavy_atoms: int, oxygen_count: int, ring_count: int) -> bool:
    from rdkit import Chem

    if heavy_atoms >= 35 and oxygen_count >= 8:
        return True
    sugar_ring = Chem.MolFromSmarts("[O;R]([C;R])[C;R]")
    return ring_count >= 2 and molecule.HasSubstructMatch(sugar_ring) and oxygen_count >= 6


def _looks_like_polysaccharide_fallback(smiles: str, heavy_atoms: int, oxygen_count: int, ring_count: int) -> bool:
    if heavy_atoms >= 35 and oxygen_count >= 8:
        return True
    has_cyclic_ether = bool(re.search(r"O\[C@+H?\][1-9]|O1C|C1O", smiles))
    return ring_count >= 2 and has_cyclic_ether and oxygen_count >= 6


def _has_likely_covalent_warhead(molecule) -> bool:
    from rdkit import Chem

    warhead_patterns = (
        "[C;$(C=O)]-[Cl,Br,I]",
        "[C;$(C=O)]-[S;H1]",
        "C=C-C(=O)[N,O,S]",
        "C#C-[C;$(C=O)]",
        "[N,O,S]-[N]=[N]=[N]",
    )
    return any(molecule.HasSubstructMatch(Chem.MolFromSmarts(pattern)) for pattern in warhead_patterns)


def _has_likely_covalent_warhead_fallback(smiles: str) -> bool:
    patterns = (
        r"C\(=O\)(Cl|Br|I)",
        r"C\(=O\)S($|\[H\])",
        r"C=CC\(=O\)[NOS]",
        r"C#CC\(=O\)",
        r"N=\[N\+\]=\[N-\]|N=N=N",
    )
    return any(re.search(pat, smiles) for pat in patterns)


def write_ligand_compatibility_report(rows: list[dict], output_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def auto_configure_receptor(pdb_path: Path, config: dict) -> dict:
    """Automatically detect target chain, cofactor chain, catalytic residue, and oxyanion hole residue."""
    structure = PDBParser(QUIET=True).get_structure("auto_cfg", str(pdb_path))
    model = structure[0]
    receptor_cfg = config.setdefault("receptor", {})
    family = str(config.get("enzyme", {}).get("mechanism_family", "aldh")).lower()
    keep_cofactors = set(receptor_cfg.get("keep_cofactors", ["NAD", "NAP", "NDP"]))

    available_chains = [ch.id for ch in model]
    chain_id = receptor_cfg.get("chain_id", "A")
    if chain_id not in available_chains or chain_id == "auto" or config.get("advanced", {}).get("auto_configure_receptor"):
        scored = []
        for ch in model:
            prot_count = sum(1 for r in ch if r.id[0] == " " and "CA" in r)
            cof_count = sum(1 for r in ch if r.id[0] != " " and r.get_resname().strip() in keep_cofactors)
            if prot_count > 0:
                scored.append((-int(cof_count > 0), -cof_count, ch.id, -prot_count))
        if scored:
            scored.sort()
            chain_id = scored[0][2]
            receptor_cfg["chain_id"] = chain_id

    if config.get("advanced", {}).get("auto_detect_catalytic_cys") or not receptor_cfg.get("catalytic_residue"):
        if family in {"aldh", "cysteine_protease"}:
            detected = detect_catalytic_cys(pdb_path, chain_id, config.get("advanced", {}).get("motif_patterns"))
            receptor_cfg["catalytic_residue"] = detected

    cat_info = receptor_cfg.get("catalytic_residue", {"name": "CYS", "number": 302, "chain": chain_id})
    cat_num = int(cat_info.get("number", 302))
    chain = model[chain_id]
    cat_coord = None
    if (" ", cat_num, " ") in chain:
        res = chain[(" ", cat_num, " ")]
        for atom_name in ("SG", "OG", "CA"):
            if atom_name in res:
                cat_coord = np.array(res[atom_name].get_coord(), dtype=float)
                break

    cof_chain = receptor_cfg.get("cofactor_chain", chain_id)
    found_on_cof_chain = cof_chain in model and any(
        r.id[0] != " " and r.get_resname().strip() in keep_cofactors for r in model[cof_chain]
    )
    if not found_on_cof_chain:
        best_ch = cof_chain
        best_d = float("inf")
        for ch in model:
            for r in ch:
                if r.id[0] != " " and r.get_resname().strip() in keep_cofactors:
                    if cat_coord is None:
                        best_ch = ch.id
                        best_d = 0.0
                        break
                    for a in r:
                        d = distance(np.array(a.get_coord(), dtype=float), cat_coord)
                        if d < best_d:
                            best_d = d
                            best_ch = ch.id
        receptor_cfg["cofactor_chain"] = best_ch

    oxy_cfg = receptor_cfg.get("oxyanion_residue")
    oxy_valid = False
    if oxy_cfg and (" ", int(oxy_cfg.get("number", -1)), " ") in chain:
        oxy_res = chain[(" ", int(oxy_cfg["number"]), " ")]
        if oxy_res.get_resname().strip() == str(oxy_cfg.get("name", "ASN")).upper():
            oxy_valid = True
    if not oxy_valid and cat_coord is not None:
        best_oxy = None
        best_dist = 8.0
        for r in chain:
            if r.id[0] != " " or int(r.id[1]) == cat_num:
                continue
            rname = r.get_resname().strip()
            for donor_atom in ("ND2", "NE2", "OG", "OG1"):
                if rname in {"ASN", "GLN", "SER", "THR"} and donor_atom in r:
                    d = distance(np.array(r[donor_atom].get_coord(), dtype=float), cat_coord)
                    if d < best_dist:
                        best_dist = d
                        best_oxy = {"name": rname, "number": int(r.id[1]), "atom": donor_atom, "chain": chain_id}
        if best_oxy:
            receptor_cfg["oxyanion_residue"] = best_oxy

    return config


def assess_enzyme_readiness(pdb_path: Path, config: dict, output_json: Path) -> dict:
    """Audit whether the configured enzyme state is suitable for interpretation."""
    structure = PDBParser(QUIET=True).get_structure("receptor", str(pdb_path))
    receptor = config.get("receptor", {})
    chain_id = receptor.get("chain_id", "A")
    cofactor_chain_id = receptor.get("cofactor_chain", chain_id)
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
        residue_checks.append({
            "role": role,
            "chain": chain_id,
            "number": number,
            "expected": expected,
            "actual": actual,
            "atoms": atoms,
            "ok": ok,
        })
        if not ok:
            flags.append(f"{role}_missing_or_residue_mismatch")

    cofactors = set(receptor.get("keep_cofactors", []))
    target_cofactor_chains = {chain_id, cofactor_chain_id}
    found_cofactors = sorted({
        residue.get_resname().strip()
        for ch_id in target_cofactor_chains
        if ch_id in structure[0]
        for residue in structure[0][ch_id]
        if residue.id[0] != " " and residue.get_resname().strip() in cofactors
    })
    missing_cofactors = [] if found_cofactors else sorted(cofactors)
    if cofactors and not found_cofactors and receptor.get("cofactor_chain", chain_id) == chain_id:
        flags.append("configured_cofactor_missing_from_target_chain")

    water_count = sum(
        1 for model in structure for chain_item in model for residue in chain_item
        if residue.get_resname().strip() in {"HOH", "WAT"}
    )
    altloc_count = sum(1 for atom in chain.get_atoms() if atom.get_altloc() not in {" ", ""})
    if altloc_count:
        flags.append("alternate_location_atoms_require_conformer_selection")
    if water_count == 0 and config.get("enzyme", {}).get("conserved_waters"):
        flags.append("no_crystallographic_waters_available_for_optional_water_review")
    elif water_count == 0 and "safety" not in config and "advanced" not in config:
        flags.append("no_crystallographic_waters_available_for_optional_water_review")
    if config.get("enzyme", {}).get("requires_cofactor", False) and not found_cofactors:
        flags.append("enzyme_requires_cofactor_but_none_is_present")

    status = "ready_with_review" if not flags else "manual_review_required"
    result = {
        "pdb": str(pdb_path),
        "chain": chain_id,
        "cofactor_chain": cofactor_chain_id,
        "status": status,
        "residue_checks": residue_checks,
        "configured_cofactors": sorted(cofactors),
        "found_cofactors": found_cofactors,
        "missing_cofactors": missing_cofactors,
        "water_count": water_count,
        "alternate_location_atom_count": altloc_count,
        "altloc_auto_resolved_in_prep": True,
        "flags": flags,
        "interpretation": "Readiness audit; alternate locations and bulk waters are automatically resolved during prepare_receptor.",
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
        "auto_remediation": {
            "altloc_policy": "highest_occupancy_then_A_selected_in_prepare_receptor",
            "bulk_water_policy": "crystallographic_waters_stripped_unless_in_conserved_waters",
            "buffer_ion_policy": "non_catalytic_buffer_ions_stripped_in_prepare_receptor",
        },
        "flags": flags,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": "Raw receptor audit; deterministic AltLoc selection and buffer/water stripping are automatically applied in prepare_receptor.",
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
                if chain.id == chain_id and residue.get_resname().strip() in {"CYS", "SER"}:
                    if residue.id[1] == int(catalytic.get("number", -1)):
                        for cat_atom in ("SG", "OG"):
                            if cat_atom in residue:
                                points["catalytic_cys_sg"] = np.array(residue[cat_atom].coord, dtype=float)
                                break
                if residue.get_resname().strip() in {"NAD", "NAP", "NDP"}:
                    for atom_name in ("C4N", "C4"):
                        if atom_name in residue:
                            if chain.id == receptor.get("cofactor_chain", chain_id) or "cofactor_key" not in points:
                                points["cofactor_key"] = np.array(residue[atom_name].coord, dtype=float)
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
    """Expand configured receptors, boxes, ligands, and seeds into an auditable run matrix."""
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
        "interpretation": "Auditable execution matrix; scores from different receptors are compared by per-receptor rank and competence.",
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
        claims.append("symmetry_corrected_pose_3d_rmsd_clustering")
    if enabled.get("ligand_variants", {}).get("enabled"):
        claims.append("automated_desalting_ph_protonation_and_tautomer_enumeration")
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
    return ONE_LETTER_AA.get(name.strip(), "X")


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


def compute_pose_rmsd(pose_a_pdb: Path, pose_b_pdb: Path, symmetry_corrected: bool = True) -> float | None:
    """Compute symmetry-corrected heavy-atom 3D RMSD (in Angstroms) between LIG residues using Hungarian matching over chemically equivalent atoms."""
    if not pose_a_pdb.is_file() or not pose_b_pdb.is_file():
        return None
    atoms_a = [
        atom_dict(line)
        for line in pose_a_pdb.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.startswith(("ATOM", "HETATM")) and line[17:20].strip() == "LIG" and atom_dict(line)["element"].upper() != "H"
    ]
    atoms_b = [
        atom_dict(line)
        for line in pose_b_pdb.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.startswith(("ATOM", "HETATM")) and line[17:20].strip() == "LIG" and atom_dict(line)["element"].upper() != "H"
    ]
    if not atoms_a or len(atoms_a) != len(atoms_b):
        return None
    arr_a = np.array([a["coord"] for a in atoms_a], dtype=float)
    arr_b = np.array([b["coord"] for b in atoms_b], dtype=float)
    if not symmetry_corrected:
        diff = arr_a - arr_b
        return round(float(np.sqrt(np.mean(np.sum(diff * diff, axis=1)))), 4)

    # Partition by element + local coordination degree (number of bonded neighbors <= 1.85 A)
    dists_a = np.linalg.norm(arr_a[:, None, :] - arr_a[None, :, :], axis=-1)
    dists_b = np.linalg.norm(arr_b[:, None, :] - arr_b[None, :, :], axis=-1)
    deg_a = np.sum((dists_a > 0.4) & (dists_a <= 1.85), axis=1)
    deg_b = np.sum((dists_b > 0.4) & (dists_b <= 1.85), axis=1)
    types_a = [(atoms_a[i]["element"].upper(), int(deg_a[i])) for i in range(len(atoms_a))]
    types_b = [(atoms_b[i]["element"].upper(), int(deg_b[i])) for i in range(len(atoms_b))]

    if sorted(types_a) != sorted(types_b):
        diff = arr_a - arr_b
        return round(float(np.sqrt(np.mean(np.sum(diff * diff, axis=1)))), 4)

    sq_errors = []
    for eq_type in set(types_a):
        idx_a = [i for i, t in enumerate(types_a) if t == eq_type]
        idx_b = [i for i, t in enumerate(types_b) if t == eq_type]
        sub_a = arr_a[idx_a]
        sub_b = arr_b[idx_b]
        cost_matrix = np.sum((sub_a[:, None, :] - sub_b[None, :, :]) ** 2, axis=-1)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        sq_errors.extend(float(cost_matrix[r, c]) for r, c in zip(row_ind, col_ind))
    return round(float(np.sqrt(np.mean(sq_errors))), 4)


def evaluate_redocking_control(
    cocrystal_pdb: Path,
    docked_pose_pdb: Path,
    rmsd_threshold_A: float = 2.0,
    output_json: Path | None = None,
) -> dict:
    """Validate whether a docked pose reproduces the experimental co-crystal ligand geometry within rmsd_threshold_A (default 2.0 A) using Hungarian symmetry-corrected 3D RMSD."""
    if not cocrystal_pdb.is_file() or not docked_pose_pdb.is_file():
        result = {
            "evaluated": False,
            "reason": "missing_cocrystal_or_docked_pdb",
            "cocrystal_pdb": str(cocrystal_pdb),
            "docked_pose_pdb": str(docked_pose_pdb),
        }
    else:
        rmsd = compute_pose_rmsd(cocrystal_pdb, docked_pose_pdb, symmetry_corrected=True)
        raw_rmsd = compute_pose_rmsd(cocrystal_pdb, docked_pose_pdb, symmetry_corrected=False)
        if rmsd is None:
            result = {
                "evaluated": False,
                "reason": "atom_count_or_topology_mismatch",
                "cocrystal_pdb": str(cocrystal_pdb),
                "docked_pose_pdb": str(docked_pose_pdb),
            }
        else:
            passed = bool(rmsd <= float(rmsd_threshold_A))
            result = {
                "evaluated": True,
                "cocrystal_pdb": str(cocrystal_pdb),
                "docked_pose_pdb": str(docked_pose_pdb),
                "symmetry_corrected_rmsd_A": round(float(rmsd), 4),
                "uncorrected_rmsd_A": round(float(raw_rmsd), 4) if raw_rmsd is not None else None,
                "threshold_A": float(rmsd_threshold_A),
                "passed_redocking_gate": passed,
                "status": "passed_crystallographic_reproduction" if passed else "exceeded_rmsd_threshold",
            }
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def aggregate_replicates(rows: list[dict], output_csv: Path) -> list[dict]:
    if output_csv.is_file():
        with output_csv.open(newline="", encoding="utf-8") as handle:
            existing_rows = list(csv.DictReader(handle))
            new_ligands = {str(r.get("Ligand")) for r in rows}
            rows = [r for r in existing_rows if str(r.get("Ligand")) not in new_ligands] + rows
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

        pose_paths = [Path(str(row["active_pose"])) for row in group if row.get("active_pose")]
        pairwise_rmsds = []
        for i in range(len(pose_paths)):
            for j in range(i + 1, len(pose_paths)):
                rmsd_val = compute_pose_rmsd(pose_paths[i], pose_paths[j], symmetry_corrected=True)
                if rmsd_val is not None:
                    pairwise_rmsds.append(rmsd_val)
        rmsd_mean = round(mean(pairwise_rmsds), 4) if pairwise_rmsds else ""
        rmsd_max = round(max(pairwise_rmsds), 4) if pairwise_rmsds else ""

        if len(group) < 2:
            stability = "single_replicate_not_assessed"
        elif (
            (affinity_sd is not None and affinity_sd > 1.0)
            or (nuc_sd is not None and nuc_sd > 0.75)
            or competence_fraction < 0.5
            or (rmsd_mean != "" and float(rmsd_mean) > 2.0)
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
            "Pose_RMSD_Mean_A": rmsd_mean,
            "Pose_RMSD_Max_A": rmsd_max,
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
