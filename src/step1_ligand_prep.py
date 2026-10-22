from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import requests

from .common import (
    HAS_MEEKO,
    HAS_RDKIT,
    enumerate_ph_protonation_states,
    generate_fallback_3d_coords,
    parse_smiles_tokens,
    standardize_and_desalt_smiles,
    write_fallback_pdbqt,
    write_fallback_sdf,
)

LOGGER = logging.getLogger(__name__)


def enumerate_ligand_variants(
    smiles: str,
    max_variants: int = 8,
    ph: float = 7.4,
) -> list[tuple[str, str]]:
    """Return unique desalted, pH-protonated, and tautomer variants."""
    desalt_info = standardize_and_desalt_smiles(smiles)
    base_smiles = desalt_info["desalted_smiles"]
    variants: list[tuple[str, str]] = [("parent", base_smiles)]
    seen = {base_smiles}

    for state in enumerate_ph_protonation_states(base_smiles, ph=ph, max_states=max_variants):
        candidate = state["smiles"]
        if candidate not in seen and len(variants) < max_variants:
            seen.add(candidate)
            variants.append((state["label"], candidate))

    if HAS_RDKIT:
        try:
            from rdkit import Chem
            from rdkit.Chem.MolStandardize import rdMolStandardize

            mol = Chem.MolFromSmiles(base_smiles)
            if mol is not None:
                enumerator = rdMolStandardize.TautomerEnumerator()
                for index, tautomer in enumerate(enumerator.Enumerate(mol)):
                    candidate = Chem.MolToSmiles(tautomer, canonical=True)
                    if candidate not in seen:
                        seen.add(candidate)
                        variants.append((f"tautomer_{index + 1}", candidate))
                        if len(variants) >= max_variants:
                            break
        except (ImportError, AttributeError):
            LOGGER.info("RDKit tautomer enumeration is unavailable; using pH microstates and parent ligand.")
    return variants[:max_variants]


def discover_raw_ligands(raw_ligands_dir: Path, configured_ligands: list[dict] | None = None) -> list[dict]:
    """Automatically merge configured ligands with any .smi/.csv/.txt files in data/02_raw_ligands/."""
    merged: list[dict] = []
    seen_names: set[str] = set()
    for item in configured_ligands or []:
        name = str(item.get("name", "")).strip()
        if name and name not in seen_names:
            seen_names.add(name)
            merged.append(dict(item))
    if not raw_ligands_dir.is_dir():
        return merged

    for path in sorted(raw_ligands_dir.iterdir()):
        if path.name.startswith(".") or path.suffix.lower() not in {".smi", ".csv", ".txt"}:
            continue
        if path.suffix.lower() == ".csv":
            with path.open(newline="", encoding="utf-8", errors="replace") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    name = (row.get("name") or row.get("Name") or row.get("Ligand") or "").strip()
                    smiles = (row.get("smiles") or row.get("SMILES") or "").strip()
                    if name and smiles and name not in seen_names:
                        seen_names.add(name)
                        merged.append({"name": name, "smiles": smiles, "source": str(path)})
        else:
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                text = line.strip()
                if not text or text.startswith("#"):
                    continue
                if "=" in text:
                    name, smiles = [p.strip() for p in text.split("=", 1)]
                else:
                    parts = text.split()
                    if len(parts) >= 2:
                        smiles, name = parts[0].strip(), parts[1].strip()
                    else:
                        continue
                if name and smiles and name not in seen_names:
                    seen_names.add(name)
                    merged.append({"name": name, "smiles": smiles, "source": str(path)})
    return merged


def fetch_smiles(name: str) -> str:
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{requests.utils.quote(name)}/property/CanonicalSMILES/JSON"
    )
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    properties = response.json()["PropertyTable"]["Properties"]
    if not properties:
        raise ValueError(f"PubChem returned no SMILES for {name}")
    smiles = properties[0].get("ConnectivitySMILES") or properties[0].get("CanonicalSMILES")
    if not smiles:
        raise ValueError(f"PubChem returned no SMILES for {name}")
    return smiles


def prepare_ligand(
    name: str,
    smiles: str | None,
    output_dir: Path,
    auto_desalt: bool = True,
) -> Path:
    resolved_smiles = smiles or fetch_smiles(name)
    if auto_desalt:
        desalt_info = standardize_and_desalt_smiles(resolved_smiles)
        if desalt_info["was_desalted"]:
            LOGGER.info(
                "Auto-desalted %s: kept parent '%s', stripped counterions %s",
                name,
                desalt_info["desalted_smiles"],
                desalt_info["stripped_counterions"],
            )
            resolved_smiles = desalt_info["desalted_smiles"]

    output_dir.mkdir(parents=True, exist_ok=True)
    sdf_path = output_dir / f"{name}.sdf"
    pdbqt_path = output_dir / f"{name}.pdbqt"

    if HAS_RDKIT and HAS_MEEKO:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(resolved_smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES for {name}: {resolved_smiles}")
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xA1D0
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise RuntimeError(f"Could not generate a 3D conformer for {name}")
        if AllChem.MMFFHasAllMoleculeParams(mol):
            result = AllChem.MMFFOptimizeMolecule(mol)
            LOGGER.info("%s MMFF optimization status=%s", name, result)

        writer = Chem.SDWriter(str(sdf_path))
        writer.write(mol)
        writer.close()
        return _write_pdbqt(name, mol, output_dir)

    parsed = parse_smiles_tokens(resolved_smiles)
    atom_records = generate_fallback_3d_coords(parsed, seed=0xA1D0, conformer_index=0)
    write_fallback_sdf(name, atom_records, parsed["bonds"], sdf_path)
    return write_fallback_pdbqt(name, atom_records, pdbqt_path)


def prepare_ligand_variants(
    name: str,
    smiles: str,
    output_dir: Path,
    max_variants: int = 8,
    ph: float = 7.4,
) -> list[Path]:
    paths = []
    manifest = []
    for label, variant_smiles in enumerate_ligand_variants(smiles, max_variants=max_variants, ph=ph):
        variant_name = f"{name}__{label}"
        pdbqt_path = prepare_ligand(variant_name, variant_smiles, output_dir, auto_desalt=True)
        paths.append(pdbqt_path)
        manifest.append({
            "parent_ligand": name,
            "variant_name": variant_name,
            "label": label,
            "smiles": variant_smiles,
            "pdbqt": str(pdbqt_path),
            "ph": float(ph),
        })
    if manifest:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{name}__variants_manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
    return paths


def prepare_polysaccharide_conformers(
    name: str,
    smiles: str,
    output_dir: Path,
    num_conformers: int = 24,
    rmsd_threshold: float = 0.5,
) -> tuple[list[Path], Path]:
    """Generate an auditable conformer ensemble for sugar-rich ligands."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    pdbqt_paths = []

    if HAS_RDKIT and HAS_MEEKO:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES for {name}: {smiles}")
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        unassigned = [idx for idx, code in Chem.FindMolChiralCenters(mol, includeUnassigned=True) if code == "?"]
        if unassigned:
            raise ValueError(f"{name} has unassigned stereocenters at atom indices {unassigned}")
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xA1D0
        params.pruneRmsThresh = rmsd_threshold
        params.useSmallRingTorsions = True
        conformer_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=params))
        if not conformer_ids:
            raise RuntimeError(f"Could not generate sugar conformers for {name}")
        for ordinal, conformer_id in enumerate(conformer_ids, 1):
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol, confId=conformer_id, maxIters=500)
            conformer_mol = Chem.Mol(mol)
            conformer_mol.RemoveAllConformers()
            conformer_mol.AddConformer(mol.GetConformer(conformer_id), assignId=True)
            conformer_name = f"{name}__sugarconf{ordinal:03d}"
            sdf_path = output_dir / f"{conformer_name}.sdf"
            writer = Chem.SDWriter(str(sdf_path))
            writer.write(conformer_mol)
            writer.close()
            pdbqt_path = _write_pdbqt(conformer_name, conformer_mol, output_dir)
            pdbqt_paths.append(pdbqt_path)
            manifest.append({
                "name": conformer_name,
                "conformer_id": int(conformer_id),
                "sdf": str(sdf_path),
                "pdbqt": str(pdbqt_path),
                "stereochemistry_preserved": True,
            })
    else:
        parsed = parse_smiles_tokens(smiles)
        if parsed["unassigned_chiral"]:
            raise ValueError(f"{name} has unassigned stereocenters at atom indices {parsed['unassigned_chiral']}")
        for ordinal in range(1, max(1, int(num_conformers)) + 1):
            conformer_name = f"{name}__sugarconf{ordinal:03d}"
            sdf_path = output_dir / f"{conformer_name}.sdf"
            pdbqt_path = output_dir / f"{conformer_name}.pdbqt"
            atom_records = generate_fallback_3d_coords(parsed, seed=0xA1D0, conformer_index=ordinal)
            write_fallback_sdf(conformer_name, atom_records, parsed["bonds"], sdf_path)
            write_fallback_pdbqt(conformer_name, atom_records, pdbqt_path)
            pdbqt_paths.append(pdbqt_path)
            manifest.append({
                "name": conformer_name,
                "conformer_id": ordinal - 1,
                "sdf": str(sdf_path),
                "pdbqt": str(pdbqt_path),
                "stereochemistry_preserved": True,
            })

    manifest_path = output_dir / f"{name}__sugar_conformers.json"
    manifest_path.write_text(
        json.dumps(
            {
                "ligand": name,
                "input_smiles": smiles,
                "conformer_count": len(manifest),
                "sampling": "RDKit ETKDGv3 + small-ring torsions + MMFF when parameterized",
                "limitations": [
                    "Not a glycosidic free-energy ensemble",
                    "Validate with carbohydrate force fields or MD before quantitative claims",
                ],
                "conformers": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return pdbqt_paths, manifest_path


def _write_pdbqt(name: str, mol, output_dir: Path) -> Path:
    from rdkit import Chem

    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
    except ImportError as exc:
        raise RuntimeError("Meeko is required for ligand PDBQT conversion.") from exc
    if not hasattr(Chem.Mol, "HasQuery"):
        setattr(Chem.Mol, "HasQuery", lambda self: False)
    setup = MoleculePreparation().prepare(mol)
    if not setup:
        raise RuntimeError(f"Meeko produced no setup for {name}")
    pdbqt_string, *_ = PDBQTWriterLegacy.write_string(setup[0])
    path = output_dir / f"{name}.pdbqt"
    path.write_text(pdbqt_string, encoding="utf-8")
    return path
