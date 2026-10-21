from __future__ import annotations

import logging
import json
from pathlib import Path

import requests

LOGGER = logging.getLogger(__name__)


def enumerate_ligand_variants(smiles: str, max_variants: int = 8) -> list[tuple[str, str]]:
    """Return unique neutral/tautomer variants when RDKit supports them."""
    from rdkit import Chem
    variants = [("parent", smiles)]
    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            enumerator = rdMolStandardize.TautomerEnumerator()
            for index, tautomer in enumerate(enumerator.Enumerate(mol)):
                candidate = Chem.MolToSmiles(tautomer, canonical=True)
                if candidate not in {value for _, value in variants}:
                    variants.append((f"tautomer_{index + 1}", candidate))
                    if len(variants) >= max_variants:
                        break
    except (ImportError, AttributeError):
        LOGGER.info("RDKit tautomer enumeration is unavailable; using parent ligand only.")
    return variants


def fetch_smiles(name: str) -> str:
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{requests.utils.quote(name)}/property/CanonicalSMILES/JSON"
    )
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    properties = response.json()["PropertyTable"]["Properties"]
    if not properties or not properties[0].get("ConnectivitySMILES"):
        raise ValueError(f"PubChem returned no SMILES for {name}")
    return properties[0]["ConnectivitySMILES"]


def prepare_ligand(name: str, smiles: str | None, output_dir: Path) -> Path:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise RuntimeError("RDKit is required for ligand preparation.") from exc

    resolved_smiles = smiles or fetch_smiles(name)
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

    output_dir.mkdir(parents=True, exist_ok=True)
    sdf_path = output_dir / f"{name}.sdf"
    writer = Chem.SDWriter(str(sdf_path))
    writer.write(mol)
    writer.close()

    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
    except ImportError as exc:
        raise RuntimeError("Meeko is required for ligand PDBQT conversion.") from exc
    # Older RDKit builds used on Windows may lack the method expected by
    # recent Meeko releases; ordinary SMILES-derived molecules have no query
    # atoms, so the compatibility method is safe for this preparation path.
    if not hasattr(Chem.Mol, "HasQuery"):
        setattr(Chem.Mol, "HasQuery", lambda self: False)
    preparation = MoleculePreparation()
    setup = preparation.prepare(mol)
    if not setup:
        raise RuntimeError(f"Meeko produced no setup for {name}")
    pdbqt_string, *_ = PDBQTWriterLegacy.write_string(setup[0])
    pdbqt_path = output_dir / f"{name}.pdbqt"
    pdbqt_path.write_text(pdbqt_string, encoding="utf-8")
    return pdbqt_path


def prepare_ligand_variants(name: str, smiles: str, output_dir: Path, max_variants: int = 8) -> list[Path]:
    paths = []
    for label, variant_smiles in enumerate_ligand_variants(smiles, max_variants):
        variant_name = f"{name}__{label}"
        paths.append(prepare_ligand(variant_name, variant_smiles, output_dir))
    return paths


def prepare_polysaccharide_conformers(
    name: str,
    smiles: str,
    output_dir: Path,
    num_conformers: int = 24,
    rmsd_threshold: float = 0.5,
) -> tuple[list[Path], Path]:
    """Generate an auditable conformer ensemble for sugar-rich ligands.

    This preserves the input stereochemistry and samples 3-D conformers; it
    does not model glycosidic kinetics or replace carbohydrate MD.
    """
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
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    pdbqt_paths = []
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
    manifest_path = output_dir / f"{name}__sugar_conformers.json"
    manifest_path.write_text(json.dumps({
        "ligand": name,
        "input_smiles": smiles,
        "conformer_count": len(manifest),
        "sampling": "RDKit ETKDGv3 + small-ring torsions + MMFF when parameterized",
        "limitations": [
            "Not a glycosidic free-energy ensemble",
            "Validate with carbohydrate force fields or MD before quantitative claims",
        ],
        "conformers": manifest,
    }, indent=2), encoding="utf-8")
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
