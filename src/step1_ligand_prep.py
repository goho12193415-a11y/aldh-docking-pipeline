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
    enumerate_stereoisomers_and_tautomers,
    generate_fallback_3d_coords,
    parse_mol2_block,
    parse_mol2_file,
    parse_pdb_ligand_file,
    parse_sdf_block,
    parse_sdf_file,
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

    for iso in enumerate_stereoisomers_and_tautomers(base_smiles, max_isomers=max_variants):
        candidate = iso["smiles"]
        if candidate not in seen and len(variants) < max_variants:
            seen.add(candidate)
            variants.append((iso["label"], candidate))

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
    """Automatically merge configured ligands with any .mol2/.sdf/.mol/.pdb/.pdbqt/.smi/.csv/.txt files in data/02_raw_ligands/."""
    merged: list[dict] = []
    seen_names: set[str] = set()
    for item in configured_ligands or []:
        entry = dict(item)
        # Support file/mol2 path passed either under "mol2"/"file" key or as a path in "smiles"
        file_ref = entry.get("mol2") or entry.get("file") or ""
        smiles_ref = str(entry.get("smiles") or "").strip()
        if not file_ref and smiles_ref.lower().endswith((".mol2", ".sdf", ".mol", ".pdb", ".pdbqt")):
            file_ref = smiles_ref
            entry.pop("smiles", None)
        if file_ref:
            entry["file"] = str(file_ref)
        name = str(entry.get("name") or (Path(str(file_ref)).stem if file_ref else "")).strip()
        if name and name not in seen_names:
            entry["name"] = name
            seen_names.add(name)
            merged.append(entry)
    if not raw_ligands_dir.is_dir():
        return merged

    for path in sorted(raw_ligands_dir.iterdir()):
        if path.name.startswith("."):
            continue
        suffix = path.suffix.lower()
        if suffix == ".mol2":
            try:
                mol2_entries = parse_mol2_file(path, auto_desalt=True)
                for m_entry in mol2_entries:
                    m_name = m_entry["name"]
                    if m_name not in seen_names:
                        seen_names.add(m_name)
                        merged.append({
                            "name": m_name,
                            "file": str(path),
                            "mol2_block": m_entry["mol2_block"],
                            "format": "mol2",
                            "source": str(path),
                        })
            except Exception as exc:
                LOGGER.warning("Skipping invalid .mol2 file %s: %s", path, exc)
        elif suffix in {".sdf", ".mol"}:
            try:
                sdf_entries = parse_sdf_file(path, auto_desalt=True)
                for s_entry in sdf_entries:
                    s_name = s_entry["name"]
                    if s_name not in seen_names:
                        seen_names.add(s_name)
                        merged.append({
                            "name": s_name,
                            "file": str(path),
                            "sdf_block": s_entry["sdf_block"],
                            "format": suffix.lstrip("."),
                            "source": str(path),
                        })
            except Exception as exc:
                LOGGER.warning("Skipping invalid %s file %s: %s", suffix, path, exc)
        elif suffix in {".pdb", ".pdbqt"}:
            name = path.stem
            if name not in seen_names:
                seen_names.add(name)
                merged.append({"name": name, "file": str(path), "format": suffix.lstrip("."), "source": str(path)})
        elif suffix == ".csv":
            with path.open(newline="", encoding="utf-8", errors="replace") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    name = (row.get("name") or row.get("Name") or row.get("Ligand") or "").strip()
                    smiles = (row.get("smiles") or row.get("SMILES") or "").strip()
                    mol2_col = (row.get("mol2") or row.get("MOL2") or row.get("file") or row.get("File") or "").strip()
                    if mol2_col and not name:
                        name = Path(mol2_col).stem
                    if name and name not in seen_names:
                        if mol2_col:
                            seen_names.add(name)
                            merged.append({"name": name, "file": mol2_col, "source": str(path)})
                        elif smiles:
                            seen_names.add(name)
                            merged.append({"name": name, "smiles": smiles, "source": str(path)})
        elif suffix in {".smi", ".txt"}:
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                text = line.strip()
                if not text or text.startswith("#"):
                    continue
                if "=" in text:
                    name, val = [p.strip() for p in text.split("=", 1)]
                else:
                    parts = text.split()
                    if len(parts) >= 2:
                        val, name = parts[0].strip(), parts[1].strip()
                    elif len(parts) == 1 and parts[0].lower().endswith((".mol2", ".sdf", ".mol", ".pdb", ".pdbqt")):
                        val = parts[0].strip()
                        name = Path(val).stem
                    else:
                        continue
                if name and val and name not in seen_names:
                    seen_names.add(name)
                    if val.lower().endswith((".mol2", ".sdf", ".mol", ".pdb", ".pdbqt")):
                        merged.append({"name": name, "file": val, "source": str(path)})
                    else:
                        merged.append({"name": name, "smiles": val, "source": str(path)})
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


def prepare_ligand_from_mol2(
    name: str,
    mol2_path_or_block: Path | str,
    output_dir: Path,
    auto_desalt: bool = True,
) -> Path:
    """Prepare a 3D ligand directly from a Tripos .mol2 file or @<TRIPOS>MOLECULE text block, preserving 3D coordinates and compiling a flexible BRANCH PDBQT."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sdf_path = output_dir / f"{name}.sdf"
    pdbqt_path = output_dir / f"{name}.pdbqt"

    if isinstance(mol2_path_or_block, Path) or (
        isinstance(mol2_path_or_block, str) and "@<TRIPOS>" not in mol2_path_or_block and Path(mol2_path_or_block).is_file()
    ):
        m_path = Path(mol2_path_or_block)
        entries = parse_mol2_file(m_path, auto_desalt=auto_desalt)
        matched = next((e for e in entries if e["name"] == name), entries[0])
    else:
        matched = parse_mol2_block(str(mol2_path_or_block), default_name=name, auto_desalt=auto_desalt)

    if matched.get("was_desalted"):
        LOGGER.info("Auto-desalted MOL2 ligand %s: stripped counterions %s", name, matched.get("stripped_counterions"))

    write_fallback_sdf(name, matched["atoms"], matched["bonds"], sdf_path)
    return write_fallback_pdbqt(name, matched["atoms"], pdbqt_path, bonds=matched["bonds"])


def prepare_ligand_from_sdf_or_pdb(
    name: str,
    file_or_block: Path | str,
    output_dir: Path,
    fmt: str = "sdf",
    auto_desalt: bool = True,
) -> Path:
    """Prepare a 3D ligand directly from an .sdf/.mol file/block or .pdb file, preserving 3D coordinates and compiling a flexible BRANCH PDBQT."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sdf_path = output_dir / f"{name}.sdf"
    pdbqt_path = output_dir / f"{name}.pdbqt"
    if fmt == "pdb":
        matched = parse_pdb_ligand_file(Path(file_or_block), default_name=name, auto_desalt=auto_desalt)
    elif isinstance(file_or_block, Path) or (isinstance(file_or_block, str) and "V2000" not in file_or_block and Path(file_or_block).is_file()):
        entries = parse_sdf_file(Path(file_or_block), auto_desalt=auto_desalt)
        matched = next((e for e in entries if e["name"] == name), entries[0])
    else:
        matched = parse_sdf_block(str(file_or_block), default_name=name, auto_desalt=auto_desalt)
    write_fallback_sdf(name, matched["atoms"], matched["bonds"], sdf_path)
    return write_fallback_pdbqt(name, matched["atoms"], pdbqt_path, bonds=matched["bonds"])


def prepare_ligand(
    name: str,
    smiles: str | None = None,
    output_dir: Path = Path("data/04_prepped_ligands"),
    auto_desalt: bool = True,
    file_path: str | Path | None = None,
    mol2_block: str | None = None,
    sdf_block: str | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if mol2_block:
        return prepare_ligand_from_mol2(name, mol2_block, output_dir, auto_desalt=auto_desalt)
    if sdf_block:
        return prepare_ligand_from_sdf_or_pdb(name, sdf_block, output_dir, fmt="sdf", auto_desalt=auto_desalt)

    candidate_file = file_path or (
        smiles if (smiles and str(smiles).strip().lower().endswith((".mol2", ".sdf", ".mol", ".pdb", ".pdbqt"))) else None
    )
    if candidate_file is not None:
        c_path = Path(candidate_file)
        if c_path.is_file():
            ext = c_path.suffix.lower()
            if ext == ".mol2":
                return prepare_ligand_from_mol2(name, c_path, output_dir, auto_desalt=auto_desalt)
            if ext in {".sdf", ".mol"}:
                return prepare_ligand_from_sdf_or_pdb(name, c_path, output_dir, fmt="sdf", auto_desalt=auto_desalt)
            if ext == ".pdb":
                return prepare_ligand_from_sdf_or_pdb(name, c_path, output_dir, fmt="pdb", auto_desalt=auto_desalt)
            if ext == ".pdbqt":
                dest = output_dir / f"{name}.pdbqt"
                if c_path.resolve() != dest.resolve():
                    dest.write_text(c_path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
                return dest

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
    return write_fallback_pdbqt(name, atom_records, pdbqt_path, bonds=parsed["bonds"])


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
            write_fallback_pdbqt(conformer_name, atom_records, pdbqt_path, bonds=parsed["bonds"])
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


def prepare_ring_conformer_ensemble(
    name: str,
    smiles: str,
    output_dir: Path,
    num_conformers: int = 4,
    rmsd_threshold: float = 0.5,
) -> tuple[list[Path], Path]:
    """Generate RMSD-diverse 3D ring/macrocycle conformer PDBQT files for ensemble docking in Vina."""
    output_dir.mkdir(parents=True, exist_ok=True)
    desalt_info = standardize_and_desalt_smiles(smiles)
    clean_smiles = desalt_info["desalted_smiles"]
    manifest = []
    pdbqt_paths: list[Path] = []

    if HAS_RDKIT and HAS_MEEKO:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(clean_smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES for {name}: {clean_smiles}")
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xA1D0
        params.pruneRmsThresh = rmsd_threshold
        params.useSmallRingTorsions = True
        if hasattr(params, "useMacrocycleTorsions"):
            params.useMacrocycleTorsions = True
        conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=max(1, int(num_conformers)), params=params))
        if not conf_ids:
            conf_ids = [AllChem.EmbedMolecule(mol, params)]
        for ordinal, cid in enumerate(conf_ids, 1):
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol, confId=cid, maxIters=400)
            conf_mol = Chem.Mol(mol)
            conf_mol.RemoveAllConformers()
            conf_mol.AddConformer(mol.GetConformer(cid), assignId=True)
            conf_name = f"{name}__ringconf{ordinal:03d}"
            sdf_path = output_dir / f"{conf_name}.sdf"
            writer = Chem.SDWriter(str(sdf_path))
            writer.write(conf_mol)
            writer.close()
            pdbqt_path = _write_pdbqt(conf_name, conf_mol, output_dir)
            pdbqt_paths.append(pdbqt_path)
            manifest.append({
                "name": conf_name,
                "conformer_id": int(cid),
                "sdf": str(sdf_path),
                "pdbqt": str(pdbqt_path),
            })
    else:
        parsed = parse_smiles_tokens(clean_smiles)
        for ordinal in range(1, max(1, int(num_conformers)) + 1):
            conf_name = f"{name}__ringconf{ordinal:03d}"
            sdf_path = output_dir / f"{conf_name}.sdf"
            pdbqt_path = output_dir / f"{conf_name}.pdbqt"
            atom_records = generate_fallback_3d_coords(parsed, seed=0xA1D0, conformer_index=ordinal)
            write_fallback_sdf(conf_name, atom_records, parsed["bonds"], sdf_path)
            write_fallback_pdbqt(conf_name, atom_records, pdbqt_path, bonds=parsed["bonds"])
            pdbqt_paths.append(pdbqt_path)
            manifest.append({
                "name": conf_name,
                "conformer_id": ordinal - 1,
                "sdf": str(sdf_path),
                "pdbqt": str(pdbqt_path),
            })

    manifest_path = output_dir / f"{name}__ring_ensemble.json"
    manifest_path.write_text(
        json.dumps(
            {
                "ligand": name,
                "input_smiles": clean_smiles,
                "conformer_count": len(manifest),
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
