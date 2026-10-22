from __future__ import annotations

import json
import re
from pathlib import Path

from .common import atom_dict


def write_qmmm_input(complex_pdb: Path, output_dir: Path, qmmm_config: dict) -> Path:
    """Extract real 3D coordinates of the QM region (ligand + catalytic residues/cofactor) into an ORCA input file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    charge = int(qmmm_config.get("charge", 0))
    multiplicity = int(qmmm_config.get("multiplicity", 1))
    qm_residues = qmmm_config.get("qm_residues", ["CYS302", "NAD5501"])
    target_Numbers = set()
    target_names = set()
    for token in qm_residues:
        m = re.match(r"^([A-Za-z]+)(\d+)$", str(token).strip())
        if m:
            target_names.add(m.group(1).upper())
            target_Numbers.add(int(m.group(2)))

    xyz_lines = []
    if complex_pdb.is_file():
        for line in complex_pdb.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.startswith(("ATOM", "HETATM")):
                continue
            atom = atom_dict(line)
            is_ligand = atom["resname"] == "LIG"
            is_qm_res = (atom["resseq"] in target_Numbers) or (
                atom["resname"] in target_names and atom["resname"] in {"NAD", "NAP", "NDP"}
            )
            if is_ligand or is_qm_res:
                # For NAD, keep nicotinamide ring and ribose near reaction center to avoid dangling phosphates
                if atom["resname"] in {"NAD", "NAP", "NDP"} and not atom["name"].startswith(("C", "N", "O")):
                    continue
                x, y, z = atom["coord"]
                elem = atom["element"].strip().capitalize() or "C"
                if len(elem) > 2 or not elem.isalpha():
                    elem = elem[:1] or "C"
                xyz_lines.append(f"  {elem:<2s}  {x:10.4f}  {y:10.4f}  {z:10.4f}")

    path = output_dir / f"{complex_pdb.stem}_qmmm.inp"
    coord_block = "\n".join(xyz_lines) + "\n" if xyz_lines else ""
    path.write_text(
        "! B3LYP D3BJ def2-SVP TightSCF Opt\n"
        f"# QM region candidates: {', '.join(str(r) for r in qm_residues)}\n"
        f"# Complex source: {complex_pdb}\n"
        f"# Extracted QM atom count: {len(xyz_lines)}\n"
        "%pal nprocs 4 end\n"
        "%maxcore 2000\n"
        f"* xyz {charge} {multiplicity}\n"
        f"{coord_block}"
        "*\n",
        encoding="utf-8",
    )
    (output_dir / f"{complex_pdb.stem}_qmmm_manifest.json").write_text(
        json.dumps(
            {
                "complex_pdb": str(complex_pdb),
                "orca_input": str(path),
                "charge": charge,
                "multiplicity": multiplicity,
                "qm_residues": list(qm_residues),
                "extracted_qm_atom_count": len(xyz_lines),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path
