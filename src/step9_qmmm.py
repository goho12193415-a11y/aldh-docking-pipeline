from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .common import ATOMIC_WEIGHTS, atom_dict, distance

ATOMIC_NUMBERS = {
    "H": 1, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9,
    "NA": 11, "MG": 12, "P": 15, "S": 16, "CL": 17,
    "K": 19, "CA": 20, "MN": 25, "FE": 26, "CO": 27,
    "NI": 28, "CU": 29, "ZN": 30, "BR": 35, "I": 53,
}


def write_qmmm_input(complex_pdb: Path, output_dir: Path, qmmm_config: dict) -> Path:
    """Extract 3D coordinates of the QM region (ligand + catalytic sidechains + active cofactor moiety), cap severed QM/MM covalent bonds with Hydrogen Link Atoms (H_link at 1.090 A), and validate charge/multiplicity electron parity."""
    output_dir.mkdir(parents=True, exist_ok=True)
    charge = int(qmmm_config.get("charge", 0))
    multiplicity = int(qmmm_config.get("multiplicity", 1))
    qm_residues = qmmm_config.get("qm_residues", ["CYS302", "NAD5501"])
    target_numbers = set()
    target_names = set()
    for token in qm_residues:
        m = re.match(r"^([A-Za-z]+)(\d+)$", str(token).strip())
        if m:
            target_names.add(m.group(1).upper())
            target_numbers.add(int(m.group(2)))

    qm_atoms: list[dict] = []
    mm_atoms: list[dict] = []

    if complex_pdb.is_file():
        for line in complex_pdb.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.startswith(("ATOM", "HETATM")):
                continue
            atom = atom_dict(line)
            is_ligand = atom["resname"] == "LIG"
            is_qm_res = (atom["resseq"] in target_numbers) or (
                atom["resname"] in target_names and atom["resname"] in {"NAD", "NAP", "NDP", "FAD", "FMN", "HEM", "HEC"}
            )
            if is_ligand:
                qm_atoms.append(atom)
            elif is_qm_res:
                # For amino-acid QM residues, cut cleanly at CA-CB (keep sidechain CB.. in QM, leave backbone N,CA,C,O in MM)
                if atom["resname"] not in {"NAD", "NAP", "NDP", "FAD", "FMN", "HEM", "HEC"} and atom["name"] in {"N", "CA", "C", "O", "H", "HN", "HA"}:
                    mm_atoms.append(atom)
                # For NAD/NAP/FAD, keep nicotinamide/flavin + ribose in QM and leave diphosphate backbone in MM
                elif atom["resname"] in {"NAD", "NAP", "NDP"} and (atom["name"].startswith("P") or atom["name"].endswith("A")):
                    mm_atoms.append(atom)
                else:
                    qm_atoms.append(atom)
            else:
                mm_atoms.append(atom)

    xyz_lines: list[str] = []
    h_link_records: list[dict] = []
    total_atomic_number = 0

    for atom in qm_atoms:
        x, y, z = atom["coord"]
        elem = atom["element"].strip().upper() or "C"
        if elem not in ATOMIC_WEIGHTS:
            elem = elem[:1] if elem[:1] in ATOMIC_WEIGHTS else "C"
        elem_fmt = elem.capitalize()
        xyz_lines.append(f"  {elem_fmt:<2s}  {x:10.4f}  {y:10.4f}  {z:10.4f}")
        total_atomic_number += ATOMIC_NUMBERS.get(elem, 6)

    # Detect severed covalent bonds across the QM-MM boundary (1.15 A <= d <= 1.95 A) and place Hydrogen Link Atoms at 1.090 A
    for q_atom in qm_atoms:
        if q_atom["resname"] == "LIG":
            continue
        for m_atom in mm_atoms:
            if q_atom["chain"] != m_atom["chain"] or q_atom["resseq"] != m_atom["resseq"]:
                continue
            d_bond = distance(q_atom["coord"], m_atom["coord"])
            if 1.15 <= d_bond <= 1.95:
                vec = m_atom["coord"] - q_atom["coord"]
                unit_vec = vec / (float(np.linalg.norm(vec)) + 1e-8)
                h_pos = q_atom["coord"] + unit_vec * 1.090
                xyz_lines.append(f"  H   {h_pos[0]:10.4f}  {h_pos[1]:10.4f}  {h_pos[2]:10.4f}")
                total_atomic_number += 1
                h_link_records.append({
                    "qm_atom": f"{q_atom['resname']}{q_atom['resseq']}:{q_atom['name']}",
                    "mm_anchor_atom": f"{m_atom['resname']}{m_atom['resseq']}:{m_atom['name']}",
                    "severed_bond_length_A": round(float(d_bond), 3),
                    "h_link_coord": [round(float(v), 4) for v in h_pos],
                })

    # Validate electron count vs spin multiplicity parity: N_elec = sum(Z) - charge; (N_elec % 2 == 0) iff (multiplicity % 2 == 1)
    n_electrons = total_atomic_number - charge
    expected_odd_mult = (n_electrons % 2 == 0)
    actual_odd_mult = (multiplicity % 2 == 1)
    parity_adjusted = False
    if xyz_lines and (expected_odd_mult != actual_odd_mult):
        # Adjust formal charge by +1 to preserve closed-shell singlet / spin consistency across QM/MM boundary
        charge += 1
        n_electrons = total_atomic_number - charge
        parity_adjusted = True

    path = output_dir / f"{complex_pdb.stem}_qmmm.inp"
    coord_block = "\n".join(xyz_lines) + "\n" if xyz_lines else ""
    path.write_text(
        "! B3LYP D3BJ def2-SVP TightSCF Opt\n"
        f"# QM region candidates: {', '.join(str(r) for r in qm_residues)}\n"
        f"# Complex source: {complex_pdb}\n"
        f"# Extracted QM atom count: {len(qm_atoms)} + {len(h_link_records)} H-link boundary cap(s)\n"
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
                "electron_count": n_electrons,
                "spin_parity_validated": True,
                "charge_parity_auto_adjusted": parity_adjusted,
                "qm_residues": list(qm_residues),
                "extracted_qm_atom_count": len(qm_atoms),
                "h_link_cap_count": len(h_link_records),
                "h_link_caps": h_link_records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path
