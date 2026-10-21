from __future__ import annotations

from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser

def locate_atom(pdb_path: Path, resname: str, atom_name: str, resseq: int | None, chain_id: str) -> np.ndarray:
    structure = PDBParser(QUIET=True).get_structure("structure", str(pdb_path))
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            for residue in chain:
                if residue.get_resname().strip() != resname:
                    continue
                if resseq is not None and residue.id[1] != resseq:
                    continue
                if atom_name in residue:
                    return residue[atom_name].coord.astype(float)
    raise ValueError(f"Could not locate {resname} {atom_name} in chain {chain_id}.")


def locate_box(
    pdb_path: Path,
    chain_id: str,
    cys_number: int,
    box_size: tuple[float, float, float],
    offset_outward: float = 2.0,
    center_mode: str = "midpoint_if_cofactor",
) -> dict[str, float]:
    sg = locate_atom(pdb_path, "CYS", "SG", cys_number, chain_id)
    cofactor = None
    fallback_cofactor = None
    structure = PDBParser(QUIET=True).get_structure("structure", str(pdb_path))
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname().strip() in {"NAD", "NAP", "NDP"}:
                    for atom_name in ("C4N", "C4"):
                        if atom_name in residue:
                            candidate = residue[atom_name].coord.astype(float)
                            if chain.id == chain_id:
                                cofactor = candidate
                            elif fallback_cofactor is None:
                                fallback_cofactor = candidate
                            break
                    if cofactor is not None:
                        break
            if cofactor is not None:
                break
    cofactor = cofactor if cofactor is not None else fallback_cofactor
    if cofactor is not None and center_mode == "midpoint_if_cofactor":
        center = (sg + cofactor) / 2
    elif cofactor is not None:
        direction = cofactor - sg
        norm = np.linalg.norm(direction)
        center = sg if norm == 0 else sg + direction / norm * offset_outward
    else:
        center = sg
    return {
        "center_x": round(float(center[0]), 3),
        "center_y": round(float(center[1]), 3),
        "center_z": round(float(center[2]), 3),
        "size_x": box_size[0],
        "size_y": box_size[1],
        "size_z": box_size[2],
    }
