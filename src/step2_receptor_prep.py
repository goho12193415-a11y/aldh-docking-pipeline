from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from Bio.PDB import PDBIO, PDBParser, Select, Superimposer

from .common import require_executable

LOGGER = logging.getLogger(__name__)
COFACTORS = {"NAD", "NAP", "NDP"}


class ReceptorSelect(Select):
    def __init__(self, chain_id: str, keep_cofactors: set[str], cofactor_chain: str | None = None):
        self.chain_id = chain_id
        self.keep_cofactors = keep_cofactors
        self.cofactor_chain = cofactor_chain or chain_id

    def accept_chain(self, chain) -> int:
        return 1

    def accept_residue(self, residue) -> int:
        name = residue.get_resname().strip()
        if residue.id[0] == " ":
            return int(residue.get_parent().id == self.chain_id)
        return int(name in self.keep_cofactors and residue.get_parent().id == self.cofactor_chain)


def transplant_cofactor(target_pdb: Path, template_pdb: Path, output_pdb: Path, chain_id: str, cofactor_names: set[str]) -> None:
    parser = PDBParser(QUIET=True)
    target = parser.get_structure("target", str(target_pdb))
    template = parser.get_structure("template", str(template_pdb))
    target_chain = target[0][chain_id]
    template_chain = template[0][chain_id]
    if any(residue.get_resname().strip() in cofactor_names for residue in target_chain):
        io = PDBIO()
        io.set_structure(target)
        io.save(str(output_pdb))
        return
    target_ca = [r["CA"] for r in target_chain if "CA" in r and r.id[0] == " "]
    template_ca = [r["CA"] for r in template_chain if "CA" in r and r.id[0] == " "]
    count = min(len(target_ca), len(template_ca))
    if count < 10:
        raise ValueError("Not enough common C-alpha atoms for cofactor transplantation.")
    superimposer = Superimposer()
    superimposer.set_atoms(target_ca[:count], template_ca[:count])
    template_cofactors = [
        residue
        for template_model in template
        for template_source_chain in template_model
        for residue in template_source_chain
        if residue.get_resname().strip() in cofactor_names
    ]
    for residue in template_cofactors:
        copied = residue.copy()
        superimposer.apply(copied)
        copied.id = (copied.id[0], copied.id[1], copied.id[2])
        target_chain.add(copied)
    io = PDBIO()
    io.set_structure(target)
    io.save(str(output_pdb))


def prepare_receptor(
    input_pdb: Path,
    output_dir: Path,
    chain_id: str,
    keep_cofactors: set[str],
    template_pdb: Path | None = None,
    cofactor_chain: str | None = None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned_pdb = output_dir / "receptor_cleaned.pdb"
    source_pdb = input_pdb
    if template_pdb:
        source_pdb = output_dir / "receptor_with_transplanted_cofactor.pdb"
        transplant_cofactor(input_pdb, template_pdb, source_pdb, chain_id, keep_cofactors)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("receptor", str(source_pdb))
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(cleaned_pdb), ReceptorSelect(chain_id, keep_cofactors, cofactor_chain))

    obabel = require_executable("obabel")
    pdbqt_path = output_dir / "receptor.pdbqt"
    command = [
        obabel, "-ipdb", str(cleaned_pdb), "-opdbqt", "-O", str(pdbqt_path),
        "-xr", "--partialcharge", "gasteiger",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"OpenBabel failed:\n{result.stdout}\n{result.stderr}")
    return cleaned_pdb, pdbqt_path
