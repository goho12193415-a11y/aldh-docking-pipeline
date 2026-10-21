from __future__ import annotations

from pathlib import Path


def write_qmmm_input(complex_pdb: Path, output_dir: Path, qmmm_config: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    charge = int(qmmm_config.get("charge", 0))
    multiplicity = int(qmmm_config.get("multiplicity", 1))
    qm_residues = qmmm_config.get("qm_residues", ["CYS302", "NAD5501"])
    path = output_dir / f"{complex_pdb.stem}_qmmm.inp"
    path.write_text(
        "! B3LYP def2-SVP TightSCF\n"
        f"* xyz {charge} {multiplicity}\n"
        f"# QM region candidates: {', '.join(qm_residues)}\n"
        f"# Complex source: {complex_pdb}\n"
        "*\n"
        "%pal nprocs 4 end\n"
        "%maxcore 2000\n",
        encoding="utf-8",
    )
    return path
