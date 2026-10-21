from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import pandas as pd

from .common import atom_dict, distance, require_executable

LOGGER = logging.getLogger(__name__)
HYDROPHOBIC_RES = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "TYR"}
INTERACTION_COLUMNS = [
    "Type", "Ligand_Atom", "Receptor_Res", "Receptor_Atom", "Distance_A"
]


def parse_complex(path: Path) -> tuple[list[dict], list[dict]]:
    receptor, ligand = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        atom = atom_dict(line)
        (ligand if atom["resname"] == "LIG" else receptor).append(atom)
    return receptor, ligand


def analyze_interactions(complex_pdb: Path, output_csv: Path, cys_res_num: int) -> pd.DataFrame:
    receptor, ligand = parse_complex(complex_pdb)
    lig_o = next((a for a in ligand if a["element"].upper() == "O"), None)
    lig_c = [a for a in ligand if a["element"].upper() == "C"]
    if lig_o is None or not lig_c:
        raise ValueError(f"Could not identify ligand carbonyl atoms in {complex_pdb}")
    carbonyl_c = min(lig_c, key=lambda a: distance(a["coord"], lig_o["coord"]))
    aliphatic = [a for a in lig_c if a["serial"] != carbonyl_c["serial"]]
    rows = []
    for atom in receptor:
        if atom["resname"] == "CYS" and atom["resseq"] == cys_res_num and atom["name"] == "SG":
            rows.append({"Type": "Catalytic_Attack", "Ligand_Atom": carbonyl_c["name"], "Receptor_Res": f"CYS{atom['resseq']}", "Receptor_Atom": "SG", "Distance_A": round(distance(carbonyl_c["coord"], atom["coord"]), 3)})
        if atom["resname"] in {"NAD", "NAP", "NDP"} and atom["name"] in {"C4N", "C4"}:
            rows.append({"Type": "Hydride_Transfer", "Ligand_Atom": carbonyl_c["name"], "Receptor_Res": f"{atom['resname']}{atom['resseq']}", "Receptor_Atom": atom["name"], "Distance_A": round(distance(carbonyl_c["coord"], atom["coord"]), 3)})
        if atom["element"].upper() in {"N", "O", "S"} and distance(lig_o["coord"], atom["coord"]) <= 3.5:
            rows.append({"Type": "Hydrogen_Bond", "Ligand_Atom": lig_o["name"], "Receptor_Res": f"{atom['resname']}{atom['resseq']}", "Receptor_Atom": atom["name"], "Distance_A": round(distance(lig_o["coord"], atom["coord"]), 3)})
    hits = {}
    for atom in receptor:
        if atom["resname"] in HYDROPHOBIC_RES and atom["element"].upper() == "C" and atom["name"] not in {"C", "CA"}:
            for lc in aliphatic:
                d = distance(lc["coord"], atom["coord"])
                key = (atom["chain"], atom["resname"], atom["resseq"])
                if 3.3 <= d <= 4.0 and (key not in hits or d < hits[key]["Distance_A"]):
                    hits[key] = {"Type": "Hydrophobic_Contact", "Ligand_Atom": lc["name"], "Receptor_Res": f"{atom['resname']}{atom['resseq']}", "Receptor_Atom": atom["name"], "Distance_A": round(d, 3)}
    rows.extend(hits.values())
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=INTERACTION_COLUMNS)
    frame.to_csv(output_csv, index=False)
    return frame


def generate_pymol_script(complex_pdb: Path, interactions: pd.DataFrame, figures_dir: Path, prefix: str, pymol_executable: str = "pymol") -> Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    script = figures_dir / f"{prefix}_render.pml"
    pse = figures_dir / f"{prefix}_session.pse"
    png = figures_dir / f"{prefix}_300dpi.png"
    cat_resi = "+".join(
        str(value)
        for value in interactions.loc[
            interactions["Type"] == "Catalytic_Attack", "Receptor_Res"
        ].str.extract(r"(\d+)")[0].dropna().unique()
    ) or "none"
    hbond_resi = "+".join(
        str(value)
        for value in interactions.loc[
            interactions["Type"] == "Hydrogen_Bond", "Receptor_Res"
        ].str.extract(r"(\d+)")[0].dropna().unique()
    ) or "none"
    hydro_resi = "+".join(
        str(value)
        for value in interactions.loc[
            interactions["Type"] == "Hydrophobic_Contact", "Receptor_Res"
        ].str.extract(r"(\d+)")[0].dropna().unique()
    ) or "none"
    complex_path = complex_pdb.resolve()
    pse_path = pse.resolve()
    png_path = png.resolve()
    script.write_text(f'''reinitialize
load "{complex_path.as_posix()}", complex
bg_color white
set ray_opaque_background, off
hide everything, all
show cartoon, polymer.protein
color gray85, polymer.protein
select ligand, resn LIG
show sticks, ligand
show spheres, ligand
color cyan, ligand and elem C
select cofactor, resn NAD+NAP+NDP
show sticks, cofactor
color tv_yellow, cofactor
select cat_cys, resn CYS and resi {cat_resi}
show sticks, cat_cys
color magenta, cat_cys
select hbond_residues, polymer.protein and resi {hbond_resi}
show sticks, hbond_residues
color slate, hbond_residues
select hydro_residues, polymer.protein and resi {hydro_resi}
show sticks, hydro_residues
color smudge, hydro_residues
create hydro_pocket_surf, hydro_residues
show surface, hydro_pocket_surf
set transparency, 72, hydro_pocket_surf
''', encoding="utf-8")
    with script.open("a", encoding="utf-8") as handle:
        for index, row in interactions.iterrows():
            if row["Type"] in {"Catalytic_Attack", "Hydrogen_Bond"}:
                resi = "".join(ch for ch in row["Receptor_Res"] if ch.isdigit())
                color = "magenta" if row["Type"] == "Catalytic_Attack" else "tv_orange"
                handle.write(f'distance d{index}, (ligand and name {row["Ligand_Atom"]}), (resi {resi} and name {row["Receptor_Atom"]})\ncolor {color}, d{index}\n')
        handle.write(f'orient ligand\nzoom (ligand or cat_cys), 5.0\nclip slab, 18\nsave {pse_path.as_posix()}\nray 2400, 1800\npng {png_path.as_posix()}, dpi=300\nquit\n')
    resolved = require_executable(pymol_executable)
    executable_path = Path(resolved)
    if executable_path.name.lower() == "pymolwin.exe":
        bundled_python = executable_path.with_name("python.exe")
        if not bundled_python.exists():
            raise FileNotFoundError(f"PyMOL bundled Python was not found beside {resolved}")
        command = [str(bundled_python), "-m", "pymol", "-cq", str(script)]
    else:
        command = [resolved, "-cq", str(script)]
    subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return script
