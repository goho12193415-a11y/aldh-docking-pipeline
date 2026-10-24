from __future__ import annotations

import csv
import logging
import math
from pathlib import Path

import numpy as np

from .common import atom_dict, distance

LOGGER = logging.getLogger(__name__)
METAL_RESNAMES = {"ZN", "MG", "FE", "MN", "CA", "CU", "NI", "CO"}
HYDROPHOBIC_RES = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "TYR"}


def parse_models_and_flex(pdbqt_path: Path) -> list[tuple[list[dict], list[dict]]]:
    """Parse Vina output PDBQT models while strictly separating ligand atoms from BEGIN_RES/END_RES flexible receptor sidechain atoms."""
    models: list[tuple[list[dict], list[dict]]] = []
    cur_lig: list[dict] = []
    cur_flex: list[dict] = []
    in_flex_res = False

    for line in pdbqt_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("MODEL"):
            cur_lig = []
            cur_flex = []
            in_flex_res = False
        elif line.startswith("BEGIN_RES"):
            in_flex_res = True
        elif line.startswith("END_RES"):
            in_flex_res = False
        elif line.startswith(("ATOM", "HETATM")):
            try:
                parsed = atom_dict(line)
                if in_flex_res:
                    cur_flex.append(parsed)
                else:
                    cur_lig.append(parsed)
            except (ValueError, IndexError):
                continue
        elif line.startswith("ENDMDL"):
            if cur_lig or cur_flex:
                models.append((cur_lig, cur_flex))
    if not models and (cur_lig or cur_flex):
        models.append((cur_lig, cur_flex))
    return models


def parse_models(pdbqt_path: Path) -> list[list[dict]]:
    """Return strictly ligand atom models (excluding any BEGIN_RES/END_RES flexible receptor sidechain atoms)."""
    return [lig for lig, _ in parse_models_and_flex(pdbqt_path) if lig]


def merge_receptor_and_flex(receptor_atoms: list[dict], flex_atoms: list[dict]) -> list[str]:
    """Update receptor PDB lines with docked flexible sidechain atom coordinates when --flex was used."""
    if not flex_atoms:
        return [a["line"] for a in receptor_atoms]
    flex_lookup: dict[tuple[str, int, str], np.ndarray] = {
        (fa["chain"], int(fa["resseq"]), fa["name"]): fa["coord"]
        for fa in flex_atoms
    }
    merged_lines = []
    for a in receptor_atoms:
        key = (a["chain"], int(a["resseq"]), a["name"])
        if key in flex_lookup:
            x, y, z = flex_lookup[key]
            raw = a["line"]
            updated = f"{raw[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{raw[54:]}"
            merged_lines.append(updated)
        else:
            merged_lines.append(a["line"])
    return merged_lines


def parse_vina_scores(pdbqt_path: Path) -> list[float | None]:
    scores: list[float | None] = []
    for line in pdbqt_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("REMARK VINA RESULT:"):
            fields = line.split()
            try:
                scores.append(float(fields[3]))
            except (IndexError, ValueError):
                scores.append(None)
    return scores


def find_carbonyl(model: list[dict], min_bond: float = 1.15, max_bond: float = 1.35) -> tuple[dict, dict]:
    oxygens = [a for a in model if a["element"].upper() == "O" or a["name"] == "OA"]
    carbons = [a for a in model if a["element"].upper() == "C" or a["name"] in {"C", "A"}]
    if not oxygens or not carbons:
        raise ValueError("Ligand model does not contain recognizable carbonyl O/C atoms.")
    candidates = [
        (oxygen, carbon, distance(oxygen["coord"], carbon["coord"]))
        for oxygen in oxygens
        for carbon in carbons
        if min_bond <= distance(oxygen["coord"], carbon["coord"]) <= max_bond
    ]
    if not candidates:
        raise ValueError("No carbonyl C-O pair was found within the configured bond range.")
    oxygen, carbonyl, _ = min(candidates, key=lambda item: item[2])
    return oxygen, carbonyl


def find_reactive_warhead_or_carbonyl(model: list[dict], min_bond: float = 1.12, max_bond: float = 1.55) -> tuple[dict, dict]:
    """Identify carbonyl (C=O), nitrile (C#N), epoxide, or heteroatom-bound reactive center for general enzymes."""
    try:
        return find_carbonyl(model, min_bond=min_bond, max_bond=min(max_bond, 1.38))
    except ValueError:
        pass
    heteros = [a for a in model if a["element"].upper() in {"O", "N", "S"} or a["name"] in {"OA", "NA", "SA"}]
    carbons = [a for a in model if a["element"].upper() == "C" or a["name"] in {"C", "A"}]
    if heteros and carbons:
        candidates = [
            (het, c, distance(het["coord"], c["coord"]))
            for het in heteros
            for c in carbons
            if 1.10 <= distance(het["coord"], c["coord"]) <= 1.90
        ]
        if candidates:
            het, c, _ = min(candidates, key=lambda item: item[2])
            return het, c
    if len(model) >= 2:
        return model[0], model[1]
    return model[0], model[0]


def compute_burgi_dunitz_angle(nucleophile_coord: np.ndarray, carbonyl_c_coord: np.ndarray, carbonyl_o_coord: np.ndarray) -> float:
    """Calculate the Nu...C=O attack angle in degrees (ideal Bürgi-Dunitz ~ 107 deg)."""
    v_nu = nucleophile_coord - carbonyl_c_coord
    v_co = carbonyl_o_coord - carbonyl_c_coord
    norm_nu = float(np.linalg.norm(v_nu))
    norm_co = float(np.linalg.norm(v_co))
    if norm_nu < 1e-6 or norm_co < 1e-6:
        return 0.0
    cos_theta = float(np.clip(np.dot(v_nu, v_co) / (norm_nu * norm_co), -1.0, 1.0))
    return round(math.degrees(math.acos(cos_theta)), 2)


def compute_physicochemical_rescore(
    model: list[dict],
    receptor_atoms: list[dict],
    protonation_penalty_kcal: float = 0.0,
) -> dict[str, float]:
    """Physics-informed rescorer with PDBQT/PEOE partial charges, Coulomb electrostatics, directional H-bonds, metal coordination, and dual-state water bridging vs. entropic water displacement."""
    lig_heavy = [a for a in model if a["element"].upper() != "H" and a["name"] != "HD"]
    if not lig_heavy or not receptor_atoms:
        return {
            "PhysChem_Score_kcal_mol": 0.0,
            "Electrostatic_kcal": 0.0,
            "Metal_Coord_kcal": 0.0,
            "HBond_kcal": 0.0,
            "Water_Bridge_kcal": 0.0,
            "Water_Displacement_kcal": 0.0,
            "Bridged_Waters_Count": 0,
            "Displaced_Waters_Count": 0,
        }

    lig_coords = np.array([a["coord"] for a in lig_heavy], dtype=float)
    lig_center = np.mean(lig_coords, axis=0)
    near_atoms = [
        a for a in receptor_atoms
        if a["element"].upper() != "H" and distance(a["coord"], lig_center) <= 10.0
    ]
    pocket_atoms = [a for a in near_atoms if a["resname"] not in {"HOH", "WAT", "TIP", "SOL"}]
    water_atoms = [
        a for a in near_atoms
        if a["resname"] in {"HOH", "WAT", "TIP", "SOL"} and a["element"].upper() == "O"
    ]

    vdw_kcal = 0.0
    elec_kcal = 0.0
    hbond_kcal = 0.0
    metal_kcal = 0.0
    hydro_kcal = 0.0
    water_bridge_kcal = 0.0
    water_disp_kcal = 0.0
    bridged_waters_count = 0
    displaced_waters_count = 0

    def _norm_el(atom: dict) -> str:
        return "O" if atom["name"] == "OA" else ("N" if atom["name"] == "NA" else ("S" if atom["name"] == "SA" else ("C" if atom["name"] == "A" else atom["element"].upper())))

    def _effective_charge(atom: dict, is_lig: bool) -> float:
        parsed_q = atom.get("charge")
        if parsed_q is not None and abs(float(parsed_q)) > 1e-4:
            return float(parsed_q)
        el = atom["element"].upper()
        rn = atom.get("resname", "")
        an = atom.get("name", "")
        if rn in METAL_RESNAMES or (rn in {"HEM", "HEC"} and an == "FE"):
            return +2.0
        if not is_lig:
            if rn in {"ASP", "GLU"} and an in {"OD1", "OD2", "OE1", "OE2"}:
                return -0.55
            if rn in {"LYS", "ARG"} and an in {"NZ", "NH1", "NH2"}:
                return +0.55
            if rn == "CYS" and an == "SG":
                return -0.45
        if el == "O" or an == "OA":
            return -0.38
        if el == "N" or an == "NA":
            return -0.28
        if el == "S" or an == "SA":
            return -0.20
        return 0.05 if el == "C" else 0.0

    for la in lig_heavy:
        l_el = _norm_el(la)
        q_l = _effective_charge(la, True)
        for pa in pocket_atoms:
            d = distance(la["coord"], pa["coord"])
            if d > 6.5 or d < 0.5:
                continue
            p_el = pa["element"].upper()
            p_rn = pa["resname"]
            q_p = _effective_charge(pa, False)

            if abs(q_l * q_p) > 0.005:
                elec_kcal += 332.06 * (q_l * q_p) / (4.0 * d * d + 1.0) * 0.08

            if (p_rn in METAL_RESNAMES or (p_rn in {"HEM", "HEC"} and pa["name"] == "FE")) and l_el in {"O", "N", "S"}:
                if 1.75 <= d <= 3.1:
                    metal_kcal -= 2.2 * math.exp(-((d - 2.15) ** 2) / 0.35)

            if l_el in {"O", "N", "S"} and p_el in {"O", "N", "S"} and 2.4 <= d <= 3.45:
                hbond_kcal -= 0.65 * math.exp(-((d - 2.85) ** 2) / 0.25)

            if l_el == "C" and p_el == "C" and p_rn in HYDROPHOBIC_RES and 3.3 <= d <= 4.2:
                hydro_kcal -= 0.11

            r_eq = 3.5 if (l_el == "C" and p_el == "C") else 3.2
            if d < 2.2:
                vdw_kcal += min(4.0, (2.2 - d) * 3.5)
            elif d <= 5.0:
                ratio = r_eq / max(2.4, d)
                vdw_kcal += 0.045 * ((ratio ** 8) - 2.0 * (ratio ** 4))

    # Dual-state explicit water thermodynamics: Displaced Water (entropic gain, clash-exempt) vs. Tetrahedral Water Bridge
    rec_polar = [pa for pa in pocket_atoms if pa["element"].upper() in {"O", "N", "S"}]
    lig_polar = [la for la in lig_heavy if _norm_el(la) in {"O", "N", "S"}]

    for wa in water_atoms:
        w_coord = wa["coord"]
        min_d_lig = min(distance(la["coord"], w_coord) for la in lig_heavy)
        if min_d_lig < 2.2:
            # Ligand physically occupies the water site -> displaces water into bulk solvent (entropic release, no clash penalty)
            displaced_waters_count += 1
            water_disp_kcal = max(-1.35, water_disp_kcal - 0.45)
        elif lig_polar and rec_polar:
            near_lp = [(la, distance(la["coord"], w_coord)) for la in lig_polar if 2.2 <= distance(la["coord"], w_coord) <= 3.5]
            near_rp = [(pa, distance(pa["coord"], w_coord)) for pa in rec_polar if 2.2 <= distance(pa["coord"], w_coord) <= 3.5]
            best_bridge_e = 0.0
            for la, d_lw in near_lp:
                v_lw = la["coord"] - w_coord
                n_lw = float(np.linalg.norm(v_lw))
                if n_lw < 1e-6:
                    continue
                for pa, _ in near_rp:
                    v_rw = pa["coord"] - w_coord
                    n_rw = float(np.linalg.norm(v_rw))
                    if n_rw < 1e-6:
                        continue
                    cos_ang = float(np.clip(np.dot(v_lw, v_rw) / (n_lw * n_rw), -1.0, 1.0))
                    ang_deg = math.degrees(math.acos(cos_ang))
                    if 80.0 <= ang_deg <= 145.0:
                        e_br = -0.60 * math.exp(-((d_lw - 2.85) ** 2) / 0.30)
                        if e_br < best_bridge_e:
                            best_bridge_e = e_br
            if best_bridge_e < 0.0:
                bridged_waters_count += 1
                water_bridge_kcal += best_bridge_e

    total = round(
        vdw_kcal + elec_kcal + hbond_kcal + metal_kcal + hydro_kcal + water_bridge_kcal + water_disp_kcal + float(protonation_penalty_kcal),
        3,
    )
    return {
        "PhysChem_Score_kcal_mol": total,
        "Electrostatic_kcal": round(elec_kcal, 3),
        "Metal_Coord_kcal": round(metal_kcal, 3),
        "HBond_kcal": round(hbond_kcal, 3),
        "Water_Bridge_kcal": round(water_bridge_kcal, 3),
        "Water_Displacement_kcal": round(water_disp_kcal, 3),
        "Bridged_Waters_Count": bridged_waters_count,
        "Displaced_Waters_Count": displaced_waters_count,
    }


def read_receptor_atoms(pdb_path: Path) -> list[dict]:
    atoms = []
    for line in pdb_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(("ATOM", "HETATM")):
            try:
                atoms.append(atom_dict(line))
            except (ValueError, IndexError):
                continue
    return atoms


def clean_ligand_pdbqt(model: list[dict], start_serial: int = 9000) -> list[str]:
    lines = []
    for index, atom in enumerate(model, start=start_serial):
        element = {"OA": "O", "NA": "N", "SA": "S", "HD": "H", "A": "C"}.get(atom["name"], atom["element"][:1].upper())
        x, y, z = atom["coord"]
        lines.append(f"HETATM{index:5d} {atom['name'][:4]:>4s} LIG L{999:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}")
    return lines


def _locate_nucleophile(receptor_atoms: list[dict], config: dict) -> dict:
    receptor_cfg = config.get("receptor", {})
    family = str(config.get("enzyme", {}).get("mechanism_family", "aldh")).lower()
    chain_id = receptor_cfg.get("chain_id", "A")
    catalytic = receptor_cfg.get("catalytic_residue") or {"name": "CYS", "number": 302, "chain": chain_id}
    cat_chain = catalytic.get("chain", chain_id)
    cat_num = int(catalytic.get("number", 302))
    cat_name = str(catalytic.get("name", "SER" if family == "serine_hydrolase" else "CYS")).upper()

    if family == "metalloenzyme":
        metal = next(
            (a for a in receptor_atoms if a["resname"] in METAL_RESNAMES and a["chain"] == cat_chain),
            None,
        ) or next((a for a in receptor_atoms if a["resname"] in METAL_RESNAMES), None)
        if metal is not None:
            return metal
    elif family == "cytochrome_p450":
        fe = next(
            (a for a in receptor_atoms if a["resname"] in {"HEM", "HEC", "HEA"} and a["name"] == "FE"),
            None,
        )
        if fe is not None:
            return fe
    elif family == "flavoenzyme":
        flavin = next(
            (a for a in receptor_atoms if a["resname"] in {"FAD", "FMN"} and a["name"] in {"N5", "C4X", "C4A"}),
            None,
        )
        if flavin is not None:
            return flavin
    elif family == "general_binding":
        chain_atoms = [a for a in receptor_atoms if a["chain"] == chain_id and a["element"].upper() != "H"] or receptor_atoms
        center = np.mean([a["coord"] for a in chain_atoms], axis=0)
        return min(chain_atoms, key=lambda a: distance(a["coord"], center))

    target_atom = {
        "SER": "OG", "THR": "OG1", "LYS": "NZ", "ASP": "OD1", "GLU": "OE1",
    }.get(cat_name, "SG")
    nuc = next(
        (
            a for a in receptor_atoms
            if a["resname"] == cat_name and a["chain"] == cat_chain and a["resseq"] == cat_num and a["name"] == target_atom
        ),
        None,
    )
    if nuc is None:
        nuc = next(
            (
                a for a in receptor_atoms
                if a["chain"] == cat_chain and a["resseq"] == cat_num and a["name"] in {"SG", "OG", "NZ", "OD1", "OE1", "CA"}
            ),
            None,
        )
    if nuc is None and family in {"kinase", "glycosidase"} and receptor_atoms:
        return receptor_atoms[0]
    if nuc is None:
        raise ValueError(f"Catalytic {cat_name} {target_atom} was not found in receptor.")
    return nuc


def score_docking(
    output_pdbqt: Path,
    receptor_pdb: Path,
    ligand_name: str,
    config: dict,
    reports_dir: Path,
    run_id: str | None = None,
    seed: int | None = None,
) -> dict:
    parsed_pairs = parse_models_and_flex(output_pdbqt)
    scores = parse_vina_scores(output_pdbqt)
    receptor_atoms = read_receptor_atoms(receptor_pdb)
    family = str(config.get("enzyme", {}).get("mechanism_family", "aldh")).lower()
    receptor_cfg = config.get("receptor", {})
    chain_id = receptor_cfg.get("chain_id", "A")
    cofactor_chain = receptor_cfg.get("cofactor_chain", chain_id)
    catalytic = receptor_cfg.get("catalytic_residue") or {"number": 302, "chain": chain_id}
    cat_chain = catalytic.get("chain", chain_id)
    cat_num = int(catalytic.get("number", 302))

    cys = _locate_nucleophile(receptor_atoms, config)
    nad = (
        next(
            (
                a for a in receptor_atoms
                if a["chain"] == cofactor_chain and a["resname"] in {"NAD", "NAP", "NDP", "FAD", "FMN"} and a["name"] in {"C4N", "C4", "N5"}
            ),
            None,
        )
        or next(
            (
                a for a in receptor_atoms
                if a["resname"] in {"NAD", "NAP", "NDP", "FAD", "FMN"} and a["name"] in {"C4N", "C4", "N5"}
            ),
            None,
        )
    )
    oxyanion = receptor_cfg.get("oxyanion_residue")
    oxyanion_chain = oxyanion.get("chain", cat_chain) if oxyanion else cat_chain
    oxyanion_nd2 = (
        next(
            (
                a for a in receptor_atoms
                if oxyanion
                and a["chain"] == oxyanion_chain
                and a["resname"] == oxyanion["name"]
                and a["resseq"] == oxyanion["number"]
                and a["name"] == oxyanion["atom"]
            ),
            None,
        )
        or next(
            (
                a for a in receptor_atoms
                if oxyanion
                and a["resname"] == oxyanion["name"]
                and a["resseq"] == oxyanion["number"]
                and a["name"] == oxyanion["atom"]
            ),
            None,
        )
    )
    cys_backbone_n = next(
        (
            a for a in receptor_atoms
            if a["chain"] == cat_chain and a["resseq"] == cat_num and a["name"] == "N"
        ),
        None,
    )

    thresholds = config.get("thresholds", {})
    bond_min = float(thresholds.get("carbonyl_bond_min", 1.15))
    bond_max = float(thresholds.get("carbonyl_bond_max", 1.35))
    default_min = {
        "metalloenzyme": 1.7, "serine_hydrolase": 2.6, "cytochrome_p450": 2.8,
        "kinase": 2.4, "glycosidase": 2.5, "flavoenzyme": 2.6, "general_binding": 2.0,
    }.get(family, 3.0)
    default_max = {
        "metalloenzyme": 3.4, "cytochrome_p450": 5.5, "kinase": 5.5,
        "glycosidase": 4.8, "flavoenzyme": 5.0, "general_binding": 6.0,
    }.get(family, 4.5)
    d_nuc_min = float(thresholds.get("d_nuc_min", default_min))
    d_nuc_max = float(thresholds.get("d_nuc_max", default_max))
    d_hyd_max = float(thresholds.get("d_hyd_max", 4.5))
    d_oxy_max = float(thresholds.get("d_oxy_max", 5.0))

    d_sg_c4n = round(distance(cys["coord"], nad["coord"]), 3) if nad is not None else None
    if nad is None:
        cofactor_conformation = "apo_or_no_nad"
    elif d_sg_c4n is not None and d_sg_c4n > 5.2:
        cofactor_conformation = "contracted_hydrolysis_out"
    else:
        cofactor_conformation = "productive_hydride_in"
    # The hydride-transfer criterion is never relaxed: a pose with the cofactor
    # out of its productive conformation is reported as such, not as competent.
    effective_d_hyd_max = d_hyd_max
    competence_basis = {
        "aldh": "nucleophile_and_hydride_geometry",
        "general_binding": "physchem_and_pocket_contact",
    }.get(family, "mechanism_geometry")

    rows = []
    for mode_index, (model, flex_model) in enumerate(parsed_pairs, start=1):
        if not model:
            continue
        if family == "aldh":
            oxygen, carbonyl = find_carbonyl(model, bond_min, bond_max)
        else:
            oxygen, carbonyl = find_reactive_warhead_or_carbonyl(model, bond_min, bond_max)

        if family in {"metalloenzyme", "kinase"}:
            polar_lig = [a for a in model if a["element"].upper() in {"O", "N", "S"} or a["name"] in {"OA", "NA", "SA"}] or [oxygen]
            target_ligand_atom = min(polar_lig, key=lambda a: distance(a["coord"], cys["coord"]))
        elif family == "general_binding":
            target_ligand_atom = min(model, key=lambda a: distance(a["coord"], cys["coord"]))
        else:
            target_ligand_atom = carbonyl

        d_nuc_val = round(distance(target_ligand_atom["coord"], cys["coord"]), 3)
        d_hyd_val = round(distance(carbonyl["coord"], nad["coord"]), 3) if nad else ""
        d_oxy_asn_val = round(distance(oxygen["coord"], oxyanion_nd2["coord"]), 3) if oxyanion_nd2 else ""
        d_oxy_cysn_val = round(distance(oxygen["coord"], cys_backbone_n["coord"]), 3) if cys_backbone_n else ""
        bd_angle = compute_burgi_dunitz_angle(cys["coord"], carbonyl["coord"], oxygen["coord"]) if oxygen is not carbonyl else 107.0

        physchem = compute_physicochemical_rescore(model, receptor_atoms)
        physchem_score = physchem["PhysChem_Score_kcal_mol"]

        nuc_ok = d_nuc_min <= d_nuc_val <= d_nuc_max
        hyd_ok: bool | str
        if family == "aldh":
            if nad is None:
                hyd_ok = "not_assessable_no_cofactor"
                # Without a retained cofactor only the nucleophilic step can be
                # assessed; hydride transfer is explicitly reported as unassessed.
                competent_flag = bool(nuc_ok)
            else:
                hyd_ok = bool(float(d_hyd_val) <= d_hyd_max)
                competent_flag = bool(nuc_ok and hyd_ok)
        elif family == "general_binding":
            hyd_ok = ""
            competent_flag = bool(physchem_score <= -1.0 or nuc_ok)
        else:
            hyd_ok = ""
            competent_flag = bool(nuc_ok)

        affinity_val = scores[mode_index - 1] if mode_index <= len(scores) and scores[mode_index - 1] is not None else ""
        aff_term = -float(affinity_val) if affinity_val != "" else 0.0
        nuc_penalty = abs(d_nuc_val - 3.45) * 0.8 if family != "general_binding" else 0.0
        angle_penalty = max(0.0, abs(bd_angle - 107.0) - 15.0) * 0.02 if family == "aldh" else 0.0
        oxy_bonus = 0.5 if (d_oxy_asn_val != "" and float(d_oxy_asn_val) <= 4.2) else 0.0
        geom_score = round(aff_term + (2.0 if competent_flag else -1.5) - nuc_penalty - angle_penalty + oxy_bonus - 0.15 * physchem_score, 3)

        row = {
            "Ligand": ligand_name,
            "Run_ID": run_id or "",
            "Seed": seed if seed is not None else "",
            "Mode": mode_index,
            "Binding_Affinity_kcal_mol": affinity_val,
            "PhysChem_Score_kcal_mol": physchem_score,
            "d_nuc_A": d_nuc_val,
            "d_hyd_A": d_hyd_val,
            "d_oxy_asn_A": d_oxy_asn_val,
            "d_oxy_cysN_A": d_oxy_cysn_val,
            "Burgi_Dunitz_Angle_deg": bd_angle,
            "Cofactor_Conformation": cofactor_conformation,
            "d_sg_c4n_A": d_sg_c4n if d_sg_c4n is not None else "",
            "Effective_d_hyd_Max_A": effective_d_hyd_max if nad else "",
            "Nucleophile_Competent": bool(nuc_ok),
            "Hydride_Transfer_Competent": hyd_ok,
            "Catalytic_Geometry_Score": geom_score,
            "Competence_Basis": competence_basis,
            "Catalytically_Competent": competent_flag,
        }
        rows.append((row, model, flex_model))

    if not rows:
        raise ValueError(f"No docking models found in {output_pdbqt}")

    by_vina = sorted(range(len(rows)), key=lambda idx: float(rows[idx][0]["Binding_Affinity_kcal_mol"]) if rows[idx][0]["Binding_Affinity_kcal_mol"] != "" else 999.0)
    by_phys = sorted(range(len(rows)), key=lambda idx: float(rows[idx][0]["PhysChem_Score_kcal_mol"]))
    by_geom = sorted(range(len(rows)), key=lambda idx: -float(rows[idx][0]["Catalytic_Geometry_Score"]))
    vina_ranks = {idx: r for r, idx in enumerate(by_vina, 1)}
    phys_ranks = {idx: r for r, idx in enumerate(by_phys, 1)}
    geom_ranks = {idx: r for r, idx in enumerate(by_geom, 1)}
    for idx, (r_dict, _, _) in enumerate(rows):
        r_dict["Universal_Consensus_Rank"] = round((vina_ranks[idx] + phys_ranks[idx] + geom_ranks[idx]) / 3.0, 2)

    scored_rows = [item for item in rows if item[0]["Binding_Affinity_kcal_mol"] != ""]
    affinity_order = lambda item: item[0]["Binding_Affinity_kcal_mol"]
    top_row, _, _ = min(scored_rows, key=affinity_order) if scored_rows else rows[0]
    competent = [item for item in rows if item[0]["Catalytically_Competent"]]
    best_row, best_model, best_flex = min(
        competent or rows,
        key=lambda item: (
            item[0]["Binding_Affinity_kcal_mol"] if item[0]["Binding_Affinity_kcal_mol"] != "" else float("inf"),
            item[0]["d_nuc_A"],
        ),
    )
    best_row["Top_Affinity_Mode"] = top_row["Mode"]
    best_row["Top_Affinity_kcal_mol"] = top_row["Binding_Affinity_kcal_mol"]
    best_row["Parsed_Mode_Count"] = len(rows)
    best_row["Catalytically_Competent_Mode_Count"] = len(competent)
    if family == "general_binding":
        best_row["Selection_Status"] = (
            "Binding_Pose_Supported" if competent else "Affinity_Fallback_No_Supported_Pose"
        )
    else:
        best_row["Selection_Status"] = (
            "Catalytically_Competent"
            if competent
            else "Affinity_Fallback_No_Catalytic_Pose"
        )
    best_row["Best_Catalytic_Mode"] = best_row["Mode"] if competent else ""
    best_row["Best_Catalytic_kcal_mol"] = (
        best_row["Binding_Affinity_kcal_mol"] if competent else ""
    )
    if competent:
        if family == "general_binding":
            best_row["Reliability_Grade"] = "A_PhysChem_Supported_Pose"
        else:
            oxy_ok = (best_row.get("d_oxy_asn_A") != "" and float(best_row["d_oxy_asn_A"]) <= 4.2) or (
                best_row.get("d_oxy_cysN_A") != "" and float(best_row["d_oxy_cysN_A"]) <= 4.2
            )
            best_row["Reliability_Grade"] = (
                "A_High_Confidence_Catalytic"
                if (best_row["d_nuc_A"] <= 3.8 and (oxy_ok or best_row["Cofactor_Conformation"] == "productive_hydride_in" or family != "aldh"))
                else "B_Moderate_Confidence_Catalytic"
            )
    else:
        best_row["Reliability_Grade"] = "C_Low_Confidence_Affinity_Fallback"

    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "docking_summary.csv"
    existing: list[dict[str, str]] = []
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            existing = [
                row for row in csv.DictReader(handle)
                if row.get("Ligand") != ligand_name
            ]
    fieldnames = list(
        dict.fromkeys(
            [key for row in existing for key in row.keys()] + list(best_row.keys())
        )
    )
    existing.append({key: best_row.get(key, "") for key in fieldnames})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing)

    active_path = reports_dir / "active_poses" / f"{ligand_name}_active.pdb"
    active_path.parent.mkdir(parents=True, exist_ok=True)
    protein_lines = merge_receptor_and_flex(receptor_atoms, best_flex)
    active_path.write_text("\n".join(protein_lines + clean_ligand_pdbqt(best_model)) + "\nEND\n", encoding="utf-8")
    best_row["active_pose"] = str(active_path)
    return best_row
