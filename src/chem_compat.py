from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

ATOMIC_WEIGHTS = {
    "H": 1.008, "LI": 6.94, "BE": 9.012, "B": 10.81, "C": 12.011,
    "N": 14.007, "O": 15.999, "F": 18.998, "NA": 22.990, "MG": 24.305,
    "AL": 26.982, "SI": 28.085, "P": 30.974, "S": 32.06, "CL": 35.45,
    "K": 39.098, "CA": 40.078, "ZN": 65.38, "FE": 55.845, "CU": 63.546,
    "MN": 54.938, "CO": 58.933, "NI": 58.693, "BR": 79.904, "I": 126.904,
    "PT": 195.084, "AU": 196.967, "HG": 200.592, "PB": 207.2,
}

DEFAULT_VALENCE = {
    "C": 4, "N": 3, "O": 2, "S": 2, "P": 3, "B": 3, "F": 1, "CL": 1, "BR": 1, "I": 1,
}

try:
    from rdkit import Chem as _rdkit_Chem  # type: ignore
    HAS_RDKIT = True
except ImportError:
    _rdkit_Chem = None
    HAS_RDKIT = False

try:
    import meeko as _meeko_mod  # type: ignore
    HAS_MEEKO = True
except ImportError:
    _meeko_mod = None
    HAS_MEEKO = False


def parse_smiles_tokens(smiles: str) -> dict:
    """Parse a SMILES string into atoms, bonds, rings, charges, isotopes, and stereocenters."""
    s = (smiles or "").strip()
    if not s or any(ch.isspace() for ch in s):
        raise ValueError(f"Invalid SMILES: {smiles}")
    fragments = [frag for frag in s.split(".") if frag]
    if not fragments:
        raise ValueError(f"Invalid SMILES: {smiles}")

    atoms: list[dict] = []
    bonds: list[tuple[int, int, int, bool]] = []
    ring_openings: dict[str, tuple[int, int]] = {}
    branch_stack: list[int] = []
    prev_atom: int | None = None
    pending_bond_order = 1

    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == ".":
            prev_atom = None
            pending_bond_order = 1
            i += 1
            continue
        if ch == "(":
            if prev_atom is None:
                raise ValueError(f"Invalid branch opening in SMILES: {smiles}")
            branch_stack.append(prev_atom)
            i += 1
            continue
        if ch == ")":
            if not branch_stack:
                raise ValueError(f"Unbalanced branch parenthesis in SMILES: {smiles}")
            prev_atom = branch_stack.pop()
            i += 1
            continue
        if ch in "-=#:":
            pending_bond_order = {"-": 1, "=": 2, "#": 3, ":": 1}[ch]
            i += 1
            continue
        if ch in "/\\":
            i += 1
            continue
        if ch == "%" and i + 2 < n and s[i + 1:i + 3].isdigit():
            ring_id = s[i + 1:i + 3]
            if prev_atom is None:
                raise ValueError(f"Invalid ring closure in SMILES: {smiles}")
            if ring_id in ring_openings:
                open_atom, open_order = ring_openings.pop(ring_id)
                bonds.append((open_atom, prev_atom, max(open_order, pending_bond_order), True))
            else:
                ring_openings[ring_id] = (prev_atom, pending_bond_order)
            pending_bond_order = 1
            i += 3
            continue
        if ch.isdigit():
            ring_id = ch
            if prev_atom is None:
                raise ValueError(f"Invalid ring closure in SMILES: {smiles}")
            if ring_id in ring_openings:
                open_atom, open_order = ring_openings.pop(ring_id)
                bonds.append((open_atom, prev_atom, max(open_order, pending_bond_order), True))
            else:
                ring_openings[ring_id] = (prev_atom, pending_bond_order)
            pending_bond_order = 1
            i += 1
            continue
        if ch == "[":
            end = s.find("]", i)
            if end == -1:
                raise ValueError(f"Unclosed bracket in SMILES: {smiles}")
            body = s[i + 1:end]
            m = re.match(r"^(\d+)?([A-Za-z][a-z]?)(@@|@)?(H\d*)?([+-]\d*|[+-]{2,})?$", body)
            if not m:
                raise ValueError(f"Invalid bracket atom [{body}] in SMILES: {smiles}")
            isotope_str, sym, chiral, h_str, charge_str = m.groups()
            isotope = int(isotope_str) if isotope_str else 0
            aromatic = sym.islower()
            element = sym.upper()
            if element not in ATOMIC_WEIGHTS:
                raise ValueError(f"Unknown element {element} in SMILES: {smiles}")
            explicit_h = 0
            if h_str:
                explicit_h = int(h_str[1:]) if len(h_str) > 1 else 1
            charge = 0
            if charge_str:
                if charge_str in {"+", "-"}:
                    charge = 1 if charge_str == "+" else -1
                elif charge_str.startswith("+") and charge_str[1:].isdigit():
                    charge = int(charge_str[1:])
                elif charge_str.startswith("-") and charge_str[1:].isdigit():
                    charge = -int(charge_str[1:])
                elif set(charge_str) == {"+"}:
                    charge = len(charge_str)
                elif set(charge_str) == {"-"}:
                    charge = -len(charge_str)
            idx = len(atoms)
            atoms.append({
                "index": idx,
                "symbol": element,
                "aromatic": aromatic,
                "isotope": isotope,
                "chiral": chiral or "",
                "explicit_h": explicit_h,
                "charge": charge,
                "bracket": True,
                "radical": 0,
            })
            if prev_atom is not None:
                bonds.append((prev_atom, idx, pending_bond_order, False))
            prev_atom = idx
            pending_bond_order = 1
            i = end + 1
            continue
        if s[i:i + 2] in {"Cl", "Br"}:
            sym = s[i:i + 2]
            step = 2
        elif ch in "BCNOPSFIbcnosp":
            sym = ch
            step = 1
        else:
            raise ValueError(f"Unrecognized character '{ch}' in SMILES: {smiles}")
        aromatic = sym.islower()
        element = sym.upper()
        idx = len(atoms)
        atoms.append({
            "index": idx,
            "symbol": element,
            "aromatic": aromatic,
            "isotope": 0,
            "chiral": "",
            "explicit_h": None,
            "charge": 0,
            "bracket": False,
            "radical": 0,
        })
        if prev_atom is not None:
            order = 1 if not (aromatic and atoms[prev_atom]["aromatic"]) else 1
            bonds.append((prev_atom, idx, pending_bond_order if pending_bond_order > 1 else order, False))
        prev_atom = idx
        pending_bond_order = 1
        i += step

    if branch_stack or ring_openings:
        raise ValueError(f"Unclosed rings or branches in SMILES: {smiles}")
    if not atoms:
        raise ValueError(f"No atoms parsed from SMILES: {smiles}")

    adj: dict[int, list[tuple[int, int]]] = {a["index"]: [] for a in atoms}
    for u, v, order, _ in bonds:
        adj[u].append((v, order))
        adj[v].append((u, order))

    total_h = 0
    for a in atoms:
        idx = a["index"]
        bond_sum = sum(order for _, order in adj[idx]) + (1 if a["aromatic"] else 0)
        if a["explicit_h"] is not None:
            h_count = a["explicit_h"]
        else:
            val = DEFAULT_VALENCE.get(a["symbol"], 0)
            h_count = max(0, val - bond_sum + a["charge"])
        a["implicit_h"] = h_count
        total_h += h_count

    heavy_atoms = [a for a in atoms if a["symbol"] != "H"]
    ring_count = sum(1 for _, _, _, is_ring in bonds if is_ring)

    unassigned_chiral: list[int] = []
    assigned_chiral: list[int] = []
    for a in atoms:
        if a["chiral"] in {"@", "@@"}:
            assigned_chiral.append(a["index"])
        elif a["symbol"] == "C" and not a["aromatic"]:
            neighbors = adj[a["index"]]
            if len(neighbors) >= 3 and ring_count >= 2:
                neighbor_syms = sorted(atoms[nb]["symbol"] + str(nb) for nb, _ in neighbors)
                if len(set(neighbor_syms)) == len(neighbor_syms) and any(atoms[nb]["symbol"] == "O" for nb, _ in neighbors):
                    unassigned_chiral.append(a["index"])

    rotatable_bonds = 0
    for u, v, order, is_ring in bonds:
        if is_ring or order != 1:
            continue
        if atoms[u]["symbol"] == "H" or atoms[v]["symbol"] == "H":
            continue
        u_heavy_deg = sum(1 for nb, _ in adj[u] if atoms[nb]["symbol"] != "H")
        v_heavy_deg = sum(1 for nb, _ in adj[v] if atoms[nb]["symbol"] != "H")
        if u_heavy_deg > 1 and v_heavy_deg > 1:
            rotatable_bonds += 1

    mw = sum(ATOMIC_WEIGHTS.get(a["symbol"], 12.0) for a in atoms) + total_h * ATOMIC_WEIGHTS["H"]

    return {
        "smiles": s,
        "fragments": fragments,
        "atoms": atoms,
        "bonds": bonds,
        "adj": adj,
        "heavy_atom_count": len(heavy_atoms),
        "ring_count": ring_count,
        "rotatable_bond_count": rotatable_bonds,
        "formal_charge": sum(a["charge"] for a in atoms),
        "molecular_weight": round(mw, 3),
        "assigned_chiral": assigned_chiral,
        "unassigned_chiral": unassigned_chiral,
    }


def standardize_and_desalt_smiles(smiles: str) -> dict:
    """Automatically strip counterions/salts and return the largest organic parent fragment."""
    raw = (smiles or "").strip()
    parts = [p.strip() for p in raw.split(".") if p.strip()]
    if len(parts) <= 1:
        return {
            "original_smiles": raw,
            "desalted_smiles": raw,
            "stripped_counterions": [],
            "fragment_count": len(parts),
            "was_desalted": False,
        }

    if HAS_RDKIT:
        from rdkit import Chem
        scored_parts = []
        stripped = []
        for part in parts:
            mol = Chem.MolFromSmiles(part)
            if mol is None:
                continue
            heavy = mol.GetNumHeavyAtoms()
            has_carbon = any(a.GetSymbol() == "C" for a in mol.GetAtoms())
            scored_parts.append((int(has_carbon), heavy, part))
        if scored_parts:
            scored_parts.sort(key=lambda item: (item[0], item[1]), reverse=True)
            best = scored_parts[0][2]
            for _, _, p in scored_parts[1:]:
                stripped.append(p)
            return {
                "original_smiles": raw,
                "desalted_smiles": best,
                "stripped_counterions": stripped,
                "fragment_count": len(parts),
                "was_desalted": True,
            }

    scored_fallback = []
    for part in parts:
        parsed = parse_smiles_tokens(part)
        has_c = any(a["symbol"] == "C" for a in parsed["atoms"])
        scored_fallback.append((int(has_c), parsed["heavy_atom_count"], part))
    scored_fallback.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_part = scored_fallback[0][2]
    stripped_parts = [item[2] for item in scored_fallback[1:]]
    return {
        "original_smiles": raw,
        "desalted_smiles": best_part,
        "stripped_counterions": stripped_parts,
        "fragment_count": len(parts),
        "was_desalted": True,
    }


def enumerate_ph_protonation_states(smiles: str, ph: float = 7.4, max_states: int = 6) -> list[dict]:
    """Generate dominant protonation microstates around experimental pH using empirical Henderson-Hasselbalch rules."""
    desalted = standardize_and_desalt_smiles(smiles)["desalted_smiles"]
    states: list[dict] = [{
        "label": "parent",
        "smiles": desalted,
        "ph": float(ph),
        "rule": "input_or_desalted_parent",
        "estimated_fraction": 1.0,
    }]
    seen = {desalted}

    if "C(=O)O" in desalted and not desalted.endswith("C(=O)OC"):
        if ph >= 5.0:
            deprot = re.sub(r"C\(=O\)\[O-\]|C\(=O\)O(?![A-Za-z0-9(\[])", "C(=O)[O-]", desalted)
            if deprot not in seen:
                seen.add(deprot)
                states.append({
                    "label": "carboxylate_deprotonated",
                    "smiles": deprot,
                    "ph": float(ph),
                    "rule": "carboxylic_acid_pka_4.2_deprotonation",
                    "estimated_fraction": round(1.0 / (1.0 + 10 ** (4.2 - ph)), 4),
                })
        if ph <= 5.5 and "C(=O)[O-]" in desalted:
            prot = desalted.replace("C(=O)[O-]", "C(=O)O")
            if prot not in seen:
                seen.add(prot)
                states.append({
                    "label": "carboxylic_acid_neutral",
                    "smiles": prot,
                    "ph": float(ph),
                    "rule": "carboxylic_acid_pka_4.2_protonation",
                    "estimated_fraction": round(1.0 / (1.0 + 10 ** (ph - 4.2)), 4),
                })

    if ph <= 9.5 and (desalted.endswith("CN") or desalted.startswith("NC")):
        prot_amine = re.sub(r"N$", "[NH3+]", desalted)
        if prot_amine == desalted and desalted.startswith("NC"):
            prot_amine = "[NH3+]" + desalted[1:]
        if prot_amine not in seen:
            try:
                parse_smiles_tokens(prot_amine)
                seen.add(prot_amine)
                states.append({
                    "label": "ammonium_protonated",
                    "smiles": prot_amine,
                    "ph": float(ph),
                    "rule": "aliphatic_amine_pka_9.8_protonation",
                    "estimated_fraction": round(1.0 / (1.0 + 10 ** (ph - 9.8)), 4),
                })
            except ValueError:
                pass
    elif "[NH3+]" in desalted and ph >= 6.0:
        neutral_amine = desalted.replace("[NH3+]", "N")
        if neutral_amine not in seen:
            seen.add(neutral_amine)
            states.append({
                "label": "amine_neutral_microstate",
                "smiles": neutral_amine,
                "ph": float(ph),
                "rule": "aliphatic_amine_pka_9.8_neutral_microstate",
                "estimated_fraction": round(1.0 / (1.0 + 10 ** (9.8 - ph)), 4),
            })

    if 6.5 <= ph <= 10.0 and ("CS" in desalted and not re.search(r"CS[C(]", desalted)):
        thiolate = re.sub(r"S$", "[S-]", desalted)
        if thiolate not in seen:
            seen.add(thiolate)
            states.append({
                "label": "thiolate_microstate",
                "smiles": thiolate,
                "ph": float(ph),
                "rule": "thiol_pka_8.3_deprotonation",
                "estimated_fraction": round(1.0 / (1.0 + 10 ** (8.3 - ph)), 4),
            })

    if re.search(r"CC=O$", desalted):
        enol = re.sub(r"CC=O$", "C=CO", desalted)
        if enol not in seen:
            seen.add(enol)
            states.append({
                "label": "enol_tautomer_minor",
                "smiles": enol,
                "ph": float(ph),
                "rule": "carbonyl_enol_tautomer",
                "estimated_fraction": 0.01,
            })

    return states[:max_states]


def generate_fallback_3d_coords(parsed: dict, seed: int = 0xA1D0, conformer_index: int = 0) -> list[dict]:
    """Generate realistic 3D atomic coordinates and bond lengths (including C=O 1.22 A) when RDKit is unavailable."""
    rng = np.random.default_rng(seed + conformer_index * 101)
    atoms = parsed["atoms"]
    adj = parsed["adj"]
    coords: dict[int, np.ndarray] = {}

    for start_idx in range(len(atoms)):
        if start_idx in coords:
            continue
        coords[start_idx] = np.array([0.0, 0.0, 0.0], dtype=float) + rng.normal(0.0, 0.02, size=3)
        queue = [start_idx]
        step_counter = 0
        while queue:
            u = queue.pop(0)
            u_pos = coords[u]
            neighbors = adj[u]
            for branch_idx, (v, order) in enumerate(neighbors):
                if v in coords:
                    continue
                step_counter += 1
                u_sym = atoms[u]["symbol"]
                v_sym = atoms[v]["symbol"]
                if {u_sym, v_sym} == {"C", "O"} and order == 2:
                    bond_len = 1.225
                elif order == 2:
                    bond_len = 1.34
                elif order == 3:
                    bond_len = 1.20
                else:
                    bond_len = 1.52 if {u_sym, v_sym} == {"C"} else 1.43

                torsion = (step_counter * 1.0472) + (branch_idx * 2.0944) + (conformer_index * 0.55)
                tilt = 0.615 if order == 1 else 0.523
                direction = np.array([
                    math.cos(tilt),
                    math.sin(tilt) * math.cos(torsion),
                    math.sin(tilt) * math.sin(torsion),
                ], dtype=float)
                direction /= np.linalg.norm(direction)
                coords[v] = u_pos + direction * bond_len
                queue.append(v)

    atom_records = []
    for idx, a in enumerate(atoms):
        pos = coords.get(idx, np.zeros(3, dtype=float))
        sym = a["symbol"]
        if sym == "O":
            ad_type = "OA"
        elif sym == "N":
            ad_type = "NA" if a["implicit_h"] == 0 else "N"
        elif sym == "S":
            ad_type = "SA" if a["implicit_h"] == 0 else "S"
        elif sym == "C" and a["aromatic"]:
            ad_type = "A"
        else:
            ad_type = sym
        atom_records.append({
            "serial": idx + 1,
            "name": f"{sym}{idx + 1}"[:4],
            "symbol": sym,
            "ad_type": ad_type,
            "coord": pos,
            "charge": float(a["charge"]),
        })
    return atom_records


def write_fallback_sdf(name: str, atom_records: list[dict], bonds: list[tuple[int, int, int, bool]], sdf_path: Path) -> Path:
    sdf_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        name,
        "  ALDH_Docking_Pipeline_3D",
        "",
        f"{len(atom_records):3d}{len(bonds):3d}  0  0  0  0  0  0  0  0999 V2000",
    ]
    for rec in atom_records:
        x, y, z = rec["coord"]
        lines.append(f"{x:10.4f}{y:10.4f}{z:10.4f} {rec['symbol']:<3s} 0  0  0  0  0  0  0  0  0  0  0  0")
    for u, v, order, _ in bonds:
        lines.append(f"{u + 1:3d}{v + 1:3d}{order:3d}  0  0  0  0")
    lines.append("M  END")
    lines.append("$$$$")
    sdf_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sdf_path


def write_fallback_pdbqt(name: str, atom_records: list[dict], pdbqt_path: Path) -> Path:
    pdbqt_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"REMARK  Name = {name}",
        "ROOT",
    ]
    for rec in atom_records:
        x, y, z = rec["coord"]
        lines.append(
            f"HETATM{rec['serial']:5d} {rec['name']:>4s} UNL     1    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00    {rec['charge']:+6.3f} {rec['ad_type']:<2s}"
        )
    lines.extend([
        "ENDROOT",
        "TORSDOF 0",
    ])
    pdbqt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return pdbqt_path
