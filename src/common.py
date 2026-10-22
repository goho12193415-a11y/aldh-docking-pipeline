from __future__ import annotations

import itertools
import logging
import math
import re
import shutil
import site
import sys
import sysconfig
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

ONE_LETTER_AA = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U",
}

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

# Gasteiger-Marsili PEOE parameters (a_i, b_i, c_i) in eV for orbital electronegativity chi = a + b*q + c*q^2
PEOE_PARAMS = {
    "H": (7.17, 6.24, -0.56),
    "C_sp3": (7.98, 9.18, 1.88),
    "C_sp2": (8.79, 9.32, 1.51),
    "N_sp3": (11.54, 10.82, 1.36),
    "N_sp2": (12.87, 11.15, 0.85),
    "O_sp3": (14.18, 12.92, 1.39),
    "O_sp2": (17.07, 13.79, 0.47),
    "S": (10.14, 9.13, 1.38),
    "P": (8.90, 8.24, 0.96),
    "F": (14.66, 13.85, 2.31),
    "CL": (11.00, 9.69, 1.35),
    "BR": (10.08, 8.47, 1.16),
    "I": (9.90, 7.96, 0.96),
}

MODEL_PKA = {
    "ASP": 3.80,
    "GLU": 4.50,
    "HIS": 6.50,
    "CYS": 8.30,
    "TYR": 10.00,
    "LYS": 10.50,
    "ARG": 12.50,
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


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def require_executable(name_or_path: str) -> str:
    resolved = shutil.which(name_or_path)
    if resolved:
        return resolved
    path = Path(name_or_path)
    if path.exists():
        return str(path)
    if not path.is_absolute():
        candidates = [
            Path(sysconfig.get_path("scripts")) / name_or_path,
            Path(site.getuserbase()) / "Scripts" / name_or_path,
            Path(sys.executable).parent / name_or_path,
            Path(sys.executable).parent / "Scripts" / name_or_path,
        ]
        candidates.extend(
            base / "Scripts" / name_or_path
            for base in Path(site.getuserbase()).glob("Python*")
        )
        for candidate in candidates:
            for actual in (candidate, candidate.with_suffix(".exe")):
                if actual.exists():
                    return str(actual)
    raise FileNotFoundError(
        f"Required executable was not found: {name_or_path}. "
        "Install the dependency or update pipeline_config.yaml."
    )


def atom_dict(line: str) -> dict[str, Any]:
    """Parse ATOM/HETATM line from either PDB or PDBQT (extracting real partial charge when present)."""
    element = line[76:78].strip() if len(line) >= 78 else ""
    charge: float | None = None
    if len(line) >= 76:
        q_field = line[66:76].strip()
        if q_field:
            try:
                charge = float(q_field)
            except ValueError:
                charge = None
    raw_name = line[12:16].strip()
    inferred_el = element or (raw_name[:1] if raw_name else "C")
    if inferred_el.upper() in {"OA", "NA", "SA", "HD"}:
        inferred_el = {"OA": "O", "NA": "N", "SA": "S", "HD": "H"}[inferred_el.upper()]
    return {
        "record": line[0:6].strip(),
        "serial": int(line[6:11].strip()),
        "name": raw_name,
        "resname": line[17:20].strip(),
        "chain": line[21].strip(),
        "resseq": int(line[22:26].strip()),
        "coord": np.array(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])],
            dtype=float,
        ),
        "charge": charge,
        "element": inferred_el,
        "line": line.rstrip("\n"),
    }


def compute_peoe_charges(atoms: list[dict], adj: dict[int, list[tuple[int, int]]], iterations: int = 6) -> list[float]:
    """Iterative Gasteiger-Marsili Partial Equalization of Orbital Electronegativity (PEOE) topological charge solver."""
    charges = [float(a.get("charge", 0.0)) for a in atoms]
    params = []
    for a in atoms:
        idx = a["index"]
        sym = a["symbol"].upper()
        max_order = max((order for _, order in adj.get(idx, [])), default=1)
        is_sp2 = a.get("aromatic", False) or max_order >= 2
        if sym in {"C", "N", "O"}:
            key = f"{sym}_{'sp2' if is_sp2 else 'sp3'}"
            params.append(PEOE_PARAMS[key])
        else:
            params.append(PEOE_PARAMS.get(sym, (8.5, 8.5, 1.0)))

    damping = 0.5
    for _ in range(iterations):
        chi = [
            a_p + b_p * q + c_p * q * q
            for (a_p, b_p, c_p), q in zip(params, charges)
        ]
        dq = [0.0] * len(atoms)
        for u, neighbors in adj.items():
            for v, _ in neighbors:
                if u >= v:
                    continue
                if chi[u] > chi[v]:
                    denom = sum(params[v]) if sum(params[v]) > 1e-3 else 20.0
                    shift = damping * (chi[u] - chi[v]) / denom
                else:
                    denom = sum(params[u]) if sum(params[u]) > 1e-3 else 20.0
                    shift = damping * (chi[u] - chi[v]) / denom
                dq[u] -= shift
                dq[v] += shift
        for idx in range(len(charges)):
            charges[idx] = round(charges[idx] + dq[idx], 4)
        damping *= 0.5
    return charges


class FallbackAtom:
    def __init__(
        self,
        name: str,
        coord: np.ndarray,
        bfactor: float = 0.0,
        occupancy: float = 1.0,
        altloc: str = " ",
        fullname: str | None = None,
        serial_number: int = 1,
        element: str = "",
        record: str = "ATOM",
    ) -> None:
        self.name = name.strip()
        self.fullname = fullname if fullname is not None else f" {self.name:<3s}"[:4]
        self.coord = np.array(coord, dtype=float)
        self.bfactor = float(bfactor)
        self.occupancy = float(occupancy)
        self.altloc = altloc if altloc else " "
        self.serial_number = int(serial_number)
        self.element = (element or self.name[:1]).strip().upper()
        self.record = record.strip() or "ATOM"
        self.parent: FallbackResidue | None = None

    def get_name(self) -> str:
        return self.name

    def get_coord(self) -> np.ndarray:
        return self.coord

    def get_bfactor(self) -> float:
        return self.bfactor

    def get_occupancy(self) -> float:
        return self.occupancy

    def get_altloc(self) -> str:
        return self.altloc

    def set_altloc(self, value: str) -> None:
        self.altloc = value if value else " "

    def set_occupancy(self, value: float) -> None:
        self.occupancy = float(value)

    def get_parent(self) -> FallbackResidue | None:
        return self.parent

    def transform(self, rot: np.ndarray, tran: np.ndarray) -> None:
        self.coord = np.dot(self.coord, rot) + tran

    def copy(self) -> FallbackAtom:
        return FallbackAtom(
            name=self.name,
            coord=self.coord.copy(),
            bfactor=self.bfactor,
            occupancy=self.occupancy,
            altloc=self.altloc,
            fullname=self.fullname,
            serial_number=self.serial_number,
            element=self.element,
            record=self.record,
        )


class FallbackResidue:
    def __init__(self, res_id: tuple[str, int, str], resname: str, segid: str = "    ") -> None:
        self.id = (str(res_id[0]), int(res_id[1]), str(res_id[2]))
        self.resname = resname.strip()
        self.segid = segid
        self._atoms: list[FallbackAtom] = []
        self.parent: FallbackChain | None = None

    def add(self, atom: FallbackAtom) -> None:
        atom.parent = self
        self._atoms.append(atom)

    def get_resname(self) -> str:
        return self.resname

    def get_parent(self) -> FallbackChain | None:
        return self.parent

    def get_atoms(self) -> Iterator[FallbackAtom]:
        return iter(self._atoms)

    def get_unpacked_list(self) -> list[FallbackAtom]:
        return list(self._atoms)

    def __iter__(self) -> Iterator[FallbackAtom]:
        return iter(self._atoms)

    def __contains__(self, atom_name: str) -> bool:
        target = atom_name.strip()
        return any(atom.name == target for atom in self._atoms)

    def __getitem__(self, atom_name: str) -> FallbackAtom:
        target = atom_name.strip()
        selected = [atom for atom in self._atoms if atom.name == target]
        if not selected:
            raise KeyError(atom_name)
        selected.sort(
            key=lambda a: (
                0 if a.altloc in {" ", "", "A"} else 1,
                -a.occupancy,
                a.altloc,
            )
        )
        return selected[0]

    def copy(self) -> FallbackResidue:
        cloned = FallbackResidue(self.id, self.resname, self.segid)
        for atom in self._atoms:
            cloned.add(atom.copy())
        return cloned


class FallbackChain:
    def __init__(self, chain_id: str) -> None:
        self.id = chain_id
        self._residues: list[FallbackResidue] = []
        self._index: dict[tuple[str, int, str], FallbackResidue] = {}
        self.parent: FallbackModel | None = None

    def add(self, residue: FallbackResidue) -> None:
        residue.parent = self
        if residue.id in self._index:
            existing = self._index[residue.id]
            for atom in residue:
                existing.add(atom)
            return
        self._residues.append(residue)
        self._index[residue.id] = residue

    def get_atoms(self) -> Iterator[FallbackAtom]:
        for residue in self._residues:
            for atom in residue:
                yield atom

    def get_residues(self) -> Iterator[FallbackResidue]:
        return iter(self._residues)

    def __iter__(self) -> Iterator[FallbackResidue]:
        return iter(self._residues)

    def __contains__(self, res_id: tuple[str, int, str] | int) -> bool:
        if isinstance(res_id, int):
            return (" ", res_id, " ") in self._index
        return res_id in self._index

    def __getitem__(self, res_id: tuple[str, int, str] | int) -> FallbackResidue:
        if isinstance(res_id, int):
            res_id = (" ", res_id, " ")
        if res_id not in self._index:
            raise KeyError(res_id)
        return self._index[res_id]


class FallbackModel:
    def __init__(self, serial: int = 0) -> None:
        self.id = serial
        self._chains: list[FallbackChain] = []
        self._index: dict[str, FallbackChain] = {}
        self.parent: FallbackStructure | None = None

    def add(self, chain: FallbackChain) -> None:
        chain.parent = self
        self._chains.append(chain)
        self._index[chain.id] = chain

    def get_chains(self) -> Iterator[FallbackChain]:
        return iter(self._chains)

    def get_atoms(self) -> Iterator[FallbackAtom]:
        for chain in self._chains:
            yield from chain.get_atoms()

    def __iter__(self) -> Iterator[FallbackChain]:
        return iter(self._chains)

    def __contains__(self, chain_id: str) -> bool:
        return chain_id in self._index

    def __getitem__(self, chain_id: str) -> FallbackChain:
        if chain_id not in self._index:
            raise KeyError(chain_id)
        return self._index[chain_id]


class FallbackStructure:
    def __init__(self, structure_id: str) -> None:
        self.id = structure_id
        self._models: list[FallbackModel] = []

    def add(self, model: FallbackModel) -> None:
        model.parent = self
        self._models.append(model)

    def get_atoms(self) -> Iterator[FallbackAtom]:
        for model in self._models:
            yield from model.get_atoms()

    def __iter__(self) -> Iterator[FallbackModel]:
        return iter(self._models)

    def __getitem__(self, index: int) -> FallbackModel:
        return self._models[index]


class FallbackSelect:
    def accept_model(self, model) -> int:
        return 1

    def accept_chain(self, chain) -> int:
        return 1

    def accept_residue(self, residue) -> int:
        return 1

    def accept_atom(self, atom) -> int:
        return 1


class FallbackPDBParser:
    def __init__(self, QUIET: bool = True, PERMISSIVE: bool = True) -> None:
        self.quiet = QUIET
        self.permissive = PERMISSIVE

    def get_structure(self, structure_id: str, file_path: str | Path) -> FallbackStructure:
        path = Path(file_path)
        text = path.read_text(encoding="utf-8", errors="replace")
        structure = FallbackStructure(structure_id)
        current_model: FallbackModel | None = None
        model_serial = 0

        for line in text.splitlines():
            record = line[0:6].strip()
            if record == "MODEL":
                current_model = FallbackModel(model_serial)
                structure.add(current_model)
                model_serial += 1
            elif record in {"ATOM", "HETATM"}:
                if current_model is None:
                    current_model = FallbackModel(0)
                    structure.add(current_model)
                try:
                    serial = int(line[6:11].strip() or "1")
                except ValueError:
                    serial = 1
                fullname = line[12:16]
                name = fullname.strip()
                altloc = line[16:17] if len(line) > 16 else " "
                resname = line[17:20].strip()
                chain_id = line[21:22].strip() or "A"
                try:
                    resseq = int(line[22:26].strip())
                except ValueError:
                    continue
                icode = line[26:27] if len(line) > 26 else " "
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    continue
                try:
                    occupancy = float(line[54:60].strip()) if len(line) >= 60 and line[54:60].strip() else 1.0
                except ValueError:
                    occupancy = 1.0
                try:
                    bfactor = float(line[60:66].strip()) if len(line) >= 66 and line[60:66].strip() else 0.0
                except ValueError:
                    bfactor = 0.0
                element = line[76:78].strip() if len(line) >= 78 else ""
                if not element:
                    stripped = "".join(ch for ch in name if ch.isalpha())
                    element = stripped[:1].upper() if stripped else "C"
                hetflag = " " if record == "ATOM" else ("W" if resname in {"HOH", "WAT"} else f"H_{resname}")
                res_id = (hetflag, resseq, icode if icode else " ")

                if chain_id not in current_model:
                    current_model.add(FallbackChain(chain_id))
                chain = current_model[chain_id]
                if res_id not in chain:
                    chain.add(FallbackResidue(res_id, resname))
                residue = chain[res_id]
                residue.add(
                    FallbackAtom(
                        name=name,
                        coord=np.array([x, y, z], dtype=float),
                        bfactor=bfactor,
                        occupancy=occupancy,
                        altloc=altloc,
                        fullname=fullname,
                        serial_number=serial,
                        element=element,
                        record=record,
                    )
                )
        if not structure._models:
            structure.add(FallbackModel(0))
        return structure


class FallbackPDBIO:
    def __init__(self) -> None:
        self.structure: FallbackStructure | None = None

    def set_structure(self, structure: FallbackStructure) -> None:
        self.structure = structure

    def save(self, file_path: str | Path, select: FallbackSelect | None = None) -> None:
        if self.structure is None:
            raise ValueError("No structure set on PDBIO.")
        selector = select or FallbackSelect()
        lines: list[str] = []
        serial = 1
        for model in self.structure:
            if hasattr(selector, "accept_model") and not selector.accept_model(model):
                continue
            for chain in model:
                if hasattr(selector, "accept_chain") and not selector.accept_chain(chain):
                    continue
                for residue in chain:
                    if hasattr(selector, "accept_residue") and not selector.accept_residue(residue):
                        continue
                    record = "ATOM" if residue.id[0] == " " else "HETATM"
                    for atom in residue:
                        if hasattr(selector, "accept_atom") and not selector.accept_atom(atom):
                            continue
                        atom_name = atom.get_name()
                        element = (getattr(atom, "element", "") or atom_name[:1]).strip().upper()
                        if len(atom_name) < 4 and len(element) == 1:
                            formatted_name = f" {atom_name:<3s}"
                        else:
                            formatted_name = f"{atom_name:<4s}"[:4]
                        altloc = atom.get_altloc() if atom.get_altloc() else " "
                        icode = residue.id[2] if len(residue.id) > 2 and residue.id[2] else " "
                        x, y, z = [float(v) for v in atom.get_coord()]
                        occ = float(atom.get_occupancy()) if atom.get_occupancy() is not None else 1.0
                        bfac = float(atom.get_bfactor()) if atom.get_bfactor() is not None else 0.0
                        lines.append(
                            f"{record:<6s}{serial:5d} {formatted_name}{altloc:1s}{residue.get_resname():>3s} "
                            f"{chain.id:1s}{int(residue.id[1]):4d}{icode:1s}   "
                            f"{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{bfac:6.2f}          {element:>2s}  "
                        )
                        serial += 1
        lines.append("END")
        Path(file_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


class FallbackSuperimposer:
    def __init__(self) -> None:
        self.rotran: tuple[np.ndarray, np.ndarray] | None = None
        self.rms: float = 0.0

    def set_atoms(self, fixed: Iterable[FallbackAtom], moving: Iterable[FallbackAtom]) -> None:
        fixed_coords = np.array([atom.get_coord() for atom in fixed], dtype=float)
        moving_coords = np.array([atom.get_coord() for atom in moving], dtype=float)
        if len(fixed_coords) != len(moving_coords) or len(fixed_coords) == 0:
            raise ValueError("Fixed and moving atom lists must have the same non-zero length.")
        fixed_center = np.mean(fixed_coords, axis=0)
        moving_center = np.mean(moving_coords, axis=0)
        f_centered = fixed_coords - fixed_center
        m_centered = moving_coords - moving_center
        cov = np.dot(np.transpose(m_centered), f_centered)
        u, _, vt = np.linalg.svd(cov)
        d = np.sign(np.linalg.det(np.dot(u, vt)))
        diag = np.diag([1.0, 1.0, d])
        rot = np.dot(u, np.dot(diag, vt))
        tran = fixed_center - np.dot(moving_center, rot)
        transformed = np.dot(moving_coords, rot) + tran
        diff = fixed_coords - transformed
        self.rms = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
        self.rotran = (rot, tran)

    def apply(self, entity) -> None:
        if self.rotran is None:
            raise RuntimeError("Superimposer.set_atoms must be called before apply.")
        rot, tran = self.rotran
        if hasattr(entity, "get_atoms"):
            for atom in entity.get_atoms():
                atom.transform(rot, tran)
        elif hasattr(entity, "__iter__"):
            for atom in entity:
                atom.transform(rot, tran)
        elif hasattr(entity, "transform"):
            entity.transform(rot, tran)


try:
    from Bio.PDB import PDBIO, PDBParser, Select, Superimposer  # type: ignore
    USING_BIOPYTHON = True
except ImportError:
    PDBParser = FallbackPDBParser  # type: ignore
    PDBIO = FallbackPDBIO  # type: ignore
    Select = FallbackSelect  # type: ignore
    Superimposer = FallbackSuperimposer  # type: ignore
    USING_BIOPYTHON = False


def align_ca_residues(target_residues: list, template_residues: list) -> list[tuple[object, object]]:
    """Align two lists of standard amino-acid residues with CA atoms using Needleman-Wunsch."""
    target_seq = [ONE_LETTER_AA.get(r.get_resname().strip(), "X") for r in target_residues]
    template_seq = [ONE_LETTER_AA.get(r.get_resname().strip(), "X") for r in template_residues]
    n, m = len(target_seq), len(template_seq)
    if n == 0 or m == 0:
        return []
    match_score = 3
    mismatch_score = -1
    gap_penalty = -2
    dp = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(1, n + 1):
        dp[i, 0] = i * gap_penalty
    for j in range(1, m + 1):
        dp[0, j] = j * gap_penalty
    for i in range(1, n + 1):
        a = target_seq[i - 1]
        for j in range(1, m + 1):
            b = template_seq[j - 1]
            score = match_score if a == b and a != "X" else mismatch_score
            dp[i, j] = max(
                dp[i - 1, j - 1] + score,
                dp[i - 1, j] + gap_penalty,
                dp[i, j - 1] + gap_penalty,
            )
    i, j = n, m
    pairs = []
    while i > 0 and j > 0:
        a, b = target_seq[i - 1], template_seq[j - 1]
        score = match_score if a == b and a != "X" else mismatch_score
        if dp[i, j] == dp[i - 1, j - 1] + score:
            pairs.append((target_residues[i - 1], template_residues[j - 1]))
            i -= 1
            j -= 1
        elif dp[i, j] == dp[i - 1, j] + gap_penalty:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    identical_or_aligned = [
        (r1, r2) for r1, r2 in pairs
        if ONE_LETTER_AA.get(r1.get_resname().strip(), "X") == ONE_LETTER_AA.get(r2.get_resname().strip(), "Y")
    ]
    if len(identical_or_aligned) >= 10:
        return identical_or_aligned
    return pairs


def estimate_receptor_microenvironment_pka(
    pdb_path: Path,
    chain_id: str = "A",
    ph: float = 7.4,
    catalytic_resseq: int | None = None,
) -> dict:
    """Estimate 3D microenvironment pKa shifts (PROPKA-inspired empirical model) for active-site ionizable residues."""
    structure = PDBParser(QUIET=True).get_structure("pka_audit", str(pdb_path))
    if chain_id not in structure[0]:
        chain_id = next(iter(structure[0])).id
    chain = structure[0][chain_id]

    all_heavy = []
    for model in structure:
        for ch in model:
            for r in ch:
                if r.get_resname().strip() in {"HOH", "WAT"}:
                    continue
                for a in r:
                    elem = (getattr(a, "element", "") or a.get_name()[:1]).strip().upper()
                    if elem != "H":
                        all_heavy.append((ch.id, r.get_resname().strip(), int(r.id[1]), a.get_name().strip(), elem, np.array(a.get_coord(), dtype=float)))

    cat_coord = None
    if catalytic_resseq is not None and (" ", int(catalytic_resseq), " ") in chain:
        cat_res = chain[(" ", int(catalytic_resseq), " ")]
        for aname in ("SG", "OG", "NZ", "OD1", "OE1", "CA"):
            if aname in cat_res:
                cat_coord = np.array(cat_res[aname].get_coord(), dtype=float)
                break

    key_atom_map = {
        "ASP": "OD1", "GLU": "OE1", "HIS": "NE2", "CYS": "SG", "TYR": "OH", "LYS": "NZ", "ARG": "CZ",
    }
    acidic_types = {"ASP", "GLU", "CYS", "TYR"}
    records = []

    for r in chain:
        if r.id[0] != " ":
            continue
        rname = r.get_resname().strip()
        if rname not in MODEL_PKA:
            continue
        resseq = int(r.id[1])
        target_aname = key_atom_map[rname]
        if target_aname not in r:
            continue
        coord = np.array(r[target_aname].get_coord(), dtype=float)
        if cat_coord is not None and distance(coord, cat_coord) > 12.0 and resseq != catalytic_resseq:
            continue

        neighbors_6a = [item for item in all_heavy if not (item[0] == chain_id and item[2] == resseq) and distance(item[5], coord) <= 6.0]
        burial_fraction = min(1.0, max(0.0, (len(neighbors_6a) - 12.0) / 28.0))

        hbond_neighbors = [
            item for item in neighbors_6a
            if item[4] in {"N", "O", "S"} and distance(item[5], coord) <= 3.6
        ]

        pos_neighbors = [
            item for item in neighbors_6a
            if (
                item[1] in {"NAD", "NAP", "NDP", "ZN", "MG", "FE", "MN", "CA", "CU", "LYS", "ARG", "HIS"}
                and distance(item[5], coord) <= 5.5
            )
        ]

        pka_model = MODEL_PKA[rname]
        delta_hbond = -0.55 * len(hbond_neighbors) if rname in acidic_types else +0.40 * len(hbond_neighbors)
        delta_elec = -0.65 * min(3, len(pos_neighbors)) if rname in acidic_types else -0.35 * min(2, len(pos_neighbors))
        if rname == "CYS" and resseq == catalytic_resseq:
            delta_elec -= 1.10
        delta_desolv = +0.80 * burial_fraction if rname in {"ASP", "GLU"} else (-0.40 * burial_fraction if rname == "CYS" else 0.0)
        pka_pred = round(pka_model + delta_hbond + delta_elec + delta_desolv, 2)

        if rname in acidic_types:
            deprot_frac = round(1.0 / (1.0 + 10.0 ** (pka_pred - ph)), 4)
            assigned_state = f"{rname}_deprotonated_anion" if deprot_frac >= 0.5 else f"{rname}_neutral"
            if rname == "CYS":
                assigned_state = "CYM_thiolate(-1)" if deprot_frac >= 0.5 else "CYS_thiol(0)"
        else:
            prot_frac = round(1.0 / (1.0 + 10.0 ** (ph - pka_pred)), 4)
            if rname == "HIS":
                if prot_frac >= 0.5:
                    assigned_state = "HIP_protonated(+1)"
                else:
                    nd1_hb = sum(1 for item in hbond_neighbors if "ND1" in r and distance(item[5], np.array(r["ND1"].get_coord(), dtype=float)) <= 3.5)
                    assigned_state = "HID_neutral" if nd1_hb > 0 else "HIE_neutral"
            else:
                assigned_state = f"{rname}_protonated(+1)" if prot_frac >= 0.5 else f"{rname}_neutral(0)"

        records.append({
            "chain": chain_id,
            "residue": rname,
            "number": resseq,
            "is_catalytic": resseq == catalytic_resseq,
            "model_pka": pka_model,
            "predicted_pka": pka_pred,
            "burial_fraction": round(burial_fraction, 3),
            "hbond_neighbor_count": len(hbond_neighbors),
            "electrostatic_stabilizer_count": len(pos_neighbors),
            "assigned_state_at_ph": assigned_state,
            "experimental_ph": float(ph),
        })

    return {
        "pdb": str(pdb_path),
        "chain": chain_id,
        "experimental_ph": float(ph),
        "active_site_residue_count": len(records),
        "residues": records,
    }


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


def _prot_penalty_kcal(fraction: float) -> float:
    f = max(1e-6, min(1.0, float(fraction)))
    return round(-0.592 * math.log(f), 3)


def enumerate_ph_protonation_states(smiles: str, ph: float = 7.4, max_states: int = 8) -> list[dict]:
    """Generate single-site and coupled multi-site (zwitterionic/multiprotic) protonation microstates with Boltzmann penalties."""
    desalted = standardize_and_desalt_smiles(smiles)["desalted_smiles"]
    states: list[dict] = [{
        "label": "parent",
        "smiles": desalted,
        "ph": float(ph),
        "rule": "input_or_desalted_parent",
        "estimated_fraction": 1.0,
        "protonation_penalty_kcal_mol": 0.0,
    }]
    seen = {desalted}

    # Build per-site transformation options for combinatorial coupled microstate enumeration
    site_transforms: list[list[tuple[str, str, str, float]]] = []

    has_ewg = any(hal in desalted for hal in ("F", "Cl", "Br", "[N+](=O)[O-]"))
    pka_cooh = 2.8 if has_ewg else 4.2
    frac_coo_neg = round(1.0 / (1.0 + 10 ** (pka_cooh - ph)), 4)
    frac_cooh_neu = round(1.0 - frac_coo_neg, 4)
    if "C(=O)O" in desalted and not desalted.endswith("C(=O)OC"):
        site_transforms.append([
            ("cooh_neutral", "C(=O)O", "C(=O)O", frac_cooh_neu),
            ("carboxylate_deprotonated", "C(=O)O", "C(=O)[O-]", frac_coo_neg),
        ])
    elif "C(=O)[O-]" in desalted:
        site_transforms.append([
            ("carboxylate_deprotonated", "C(=O)[O-]", "C(=O)[O-]", frac_coo_neg),
            ("carboxylic_acid_neutral", "C(=O)[O-]", "C(=O)O", frac_cooh_neu),
        ])

    frac_nh3_pos = round(1.0 / (1.0 + 10 ** (ph - 9.8)), 4)
    frac_nh2_neu = round(1.0 - frac_nh3_pos, 4)
    if "[NH3+]" in desalted:
        site_transforms.append([
            ("ammonium_protonated", "[NH3+]", "[NH3+]", frac_nh3_pos),
            ("amine_neutral_microstate", "[NH3+]", "N", frac_nh2_neu),
        ])
    elif desalted.startswith("NC") or desalted.endswith("CN"):
        target_tok = "NC" if desalted.startswith("NC") else "CN"
        repl_tok = "[NH3+]C" if desalted.startswith("NC") else "C[NH3+]"
        site_transforms.append([
            ("amine_neutral_microstate", target_tok, target_tok, frac_nh2_neu),
            ("ammonium_protonated", target_tok, repl_tok, frac_nh3_pos),
        ])

    if "C(=O)NO" in desalted:
        frac_hyd = round(1.0 / (1.0 + 10 ** (8.8 - ph)), 4)
        site_transforms.append([
            ("hydroxamic_neutral", "C(=O)NO", "C(=O)NO", round(1.0 - frac_hyd, 4)),
            ("hydroxamate_chelator_anion", "C(=O)NO", "C(=O)N[O-]", frac_hyd),
        ])

    if 6.5 <= ph <= 10.0 and ("CS" in desalted and not re.search(r"CS[C(]", desalted)):
        frac_thiolate = round(1.0 / (1.0 + 10 ** (8.3 - ph)), 4)
        site_transforms.append([
            ("thiol_neutral", "CS", "CS", round(1.0 - frac_thiolate, 4)),
            ("thiolate_microstate", "CS", "C[S-]", frac_thiolate),
        ])

    if site_transforms:
        for combo in itertools.product(*site_transforms):
            cand = desalted
            labels = []
            joint_prob = 1.0
            for lbl, src_pat, dst_pat, prob in combo:
                cand = cand.replace(src_pat, dst_pat, 1)
                labels.append(lbl)
                joint_prob *= prob
            joint_prob = round(joint_prob, 4)
            if cand not in seen and joint_prob >= 1e-4:
                try:
                    parse_smiles_tokens(cand)
                    seen.add(cand)
                    combined_label = "__".join(l for l in labels if not l.endswith("_neutral") or len(labels) == 1) or "neutral_microstate"
                    states.append({
                        "label": combined_label,
                        "smiles": cand,
                        "ph": float(ph),
                        "rule": "coupled_henderson_hasselbalch_microstate",
                        "estimated_fraction": joint_prob,
                        "protonation_penalty_kcal_mol": _prot_penalty_kcal(joint_prob),
                    })
                except ValueError:
                    pass

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
                "protonation_penalty_kcal_mol": _prot_penalty_kcal(0.01),
            })

    return states[:max_states]


def generate_fallback_3d_coords(parsed: dict, seed: int = 0xA1D0, conformer_index: int = 0) -> list[dict]:
    """Generate realistic 3D atomic coordinates with Gasteiger-Marsili PEOE partial charges when RDKit is unavailable."""
    rng = np.random.default_rng(seed + conformer_index * 101)
    atoms = parsed["atoms"]
    adj = parsed["adj"]
    peoe_q = compute_peoe_charges(atoms, adj, iterations=6)
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
            "charge": float(peoe_q[idx]),
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
