from __future__ import annotations

import csv
import json
import math
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common import atom_dict, distance
from src.mechanism_templates import list_supported_mechanisms


def _canvas_sketch_line(
    canvas: tk.Canvas,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str = "#2B2520",
    width: int = 2,
    seed: int = 1,
    dash: tuple[int, ...] | None = None,
) -> None:
    """Draw a hand-sketched fountain-pen stroke using two slightly jittered passes."""
    mx = (x1 + x2) * 0.5 + math.sin(seed * 7.13) * 1.5
    my = (y1 + y2) * 0.5 + math.cos(seed * 5.31) * 1.5
    kwargs = {"fill": color, "width": width, "smooth": True, "capstyle": "round"}
    if dash:
        kwargs["dash"] = dash
    canvas.create_line(x1, y1, mx, my, x2, y2, **kwargs)
    if not dash:
        canvas.create_line(
            x1 + 0.6, y1 - 0.5,
            mx - 0.8, my + 0.7,
            x2 - 0.5, y2 + 0.5,
            fill=color, width=max(1, width - 1), smooth=True, capstyle="round",
        )


def _canvas_sketch_box(
    canvas: tk.Canvas,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    fill: str = "#FBF8F1",
    outline: str = "#2B2520",
    width: int = 2,
    seed: int = 10,
) -> None:
    """Draw a hand-illustrated notebook card with subtle corner over-strokes."""
    canvas.create_rectangle(x1, y1, x2, y2, fill=fill, outline="")
    _canvas_sketch_line(canvas, x1 - 2, y1, x2 + 2, y1, outline, width, seed + 1)
    _canvas_sketch_line(canvas, x2, y1 - 2, x2, y2 + 2, outline, width, seed + 2)
    _canvas_sketch_line(canvas, x2 + 2, y2, x1 - 2, y2, outline, width, seed + 3)
    _canvas_sketch_line(canvas, x1, y2 + 2, x1, y1 - 2, outline, width, seed + 4)


class Interactive3DPocketWindow(tk.Toplevel):
    """Hand-drawn parchment-and-ink 3D molecular active-site sketchbook viewer (钢笔淡彩 / 炭铅素描三维速写台)."""

    def __init__(self, master: tk.Misc, complex_pdb: Path, box_audit_json: Path | None, catalytic_num: int = 302) -> None:
        super().__init__(master)
        self.title(f"活性口袋三维钢笔速写稿 — {complex_pdb.name}")
        self.geometry("960x720")
        self.configure(bg="#F4EFE0")

        self.yaw = 0.45
        self.pitch = 0.30
        self.zoom = 26.0
        self.last_mouse = (0, 0)
        self.show_box_var = tk.BooleanVar(value=True)
        self.show_pocket_var = tk.BooleanVar(value=True)
        self.show_hbonds_var = tk.BooleanVar(value=True)
        self.wash_mode_var = tk.BooleanVar(value=True)

        self.lig_atoms: list[dict] = []
        self.cat_atoms: list[dict] = []
        self.cof_atoms: list[dict] = []
        self.pocket_atoms: list[dict] = []
        self.box_info: dict | None = None
        self.center = np.zeros(3, dtype=float)

        self._load_structure(complex_pdb, box_audit_json, catalytic_num)
        self._build_controls()
        self.canvas = tk.Canvas(self, bg="#F4EFE0", highlightthickness=0, cursor="fleur")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Button-4>", lambda _e: self._zoom_by(1.12))
        self.canvas.bind("<Button-5>", lambda _e: self._zoom_by(0.88))
        self.canvas.bind("<Configure>", lambda _e: self._redraw())
        self.after(50, self._redraw)

    def _load_structure(self, complex_pdb: Path, box_audit_json: Path | None, catalytic_num: int) -> None:
        all_receptor = []
        for line in complex_pdb.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.startswith(("ATOM", "HETATM")):
                continue
            a = atom_dict(line)
            if a["element"].upper() == "H":
                continue
            if a["resname"] == "LIG":
                self.lig_atoms.append(a)
            elif a["resname"] in {"NAD", "NAP", "NDP", "FAD", "FMN", "HEM", "HEC", "ZN", "MG", "FE", "MN"}:
                self.cof_atoms.append(a)
            elif a["resseq"] == catalytic_num:
                self.cat_atoms.append(a)
            else:
                all_receptor.append(a)

        if self.lig_atoms:
            self.center = np.mean([a["coord"] for a in self.lig_atoms], axis=0)
        elif self.cat_atoms:
            self.center = np.mean([a["coord"] for a in self.cat_atoms], axis=0)

        self.pocket_atoms = [
            a for a in all_receptor
            if distance(a["coord"], self.center) <= 6.2 and a["name"] in {"CA", "CB", "CG", "CD", "ND2", "OD1", "OE1", "SG", "OG", "NZ", "OH"}
        ]
        self.cof_atoms = [a for a in self.cof_atoms if distance(a["coord"], self.center) <= 10.5]

        if box_audit_json and box_audit_json.is_file():
            try:
                data = json.loads(box_audit_json.read_text(encoding="utf-8"))
                self.box_info = data.get("box")
            except Exception:
                self.box_info = None

    def _build_controls(self) -> None:
        bar = tk.Frame(self, bg="#EAE1CD", padx=14, pady=8, highlightbackground="#2B2520", highlightthickness=1)
        bar.pack(fill="x")
        tk.Label(
            bar,
            text="按住左键旋转速写视角 · 滚轮缩放  |  靛蓝: 配体(LIG)   朱砂: 催化残基   赭石: 辅因子/金属   炭铅: 口袋骨架",
            bg="#EAE1CD",
            fg="#2B2520",
            font=("Georgia", 10, "bold"),
        ).pack(side="left")
        tk.Button(bar, text="归位视角", command=self._reset_view, bg="#FBF8F1", fg="#2B2520", relief="solid", bd=1, padx=10).pack(side="right", padx=4)
        tk.Checkbutton(bar, text="淡彩上色", variable=self.wash_mode_var, command=self._redraw, bg="#EAE1CD", fg="#2B2520", selectcolor="#FBF8F1").pack(side="right", padx=6)
        tk.Checkbutton(bar, text="尺规 Box", variable=self.show_box_var, command=self._redraw, bg="#EAE1CD", fg="#2B2520", selectcolor="#FBF8F1").pack(side="right", padx=6)
        tk.Checkbutton(bar, text="距离批注", variable=self.show_hbonds_var, command=self._redraw, bg="#EAE1CD", fg="#2B2520", selectcolor="#FBF8F1").pack(side="right", padx=6)
        tk.Checkbutton(bar, text="口袋残基", variable=self.show_pocket_var, command=self._redraw, bg="#EAE1CD", fg="#2B2520", selectcolor="#FBF8F1").pack(side="right", padx=6)

    def _reset_view(self) -> None:
        self.yaw = 0.45
        self.pitch = 0.30
        self.zoom = 26.0
        self._redraw()

    def _on_press(self, event) -> None:
        self.last_mouse = (event.x, event.y)

    def _on_drag(self, event) -> None:
        dx = event.x - self.last_mouse[0]
        dy = event.y - self.last_mouse[1]
        self.last_mouse = (event.x, event.y)
        self.yaw += dx * 0.012
        self.pitch = max(-1.45, min(1.45, self.pitch + dy * 0.012))
        self._redraw()

    def _on_wheel(self, event) -> None:
        factor = 1.12 if event.delta > 0 else 0.88
        self._zoom_by(factor)

    def _zoom_by(self, factor: float) -> None:
        self.zoom = max(8.0, min(85.0, self.zoom * factor))
        self._redraw()

    def _project(self, coord: np.ndarray, w: int, h: int) -> tuple[float, float, float]:
        rel = coord - self.center
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        x1 = rel[0] * cy - rel[2] * sy
        z1 = rel[0] * sy + rel[2] * cy
        y2 = rel[1] * cp - z1 * sp
        z2 = rel[1] * sp + z1 * cp
        persp = 45.0 / max(15.0, 45.0 - z2)
        sx = w * 0.5 + x1 * self.zoom * persp
        sy_scr = h * 0.5 - y2 * self.zoom * persp
        return sx, sy_scr, z2

    def _redraw(self) -> None:
        self.canvas.delete("all")
        w = max(400, self.canvas.winfo_width())
        h = max(300, self.canvas.winfo_height())

        # Draw faint notebook millimeter grid & hand-sketched frame border
        for gx in range(28, w, 28):
            self.canvas.create_line(gx, 0, gx, h, fill="#E5DEC9", width=1)
        for gy in range(28, h, 28):
            self.canvas.create_line(0, gy, w, gy, fill="#E5DEC9", width=1)
        _canvas_sketch_line(self.canvas, 14, 14, w - 14, 14, "#2B2520", 2, 1)
        _canvas_sketch_line(self.canvas, w - 14, 14, w - 14, h - 14, "#2B2520", 2, 2)
        _canvas_sketch_line(self.canvas, w - 14, h - 14, 14, h - 14, "#2B2520", 2, 3)
        _canvas_sketch_line(self.canvas, 14, h - 14, 14, 14, "#2B2520", 2, 4)
        self.canvas.create_text(26, h - 26, text="Plate I — 活性口袋三维钢笔淡彩速写", anchor="w", fill="#655B4E", font=("Georgia", 10, "italic"))

        if self.show_box_var.get() and self.box_info:
            bx, by, bz = self.box_info.get("center_x", 0.0), self.box_info.get("center_y", 0.0), self.box_info.get("center_z", 0.0)
            hx = self.box_info.get("size_x", 18.0) * 0.5
            hy = self.box_info.get("size_y", 18.0) * 0.5
            hz = self.box_info.get("size_z", 18.0) * 0.5
            corners = [
                np.array([bx + dx, by + dy, bz + dz], dtype=float)
                for dx in (-hx, hx) for dy in (-hy, hy) for dz in (-hz, hz)
            ]
            proj_c = [self._project(c, w, h) for c in corners]
            edges = [
                (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
                (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
            ]
            for idx_e, (i, j) in enumerate(edges):
                _canvas_sketch_line(self.canvas, proj_c[i][0], proj_c[i][1], proj_c[j][0], proj_c[j][1], "#8C7F6B", 1, idx_e + 20, dash=(4, 4))

        def _draw_group_bonds(atoms: list[dict], color: str, width: int, cutoff: float = 1.95, seed_base: int = 100) -> None:
            for i in range(len(atoms)):
                for j in range(i + 1, len(atoms)):
                    if distance(atoms[i]["coord"], atoms[j]["coord"]) <= cutoff:
                        x1, y1, _ = self._project(atoms[i]["coord"], w, h)
                        x2, y2, _ = self._project(atoms[j]["coord"], w, h)
                        _canvas_sketch_line(self.canvas, x1, y1, x2, y2, color, width, seed_base + i * 17 + j)

        if self.show_pocket_var.get():
            _draw_group_bonds(self.pocket_atoms, "#7A7062", 2, cutoff=1.9, seed_base=100)
        _draw_group_bonds(self.cof_atoms, "#9C6B21", 2, cutoff=1.85, seed_base=300)
        _draw_group_bonds(self.cat_atoms, "#A8382B", 3, cutoff=1.95, seed_base=500)
        _draw_group_bonds(self.lig_atoms, "#2B2520", 3, cutoff=1.85, seed_base=700)

        if self.show_hbonds_var.get() and self.lig_atoms:
            for la in self.lig_atoms:
                for ca in self.cat_atoms:
                    if ca["name"] in {"SG", "OG", "NZ", "OD1", "OE1"} and la["element"].upper() == "C":
                        d = distance(la["coord"], ca["coord"])
                        if d <= 5.0:
                            x1, y1, _ = self._project(la["coord"], w, h)
                            x2, y2, _ = self._project(ca["coord"], w, h)
                            _canvas_sketch_line(self.canvas, x1, y1, x2, y2, "#A8382B", 2, 901, dash=(5, 3))
                            self.canvas.create_text((x1 + x2) * 0.5, (y1 + y2) * 0.5 - 8, text=f"d_nuc = {d:.2f} Å", fill="#A8382B", font=("Georgia", 10, "bold italic"))
                if la["element"].upper() in {"O", "N", "S"}:
                    for pa in self.pocket_atoms + self.cat_atoms + self.cof_atoms:
                        if pa["element"].upper() in {"O", "N", "S"}:
                            d = distance(la["coord"], pa["coord"])
                            if 2.3 <= d <= 3.5:
                                x1, y1, _ = self._project(la["coord"], w, h)
                                x2, y2, _ = self._project(pa["coord"], w, h)
                                _canvas_sketch_line(self.canvas, x1, y1, x2, y2, "#3D6349", 1, 950, dash=(3, 3))
                                self.canvas.create_text((x1 + x2) * 0.5, (y1 + y2) * 0.5 + 8, text=f"{d:.2f} Å", fill="#3D6349", font=("Georgia", 8, "italic"))

        render_queue = []
        if self.show_pocket_var.get():
            for a in self.pocket_atoms:
                render_queue.append((a, "#D2C9B8", 4, f"{a['resname']}{a['resseq']}" if a["name"] == "CA" else ""))
        for a in self.cof_atoms:
            render_queue.append((a, "#E6B86A", 6, f"{a['resname']}:{a['name']}" if a["name"] in {"C4N", "N5", "FE"} else ""))
        for a in self.cat_atoms:
            render_queue.append((a, "#D97B6C", 7, f"{a['resname']}{a['resseq']}:{a['name']}" if a["name"] in {"SG", "OG", "CA"} else ""))
        for a in self.lig_atoms:
            el = a["element"].upper()
            col = "#DF7A6B" if el == "O" else ("#6D93B0" if el == "N" else "#5B7D95")
            render_queue.append((a, col, 8, a["name"]))

        projected_atoms = []
        for atom, col, radius, label in render_queue:
            sx, sy, z_depth = self._project(atom["coord"], w, h)
            projected_atoms.append((z_depth, sx, sy, col, radius, label))
        projected_atoms.sort(key=lambda item: item[0])

        wash = self.wash_mode_var.get()
        for _, sx, sy, col, radius, label in projected_atoms:
            if wash:
                self.canvas.create_oval(sx - radius + 1.5, sy - radius + 1.2, sx + radius + 1.5, sy + radius + 1.2, fill=col, outline="")
            else:
                self.canvas.create_oval(sx - radius, sy - radius, sx + radius, sy + radius, fill="#F4EFE0", outline="")
            self.canvas.create_oval(sx - radius, sy - radius, sx + radius, sy + radius, fill="", outline="#2B2520", width=1.5)
            self.canvas.create_arc(sx - radius * 0.7, sy - radius * 0.7, sx + radius * 0.7, sy + radius * 0.7, start=210, extent=110, style="arc", outline="#2B2520", width=1)
            if label:
                self.canvas.create_text(sx + radius + 4, sy - 3, text=label, anchor="w", fill="#2B2520", font=("Georgia", 9))


class DockingWorkbench(tk.Tk):
    """Interactive 4-page scientific workbench for the Universal ALDH & Enzyme Docking Pipeline v3.0."""

    COLORS = {
        "navy": "#2B2520",
        "navy_2": "#423930",
        "blue": "#2F4860",
        "cyan": "#3D6349",
        "green": "#3D6349",
        "amber": "#B57B28",
        "red": "#A8382B",
        "ink": "#2B2520",
        "muted": "#655B4E",
        "line": "#D8CEB8",
        "paper": "#F6F1E4",
        "white": "#FBF8F1",
    }

    def __init__(self) -> None:
        super().__init__()
        self.title("分子对接与催化构象实验手札 · 手绘工作台")
        self.geometry("1260x820")
        self.minsize(1040, 680)
        self.process: subprocess.Popen[str] | None = None
        self.messages: queue.Queue[tuple[str, object]] = queue.Queue()
        self.root_var = tk.StringVar(value=str(PROJECT_ROOT))
        self.config_var = tk.StringVar(value=str(PROJECT_ROOT / "config" / "pipeline_config.yaml"))
        self.quick_receptor_var = tk.StringVar(value=str(PROJECT_ROOT / "data" / "01_raw_receptors" / "receptor.pdb"))
        self.quick_ligands_var = tk.StringVar(value="Hexanal=CCCCC=O; Octanal=CCCCCCC=O; Nonanal=CCCCCCCC=O")
        self.status_var = tk.StringVar(value="就绪")
        self.gate_var = tk.StringVar(value="安全闸门：未检查")
        self.audit_var = tk.StringVar(value="尚未读取审计结果")
        self.plan_var = tk.StringVar(value="执行计划：未生成")
        self.result_var = tk.StringVar(value="结果：未生成")

        # Page 2 variables
        self.chain_var = tk.StringVar(value="A")
        self.cof_chain_var = tk.StringVar(value="A")
        self.cat_res_name_var = tk.StringVar(value="CYS")
        self.cat_res_num_var = tk.StringVar(value="302")
        self.oxy_res_name_var = tk.StringVar(value="ASN")
        self.oxy_res_num_var = tk.StringVar(value="169")
        self.oxy_atom_var = tk.StringVar(value="ND2")
        self.mechanism_var = tk.StringVar(value="aldh")
        self.ph_var = tk.StringVar(value="7.4")
        self.auto_chem_var = tk.BooleanVar(value=True)
        self.auto_cys_var = tk.BooleanVar(value=True)
        self.bridging_water_var = tk.BooleanVar(value=False)

        # Page 3 variables
        self.box_x_var = tk.StringVar(value="18.0")
        self.box_y_var = tk.StringVar(value="18.0")
        self.box_z_var = tk.StringVar(value="18.0")
        self.center_mode_var = tk.StringVar(value="midpoint_if_cofactor")
        self.exhaustiveness_var = tk.StringVar(value="16")
        self.num_modes_var = tk.StringVar(value="20")
        self.replicates_var = tk.StringVar(value="1")
        self.refine_propagate_var = tk.BooleanVar(value=False)
        self.dock_variants_var = tk.BooleanVar(value=False)
        self.auto_pockets_var = tk.BooleanVar(value=False)

        self.nav_frames: dict[str, tk.Frame] = {}
        self.nav_labels: dict[str, tk.Label] = {}
        self.pages: dict[str, ttk.Frame] = {}
        self.results_tree: ttk.Treeview | None = None

        self._configure_theme()
        self._build_ui()
        self._load_config_into_vars()
        self.after(100, self._drain_messages)
        self.after(400, self._refresh_gate)
        self.after(600, self._refresh_results)

    def _configure_theme(self) -> None:
        self.configure(bg=self.COLORS["paper"])
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background=self.COLORS["paper"])
        style.configure("Card.TFrame", background=self.COLORS["white"])
        style.configure("TLabel", background=self.COLORS["paper"], foreground=self.COLORS["ink"], font=("Georgia", 10))
        style.configure("Card.TLabel", background=self.COLORS["white"], foreground=self.COLORS["ink"], font=("Georgia", 10))
        style.configure("Muted.TLabel", background=self.COLORS["paper"], foreground=self.COLORS["muted"], font=("Georgia", 9))
        style.configure("CardMuted.TLabel", background=self.COLORS["white"], foreground=self.COLORS["muted"], font=("Georgia", 9))
        style.configure("Title.TLabel", background=self.COLORS["paper"], foreground=self.COLORS["navy"], font=("Georgia", 20, "bold"))
        style.configure("Section.TLabel", background=self.COLORS["white"], foreground=self.COLORS["navy"], font=("Georgia", 12, "bold"))
        style.configure("Primary.TButton", background=self.COLORS["blue"], foreground=self.COLORS["white"], borderwidth=0, padding=(14, 8), font=("Georgia", 10, "bold"))
        style.map("Primary.TButton", background=[("active", "#1E67C5")])
        style.configure("Danger.TButton", background=self.COLORS["red"], foreground=self.COLORS["white"], borderwidth=0, padding=(12, 8), font=("Georgia", 10, "bold"))
        style.configure("Secondary.TButton", background=self.COLORS["white"], foreground=self.COLORS["navy"], bordercolor=self.COLORS["line"], padding=(12, 8))
        style.configure("TEntry", fieldbackground=self.COLORS["white"], bordercolor=self.COLORS["line"], padding=6)

    def _build_ui(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_main()

    def _build_sidebar(self) -> None:
        sidebar = tk.Frame(self, bg=self.COLORS["navy"], width=230)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)
        tk.Label(sidebar, text="FOLIO", bg=self.COLORS["navy"], fg=self.COLORS["white"], font=("Georgia", 20, "bold")).pack(anchor="w", padx=24, pady=(26, 0))
        tk.Label(sidebar, text="纸本钢笔淡彩 · 实验手札台", bg=self.COLORS["navy"], fg="#C9BFA8", font=("Georgia", 9, "bold")).pack(anchor="w", padx=24, pady=(0, 24))
        nav_items = [
            ("quick", "卷一 · 标本入册"),
            ("prep", "卷二 · 活性尺规"),
            ("dock", "卷三 · 构象搜寻"),
            ("results", "卷四 · 台账与速写"),
        ]
        for key, label in nav_items:
            frame = tk.Frame(sidebar, bg=self.COLORS["navy"], padx=18, pady=11, cursor="hand2")
            frame.pack(fill="x", padx=10, pady=2)
            lbl = tk.Label(frame, text=label, bg=frame["bg"], fg="#E6DFCE", font=("Georgia", 10), cursor="hand2")
            lbl.pack(anchor="w")
            frame.bind("<Button-1>", lambda _e, k=key: self._switch_page(k))
            lbl.bind("<Button-1>", lambda _e, k=key: self._switch_page(k))
            self.nav_frames[key] = frame
            self.nav_labels[key] = lbl

        tk.Frame(sidebar, bg="#5C5042", height=1).pack(fill="x", padx=24, pady=22)
        tk.Label(sidebar, text="当前项目目录", bg=self.COLORS["navy"], fg="#C9BFA8", font=("Georgia", 8, "bold")).pack(anchor="w", padx=24)
        tk.Label(sidebar, textvariable=self.root_var, wraplength=180, justify="left", bg=self.COLORS["navy"], fg="#F6F1E4", font=("Georgia", 8)).pack(anchor="w", padx=24, pady=(6, 0))
        tk.Label(sidebar, text="9大机制 / 3D-pKa / 物理重打分已启用", bg=self.COLORS["navy"], fg="#D8B372", font=("Georgia", 8)).pack(anchor="w", padx=24, pady=(18, 0))

    def _build_main(self) -> None:
        main = ttk.Frame(self, padding=(24, 20, 24, 16))
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(3, weight=1)

        header = ttk.Frame(main)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="分子对接与催化构象实验手札", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="覆盖 9 大酶与通用靶点家族、3D 微环境 pKa、侧链自动修复、物理化学能量重打分、L-BFGS-B 精修与交互式 3D 视图。", style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=(3, 0))
        btn_bar = ttk.Frame(header)
        btn_bar.grid(row=0, column=1, rowspan=2)
        ttk.Button(btn_bar, text="停止任务", style="Danger.TButton", command=self._stop_worker).pack(side="left", padx=(0, 8))
        ttk.Button(btn_bar, text="打开结果目录", style="Secondary.TButton", command=lambda: self._open_folder("reports")).pack(side="left")

        status = ttk.Frame(main)
        status.grid(row=1, column=0, sticky="ew", pady=(16, 12))
        for index, (label, value, color) in enumerate((
            ("流程状态", self.status_var, self.COLORS["blue"]),
            ("安全闸门", self.gate_var, self.COLORS["green"]),
            ("受体与 Box 审计", self.audit_var, self.COLORS["cyan"]),
            ("结果与可靠性", self.result_var, self.COLORS["amber"]),
        )):
            card = tk.Frame(status, bg=self.COLORS["white"], highlightbackground=self.COLORS["line"], highlightthickness=1)
            card.grid(row=0, column=index, sticky="ew", padx=(0 if index == 0 else 8, 0))
            status.columnconfigure(index, weight=1)
            tk.Frame(card, bg=color, width=5).pack(side="left", fill="y")
            inner = tk.Frame(card, bg=self.COLORS["white"], padx=11, pady=8)
            inner.pack(side="left", fill="both", expand=True)
            tk.Label(inner, text=label, bg=self.COLORS["white"], fg=self.COLORS["muted"], font=("Georgia", 8, "bold")).pack(anchor="w")
            tk.Label(inner, textvariable=value, bg=self.COLORS["white"], fg=self.COLORS["ink"], font=("Georgia", 9, "bold"), wraplength=210, justify="left").pack(anchor="w", pady=(2, 0))

        self.page_container = ttk.Frame(main)
        self.page_container.grid(row=2, column=0, sticky="ew")
        self.page_container.columnconfigure(0, weight=1)

        self._build_page_quick()
        self._build_page_prep()
        self._build_page_dock()
        self._build_page_results()

        lower = ttk.PanedWindow(main, orient="vertical")
        lower.grid(row=3, column=0, sticky="nsew", pady=(12, 0))
        actions = ttk.Frame(lower, style="Card.TFrame", padding=12)
        row = ttk.Frame(actions, style="Card.TFrame")
        row.pack(anchor="w", fill="x")
        ttk.Label(row, text="快捷控制：", style="Card.TLabel").pack(side="left", padx=(0, 8))
        for text, command in (
            ("依赖预检", lambda: self._run("--preflight")),
            ("仅做自动准备与审计", lambda: self._run("--skip-vina")),
            ("标准全流程运行", lambda: self._run("")),
            ("强制重跑(忽略缓存)", lambda: self._run("--no-cache")),
            ("验收全部输出", self._validate),
            ("刷新结果表", self._refresh_results),
        ):
            ttk.Button(row, text=text, style="Secondary.TButton", command=command).pack(side="left", padx=(0, 6))

        log_frame = ttk.Frame(lower, style="Card.TFrame", padding=10)
        log_frame.rowconfigure(1, weight=1)
        log_frame.columnconfigure(0, weight=1)
        ttk.Label(log_frame, text="实验过程笔录", style="Section.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 6))
        self.log = tk.Text(log_frame, wrap="none", state="disabled", bg="#FAF5E8", fg="#2B2520", insertbackground="#FFFFFF", relief="flat", padx=12, pady=8, height=10, font=("Consolas", 9))
        self.log.grid(row=1, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        scroll.grid(row=1, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scroll.set)
        lower.add(actions, weight=0)
        lower.add(log_frame, weight=1)

        self._switch_page("quick")

    def _build_page_quick(self) -> None:
        page = ttk.Frame(self.page_container, style="Card.TFrame", padding=18)
        page.columnconfigure(1, weight=1)
        ttk.Label(page, text="卷一 · 标本入册（放入蛋白与配体标本）", style="Section.TLabel").grid(row=0, column=0, columnspan=4, sticky="w")
        ttk.Label(page, text="支持同时选择多个受体 PDB，配体支持直接导入 .mol2 / .sdf 三维结构文件、CSV/SMI 表格或 名称=SMILES。", style="CardMuted.TLabel").grid(row=1, column=0, columnspan=4, sticky="w", pady=(3, 12))
        ttk.Label(page, text="受体 PDB", style="Card.TLabel").grid(row=2, column=0, sticky="w")
        ttk.Entry(page, textvariable=self.quick_receptor_var).grid(row=2, column=1, columnspan=2, sticky="ew", padx=10)
        ttk.Button(page, text="浏览 PDB(可多选)", style="Secondary.TButton", command=self._choose_quick_receptor).grid(row=2, column=3)

        ttk.Label(page, text="配体列表", style="Card.TLabel").grid(row=3, column=0, sticky="nw", pady=(10, 0))
        ttk.Entry(page, textvariable=self.quick_ligands_var).grid(row=3, column=1, sticky="ew", padx=10, pady=(10, 0))
        ttk.Button(page, text="导入 MOL2/SDF/CSV", style="Secondary.TButton", command=self._import_ligand_file).grid(row=3, column=2, padx=(0, 8), pady=(10, 0))
        ttk.Button(page, text="一键自动配置并运行", style="Primary.TButton", command=self._quick_run).grid(row=3, column=3, pady=(10, 0))
        self.pages["quick"] = page

    def _build_page_prep(self) -> None:
        page = ttk.Frame(self.page_container, style="Card.TFrame", padding=18)
        for col in range(6):
            page.columnconfigure(col, weight=1)
        ttk.Label(page, text="卷二 · 活性尺规（位点勘定与质子化微态）", style="Section.TLabel").grid(row=0, column=0, columnspan=6, sticky="w")
        ttk.Label(page, text="支持 9 类酶与通用靶点机制，自动重建缺失侧链重原子、清理 AltLoc、探测桥联结晶水并预测口袋 3D 微环境 pKa。", style="CardMuted.TLabel").grid(row=1, column=0, columnspan=6, sticky="w", pady=(3, 12))

        ttk.Label(page, text="机制家族", style="Card.TLabel").grid(row=2, column=0, sticky="w")
        mech_box = ttk.Combobox(page, textvariable=self.mechanism_var, values=list_supported_mechanisms(), state="readonly", width=18)
        mech_box.grid(row=2, column=1, sticky="w", padx=6)
        ttk.Label(page, text="蛋白链 ID", style="Card.TLabel").grid(row=2, column=2, sticky="w")
        ttk.Entry(page, textvariable=self.chain_var, width=8).grid(row=2, column=3, sticky="w", padx=6)
        ttk.Label(page, text="辅因子链 ID", style="Card.TLabel").grid(row=2, column=4, sticky="w")
        ttk.Entry(page, textvariable=self.cof_chain_var, width=8).grid(row=2, column=5, sticky="w", padx=6)

        ttk.Label(page, text="催化残基名/号", style="Card.TLabel").grid(row=3, column=0, sticky="w", pady=(10, 0))
        cat_frame = ttk.Frame(page, style="Card.TFrame")
        cat_frame.grid(row=3, column=1, sticky="w", padx=6, pady=(10, 0))
        ttk.Entry(cat_frame, textvariable=self.cat_res_name_var, width=6).pack(side="left")
        ttk.Entry(cat_frame, textvariable=self.cat_res_num_var, width=7).pack(side="left", padx=(4, 0))

        ttk.Label(page, text="氧阴离子穴残基/号/原子", style="Card.TLabel").grid(row=3, column=2, sticky="w", pady=(10, 0))
        oxy_frame = ttk.Frame(page, style="Card.TFrame")
        oxy_frame.grid(row=3, column=3, sticky="w", padx=6, pady=(10, 0))
        ttk.Entry(oxy_frame, textvariable=self.oxy_res_name_var, width=6).pack(side="left")
        ttk.Entry(oxy_frame, textvariable=self.oxy_res_num_var, width=6).pack(side="left", padx=4)
        ttk.Entry(oxy_frame, textvariable=self.oxy_atom_var, width=6).pack(side="left")

        ttk.Label(page, text="实验 pH", style="Card.TLabel").grid(row=3, column=4, sticky="w", pady=(10, 0))
        ttk.Entry(page, textvariable=self.ph_var, width=8).grid(row=3, column=5, sticky="w", padx=6, pady=(10, 0))

        btn_row = ttk.Frame(page, style="Card.TFrame")
        btn_row.grid(row=4, column=0, columnspan=6, sticky="w", pady=(12, 0))
        ttk.Checkbutton(btn_row, text="自动去盐与微态枚举", variable=self.auto_chem_var).pack(side="left", padx=(0, 12))
        ttk.Checkbutton(btn_row, text="自动识别催化位点", variable=self.auto_cys_var).pack(side="left", padx=(0, 12))
        ttk.Checkbutton(btn_row, text="自动探测并保留活性位点桥联水", variable=self.bridging_water_var).pack(side="left", padx=(0, 12))
        ttk.Button(btn_row, text="从当前 PDB 自动探测位点", style="Secondary.TButton", command=self._auto_detect_from_pdb).pack(side="left", padx=(0, 8))
        ttk.Button(btn_row, text="保存到配置文件", style="Primary.TButton", command=self._save_vars_to_config).pack(side="left")
        self.pages["prep"] = page

    def _build_page_dock(self) -> None:
        page = ttk.Frame(self.page_container, style="Card.TFrame", padding=18)
        for col in range(6):
            page.columnconfigure(col, weight=1)
        ttk.Label(page, text="卷三 · 构象搜寻（活性腔体尺规与采样精度）", style="Section.TLabel").grid(row=0, column=0, columnspan=6, sticky="w")
        ttk.Label(page, text="支持已知位点聚焦与无先验全蛋白 3D 空腔扫描（盲对接），内置物理化学能量重打分与 L-BFGS-B 局部精修。", style="CardMuted.TLabel").grid(row=1, column=0, columnspan=6, sticky="w", pady=(3, 12))

        ttk.Label(page, text="Box 尺寸 X/Y/Z (Å)", style="Card.TLabel").grid(row=2, column=0, sticky="w")
        box_frame = ttk.Frame(page, style="Card.TFrame")
        box_frame.grid(row=2, column=1, sticky="w", padx=6)
        ttk.Entry(box_frame, textvariable=self.box_x_var, width=6).pack(side="left")
        ttk.Entry(box_frame, textvariable=self.box_y_var, width=6).pack(side="left", padx=4)
        ttk.Entry(box_frame, textvariable=self.box_z_var, width=6).pack(side="left")

        ttk.Label(page, text="Box 中心策略", style="Card.TLabel").grid(row=2, column=2, sticky="w")
        ttk.Combobox(page, textvariable=self.center_mode_var, values=["midpoint_if_cofactor", "catalytic_focused", "outward_offset"], state="readonly", width=18).grid(row=2, column=3, sticky="w", padx=6)

        ttk.Label(page, text="Exhaustiveness / Modes", style="Card.TLabel").grid(row=2, column=4, sticky="w")
        vina_frame = ttk.Frame(page, style="Card.TFrame")
        vina_frame.grid(row=2, column=5, sticky="w", padx=6)
        ttk.Entry(vina_frame, textvariable=self.exhaustiveness_var, width=6).pack(side="left")
        ttk.Entry(vina_frame, textvariable=self.num_modes_var, width=6).pack(side="left", padx=(4, 0))

        ttk.Label(page, text="重复 Seed 次数", style="Card.TLabel").grid(row=3, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(page, textvariable=self.replicates_var, width=8).grid(row=3, column=1, sticky="w", padx=6, pady=(10, 0))

        opt_row = ttk.Frame(page, style="Card.TFrame")
        opt_row.grid(row=3, column=2, columnspan=4, sticky="w", pady=(10, 0))
        ttk.Checkbutton(opt_row, text="精修构象用于下游分析（默认不覆盖对接构象）", variable=self.refine_propagate_var).pack(side="left", padx=(0, 10))
        ttk.Checkbutton(opt_row, text="对接全部微态变体", variable=self.dock_variants_var).pack(side="left", padx=(0, 10))
        ttk.Checkbutton(opt_row, text="全蛋白自动发现多口袋", variable=self.auto_pockets_var).pack(side="left", padx=(0, 10))
        ttk.Button(opt_row, text="保存并启动对接", style="Primary.TButton", command=self._save_and_run).pack(side="left")
        self.pages["dock"] = page

    def _build_page_results(self) -> None:
        page = ttk.Frame(self.page_container, style="Card.TFrame", padding=16)
        page.columnconfigure(0, weight=1)
        header_row = ttk.Frame(page, style="Card.TFrame")
        header_row.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        header_row.columnconfigure(0, weight=1)
        ttk.Label(header_row, text="卷四 · 实验台账与三维钢笔速写", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        btn_box = ttk.Frame(header_row, style="Card.TFrame")
        btn_box.grid(row=0, column=1, sticky="e")
        ttk.Button(btn_box, text="展阅手绘实验手札(HTML)", style="Primary.TButton", command=self._open_html_report).pack(side="left", padx=(0, 6))
        ttk.Button(btn_box, text="开启三维钢笔速写台", style="Secondary.TButton", command=self._open_interactive_3d_viewer).pack(side="left", padx=(0, 6))
        ttk.Button(btn_box, text="预览 300DPI 渲染图", style="Secondary.TButton", command=self._preview_selected_figure).pack(side="left", padx=(0, 6))
        ttk.Button(btn_box, text="查看 14 项系统覆盖矩阵", style="Secondary.TButton", command=self._show_coverage_dialog).pack(side="left")

        columns = (
            "Receptor", "Ligand", "Mode", "Binding_Affinity_kcal_mol", "PhysChem_Score_kcal_mol",
            "Universal_Consensus_Rank", "d_nuc_A", "d_hyd_A", "d_oxy_asn_A",
            "Catalytic_Geometry_Score", "Catalytically_Competent", "Reliability_Grade",
        )
        tree = ttk.Treeview(page, columns=columns, show="headings", height=5)
        headings = {
            "Receptor": "受体",
            "Ligand": "配体",
            "Mode": "构象",
            "Binding_Affinity_kcal_mol": "Vina(kcal/mol)",
            "PhysChem_Score_kcal_mol": "物理重打分",
            "Universal_Consensus_Rank": "Borda共识秩",
            "d_nuc_A": "d_nuc(Å)",
            "d_hyd_A": "d_hyd(Å)",
            "d_oxy_asn_A": "d_oxy(Å)",
            "Catalytic_Geometry_Score": "几何总分",
            "Catalytically_Competent": "催化合格",
            "Reliability_Grade": "可靠性等级",
        }
        for col in columns:
            tree.heading(col, text=headings.get(col, col))
            tree.column(col, width=90 if col != "Reliability_Grade" else 170, anchor="center")
        tree.grid(row=1, column=0, sticky="ew")
        self.results_tree = tree
        self.pages["results"] = page

    def _open_html_report(self) -> None:
        root = Path(self.root_var.get())
        cn_html = root / "reports" / "对接结果可视化总览报告.html"
        en_html = root / "reports" / "visual_summary_report.html"
        target = cn_html if cn_html.is_file() else en_html
        if not target.is_file():
            messagebox.showinfo("提示", "尚未生成可视化 HTML 报告，请先运行一次对接。")
            return
        webbrowser.open(target.resolve().as_uri())

    def _open_interactive_3d_viewer(self) -> None:
        selected = self.results_tree.selection() if self.results_tree else ()
        ligand = None
        if selected and self.results_tree:
            values = self.results_tree.item(selected[0], "values")
            if values:
                ligand = str(values[1]) if len(values) > 1 and self.results_tree["columns"][0] == "Receptor" else str(values[0])
        if not ligand and self.results_tree:
            children = self.results_tree.get_children()
            if children:
                vals0 = self.results_tree.item(children[0], "values")
                ligand = str(vals0[1]) if len(vals0) > 1 and self.results_tree["columns"][0] == "Receptor" else str(vals0[0])
        if not ligand:
            messagebox.showinfo("提示", "请先运行对接并选择一条配体结果。")
            return
        root = Path(self.root_var.get())
        refined = root / "reports" / "active_poses" / f"{ligand}_refined.pdb"
        active = refined if refined.is_file() else (root / "reports" / "active_poses" / f"{ligand}_active.pdb")
        if not active.is_file():
            messagebox.showwarning("未找到结构", f"未找到复合物结构：{active}")
            return
        box_json = root / "reports" / "docking_box_audit.json"
        cat_num = int(self.cat_res_num_var.get().strip() or 302)
        Interactive3DPocketWindow(self, active, box_json, catalytic_num=cat_num)

    def _switch_page(self, page_key: str) -> None:
        for key, frame in self.pages.items():
            if key == page_key:
                frame.grid(row=0, column=0, sticky="ew")
            else:
                frame.grid_remove()
        for key, nav_frame in self.nav_frames.items():
            active = key == page_key
            bg = self.COLORS["navy_2"] if active else self.COLORS["navy"]
            fg = self.COLORS["white"] if active else "#C7D6E5"
            nav_frame.configure(bg=bg)
            self.nav_labels[key].configure(bg=bg, fg=fg, font=("Georgia", 10, "bold" if active else "normal"))
        if page_key == "results":
            self._refresh_results()

    def _load_config_into_vars(self) -> None:
        cfg_path = Path(self.config_var.get())
        if not cfg_path.is_file():
            return
        try:
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            rec = cfg.get("receptor", {})
            self.chain_var.set(str(rec.get("chain_id", "A")))
            self.cof_chain_var.set(str(rec.get("cofactor_chain", "A")))
            cat = rec.get("catalytic_residue", {})
            self.cat_res_name_var.set(str(cat.get("name", "CYS")))
            self.cat_res_num_var.set(str(cat.get("number", 302)))
            oxy = rec.get("oxyanion_residue", {})
            self.oxy_res_name_var.set(str(oxy.get("name", "ASN")))
            self.oxy_res_num_var.set(str(oxy.get("number", 169)))
            self.oxy_atom_var.set(str(oxy.get("atom", "ND2")))
            enz = cfg.get("enzyme", {})
            self.mechanism_var.set(str(enz.get("mechanism_family", "aldh")))
            self.bridging_water_var.set(bool(enz.get("auto_retain_bridging_waters", False)))
            safety = cfg.get("safety", {})
            self.ph_var.set(str(safety.get("experimental_ph", 7.4)))
            self.auto_chem_var.set(bool(safety.get("auto_remediate_chemical_state", True)))
            box = cfg.get("docking_box", {})
            self.box_x_var.set(str(box.get("size_x", 18.0)))
            self.box_y_var.set(str(box.get("size_y", 18.0)))
            self.box_z_var.set(str(box.get("size_z", 18.0)))
            self.center_mode_var.set(str(box.get("center_mode", "midpoint_if_cofactor")))
            self.auto_pockets_var.set(bool(box.get("auto_discover_pockets", False)))
            vina = cfg.get("vina", {})
            self.exhaustiveness_var.set(str(vina.get("exhaustiveness", 16)))
            self.num_modes_var.set(str(vina.get("num_modes", 20)))
            adv = cfg.get("advanced", {})
            self.replicates_var.set(str(adv.get("replicates", {}).get("count", 1)))
            self.auto_cys_var.set(bool(adv.get("auto_detect_catalytic_cys", False)))
            self.dock_variants_var.set(bool(adv.get("ligand_variants", {}).get("dock_variants", False)))
            self.refine_propagate_var.set(bool(adv.get("pose_refinement", {}).get("propagate_to_downstream", False)))
        except Exception as exc:
            self._append(f"[WARN] 读取配置失败: {exc}\n")

    def _save_vars_to_config(self) -> None:
        cfg_path = Path(self.config_var.get())
        if not cfg_path.is_file():
            return
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        cfg.setdefault("receptor", {})["chain_id"] = self.chain_var.get().strip() or "A"
        cfg["receptor"]["cofactor_chain"] = self.cof_chain_var.get().strip() or "A"
        cfg["receptor"]["catalytic_residue"] = {
            "name": self.cat_res_name_var.get().strip() or "CYS",
            "number": int(self.cat_res_num_var.get().strip() or 302),
            "chain": self.chain_var.get().strip() or "A",
        }
        cfg["receptor"]["oxyanion_residue"] = {
            "name": self.oxy_res_name_var.get().strip() or "ASN",
            "number": int(self.oxy_res_num_var.get().strip() or 169),
            "atom": self.oxy_atom_var.get().strip() or "ND2",
        }
        cfg.setdefault("enzyme", {})["mechanism_family"] = self.mechanism_var.get().strip() or "aldh"
        cfg["enzyme"]["auto_retain_bridging_waters"] = bool(self.bridging_water_var.get())
        cfg.setdefault("safety", {})["experimental_ph"] = float(self.ph_var.get().strip() or 7.4)
        cfg["safety"]["auto_remediate_chemical_state"] = bool(self.auto_chem_var.get())
        cfg.setdefault("docking_box", {})["size_x"] = float(self.box_x_var.get().strip() or 18.0)
        cfg["docking_box"]["size_y"] = float(self.box_y_var.get().strip() or 18.0)
        cfg["docking_box"]["size_z"] = float(self.box_z_var.get().strip() or 18.0)
        cfg["docking_box"]["center_mode"] = self.center_mode_var.get().strip() or "midpoint_if_cofactor"
        cfg["docking_box"]["auto_discover_pockets"] = bool(self.auto_pockets_var.get())
        cfg.setdefault("vina", {})["exhaustiveness"] = int(self.exhaustiveness_var.get().strip() or 16)
        cfg["vina"]["num_modes"] = int(self.num_modes_var.get().strip() or 20)
        adv = cfg.setdefault("advanced", {})
        adv["auto_detect_catalytic_cys"] = bool(self.auto_cys_var.get())
        adv.setdefault("pose_refinement", {})["propagate_to_downstream"] = bool(self.refine_propagate_var.get())
        adv.setdefault("replicates", {})["count"] = int(self.replicates_var.get().strip() or 1)
        adv.setdefault("ligand_variants", {})["dock_variants"] = bool(self.dock_variants_var.get())
        with cfg_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(cfg, handle, sort_keys=False, allow_unicode=True)
        self._append(f"[INFO] 已保存配置至 {cfg_path}\n")

    def _save_and_run(self) -> None:
        self._save_vars_to_config()
        self._run("")

    def _auto_detect_from_pdb(self) -> None:
        from src.step7_advanced import auto_configure_receptor

        pdb_str = self.quick_receptor_var.get().strip()
        pdb_path = Path(pdb_str) if pdb_str else Path(self.root_var.get()) / "data" / "01_raw_receptors" / "receptor.pdb"
        if not pdb_path.is_file():
            messagebox.showerror("文件不存在", f"未找到受体 PDB：{pdb_path}")
            return
        mock_cfg = {
            "receptor": {"chain_id": "auto", "keep_cofactors": ["NAD", "NAP", "NDP", "FAD", "FMN", "HEM"]},
            "enzyme": {"mechanism_family": self.mechanism_var.get() or "aldh"},
            "advanced": {"auto_detect_catalytic_cys": True, "auto_configure_receptor": True},
        }
        auto_configure_receptor(pdb_path, mock_cfg)
        rec = mock_cfg["receptor"]
        self.chain_var.set(str(rec.get("chain_id", "A")))
        self.cof_chain_var.set(str(rec.get("cofactor_chain", "A")))
        cat = rec.get("catalytic_residue", {})
        if cat:
            self.cat_res_name_var.set(str(cat.get("name", "CYS")))
            self.cat_res_num_var.set(str(cat.get("number", 302)))
        oxy = rec.get("oxyanion_residue", {})
        if oxy:
            self.oxy_res_name_var.set(str(oxy.get("name", "ASN")))
            self.oxy_res_num_var.set(str(oxy.get("number", 169)))
            self.oxy_atom_var.set(str(oxy.get("atom", "ND2")))
        self._append(
            f"[AUTO] 已自动识别受体位点: 链={self.chain_var.get()}, 辅因子链={self.cof_chain_var.get()}, "
            f"催化残基={self.cat_res_name_var.get()}{self.cat_res_num_var.get()}, "
            f"氧阴离子穴={self.oxy_res_name_var.get()}{self.oxy_res_num_var.get()}:{self.oxy_atom_var.get()}\n"
        )

    def _choose_quick_receptor(self) -> None:
        selected = filedialog.askopenfilenames(
            initialdir=str(Path(self.root_var.get()) / "data" / "01_raw_receptors"),
            title="选择一个或多个受体 PDB 文件（支持多选）",
            filetypes=[("蛋白质结构 .pdb / .cif (支持多选)", "*.pdb *.cif *.mmcif"), ("All files", "*.*")],
        )
        if selected:
            self.quick_receptor_var.set("; ".join(selected))
            self._append(f"[INFO] 已选择 {len(selected)} 个受体 PDB 文件。\n")

    def _import_ligand_file(self) -> None:
        selected_files = filedialog.askopenfilenames(
            initialdir=str(Path(self.root_var.get()) / "data"),
            title="选择配体文件（支持多选 .mol2 / .sdf / .pdbqt / .csv / .smi）",
            filetypes=[
                ("所有支持的配体格式 (.mol2/.sdf/.pdbqt/.csv/.smi)", "*.mol2 *.sdf *.mol *.pdb *.pdbqt *.csv *.smi *.txt"),
                ("Tripos MOL2 3D 配体 (*.mol2)", "*.mol2"),
                ("SDF/MOL/PDBQT 3D 配体", "*.sdf *.mol *.pdbqt"),
                ("SMILES / CSV 列表", "*.csv *.smi *.txt"),
                ("All files", "*.*"),
            ],
        )
        if not selected_files:
            return
        entries = []
        for selected in selected_files:
            path = Path(selected)
            ext = path.suffix.lower()
            if ext in {".mol2", ".sdf", ".mol", ".pdb", ".pdbqt"}:
                entries.append(f"{path.stem}={path}")
                continue
            if ext == ".csv":
                with path.open(newline="", encoding="utf-8", errors="replace") as handle:
                    for row in csv.DictReader(handle):
                        n = (row.get("name") or row.get("Name") or row.get("Ligand") or "").strip()
                        s = (row.get("mol2") or row.get("MOL2") or row.get("file") or row.get("smiles") or row.get("SMILES") or "").strip()
                        if n and s:
                            entries.append(f"{n}={s}")
            else:
                for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                    t = line.strip()
                    if not t or t.startswith("#"):
                        continue
                    if "=" in t:
                        entries.append(t)
                    else:
                        parts = t.split()
                        if len(parts) >= 2:
                            entries.append(f"{parts[1]}={parts[0]}")
                        elif len(parts) == 1 and parts[0].lower().endswith((".mol2", ".sdf", ".mol", ".pdbqt")):
                            entries.append(f"{Path(parts[0]).stem}={parts[0]}")
        if entries:
            self.quick_ligands_var.set("; ".join(entries))
            self._append(f"[INFO] 已成功导入 {len(entries)} 个配体（支持 .mol2 / SMILES）。\n")
            return
        path = Path(selected_files[0])
        if path.suffix.lower() == ".csv":
            with path.open(newline="", encoding="utf-8", errors="replace") as handle:
                for row in csv.DictReader(handle):
                    n = (row.get("name") or row.get("Name") or row.get("Ligand") or "").strip()
                    s = (row.get("smiles") or row.get("SMILES") or "").strip()
                    if n and s:
                        entries.append(f"{n}={s}")
        else:
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                t = line.strip()
                if not t or t.startswith("#"):
                    continue
                if "=" in t:
                    entries.append(t)
                else:
                    parts = t.split()
                    if len(parts) >= 2:
                        entries.append(f"{parts[1]}={parts[0]}")
        if entries:
            self.quick_ligands_var.set("; ".join(entries))
            self._append(f"[INFO] 已从 {path.name} 导入 {len(entries)} 个配体。\n")

    def _quick_run(self) -> None:
        if self._busy():
            return
        receptor = self.quick_receptor_var.get().strip()
        ligands = [line.strip() for line in self.quick_ligands_var.get().replace(";", "\n").splitlines() if line.strip()]
        if not receptor or not ligands:
            messagebox.showerror("输入不完整", "请选择受体 PDB，并至少填写一个配体。")
            return
        root = Path(self.root_var.get())
        output = root / "config" / "quick_config.yaml"
        command = [sys.executable, str(root / "scripts" / "quick_start.py"), "--project", str(root), "--receptor", receptor]
        for ligand in ligands:
            command.extend(["--ligand", ligand])
        command.extend(["--output", str(output)])
        self._start_worker(command, root, on_success=lambda: self._start_worker([sys.executable, str(root / "main.py"), "--config", str(output)], root))

    def _run(self, argument: str) -> None:
        if self._busy():
            return
        root = Path(self.root_var.get())
        command = [sys.executable, str(root / "main.py"), "--config", self.config_var.get()]
        if argument:
            command.append(argument)
        self._start_worker(command, root)

    def _validate(self) -> None:
        if self._busy():
            return
        root = Path(self.root_var.get())
        self._start_worker([sys.executable, str(root / "scripts" / "validate_outputs.py"), "--root", str(root)], root)

    def _stop_worker(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.status_var.set("已终止")
            self._append("[WARN] 用户已手动终止当前运行任务。\n")

    def _start_worker(self, command: list[str], cwd: Path, on_success=None) -> None:
        if not (cwd / "main.py").is_file():
            messagebox.showerror("项目目录无效", "所选目录中没有 main.py。")
            return
        self._clear_log()
        self.status_var.set("运行中…")
        self._append(">>> " + subprocess.list2cmdline(command) + "\n")
        threading.Thread(target=self._worker, args=(command, cwd, on_success), daemon=True).start()

    def _worker(self, command: list[str], cwd: Path, on_success=None) -> None:
        try:
            self.process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            assert self.process.stdout is not None
            for line in self.process.stdout:
                self.messages.put(("log", line))
            code = self.process.wait()
            if code == 0 and on_success:
                self.messages.put(("chain", on_success))
            self.messages.put(("done", str(code)))
        except Exception as exc:
            self.messages.put(("error", str(exc)))

    def _drain_messages(self) -> None:
        try:
            while True:
                kind, value = self.messages.get_nowait()
                if kind == "log":
                    self._append(str(value))
                elif kind == "done":
                    self.status_var.set("已完成" if value == "0" else f"失败（退出码 {value}）")
                    self._refresh_gate()
                    self._refresh_results()
                elif kind == "chain" and callable(value):
                    value()
                else:
                    self.status_var.set("启动失败")
                    self._append(f"[ERROR] {value}\n")
        except queue.Empty:
            pass
        self.after(100, self._drain_messages)

    def _refresh_gate(self) -> None:
        path = Path(self.root_var.get()) / "reports" / "capability_gate.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            status = data.get("interpretation_status", "unknown").replace("_", " ")
            labels = {"exploratory override": "探索性覆盖", "standard or reviewed": "标准通过(零阻断)"}
            self.gate_var.set(labels.get(status, status))
        except (FileNotFoundError, json.JSONDecodeError):
            self.gate_var.set("安全闸门：未检查")

    def _refresh_results(self) -> None:
        reports = Path(self.root_var.get()) / "reports"
        try:
            state = json.loads((reports / "receptor_state.json").read_text(encoding="utf-8"))
            box = json.loads((reports / "docking_box_audit.json").read_text(encoding="utf-8"))
            state_label = {"ready_for_preparation": "受体已自动清理就绪", "manual_review_required": "受体需人工复核"}.get(state.get("status"), state.get("status", "未知"))
            box_label = {"box_passed": "Box 覆盖通过", "manual_review_required": "Box 需人工复核"}.get(box.get("status"), box.get("status", "未知"))
            self.audit_var.set(f"{state_label}；{box_label}")
        except (FileNotFoundError, json.JSONDecodeError):
            self.audit_var.set("尚未读取受体/Box 审计")
        try:
            plan = json.loads((reports / "execution_plan.json").read_text(encoding="utf-8"))
            self.plan_var.set(f"{plan.get('receptor_count', 0)} 受体 × {plan.get('ligand_count', 0)} 配体；共 {plan.get('planned_run_count', 0)} 次")
        except (FileNotFoundError, json.JSONDecodeError):
            self.plan_var.set("执行计划：未生成")

        summary_csv = reports / "docking_summary.csv"
        if summary_csv.is_file():
            try:
                with summary_csv.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                competent = sum(row.get("Catalytically_Competent") == "True" for row in rows)
                self.result_var.set(f"共 {len(rows)} 个配体；催化合格 {competent}/{len(rows)}")
                if self.results_tree is not None:
                    for item in self.results_tree.get_children():
                        self.results_tree.delete(item)
                    for row in rows:
                        values = tuple(row.get(col, "") for col in self.results_tree["columns"])
                        self.results_tree.insert("", "end", values=values)
            except OSError:
                self.result_var.set("结果：读取失败")
        else:
            self.result_var.set("结果：未生成")

    def _preview_selected_figure(self) -> None:
        if self.results_tree is None:
            return
        selected = self.results_tree.selection()
        ligand = None
        if selected:
            values = self.results_tree.item(selected[0], "values")
            if values:
                ligand = str(values[1]) if len(values) > 1 and self.results_tree["columns"][0] == "Receptor" else str(values[0])
        if not ligand:
            children = self.results_tree.get_children()
            if children:
                vals0 = self.results_tree.item(children[0], "values")
                ligand = str(vals0[1]) if len(vals0) > 1 and self.results_tree["columns"][0] == "Receptor" else str(vals0[0])
        if not ligand:
            messagebox.showinfo("提示", "请先运行对接并选择一条配体结果。")
            return
        png_path = Path(self.root_var.get()) / "reports" / "figures" / f"{ligand}_300dpi.png"
        if not png_path.is_file():
            messagebox.showwarning("未找到图像", f"未找到渲染图：{png_path}")
            return
        top = tk.Toplevel(self)
        top.title(f"{ligand} 活性位点 300DPI 渲染预览")
        top.geometry("860x660")
        try:
            from PIL import Image, ImageTk

            img = Image.open(png_path)
            img.thumbnail((820, 610))
            photo = ImageTk.PhotoImage(img)
            lbl = tk.Label(top, image=photo)
            lbl.image = photo  # type: ignore[attr-defined]
            lbl.pack(expand=True, fill="both", padx=10, pady=10)
        except Exception:
            photo = tk.PhotoImage(file=str(png_path)).subsample(3, 3)
            lbl = tk.Label(top, image=photo)
            lbl.image = photo  # type: ignore[attr-defined]
            lbl.pack(expand=True, fill="both", padx=10, pady=10)

    def _show_coverage_dialog(self) -> None:
        cov_path = Path(self.root_var.get()) / "reports" / "system_coverage.json"
        if not cov_path.is_file():
            messagebox.showinfo("提示", "尚未生成 system_coverage.json。")
            return
        data = json.loads(cov_path.read_text(encoding="utf-8"))
        top = tk.Toplevel(self)
        top.title("14 项全流程自动化覆盖矩阵 (v3.0)")
        top.geometry("780x460")
        tree = ttk.Treeview(top, columns=("item", "status", "external"), show="headings")
        tree.heading("item", text="模块项")
        tree.heading("status", text="当前自动化状态")
        tree.heading("external", text="内置执行状态")
        tree.column("item", width=190)
        tree.column("status", width=400)
        tree.column("external", width=140, anchor="center")
        for row in data.get("items", []):
            tree.insert("", "end", values=(row.get("item"), row.get("status"), "已全内置闭环"))
        tree.pack(expand=True, fill="both", padx=12, pady=12)

    def _busy(self) -> bool:
        if self.process and self.process.poll() is None:
            messagebox.showwarning("任务运行中", "请等待当前任务完成或点击“停止任务”。")
            return True
        return False

    def _append(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _open_folder(self, relative: str) -> None:
        target = Path(self.root_var.get()) / relative
        target.mkdir(parents=True, exist_ok=True)
        if hasattr(os, "startfile"):
            os.startfile(str(target))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target)])


if __name__ == "__main__":
    DockingWorkbench().mainloop()
