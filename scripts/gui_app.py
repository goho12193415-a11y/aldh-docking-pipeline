from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DockingWorkbench(tk.Tk):
    """Compact scientific workbench for the existing CLI pipeline."""

    COLORS = {
        "navy": "#10233F",
        "navy_2": "#173557",
        "blue": "#2F80ED",
        "cyan": "#35B8C8",
        "green": "#2E9B68",
        "amber": "#D88A28",
        "red": "#C94C4C",
        "ink": "#182433",
        "muted": "#6C7A89",
        "line": "#DCE4EC",
        "paper": "#F4F7FA",
        "white": "#FFFFFF",
    }

    def __init__(self) -> None:
        super().__init__()
        self.title("ALDH 分子对接工作台")
        self.geometry("1180x760")
        self.minsize(980, 650)
        self.process: subprocess.Popen[str] | None = None
        self.messages: queue.Queue[tuple[str, str]] = queue.Queue()
        self.root_var = tk.StringVar(value=str(PROJECT_ROOT))
        self.config_var = tk.StringVar(value=str(PROJECT_ROOT / "config" / "pipeline_config.yaml"))
        self.quick_receptor_var = tk.StringVar()
        self.quick_ligands_var = tk.StringVar(value="Hexanal=CCCCC=O")
        self.status_var = tk.StringVar(value="就绪")
        self.gate_var = tk.StringVar(value="安全闸门：未检查")
        self.audit_var = tk.StringVar(value="尚未读取审计结果")
        self.plan_var = tk.StringVar(value="执行计划：未生成")
        self.result_var = tk.StringVar(value="结果：未生成")
        self._configure_theme()
        self._build_ui()
        self.after(100, self._drain_messages)
        self.after(500, self._refresh_gate)
        self.after(700, self._refresh_results)

    def _configure_theme(self) -> None:
        self.configure(bg=self.COLORS["paper"])
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background=self.COLORS["paper"])
        style.configure("Card.TFrame", background=self.COLORS["white"])
        style.configure("TLabel", background=self.COLORS["paper"], foreground=self.COLORS["ink"], font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background=self.COLORS["paper"], foreground=self.COLORS["muted"], font=("Segoe UI", 9))
        style.configure("Title.TLabel", background=self.COLORS["paper"], foreground=self.COLORS["navy"], font=("Segoe UI", 22, "bold"))
        style.configure("Section.TLabel", background=self.COLORS["white"], foreground=self.COLORS["navy"], font=("Segoe UI", 12, "bold"))
        style.configure("Nav.TLabel", background=self.COLORS["navy"], foreground="#DCE9F5", font=("Segoe UI", 10))
        style.configure("NavActive.TLabel", background=self.COLORS["navy_2"], foreground=self.COLORS["white"], font=("Segoe UI", 10, "bold"))
        style.configure("Primary.TButton", background=self.COLORS["blue"], foreground=self.COLORS["white"], borderwidth=0, padding=(15, 9), font=("Segoe UI", 10, "bold"))
        style.map("Primary.TButton", background=[("active", "#1E67C5")])
        style.configure("Secondary.TButton", background=self.COLORS["white"], foreground=self.COLORS["navy"], bordercolor=self.COLORS["line"], padding=(12, 8))
        style.configure("TEntry", fieldbackground=self.COLORS["white"], bordercolor=self.COLORS["line"], padding=7)
        style.configure("TNotebook", background=self.COLORS["paper"], borderwidth=0)
        style.configure("TNotebook.Tab", padding=(14, 8), background=self.COLORS["paper"])
        style.configure("TProgressbar", troughcolor="#E6EDF4", background=self.COLORS["cyan"], borderwidth=0)

    def _build_ui(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_main()

    def _build_sidebar(self) -> None:
        sidebar = tk.Frame(self, bg=self.COLORS["navy"], width=220)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)
        tk.Label(sidebar, text="ALDH", bg=self.COLORS["navy"], fg=self.COLORS["white"], font=("Segoe UI", 24, "bold")).pack(anchor="w", padx=24, pady=(28, 0))
        tk.Label(sidebar, text="分子对接工作台", bg=self.COLORS["navy"], fg="#8FB5D8", font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=26, pady=(0, 28))
        for label, active in (("01  快速实验", True), ("02  结构准备", False), ("03  对接运行", False), ("04  结果与审计", False)):
            frame = tk.Frame(sidebar, bg=self.COLORS["navy_2"] if active else self.COLORS["navy"], padx=18, pady=11)
            frame.pack(fill="x", padx=10, pady=2)
            tk.Label(frame, text=label, bg=frame["bg"], fg=self.COLORS["white"] if active else "#C7D6E5", font=("Segoe UI", 10, "bold" if active else "normal")).pack(anchor="w")
        tk.Frame(sidebar, bg="#28486A", height=1).pack(fill="x", padx=24, pady=25)
        tk.Label(sidebar, text="当前项目", bg=self.COLORS["navy"], fg="#8FB5D8", font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=26)
        tk.Label(sidebar, textvariable=self.root_var, wraplength=170, justify="left", bg=self.COLORS["navy"], fg="#DCE9F5", font=("Segoe UI", 8)).pack(anchor="w", padx=26, pady=(8, 0))
        tk.Label(sidebar, text="自动安全闸门已启用", bg=self.COLORS["navy"], fg="#7BD6A7", font=("Segoe UI", 8)).pack(anchor="w", padx=26, pady=(22, 0))

    def _build_main(self) -> None:
        main = ttk.Frame(self, padding=(28, 24, 28, 18))
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(4, weight=1)

        header = ttk.Frame(main)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="新建分子对接实验", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="只需提供两个输入，程序会自动完成准备、适用性检查和结果审计。", style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Button(header, text="打开结果", style="Secondary.TButton", command=lambda: self._open_folder("reports")).grid(row=0, column=1, rowspan=2, padx=(12, 0))

        status = ttk.Frame(main)
        status.grid(row=1, column=0, sticky="ew", pady=(22, 14))
        for index, (label, value, color) in enumerate((("流程状态", "就绪", self.COLORS["blue"]), ("安全闸门", self.gate_var, self.COLORS["green"]), ("输出状态", "已审计", self.COLORS["cyan"]))):
            card = tk.Frame(status, bg=self.COLORS["white"], highlightbackground=self.COLORS["line"], highlightthickness=1)
            card.grid(row=0, column=index, sticky="ew", padx=(0 if index == 0 else 10, 0))
            status.columnconfigure(index, weight=1)
            tk.Frame(card, bg=color, width=5).pack(side="left", fill="y")
            inner = tk.Frame(card, bg=self.COLORS["white"], padx=13, pady=9)
            inner.pack(side="left", fill="both", expand=True)
            tk.Label(inner, text=label, bg=self.COLORS["white"], fg=self.COLORS["muted"], font=("Segoe UI", 8, "bold")).pack(anchor="w")
            if isinstance(value, tk.StringVar):
                tk.Label(inner, textvariable=value, bg=self.COLORS["white"], fg=self.COLORS["ink"], font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(3, 0))
            else:
                tk.Label(inner, text=value, bg=self.COLORS["white"], fg=self.COLORS["ink"], font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(3, 0))

        quick = ttk.Frame(main, style="Card.TFrame", padding=20)
        quick.grid(row=2, column=0, sticky="ew")
        quick.columnconfigure(1, weight=1)
        ttk.Label(quick, text="快速实验", style="Section.TLabel").grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(quick, text="选择一个受体，并按“名称=SMILES”填写一个或多个配体。", style="Muted.TLabel").grid(row=1, column=0, columnspan=3, sticky="w", pady=(3, 16))
        ttk.Label(quick, text="受体 PDB").grid(row=2, column=0, sticky="w")
        ttk.Entry(quick, textvariable=self.quick_receptor_var).grid(row=2, column=1, sticky="ew", padx=12)
        ttk.Button(quick, text="选择文件", style="Secondary.TButton", command=self._choose_quick_receptor).grid(row=2, column=2)
        ttk.Label(quick, text="配体列表").grid(row=3, column=0, sticky="nw", pady=(12, 0))
        ttk.Entry(quick, textvariable=self.quick_ligands_var).grid(row=3, column=1, sticky="ew", padx=12, pady=(12, 0))
        ttk.Label(quick, text="示例：Hexanal=CCCCC=O", style="Muted.TLabel").grid(row=4, column=1, sticky="w", padx=12, pady=(3, 0))
        ttk.Button(quick, text="生成并运行", style="Primary.TButton", command=self._quick_run).grid(row=3, column=2, rowspan=2, pady=(12, 0))

        audit = ttk.Frame(main, style="Card.TFrame", padding=14)
        audit.grid(row=3, column=0, sticky="ew", pady=(14, 0))
        audit.columnconfigure(0, weight=1)
        audit.columnconfigure(1, weight=1)
        audit.columnconfigure(2, weight=1)
        ttk.Label(audit, text="结果与审计摘要", style="Section.TLabel").grid(row=0, column=0, columnspan=3, sticky="w")
        for index, (label, variable) in enumerate((
            ("受体与 box", self.audit_var),
            ("执行计划", self.plan_var),
            ("结果稳定性", self.result_var),
        )):
            card = tk.Frame(audit, bg=self.COLORS["paper"], highlightbackground=self.COLORS["line"], highlightthickness=1)
            card.grid(row=1, column=index, sticky="ew", padx=(0 if index == 0 else 8, 0), pady=(9, 0))
            tk.Label(card, text=label, bg=self.COLORS["paper"], fg=self.COLORS["muted"], font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=10, pady=(7, 0))
            tk.Label(card, textvariable=variable, bg=self.COLORS["paper"], fg=self.COLORS["ink"], font=("Segoe UI", 9, "bold"), wraplength=230, justify="left").pack(anchor="w", padx=10, pady=(3, 8))

        lower = ttk.PanedWindow(main, orient="vertical")
        lower.grid(row=4, column=0, sticky="nsew", pady=(16, 0))
        actions = ttk.Frame(lower, style="Card.TFrame", padding=14)
        ttk.Label(actions, text="高级控制", style="Section.TLabel").pack(anchor="w")
        ttk.Label(actions, text="只有需要检查或调整自动流程时才使用这些选项。", style="Muted.TLabel").pack(anchor="w", pady=(3, 8))
        row = ttk.Frame(actions, style="Card.TFrame")
        row.pack(anchor="w")
        for text, command in (("依赖预检", lambda: self._run("--preflight")), ("只做准备", lambda: self._run("--skip-vina")), ("完整运行", lambda: self._run("")), ("验收输出", self._validate)):
            ttk.Button(row, text=text, style="Secondary.TButton", command=command).pack(side="left", padx=(0, 7))
        ttk.Button(row, text="配置目录", style="Secondary.TButton", command=lambda: self._open_folder("config")).pack(side="left")
        log_frame = ttk.Frame(lower, style="Card.TFrame", padding=12)
        log_frame.rowconfigure(1, weight=1)
        log_frame.columnconfigure(0, weight=1)
        ttk.Label(log_frame, text="运行日志", style="Section.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 8))
        self.log = tk.Text(log_frame, wrap="none", state="disabled", bg="#101B2B", fg="#D8E6F3", insertbackground="#FFFFFF", relief="flat", padx=12, pady=10, font=("Consolas", 9))
        self.log.grid(row=1, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        scroll.grid(row=1, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scroll.set)
        lower.add(actions, weight=0)
        lower.add(log_frame, weight=1)
        ttk.Label(main, textvariable=self.status_var, style="Muted.TLabel").grid(row=5, column=0, sticky="w", pady=(9, 0))

    def _choose_quick_receptor(self) -> None:
        selected = filedialog.askopenfilename(initialdir=str(Path(self.root_var.get()) / "data" / "01_raw_receptors"), filetypes=[("PDB structure", "*.pdb"), ("All files", "*.*")])
        if selected:
            self.quick_receptor_var.set(selected)

    def _quick_run(self) -> None:
        if self._busy():
            return
        receptor = self.quick_receptor_var.get().strip()
        ligands = [line.strip() for line in self.quick_ligands_var.get().replace(";", "\n").splitlines() if line.strip()]
        if not receptor or not ligands:
            messagebox.showerror("输入不完整", "请选择受体 PDB，并至少填写一个“名称=SMILES”配体。")
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
                    self._append(value)
                elif kind == "done":
                    self.status_var.set("已完成" if value == "0" else f"失败（退出码 {value}）")
                    self._refresh_gate()
                    self._refresh_results()
                elif kind == "chain":
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
            labels = {"exploratory override": "探索性覆盖", "standard or reviewed": "标准/已复核"}
            self.gate_var.set(labels.get(status, status))
        except (FileNotFoundError, json.JSONDecodeError):
            self.gate_var.set("安全闸门：未检查")

    def _refresh_results(self) -> None:
        reports = Path(self.root_var.get()) / "reports"
        try:
            state = json.loads((reports / "receptor_state.json").read_text(encoding="utf-8"))
            box = json.loads((reports / "docking_box_audit.json").read_text(encoding="utf-8"))
            state_label = {"ready_for_preparation": "受体准备就绪", "manual_review_required": "受体需人工复核"}.get(state.get("status"), state.get("status", "未知"))
            box_label = {"box_passed": "box 覆盖通过", "manual_review_required": "box 需人工复核"}.get(box.get("status"), box.get("status", "未知"))
            self.audit_var.set(f"{state_label}；{box_label}")
        except (FileNotFoundError, json.JSONDecodeError):
            self.audit_var.set("尚未读取受体/box 审计")
        try:
            plan = json.loads((reports / "execution_plan.json").read_text(encoding="utf-8"))
            self.plan_var.set(f"{plan.get('receptor_count', 0)} 个受体 × {plan.get('ligand_count', 0)} 个配体；计划 {plan.get('planned_run_count', 0)} 次")
        except (FileNotFoundError, json.JSONDecodeError):
            self.plan_var.set("执行计划：未生成")
        try:
            import csv
            rows = list(csv.DictReader((reports / "consensus_summary.csv").open(newline="", encoding="utf-8")))
            unstable = sum(row.get("Stability_Status") == "unstable_manual_review_required" for row in rows)
            self.result_var.set(f"{len(rows)} 个配体；不稳定 {unstable} 个")
        except (FileNotFoundError, OSError):
            try:
                rows = list(csv.DictReader((reports / "docking_summary.csv").open(newline="", encoding="utf-8")))
                fallback = sum(row.get("Selection_Status") == "Affinity_Fallback_No_Catalytic_Pose" for row in rows)
                self.result_var.set(f"{len(rows)} 个配体；催化 fallback {fallback} 个")
            except (FileNotFoundError, OSError):
                self.result_var.set("结果：未生成")

    def _busy(self) -> bool:
        if self.process and self.process.poll() is None:
            messagebox.showwarning("任务运行中", "请等待当前任务完成。")
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
        os.startfile(str(target))


if __name__ == "__main__":
    DockingWorkbench().mainloop()
