# ALDH 分子对接工作台图形界面使用说明

当前 GUI 已提供完整中文界面。技术字段（例如 `SMILES`、`PDB`、`Vina`、`PDBQT` 和 `reports`）保留英文，是文件格式、软件名称或项目目录名称。

## 1. 启动

在 Windows 中双击项目根目录的：

```text
启动可视化程序.bat
```

也可以在 PowerShell 中运行：

```powershell
cd "C:\Users\go ho\Desktop\ALDH_Docking_Pipeline"
python scripts\gui_app.py
```

图形界面使用 Python 自带的 Tkinter，不需要额外安装 Web 框架。

## 2. 运行前准备

### 最简单的快速模式

GUI 顶部“快速实验”只需要两项：

1. 选择受体 PDB；
2. 每行填写一个 `名称=SMILES`，例如 `Hexanal=CCCCC=O`。

点击“一键生成并运行”后，程序会自动生成 `config\quick_config.yaml`，自动选择主要蛋白链，并尝试识别催化 Cys，然后直接运行完整流程。只有在自动识别失败或体系特殊时，才需要进入高级配置。

命令行等价方式：

```powershell
python scripts\quick_start.py --receptor data\01_raw_receptors\receptor.pdb --ligand Hexanal=CCCCC=O
```

快速模式会自动检查各条链的辅因子和催化残基，并将选择理由写入 `config\quick_config.yaml`。运行完成后，可在 `reports` 中查看 `receptor_state.json`、`capability_gate.json` 和 `run_manifest.json`，确认程序没有把需要专用方法的体系当成普通小分子结果解释。

1. 将受体 PDB 放入 `data\01_raw_receptors\receptor.pdb`，或在配置文件中修改 `receptor.input_pdb`。
2. 在 `config\pipeline_config.yaml` 中填写蛋白链、催化残基、辅因子、配体名称和 SMILES。
3. 确认 `vina.executable`、PyMOL 路径和 `obabel` 可执行文件可用。
4. 先点击 **预检依赖**，确认所有依赖显示 `OK`。

## 3. 四个主要按钮

| 按钮 | 用途 |
|---|---|
| 预检依赖 | 检查受体、Python 包、Vina、OpenBabel 和 PyMOL |
| 只做准备 | 生成配体/受体 PDBQT、兼容性报告和酶 readiness 报告，不运行 Vina |
| 完整运行 | 执行配体准备、受体准备、box 定位、Vina、机制筛选、相互作用分析和 PyMOL 输出 |
| 验收输出 | 检查 summary、active pose、interaction CSV、PML/PSE/PNG 及审计报告是否完整 |

运行日志会显示在窗口下方。任务运行期间不要重复点击其他运行按钮；程序会阻止并行启动，以免覆盖相同输出文件。

程序默认启用自动适用性闸门。多糖、金属配合物、潜在共价体系等会在 Vina 前被阻止，而不是生成容易误读的标准结果。需要探索性运行时，请使用命令行 `python main.py --allow-specialized`，并查看 `reports\capability_gate.json`。

## 4. 结果位置

- `reports\docking_summary.csv`：主要对接汇总；
- `reports\active_poses\`：筛选出的复合物 PDB；
- `reports\interactions\`：相互作用明细；
- `reports\figures\`：PyMOL 脚本、会话和 PNG；
- `reports\ligand_compatibility.json`：配体适用性审计；
- `reports\enzyme_readiness.json`：酶活性位点/辅因子/水/alternate location 审计；
- `reports\capability_report.json`：本次运行允许和不允许的科学解释；
- `reports\chemical_state.json`：实验 pH、酸碱位点、盐形式和显式电荷审计；
- `reports\docking_box_audit.json`：催化位点、辅因子关键原子和 box 尺寸覆盖审计；
- `reports\execution_plan.json`：受体、配体和 seed 组合的自动执行计划；
- `reports\consensus_summary.csv`：重复 seed 的波动和稳定性审计；
- `reports\structure_repair_audit.json`：受体缺失原子、gap 和 alternate location 审计；
- `reports\multi_box_plan.json`：多 box 计划；
- `reports\flexibility_plan.json`：柔性残基和保守水计划；
- `reports\engine_consensus_plan.json`：多引擎计划；
- `reports\workflow_readiness.json`：外部工作流状态；
- `reports\run_history.sqlite3`：运行历史。
- `reports\system_coverage.json`：14 项升级内容的覆盖矩阵。
- `reports\receptor_state.json`：受体 alternate location、辅因子、金属和水审计；
- `reports\run_manifest.json`：本次配置指纹、机制模板和运行模式；
- `data\04_prepped_ligands\polysaccharides\`：多糖构象集合和 manifest；
- `reports\polysaccharides\`：多糖逐构象对接结果。

## 5. 如何解释结果

`Catalytically_Competent=True` 仅表示该构象满足当前配置的几何筛选，不等于真实酶活性。`Affinity_Fallback_No_Catalytic_Pose` 表示没有构象满足严格催化几何，只导出了最低 Vina 能量参考构象。

对于多糖、金属配合物、自由基、强多电荷体系、潜在共价抑制剂、超大柔性配体，必须先阅读 `ligand_compatibility.json` 和 `capability_report.json`。如果出现 `specialized_method_required` 或 `block_interpretation`，不能只依据 Vina 分数下结论。

如果配体含有可电离基团、盐形式或显式电荷，还应查看 `chemical_state.json`。`safety.experimental_ph` 只记录实验 pH 并触发审计，不代表程序已经完成 pKa 预测或可靠的质子化枚举。

如果启用了多受体或重复 seed，请先查看 `execution_plan.json`。它用于确认组合数量和输入路径；不同受体的分数仍需分结构比较，不能直接混合解释。

重复 seed 运行后，还应查看 `consensus_summary.csv`。如果 `Stability_Status` 为 `unstable_manual_review_required`，说明不同 seed 的分数或催化几何波动较大，不能仅依据均值排序。

GUI 中央的“结果与审计摘要”会自动显示受体/box 审计状态、执行计划规模和结果稳定性提示；它是快速导航，不替代 `reports` 下的详细 JSON/CSV 报告。

## 6. 常见问题

### 窗口无法启动

确认 Python 可用，并在 PowerShell 中执行：

```powershell
python -c "import tkinter; print('Tkinter OK')"
```

### 预检失败

按日志中标出的路径修正 `config\pipeline_config.yaml`，或安装缺少的工具。GUI 不会跳过依赖检查。

### 完整运行失败

先点击 **只做准备** 定位问题，再查看日志和 `reports\` 中的审计文件。不要直接删除全部结果；保留失败日志有助于追踪输入或工具问题。
