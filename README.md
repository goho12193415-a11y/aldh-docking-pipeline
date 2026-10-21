# ALDH Docking Pipeline

这是一个面向 Windows 的机制引导型 AutoDock Vina 管道，覆盖：

1. PubChem/RDKit/Meeko 配体准备；
2. 受体清理、辅酶保留或基于 C-alpha 对齐的 NAD 移植；
3. 基于催化 Cys 与 NAD C4/C4N 的 docking box 定位；
4. 安全的 Windows Vina 调用；
5. 全部 docking mode 的催化几何重排序；
6. 相互作用表、标准化复合物 PDB、PyMOL 脚本和高清渲染。

## 环境

推荐使用 Conda：

```powershell
conda env create -f environment.yml
conda activate aldh-docking
```

必须另外准备：

- AutoDock Vina 1.2.x，并在 `config/pipeline_config.yaml` 设置路径；
- OpenBabel 的 `obabel` 命令；
- PyMOL 3.x，用于自动生成 PNG/PSE。Windows 可安装：

```powershell
winget install --id Schrodinger.PyMOL
```

默认配置使用 PATH 中的 `vina` 和 `pymol` 命令，适合跨机器运行。如果可执行文件不在 PATH 中，请把这两个字段改为绝对路径。Windows 下如果使用 `vina.exe`，也可以直接填写 `vina.exe`。

如果当前配置的 PyMOL 入口是 `PyMOLWin.exe`，管道会自动调用其旁边的 Python 解释器进行无头渲染。

## 输入

将受体放入 `data/01_raw_receptors/receptor.pdb`，然后编辑：

- `config/pipeline_config.yaml`
- 受体链、催化 Cys 编号、oxyanion-hole 残基；
- 配体名称和 SMILES；
- Vina 路径。

AlphaFold apo 结构可设置 `receptor.template_pdb`，模板应包含同源 NAD/NAP/NDP。

也可以直接下载 RCSB 结构，例如：

```powershell
python scripts/download_pdb.py 1O01
```

## 运行

### 图形界面

Windows 用户可以双击项目根目录的 `启动可视化程序.bat`，或运行：

```powershell
python scripts\gui_app.py
```

GUI 支持选择项目目录/配置文件、预检依赖、只做准备、完整运行、验收输出和打开结果目录。完整的按钮说明、结果位置和常见问题见 [`GUI_USER_GUIDE.md`](GUI_USER_GUIDE.md)。

推荐使用 GUI 顶部的“快速实验”：只选择受体 PDB，并填写 `名称=SMILES`，点击“一键生成并运行”。程序会自动选择主要蛋白链、尝试识别催化 Cys、生成 `config/quick_config.yaml`，然后执行完整流程。高级配置仍可在下方手动调整。

先运行预检：

```powershell
python main.py --preflight
```

先只验证配体和受体准备：

```powershell
python main.py --config config/pipeline_config.yaml --skip-vina
```

完整运行：

```powershell
python main.py --config config/pipeline_config.yaml
```

运行结果验收：

```powershell
python scripts/validate_outputs.py
```

`Catalytically_Competent=True` 要求亲核进攻距离合格，并且在受体保留 NAD/NADP 时同时满足羰基碳到 `C4N/C4` 的氢负离子转移距离阈值。如果没有任何 mode 同时满足，管道会明确写入 `Selection_Status=Affinity_Fallback_No_Catalytic_Pose`；此时导出的结构只能作为最低结合能参考构象，不能称为催化活性构象。

输出位于 `reports/`：

- `docking_summary.csv`
- `active_poses/*_active.pdb`
- `interactions/*_interactions.csv`
- `figures/*_render.pml`
- `figures/*_session.pse`
- `figures/*_300dpi.png`

## 重要限制

该项目不会在缺少 RDKit、Meeko、OpenBabel、Vina 或输入 PDB 时伪造成功结果；会直接给出可定位的错误。催化几何判据是结构筛选规则，不等价于实验活性或真实反应能垒，最终结论应结合晶体学、QM/MM 或实验验证。

## 高级升级开关

`config/pipeline_config.yaml` 的 `advanced` 区域提供：

- 催化 Cys 基序自动识别；
- 配体互变异构体枚举接口；
- 配体复杂度、金属、自由基、多电荷和潜在共价反应预检；
- 多糖/寡糖识别与专用流程阻断提示；
- 多糖立体化学检查、糖样多构象生成和逐构象对接入口；
- 酶活性位点、辅因子、结晶水和 alternate location readiness 审计；
- 多受体 ensemble 配置入口；
- 重复对接 seed 配置和共识统计接口；
- 可选 PLIP 交叉验证；
- MD 准备清单和 manifest；
- QM/MM/ORCA 输入草稿；
- 实验活性 CSV 对齐评估接口。

这些模块不会在缺少 PLIP、GROMACS、ORCA 或实验数据时伪造结果；它们只生成可审计的准备文件或明确报告不可用原因。

当前版本中，重复对接会实际传递 Vina seed，并输出均值、标准差、催化构象比例和 `Consensus_Score`；多受体 ensemble 已加入配置/预检入口，实际批量 ensemble 运行仍建议按受体分别执行并比较结果，避免把不同结构的分数直接混为一个物理量。

每次运行还会生成 `reports/ligand_compatibility.json`。`standard_small_molecule` 只表示适合标准 Vina 初筛，不表示实验活性；含金属、自由基、多重电荷、超大/高度柔性或潜在共价反应的配体会标记为 `manual_review_required`，不会被静默伪装成普通小分子结果。

### 多糖和寡糖的边界

当前项目**不能完全覆盖多糖分子对接**。它可以在 RDKit/Meeko 能处理输入时生成初步构象并运行 Vina，但这不等于多糖结果可靠。多糖通常需要糖残基级别的立体化学核对、糖苷键构象采样、分支构象、质子化/离子状态、溶剂和受体柔性处理；超出标准小分子 Vina 的稳健范围。检测到糖样分子后会在兼容性报告中标记 `specialized_method_required`，建议使用糖类专用构象生成、RosettaCarbohydrate、GLYCAM/Amber、MD 或实验结构约束进行复核。

启用 `advanced.polysaccharide` 后，项目会拒绝未指定手性中心的糖样 SMILES，使用 ETKDGv3/小环扭转约束生成一组带来源记录的构象，写入 `data/04_prepped_ligands/polysaccharides/*__sugar_conformers.json`，并可逐构象调用 Vina，结果写入 `reports/polysaccharides/`。逐构象结果不会被伪装成一个“最佳结合能”；正式定量结论仍需要 GLYCAM/Amber、MD、实验结构或其他糖类专用方法复核。

同时生成 `reports/capability_report.json`，给出本次配置实际支持的声明、不能支持的声明，以及 `allow_initial_screening`、`allow_with_review` 或 `block_interpretation` 建议。这样“程序能运行”和“结果能解释”被明确分开。

## 当前自动化改进层

每次运行还会生成以下可追溯审计文件：

- `reports/receptor_state.json`：alternate location、辅因子、金属和水的受体状态审计；
- `reports/chemical_state.json`：实验 pH、酸碱位点、盐形式和显式电荷的化学状态审计；
- `reports/docking_box_audit.json`：催化 Cys、辅因子关键原子和 box 尺寸覆盖审计；
- `reports/execution_plan.json`：受体、配体、构象/seed 组合的结构化执行计划；
- `reports/consensus_summary.csv`：重复 seed 的均值、标准差、催化合格比例和稳定性等级；
- `reports/structure_repair_audit.json`：缺失主链原子、序列 gap 和 alternate location 审计；
- `reports/multi_box_plan.json`：多 box 实验计划；
- `reports/flexibility_plan.json`：柔性残基和保守水配置审计；
- `reports/engine_consensus_plan.json`：多评分引擎计划，按引擎排名而非直接平均原始分数；
- `reports/workflow_readiness.json`：多糖、共价、MD、QM/MM、PLIP 和实验评估接口状态；
- `reports/run_history.sqlite3`：本地运行历史。
- `reports/system_coverage.json`：14 项空白的机器可读覆盖矩阵。
- `reports/enzyme_readiness.json`：催化残基、辅因子和结构准备状态；
- `reports/capability_gate.json`：是否阻止了专用体系的标准解释；
- `reports/run_manifest.json`：配置 SHA-256、机制模板、闸门状态和运行模式。

机制模板位于 `config/mechanisms/`。目前 ALDH 模板是已实现评分的安全路径；丝氨酸水解酶、半胱氨酸蛋白酶和金属酶模板目前只提供准备/审计接口，默认不会被当作 ALDH 规则运行。严格闸门会阻止这些模板进入标准 docking；只有显式 `--allow-specialized` 才能进行探索性运行。

快速实验会比较每条蛋白链的辅因子、催化 Cys 和蛋白规模，并用确定性规则选择目标链；候选证据会写入 `config/quick_config.yaml` 的 `project.auto_selection`。

配置中的 `safety.experimental_ph` 用于记录实验 pH，并触发保守的配体状态检查。项目会识别潜在酸碱位点、盐/多片段输入和显式电荷，但不会伪造 pKa 或自动生成经过验证的质子化组合；出现这些情况时会把配体升级为 `manual_review_required`。

程序还会在 Vina 前检查 docking box 是否覆盖催化 Cys 和辅因子关键原子，以及尺寸是否适合聚焦活性位点采样。覆盖失败会触发严格闸门，而不是继续输出容易误读的结果。

当启用 `advanced.receptor_ensemble` 或重复 seed 时，程序会自动生成 `execution_plan.json`，列出每个受体、配体和 replicate 的 `run_id`、seed 与比较组。该文件是执行计划和审计记录，不会把不同受体的 Vina 分数直接合并成一个物理量；真正的 ensemble 批量执行和跨结构 pose 一致性分析仍属于后续扩展。

当 `advanced.replicates.count > 1` 时，`consensus_summary.csv` 还会报告 `Affinity_SD_kcal_mol`、`d_nuc_SD_A`、`d_hyd_SD_A` 和 `Stability_Status`。`unstable_manual_review_required` 表示不同 seed 的结果不一致，不能只按平均分数下结论；这是保守启发式审计，不是统计置信区间。

GUI 的“结果与审计摘要”会自动读取受体状态、box 审计、执行计划和结果汇总：重复 seed 时显示不稳定配体数量；单次运行时显示催化 fallback 数量。详细结论仍以 `reports` 中的原始报告为准。

本轮系统升级还会为受体修复、柔性受体、多 box、多评分引擎、外部 MD/QM/MM/共价工作流和运行历史生成独立报告。它们优先提供真实的审计、计划和阻断条件；没有实际外部工具或成功轨迹时，不会把 manifest 当成模拟结果。

`system_coverage.json` 会明确区分“已实现审计”“已生成计划”“外部工作流接口”和“仍需外部验证”，因此项目不会把 14 项空白简单标记为已经完成。

## 自动适用性闸门

默认 `safety.strict_capability_gate: true`。程序会在 Vina 前自动检查配体和酶体系：

- `standard_small_molecule`：允许标准流程；
- `manual_review_required`：允许运行，但报告会标记需要人工复核；
- `specialized_method_required`：默认阻止标准对接，避免生成容易被误解的结果。

如果只是进行非标准探索性尝试，必须显式运行 `python main.py --allow-specialized`。程序会生成 `reports/capability_gate.json` 并标记 `exploratory_override`；这不代表结果变得可靠。

### 酶类对接优化边界

酶类任务会额外生成 `reports/enzyme_readiness.json`，检查配置的催化/酸碱/稳定残基是否存在且残基名匹配、目标链是否保留所需辅因子、结构中的结晶水数量、alternate location，以及是否声明需要辅因子。`enzyme.flexible_residues` 和 `enzyme.conserved_waters` 是后续柔性残基/保守水建模的显式配置入口；当前 readiness 报告不会假装完成质子化、金属配位、诱导契合或真实反应状态验证。
