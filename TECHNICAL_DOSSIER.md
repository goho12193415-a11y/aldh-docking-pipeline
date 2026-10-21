# ALDH Docking Pipeline 技术档案

## 1. 项目定位

本项目是一个面向 Windows 的、带科学安全闸门的机制引导型分子对接科研原型。

当前最适合的任务：

- ALDH/醛脱氢酶体系的机制引导小分子初筛；
- 普通蛋白—小分子 docking 的自动准备、运行、筛选和可视化；
- 催化残基、辅因子、docking box 和配体输入的自动审计；
- 生成可复现的 docking 记录、结果报告和结构图；
- 为多糖、金属配体、共价配体、MD、QM/MM 和实验校准提供安全的扩展入口。

当前不应将本项目描述为：

- 任意酶和任意配体都能可靠处理的通用平台；
- 可以直接给出实验结合自由能的工具；
- 可以直接证明酶活性、共价反应或反应能垒的工具；
- 已经完成真实 MD、QM/MM、金属配位或多引擎 docking 的平台。

核心原则：

> 程序能够运行，不等于结果能够解释；结果能够解释，也不等于结论已经被实验验证。

---

## 2. 当前能力等级

| 能力范围 | 当前状态 |
|---|---|
| ALDH 专用机制引导 docking | 已实现，当前核心能力 |
| 普通蛋白—小分子初筛 | 可用，但仍需人工复核 |
| 受体/配体自动准备 | 已实现基础流程 |
| 催化几何筛选 | 已实现 ALDH 规则 |
| 多糖识别和构象准备 | 已实现初步流程，不等于糖类专用可靠模拟 |
| 金属/盐/自由基/多电荷识别 | 已实现审计和安全阻断 |
| 受体状态审计 | 已实现 |
| docking box 覆盖审计 | 已实现 |
| 重复 seed 稳定性统计 | 已实现基础版本 |
| 多受体 execution plan | 已实现计划层，未完成全矩阵实际执行 |
| 通用酶机制模板 | 已有框架；只有 ALDH 评分真正实现 |
| pKa/质子化自动枚举 | 尚未完成；当前是化学状态审计 |
| 受体自动修复 | 尚未完成；当前是修复需求审计 |
| 真实 MD/QM/MM/共价计算 | 尚未完成；当前是准备和 readiness 层 |
| GUI 结果分析 | 已有摘要卡片，仍非完整三维分析工作台 |

---

## 3. 目录结构

```text
ALDH_Docking_Pipeline/
├── config/
│   ├── pipeline_config.yaml
│   └── mechanisms/
│       ├── aldh.yaml
│       ├── serine_hydrolase.yaml
│       ├── cysteine_protease.yaml
│       └── metalloenzyme.yaml
├── data/
│   ├── 01_raw_receptors/
│   ├── 03_prepped_receptors/
│   ├── 04_prepped_ligands/
│   └── 05_docking_outputs/
├── reports/
│   ├── docking_summary.csv
│   ├── consensus_summary.csv
│   ├── ligand_compatibility.json
│   ├── chemical_state.json
│   ├── receptor_quality.json
│   ├── receptor_state.json
│   ├── structure_repair_audit.json
│   ├── enzyme_readiness.json
│   ├── docking_box_audit.json
│   ├── multi_box_plan.json
│   ├── flexibility_plan.json
│   ├── engine_consensus_plan.json
│   ├── execution_plan.json
│   ├── workflow_readiness.json
│   ├── capability_report.json
│   ├── capability_gate.json
│   ├── system_coverage.json
│   ├── run_manifest.json
│   └── run_history.sqlite3
├── scripts/
│   ├── gui_app.py
│   ├── quick_start.py
│   ├── validate_outputs.py
│   ├── evaluate_experimental.py
│   └── download_pdb.py
├── src/
│   ├── common.py
│   ├── mechanism_templates.py
│   ├── system_upgrade.py
│   ├── step1_ligand_prep.py
│   ├── step2_receptor_prep.py
│   ├── step3_box_locator.py
│   ├── step4_vina_runner.py
│   ├── step5_mechanism_scorer.py
│   ├── step6_interaction_viz.py
│   ├── step7_advanced.py
│   ├── step8_dynamics.py
│   ├── step9_qmmm.py
│   └── step10_evaluation.py
├── tests/
│   └── test_geometry.py
├── main.py
├── README.md
├── GUI_USER_GUIDE.md
├── requirements.txt
├── environment.yml
└── 启动可视化程序.bat
```

运行产生的 `data/` 和 `reports/` 文件属于实验产物，不应直接当作源代码维护。继续开发时，应优先修改 `src/`、`scripts/`、`config/`、`tests/` 和文档。

---

## 4. 主流程

### 4.1 CLI 流程

入口：

```powershell
python main.py --config config\pipeline_config.yaml
```

主流程顺序：

1. 读取 YAML 配置；
2. 加载机制模板；
3. 如果开启，自动识别催化 Cys；
4. 生成 docking box；
5. 审计受体结构质量和修复需求；
6. 审计配体兼容性、酸碱位点、盐形式和电荷状态；
7. 审计酶活性位点、辅因子、结晶水和 alternate location；
8. 审计 docking box 是否覆盖关键原子；
9. 生成多受体/多 seed execution plan；
10. 生成多 box、柔性受体、多引擎和外部工作流计划；
11. 生成能力报告和覆盖矩阵；
12. 通过安全闸门后准备受体和配体；
13. 调用 AutoDock Vina；
14. 解析所有 docking modes；
15. 按 ALDH 催化几何重新筛选 pose；
16. 输出 interaction CSV、active pose、PyMOL PML/PSE/PNG；
17. 可选生成重复 seed 汇总、PLIP/MD/QM/MM 准备文件；
18. 写入 run manifest 和 SQLite 运行历史。

### 4.2 常用命令

```powershell
# 依赖和输入预检
python main.py --preflight

# 只执行受体/配体准备和审计，不运行 Vina
python main.py --config config\pipeline_config.yaml --skip-vina

# 完整标准流程
python main.py --config config\pipeline_config.yaml

# 显式允许专用体系的探索性运行
python main.py --config config\pipeline_config.yaml --allow-specialized

# 输出验收
python scripts\validate_outputs.py

# 快速生成配置
python scripts\quick_start.py `
  --receptor data\01_raw_receptors\receptor.pdb `
  --ligand Hexanal=CCCCC=O
```

### 4.3 GUI

```powershell
python scripts\gui_app.py
```

也可以双击：

```text
启动可视化程序.bat
```

GUI 默认中文，提供：

- 快速实验；
- 依赖预检；
- 只做准备；
- 完整运行；
- 输出验收；
- 结果目录和配置目录打开；
- 运行日志；
- 结果与审计摘要；
- 安全闸门状态。

---

## 5. 配置说明

主配置文件：

```text
config/pipeline_config.yaml
```

### 5.1 safety

```yaml
safety:
  strict_capability_gate: true
  experimental_ph: 7.4
```

- `strict_capability_gate: true`：专用体系或未实现机制默认阻断；
- `experimental_ph`：实验 pH 记录和化学状态审计参数，不是 pKa 计算。

### 5.2 receptor

关键字段：

- `input_pdb`：受体输入 PDB；
- `chain_id`：目标蛋白链；
- `cofactor_chain`：辅因子所在链；
- `keep_cofactors`：要保留的辅因子；
- `catalytic_residue`：催化残基；
- `oxyanion_residue`：氧阴离子穴残基；
- `template_pdb`：可选同源模板。

### 5.3 enzyme

```yaml
enzyme:
  mechanism_family: aldh
  requires_cofactor: true
  flexible_residues: []
  conserved_waters: []
```

目前 `aldh` 是唯一具有完整机制评分路径的模板。其他模板默认只提供审计和安全阻断。

### 5.4 docking_box

```yaml
docking_box:
  size_x: 18.0
  size_y: 18.0
  size_z: 18.0
  center_mode: midpoint_if_cofactor
  candidates: []
```

`candidates` 可以配置多个 box，但当前主要生成多 box 计划，不会自动将不同 box 的原始分数混合。

### 5.5 advanced

主要功能：

- `auto_detect_catalytic_cys`
- `ligand_variants`
- `polysaccharide`
- `receptor_ensemble`
- `replicates`
- `plip`
- `md`
- `qmmm`
- `covalent`
- `engines`
- `evaluation`

启用外部流程前，必须确认外部软件、参数和结果导入方式已经配置。

---

## 6. 核心算法和科学规则

### 6.1 配体兼容性

当前识别：

- 金属/无机元素；
- 同位素；
- 自由基；
- 大分子；
- 高柔性配体；
- 多重电荷；
- 糖样分子；
- 潜在共价 warhead；
- 酸碱位点；
- 盐和多片段输入。

状态：

- `standard_small_molecule`
- `manual_review_required`
- `specialized_method_required`

普通硫醚不能仅因为含硫而被误判为共价 warhead，测试已经覆盖该行为。

### 6.2 ALDH 催化筛选

当前核心几何：

- 配体羰基 C—O 键长度必须在配置区间；
- 羰基 C 到催化 Cys-SG 的距离必须满足 `d_nuc_min` 和 `d_nuc_max`；
- 存在 NAD/NAP/NDP 时，羰基 C 到 `C4N/C4` 的距离必须小于 `d_hyd_max`；
- 有合格 pose 时标记 `Catalytically_Competent`；
- 没有合格 pose 时标记 `Affinity_Fallback_No_Catalytic_Pose`。

注意：催化几何合格只代表“满足筛选规则”，不证明真实反应发生。

### 6.3 安全闸门

严格闸门会阻止：

- 专用方法需求的配体；
- 未实现机制模板；
- box 关键原子未覆盖；
- 明显不完整的外部工作流条件。

显式 `--allow-specialized` 只代表探索性覆盖，不代表结果科学可靠。

---

## 7. 报告文件数据契约

| 文件 | 作用 |
|---|---|
| `docking_summary.csv` | 每个配体主结果、affinity、催化状态和 active pose |
| `consensus_summary.csv` | 重复 seed 均值、标准差、催化比例和稳定性 |
| `ligand_compatibility.json` | 配体化学复杂度和适用性 |
| `chemical_state.json` | pH、酸碱位点、盐形式和电荷审计 |
| `receptor_quality.json` | B-factor/pLDDT proxy |
| `receptor_state.json` | 辅因子、金属、水和 alternate location 审计 |
| `structure_repair_audit.json` | 缺失主链原子、序列 gap、alternate location |
| `enzyme_readiness.json` | 催化残基和辅因子 readiness |
| `docking_box_audit.json` | 关键原子是否位于 box 内 |
| `execution_plan.json` | 受体 × 配体 × seed 计划 |
| `multi_box_plan.json` | 多 box 计划 |
| `flexibility_plan.json` | 柔性残基和保守水计划 |
| `engine_consensus_plan.json` | 多引擎计划和分数归一化规则 |
| `workflow_readiness.json` | MD/QM/MM/共价/PLIP/实验接口状态 |
| `capability_report.json` | 当前运行允许和不允许的科学声明 |
| `capability_gate.json` | 是否阻断、是否 override、阻断原因 |
| `run_manifest.json` | 配置 SHA-256、运行模式、报告路径和审计状态 |
| `system_coverage.json` | 14 项升级的覆盖矩阵 |
| `run_history.sqlite3` | 本地运行历史 |

---

## 8. 14 项空白的当前真实状态

| # | 空白 | 当前实现 | 后续真正完成条件 |
|---:|---|---|---|
| 1 | 质子化/pKa | 化学状态审计 | 集成可靠 pKa/质子化工具并验证电荷 |
| 2 | 受体自动修复 | 修复需求审计 | 集成 PDBFixer/Modeller 等并做结构质量复核 |
| 3 | 通用酶机制 | 模板框架，ALDH 已实现 | 每类机制都有专用 box、几何和评分器 |
| 4 | 多受体 ensemble | execution plan | 实际执行矩阵和跨结构 pose 一致性分析 |
| 5 | 多 box/口袋 | 多 box 计划 | 集成口袋发现和独立 box 批量 docking |
| 6 | 受体柔性 | 柔性配置审计 | rotamer/柔性受体引擎和局部优化 |
| 7 | 金属/水/辅因子 | 检测和审计 | 价态、配位、参数化和水网络验证 |
| 8 | 多引擎共识 | 引擎计划和分数隔离 | 真实多引擎运行、pose 映射和排名校准 |
| 9 | 多糖 | 构象生成和安全闸门 | 糖专用参数、采样、MD 和分支处理 |
| 10 | 共价 docking | warhead 检测和 readiness | 共价前后状态、成键和 QM/MM |
| 11 | MD/QM/MM/PLIP | 准备和接口层 | 工具检测、真实运行、轨迹分析和结果回写 |
| 12 | 实验校准 | Pearson/Spearman/ROC/PR/enrichment/bootstrap 工具 | 外部数据、scaffold split 和独立验证 |
| 13 | GUI 分析 | 摘要卡片和日志 | 三维预览、结果表、box 预览、历史对比 |
| 14 | 项目管理 | SQLite 运行历史 | 版本隔离、断点恢复、失败重试和任务队列 |

---

## 9. 当前已知限制和技术债

1. `Consensus_Score` 仍是研究型组合分数，不是多引擎物理共识；
2. 多受体计划不会自动执行完整 receptor × ligand × seed 矩阵；
3. `assess_enzyme_readiness()` 对跨链辅因子的科学语义仍需更精细；
4. 辅因子移植仍缺少序列对齐、RMSD 门槛和碰撞检查；
5. interaction 分析仍有按残基编号识别造成链歧义的风险；
6. 多糖没有真实端到端专用力场/MD 验证；
7. PLIP、GROMACS/OpenMM、ORCA 等外部程序未必安装；
8. GUI 尚未提供真正的三维交互式结构查看；
9. 运行历史目前是轻量 SQLite，不是完整实验数据库；
10. 当前项目没有 Git 仓库，继续开发时建议先建立版本控制。

---

## 10. 环境和依赖

已验证环境：

- Windows；
- Python 3.11；
- RDKit 约 2022.09.5；
- NumPy 1.26.4；
- Meeko；
- Biopython；
- pandas；
- PyYAML；
- OpenBabel；
- AutoDock Vina 1.2.7；
- Schrodinger PyMOL Windows 版。

重要兼容性注意：

- Python PyMOL wheel 可能引入 NumPy 2.x，导致 RDKit 二进制冲突；
- 当前推荐使用独立 PyMOL Windows 程序进行无头渲染；
- 旧 RDKit 与 Meeko 之间存在 `Mol.HasQuery` 兼容 workaround；
- 不要在没有确认版本兼容性的情况下升级 RDKit、NumPy 或 PyMOL。

---

## 11. 验证状态

最近验证结果：

```text
23 项 unittest：通过
Python 编译检查：通过
Tkinter 检查：通过
main.py --preflight：通过
main.py --skip-vina：通过
scripts/validate_outputs.py：通过
自动催化 Cys 流程：通过
自动链选择：通过
严格 capability gate：通过
14 项 system coverage：已生成
```

标准样例：

- 受体：`1O01` 派生样例；
- 配体：Hexanal、Nonanal、Octanal；
- 当前样例 docking 结果均为 `Affinity_Fallback_No_Catalytic_Pose`；
- 这些结果不能称为催化活性构象。

---

## 12. 建议的后续开发顺序

### 第一阶段：科学输入可靠性

1. 集成 pKa/质子化枚举；
2. 集成受体结构修复；
3. 统一金属、辅因子和保守水的状态模型；
4. 增加结构碰撞和缺失原子阻断。

### 第二阶段：真正批量运行

1. 执行 receptor × box × ligand × conformer × seed 矩阵；
2. 输出每次运行独立目录；
3. 加入失败重试和断点恢复；
4. 做跨受体 pose 聚类和稳定性分析。

### 第三阶段：机制扩展

1. 丝氨酸水解酶；
2. 半胱氨酸蛋白酶；
3. 金属酶；
4. 激酶；
5. 糖苷酶；
6. 共价抑制剂。

### 第四阶段：真实物理和实验闭环

1. 多引擎 docking；
2. PLIP；
3. MD；
4. MM/GBSA 或 MM/PBSA；
5. QM/MM；
6. 实验数据校准和独立验证。

### 第五阶段：GUI 和工程化

1. 三维结构预览；
2. docking box 可视化；
3. 结果表格和 pose 切换；
4. 运行历史比较；
5. 取消、重试和断点恢复；
6. 建立 Git 仓库和持续集成。

---

## 13. 交接给下一位开发者的最短说明

先执行：

```powershell
cd "C:\Users\go ho\Desktop\ALDH_Docking_Pipeline"
python main.py --preflight
python main.py --config config\pipeline_config.yaml --skip-vina
python scripts\validate_outputs.py
python -m unittest discover -s tests -v
```

然后阅读以下文件：

1. `README.md`
2. `TECHNICAL_DOSSIER.md`
3. `config/pipeline_config.yaml`
4. `main.py`
5. `src/step7_advanced.py`
6. `src/system_upgrade.py`
7. `reports/system_coverage.json`
8. `reports/capability_report.json`
9. `reports/capability_gate.json`

继续开发时必须遵守：

- 不要把审计文件写成“已完成真实模拟”的表述；
- 不要删除现有安全闸门；
- 新增机制必须同时新增测试、能力声明和边界说明；
- 新增外部工具必须加入 preflight；
- 任何新结果都要写入 run manifest；
- 不要把不同受体或不同评分引擎的原始分数直接平均；
- 先保证结果可追溯，再增加自动化程度。

