from __future__ import annotations

from pathlib import Path

import yaml

BUILTIN_MECHANISM_TEMPLATES: dict[str, dict] = {
    "aldh": {
        "name": "aldh",
        "display_name": "醛脱氢酶（ALDH）",
        "status": "implemented",
        "description": "催化半胱氨酸与醛底物发生亲核加成，结合氧阴离子穴稳定化，并由 NAD/NADP 辅因子参与氢负离子转移（自动识别辅因子 in/out 构象）。",
        "required_roles": {
            "catalytic_residue": {"residue_names": ["CYS"], "required_atoms": ["SG"]},
            "cofactor": {"residue_names": ["NAD", "NAP", "NDP"], "required_atoms": ["C4N", "C4"]},
        },
        "box_strategy": "catalytic_residue_to_cofactor_midpoint",
        "scoring_strategy": "aldh_nucleophile_carbonyl_hydride_geometry",
        "default_thresholds": {"d_nuc_min": 3.0, "d_nuc_max": 4.5, "d_hyd_max": 4.5, "d_oxy_max": 5.0},
        "interpretation_limits": [
            "Vina affinity is not a reaction barrier or binding free energy.",
            "Catalytic geometry is not proof of covalent chemistry.",
        ],
    },
    "serine_hydrolase": {
        "name": "serine_hydrolase",
        "display_name": "丝氨酸水解酶",
        "status": "implemented",
        "description": "催化丝氨酸（SER-OG）在组氨酸-天冬氨酸/谷氨酸催化三联体及氧阴离子穴协同下对亲电中心发起亲核进攻。",
        "required_roles": {
            "catalytic_residue": {"residue_names": ["SER"], "required_atoms": ["OG"]},
            "acid_base_residue": {"residue_names": ["HIS", "ASP", "GLU"], "required_atoms": ["NE2", "ND1", "OD1", "OE1"]},
        },
        "box_strategy": "catalytic_residue_center",
        "scoring_strategy": "serine_hydrolase_triad_geometry",
        "default_thresholds": {"d_nuc_min": 2.6, "d_nuc_max": 4.5, "d_oxy_max": 4.2},
        "interpretation_limits": [
            "Geometric competence filters pre-reactive poses and does not prove covalent acylation kinetics.",
        ],
    },
    "cysteine_protease": {
        "name": "cysteine_protease",
        "display_name": "半胱氨酸蛋白酶",
        "status": "implemented",
        "description": "催化半胱氨酸（CYS-SG）与组氨酸（HIS-ND1/NE2）形成离子对，对底物羰基或亲电弹头碳原子进行亲核进攻。",
        "required_roles": {
            "catalytic_residue": {"residue_names": ["CYS"], "required_atoms": ["SG"]},
            "acid_base_residue": {"residue_names": ["HIS"], "required_atoms": ["ND1", "NE2"]},
        },
        "box_strategy": "catalytic_residue_center",
        "scoring_strategy": "cysteine_protease_dyad_geometry",
        "default_thresholds": {"d_nuc_min": 2.8, "d_nuc_max": 4.5, "d_oxy_max": 4.5},
        "interpretation_limits": [
            "Pre-reactive dyad geometry does not replace covalent bond formation energetics.",
        ],
    },
    "metalloenzyme": {
        "name": "metalloenzyme",
        "display_name": "金属酶",
        "status": "implemented",
        "description": "活性中心金属离子（Zn2+、Mg2+、Fe2+/3+、Mn2+ 等）通过第一配位层与配体杂原子（O/N/S）形成配位极化作用。",
        "required_roles": {
            "metal_center": {"residue_names": ["ZN", "MG", "FE", "MN", "CA", "CU", "NI", "CO"], "required_atoms": []},
        },
        "box_strategy": "metal_center",
        "scoring_strategy": "metalloenzyme_coordination_geometry",
        "default_thresholds": {"d_coord_min": 1.7, "d_coord_max": 3.4},
        "interpretation_limits": [
            "Coordination geometry screening identifies metal-chelating poses but does not replace quantum polarization.",
        ],
    },
    "cytochrome_p450": {
        "name": "cytochrome_p450",
        "display_name": "细胞色素 P450 / 血红素氧化酶",
        "status": "implemented",
        "description": "血红素辅因子（HEM/HEC）中心铁离子（FE）作为催化中心，评估底物可氧化位点到血红素铁的距离与取向。",
        "required_roles": {
            "cofactor": {"residue_names": ["HEM", "HEC", "HEA"], "required_atoms": ["FE"]},
        },
        "box_strategy": "heme_iron_center",
        "scoring_strategy": "p450_heme_iron_oxidation_geometry",
        "default_thresholds": {"d_nuc_min": 2.8, "d_nuc_max": 5.5},
        "interpretation_limits": [
            "Geometric proximity to heme Fe identifies accessible oxidation sites but does not replace DFT spin-state barriers.",
        ],
    },
    "kinase": {
        "name": "kinase",
        "display_name": "蛋白激酶（ATP/变构口袋）",
        "status": "implemented",
        "description": "针对激酶催化域（催化 LYS、DFG 基序 ASP 及铰链区 Hinge 主链氢键）定位结合口袋并评估铰链区氢键与极性锚定。",
        "required_roles": {
            "catalytic_residue": {"residue_names": ["LYS", "ASP", "GLU"], "required_atoms": ["NZ", "OD1", "OE1", "CA"]},
        },
        "box_strategy": "kinase_cleft_center",
        "scoring_strategy": "kinase_hinge_and_dfg_geometry",
        "default_thresholds": {"d_nuc_min": 2.4, "d_nuc_max": 5.5},
        "interpretation_limits": [
            "DFG-in/DFG-out conformational state of the input receptor governs inhibitor class compatibility.",
        ],
    },
    "glycosidase": {
        "name": "glycosidase",
        "display_name": "糖苷酶 / 碳水化合物活性酶",
        "status": "implemented",
        "description": "催化双羧酸残基（ASP/GLU 亲核体与酸碱残基）协同作用于糖苷键或底物极性中心。",
        "required_roles": {
            "catalytic_residue": {"residue_names": ["ASP", "GLU"], "required_atoms": ["OD1", "OD2", "OE1", "OE2"]},
            "acid_base_residue": {"residue_names": ["ASP", "GLU"], "required_atoms": ["OD1", "OD2", "OE1", "OE2"]},
        },
        "box_strategy": "catalytic_residue_center",
        "scoring_strategy": "glycosidase_dyad_geometry",
        "default_thresholds": {"d_nuc_min": 2.5, "d_nuc_max": 4.8},
        "interpretation_limits": [
            "Glycosidic ring distortion requires carbohydrate force-field or QM/MM validation.",
        ],
    },
    "flavoenzyme": {
        "name": "flavoenzyme",
        "display_name": "黄素酶（FAD/FMN 依赖型氧化还原酶）",
        "status": "implemented",
        "description": "以 FAD 或 FMN 辅因子异咯嗪环（N5/C4X/C4A）为反应核心，评估底物反应位点到 N5/C4X 的氢转移/电子转移几何。",
        "required_roles": {
            "cofactor": {"residue_names": ["FAD", "FMN"], "required_atoms": ["N5", "C4X", "C4A"]},
        },
        "box_strategy": "flavin_n5_center",
        "scoring_strategy": "flavoenzyme_n5_transfer_geometry",
        "default_thresholds": {"d_nuc_min": 2.6, "d_nuc_max": 5.0, "d_hyd_max": 5.0},
        "interpretation_limits": [
            "Flavin redox state affects active-site electrostatics.",
        ],
    },
    "general_binding": {
        "name": "general_binding",
        "display_name": "通用蛋白-配体结合（无先验位点/自动口袋发现）",
        "status": "implemented",
        "description": "面向非酶靶点（受体、通道、PPI 界面、变构口袋）或未知位点体系，自动通过三维网格空腔扫描识别结合口袋并结合物理化学能量重打分。",
        "required_roles": {},
        "box_strategy": "auto_cavity_discovery",
        "scoring_strategy": "universal_physicochemical_and_contact_scoring",
        "default_thresholds": {"d_nuc_min": 2.0, "d_nuc_max": 6.0},
        "interpretation_limits": [
            "Universal cavity docking ranks poses by Vina affinity, physicochemical rescore, and pocket burial without assuming covalent catalysis.",
        ],
    },
}


def list_supported_mechanisms() -> list[str]:
    return list(BUILTIN_MECHANISM_TEMPLATES.keys())


def load_mechanism_template(project_root: Path, family: str) -> dict:
    """Load a mechanism template from YAML or built-in universal mechanism registry."""
    if not family:
        family = "aldh"
    family_key = family.strip().lower()
    path = project_root / "config" / "mechanisms" / f"{family_key}.yaml"
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            template = yaml.safe_load(handle) or {}
        template["path"] = str(path)
        template["family"] = family_key
        return template
    if family_key in BUILTIN_MECHANISM_TEMPLATES:
        template = dict(BUILTIN_MECHANISM_TEMPLATES[family_key])
        template["path"] = f"builtin://{family_key}"
        template["family"] = family_key
        return template
    raise FileNotFoundError(
        f"Unknown mechanism family '{family}'. Expected one of {list_supported_mechanisms()} or a template at {path}."
    )


def mechanism_capability(template: dict) -> dict:
    status = template.get("status", "audit_only")
    return {
        "family": template.get("family"),
        "display_name": template.get("display_name", template.get("family")),
        "status": status,
        "implemented_scoring": status == "implemented",
        "box_strategy": template.get("box_strategy"),
        "scoring_strategy": template.get("scoring_strategy"),
        "default_thresholds": template.get("default_thresholds", {}),
        "interpretation_limits": template.get("interpretation_limits", []),
    }
