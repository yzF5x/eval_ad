import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


@dataclass(frozen=True)
class AppConfig:
    OPENROUTER_API_KEY: str
    api_model: str
    dataset_path: str
    save_dir: str
    replace_path: str
    max_size: int
    merged_patch_size: int
    with_tag: bool
    attention_threshold: float
    threshold_low: int
    threshold_high: int
    gt_binary_threshold: int
    sc_low_threshold: float
    sc_high_threshold: float
    model_path: str
    max_new_tokens: int
    save_attentions: bool
    overwrite: bool
    use_qkvfp32_monkey_patch: bool
    generated_dir: str
    global_save_fig: bool
    normal_set_zero: bool
    return_aggregate: bool
    dataset: str
    api_base: str


_CONFIG_SINGLETON: AppConfig | None = None


def _as_bool(value: Any, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"配置项 {key} 必须是布尔值")


def _as_int(value: Any, key: str) -> int:
    try:
        return int(value)
    except Exception as exc:
        raise ValueError(f"配置项 {key} 必须是整数") from exc


def _as_float(value: Any, key: str) -> float:
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"配置项 {key} 必须是浮点数") from exc


def _as_str(value: Any, key: str) -> str:
    if value is None:
        raise ValueError(f"配置项 {key} 不能为空")
    text = str(value).strip()
    if not text:
        raise ValueError(f"配置项 {key} 不能为空")
    return text


def _read_config_file(config_path: str | os.PathLike[str]) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as f:
        if suffix in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError("未安装 PyYAML，无法读取 YAML 配置。请先安装 pyyaml。")
            data = yaml.safe_load(f) or {}
        elif suffix == ".json":
            data = json.load(f) or {}
        else:
            raise ValueError(f"不支持的配置文件格式: {suffix}，请使用 .yaml/.yml/.json")
    if not isinstance(data, dict):
        raise ValueError("配置文件根节点必须是对象（dict）")
    return data


def _apply_env_overrides(data: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(data)
    env_val = os.getenv("ENV_OPENROUTER_API_KEY")
    if env_val is not None:
        merged["OPENROUTER_API_KEY"] = env_val
    return merged


def _validate_and_build(data: Dict[str, Any]) -> AppConfig:
    required = [
        "OPENROUTER_API_KEY",
        "api_model",
        "dataset_path",
        "save_dir",
        "replace_path",
        "max_size",
        "merged_patch_size",
        "with_tag",
    ]
    for key in required:
        if key not in data:
            raise KeyError(f"缺少必填配置项: {key}")
    config = AppConfig(
        OPENROUTER_API_KEY=_as_str(data.get("OPENROUTER_API_KEY"), "OPENROUTER_API_KEY"),
        api_model=_as_str(data.get("api_model"), "api_model"),
        dataset_path=_as_str(data.get("dataset_path"), "dataset_path"),
        save_dir=_as_str(data.get("save_dir"), "save_dir"),
        replace_path=_as_str(data.get("replace_path"), "replace_path"),
        max_size=_as_int(data.get("max_size"), "max_size"),
        merged_patch_size=_as_int(data.get("merged_patch_size"), "merged_patch_size"),
        with_tag=_as_bool(data.get("with_tag"), "with_tag"),
        attention_threshold=_as_float(data.get("attention_threshold", 0.17), "attention_threshold"),
        threshold_low=_as_int(data.get("threshold_low", 0), "threshold_low"),
        threshold_high=_as_int(data.get("threshold_high", 1), "threshold_high"),
        gt_binary_threshold=_as_int(data.get("gt_binary_threshold", 128), "gt_binary_threshold"),
        sc_low_threshold=_as_float(data.get("sc_low_threshold", 0.1), "sc_low_threshold"),
        sc_high_threshold=_as_float(data.get("sc_high_threshold", 0.2), "sc_high_threshold"),
        model_path=_as_str(data.get("model_path", ""), "model_path"),
        max_new_tokens=_as_int(data.get("max_new_tokens", 1024), "max_new_tokens"),
        save_attentions=_as_bool(data.get("save_attentions", False), "save_attentions"),
        overwrite=_as_bool(data.get("overwrite", False), "overwrite"),
        use_qkvfp32_monkey_patch=_as_bool(data.get("use_qkvfp32_monkey_patch", True), "use_qkvfp32_monkey_patch"),
        generated_dir=_as_str(data.get("generated_dir", data.get("save_dir", "")), "generated_dir"),
        global_save_fig=_as_bool(data.get("global_save_fig", True), "global_save_fig"),
        normal_set_zero=_as_bool(data.get("normal_set_zero", False), "normal_set_zero"),
        return_aggregate=_as_bool(data.get("return_aggregate", True), "return_aggregate"),
        dataset=_as_str(data.get("dataset", "mvtec"), "dataset"),
        api_base=_as_str(data.get("api_base", "https://openrouter.ai/api/v1/chat/completions"), "api_base"),
    )
    if config.threshold_low >= config.threshold_high:
        raise ValueError("配置项 threshold_low 必须小于 threshold_high")
    if not (0.0 <= config.attention_threshold <= 1.0):
        raise ValueError("配置项 attention_threshold 必须在 [0, 1] 范围内")
    if not (0 <= config.gt_binary_threshold <= 255):
        raise ValueError("配置项 gt_binary_threshold 必须在 [0, 255] 范围内")
    if not (0.0 <= config.sc_low_threshold <= 1.0 and 0.0 <= config.sc_high_threshold <= 1.0):
        raise ValueError("配置项 sc_low_threshold/sc_high_threshold 必须在 [0, 1] 范围内")
    if config.sc_low_threshold >= config.sc_high_threshold:
        raise ValueError("配置项 sc_low_threshold 必须小于 sc_high_threshold")
    return config


def initialize_config(config_path: str = "config/config.yaml", force_reload: bool = False) -> AppConfig:
    global _CONFIG_SINGLETON
    if _CONFIG_SINGLETON is not None and not force_reload:
        return _CONFIG_SINGLETON
    raw = _read_config_file(config_path)
    merged = _apply_env_overrides(raw)
    _CONFIG_SINGLETON = _validate_and_build(merged)
    return _CONFIG_SINGLETON


def get_config() -> AppConfig:
    if _CONFIG_SINGLETON is None:
        raise RuntimeError("配置尚未初始化，请先调用 initialize_config()")
    return _CONFIG_SINGLETON

def load_dataset(dataset_path):
    import pandas as pd

    if dataset_path.endswith(".csv"):
        dataset = pd.read_csv(dataset_path)
    elif dataset_path.endswith(".xlsx"):
        dataset = pd.read_excel(dataset_path)
    elif dataset_path.endswith(".tsv"):
        dataset = pd.read_csv(dataset_path, sep="\t")
 
    return dataset


def load_model(model_path):
    from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

    lower_model_path = model_path.lower()
    if "qwen2.5-vl" in lower_model_path or "qwen2_5" in lower_model_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto",attn_implementation="sdpa")
        processor = AutoProcessor.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        raise ValueError(f"Unsupported model path: {model_path}")
    return model , processor , tokenizer
