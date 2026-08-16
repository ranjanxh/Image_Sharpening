"""Config loading for the sharpening pipeline.

Everything path- or hyperparameter-related lives in a YAML file
(configs/default.yaml). No module in this package hardcodes an absolute
path -- paths in the config are resolved relative to a `project_root`
(the repo root by default, overridable via --project-root on any CLI).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclasses.dataclass
class PathsConfig:
    restormer_repo: str
    restormer_weights: str
    dataset_root: str
    checkpoints_dir: str
    inference_output_dir: str
    logs_dir: str
    results_dir: str


@dataclasses.dataclass
class DataConfig:
    train_blurry_subdir: str
    train_sharp_subdir: str
    test_blurry_subdir: str
    test_sharp_subdir: str
    blurry_stem_suffixes: list[str]
    sharp_stem_suffixes: list[str]
    image_size: list[int]
    batch_size: int
    num_workers: int
    gaussian_blur_augmentation: bool


@dataclasses.dataclass
class TeacherConfig:
    in_channels: int
    out_channels: int
    dim: int
    num_blocks: list[int]
    num_refinement_blocks: int
    heads: list[int]
    ffn_expansion_factor: float
    bias: bool
    layer_norm_type: str
    dual_pixel_task: bool
    img_multiple_of: int


@dataclasses.dataclass
class StudentConfig:
    in_channels: int
    out_channels: int
    base_channels: int


@dataclasses.dataclass
class ModelConfig:
    teacher: TeacherConfig
    student: StudentConfig


@dataclasses.dataclass
class LossesConfig:
    weights: dict[str, float]
    enabled: dict[str, bool]
    kl_temperature: float


@dataclasses.dataclass
class SchedulerConfig:
    t_0: int
    t_mult: int
    eta_min: float


@dataclasses.dataclass
class TrainingConfig:
    num_epochs: int
    learning_rate: float
    gradient_accumulation_steps: int
    log_interval: int
    eval_interval: int
    ema_decay: float
    seed: int
    scheduler: SchedulerConfig


@dataclasses.dataclass
class InferenceConfig:
    target_output_resolution: list[int]
    use_tiling: bool
    tiling_patch_size: int
    tiling_overlap: int


@dataclasses.dataclass
class Config:
    paths: PathsConfig
    data: DataConfig
    model: ModelConfig
    losses: LossesConfig
    training: TrainingConfig
    inference: InferenceConfig
    device: str
    project_root: Path

    def resolve(self, relative_path: str) -> Path:
        """Resolve a config path (relative or absolute) against project_root."""
        p = Path(relative_path)
        return p if p.is_absolute() else (self.project_root / p)


def _dict_to_dataclass(cls, data: dict[str, Any]):
    field_types = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs = {}
    for name, value in data.items():
        if name not in field_types:
            continue
        kwargs[name] = value
    return cls(**kwargs)


def load_config(config_path: str | Path, project_root: str | Path | None = None) -> Config:
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    root = Path(project_root).resolve() if project_root is not None else REPO_ROOT

    paths = _dict_to_dataclass(PathsConfig, raw["paths"])
    data = _dict_to_dataclass(DataConfig, raw["data"])
    teacher = _dict_to_dataclass(TeacherConfig, raw["model"]["teacher"])
    student = _dict_to_dataclass(StudentConfig, raw["model"]["student"])
    model = ModelConfig(teacher=teacher, student=student)
    losses = _dict_to_dataclass(LossesConfig, raw["losses"])
    scheduler = _dict_to_dataclass(SchedulerConfig, raw["training"]["scheduler"])
    training_raw = dict(raw["training"])
    training_raw["scheduler"] = scheduler
    training = _dict_to_dataclass(TrainingConfig, training_raw)
    inference = _dict_to_dataclass(InferenceConfig, raw["inference"])

    return Config(
        paths=paths,
        data=data,
        model=model,
        losses=losses,
        training=training,
        inference=inference,
        device=raw.get("device", "auto"),
        project_root=root,
    )


def resolve_device(requested: str):
    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)
