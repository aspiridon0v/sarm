from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GeneralConfig:
    project_name: str = "ethrc_sarm"
    task_name: str = "Fold Towel"
    repo_id_sparse: str = "ETHRC/towel_base_with_rewards"
    state_norm_path: str | Path = "data/towel_base_with_rewards.json"
    camera_names: list[str] = field(
        default_factory=lambda: [
            "observation.images.left_wrist",
            "observation.images.right_wrist",
            "observation.images.topdown",
        ]
    )
    seed: int = 42
    wandb_entity: str | None = None  # Set to None to use default account


@dataclass
class ModelConfig:
    d_model: int = 768
    n_layers: int = 8
    n_heads: int = 12
    dropout: float = 0.1
    horizon: int = 8
    n_obs_steps: int = 8
    max_rewind_steps: int = 4
    frame_gap: int = 30
    state_dim: int = 14
    clip_weights_path: str = "checkpoints/clip_vit_b32_openai.npz"
    clip_preprocess_chunk_size: int = 32  # Images to process at once during resize (lower = less memory)
    model_path: str | Path | None = None
    num_classes_sparse: int = 5
    sparse_annotation_list: list[str] = field(
        default_factory=lambda: [
            "Grasp right corner",
            "Grasp left corner",
            "Fold towel horizontally",
            "Grasp right edge",
            "Fold towel vertically",
        ]
    )
    resume_from_checkpoint: bool = False
    progress_checkpoint_path: str | None = "checkpoints/prg_t-2026.01.23-02.41.26-s-5000-b48.eqx"
    stage_checkpoint_path: str | None = "checkpoints/stg_t-2026.01.23-02.41.26-s-5000-b48.eqx"


@dataclass
class OptimizerConfig:
    lr: float = 5e-5
    weight_decay: float = 5e-3
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    warmup_steps: int = 100
    total_steps: int = 5000


@dataclass
class TrainConfig:
    num_epochs: int = 2
    grad_clip: float = 1.0
    log_every: int = 5  # in steps
    eval_every: int = 20  # in steps
    save_every: int = 1000
    val_portion: float = 0.1  # portion of the dataset to use for validation
    dense_shema: bool = False


@dataclass
class TrainLoaderConfig:
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True
    pin_memory: bool = False
    persistant_workers: bool = False


@dataclass
class LoggingConfig:
    level: str = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
    format: str = "%(asctime)s | %(name)s:%(lineno)d | %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"


@dataclass
class SarmConfig:
    general_config: GeneralConfig = GeneralConfig()
    model_config: ModelConfig = ModelConfig()
    optimizer_config: OptimizerConfig = OptimizerConfig()
    train_config: TrainConfig = TrainConfig()
    train_loader_config: TrainLoaderConfig = TrainLoaderConfig()
    logging_config: LoggingConfig = LoggingConfig()
