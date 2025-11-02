from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GeneralConfig:
    oroject_name: str = "ethrc_sarm"
    task_name: str = "Fold Towel"
    repo_id_sparse: str = "ETHRC/piper_towel_v0_with_rewards"
    state_norm_path: str | Path = ""  # TODO: add path to state normalization
    camera_names: list[str] = field(
        default_factory=lambda: [
            "observation.images.wrist1",
            "observation.images.wrist2",
            "observation.images.stereo",
        ]
    )
    seed: int = 42


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
    checkpoint_path: str | Path | None = None


@dataclass
class OptimizerConfig:
    lr: float = 5e-5
    weight_decay: float = 5e-3
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    warmup_steps: int = 1000
    total_steps: int = 100000


@dataclass
class TrainConfig:
    num_epochs: int = 2
    grad_clip: float = 2.0
    log_every: int = 50  # in steps
    eval_every: int = 1  # in epochs
    save_every: int = 5000
    val_portion: float = 0.1  # portion of the dataset to use for validation


@dataclass
class TrainLoaderConfig:
    batch_size: int = 32
    num_workers: int = 6
    shuffle: bool = True
    pin_memory: bool = True
    persistant_workers: bool = True


@dataclass
class SarmConfig:
    general_config: GeneralConfig = GeneralConfig()
    model_config: ModelConfig = ModelConfig()
    optimizer_config: OptimizerConfig = OptimizerConfig()
    train_config: TrainConfig = TrainConfig()
    train_loader_config: TrainLoaderConfig = TrainLoaderConfig()
