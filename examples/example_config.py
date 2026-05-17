from dataclasses import dataclass, field

import dacite
import tyro
import yaml

from diffcollision import DiffCollisionConfig, RS1DistConfig


@dataclass
class MainConfig:
    seed: int = 0
    device: str = "cuda:0"
    dtype: str = "double"
    n_thread: int = 16

    exp: str = "debug"
    log_dir: str = "output/debug"
    asset_dir: str = "examples/assets/object/DGN_5k/processed_data"
    n_prob: int = 100
    prob_rand: list[str] | None = field(default_factory=lambda: ["obj", "scale", "tp"])
    obj: list[str] = field(
        default_factory=lambda: [
            "sem_Gun_8834a85c572d88802a23d93958262ccc",
            "mujoco_Perricone_MD_OVM",
        ]
    )
    scale: list[float] = field(default_factory=lambda: [0.01, 0.115])
    convex: list[bool] = field(default_factory=lambda: [True, True])
    tp: list[str] = field(default_factory=lambda: ["v", "v"])
    n_tp: int = 1024
    tp_check: bool = True

    step_r: float = 10.0
    step_t: float = 0.01
    iter: int = 2000
    upd1: bool = False
    target_margin: float = 0.001

    dcd: DiffCollisionConfig = field(default_factory=RS1DistConfig)

    vis: bool = False
    vis_sample: bool = False

    @classmethod
    def from_dict(cls, config_dict: dict) -> "MainConfig":
        return dacite.from_dict(
            cls,
            config_dict,
            config=dacite.Config(strict=True, strict_unions_match=True),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "MainConfig":
        with open(path, "r") as f:
            return cls.from_dict(yaml.safe_load(f))

    def cli(self) -> "MainConfig":
        return tyro.cli(MainConfig, default=self)
