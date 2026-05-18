from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from collections.abc import Sequence
import shlex

import dacite
from loguru import logger
import numpy as np
import tyro
import yaml

from diffcollision import DiffCollisionConfig, RS1DistConfig


_LOGURU_FILE_SINKS: dict[Path, int] = {}
_DEFAULT_EXP = "debug"
_DEFAULT_LOG_DIR = "output/debug"


def _to_yaml_data(value):
    if is_dataclass(value):
        return {
            field.name: _to_yaml_data(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {_to_yaml_data(k): _to_yaml_data(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_yaml_data(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass
class MainConfig:
    seed: int = 0
    device: str = "cuda:0"
    dtype: str = "double"
    n_thread: int = 16

    exp: str = _DEFAULT_EXP
    log_dir: str = _DEFAULT_LOG_DIR
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

    def to_dict(self) -> dict:
        return _to_yaml_data(self)

    def save(self, cli_args: Sequence[str] | None = None) -> None:
        config_dir = Path(self.log_dir) / "config"
        config_dir.mkdir(parents=True, exist_ok=True)

        with open(config_dir / "final_config.yaml", "w") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False)

        args = [] if cli_args is None else list(cli_args)
        cli_args_text = shlex.join(args)
        if cli_args_text:
            cli_args_text += "\n"
        (config_dir / "cli_args.txt").write_text(cli_args_text)

    def setup_logging(self) -> Path:
        exp_dir = Path(self.log_dir)
        exp_dir.mkdir(parents=True, exist_ok=True)
        log_path = exp_dir / "loguru.log"
        resolved_path = log_path.resolve()
        if resolved_path not in _LOGURU_FILE_SINKS:
            _LOGURU_FILE_SINKS[resolved_path] = logger.add(log_path, enqueue=True)
        return log_path

    def sync_log_dir_with_exp(self, cli_args: Sequence[str] | None = None) -> None:
        args = [] if cli_args is None else list(cli_args)
        has_cli_log_dir = any(
            arg in {"--log-dir", "--log_dir"}
            or arg.startswith("--log-dir=")
            or arg.startswith("--log_dir=")
            for arg in args
        )
        if not has_cli_log_dir and self.log_dir == _DEFAULT_LOG_DIR:
            self.log_dir = str(Path("output") / self.exp)

    def prepare_experiment(self, cli_args: Sequence[str] | None = None) -> None:
        self.sync_log_dir_with_exp(cli_args)
        log_path = self.setup_logging()
        self.save(cli_args)
        logger.info(f"Saving loguru output to {log_path}")

    def cli(self) -> "MainConfig":
        return tyro.cli(MainConfig, default=self)
