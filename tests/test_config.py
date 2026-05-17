from dataclasses import dataclass, field
import dacite
from diffcollision import DiffCollisionConfig, RS1DistConfig, RS1DirConfig, FDConfig
from diffcollision import build_diffcoll_config


def test_config():
    config_dict = {
        "type": "RS1Dir",
        "egt_step_r": 1.0,
        "egt_step_t": 0.1,
        "enable_debug": True,
    }
    config = build_diffcoll_config(config_dict)
    assert isinstance(config, RS1DirConfig)
    assert config.egt_step_r == 1.0
    assert config.egt_step_t == 0.1
    assert config.enable_debug is True

    config_dict = {
        "type": "FD",
        "egt_step_r": 1.0,
        "egt_step_t": 0.1,
        "enable_debug": True,
    }
    config = build_diffcoll_config(config_dict)
    assert isinstance(config, FDConfig)
    assert config.egt_step_r == 1.0
    assert config.egt_step_t == 0.1
    assert config.enable_debug is True


def test_config_downstream():
    @dataclass
    class MyConfig:
        other: str = "test"
        collision: DiffCollisionConfig = field(default_factory=RS1DistConfig)

    config_dict = {
        "other": "111",
        "collision": {
            "type": "RS1Dir",
            "egt_step_r": 1.0,
            "egt_step_t": 1000.0,
            "enable_debug": True,
        },
    }
    config = dacite.from_dict(
        MyConfig,
        config_dict,
        config=dacite.Config(strict=True, strict_unions_match=True),
    )
    assert isinstance(config.collision, RS1DirConfig)
    assert config.collision.egt_step_t == 1000.0
