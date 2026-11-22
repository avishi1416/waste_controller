import pytest

from wastectl.cli import build_parser, main
from wastectl.config import load_config


def test_config_has_owner_decisions():
    cfg = load_config()
    assert cfg["classes"] == ["cardboard", "soft_plastic", "rigid_plastic", "metal", "other"]
    assert cfg["purity"]["target_stream"] in cfg["classes"]
    assert cfg["purity"]["alert_below"] == 0.90
    assert cfg["purity"]["conf_draw"] <= cfg["purity"]["conf_count"]


def test_config_missing_keys_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("seed: 1\n")
    with pytest.raises(ValueError):
        load_config(bad)


def test_unbuilt_command_fails_loudly():
    assert main(["bench"]) == 2


def test_parser_rejects_unknown_model():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["train", "vae"])
