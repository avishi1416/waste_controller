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
    assert main(["demo"]) == 2


def test_missing_weights_is_a_clean_error(tmp_path, capsys):
    import yaml

    from wastectl.config import DEFAULT_CONFIG

    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    cfg["paths"]["weights"] = str(tmp_path / "no_weights_here")  # independent of what's trained on disk
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    img = tmp_path / "x.jpg"
    img.write_bytes(b"")
    assert main(["--config", str(cfg_path), "infer", str(img)]) == 1
    assert "not trained yet" in capsys.readouterr().err


def test_parser_rejects_unknown_model():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["train", "vae"])
