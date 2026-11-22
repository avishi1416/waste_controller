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
    from wastectl.vision.pipeline import Pipeline  # noqa: F401  (needs ultralytics importable)

    img = tmp_path / "x.jpg"
    img.write_bytes(b"")
    assert main(["infer", str(img)]) == 1
    assert "not trained yet" in capsys.readouterr().err


def test_parser_rejects_unknown_model():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["train", "vae"])
