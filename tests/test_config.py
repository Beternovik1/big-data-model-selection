"""Tests for strict experiment configuration loading."""

from pathlib import Path

import pytest
import yaml

from bigdata_model_selection.config import load_config

CONFIG_PATH = Path("configs/experiments/datasets.yaml")


def test_load_config_authoritative_contract_returns_typed_config() -> None:
    """The tracked contract should expose all fixed scientific settings."""
    config = load_config(CONFIG_PATH)

    assert config.random_state == 42
    assert config.team_ids == ("edgar", "luis", "fercho", "isaac", "caleb", "michelle")
    assert config.datasets["rlcp"].split_strategy == "connected_components"
    assert config.datasets["kdd"].split_strategy == "duplicate_groups"
    assert config.datasets["higgs"].split["official_test_size"] == 500_000
    assert config.datasets["epsilon"].labels == {"1": "1", "-1": "-1"}


def test_load_config_unknown_key_raises_value_error(tmp_path: Path) -> None:
    """Unknown fields must not be silently ignored."""
    data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    data["unexpected"] = True
    path = tmp_path / "unknown.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ValueError, match=r"unknown=\['unexpected'\]"):
        load_config(path)


def test_load_config_non_nested_tiers_raises_value_error(tmp_path: Path) -> None:
    """Tier sizes must encode a strictly increasing nesting order."""
    data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    data["datasets"]["epsilon"]["tiers"] = [30_000, 10_000]
    path = tmp_path / "tiers.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ValueError, match="strictly increasing"):
        load_config(path)


def test_load_config_unsafe_yaml_tag_raises_yaml_error(tmp_path: Path) -> None:
    """Unsafe object constructors must be rejected by safe YAML loading."""
    path = tmp_path / "unsafe.yaml"
    path.write_text("!!python/object/apply:os.system ['false']", encoding="utf-8")

    with pytest.raises(yaml.YAMLError):
        load_config(path)


def test_load_config_invalid_split_ratio_raises_value_error(tmp_path: Path) -> None:
    """Ratio-based strategies must explicitly cover the entire population."""
    data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    data["datasets"]["kdd"]["split"]["train_ratio"] = 0.60
    path = tmp_path / "ratios.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ValueError, match="ratios must sum to 1"):
        load_config(path)
