"""Strict loading of authoritative dataset preparation contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

EXPECTED_DATASETS = {"rlcp", "kdd", "higgs", "epsilon"}
EXPECTED_TEAMS = ("edgar", "luis", "fercho", "isaac", "caleb", "michelle")
FORMATS = {"nested_zip_csv", "gzip_csv", "bzip2_libsvm"}
STRATEGIES = {
    "connected_components",
    "duplicate_groups",
    "tail_test_stratified_validation",
    "official_test_stratified_validation",
}
ROOT_KEYS = {"version", "random_state", "team_ids", "datasets"}
DATASET_KEYS = {"format", "sources", "output_dir", "split", "tiers", "schema"}
SOURCE_KEYS = {"path", "partition", "expected_rows"}
SCHEMA_KEYS = {"feature_count", "label_position", "labels", "columns", "block_count"}
RATIO_KEYS = {"strategy", "train_ratio", "validation_ratio", "internal_test_ratio"}
SIZE_KEYS = {"strategy", "development_rows", "validation_size", "official_test_size"}
VALIDATION_KEYS = {"strategy", "validation_size"}


@dataclass(frozen=True)
class SourceConfig:
    """One immutable raw source contract."""

    path: Path
    partition: str
    expected_rows: int | None


@dataclass(frozen=True)
class DatasetConfig:
    """Validated preparation settings for one dataset."""

    name: str
    format: str
    sources: tuple[SourceConfig, ...]
    output_dir: Path
    split_strategy: str
    split: dict[str, int | float | str]
    tiers: tuple[int, ...]
    feature_count: int
    label_position: str
    labels: dict[str, str]
    columns: tuple[str, ...]
    block_count: int | None


@dataclass(frozen=True)
class ExperimentsConfig:
    """Validated top-level experiment preparation contract."""

    random_state: int
    team_ids: tuple[str, ...]
    datasets: dict[str, DatasetConfig]


def _mapping(value: Any, location: str) -> dict[str, Any]:
    """Require a string-keyed mapping."""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{location} must be a mapping with string keys")
    return value


def _exact_keys(data: dict[str, Any], expected: set[str], location: str) -> None:
    """Reject missing and unknown schema keys."""
    missing = expected - data.keys()
    unknown = data.keys() - expected
    if missing or unknown:
        raise ValueError(f"{location} keys invalid; missing={sorted(missing)}, unknown={sorted(unknown)}")


def _positive_int(value: Any, location: str, *, nullable: bool = False) -> int | None:
    """Require a positive integer, optionally allowing null."""
    if nullable and value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{location} must be a positive integer")
    return value


def _load_source(value: Any, location: str) -> SourceConfig:
    """Validate one source declaration."""
    data = _mapping(value, location)
    _exact_keys(data, SOURCE_KEYS, location)
    if not isinstance(data["path"], str) or not data["path"]:
        raise ValueError(f"{location}.path must be a non-empty string")
    if not isinstance(data["partition"], str) or not data["partition"]:
        raise ValueError(f"{location}.partition must be a non-empty string")
    expected = _positive_int(data["expected_rows"], f"{location}.expected_rows", nullable=True)
    return SourceConfig(Path(data["path"]), data["partition"], expected)


def _validate_split(value: Any, location: str) -> dict[str, int | float | str]:
    """Validate strategy-specific split parameters without fallback."""
    data = _mapping(value, location)
    strategy = data.get("strategy")
    if strategy not in STRATEGIES:
        raise ValueError(f"{location}.strategy is unsupported: {strategy!r}")
    expected = RATIO_KEYS if strategy in {"connected_components", "duplicate_groups"} else SIZE_KEYS
    if strategy == "official_test_stratified_validation":
        expected = VALIDATION_KEYS
    _exact_keys(data, expected, location)
    if expected == RATIO_KEYS:
        ratios = [data[key] for key in ("train_ratio", "validation_ratio", "internal_test_ratio")]
        if any(not isinstance(item, (int, float)) or isinstance(item, bool) or item <= 0 for item in ratios):
            raise ValueError(f"{location} ratios must be positive numbers")
        if abs(sum(ratios) - 1.0) > 1e-12:
            raise ValueError(f"{location} ratios must sum to 1")
    else:
        for key in expected - {"strategy"}:
            _positive_int(data[key], f"{location}.{key}")
    return dict(data)


def _load_dataset(name: str, value: Any) -> DatasetConfig:
    """Validate one complete dataset contract."""
    data = _mapping(value, f"datasets.{name}")
    _exact_keys(data, DATASET_KEYS, f"datasets.{name}")
    if data["format"] not in FORMATS:
        raise ValueError(f"datasets.{name}.format is unsupported: {data['format']!r}")
    if not isinstance(data["sources"], list) or not data["sources"]:
        raise ValueError(f"datasets.{name}.sources must be a non-empty list")
    sources = tuple(_load_source(item, f"datasets.{name}.sources[{index}]") for index, item in enumerate(data["sources"]))
    if not isinstance(data["output_dir"], str) or not data["output_dir"]:
        raise ValueError(f"datasets.{name}.output_dir must be a non-empty string")
    split = _validate_split(data["split"], f"datasets.{name}.split")
    tiers = data["tiers"]
    if not isinstance(tiers, list) or any(_positive_int(item, f"datasets.{name}.tiers") is None for item in tiers):
        raise ValueError(f"datasets.{name}.tiers must contain positive integers")
    if not tiers or tiers != sorted(set(tiers)):
        raise ValueError(f"datasets.{name}.tiers must be strictly increasing and unique")
    schema = _mapping(data["schema"], f"datasets.{name}.schema")
    _exact_keys(schema, SCHEMA_KEYS, f"datasets.{name}.schema")
    if schema["label_position"] not in {"first", "last"}:
        raise ValueError(f"datasets.{name}.schema.label_position must be first or last")
    if not isinstance(schema["labels"], dict) or not schema["labels"] or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in schema["labels"].items()
    ):
        raise ValueError(f"datasets.{name}.schema.labels must be a non-empty mapping")
    if not isinstance(schema["columns"], list) or not all(isinstance(item, str) for item in schema["columns"]):
        raise ValueError(f"datasets.{name}.schema.columns must be a string list")
    return DatasetConfig(
        name=name, format=data["format"], sources=sources, output_dir=Path(data["output_dir"]),
        split_strategy=str(split["strategy"]), split=split,
        tiers=tuple(tiers), feature_count=int(_positive_int(schema["feature_count"], f"datasets.{name}.schema.feature_count")),
        label_position=schema["label_position"], labels={str(key): str(item) for key, item in schema["labels"].items()},
        columns=tuple(schema["columns"]), block_count=_positive_int(schema["block_count"], f"datasets.{name}.schema.block_count", nullable=True),
    )


def load_config(path: Path) -> ExperimentsConfig:
    """Load and strictly validate an experiment dataset YAML file.

    Args:
        path: YAML configuration path.

    Returns:
        The immutable validated experiment configuration.

    Raises:
        ValueError: If YAML content violates the explicit schema.
        yaml.YAMLError: If YAML syntax or tags are unsafe or invalid.
    """
    with path.open("r", encoding="utf-8") as stream:
        root = _mapping(yaml.safe_load(stream), "root")
    _exact_keys(root, ROOT_KEYS, "root")
    if root["version"] != 1 or root["random_state"] != 42:
        raise ValueError("version must be 1 and random_state must be 42")
    if not isinstance(root["team_ids"], list) or tuple(root["team_ids"]) != EXPECTED_TEAMS:
        raise ValueError(f"team_ids must be exactly {list(EXPECTED_TEAMS)}")
    datasets = _mapping(root["datasets"], "datasets")
    if set(datasets) != EXPECTED_DATASETS:
        raise ValueError(f"datasets must be exactly {sorted(EXPECTED_DATASETS)}")
    return ExperimentsConfig(42, EXPECTED_TEAMS, {name: _load_dataset(name, value) for name, value in datasets.items()})
