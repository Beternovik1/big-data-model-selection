"""Synthetic compressed-fixture tests for dataset preparation."""

from __future__ import annotations

import bz2
import csv
import gzip
import io
import json
import zipfile
from pathlib import Path

import pytest

from bigdata_model_selection.config import DatasetConfig, ExperimentsConfig, SourceConfig
from bigdata_model_selection.preparation import prepare_dataset

TEAMS = ("edgar", "luis", "fercho", "isaac", "caleb", "michelle")
RLCP_COLUMNS = ("id_1", "id_2", "cmp_fname_c1", "cmp_fname_c2", "cmp_lname_c1", "cmp_lname_c2", "cmp_sex", "cmp_bd", "cmp_bm", "cmp_by", "cmp_plz", "is_match")


def _experiment(dataset: DatasetConfig) -> ExperimentsConfig:
    """Wrap one synthetic contract for the public preparation boundary."""
    return ExperimentsConfig(42, TEAMS, {dataset.name: dataset})


def _dataset(
    name: str,
    format_name: str,
    sources: tuple[SourceConfig, ...],
    strategy: str,
    split: dict[str, int | float | str],
    tiers: tuple[int, ...],
    feature_count: int,
    labels: dict[str, str],
    label_position: str,
    columns: tuple[str, ...] = (),
    block_count: int | None = None,
) -> DatasetConfig:
    """Create a small validated-equivalent dataset contract."""
    return DatasetConfig(name, format_name, sources, Path("processed") / name, strategy, split, tiers, feature_count, label_position, labels, columns, block_count)


def _manifest_ids(path: Path) -> set[str]:
    """Read stable IDs from a generated sidecar manifest."""
    with path.open(encoding="utf-8", newline="") as stream:
        return {row["observation_id"] for row in csv.DictReader(stream)}


def _assert_common_guards(output: Path, tiers: tuple[int, ...]) -> None:
    """Assert shared split disjointness, nesting, metadata, and cleanup."""
    validation = _manifest_ids(output / "validation.manifest.csv")
    test_name = "official_test" if (output / "official_test.manifest.csv").exists() else "internal_test"
    test = _manifest_ids(output / f"{test_name}.manifest.csv")
    tier_ids = [_manifest_ids(output / f"train_{tier}.manifest.csv") for tier in tiers]
    assert validation.isdisjoint(test)
    assert all(validation.isdisjoint(ids) and test.isdisjoint(ids) for ids in tier_ids)
    assert all(tier_ids[index] < tier_ids[index + 1] for index in range(len(tier_ids) - 1))
    assert [len(ids) for ids in tier_ids] == list(tiers)
    assert (output / ".complete").read_text(encoding="ascii") == "complete\n"
    assert not (output / ".work").exists()
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert all(metadata["guards"].values())
    assert metadata["elapsed_seconds"] >= 0
    assert set(json.loads((output / "team_manifest.json").read_text())["team_ids"]) == set(TEAMS)


def _write_rlcp(path: Path, rows: list[list[str]], *, unsafe: bool = False) -> None:
    """Create a nested ZIP fixture matching RLCP's archive shape."""
    nested_buffer = io.BytesIO()
    with zipfile.ZipFile(nested_buffer, "w", zipfile.ZIP_DEFLATED) as nested:
        content = io.StringIO()
        writer = csv.writer(content, lineterminator="\n")
        writer.writerow(RLCP_COLUMNS)
        writer.writerows(rows)
        nested.writestr("../block.csv" if unsafe else "block.csv", content.getvalue())
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as outer:
        outer.writestr("block_1.zip", nested_buffer.getvalue())


def test_prepare_dataset_rlcp_keeps_connected_entities_together(tmp_path: Path) -> None:
    """Typed bipartite connected components must never cross evaluations."""
    raw = tmp_path / "rlcp.zip"
    rows = []
    for index in range(12):
        left = f"L{index // 2}"
        right = f"R{index}"
        rows.append([left, right, "1", "", "1", "", "1", "1", "1", "1", "1", "TRUE" if index % 2 else "FALSE"])
    _write_rlcp(raw, rows)
    config = _dataset("rlcp", "nested_zip_csv", (SourceConfig(Path("rlcp.zip"), "all", None),), "connected_components", {"strategy": "connected_components", "train_ratio": 0.5, "validation_ratio": 0.25, "internal_test_ratio": 0.25}, (2, 4), 10, {"TRUE": "TRUE", "FALSE": "FALSE"}, "last", RLCP_COLUMNS, 1)

    output = prepare_dataset(_experiment(config), "rlcp", tmp_path)

    _assert_common_guards(output, config.tiers)
    entity_splits: dict[str, str] = {}
    for split in ("train", "validation", "internal_test"):
        with (output / f"{split}.csv").open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                for entity in (f"left:{row['id_1']}", f"right:{row['id_2']}"):
                    assert entity not in entity_splits or entity_splits[entity] == split
                    entity_splits[entity] = split
    generated = list(output.glob("*.csv"))
    assert all(path.stat().st_size > 0 for path in generated)
    assert json.loads((output / "metadata.json").read_text())["group_ratio_deviation_rows"] is not None


def test_prepare_dataset_kdd_groups_exact_duplicates_and_maps_binary_labels(tmp_path: Path) -> None:
    """Exact duplicate KDD records must share a split and retain source labels."""
    raw = tmp_path / "kdd.gz"
    records = []
    for index in range(10):
        label = "normal." if index % 2 else "smurf."
        record = ",".join([str(index)] * 41 + [label])
        records.extend([record, record])
    with gzip.open(raw, "wt", encoding="utf-8", newline="") as stream:
        stream.write("\n".join(records) + "\n")
    config = _dataset("kdd", "gzip_csv", (SourceConfig(Path("kdd.gz"), "all", 20),), "duplicate_groups", {"strategy": "duplicate_groups", "train_ratio": 0.6, "validation_ratio": 0.2, "internal_test_ratio": 0.2}, (4, 8), 41, {"normal.": "normal", "*": "attack"}, "last")

    output = prepare_dataset(_experiment(config), "kdd", tmp_path)

    _assert_common_guards(output, config.tiers)
    metadata = json.loads((output / "metadata.json").read_text())
    labels = {label for counts in metadata["class_counts"].values() for label in counts}
    assert labels == {"normal", "attack"}
    source_labels = {label for counts in metadata["source_class_counts"].values() for label in counts}
    assert source_labels == {"normal.", "smurf."}
    split_lines = {}
    for split in ("train", "validation", "internal_test"):
        for line in (output / f"{split}.csv").read_text().splitlines():
            assert line not in split_lines or split_lines[line] == split
            split_lines[line] = split


def test_prepare_dataset_higgs_preserves_tail_and_normalizes_labels(tmp_path: Path) -> None:
    """HIGGS must reserve the source tail and serialize labels as 0 or 1."""
    raw = tmp_path / "higgs.gz"
    lines = [",".join(["0.0" if index % 2 == 0 else "1.000"] + [str(index)] * 28) for index in range(20)]
    with gzip.open(raw, "wt", encoding="utf-8", newline="") as stream:
        stream.write("\n".join(lines) + "\n")
    config = _dataset("higgs", "gzip_csv", (SourceConfig(Path("higgs.gz"), "all", 20),), "tail_test_stratified_validation", {"strategy": "tail_test_stratified_validation", "development_rows": 16, "validation_size": 4, "official_test_size": 4}, (4, 8), 28, {"0": "0", "1": "1", "0.0": "0", "1.000": "1"}, "first")

    output = prepare_dataset(_experiment(config), "higgs", tmp_path)

    _assert_common_guards(output, config.tiers)
    official_ids = _manifest_ids(output / "official_test.manifest.csv")
    assert official_ids == {f"higgs.gz:{index}" for index in range(17, 21)}
    assert {line.split(",", 1)[0] for line in (output / "official_test.csv").read_text().splitlines()} == {"0", "1"}


def _epsilon_line(label: str, offset: int, features: int = 4) -> str:
    """Create one dense ordered LIBSVM row."""
    return " ".join([label] + [f"{index}:{offset + index}.0" for index in range(1, features + 1)])


def test_prepare_dataset_epsilon_preserves_official_test_and_source_labels(tmp_path: Path) -> None:
    """Epsilon preparation must not require positive-class semantics."""
    train = tmp_path / "epsilon.bz2"
    test = tmp_path / "epsilon.t.bz2"
    with bz2.open(train, "wt", encoding="utf-8") as stream:
        stream.write("\n".join(_epsilon_line("1" if index % 2 else "-1", index) for index in range(16)) + "\n")
    with bz2.open(test, "wt", encoding="utf-8") as stream:
        stream.write("\n".join(_epsilon_line("1" if index % 2 else "-1", index + 20) for index in range(4)) + "\n")
    sources = (SourceConfig(Path("epsilon.bz2"), "official_train", 16), SourceConfig(Path("epsilon.t.bz2"), "official_test", 4))
    config = _dataset("epsilon", "bzip2_libsvm", sources, "official_test_stratified_validation", {"strategy": "official_test_stratified_validation", "validation_size": 4}, (4, 8), 4, {"1": "1", "-1": "-1"}, "first")

    output = prepare_dataset(_experiment(config), "epsilon", tmp_path)

    _assert_common_guards(output, config.tiers)
    assert _manifest_ids(output / "official_test.manifest.csv") == {f"official_test:{index}" for index in range(1, 5)}
    with (output / "official_test.manifest.csv").open(encoding="utf-8") as stream:
        assert {row["source_label"] for row in csv.DictReader(stream)} == {"1", "-1"}


def test_prepare_dataset_repeated_run_is_deterministic(tmp_path: Path) -> None:
    """A repeated run must publish identical memberships despite fresh work state."""
    raw = tmp_path / "kdd.gz"
    records = [",".join([str(index)] * 41 + ["normal." if index % 2 else "attack."]) for index in range(20)]
    with gzip.open(raw, "wt", encoding="utf-8") as stream:
        stream.write("\n".join(records) + "\n")
    config = _dataset("kdd", "gzip_csv", (SourceConfig(Path("kdd.gz"), "all", 20),), "duplicate_groups", {"strategy": "duplicate_groups", "train_ratio": 0.6, "validation_ratio": 0.2, "internal_test_ratio": 0.2}, (4, 8), 41, {"normal.": "normal", "*": "attack"}, "last")

    output = prepare_dataset(_experiment(config), "kdd", tmp_path)
    first = (output / "train_8.manifest.csv").read_bytes()
    prepare_dataset(_experiment(config), "kdd", tmp_path)

    assert (output / "train_8.manifest.csv").read_bytes() == first


def test_prepare_dataset_unsafe_nested_zip_leaves_no_completion_marker(tmp_path: Path) -> None:
    """ZIP traversal and partial publication must both fail closed."""
    raw = tmp_path / "rlcp.zip"
    _write_rlcp(raw, [["L", "R", "1", "", "1", "", "1", "1", "1", "1", "1", "TRUE"]], unsafe=True)
    config = _dataset("rlcp", "nested_zip_csv", (SourceConfig(Path("rlcp.zip"), "all", None),), "connected_components", {"strategy": "connected_components", "train_ratio": 0.6, "validation_ratio": 0.2, "internal_test_ratio": 0.2}, (1,), 10, {"TRUE": "TRUE", "FALSE": "FALSE"}, "last", RLCP_COLUMNS, 1)

    with pytest.raises(ValueError, match="unsafe ZIP member"):
        prepare_dataset(_experiment(config), "rlcp", tmp_path)

    assert not (tmp_path / "processed/rlcp/.complete").exists()


def test_prepare_dataset_tier_larger_than_train_fails_explicitly(tmp_path: Path) -> None:
    """Impossible exact tiers must fail rather than truncating silently."""
    raw = tmp_path / "higgs.gz"
    lines = [",".join([str(index % 2)] + ["0"] * 28) for index in range(10)]
    with gzip.open(raw, "wt", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    config = _dataset("higgs", "gzip_csv", (SourceConfig(Path("higgs.gz"), "all", 10),), "tail_test_stratified_validation", {"strategy": "tail_test_stratified_validation", "development_rows": 8, "validation_size": 2, "official_test_size": 2}, (7,), 28, {"0": "0", "1": "1"}, "first")

    with pytest.raises(ValueError, match="exceeds train rows"):
        prepare_dataset(_experiment(config), "higgs", tmp_path)

    assert not (tmp_path / "processed/higgs/.complete").exists()


def test_prepare_dataset_source_count_mismatch_fails_explicitly(tmp_path: Path) -> None:
    """A configured source count mismatch must invalidate publication."""
    raw = tmp_path / "kdd.gz"
    record = ",".join(["0"] * 41 + ["normal."])
    with gzip.open(raw, "wt", encoding="utf-8") as stream:
        stream.write(record + "\n")
    config = _dataset("kdd", "gzip_csv", (SourceConfig(Path("kdd.gz"), "all", 2),), "duplicate_groups", {"strategy": "duplicate_groups", "train_ratio": 0.6, "validation_ratio": 0.2, "internal_test_ratio": 0.2}, (1,), 41, {"normal.": "normal", "*": "attack"}, "last")

    with pytest.raises(ValueError, match="expected 2 rows, found 1"):
        prepare_dataset(_experiment(config), "kdd", tmp_path)

    assert not (tmp_path / "processed/kdd/.complete").exists()
