"""Streaming split and nested-sample generation for supported datasets."""

from __future__ import annotations

import bz2
import csv
import gzip
import hashlib
import io
import json
import logging
import math
import os
import resource
import shutil
import sqlite3
import time
import zipfile
from array import array
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterator, TextIO

import numpy as np

from bigdata_model_selection.config import DatasetConfig, ExperimentsConfig

logger = logging.getLogger(__name__)

SPLIT_TRAIN = 0
SPLIT_VALIDATION = 1
SPLIT_INTERNAL_TEST = 2
SPLIT_OFFICIAL_TEST = 3
SPLIT_NAMES = ("train", "validation", "internal_test", "official_test")
MANIFEST_HEADER = ("observation_id", "logical_label", "source_label")


@dataclass(frozen=True)
class NativeRow:
    """One validated native-format record without retained feature parsing."""

    ordinal: int
    observation_id: str
    native_line: str
    source_label: str
    logical_label: str
    group_key: str | None = None
    source_partition: str | None = None


class DisjointSet:
    """Memory-conscious union-find for typed RLCP entity nodes."""

    def __init__(self) -> None:
        self._index: dict[str, int] = {}
        self._parent = array("I")
        self._rank = bytearray()

    def _add(self, key: str) -> int:
        index = self._index.get(key)
        if index is not None:
            return index
        index = len(self._parent)
        self._index[key] = index
        self._parent.append(index)
        self._rank.append(0)
        return index

    def find(self, index: int) -> int:
        """Return a node's canonical component with path compression."""
        while self._parent[index] != index:
            self._parent[index] = self._parent[self._parent[index]]
            index = self._parent[index]
        return index

    def union(self, left: str, right: str) -> None:
        """Join two typed entity nodes by rank."""
        left_root = self.find(self._add(left))
        right_root = self.find(self._add(right))
        if left_root == right_root:
            return
        if self._rank[left_root] < self._rank[right_root]:
            left_root, right_root = right_root, left_root
        self._parent[right_root] = left_root
        if self._rank[left_root] == self._rank[right_root]:
            self._rank[left_root] += 1

    def component(self, key: str) -> int:
        """Return the component for an entity seen during union construction."""
        try:
            return self.find(self._index[key])
        except KeyError as error:
            raise ValueError(f"RLCP entity missing from first pass: {key}") from error


def _safe_member(name: str) -> None:
    """Reject archive members that could escape their logical root."""
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"unsafe ZIP member path: {name}")


def _logical_label(source: str, config: DatasetConfig) -> str:
    """Map an exact source label or the explicit wildcard."""
    if source in config.labels:
        return config.labels[source]
    if "*" in config.labels:
        return config.labels["*"]
    raise ValueError(f"{config.name}: unsupported source label {source!r}")


def _csv_row(line: str, expected: int, location: str) -> list[str]:
    """Parse one CSV record and enforce its exact field count."""
    try:
        fields = next(csv.reader([line]))
    except csv.Error as error:
        raise ValueError(f"{location}: invalid CSV record") from error
    if len(fields) != expected:
        raise ValueError(f"{location}: expected {expected} fields, found {len(fields)}")
    return fields


def _gzip_rows(config: DatasetConfig, root: Path) -> Iterator[NativeRow]:
    """Stream and validate KDD or HIGGS gzip CSV records."""
    source = config.sources[0]
    path = root / source.path
    count = 0
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        for count, raw_line in enumerate(stream, 1):
            line = raw_line.rstrip("\r\n")
            fields = _csv_row(line, config.feature_count + 1, f"{path}:{count}")
            source_label = fields[0] if config.label_position == "first" else fields[-1]
            if config.name == "higgs":
                try:
                    numeric = float(source_label)
                except ValueError as error:
                    raise ValueError(f"{path}:{count}: HIGGS label must be numeric") from error
                if not math.isfinite(numeric) or numeric not in {0.0, 1.0}:
                    raise ValueError(f"{path}:{count}: HIGGS label must be exactly 0 or 1")
                logical = str(int(numeric))
                logical = _logical_label(logical, config)
                fields[0] = logical
                line = ",".join(fields)
            else:
                logical = _logical_label(source_label, config)
            record_key = line if config.name == "kdd" else None
            yield NativeRow(count, f"{source.path.name}:{count}", line, source_label, logical, record_key)
    if source.expected_rows is not None and count != source.expected_rows:
        raise ValueError(f"{path}: expected {source.expected_rows} rows, found {count}")


def _epsilon_rows(config: DatasetConfig, root: Path) -> Iterator[NativeRow]:
    """Stream exact dense LIBSVM epsilon partitions."""
    partitions = [source.partition for source in config.sources]
    if len(partitions) != len(set(partitions)):
        raise ValueError("epsilon: source partition names must be unique")
    global_ordinal = 0
    for source in config.sources:
        count = 0
        path = root / source.path
        with bz2.open(path, "rt", encoding="utf-8", newline="") as stream:
            for count, raw_line in enumerate(stream, 1):
                global_ordinal += 1
                line = raw_line.rstrip("\r\n")
                parts = line.split(maxsplit=1)
                if len(parts) != 2 or not parts[1]:
                    raise ValueError(f"{path}:{count}: expected a label and LIBSVM feature payload")
                source_label = parts[0]
                logical = _logical_label(source_label, config)
                observation_id = f"{source.partition}:{count}"
                yield NativeRow(
                    global_ordinal,
                    observation_id,
                    line,
                    source_label,
                    logical,
                    source.partition,
                    source.partition,
                )
        if source.expected_rows is not None and count != source.expected_rows:
            raise ValueError(f"{path}: expected {source.expected_rows} rows, found {count}")


def _rlcp_rows(config: DatasetConfig, root: Path) -> Iterator[NativeRow]:
    """Stream nested RLCP ZIP members without extracting them."""
    path = root / config.sources[0].path
    ordinal = 0
    with zipfile.ZipFile(path) as outer:
        for info in outer.infolist():
            _safe_member(info.filename)
        expected = [f"block_{index}.zip" for index in range(1, (config.block_count or 0) + 1)]
        block_infos = [info for info in outer.infolist() if info.filename.lower().endswith(".zip")]
        available = {PurePosixPath(info.filename).name: info for info in block_infos}
        if len(available) != len(block_infos) or set(available) != set(expected):
            raise ValueError(f"{path}: expected nested blocks {expected}")
        for block_name in expected:
            with outer.open(available[block_name]) as nested_stream, zipfile.ZipFile(nested_stream) as nested:
                for info in nested.infolist():
                    _safe_member(info.filename)
                csv_members = [info for info in nested.infolist() if info.filename.lower().endswith(".csv")]
                if len(csv_members) != 1:
                    raise ValueError(f"{block_name}: expected exactly one CSV member")
                with nested.open(csv_members[0]) as binary, io.TextIOWrapper(binary, encoding="utf-8", newline="") as text:
                    reader = csv.reader(text)
                    header = tuple(next(reader, ()))
                    if header != config.columns:
                        raise ValueError(f"{block_name}: RLCP header mismatch")
                    for block_ordinal, fields in enumerate(reader, 1):
                        ordinal += 1
                        if len(fields) != len(config.columns):
                            raise ValueError(f"{block_name}:{block_ordinal}: RLCP field count mismatch")
                        source_label = fields[-1]
                        logical = _logical_label(source_label, config)
                        row_id = f"{path.name}:{block_name}:{csv_members[0].filename}:{block_ordinal}"
                        group = f"left:{fields[0]}\0right:{fields[1]}"
                        yield NativeRow(
                            ordinal,
                            row_id,
                            ",".join(fields),
                            source_label,
                            logical,
                            group,
                            block_name,
                        )


def _rows(config: DatasetConfig, root: Path) -> Iterator[NativeRow]:
    """Route a validated contract to its explicit native reader."""
    if config.format == "nested_zip_csv":
        yield from _rlcp_rows(config, root)
    elif config.format == "gzip_csv":
        yield from _gzip_rows(config, root)
    elif config.format == "bzip2_libsvm":
        yield from _epsilon_rows(config, root)
    else:
        raise ValueError(f"unsupported reader format: {config.format}")


def _write_compact(path: Path, values: Iterator[int], typecode: str) -> int:
    """Persist compact integers incrementally and return their count."""
    count = 0
    chunk = array(typecode)
    with path.open("wb") as stream:
        for value in values:
            chunk.append(value)
            count += 1
            if len(chunk) >= 65_536:
                chunk.tofile(stream)
                chunk = array(typecode)
        chunk.tofile(stream)
    return count


def _label_scan(config: DatasetConfig, root: Path, work: Path) -> tuple[np.memmap, list[str]]:
    """Validate all records and persist only compact logical label codes."""
    label_names = sorted(set(config.labels.values()))
    codes = {label: index for index, label in enumerate(label_names)}
    count = _write_compact(work / "labels.bin", (codes[row.logical_label] for row in _rows(config, root)), "B")
    if count == 0:
        raise ValueError(f"{config.name}: source contains no rows")
    return np.memmap(work / "labels.bin", dtype=np.uint8, mode="r", shape=(count,)), label_names


def _group_ids_rlcp(config: DatasetConfig, root: Path, work: Path) -> np.memmap:
    """Build typed bipartite components and persist each row's root ID."""
    dsu = DisjointSet()
    for row in _rows(config, root):
        assert row.group_key is not None
        left, right = row.group_key.split("\0", 1)
        dsu.union(left, right)
    count = _write_compact(
        work / "groups.bin",
        (dsu.component(row.group_key.split("\0", 1)[0]) for row in _rows(config, root) if row.group_key),
        "I",
    )
    return np.memmap(work / "groups.bin", dtype=np.uint32, mode="r", shape=(count,))


def _group_ids_kdd(config: DatasetConfig, root: Path, work: Path) -> np.memmap:
    """Assign exact duplicate records through a disk-backed SQLite key table."""
    database = sqlite3.connect(work / "duplicates.sqlite3")
    database.execute("CREATE TABLE records (id INTEGER PRIMARY KEY, content TEXT NOT NULL UNIQUE)")

    def ids() -> Iterator[int]:
        for row in _rows(config, root):
            assert row.group_key is not None
            cursor = database.execute("INSERT OR IGNORE INTO records(content) VALUES (?)", (row.group_key,))
            group_id = cursor.lastrowid if cursor.rowcount else database.execute(
                "SELECT id FROM records WHERE content = ?", (row.group_key,)
            ).fetchone()[0]
            yield int(group_id)
        database.commit()

    try:
        count = _write_compact(work / "groups.bin", ids(), "I")
    finally:
        database.close()
    return np.memmap(work / "groups.bin", dtype=np.uint32, mode="r", shape=(count,))


def _seeded_rank(seed: int, namespace: str, value: int) -> int:
    """Return a stable rank independent of Python hash randomization."""
    payload = f"{seed}:{namespace}:{value}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _grouped_splits(
    groups: np.ndarray,
    labels: np.ndarray,
    config: DatasetConfig,
    seed: int,
) -> tuple[np.ndarray, dict[str, int], dict[str, object]]:
    """Assign complete groups toward row and binary-class targets."""
    unique, counts = np.unique(groups, return_counts=True)
    ratios = np.array([config.split["train_ratio"], config.split["validation_ratio"], config.split["internal_test_ratio"]])
    targets = ratios * len(groups)
    label_count = int(labels.max()) + 1
    group_labels = np.bincount(
        groups.astype(np.int64) * label_count + labels.astype(np.int64),
        minlength=(int(unique.max()) + 1) * label_count,
    ).reshape(-1, label_count)[unique]
    class_targets = np.outer(ratios, group_labels.sum(axis=0))
    actual = np.zeros(3, dtype=np.int64)
    actual_labels = np.zeros_like(class_targets)
    group_split = np.zeros(int(unique.max()) + 1, dtype=np.uint8)
    order = sorted(
        range(len(unique)),
        key=lambda index: (-int(counts[index]), _seeded_rank(seed, config.name, int(unique[index]))),
    )
    for index in order:
        scores = []
        for split_index in range(3):
            candidate_rows = actual.copy()
            candidate_labels = actual_labels.copy()
            candidate_rows[split_index] += counts[index]
            candidate_labels[split_index] += group_labels[index]
            row_error = np.sum(((candidate_rows - targets) / max(len(groups), 1)) ** 2)
            class_error = np.sum(
                ((candidate_labels - class_targets) / np.maximum(class_targets, 1)) ** 2
            )
            scores.append((float(row_error + class_error), split_index))
        split = min(scores)[1]
        group_split[int(unique[index])] = split
        actual[split] += int(counts[index])
        actual_labels[split] += group_labels[index]
    memberships = group_split[groups]
    diagnostics = {
        "largest_group_rows": int(counts.max()),
        "split_class_counts": {
            SPLIT_NAMES[index]: {str(code): int(value) for code, value in enumerate(actual_labels[index])}
            for index in range(3)
        },
    }
    return memberships, {SPLIT_NAMES[index]: int(value) for index, value in enumerate(actual)}, diagnostics


def _source_block_splits(
    rows: Iterator[NativeRow], config: DatasetConfig
) -> tuple[np.ndarray, dict[str, int]]:
    """Assign RLCP rows to the configured source blocks."""
    block_to_split = {
        block: split
        for split, key in (
            (SPLIT_TRAIN, "train_blocks"),
            (SPLIT_VALIDATION, "validation_blocks"),
            (SPLIT_INTERNAL_TEST, "internal_test_blocks"),
        )
        for block in config.split[key]
    }
    memberships = []
    for row in rows:
        if row.source_partition not in block_to_split:
            raise ValueError(f"rlcp: source block is not assigned: {row.source_partition!r}")
        memberships.append(block_to_split[row.source_partition])
    values = np.asarray(memberships, dtype=np.uint8)
    counts = Counter(SPLIT_NAMES[int(value)] for value in values)
    return values, dict(counts)


def _source_block_overlap(
    config: DatasetConfig,
    root: Path,
    split_by_block: dict[str, int],
    work: Path,
) -> dict[str, int]:
    """Count typed RLCP entity IDs shared across configured source splits."""
    database = sqlite3.connect(work / "entity_overlap.sqlite3")
    database.execute("CREATE TABLE entities (entity TEXT PRIMARY KEY, mask INTEGER NOT NULL)")
    try:
        for row in _rows(config, root):
            split = split_by_block[row.source_partition]
            left, right = row.group_key.split("\0", 1)
            for entity in (left, right):
                database.execute(
                    "INSERT INTO entities(entity, mask) VALUES (?, ?) "
                    "ON CONFLICT(entity) DO UPDATE SET mask = mask | excluded.mask",
                    (entity, 1 << split),
                )
        database.commit()
        masks = Counter()
        for (mask,) in database.execute("SELECT mask FROM entities"):
            if mask & 1 and mask & 2:
                masks["train_validation"] += 1
            if mask & 1 and mask & 4:
                masks["train_internal_test"] += 1
            if mask & 2 and mask & 4:
                masks["validation_internal_test"] += 1
        return dict(masks)
    finally:
        database.close()


def _quota(counts: np.ndarray, requested: int) -> np.ndarray:
    """Allocate an exact proportional stratified size by largest remainder."""
    if requested > int(counts.sum()):
        raise ValueError(f"requested size {requested} exceeds available train rows {int(counts.sum())}")
    exact = counts * (requested / counts.sum())
    quota = np.floor(exact).astype(np.int64)
    remainder = requested - int(quota.sum())
    order = np.argsort(-(exact - quota), kind="stable")
    for index in order[:remainder]:
        quota[index] += 1
    if np.any(quota > counts):
        raise ValueError(f"cannot produce exact stratified size {requested}")
    return quota


def _select_stratified(mask: np.ndarray, labels: np.ndarray, size: int, seed: int) -> np.ndarray:
    """Select an exact seeded stratified subset as a boolean membership array."""
    selected = np.zeros(len(labels), dtype=bool)
    label_count = int(labels.max()) + 1
    counts = np.array([np.count_nonzero(mask & (labels == code)) for code in range(label_count)])
    quotas = _quota(counts, size)
    for code, requested in enumerate(quotas):
        eligible = np.flatnonzero(mask & (labels == code))
        rng = np.random.default_rng(seed + code)
        chosen = rng.choice(eligible, int(requested), replace=False, shuffle=False)
        selected[chosen] = True
    return selected


def _fixed_splits(config: DatasetConfig, labels: np.ndarray, seed: int) -> tuple[np.ndarray, dict[str, int]]:
    """Build official-test-aware HIGGS or epsilon memberships."""
    memberships = np.full(len(labels), SPLIT_TRAIN, dtype=np.uint8)
    if config.name == "higgs":
        development = int(config.split["development_rows"])
        official_test = int(config.split["official_test_size"])
        if development + official_test != len(labels):
            raise ValueError("higgs: configured development and official test rows do not match source")
        memberships[development:] = SPLIT_OFFICIAL_TEST
        validation_pool = np.arange(len(labels)) < development
    else:
        train_rows = config.sources[0].expected_rows
        if train_rows is None or train_rows >= len(labels):
            raise ValueError("epsilon: official train/test source counts are required")
        memberships[train_rows:] = SPLIT_OFFICIAL_TEST
        validation_pool = np.arange(len(labels)) < train_rows
    validation = _select_stratified(validation_pool, labels, int(config.split["validation_size"]), seed)
    memberships[validation] = SPLIT_VALIDATION
    counts = Counter(SPLIT_NAMES[int(value)] for value in memberships)
    return memberships, dict(counts)


def _tier_memberships(train: np.ndarray, labels: np.ndarray, tiers: tuple[int, ...], seed: int) -> list[np.ndarray]:
    """Generate exact nested, seeded, stratified train-only tiers."""
    if tiers[-1] > int(train.sum()):
        raise ValueError(f"requested tier {tiers[-1]} exceeds train rows {int(train.sum())}")
    memberships = [np.zeros(len(labels), dtype=bool) for _ in tiers]
    label_count = int(labels.max()) + 1
    counts = np.array([np.count_nonzero(train & (labels == code)) for code in range(label_count)])
    quotas = [_quota(counts, tier) for tier in tiers]
    for code in range(label_count):
        eligible = np.flatnonzero(train & (labels == code))
        rng = np.random.default_rng(seed + 10_000 + code)
        ranked = rng.choice(eligible, int(quotas[-1][code]), replace=False, shuffle=True)
        for tier_index, quota in enumerate(quotas):
            memberships[tier_index][ranked[: int(quota[code])]] = True
    return memberships


def _native_extension(config: DatasetConfig) -> str:
    """Return an unambiguous native-format extension."""
    return "libsvm" if config.format == "bzip2_libsvm" else "csv"


def _publish_rows(
    config: DatasetConfig,
    root: Path,
    publish: Path,
    memberships: np.ndarray,
    tiers: list[np.ndarray],
) -> tuple[dict[str, Counter[str]], dict[str, Counter[str]], list[str]]:
    """Stream records once into split, tier, and sidecar manifest files."""
    extension = _native_extension(config)
    names = [name for code, name in enumerate(SPLIT_NAMES) if np.any(memberships == code)]
    tier_names = [f"train_{size}" for size in config.tiers]
    class_counts = {name: Counter() for name in names + tier_names}
    source_class_counts = {name: Counter() for name in names + tier_names}
    with ExitStack() as stack:
        data = {name: stack.enter_context((publish / f"{name}.{extension}").open("w", encoding="utf-8", newline="")) for name in names + tier_names}
        manifests = {name: csv.writer(stack.enter_context((publish / f"{name}.manifest.csv").open("w", encoding="utf-8", newline=""))) for name in names + tier_names}
        for writer in manifests.values():
            writer.writerow(MANIFEST_HEADER)
        if config.name == "rlcp":
            header = ",".join(config.columns) + "\n"
            for stream in data.values():
                stream.write(header)
        seen = 0
        for index, row in enumerate(_rows(config, root)):
            seen += 1
            split_name = SPLIT_NAMES[int(memberships[index])]
            _write_output_row(data[split_name], manifests[split_name], row)
            class_counts[split_name][row.logical_label] += 1
            source_class_counts[split_name][row.source_label] += 1
            for tier_index, tier in enumerate(tiers):
                if tier[index]:
                    tier_name = tier_names[tier_index]
                    _write_output_row(data[tier_name], manifests[tier_name], row)
                    class_counts[tier_name][row.logical_label] += 1
                    source_class_counts[tier_name][row.source_label] += 1
        if seen != len(memberships):
            raise ValueError(f"{config.name}: row count changed between passes")
    return class_counts, source_class_counts, names


def _write_output_row(stream: TextIO, manifest: csv.writer, row: NativeRow) -> None:
    """Write one native record and its compact provenance sidecar."""
    stream.write(row.native_line + "\n")
    manifest.writerow((row.observation_id, row.logical_label, row.source_label))


def _team_manifest(config: ExperimentsConfig, dataset: DatasetConfig, split_names: list[str]) -> dict[str, object]:
    """Build one shared team manifest without copying any data files."""
    extension = _native_extension(dataset)
    evaluation = {
        name: {"data": f"{name}.{extension}", "manifest": f"{name}.manifest.csv"}
        for name in split_names if name != "train"
    }
    tiers = {
        str(size): {"data": f"train_{size}.{extension}", "manifest": f"train_{size}.manifest.csv"}
        for size in dataset.tiers
    }
    return {"team_ids": list(config.team_ids), "shared_evaluation": evaluation, "train_tiers": tiers}


def _peak_rss_mib() -> float | None:
    """Return process peak RSS in MiB where the platform exposes it."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / 1024 if usage > 0 else None


def prepare_dataset(config: ExperimentsConfig, dataset_name: str, root: Path = Path(".")) -> Path:
    """Prepare one dataset with shared splits and exact nested train tiers.

    Args:
        config: Validated experiment configuration.
        dataset_name: One configured dataset name.
        root: Repository root used to resolve relative paths.

    Returns:
        The published dataset output directory.

    Raises:
        ValueError: If source data or scientific guards violate the contract.
        KeyError: If the dataset name is not configured.
    """
    dataset = config.datasets[dataset_name]
    output = root / dataset.output_dir
    work = output / ".work"
    publish = work / "publish"
    start = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    (output / ".complete").unlink(missing_ok=True)
    shutil.rmtree(work, ignore_errors=True)
    publish.mkdir(parents=True)
    logger.info("Preparing %s with random_state=%d", dataset_name, config.random_state)
    try:
        labels, label_names = _label_scan(dataset, root, work)
        split_diagnostics: dict[str, object] = {}
        if dataset.split_strategy == "source_blocks":
            memberships, actual_splits = _source_block_splits(_rows(dataset, root), dataset)
            split_by_block = {
                block: split
                for split, key in (
                    (SPLIT_TRAIN, "train_blocks"),
                    (SPLIT_VALIDATION, "validation_blocks"),
                    (SPLIT_INTERNAL_TEST, "internal_test_blocks"),
                )
                for block in dataset.split[key]
            }
            split_diagnostics["entity_overlap_counts"] = _source_block_overlap(dataset, root, split_by_block, work)
        elif dataset.split_strategy == "connected_components":
            groups = _group_ids_rlcp(dataset, root, work)
            memberships, actual_splits, split_diagnostics = _grouped_splits(groups, labels, dataset, config.random_state)
        elif dataset.split_strategy == "duplicate_groups":
            groups = _group_ids_kdd(dataset, root, work)
            memberships, actual_splits, split_diagnostics = _grouped_splits(groups, labels, dataset, config.random_state)
        elif dataset.split_strategy in {"tail_test_stratified_validation", "official_test_stratified_validation"}:
            memberships, actual_splits = _fixed_splits(dataset, labels, config.random_state)
        else:
            raise ValueError(f"unsupported split strategy: {dataset.split_strategy}")
        if len(memberships) != len(labels):
            raise ValueError(f"{dataset.name}: membership length mismatch")
        required_splits = {SPLIT_TRAIN, SPLIT_VALIDATION}
        required_splits.add(
            SPLIT_INTERNAL_TEST
            if dataset.split_strategy in {"connected_components", "source_blocks", "duplicate_groups"}
            else SPLIT_OFFICIAL_TEST
        )
        if any(not np.any(memberships == split) for split in required_splits):
            raise ValueError(f"{dataset.name}: split strategy produced an empty required split")
        train = memberships == SPLIT_TRAIN
        tiers = _tier_memberships(train, labels, dataset.tiers, config.random_state)
        if any(np.any(tier & ~train) for tier in tiers) or any(np.any(tiers[index] & ~tiers[index + 1]) for index in range(len(tiers) - 1)):
            raise ValueError(f"{dataset.name}: nested train-only tier guard failed")
        class_counts, source_class_counts, split_names = _publish_rows(dataset, root, publish, memberships, tiers)
        metadata = {
            "dataset": dataset.name,
            "random_state": config.random_state,
            "strategy": dataset.split_strategy,
            "rows": len(labels),
            "logical_labels": label_names,
            "requested_tiers": list(dataset.tiers),
            "actual_tiers": {str(size): int(tier.sum()) for size, tier in zip(dataset.tiers, tiers, strict=True)},
            "requested_split": dataset.split,
            "actual_splits": actual_splits,
            "class_counts": {name: dict(counts) for name, counts in class_counts.items()},
            "source_class_counts": {name: dict(counts) for name, counts in source_class_counts.items()},
            "guards": {
                "deterministic_memberships": True,
                "no_overlap": True,
                "requested_sizes_within_train": True,
                "tiers_nested": True,
                "tiers_train_only": True,
                "unique_observation_ids": True,
            },
            "group_ratio_deviation_rows": _ratio_deviations(dataset, actual_splits, len(labels)),
            "split_diagnostics": split_diagnostics,
            "elapsed_seconds": time.monotonic() - start,
            "peak_rss_mib": _peak_rss_mib(),
        }
        (publish / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (publish / "team_manifest.json").write_text(json.dumps(_team_manifest(config, dataset, split_names), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for path in publish.iterdir():
            os.replace(path, output / path.name)
        (output / ".complete").write_text("complete\n", encoding="ascii")
    except Exception:
        logger.exception("Preparation failed for %s", dataset_name)
        raise
    else:
        shutil.rmtree(work)
    return output


def _ratio_deviations(config: DatasetConfig, actual: dict[str, int], total: int) -> dict[str, float] | None:
    """Report unavoidable grouped-split deviations from requested row ratios."""
    if config.split_strategy not in {"connected_components", "duplicate_groups"}:
        return None
    ratio_keys = {"train": "train_ratio", "validation": "validation_ratio", "internal_test": "internal_test_ratio"}
    return {name: actual.get(name, 0) - float(config.split[key]) * total for name, key in ratio_keys.items()}
