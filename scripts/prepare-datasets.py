#!/usr/bin/env python3
"""Prepare one configured dataset without training or transformations."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from bigdata_model_selection.config import load_config
from bigdata_model_selection.preparation import prepare_dataset


def main() -> None:
    """Load the authoritative contract and prepare the selected dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("rlcp", "kdd", "higgs", "epsilon"))
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/datasets.yaml"))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    prepare_dataset(load_config(args.config), args.dataset, args.root)


if __name__ == "__main__":
    main()
