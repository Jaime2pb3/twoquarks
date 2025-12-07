"""
Simple logging utilities for experiments.

Provides a minimal CSVLogger that creates the output directory if needed,
writes a header once, and appends rows as dictionaries.
"""

from __future__ import annotations

import csv
import os
from typing import Mapping, Sequence


class CSVLogger:
    """
    Minimal CSV logger for experiment metrics.

    Parameters
    ----------
    filepath : str
        Path to the CSV file to create/append.
    fieldnames : Sequence[str]
        Ordered list of column names to use as the CSV header.

    Notes
    -----
    - The directory containing `filepath` is created if it does not exist.
    - The file is overwritten when the logger is constructed.
    - Each call to `log` appends a single row.
    """

    def __init__(self, filepath: str, fieldnames: Sequence[str]) -> None:
        self.filepath = filepath
        # Handle the case where filepath is in the current directory.
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        self.fieldnames = list(fieldnames)
        self._init_file()

    def _init_file(self) -> None:
        """Create or overwrite the CSV file and write the header row."""
        with open(self.filepath, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def log(self, row_dict: Mapping[str, object]) -> None:
        """
        Append a single row to the CSV file.

        Parameters
        ----------
        row_dict : Mapping[str, object]
            Dictionary mapping field name to value. Missing keys will be
            written as empty cells; extra keys are ignored.
        """
        # Project to known fieldnames to avoid surprises.
        row = {k: row_dict.get(k, "") for k in self.fieldnames}
        with open(self.filepath, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)
