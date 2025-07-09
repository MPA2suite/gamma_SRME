import os
from collections.abc import Callable
from glob import glob
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm


def glob2df(
    file_pattern: str,
    data_loader: Callable[[Any], pd.DataFrame] = None,
    pbar: bool = True,
    max_files=None,
    **load_options: Any,
) -> pd.DataFrame:
    """Merge multiple data files matching a glob pattern into a single dataframe.

    Args:
        file_pattern (str): Glob pattern for file matching (e.g., '*.csv').
        data_loader (Callable[[Any], pd.DataFrame], optional): Function for loading
            individual files. Defaults to pd.read_csv for CSVs, otherwise pd.read_json.
        show_progress (bool, optional): Show progress bar during file loading. Defaults to True.
        **load_options: Additional options passed to the data loader (like pd.read_csv or pd.read_json).

    Returns:
        pd.DataFrame: A single DataFrame combining the data from all matching files.

    Raises:
        FileNotFoundError: If no files match the given glob pattern.
    """
    # Choose the appropriate data loading function based on file extension if not provided
    if data_loader is None:
        if ".csv" in file_pattern.lower():
            data_loader = pd.read_csv
        else:
            data_loader = pd.read_json

    # Find all files matching the given pattern
    matched_files = glob(file_pattern)
    if not matched_files:
        raise FileNotFoundError(f"No files matched the pattern: {file_pattern}")

    if max_files is not None:
        max_index = min(len(matched_files), max_files)
        matched_files = matched_files[:max_index]

    # Load data from each file into a dataframe
    dataframes = []
    for file_path in tqdm(matched_files, disable=not pbar):
        df = data_loader(file_path, **load_options)
        dataframes.append(df)

    # Combine all loaded dataframes into one
    combined_df = pd.concat(dataframes, ignore_index=True)

    return combined_df
