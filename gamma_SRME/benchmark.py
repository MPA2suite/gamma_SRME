import traceback
import warnings
from typing import Any

import numpy as np
import pandas as pd

from gamma_SRME.phonons import calc_gamma_grun_dict
from gamma_SRME.utils import check_imaginary_freqs


DEFAULT_LIST2NP_COLS = {
    "max_stress",
    "weights",
    "q_points",
    "frequencies",
    "gruneisen",
    "frequencies_plus",
    "frequencies_minus",
    "mode_gamma",
}


def fill_na_in_list(lst: list, y: Any) -> np.ndarray:
    return np.asarray([y if pd.isna(x) else x for x in lst])


def process_gruneisen_descriptors(
    df_mlp_filtered: pd.DataFrame,
    df_dft_results: pd.DataFrame,
    temperatures: list[int] | None = None,
) -> pd.DataFrame:
    # df_mlp_filtered = df_mlp_filtered.map(np.asarray)
    # df_dft_results = df_dft_results.map(np.asarray)

    from gamma_SRME import TEMPERATURES

    if temperatures is None:
        temperatures = TEMPERATURES

    mlp_list2np_cols = [col for col in DEFAULT_LIST2NP_COLS if col in df_mlp_filtered]
    df_mlp_filtered[mlp_list2np_cols] = df_mlp_filtered[mlp_list2np_cols].map(
        np.asarray
    )

    df_mlp_filtered["gruneisen"] = df_mlp_filtered["gruneisen"].map(
        lambda x: np.asarray(x, dtype=float)
    )

    dft_list2np_cols = [col for col in DEFAULT_LIST2NP_COLS if col in df_dft_results]
    df_dft_results[dft_list2np_cols] = df_dft_results[dft_list2np_cols].map(np.asarray)

    # Remove precomputed columns
    columns_to_remove = ["SRD", "SRE", "SRME", "DFT_mode_gamma"]
    if any([col in df_mlp_filtered for col in columns_to_remove]):
        df_mlp_filtered = df_mlp_filtered.drop(
            columns=[col for col in columns_to_remove if col in df_mlp_filtered.columns]
        )

    if "mode_gamma" not in df_mlp_filtered:
        df_mlp_filtered["mode_gamma"] = df_mlp_filtered.apply(
            lambda x: calc_gamma_grun_dict(x, temperatures=temperatures), axis=1
        )

    if "DFT_mode_gamma" not in df_mlp_filtered:
        df_dft_results["mode_gamma"] = df_dft_results.apply(
            lambda x: calc_gamma_grun_dict(x, temperatures=temperatures), axis=1
        )

    df_mlp_filtered["gamma"] = df_mlp_filtered["mode_gamma"].apply(
        lambda x: (
            np.sum(x, axis=tuple(range(1, x.ndim)))
            if isinstance(x, np.ndarray)
            else np.nan
        )
    )

    df_dft_results["gamma"] = df_dft_results["mode_gamma"].apply(
        lambda x: (
            np.sum(x, axis=tuple(range(1, x.ndim)))
            if isinstance(x, np.ndarray)
            else np.nan
        )
    )

    df_mlp_filtered["SRD"] = (
        2
        * (
            df_mlp_filtered["gamma"].apply(np.abs)
            - df_dft_results["gamma"].apply(np.abs)
        )
        / (
            df_mlp_filtered["gamma"].apply(np.abs)
            + df_dft_results["gamma"].apply(np.abs)
        )
    )

    # turn temperature list to the first temperature (300K) TODO: allow multiple temperatures to be tested
    df_mlp_filtered["SRD"] = df_mlp_filtered["SRD"].apply(
        lambda x: x[0] if not isinstance(x, float) else x
    )

    # We substitute NaN values with 0 predicted conductivity, yielding -2 for SRD
    df_mlp_filtered["SRD"] = df_mlp_filtered["SRD"].fillna(-2)

    df_mlp_filtered["SRE"] = df_mlp_filtered["SRD"].abs()

    df_mlp_filtered["SRME"] = calculate_SRME_dataframes(df_mlp_filtered, df_dft_results)

    df_mlp_filtered["DFT_gamma"] = df_dft_results["gamma"]

    df_mlp_filtered["DFT_mode_gamma"] = df_dft_results["mode_gamma"]

    columns_to_remove = []
    df_mlp_filtered = df_mlp_filtered.drop(
        columns=[col for col in columns_to_remove if col in df_mlp_filtered]
    )

    # TODO: Add column reason for SRME = 2

    # TODO: round to 4-5 decimals

    return df_mlp_filtered


def get_metrics(df_mlp_filtered: pd.DataFrame) -> tuple[float, float, float, float]:
    if "SRE" in df_mlp_filtered:
        mSRE = df_mlp_filtered["SRE"].mean()
        rmseSRE = ((df_mlp_filtered["SRE"] - mSRE) ** 2).mean() ** 0.5
    else:
        mSRE = rmseSRE = np.nan

    if "SRME" in df_mlp_filtered:
        mSRME = df_mlp_filtered["SRME"].mean()
        rmseSRME = ((df_mlp_filtered["SRME"] - mSRME) ** 2).mean() ** 0.5
    else:
        mSRME = rmseSRME = np.nan

    return mSRE, mSRME, rmseSRE, rmseSRME


def get_success_metrics(df_mlp):
    df_mlp_reduced = df_mlp[df_mlp["SRME"] != 2.0]
    mSRE = df_mlp_reduced["SRE"].mean()
    mSRME = df_mlp_reduced["SRME"].mean()
    return mSRE, mSRME


def calculate_kappa_ave(kappa: np.ndarray) -> float | np.ndarray:
    if np.any(pd.isna(kappa)):
        return np.nan
    _kappa = np.asarray(kappa)

    try:
        kappa_ave = _kappa[..., :3].mean(axis=-1)
    except Exception as e:
        warnings.warn(f"Failed to calculate kappa_ave: {e!r}")
        warnings.warn(traceback.format_exc())
        return np.nan

    return kappa_ave


def calculate_SRME_dataframes(
    df_mlp: pd.DataFrame, df_dft: pd.DataFrame
) -> list[float]:
    srme_list = []
    for idx, row_mlp in df_mlp.iterrows():
        row_dft = df_dft.loc[idx]
        try:
            if row_mlp.get("imaginary_freqs"):
                if row_mlp["imaginary_freqs"] in ["True", True]:
                    srme_list.append(2)
                    continue
            if "relaxed_space_group_number" in row_mlp:
                if "initial_space_group_number" in row_mlp:
                    if (
                        row_mlp["relaxed_space_group_number"]
                        != row_mlp["initial_space_group_number"]
                    ):
                        srme_list.append(2)
                        continue
                elif "symm.no" in row_dft:
                    if row_mlp["relaxed_space_group_number"] != row_dft["symm.no"]:
                        srme_list.append(2)
                        continue
            result = calculate_grun_SRME(row_mlp, row_dft)
            srme_list.append(result[0])  # append the first temperature SRME

            # Idea: Multiple temperature tests.
        except Exception as e:
            warnings.warn(f"Failed to calculate SRME for {idx}: {e!r}")
            warnings.warn(traceback.format_exc())
            srme_list.append(2)

    return srme_list


def calculate_SRME(mlp_dict: pd.Series, dft_dict: pd.Series) -> list[float]:
    if np.all(pd.isna(mlp_dict["mode_gamma"])):
        return [2]
    if np.all(pd.isna(dft_dict["mode_gamma"])):
        return [2]  # np.nan
    if np.all(pd.isna(mlp_dict["gamma"])):
        return [2]
    if np.all(pd.isna(dft_dict["gamma"])):
        return [2]  # np.nan
    if np.any(pd.isna(mlp_dict["weights"])):
        return [2]  # np.nan

    mlp_mode_gamma = mlp_dict["mode_gamma"]
    dft_mode_gamma = dft_dict["mode_gamma"]

    # calculating microscopic error for all temperatures
    microscopic_error = np.abs(mlp_mode_gamma - dft_mode_gamma).sum(
        axis=tuple(range(1, mlp_mode_gamma.ndim))
    )  # summing qpoints and bands, but not temperatures

    SRME = 2 * microscopic_error / (mlp_dict["gamma"] + dft_dict["gamma"])

    return SRME


def calculate_grun_SRME(mlp_dict: pd.Series, dft_dict: pd.Series) -> list[float]:
    if np.all(pd.isna(mlp_dict["mode_gamma"])):
        return [2]
    if np.all(pd.isna(dft_dict["mode_gamma"])):
        return [2]  # np.nan
    if np.all(pd.isna(mlp_dict["gamma"])):
        return [2]
    if np.all(pd.isna(dft_dict["gamma"])):
        return [2]  # np.nan
    if np.any(pd.isna(mlp_dict["weights"])):
        return [2]  # np.nan

    mlp_mode_gamma = mlp_dict["mode_gamma"]
    dft_mode_gamma = dft_dict["mode_gamma"]

    # calculating microscopic error for all temperatures
    microscopic_error = np.abs(mlp_mode_gamma - dft_mode_gamma).sum(
        axis=tuple(range(1, mlp_mode_gamma.ndim))
    )  # summing qpoints and bands, but not temperatures

    mlp_abs_sum = np.abs(mlp_mode_gamma).sum(axis=tuple(range(1, mlp_mode_gamma.ndim)))

    dft_abs_sum = np.abs(dft_mode_gamma).sum(axis=tuple(range(1, mlp_mode_gamma.ndim)))

    SRME = 2 * microscopic_error / (mlp_abs_sum + dft_abs_sum)

    return SRME
