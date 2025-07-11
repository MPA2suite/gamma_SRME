import os

from gamma_SRME.benchmark import (
    calculate_SRME,
    get_metrics,
)
from gamma_SRME.data import glob2df
from gamma_SRME.relax import two_stage_relax, NO_TILT_MASK

# TODO: create separete phonopy_utils, such that code does not depend on phono3py

from gamma_SRME.utils import (
    check_imaginary_freqs,
    aseatoms2str,
    str2aseatoms,
    log_message,
)

PKG_NAME = "gamma_SRME"
__version__ = "0.0.1"

PKG_DIR = os.path.dirname(__file__)
# repo root directory if editable install, TODO: else the package dir
ROOT = os.path.dirname(PKG_DIR)
DATA_DIR = f"{ROOT}/data"  # directory to store default data
DATA_DIR = f"{DATA_DIR}"
STRUCTURES_FILE = "phononDB-PBE-structures.extxyz"
STRUCTURES = f"{DATA_DIR}/{STRUCTURES_FILE}"

DFT_NAC_REF_FILE = "gruneisen_DFT_NAC-Gonze.json.gz"
DFT_NONAC_REF_FILE = "gruneisen_DFT_noNAC.json.gz"
DFT_NAC_REF = f"{DATA_DIR}/{DFT_NAC_REF_FILE}"
DFT_NONAC_REF = f"{DATA_DIR}/{DFT_NONAC_REF_FILE}"

# DFT_NONAC_REF = "/mnt/scratch2/q13camb_scratch/bp443/foundational_TC/visionre_thermal_expansion/mlps/3rd_dft_nonac_gruneisen.json.gz"


###
pkg_is_editable = True


######
TEMPERATURES = [300]
ID = "mp_id"


__all__ = [
    "calculate_mode_kappa_TOT",
    "calculate_kappa_ave",
    "calculate_SRME",
    "get_metrics",
    "glob2df",
    "two_stage_relax",
    "check_imaginary_freqs",
    "aseatoms2str",
    "str2aseatoms",
    "log_message",
    "log_symmetry",
    "get_spacegroup_number",
    "NO_TILT_MASK",
]
