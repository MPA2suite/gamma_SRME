import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable
from functools import partial
import traceback

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from tqdm import tqdm
from phonopy.api_phonopy import Phonopy
from phonopy.api_gruneisen import PhonopyGruneisen
from phonopy.phonon.thermal_properties import mode_cv
from phonopy.units import EvTokJmol, THzToEv
from phonopy.api_qha import PhonopyQHA


from gamma_SRME.phonopy_utils import (
    aseatoms2phonopy,
    get_chemical_formula,
    load_phonopy,
)
from gamma_SRME.utils import (
    MODE_KAPPA_THRESHOLD,
    log_message,
    scale_atoms_volume,
)
from gamma_SRME.relax import simple_relax

FREQ_CUTOFF = 1e-3


def calculate_fc2_phonopy_set(
    phonons: Phonopy,
    calculator: Calculator,
    log: bool = True,
    pbar_kwargs: dict[str, Any] = {},
) -> np.ndarray:
    # calculate FC2 force set

    log_message(
        f"Computing FC2 force set in {get_chemical_formula(phonons)}.", output=log
    )

    forces = []
    nat = len(phonons.supercell)

    for sc in tqdm(
        phonons.supercells_with_displacements,
        desc=f"FC2 calculation: {get_chemical_formula(phonons,empirical=True)}",
        **pbar_kwargs,
    ):
        if sc is not None:
            atoms = Atoms(sc.symbols, cell=sc.cell, positions=sc.positions, pbc=True)
            atoms.calc = calculator
            f = atoms.get_forces()
        else:
            f = np.zeros((nat, 3))
        forces.append(f)

    # append forces
    force_set = np.array(forces)
    phonons.forces = force_set
    return force_set


def init_phonopy(
    atoms: Atoms,
    fc2_supercell: np.ndarray | None = None,
    primitive_matrix: Any = "auto",
    log: str | Path | bool = True,
    symprec: float = 1e-5,
    displacement_distance: float = 0.03,
    **kwargs: Any,
) -> tuple[Phonopy, list[Any]]:

    if not log:
        log_level = 0
    elif log is not None:
        log_level = 1

    if fc2_supercell is not None:
        _fc2_supercell = fc2_supercell
    else:
        if "fc2_supercell" in atoms.info.keys():
            _fc2_supercell = atoms.info["fc2_supercell"]
        else:
            raise ValueError(
                f'{atoms.get_chemical_formula(mode="metal")=} "fc2_supercell" was not found in atoms.info and was not provided as an argument when calculating force sets.'
            )

    # Initialise Phonopy object
    phonons = aseatoms2phonopy(
        atoms,
        fc2_supercell=_fc2_supercell,
        primitive_matrix=primitive_matrix,
        symprec=symprec,
        log_level=log_level,
        **kwargs,
    )

    phonons.generate_displacements(distance=displacement_distance)

    return phonons


def get_fc2_and_freqs(
    phonons: Phonopy,
    calculator: Calculator | None = None,
    q_mesh: np.ndarray | None = None,
    symmetrize_fc2=True,
    log: str | Path | bool = True,
    pbar_kwargs: dict[str, Any] = {"leave": False},
    **kwargs: Any,
) -> tuple[Phonopy, np.ndarray, np.ndarray]:

    if calculator is None:
        raise ValueError(
            f'{get_chemical_formula(phonons)} "calculator" was provided when calculating fc2 force sets.'
        )

    fc2_set = calculate_fc2_phonopy_set(
        phonons, calculator, log=log, pbar_kwargs=pbar_kwargs
    )

    phonons.produce_force_constants(show_drift=False)

    if symmetrize_fc2:
        phonons.symmetrize_force_constants(show_drift=False)

    if q_mesh is not None:
        phonons.run_mesh(q_mesh, **kwargs)
        freqs = phonons.get_mesh_dict()["frequencies"]
    else:
        freqs = []

    return phonons, fc2_set, freqs


def load_force_sets(phonons: Phonopy, fc2_set: np.ndarray) -> Phonopy:
    phonons.forces = fc2_set
    phonons.produce_force_constants(symmetrize_fc2=True)
    return phonons


def gruneisen_scaled(
    origin: Phonopy,
    plus: Phonopy,
    minus: Phonopy,
    q_mesh: np.ndarray,
):

    grun = PhonopyGruneisen(phonon=origin, phonon_plus=plus, phonon_minus=minus)

    grun.set_mesh(mesh=q_mesh)

    q_points, weights, frequeincies, eigenvectors, gruneisen = grun.get_mesh()

    gru_dict = {
        "q_points": q_points,
        "weights": weights,
        "frequencies": frequeincies,
        "eigenvectors": eigenvectors,
        "gruneisen": gruneisen,
    }

    return gru_dict


def calculate_gruneisen(
    atoms_origin: Atoms,
    atoms_plus: Atoms,
    atoms_minus: Atoms,
    calculator: Calculator | None = None,
    fc2_supercell: np.ndarray | None = None,
    q_mesh: np.ndarray | None = None,
    const_volume_relax_fun: Callable | None = None,  # function to optimize the
    primitive_matrix: Any = "auto",
    log: str | Path | bool = True,
    symprec: float = 1e-5,
    displacement_distance: float = 0.03,
    relax_kwargs: dict[str, Any] | None = None,
    save: bool = False,
    **kwargs: Any,
):

    if fc2_supercell is not None:
        _fc2_supercell = fc2_supercell
    else:
        if "fc2_supercell" in atoms_origin.info.keys():
            _fc2_supercell = atoms_origin.info["fc2_supercell"]
        else:
            raise ValueError(
                f'{atoms_origin.get_chemical_formula(mode="metal")=} "fc2_supercell" was not found in atoms.info and was not provided as an argument when calculating force sets.'
            )

    if calculator is not None:
        _calculator = calculator
    else:
        if getattr(atoms_origin, "calc", None) is not None:
            _calculator = atoms_origin.calc
        else:
            raise ValueError(
                f'{atoms_origin.get_chemical_formula(mode="metal")=} "calculator" was not found in atoms object and was not provided as an argument when calculating force sets.'
            )

    if q_mesh is not None:
        _q_mesh = q_mesh
    else:
        if "q_mesh" in atoms_origin.info.keys():
            _q_mesh = atoms_origin.info["q_mesh"]
        else:
            raise ValueError(
                f'{atoms_origin.get_chemical_formula(mode="metal")=} "q_mesh" was not found in atoms.info and was not provided as an argument when calculating force sets.'
            )

    if relax_kwargs is None:
        _relax_kwargs = {
            "constant_volume": True,
            "fmax": 1e-4,
            "allow_tilt": True,
            "log": log,
        }
    else:
        _relax_kwargs = relax_kwargs

    if const_volume_relax_fun is None:
        _const_volume_relax_fun = partial(simple_relax, **_relax_kwargs)
    else:
        _const_volume_relax_fun = const_volume_relax_fun

    atoms_dict = {
        "atoms_plus": atoms_plus.copy(),
        "atoms_minus": atoms_minus.copy(),
    }

    origin = init_phonopy(
        atoms_origin,
        fc2_supercell=_fc2_supercell,
        log=log,
        symprec=symprec,
        displacement_distance=displacement_distance,
        primitive_matrix=primitive_matrix,
        **kwargs,
    )

    plus = init_phonopy(
        atoms_plus,
        fc2_supercell=_fc2_supercell,
        log=log,
        symprec=symprec,
        displacement_distance=displacement_distance,
        primitive_matrix=primitive_matrix,
        **kwargs,
    )

    minus = init_phonopy(
        atoms_minus,
        fc2_supercell=_fc2_supercell,
        log=log,
        symprec=symprec,
        displacement_distance=displacement_distance,
        primitive_matrix=primitive_matrix,
        **kwargs,
    )

    origin, _, _ = get_fc2_and_freqs(origin, _calculator, q_mesh=_q_mesh, log=log)
    plus, _, freqs_plus = get_fc2_and_freqs(plus, _calculator, q_mesh=_q_mesh, log=log)
    minus, _, freqs_minus = get_fc2_and_freqs(
        minus, _calculator, q_mesh=_q_mesh, log=log
    )

    grun_dict = gruneisen_scaled(origin, plus, minus, _q_mesh)

    grun_dict.update(atoms_dict)

    grun_dict["frequencies_plus"] = freqs_plus
    grun_dict["frequencies_minus"] = freqs_minus

    return grun_dict


def gruneisen_load(
    origin_yaml,
    plus_yaml,
    minus_yaml,
    q_mesh,
    nac_method="gonze",
    fc_calculator_origin=None,
):
    origin = load_phonopy(origin_yaml, fc_calculator=fc_calculator_origin)
    plus = load_phonopy(plus_yaml)
    minus = load_phonopy(minus_yaml)

    origin.symmetrize_force_constants(level=3)
    plus.symmetrize_force_constants(level=3)
    minus.symmetrize_force_constants(level=3)

    if nac_method is None or nac_method is False:
        origin.nac_params = None
        plus.nac_params = None
        minus.nac_params = None

    if nac_method in ["wang", "Wang"] and origin.nac_params is not None:
        origin.nac_params["method"] = "wang"
        plus.nac_params["method"] = "wang"
        minus.nac_params["method"] = "wang"

    if nac_method in ["gonze", "Gonze"] and origin.nac_params is not None:
        origin.nac_params["method"] = "gonze"
        plus.nac_params["method"] = "gonze"
        minus.nac_params["method"] = "gonze"

    grun_dict = gruneisen_scaled(origin, plus, minus, q_mesh)

    origin.run_mesh(q_mesh, with_eigenvectors=False)
    plus.run_mesh(q_mesh, with_eigenvectors=False)
    minus.run_mesh(q_mesh, with_eigenvectors=False)

    plus_dict = plus.get_mesh_dict()
    minus_dict = minus.get_mesh_dict()

    grun_dict["frequencies_plus"] = plus_dict["frequencies"]
    grun_dict["frequencies_minus"] = minus_dict["frequencies"]

    return grun_dict


def calculate_mode_cv(freqs, weights, temperatures, frequency_threshold=FREQ_CUTOFF):
    mode_cv_list = []

    # print(mode_cv())

    freqs_local = deepcopy(freqs)
    weigth_sum = np.sum(weights)

    if weigth_sum is None:
        return np.nan

    for index, freq in enumerate(freqs_local):
        freq[freq < frequency_threshold] = np.nan
        if index == 0:
            freq[:3] = np.nan
        mcv = (
            mode_cv(
                np.asarray(temperatures)[:, np.newaxis], freq[np.newaxis, :] * THzToEv
            )
            * weights[index]
            * EvTokJmol
            / weigth_sum
        )
        mcv = mcv * 1000  # kj/K to J/K
        mcv[np.isnan(mcv)] = 0

        mode_cv_list.append(mcv)

    return np.array(mode_cv_list).transpose((1, 0, 2))


def calc_mode_cv_grun_dict(
    grun_dict, temperatures, frequency_threshold=FREQ_CUTOFF, dict_update=False
):
    mode_cv_array = calculate_mode_cv(
        grun_dict["frequencies"],
        grun_dict["weights"],
        temperatures,
        frequency_threshold,
    )

    if dict_update:
        grun_dict.update({"heat_capacity": mode_cv_array})

    return mode_cv_array


def calc_gamma_grun_dict(
    grun_dict, temperatures, frequency_threshold=FREQ_CUTOFF, dict_update=False
):
    try:
        mode_cv_array = calc_mode_cv_grun_dict(
            grun_dict, temperatures, frequency_threshold, dict_update=dict_update
        )
    except Exception as exc:
        traceback.print_exc()
        mode_cv_array = np.nan

    if grun_dict["gruneisen"] is None:
        print(
            f'{grun_dict.get("name","")} "gruneisen" is None in grun_dict, returning NaN'
        )
        return np.nan

    try:
        mode_gamma = (
            mode_cv_array * grun_dict["gruneisen"][np.newaxis, :, :]
        ) / mode_cv_array.sum(axis=(1, 2), keepdims=True)

        # Set the acoustic modes to 0
        mode_gamma[:, 0, :3] = 0

        if dict_update:
            grun_dict.update({"mode_gamma": mode_gamma})
    except Exception as exc:
        print(f"Failed to calculate gamma {exc!r}")
        traceback.print_exc()

        if "name" in grun_dict.keys():
            print(f"{grun_dict['name']}")
        mode_gamma = np.nan

    return mode_gamma


def compute_qha(
    atoms: Atoms,
    calculator: Calculator | None = None,
    fc2_supercell: np.ndarray | None = None,
    q_mesh: np.ndarray | None = None,
    temperatures: list[int] = [300],
    const_volume_relax_fun: Callable | None = None,  # function to optimize the
    primitive_matrix: Any = "auto",
    log: str | Path | bool = True,
    symprec: float = 1e-5,
    displacement_distance: float = 0.03,
    scale_factor: float = 0.01,  # volume change by 1+scale_factor and 1-scale_factor
    scale_number: int = 5,  # number of scaled volumes (total 2*scale_number+1)
    relax_kwargs: dict[str, Any] | None = None,
    eos: str = "vinet",
    **kwargs: Any,
):

    if fc2_supercell is not None:
        _fc2_supercell = fc2_supercell
    else:
        if "fc2_supercell" in atoms.info.keys():
            _fc2_supercell = atoms.info["fc2_supercell"]
        else:
            raise ValueError(
                f'{atoms.get_chemical_formula(mode="metal")=} "fc2_supercell" was not found in atoms.info and was not provided as an argument when calculating force sets.'
            )

    if calculator is not None:
        _calculator = calculator
    else:
        if getattr(atoms, "calc", None) is not None:
            _calculator = atoms.calc
        else:
            raise ValueError(
                f'{atoms.get_chemical_formula(mode="metal")=} "calculator" was not found in atoms object and was not provided as an argument when calculating force sets.'
            )

    if q_mesh is not None:
        _q_mesh = q_mesh
    else:
        if "q_mesh" in atoms.info.keys():
            _q_mesh = atoms.info["q_mesh"]
        else:
            raise ValueError(
                f'{atoms.get_chemical_formula(mode="metal")=} "q_mesh" was not found in atoms.info and was not provided as an argument when calculating force sets.'
            )

    if relax_kwargs is None:
        _relax_kwargs = {
            "constant_volume": True,
            "fmax": 1e-4,
            "allow_tilt": True,
            "log": log,
        }
    else:
        _relax_kwargs = relax_kwargs

    if const_volume_relax_fun is None:
        _const_volume_relax_fun = partial(simple_relax, **_relax_kwargs)
    else:
        _const_volume_relax_fun = const_volume_relax_fun

    scale_factor_arr = np.arange(-scale_number, scale_number + 1) * scale_factor
    atoms_scaled_list = [
        scale_atoms_volume(atoms, sf, copy=True) for sf in scale_factor_arr
    ]

    atoms_dict = {}
    energies = []
    for index, atoms_scaled in enumerate(atoms_scaled_list):
        atoms_scaled.calc = _calculator
        energies.append(atoms_scaled.get_potential_energy())
        _const_volume_relax_fun(atoms_scaled)
        atoms_dict[f"atoms_{scale_factor_arr[index]*100}"] = atoms_scaled.copy()

    scaled_phonons = [
        init_phonopy(
            atoms_scaled,
            fc2_supercell=_fc2_supercell,
            log=log,
            symprec=symprec,
            displacement_distance=displacement_distance,
            primitive_matrix=primitive_matrix,
            **kwargs,
        )
        for atoms_scaled in atoms_scaled_list
    ]

    thermal_dict = {
        "temperatures": temperatures,
        "electronic_energies": energies,
        "free_energy": [],
        "entropy": [],
        "cv": [],
        "volumes": [],
    }

    freq_dict = {
        "frequencies": [],
    }

    for index, scaled_phonon in enumerate(scaled_phonons):
        scaled_phonon, _, freqs = get_fc2_and_freqs(
            scaled_phonon, _calculator, q_mesh=_q_mesh, log=log
        )
        scaled_phonon.run_thermal_properties(temperatures=temperatures)

        thermal_prop = scaled_phonon.get_thermal_properties_dict()

        thermal_dict["free_energy"].append(thermal_prop["free_energy"])
        thermal_dict["entropy"].append(thermal_prop["entropy"])
        thermal_dict["cv"].append(thermal_prop["heat_capacity"])
        thermal_dict["volumes"].append(scaled_phonon.unitcell.volume)

        freq_dict["frequencies"].append(freqs)

    thermal_dict["free_energy"] = np.array(thermal_dict["free_energy"]).transpose()
    thermal_dict["entropy"] = np.array(thermal_dict["entropy"]).transpose()
    thermal_dict["cv"] = np.array(thermal_dict["cv"]).transpose()

    pqha = PhonopyQHA(**thermal_dict, eos=eos)

    qha_dict = {
        "bulk_modulus": pqha.bulk_modulus,
        "thermal_expansion": pqha.thermal_expansion,
        "gruneisen": pqha.gruneisen_temperature,
        "helmholtz": pqha.helmholtz_volume,
        "volume_temperature": pqha.volume_temperature,
        "gibbs_temperature": pqha.gibbs_temperature,
        "bulk_modulus_temperature": pqha.bulk_modulus_temperature,
        "heat_capacity_P_numerical": pqha.heat_capacity_P_numerical,
        "heat_capacity_P_polyfit": pqha.heat_capacity_P_polyfit,
    }

    qha_dict = qha_dict | freq_dict | thermal_dict  # | atoms_dict

    return qha_dict


def load_qha(
    scaled_yaml: list[str] | list[Path],
    scaled_energies: list[float],
    q_mesh: np.ndarray | None = None,
    temperatures: list[int] = [300],
    # primitive_matrix: Any = "auto",
    log: str | Path | bool = True,
    eos: str = "vinet",
    **kwargs: Any,
):

    if q_mesh is not None:
        _q_mesh = q_mesh
    else:
        raise ValueError(
            f'"q_mesh" was not provided as an argument when calculating force sets.'
        )

    scaled_phonons = [
        load_phonopy(scaled_yaml[i], fc_calculator="symfc")
        for i in tqdm(range(len(scaled_yaml)), desc="Load phonopy yaml")
    ]

    thermal_dict = {
        "temperatures": temperatures,
        "electronic_energies": scaled_energies,
        "free_energy": [],
        "entropy": [],
        "cv": [],
        "volumes": [],
    }

    freq_dict = {
        "frequencies": [],
    }

    for index, scaled_phonon in enumerate(
        tqdm(scaled_phonons, desc="Run thermal properties")
    ):
        scaled_phonon.run_mesh(
            _q_mesh,
            with_eigenvectors=False,
            with_group_velocities=False,
            is_gamma_center=True,
        )
        mesh_dict = scaled_phonon.get_mesh_dict()
        freqs = mesh_dict["frequencies"]

        scaled_phonon.run_thermal_properties(temperatures=temperatures)

        thermal_prop = scaled_phonon.get_thermal_properties_dict()

        thermal_dict["free_energy"].append(thermal_prop["free_energy"])
        thermal_dict["entropy"].append(thermal_prop["entropy"])
        thermal_dict["cv"].append(thermal_prop["heat_capacity"])
        thermal_dict["volumes"].append(scaled_phonon.unitcell.volume)

        freq_dict["frequencies"].append(freqs)

    thermal_dict["free_energy"] = np.array(thermal_dict["free_energy"]).transpose()
    thermal_dict["entropy"] = np.array(thermal_dict["entropy"]).transpose()
    thermal_dict["cv"] = np.array(thermal_dict["cv"]).transpose()

    pqha = PhonopyQHA(**thermal_dict, eos=eos)

    qha_dict = {
        "bulk_modulus": pqha.bulk_modulus,
        "thermal_expansion": pqha.thermal_expansion,
        "gruneisen": pqha.gruneisen_temperature,
        "helmholtz": pqha.helmholtz_volume,
        "volume_temperature": pqha.volume_temperature,
        "gibbs_temperature": pqha.gibbs_temperature,
        "bulk_modulus_temperature": pqha.bulk_modulus_temperature,
        "heat_capacity_P_numerical": pqha.heat_capacity_P_numerical,
        "heat_capacity_P_polyfit": pqha.heat_capacity_P_polyfit,
    }

    qha_dict = qha_dict | freq_dict | thermal_dict

    return qha_dict
