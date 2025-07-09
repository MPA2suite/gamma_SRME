from ase import Atoms
from phono3py.api_phono3py import Phono3py
from gamma_SRME.phonopy_utils import aseatoms2phonoatoms

from phonopy import Phonopy


def phono3py2aseatoms(ph3: Phono3py) -> Atoms:
    phonopy_atoms = ph3.unitcell
    atoms = Atoms(
        phonopy_atoms.symbols,
        cell=phonopy_atoms.cell,
        positions=phonopy_atoms.positions,
        pbc=True,
    )

    if ph3.supercell_matrix is not None:
        atoms.info["fc3_supercell"] = ph3.supercell_matrix

    if ph3.phonon_supercell_matrix is not None:
        atoms.info["fc2_supercell"] = ph3.phonon_supercell_matrix

    if ph3.primitive_matrix is not None:
        atoms.info["primitive_matrix"] = ph3.primitive_matrix

    if ph3.mesh_numbers is not None:
        atoms.info["q_mesh"] = ph3.mesh_numbers

    # TODO : Non-default values and BORN charges to be added

    return atoms


def get_chemical_formula(ph3: Phono3py, mode="metal", **kwargs):
    unitcell = ph3.unitcell
    atoms = Atoms(
        unitcell.symbols, cell=unitcell.cell, positions=unitcell.positions, pbc=True
    )
    return atoms.get_chemical_formula(mode=mode, **kwargs)


def phono3py2phonopy(ph3: Phono3py, include_forces=False) -> Phonopy:
    phonopy_atoms = aseatoms2phonoatoms(ph3.unitcell)
    phonopy = Phonopy(
        phonopy_atoms,
        supercell_matrix=ph3.phonon_supercell_matrix,
        primitive_matrix=ph3.primitive_matrix,
        nac_params=ph3.nac_params,
    )
    if include_forces and ph3.phonon_forces is not None:
        # phonopy.generate_displacements(distance=0.03)
        phonopy.displacements = ph3.phonon_displacements
        phonopy.forces = ph3.phonon_forces
        phonopy.produce_force_constants(forces=ph3.phonon_forces, fc_calculator="symfc")

    # if ph3.fc2 is not None:
    #    phonopy.set_force_constants(ph3.fc2)
    # else:
    #    if ph3.phonon_forces is not None:
    #        ph3.produce_fc2(symmetrize_fc2=True)
    #        phonopy.set_force_constants(ph3.fc2)

    return phonopy
