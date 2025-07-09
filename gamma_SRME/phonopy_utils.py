from phonopy.structure.atoms import PhonopyAtoms
from phonopy.api_phonopy import Phonopy
from ase import Atoms
import numpy as np

def aseatoms2phonoatoms(atoms):
    phonoatoms = PhonopyAtoms(
        atoms.symbols, cell=atoms.cell, positions=atoms.positions, pbc=True
    )
    return phonoatoms

def phonoatoms2aseatoms(phonoatoms):
    atoms = Atoms(
        phonoatoms.symbols, cell=phonoatoms.cell, positions=phonoatoms.positions, pbc=True
    )
    return atoms


def aseatoms2phonopy(
    atoms, fc2_supercell, primitive_matrix=None, nac_params=None, symprec=1e-5,**kwargs
) -> Phonopy:
    unitcell = aseatoms2phonoatoms(atoms)
    return Phonopy(
        unitcell=unitcell,
        supercell_matrix=fc2_supercell,
        primitive_matrix=primitive_matrix,
        nac_params = nac_params,
        symprec=symprec,
        **kwargs,
    )

def phonopy2aseatoms(phonons: Phonopy) -> Atoms:
    phonopy_atoms = phonons.unitcell
    atoms = Atoms(
        phonopy_atoms.symbols,
        cell=phonopy_atoms.cell,
        positions=phonopy_atoms.positions,
        pbc=True,
    )

    if phonons.supercell_matrix is not None:
        atoms.info["fc2_supercell"] = phonons.supercell_matrix

    if phonons.primitive_matrix is not None:
        atoms.info["primitive_matrix"] = phonons.primitive_matrix

    if phonons.mesh_numbers is not None:
        atoms.info["q_mesh"] = phonons.mesh_numbers

    # TODO : Non-default values and BORN charges to be added

    return atoms


def get_chemical_formula(phonons: Phonopy, mode="metal", **kwargs):
    unitcell = phonons.unitcell
    atoms = Atoms(
        unitcell.symbols, cell=unitcell.cell, positions=unitcell.positions, pbc=True
    )
    return atoms.get_chemical_formula(mode=mode, **kwargs)

def load_phonopy(yaml_file, **kwargs):
    from phonopy.cui.load import load
    return load(yaml_file, **kwargs)

def phonopy_reinitiate(ph: Phonopy, include_forces = False,include_fc=False, **kwargs) -> Phonopy:
    #unitcell = aseatoms2phonoatoms()

    #print(unitcell.scaled_positions)
    input_kwargs = {
        'unitcell': ph.unitcell,
        'supercell_matrix': ph.supercell_matrix,
        'primitive_matrix': ph.primitive_matrix,
        'nac_params': ph.nac_params,
    }
    input_kwargs.update(kwargs)
    phonopy = Phonopy(**input_kwargs)

    if include_forces and ph.forces is not None:
        #phonopy.generate_displacements(distance=0.03)
        phonopy.displacements = ph.displacements
        phonopy.forces = ph.forces
        phonopy.produce_force_constants(forces=ph.forces,fc_calculator='symfc')

    if include_fc and ph.force_constants is not None:
        phonopy.force_constants = ph.force_constants

    return phonopy