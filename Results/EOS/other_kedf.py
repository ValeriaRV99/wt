#!/usr/bin/env python
# coding: utf-8
# %%
import numpy as np
from matplotlib import pyplot as plt

from dftpy.ions import Ions
from dftpy.field import DirectField
from dftpy.grid import DirectGrid
from dftpy.functional import LocalPseudo, Functional, TotalFunctional, ExternalPotential
from dftpy.formats import io
from dftpy.math_utils import ecut2nr
from dftpy.optimization import Optimization


from ase.build import bulk
from ase.io import read
from ase.lattice.spacegroup import Spacegroup
from ase.lattice.spacegroup import crystal
from ase import Atoms
from scipy.optimize import minimize


# %%
def get_energy(material, KEDF, a, b, PP_list):
    import copy
    PP_list=PP_list

    if KEDF=='MGP':
        KE = Functional(type='KEDF',name=KEDF, lumpfactor=[a,b])
    else:
        KE = Functional(type='KEDF',name=KEDF)
    if KEDF=='TFvW':
        KE.options.update({'y':0.2})
    XC = Functional(type='XC',name='LDA')
    HARTREE = Functional(type='HARTREE')
    ions = Ions.from_ase(material)
    cell = ions.get_cell()
    nr = ecut2nr(ecut=50, lattice=ions.cell)
    grid = DirectGrid(lattice=ions.cell, nr=nr)
    PSEUDO = LocalPseudo(grid = grid, ions=ions, PP_list=PP_list, rcut=20)
    rho_ini = DirectField(grid=grid)
    rho_ini[:] = ions.get_ncharges()/ions.cell.volume
    
    realevaluator = TotalFunctional(KE=KE, XC=XC, HARTREE=HARTREE, PSEUDO=PSEUDO)
    optimization_options = {'econv' : 1e-5*ions.nat}
    optreal = Optimization(EnergyEvaluator=realevaluator, optimization_options = optimization_options, 
                       optimization_method = 'TN')
    realrho = optreal.optimize_rho(guess_rho=rho_ini)
    realenergy = realevaluator.Energy(rho=realrho, ions=ions)
    ke_en = KE(realrho).energy
    return np.asarray(realenergy), np.asarray(ke_en), realrho

