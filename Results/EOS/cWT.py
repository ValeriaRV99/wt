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
from ase.io.trajectory import Trajectory
from ase.lattice.spacegroup import Spacegroup
from ase.lattice.spacegroup import crystal

from sklearn.model_selection import train_test_split 
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process.kernels import Exponentiation, RationalQuadratic, Matern
from sklearn.gaussian_process import GaussianProcessRegressor

from dscribe.descriptors import CoulombMatrix, SineMatrix, EwaldSumMatrix, SOAP
from ase import Atoms


def get_energy(material, path_rho, PP_list, rho0, i,r):
    import copy
    ml_material = copy.deepcopy(material)
    of_material = copy.deepcopy(material)
#     phase = phase
    print('init_material', of_material.get_cell())

    PP_list = PP_list
    XC = Functional(type='XC',name='LDA')
    HARTREE = Functional(type='HARTREE')
    pred_rho00 = np.asarray(rho0) 
    pred_energy = []
    Volume = []
    ke_en = []
    print(of_material)
    for j, d in enumerate(r):
        pred_KE = Functional(type='KEDF',name='WT', rho0=pred_rho00[i][j])
        ions = Ions.from_ase(of_material)
        cell = ions.get_cell()
        ions.set_cell(cell * d, scale_atoms=True)
        if path_rho is None:
            nr = ecut2nr(ecut=25, lattice=ions.cell)
            grid = DirectGrid(lattice=ions.cell, nr=nr)
            PSEUDO = LocalPseudo(grid = grid, ions=ions, PP_list=PP_list, rcut=20)

            rho_ini = DirectField(grid=grid)
            rho_ini[:] = ions.get_ncharges()/ions.cell.volume
            predevaluator = TotalFunctional(KE=pred_KE, XC=XC, HARTREE=HARTREE, PSEUDO=PSEUDO)
            optimization_options = {'econv' : 1e-5*ions.nat}
            optpred = Optimization(EnergyEvaluator=predevaluator, optimization_options = optimization_options, 
                               optimization_method = 'TN')
            predrho = optpred.optimize_rho(guess_rho=rho_ini)
        else:
            rho = io.read_density(path_rho+str(j)+'.xsf')
            nr = rho.grid.nr
            grid = DirectGrid(lattice=ions.cell, nr=nr)
            PSEUDO = LocalPseudo(grid = grid, ions=ions, PP_list=PP_list, rcut=20)
#             rho_ini = DirectField(grid=grid)
#             rho_ini[:] = ions.get_ncharges()/ions.cell.volume
            predevaluator = TotalFunctional(KE=pred_KE, XC=XC, HARTREE=HARTREE, PSEUDO=PSEUDO)
            predrho = rho
        predenergy = predevaluator.Energy(rho=predrho, ions=ions)
        vol = ions.get_volume()
        ke = pred_KE(predrho).energy
#         predrho.write('/Users/valeria/Documents/aiWT/Model_ML_function/Densitie_DEN/rho_'+str(i)+'_'+str(j)+'.xsf', ions=ions)

#         predrho.write('/home/valeria/Documents/DFTPY/cWT-KEDF/Phases/Model_ML_function/Results_test/pred_rWT_rho0_'+str(phase)+'_'+str(i)+'.xsf', ions=ions)
#         predrho.write('/home/valeria/Documents/DFTPY/cWT-KEDF/Phases/Model_ML_function/Results_test/pred_rWT_rho0_'+str(phase)+'_'+str(i)+'.cube', ions=ions)
        pred_energy.append(predenergy)        
        Volume.append(vol)
        ke_en.append(ke)
    return np.asarray(pred_energy), np.asarray(ke_en), np.asarray(Volume)

