# +
import numpy as np
from matplotlib import pyplot as plt

import dftpy
from dftpy.ions import Ions
from dftpy.field import DirectField
from dftpy.grid import DirectGrid
from dftpy.functional import LocalPseudo, Functional, TotalFunctional, ExternalPotential
from dftpy.formats import io
from dftpy.math_utils import ecut2nr
from dftpy.time_data import TimeData
from dftpy.optimization import Optimization
from dftpy.mpi import sprint
from dftpy.formats import io

from qepy.calculator import QEpyCalculator 
from qepy.io import QEInput
from qepy.driver import Driver

from ase.build import bulk
from ase.io.trajectory import Trajectory
from ase.io import read
from ase.units import kJ
from ase.eos import EquationOfState

import pickle
# -

from scipy.optimize import minimize
try:
    from mpi4py import MPI
    comm=MPI.COMM_WORLD
except:
    comm=None
path_pp='/home/valeria/Desktop/programs/dftpy/examples/ofpp/EAC/upf/blps/'
file1='si.lda.upf'
PP_list = {'Si': path_pp+file1}

# +
qe_options = {
        '&control': {
            'calculation': "'scf'",
            'prefix': "'si'",
            'pseudo_dir': "'/home/valeria/Desktop/programs/dftpy/examples/ofpp/EAC/upf/blps/'",
            'restart_mode': "'from_scratch'"},
        '&system': {
            'ibrav' : 1,
            'nat': 8,
            'degauss': 0.005,
            'ecutwfc': 60,
            'occupations': "'smearing'"},
        'atomic_species': ['Si  28.08 si.lda.upf'],
        'cell_parameters angstrom': ['5.43    0.0000000000000000    0.0000000000000003',
                                     '0.0000000000000009    5.43    0.0000000000000003',
                                     '0.0000000000000000    0.0000000000000000    5.43'],
         'k_points automatic': ['20 20 20 1 1 1']}

# KS DFT
l = np.linspace(0.8, 1.4, 30)
# RHO = []
ks_ke = []
for i in np.arange(0, len(l), 1):
        n = int(i)
        X = l[n]
        si = bulk('Si', 'diamond', a=5.43, cubic=True)
        cell = si.get_cell()
        si.set_cell(cell * X, scale_atoms=True)
        qe_options = QEInput.update_atoms(si, qe_options = qe_options,  extrapolation=False)
        QEInput().write_qe_input("/home/valeria/Documents/DFTPY/Fitting_densities/Si8-CD/DATA2/Si"+str(n)+".in", qe_options=qe_options)
        driver = Driver('/home/valeria/Documents/DFTPY/Fitting_densities/Si8-CD/DATA2/Si'+str(n)+'.in', comm=comm, logfile='/home/valeria/Documents/DFTPY/Fitting_densities/Si8-CD/DATA2/Si'+str(n)+'.out')
        driver.scf()
        driver.calc_energy()
        D = driver.get_output()
        k = [match for match in D if "Kinetic energy" in match]
        KS_KE = str(k).split()[6]
        rho = driver.data2field(driver.get_density())
        ions = driver.get_dftpy_ions()
        rho.write('/home/valeria/Documents/DFTPY/Fitting_densities/Si8-CD/DATA2/rho0'+str(i)+'.xsf', ions=ions)
        driver.stop()
#         RHO.append(rho)
        ks_ke.append(KS_KE)
# -

KS_KE = []
for i in np.arange(0,len(l),1):
    ke = float(ks_ke[int(i)])
    KS_KE.append(ke)

np.save("/home/valeria/Documents/DFTPY/Fitting_densities/Si8-CD/DATA2/ks_ke.npy", np.asarray(KS_KE))

var = np.asarray(KS_KE)
with open('ks_ke.pkl', 'wb') as f:
    pickle.dump(var,f)

with open('ks_ke.pkl','rb') as f:
    ks_ke = pickle.load(f)


class setting:
    def sett(ions, rho, scaling):
        ions = ions
        rho = rho
        scaling = scaling
        cell = ions.get_cell()
        ions.set_cell(cell * scaling, scale_atoms=True) 
        nr = ecut2nr(ecut=147, lattice=ions.cell)
        grid = DirectGrid(lattice=ions.cell, nr=[len(rho),len(rho),len(rho)])
        PSEUDO = LocalPseudo(grid = grid, ions=ions, PP_list=PP_list)  
        XC = Functional(type='XC',name='LDA')
        HARTREE = Functional(type='HARTREE')
        return PSEUDO, HARTREE, XC


def min_energy(x0, *args):
        KS_KE, KS_PE, rho = args
        KE = Functional(type='KEDF',name='WT', rho0=x0) 
        OF_KE = KE(rho).energy
#         pot = KE(rho, calctype = {'V'}).potential
#         OF_PE =  pot - pot.min()
#         pot = rho*(KS_PE - OF_PE)
        diff = 0.1*(float(OF_KE)-float(KS_KE)*(1/2))**2 #+ 0.1*(pot.integral())**2
#         print(x0)
        return diff


ions = Ions.from_ase(bulk('Si', 'diamond', a=5.43, cubic=True))
# i = 2
l = np.linspace(0.8, 1.1, 10)
delta_E = []
Rho0 = []
for n in np.arange(0,len(l),1):
    i = int(n)
    rho = io.read_density('rho0'+str(i)+'.xsf')
#     rho = RHO[int(i)]
    scaling = l[i]

    PSEUDO, HARTREE, XC = setting.sett(ions,rho,scaling)
    H_PE = HARTREE(rho, calctype = {'V'}).potential
    XC_PE = XC(rho, calctype = {'V'}).potential
    PP_PE = PSEUDO(rho, calctype = {'V'}).potential
    
    KS_KE = ks_ke[int(i)]
    Ef_PE = H_PE + XC_PE + PP_PE
    KS_PE = Ef_PE - Ef_PE.min()
    
    bnds = ((0, None))
    minn = minimize(min_energy, 0.02, args = (KS_KE, KS_PE, rho), method='Nelder-Mead',
               options={'xtol': 1e-4,'disp': True})
    delta_E.append(minn.fun)
    Rho0.append(minn.x)
    print(minn.x)

plt.plot(Rho0,delta_E)

# +
rh0 = []
for i in np.arange(0,len(Rho0),1):
    r = float(Rho0[int(i)])
    rh0.append(r)
    
rho0 = np.asarray(rh0)

# +


ions = Ions.from_ase(bulk('Si', 'diamond', a=5.43, cubic=True))
# rho0 = 24 /ions.cell.volume
l = np.linspace(0.8, 1.1, 10)

XC = Functional(type='XC',name='LDA')
HARTREE = Functional(type='HARTREE')

cell = ions.get_cell()
traj = Trajectory('ions.traj', 'w')
ENERGY2 = []
VOLUME2 = []
KIN = []
for x in np.arange(0, 9, 1):
        i = int(x)
        
        ions.set_cell(cell * l[i], scale_atoms=True)
        rho = io.read_density('rho0'+str(i)+'.xsf')

        KE = Functional(type='KEDF',name='WT', rho0=rho0[i])  
        nr = ecut2nr(ecut=25, lattice=ions.cell)
        grid = DirectGrid(lattice=ions.cell, nr=[len(rho),len(rho),len(rho)])

        PSEUDO = LocalPseudo(grid = grid, ions=ions, PP_list=PP_list)
        evaluator = TotalFunctional(KE=KE, XC=XC, HARTREE=HARTREE, PSEUDO=PSEUDO)

#         rho_ini = DirectField(grid=grid)
#         rho_ini[:] = ions.get_ncharges()/ions.cell.volume

#         optimization_options = {'econv' : 1e-5*ions.nat}
#         opt = Optimization(EnergyEvaluator=evaluator, optimization_options = optimization_options, 
#                            optimization_method = 'TN')
#         rho = opt.optimize_rho(guess_rho=rho_ini)
        energy2 = evaluator.Energy(rho=rho, ions=ions)
        vol2=ions.get_volume()
        ENERGY2.append(energy2)
        VOLUME2.append(vol2)
# -

rho0_exact = 32.0 / np.asarray(VOLUME2)
plt.plot(np.asarray(VOLUME2)*0.529177**3/8, rho0[:-1],'b<--',label='cWT')
plt.plot(np.asarray(VOLUME2)*0.529177**3/8, rho0_exact,'r<--',label='exact') # b<--


rho0_exact = 32.0 / np.asarray(VOLUME2)
plt.plot(np.asarray(VOLUME2)*0.529177**3/8, rho0[:-1],'b<--',label='cWT')
plt.plot(np.asarray(VOLUME2)*0.529177**3/8, rho0_exact,'r<--',label='exact') # b<--




# +
# Efective potential
Ef_PE = H_PE + XC_PE + PP_PE
KS_PE = Ef_PE - Ef_PE.min()
# Evaluation of the functional with the density obtained from KS calculation

rho = RHO[int(i)]
H_PE = HARTREE(rho, calctype = {'V'}).potential
XC_PE = XC(rho, calctype = {'V'}).potential
PP_PE = PSEUDO(rho, calctype = {'V'}).potential
# Kinetic and potential energy of the WT nonlocal KEDF
#         KE.options.update({'x, y':0})

OF_PE = KE(rho, calctype = {'V'}).potential - KE(rho, calctype = {'V'}).potential.min()


# -

def min_energy(x0):
        DIFF = []
        l = np.linspace(0.8, 1.1, 10)
        # for i in np.arange(0,len(l), 1):
        n = int(0)
        X = l[n]
        # OF DFT
        ions = Ions.from_ase(bulk('Si', 'diamond', a=5.43, cubic=True))
        cell = ions.get_cell()
        ions.set_cell(cell * X, scale_atoms=True)
        XC = Functional(type='XC',name='LDA')
        HARTREE = Functional(type='HARTREE')
        KE = Functional(type='KEDF',name='WT', rho0=x0)  
        nr = ecut2nr(ecut=470, lattice=ions.cell)
        grid = DirectGrid(lattice=ions.cell, nr=nr)

        PSEUDO = LocalPseudo(grid = grid, ions=ions, PP_list=PP_list)
        evaluator = TotalFunctional(KE=KE, XC=XC, HARTREE=HARTREE, PSEUDO=PSEUDO)
        rho_ini = DirectField(grid=grid)
        rho_ini[:] = ions.get_ncharges()/ions.cell.volume

        optimization_options = {'econv' : 1e-5*ions.nat}
        opt = Optimization(EnergyEvaluator=evaluator, optimization_options = optimization_options, 
                           optimization_method = 'TN')
        rho = opt.optimize_rho(guess_rho=rho_ini)
        OF_KE = KE(rho).energy
        OF_PE = KE(rho).potential - KE(rho).potential.mean()
#         ene2 = evaluator.Energy(rho=rho, ions=ions)
#         vol2 = ions.get_volume()

        qe_options = {
                '&control': {
                    'calculation': "'scf'",
                    'prefix': "'si'",
                    'pseudo_dir': "'/home/valeria/Desktop/programs/dftpy/examples/DATA/ofpp/'",
                    'restart_mode': "'from_scratch'"},
                '&system': {
                    'ibrav' : 1,
                    'nat': 8,
                    'degauss': 0.005,
                    'ecutwfc': 60,
                    'occupations': "'smearing'"},
                'atomic_species': ['Si  28.08 si.lda.upf'],
                'cell_parameters angstrom': ['5.43    0.0000000000000000    0.0000000000000003',
                                             '0.0000000000000009    5.43    0.0000000000000003',
                                             '0.0000000000000000    0.0000000000000000    5.43'],
                 'k_points automatic': ['2 2 2 1 1 1']}

        # KS DFT
        si = bulk('Si', 'diamond', a=5.43, cubic=True)
        si.set_cell(cell * X, scale_atoms=True)
        qe_options = QEInput.update_atoms(si, qe_options = qe_options,  extrapolation=False)
        QEInput().write_qe_input("Si"+str(n)+".in", qe_options=qe_options)
        driver = Driver('Si'+str(n)+'.in', comm=comm, logfile='Si'+str(n)+'.out')
        driver.scf()
        driver.calc_energy()

        D = driver.get_output()
        k = [match for match in D if "Kinetic energy" in match]
        KS_KE = str(k).split()[6]
        v_men = driver.get_effective_potential().mean()
        KS_PE = driver.data2field(driver.get_effective_potential() - v_men)

        r = driver.data2field(driver.get_density())
        dif = r*(KS_PE  - OF_PE)**2
        diff = (float(KS_KE)*2-float(OF_KE))**2 + dif.integral()
        return diff
minimize(min_energy, 0.02, method='nelder-mead',
               options={'xatol': 1e-4,'disp': True})



