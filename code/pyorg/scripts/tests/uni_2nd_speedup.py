"""
    Measures the speed-up for computing univarite 2nd oder models and simulate CSRV instances

"""

################# Package import

import os
import sys
import math
import time
import numpy as np
import multiprocessing as mp
from scipy.optimize import curve_fit
from pyorg.surf.model import ModelCSRV, gen_tlist
from pyorg.surf.utils import disperse_io
from matplotlib import pyplot as plt, rcParams
plt.switch_backend('agg')

###### Global variables

__author__ = 'Antonio Martinez-Sanchez'


########################################################################################
# PARAMETERS
########################################################################################

try:
    root_path = sys.argv[1]
except IndexError:
    root_path = os.path.split(os.path.abspath(__file__))[0] + '/../../../tests'
out_dir = root_path + '/results'

# Synthetic data generation variables
sdat_surf = root_path + '/../pyorg/surf/test/in/sph_rad_5_surf.vtp'
sdat_tomo_shape = (500, 500, 100)
sdat_n_tomos = 5
sdat_n_sims = None # 20
sdat_n_part_tomo = 600 # 200

# Analysis variables
ana_npr_rg = [1, 2, 4, 8, 16, 24, 32, 36] # [1, 2, 4, 16] # It must start with 1
ana_rad_rg = np.arange(4, 250, 1) # np.arange(4, 180, 3)
ana_shell_thick = None
ana_fmm = False # True

# Plotting settings

rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14
rcParams['patch.linewidth'] = 2

########################################################################################
# HELPING FUNCTIONS
########################################################################################

def gen_rect_voi_array(shape):
    """
    Generates a rectangular array VOI
    :param shape: 3-tuple with the length of the three rectangle sides
    :return: a binary ndarray object
    """
    seg = np.zeros(shape=np.asarray(shape) + 1, dtype=bool)
    seg[1:shape[0], 1:shape[1], 1:shape[2]] = True
    return seg

def amdahls(x, p):
    """
    Computes Amdal's Law speed-up
    :param x: is the speedup of the part of the task that benefits from improved system resources
    :param p: is the proportion of execution time that the part benefiting from improved resources originally occupied
    :return: the computed speed-up
    """
    return 1. / (1. - p + p/x)

########################################################################################
# MAIN ROUTINE
########################################################################################

########## Print initial message

print('Test for measuring univariate 2nd order and simulations computation speed-up.')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('\tSynthetic data generations settings: ')
print('\t\t-Particle surface path: ' + str(sdat_surf))
print('\t\t-Tomogram shape: ' + str(sdat_tomo_shape))
print('\t\t-Number of tomograms: ' + str(sdat_n_tomos))
if sdat_n_sims is None:
    print('\t\t-Number of simulations per tomogram are set to the number of processess.')
else:
    print('\t\t-Number of simulations per tomogram: ' + str(sdat_n_sims))
print('\t\t-Number of particles per tomogram: ' + str(sdat_n_part_tomo))
print('\tAnalysis settings: ')
print('\t\t-Number of parallel processes to check: ' + str(ana_npr_rg))
print('\t\t-Scale samplings array: ' + str(ana_rad_rg))
if ana_shell_thick is None:
    print('\t\t-Functions L is computed.')
else:
    print('\t\t-Function O is computed with shell thickness: ' + str(ana_shell_thick))
if ana_fmm:
    print('\t\t-Geodesic metric.')
else:
    print('\t\t-Euclidean metric.')
print('')

######### Main process

print('Main Routine: ')

print('\t-Initialization...')
voi = gen_rect_voi_array(sdat_tomo_shape)
part = disperse_io.load_poly(sdat_surf)
model_csrv = ModelCSRV()
ltomos_csrv = gen_tlist(sdat_n_tomos, sdat_n_part_tomo, model_csrv, voi, sdat_surf, mode_emb='center',
                        npr=max(ana_rad_rg))
cu_i = 1. / float(sdat_n_tomos * sdat_n_part_tomo)
cpus = mp.cpu_count()
print('\t\t+CPUs found: ' + str(cpus))

# Loop for the of processors
print('\t-Measurements loops: ')
comp_times = np.zeros(shape=len(ana_npr_rg), dtype=np.float32)
sim_times = np.zeros(shape=len(ana_npr_rg), dtype=np.float32)
for i, npr in enumerate(ana_npr_rg):

    print('\t\t+Number of processes: ' + str(npr))

    # Computations loop
    comp_time, sim_time = 0, 0
    for tkey in ltomos_csrv.get_tomo_fname_list():
        hold_time = time.time()
        hold_tomo = ltomos_csrv.get_tomo_by_key(tkey)
        hold_tomo.compute_uni_2nd_order(ana_rad_rg, thick=None, border=True, conv_iter=None, max_iter=None, fmm=ana_fmm,
                                        npr=npr)
        comp_time += (time.time() - hold_time)
        if sdat_n_sims is None:
            hold_n_sims = npr
        else:
            hold_n_sims = sdat_n_sims
        cu_sim_i = 1. / float(sdat_n_tomos * sdat_n_part_tomo * hold_n_sims)
        hold_time = time.time()
        hold_sim = hold_tomo.simulate_uni_2nd_order(hold_n_sims, model_csrv, part, 'center', ana_rad_rg, thick=None,
                                                    border=True, conv_iter=None, max_iter=None, fmm=ana_fmm,
                                                    npr=npr)
        sim_time += (time.time() - hold_time)
    comp_times[i], sim_times[i] = comp_time * cu_i, sim_time * cu_sim_i
    print('\t\t\t*Computation time per c.u.: ' + str(comp_times[i]) + ' [secs]')
    print('\t\t\t*Computation time per c.u. and null-model simulations time: ' + str(sim_times[i]) + ' [secs]')

print('\tPlotting: ')

# plt.figure()
# plt.xlabel('# processes')
# plt.ylabel('Time/c.u. [s]')
# plt.plot(ana_npr_rg, comp_times, linewidth=2.0, linestyle='-', color='b', label='C')
# plt.plot(ana_npr_rg, sim_times, linewidth=2.0, linestyle='-', color='g', label='C+S')
# plt.tight_layout()
# plt.legend(loc=0)
# if out_dir is not None:
#     out_fig_times = out_dir + '/times.png'
#     print '\t\t-Storing the time figure in: ' + out_fig_times
#     plt.savefig(out_fig_times)
# else:
#     plt.show(block=True)
# plt.close()

# Speed up fitting:
processes = np.asarray(ana_npr_rg, dtype=float)
processes_ex = np.logspace(0, np.log2(cpus), num=50, base=2)
sup_comp = comp_times[0] / comp_times
sup_sim = sim_times[0] / sim_times
popt_comp, pcov_comp = curve_fit(amdahls, processes, sup_comp)
popt_sim, pcov_sim = curve_fit(amdahls, processes, sup_sim)
sup_comp_f = amdahls(processes_ex, popt_comp)
sup_sim_f = amdahls(processes_ex, popt_sim)

fig, ax1 = plt.subplots()
ax1.set_xlabel('# processes')
ax1.set_ylabel('Time/c.u. [s]')
# ax1.set_xlim((1, processes_ex.max()))
ax1.plot(ana_npr_rg, comp_times, linewidth=2.0, linestyle='--', color='b', label='C Time')
ax1.plot(ana_npr_rg, sim_times, linewidth=2.0, linestyle='--', color='g', label='C&S Time')
ax2 = ax1.twinx()
ax2.set_ylabel('Speedup')
# plt.plot(processes_ex, processes_ex, linewidth=1.0, linestyle='--', color='k', label='IDEAL')
# plt.plot((16, 16), (0, 16), linewidth=1.0, linestyle='-.', color='k')
# plt.plot((36, 36), (0, 36), linewidth=1.0, linestyle='-.', color='k')
ax2.plot(processes, sup_comp, linewidth=4.0, linestyle='-', marker='*', color='b', label='C Speedup')
# ax2.plot(processes_ex, sup_comp_f, linewidth=2.0, linestyle='-', color='b', label='C Speedup')
ax2.plot(processes, sup_sim, linewidth=4.0, linestyle='-', marker='s', color='g', label='C&S Speedup')
# ax2.plot(processes_ex, sup_sim_f, linewidth=2.0, linestyle='-', color='g', label='C&S Speedup')
# ax2.set_ylim((1, processes_ex.max()))
fig.tight_layout()
# fig.legend(loc=9)
if out_dir is not None:
    out_fig_speed = out_dir + '/speed_up_time.png'
    print('\t\t-Storing the time figure in: ' + out_fig_speed)
    plt.savefig(out_fig_speed)
else:
    plt.show(block=True)
plt.close()

print('Terminated. (' + time.strftime("%c") + ')')
