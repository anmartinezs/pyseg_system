
import vtk
import math
import time
import numpy as np
from unittest import TestCase
from ..surface import ListShellNhood, ListSphereNhood
from ..model import ModelCSRV
from ..utils import disperse_io, iso_surface, poly_decimate
from matplotlib import pyplot as plt, rcParams

# Global variables
RAD_RG = np.arange(2, 80, 3)
SHELL_THICKS = (3, 6, 9, 12)
THICK_COLORS = ('b', 'r', 'g', 'c')
CONV_ITER = 1000 # 100 # 1000
MAX_ITER = 100000
N_SIMS = 100 # 1000
PERCENT = 5 # %
BAR_WIDTH = .4
OUT_DIR = './surf/test/out/nhood_vol'

rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14
rcParams['patch.linewidth'] = 2

class TestListNhoodVol(TestCase):

    # Check precision for Nhoods using VOIs as 3D surfaces and Monte Carlo method
    def test_volume_MCS(self):

        # return

        # VOI Creation
        # cuber = vtk.vtkCubeSource()
        # off = RAD_RG.max()
        # cuber.SetBounds(-off - SHELL_THICKS[3], off + SHELL_THICKS[3], -off - SHELL_THICKS[3], off + SHELL_THICKS[3],
        #                 0, off + SHELL_THICKS[3])
        # cuber.Update()
        # voi = cuber.GetOutput()
        # orienter = vtk.vtkPolyDataNormals()
        # orienter.SetInputData(voi)
        # orienter.AutoOrientNormalsOn()
        # orienter.Update()
        # voi = orienter.GetOutput()
        # disperse_io.save_vtp(voi, OUT_DIR + '/cube_voi.vtp')
        # voi_center = (0, 0, 0)
        r_m_h = 5
        rad_max = RAD_RG.max()
        max_size = int(math.ceil(2 * rad_max + SHELL_THICKS[3]))
        seg = np.ones(shape=(max_size, max_size, max_size), dtype=bool)
        voi_center = np.asarray((.5 * seg.shape[0], .5 * seg.shape[1], .5 * seg.shape[2]))
        X = np.meshgrid(np.arange(seg.shape[0]), np.arange(seg.shape[1]), np.arange(seg.shape[2]))[0]
        seg[X > voi_center[0] + r_m_h] = False
        voi = iso_surface(seg, .5, closed=True, normals='outwards')
        voi = poly_decimate(voi, .9)

        # Computations loop
        sph_errs = np.zeros(shape=(len(RAD_RG), N_SIMS), dtype=np.float32)
        she_errs = np.zeros(shape=(len(RAD_RG), N_SIMS), dtype=np.float32)
        sph_times = np.zeros(shape=N_SIMS, dtype=np.float32)
        she_times = np.zeros(shape=N_SIMS, dtype=np.float32)
        for i in range(N_SIMS):
            # Computations
            hold_time = time.time()
            spheres = ListSphereNhood(center=voi_center, radius_rg=RAD_RG, voi=voi, conv_iter=CONV_ITER,
                                      max_iter=MAX_ITER)
            sph_vol = np.asarray(spheres.get_volumes())
            sph_times[i] = time.time() - hold_time
            hold_time = time.time()
            shells = ListShellNhood(center=voi_center, radius_rg=RAD_RG, voi=voi, conv_iter=CONV_ITER,
                                    max_iter=MAX_ITER, thick=SHELL_THICKS[0])
            she_vol = np.asarray(shells.get_volumes())
            she_times[i] = time.time() - hold_time

            # Computing the errors
            h_arr = RAD_RG - r_m_h
            sph_volr = np.pi * ((4. / 3.)*(RAD_RG**3) - RAD_RG*(h_arr**2) + (h_arr**3)*(1./3.))
            rad_rg_1, rad_rg_2 = RAD_RG - .5*SHELL_THICKS[0], RAD_RG + .5*SHELL_THICKS[0]
            h_arr_1, h_arr_2 = rad_rg_1 - r_m_h, rad_rg_2 - r_m_h
            she_volr = np.pi * ((4. / 3.)*(rad_rg_2**3) - rad_rg_2*(h_arr_2**2) + (h_arr_2**3)*(1./3.)
                                - (4. / 3.)*(rad_rg_1**3) + rad_rg_1*(h_arr_1**2) - (h_arr_1**3)*(1./3.))
            # sph_volr =  (4. / 3.) * np.pi * (RAD_RG ** 3)
            # she_volr =  (4. / 3.) * np.pi * \
            #           ((RAD_RG + 0.5 * SHELL_THICKS[
            # 0]) ** 3 - (RAD_RG - 0.5 * SHELL_THICKS[0]) ** 3)
            sph_errs[:, i] = 100. * (sph_vol - sph_volr) / sph_volr
            she_errs[:, i] = 100. * (she_vol - she_volr) / she_volr

        # Plotting error
        plt.figure()
        # plt.title('Averaged fractional error for volume estimation (MCS-SPHERE)')
        plt.xlabel('Scale')
        plt.ylabel('E [%]')
        ic_low = np.percentile(sph_errs, PERCENT, axis=1, interpolation='linear')
        ic_med = np.percentile(sph_errs, 50, axis=1, interpolation='linear')
        ic_high = np.percentile(sph_errs, 100 - PERCENT, axis=1, interpolation='linear')
        plt.fill_between(RAD_RG, ic_low, ic_high, alpha=.5, color='gray', edgecolor='w')
        plt.plot(RAD_RG, np.zeros(shape=len(RAD_RG)), 'k--', linewidth=1)
        plt.plot(RAD_RG, ic_low, linewidth=2, color='k', linestyle='--')
        plt.plot(RAD_RG, ic_med, linewidth=2, color='k')
        plt.plot(RAD_RG, ic_high, linewidth=2, color='k', linestyle='--')
        plt.xlim(r_m_h, RAD_RG.max())
        # plt.ylim(-5, 5)
        plt.ylim(-25, 25)
        plt.tight_layout()
        # plt.show(block=True)
        plt.savefig(OUT_DIR + '/mcs_sph_err_' + str(CONV_ITER) + '.png')
        plt.close()
        plt.figure()
        # plt.title('Averaged fractional error for volume estimation (MCS-SHELL)')
        plt.xlabel('Scale')
        plt.ylabel('E [%]')
        ic_low = np.percentile(she_errs, PERCENT, axis=1, interpolation='linear')
        ic_med = np.percentile(she_errs, 50, axis=1, interpolation='linear')
        ic_high = np.percentile(she_errs, 100 - PERCENT, axis=1, interpolation='linear')
        plt.fill_between(RAD_RG, ic_low, ic_high, alpha=.5, color='gray', edgecolor='w')
        plt.plot(RAD_RG, np.zeros(shape=len(RAD_RG)), 'k--', linewidth=1)
        plt.plot(RAD_RG, ic_low, linewidth=2, color='k', linestyle='--')
        plt.plot(RAD_RG, ic_med, linewidth=2, color='k')
        plt.plot(RAD_RG, ic_high, linewidth=2, color='k', linestyle='--')
        plt.xlim(r_m_h, RAD_RG.max())
        # plt.ylim(-5, 5)
        plt.ylim(-25, 25)
        plt.tight_layout()
        # plt.show(block=True)
        plt.savefig(OUT_DIR + '/mcs_she_err_' + str(CONV_ITER) + '.png')
        plt.close()

        # Plotting times
        plt.figure()
        # plt.title('Averaged time for computing unit')
        plt.ylabel('Time [s]')
        sph_ic_low = np.percentile(sph_times, PERCENT, axis=0, interpolation='linear')
        sph_ic_med = np.percentile(sph_times, 50, axis=0, interpolation='linear')
        sph_ic_high = np.percentile(sph_times, 100 - PERCENT, axis=0, interpolation='linear')
        she_ic_low = np.percentile(she_times, PERCENT, axis=0, interpolation='linear')
        she_ic_med = np.percentile(she_times, 50, axis=0, interpolation='linear')
        she_ic_high = np.percentile(she_times, 100 - PERCENT, axis=0, interpolation='linear')
        plt.bar(1, sph_ic_med, BAR_WIDTH, color='blue', linewidth=2, edgecolor='k')
        plt.errorbar(1, sph_ic_med,
                     yerr=np.asarray([[sph_ic_med - sph_ic_low, sph_ic_high - sph_ic_med], ]).reshape(2, 1),
                     ecolor='k', elinewidth=4, capthick=4, capsize=8)
        plt.bar(2, she_ic_med, BAR_WIDTH, color='blue', linewidth=2, edgecolor='k')
        plt.errorbar(2, she_ic_med,
                     yerr=np.asarray([[she_ic_med - she_ic_low, she_ic_high - she_ic_med], ]).reshape(2, 1),
                     ecolor='k', elinewidth=4, capthick=4, capsize=8)
        plt.xticks((1, 2), ('SPHERE', 'SHELL'))
        plt.xlim(0.5, 2.5)
        plt.ylim(0, 1.8)
        plt.tight_layout()
        # plt.show(block=True)
        plt.savefig(OUT_DIR + '/mcs_time_' + str(CONV_ITER) + '.png')
        plt.close()

    # Check precision for Nhoods using VOIs as 3D arrays and Monte Carlo method
    def test_volume_MCA(self):

        # return

        # VOI array creation
        rad_max = RAD_RG.max()
        max_size = int(math.ceil(2 * rad_max + SHELL_THICKS[3]))
        voi = np.ones(shape=(max_size, max_size, max_size), dtype=bool)
        voi_center = (.5 * voi.shape[0], .5 * voi.shape[1], .5 * voi.shape[2])
        X = np.meshgrid(np.arange(voi.shape[0]), np.arange(voi.shape[1]), np.arange(voi.shape[2]))[0]
        voi[X > voi_center[0]] = False

        # Computations loop
        sph_errs = np.zeros(shape=(len(RAD_RG), N_SIMS), dtype=np.float32)
        she_errs = np.zeros(shape=(len(RAD_RG), N_SIMS), dtype=np.float32)
        sph_times = np.zeros(shape=N_SIMS, dtype=np.float32)
        she_times = np.zeros(shape=N_SIMS, dtype=np.float32)
        for i in range(N_SIMS):
            # Computations
            hold_time = time.time()
            spheres = ListSphereNhood(center=voi_center, radius_rg=RAD_RG, voi=voi, conv_iter=CONV_ITER,
                                      max_iter=MAX_ITER)
            sph_vol = np.asarray(spheres.get_volumes())
            sph_times[i] = time.time() - hold_time
            hold_time = time.time()
            shells = ListShellNhood(center=voi_center, radius_rg=RAD_RG, voi=voi, conv_iter=CONV_ITER,
                                    max_iter=MAX_ITER, thick=SHELL_THICKS[0])
            she_vol = np.asarray(shells.get_volumes())
            she_times[i] = time.time() - hold_time

            # Computing the errors
            sph_volr = .5 * (4. / 3.) * np.pi * (RAD_RG ** 3)
            she_volr = .5 * (4. / 3.) * np.pi * \
                       ((RAD_RG + 0.5 * SHELL_THICKS[0]) ** 3 - (RAD_RG - 0.5 * SHELL_THICKS[0]) ** 3)
            sph_errs[:, i] = 100. * (sph_vol - sph_volr) / sph_volr
            she_errs[:, i] = 100. * (she_vol - she_volr) / she_volr

        # Plotting error
        plt.figure()
        # plt.title('Averaged fractional error for volume estimation (MCA-SPHERE)')
        plt.xlabel('Scale')
        plt.ylabel('E [%]')
        ic_low = np.percentile(sph_errs, PERCENT, axis=1, interpolation='linear')
        ic_med = np.percentile(sph_errs, 50, axis=1, interpolation='linear')
        ic_high = np.percentile(sph_errs, 100 - PERCENT, axis=1, interpolation='linear')
        plt.fill_between(RAD_RG, ic_low, ic_high, alpha=.5, color='gray', edgecolor='w')
        plt.plot(RAD_RG, np.zeros(shape=len(RAD_RG)), 'k--', linewidth=1)
        plt.plot(RAD_RG, ic_low, linewidth=2, color='k', linestyle='--')
        plt.plot(RAD_RG, ic_med, linewidth=2, color='k')
        plt.plot(RAD_RG, ic_high, linewidth=2, color='k', linestyle='--')
        plt.xlim(RAD_RG.min(), RAD_RG.max())
        plt.ylim(-100, 100)
        plt.tight_layout()
        # plt.show(block=True)
        plt.savefig(OUT_DIR + '/mca_sph_err_' + str(CONV_ITER) + '.png')
        plt.close()
        plt.figure()
        # plt.title('Averaged fractional error for volume estimation (MCA-SHELL)')
        plt.xlabel('Scale')
        plt.ylabel('E [%]')
        ic_low = np.percentile(she_errs, PERCENT, axis=1, interpolation='linear')
        ic_med = np.percentile(she_errs, 50, axis=1, interpolation='linear')
        ic_high = np.percentile(she_errs, 100 - PERCENT, axis=1, interpolation='linear')
        plt.fill_between(RAD_RG, ic_low, ic_high, alpha=.5, color='gray', edgecolor='w')
        plt.plot(RAD_RG, np.zeros(shape=len(RAD_RG)), 'k--', linewidth=1)
        plt.plot(RAD_RG, ic_low, linewidth=2, color='k', linestyle='--')
        plt.plot(RAD_RG, ic_med, linewidth=2, color='k')
        plt.plot(RAD_RG, ic_high, linewidth=2, color='k', linestyle='--')
        plt.xlim(RAD_RG.min(), RAD_RG.max())
        plt.ylim(-100, 100)
        plt.tight_layout()
        # plt.show(block=True)
        plt.savefig(OUT_DIR + '/mca_she_err_' + str(CONV_ITER) + '.png')
        plt.close()

        # Plotting times
        plt.figure()
        # plt.title('Averaged time for computing unit')
        plt.ylabel('Time [s]')
        sph_ic_low = np.percentile(sph_times, PERCENT, axis=0, interpolation='linear')
        sph_ic_med = np.percentile(sph_times, 50, axis=0, interpolation='linear')
        sph_ic_high = np.percentile(sph_times, 100 - PERCENT, axis=0, interpolation='linear')
        she_ic_low = np.percentile(she_times, PERCENT, axis=0, interpolation='linear')
        she_ic_med = np.percentile(she_times, 50, axis=0, interpolation='linear')
        she_ic_high = np.percentile(she_times, 100 - PERCENT, axis=0, interpolation='linear')
        plt.bar(1, sph_ic_med, BAR_WIDTH, color='blue', linewidth=2, edgecolor='k')
        plt.errorbar(1, sph_ic_med,
                     yerr=np.asarray([[sph_ic_med - sph_ic_low, sph_ic_high - sph_ic_med], ]).reshape(2, 1),
                     ecolor='k', elinewidth=4, capthick=4, capsize=8)
        plt.bar(2, she_ic_med, BAR_WIDTH, color='blue', linewidth=2, edgecolor='k')
        plt.errorbar(2, she_ic_med,
                     yerr=np.asarray([[she_ic_med - she_ic_low, she_ic_high - she_ic_med], ]).reshape(2, 1),
                     ecolor='k', elinewidth=4, capthick=4, capsize=8)
        plt.xticks((1, 2), ('SPHERE', 'SHELL'))
        plt.ylim(0, 1.8)
        plt.tight_layout()
        # plt.show(block=True)
        plt.savefig(OUT_DIR + '/mca_time_' + str(CONV_ITER) + '.png')
        plt.close()

    # Check precision for Nhoods using VOIs as 3D arrays and Direct Sum method
    def test_volume_DSA(self):

        # return

        # VOI array creation
        r_m_h = 5
        rad_max = RAD_RG.max()
        max_size = int(math.ceil(2 * rad_max + SHELL_THICKS[2]))
        voi = np.ones(shape=(max_size, max_size, max_size), dtype=bool)
        voi_center = (.5 * voi.shape[0], .5 * voi.shape[1], .5 * voi.shape[2])
        X = np.meshgrid(np.arange(voi.shape[0]), np.arange(voi.shape[1]), np.arange(voi.shape[2]))[0]
        # voi[X > voi_center[0]] = False
        voi[X > voi_center[0] + r_m_h] = False

        # Computations loop
        # Computations for sphere
        hold_time = time.time()
        spheres = ListSphereNhood(center=voi_center, radius_rg=RAD_RG, voi=voi, conv_iter=None,
                                  max_iter=None)
        sph_vol = np.asarray(spheres.get_volumes())
        sph_times = time.time() - hold_time
        hold_time = time.time()
        # sph_volr = .5 * (4. / 3.) * np.pi * (RAD_RG ** 3)
        h_arr = RAD_RG - r_m_h
        sph_volr = np.pi * ((4. / 3.) * (RAD_RG ** 3) - RAD_RG * (h_arr ** 2) + (h_arr ** 3) * (1. / 3.))
        sph_errs = 100. * (sph_vol - sph_volr) / sph_volr
        # Computations for shell
        she_errs = np.zeros(shape=(len(RAD_RG), len(SHELL_THICKS)), dtype=np.float32)
        she_times = np.zeros(shape=N_SIMS, dtype=np.float32)
        for i, thick in enumerate(SHELL_THICKS):
            shells = ListShellNhood(center=voi_center, radius_rg=RAD_RG, voi=voi, conv_iter=None,
                                    max_iter=None, thick=thick)
            she_vol = np.asarray(shells.get_volumes())
            she_times[i] = time.time() - hold_time
            # she_volr = .5 * (4. / 3.) * np.pi * ((RAD_RG + 0.5 * thick) ** 3 - (RAD_RG - 0.5 * thick) ** 3)
            rad_rg_1, rad_rg_2 = RAD_RG - .5 * thick, RAD_RG + .5 * thick
            h_arr_1, h_arr_2 = rad_rg_1 - r_m_h, rad_rg_2 - r_m_h
            she_volr = np.pi * ((4. / 3.) * (rad_rg_2 ** 3) - rad_rg_2 * (h_arr_2 ** 2) + (h_arr_2 ** 3) * (1. / 3.)
                                - (4. / 3.) * (rad_rg_1 ** 3) + rad_rg_1 * (h_arr_1 ** 2) - (h_arr_1 ** 3) * (1. / 3.))
            she_errs[:, i] = 100. * (she_vol - she_volr) / she_volr

        # Plotting error
        plt.figure()
        # plt.title('Averaged fractional error for volume estimation (DSA-SPHERE)')
        plt.xlabel('Scale')
        plt.ylabel('E [%]')
        plt.plot(RAD_RG, np.zeros(shape=len(RAD_RG)), 'k--', linewidth=1)
        plt.plot(RAD_RG, sph_errs, linewidth=2, color='k')
        plt.xlim(r_m_h, RAD_RG.max())
        plt.ylim(-10, 10)
        plt.tight_layout()
        # plt.show(block=True)
        plt.savefig(OUT_DIR + '/dsa_sph_err_' + str(CONV_ITER) + '.png')
        plt.close()
        plt.figure()
        # plt.title('Averaged fractional error for volume estimation (DSA-SHELL)')
        plt.xlabel('Scale')
        plt.ylabel('E [%]')
        plt.plot(RAD_RG, np.zeros(shape=len(RAD_RG)), 'k--', linewidth=1)
        for i, thick in enumerate(SHELL_THICKS):
            plt.plot(RAD_RG, she_errs[:, i], linewidth=2, color=THICK_COLORS[i], label=str(thick))
        plt.xlim(r_m_h, RAD_RG.max())
        plt.ylim(-10, 10)
        plt.legend(title='Thickness', loc=4)
        plt.tight_layout()
        plt.savefig(OUT_DIR + '/dsa_she_err_' + str(CONV_ITER) + '.png')
        # plt.show(block=True)
        plt.close()

        # Plotting times
        plt.figure()
        # plt.title('Averaged time for computing unit')
        plt.ylabel('Time [s]')
        plt.bar(1, sph_times, BAR_WIDTH, color='blue', linewidth=2, edgecolor='k')
        plt.bar(2, she_times.mean(), BAR_WIDTH, color='blue', linewidth=2, edgecolor='k')
        plt.xticks((1, 2), ('SPHERE', 'SHELL'))
        plt.ylim(0, 1.8)
        plt.tight_layout()
        plt.savefig(OUT_DIR + '/dsa_time_' + str(CONV_ITER) + '.png')
        # plt.show(block=True)
        plt.close()


    # Check precision for Nhoods using VOIs as 3D arrays and Direct Sum method
    def test_times(self):

        # return

        # VOI array creation
        N_SIMS = 1
        rad_max = RAD_RG.max()
        max_size = int(math.ceil(2 * 200 + SHELL_THICKS[2]))
        voi = np.ones(shape=(max_size, max_size, max_size), dtype=bool)
        voi_center = (.5 * voi.shape[0], .5 * voi.shape[1], .5 * voi.shape[2])
        X = np.meshgrid(np.arange(voi.shape[0]), np.arange(voi.shape[1]), np.arange(voi.shape[2]))[0]
        voi[X > voi_center[0]] = False
        voi_surf = iso_surface(voi, .5, closed=True, normals='outwards')
        voi_surf = poly_decimate(voi_surf, .9)

        # Loop for maximum radius
        n_rads_arr = np.arange(15, 200, 5)
        sph_times_dsa = np.zeros(shape=(len(n_rads_arr), N_SIMS), dtype=np.float32)
        she_times_dsa = np.zeros(shape=(len(n_rads_arr), N_SIMS), dtype=np.float32)
        sph_times_mcs = np.zeros(shape=(len(n_rads_arr), N_SIMS), dtype=np.float32)
        she_times_mcs = np.zeros(shape=(len(n_rads_arr), N_SIMS), dtype=np.float32)
        # Loop for the number of particles within the neighborhod
        for i, max_rad in enumerate(n_rads_arr[1:]):
            rad_rg = np.arange(10, max_rad, 5)
            tot_vol, tot_area = 0., 0.
            for rad in rad_rg:
                tot_vol += ((4./3.) * np.pi * rad * rad * rad)
                tot_area += (4. * np.pi * rad * rad)
            # Loop for the number of simulation
            for j in range(N_SIMS):
                # Computations for DSA
                hold_time = time.time()
                spheres = ListSphereNhood(center=voi_center, radius_rg=rad_rg, voi=voi, conv_iter=None,
                                          max_iter=None)
                spheres.get_volumes()
                sph_times_dsa[i, j] = time.time() - hold_time
                hold_time = time.time()
                shells = ListShellNhood(center=voi_center, radius_rg=rad_rg, voi=voi, conv_iter=None,
                                          max_iter=None, thick=6)
                shells.get_volumes()
                she_times_dsa[i, j] = time.time() - hold_time
                # Computations for MCS
                hold_time = time.time()
                spheres = ListSphereNhood(center=voi_center, radius_rg=rad_rg, voi=voi_surf, conv_iter=CONV_ITER,
                                          max_iter=MAX_ITER)
                spheres.get_volumes()
                sph_times_mcs[i, j] = time.time() - hold_time
                hold_time = time.time()
                shells = ListShellNhood(center=voi_center, radius_rg=rad_rg, voi=voi_surf, conv_iter=CONV_ITER,
                                        max_iter=MAX_ITER, thick=6)
                shells.get_volumes()
                she_times_mcs[i, j] = time.time() - hold_time
            sph_times_dsa[i, :] /= float(tot_vol)
            she_times_dsa[i, :] /= float(tot_area)
            sph_times_mcs[i, :] /= float(tot_vol)
            she_times_mcs[i, :] /= float(tot_area)

        # Plotting times
        plt.figure()
        plt.title('HALF-SPHERE')
        plt.ylabel('Time/v.u. [s]')
        plt.xlabel('Maximum scale')
        # ic_low = np.percentile(sph_times_dsa, PERCENT, axis=1, interpolation='linear')
        # ic_med = np.percentile(sph_times_dsa, 50, axis=1, interpolation='linear')
        # ic_high = np.percentile(sph_times_dsa, 100 - PERCENT, axis=1, interpolation='linear')
        # plt.plot(n_rads_arr, ic_med, color='blue', linewidth=2, label='3D Array')
        plt.plot(n_rads_arr, sph_times_dsa[:, 0], color='blue', linewidth=2, label='3D Array')
        # plt.fill_between(n_rads_arr, ic_low, ic_high, alpha=0.5, color='blue', edgecolor='w')
        # ic_low = np.percentile(sph_times_mcs, PERCENT, axis=1, interpolation='linear')
        # ic_med = np.percentile(sph_times_mcs, 50, axis=1, interpolation='linear')
        # ic_high = np.percentile(sph_times_mcs, 100 - PERCENT, axis=1, interpolation='linear')
        # plt.plot(n_rads_arr, ic_med, color='red', linewidth=2, linestyle='--', label='MCS')
        plt.plot(n_rads_arr, sph_times_mcs[:, 0], color='red', linewidth=2, label='MCS')
        # plt.fill_between(n_rads_arr, ic_low, ic_high, alpha=0.5, color='red', edgecolor='w')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.legend(loc=1)
        plt.tight_layout()
        # plt.show(block=True)
        plt.savefig(OUT_DIR + '/times_vol_sphere.png')
        plt.close()

        # Plotting times
        plt.figure()
        plt.title('HALF-SHELL')
        plt.ylabel('Time/a.u. [s]')
        plt.xlabel('Maximum scale')
        # ic_low = np.percentile(she_times_dsa, PERCENT, axis=1, interpolation='linear')
        # ic_med = np.percentile(she_times_dsa, 50, axis=1, interpolation='linear')
        # ic_high = np.percentile(she_times_dsa, 100 - PERCENT, axis=1, interpolation='linear')
        # plt.plot(n_rads_arr, ic_med, color='blue', linewidth=2, label='3D Array')
        plt.plot(n_rads_arr, she_times_dsa[:, 0], color='blue', linewidth=2, label='3D Array')
        # plt.fill_between(n_rads_arr, ic_low, ic_high, alpha=0.5, color='blue', edgecolor='w')
        # ic_low = np.percentile(she_times_mcs, PERCENT, axis=1, interpolation='linear')
        # ic_med = np.percentile(she_times_mcs, 50, axis=1, interpolation='linear')
        # ic_high = np.percentile(she_times_mcs, 100 - PERCENT, axis=1, interpolation='linear')
        # plt.plot(n_rads_arr, ic_med, color='red', linewidth=2, linestyle='--', label='MCS')
        plt.plot(n_rads_arr, she_times_mcs[:, 0], color='red', linewidth=2, label='MCS')
        # plt.fill_between(n_rads_arr, ic_low, ic_high, alpha=0.5, color='red', edgecolor='w')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.legend(loc=1)
        plt.tight_layout()
        # plt.show(block=True)
        plt.savefig(OUT_DIR + '/times_vol_shell.png')
        plt.close()