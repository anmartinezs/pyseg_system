
import vtk
import math
import time
import numpy as np
from unittest import TestCase
from ..surface import ListShellNhood, ListSphereNhood
from ..utils import disperse_io, iso_surface, poly_decimate
from ..utils import points_to_poly
from matplotlib import pyplot as plt, rcParams

# Global variables
RAD = 50
SHELL_THICK = 3
THICK_COLORS = ('b', 'r', 'g', 'c')
N_PARTS = 10000
CONV_ITER = 1000
MAX_ITER = 100000
N_SIMS = 1000
PERCENT = 5 # %
BAR_WIDTH = .4
OUT_DIR = './surf/test/out/nhood_num'

rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14
rcParams['patch.linewidth'] = 2

#######################################################################################################################
# GLOBAL FUNCTIONS
#######################################################################################################################

def gen_points_sph(n_points, rad):
    """
    Generates points uniformly distributed on a sphere centered at (0, 0, 0)
    :param n_points: number of points to generate
    :param rad: sphere radius
    :return: an array with shape [n_point, 3] with the point coordinates
    """
    coords = np.zeros(shape=(n_points, 3), dtype=np.float32)
    for i in range(n_points):
        u = np.random.rand(1)[0]
        u = rad * np.cbrt(u)
        X = np.random.randn(1, 3)[0]
        norm = u / np.linalg.norm(X)
        coords[i, :] = X * norm
    return coords


def gen_points_she(n_points, rad):
    """
    Generates points uniformly distributed on the sphere surface centered at (0, 0, 0)
    :param n_points: number of points to generate
    :param rad: sphere radius
    :return: an array with shape [n_point, 3] with the point coordinates
    """
    coords = np.zeros(shape=(n_points, 3), dtype=np.float32)
    for i in range(n_points):
        X = np.random.randn(1, 3)[0]
        norm = rad / np.linalg.norm(X)
        coords[i, :] = X * norm
    return coords


#######################################################################################################################
# TEST CLASSES
#######################################################################################################################


class TestListNhoodNum(TestCase):

    # Check precision and time for Nhoods using VOIs as 3D surfaces and Monte Carlo method
    def test_num_surf(self):

        # VOI Creation
        # cuber = vtk.vtkCubeSource()
        # off, max_thick = RAD, SHELL_THICK
        # cuber.SetBounds(-off - max_thick, off + max_thick, -off - max_thick, off + max_thick,
        #                 -off - max_thick, off + max_thick)
        # cuber.Update()
        # voi = cuber.GetOutput()
        # orienter = vtk.vtkPolyDataNormals()
        # orienter.SetInputData(voi)
        # orienter.AutoOrientNormalsOn()
        # orienter.Update()
        # voi = orienter.GetOutput()
        # disperse_io.save_vtp(voi, OUT_DIR + '/cube_voi.vtp')
        # voi_center = (0, 0, 0)
        max_size = int(math.ceil(2 * RAD + SHELL_THICK))
        seg = np.ones(shape=(max_size, max_size, max_size), dtype=bool)
        voi_center = (.5 * seg.shape[0], .5 * seg.shape[1], .5 * seg.shape[2])
        X = np.meshgrid(np.arange(seg.shape[0]), np.arange(seg.shape[1]), np.arange(seg.shape[2]))[0]
        seg[X > voi_center[0]] = False
        voi = iso_surface(seg, .5, closed=True, normals='outwards')
        voi = poly_decimate(voi, .9)

        # Particles coordinates generation (just for storing)
        coords_sph, coords_she = gen_points_sph(N_PARTS, RAD), gen_points_she(N_PARTS, RAD)
        coords_sph, coords_she = points_to_poly(coords_sph), points_to_poly(coords_she)
        disperse_io.save_vtp(coords_sph, OUT_DIR + '/points_sph.vtp')
        disperse_io.save_vtp(coords_she, OUT_DIR + '/points_she.vtp')

        # Computations loop
        sph_errs = np.zeros(shape=N_SIMS, dtype=np.float32)
        she_errs = np.zeros(shape=N_SIMS, dtype=np.float32)
        sph_times = np.zeros(shape=N_SIMS, dtype=np.float32)
        she_times = np.zeros(shape=N_SIMS, dtype=np.float32)
        for i in range(N_SIMS):
            coords_sph, coords_she = gen_points_sph(N_PARTS, RAD) + voi_center, \
                                     gen_points_she(N_PARTS, RAD) + voi_center
            # Computations
            hold_time = time.time()
            spheres = ListSphereNhood(center=voi_center, radius_rg=[RAD,], voi=voi, conv_iter=CONV_ITER,
                                      max_iter=MAX_ITER)
            sph_num = np.asarray(spheres.get_nums_embedded(coords_sph))
            sph_times[i] = time.time() - hold_time
            hold_time = time.time()
            shells = ListShellNhood(center=voi_center, radius_rg=[RAD,], voi=voi, conv_iter=CONV_ITER,
                                    max_iter=MAX_ITER, thick=SHELL_THICK)
            she_num = np.asarray(shells.get_nums_embedded(coords_she))
            she_times[i] = time.time() - hold_time

            # Computing the errors
            sph_numr, she_numr = len(coords_sph), len(coords_she)
            sph_errs[i] = 100. * (sph_num - sph_numr) / sph_numr
            she_errs[i] = 100. * (she_num - she_numr) / she_numr
        sph_times /= float(N_PARTS)
        she_times /= float(N_PARTS)

        # Plotting error
        plt.figure()
        # plt.title('Precision counting the number of embedded particles')
        plt.ylabel('E [%]')
        sph_ic_low = np.percentile(sph_errs, PERCENT, axis=0, interpolation='linear')
        sph_ic_med = np.percentile(sph_errs, 50, axis=0, interpolation='linear')
        sph_ic_high = np.percentile(sph_errs, 100 - PERCENT, axis=0, interpolation='linear')
        she_ic_low = np.percentile(she_errs, PERCENT, axis=0, interpolation='linear')
        she_ic_med = np.percentile(she_errs, 50, axis=0, interpolation='linear')
        she_ic_high = np.percentile(she_errs, 100 - PERCENT, axis=0, interpolation='linear')
        plt.bar(1, sph_ic_med, BAR_WIDTH, color='blue', linewidth=2, edgecolor='k')
        plt.errorbar(1, sph_ic_med,
                     yerr=np.asarray([[sph_ic_med - sph_ic_low, sph_ic_high - sph_ic_med], ]).reshape(2, 1),
                     ecolor='k', elinewidth=4, capthick=4, capsize=8)
        plt.bar(2, she_ic_med, BAR_WIDTH, color='blue', linewidth=2, edgecolor='k')
        plt.errorbar(2, she_ic_med,
                     yerr=np.asarray([[she_ic_med - she_ic_low, she_ic_high - she_ic_med], ]).reshape(2, 1),
                     ecolor='k', elinewidth=4, capthick=4, capsize=8)
        plt.xticks((1, 2), ('SPHERE', 'SHELL'))
        # plt.xlim(-100, 100)
        plt.tight_layout()
        # plt.show(block=True)
        plt.savefig(OUT_DIR + '/surf_error_' + str(N_PARTS) + '.png')
        plt.close()

        # Plotting times
        plt.figure()
        # plt.title('Accumulated time for counting the number of embedded particles')
        plt.ylabel('Time/particle [s]')
        sph_ic_low = np.percentile(sph_times, PERCENT, axis=0, interpolation='linear')
        sph_ic_med = np.percentile(sph_times, 50, axis=0, interpolation='linear')
        sph_ic_high = np.percentile(sph_times, 100 - PERCENT, axis=0, interpolation='linear')
        she_ic_low = np.percentile(she_times, PERCENT, axis=0, interpolation='linear')
        she_ic_med = np.percentile(she_times, 50, axis=0, interpolation='linear')
        she_ic_high = np.percentile(she_times, 100 - PERCENT, axis=0, interpolation='linear')
        plt.bar(1, sph_ic_med, BAR_WIDTH, color='blue', linewidth=2)
        plt.errorbar(1, sph_ic_med,
                     yerr=np.asarray([[sph_ic_med - sph_ic_low, sph_ic_high - sph_ic_med], ]).reshape(2, 1),
                     ecolor='k', elinewidth=4, capthick=4, capsize=8)
        plt.bar(2, she_ic_med, BAR_WIDTH, color='blue', linewidth=2)
        plt.errorbar(2, she_ic_med,
                     yerr=np.asarray([[she_ic_med - she_ic_low, she_ic_high - she_ic_med], ]).reshape(2, 1),
                     ecolor='k', elinewidth=4, capthick=4, capsize=8)
        plt.xticks((1, 2), ('SPHERE', 'SHELL'))
        plt.xlim(0.5, 2.5)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        # plt.ylim(0, 0.00006)
        # plt.ylim(0, 1.3)
        plt.tight_layout()
        # plt.show(block=True)
        plt.savefig(OUT_DIR + '/surf_time_' + str(N_PARTS) + '.png')
        plt.close()

    # Check precision and time for Nhoods using VOIs as 3D arrays and Monte Carlo method
    def test_num_array(self):

        # VOI array creation
        max_size = int(math.ceil(2 * RAD + SHELL_THICK))
        voi = np.ones(shape=(max_size, max_size, max_size), dtype=bool)
        voi_center = (.5 * voi.shape[0], .5 * voi.shape[1], .5 * voi.shape[2])

        # Particles coordinates generation (just for storing)
        coords_sph, coords_she = gen_points_sph(N_PARTS, RAD), gen_points_she(N_PARTS, RAD)
        coords_sph, coords_she = points_to_poly(coords_sph), points_to_poly(coords_she)
        disperse_io.save_vtp(coords_sph, OUT_DIR + '/points_sph.vtp')
        disperse_io.save_vtp(coords_she, OUT_DIR + '/points_she.vtp')

        # Computations loop
        sph_errs = np.zeros(shape=N_SIMS, dtype=np.float32)
        she_errs = np.zeros(shape=N_SIMS, dtype=np.float32)
        sph_times = np.zeros(shape=N_SIMS, dtype=np.float32)
        she_times = np.zeros(shape=N_SIMS, dtype=np.float32)
        for i in range(N_SIMS):
            coords_sph, coords_she = gen_points_sph(N_PARTS, RAD) + voi_center, gen_points_she(N_PARTS, RAD) + voi_center
            # Computations
            hold_time = time.time()
            spheres = ListSphereNhood(center=voi_center, radius_rg=[RAD, ], voi=voi, conv_iter=None, max_iter=None)
            sph_num = np.asarray(spheres.get_nums_embedded(coords_sph))
            sph_times[i] = time.time() - hold_time
            hold_time = time.time()
            shells = ListShellNhood(center=voi_center, radius_rg=[RAD, ], voi=voi, conv_iter=None, max_iter=None,
                                    thick=SHELL_THICK)
            she_num = np.asarray(shells.get_nums_embedded(coords_she))
            she_times[i] = time.time() - hold_time

            # Computing the errors
            sph_numr, she_numr = len(coords_sph), len(coords_she)
            sph_errs[i] = 100. * (sph_num - sph_numr) / sph_numr
            she_errs[i] = 100. * (she_num - she_numr) / she_numr
        sph_times /= float(N_PARTS)
        she_times /= float(N_PARTS)

        # Plotting error
        plt.figure()
        # plt.title('Precision for counting the number of embedded particles')
        plt.ylabel('E [%]')
        sph_ic_low = np.percentile(sph_errs, PERCENT, axis=0, interpolation='linear')
        sph_ic_med = np.percentile(sph_errs, 50, axis=0, interpolation='linear')
        sph_ic_high = np.percentile(sph_errs, 100 - PERCENT, axis=0, interpolation='linear')
        she_ic_low = np.percentile(she_errs, PERCENT, axis=0, interpolation='linear')
        she_ic_med = np.percentile(she_errs, 50, axis=0, interpolation='linear')
        she_ic_high = np.percentile(she_errs, 100 - PERCENT, axis=0, interpolation='linear')
        plt.bar(1, sph_ic_med, BAR_WIDTH, color='blue', linewidth=2, edgecolor='k')
        plt.errorbar(1, sph_ic_med,
                     yerr=np.asarray([[sph_ic_med - sph_ic_low, sph_ic_high - sph_ic_med], ]).reshape(2, 1),
                     ecolor='k', elinewidth=4, capthick=4, capsize=8)
        plt.bar(2, she_ic_med, BAR_WIDTH, color='blue', linewidth=2, edgecolor='k')
        plt.errorbar(2, she_ic_med,
                     yerr=np.asarray([[she_ic_med - she_ic_low, she_ic_high - she_ic_med], ]).reshape(2, 1),
                     ecolor='k', elinewidth=4, capthick=4, capsize=8)
        plt.xticks((1, 2), ('SPHERE', 'SHELL'))
        # plt.xlim(-100, 100)
        plt.tight_layout()
        # plt.show(block=True)
        plt.savefig(OUT_DIR + '/array_error_' + str(N_PARTS) + '.png')
        plt.close()

        # Plotting times
        plt.figure()
        # plt.title('Accumulated time for counting the number of embedded particles')
        plt.ylabel('Time/particle [s]')
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
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        # plt.ylim(0, 0.00006)
        plt.tight_layout()
        # plt.show(block=True)
        plt.savefig(OUT_DIR + '/array_time_' + str(N_PARTS) + '.png')
        plt.close()


    def test_times(self):

        # VOI Creation
        max_size = int(math.ceil(2 * RAD + SHELL_THICK))
        voi = np.ones(shape=(max_size, max_size, max_size), dtype=bool)
        voi_center = (.5 * voi.shape[0], .5 * voi.shape[1], .5 * voi.shape[2])
        X = np.meshgrid(np.arange(voi.shape[0]), np.arange(voi.shape[1]), np.arange(voi.shape[2]))[0]
        voi[X > voi_center[0]] = False
        voi_surf = iso_surface(voi, .5, closed=True, normals='outwards')
        voi_surf = poly_decimate(voi_surf, .9)

        # Computations loop
        # n_parts_arr = np.logspace(1, str(N_PARTS).count('0'), num=str(N_PARTS).count('0')).astype(int)
        n_parts_arr = np.logspace(1, str(N_PARTS).count('0'), num=4*str(N_PARTS).count('0')).astype(int)
        sph_times = np.zeros(shape=(len(n_parts_arr), N_SIMS), dtype=np.float32)
        she_times = np.zeros(shape=(len(n_parts_arr), N_SIMS), dtype=np.float32)
        sph_times_surf = np.zeros(shape=(len(n_parts_arr), N_SIMS), dtype=np.float32)
        she_times_surf = np.zeros(shape=(len(n_parts_arr), N_SIMS), dtype=np.float32)
        # Loop for the number of particles within the neighborhod
        for i, n_parts in enumerate(n_parts_arr):
            # Loop for the number of simulation
            for j in range(N_SIMS):
                coords_sph, coords_she = gen_points_sph(n_parts, RAD) + voi_center, \
                                         gen_points_she(n_parts, RAD) + voi_center
                # Computations
                hold_time = time.time()
                spheres = ListSphereNhood(center=voi_center, radius_rg=[RAD, ], voi=voi, conv_iter=CONV_ITER,
                                          max_iter=MAX_ITER)
                spheres.get_nums_embedded(coords_she)
                sph_times[i, j] = time.time() - hold_time
                hold_time = time.time()
                shells = ListShellNhood(center=voi_center, radius_rg=[RAD, ], voi=voi, conv_iter=CONV_ITER,
                                        max_iter=MAX_ITER, thick=SHELL_THICK)
                shells.get_nums_embedded(coords_she)
                she_times[i, j] = time.time() - hold_time
                hold_time = time.time()
                spheres = ListSphereNhood(center=voi_center, radius_rg=[RAD, ], voi=voi_surf, conv_iter=CONV_ITER,
                                          max_iter=MAX_ITER)
                spheres.get_nums_embedded(coords_she)
                sph_times_surf[i, j] = time.time() - hold_time
                hold_time = time.time()
                shells = ListShellNhood(center=voi_center, radius_rg=[RAD, ], voi=voi_surf, conv_iter=CONV_ITER,
                                        max_iter=MAX_ITER, thick=SHELL_THICK)
                shells.get_nums_embedded(coords_she)
                she_times_surf[i, j] = time.time() - hold_time
            sph_times[i, :] /= float(n_parts)
            she_times[i, :] /= float(n_parts)
            sph_times_surf[i, :] /= float(n_parts)
            she_times_surf[i, :] /= float(n_parts)

        # Plotting times
        plt.figure()
        plt.title('SPHERE')
        plt.ylabel('Time/particle [s]')
        plt.xlabel('#particles in VOI')
        ic_low = np.percentile(sph_times, PERCENT, axis=1, interpolation='linear')
        ic_med = np.percentile(sph_times, 50, axis=1, interpolation='linear')
        ic_high = np.percentile(sph_times, 100 - PERCENT, axis=1, interpolation='linear')
        plt.plot(n_parts_arr, ic_med, color='blue', linewidth=2, label='ARRAY')
        plt.fill_between(n_parts_arr, ic_low, ic_high, alpha=0.5, color='blue', edgecolor='w')
        ic_low = np.percentile(sph_times_surf, PERCENT, axis=1, interpolation='linear')
        ic_med = np.percentile(sph_times_surf, 50, axis=1, interpolation='linear')
        ic_high = np.percentile(sph_times_surf, 100 - PERCENT, axis=1, interpolation='linear')
        plt.plot(n_parts_arr, ic_med, color='red', linewidth=2, linestyle='--', label='SURFACE')
        plt.fill_between(n_parts_arr, ic_low, ic_high, alpha=0.5, color='red', edgecolor='w')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.xscale('log')
        plt.legend(loc=1)
        plt.tight_layout()
        # plt.show(block=True)
        plt.savefig(OUT_DIR + '/times_sphere_' + str(N_PARTS) + '.png')
        plt.close()

        # Plotting times
        plt.figure()
        plt.title('SHELL')
        plt.ylabel('Time/particle [s]')
        plt.xlabel('#particles in VOI')
        ic_low = np.percentile(she_times, PERCENT, axis=1, interpolation='linear')
        ic_med = np.percentile(she_times, 50, axis=1, interpolation='linear')
        ic_high = np.percentile(she_times, 100 - PERCENT, axis=1, interpolation='linear')
        plt.plot(n_parts_arr, ic_med, color='blue', linewidth=2, label='ARRAY')
        plt.fill_between(n_parts_arr, ic_low, ic_high, alpha=0.5, color='blue', edgecolor='w')
        ic_low = np.percentile(she_times_surf, PERCENT, axis=1, interpolation='linear')
        ic_med = np.percentile(she_times_surf, 50, axis=1, interpolation='linear')
        ic_high = np.percentile(she_times_surf, 100 - PERCENT, axis=1, interpolation='linear')
        plt.plot(n_parts_arr, ic_med, color='red', linewidth=2, linestyle='--', label='SURFACE')
        plt.fill_between(n_parts_arr, ic_low, ic_high, alpha=0.5, color='red', edgecolor='w')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.xscale('log')
        plt.legend(loc=1)
        plt.tight_layout()
        # plt.show(block=True)
        plt.savefig(OUT_DIR + '/times_shell_' + str(N_PARTS) + '.png')
        plt.close()

