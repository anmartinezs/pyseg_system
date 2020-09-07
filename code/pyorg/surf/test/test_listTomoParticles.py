import time
import math
import os, shutil
import numpy as np
from unittest import TestCase
from ..model import ModelCSRV, ModelSRPV, Model2CCSRV, gen_tlist, gen_tlist2
from ..utils import iso_surface, poly_decimate, disperse_io, stat_dict_to_mat
from ..surface import Particle, TomoParticles, ListTomoParticles
from matplotlib import pyplot as plt, rcParams

# ListTomoParticles variables
PARTICLE_SURF = './surf/test/in/sph_rad_5_surf.vtp'
PARTICLE_SURF_SHELL = './surf/test/in/sph_rad_0.5_surf.vtp'
TOMO_SHAPE = (500, 500, 100)
N_TOMOS = 5
N_SIMS = 20
N_PART_TOMO = 200 # 200 # 600
N_CYCLES_TOMO = (4, 4, 1)
SIN_T = 0.8
DST_BI = 40 # 10
STD_BI = 5 # 1

# Analysis variables
RAD_RG = np.arange(4, 180, 5) # np.arange(4, 180, 3)
SHELL_THICK = 6
BORDER = True
CONV_ITER = 1000
MAX_ITER = 100000
N_PROCESSORS = 10 # None means Auto
DMAPS = True
OUT_DIR = './surf/test/out/uni_2nd'
OUT_DIR_BI = './surf/test/out/bi_2nd'

# Statistics variables
PERCENTILE = 5 # %
BAR_WIDTH = .4

# VOI shell thick
VOI_SHELL_THICK = 3
VOI_SHELL_STD = 15

rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14
rcParams['patch.linewidth'] = 2

# Clean an directory contents (directory is preserved)
# dir: directory path
def clean_dir(dir):
    for root, dirs, files in os.walk(dir):
#         for f in files:
#             os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d), ignore_errors=True)


class TestListTomoParticles(TestCase):

    def __init__(self, *args, **kwargs):
        super(TestListTomoParticles, self).__init__(*args, **kwargs)
        clean_dir(OUT_DIR)

    @staticmethod
    def gen_rect_voi_surface(shape, iso_th=.5, dec=.9):
        """
        Generates a rectangular surface VOI
        :param shape: 3-tuple with the length of the three rectangle sides
        :param iso_th: threshold for the isosurface (default 0.5)
        :param dec: triangles decimation factor (default 0.9)
        :return: a vtkPolyData object
        """
        seg = np.zeros(shape=np.asarray(shape)+2, dtype=np.float32)
        seg[1:shape[0]+1, 1:shape[1]+1, 1:shape[2]+1] = 1
        voi = iso_surface(seg, iso_th, closed=True, normals='outwards')
        return poly_decimate(voi, dec)

    @staticmethod
    def gen_rect_voi_array(shape):
        """
        Generates a rectangular array VOI
        :param shape: 3-tuple with the length of the three rectangle sides
        :return: a binary ndarray object
        """
        seg = np.zeros(shape=np.asarray(shape) + 1, dtype=np.bool)
        seg[1:shape[0], 1:shape[1], 1:shape[2]] = True
        return seg

    @staticmethod
    def gen_shell_voi_array(shape, center, rad, thick):
        """
        Generates a spherical shell-like array VOI
        :param shape: 3-tuple with the length of the three rectangle sides
        :param center: sphere center
        :param rad: sphere radius
        :param thick: shell thickness
        :return: a binary ndarray object
        """
        [Y, X, Z] = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
        hold_x, hold_y, hold_z = X - center[0], Y - center[1], Z - center[2]
        dst_field_sq = hold_x*hold_x + hold_y*hold_y + hold_z*hold_z
        rad_1, rad_2 = rad - .5 * thick, rad + .5 * thick
        return (dst_field_sq >= rad_1**2) & (dst_field_sq <= rad_2**2)

    @staticmethod
    def gen_shell_coords(n_coords, center, rad, g_sd, voi):
        """
        Generates a list with the coordinates placed on 6 poles of a sphere surface.
        For each pole, the coordinates are distributed as a Gaussian around the pole.
        :param n_coords: number of coordinates to generate
        :param center: sphere center
        :param rad: sphere radius
        :param g_sd: standard deviation for the Gaussian distributions at the poles
        :param voi: if not None (default) only those inside the binary voi are instered
        :return: a list of coordinates, it will try to have the length equal to n_coords
        """

        # Initialization
        coords = np.zeros(shape=(n_coords, 3), dtype=np.float32)
        poles = [(0, 0),
                 (90, 0), (90, 90), (90, 180), (90, 270),
                 (180, 0)]
        # poles = [(90, 0), (90, 180)]

        # Generate the preliminary coordinates
        langs = list()
        for pole in poles:
            langs.append(np.random.normal(pole, g_sd, size=(n_coords, 2)))
        langs = np.concatenate(langs)
        rnd_ids = np.random.randint(0, len(langs), size=len(langs))
        langs = langs[rnd_ids]

        # Coordinates loop
        n_added = 0
        for i, lang in enumerate(langs):
            rho, phi = np.radians(lang)
            x = rad * math.sin(rho) * math.cos(phi)
            y = rad * math.sin(rho) * math.sin(phi)
            z = rad * math.cos(rho)
            x, y, z = center[0] + x, center[1] + y, center[2] + z
            x, y, z = int(round(x)), int(round(y)), int(round(z))
            if (x >= 0) and (y >= 0) and (z >= 0) and (x < voi.shape[0]) and (y < voi.shape[1]) and (z < voi.shape[2]):
                coords[n_added, :] = (x, y, z)
                n_added += 1
            if n_added >= n_coords:
                return coords
        return coords

    def gen_tlist_coords(self, lcoords, voi, part, mode_emb='full'):
        """
        Generates an instance of a ListTomoParticles for testing with already generated patterns
        :param lcoords: list of arrays with the coordinates for the different tomograms
        :param voi: VOI
        :param part: particle shape
        :param mode_emb: model for embedding, valid: 'full' (default) and 'center'
        :return:
        """

        # Initialization
        ltomos = ListTomoParticles()

        # Serial instance generation
        for tomo_id, coords in enumerate(lcoords):

            # Initialization
            tomo = TomoParticles('tomo_coords_' + str(tomo_id), -1, voi)

            # Generations loop
            for coord in coords:

                # Particle construction
                try:
                    hold_part = Particle(part, center=(0, 0, 0))
                except Exception:
                    continue

                # Random rigid body transformation
                hold_part.translation(coord[0], coord[1], coord[2])

                # Checking embedding and no overlapping
                try:
                    tomo.insert_particle(hold_part, check_bounds=True, mode=mode_emb, check_inter=True)
                except Exception:
                    continue

            ltomos.add_tomo(tomo)

        return ltomos

    def test_compute_uni_2nd_mcs(self):

        # return

        # Initialization
        voi = self.gen_rect_voi_surface(TOMO_SHAPE)
        # voi_arr = self.gen_rect_voi_array(TOMO_SHAPE)
        part = disperse_io.load_poly(PARTICLE_SURF)

        # Generate models
        model_csrv, model_sprv = ModelCSRV(), ModelSRPV(n_cycles=N_CYCLES_TOMO, sin_t=SIN_T)

        # Generate instances
        out_temp = OUT_DIR + '/temp_mcs'
        clean_dir(out_temp)
        os.makedirs(out_temp)
        hold_time = time.time()
        ltomos_csrv = gen_tlist(N_TOMOS, N_PART_TOMO, model_csrv, voi, PARTICLE_SURF, mode_emb='center',
                                npr=N_PROCESSORS, tmp_folder=out_temp)
        ltomos_sprv = gen_tlist(N_TOMOS, N_PART_TOMO, model_sprv, voi, PARTICLE_SURF, mode_emb='center',
                                npr=N_PROCESSORS, tmp_folder=out_temp)
        sprv_den = ltomos_sprv.get_tomo_list()[0].compute_global_density()
        print('Time to generate the tomos: ' + str(time.time() - hold_time))

        # Storing tomograms list
        ltomos_csrv.pickle(OUT_DIR + '/tomos_csrv_mcs_tpl.pkl')
        ltomos_sprv.pickle(OUT_DIR + '/tomos_srpv_mcs_tpl.pkl')
        ltomos_csrv.store_appended_tomos(OUT_DIR, out_stem='csrv_mcs_', mode='surface')
        ltomos_sprv.store_appended_tomos(OUT_DIR, out_stem='srpv_mcs_', mode='surface')

        # Compute the matrices
        hold_time = time.time()
        csrv_exp_mat = np.zeros(shape=(N_TOMOS, len(RAD_RG)), dtype=np.float32)
        for i, tkey in enumerate(ltomos_csrv.get_tomo_fname_list()):
            hold_tomo = ltomos_csrv.get_tomo_by_key(tkey)
            hold_exp = hold_tomo.compute_uni_2nd_order(RAD_RG, thick=None, border=BORDER,
                                                       conv_iter=CONV_ITER, max_iter=MAX_ITER,
                                                       npr=N_PROCESSORS, tmp_folder=out_temp)
            csrv_exp_mat[i, :] = hold_exp
        csrv_time = time.time() - hold_time

        hold_time = time.time()
        sprv_exp_mat = np.zeros(shape=(N_TOMOS, len(RAD_RG)), dtype=np.float32)
        for i, tkey in enumerate(ltomos_sprv.get_tomo_fname_list()):
            hold_tomo = ltomos_sprv.get_tomo_by_key(tkey)
            hold_exp = hold_tomo.compute_uni_2nd_order(RAD_RG, thick=None, border=BORDER,
                                                       conv_iter=CONV_ITER, max_iter=MAX_ITER,
                                                       npr=N_PROCESSORS, tmp_folder=out_temp)
            sprv_exp_mat[i, :] = hold_exp
        sprv_time = time.time() - hold_time
        hold_time = time.time()
        sprv_exp_shell_mat = np.zeros(shape=(N_TOMOS, len(RAD_RG)), dtype=np.float32)
        for i, tkey in enumerate(ltomos_sprv.get_tomo_fname_list()):
            hold_tomo = ltomos_sprv.get_tomo_by_key(tkey)
            hold_exp = hold_tomo.compute_uni_2nd_order(RAD_RG, thick=SHELL_THICK, border=BORDER,
                                                       conv_iter=CONV_ITER, max_iter=MAX_ITER,
                                                       npr=N_PROCESSORS, tmp_folder=out_temp)
            sprv_exp_shell_mat[i, :] = hold_exp
        sprv_shell_time = time.time() - hold_time

        # Compute the simulations
        hold_time = time.time()
        csrv_sim_mat = np.zeros(shape=(N_SIMS * N_TOMOS, len(RAD_RG)), dtype=np.float32)
        for i, tkey in enumerate(ltomos_csrv.get_tomo_fname_list()):
            hold_tomo = ltomos_csrv.get_tomo_by_key(tkey)
            hold_sim = hold_tomo.simulate_uni_2nd_order(N_SIMS, model_csrv, part, 'center', RAD_RG, thick=None,
                                                        border=BORDER, conv_iter=CONV_ITER, max_iter=MAX_ITER,
                                                        npr=N_PROCESSORS, tmp_folder=out_temp)
            start_id = i * N_SIMS
            end_id = start_id + N_SIMS
            csrv_sim_mat[start_id:end_id] = hold_sim
        csrv_sim_time = time.time() - hold_time
        hold_time = time.time()
        csrv_sim_shell_mat = np.zeros(shape=(N_SIMS * N_TOMOS, len(RAD_RG)), dtype=np.float32)
        for i, tkey in enumerate(ltomos_csrv.get_tomo_fname_list()):
            hold_tomo = ltomos_csrv.get_tomo_by_key(tkey)
            hold_sim = hold_tomo.simulate_uni_2nd_order(N_SIMS, model_csrv, part, 'center', RAD_RG, thick=SHELL_THICK,
                                                        border=BORDER, conv_iter=CONV_ITER, max_iter=MAX_ITER,
                                                        npr=N_PROCESSORS, tmp_folder=out_temp)
            start_id = i * N_SIMS
            end_id = start_id + N_SIMS
            csrv_sim_shell_mat[start_id:end_id] = hold_sim
        csrv_sim_shell_time = time.time() - hold_time

        # Compute statistics

        csrv_exp = np.percentile(csrv_exp_mat, 50, axis=0)
        sprv_exp = np.percentile(sprv_exp_mat, 50, axis=0)
        sprv_exp_shell = np.percentile(sprv_exp_shell_mat, 50, axis=0)
        csrv_high = np.percentile(csrv_sim_mat, 100 - PERCENTILE, axis=0)
        csrv_med = np.percentile(csrv_sim_mat, 50, axis=0)
        csrv_low = np.percentile(csrv_sim_mat, PERCENTILE, axis=0)
        csrv_high_shell = np.percentile(csrv_sim_shell_mat, 100 - PERCENTILE, axis=0)
        csrv_med_shell = np.percentile(csrv_sim_shell_mat, 50, axis=0)
        csrv_low_shell = np.percentile(csrv_sim_shell_mat, PERCENTILE, axis=0)

        # Plotting
        plt.figure()
        plt.title('Model CSRV')
        plt.ylabel('Ripley\'s L')
        plt.xlabel('Scale')
        plt.fill_between(RAD_RG, csrv_low, csrv_high, alpha=.5, color='gray', edgecolor='w')
        plt.plot(RAD_RG, np.zeros(shape=len(RAD_RG)), 'k--', linewidth=1)
        plt.plot(RAD_RG, csrv_low, linewidth=2, color='k', linestyle='--')
        plt.plot(RAD_RG, csrv_med, linewidth=2, color='k')
        plt.plot(RAD_RG, csrv_high, linewidth=2, color='k', linestyle='--')
        plt.plot(RAD_RG, csrv_exp, linewidth=2, color='b')
        plt.xlim(RAD_RG.min(), RAD_RG.max())
        # plt.ylim(-10, 10)
        plt.tight_layout()
        # plt.show(block=True)
        plt.savefig(OUT_DIR + '/mcs_csrv_L_' + str(N_PART_TOMO) + '_' + str(N_PROCESSORS) + '_' +
                    str(N_PROCESSORS) + '.png')
        plt.close()
        plt.figure()
        plt.title('Model SRPV')
        plt.ylabel('Ripley\'s L')
        plt.xlabel('Scale')
        plt.fill_between(RAD_RG, csrv_low, csrv_high, alpha=.5, color='gray', edgecolor='w')
        plt.plot(RAD_RG, np.zeros(shape=len(RAD_RG)), 'k--', linewidth=1)
        plt.plot(RAD_RG, csrv_low, linewidth=2, color='k', linestyle='--')
        plt.plot(RAD_RG, csrv_med, linewidth=2, color='k')
        plt.plot(RAD_RG, csrv_high, linewidth=2, color='k', linestyle='--')
        plt.plot(RAD_RG, sprv_exp, linewidth=2, color='b')
        plt.xlim(RAD_RG.min(), RAD_RG.max())
        idx = np.argmax(np.asarray(N_CYCLES_TOMO, dtype=float))
        ds = 2 * (TOMO_SHAPE[idx] / (float(N_CYCLES_TOMO[idx]) * np.pi)) * np.arccos(SIN_T)
        plt.gca().axvline(x=.5 * ds, linewidth=2, linestyle='-.', color='k')
        dd = TOMO_SHAPE[idx] / float(N_CYCLES_TOMO[idx])
        plt.gca().axvline(x=dd, linewidth=2, linestyle='-.', color='k')
        plt.tight_layout()
        # plt.show(block=True)
        plt.savefig(OUT_DIR + '/mcs_sprv_L_' + str(N_PART_TOMO) + '_' + str(N_PROCESSORS) + '_' +
                    str(N_PROCESSORS) + '.png')
        plt.close()
        plt.close()
        # plt.figure()
        fig, (ax, ax2) = plt.subplots(2, 1, sharex=True)
        # plt.title('Model SRPV')
        # plt.ylabel('Ripley\'s O')
        # plt.xlabel('Scale')
        fig.suptitle('Model SRPV')
        ax.set_ylabel('Ripley\'s O')
        ax2.set_ylabel('Ripley\'s O')
        plt.xlabel('Scale')
        # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.set_ylim(1.25e-4, 2e-4)
        ax2.set_ylim(0, .25e-4)
        # hide the spines between ax and ax2
        ax.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax.xaxis.tick_top()
        ax.tick_params(labeltop='off')  # don't put tick labels at the top
        ax2.xaxis.tick_bottom()
        d = .015  # how big to make the diagonal lines in axes coordinates
        # arguments to pass to plot, just so we don't keep repeating them
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
        ax.plot((+d, -d), (-d, +d), **kwargs)  # top-left diagonal
        ax.plot((1 + d, 1 - d), (-d, +d), **kwargs)  # top-right diagonal
        kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
        ax2.plot((+d, -d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
        ax2.plot((1 + d, 1 - d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
        # plt.fill_between(RAD_RG, csrv_shell_low, csrv_shell_high, alpha=.5, color='gray', edgecolor='w')
        # plt.plot((RAD_RG.min(), RAD_RG.max()), (sprv_den, sprv_den), 'k--', linewidth=1)
        # plt.plot(RAD_RG, csrv_shell_low, linewidth=2, color='k', linestyle='--')
        # plt.plot(RAD_RG, csrv_shell_med, linewidth=2, color='k')
        # plt.plot(RAD_RG, csrv_shell_high, linewidth=2, color='k', linestyle='--')
        # plt.plot(RAD_RG, sprv_shell_exp, linewidth=2, color='b')
        ax.fill_between(RAD_RG, csrv_low_shell, csrv_high_shell, alpha=.5, color='gray', edgecolor='w')
        ax.plot((RAD_RG.min(), RAD_RG.max()), (sprv_den, sprv_den), 'k--', linewidth=1)
        ax.plot(RAD_RG, csrv_low_shell, linewidth=2, color='k', linestyle='--')
        ax.plot(RAD_RG, csrv_med_shell, linewidth=2, color='k')
        ax.plot(RAD_RG, csrv_high_shell, linewidth=2, color='k', linestyle='--')
        ax.plot(RAD_RG, sprv_exp_shell, linewidth=2, color='b')
        ax2.fill_between(RAD_RG, csrv_low_shell, csrv_high_shell, alpha=.5, color='gray', edgecolor='w')
        ax2.plot((RAD_RG.min(), RAD_RG.max()), (sprv_den, sprv_den), 'k--', linewidth=1)
        ax2.plot(RAD_RG, csrv_low_shell, linewidth=2, color='k', linestyle='--')
        ax2.plot(RAD_RG, csrv_med_shell, linewidth=2, color='k')
        ax2.plot(RAD_RG, csrv_high_shell, linewidth=2, color='k', linestyle='--')
        ax2.plot(RAD_RG, sprv_exp_shell, linewidth=2, color='b')
        # plt.xlim(RAD_RG.min(), RAD_RG.max())
        plt.xlim(RAD_RG.min(), RAD_RG.max())
        plt.xlim(RAD_RG.min(), RAD_RG.max())
        plt.gca().axvline(x=ds, linewidth=2, linestyle='-.', color='k')
        plt.gca().axvline(x=dd, linewidth=2, linestyle='-.', color='k')
        plt.tight_layout()
        # plt.show(block=True)
        plt.savefig(OUT_DIR + '/mcs_sprv_shell_O_' + str(N_PART_TOMO) + '_' + str(N_PROCESSORS) + '_' +
                    str(N_PROCESSORS) + '.png')
        plt.close()

        # Plotting times
        plt.figure()
        plt.title('Averaged time per computing unit')
        plt.ylabel('Time [secs]')
        plt.bar(1, csrv_time / (N_TOMOS*N_PART_TOMO), BAR_WIDTH, color='blue', linewidth=2)
        plt.bar(2, sprv_time / (N_TOMOS*N_PART_TOMO), BAR_WIDTH, color='blue', linewidth=2)
        plt.bar(3, sprv_shell_time / (N_TOMOS*N_PART_TOMO), BAR_WIDTH, color='blue', linewidth=2)
        plt.bar(4, csrv_sim_time / (N_SIMS * N_TOMOS * N_PART_TOMO), BAR_WIDTH, color='blue', linewidth=2)
        plt.bar(5, csrv_sim_shell_time / (N_SIMS * N_TOMOS * N_PART_TOMO), BAR_WIDTH, color='blue', linewidth=2)
        plt.xticks((1, 2, 3, 4, 5), ('CSRV', 'SPRV', 'SPRV_SHELL', 'CSRV\'', 'SPRV_SHELL\''))
        plt.xlim(0.5, 5.5)
        plt.tight_layout()
        # plt.show(block=True)
        plt.savefig(OUT_DIR + '/mcs_time_' + str(N_PART_TOMO) + '_' + str(N_PROCESSORS) + '_' +
                    str(N_PROCESSORS) + '.png')
        plt.close()

    def test_compute_uni_2nd_mca(self):

        # return

        # Initialization
        voi = self.gen_rect_voi_array(TOMO_SHAPE)
        part = disperse_io.load_poly(PARTICLE_SURF)
        disperse_io.save_numpy(voi, OUT_DIR + '/voi_mca.mrc')

        # Generate models
        model_csrv, model_sprv = ModelCSRV(), ModelSRPV(n_cycles=N_CYCLES_TOMO, sin_t=SIN_T)

        # Generate instances
        hold_time = time.time()
        ltomos_csrv = gen_tlist(N_TOMOS, N_PART_TOMO, model_csrv, voi, PARTICLE_SURF, mode_emb='center',
                                npr=N_PROCESSORS)
        ltomos_sprv = gen_tlist(N_TOMOS, N_PART_TOMO, model_sprv, voi, PARTICLE_SURF, mode_emb='center',
                                npr=N_PROCESSORS)
        print('Time to generate the tomos: ' + str(time.time() - hold_time))

        # Storing tomograms list
        ltomos_csrv.pickle(OUT_DIR + '/tomos_csrv_mca_tpl.pkl')
        ltomos_sprv.pickle(OUT_DIR + '/tomos_srpv_mca_tpl.pkl')
        ltomos_csrv.store_appended_tomos(OUT_DIR, out_stem='csrv_mca_', mode='surface')
        ltomos_sprv.store_appended_tomos(OUT_DIR, out_stem='srpv_mca_', mode='surface')

        # Compute the matrices
        hold_time = time.time()
        csrv_exp_mat = np.zeros(shape=(N_TOMOS, len(RAD_RG)), dtype=np.float32)
        for i, tkey in enumerate(ltomos_csrv.get_tomo_fname_list()):
            hold_tomo = ltomos_csrv.get_tomo_by_key(tkey)
            hold_exp = hold_tomo.compute_uni_2nd_order(RAD_RG, thick=None, border=BORDER,
                                                       conv_iter=CONV_ITER, max_iter=MAX_ITER,
                                                       npr=N_PROCESSORS)
            csrv_exp_mat[i, :] = hold_exp
        csrv_time = time.time() - hold_time
        hold_time = time.time()
        sprv_exp_mat = np.zeros(shape=(N_TOMOS, len(RAD_RG)), dtype=np.float32)
        for i, tkey in enumerate(ltomos_sprv.get_tomo_fname_list()):
            hold_tomo = ltomos_sprv.get_tomo_by_key(tkey)
            hold_exp = hold_tomo.compute_uni_2nd_order(RAD_RG, thick=None, border=BORDER,
                                                       conv_iter=CONV_ITER, max_iter=MAX_ITER,
                                                       npr=N_PROCESSORS)
            sprv_exp_mat[i, :] = hold_exp
        sprv_time = time.time() - hold_time
        hold_time = time.time()
        sprv_exp_shell_mat = np.zeros(shape=(N_TOMOS, len(RAD_RG)), dtype=np.float32)
        for i, tkey in enumerate(ltomos_sprv.get_tomo_fname_list()):
            hold_tomo = ltomos_sprv.get_tomo_by_key(tkey)
            hold_exp = hold_tomo.compute_uni_2nd_order(RAD_RG, thick=SHELL_THICK, border=BORDER,
                                                       conv_iter=CONV_ITER, max_iter=MAX_ITER,
                                                       npr=N_PROCESSORS)
            sprv_exp_shell_mat[i, :] = hold_exp
        sprv_time_shell = time.time() - hold_time

        # Compute the simulations
        hold_time = time.time()
        csrv_sim_mat = np.zeros(shape=(N_SIMS * N_TOMOS, len(RAD_RG)), dtype=np.float32)
        for i, tkey in enumerate(ltomos_csrv.get_tomo_fname_list()):
            hold_tomo = ltomos_csrv.get_tomo_by_key(tkey)
            hold_sim = hold_tomo.simulate_uni_2nd_order(N_SIMS, model_csrv, part, 'center', RAD_RG, thick=None,
                                                        border=BORDER, conv_iter=CONV_ITER, max_iter=MAX_ITER,
                                                        npr=N_PROCESSORS)
            start_id = i * N_SIMS
            end_id = start_id + N_SIMS
            csrv_sim_mat[start_id:end_id] = hold_sim
        csrv_sim_time = time.time() - hold_time
        hold_time = time.time()
        csrv_sim_shell_mat = np.zeros(shape=(N_SIMS * N_TOMOS, len(RAD_RG)), dtype=np.float32)
        for i, tkey in enumerate(ltomos_csrv.get_tomo_fname_list()):
            hold_tomo = ltomos_csrv.get_tomo_by_key(tkey)
            hold_sim = hold_tomo.simulate_uni_2nd_order(N_SIMS, model_csrv, part, 'center', RAD_RG, thick=SHELL_THICK,
                                                        border=BORDER, conv_iter=CONV_ITER, max_iter=MAX_ITER,
                                                        npr=N_PROCESSORS)
            start_id = i * N_SIMS
            end_id = start_id + N_SIMS
            csrv_sim_shell_mat[start_id:end_id] = hold_sim
        csrv_sim_shell_time = time.time() - hold_time

        # Compute statistics
        csrv_exp = np.percentile(csrv_exp_mat, 50, axis=0)
        sprv_exp = np.percentile(sprv_exp_mat, 50, axis=0)
        sprv_exp_shell = np.percentile(sprv_exp_shell_mat, 50, axis=0)
        csrv_high = np.percentile(csrv_sim_mat, 100 - PERCENTILE, axis=0)
        csrv_med = np.percentile(csrv_sim_mat, 50, axis=0)
        csrv_low = np.percentile(csrv_sim_mat, PERCENTILE, axis=0)
        csrv_high_shell = np.percentile(csrv_sim_shell_mat, 100 - PERCENTILE, axis=0)
        csrv_med_shell = np.percentile(csrv_sim_shell_mat, 50, axis=0)
        csrv_low_shell = np.percentile(csrv_sim_shell_mat, PERCENTILE, axis=0)

        # Plotting
        plt.figure()
        plt.title('Model CSRV')
        plt.ylabel('Ripley\'s L')
        plt.xlabel('Scale')
        plt.fill_between(RAD_RG, csrv_low, csrv_high, alpha=.5, color='gray', edgecolor='w')
        plt.plot(RAD_RG, np.zeros(shape=len(RAD_RG)), 'k--', linewidth=1)
        plt.plot(RAD_RG, csrv_low, linewidth=2, color='k', linestyle='--')
        plt.plot(RAD_RG, csrv_med, linewidth=2, color='k')
        plt.plot(RAD_RG, csrv_high, linewidth=2, color='k', linestyle='--')
        plt.plot(RAD_RG, csrv_exp, linewidth=2, color='b')
        plt.xlim(RAD_RG.min(), RAD_RG.max())
        # plt.ylim(-10, 10)
        plt.tight_layout()
        # plt.show(block=True)
        plt.savefig(OUT_DIR + '/mca_csrv_L_' + str(N_PART_TOMO) + '_' + str(N_PROCESSORS) + '_' +
                    str(N_PROCESSORS) + '.png')
        plt.close()
        plt.figure()
        plt.title('Model SRPV')
        plt.ylabel('Ripley\'s L')
        plt.xlabel('Scale')
        plt.fill_between(RAD_RG, csrv_low, csrv_high, alpha=.5, color='gray', edgecolor='w')
        plt.plot(RAD_RG, np.zeros(shape=len(RAD_RG)), 'k--', linewidth=1)
        plt.plot(RAD_RG, csrv_low, linewidth=2, color='k', linestyle='--')
        plt.plot(RAD_RG, csrv_med, linewidth=2, color='k')
        plt.plot(RAD_RG, csrv_high, linewidth=2, color='k', linestyle='--')
        plt.plot(RAD_RG, sprv_exp, linewidth=2, color='b')
        plt.xlim(RAD_RG.min(), RAD_RG.max())
        plt.tight_layout()
        # plt.show(block=True)
        plt.savefig(OUT_DIR + '/mca_sprv_L_' + str(N_PART_TOMO) + '_' + str(N_PROCESSORS) + '_' +
                    str(N_PROCESSORS) + '.png')

        plt.close()
        plt.figure()
        plt.title('Model SRPV')
        plt.ylabel('Ripley\'s L')
        plt.xlabel('Scale')
        plt.fill_between(RAD_RG, csrv_low_shell, csrv_high_shell, alpha=.5, color='gray', edgecolor='w')
        plt.plot(RAD_RG, np.zeros(shape=len(RAD_RG)), 'k--', linewidth=1)
        plt.plot(RAD_RG, csrv_low_shell, linewidth=2, color='k', linestyle='--')
        plt.plot(RAD_RG, csrv_med_shell, linewidth=2, color='k')
        plt.plot(RAD_RG, csrv_high_shell, linewidth=2, color='k', linestyle='--')
        plt.plot(RAD_RG, sprv_exp_shell, linewidth=2, color='b')
        plt.xlim(RAD_RG.min(), RAD_RG.max())
        plt.tight_layout()
        # plt.show(block=True)
        plt.savefig(OUT_DIR + '/mca_sprv_O_' + str(N_PART_TOMO) + '_' + str(N_PROCESSORS) + '_' +
                    str(N_PROCESSORS) + '.png')

        # Plotting times
        plt.figure()
        plt.title('Time for computing and simulating Ripley\'s L')
        plt.ylabel('Time (seconds)')
        plt.bar(1, csrv_time / N_TOMOS, BAR_WIDTH, color='blue', linewidth=2)
        plt.bar(2, sprv_time / N_TOMOS, BAR_WIDTH, color='blue', linewidth=2)
        plt.bar(3, csrv_sim_time / (N_SIMS * N_TOMOS), BAR_WIDTH, color='blue', linewidth=2)
        plt.bar(4, csrv_sim_shell_time / (N_SIMS * N_TOMOS), BAR_WIDTH, color='blue', linewidth=2)
        plt.xticks((1, 2, 3, 4), ('CSRV', 'SPRV', 'CSRV\'', 'SPRV\''))
        plt.xlim(0.5, 4.5)
        plt.tight_layout()
        # plt.show(block=True)
        plt.savefig(OUT_DIR + '/mca_time_L_' + str(N_PART_TOMO) + '_' + str(N_PROCESSORS) + '_' +
                    str(N_PROCESSORS) + '.png')
        plt.close()

    def test_compute_uni_2nd_dsa(self):

        return

        # Initialization
        voi = self.gen_rect_voi_array(TOMO_SHAPE)
        part = disperse_io.load_poly(PARTICLE_SURF)

        # Generate models
        model_csrv, model_sprv = ModelCSRV(), ModelSRPV(n_cycles=N_CYCLES_TOMO, sin_t=SIN_T)

        # Generate instances
        hold_time = time.time()
        ltomos_csrv = gen_tlist(N_TOMOS, N_PART_TOMO, model_csrv, voi, PARTICLE_SURF, mode_emb='center',
                                npr=N_PROCESSORS)
        ltomos_sprv = gen_tlist(N_TOMOS, N_PART_TOMO, model_sprv, voi, PARTICLE_SURF, mode_emb='center',
                                npr=N_PROCESSORS)
        sprv_den = ltomos_sprv.get_tomo_list()[0].compute_global_density()
        print('Time to generate the tomos: ' + str(time.time() - hold_time))

        # Storing tomograms list
        ltomos_csrv.pickle(OUT_DIR + '/tomos_csrv_dsa_tpl.pkl')
        ltomos_sprv.pickle(OUT_DIR + '/tomos_srpv_mca_tpl.pkl')
        ltomos_csrv.store_appended_tomos(OUT_DIR, out_stem='csrv_dsa_', mode='surface')
        ltomos_sprv.store_appended_tomos(OUT_DIR, out_stem='srpv_dsa_', mode='surface')

        # Compute the matrices
        hold_time = time.time()
        csrv_exp_mat = np.zeros(shape=(N_TOMOS, len(RAD_RG)), dtype=np.float32)
        for i, tkey in enumerate(ltomos_csrv.get_tomo_fname_list()):
            hold_tomo = ltomos_csrv.get_tomo_by_key(tkey)
            hold_exp = hold_tomo.compute_uni_2nd_order(RAD_RG, thick=None, border=BORDER,
                                                       conv_iter=None, max_iter=None,
                                                       npr=N_PROCESSORS)
            csrv_exp_mat[i, :] = hold_exp
        csrv_time = time.time() - hold_time
        hold_time = time.time()
        sprv_exp_mat = np.zeros(shape=(N_TOMOS, len(RAD_RG)), dtype=np.float32)
        for i, tkey in enumerate(ltomos_sprv.get_tomo_fname_list()):
            hold_tomo = ltomos_sprv.get_tomo_by_key(tkey)
            hold_exp = hold_tomo.compute_uni_2nd_order(RAD_RG, thick=None, border=BORDER,
                                                       conv_iter=None, max_iter=None,
                                                       npr=N_PROCESSORS)
            sprv_exp_mat[i, :] = hold_exp
        sprv_time = time.time() - hold_time
        hold_time = time.time()
        sprv_exp_shell_mat = np.zeros(shape=(N_TOMOS, len(RAD_RG)), dtype=np.float32)
        for i, tkey in enumerate(ltomos_sprv.get_tomo_fname_list()):
            hold_tomo = ltomos_sprv.get_tomo_by_key(tkey)
            hold_exp = hold_tomo.compute_uni_2nd_order(RAD_RG, thick=SHELL_THICK, border=BORDER,
                                                       conv_iter=None, max_iter=None,
                                                       npr=N_PROCESSORS)
            sprv_exp_shell_mat[i, :] = hold_exp
        sprv_shell_time = time.time() - hold_time

        # Compute the simulations
        hold_time = time.time()
        csrv_sim_mat = np.zeros(shape=(N_SIMS * N_TOMOS, len(RAD_RG)), dtype=np.float32)
        for i, tkey in enumerate(ltomos_csrv.get_tomo_fname_list()):
            hold_tomo = ltomos_csrv.get_tomo_by_key(tkey)
            hold_sim = hold_tomo.simulate_uni_2nd_order(N_SIMS, model_csrv, part, 'center', RAD_RG, thick=None,
                                                        border=BORDER, conv_iter=None, max_iter=None,
                                                        npr=N_PROCESSORS)
            start_id = i * N_SIMS
            end_id = start_id + N_SIMS
            csrv_sim_mat[start_id:end_id] = hold_sim
        csrv_sim_time = time.time() - hold_time
        hold_time = time.time()
        csrv_sim_shell_mat = np.zeros(shape=(N_SIMS * N_TOMOS, len(RAD_RG)), dtype=np.float32)
        for i, tkey in enumerate(ltomos_csrv.get_tomo_fname_list()):
            hold_tomo = ltomos_csrv.get_tomo_by_key(tkey)
            hold_sim = hold_tomo.simulate_uni_2nd_order(N_SIMS, model_csrv, part, 'center', RAD_RG, thick=SHELL_THICK,
                                                        border=BORDER, conv_iter=None, max_iter=None,
                                                        npr=N_PROCESSORS)
            start_id = i * N_SIMS
            end_id = start_id + N_SIMS
            csrv_sim_shell_mat[start_id:end_id] = hold_sim
        csrv_sim_shell_time = time.time() - hold_time

        # Compute statistics
        csrv_exp = np.percentile(csrv_exp_mat, 50, axis=0)
        sprv_exp = np.percentile(sprv_exp_mat, 50, axis=0)
        sprv_shell_exp = np.percentile(sprv_exp_shell_mat, 50, axis=0)
        csrv_high = np.percentile(csrv_sim_mat, 100-PERCENTILE, axis=0)
        csrv_med = np.percentile(csrv_sim_mat, 50, axis=0)
        csrv_low = np.percentile(csrv_sim_mat, PERCENTILE, axis=0)
        csrv_shell_high = np.percentile(csrv_sim_shell_mat, 100 - PERCENTILE, axis=0)
        csrv_shell_med = np.percentile(csrv_sim_shell_mat, 50, axis=0)
        csrv_shell_low = np.percentile(csrv_sim_shell_mat, PERCENTILE, axis=0)

        # Plotting
        plt.figure()
        plt.title('Model CSRV')
        plt.ylabel('Ripley\'s L')
        plt.xlabel('Scale')
        plt.fill_between(RAD_RG, csrv_low, csrv_high, alpha=.5, color='gray', edgecolor='w')
        plt.plot(RAD_RG, np.zeros(shape=len(RAD_RG)), 'k--', linewidth=1)
        plt.plot(RAD_RG, csrv_low, linewidth=2, color='k', linestyle='--')
        plt.plot(RAD_RG, csrv_med, linewidth=2, color='k')
        plt.plot(RAD_RG, csrv_high, linewidth=2, color='k', linestyle='--')
        plt.plot(RAD_RG, csrv_exp, linewidth=2, color='b')
        plt.xlim(RAD_RG.min(), RAD_RG.max())
        # plt.ylim(-10, 10)
        plt.tight_layout()
        # plt.show(block=True)
        plt.savefig(OUT_DIR + '/dsa_csrv_L_' + str(N_PART_TOMO) + '_' + str(N_PROCESSORS) + '_' +
                    str(N_PROCESSORS) + '.png')
        plt.close()
        plt.figure()
        plt.title('Model SRPV')
        plt.ylabel('Ripley\'s L')
        plt.xlabel('Scale')
        plt.fill_between(RAD_RG, csrv_low, csrv_high, alpha=.5, color='gray', edgecolor='w')
        plt.plot(RAD_RG, np.zeros(shape=len(RAD_RG)), 'k--', linewidth=1)
        plt.plot(RAD_RG, csrv_low, linewidth=2, color='k', linestyle='--')
        plt.plot(RAD_RG, csrv_med, linewidth=2, color='k')
        plt.plot(RAD_RG, csrv_high, linewidth=2, color='k', linestyle='--')
        plt.plot(RAD_RG, sprv_exp, linewidth=2, color='b')
        plt.xlim(RAD_RG.min(), RAD_RG.max())
        idx = np.argmax(np.asarray(N_CYCLES_TOMO, dtype=float))
        ds = 2 * (TOMO_SHAPE[idx] / (float(N_CYCLES_TOMO[idx]) * np.pi)) * np.arccos(SIN_T)
        plt.gca().axvline(x=.5 * ds, linewidth=2, linestyle='-.', color='k')
        dd = TOMO_SHAPE[idx] / float(N_CYCLES_TOMO[idx])
        plt.gca().axvline(x=dd, linewidth=2, linestyle='-.', color='k')
        plt.tight_layout()
        # plt.show(block=True)
        plt.savefig(OUT_DIR + '/dsa_sprv_L_' + str(N_PART_TOMO) + '_' + str(N_PROCESSORS) + '_' +
                    str(N_PROCESSORS) + '.png')
        plt.close()
        # plt.figure()
        fig, (ax, ax2) = plt.subplots(2, 1, sharex=True)
        # plt.title('Model SRPV')
        # plt.ylabel('Ripley\'s O')
        # plt.xlabel('Scale')
        fig.suptitle('Model SRPV')
        ax.set_ylabel('Ripley\'s O')
        ax2.set_ylabel('Ripley\'s O')
        plt.xlabel('Scale')
        # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.set_ylim(1.25e-4, 2e-4)
        ax2.set_ylim(0, .25e-4)
        # hide the spines between ax and ax2
        ax.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax.xaxis.tick_top()
        ax.tick_params(labeltop='off')  # don't put tick labels at the top
        ax2.xaxis.tick_bottom()
        d = .015  # how big to make the diagonal lines in axes coordinates
        # arguments to pass to plot, just so we don't keep repeating them
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
        ax.plot((+d, -d), (-d, +d), **kwargs)  # top-left diagonal
        ax.plot((1 + d, 1 - d), (-d, +d), **kwargs)  # top-right diagonal
        kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
        ax2.plot((+d, -d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
        ax2.plot((1 + d, 1 - d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
        # plt.fill_between(RAD_RG, csrv_shell_low, csrv_shell_high, alpha=.5, color='gray', edgecolor='w')
        # plt.plot((RAD_RG.min(), RAD_RG.max()), (sprv_den, sprv_den), 'k--', linewidth=1)
        # plt.plot(RAD_RG, csrv_shell_low, linewidth=2, color='k', linestyle='--')
        # plt.plot(RAD_RG, csrv_shell_med, linewidth=2, color='k')
        # plt.plot(RAD_RG, csrv_shell_high, linewidth=2, color='k', linestyle='--')
        # plt.plot(RAD_RG, sprv_shell_exp, linewidth=2, color='b')
        ax.fill_between(RAD_RG, csrv_shell_low, csrv_shell_high, alpha=.5, color='gray', edgecolor='w')
        ax.plot((RAD_RG.min(), RAD_RG.max()), (sprv_den, sprv_den), 'k--', linewidth=1)
        ax.plot(RAD_RG, csrv_shell_low, linewidth=2, color='k', linestyle='--')
        ax.plot(RAD_RG, csrv_shell_med, linewidth=2, color='k')
        ax.plot(RAD_RG, csrv_shell_high, linewidth=2, color='k', linestyle='--')
        ax.plot(RAD_RG, sprv_shell_exp, linewidth=2, color='b')
        ax2.fill_between(RAD_RG, csrv_shell_low, csrv_shell_high, alpha=.5, color='gray', edgecolor='w')
        ax2.plot((RAD_RG.min(), RAD_RG.max()), (sprv_den, sprv_den), 'k--', linewidth=1)
        ax2.plot(RAD_RG, csrv_shell_low, linewidth=2, color='k', linestyle='--')
        ax2.plot(RAD_RG, csrv_shell_med, linewidth=2, color='k')
        ax2.plot(RAD_RG, csrv_shell_high, linewidth=2, color='k', linestyle='--')
        ax2.plot(RAD_RG, sprv_shell_exp, linewidth=2, color='b')
        # plt.xlim(RAD_RG.min(), RAD_RG.max())
        plt.xlim(RAD_RG.min(), RAD_RG.max())
        plt.xlim(RAD_RG.min(), RAD_RG.max())
        plt.gca().axvline(x=ds, linewidth=2, linestyle='-.', color='k')
        plt.gca().axvline(x=dd, linewidth=2, linestyle='-.', color='k')
        plt.tight_layout()
        # plt.show(block=True)
        plt.savefig(OUT_DIR + '/dsa_sprv_shell_O_' + str(N_PART_TOMO) + '_' + str(N_PROCESSORS) + '_' +
                    str(N_PROCESSORS) + '.png')
        plt.close()

        # Plotting times
        plt.figure()
        plt.title('Averaged time per computing unit')
        plt.ylabel('Time [secs]')
        plt.bar(1, csrv_time/(N_TOMOS*N_PART_TOMO), BAR_WIDTH, color='blue', linewidth=2)
        plt.bar(2, sprv_time/(N_TOMOS*N_PART_TOMO), BAR_WIDTH, color='blue', linewidth=2)
        plt.bar(3, sprv_shell_time/(N_TOMOS*N_PART_TOMO), BAR_WIDTH, color='blue', linewidth=2)
        plt.bar(4, csrv_sim_time/(N_SIMS*N_TOMOS*N_PART_TOMO), BAR_WIDTH, color='blue', linewidth=2)
        plt.bar(5, csrv_sim_shell_time/(N_SIMS * N_TOMOS * N_PART_TOMO), BAR_WIDTH, color='blue', linewidth=2)
        plt.xticks((1, 2, 3, 4, 5), ('CSRV_L', 'SPRV_L', 'SPRV_O', 'CSRV\'', 'SPRV_O\''))
        plt.xlim(0.5, 5.5)
        plt.tight_layout()
        # plt.show(block=True)
        plt.savefig(OUT_DIR + '/dsa_time_L_O_' + str(N_PART_TOMO) + '_' + str(N_PROCESSORS) + '_' +
                    str(N_PROCESSORS) + '.png')
        plt.close()

    def test_compute_bi_2nd_dsa(self):

        return

        # Initialization
        voi = self.gen_rect_voi_array(TOMO_SHAPE)
        part = disperse_io.load_poly(PARTICLE_SURF)

        # Generate models
        model_csrv, model_c2rv = ModelCSRV(), Model2CCSRV(dst=DST_BI, std=STD_BI)
        model_c2rv.set_voi(voi)
        model_c2rv.set_part(part)

        # Generate instances
        hold_time = time.time()
        ltomos_csrv, ltomos_c2rv = gen_tlist2(N_TOMOS, N_PART_TOMO, model_c2rv, mode_emb='center', npr=1)
        print('Time to generate the tomos: ' + str(time.time() - hold_time))

        # Storing tomograms list
        ltomos_csrv.pickle(OUT_DIR + '/tomos_csrv_bi_tpl.pkl')
        ltomos_c2rv.pickle(OUT_DIR + '/tomos_c2pv_bi_tpl.pkl')
        ltomos_csrv.store_appended_tomos(OUT_DIR, out_stem='csrv_bi_', mode='surface')
        ltomos_c2rv.store_appended_tomos(OUT_DIR, out_stem='c2pv_bi_', mode='surface')

        # Compute the matrices
        hold_time = time.time()
        csrv_exp_mat = np.zeros(shape=(N_TOMOS, len(RAD_RG)), dtype=np.float32)
        for i, tkey in enumerate(ltomos_csrv.get_tomo_fname_list()):
            hold_tomo_1, hold_tomo_2 = ltomos_csrv.get_tomo_by_key(tkey), ltomos_c2rv.get_tomo_by_key(tkey)
            hold_exp = hold_tomo_1.compute_bi_2nd_order(hold_tomo_2, RAD_RG, thick=None, border=BORDER,
                                                        conv_iter=None, max_iter=None,
                                                        npr=N_PROCESSORS)
            csrv_exp_mat[i, :] = hold_exp
        csrv_time = time.time() - hold_time
        hold_time = time.time()
        csrv_exp_fmm_mat = np.zeros(shape=(N_TOMOS, len(RAD_RG)), dtype=np.float32)
        for i, tkey in enumerate(ltomos_csrv.get_tomo_fname_list()):
            hold_tomo_1, hold_tomo_2 = ltomos_csrv.get_tomo_by_key(tkey), ltomos_c2rv.get_tomo_by_key(tkey)
            hold_exp = hold_tomo_1.compute_bi_2nd_order(hold_tomo_2, RAD_RG, thick=None, border=BORDER,
                                                        conv_iter=None, max_iter=None, fmm=True,
                                                        npr=N_PROCESSORS)
            csrv_exp_fmm_mat[i, :] = hold_exp
        csrv_fmm_time = time.time() - hold_time

        # Compute the simulations
        hold_time = time.time()
        csrv_sim_mat = np.zeros(shape=(N_SIMS * N_TOMOS, len(RAD_RG)), dtype=np.float32)
        for i, tkey in enumerate(ltomos_csrv.get_tomo_fname_list()):
            hold_tomo_1, hold_tomo_2 = ltomos_csrv.get_tomo_by_key(tkey), ltomos_c2rv.get_tomo_by_key(tkey)
            hold_sim = hold_tomo_1.simulate_bi_2nd_order(hold_tomo_2, N_SIMS, model_csrv, part, 'center',
                                                         RAD_RG, thick=None, border=BORDER,
                                                         conv_iter=None, max_iter=None,
                                                         npr=N_PROCESSORS)
            start_id = i * N_SIMS
            end_id = start_id + N_SIMS
            csrv_sim_mat[start_id:end_id] = hold_sim
        csrv_sim_time = time.time() - hold_time
        hold_time = time.time()
        csrv_sim_fmm_mat = np.zeros(shape=(N_SIMS * N_TOMOS, len(RAD_RG)), dtype=np.float32)
        for i, tkey in enumerate(ltomos_csrv.get_tomo_fname_list()):
            hold_tomo_1, hold_tomo_2 = ltomos_csrv.get_tomo_by_key(tkey), ltomos_c2rv.get_tomo_by_key(tkey)
            hold_sim = hold_tomo_1.simulate_bi_2nd_order(hold_tomo_2, N_SIMS, model_csrv, part, 'center',
                                                         RAD_RG, thick=None, border=BORDER,
                                                         conv_iter=None, max_iter=None, fmm=True,
                                                         npr=N_PROCESSORS)
            start_id = i * N_SIMS
            end_id = start_id + N_SIMS
            csrv_sim_fmm_mat[start_id:end_id] = hold_sim
        csrv_sim_fmm_time = time.time() - hold_time

        # Compute statistics
        csrv_exp = np.percentile(csrv_exp_mat, 50, axis=0)
        csrv_exp_fmm = np.percentile(csrv_exp_fmm_mat, 50, axis=0)
        csrv_high = np.percentile(csrv_sim_mat, 100-PERCENTILE, axis=0)
        csrv_med = np.percentile(csrv_sim_mat, 50, axis=0)
        csrv_low = np.percentile(csrv_sim_mat, PERCENTILE, axis=0)
        csrv_high_fmm = np.percentile(csrv_sim_fmm_mat, 100 - PERCENTILE, axis=0)
        csrv_med_fmm = np.percentile(csrv_sim_fmm_mat, 50, axis=0)
        csrv_low_fmm = np.percentile(csrv_sim_fmm_mat, PERCENTILE, axis=0)

        # Plotting
        plt.figure()
        plt.title('Model C2RPV')
        plt.ylabel('Ripley\'s L (DT)')
        plt.xlabel('Scale')
        plt.fill_between(RAD_RG, csrv_low, csrv_high, alpha=.5, color='gray', edgecolor='w')
        plt.plot(RAD_RG, np.zeros(shape=len(RAD_RG)), 'k--', linewidth=1)
        plt.plot(RAD_RG, csrv_low, linewidth=2, color='k', linestyle='--')
        plt.plot(RAD_RG, csrv_med, linewidth=2, color='k')
        plt.plot(RAD_RG, csrv_high, linewidth=2, color='k', linestyle='--')
        plt.plot(RAD_RG, csrv_exp, linewidth=2, color='b')
        plt.gca().axvline(x=DST_BI - 3.*STD_BI, linewidth=1, linestyle='-.', color='k')
        plt.gca().axvline(x=DST_BI, linewidth=2, linestyle='-.', color='k')
        plt.gca().axvline(x=DST_BI + 3.*STD_BI, linewidth=1, linestyle='-.', color='k')
        plt.xlim(RAD_RG.min(), RAD_RG.max())
        plt.tight_layout()
        # plt.show(block=True)
        plt.savefig(OUT_DIR + '/bi_c2rv_L_' + str(N_PART_TOMO) + '_' + str(N_PROCESSORS) + '_' +
                    str(N_PROCESSORS) + '.png')
        plt.close()
        plt.figure()
        plt.title('Model C2RPV')
        plt.ylabel('Ripley\'s L (FMM)')
        plt.xlabel('Scale')
        plt.fill_between(RAD_RG, csrv_low_fmm, csrv_high_fmm, alpha=.5, color='gray', edgecolor='w')
        plt.plot(RAD_RG, np.zeros(shape=len(RAD_RG)), 'k--', linewidth=1)
        plt.plot(RAD_RG, csrv_low_fmm, linewidth=2, color='k', linestyle='--')
        plt.plot(RAD_RG, csrv_med_fmm, linewidth=2, color='k')
        plt.plot(RAD_RG, csrv_high_fmm, linewidth=2, color='k', linestyle='--')
        plt.plot(RAD_RG, csrv_exp_fmm, linewidth=2, color='b')
        plt.gca().axvline(x=DST_BI - 3.*STD_BI, linewidth=1, linestyle='-.', color='k')
        plt.gca().axvline(x=DST_BI, linewidth=2, linestyle='-.', color='k')
        plt.gca().axvline(x=DST_BI + 3.*STD_BI, linewidth=1, linestyle='-.', color='k')
        plt.xlim(RAD_RG.min(), RAD_RG.max())
        plt.tight_layout()
        # plt.show(block=True)
        plt.savefig(OUT_DIR + '/bi_c2rv_L_fmm_' + str(N_PART_TOMO) + '_' + str(N_PROCESSORS) + '_' +
                    str(N_PROCESSORS) + '.png')
        plt.close()

        # Plotting times
        plt.figure()
        plt.title('Averaged time per computing unit')
        plt.ylabel('Time [secs]')
        plt.bar(1, csrv_time/(N_TOMOS*N_PART_TOMO), BAR_WIDTH, color='blue', linewidth=2)
        plt.bar(2, csrv_fmm_time/(N_TOMOS*N_PART_TOMO), BAR_WIDTH, color='blue', linewidth=2)
        plt.bar(3, csrv_sim_time/(N_SIMS*N_TOMOS*N_PART_TOMO), BAR_WIDTH, color='blue', linewidth=2)
        plt.bar(4, csrv_sim_fmm_time / (N_SIMS * N_TOMOS * N_PART_TOMO), BAR_WIDTH, color='blue', linewidth=2)
        plt.xticks((1, 2, 3, 4), ('C2RPV', 'C2RPV_FMM', 'C2RPV\'', 'C2RPV_FMM\''))
        plt.xlim(0.5, 4.5)
        plt.tight_layout()
        # plt.show(block=True)
        plt.savefig(OUT_DIR + '/bi_time_' + str(N_PART_TOMO) + '_' + str(N_PROCESSORS) + '_' +
                    str(N_PROCESSORS) + '.png')
        plt.close()

    def test_compute_uni_2nd_fmm(self):

        return

        # Initialization
        hold_shape = (TOMO_SHAPE[0], TOMO_SHAPE[1], .33 * TOMO_SHAPE[2])
        rad, center = max(hold_shape) * .1, np.round(.5 * np.asarray(hold_shape)).astype(np.int)
        rad_rg = np.arange(SHELL_THICK+1, rad * np.pi, SHELL_THICK)
        voi = self.gen_shell_voi_array(hold_shape, center, rad, VOI_SHELL_THICK)
        disperse_io.save_numpy(voi, OUT_DIR + '/voi_shell.mrc')
        exps_coords = list()
        for i in range(N_TOMOS):
            exps_coords.append(self.gen_shell_coords(N_PART_TOMO, center, rad, VOI_SHELL_STD, voi))
        part = disperse_io.load_poly(PARTICLE_SURF_SHELL)
        voi_shell_std_rad = math.radians(VOI_SHELL_STD)

        # Generate instances
        model_csrv = ModelCSRV()
        hold_time = time.time()
        ltomos_shell = self.gen_tlist_coords(exps_coords, voi, part, mode_emb='center')
        print('Time to generate the tomos: ' + str(time.time() - hold_time))

        # Storing tomograms list
        ltomos_shell.pickle(OUT_DIR + '/tomos_shell_tpl.pkl')
        ltomos_shell.store_appended_tomos(OUT_DIR, out_stem='shell_', mode='surface')

        # Compute the matrices
        hold_time = time.time()
        dt_exp_mat = np.zeros(shape=(N_TOMOS, len(rad_rg)), dtype=np.float32)
        for i, tkey in enumerate(ltomos_shell.get_tomo_fname_list()):
            hold_tomo = ltomos_shell.get_tomo_by_key(tkey)
            hold_n_parts = hold_tomo.get_num_particles()
            print('N particles tomo ' + str(i) + ': ' + str(hold_n_parts))
            hold_exp = hold_tomo.compute_uni_2nd_order(rad_rg, thick=None, border=BORDER,
                                                       conv_iter=None, max_iter=None,
                                                       npr=N_PROCESSORS)
            dt_exp_mat[i, :] = hold_exp
        dt_time = time.time() - hold_time
        hold_time = time.time()
        dt_exp_shell_mat = np.zeros(shape=(N_TOMOS, len(rad_rg)), dtype=np.float32)
        for i, tkey in enumerate(ltomos_shell.get_tomo_fname_list()):
            hold_tomo = ltomos_shell.get_tomo_by_key(tkey)
            hold_exp = hold_tomo.compute_uni_2nd_order(rad_rg, thick=SHELL_THICK, border=BORDER,
                                                       conv_iter=None, max_iter=None,
                                                       npr=N_PROCESSORS)
            dt_exp_shell_mat[i, :] = hold_exp
        dt_shell_time = time.time() - hold_time
        hold_time = time.time()
        fmm_exp_mat = np.zeros(shape=(N_TOMOS, len(rad_rg)), dtype=np.float32)
        for i, tkey in enumerate(ltomos_shell.get_tomo_fname_list()):
            hold_tomo = ltomos_shell.get_tomo_by_key(tkey)
            hold_exp = hold_tomo.compute_uni_2nd_order(rad_rg, thick=None, border=BORDER,
                                                       conv_iter=None, max_iter=None, fmm=True,
                                                       npr=N_PROCESSORS)
            fmm_exp_mat[i, :] = hold_exp
        fmm_time = time.time() - hold_time
        fmm2d_exp_mat = np.zeros(shape=(N_TOMOS, len(rad_rg)), dtype=np.float32)
        for i, tkey in enumerate(ltomos_shell.get_tomo_fname_list()):
            hold_tomo = ltomos_shell.get_tomo_by_key(tkey)
            hold_exp = hold_tomo.compute_uni_2nd_order(rad_rg, thick=None, border=BORDER,
                                                       conv_iter=None, max_iter=None, fmm=True, dimen=2,
                                                       npr=N_PROCESSORS)
            fmm2d_exp_mat[i, :] = hold_exp
        hold_time = time.time()
        fmm_exp_shell_mat = np.zeros(shape=(N_TOMOS, len(rad_rg)), dtype=np.float32)
        for i, tkey in enumerate(ltomos_shell.get_tomo_fname_list()):
            hold_tomo = ltomos_shell.get_tomo_by_key(tkey)
            hold_exp = hold_tomo.compute_uni_2nd_order(rad_rg, thick=SHELL_THICK, border=BORDER,
                                                       conv_iter=None, max_iter=None, fmm=True,
                                                       npr=N_PROCESSORS)
            fmm_exp_shell_mat[i, :] = hold_exp
        fmm_shell_time = time.time() - hold_time

        # Compute the simulations
        hold_time = time.time()
        fmm_sim_mat = np.zeros(shape=(N_SIMS * N_TOMOS, len(rad_rg)), dtype=np.float32)
        for i, tkey in enumerate(ltomos_shell.get_tomo_fname_list()):
            hold_tomo = ltomos_shell.get_tomo_by_key(tkey)
            hold_sim = hold_tomo.simulate_uni_2nd_order(N_SIMS, model_csrv, part, 'center', rad_rg, thick=None,
                                                        border=BORDER, conv_iter=None, max_iter=None, fmm=True,
                                                        npr=N_PROCESSORS)
            start_id = i * N_SIMS
            end_id = start_id + N_SIMS
            fmm_sim_mat[start_id:end_id] = hold_sim
        fmm_sim_time = time.time() - hold_time
        fmm2d_sim_mat = np.zeros(shape=(N_SIMS * N_TOMOS, len(rad_rg)), dtype=np.float32)
        for i, tkey in enumerate(ltomos_shell.get_tomo_fname_list()):
            hold_tomo = ltomos_shell.get_tomo_by_key(tkey)
            hold_sim = hold_tomo.simulate_uni_2nd_order(N_SIMS, model_csrv, part, 'center', rad_rg, thick=None,
                                                        border=BORDER, conv_iter=None, max_iter=None, fmm=True, dimen=2,
                                                        npr=N_PROCESSORS)
            start_id = i * N_SIMS
            end_id = start_id + N_SIMS
            fmm2d_sim_mat[start_id:end_id] = hold_sim
        hold_time = time.time()
        fmm_sim_dt_mat = np.zeros(shape=(N_SIMS * N_TOMOS, len(rad_rg)), dtype=np.float32)
        for i, tkey in enumerate(ltomos_shell.get_tomo_fname_list()):
            hold_tomo = ltomos_shell.get_tomo_by_key(tkey)
            hold_sim = hold_tomo.simulate_uni_2nd_order(N_SIMS, model_csrv, part, 'center', rad_rg, thick=None,
                                                        border=BORDER, conv_iter=None, max_iter=None, fmm=False,
                                                        npr=N_PROCESSORS)
            start_id = i * N_SIMS
            end_id = start_id + N_SIMS
            fmm_sim_dt_mat[start_id:end_id] = hold_sim
        fmm_sim_dt_time = time.time() - hold_time
        shell_den = 0
        hold_time = time.time()
        fmm_sim_shell_mat = np.zeros(shape=(N_SIMS * N_TOMOS, len(rad_rg)), dtype=np.float32)
        for i, tkey in enumerate(ltomos_shell.get_tomo_fname_list()):
            hold_tomo = ltomos_shell.get_tomo_by_key(tkey)
            hold_sim = hold_tomo.simulate_uni_2nd_order(N_SIMS, model_csrv, part, 'center', rad_rg, thick=SHELL_THICK,
                                                        border=BORDER, conv_iter=None, max_iter=None, fmm=True,
                                                        npr=N_PROCESSORS)
            start_id = i * N_SIMS
            end_id = start_id + N_SIMS
            fmm_sim_shell_mat[start_id:end_id] = hold_sim
            shell_den += hold_tomo.compute_global_density()
        fmm_sim_shell_time = time.time() - hold_time
        shell_den /= float(len(ltomos_shell.get_tomo_fname_list()))

        # Compute statistics
        dt_exp = np.percentile(dt_exp_mat, 50, axis=0)
        dt_exp_shell = np.percentile(dt_exp_shell_mat, 50, axis=0)
        fmm_exp = np.percentile(fmm_exp_mat, 50, axis=0)
        fmm2d_exp = np.percentile(fmm2d_exp_mat, 50, axis=0)
        fmm_exp_shell = np.percentile(fmm_exp_shell_mat, 50, axis=0)
        fmm_high = np.percentile(fmm_sim_mat, 100-PERCENTILE, axis=0)
        fmm_med = np.percentile(fmm_sim_mat, 50, axis=0)
        fmm_low = np.percentile(fmm_sim_mat, PERCENTILE, axis=0)
        fmm2d_high = np.percentile(fmm2d_sim_mat, 100 - PERCENTILE, axis=0)
        fmm2d_med = np.percentile(fmm2d_sim_mat, 50, axis=0)
        fmm2d_low = np.percentile(fmm2d_sim_mat, PERCENTILE, axis=0)
        dt_high = np.percentile(fmm_sim_dt_mat, 100 - PERCENTILE, axis=0)
        dt_med = np.percentile(fmm_sim_dt_mat, 50, axis=0)
        dt_low = np.percentile(fmm_sim_dt_mat, PERCENTILE, axis=0)
        fmm_high_shell = np.percentile(fmm_sim_shell_mat, 100 - PERCENTILE, axis=0)
        fmm_med_shell = np.percentile(fmm_sim_shell_mat, 50, axis=0)
        fmm_low_shell = np.percentile(fmm_sim_shell_mat, PERCENTILE, axis=0)

        # Plotting
        plt.figure()
        plt.title('Model SPRV in shell VOI')
        plt.ylabel('Ripley\'s L')
        plt.xlabel('Scale')
        plt.fill_between(rad_rg, fmm_low, fmm_high, alpha=.5, color='gray', edgecolor='w')
        plt.plot(rad_rg, np.zeros(shape=len(rad_rg)), 'k--', linewidth=1)
        plt.plot(rad_rg, fmm_low, linewidth=2, color='k', linestyle='--')
        plt.plot(rad_rg, fmm_med, linewidth=2, color='k')
        plt.plot(rad_rg, fmm_high, linewidth=2, color='k', linestyle='--')
        plt.plot(rad_rg, fmm_exp, linewidth=2, color='b', label='FMM')
        plt.plot(rad_rg, dt_exp, linewidth=2, color='b', linestyle='--', label='DT')
        plt.gca().axvline(x=3. * voi_shell_std_rad * rad, linewidth=2, linestyle='-.', color='k')
        plt.gca().axvline(x=.5 * np.pi * rad, linewidth=2, linestyle='-.', color='k')
        plt.xlim(rad_rg.min(), rad_rg.max())
        # plt.ylim(-10, 10)
        plt.legend(loc=1)
        plt.tight_layout()
        # plt.show(block=True)
        plt.savefig(OUT_DIR + '/fmm_shell_L_' + str(N_PART_TOMO) + '_' + str(N_PROCESSORS) + '_' +
                    str(N_PROCESSORS) + '.png')
        plt.close()
        plt.figure()
        plt.title('Model SPRV in shell VOI (2D)')
        plt.ylabel('Ripley\'s L')
        plt.xlabel('Scale')
        plt.fill_between(rad_rg, fmm2d_low, fmm2d_high, alpha=.5, color='gray', edgecolor='w')
        plt.plot(rad_rg, np.zeros(shape=len(rad_rg)), 'k--', linewidth=1)
        plt.plot(rad_rg, fmm2d_low, linewidth=2, color='k', linestyle='--')
        plt.plot(rad_rg, fmm2d_med, linewidth=2, color='k')
        plt.plot(rad_rg, fmm2d_high, linewidth=2, color='k', linestyle='--')
        plt.plot(rad_rg, fmm2d_exp, linewidth=2, color='b', label='FMM')
        plt.gca().axvline(x=3. * voi_shell_std_rad * rad, linewidth=2, linestyle='-.', color='k')
        plt.gca().axvline(x=.5 * np.pi * rad, linewidth=2, linestyle='-.', color='k')
        plt.xlim(rad_rg.min(), rad_rg.max())
        # plt.ylim(-10, 10)
        plt.legend(loc=1)
        plt.tight_layout()
        # plt.show(block=True)
        plt.savefig(OUT_DIR + '/fmm2d_shell_L_' + str(N_PART_TOMO) + '_' + str(N_PROCESSORS) + '_' +
                    str(N_PROCESSORS) + '.png')
        plt.close()
        plt.figure()
        plt.title('Model SPRV in shell VOI')
        plt.ylabel('Ripley\'s O')
        plt.xlabel('Scale')
        plt.fill_between(rad_rg, fmm_low_shell, fmm_high_shell, alpha=.5, color='gray', edgecolor='w')
        plt.plot(rad_rg, shell_den * np.ones(shape=len(rad_rg)), 'k--', linewidth=1)
        plt.plot(rad_rg, fmm_low_shell, linewidth=2, color='k', linestyle='--')
        plt.plot(rad_rg, fmm_med_shell, linewidth=2, color='k')
        plt.plot(rad_rg, fmm_high_shell, linewidth=2, color='k', linestyle='--')
        plt.plot(rad_rg, fmm_exp_shell, linewidth=2, color='b', label='FMM')
        plt.plot(rad_rg, dt_exp_shell, linewidth=2, color='b', linestyle='--', label='DT')
        plt.gca().axvline(x=3. * voi_shell_std_rad * rad, linewidth=2, linestyle='-.', color='k')
        plt.gca().axvline(x=.5 * np.pi * rad, linewidth=2, linestyle='-.', color='k')
        plt.xlim(rad_rg.min(), rad_rg.max())
        # plt.ylim(-10, 10)
        plt.legend(loc=1)
        plt.tight_layout()
        # plt.show(block=True)
        plt.savefig(OUT_DIR + '/fmm_shell_O_' + str(N_PART_TOMO) + '_' + str(N_PROCESSORS) + '_' +
                    str(N_PROCESSORS) + '.png')
        plt.close()

        # Plotting times
        plt.figure()
        plt.title('Averaged time per computing unit')
        plt.ylabel('Time [secs]')
        plt.bar(1, dt_time/(N_TOMOS*hold_n_parts), BAR_WIDTH, color='blue', linewidth=2)
        plt.bar(2, fmm_time/(N_TOMOS*hold_n_parts), BAR_WIDTH, color='blue', linewidth=2)
        plt.bar(2, dt_shell_time / (N_TOMOS*hold_n_parts), BAR_WIDTH, color='blue', linewidth=2)
        plt.bar(3, fmm_shell_time/(N_TOMOS*hold_n_parts), BAR_WIDTH, color='blue', linewidth=2)
        plt.bar(4, fmm_sim_dt_time / (N_SIMS*N_TOMOS*hold_n_parts), BAR_WIDTH, color='blue', linewidth=2)
        plt.bar(5, fmm_sim_time / (N_SIMS*N_TOMOS*hold_n_parts), BAR_WIDTH, color='blue', linewidth=2)
        plt.bar(6, fmm_sim_shell_time / (N_SIMS * N_TOMOS*hold_n_parts), BAR_WIDTH, color='blue', linewidth=2)
        plt.xticks((1, 2, 3, 4, 5, 6, 7), ('DT', 'FMM', 'DT_SHELL', 'FMM_SHELL', 'DT\'', 'FMM\'', 'FMM_SHELL\''))
        plt.xlim(0.5, 6.5)
        plt.tight_layout()
        # plt.show(block=True)
        plt.savefig(OUT_DIR + '/time_fmm_L_' + str(N_PART_TOMO) + '_' + str(N_PROCESSORS) + '_' +
                    str(N_PROCESSORS) + '.png')
        plt.close()