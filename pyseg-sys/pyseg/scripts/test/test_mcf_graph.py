"""

    Tests for testing DisPerSe and GraphMCF classes (mcf_graph script) using phantoms


"""

__author__ = 'Antonio Martinez-Sanchez'

from unittest import TestCase
from pyseg.scripts import mcf_graph
from pyseg.factory import Torus
from pyseg.factory import Grid3D
from pyseg.factory import unpickle_obj
from pyseg.globals import *
import pyseg.disperse_io as disperse_io
from matplotlib import pyplot as plt, rcParams

rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14


#### Torus Test variables
TO_SKIP = False
TO_RESOLUTION = 1
TO_NSIG_PER = 0
TO_V_PERSISTENCE = 0.1
TO_NOISE_MODE = 'normal'
TO_STD_NOISE = 0.1 # If None then SNR is infinity (no noise is added to ground truth)
TO_TILT_ANG = 60
TO_ROT_ANG = 0
TO_R = 25
TO_r = 2.5
TO_NUM_FEAT = 10
TO_RANG_SCALES = (2, 6)
TO_RAND_FACTOR = 0.2
TO_GTRUTH_FILE = './in/torus_gturth.fits'
TO_TOMO_FILE = './in/torus_tomo.fits'
TO_MASK_FILE = './in/torus_mask.fits'
TO_VTP_FILE = './in/torus_feat.vtp'
TO_PKL_FILE = './in/torus.pkl'
TO_EPS = 3
# For debugging
# TO_GRAPHMCF = './out/torus_tomo.pkl'
TO_GRAPHMCF = None
# TO_TORUS_PKL = './in/torus.pkl'
TO_TORUS_PKL = None

# mcf_graph configuration
MCF_OUT_DIR = './out/'
PAIR_PROP = STR_FIELD_VALUE

#### Grid 3D test variables
G3_SKIP = True
G3_RESOLUTION = 1
G3_NSIG_PER = 0.05
G3_PERSISTENCE = 0.01
G3_NOISE_MODE = 'normal'
# G3_STD_NOISE = np.arange(0, 1.5, 0.1)
# G3_STD_NOISE = np.arange(0, 1, 0.05)
G3_STD_NOISE = np.linspace(0.03, 1.0, 15)[::-1]# (1.3, 1., 0.6, 0.3, 0.1, 0.01) # (0.3, 0.1)
G3_TILT_ANG = None
G3_ROT_ANG = None
G3_L = (6, 6, 3) # (3, 3, 3) # (2, 2, 2) # (3, 3, 2)
G3_SP = 8 # 4
G3_EV = 0.3 # 0.5
G3_GBP_CUT = 0.3
G3_FEAT = 0.05
# G3_THICK = (5, 5)
G3_GTRUTH_FILE = './in/grid3_gturth.fits'
G3_TOMO_FILE = './in/grid3_tomo.fits'
G3_MASK_FILE = './in/grid3_mask.fits'
G3_VTP_FILE = './in/grid3_feat.vtp'
G3_PKL_FILE = './in/grid3.pkl'
G3_EPS = 3
G3_NUM_REP = 10 # 3 # 10
# For debugging
# G3_GRAPHMCF = './out/grid3d_tomo.pkl'
G3_GRAPHMCF = None
# G3_TORUS_PKL = './in/grid3d.pkl'
G3_TORUS_PKL = None

#### Grid 3D connectivity test variables
G3_CONN_SKIP = False
G3_EDG_VER_RATIO = 2.0
G3_RES_PER = 1.2

class TestMcfGraph(TestCase):

    def test_Torus(self):

        print 'TORUS TEST:'

        if TO_SKIP:
            print '\tSkipping...'
            return

        if TO_TORUS_PKL is None:
            # Synthetic phantom generation
            print '\tGenerating the synthetic phantom...'
            torus = Torus()
            torus.set_parameters(TO_R, TO_r, TO_NUM_FEAT, TO_RAND_FACTOR, TO_RANG_SCALES)
            torus.build()
            if TO_STD_NOISE is not None:
                torus.add_noise(TO_NOISE_MODE, TO_STD_NOISE)
            if (TO_TILT_ANG is not None) and (TO_ROT_ANG is not None):
                torus.add_mw(TO_TILT_ANG, TO_ROT_ANG)
            torus.save_gtruth(TO_GTRUTH_FILE)
            torus.save_tomo(TO_TOMO_FILE)
            torus.save_mask(TO_MASK_FILE)
            torus.save_vtp(TO_VTP_FILE)
            torus.pickle(TO_PKL_FILE)
        else:
            print '\tUnpickling Torus...'
            torus = unpickle_obj(TO_TORUS_PKL)

        print '\tTesting Torus phantom with parameters: '
        print '\t\tMajor radius: ' + str(TO_R)
        print '\t\tMinor radius: ' + str(TO_r)
        print '\t\tRange for scales: ' + str(TO_RANG_SCALES)
        print '\t\tRandomness factor: ' + str(TO_RAND_FACTOR)
        print '\t\tNumber of features: ' + str(TO_NUM_FEAT)
        if TO_STD_NOISE is None:
            print '\t\tSNR: infinity'
        else:
            print '\t\tSNR: ' + str(round(torus.get_snr(), 2)) + ' dB'
        if (TO_TILT_ANG is not None) and (TO_ROT_ANG is not None):
            print '\t\tMissing wedge with tilt ' + str(TO_TILT_ANG) + 'deg and rotation '\
                  + str(TO_ROT_ANG) + 'deg'

        # Obtaining the GraphMCF
        if TO_GRAPHMCF is None:
            print '\tObtaining the GraphMCF...'
            main_args = ['-i', TO_TOMO_FILE, '-o', MCF_OUT_DIR, '-m', TO_MASK_FILE,
                        '-r', str(TO_RESOLUTION), '-N', str(TO_NSIG_PER),
                        '-s', TO_RANG_SCALES[0], '-v']
            mcf_graph.main(main_args)

            # Load GraphMCF
            print '\tLoading GraphMCF...'
            path, file_n = os.path.split(TO_TOMO_FILE)
            stem, _ = os.path.splitext(file_n)
            graph_mcf = unpickle_obj(MCF_OUT_DIR + stem + '.pkl')
        else:
            print '\tUnpickling GraphMCF...'
            graph_mcf = unpickle_obj(TO_GRAPHMCF)

        # Threshold to keep just the TO_NUM_FEAT with highest persistence
        print '\tTopological simplification...'
        graph_mcf.topological_simp(TO_V_PERSISTENCE)
        disperse_io.save_vtp(graph_mcf.get_vtp(av_mode=True, edges=True), './out/hold.vtp')
        disperse_io.save_vtp(graph_mcf.get_scheme_vtp(nodes=True, edges=True),
                             './out/hold_sch.vtp')
        disperse_io.save_numpy(graph_mcf.print_vertices(th_den=-1), './out/hold_seg.vti')

        # Preserves the TO_NUM_FEAT highest persistence vertices and th TO_NUM_FEAT with
        # lower field value
        print '\tGraph filtration...'
        graph_mcf.threshold_vertices_n(TO_NUM_FEAT, STR_V_PER)
        graph_mcf.threshold_edges_n(TO_NUM_FEAT, STR_FIELD_VALUE, mode='low')
        vertices = graph_mcf.get_vertices_list()
        points = np.zeros(shape=(len(vertices), 3), dtype=np.float)
        for i, v in enumerate(vertices):
            points[i, :] = graph_mcf.get_vertex_coords(v)
        disperse_io.save_vtp(graph_mcf.get_vtp(av_mode=True, edges=True), './out/hold2.vtp')
        disperse_io.save_vtp(graph_mcf.get_scheme_vtp(nodes=True, edges=True),
                             './out/hold2_sch.vtp')
        disperse_io.save_numpy(graph_mcf.print_vertices(th_den=-1), './out/hold_seg2.vti')

        # Comparing GraphMCF with ground truth
        print '\tComparing with Ground Truth...'
        t_n, f_p = torus.check_feat_localization(points, TO_EPS)

        # Printing the result
        print ''
        print '\tRESULTS: '
        print 'Number of phantom features: ' + str(torus.get_num_features())
        print 'Number of detected features: ' + str(len(vertices))
        print 'Number of true negatives: ' + str(np.sum(t_n))
        print 'Number of false positives: ' + str(np.sum(f_p))

        # Test assertions
        error_msg = 'True negatives array does not match the number of features.'
        self.assertEqual(len(t_n), torus.get_num_features(), error_msg)
        error_msg = 'False positives array does not match the number of features.'
        self.assertEqual(len(f_p), torus.get_num_features(), error_msg)

    def test_Grid3D(self):

        print '3D GRID TEST:'

        if G3_SKIP:
            print '\tSkipping...'
            return

        hold_snr = (-1) * np.ones(shape=(G3_NUM_REP, len(G3_STD_NOISE)), dtype=np.float)
        hold_t_n = (-1) * np.ones(shape=(G3_NUM_REP, len(G3_STD_NOISE)), dtype=np.float)
        hold_f_p = (-1) * np.ones(shape=(G3_NUM_REP, len(G3_STD_NOISE)), dtype=np.float)
        hold_d_e = (-1) * np.ones(shape=(G3_NUM_REP, len(G3_STD_NOISE)), dtype=np.float)
        hold_avd_in = (-1) * np.ones(shape=(G3_NUM_REP, len(G3_STD_NOISE)), dtype=np.float)
        hold_avd_g = (-1) * np.ones(shape=(G3_NUM_REP, len(G3_STD_NOISE)), dtype=np.float)
        thick = G3_SP * G3_FEAT
        thick2 = 2 * math.ceil(thick * G3_RESOLUTION)
        # thick2 = math.ceil(thick * G3_RESOLUTION)


        # Number of repetitions loop
        for i in range(G3_NUM_REP):
            # for j in range(len(G3_STD_NOISE)):
            j = 0

            # Noise loop
            while j < len(G3_STD_NOISE):

                if G3_TORUS_PKL is None:
                    # Synthetic phantom generation
                    print '\tGenerating the synthetic phantom...'
                    grid = Grid3D()
                    grid.set_parameters(G3_L, G3_SP, (thick, thick), G3_EV)
                    grid.build()
                    if (G3_STD_NOISE[j] is not None) and (G3_STD_NOISE[j] > 0):
                        grid.add_noise(G3_NOISE_MODE, G3_STD_NOISE[j])
                    if (G3_TILT_ANG is not None) and (G3_ROT_ANG is not None):
                        grid.add_mw(G3_TILT_ANG, G3_ROT_ANG)
                    grid.save_gtruth(G3_GTRUTH_FILE)
                    grid.save_tomo(G3_TOMO_FILE)
                    grid.save_mask(G3_MASK_FILE)
                    grid.save_vtp(G3_VTP_FILE)
                    grid.pickle(G3_PKL_FILE)
                else:
                    print '\tUnpickling Torus...'
                    grid = unpickle_obj(G3_TORUS_PKL)

                print '\tTesting Grid 3D phantom with parameters: '
                snr = round(grid.get_snr(), 2)
                print '\t\tNumber of lines (x, y, z): ' + str(G3_L * G3_RESOLUTION) + ' nm'
                print '\t\tLines spacing: ' + str(G3_SP * G3_RESOLUTION) + ' nm'
                print '\t\tRange for thickness: ' + str(thick2) \
                      + ' vox' + ' (' + str(round(thick2*G3_RESOLUTION,2)) + ' nm)'
                if (G3_STD_NOISE[j] is None) or (snr <= 0):
                    print '\t\tSNR: infinity'
                else:
                    print '\t\tSNR: ' + str(snr) + ' (' + str(round(10*math.log10(snr),2)) + ' dB)'
                if (G3_TILT_ANG is not None) and (G3_ROT_ANG is not None):
                    print '\t\tMissing wedge with tilt ' + str(G3_TILT_ANG) + 'deg and rotation '\
                          + str(G3_ROT_ANG) + 'deg'
                print '\t\tIteration: ' + str(i+1) + '/' + str(G3_NUM_REP)

                # Obtaining the GraphMCF
                if G3_GRAPHMCF is None:
                    print '\tObtaining the GraphMCF...'
                    try:
                        thick2 = 1.0
                        main_args = ['-i', G3_TOMO_FILE, '-o', MCF_OUT_DIR, '-m', G3_MASK_FILE,
                                    '-r', str(G3_RESOLUTION), '-N', str(G3_NSIG_PER),
                                    '-s', 0.5 * thick2, '-v']
                        mcf_graph.main(main_args)
                    except:
                        print 'Error running mcf_graph script, skipping iteration...'
                        continue

                    # Load GraphMCF
                    print '\tLoading GraphMCF...'
                    path, file_n = os.path.split(G3_TOMO_FILE)
                    stem, _ = os.path.splitext(file_n)
                    graph_mcf = unpickle_obj(MCF_OUT_DIR + stem + '.pkl')
                else:
                    print '\tUnpickling GraphMCF...'
                    graph_mcf = unpickle_obj(G3_GRAPHMCF)

                print '\tTopological simplification...'
                # Make topological simplification until fitting the number of features
                graph_mcf.topological_simp(0, n=grid.get_num_features(), prop_ref=PAIR_PROP)
                # Filter edges until fitting the number of edges
                graph_mcf.threshold_edges_n(grid.get_num_edges(), STR_FIELD_VALUE, mode='low')
                skel = graph_mcf.get_vtp(av_mode=True, edges=True)
                disperse_io.save_vtp(skel, './out/hold.vtp')
                disperse_io.save_vtp(graph_mcf.get_scheme_vtp(nodes=True, edges=True),
                                     './out/hold_sch.vtp')
                # disperse_io.save_numpy(graph_mcf.print_vertices(th_den=-1), './out/hold_seg.vti')
                grid.save_gtruth('./out/hold_gtruth.mrc')
                grid.save_mask('./out/hold_mask.mrc')
                grid.save_tomo('./out/hold_tomo.mrc')

                # Getting point, degrees form the graph
                vertices = graph_mcf.get_vertices_list()
                n_points = len(vertices)
                points = np.zeros(shape=(n_points, 3), dtype=np.float)
                degrees = np.zeros(shape=n_points, dtype=np.int)
                for k, v in enumerate(vertices):
                    points[k] = graph_mcf.get_vertex_coords(v)
                    neighs, _ = graph_mcf.get_vertex_neighbours(v.get_id())
                    degrees[k] = len(neighs)

                # Comparing GraphMCF with ground truth
                print '\tComparing with Ground Truth...'
                t_n, f_p, d_e = grid.hard_test(points, degrees, G3_EPS)
                avd_in, avd_g = grid.soft_test(skel)

                hold_snr[i, j] = snr
                hold_t_n[i, j] = np.round(t_n, 2)
                hold_f_p[i, j] = np.round(f_p, 2)
                hold_d_e[i, j] = np.round(d_e, 2)
                hold_avd_in[i, j] = np.round(avd_g, 2)
                hold_avd_g[i, j] = np.round(avd_in, 2)

                j += 1

        # Printing the result
        print ''
        print '\tRESULTS: '
        for i in range(len(G3_STD_NOISE)):
            print '\tIteration: ' + str(i+1)
            print '\tSNR: [' + str(hold_snr[:, i].min()) + ', ' \
                  + str(hold_snr[:, i].mean()) + ', ' + str(hold_snr[:, i].max()) + ']'
            print '\tFraction of true negatives: [' + str(hold_t_n[:, i].min()) + ', ' \
                  + str(hold_t_n[:, i].mean()) + ', ' + str(hold_t_n[:, i].max()) + ']'
            print '\tFraction of false positives: [' + str(hold_f_p[:, i].min()) + ', ' \
                  + str(hold_f_p[:, i].mean()) + ', ' + str(hold_f_p[:, i].max()) + ']'
            print '\tFraction of error degrees: [' + str(hold_d_e[:, i].min()) + ', ' \
                  + str(hold_d_e[:, i].mean()) + ', ' + str(hold_d_e[:, i].max()) + ']'
            print '\tAverage distance to ground truth: [' + str(hold_avd_g[:, i].min()) + ', ' \
                  + str(hold_avd_g[:, i].mean()) + ', ' + str(hold_avd_g[:, i].max()) + ']'
            print '\tAverage distance to input skeleton: [' + str(hold_avd_in[:, i].min()) + ', ' \
                  + str(hold_avd_in[:, i].mean()) + ', ' + str(hold_avd_in[:, i].max()) + ']'

        # Plotting the results
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.mean(hold_snr, axis=0), np.mean(hold_d_e, axis=0), color='blue', linewidth=2)
        plt.plot(np.mean(hold_snr, axis=0), np.mean(hold_t_n, axis=0), color='red', linewidth=2)
        plt.plot(np.mean(hold_snr, axis=0), np.mean(hold_f_p, axis=0), color='yellow', linewidth=2)
        ax.set_xscale('log')
        plt.xlabel('SNR')
        plt.show(block=True)
        plt.close()


        # # Test assertions
        # error_msg = 'Invalid value for t_n.'
        # self.assertGreaterEqual(hold_t_n.min(), 0, error_msg)
        # self.assertLessEqual(hold_t_n.max(), 1, error_msg)
        # error_msg = 'Invalid value for f_p.'
        # self.assertGreaterEqual(hold_f_p.min(), 0, error_msg)
        # self.assertLessEqual(hold_f_p.max(), 1, error_msg)
        # error_msg = 'Invalid value for d_e.'
        # self.assertGreaterEqual(hold_d_e.min(), 0, error_msg)
        # self.assertLessEqual(hold_d_e.max(), 1, error_msg)
        # nx, ny, nz = grid.get_shape()
        # d_max = math.sqrt(nx*nx + ny*ny * nz*nz)
        # error_msg = 'Invalid value for avd_in.'
        # self.assertGreaterEqual(hold_avd_in.min(), 0, error_msg)
        # self.assertLessEqual(hold_avd_in.max(), d_max, error_msg)
        # error_msg = 'Invalid value for avd_g.'
        # self.assertGreaterEqual(hold_avd_g.min(), 0, error_msg)
        # self.assertLessEqual(hold_avd_g.max(), d_max, error_msg)

    def test_Grid3DConn(self):

        print '3D GRID CONNECTIVITY TEST:'

        if G3_CONN_SKIP:
            print '\tSkipping...'
            return

        hold_snr = (-1) * np.ones(shape=(G3_NUM_REP, len(G3_STD_NOISE)), dtype=np.float)
        hold_t_p = (-1) * np.ones(shape=(G3_NUM_REP, len(G3_STD_NOISE)), dtype=np.float)
        hold_f_p = (-1) * np.ones(shape=(G3_NUM_REP, len(G3_STD_NOISE)), dtype=np.float)
        hold_f_n = (-1) * np.ones(shape=(G3_NUM_REP, len(G3_STD_NOISE)), dtype=np.float)
        hold_p_e = (-1) * np.ones(shape=(G3_NUM_REP, len(G3_STD_NOISE)), dtype=np.float)
        thick = G3_SP * G3_FEAT
        thick2 = 2 * math.ceil(thick * G3_RESOLUTION)


        # Number of repetitions loop
        for i in range(G3_NUM_REP):
            # for j in range(len(G3_STD_NOISE)):
            j = 0

            # Noise loop
            while j < len(G3_STD_NOISE):

                if G3_TORUS_PKL is None:
                    # Synthetic phantom generation
                    print '\tGenerating the synthetic phantom...'
                    grid = Grid3D()
                    grid.set_parameters(G3_L, G3_SP, (thick, thick), G3_EV)
                    grid.build()
                    if (G3_STD_NOISE[j] is not None) and (G3_STD_NOISE[j] > 0):
                        grid.add_noise(G3_NOISE_MODE, G3_STD_NOISE[j])
                    if (G3_TILT_ANG is not None) and (G3_ROT_ANG is not None):
                        grid.add_mw(G3_TILT_ANG, G3_ROT_ANG)
                    grid.save_gtruth(G3_GTRUTH_FILE)
                    grid.save_tomo(G3_TOMO_FILE)
                    grid.save_mask(G3_MASK_FILE)
                    grid.save_vtp(G3_VTP_FILE)
                    grid.pickle(G3_PKL_FILE)
                else:
                    print '\tUnpickling Torus...'
                    grid = unpickle_obj(G3_TORUS_PKL)

                print '\tTesting Grid 3D phantom with parameters: '
                snr = grid.get_snr()
                print '\t\tNumber of lines (x, y, z): ' + str(G3_L * G3_RESOLUTION) + ' nm'
                print '\t\tLines spacing: ' + str(G3_SP * G3_RESOLUTION) + ' nm'
                print '\t\tRange for thickness: ' + str(thick2) \
                      + ' vox' + ' (' + str(round(thick2*G3_RESOLUTION,2)) + ' nm)'
                if (G3_STD_NOISE[j] is None) or (snr <= 0):
                    print '\t\tSNR: infinity'
                else:
                    print '\t\tSNR: ' + str(snr) + ' (' + str(round(10*math.log10(snr),2)) + ' dB)'
                if (G3_TILT_ANG is not None) and (G3_ROT_ANG is not None):
                    print '\t\tMissing wedge with tilt ' + str(G3_TILT_ANG) + 'deg and rotation '\
                          + str(G3_ROT_ANG) + 'deg'
                print '\t\tIteration: ' + str(i+1) + '/' + str(G3_NUM_REP)

                # Obtaining the GraphMCF
                if G3_GRAPHMCF is None:
                    print '\tObtaining the GraphMCF...'
                    try:
                        thick2 = 1
                        main_args = ['-i', G3_TOMO_FILE, '-o', MCF_OUT_DIR, '-m', G3_MASK_FILE,
                                    '-r', str(G3_RESOLUTION), '-N', str(G3_NSIG_PER),
                                    '-s', 0.5 * thick2, '-v']
                        mcf_graph.main(main_args)
                    except:
                        print 'Error running mcf_graph script, skipping iteration...'
                        continue

                    # Load GraphMCF
                    print '\tLoading GraphMCF...'
                    path, file_n = os.path.split(G3_TOMO_FILE)
                    stem, _ = os.path.splitext(file_n)
                    graph_mcf = unpickle_obj(MCF_OUT_DIR + stem + '.pkl')
                else:
                    print '\tUnpickling GraphMCF...'
                    graph_mcf = unpickle_obj(G3_GRAPHMCF)

                print '\tTopological simplification...'
                # Make topological simplification until fitting the number of features times residues percentage
                graph_mcf.topological_simp(0, n=grid.get_num_features()*G3_RES_PER, prop_ref=PAIR_PROP)
                # Filter edges until fitting the number of edges
                n_edges = float(len(graph_mcf.get_vertices_list())) * G3_EDG_VER_RATIO
                i_edges = len(graph_mcf.get_edges_list())
                if n_edges < i_edges:
                    graph_mcf.threshold_edges_n(grid.get_num_edges(), STR_FIELD_VALUE, mode='low')
                else:
                    print 'WARNING: Graph cannot be simplify to ' + str(n_edges) + ' edges, since it has only ' + str(i_edges)
                skel = graph_mcf.get_vtp(av_mode=True, edges=True)
                stem = 'grid_conn_snr_' + str(round(snr, 2))
                disperse_io.save_vtp(skel, './out/' + stem + '_skel.vtp')
                disperse_io.save_vtp(graph_mcf.get_scheme_vtp(nodes=True, edges=True),
                                     './out/' + stem + '_sch.vtp')
                # disperse_io.save_numpy(graph_mcf.print_vertices(th_den=-1), './out/hold_seg.vti')
                grid.save_gtruth('./out/' + stem + '_gtruth.mrc')
                grid.save_mask('./out/' + stem + '_mask.mrc')
                grid.save_tomo('./out/' + stem + '_tomo.mrc')

                # Getting the starting points for tracing
                labels = list()
                vertices = graph_mcf.get_vertices_list()
                n_points = float(len(vertices))
                # points = np.zeros(shape=(n_points, 3), dtype=np.float)
                gxs, gys, gzs = grid.get_grid_points()
                grid_vs = -1 * np.ones(shape=gxs.shape, dtype=np.int)
                good_picks, bad_picks, fp_picks, tot_picks = 0., 0., 0., float(np.asarray(gxs.shape).prod())
                for x in range(gxs.shape[0]):
                    for y in range(gys.shape[1]):
                        for z in range(gzs.shape[2]):
                            labels.append(str(x) + '_' + str(y) + '_' + str(z))
                            hold_min = np.finfo(np.float).max
                            g_point = np.asarray((gxs[x, y, z], gys[x, y, z], gzs[x, y, z]), dtype=np.float32)
                            pick_found = False
                            for v in vertices:
                                point = np.asarray(graph_mcf.get_vertex_coords(v), dtype=np.float32)
                                dst = g_point - point
                                dst = math.sqrt((dst*dst).sum())
                                if hold_min > dst:
                                    if dst < G3_EPS:
                                        grid_vs[x, y, z] = int(v.get_id())
                                        pick_found = True
                                    hold_min = dst
                            if pick_found:
                                good_picks += 1
                            else:
                                bad_picks += 1
                # Computing the false positives
                for v in vertices:
                    point = np.asarray(graph_mcf.get_vertex_coords(v), dtype=np.float32)
                    if not grid.in_feature(point, G3_EPS):
                        fp_picks += 1

                # Trace the path to closest neighbours
                graph_mcf.compute_graph_gt()
                tot_path, good_path = 0., 0.
                for x in range(gxs.shape[0]):
                    for y in range(gys.shape[1]):
                        for z in range(gzs.shape[2]):
                            # Get staring vertex and neighbours (target vertices)
                            v = grid_vs[x, y, z]
                            lbl = str(x) + '_' + str(y) + '_' + str(z)
                            if v != -1:
                                if x == 0:
                                    nx0 = None
                                else:
                                    nx0 = grid_vs[x-1, y, z]
                                if x == gxs.shape[0] - 1:
                                    nx1 = None
                                else:
                                    nx1 = grid_vs[x+1, y, z]
                                if y == 0:
                                    ny0 = None
                                else:
                                    ny0 = grid_vs[x, y-1, z]
                                if y == gys.shape[1] - 1:
                                    ny1 = None
                                else:
                                    ny1 = grid_vs[x, y+1, z]
                                if z == 0:
                                    nz0 = None
                                else:
                                    nz0 = grid_vs[x, y, z-1]
                                if z == gzs.shape[2] - 1:
                                    nz1 = None
                                else:
                                    nz1 = grid_vs[x, y, z+1]
                                # Check path to neighbours
                                neighbours = [nx0, nx1, ny0, ny1, nz0, nz1]
                                for neigh in neighbours:
                                    if (neigh is not None) and (v is not neigh):
                                        tot_path += 1
                                        # Computing the arc path
                                        v_path = graph_mcf.find_shortest_path(v, neigh, prop_key=SGT_EDGE_LENGTH)
                                        if (v_path is not None) and (len(v_path) > 0):
                                            # Connectivity test
                                            found = False
                                            e_point = graph_mcf.get_vertex_coords(graph_mcf.get_vertex(v_path[-1]))
                                            for p in v_path:
                                                if found:
                                                    break
                                                p_point = graph_mcf.get_vertex_coords(graph_mcf.get_vertex(p))
                                                lbl_count = 0
                                                for hold_x in range(gxs.shape[0]):
                                                    if found:
                                                        break
                                                    for hold_y in range(gys.shape[1]):
                                                        if found:
                                                            break
                                                        for hold_z in range(gzs.shape[2]):
                                                            if found:
                                                                break
                                                            hold_lbl = labels[lbl_count]
                                                            lbl_count += 1
                                                            if lbl == hold_lbl:
                                                                continue
                                                            hold_point = np.asarray((gxs[hold_x][hold_y][hold_z],
                                                                                     gys[hold_x][hold_y][hold_z],
                                                                                     gzs[hold_x][hold_y][hold_z]),
                                                                                    dtype=np.float32)
                                                            # Check if end point condition
                                                            dst = hold_point - e_point
                                                            dst = math.sqrt((dst*dst).sum())
                                                            if dst < G3_EPS:
                                                                good_path += 1
                                                                found = True
                                                            else:
                                                                # Check if bad point
                                                                dst = hold_point - p_point
                                                                dst = math.sqrt((dst*dst).sum())
                                                                if dst < G3_EPS:
                                                                    if p == v_path[-1]:
                                                                        good_path += 1
                                                                    found = True

                hold_snr[i, j] = snr
                if good_picks > tot_picks:
                    good_picks = tot_picks
                if bad_picks > tot_picks:
                    bad_picks = tot_picks
                if tot_picks <= 0:
                    hold_t_p[i, j] = 0
                    hold_f_n[i, j] = 0
                else:
                    hold_t_p[i, j] = good_picks / tot_picks
                    hold_f_n[i, j] = bad_picks / tot_picks
                if n_points <= 0:
                    hold_f_p[i, j] = 0
                else:
                    hold_f_p[i, j] = fp_picks / n_points
                if tot_path <= 0:
                    hold_p_e[i, j] = 0
                else:
                    hold_p_e[i, j] = good_path / tot_path

                j += 1

        # Printing the result
        print ''
        print '\tRESULTS: '
        for i in range(len(G3_STD_NOISE)):
            print '\tIteration: ' + str(i+1)
            print '\tSNR: [' + str(hold_snr[:, i].min()) + ', ' \
                  + str(hold_snr[:, i].mean()) + ', ' + str(hold_snr[:, i].max()) + ']'
            print '\tTrue positive picked: [' + str(hold_t_p[:, i].min()) + ', ' \
                  + str(hold_t_p[:, i].mean()) + ', ' + str(hold_t_p[:, i].max()) + ']'
            print '\tFalse positive picked: [' + str(hold_f_p[:, i].min()) + ', ' \
                  + str(hold_f_p[:, i].mean()) + ', ' + str(hold_f_p[:, i].max()) + ']'
            print '\tFalse negative picked: [' + str(hold_f_n[:, i].min()) + ', ' \
                  + str(hold_f_n[:, i].mean()) + ', ' + str(hold_f_n[:, i].max()) + ']'
            print '\tFraction of correctly tracked paths: [' + str(hold_p_e[:, i].min()) + ', ' \
                  + str(hold_p_e[:, i].mean()) + ', ' + str(hold_p_e[:, i].max()) + ']'

        # Storing the results
        np.savez('./out/grid3d_arrays.npz', hold_snr, hold_t_p, hold_p_e, hold_f_p, hold_f_n)

        # Plotting the results
        snr_mean = np.mean(hold_snr, axis=0)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(snr_mean, np.mean(hold_t_p, axis=0), color='blue', linestyle='-', marker='o', linewidth=2, label='TP')
        # plt.fill_between(snr_mean, np.percentile(hold_t_p, 5, axis=0), np.percentile(hold_t_p, 95, axis=0), alpha=0.5, color='blue', edgecolor='w')
        plt.plot(snr_mean, np.mean(hold_p_e, axis=0), color='red', linestyle='-', marker='o', linewidth=2, label='Tpath')
        # plt.fill_between(snr_mean, np.percentile(hold_p_e, 5, axis=0), np.percentile(hold_p_e, 5, axis=0), alpha=0.5, color='red', edgecolor='w')
        plt.plot(snr_mean, np.mean(hold_f_p, axis=0), color='yellow', linestyle='-', marker='o', linewidth=2, label='FP')
        # plt.fill_between(snr_mean, np.percentile(hold_f_p, 5, axis=0), np.percentile(hold_f_p, 95, axis=0), alpha=0.5, color='yellow', edgecolor='w')
        plt.plot(snr_mean, np.mean(hold_f_n, axis=0), color='cyan', linestyle='-', marker='o', linewidth=2, label='FN')
        # plt.fill_between(snr_mean, np.percentile(hold_f_n, 5, axis=0), np.percentile(hold_f_n, 95, axis=0), alpha=0.5, color='cyan', edgecolor='w')
        plt.plot(snr_mean, np.ones(shape=hold_p_e.shape[1]), color='black', linestyle='--', linewidth=1)
        plt.plot(snr_mean, np.zeros(shape=hold_p_e.shape[1]), color='black', linestyle='--', linewidth=1)
        plt.ylim((-0.1, 1.1))
        plt.xlim((snr_mean[0], snr_mean[-1]))
        ax.set_xscale('log')
        plt.xlabel('SNR')
        plt.legend(loc=7)
        # ref_FR_ssup_mb.sh
        plt.tight_layout()
        plt.savefig('./out/test_grid_conn.png', dpi=600)
        plt.close()


        # # Test assertions
        # error_msg = 'Invalid value for t_n.'
        # self.assertGreaterEqual(hold_t_n.min(), 0, error_msg)
        # self.assertLessEqual(hold_t_n.max(), 1, error_msg)
        # error_msg = 'Invalid value for f_p.'
        # self.assertGreaterEqual(hold_f_p.min(), 0, error_msg)
        # self.assertLessEqual(hold_f_p.max(), 1, error_msg)
        # error_msg = 'Invalid value for d_e.'
        # self.assertGreaterEqual(hold_d_e.min(), 0, error_msg)
        # self.assertLessEqual(hold_d_e.max(), 1, error_msg)
        # nx, ny, nz = grid.get_shape()
        # d_max = math.sqrt(nx*nx + ny*ny * nz*nz)
        # error_msg = 'Invalid value for avd_in.'
        # self.assertGreaterEqual(hold_avd_in.min(), 0, error_msg)
        # self.assertLessEqual(hold_avd_in.max(), d_max, error_msg)
        # error_msg = 'Invalid value for avd_g.'
        # self.assertGreaterEqual(hold_avd_g.min(), 0, error_msg)
        # self.assertLessEqual(hold_avd_g.max(), d_max, error_msg)
