"""

    Tests for testing bio-material tracking by GraphMCF


"""

__author__ = 'Antonio Martinez-Sanchez'

import time
import sys
from pyseg.scripts import mcf_graph
from pyseg.factory import Grid3D
from pyseg.factory import unpickle_obj
from pyseg.globals import *
import pyseg.disperse_io as disperse_io
from matplotlib import pyplot as plt, rcParams
import multiprocessing as mp

### INPUT SETTINGS ##############

# Short version is 'False' (default) takes few minutes but is enough for testing
# functionality, long version 'True' produces stronger statistics but takes a few
# hours and require high memory resources
try:
    if sys.argv[1] == 'do_long':
        do_long = True
    else:
        do_long = False
except IndexError:
    do_long = False
ROOT_DIR = os.path.split(os.path.abspath(__file__))[0] + '/../../../tests'
MCF_OUT_DIR = ROOT_DIR + '/results/tracing_grid'

#################################

rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14

# mcf_graph configuration
PAIR_PROP = STR_FIELD_VALUE

#### Grid 3D test variables
G3_SKIP = True
G3_RESOLUTION = 1
G3_NSIG_PER = 0.05
G3_PERSISTENCE = 0.01
G3_NOISE_MODE = 'normal'
 # np.linspace(0.03, 1.0, 15)[::-1]# (1.3, 1., 0.6, 0.3, 0.1, 0.01) # (0.3, 0.1)
if do_long:
    G3_L = (6, 6, 3)
    G3_STD_NOISE = np.linspace(0.03, 1.0, 15)[::-1]
    G3_NUM_REP = 10
    G3_NPR = 10
else:
    G3_L = (3, 3, 3)  # (3, 3, 2)
    G3_STD_NOISE = np.linspace(0.03, 1.0, 10)[::-1]
    G3_NUM_REP = 3
    G3_NPR = 3
G3_SP = 8 # 4
G3_EV = 0.3 # 0.5
G3_GBP_CUT = 0.3
G3_FEAT = 0.05
G3_EPS = 3
# For picking
G3_GRAPHMCF_PKL = True # True
G3_EDG_VER_RATIO = 6.0
G3_RES_PER = 1.2
# For DoG
DOG_SG1 = 1.5
DOG_K = 1.1

#################################################################################################
# Parallel process for computing the graph
#################################################################################################

def pr_mcf_graph(pr_id, ids, std, mcf_out_dir, res, dims, eps, sp, ev, nsig_per, res_per, ev_ratio):

    # To avoid repeated simulations
    if pr_id < -1:
        np.random.seed(0)
    else:
        np.random.seed(pr_id)

    for idx in ids:

        g3_tomo_file = MCF_OUT_DIR + '/grid_noise_std_' + str(std) + '_it_' + str(idx) + '.fits'
        g3_mask_file = MCF_OUT_DIR + '/grid_mask_std_' + str(std) + '_it_' + str(idx) + '_mask.fits'
        # g3_graph_pkl = MCF_OUT_DIR + '/grid_graph_std_' + str(G3_STD_NOISE[j]) + '_it_' + str(i) + '.pkl'
        g3_graph_vtp = MCF_OUT_DIR + '/grid_graph_skel_' + str(std) + '_it_' + str(idx) + '.vtp'
        g3_graph_sch = MCF_OUT_DIR + '/grid_graph_sch_' + str(std) + '_it_' + str(idx) + '.vtp'
        g3_grid_pkl = MCF_OUT_DIR + '/grid_' + str(std) + '_it_' + str(idx) + '.pkl'
        mcf_out_dir2 = MCF_OUT_DIR + '/graph_' + str(std) + '_it_' + str(idx)
        if os.path.exists(mcf_out_dir2):
            clean_dir(mcf_out_dir2)
        else:
            os.makedirs(mcf_out_dir2)
        g3_graph_pkl = mcf_out_dir2 + '/' + os.path.splitext(os.path.split(g3_tomo_file)[1])[0] + '.pkl'

        grid = Grid3D()
        grid.set_parameters(dims, sp, (thick, thick), ev)
        grid.build()
        grid.add_noise(G3_NOISE_MODE, G3_STD_NOISE[std])
        grid.save_tomo(g3_tomo_file)
        grid.save_mask(g3_mask_file)
        grid.pickle(g3_grid_pkl)

        thick2 = 1
        main_args = ['-i', g3_tomo_file, '-o', mcf_out_dir2, '-m', g3_mask_file,
                     '-r', str(res), '-N', str(nsig_per), '-S', 3,
                     '-s', 0.5 * thick2] # , '-v']
        mcf_graph.main(main_args)

        # Load GraphMCF
        print '\tPROCESS[' + str(pr_id) + ']: Unpickling the graph: ' + g3_graph_pkl
        graph_mcf = unpickle_obj(g3_graph_pkl)

        # Make topological simplification until fitting the number of features times residues percentage
        print '\tPROCESS[' + str(pr_id) + ']: Simplifying the graph...'
        graph_mcf.topological_simp(0, n=grid.get_num_features()*res_per, prop_ref=STR_FIELD_VALUE)
        # Filter edges until fitting the number of edges
        # n_edges = float(len(graph_mcf.get_vertices_list())) * ev_ratio
        # i_edges = len(graph_mcf.get_edges_list())
        # if n_edges < i_edges:
        n_edges = int(math.ceil(grid.get_num_edges()*res_per))
        print '\t\t-Number of edges to keep: ' + str(n_edges)
        graph_mcf.threshold_edges_n(n_edges, STR_FIELD_VALUE, mode='low')
        # else:
        #     print 'WARNING [process:' + str(pr_id) + ']: Graph cannot be simplify to ' + str(n_edges) + \
        #           ' edges, since it has only ' + str(i_edges)
        disperse_io.save_vtp(graph_mcf.get_vtp(av_mode=True, edges=True), g3_graph_vtp)
        disperse_io.save_vtp(graph_mcf.get_scheme_vtp(nodes=True, edges=True), g3_graph_sch)
        graph_mcf.pickle(g3_graph_pkl)

    if pr_id < 0:
        return -1
    else:
        print '\tFinishing PROCESS[' + str(pr_id) + '] successfully!'
        return pr_id

#################################################################################################
# Main Routine
#################################################################################################

print 'Evaluating GraphMCF performance with synthetic data.'
print '\tAuthor: ' + __author__
print '\tDate: ' + time.strftime("%c") + '\n'
print 'Running main loop:'

hold_snr = (-1) * np.zeros(shape=(G3_NUM_REP, len(G3_STD_NOISE)), dtype=np.float)
hold_t_p = (-1) * np.zeros(shape=(G3_NUM_REP, len(G3_STD_NOISE)), dtype=np.float)
hold_f_p = (-1) * np.ones(shape=(G3_NUM_REP, len(G3_STD_NOISE)), dtype=np.float)
hold_f_n = (-1) * np.ones(shape=(G3_NUM_REP, len(G3_STD_NOISE)), dtype=np.float)
hold_p_e = (-1) * np.zeros(shape=(G3_NUM_REP, len(G3_STD_NOISE)), dtype=np.float)
hold_pp_e = (-1) * np.zeros(shape=(G3_NUM_REP, len(G3_STD_NOISE)), dtype=np.float)
hold_b_e = (-1) * np.zeros(shape=(G3_NUM_REP, len(G3_STD_NOISE)), dtype=np.float)
hold_snr = np.zeros(shape=(G3_NUM_REP, len(G3_STD_NOISE)), dtype=np.float)
thick = G3_SP * G3_FEAT
thick2 = 2 * math.ceil(thick * G3_RESOLUTION)

# Noise loop for computing the graphs
for j in range(len(G3_STD_NOISE)):

    print '\tProcessing noise entry: ' + str(j) + ' of ' + str(len(G3_STD_NOISE))

    if G3_GRAPHMCF_PKL:

        print '\tGenerating grids and graphs...'
        # MULTIPROCESSING
        if G3_NPR <= 1:
            pr_mcf_graph(0, range(G3_NUM_REP), j, MCF_OUT_DIR, G3_RESOLUTION, G3_L, G3_EPS, G3_SP, G3_EV,
                         G3_NSIG_PER, G3_RES_PER, G3_EDG_VER_RATIO)
        else:
            processes = list()
            spl_ids = np.array_split(range(G3_NUM_REP), G3_NPR)
            for pr_id, ids in zip(range(G3_NPR), spl_ids):
                pr = mp.Process(target=pr_mcf_graph, args=(pr_id, ids, j, MCF_OUT_DIR, G3_RESOLUTION,
                                                           G3_L, G3_EPS, G3_SP,
                                                           G3_EV, G3_NSIG_PER, G3_RES_PER, G3_EDG_VER_RATIO))
                pr.start()
                processes.append(pr)
            pr_results = list()
            for pr in processes:
                pr.join()
                pr_results.append(pr.exitcode)
            # for pr_id in range(len(processes)):
            #     if pr_id != pr_results[pr_id]:
            #         error_msg = 'Process ' + str(pr_id) + ' exited unexpectedly!'
            #         sys.exit(-1)
            print '\t\t-All processes are finished!'
    else:
        print '\t\t-WARNING: tomograms are loaded from a previous running instance so some settings may not fit!'

    print '\tGraphs computed!'

    # Repetitions loop
    for i in range(G3_NUM_REP):

        # Paths
        g3_tomo_file = MCF_OUT_DIR + '/grid_noise_std_' + str(j) + '_it_' + str(i) + '.fits'
        g3_grid_pkl = MCF_OUT_DIR + '/grid_' + str(j) + '_it_' + str(i) + '.pkl'
        mcf_out_dir2 = MCF_OUT_DIR + '/graph_' + str(j) + '_it_' + str(i)
        g3_graph_pkl = mcf_out_dir2 + '/' + os.path.splitext(os.path.split(g3_tomo_file)[1])[0] + '.pkl'

        # Synthetic phantom generation
        print '\tLoading the grid with STD=' + str(G3_STD_NOISE[j]) + ' and repetition ' + str(i) + '.'
        grid = unpickle_obj(g3_grid_pkl)
        hold_snr[i, j] = grid.get_snr()

# Noise loop for data analysis
for j in range(len(G3_STD_NOISE)):

    # Repetitions loop
    for i in range(G3_NUM_REP):

        # Paths
        g3_tomo_file = MCF_OUT_DIR + '/grid_noise_std_' + str(j) + '_it_' + str(i) + '.fits'
        g3_mask_file = MCF_OUT_DIR + '/grid_mask_std_' + str(j) + '_it_' + str(i) + '_mask.fits'
        g3_graph_vtp = MCF_OUT_DIR + '/grid_graph_skel_' + str(j) + '_it_' + str(i) + '.vtp'
        g3_graph_sch = MCF_OUT_DIR + '/grid_graph_sch_' + str(j) + '_it_' + str(i) + '.vtp'
        g3_grid_pkl = MCF_OUT_DIR + '/grid_' + str(j) + '_it_' + str(i) + '.pkl'
        mcf_out_dir2 = MCF_OUT_DIR + '/graph_' + str(j) + '_it_' + str(i)
        g3_graph_pkl = mcf_out_dir2 + '/' + os.path.splitext(os.path.split(g3_tomo_file)[1])[0] + '.pkl'
        g3_dog_file = MCF_OUT_DIR + '/grid_noise_std_' + str(j) + '_it_' + str(i) + '_dog.mrc'

        # print '\tComputing DoG metrics...'
        # n_points_dog = grid.get_num_features() * G3_RES_PER
        # g3_tomo_dog = dog_operator(lin_map(g3_tomo_file, lb=1, ub=0), DOG_SG1, DOG_K)
        # disperse_io.save_numpy(g3_tomo_dog, g3_dog_file)
        # g3_dog_peaks = find_region_peaks_by_num(g3_tomo_dog, n_points_dog)
        # labels_dog, v_tps_dog = list(), list()
        # gxs, gys, gzs = grid.get_grid_points()
        # grid_vs = -1 * np.ones(shape=gxs.shape, dtype=np.int)
        # good_picks_dog, bad_picks_dog, fp_picks_dog, tot_picks_dog = 0., 0., 0., float(np.asarray(gxs.shape).prod())
        # for x in range(gxs.shape[0]):
        #     for y in range(gys.shape[1]):
        #         for z in range(gzs.shape[2]):
        #             hold_min = np.finfo(np.float).max
        #             g_point = np.asarray((gxs[x, y, z], gys[x, y, z], gzs[x, y, z]), dtype=np.float32)
        #             pick_found = False
        #             for point in g3_dog_peaks:
        #                 dst = g_point - point
        #                 dst = math.sqrt((dst*dst).sum())
        #                 if hold_min > dst:
        #                     if dst < G3_EPS:
        #                         grid_vs[x, y, z] = int(v.get_id())
        #                         pick_found = True
        #                     hold_min = dst
        #             if pick_found:
        #                 good_picks_dog += 1
        #             else:
        #                 bad_picks_dog += 1
        # # Computing the false positives
        # for point in g3_dog_peaks:
        #     if not grid.in_feature(point, G3_EPS):
        #         fp_picks_dog += 1

        print '\tUnpickling GraphMCF...'
        graph_mcf = unpickle_obj(g3_graph_pkl)

        # Getting the starting points for tracing
        labels, v_tps = list(), list()
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
                        v_tps.append(grid_vs[x, y, z])
                        good_picks += 1
                    else:
                        bad_picks += 1
        # Computing the false positives
        v_fps = list()
        for v in vertices:
            point = np.asarray(graph_mcf.get_vertex_coords(v), dtype=np.float32)
            if not grid.in_feature(point, G3_EPS):
                fp_picks += 1
                v_fps.append(int(v.get_id()))

        # Trace the path to closest neighbours
        pairs = list()
        graph_mcf.compute_graph_gt()
        tot_path, good_path, bad_path = 0., 0., 0.
        for x in range(gxs.shape[0]):
            for y in range(gys.shape[1]):
                for z in range(gzs.shape[2]):
                    # Get starting vertex and neighbours (target vertices)
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
                        # Check paths
                        neighbours = [nx0, nx1, ny0, ny1, nz0, nz1]
                        for t in v_tps:
                            if not ((str(v) + '_' + str(t)) in pairs):
                                pairs.append(str(v) + '_' + str(t))
                                pairs.append(str(t) + '_' + str(v))
                                if t in neighbours:
                                    tot_path += 1
                                    # Computing the arc path
                                    v_path, e_path = graph_mcf.find_shortest_path(v, t, prop_key=SGT_EDGE_LENGTH)
                                    found = False
                                    if v_path is not None:
                                        if len(v_path) > 2:
                                            for p in v_path[1:-1]:
                                                if p in neighbours:
                                                    found = True
                                                    break
                                    else:
                                        found = True
                                    if not found:
                                        good_path += 1
                                else:
                                    v_path, e_path = graph_mcf.find_shortest_path(v, t, prop_key=SGT_EDGE_LENGTH)
                                    found = True
                                    if v_path is not None:
                                        if len(v_path) >= 2:
                                            for p in v_path[1:]:
                                                hold_point = graph_mcf.get_vertex_coords(graph_mcf.get_vertex(p))
                                                if not grid.in_grid(hold_point):
                                                    found = False
                                                    break
                                                if (p in neighbours) or (p in v_tps):
                                                    break
                                        # if found:
                                        #     for e in e_path:
                                        #         hold_point = graph_mcf.get_edge_coords(graph_mcf.get_edge(e))
                                        #         if not grid.in_grid(hold_point):
                                        #             found = False
                                        #             break
                                    if not found:
                                        bad_path += 1

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
            hold_pp_e[i, j] = 0
            hold_b_e[i, j] = 0
        else:
            # hold_p_e[i, j] = good_path / tot_path
            hold_pp_e[i, j] = good_path / tot_path
            hold_p_e[i, j] = good_path / grid.get_num_edges()
            # if (bad_path_fp+good_path_fp) > 0:
            if bad_path > 0:
                # hold_b_e[i, j] = bad_path_fp / (bad_path_fp + good_path_fp)
                hold_b_e[i, j] = bad_path / grid.get_num_edges()
            else:
                hold_b_e[i, j] = 0

    # Printing the result
    print ''
    print '\tRESULTS: '
    print '\tSNR: [' + str(hold_snr[:, j].min()) + ', ' + str(hold_snr[:, j].mean()) + ', ' + str(hold_snr[:, j].max()) + ']'
    print '\tTrue positive picked: [' + str(hold_t_p[:, j].min()) + ', ' + str(hold_t_p[:, j].mean()) + ', ' + str(hold_t_p[:, j].max()) + ']'
    print '\tFalse positive picked: [' + str(hold_f_p[:, j].min()) + ', ' + str(hold_f_p[:, j].mean()) + ', ' + str(hold_f_p[:, j].max()) + ']'
    print '\tFalse negative picked: [' + str(hold_f_n[:, j].min()) + ', ' + str(hold_f_n[:, j].mean()) + ', ' + str(hold_f_n[:, j].max()) + ']'
    print '\tFraction of correctly tracked paths over corrected ground truth: [' + str(hold_pp_e[:, j].min()) + ', ' + str(hold_pp_e[:, j].mean()) + ', ' + str(hold_pp_e[:, j].max()) + ']'
    print '\tFraction of correctly tracked paths: [' + str(hold_p_e[:, j].min()) + ', ' + str(hold_p_e[:, j].mean()) + ', ' + str(hold_p_e[:, j].max()) + ']'
    print '\tFraction of bad tracked paths: [' + str(hold_b_e[:, j].min()) + ', ' + str(hold_b_e[:, j].mean()) + ', ' + str(hold_b_e[:, j].max()) + ']'

# Storing the results
np.savez(MCF_OUT_DIR + '/grid3d_arrays.npz', hold_snr, hold_t_p, hold_p_e, hold_f_p, hold_f_n, hold_b_e)

# Plotting the results
snr_mean = np.mean(hold_snr, axis=0)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.errorbar(snr_mean, np.mean(hold_t_p, axis=0), yerr=np.std(hold_t_p, axis=0)/math.sqrt(G3_NUM_REP), color='blue', linestyle='-', linewidth=1.5, label='TP')
# plt.plot(snr_mean, np.mean(hold_t_p, axis=0), color='blue', linestyle='-', linewidth=2, label='TP')
# plt.fill_between(snr_mean, np.mean(hold_t_p, axis=0)-.5*np.std(hold_t_p, axis=0)/math.sqrt(G3_NUM_REP), np.mean(hold_t_p, axis=0)+.5*np.std(hold_t_p, axis=0)/math.sqrt(G3_NUM_REP), alpha=0.5, color='blue', edgecolor='w')
plt.errorbar(snr_mean, np.mean(hold_pp_e, axis=0), yerr=np.std(hold_pp_e, axis=0)/math.sqrt(G3_NUM_REP), color='red', linestyle='-.', linewidth=1.5, label='TP arcs corrected')
plt.errorbar(snr_mean, np.mean(hold_p_e, axis=0), yerr=np.std(hold_p_e, axis=0)/math.sqrt(G3_NUM_REP), color='red', linestyle='-', linewidth=1.5, label='TP arcs')
# plt.plot(snr_mean, np.mean(hold_p_e, axis=0), color='red', linestyle='-', linewidth=2, label='TP arcs')
# plt.fill_between(snr_mean, np.mean(hold_p_e, axis=0)-.5*np.std(hold_p_e, axis=0)/math.sqrt(G3_NUM_REP), np.mean(hold_p_e, axis=0)+.5*np.std(hold_p_e, axis=0)/math.sqrt(G3_NUM_REP), alpha=0.5, color='red', edgecolor='w')
plt.errorbar(snr_mean, np.mean(hold_f_p, axis=0), yerr=np.std(hold_f_p, axis=0)/math.sqrt(G3_NUM_REP), color='yellow', linestyle='-', linewidth=1.5, label='FP')
# plt.plot(snr_mean, np.mean(hold_f_p, axis=0), color='yellow', linestyle='-', linewidth=2, label='FP')
# plt.fill_between(snr_mean, np.mean(hold_f_p, axis=0)-.5*np.std(hold_f_p, axis=0)/math.sqrt(G3_NUM_REP), np.mean(hold_f_p, axis=0)+.5*np.std(hold_f_p, axis=0)/math.sqrt(G3_NUM_REP), alpha=0.5, color='yellow', edgecolor='w')
plt.errorbar(snr_mean, np.mean(hold_f_n, axis=0), yerr=np.std(hold_f_n, axis=0)/math.sqrt(G3_NUM_REP), color='cyan', linestyle='-', linewidth=1.5, label='FN')
# plt.plot(snr_mean, np.mean(hold_f_n, axis=0), color='cyan', linestyle='-', linewidth=2, label='FN')
# plt.fill_between(snr_mean, np.mean(hold_f_n, axis=0)-0.5*np.std(hold_f_n, axis=0)/math.sqrt(G3_NUM_REP), np.mean(hold_f_n, axis=0)+0.5*np.std(hold_f_n, axis=0)/math.sqrt(G3_NUM_REP), alpha=0.5, color='cyan', edgecolor='w')
plt.errorbar(snr_mean, np.mean(hold_b_e, axis=0), yerr=np.std(hold_b_e, axis=0)/math.sqrt(G3_NUM_REP), color='magenta', linestyle='-', linewidth=1.5, label='FP arcs')
# plt.plot(snr_mean, np.mean(hold_b_e, axis=0), color='magenta', linestyle='-', linewidth=2, label='FP arcs')
# plt.fill_between(snr_mean, np.mean(hold_b_e, axis=0)-0.5*np.std(hold_b_e, axis=0)/math.sqrt(G3_NUM_REP), np.mean(hold_b_e, axis=0)+0.5*np.std(hold_b_e, axis=0)/math.sqrt(G3_NUM_REP), alpha=0.5, color='magenta', edgecolor='w')
plt.plot(snr_mean, np.ones(shape=hold_p_e.shape[1]), color='black', linestyle='--', linewidth=1)
plt.plot(snr_mean, np.zeros(shape=hold_p_e.shape[1]), color='black', linestyle='--', linewidth=1)
plt.ylim((-0.1, 1.1))
plt.xlim((snr_mean[0], snr_mean[-1]))
ax.set_xscale('log')
plt.xlabel('SNR')
plt.legend(loc=7)
# ref_FR_ssup_mb.sh
plt.tight_layout()
plt.savefig(MCF_OUT_DIR + '/test_grid_conn.png', dpi=600)
plt.close()

print 'Terminated successfully. (' + time.strftime("%c") + ')'
