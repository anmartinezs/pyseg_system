"""

    Script for filter a MbGraphMCF object by extracting a filament network

    Input:  - A STAR file with a list of graphs (MbGraphMCF) as it is returned by
              by mb_graph.py script:
            	+ Density map tomogram
            	+ Segmentation tomogram
                + MbGraphMCF object
            - XML files for segmenting source and target vertices
            - Filament settings

    Output: - A STAR file with filtered version of the input MbGraphMCF with filaments network
            - Additional files for visualization

"""

__author__ = 'Antonio Martinez-Sanchez'

# ################ Package import
import argparse
import gc
import os
import time
import pyseg as ps
import sys
try:
    import pickle as pickle
except:
    import pickle

########## Global variables

########################################################################################
# PARAMETERS
########################################################################################

####### Input data

ROOT_PATH = '../' # Data path

# Input STAR file with the GraphMCF pickles
in_star = ROOT_PATH + 'graphs/test_mb_graph1.star' # The outuput of mb_graph.py

# Sources slice XML file
in_sources = ROOT_PATH + 'tracing/fils/in/mb_sources.xml'

# Targets slice XML file
in_targets =  ROOT_PATH + 'tracing/fils/in/no_mb_targets.xml'

####### Output data

out_dir = ROOT_PATH + 'fils/out' # '/fils/cyto'
out_int_g = True

####### Filament geometrical parameters

g_rg_len = [1, 60] # [1, 50] # [1, 75] # nm (>0)
g_rg_sin = [0, 3]
g_rg_eud = [1, 25] # [1, 15] # [1, 25] # nm (>0)

####### Graph thresholding (Advanced parameters)

th_mode = 'in' # 'out'

########################################################################################
# MAIN ROUTINE
########################################################################################

# Get them from the command line if they were passed through it
parser = argparse.ArgumentParser()
parser.add_argument('--inStar', default=in_star, help='Input star file.')
parser.add_argument('--outDir', default=out_dir, help='Output subtomograms directory.')
parser.add_argument('--inSources', default=in_sources, help='Sources xml file.')
parser.add_argument('--inTargets', default=in_targets, help='Targets xml file.')
parser.add_argument('--thMode', default=th_mode, choices=['in', 'out'],
                    help='Orientation with respect to the membrane/filament.')
parser.add_argument('--gRgLen', nargs='+', type=int, default=g_rg_len,
                    help='Geodesic distance trough the graph between source and target vertices in nm.')
parser.add_argument('--gRgSin', nargs='+', type=int, default=g_rg_sin,
                    help='Filament sinuosity, geodesic/euclidean distances ratio.')
parser.add_argument('--gRgEud', nargs='+', type=int, default=g_rg_eud,
                    help='Euclidean distance between source and target vertices in nm.')
args = parser.parse_args()

in_star = args.inStar
out_dir = args.outDir
in_sources = args.inSources
in_targets = args.inTargets
th_mode = args.thMode
g_rg_len = args.gRgLen
g_rg_sin = args.gRgSin
g_rg_eud = args.gRgEud

# Print initial message
print(f'{time.strftime( "%Y-%m-%d %H:%M:%S", time.localtime() )} Filtering a MbGraphMCF by filaments network.', file=sys.stdout, flush=True)
print(f'\tAuthor: ' + __author__, file=sys.stdout, flush=True)
print(f'\tDate: ' + time.strftime("%c") + '\n', file=sys.stdout, flush=True)
print(f'Options:', file=sys.stdout, flush=True)
print(f'\tInput SynGraphMCF STAR file: ' + str(in_star), file=sys.stdout, flush=True)
print(f'\tSources slice file: ' + in_sources, file=sys.stdout, flush=True)
print(f'\tTargets slice file: ' + in_targets, file=sys.stdout, flush=True)
print(f'\tOutput directory: ' + out_dir, file=sys.stdout, flush=True)
if out_int_g:
    print(f'\t\t-Intermediate graph will also be stored.', file=sys.stdout, flush=True)
print(f'\tFilament geometrical parameters: ', file=sys.stdout, flush=True)
print(f'\t\t-Lengths range: ' + str(g_rg_len) + ' nm', file=sys.stdout, flush=True)
print(f'\t\t-Euclidean distance range: ' + str(g_rg_eud) + ' nm', file=sys.stdout, flush=True)
if g_rg_sin:
    print(f'\t\t-Sinuosity range (inclusive): ' + str(g_rg_sin), file=sys.stdout, flush=True)
else:
    print(f'\t\t-Sinuosity range (exclusive): ' + str(g_rg_sin), file=sys.stdout, flush=True)
print(f'', file=sys.stdout, flush=True)

# Getting stem strings
s_stem = os.path.splitext(os.path.split(in_sources)[1])[0]
t_stem = os.path.splitext(os.path.split(in_targets)[1])[0]

print(f'{time.strftime( "%Y-%m-%d %H:%M:%S", time.localtime() )} Paring input star file...', file=sys.stdout, flush=True)
star = ps.sub.Star()
star.load(in_star)
in_graph_l = star.get_column_data('_psGhMCFPickle')
star.add_column('_psGhMCFPickle')
del_rows = list()

# Loop for processing the input data
print(f'{time.strftime( "%Y-%m-%d %H:%M:%S", time.localtime() )} Running main loop: ', file=sys.stdout, flush=True)
for row, in_graph in enumerate(in_graph_l):

    print(f'\t{time.strftime( "%Y-%m-%d %H:%M:%S", time.localtime() )} Loading input pickle: ' + in_graph, file=sys.stdout, flush=True)
    graph = ps.factory.unpickle_obj(in_graph)
    graph.compute_vertices_dst()
    graph.compute_vertices_fwdst()

    print(f'\t{time.strftime( "%Y-%m-%d %H:%M:%S", time.localtime() )} Compute GraphGT...', file=sys.stdout, flush=True)
    graph_gtt = ps.graph.GraphGT(graph)
    graph_gt = graph_gtt.get_gt()

    print(f'\t{time.strftime( "%Y-%m-%d %H:%M:%S", time.localtime() )} Processing sources slice: ' + in_sources, file=sys.stdout, flush=True)
    s_slice = ps.xml_io.SliceSet(in_sources)
    try:
        _, s_ids, _ = graph.get_cloud_mb_slice(s_slice.get_slices_list()[0], cont_mode=False, graph_gt=graph_gt)
    except ValueError:
        print(f'WARNING: no points found in the slice, this row will be deleted from the STAR file!', file=sys.stdout, flush=True)
        del_rows.append(row)
        continue

    print(f'\t{time.strftime( "%Y-%m-%d %H:%M:%S", time.localtime() )} Processing targets slice: ' + in_targets, file=sys.stdout, flush=True)
    t_slice = ps.xml_io.SliceSet(in_targets)
    try:
        _, t_ids, _ = graph.get_cloud_mb_slice(t_slice.get_slices_list()[0], cont_mode=False, graph_gt=graph_gt)
    except ValueError:
        print(f'WARNING: no points found in the slice, this row will be deleted from the STAR file!', file=sys.stdout, flush=True)
        del_rows.append(row)
        continue

    print(f'\t{time.strftime( "%Y-%m-%d %H:%M:%S", time.localtime() )} Building the filament network in lengths range ' + str(g_rg_len) + ' nm and sinuosity range ' + \
          str(g_rg_sin) + '...', file=sys.stdout, flush=True)
    net = graph.gen_fil_network(s_ids, t_ids, g_rg_len, g_rg_sin, graph_gt=graph_gt, key_l=ps.globals.SGT_EDGE_LENGTH)

    print(f'\t{time.strftime( "%Y-%m-%d %H:%M:%S", time.localtime() )} Filtering filaments by Euclidean distance between extrema with range ' + str(g_rg_eud) + ' nm', file=sys.stdout, flush=True)
    net.filter_by_eud(g_rg_eud[0], g_rg_eud[1], inc=True)

    print(f'\t{time.strftime( "%Y-%m-%d %H:%M:%S", time.localtime() )} Filtering the Graph...', file=sys.stdout, flush=True)
    net.filter_graph_mcf(graph, mode=th_mode)

    print(f'\t{time.strftime( "%Y-%m-%d %H:%M:%S", time.localtime() )} Adding filamentness to the Graph...', file=sys.stdout, flush=True)
    net.add_vprop_graph_mcf(graph, key_p=ps.filament.globals.STR_FIL_FNESS, mode='max')

    print(f'\t{time.strftime( "%Y-%m-%d %H:%M:%S", time.localtime() )} Computing output paths...', file=sys.stdout, flush=True)
    p_stem = os.path.splitext(os.path.split(in_graph)[1])[0]
    out_stem = out_dir + '/' + p_stem + '_fil_' + s_stem + '_to_' + t_stem + '_net'

    out_pkl = out_stem + '.pkl'
    print(f'\t{time.strftime( "%Y-%m-%d %H:%M:%S", time.localtime() )} Storing filtered graph as: ' + out_pkl, file=sys.stdout, flush=True)
    graph.pickle(out_pkl)
    star.set_element('_psGhMCFPickle', row, out_pkl)

    if out_int_g == 1:
        print(f'\t{time.strftime( "%Y-%m-%d %H:%M:%S", time.localtime() )} Storing intermediate graphs...', file=sys.stdout, flush=True)
        ps.disperse_io.save_vtp(graph.get_vtp(av_mode=True, edges=True), out_stem + '_edges.vtp')
        ps.disperse_io.save_vtp(graph.get_vtp(av_mode=True, edges=True), out_stem + '_edges2.vtp')
        # ps.disperse_io.save_vtp(graph.get_scheme_vtp(nodes=True, edges=True), out_stem + '_sch.vtp')

    print(f'\t{time.strftime( "%Y-%m-%d %H:%M:%S", time.localtime() )} Storing filament network in: ' + out_stem + '.vtp', file=sys.stdout, flush=True)
    ps.disperse_io.save_vtp(net.get_vtp(), out_stem+'.vtp')

    gc.collect()

if len(del_rows) > 0:
    print(f'\t{time.strftime( "%Y-%m-%d %H:%M:%S", time.localtime() )} Deleting empty rows...', file=sys.stdout, flush=True)
    star.del_rows(del_rows)

out_star = out_dir + '/fil_' + s_stem + '_to_' + t_stem + '_net.star'
print(f'\t{time.strftime( "%Y-%m-%d %H:%M:%S", time.localtime() )} Storing output STAR file in: ' + out_star, file=sys.stdout, flush=True)
star.store(out_star)

print(f'Terminated {os.path.basename(__file__)}. ({time.strftime( "%Y-%m-%d %H:%M:%S", time.localtime() )})', file=sys.stdout, flush=True)
