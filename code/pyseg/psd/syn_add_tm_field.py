"""

    Script for inserting template matching results (scores y angles) in a SynGraphMCF

    Input:  - SynGraphMCF
            - Template matching output(s) (Pytom)
            - Refinement parameters

    Output: - Graph with template matching information inserted as vertex and edge properties

"""

__author__ = 'Antonio Martinez-Sanchez'

# ################ Package import

import time
import numpy as np
import pyseg as ps
import os
try:
    import pickle as pickle
except:
    import pickle

########## Global variables

# Membrane segmentation: 1-mb, 2-cito, 3-ext
SEG_MB = 1
SEG_MB_IN = 2
SEG_MB_OUT = 3
SVOL_OFF = 3

RD_2_DG = 180./np.pi

########################################################################################
# PARAMETERS
########################################################################################

####### Input data

ROOT_PATH = '/home/martinez/workspace/disperse/data/psd_an1'

# Input graph
in_graph = ROOT_PATH+'/ex/syn/graphs_2/syn_14_9_bin2_rot_crop2.pkl'

# Input template matching scores
in_scores = (ROOT_PATH + '/in/zd/bin2/syn_14_9_bin2_rot_crop2_tm_ampar_gcc.fits',
             ROOT_PATH + '/in/zd/bin2/syn_14_9_bin2_rot_crop2_tm_nmdar_gcc.fits',
             ROOT_PATH + '/in/zd/bin2/syn_14_9_bin2_rot_crop2_tm_camkii_gcc.fits'
             )

# Input template matching angles
in_phis = (ROOT_PATH + '/in/zd/bin2/syn_14_9_bin2_rot_crop2_tm_ampar_phi.fits',
             ROOT_PATH + '/in/zd/bin2/syn_14_9_bin2_rot_crop2_tm_nmdar_phi.fits',
             ROOT_PATH + '/in/zd/bin2/syn_14_9_bin2_rot_crop2_tm_camkii_phi.fits'
             )
in_psis = (ROOT_PATH + '/in/zd/bin2/syn_14_9_bin2_rot_crop2_tm_ampar_psi.fits',
             ROOT_PATH + '/in/zd/bin2/syn_14_9_bin2_rot_crop2_tm_nmdar_psi.fits',
             ROOT_PATH + '/in/zd/bin2/syn_14_9_bin2_rot_crop2_tm_camkii_psi.fits'
             )
in_thes = (ROOT_PATH + '/in/zd/bin2/syn_14_9_bin2_rot_crop2_tm_ampar_the.fits',
             ROOT_PATH + '/in/zd/bin2/syn_14_9_bin2_rot_crop2_tm_nmdar_the.fits',
             ROOT_PATH + '/in/zd/bin2/syn_14_9_bin2_rot_crop2_tm_camkii_the.fits'
             )

# List of template normals
t_normals = ([0, 0, 1],
             [0, 0, 1],
             [0, 0, 1] # TODO: for CamKII this is not reliable
             )

# Template names
t_names = ('ampar',
           'nmdar',
           'camkii'
           )


####### Output data

output_dir = ROOT_PATH+'/ex/syn/tm/graphs_2'
store_lvl = 2

####### Parameters

diam_nhood = 10 # Neighborhood diameter (nm)

########################################################################################
# MAIN ROUTINE
########################################################################################

# Print initial message
print('Inserting template matching output to GraphMCF properties.')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('Input graphs: ' + str(in_graph))
print('Input scores: ' + str(in_scores))
print('Input angles phi: ' + str(in_phis))
print('Input angles psi: ' + str(in_psis))
print('Input angles theta: ' + str(in_thes))
print('Output directory: ' + str(output_dir))
print('Options for templates:')
print('\tNeighborhood diameter: ' + str(diam_nhood) + ' nm')
print('\tStore level: ' + str(store_lvl))
print('')

print('\tProcessing the input graph: ' + in_graph)

print('\tUnpicking graph...')
path, fname = os.path.split(in_graph)
stem_name, _ = os.path.splitext(fname)
graph = ps.factory.unpickle_obj(in_graph)

print('Template matching loop:')
for (in_score, in_phi, in_psi, in_the, t_name, t_normal) in zip(in_scores, in_phis, in_psis, in_thes,
                                                                t_names, t_normals):

    print('\t\tAdding cross-correlation map with name ' + t_name + ': ' + str((in_score, in_phi, in_psi, in_the)))
    print('\t\t\t-Normal: ' + str(t_normal))

    print('\t\tLoading input tomograms...')
    scores = ps.disperse_io.load_tomo(in_score)
    phi = ps.disperse_io.load_tomo(in_phi) * RD_2_DG
    psi = ps.disperse_io.load_tomo(in_psi) * RD_2_DG
    the = ps.disperse_io.load_tomo(in_the) * RD_2_DG

    print('\t\tAdding scores in angles (as rotated template normal)...')
    graph.add_tm_field_eu(t_name, scores, phi, psi, the, t_normal, diam_nhood)

out_pkl = output_dir +  '/' + fname
print('\t\tPickling the graph as: ' + out_pkl)
graph.pickle(out_pkl)

print('\tSaving graphs at level ' + str(store_lvl) + '...')
if store_lvl > 0:
    ps.disperse_io.save_vtp(graph.get_vtp(av_mode=True, edges=True),
                            output_dir + '/' + stem_name + '_edges.vtp')
if store_lvl > 1:
    ps.disperse_io.save_vtp(graph.get_vtp(av_mode=False, edges=True),
                                output_dir + '/' + stem_name + '_edges_2.vtp')
if store_lvl > 2:
    ps.disperse_io.save_vtp(graph.get_scheme_vtp(nodes=True, edges=True),
                            output_dir + '/' + stem_name + '_sch.vtp')

print('Successfully terminated. (' + time.strftime("%c") + ')')
