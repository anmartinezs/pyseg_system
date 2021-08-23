"""

    Script for inserting template matching results (scores y angles) in a GraphMCF

    Input:  - GraphMCF
            - Template matching output (Pytom)
            - Refinement parameters

    Output: - Graph with template matching information inserted as vertex and edge properties

"""

__author__ = 'Antonio Martinez-Sanchez'

# ################ Package import

import time
import pyseg as ps
import scipy as sp
import os
import csv
import sys
import numpy as np
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

ROOT_PATH = '/home/martinez/workspace/disperse/data/chlamy'

# Input graphs
in_graphs = (ROOT_PATH + '/g/T1L1_b4_M1_den.pkl',
             )

# Input template matching scores
in_scores = (ROOT_PATH + '/s/T1L1_M1_crop_b4_b6f_cc.fits',
             ROOT_PATH + '/s/T1L1_M1_crop_b4_atp_cc.fits',
             ROOT_PATH + '/s/T1L1_M1_crop_b4_ribo_cc.fits'
             )

# Input template matching angles
in_angles = (ROOT_PATH + '/s/T1L1_M1_crop_b4_b6f_ang.fits',
             ROOT_PATH + '/s/T1L1_M1_crop_b4_atp_ang.fits',
             ROOT_PATH + '/s/T1L1_M1_crop_b4_ribo_ang.fits'
             )

# List of template normals
t_normals = ([0, 0, 1],
             [0, 0, 1],
             [0, 0, 1]
             )

# Template names
t_names = ('b6f',
           'atp',
           'ribo'
           )

# List of angles
in_ang_lut = '/home/martinez/pool/bmsan-apps/pytom/testing/hpc/pytom/angles/angleLists/angles_12.85_7112.em'


####### Output data

output_dir = '/home/martinez/workspace/disperse/data/chlamy/field'
store_lvl = 2

####### Parameters

diam_nhood = 3 # Neighborhood diameter (nm)
off_set = (0, 0, 0)
binn = 1.

########################################################################################
# MAIN ROUTINE
########################################################################################

# Print initial message
print('Inserting template matching output to GraphMCF properties.')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('Input graphs: ' + str(in_graphs))
print('Input scores: ' + str(in_scores))
print('Input angles: ' + str(in_angles))
print('Output directory: ' + str(output_dir))
print('List for the angles: ' + str(in_ang_lut))
print('Options for templates:')
print('\tNeighborhood diameter: ' + str(diam_nhood) + ' nm')
print('\tStore level: ' + str(store_lvl))
print('')

if in_ang_lut is not None:
    print('Loading angles list...')
    ang_lut = ps.disperse_io.load_tomo(in_ang_lut)

print('Main loop:')
for in_graph in in_graphs:

    print('\tProcessing the input graph: ' + in_graph)

    print('\tUnpicking graph...')
    path, fname = os.path.split(in_graph)
    stem_name, _ = os.path.splitext(fname)
    graph = ps.factory.unpickle_obj(in_graph)

    print('Template matching loop:')
    for (in_score, in_angle, t_name, t_normal) in zip(in_scores, in_angles, t_names, t_normals):

        if in_angle is None:
            print('\t\tAdding cross-correlation map without angles ' + t_name + ': ' + str(in_score))

            print('\t\tLoading input tomogram...')
            scores = ps.disperse_io.load_tomo(in_score)

            print('\t\tAdding scores in angles (as rotated template normal)...')
            graph.add_scalar_field(scores, t_name, neigh=diam_nhood, mode='max', offset=off_set, bin=binn)

        else:
            print('\t\tAdding cross-correlation map with name ' + t_name + ': ' + str((in_score, in_angle)))
            print('\t\t\t-Normal: ' + str(t_normal))

            print('\t\tLoading input tomograms...')
            scores = ps.disperse_io.load_tomo(in_score)
            angles = ps.disperse_io.load_tomo(in_angle).astype(np.int)
            ang_lut = ps.disperse_io.load_tomo(in_ang_lut)

            print('\t\tAdding scores in angles (as rotated template normal)...')
            graph.add_tm_field(t_name, scores, angles, ang_lut*RD_2_DG, t_normal, diam_nhood, off_set, binn)

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
