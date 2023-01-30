"""

    Add segmentation labels to GraphMCF objects in a STAR file which pairs them with the segmentation

    Input:  - The STAR file

    Output: - New graph are pickled with the segmentation information

"""

__author__ = 'Antonio Martinez-Sanchez'

# ################ Package import

import time
import pyseg as ps
import scipy as sp
import os
import sys
import numpy as np
try:
    import pickle as pickle
except:
    import pickle

########## Global variables

########################################################################################
# PARAMETERS
########################################################################################

####### Input data

ROOT_PATH = '/fs/pool/pool-lucic2'

# Input STAR file with segmentations
in_star = ROOT_PATH + '/antonio/workspace/psd_an/in/syn_seg_ves_glur_graph.star'
in_g_star = ROOT_PATH + '/antonio/workspace/psd_an/in/syn_seg_glur.star'

####### Output data

out_dir = ROOT_PATH+'/ex/syn/graphs_ves'
out_sufix = 'ves'

####### Graph settings

gh_pname = 'syn_seg'
gh_clean = False

####### Segmentation pre-processing

sg_th = 8 # > is considered fg, if None segmentation labels are considered
sg_lbl = 6 # label for be set on the graph, only applicable if sg_th is not None
sg_ref = True # if True then segmentation is done in 'rlnMircographName' tomogram space, False for 'psSegImage' space
sg_close = 1

########################################################################################
# MAIN ROUTINE
########################################################################################

# Print initial message
print('Adding segmentation to GraphMCF objects.')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('Options:')
# print '\tDisPerSe persistence threshold (nsig): ' + str(nsig)
print('\tSTAR file with the GraphMCF and segmentation pairs: ' + in_star)
print('\tSTAR file with the GraphMCF info: ' + in_g_star)
print('\tOutput directory: ' + out_dir)
print('\t\t-Files sufix: ' + out_sufix)
print('\tGraph settings: ')
print('\t\t-Property name: ' + gh_pname)
if gh_clean:
    print('\t\t-Clean old values for the selected property.')
print('\tSegmentation processing: ')
if sg_th is None:
    print('\t\t-Using segmentation labels.')
else:
    print('\t\t-Segmentation threshold: ' + str(sg_th))
    print('\t\t-Segmentation label: ' + str(sg_lbl))
if sg_ref:
    print('\t\t-Reference space \'rlnMicrographName\'')
else:
    print('\t\t-Reference space \'psSegImage\'')
print('\t\t-Iterations for post closing: ' + str(sg_close))
print('')

print('Loading the input star file...')
star, graph_star = ps.sub.Star(), ps.sub.Star()
star.load(in_star)
graph_star.load(in_g_star)
if not star.has_column('_psGhMCFPickle'):
    print('ERROR: input pairs STAR file has no \'psGhMCFPickle\' column.')
    print('Un-successfully terminated. (' + time.strftime("%c") + ')')
    sys.exit(-1)
if not star.has_column('_psSegImage'):
    print('ERROR: input pairs STAR file has no \'psSegImage\' column.')
    print('Un-successfully terminated. (' + time.strftime("%c") + ')')
    sys.exit(-1)
if not graph_star.has_column('_rlnMicrographName'):
    print('ERROR: input graph STAR file has no \'rlnMicrographName\' column.')
    print('Un-successfully terminated. (' + time.strftime("%c") + ')')
    sys.exit(-1)
if not graph_star.has_column('_psGhMCFPickle'):
    print('ERROR: input graph STAR file has no \'psGhMCFPickle\' column.')
    print('Un-successfully terminated. (' + time.strftime("%c") + ')')
    sys.exit(-1)
graph_list = graph_star.get_column_data('_psGhMCFPickle')

# Loop for processing the input data
print('Running main loop: ')
for row in range(star.get_nrows()):

    seg_file, graph_file = star.get_element('_psSegImage', row), star.get_element('_psGhMCFPickle', row)
    print('\tPre-processing segmentation tomogram: ' + seg_file)
    mic_file = graph_star.get_element('_rlnMicrographName', row)
    try:
        seg = ps.disperse_io.load_tomo(seg_file).astype(np.uint16)
    except IOError:
        print('WARNING: input tomograms ' + seg_file + ' could not be read!')
        continue
    try:
        mic = ps.disperse_io.load_tomo(mic_file, mmap=True)
    except IOError:
        print('WARNING: input tomograms ' + mic_file + ' could not be read!')
        continue

    try:
        segg_row = graph_list.index(graph_file)
    except ValueError:
        print('WARNING: graph ' + graph_file + ' where not found on graphs STAR file!')
        continue
    segg_fname = graph_star.get_element('_psSegImage', segg_row)

    if sg_ref:
        print('\tApplying rigid body transformation to fit segmentation...')
        if sg_th is None:
            p_ids = (np.arange(seg.shape[0]), np.arange(seg.shape[1]), np.arange(seg.shape[2]))
        else:
            p_ids = np.where(seg > sg_th)
        segg = ps.disperse_io.load_tomo(segg_fname)
        if os.path.splitext(segg_fname)[1] == '.fits':
            segg = segg.swapaxes(0, 1)
        seg = np.zeros(shape=segg.shape, dtype=np.uint16)
        mic_c = np.asarray((.5*mic.shape[0], .5*mic.shape[1], .5*mic.shape[2]), dtype=float)
        for i in range(len(p_ids[0])):
            point = np.asarray((p_ids[0][i], p_ids[1][i], p_ids[2][i]), dtype=float)
            # Segmentation rigid body transformations
            # Centering
            point -= mic_c
            # Rotation
            try:
                rot, tilt, psi = graph_star.get_element('_psSegRot', segg_row), \
                                 graph_star.get_element('_psSegTilt', segg_row), \
                                 graph_star.get_element('_psSegPsi', segg_row)
                M = ps.globals.rot_mat_relion(rot, tilt, psi, deg=True)
                hold = (M * point.reshape(3, 1)).reshape(3)[0]
                point[0], point[1], point[2] = hold[0, 0], hold[0, 1], hold[0, 2]
            except KeyError:
                pass
            # Un-centering
            point += mic_c
            # Cropping
            try:
                offy, offx, offz = graph_star.get_element('_psSegOffX', segg_row), \
                                   graph_star.get_element('_psSegOffY', segg_row), \
                                   graph_star.get_element('_psSegOffZ', segg_row)
                point -= np.asarray((offx, offy, offz), dtype=float)
            except KeyError:
                pass
            # Assign the label
            x, y, z = int(round(point[0])), int(round(point[1])), int(round(point[2]))
            if (x >= 0) and (x < seg.shape[0]) and (y >= 0) and (y < seg.shape[1]) and (z >= 0) and (z < seg.shape[2]):
                seg[x, y, z] = sg_lbl
        if sg_close > 0:
            print('\t\t-Closing...')
            hold_seg = sp.ndimage.morphology.binary_closing(seg==sg_lbl, structure=None, iterations=sg_close)
            seg = np.zeros(shape=hold_seg.shape, dtype=np.uint16)
            seg[hold_seg > 0] = sg_lbl
        if os.path.splitext(segg_fname)[1] == '.fits':
            seg = seg.swapaxes(0, 1)
        in_seg_file = os.path.splitext(os.path.split(seg_file)[1])[0]
        out_seg_file = out_dir + '/' + in_seg_file + '_' + out_sufix + '_seg.vti'
        print('\t\t-Storing transformed segmentation in: ' + out_seg_file)
        ps.disperse_io.save_numpy(seg, out_seg_file)

    print('\tLoading the graph...')
    graph = ps.factory.unpickle_obj(graph_file)

    print('\tAdding segmentation to graph...')
    graph.add_scalar_field_nn(seg, name=gh_pname, clean=gh_clean, bg=0)

    in_graph_stem = os.path.splitext(os.path.split(graph_file)[1])[0]
    out_graph_file = out_dir + '/' + in_graph_stem + '_' + out_sufix + '.pkl'
    print('\tPickling updated graph in: ' + out_graph_file)
    graph.pickle(out_graph_file)
    graph_star.set_element('_psGhMCFPickle', segg_row, out_graph_file)
    ps.disperse_io.save_vtp(graph.get_vtp(av_mode=True, edges=True),
                            out_dir + '/' + in_graph_stem + '_' + out_sufix + '_edges.vtp')

in_star_stem = os.path.splitext(os.path.split(in_star)[1])[0]
out_star = out_dir + '/' + in_star_stem + '_' + out_sufix + '.star'
print('\tStoring output STAR file in: ' + out_star)
graph_star.store(out_star)

print('Terminated. (' + time.strftime("%c") + ')')