# -*- coding: utf-8 -*-
"""Prepares input files for ltomos_generator_translocon."""
from __future__ import division

import os
import logging
import time
import csv
import itertools
import operator

from datetime import timedelta

# import pyseg as ps
from pyorg import sub

# filepaths
ROOT_PATH = '/fs/pool/pool-lucic2/antonio/ribo_johannes/lum_ext_repick/FTR/stat'

# Input STAR file
in_star = ROOT_PATH + '/in/in_ltomos_FTR_dp8.star'

# Output STAR file
out_star = ROOT_PATH + '/in/in_seg_FTR_dp8.star'

# Parameters
no_rt = False
swap_seg_xy = True

def main():

    start_time = time.time()

    print 'Loading star file: {}.'.format(in_star)

    # Read micrograph names
    star_ltomos = sub.Star()
    star_ltomos.load(in_star)
    mics, imgs, ltomos = list(), list(), star_ltomos.get_column_data('_psStarFile')
    for ltomo in ltomos:
        hold_star = sub.Star()
        hold_star.load(ltomo)
        mics += hold_star.get_column_data('_rlnMicrographName')
        imgs += hold_star.get_column_data('_rlnImageName')

    # Find segmentation entries
    segs = dict()
    for img, mic in zip(imgs, mics):
        path_mic, stem_mic = os.path.split(mic)
        path_mic = path_mic.replace('/oriented_ribo_bin2', '')
        name = stem_mic.split('_')
        seg_star = sub.Star()
        seg_star.load(path_mic + '/graph_ribo/' + name[0] + '_' + name[1] + '_mb_graph.star')
        for row in range(seg_star.get_nrows()):
            seg_file = seg_star.get_element(key='_psSegImage', row=row)
            try:
                hold_dic = segs[seg_file]
                continue
            except KeyError:
                segs[seg_file] = dict()
                hold_dic = segs[seg_file]
            print '\t-Adding entry for file: ' + in_star
            hold_dic['_psSegImage'] = seg_file
            hold_dic['_rlnMicrographName'] = mic # seg_star.get_element(key='_rlnMicrographName', row=row)
            hold_dic['_psGhMCFPickle'] = seg_star.get_element(key='_psGhMCFPickle', row=row)
            if no_rt:
                hold_dic['_psSegOffX'] = 0
                hold_dic['_psSegOffY'] = 0
                hold_dic['_psSegOffZ'] = 0
                hold_dic['_psSegRot'] = 0
                hold_dic['_psSegTilt'] = 0
                hold_dic['_psSegPsi'] = 0
            else:
                hold_dic['_psSegOffX'] = seg_star.get_element(key='_psSegOffX', row=row)
                hold_dic['_psSegOffY'] = seg_star.get_element(key='_psSegOffY', row=row)
                hold_dic['_psSegOffZ'] = seg_star.get_element(key='_psSegOffZ', row=row)
                hold_dic['_psSegRot'] = seg_star.get_element(key='_psSegRot', row=row)
                hold_dic['_psSegTilt'] = seg_star.get_element(key='_psSegTilt', row=row)
                hold_dic['_psSegPsi'] = seg_star.get_element(key='_psSegPsi', row=row)

    # From dict to Star
    seg_star = sub.Star()
    seg_star.add_column('_psSegImage')
    seg_star.add_column('_rlnMicrographName')
    seg_star.add_column('_psGhMCFPickle')
    seg_star.add_column('_psSegOffX')
    seg_star.add_column('_psSegOffY')
    seg_star.add_column('_psSegOffZ')
    seg_star.add_column('_psSegRot')
    seg_star.add_column('_psSegTilt')
    seg_star.add_column('_psSegPsi')
    for key, hold_row in zip(segs.iterkeys(), segs.itervalues()):
        if swap_seg_xy:
            hold = hold_row['_psSegOffX']
            hold_row['_psSegOffX'] = hold_row['_psSegOffY']
            hold_row['_psSegOffY'] = hold
        seg_star.add_row(**hold_row)

    # Store the Star file
    print 'Storing output Star file in: ' + out_star
    seg_star.store(out_star)

    print 'Finished. Runtime {}.'.format(str(timedelta(seconds=time.time()-start_time)))

if __name__ == "__main__":
    main()