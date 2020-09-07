# -*- coding: utf-8 -*-
"""Creates a segmentaion file for statistical analysis containg the global offsets."""


import os
import logging
import time
import csv
import itertools
import operator

from datetime import timedelta

# import pyseg as ps
from pyorg import sub

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

# filepaths
path = '/fs/pool/pool-lucic2/antonio/ribo_johannes/lum_ext_repick'
star_path = path + '/stat/in/split_hold_ABC/run5_it050_data_kB_bin4.star'

path = '/fs/pool/pool-lucic2/antonio/ribo_johannes/lum_ext_repick'

export_star_path = path + '/stat/in/run5_it050_data_kB_seg.star'
out_part_seg = path + '/stat/in/2_run5_it050_data_kB_bin4_pseg.star'
export_csv_path = path + '/stat/in/csv_B'

swap = True
apply_pyseg_offsets = True
apply_relion_offsets = True

log_path = path
log_filename = 'statistical_analysis_export.log'

binning_particles = 2 # once binned
binning_segmentation = 4 # twice binned

def main():
    start_time = time.time()
    logger = logging.getLogger()
    add_file_logger(logger)

    logger.info('Loading star file: {}.'.format(star_path))

    star = sub.Star()
    star.load(star_path)
    micnames = star.get_column_data('_rlnMicrographName')
    imagenames = star.get_column_data('_rlnImageName')

    xcoords = star.get_column_data('_rlnCoordinateX')
    ycoords = star.get_column_data('_rlnCoordinateY')
    zcoords = star.get_column_data('_rlnCoordinateZ')
    xoffsets = star.get_column_data('_rlnOriginX')
    yoffsets = star.get_column_data('_rlnOriginY')
    zoffsets = star.get_column_data('_rlnOriginZ')

    rots = star.get_column_data('_rlnAngleRot')
    tilts = star.get_column_data('_rlnAngleTilt')
    psis = star.get_column_data('_rlnAnglePsi')
    tiltpriors = star.get_column_data('_rlnAngleTiltPrior')
    psipriors = star.get_column_data('_rlnAnglePsiPrior')
    star_part = star

    # loads segmentation file information pyseg offsets from graph data
    coords = {}
    mics = set()
    for mic in micnames:
        path, stem_tomo = os.path.split(mic)
        name = stem_tomo.split('_')
        mics.add(path + '/graph_ribo/' + name[0] + '_' + name[1] + '_mb_graph.star')

    offsets = {}
    segmentations = {}
    for mic in mics:
        star = sub.Star()
        star.load(mic)
        mnames = star.get_column_data('_rlnMicrographName')
        segnames = star.get_column_data('_psSegImage')
        xpsegs = star.get_column_data('_psSegOffX')
        ypsegs = star.get_column_data('_psSegOffY')
        zpsegs = star.get_column_data('_psSegOffZ')
        for idx, micrograph in enumerate(mnames):
            path, stem_tomo = os.path.split(micrograph)
            path = path.split('/')[7]
            name = stem_tomo.split('_')
            name[3] = name[3].split('.')
            name = path + '/' + name[0] + '_' + name[1] + '_' + name[2] + '_' + name[3][0]
            if apply_pyseg_offsets:
                offsets[name] = (xpsegs[idx], ypsegs[idx], zpsegs[idx])
            else:
                offsets[name] = (0,0,0)
            segmentations[name] = segnames[idx]

    vesicles = []
    for imagename, micname, x, y, z, xoff, yoff, zoff, rot, tilt, psi, tiltprior, psiprior in zip(imagenames, micnames, xcoords, ycoords, zcoords, xoffsets, yoffsets, zoffsets,
                                                    rots, tilts, psis, tiltpriors, psipriors):

        path, stem_tomo = os.path.split(imagename)
        path = path.split('/')[7]
        name = stem_tomo.split('_')
        name = name[0] + '_' + name[1] + '_' + name[2] + '_' + name[3]
        image_base, file_ext = os.path.splitext(stem_tomo)

        name = path + '/' + name
        xcoord, ycoord, zcoord = offsets[name]

        vesicles.append((segmentations[name].replace('_seg', ''), segmentations[name], xcoord, ycoord, zcoord, 0, 0, 0))

    vesicles_set = set(vesicles)

    export_hits_as_star(export_star_path, vesicles_set)

    for idx, micname in enumerate(micnames):
        stem, fname = os.path.split(micname)
        new_fname = stem + '/oriented_ribo_bin2/' + fname
        star_part.set_element(key='_rlnMicrographName', row=idx, val=new_fname)
    star_part.store(out_part_seg)

    coords = {}
    for x, y, z, rot, tilt, psi, xoffset, yoffset, zoffset, imagename in zip(xcoords, ycoords, zcoords,
                                                                                          rots, tilts, psis, xoffsets,
                                                                                          yoffsets, zoffsets,
                                                                                          imagenames):

        path, stem_tomo = os.path.split(imagename)
        path = path.split('/')[7]
        name = stem_tomo.split('_')
        name = name[0] + '_' + name[1] + '_' + name[2] + '_' + name[3]

        name = path + '/' + name

        image_base, file_ext = os.path.splitext(stem_tomo)

        if name not in coords:
            coords[name] = []

        # Imod binned format without origin
        # print xoffset, yoffset, zoffset
        hold = x
        x = y
        y = hold
        x -= (.25 * xoffset)
        y -= (.25 * yoffset)
        z -= (.25 * zoffset)
        coords[name].append((x, y, z, rot, tilt, psi, image_base.rsplit('_', 1)[-1]))

    for name, particles in coords.items():
        write_coords(particles, export_csv_path + '/' + str(name) + '_peaks_imod.csv')

    logger.info('Finished. Runtime {}.'.format(str(timedelta(seconds=time.time()-start_time))))

def add_file_logger(logger):
    fileHandler = logging.FileHandler("{0}/{1}".format(log_path, log_filename, mode='w'))
    logger.addHandler(fileHandler)

def export_hits_as_star(filename, list):
    star = create_star_stub()
    for micname, seg, x, y, z, rot, tilt, psi in list:
        row = {}
        row['_rlnMicrographName'] = micname
        row['_psSegImage'] = seg
        if swap:
            row['_psSegOffX'] = y
            row['_psSegOffY'] = x
        else:
            row['_psSegOffX'] = x
            row['_psSegOffY'] = y
        row['_psSegOffZ'] = z
        row['_psSegRot'] = rot
        row['_psSegTilt'] = tilt
        row['_psSegPsi'] = psi
        star.add_row(**row)
    star.store(filename)

def create_star_stub():
    star = sub.Star()
    star.add_column('_rlnMicrographName')
    star.add_column('_psSegImage')
    star.add_column('_psSegOffX')
    star.add_column('_psSegOffY')
    star.add_column('_psSegOffZ')
    star.add_column('_psSegRot')
    star.add_column('_psSegTilt')
    star.add_column('_psSegPsi')
    return star

def write_coords(coords, out_path):
    csv_path, _ = os.path.split(out_path)

    if not os.path.exists(csv_path):
        os.makedirs(csv_path)

    f = open(out_path, "wb")
    writer = csv.writer(f, delimiter='\t')
    #if swap_xy:
    #    writer.writerow(['PositionY', 'PositionX', 'PositionZ', 'Rot', 'Tilt', 'Psi'])
    #else:
    #    writer.writerow(['PositionX', 'PositionY', 'PositionZ', 'Rot', 'Tilt', 'Psi'])

    writer.writerows(coords)
    f.close()

if __name__ == "__main__":
    main()