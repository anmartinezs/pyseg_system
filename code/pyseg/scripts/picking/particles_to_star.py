''' Script for generating a list of coordinates (with Euler angles) and their correspodent reconstructed particles to STAR file '''
''' % generate folders corresponding to tomogram names in rln folder '''

import os
import csv
import random
import pyseg as ps
import scipy as sp

# I/O files
in_coords = '/fs/pool/pool-lucic2/in_situ_mito/mb_cont/out/11-dose_filt_imod.csv'
in_part_stem = '/fs/pool/pool-plitzko/Saikat/Tomography/Titan2/161220/t11_dosefilt/t11/particles_mb_cont/11-dose_filt-'
in_tomo = '/fs/pool/pool-plitzko/Saikat/Tomography/Titan2/161220/t11_dosefilt/t11/11-dose_filt.rec'
in_mask_norm = '/fs/pool/pool-lucic2/in_situ_er/mb_seg/in/mask_160_r60.mrc'
in_ctf = '/fs/pool/pool-plitzko/Saikat/Tomography/Titan2/161220/t11_dosefilt/t11/ctf_wedge_dose/11-dose_filt_wedge_160_35.mrc'
out_part_dir = '/fs/pool/pool-lucic2/in_situ_mito/mb_rec/parts/11-dose_filt'
out_star = '/fs/pool/pool-lucic2/in_situ_mito/mb_rec/parts/particles_1_11-dose_filt_prior_rot_rnd.star'

# Parameters
csv_delimiter = ' '
do_ang = True
do_prior = True
do_rnd = True
do_sg = False
do_norm = True

print 'Intilizing output STAR file...'
star = ps.sub.Star()
star.add_column('_rlnMicrographName')
star.add_column('_rlnImageName')
star.add_column('_rlnCtfImage')
star.add_column('_rlnCoordinateX')
star.add_column('_rlnCoordinateY')
star.add_column('_rlnCoordinateZ')
if do_ang:
    star.add_column('_rlnAngleRot')
    star.add_column('_rlnAngleTilt')
    star.add_column('_rlnAnglePsi')
    if do_prior:
        star.add_column('_rlnAngleTiltPrior')
        star.add_column('_rlnAnglePsiPrior')
mask = None
if in_mask_norm is not None:
    mask = ps.disperse_io.load_tomo(in_mask_norm)

print 'Processing file: ' + in_coords
with open(in_coords, 'r') as in_file:
    lines = in_file.readlines()
    nlines = len(lines)
    ndigits = len(str(nlines))
    print 'Number of particles to process ' + str(nlines)
    for i, line in enumerate(lines):
        part_id = str(i+1).zfill(ndigits)
        part_name = in_part_stem + part_id + '.mrc'
        print '\t-Processing particle: ' + part_name
        if not os.path.exists(part_name):
            print 'WARNING: particle reconstruction file ' + part_name + ' was not found!'
            continue
            # raise IOError
        part = ps.disperse_io.load_tomo(part_name)
        if do_sg:
            part = sp.ndimage.filters.gaussian_filter(part, do_sg)
        if do_norm and (mask is not None):
            part = ps.sub.relion_norm(part, mask)
        it = [float(s) for s in line.split()]
        row = dict()
        row['_rlnMicrographName'] = in_tomo
        row['_rlnCtfImage'] = in_ctf
        row['_rlnCoordinateX'] = it[0]
        row['_rlnCoordinateY'] = it[1]
        row['_rlnCoordinateZ'] = it[2]
        if do_ang:
            if do_rnd:
                row['_rlnAngleRot'] = 180. * random.random()
            else:
                row['_rlnAngleRot'] = it[3]
            row['_rlnAngleTilt'] = it[4]
            row['_rlnAnglePsi'] = it[5]
            if do_prior:
                row['_rlnAngleTiltPrior'] = it[4]
                row['_rlnAnglePsiPrior'] = it[5]
        out_part_name = out_part_dir + '/particle_rln_' + part_id + '.mrc'
        ps.disperse_io.save_numpy(part, out_part_name)
        row['_rlnImageName'] = out_part_name
        star.add_row(**row)

print 'Storing file: ' + out_star
star.store(out_star)
print 'Succesfully terminated!'