"""
Set of classes for particle classification

# Author: Antonio Martinez-Sanchez (Max Planck Institute for Biochemistry)
# Date: 1.03.17
"""

__author__ = 'Antonio Martinez-Sanchez'


# import cv2

import gc
import os
import sys
import pyto
import copy
import itertools as it
from pyseg.globals import *
from pyseg import disperse_io
from .star import Star, relion_norm
from imageio import imread, imwrite
# from scipy.misc import imsave, imread # From PYthon 2.7
import numpy as np
from scipy import signal as sgn
import multiprocessing as mp
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import LocallyLinearEmbedding
from .variables import RadialAvg3D

from pyseg.pexceptions import *

try:
    import pickle as pickle
except:
    import pickle

###########################################################################################
# Global functionality
###########################################################################################

# Returns the a subvolume of a tomogram from a center and a shape
# tomo: input tomogram
# sub_pt: subtomogram center point
# sub_shape: output subtomogram shape
# halo_val: value for the halos (default 0)
# Returns: a copy with the subvolume, halos are filled up with 'halo_val'
def get_tomo_sub(tomo, sub_pt, sub_shape, halo_val=0):

    # Initialization
    nx, ny, nz = sub_shape[0], sub_shape[1], sub_shape[2]
    mx, my, mz = tomo.shape[0], tomo.shape[1], tomo.shape[2]
    mx1, my1, mz1 = mx-1, my-1, mz-1
    hl_x, hl_y, hl_z = int(nx*.5), int(ny*.5), int(nz*.5)
    x, y, z = int(round(sub_pt[0])), int(round(sub_pt[1])), int(round(sub_pt[2]))

    # Compute bounding restrictions
    dif_l_x, dif_l_y, dif_l_z = 0, 0, 0
    dif_h_x, dif_h_y, dif_h_z = nx, ny, nz
    if nx%2 == 0:
        off_l_x, off_h_x = x-hl_x, x+hl_x
    else:
        off_l_x, off_h_x = x-hl_x+1, x+hl_x+1
    if ny%2 == 0:
        off_l_y, off_h_y = y-hl_y, y+hl_y
    else:
        off_l_y, off_h_y = y-hl_y+1, y+hl_y+1
    if nz%2 == 0:
        off_l_z, off_h_z = z-hl_z, z+hl_z
    else:
        off_l_z, off_h_z = z-hl_z+1, z+hl_z+1
    if off_l_x < 0:
        dif_l_x = abs(off_l_x)
        off_l_x = 0
    if off_l_y < 0:
        dif_l_y = abs(off_l_y)
        off_l_y = 0
    if off_l_z < 0:
        dif_l_z = abs(off_l_z)
        off_l_z = 0
    if off_h_x >= mx:
        dif_h_x = nx - off_h_x + mx1
        off_h_x = mx1
    if off_h_y >= my:
        dif_h_y = ny - off_h_y + my1
        off_h_y = my1
    if off_h_z >= mz:
        dif_h_z = nz - off_h_z + mz1
        off_h_z = mz1
    # Copying the data
    svol = halo_val * np.ones(shape=sub_shape, dtype=tomo.dtype)
    svol[dif_l_x:dif_h_x, dif_l_y:dif_h_y, dif_l_z:dif_h_z] = tomo[off_l_x:off_h_x, off_l_y:off_h_y, off_l_z:off_h_z]
    return svol

###########################################################################################
# Parallel processes
###########################################################################################

# Parallel process for computing the cross correlation matrix among 2D particles
# VERY IMPORTANT: function ClassStar().build_ncc_z2d() already normalized particle contrast in 3D
# pr_id: process id, if lower than zeros the serial process is assumed
# ids: ids for sym_ids array
# sym_ids: id pairs for particles
# imgs: array with the 2d images
# masks: array with the binary masks for each image
# metric: 0- cross-correlation within mask, 1- mask normalized similarity (negative squared Euclidean distance),
#         otherwise- full cross-correlation (slower than 0 but allows small disalignments between particles)
# shared_mat: output CC matrix as shared array
def pr_cc_2d(pr_id, ids, sym_ids, imgs, masks, metric, shared_mat):

    # Particles loop
    n_parts = len(imgs)
    if metric == 0:
        for idx in ids:
            pair_ids = sym_ids[int(idx)]

            ## Extract the particle information
            # Particle 1
            id_1 = pair_ids[0]
            img_1 = np.copy(imgs[id_1])
            # Particle 2
            id_2 = pair_ids[1]
            img_2 = np.copy(imgs[id_2])


            # 2D Cross-Correlation
            mask = (masks[id_1] * masks[id_2]) > 0
            n_mask = float(mask.sum())
            if n_mask > 0:
                cte = 1. / n_mask
                if cte > 0:
                    ncc = cte * np.sum(img_1[mask]*img_2[mask])

                    # Set the maximum cross correlation value
                    shared_mat[int(id_1 + id_2*n_parts)] = ncc
                    shared_mat[int(id_2 + id_1*n_parts)] = ncc

    elif metric == 1:
        for idx in ids:
            pair_ids = sym_ids[int(idx)]

            ## Extract the particle information
            # Particle 1
            id_1 = pair_ids[0]
            img_1 = np.copy(imgs[id_1])
            # Particle 2
            id_2 = pair_ids[1]
            img_2 = np.copy(imgs[id_2])


            # Simmilarity
            mask = (masks[id_1] * masks[id_2]) > 0
            n_mask = float(mask.sum())
            if n_mask > 0:
                cte = 1. / n_mask
                if cte > 0:
                    hold = img_1[mask] - img_2[mask]
                    sim = -1. * cte * (hold * hold).sum()

                    # Set the maximum cross correlation value
                    shared_mat[int(id_1 + id_2*n_parts)] = sim
                    shared_mat[int(id_2 + id_1*n_parts)] = sim

    else:
        # print 'Metric F'
        for idx in ids:
            pair_ids = sym_ids[int(idx)]

            ## Extract the particle information
            # Particle 1
            id_1 = pair_ids[0]
            img_1 = np.copy(imgs[id_1])
            # Particle 2
            id_2 = pair_ids[1]
            img_2 = np.copy(imgs[id_2])


            # 2D Cross-Correlation
            mask = (masks[id_1] * masks[id_2]) > 0
            n_mask = float(mask.sum())
            if n_mask > 0:
                cte = 1. / n_mask
                if cte > 0:
                    mask_inv = np.invert(mask)
                    img_1[mask_inv] = 0
                    img_2[mask_inv] = 0
                    ncc = cte * sgn.correlate2d(img_1, img_2, mode='full').max()

                    # Set the maximum cross correlation value
                    shared_mat[int(id_1 + id_2*n_parts)] = ncc
                    shared_mat[int(id_2 + id_1*n_parts)] = ncc

    if pr_id < 0:
        return
    else:
        sys.exit(pr_id)

# Parallel process for loading and preprocessing particles before their classification
# pr_id: process id, if lower than zeros the serial process is assumed
# p_ids: ids for particles
# mask: binary mask to be applied after rotations, its dimensions also defines subvolumes dimensions used,
#       only 3D cubes with even dimension are accepted
# star: STAR file object
# low_sg: low pass Gaussian filter sigma in voxels, it does not apply if lower or equal than 0 (default)
# rln_norm: normalize particles according Relion convention after loaded (default True)
# avg_norm: re-normalize particles after their radial averaging
# rad_3D: if False NCC are made in 2D, otherwise radial average is compensate for doing a NCC in 3D
# debug_dir: if not None (default) intermediate information will be stored in this directory, use only for debugging
#            purposes
# ref_dir: Alternative directory for looking for reference tomograms (default None)
# direct: if True then particles are directly load from input STAR file
# particles: shared array where the particles will be stored
# masks: shared array where the masks will be stored
# bin: binning factor (default None)
def pr_load_parts(pr_id, p_ids, mask, star, low_sg, rln_norm, avg_norm, rad_3D, debug_dir, ref_dir, direct, particles,
                  masks, bin=None):

    # Initialization
    if bin is not None:
        ibin = 1. / float(bin)
        # mask = sp.ndimage.zoom(mask, ibin, order=3, mode='constant', cval=0.0, prefilter=True)
        low_sg_f = ibin * float(low_sg)
    else:
        low_sg_f = float(low_sg)
    svol_sp = np.asarray(mask.shape, dtype=int)
    svol_sp2 = int(.5 * svol_sp[0])
    svol_cent = np.asarray((svol_sp2, svol_sp2, svol_sp2), dtype=np.float32)
    averager = RadialAvg3D(svol_sp, axis='z')
    part_h, part_r = averager.get_output_dim()
    part_sz = int(part_h * part_r)
    hold_mask_2d = averager.avg_vol(mask)
    ids_h, ids_r = np.where(hold_mask_2d.sum(axis=1) > 0)[0], np.where(hold_mask_2d.sum(axis=0) > 0)[0]
    rg_h, rg_r = [ids_h.min(), ids_h.max()], [ids_r.min(), ids_r.max()]
    # mask_avg = averager.avg_vol(mask, rg_h=rg_h, rg_r=rg_r) > 0.5
    mask_avg = averager.avg_vol(mask, rg_h=rg_h, rg_r=rg_r) > 0

    # Update mask with segmentation information and rotations matrix construction
    count = 1
    for row in p_ids:

        # print('Processing particle ' + str(count) + ' of ' + str(len(p_ids)) + ' for process ' + str(pr_id))
        # sys.stdout.flush()
        count += 1

        ### Getting the coordinates
        # Coordinates and rotation
        coord, angs = star.get_particle_coords(row, orig=True, rots=True)
        coord, angs = np.asarray(coord, dtype=np.float32), np.asarray(angs, dtype=np.float32)
        r3d = pyto.geometry.Rigid3D()
        r3d.q = r3d.make_r_euler(angles=np.radians(angs), mode='zyz_in_active')

        ### Getting the particle
        if not direct:
            ref_fname = star.get_element('_rlnMicrographName', row)
            # print 'DEBUG: ' + ref_fname
            if ref_dir is not None:
                ref_fname = ref_dir + '/' + os.path.split(ref_fname)[1]
            ref_tomo = disperse_io.load_tomo(ref_fname, mmap=True)
            if bin is not None:
                ref_tomo = sp.ndimage.zoom(ref_tomo, ibin, order=3, mode='constant', cval=0.0, prefilter=True)
            particle_u = get_tomo_sub(ref_tomo, coord, svol_sp, halo_val=0)
        else:
            sub_fname = star.get_element('_rlnImageName', row)
            particle_u = disperse_io.load_tomo(sub_fname)
            try:
                shift_x, shift_y, shift_z = star.get_element('_rlnOriginX', row), star.get_element('_rlnOriginY', row), \
                                            star.get_element('_rlnOriginZ', row)
                particle_u = tomo_shift(particle_u, (shift_y, shift_x, shift_z))
            except KeyError:
                pass
            # particle_u = tomo_shift(particle_u, (-shift_y, -shift_x, -shift_z))
            if bin is not None:
                particle_u = sp.ndimage.zoom(particle_u, ibin, order=3, mode='constant', cval=0.0, prefilter=True)
        particle = r3d.transformArray(particle_u, center=svol_cent, order=3, prefilter=True)
        if low_sg_f > 0:
            particle = sp.ndimage.filters.gaussian_filter(particle, low_sg)

        ### Getting the mask
        ### Particle masking and normalization
        if rln_norm:
            particle = relion_norm(particle, mask=mask)
        particle_mask = particle * mask

        ### Applying Z-axis volume compression
        particle_avg = averager.avg_vol(particle_mask, rg_h=rg_h, rg_r=rg_r)
        if rad_3D:
            rd = (np.meshgrid(np.arange(particle_avg.shape[0]), np.arange(particle_avg.shape[1]),
                              indexing='ij')[1]).astype(np.float32)
            rd = np.sqrt(rd)
            particle_avg *= rd
        if avg_norm:
            p_masked = particle_avg[mask_avg]
            m_std = p_masked.std()
            if m_std > 0:
                particle_avg = (particle_avg-p_masked.mean()) / m_std
            particle_avg[~mask_avg] = 0

        ### Store intermediate information
        if debug_dir is not None:
            hold_out = os.path.split(star.get_element('_rlnImageName', row))[1]
            stem_out = os.path.splitext(hold_out)[0]
            disperse_io.save_numpy(mask, debug_dir + '/' + stem_out + '_mask.mrc')
            disperse_io.save_numpy(seg_mask_r, debug_dir + '/' + stem_out + '_smask.mrc')
            disperse_io.save_numpy(tot_mask, debug_dir + '/' + stem_out + '_tmask.mrc')
            disperse_io.save_numpy(particle_u, debug_dir + '/' + stem_out + '_svol.mrc')
            disperse_io.save_numpy(particle, debug_dir + '/' + stem_out + '_rsvol.mrc')
            disperse_io.save_numpy(particle_mask, debug_dir + '/' + stem_out + '_mrsvol.mrc')
            imwrite(debug_dir + '/' + stem_out + '_avgz.png', lin_map(particle_avg,0,np.iinfo(np.uint16).max).astype(np.uint16))
            imwrite(debug_dir + '/' + stem_out + '_mavgz.png', lin_map(mask_avg,0,np.iinfo(np.uint16).max).astype(np.uint16))

        # Set the shared arrays
        sh_id = row * part_sz
        for j in range(part_r):
            sh_id_l, sh_id_h = sh_id+j*part_h, sh_id+(j+1)*part_h
            particles[sh_id_l:sh_id_h], masks[sh_id_l:sh_id_h] = particle_avg[:, j], mask_avg[:, j]

        # Try to clean memory
        gc.collect()

    if pr_id < 0:
        return
    else:
        sys.exit(pr_id)

# Parallel process for computing the image (3D volume Z-radially averaged) moments
# pr_id: process id, if lower than zeros the serial process is assumed
# p_ids: ids for particles
# mode: type of moments computed
# momes_sz: number of modes
# particles: array where the particles will be stored
# mask: global mask for all particles
# momes: shared array where the moments will be stored
def pr_mome_2d(pr_id, p_ids, mode, momes_sz, particles, mask, momes):

    # spatial moments
    if mode == 0:
        for idx in p_ids:
            hold_mome = cv2.moments(particles[idx])
            sh_id = idx * momes_sz
            momes[sh_id], momes[sh_id+1], momes[sh_id+2] = hold_mome['m00'], hold_mome['m10'], hold_mome['m01']
            momes[sh_id+3], momes[sh_id+4], momes[sh_id+5] = hold_mome['m20'], hold_mome['m11'], hold_mome['m02']
            momes[sh_id+6], momes[sh_id+7], momes[sh_id+8] = hold_mome['m30'], hold_mome['m21'], hold_mome['m12']
            momes[sh_id+9] = hold_mome['m03']
            # print 'Particle ' + str(idx) + ': ' + str(momes[sh_id:sh_id+9])

    # central moments
    elif mode == 1:
        for idx in p_ids:
            hold_mome = cv2.moments(particles[idx])
            sh_id = idx * momes_sz
            momes[sh_id], momes[sh_id+1], momes[sh_id+2] = hold_mome['mu20'], hold_mome['mu11'], hold_mome['mu02']
            momes[sh_id+3], momes[sh_id+4], momes[sh_id+5] = hold_mome['mu30'], hold_mome['mu21'], hold_mome['mu12']
            momes[sh_id+6] = hold_mome['mu03']
            # print 'Particle ' + str(idx) + ': ' + str(momes[sh_id:sh_id+6])

    # central normalized moments
    elif mode == 2:
        for idx in p_ids:
            hold_mome = cv2.moments(particles[idx])
            sh_id = idx * momes_sz
            momes[sh_id], momes[sh_id+1], momes[sh_id+2] = hold_mome['nu20'], hold_mome['nu11'], hold_mome['nu02']
            momes[sh_id+3], momes[sh_id+4], momes[sh_id+5] = hold_mome['nu30'], hold_mome['nu21'], hold_mome['nu12']
            momes[sh_id+6] = hold_mome['nu03']

    # Raw image analysis (masked images are flattered)
    else:
        for idx in p_ids:
            sh_id = idx * momes_sz
            particle = particles[idx]
            momes[sh_id:sh_id+momes_sz] = particle[mask]
            # print 'Particle ' + str(idx) + ': ' + str(momes[sh_id:sh_id+momes_sz])

    if pr_id < 0:
        return
    else:
        sys.exit(pr_id)

# Parallel process for sequentially loading particles and compute cross-correlation against input exemplars
# pr_id: process id, if lower than zeros the serial process is assumed
# p_ids: ids for particle
# ex_ids: exemplars ids
# ex_temps: exemplar templates ordered by ids
# mask: binary mask to be applied after rotations, its dimensions also defines subvolumes dimensions used,
#       only 3D cubes with even dimension are accepted
# star: STAR file object
# low_sg: low pass Gaussian filter sigma in voxels, it does not apply if lower or equal than 0 (default)
# rln_norm: normalize particles according Relion convention after loaded (default True)
# avg_norm: re-normalize particles after their radial averaging
# rad_3D: if False NCC are made in 2D, otherwise radial average is compensate for doing a NCC in 3D
# ref_dir: Alternative directory for looking for reference tomograms (default None)
# direct: if True then particles are directly load from input STAR file
# out_dir: output directory for storing the flattern images
# exs_cc: shared array where the CC will be stored
def pr_exemplars_cc(pr_id, p_ids, ex_temps, mask, star, low_sg, rln_norm, avg_norm, rad_3D, ref_dir, direct,
                    out_dir, exs_cc):

    # Initialization
    svol_sp = np.asarray(mask.shape, dtype=int)
    svol_sp2 = int(.5 * svol_sp[0])
    svol_cent = np.asarray((svol_sp2, svol_sp2, svol_sp2), dtype=np.float32)
    low_sg_f = float(low_sg)
    averager = RadialAvg3D(svol_sp, axis='z')
    num_exs = int(len(ex_temps))
    mask_avg = averager.avg_vol(mask, rg_h=rg_h, rg_r=rg_r) > 0.5

    # Update mask with segmentation information and rotations matrix construction
    for row in p_ids:

        # print 'Processing particle ' + str(row)

        ### Getting the coordinates
        # Coordinates and rotation
        coord, angs = star.get_particle_coords(row, orig=True, rots=True)
        coord, angs = np.asarray(coord, dtype=np.float32), np.asarray(angs, dtype=np.float32)
        r3d = pyto.geometry.Rigid3D()
        r3d.q = r3d.make_r_euler(angles=np.radians(angs), mode='zyz_in_active')

        ### Getting the particle
        if not direct:
            ref_fname = star.get_element('_rlnMicrographName', row)
            # print 'DEBUG: ' + ref_fname
            if ref_dir is not None:
                ref_fname = ref_dir + '/' + os.path.split(ref_fname)[1]
            ref_tomo = disperse_io.load_tomo(ref_fname, mmap=True)
            particle_u = get_tomo_sub(ref_tomo, coord, svol_sp, halo_val=0)
        else:
            sub_fname = star.get_element('_rlnImageName', row)
            particle_u = disperse_io.load_tomo(sub_fname, mmap=True)
            shift_x, shift_y, shift_z = star.get_element('_rlnOriginX', row), star.get_element('_rlnOriginY', row), \
                                        star.get_element('_rlnOriginZ', row)
            particle_u = tomo_shift(particle_u, (-shift_x, -shift_y, -shift_z))
            # particle_u = tomo_shift(particle_u, (-shift_y, -shift_x, -shift_z))
        particle = r3d.transformArray(particle_u, center=svol_cent, order=3, prefilter=True)
        if low_sg_f > 0:
            particle = sp.ndimage.filters.gaussian_filter(particle, low_sg)

        ### Getting the mask
        if rln_norm:
            particle = relion_norm(particle, mask=mask)
        particle_mask = particle * mask

        ### Applying Z-axis volume compression
        particle_avg = averager.avg_vol(particle_mask, rg_h=rg_h, rg_r=rg_r)
        if rad_3D:
            rd = (np.meshgrid(np.arange(particle_avg.shape[0]), np.arange(particle_avg.shape[1]),
                              indexing='ij')[1]).astype(np.float32)
            rd = np.sqrt(rd)
            particle_avg *= rd
        if avg_norm:
            p_masked = particle_avg[mask_avg]
            m_std = p_masked.std()
            if m_std > 0:
                particle_avg = (particle_avg-p_masked.mean()) / m_std
            particle_avg[~mask_avg] = 0

        # Compute CC against exemplars
        ex_cc = np.zeros(shape=num_exs, dtype=np.float32)
        for i, ex_temp in enumerate(ex_temps):
            # 2D Cross-Correlation
            n_mask = float(mask_avg.sum())
            if n_mask > 0:
                cte = 1. / n_mask
                if cte > 0:
                    ex_cc[i] = cte * np.sum(particle_avg[mask_avg] * ex_temp[mask_avg])

        if out_dir is not None:
            out_fname = out_dir + '/particle_' + str(row) + '.png'
            imwrite(out_fname, lin_map(particle_avg,0,np.iinfo(np.uint16).max).astype(np.uint16))
            out_fname = out_dir + '/particle_' + str(row) + '.npy'
            np.save(out_fname, particle_avg)

        # Set the shared arrays
        sh_id_l = row * num_exs
        sh_id_h = sh_id_l + num_exs
        # print 'len(exs_cc)=' + str(len(exs_cc)) + ', sh_id_l=' + str(sh_id_l) + ', sh_id_h=' + str(sh_id_h)
        exs_cc[sh_id_l:sh_id_h] = ex_cc

    if pr_id < 0:
        return
    else:
        sys.exit(pr_id)

###########################################################################################
# Classes
###########################################################################################

############################################################################################
# Class for particles classification in a STAR file
#
#
class ClassStar(object):

    #### Constructor area

    def __init__(self, star, particles=None):
        '''
        Constructor
        :param star: Star object
        :param particles: (optional, default None) to insert particles directly
        '''
        if not isinstance(star, Star):
            error_msg = 'Input variable star must be a Star object.'
            raise pexceptions.PySegInputError(expr='__init__ (ClassStar)', msg=error_msg)
        if particles is not None:
            if (not hasattr(particles,'__len__')) or (len(particles)<=0):
                error_msg = 'Input particles must be an interable with length larger than zero.'
                raise pexceptions.PySegInputError(expr='__init__ (ClassStar)', msg=error_msg)
            if len(particles) != star.get_nrows():
                error_msg = 'Input particles length must be equal to Star number of rows.'
                raise pexceptions.PySegInputError(expr='__init__ (ClassStar)', msg=error_msg)
        self.__star = star
        self.__cc = None
        self.__particles = particles
        self.__masks = None
        self.__momes, self.__vectors = None, None
        self.__mask_gl = None
        # AP variables
        self.__ap_classes, self.__ap_centers, self.__ap_ref_cc = None, None, None

    #### External functionality

    def get_global_mask(self):
        return self.__mask_gl

    def get_split_stars(self, parse_rln=False):
        '''
        Get a STAR file object per each class
        :param parse_rln:
        :return: a list with the STAR file objects
        '''

        if parse_rln:
            hold_star = self.__star.get_relion_parsed_copy()
        else:
            hold_star = self.__star

        return hold_star.split_class()


    def get_moments_nfeatures(self):
        if (self.__momes is None) or (self.__momes.shape[0] < 1):
            return 0
        else:
            return self.__momes[0].shape[0]

    # Add a segmentation STAR file with information that will be consider to focus the mask
    # seg_star: Star object with the segmentation information
    def add_segmentation(self, seg_star):

        # Input parsing
        if not isinstance(seg_star, Star):
            error_msg = 'Input seg_star must be pyseg.sub.Star instance'
            raise pexceptions.PySegInputError(expr='add_segmentation (ClassStar)', msg=error_msg)

        # First call creates the columns
        self.__star.add_column('_psSegImage')
        self.__star.add_column('_psSegLabel')
        self.__star.add_column('_psSegOffX')
        self.__star.add_column('_psSegOffY')
        self.__star.add_column('_psSegOffZ')
        self.__star.add_column('_psSegRot')
        self.__star.add_column('_psSegTilt')
        self.__star.add_column('_psSegPsi')

        # Add the segmentation information only on the corresponding particles (those which share the micrograph name)
        seg_mics = list(seg_star.get_column_data('_rlnMicrographName'))
        for row in range(self.__star.get_nrows()):
            try:
                row_seg = seg_mics.index(self.__star.get_element('_rlnMicrographName', row))
                try:
                    hold = seg_star.get_element('_psSegImage', row_seg)
                except KeyError:
                    hold = ''
                self.__star.set_element('_psSegImage', row, hold)
                try:
                    hold = seg_star.get_element('_psSegLabel', row_seg)
                except KeyError:
                    hold = 0
                self.__star.set_element('_psSegLabel', row, hold)
                try:
                    hold = seg_star.get_element('_psSegOffX', row_seg)
                except KeyError:
                    hold = 0
                self.__star.set_element('_psSegOffX', row, hold)
                try:
                    hold = seg_star.get_element('_psSegOffY', row_seg)
                except KeyError:
                    hold = 0
                self.__star.set_element('_psSegOffY', row, hold)
                try:
                    hold = seg_star.get_element('_psSegOffZ', row_seg)
                except KeyError:
                    hold = 0
                self.__star.set_element('_psSegOffZ', row, hold)
                try:
                    hold = seg_star.get_element('_psSegRot', row_seg)
                except KeyError:
                    hold = 0
                self.__star.set_element('_psSegRot', row, hold)
                try:
                    hold = seg_star.get_element('_psSegTilt', row_seg)
                except KeyError:
                    hold = 0
                self.__star.set_element('_psSegTilt', row, hold)
                try:
                    hold = seg_star.get_element('_psSegPsi', row_seg)
                except KeyError:
                    hold = 0
                self.__star.set_element('_psSegPsi', row, hold)
            except ValueError:
                self.__star.set_element('_psSegImage', row, '')
                self.__star.set_element('_psSegLabel', row, 0)
                self.__star.set_element('_psSegOffX', row, 0)
                self.__star.set_element('_psSegOffY', row, 0)
                self.__star.set_element('_psSegOffZ', row, 0)
                self.__star.set_element('_psSegRot', row, 0)
                self.__star.set_element('_psSegTilt', row, 0)
                self.__star.set_element('_psSegPsi', row, 0)

    def save_cc(self, fname, txt=False):
        """
        Saves CC matrix in file as numpy (2D)array
        :param fname: full path where CC 2D numpy array will be stored, see numpy.save for more details
        :param txt: if False (default) the is saved as binary data, otherwise as human readable text
        :return:
        """

        # Input parsing
        if self.__cc is None:
            error_msg = 'No CC matrix, call build_cc_*() methods first!'
            raise pexceptions.PySegInputError(expr='save_cc (ClassStar)', msg=error_msg)

        if txt:
            np.savetxt(fname, self.__cc)
        else:
            np.save(fname, self.__cc)

    # Loads numpy array from disk and store it as a CC matrix
    # fname: full path where CC 2D numpy array is stored, see numpy.load for more details
    # mmap: see numpy.load for more detail
    def load_cc(self, fname, mmap=None):

        hold = np.load(fname, mmap_mode=mmap)

        # Input parsing
        if (not isinstance(hold, np.ndarray)) or (len(hold.shape) != 2) or (hold.shape[0] != hold.shape[1]):
            error_msg = 'Input CC matrix must be a squared 2D numpy array!'
            raise pexceptions.PySegInputError(expr='load_cc (ClassStar)', msg=error_msg)

        self.__cc = hold

    # Load particles from the STAR file information and computes their mask and radial averages
    # mask: binary mask to be applied after rotations, its dimensions also defines subvolumes dimensions used,
    #       only 3D cubes with even dimension are accepted
    # low_sg: low pass Gaussian filter sigma in voxels, it does not apply if lower or equal than 0 (default)
    # rln_norm: normalize particles according Relion convention after loaded (default True)
    # avg_norm: re-normalize particles after their radial averaging (default False)
    # rad_3d: if False (default) the NCC are done in 2D, otherwise in 3D
    # npr: number of process, None (defaults) means they are adjusted automatically to the number of cores
    # debug_dir: if not None (default) intermediate information will be stored in this directory, use only for debugging
    #            purposes
    # ref_dir: Alternative directory for looking for reference tomograms (default None)
    # direct_rec: if False (default) the particles are cropped from 'rlnMicrographName' tomogram, otherwise they are
    #             directly load from 'rlnImageName'
    # bin: if not None (default) then particles are binned before being processed
    def load_particles(self, mask, low_sg=0, seg_dil_it=2, rln_norm=True, avg_norm=False, rad_3D=False, npr=None,
                       debug_dir=None, ref_dir=None, direct_rec=False, bin=None):

        # Input parsing
        binf = None
        if bin is not None:
            binf = float(bin)
            ibin = 1. / binf
        if not isinstance(mask, np.ndarray):
            error_msg = 'Input mask must a ndarray!'
            raise pexceptions.PySegInputError(expr='load_particles (ClassStar)', msg=error_msg)
        if binf is not None:
            mask = sp.ndimage.zoom(mask, ibin, order=3, mode='constant', cval=0.0, prefilter=True)
        svol_sp = np.asarray(mask.shape, dtype=int)
        if (len(svol_sp) != 3) or (svol_sp[0] != svol_sp[1]) or (svol_sp[1] != svol_sp[2]) or (svol_sp[0]%2 != 0):
            error_msg = 'Input mask must be a 3D cube with even dimension!'
            raise pexceptions.PySegInputError(expr='load_particles (ClassStar)', msg=error_msg)
        # Store intermediate information
        averager = RadialAvg3D(svol_sp, axis='z')
        if debug_dir is not None:
            disperse_io.save_numpy(averager.get_kernels(), debug_dir + '/avg_kernels.mrc')

        # Initialization
        npart = self.__star.get_nrows()
        # self.__cc = np.zeros(shape=(npart, npart), dtype=np.float32)
        self.__particles = np.zeros(shape=npart, dtype=object)
        self.__masks = np.zeros(shape=npart, dtype=object)

        # Multiprocessing
        if npr is None:
            npr = mp.cpu_count()
        processes = list()
        # Create the list on indices to split
        npart = self.__star.get_nrows()
        part_h, part_r = averager.get_output_dim()
        part_sz = int(part_h * part_r)
        particles_sh, masks_sh = mp.Array('f', int(part_h*part_r*npart)), mp.Array('f', int(part_h*part_r*npart))

        # Loading particles loop (Parallel)
        if npr <= 1:
            pr_load_parts(-1, np.arange(npart, dtype=int), mask, self.__star, float(low_sg), rln_norm,  avg_norm, rad_3D,
                          debug_dir, ref_dir, direct_rec,
                          particles_sh, masks_sh, binf)
        else:
            spl_ids = np.array_split(np.arange(npart, dtype=int), npr)
            for pr_id in range(npr):
                pr = mp.Process(target=pr_load_parts, args=(pr_id, spl_ids[pr_id], mask, self.__star, float(low_sg),
                                                            rln_norm, avg_norm, rad_3D, debug_dir, ref_dir, direct_rec,
                                                            particles_sh, masks_sh, binf))
                pr.start()
                processes.append(pr)
            pr_results = list()
            for pr in processes:
                pr.join()
                pr_results.append(pr.exitcode)
            for pr_id in range(len(processes)):
                if pr_id != pr_results[pr_id]:
                    error_msg = 'Process ' + str(pr_id) + ' exited unexpectedly!'
                    raise pexceptions.PySegInputError(expr='load_particles (ClassStar)', msg=error_msg)
            gc.collect()

        # Set class variables from multiprocess shared objects
        self.__mask_gl = np.ones(shape=(part_h, part_r), dtype=bool)
        for row in range(npart):
            hold_particle, hold_mask = np.zeros(shape=(part_h, part_r), dtype=np.float32), \
                                       np.zeros(shape=(part_h, part_r), dtype=np.float32)
            sh_id = row * part_sz
            for j in range(part_r):
                sh_id_l, sh_id_h = sh_id+j*part_h, sh_id+(j+1)*part_h
                hold_particle[:, j], hold_mask[:, j] = particles_sh[sh_id_l:sh_id_h], masks_sh[sh_id_l:sh_id_h]
            self.__particles[row], self.__masks[row] = hold_particle, hold_mask
            self.__mask_gl *= (hold_mask > 0)

    # Build the Normalized (intensity in 3D) Cross-Correlation Matrix among Z radially averaged particles
    # metric: metric used for particles affinity, valid: 'cc' (default) cross-correlation,
    #         'similarity' negative squared Euclidean distance, 'full_cc' slower than 'cc' but considers small
    #         misalignments
    # npr: number of process, None (defaults) means they are adjusted automatically to the number of cores
    def build_ncc_z2d(self, metric='cc', npr=None):

        # Input parsing
        if (self.__particles is None) or (self.__masks is None) or (len(self.__particles) != len(self.__masks)):
            error_msg = 'There is no particle to process call correctly load_particles() function first!'
            raise pexceptions.PySegInputError(expr='build_ncc_z2d (ClassStar)', msg=error_msg)
        if metric == 'cc':
            p_metric = 0
        elif metric == 'similarity':
            p_metric = 1
        elif metric == 'cc_full':
            p_metric = 2
        else:
            error_msg = 'Invalid metric: ' + str(metric)
            raise pexceptions.PySegInputError(expr='build_ncc_z2d (ClassStar)', msg=error_msg)

        # Multiprocessing
        if npr is None:
            npr = mp.cpu_count()
        processes = list()
        # Create the list on indices to split
        npart = self.__star.get_nrows()
        sym_ids = list(it.combinations(list(range(npart)), r=2))
        spl_ids = np.array_split(list(range(len(sym_ids))), npr)
        shared_mat = mp.Array('f', int(npart*npart))

        # Particles loop (Parallel)

        if npr <= 1:
            pr_cc_2d(-1, spl_ids[0], sym_ids, self.__particles, self.__masks, p_metric,
                     shared_mat)
        else:
            for pr_id in range(npr):
                pr = mp.Process(target=pr_cc_2d, args=(pr_id, spl_ids[pr_id], sym_ids, self.__particles,
                                                       self.__masks, p_metric,
                                                       shared_mat))
                pr.start()
                processes.append(pr)
            pr_results = list()
            for pr in processes:
                pr.join()
                pr_results.append(pr.exitcode)
            for pr_id in range(len(processes)):
                if pr_id != pr_results[pr_id]:
                    error_msg = 'Process ' + str(pr_id) + ' exited unexpectedly!'
                    raise pexceptions.PySegInputError(expr='build_ccc (ClassStar)', msg=error_msg)
            gc.collect()

        # Fill diagonal with the maximum value for normalized cross-correlation
        self.__cc = np.frombuffer(shared_mat.get_obj(), dtype=np.float32).reshape(npart, npart)
        if p_metric == 0:
            np.fill_diagonal(self.__cc, np.finfo(np.float32).max)
        elif p_metric == 1:
            np.fill_diagonal(self.__cc, 0)

    # Compute the image momentes for every particle
    # mode: 'spatial' spatial moments (default), 'central' central moments and 'normalized' for central normalized ones,
    #       'raw' moments are just the normalized flattered gray values of the masked particles
    # npr: number of process, None (defaults) means they are adjusted automatically to the number of cores
    def build_moments(self, mode='spatial', npr=None):

        # Input parsing
        if self.__mask_gl is None:
            error_msg = 'There is no particles load call load_particles() method first!'
            raise pexceptions.PySegInputError(expr='build_moments (ClassStar)', msg=error_msg)
        if mode == 'spatial':
            mode_p, nfeat = 0, 10
        elif mode == 'central':
            mode_p, nfeat = 1, 7
        elif mode == 'normalized':
            mode_p, nfeat = 2, 7
        elif mode == 'raw':
            mode_p, nfeat = 3, np.sum(self.__mask_gl)
        else:
            error_msg = 'Invalid option for moments mode: ' + str(mode)
            raise pexceptions.PySegInputError(expr='build_moments (ClassStar)', msg=error_msg)

        # Multiprocessing
        if npr is None:
            npr = mp.cpu_count()
        processes = list()
        # Create the list on indices to split
        npart = self.__star.get_nrows()
        sym_ids = list(np.arange(npart))
        spl_ids = np.array_split(list(range(len(sym_ids))), npr)
        shared_mat = mp.Array('f', int(nfeat*npart))

        # Particles loop (Parallel)
        if npr <= 1:
            pr_mome_2d(-1, spl_ids[0], mode_p, nfeat, self.__particles, self.__mask_gl,
                       shared_mat)
        else:
            for pr_id in range(npr):
                pr = mp.Process(target=pr_mome_2d, args=(pr_id, spl_ids[pr_id], mode_p, nfeat,
                                                         self.__particles, self.__mask_gl,
                                                         shared_mat))
                pr.start()
                processes.append(pr)
            pr_results = list()
            for pr in processes:
                pr.join()
                pr_results.append(pr.exitcode)
            for pr_id in range(len(processes)):
                if pr_id != pr_results[pr_id]:
                    error_msg = 'Process ' + str(pr_id) + ' exited unexpectedly!'
                    raise pexceptions.PySegInputError(expr='build_moments (ClassStar)', msg=error_msg)
            gc.collect()

        # Set class variable for moments
        self.__momes = np.ones(shape=npart, dtype=object)
        for i in range(npart):
            self.__momes[i] = np.asarray(shared_mat[i:i+nfeat], dtype=float)
        # self.__momes = np.frombuffer(shared_mat.get_obj(), dtype=np.float32).reshape(nfeat, npart)

    def build_vectors(self):
        '''
        Reshape each ZAV images (nxm) to an array (vector 1xnm)
        :return: None, vectors are stored in a internal variable
        '''

        # Input parsing
        if (self.__particles is None) or (len(self.__particles) <= 0):
            error_msg = 'There is no particles load call load_particles() method first!'
            raise pexceptions.PySegInputError(expr='build_vectors (ClassStar)', msg=error_msg)

        # Set class variable for moments
        img_n, img_m = self.__particles[0].shape
        npart, nfeat = len(self.__particles), img_n * img_m
        self.__vectors = np.ones(shape=(npart,nfeat), dtype=np.float32)
        for i in range(npart):
            self.__vectors[i, :] = self.__particles[i].reshape(1, nfeat)

    def save_vectors(self, fname, txt=False):
        """
        Saves vectors in file as numpy (2D)array
        :param fname: full path where vectors 2D numpy array will be stored, see numpy.save for more details
        :param txt: if False (default) the is saved as binary data, otherwise as human readable text
        :return
        """

        # Input parsing
        if self.__vectors is None:
            error_msg = 'No CC matrix, call build_vectors() methods first!'
            raise pexceptions.PySegInputError(expr='save_cc (ClassStar)', msg=error_msg)

        if txt:
            np.savetxt(fname, self.__vectors)
        else:
            np.save(fname, self.__vectors)

    # Reduces moments dimensionality (see SciPy sklearn for detailed information)
    # n_comp: number of components (moments) after the reductions, default 3, if 'mle' the Minka's MLE is used to
    #         guess the dimension
    # method: valid, 'pca' PCA and 'ltsa' locally linear embedding with local tangent space alignment algorithm
    def moments_dim_reduction(self, n_comp=3, method='pca'):

        # Methods settings
        if method == 'pca':
            ml = PCA(n_components=n_comp)
        elif method == 'ltsa':
            ml = LocallyLinearEmbedding(n_neighbors=5, n_components=n_comp, eigen_solver='dense',
                                        method='ltsa')

        # Training
        X = np.zeros(shape=(self.__momes.shape[0], self.__momes[0].shape[0]), dtype=float)
        for i in range(self.__momes.shape[0]):
            X[i, :] = np.asarray(self.__momes[i], dtype=float)
        ml.fit(X)

        # Transformation
        Xt = ml.transform(X)
        self.__momes = np.zeros(shape=Xt.shape[0], dtype=object)
        for i in range(Xt.shape[0]):
            self.__momes[i] = Xt[i, :]

    # Reduces moments dimensionality (see SciPy sklearn for detailed information)
    # n_comp: number of components (moments) after the reductions, default 3, if 'mle' the Minka's MLE is used to
    #         guess the dimension
    # method: valid, 'pca' PCA and 'ltsa' locally linear embedding with local tangent space alignment algorithm
    def vectors_dim_reduction(self, n_comp=3, method='pca'):
        '''
        Reduces vectors dimensionality (see SciPy sklearn for detailed information)
        :param n_comp: number of components (moments) after the reductions, if None then n_comp == min(n_samples, n_features) - 1
        :param method: valid, 'pca' PCA and 'ltsa' locally linear embedding with local tangent space alignment algorithm
        :return: vectors internal variable is updated, if method='pca' then percentage of variance explained
                 for each selected component
        '''

        # Methods settings
        if method == 'pca':
            ml = PCA(n_components=n_comp)
        elif method == 'ltsa':
            ml = LocallyLinearEmbedding(n_neighbors=5, n_components=n_comp, eigen_solver='dense', method='ltsa')

        # Training
        ml.fit(self.__vectors)

        # Transformation
        self.__vectors = ml.transform(self.__vectors)

        if method == 'pca':
            return ml.explained_variance_ratio_

    # Returns a .vtp file with information about moments classification
    def moments_to_vtp(self):

        # Input parsing
        if self.__momes is None:
            error_msg = 'Inputs moments are not computed yet!, class build_moments() first!'
            raise pexceptions.PySegInputError(expr='moments_to_vtp (ClassStar)', msg=error_msg)
        if self.__momes[0].shape[0] > 3:
            error_msg = 'Only moments with dimensionality lower than 3 can be represented, currently ' + \
                        str(self.__momes[0].shape) + ', call moments_pca_reduction() function for dimensionality reduction'
            raise pexceptions.PySegInputError(expr='moments_to_vtp (ClassStar)', msg=error_msg)
        # PySeg columns
        ps_cols = False
        if self.__star.has_column('_psAPClass') and self.__star.has_column('_psAPCenter'):
            ps_cols = True

        # Generate the VTK Poly Data file
        poly, points, verts = vtk.vtkPolyData(), vtk.vtkPoints(), vtk.vtkCellArray()
        klass = vtk.vtkIntArray()
        klass.SetNumberOfComponents(1), klass.SetName('rlnClassNumber')
        if ps_cols:
            klass_ap = vtk.vtkIntArray()
            klass_ap.SetNumberOfComponents(1), klass_ap.SetName('psAPClass')
            cent_ap = vtk.vtkIntArray()
            cent_ap.SetNumberOfComponents(1), cent_ap.SetName('psAPCenter')
        for row, mome in enumerate(self.__momes):
            points.InsertPoint(row, (mome[0], mome[1], mome[2]))
            verts.InsertNextCell(1)
            verts.InsertCellPoint(row)
            klass.InsertTuple(row, (self.__star.get_element('_rlnClassNumber', row),))
            if ps_cols:
                klass_ap.InsertTuple(row, (self.__star.get_element('_psAPClass', row),))
                cent_ap.InsertTuple(row, (self.__star.get_element('_psAPCenter', row),))
        poly.SetPoints(points)
        poly.SetVerts(verts)
        poly.GetPointData().AddArray(klass)
        if ps_cols:
            poly.GetPointData().AddArray(klass_ap)
            poly.GetPointData().AddArray(cent_ap)

        return poly

    def affinity_propagation(self, mode_in='ncc_2dz', damping=0.5, preference=None, max_iter=200, convergence_iter=15,
                             verbose=False):
        '''
        Classification based on Affinity Propagation (AP) algorithm (it requires call a build_*() method first)
        AP documentation in www.skit-learn.org
        :param mode_in: mode for input data, valid 'ncc_2dz' (default), 'moments' and 'vectors'
        :param damping: AP dumping parameter [0.5, 1), it controls convergence speed (default 0.5)
        :param preference: AP preference parameter (-infty, infty), the smaller the higher number of potential classes
    #             (default None, the median of the affinity class is considered)
        :param max_iter: AP maximum number of iterations parameters (default 200)
        :param convergence_iter: number of iteration for fitting the convergence criterium (default 15)
        :param verbose: if True (default False) some messages are printed out
        :return: Calassification output is stored in member variables
        '''

        # Input parsing
        if (mode_in != 'ncc_2dz') and (mode_in != 'moments') and (mode_in != 'vectors'):
            error_msg = 'Invalid input for mode_in option: ' + str(mode_in)
            raise pexceptions.PySegInputError(expr='affinity_propagation (ClassStar)', msg=error_msg)
        if (mode_in == 'ncc_2dz') and (self.__cc is None):
            error_msg = 'No CC matrix, call build_ncc_ndz() method first!'
            raise pexceptions.PySegInputError(expr='affinity_propagation (ClassStar)', msg=error_msg)
        if (mode_in == 'moments') and (self.__momes is None):
            error_msg = 'No moments computed, call build_moments() method first!'
            raise pexceptions.PySegInputError(expr='affinity_propagation (ClassStar)', msg=error_msg)
        if (mode_in == 'vectors') and (self.__vectors is None):
            error_msg = 'No vectors computed, call build_vectors() method first!'
            raise pexceptions.PySegInputError(expr='affinity_propagation (ClassStar)', msg=error_msg)

        # AP algorithm
        if mode_in == 'ncc_2dz':
            aff = AffinityPropagation(damping=damping,
                                  convergence_iter=convergence_iter,
                                  max_iter=max_iter,
                                  affinity='precomputed',
                                  preference=preference)
            aff.fit(self.__cc)
        elif mode_in == 'moments':
            aff = AffinityPropagation(damping=damping,
                                  convergence_iter=convergence_iter,
                                  max_iter=max_iter,
                                  affinity='euclidean',
                                  preference=preference)
            X = np.zeros(shape=(self.__momes.shape[0], self.__momes[0].shape[0]), dtype=float)
            for i in range(self.__momes.shape[0]):
                X[i, :] = np.asarray(self.__momes[i], dtype=float)
            aff.fit(X)
        elif mode_in == 'vectors':
            aff = AffinityPropagation(damping=damping,
                                      convergence_iter=convergence_iter,
                                      max_iter=max_iter,
                                      affinity='euclidean',
                                      preference=preference)
            aff.fit(self.__vectors)
        if aff.n_iter_ >= max_iter:
            print('WARNING: affinity_propagation (ClassStar): AP reached the maximum iterations (' + str(max_iter) + \
                ') without converging, its results might be meaningless!')
        lbls, centers = np.asarray(aff.labels_, dtype=int), np.asarray(aff.cluster_centers_indices_, dtype=int)
        class_ids = np.asarray(list(set(lbls)))
        n_classes = len(class_ids)

        if verbose:
            print('Affinity Matrix statistics:')
            print('\t\t-Percentile 5%: ' + str(np.percentile(aff.affinity_matrix_, 5)))
            print('\t\t-Median: ' + str(np.percentile(aff.affinity_matrix_, 50)))
            print('\t\t-Percentile 95%: ' + str(np.percentile(aff.affinity_matrix_, 95)))
            print('AFFINITY PROPAGATION RESULTS: ')
            print('\tNumber of classes found: ' + str(n_classes))
            for class_id in class_ids:
                print('\t\t-Number of particles in class ' + str(class_id) + ': ' + str((lbls==class_id).sum()))

        # AP classes grouped
        self.__ap_classes, self.__ap_centers = dict(), dict()
        for lbl in class_ids:
            self.__ap_classes[lbl] = list()
            self.__ap_centers[lbl] = centers[lbl]

        # Class assignment to the STAR file object
        self.__star.add_column('_psAPClass')
        self.__star.add_column('_psAPCenter', val=-1)
        for row in range(self.__star.get_nrows()):
            self.__ap_classes[lbls[row]].append(row)
            self.__star.set_element('_psAPClass', row, lbls[row])
        for i, center_id in enumerate(centers):
            self.__star.set_element('_psAPCenter', center_id, class_ids[i])

    # Affinity Propagation (AP) approximation to cluster non-metric (CC matrices) spaces in a memory safe way
    # mode_in: mode for input data, valid 'ncc_2dz' (default) and 'moments'
    # sset_size: size for random subsets, it should be considerably large thant the number of classes
    # sset_iter: number of random subsets to process before computing final exemplars, if time is not an issue then
    #            the bigger the better
    # imgs_dir: directory to store the flattern images generated (default None)
    # rest parameters: AP settings (see affinity_propagation(), build_ncc_z3d(), load_particles())
    # verbose: if True (default False) some messages are printed out
    def affinity_propagation_memsafe(self, mask, sset_size=3000, sset_iter=10, damping=0.5, preference=None, max_iter=200,
                                     convergence_iter=15, metric='cc',
                                     low_sg=0, seg_dil_it=2, rln_norm=True, avg_norm=False, rad_3D=False,
                                     direct_rec=False, ref_dir=None, imgs_dir=None,
                                     npr=None, verbose=False):

        # Input parsing
        if verbose:
            print('AP_memsafe: initialization...')
        if sset_size < 0:
            error_msg = 'Input subset size must be greater an zero!'
            raise pexceptions.PySegInputError(expr='affinity_propagation_memsafe (ClassStar)', msg=error_msg)
        elif sset_size >= self.__star.get_nrows():
            print('WARNING: it does not make sense called memory safe version of AP with the subset size greater or ' + \
                'equal to the amount of particles. Calling standard implementation instead!')
            self.affinity_propagation(damping=damping, preference=preference, max_iter=max_iter,
                                      convergence_iter=convergence_iter, verbose=verbose)

        # Getting the random subsets
        ssets, nparts = list(), self.__star.get_nrows()
        for i in range(sset_iter):
            ssets.append(np.random.randint(0, high=nparts, size=sset_size))

        # Loop with call to standard of AP on random subsets
        ex_ids = list()
        for i, sset in enumerate(ssets):
            if verbose:
                print('AP_memsafe: AP on subset ' + str(i+1) + ' of ' + str(sset_iter))
            # Building current CC matrix
            hold_cc = self.__build_ncc_z2d_sset(sset, mask, metric=metric, npr=npr,
                                                low_sg=low_sg, seg_dil_it=seg_dil_it, rln_norm=rln_norm,
                                                avg_norm=avg_norm, rad_3D=rad_3D,
                                                direct_rec=direct_rec, ref_dir=ref_dir)
            # Affinity propagation to get current exemplars
            hold_exs = self.__affinity_propagation_cc(hold_cc, damping=damping, preference=preference,
                                                   max_iter=max_iter, convergence_iter=convergence_iter)[1]
            if verbose:
                print('\t-Intermediate exemplars found: ' + str(len(hold_exs)))
            # Update list of exemplars
            for ex in hold_exs:
                ex_ids.append(sset[ex])

        # AP on intermediate exemplars
        cc = self.__build_ncc_z2d_sset(ex_ids, mask, metric=metric, npr=npr,
                                       low_sg=low_sg, seg_dil_it=seg_dil_it, rln_norm=rln_norm,
                                       avg_norm=avg_norm, rad_3D=rad_3D,
                                       direct_rec=direct_rec, ref_dir=ref_dir)
        exs = self.__affinity_propagation_cc(cc, damping=damping, preference=preference,
                                             max_iter=max_iter, convergence_iter=convergence_iter)[1]
        if verbose:
            print('Final exemplars found: ' + str(len(exs)))
        final_ex_ids = list()
        for ex in exs:
            final_ex_ids.append(ex_ids[ex])

        # Classifying all particles respect to exemplars
        self.__class_by_exemplars(final_ex_ids, mask, low_sg=low_sg, seg_dil_it=seg_dil_it, rln_norm=rln_norm,
                                  avg_norm=avg_norm, rad_3D=rad_3D, npr=npr, ref_dir=ref_dir, imgs_dir=imgs_dir,
                                  direct_rec=direct_rec)

        # Class assignment to the STAR file object
        self.__star.add_column('_psAPClass')
        self.__star.add_column('_psAPCenter', val=-1)
        self.__ap_classes, self.__ap_centers = dict(), dict()
        for row in range(self.__star.get_nrows()):
            lbl = self.__cc_rows[row, :].argmax()
            try:
                self.__ap_classes[lbl].append(row)
            except KeyError:
                self.__ap_classes[lbl] = list()
                self.__ap_classes[lbl].append(row)
            self.__star.set_element('_psAPClass', row, lbl)
        for lbl, rows in zip(iter(self.__ap_classes.keys()), iter(self.__ap_classes.values())):
            self.__ap_centers[lbl] = final_ex_ids[lbl]
            self.__star.set_element('_psAPCenter', final_ex_ids[lbl], lbl)

    def agglomerative_clustering(self, mode_in='ncc_2dz', n_clusters=2, linkage='ward', knn=None, verbose=False):
        '''
        Classification based on Agglomerative Clustering (AG) algorithm (it requires call a build_*() method first)
        AG documentation in www.skit-learn.org
        :param mode_in: mode for input data, valid 'ncc_2dz' (default), 'moments' and 'vectors'
        :param n_clusters: number of clusters to find (default 2)
        :param linkage: valid: 'ward' (default), 'complete' and 'average'
        :param knn: number of neighbours for building knn graph for recovering local connectivity (default None)
        :param verbose: if True (default False) some messages are printed out
        :return:
        '''

        # Input parsing
        if (mode_in != 'ncc_2dz') and (mode_in != 'moments') and  (mode_in != 'vectors'):
            error_msg = 'Invalid input for mode_in option: ' + str(mode_in)
            raise pexceptions.PySegInputError(expr='agglomerative_clustering (ClassStar)', msg=error_msg)
        if (mode_in == 'ncc_2dz') and (self.__cc is None):
            error_msg = 'No CC matrix, call build_ncc_ndz() method first!'
            raise pexceptions.PySegInputError(expr='agglomerative_clustering (ClassStar)', msg=error_msg)
        if (mode_in == 'moments') and (self.__momes is None):
            error_msg = 'No moments computed, call build_moments() method first!'
            raise pexceptions.PySegInputError(expr='agglomerative_clustering (ClassStar)', msg=error_msg)
        if (mode_in == 'vectors') and (self.__vectors is None):
            error_msg = 'No vectors computed, call build_vectors() method first!'
            raise pexceptions.PySegInputError(expr='affinity_propagation (ClassStar)', msg=error_msg)

        # AG algorithm
        if mode_in == 'ncc_2dz':
            X = lin_map(self.__cc, lb=1., ub=0.)
        elif mode_in == 'moments':
            X = np.zeros(shape=(self.__momes.shape[0], self.__momes[0].shape[0]), dtype=float)
            for i in range(self.__momes.shape[0]):
                X[i, :] = np.asarray(self.__momes[i], dtype=float)
        else:
            X = self.__vectors
        conn = None
        if (knn is not None) and (knn > 0):
            conn = kneighbors_graph(X, n_neighbors=knn, include_self=None)
            agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, affinity='precomputed',
                                      connectivity=conn)
        else:
            if (mode_in == 'moments') or (mode_in == 'vectors'):
                agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, affinity='euclidean')
            else:
                agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, affinity='precomputed')
        agg.fit(X)
        lbls = np.asarray(agg.labels_, dtype=int)
        class_ids = np.asarray(list(set(lbls)))
        n_classes = len(class_ids)

        if verbose:
            print('AGGLOMERATIVE CLUSTERING RESULTS: ')
            print('\tNumber of classes found: ' + str(n_classes))
            for class_id in class_ids:
                print('\t\t-Number of particles in class ' + str(class_id) + ': ' + str((lbls==class_id).sum()))

        # Class assignment to the STAR file object
        self.__star.add_column('_psAPClass')
        self.__star.add_column('_psAPCenter', val=-1)
        for row in range(self.__star.get_nrows()):
            self.__star.set_element('_psAPClass', row, lbls[row])

    def agglomerative_clustering_ref(self, n_clusters=2, mode='averages', pca_ncomp=None, linkage='ward', verbose=False):
        '''
        Classification refinement based on Agglomerative Clustering (AG) algorithm to refine Affinity Propagation results
        (it requires call a affinity_propagation() method first)
        :param n_clusters: number of clusters to find (default 2)
        :param imgs: type of image used to represent each class, either 'averages' (default) or 'exemplars'
        :param pca_ncomp: if not None (default) number of components for PCA dimensionality reduction
        :param linkage: valid: 'ward' (default), 'complete' and 'average'
        :param verbose: if True (default False) some messages are printed out
        :return:
        '''

        # Input parsing
        if len(self.__particles) <= 0:
            error_msg = 'No particles to process!'
            raise pexceptions.PySegInputError(expr='agglomerative_clustering_ref (ClassStar)', msg=error_msg)
        if not self.__star.has_column('_psAPCenter'):
            error_msg = 'No psAPCenter information, call function affinity_propagation() first!'
            raise pexceptions.PySegInputError(expr='agglomerative_clustering_ref (ClassStar)', msg=error_msg)

        # Getting the output particles
        n_dim = len(self.__particles[0].shape)
        klass_col = self.__star.get_column_data('_rlnClassNumber')
        klass_cent = self.__star.get_column_data('_psAPCenter')
        klass_ids = set(klass_col)
        hold_particles = np.zeros(shape=len(klass_ids), dtype=object)
        hold_row_ids = np.zeros(shape=len(klass_ids), dtype=int)
        d_parts, d_count = dict(), dict()
        for k_id in klass_ids:
            d_parts[k_id] = np.zeros(shape=self.__particles[0].shape, dtype=self.__particles[0].dtype)
            d_count[k_id] = 0
        for row, k_id, k_cent in zip(list(range(len(klass_col))), klass_col, klass_cent):
            d_parts[k_id] += self.__particles[row]
            d_count[k_id] += 1
            if k_cent > 0:
                hold_row_ids[k_cent] = row
        if mode == 'averages':
            for k_id in klass_ids:
                hold_avg, hold_nparts = d_parts[k_id], float(d_count[k_id])
                if hold_nparts > 0:
                    hold_avg /= hold_nparts
                    hold_particles[k_id] = hold_avg
        elif mode == 'exemplars':
            for row in range(self.__star.get_nrows()):
                k_cent = self.__star.get_element('_psAPCenter', row)
                if k_cent >= 0:
                    hold_row_ids[k_cent] = row
                    hold_particles[k_cent] = self.__particles[row]
        else:
            error_msg = 'Mode ' + str(mode) + ' is not valid!'
            raise pexceptions.PySegInputError(expr='gen_sub_star_class (ClassStar)', msg=error_msg)

        # Image to vectors
        img_n, img_m = hold_particles[0].shape
        npart, nfeat = len(hold_particles), img_n * img_m
        vectors = np.ones(shape=(npart, nfeat), dtype=np.float32)
        for i in range(npart):
            vectors[i, :] = hold_particles[i].reshape(1, nfeat)

        # PCA dimensionality reduction
        if pca_ncomp is not None:
            ml = PCA(n_components=pca_ncomp)
            ml.fit(vectors)
            evs = ml.explained_variance_ratio_
            vectors = ml.transform(vectors)

        # AG algorithm
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, affinity='euclidean')
        agg.fit(vectors)
        lbls = np.asarray(agg.labels_, dtype=int)
        class_ids = np.asarray(list(set(lbls)))
        n_classes = len(class_ids)

        if verbose:
            print('AGGLOMERATIVE REFINEMENT RESULTS: ')
            if pca_ncomp is not None:
                print('\tPCA energy kept in ' + str(pca_ncomp) + ': ' + str(round(100*evs.sum(),2)))
            print('\tNumber of classes found: ' + str(n_classes))
            for class_id in class_ids:
                print('\t\t-Number of particles in class ' + str(class_id) + ': ' + str(np.where(lbls == class_id)[0]))

        # Class assignment to the STAR file object
        self.__star.add_column('_psAPClass')
        self.__star.add_column('_psAPCenter', val=-1)
        for row in range(self.__star.get_nrows()):
            self.__star.set_element('_psAPClass', row, lbls[klass_col[row]])

    def kmeans_clustering(self, n_clusters=2, verbose=False):
        '''
        Classification based on Kmeans clustering algorithm (it requires call a build_vectors() method first)
        Kmeans documentation in www.skit-learn.org
        :param n_clusters: number of clusters to find (default 2)
        :param verbose: if True (default False) some messages are printed out
        :return:
        '''

        # Input parsing
        if self.__vectors is None:
            error_msg = 'No vectors computed, call build_vectors() method first!'
            raise pexceptions.PySegInputError(expr='kmeans_clustering (ClassStar)', msg=error_msg)

        # Kmeans algorithm
        kms = KMeans(n_clusters=n_clusters)
        kms.fit(self.__vectors)
        lbls = np.asarray(kms.labels_, dtype=int)
        class_ids = np.asarray(list(set(lbls)))
        n_classes = len(class_ids)

        if verbose:
            print('AFFINITY CLUSTERING RESULTS: ')
            print('\tNumber of classes found: ' + str(n_classes))
            for class_id in class_ids:
                print('\t\t-Number of particles in class ' + str(class_id) + ': ' + str((lbls == class_id).sum()))

        # Class assignment to the STAR file object
        self.__star.add_column('_psAPClass')
        self.__star.add_column('_psAPCenter', val=-1)
        for row in range(self.__star.get_nrows()):
            self.__star.set_element('_psAPClass', row, lbls[row])

    # Updates RELION's classification with the classes obtained here
    def update_relion_classes(self):

        # Input parsing
        if not self.__star.has_column('_psAPClass'):
            print('WARNING update_relion_classes (ClassStar): No classification done yet, so class information is not modified!')
            return
        if not self.__star.has_column('_rlnClassNumber'):
            self.__star.add_column(key='_rlnClassNumber', val=1)

        # Class updating
        for row in range(self.__star.get_nrows()):
            self.__star.set_element('_rlnClassNumber', row, self.__star.get_element('_psAPClass', row))

    # Compute CC statistics for AP classification
    # reference: reference 2D image used for classes, valid: 'exemplar' (default) and 'average'
    def compute_ccap_stat(self, reference='exemplar'):

        # Input parsing
        if (reference != 'exemplar') and (reference != 'average'):
            error_msg = 'Invalid input for mode_in option: ' + str(reference)
            raise pexceptions.PySegInputError(expr='compute_ccap_stat (ClassStar)', msg=error_msg)
        if (not self.__star.has_column('_psAPClass')) or (not self.__star.has_column('_psAPCenter')):
            error_msg = 'It requires affinity_propagation classification to be called before!'
            raise pexceptions.PySegInputError(expr='compute_ccap_stat (ClassStar)', msg=error_msg)
        if self.__cc is None:
            error_msg = 'No CC matrix, call build_ncc_ndz() method first!'
            raise pexceptions.PySegInputError(expr='compute_ccap_stat (ClassStar)', msg=error_msg)
        if self.__particles.shape[0] != self.__star.get_nrows():
            error_msg = 'Unexpected event!'
            raise pexceptions.PySegInputError(expr='compute_ccap_stat (ClassStar)', msg=error_msg)

        # Getting the reference 2D images
        particles_ref, masks_ref = np.zeros(shape=len(self.__ap_classes), dtype=object), \
                                   np.zeros(shape=len(self.__ap_classes), dtype=object)
        if reference == 'average':
            klass_col = self.__star.get_column_data('_psAPClass')
            d_parts, d_count, d_mask = dict(), dict(), dict()
            for k_id in list(self.__ap_classes.keys()):
                d_parts[k_id] = np.zeros(shape=self.__particles[0].shape, dtype=self.__particles[0].dtype)
                d_count[k_id] = 0
                d_mask[k_id] = np.ones(shape=self.__masks[0].shape, dtype=bool)
            for i, k_id in enumerate(klass_col):
                d_parts[k_id] += self.__particles[i]
                d_count[k_id] += 1
                d_mask[k_id] *= self.__masks[i]
            for k_id in list(self.__ap_classes.keys()):
                hold_avg, hold_nparts = d_parts[k_id], float(d_count[k_id])
                if hold_nparts > 0:
                    hold_avg /= hold_nparts
                particles_ref[k_id] = hold_avg
                masks_ref[k_id] = d_mask[k_id]
        elif reference == 'exemplar':
            for k_id, row in zip(list(self.__ap_centers.keys()), list(self.__ap_centers.values())):
                particles_ref[k_id] = self.__particles[row]
                masks_ref[k_id] = self.__masks[row]

        # Compute NCC to reference particle
        self.__ap_ref_cc = dict()
        for k_id in list(self.__ap_classes.keys()):
            particle_ref, mask_ref = particles_ref[k_id], masks_ref[k_id]
            hold_ccs = np.zeros(shape=len(self.__ap_classes[k_id]), dtype=float)
            for i, row in enumerate(self.__ap_classes[k_id]):
                # Extract the particle information
                hold_particle, hold_mask = self.__particles[row], self.__masks[row]
                # 2D Cross-Correlation
                mask = (hold_mask * mask_ref) > 0
                n_mask = float(mask.sum())
                if n_mask > 0:
                    cte = 1. / n_mask
                    if cte > 0:
                        hold_ccs[i] = cte * np.sum(hold_particle[mask]*particle_ref[mask])
            self.__ap_ref_cc[k_id] = hold_ccs

    # Compute CC statistics for AP classification
    # imgs_dir: directory with the particle images
    # mask_3d: mask volume in 3D
    # reference: reference 2D image used for classes, valid: 'exemplar' (default) and 'average'
    def compute_ccap_stat_dir(self, mask_3d, imgs_dir, reference='exemplar'):

        # Input parsing
        if (reference != 'exemplar') and (reference != 'average'):
            error_msg = 'Invalid input for mode_in option: ' + str(reference)
            raise pexceptions.PySegInputError(expr='compute_ccap_stat_dir (ClassStar)', msg=error_msg)
        if (not self.__star.has_column('_psAPClass')) or (not self.__star.has_column('_psAPCenter')):
            error_msg = 'It requires affinity_propagation classification to be called before!'
            raise pexceptions.PySegInputError(expr='compute_ccap_stat_dir (ClassStar)', msg=error_msg)
        in_zero = imgs_dir + '/particle_0.npy'
        if not os.path.exists(in_zero):
            error_msg = 'No image zero in images directory!'
            raise pexceptions.PySegInputError(expr='save_class_dir (ClassStar)', msg=error_msg)
        img_zero = np.load(in_zero)

        # Getting the reference 2D images
        averager = RadialAvg3D(shape=mask_3d.shape)
        mask_2d = averager.avg_vol(mask_3d) > 0.5
        n_mask = float(mask_2d.sum())
        particles_ref, masks_ref = np.zeros(shape=len(self.__ap_classes), dtype=object), \
                                   np.zeros(shape=len(self.__ap_classes), dtype=object)
        if reference == 'average':
            klass_col = self.__star.get_column_data('_psAPClass')
            d_parts, d_count, d_mask = dict(), dict(), dict()
            for k_id in list(self.__ap_classes.keys()):
                d_parts[k_id] = np.zeros(shape=img_zero.shape, dtype=imgs_dir.dtype)
                d_count[k_id] = 0
                d_mask[k_id] = np.ones(shape=mask_2d.shape, dtype=bool)
            for i, k_id in enumerate(klass_col):
                in_fname = imgs_dir + '/particle_' + str(i) + '.npy'
                hold_particle = np.load(in_fname)
                d_parts[k_id] += hold_particle
                d_count[k_id] += 1
                d_mask[k_id] *= mask_2d
            for k_id in list(self.__ap_classes.keys()):
                hold_avg, hold_nparts = d_parts[k_id], float(d_count[k_id])
                if hold_nparts > 0:
                    hold_avg /= hold_nparts
                particles_ref[k_id] = hold_avg
                masks_ref[k_id] = d_mask[k_id]
        elif reference == 'exemplar':
            for k_id, row in zip(list(self.__ap_centers.keys()), list(self.__ap_centers.values())):
                in_fname = imgs_dir + '/particle_' + str(row) + '.npy'
                hold_particle = np.load(in_fname)
                particles_ref[k_id] = hold_particle
                masks_ref[k_id] = mask_2d

        # Compute NCC to reference particle
        self.__ap_ref_cc = dict()
        for k_id in list(self.__ap_classes.keys()):
            particle_ref, mask_ref = particles_ref[k_id], masks_ref[k_id]
            hold_ccs = np.zeros(shape=len(self.__ap_classes[k_id]), dtype=float)
            for i, row in enumerate(self.__ap_classes[k_id]):
                # Extract the particle information
                in_fname = imgs_dir + '/particle_' + str(row) + '.npy'
                hold_particle = np.load(in_fname)
                # 2D Cross-Correlation
                if n_mask > 0:
                    cte = 1. / n_mask
                    if cte > 0:
                        hold_ccs[i] = cte * np.sum(hold_particle[mask_2d]*particle_ref[mask_2d])
            self.__ap_ref_cc[k_id] = hold_ccs

    # Print CC statistics for AP classification
    # percentile: percentile used to be computed
    def print_ccap_stat(self, percentile=5):

        # Input parsing
        if (percentile < 0) and (percentile > 100):
            error_msg = 'Input percentile must be between [0, 100], current ' + str(percentile)
            raise pexceptions.PySegInputError(expr='print_ccap_stat (ClassStar)', msg=error_msg)
        if self.__ap_ref_cc is None:
            error_msg = 'This method requires CCAP statistics computation, call compute_ccap_stat() first!'
            raise pexceptions.PySegInputError(expr='print_ccap_stat (ClassStar)', msg=error_msg)

        # Initialization
        per_l, per_m, per_h = float(percentile), 50., 100.-float(percentile)
        per_msg = '[' + str(per_l) + ', ' + str(per_m) + ', ' + str(per_h) + ']'

        # Printing loop
        print('Cross-Correlation Statistics for Affinity Propagation classes: ')
        for k_id in self.__ap_classes:
            hold_ccs = self.__ap_ref_cc[k_id]
            print('\t-Class ' + str(k_id) + ': ')
            print('\t\t+Mean, Std: ' + str(hold_ccs.mean()) + ', ' + str(hold_ccs.std()))
            print('\t\t+Percentiles ' + per_msg + ': ' + '[' + str(np.percentile(hold_ccs, per_l)) + ', ' + \
                  str(np.percentile(hold_ccs, per_m)) + ', ' + str(np.percentile(hold_ccs, per_h)) + ']')

    # Store particles information in a STAR file
    # out_dir: output directory
    # out_stem: output file stem
    # parse_rln: if True (default False) al non-Relion compatible columns are erased
    # mode: valid, 'split' a separated file for every class is created, 'gather' (default) all classes in the same file
    #       'particles' then particles are stored
    def save_star(self, out_dir, out_stem, parse_rln=False, mode='gather'):

        if parse_rln:
            hold_star = self.__star.get_relion_parsed_copy()
        else:
            hold_star = self.__star

        if mode == 'gather':
            hold_star.store(out_dir + '/' + out_stem + '_gather.star')

        elif mode == 'split':
            out_stars = hold_star.split_class()
            for i, star in enumerate(out_stars):
                out_file = out_dir + '/' + out_stem + '_k' + str(i) + '_split.star'
                star.store(out_file)

        # Parsing for storing particles modes
        elif self.__particles is None:
            error_msg = 'No particles to store in mode: ' + str(mode)
            raise pexceptions.PySegInputError(expr='save_star (ClassStar)', msg=error_msg)

        elif mode == 'particle':
            count, count_2 = 0, 0
            nrows = hold_star.get_nrows()
            nrows_2 = 2 * nrows
            while (count < nrows) and (count_2 < nrows_2):
                klass = hold_star.get_element('_rlnClassNumber', count)
                hold_dir = out_dir + '/' + out_stem + '_particles_k' + str(klass)
                try:
                    if len(self.__particles[count].shape) == 2:
                        part_name = hold_dir+'/particle_'+str(count)+'.png'
                        imwrite(part_name, lin_map(self.__particles[count],0,np.iinfo(np.uint16).max).astype(np.uint16))
                    else:
                        part_name = hold_dir+'/particle_'+str(count)+'.mrc'
                        disperse_io.save_numpy(self.__particles[count], part_name)
                    count += 1
                except (IOError, OSError, pexceptions.PySegInputError):
                    if not os.path.exists(hold_dir):
                        os.makedirs(hold_dir)
                count_2 += 1

        else:
            error_msg = 'Mode ' + str(mode) + ' is not valid!'
            raise pexceptions.PySegInputError(expr='save_star (ClassStar)', msg=error_msg)

    # Purge particles in classes with a small amount of particles
    # th: threshold, if a class has fewer particles than 'th' then their particles are purged
    # Return: a dictionary with the _rlnClassNumber as index and the amount particles in these purged classes
    def purge_small_classes(self, th):

        # Initialization
        d_rows, d_nparts = dict(), dict()

        # Purge
        d_fnparts = None
        if th > 0:
            hold_star = copy.deepcopy(self.__star)
            klass_col = hold_star.get_column_data('_rlnClassNumber')
            klass_ids = set(klass_col)
            for k_id in klass_ids:
                d_rows[k_id], d_nparts[k_id] = list(), 0
            for row, k_id in enumerate(klass_col):
                d_rows[k_id].append(row)
                d_nparts[k_id] += 1
            rows_to_del, d_fnparts, del_mask = list(), dict(), np.ones(shape=len(self.__particles), dtype=bool)
            for k_id in klass_ids:
                if d_nparts[k_id] < th:
                    rows_to_del += d_rows[k_id]
                else:
                    d_fnparts[k_id] = d_nparts[k_id]
            for row, k_id in enumerate(klass_col):
                if d_nparts[k_id] < th:
                    del_mask[row] = False
            # Purge STAR FILE
            hold_star.del_rows(rows_to_del)
            # Purge particles, masks and moments array
            self.__particles = self.__particles[del_mask]
            self.__masks = self.__masks[del_mask]
            if self.__momes:
                self.__momes = self.__momes[del_mask]
            self.__star = hold_star

        return d_fnparts

    # Purge particles with low cross-correlation against is AP reference
    # VERY IMPORTANT: method update_relion_classes() is always called internally
    # th: CC threshold, use method print_ccap_stat() to find its value
    # Return: a dictionary with the _rlnClassNumber as index and the amount particles in these purged classes
    def purge_low_ccap_particles(self, th):

        # Input parsing
        if self.__ap_ref_cc is None:
            error_msg = 'This method requires CCAP statistics computation, call compute_ccap_stat() first!'
            raise pexceptions.PySegInputError(expr='print_ccap_stat (ClassStar)', msg=error_msg)

        # Initialzation
        d_rows, d_nparts = dict(), dict()
        self.update_relion_classes()

        # Purge
        d_fnparts = None
        if th > 0:
            hold_star = copy.deepcopy(self.__star)
            klass_col = hold_star.get_column_data('_rlnClassNumber')
            for k_id in list(self.__ap_classes.keys()):
                d_rows[k_id], d_nparts[k_id] = list(), 0
            for row, k_id in enumerate(klass_col):
                d_rows[k_id].append(row)
                d_nparts[k_id] += 1
            rows_to_del, d_fnparts, del_mask = list(), dict(), np.ones(shape=len(self.__particles), dtype=bool)
            for k_id, klass in zip(list(self.__ap_classes.keys()), list(self.__ap_classes.values())):
                d_fnparts[k_id] = 0
                for j, row in enumerate(klass):
                    if self.__ap_ref_cc[k_id][j] < th:
                        rows_to_del += [row,]
                        del_mask[row] = False
                    else:
                        d_fnparts[k_id] += 1
            # Purge STAR FILE
            hold_star.del_rows(rows_to_del)
            # Purge particles, masks and moments array
            self.__particles = self.__particles[del_mask]
            self.__masks = self.__masks[del_mask]
            if self.__momes is not None:
                self.__momes = self.__momes[del_mask]
            self.__star = hold_star
            # Clean AP statistics
            self.__ap_classes, self.__ap_centers, self.__ap_ref_cc = None, None, None

        return d_fnparts

    # Store information related with the classification procedure
    # out_dir: output directory
    # out_stem: output file stem
    # purge_k: information related with classes whose amount of particles is smaller that this value is discarded
    #          (default 0)
    # mode: 'averages' (default) then the direct sum of every particle class is stored,
    #       'exemplars' then exemplar particles obtained by AP are stored,
    def save_class(self, out_dir, out_stem, purge_k=0, mode='averages', forceMRC=False):

        if mode == 'averages':
            hold_dir = out_dir + '/' + out_stem + '_averages'
            if not os.path.exists(hold_dir):
                os.makedirs(hold_dir)
            n_dim = len(self.__particles[0].shape)
            klass_col = self.__star.get_column_data('_rlnClassNumber')
            klass_ids = set(klass_col)
            d_parts, d_count = dict(), dict()
            for k_id in klass_ids:
                d_parts[k_id] = np.zeros(shape=self.__particles[0].shape, dtype=self.__particles[0].dtype)
                d_count[k_id] = 0
            for i, k_id in enumerate(klass_col):
                d_parts[k_id] += self.__particles[i]
                d_count[k_id] += 1
            for k_id in klass_ids:
                hold_avg, hold_nparts = d_parts[k_id], float(d_count[k_id])
                if hold_nparts > 0:
                    hold_avg /= hold_nparts
                if forceMRC:
                    disperse_io.save_numpy(hold_avg, hold_dir + '/class_k' + str(k_id) + '.mrc')
                else:
                    if n_dim == 2:
                        imwrite(hold_dir+'/class_k'+str(k_id)+'.png', lin_map(hold_avg,0,np.iinfo(np.uint16).max).astype(np.uint16))
                    else:
                        disperse_io.save_numpy(hold_avg, hold_dir+'/class_k'+str(k_id)+'.mrc')

        elif mode == 'exemplars':
            if not self.__star.has_column('_psAPCenter'):
                error_msg = 'No psAPCenter information, call function affinity_propagation() first!'
                raise pexceptions.PySegInputError(expr='save_star (ClassStar)', msg=error_msg)
            hold_dir = out_dir + '/' + out_stem + '_exemplars'
            if not os.path.exists(hold_dir):
                os.makedirs(hold_dir)
            for row in range(self.__star.get_nrows()):
                hold = self.__star.get_element('_psAPCenter', row)
                if hold >= 0:
                    if forceMRC:
                        disperse_io.save_numpy(self.__particles[row], hold_dir + '/class_k' + str(hold) + '.mrc')
                    else:
                        if len(self.__particles[row].shape) == 2:
                            imwrite(hold_dir+'/class_k'+str(hold)+'.png', lin_map(self.__particles[row],0,np.iinfo(np.uint16).max).astype(np.uint16))
                        else:
                            disperse_io.save_numpy(self.__particles[row], hold_dir+'/class_k'+str(hold)+'.mrc')

        else:
            error_msg = 'Mode ' + str(mode) + ' is not valid!'
            raise pexceptions.PySegInputError(expr='save_class (ClassStar)', msg=error_msg)

    # Store information related with the classification procedure when particles have already been saved in a
    # directory
    # imgs_dir: directory with the particle images
    # out_dir: output directory
    # out_stem: output file stem
    # mode: 'averages' (default) then the direct sum of every particle class is stored,
    #       'exemplars' then exemplar particles obtained by AP are stored,
    #       'classes' all particles are grouped by class
    def save_class_dir(self, imgs_dir, out_dir, out_stem, mode='averages'):

        # Input parsing
        if not os.path.exists(imgs_dir):
            error_msg = 'No images directory!'
            raise pexceptions.PySegInputError(expr='save_class_dir (ClassStar)', msg=error_msg)
        in_zero = imgs_dir + '/particle_0.npy'
        if not os.path.exists(in_zero):
            error_msg = 'No image zero in images directory!'
            raise pexceptions.PySegInputError(expr='save_class_dir (ClassStar)', msg=error_msg)
        img_zero = np.load(in_zero)

        if mode == 'averages':
            hold_dir = out_dir + '/' + out_stem + '_averages'
            if not os.path.exists(hold_dir):
                os.makedirs(hold_dir)
            klass_col = self.__star.get_column_data('_rlnClassNumber')
            klass_ids = set(klass_col)
            d_parts, d_count = dict(), dict()
            for k_id in klass_ids:
                d_parts[k_id] = np.zeros(shape=img_zero.shape, dtype=float)
                d_count[k_id] = 0
            for row, k_id in enumerate(klass_col):
                in_fname = imgs_dir + '/particle_' + str(row) + '.npy'
                hold_particle = np.load(in_fname)
                d_parts[k_id] += hold_particle
                d_count[k_id] += 1
            for k_id in klass_ids:
                hold_avg, hold_nparts = d_parts[k_id], float(d_count[k_id])
                if hold_nparts > 0:
                    hold_avg /= hold_nparts
                imwrite(hold_dir+'/class_k'+str(k_id)+'.png', lin_map(hold_avg,0,np.iinfo(np.uint16).max).astype(np.uint16))

        elif mode == 'exemplars':
            if not self.__star.has_column('_psAPCenter'):
                error_msg = 'No psAPCenter information, call function affinity_propagation() first!'
                raise pexceptions.PySegInputError(expr='save_class_dir (ClassStar)', msg=error_msg)
            hold_dir = out_dir + '/' + out_stem + '_exemplars'
            if not os.path.exists(hold_dir):
                os.makedirs(hold_dir)
            for row in range(self.__star.get_nrows()):
                hold = self.__star.get_element('_psAPCenter', row)
                if hold >= 0:
                    in_fname = imgs_dir + '/particle_' + str(row) + '.npy'
                    hold_particle = np.load(in_fname)
                    imwrite(hold_dir+'/class_k'+str(hold)+'.png', lin_map(hold_particle,0,np.iinfo(np.uint16).max).astype(np.uint16))

        elif mode == 'classes':
            klass_col = self.__star.get_column_data('_rlnClassNumber')
            klass_ids = set(klass_col)
            for k_id in klass_ids:
                hold_dir = out_dir + '/' + out_stem + '_class_k' + str(k_id)
                if not os.path.exists(hold_dir):
                    os.makedirs(hold_dir)
            for row, k_id in enumerate(klass_col):
                in_fname = imgs_dir + '/particle_' + str(row) + '.png'
                out_fname = out_dir + '/' + out_stem + '_class_k' + str(k_id) + '/particle_' + str(row) + '.png'
                hold_particle = imread(in_fname)
                imwrite(out_fname, lin_map(hold_particle,0,np.iinfo(np.uint16).max).astype(np.uint16))

        else:
            error_msg = 'Mode ' + str(mode) + ' is not valid!'
            raise pexceptions.PySegInputError(expr='save_class_dir (ClassStar)', msg=error_msg)


    def save_particles(self, out_dir, out_stem, masks=False, stack=False):
        """
        Store particles (and masks) in a folder when particles have already been saved in a directory
        :param out_dir: output directory
        :param out_stem: output file stem (default '')
        :param masks: if True the masks are also stored
        :param stack: if True particles are stored independently, otherwise they are stacked in a mrc file (avoided for 3D particles)
        :return:
        """

        # Input parsing
        if (self.__particles is None) and (len(self.__particles) > 0):
            return
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)


        # Storing loop
        if len(self.__particles[0].shape) > 2:
            for i, part in enumerate(self.__particles):
                hold_fname =out_dir + '/' + out_stem + '_particle_' + str(i)
                disperse_io.save_numpy(part, hold_fname+'.mrc')
                if masks:
                    disperse_io.save_numpy(self.__masks[i], hold_fname+'_mask.mrc')
        else:
            if stack:
                hold_fname =out_dir + '/' + out_stem + '_particles'
                part_x, part_y = self.__particles[0].shape
                if masks:
                    hold_stack = np.zeros(shape=(part_x, part_y, len(self.__particles)),
                                          dtype=self.__particles[0].dtype)
                    hold_stack_m = np.zeros(shape=(part_x, part_y, len(self.__particles)),
                                            dtype=self.__particles[0].dtype)
                    for i, part in enumerate(self.__particles):
                        hold_stack[:, :, i], hold_stack_m[:, :, i] = part, self.__masks[i]
                    disperse_io.save_numpy(hold_stack, hold_fname+'.mrc')
                    disperse_io.save_numpy(hold_stack_m, hold_fname+'_mask.mrc')
                else:
                    hold_stack = np.zeros(shape=(part_x, part_y, len(self.__particles)), dtype=self.__particles[0].dtype)
                    for i, part in enumerate(self.__particles):
                        hold_stack[:, :, i] = part
                    disperse_io.save_numpy(hold_stack, hold_fname+'.mrc')
            else:
                for i, part in enumerate(self.__particles):
                    hold_fname =out_dir + '/' + out_stem + '_particle_' + str(i)
                    imwrite(hold_fname+'.png', lin_map(part,0,np.iinfo(np.uint16).max).astype(np.uint16))
                    if masks:
                        imwrite(hold_fname+'_mask.png', lin_map(self.__masks[i],0,np.iinfo(np.uint16).max).astype(np.uint16))

    # fname: file name ended with .pkl
    def pickle(self, fname):
        pkl_f = open(fname, 'wb')
        try:
            pickle.dump(self, pkl_f)
        finally:
            pkl_f.close()

    # INTERNAL FUCTIONALITY

    # Build the Normalized (intensity in 3D) Cross-Correlation Matrix among Z radially averaged particles
    # for a subset of particles
    # sset_ids: particles indices of the subset
    # metric: metric used for particles affinity, valid: 'cc' (default) cross-correlation,
    #         'similarity' negative squared Euclidean distance, 'full_cc' slower than 'cc' but considers small
    #         misalignments
    # npr: number of process, None (defaults) means they are adjusted automatically to the number of cores
    # rest of parameters in load_particles
    # Returns: a cross-correlation matrix for input subset of particles
    def __build_ncc_z2d_sset(self, sset_ids, mask, metric='cc', low_sg=0, seg_dil_it=2, rln_norm=True, avg_norm=False,
                             rad_3D=False, direct_rec=False, ref_dir=None, npr=None):

        # Get subset star file object
        hold_star = self.__star.get_subset(sset_ids)
        hold_class_star = ClassStar(hold_star)

        # Load particles
        hold_class_star.load_particles(mask, low_sg=low_sg, rln_norm=rln_norm, avg_norm=avg_norm, rad_3D=rad_3D,
                                       npr=npr, ref_dir=ref_dir, direct_rec=direct_rec)
        hold_class_star.build_ncc_z2d(metric=metric, npr=npr)
        hold_cc = hold_class_star.__cc

        return hold_cc

    # Applies Affinity Propagation into a CC matrix and find
    # cc: input cross correlation matrix to cluster
    # rest: see affinity_propagation()
    # Returns: returns two lists, class labels and exemplars indices
    def __affinity_propagation_cc(self, cc, damping=0.5, preference=None, max_iter=200, convergence_iter=15,):

        # AP algorithm
        aff = AffinityPropagation(damping=damping,
                                  convergence_iter=convergence_iter,
                                  max_iter=max_iter,
                                  affinity='precomputed',
                                  preference=preference)
        aff.fit(cc)
        if aff.n_iter_ >= max_iter:
            print('WARNING: __affinity_propagation_cc (ClassStar): AP reached the maximum iterations (' + str(max_iter) + \
                ') without converging, its results might be meaningless!')
        return aff.labels_, aff.cluster_centers_indices_

    # Classify particle by maximum CC to some exemplars, then the final number equals the number of exemplars
    # ex_ids: list with exemplars id rows
    # mask: binary mask to be applied after rotations, its dimensions also defines subvolumes dimensions used,
    #       only 3D cubes with even dimension are accepted
    # low_sg: low pass Gaussian filter sigma in voxels, it does not apply if lower or equal than 0 (default)
    # rln_norm: normalize particles according Relion convention after loaded (default True)
    # avg_norm: re-normalize particles after their radial averaging (default False)
    # rad_3d: if False (default) the NCC are done in 2D, otherwise in 3D
    # npr: number of process, None (defaults) means they are adjusted automatically to the number of cores
    # direct_rec: if False (default) the particles are cropped from 'rlnMicrographName' tomogram, otherwise they are
    #             directly load from 'rlnImageName'
    # ref_dir: Alternative directory for looking for reference tomograms (default None)
    # imgs_dir: directory to store the flattern images generated (default None)
    # Returns: returns two lists, class labels and exemplars indices
    def __class_by_exemplars(self, ex_ids, mask, low_sg=0, seg_dil_it=2, rln_norm=True, avg_norm=False, rad_3D=False,
                             npr=None, direct_rec=False, ref_dir=None, imgs_dir=None):

        # Input parsing
        if not isinstance(mask, np.ndarray):
            error_msg = 'Input mask must a ndarray!'
            raise pexceptions.PySegInputError(expr='load_particles (ClassStar)', msg=error_msg)
        svol_sp = np.asarray(mask.shape, dtype=int)
        if (len(svol_sp) != 3) or (svol_sp[0] != svol_sp[1]) or (svol_sp[1] != svol_sp[2]) or (svol_sp[0]%2 != 0):
            error_msg = 'Input mask must be a 3D cube with even dimension!'
            raise pexceptions.PySegInputError(expr='load_particles (ClassStar)', msg=error_msg)

        # Computing exemplars templates
        temp_star = self.__star.get_subset(ex_ids)
        temp_class_star = ClassStar(temp_star)
        temp_class_star.load_particles(mask, low_sg=low_sg, seg_dil_it=seg_dil_it, rln_norm=rln_norm,
                                       avg_norm=avg_norm, rad_3D=rad_3D,
                                       npr=npr, ref_dir=ref_dir, direct_rec=direct_rec)
        ex_temps = temp_class_star.__particles

        # Multiprocessing
        if npr is None:
            npr = mp.cpu_count()
        processes = list()
        # Create the list on indices to split
        npart, ntemps = self.__star.get_nrows(), len(ex_temps)
        sym_ids = np.arange(npart)
        spl_ids = np.array_split(list(range(len(sym_ids))), npr)
        shared_mat_cce = mp.Array('f', int(npart*ntemps))

        # Particles loop (Parallel)
        if npr <= 1:
            pr_exemplars_cc(-1, spl_ids[0], ex_temps, mask, self.__star, low_sg, rln_norm, avg_norm, rad_3D,
                            ref_dir, direct_rec, imgs_dir,
                            shared_mat_cce)
        else:
            for pr_id in range(npr):
                pr = mp.Process(target=pr_exemplars_cc, args=(pr_id, spl_ids, ex_temps, mask, self.__star, low_sg,
                                                              rln_norm, avg_norm, rad_3D, ref_dir, direct_rec, imgs_dir,
                                                              shared_mat_cce))
                pr.start()
                processes.append(pr)
            pr_results = list()
            for pr in processes:
                pr.join()
                pr_results.append(pr.exitcode)
            for pr_id in range(len(processes)):
                if pr_id != pr_results[pr_id]:
                    error_msg = 'Process ' + str(pr_id) + ' exited unexpectedly!'
                    raise pexceptions.PySegInputError(expr='__class_by_exemplars (ClassStar)', msg=error_msg)
            gc.collect()

        # Fill diagonal with the maximum value for normalized cross-correlation
        self.__cc_rows = np.frombuffer(shared_mat_cce.get_obj(), dtype=np.float32).reshape(npart, ntemps)


    # Load particles a subset of particles from the STAR file information and computes their mask and radial averages
    # row_ids: list with the rows ids
    # mask: binary mask to be applied after rotations, its dimensions also defines subvolumes dimensions used,
    #       only 3D cubes with even dimension are accepted
    # low_sg: low pass Gaussian filter sigma in voxels, it does not apply if lower or equal than 0 (default)
    # rln_norm: normalize particles according Relion convention after loaded (default True)
    # avg_norm: re-normalize particles after their radial averaging (default False)
    # rad_3d: if False (default) the NCC are done in 2D, otherwise in 3D
    # npr: number of process, None (defaults) means they are adjusted automatically to the number of cores
    # debug_dir: if not None (default) intermediate information will be stored in this directory, use only for debugging
    #            purposes
    # ref_dir: Alternative directory for looking for reference tomograms (default None)
    # direct_rec: if False (default) the particles are cropped from 'rlnMicrographName' tomogram, otherwise they are
    #             directly load from 'rlnImageName'
    # Returns: the 2D images for the input row_ids and the correspondent masks array
    def __load_particles_rows(self, row_ids, mask, low_sg=0, seg_dil_it=2, rln_norm=True, avg_norm=False, rad_3D=False,
                              npr=None, debug_dir=None, ref_dir=None, direct_rec=False):

        # Input parsing
        if not isinstance(mask, np.ndarray):
            error_msg = 'Input mask must a ndarray!'
            raise pexceptions.PySegInputError(expr='load_particles (ClassStar)', msg=error_msg)
        svol_sp = np.asarray(mask.shape, dtype=int)
        if (len(svol_sp) != 3) or (svol_sp[0] != svol_sp[1]) or (svol_sp[1] != svol_sp[2]) or (svol_sp[0]%2 != 0):
            error_msg = 'Input mask must be a 3D cube with even dimension!'
            raise pexceptions.PySegInputError(expr='load_particles (ClassStar)', msg=error_msg)
        # Store intermediate information
        averager = RadialAvg3D(svol_sp, axis='z')
        if debug_dir is not None:
            disperse_io.save_numpy(averager.get_kernels(), debug_dir + '/avg_kernels.mrc')

        # Initialization
        npart = len(row_ids)
        ex_temps, ex_mask = np.zeros(shape=npart, dtype=object), np.zeros(shape=npart, dtype=object)

        # Multiprocessing
        if npr is None:
            npr = mp.cpu_count()
        processes = list()
        # Create the list on indices to split
        part_h, part_r = averager.get_output_dim()
        part_sz = int(part_h * part_r)
        particles_sh, masks_sh = mp.Array('f', int(part_h*part_r*npart)), mp.Array('f', int(part_h*part_r*npart))

        # Loading particles loop (Parallel)
        if npr <= 1:
            pr_load_parts(-1, row_ids, mask, self.__star, float(low_sg), rln_norm,  avg_norm, rad_3D,
                          debug_dir, ref_dir, direct_rec,
                          particles_sh, masks_sh)
        else:
            spl_ids = np.array_split(np.arange(row_ids, dtype=int), npr)
            for pr_id in range(npr):
                pr = mp.Process(target=pr_load_parts, args=(pr_id, spl_ids[pr_id], mask, self.__star, float(low_sg),
                                                            rln_norm, avg_norm, rad_3D, debug_dir, ref_dir, direct_rec,
                                                            particles_sh, masks_sh))
                pr.start()
                processes.append(pr)
            pr_results = list()
            for pr in processes:
                pr.join()
                pr_results.append(pr.exitcode)
            for pr_id in range(len(processes)):
                if pr_id != pr_results[pr_id]:
                    error_msg = 'Process ' + str(pr_id) + ' exited unexpectedly!'
                    raise pexceptions.PySegInputError(expr='load_particles (ClassStar)', msg=error_msg)
            gc.collect()

        hold_particle, hold_mask = np.zeros(shape=(part_h, part_r), dtype=np.float32), \
                                       np.zeros(shape=(part_h, part_r), dtype=np.float32)
        for row in row_ids:
            sh_id = row * part_sz
            for j in range(part_r):
                sh_id_l, sh_id_h = sh_id+j*part_h, sh_id+(j+1)*part_h
                hold_particle[:, j], hold_mask[:, j] = particles_sh[sh_id_l:sh_id_h], masks_sh[sh_id_l:sh_id_h]

        return hold_particle, hold_mask
