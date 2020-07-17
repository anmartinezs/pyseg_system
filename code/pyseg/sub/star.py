"""
Set of classes for dealing with a STAR files (Relion's format)

# Author: Antonio Martinez-Sanchez (Max Planck Institute for Biochemistry)
# Date: 1.06.16
"""

__author__ = 'Antonio Martinez-Sanchez'

import gc
import sys
import csv
import copy
import pyto
import errno
from pyseg.globals import *
from pyseg import disperse_io
import numpy as np
import scipy as sp
import multiprocessing as mp
from pyseg.pexceptions import *
import itertools as it
from variables import RadialAvg3D

###########################################################################################
# Global functionality
###########################################################################################

def relion_norm(tomo, mask=None, inv=True):
    """
    Relion tomogram normalization
    :param tomo: input tomogram
    :param mask: if None (default) the whole tomogram is used for computing the statistics otherwise just the masked region
    :param inv: if True the values are inverted (default)
    :return:
    """

    # Input parsing
    if mask is None:
        mask = np.ones(shape=tomo.shape, dtype=np.bool)

    # Inversion
    if inv:
        hold_tomo = -1. * tomo
    else:
        hold_tomo = tomo

    # Statistics
    stat_tomo = hold_tomo[mask>0]
    mn, st = stat_tomo.mean(), stat_tomo.std()

    # Histogram equalization
    tomo_out = np.zeros(shape=tomo.shape, dtype=np.float32)
    if st > 0:
        tomo_out = (hold_tomo-mn) / st
    else:
        print 'WARNING (relion_norm): standard deviation=' + str(st)

    return tomo_out

###########################################################################################
# Parallel processes
###########################################################################################

# Bin subvolumes in a particles STAR file
# pr_id: process ID
# beg_ids: beginning STAR particle row indices
# end_ids: ending STAR particle row indices
# bin: binning factor
# sv_shape: original subvolume shape
# res: original resolution vx/nm
# cutoff: cutoff in nm for low pass filtering, if None not applied
# dstar: dictionary with the STAR file data
# out_svol_dir: output directory for the binned files
# tmp_csv_dir: directory for temporary output CSV files
def pr_bin_star_svols(pr_id, beg_ids, end_ids, bin, res, sv_shape, cutoff, dstar, out_svol_dir, tmp_csv_dir):

    # Initialization
    zoom_f, nyquist = 1. / bin, 2. * res
    tmp_csv = tmp_csv_dir + '/process_' + str(pr_id) + '.csv'
    sv_shape_f = np.asarray(sv_shape, dtype=np.float32).min()
    # Low pass filter to estimation CTF correction
    if cutoff is None:
        cutoff = nyquist * float(bin)
    rad = (nyquist/cutoff) * np.min(sv_shape_f) * .5
    lpf = low_pass_fourier_3d(sv_shape, res, cutoff, rad*.1)
    clpf = lpf
    lpf = sp.fftpack.fftshift(lpf)
    sph_mask = sphere_mask(sv_shape, .4*sv_shape_f) > 0

    # Particles loop
    with open(tmp_csv, 'w') as tfile:
        writer = csv.DictWriter(tfile, fieldnames=dstar.keys())
        writer.writeheader()
        for row_id in range(beg_ids, end_ids+1):
            svol = disperse_io.load_tomo(dstar['_rlnImageName'][row_id], mmap=False)
            # Low pass filtering
            svol_fft = sp.fftpack.fftn(svol)
            svol = np.real(sp.fftpack.ifftn(svol_fft * lpf))
            hold_svol = svol[sph_mask]
            svol -= hold_svol.mean()
            svol/hold_svol.std()
            # Rescaling
            svol_bin =  sp.ndimage.interpolation.zoom(svol, zoom_f,
                                                      order=3, mode='constant', cval=0.0, prefilter=True)
            out_svol = out_svol_dir + '/particle_bin' + str(int(bin)) + '_id_' + str(row_id) + '.mrc'
            disperse_io.save_numpy(svol_bin, out_svol)
            dstar['_rlnImageName'][row_id] = out_svol
            try:
                cvol = disperse_io.load_tomo(dstar['_rlnCtfImage'][row_id], mmap=False)
                cvol *= clpf
                lx_crop, ly_crop, lz_crop = int(math.floor(.5*svol.shape[0])), int(math.floor(.5*svol.shape[1])), \
                                            int(math.floor(.5*svol.shape[2]))
                lx_crop -= int(math.floor(0.5*svol_bin.shape[0]))
                ly_crop -= int(math.floor(0.5*svol_bin.shape[1]))
                lz_crop -= int(math.floor(0.5*svol_bin.shape[2]))
                cvol_bin = cvol[lx_crop:-lx_crop, ly_crop:-ly_crop, lz_crop:-ly_crop]
                out_cvol = out_svol_dir + '/ctf_bin' + str(int(bin)) + '_id_' + str(row_id) + '.mrc'
                # disperse_io.save_numpy(lpf, out_cvol)
                disperse_io.save_numpy(cvol_bin, out_cvol)
                dstar['_rlnCtfImage'][row_id] = out_cvol
            except ValueError:
                pass
            row = dict.fromkeys(dstar)
            for key in dstar.keys():
                row[key] = dstar[key][row_id]
            writer.writerow(row)

            # print str(row_id)

    if pr_id < 0:
        return
    else:
        sys.exit(pr_id)

# Process for template matching on STAR file particles
# pr_id: process ID
# temp: template subvolume (same size as particles)
# mask: smooth borders mask (same size as particles)
# row_ids: STAR files row IDs to process
# dstar: STAR file dictionary with its data
# csvol_paths: list of paths ot CTF subvolumes
# angs_arr: array with the angular sampling
# max_shift_v: maximum shifting in vx from suvolume center to consider
# shared_ncc: shared array to store the computed NCC scores
def pr_star_tm(pr_id, row_ids, temp, mask, dstar, max_shift_v, angs_arr, shared_ncc):

    # Initialization
    n_mask = float(mask.sum())
    cte = 1. / n_mask
    svol_sp = np.asarray(temp.shape, dtype=np.int)
    svol_sp2 = int(.5 * svol_sp[0])
    svol_cent = np.asarray((svol_sp2, svol_sp2, svol_sp2), dtype=np.float32)
    psvol_paths = dstar['_rlnImageName']
    try:
        csvol_paths = dstar['_rlnCtfImage']
    except ValueError:
        csvol_paths = None
    try:
        offs_x = dstar['_rlnOriginX']
    except ValueError:
        offs_x = np.zeros(shape=len(psvol_paths), dtype=np.float32)
    try:
        offs_y = dstar['_rlnOriginY']
    except ValueError:
        offs_y = np.zeros(shape=len(psvol_paths), dtype=np.float32)
    try:
        offs_z = dstar['_rlnOriginZ']
    except ValueError:
        offs_z = np.zeros(shape=len(psvol_paths), dtype=np.float32)
    try:
        offs_rot = dstar['_rlnAngleRot']
    except ValueError:
        offs_rot = np.zeros(shape=len(psvol_paths), dtype=np.float32)
    try:
        offs_tilt = dstar['_rlnAngleTilt']
    except ValueError:
        offs_tilt = np.zeros(shape=len(psvol_paths), dtype=np.float32)
    try:
        offs_psi = dstar['_rlnAnglePsi']
    except ValueError:
        offs_psi = np.zeros(shape=len(psvol_paths), dtype=np.float32)

    # Template masking
    mtemp = temp * mask
    mask_id = mask > 0
    mask_id_sum = mask_id.sum()

    # Shifting mask
    mask_shift = sphere_mask(mask.shape, max_shift_v).astype(np.bool)

    # Particles loop
    count = 0
    for row_id in row_ids:

        # Loading subvolumes
        psvol = disperse_io.load_tomo(psvol_paths[row_id], mmap=False)
        off_shift = np.asarray((offs_y[row_id], offs_x[row_id], offs_z[row_id]))
        off_ang = np.asarray((offs_rot[row_id], offs_tilt[row_id], offs_psi[row_id]))

        # Rotate particle to reference system
        psvol = tomo_shift(psvol, off_shift)
        r3d_a = pyto.geometry.Rigid3D()
        r3d_a.q = r3d_a.make_r_euler(angles=np.radians(off_ang), mode='zyz_in_active')
        rpsvol = r3d_a.transformArray(psvol, origin=svol_cent, order=3, prefilter=True, mode='reflect')

        # Normalize particle density within the mask
        rpsvol_fg = rpsvol[mask_id]
        rpsvol_mn, rpsvol_std = rpsvol_fg.mean(), rpsvol_fg.std()
        nrpsvol = rpsvol - rpsvol_mn
        if rpsvol_std > 0:
            nrpsvol /= rpsvol_std

        # Rotate back the particles
        r3d_p = pyto.geometry.Rigid3D()
        r3d_p.q = r3d_p.make_r_euler(angles=np.radians(off_ang), mode='zyz_in_passive')
        nrpsvol = r3d_p.transformArray(nrpsvol, origin=svol_cent, order=3, prefilter=True, mode='reflect')

        # Apply CTF to reference
        if csvol_paths is not None:

            # Rotate template to particles system
            rmtemp = r3d_p.transformArray(mtemp, origin=svol_cent, order=3, prefilter=True, mode='reflect')

            # CTF
            csvol = disperse_io.load_tomo(csvol_paths[row_id], mmap=False)
            crmtemp = np.real(sp.fftpack.ifftn(sp.fftpack.fftn(rmtemp) * sp.fftpack.fftshift(csvol)))

            # Rotate back to template system
            ctemp = r3d_a.transformArray(crmtemp, origin=svol_cent, order=3, prefilter=True, mode='reflect')

        else:
            ctemp = mtemp

        # Mask corrected normalization
        ctemp_fg = ctemp[mask_id]
        ctemp_mn, ctemp_std = ctemp_fg.mean(), ctemp_fg.std()
        nctemp = ctemp - ctemp_mn
        if ctemp_std > 0:
            nctemp /= ctemp_std
        # nctemp *= mask
        print sp.__version__
        if (int(sp.__version__.split('.')[0]) < 1) and (int(sp.__version__.split('.')[1]) < 19):
            nctemp_conj = np.real(sp.fftpack.ifftn(np.conjugate(sp.fftpack.fftn(nctemp))))

        # if row_id == 10:
        #     print 'Jol'
        # else:
        #     continue

        # Rotating loop
        max_ncc = 0
        # max_rot, max_tilt, max_psi = off_ang[0], off_ang[1], off_ang[2]
        for i_ang, ang_val in enumerate(angs_arr):
            # Particles search rotation
            ang = off_ang + ang_val
            r3d = pyto.geometry.Rigid3D()
            r3d.q = r3d.make_r_euler(angles=np.radians(ang), mode='zyz_in_active')
            rnrpsvol = r3d.transformArray(nrpsvol, origin=svol_cent, order=3, prefilter=True, mode='reflect') # * mask

            # NCC
            # psvol_conv = np.real(sp.fftpack.ifftn(sp.fftpack.fftn(rnrpsvol) * nctemp_fft)) * mask_shift * mask
            # psvol_conv = sp.signal.fftconvolve(rnrpsvol, nctemp, mode='same') # * mask_shift * mask
            if (int(sp.__version__.split('.')[0]) < 1) and (int(sp.__version__.split('.')[1]) < 19):
                # psvol_conv = sp.signal.correlate(rnrpsvol*mask, nctemp*mask, mode='same') * mask_shift
                psvol_conv = sp.signal.fftconvolve(rnrpsvol*mask, nctemp_conj*mask, mode='same') * mask_shift
            else:
                psvol_conv = sp.signal.correlate(rnrpsvol*mask, nctemp*mask, mode='same', method='fft') * mask_shift # * mask
            # oi, oj, ok = np.unravel_index(np.argmax(psvol_conv), psvol_conv.shape)
            # ncc = psvol_conv[oi, oj, ok] / mask_id_sum
            ncc = psvol_conv.max() / mask_id_sum
            # if i_ang == 0:
                # disperse_io.save_numpy(rnrpsvol, '/fs/pool/pool-lucic2/antonio/ribo_johannes/lum_ext/stm/test3_ribo/svols_3000_low_bin2_lp6/particle_bin2_id_' + str(row_id) + '_tvol.mrc')
                # disperse_io.save_numpy(nctemp, '/fs/pool/pool-lucic2/antonio/ribo_johannes/lum_ext/stm/test3_ribo/svols_3000_low_bin2_lp6/particle_bin2_id_' + str(row_id) + '_rvol.mrc')
                # disperse_io.save_numpy(psvol_conv, '/fs/pool/pool-lucic2/antonio/ribo_johannes/lum_ext/stm/test3_ribo/svols_3000_low_bin2_lp6/particle_bin2_id_' + str(row_id) + '_cvol.mrc')

            # Updating maximum
            if ncc > max_ncc:
                max_ncc = ncc
                # FOR DEBUGGING
                # max_rot, max_tilt, max_psi = ang[0], ang[1], ang[2]
                # max_ox, max_oy, max_oz = oi-svol_cent[0], oj-svol_cent[1], ok-svol_cent[2]
                # rnrpsvol = tomo_shift(rnrpsvol, (max_ox, max_oy, max_oz))
                # rnrpsvol_sp = tomo_shift(np.copy(rnrpsvol), (max_oy, max_ox, max_oz)) * mask
                # st_part, st_part_sp, st_temp, st_conv = rnrpsvol, rnrpsvol_sp, nctemp, psvol_conv

            # print 'Ang: ' + str(ang_val)

        # Set the maximum NCC in the shared array
        shared_ncc[row_id] = max_ncc

        # FOR DEBUGGING
        # parts_dir = os.path.split(psvol_paths[row_id])[0]
        # disperse_io.save_numpy(st_part, parts_dir + '/tm_particle_' + str(row_id) + '.mrc')
        # disperse_io.save_numpy(st_part_sp, parts_dir + '/tm_particle_sp_' + str(row_id) + '.mrc')
        # disperse_io.save_numpy(st_temp, parts_dir + '/tm_model_' + str(row_id) + '.mrc')
        # disperse_io.save_numpy(st_conv, parts_dir + '/tm_conv_' + str(row_id) + '.mrc')

        print 'Process: ' + str(os.path.split(psvol_paths[row_id])[1]) + ', Particle: ' + str(count) + ' of ' + str(len(row_ids)) + ', ncc: ' + str(max_ncc)
        count += 1

    if pr_id < 0:
        return
    else:
        sys.exit(pr_id)

# Process for template matching by Z-axis averaging on STAR file particles
# pr_id: process ID
# temp: template subvolume (same size as particles)
# mask: smooth borders mask (same size as particles)
# row_ids: STAR files row IDs to process
# dstar: STAR file dictionary with its data
# csvol_paths: list of paths ot CTF subvolumes
# shifts_arr: array with space shiftings
# shared_ncc: shared array to store the computed NCC scores
def pr_star_tm_za(pr_id, row_ids, temp, mask, dstar, shifts_arr, shared_ncc):

    # Initialization
    averager = RadialAvg3D(mask.shape, axis='z')
    svol_sp = np.asarray(temp.shape, dtype=np.int)
    svol_sp2 = int(.5 * svol_sp[0])
    svol_cent = np.asarray((svol_sp2, svol_sp2, svol_sp2), dtype=np.float32)
    psvol_paths = dstar['_rlnImageName']
    try:
        csvol_paths = dstar['_rlnCtfImage']
    except ValueError:
        csvol_paths = None
    try:
        offs_x = dstar['_rlnOriginX']
    except ValueError:
        offs_x = np.zeros(shape=len(psvol_paths), dtype=np.float32)
    try:
        offs_y = dstar['_rlnOriginY']
    except ValueError:
        offs_y = np.zeros(shape=len(psvol_paths), dtype=np.float32)
    try:
        offs_z = dstar['_rlnOriginZ']
    except ValueError:
        offs_z = np.zeros(shape=len(psvol_paths), dtype=np.float32)
    try:
        offs_rot = dstar['_rlnAngleRot']
    except ValueError:
        offs_rot = np.zeros(shape=len(psvol_paths), dtype=np.float32)
    try:
        offs_tilt = dstar['_rlnAngleTilt']
    except ValueError:
        offs_tilt = np.zeros(shape=len(psvol_paths), dtype=np.float32)
    try:
        offs_psi = dstar['_rlnAnglePsi']
    except ValueError:
        offs_psi = np.zeros(shape=len(psvol_paths), dtype=np.float32)

    # Preprocessing the mask
    mask_id = mask > 0
    mask_avg = averager.avg_vol(mask) > 0.5
    n_mask_avg = float(mask_avg.sum())
    cte = 0.
    if n_mask_avg > 0:
        cte = 1. / n_mask_avg

    # Particles loop
    count = 0
    for row_id in row_ids:

        # Loading subvolumes
        psvol = disperse_io.load_tomo(psvol_paths[row_id], mmap=False)
        off_shift = np.asarray((offs_y[row_id], offs_x[row_id], offs_z[row_id]))
        off_ang = np.asarray((offs_rot[row_id], offs_tilt[row_id], offs_psi[row_id]))

        # Rotate particle to reference system
        psvol = tomo_shift(psvol, off_shift)
        r3d_a = pyto.geometry.Rigid3D()
        r3d_a.q = r3d_a.make_r_euler(angles=np.radians(off_ang), mode='zyz_in_active')
        rpsvol = r3d_a.transformArray(psvol, origin=svol_cent, order=3, prefilter=True, mode='reflect')

        # Normalize particle density with the mask
        rpsvol_fg = rpsvol[mask_id]
        rpsvol_mn, rpsvol_std = rpsvol_fg.mean(), rpsvol_fg.std()
        nrpsvol = rpsvol - rpsvol_mn
        if rpsvol_std > 0:
            nrpsvol /= rpsvol_std

        # Apply CTF to reference
        if csvol_paths is not None:

            # Rotate template to particles system
            r3d_p = pyto.geometry.Rigid3D()
            r3d_p.q = r3d_p.make_r_euler(angles=np.radians(off_ang), mode='zyz_in_passive')
            rmtemp = r3d_p.transformArray(temp, origin=svol_cent, order=3, prefilter=True, mode='reflect')

            # CTF
            csvol = disperse_io.load_tomo(csvol_paths[row_id], mmap=False)
            crmtemp = np.real(sp.fftpack.ifftn(sp.fftpack.fftn(rmtemp) * sp.fftpack.fftshift(csvol)))

            # Rotate back to template system
            ctemp = r3d_a.transformArray(crmtemp, origin=svol_cent, order=3, prefilter=True, mode='reflect')

        else:
            ctemp = temp

        # Mask corrected normalization
        ctemp_fg = ctemp[mask_id]
        ctemp_mn, ctemp_std = ctemp_fg.mean(), ctemp_fg.std()
        nctemp = ctemp - ctemp_mn
        if ctemp_std > 0:
            nctemp /= ctemp_std
        nctemp *= mask

        # Mask averaging
        temp_avg = averager.avg_vol(nctemp)
        t_masked = temp_avg[mask_avg]
        m_std = t_masked.std()
        if m_std > 0:
            particle_avg = (temp_avg-t_masked.mean()) / m_std
        temp_avg[~mask_avg] = 0

        # Shifting loop
        max_ncc = 0
        for shift_val in shifts_arr:

            # Shifting
            snrpsvol = tomo_shift(nrpsvol, shift_val)

            # Z-axis averaging
            particle_avg = averager.avg_vol(snrpsvol)
            p_masked = particle_avg[mask_avg]
            m_std = p_masked.std()
            if m_std > 0:
                particle_avg = (particle_avg-p_masked.mean()) / m_std
            particle_avg[~mask_avg] = 0

            # 2D Cross-Correlation
            ncc = cte * np.sum(particle_avg[mask_avg]*temp_avg[mask_avg])

            # Updating maximum
            if ncc > max_ncc:
                max_ncc = ncc
                # FOR DEBUGGING
                # st_part, st_temp = particle_avg, temp_avg

            # print 'Ang: ' + str(ang_val)

        # Set the maximum NCC in the shared array
        shared_ncc[row_id] = max_ncc

        # FOR DEBUGGING
        # parts_dir = os.path.split(psvol_paths[row_id])[0]
        # sp.misc.imsave(parts_dir + '/za_particle_' + str(row_id) + '.png', st_part)
        # sp.misc.imsave(parts_dir + '/za_temp_' + str(row_id) + '.png', st_temp)

        # print 'Process: ' + str(os.path.split(psvol_paths[row_id])[1]) + ', Particle: ' + str(count) + ' of ' + str(len(row_ids)) + ', ncc: ' + str(max_ncc)
        count += 1

    if pr_id < 0:
        return
    else:
        sys.exit(pr_id)

###########################################################################################
# Class for converting data types for columns used by Relion
###########################################################################################

class RelionCols(object):

    def __init__(self):
        self.__cols = (# RELION
                       '_rlnMicrographName',
                       '_rlnCoordinateX',
                       '_rlnCoordinateY',
                       '_rlnCoordinateZ',
                       '_rlnImageName',
                       '_rlnCtfImage',
                       '_rlnGroupNumber',
                       '_rlnAngleRotPrior',
                       '_rlnAngleTiltPrior',
                       '_rlnAnglePsiPrior',
                       '_rlnAngleRot',
                       '_rlnAngleTilt',
                       '_rlnAnglePsi',
                       '_rlnOriginX',
                       '_rlnOriginY',
                       '_rlnClassNumber',
                       '_rlnNormCorrection',
                       '_rlnOriginZ',
                       '_rlnLogLikeliContribution',
                       '_rlnMaxValueProbDistribution',
                       '_rlnNrOfSignificantSamples',
                       '_rlnRandomSubset',
                       # PySeg: Graph analysis
                       '_psGhMCFPickle',
                       # PySeg: Segmentation
                       '_psSegImage',
                       '_psSegLabel',
                       '_psSegScale',
                       '_psSegRot',
                       '_psSegTilt',
                       '_psSegPsi',
                       '_psSegOffX',
                       '_psSegOffY',
                       '_psSegOffZ',
                       '_psCCScores',
                       # PySeg: Affinity Propagation
                       '_psAPClass',
                       '_psAPCenter',
                       # Microtubules:
                       '_mtCenterLine',
                       '_mtMtubesCsv'
                       )
        self.__dtypes = (str,
                         float,
                         float,
                         float,
                         str,
                         str,
                         int,
                         float,
                         float,
                         float,
                         float,
                         float,
                         float,
                         float,
                         float,
                         int,
                         float,
                         float,
                         float,
                         float,
                         int,
                         int,
                         # PySeg: Graph analysis
                         str,
                         # PySeg: Segmentation
                         str,
                         int,
                         float,
                         float,
                         float,
                         float,
                         float,
                         float,
                         float,
                         float,
                         # PySeg: Affinity Propagation
                         int,
                         int,
                         # Microtubules
                         str,
                         str
                         )
        assert len(self.__cols) == len(self.__dtypes)

    #### External functionality area

    # Returns data type used for this column in Relion, None if this is not a valid column key
    def get_dtype(self, key):
        if self.is_valid(key):
            idx = self.__cols.index(key)
            return self.__dtypes[idx]
        else:
            return None

    # Checks if a column name is valid for Relion
    def is_valid(self, key):
        try:
            self.__cols.index(key)
        except ValueError:
            return False
        return True

###########################################################################################
# Class for representing a Star file of a set of subvolumes usable by Relion
###########################################################################################

class Star(object):

    def __init__(self):
        self.__header_1 = ['\n', 'data_\n', '\n', 'loop_\n']
        self.__data = {}
        self.__dtypes = list()
        self.__rows = 0
        self.__cols = list()
        self.__checker = RelionCols()
        self.__root_dir, self.__fname = None, None

    #### Get/Set Area

    def set_root_dir(self, root_dir):
        self.__root_dir = str(root_dir)

    def get_root_dir(self):
        return self.__root_dir

    # Return the number of columns
    def get_ncols(self):
        return len(self.__cols)

    # Return the number of rows
    def get_nrows(self):
        return self.__rows

    # Returns column data in a list (if exists)
    # key: column key
    def get_column_data(self, key):
        if self.is_column(key):
            return self.__data[key]
        else:
            return None

    # Returns the existing differnt elements in a columh
    # key: column key
    def get_column_data_set(self, key):
        hold = self.get_column_data(key)
        if hold is None:
            return None
        else:
            return set(hold)

    # Returns a list with the keys of all columns
    def get_column_keys(self):
        return copy.copy(self.__cols)

    # Returns column type, None if the column is not found
    # key: column key
    def get_column_type(self, key):
        try:
            idx = self.__cols.index(key)
        except ValueError:
            return None
        return self.__dtypes[idx]

    # Returns the row and column specified element, if the key is not present a KeyError is returned
    # key: column key
    # row: row index
    def get_element(self, key, row):
        return self.__data[key][row]

    # Get the elements specified in a list of key columns and a row
    # keys: list column keys
    # row: row index
    # Returns: a list with with elements in the same order as keys
    def get_elements(self, keys, row):
        values = list()
        for key in keys:
            values.append(self.get_element(key, row))
        return values

    # Set the value of a specific column and row
    # key: column key
    # row: row index
    # val: value to be set
    def set_element(self, key, row, val):
        dtype = self.__checker.get_dtype(key)
        self.__data[key][row] = dtype(val)

    # Return geometrical information of specific particle: coordinates and rotation (optional) as 3-tuples
    # row: coordinate row
    # orig: take into account origin shifting information (default False)
    # rots: if True (default False) the rotation angles are also returned in another list
    def get_particle_coords(self, row, orig=False, rots=False):
        try:
            x = self.get_element('_rlnCoordinateX', row)
        except KeyError:
            x = 0
        try:
            y = self.get_element('_rlnCoordinateY', row)
        except KeyError:
            y = 0
        try:
            z = self.get_element('_rlnCoordinateZ', row)
        except KeyError:
            z = 0
        if orig:
            try:
                o_x = self.get_element('_rlnOriginX', row)
            except KeyError:
                o_x = 0
            try:
                o_y = self.get_element('_rlnOriginY', row)
            except KeyError:
                o_y = 0
            try:
                o_z = self.get_element('_rlnOriginZ', row)
            except KeyError:
                o_z = 0
            x -= o_x
            y -= o_y
            z -= o_z
            # x += o_x
            # y += o_y
            # z += o_z
        if rots:
            try:
                rho = self.get_element('_rlnAngleRot', row)
            except KeyError:
                rho = 0
            try:
                tilt = self.get_element('_rlnAngleTilt', row)
            except KeyError:
                tilt = 0
            try:
                psi = self.get_element('_rlnAnglePsi', row)
            except KeyError:
                psi = 0
            return (x, y, z), (rho, tilt, psi)
        else:
            return x, y, z

    # Return particles geometrical information: coordinates and rotation (optional)
    # orig: take into account origin shifting information (default False)
    # rots: if True (default False) the rotation angles are also returned in another list
    def get_particles_coords(self, orig=False, rots=False):
        coords, angs = list(), list()
        for row in range(self.get_nrows()):
            x, y, z = self.get_element('_rlnCoordinateX', row), self.get_element('_rlnCoordinateY', row), \
                      self.get_element('_rlnCoordinateZ', row)
            if orig:
                o_x, o_y, o_z = self.get_element('_rlnOriginX', row), self.get_element('_rlnOriginY', row), \
                                self.get_element('_rlnOriginZ', row)
                x -= o_x
                y -= o_y
                z -= o_z
            coords.append(np.asarray((x, y, z), dtype=np.float))
            if rots:
                rho, tilt, psi = self.get_element('_rlnAngleRot', row), self.get_element('_rlnAngleTilt', row), \
                                 self.get_element('_rlnAnglePsi', row)
                angs.append(np.asarray((rho, tilt, psi), dtype=np.float))
        if rots:
            return coords, angs
        else:
            return coords

    # Set the data of a column in one call
    # If the column does not exist of data does not fit the number of rows it does nothing
    # key: column key
    # data: column data
    def set_column_data(self, key, dat):
        if (self.is_column(key)) and (len(dat) == self.get_nrows()):
            if isinstance(dat, np.ndarray):
                self.__data[key] = dat.tolist()
            else:
                self.__data[key] = dat[:]

    # Find the index of the first element with a value in a column
    # key: column key
    # val: row value
    # Returns: if exists the element row index is returns, otherwise a ValueError exception is thrown
    def find_element(self, key, val):
        return self.__data[key].index(val)

    # Check if a column exists
    def has_column(self, key):
        return key in self.__cols

    #### External functionality

    # Scale particles coordinates and origins
    # sf: scale factor
    def scale_coords(self, sf):
        fsf = float(sf)
        hold = np.asarray(self.get_column_data('_rlnCoordinateX'), dtype=np.float) * fsf
        self.set_column_data('_rlnCoordinateX', hold)
        hold = np.asarray(self.get_column_data('_rlnCoordinateY'), dtype=np.float) * fsf
        self.set_column_data('_rlnCoordinateY', hold)
        hold = np.asarray(self.get_column_data('_rlnCoordinateZ'), dtype=np.float) * fsf
        self.set_column_data('_rlnCoordinateZ', hold)
        hold = np.asarray(self.get_column_data('_rlnOriginX'), dtype=np.float) * fsf
        self.set_column_data('_rlnOriginX', hold)
        hold = np.asarray(self.get_column_data('_rlnOriginY'), dtype=np.float) * fsf
        self.set_column_data('_rlnOriginY', hold)
        hold = np.asarray(self.get_column_data('_rlnOriginZ'), dtype=np.float) * fsf
        self.set_column_data('_rlnOriginZ', hold)

    # Checks if a column already exist
    def is_column(self, key):
        return key in self.__cols

    # Check if a column key is compatible with Relion
    def is_relion_compatible(self, key):
        return isinstance(key, str) and self.__checker.is_valid(key) and (len(key)>=4) and (key[:4] == '_rln')

    # Counts the number of elements which satisfied several (key, val) pairs
    # pairs: list of (key,val) conditions
    def count_elements(self, pairs):

        # Intialization
        count = 0

        # All rows loop
        for row in range(self.get_nrows()):

            # Pairs condition loop
            is_hit = True
            for (pkey, pval) in pairs:
                try:
                    val = self.get_element(pkey, row)
                except KeyError:
                    is_hit = False
                    break
                if val != pval:
                    is_hit = False
                    break

            # Count if all conditions are true
            if is_hit:
                count += 1

        return count


    # Add a new column, if it already existed their values are overwritten
    # key: string key for the column, only Relion accepted key strings are valid
    # val: (default -1) if scalar value set to all rows, if iterable and has the same size of the number of rows
    #       allows to set different values to every row
    # no_parse: if True (default False) the column name is not parsed
    def add_column(self, key, val=-1, no_parse=False):

        if (no_parse is True) or (self.__checker.is_valid(key)):
            dtype = self.__checker.get_dtype(key)
            if dtype is None:
                dtype = float

            # Fill up the new row
            row = list()
            if hasattr(val, '__len__'):
                if len(val) != self.__rows:
                    error_msg = 'For multiple input values their length, ' + str(len(val)) + ' must aggree' + \
                                ' with the current number of rows, ' + str(self.__rows)
                    raise PySegInputError(expr='add_column (Star)', msg=error_msg)
                for v in val:
                    row.append(dtype(v))
            else:
                for i in range(self.__rows):
                    row.append(dtype(val))

            # Add or overwrite the column
            self.__data[key] = row
            idx = None
            try:
                idx = self.__cols.index(key)
            except ValueError:
                self.__cols.append(key)
                self.__dtypes.append(dtype)
            if idx is not None:
                self.__cols[idx] = key
                self.__dtypes[idx] = dtype

        else:
            error_msg = 'Column name ' + key + ' not accepted'
            raise PySegInputError(expr='add_column (Star)', msg=error_msg)

    # Add a new row (data entry), since no default values are imposed this call must contain a value for
    # all columns, every column key and data pair is introduced via kwargs
    def add_row(self, **kwargs):
        if kwargs is not None:
            keys, values = kwargs.keys(), kwargs.values()
            if len(keys) != self.get_ncols():
                error_msg = 'Number of columns introduced for this row, ' + str(len(keys)) + ' does not ' + \
                    ' fit the current number of columns, ' + str(self.get_ncols())
                raise PySegInputError(expr='add_row (Star)', msg=error_msg)
            for key, value in zip(keys, values):
                if self.is_column(key):
                    dtype = self.get_column_type(key)
                    self.__data[key].append(dtype(value))
                else:
                    error_msg = 'Column name ' + key + ' not present.'
                    raise PySegInputError(expr='add_row (Star)', msg=error_msg)
            self.__rows += 1

    # Delete a column
    # key: key of the column to delete
    def del_column(self, key):
        if self.is_column(key):
            idx = self.__cols.index(key)
            self.__cols.pop(idx)
            self.__dtypes.pop(idx)
            del self.__data[key]

    # Delete a set of rows
    # ids: list with the indices of the rows to delete
    def del_rows(self, ids):

        # Input parsing
        if not hasattr(ids, '__len__'):
            error_msg = 'Input ids must be a list.'
            raise PySegInputError(expr='del_rows (Star)', msg=error_msg)
        if len(ids) == 0:
            return

        # Temporary data initialization
        hold_data, hold_rows = copy.copy(self.__data), self.__rows
        self.__rows = 0
        for key in hold_data:
            self.__data[key] = list()
        ids_lut = np.zeros(shape=hold_rows, dtype=np.bool)
        for idx in ids:
            try:
                ids_lut[idx] = True
            except IndexError:
                pass

        # Loop for deleting rows
        for i in range(hold_rows):
            if not ids_lut[i]:
                # Copy row in attributes
                for key in self.__data:
                    self.__data[key].append(hold_data[key][i])
                self.__rows += 1

    # Copy data from one column to one another, if the last one did exist it is created
    # key_in: key string for input column
    # key_out: key string for output column
    # func: function (func(colum_vals)) to apply to data (default None)
    def copy_data_columns(self, key_in, key_out, func=None):
        vals = self.get_column_data(key_in)
        if vals is None:
            error_msg = 'Input column with ' + key_in + ' does not exist'
            raise PySegInputError(expr='self (Star)', msg=error_msg)
        if func is not None:
            vals = func(vals)
        self.add_column(key_out, val=vals)

    # Randomize data in a column (data will be overwriten)
    # key_in: key string with the column to randomize
    # mn_val: minimum value
    # mx_val: maximum value
    def rnd_data_column(self, key_in, mn_val, mx_val):
        nrows = self.get_nrows()
        rnd_vals = (mx_val-mn_val)*np.random.random((nrows,)) + mn_val
        self.add_column(key_in, val=rnd_vals.tolist())

    # Generates a TomoPeaks object where columns will be associated to peak attributes (Peaks coordinates are
    # set to '_rlnCoordinate(i)')
    # tomo: reference tomogram full path (only the peaks of this tomogram will be considered)
    # klass: if not None (default), it allows to pick the elements of some classes, it can be a list
    # orig: if True (default False), peaks coordinates are shifted according '_rlnOrigin(i)' columns
    # full_path: if True (default) for searching particles in the tomogram the input full path is considered,
    #            otherwise just the file name
    # micro: if True (default) the particle micrograph is checked
    def gen_tomo_peaks(self, tomo, klass=None, orig=True, full_path=True, micro=True):

        # Input parsing
        in_tomo = str(tomo)
        try:
            hold_tomo = disperse_io.load_tomo(in_tomo, mmap=True)
        except pexceptions as pe:
            raise pe
        if not full_path:
            in_tomo = os.path.split(in_tomo)[1]
        if (klass is not None) and (not hasattr(klass, '__len__')):
            in_klass = (klass,)
        else:
            in_klass = klass
        tpeaks = pyseg.sub.TomoPeaks(shape=hold_tomo.shape, name=in_tomo)
        del hold_tomo

        # Rows loop
        pid = 0
        for i in range(self.__rows):

            # Check if particle in tomogram
            if full_path:
                hold_str = self.get_element('_rlnMicrographName', i)
            else:
                hold_str = os.path.split(self.get_element('_rlnMicrographName', i))[1]
            if (not micro) or (hold_str == in_tomo) :
                # Check if it is in classes list
                if (klass is None) or (self.get_element('_rlnClassNumber', i) in klass):

                    # Crate a new peak
                    x, y, z = self.get_element('_rlnCoordinateX', i), self.get_element('_rlnCoordinateY', i), \
                              self.get_element('_rlnCoordinateZ', i)
                    if orig:
                        try:
                            ox = self.get_element('_rlnOriginX', i)
                        except KeyError:
                            ox = 0.
                        try:
                            oy = self.get_element('_rlnOriginY', i)
                        except KeyError:
                            oy = 0.
                        try:
                            oz = self.get_element('_rlnOriginZ', i)
                        except KeyError:
                            oz = 0.
                        x -= ox
                        y -= oy
                        z -= oz

                    # Add the peak to TomoPeaks
                    tpeaks.add_peak((x, y, z))

                    # Add STAR file attributes
                    if tpeaks.get_num_peaks() == 1:
                        for key in self.get_column_keys():
                            tpeaks.add_prop(key, 1, dtype=self.get_column_type(key))
                    for key in self.get_column_keys():
                        tpeaks.set_peak_prop(pid, key, self.get_element(key, i))
                    pid += 1

        return tpeaks

    # fname: full path to the input Star file to read
    def load(self, fname):

        # File reading
        lines = None
        with open(fname, 'r') as ffile:
            lines = ffile.readlines()
            ffile.close()
        if lines is None:
            error_msg = 'File ' + fname + ' was empty.'
            raise PySegInputError(expr='load (Star)', msg=error_msg)
        lidx = 0

        # Parse Header 1
        self.__header_1 = list()
        while (lidx < len(lines)) and (lines[lidx][0] != '_'):
            self.__header_1.append(lines[lidx])
            lidx += 1

        # Parse Header 2
        self.__cols = list()
        while (lidx < len(lines)) and (lines[lidx][0] == '_'):
            col = lines[lidx].split()[0]
            if self.__checker.is_valid(col) and (not self.is_column(col)):
                self.__data[col] = list()
                self.__dtypes.append(self.__checker.get_dtype(col))
                self.__cols.append(col)
            else:
                error_msg = 'Unexpected error parsing star file ' + fname + ' header.'
                raise PySegInputError(expr='load (Star)', msg=error_msg)
            lidx += 1

        # Parse data rows
        for line in lines[lidx:]:
            datas = line.split()
            # EOF condition
            if self.get_ncols() != len(datas):
                break
            for (d, dtype, col) in zip(datas, self.__dtypes, self.__cols):
                try:
                    typed_d = dtype(d)
                except ValueError:
                    if dtype is int:
                        typed_d = int(float(d))
                    else:
                        raise ValueError
                self.__data[col].append(typed_d)
            self.__rows += 1

        self.__root_dir, self.__fname = os.path.split(fname)

    # Parse class attributes into a STAR format
    # Returns: an string with the STAR file
    def to_string(self):
        hold = str()
        for line in self.__header_1:
            hold += line
        for i, key in enumerate(self.__cols):
            hold += (key + ' ' + '#' + str(i+1) + '\n')
        if self.get_ncols() > 0:
            keys = self.get_column_keys()
            for i in range(self.get_nrows()):
                for key in keys[:-1]:
                    hold += (str(self.__data[key][i]) + '\t')
                hold += (str(self.__data[keys[-1]][i]) + '\n')
        return hold + '\n'

    # Store in a STAR
    # fname: output file name
    # sv: if not None (default), it specified shape (all dimensions must be even) for subvolumes that will be stored in a sub-folder called 'sub'
    #     in the same directory as the STAR file from reference tomograms (density is inverted and normalized)
    # mask: mask applied for sub-volume normalization (default None)
    # swap_xy: swap XY coordinate only for subvolume extraction (only applies if sv is not None)
    def store(self, fname, sv=None, mask=None, swap_xy=False, del_ang=(0,0,0)):

        # Sub-volumes
        if sv is not None:

            # Parsing input shape
            if (not hasattr(sv, '__len__')) or (not(len(sv) == 3)) \
                or (sv[0]<=0) or (sv[1]<=0) or (sv[2]<=0):
                error_msg = 'Subvolume shape must be 3-tuple with non trivial values.'
                raise PySegInputError(expr='store (Star)', msg=error_msg)
            if ((sv[0]%2) != 0) or ((sv[1]%2) != 0) or ((sv[2]%2) != 0):
                error_msg = 'All subvolume dimensions must be even, current ' + str(sv)
                raise PySegInputError(expr='store (Star)', msg=error_msg)
            hl_x, hl_y, hl_z = int(sv[1]*.5), int(sv[0]*.5), int(sv[2]*.5)
            tomo_path = ''
            sv_path = os.path.split(fname)[0] + '/sub'
            try:
                os.makedirs(sv_path)
            except OSError as exception:
                if exception.errno != errno.EEXIST:
                    error_msg = 'Directory ' + sv_path + ' could not be created'
                    raise PySegInputError(expr='store (Star)', msg=error_msg)


            # Rows loop
            for i in range(self.__rows):

                # Particle coordinate
                x, y, z = self.__data['_rlnCoordinateX'][i], self.__data['_rlnCoordinateY'][i], \
                          self.__data['_rlnCoordinateZ'][i]


                # Read tomogram map (if required only)
                hold_path = self.__data['_rlnMicrographName'][i]
                if hold_path != tomo_path:
                    tomo_path = hold_path
                    try:
                        tomo = disperse_io.load_tomo(tomo_path, mmap=True)
                    except KeyError:
                        error_msg = 'Reference tomogram ' + tomo_path + ' could not be read'
                        raise PySegInputError(expr='save_subvolumes (ParticleList)', msg=error_msg)

                # Cropping
                x_l, y_l, z_l = x-hl_x+1, y-hl_y+1, z-hl_z+1
                if (x_l < 0) or (y_l < 0) or (z_l < 0):
                    hold_sub = np.zeros(shape=sv, dtype=np.float32)
                x_h, y_h, z_h = x+hl_x+1, y+hl_y+1, z+hl_z+1
                if (x_l >= tomo.shape[0]) or (y_l >= tomo.shape[1]) or (z_l >= tomo.shape[2]):
                    hold_sub = np.zeros(shape=sv, dtype=np.float32)
                if swap_xy:
                    hold_sub = tomo[x_l:x_h, y_l:y_h, z_l:z_h]
                else:
                    hold_sub = tomo[x_l:x_h, y_l:y_h, z_l:z_h]

                # Normalization
                hold_sub = relion_norm(hold_sub, mask=mask)

                # Storing the sub-volume and updating image name
                part_name = sv_path + '/' + os.path.split(self.__data['_rlnImageName'][i])[1]
                disperse_io.save_numpy(hold_sub, part_name)
                self.__data['_rlnImageName'][i] = part_name

        # Angles deletion
        if del_ang[0] > 0:
            self.__data['_rlnAngleRot'][i] = 0
        if del_ang[1] > 0:
            self.__data['_rlnAngleTilt'][i] = 0
        if del_ang[2] > 0:
            self.__data['_rlnAnglePsi'][i] = 0

        # STAR file
        with open(fname, 'w') as ffile:
            ffile.write(self.to_string())

    # Compute alignment against a reference star file
    # ref_star: reference STAR file, with the same (at least partially) particles
    # ref_vect: reference vector
    # Returns: two arrays with length equal to the number of particles, first angles miss-alignment, second shifting.
    #          Negative values represent unexpected events.
    def compute_malign(self, ref_star, ref_vect):

        # Initialization
        n_parts = self.get_nrows()
        angs, shifts = (-1.)*np.ones(shape=n_parts, dtype=np.float32), (-1.)*np.ones(shape=n_parts, dtype=np.float32)
        a_ref_vect = np.mat(np.asarray(ref_vect, dtype=np.float32)).transpose()

        # Particles loop
        for i in range(n_parts):

            # Getting particle index in reference star file
            p_name = self.get_element('_rlnImageName', i)
            try:
                j = ref_star.find_element('_rlnImageName', p_name)
            except ValueError:
                continue

            # Geting particle information
            rot, psi, tilt = self.get_element('_rlnAngleRot', i), self.get_element('_rlnAnglePsi', i), \
                             self.get_element('_rlnAngleTilt', i)
            r_rot, r_psi, r_tilt = ref_star.get_element('_rlnAngleRot', j), ref_star.get_element('_rlnAnglePsi', j), \
                                   ref_star.get_element('_rlnAngleTilt', j)

            # Computing angle miss-alignment
            # mat_1 = rot_mat_relion(rot, psi, tilt, deg=True)
            # mat_2 = rot_mat_relion(r_rot, r_psi, r_tilt, deg=True)
            mat_1 = rot_mat_relion(rot, tilt, psi, deg=True)
            mat_2 = rot_mat_relion(r_rot, r_tilt, r_psi, deg=True)
            rot_v_1 = mat_1.T * a_ref_vect
            rot_v_2 = mat_2.T * a_ref_vect
            angs[i] = math.degrees(angle_2vec_3D(rot_v_1, rot_v_2))

            # print 'P=' + str((rot, tilt, psi)) + ', R=' + str((r_rot, r_tilt, r_psi)) + ', A=' + str(angs[i]) + 'deg'

            # Compute shift miss-alignment
            c_x, c_y, c_z = self.get_element('_rlnCoordinateX', i), self.get_element('_rlnCoordinateY', i), \
                            self.get_element('_rlnCoordinateZ', i)
            try:
                o_x, o_y, o_z = self.get_element('_rlnOriginX', i), self.get_element('_rlnOriginY', i), \
                                self.get_element('_rlnOriginZ', i)
            except KeyError:
                o_x, o_y, o_z = 0., 0., 0.
            coord = np.asarray((c_x-o_x, c_y-o_y, c_z-o_z), dtype=np.float32)
            c_x, c_y, c_z = ref_star.get_element('_rlnCoordinateX', j), ref_star.get_element('_rlnCoordinateY', j), \
                            ref_star.get_element('_rlnCoordinateZ', j)
            try:
                o_x, o_y, o_z = ref_star.get_element('_rlnOriginX', j), ref_star.get_element('_rlnOriginY', j), \
                                ref_star.get_element('_rlnOriginZ', j)
            except KeyError:
                o_x, o_y, o_z = 0., 0., 0.
            coord_r = np.asarray((c_x-o_x, c_y-o_y, c_z-o_z), dtype=np.float32)
            hold = coord - coord_r
            shifts[i] = math.sqrt((hold * hold).sum())

        return angs, shifts

    # Group particles into groups
    # min_gp: (default None) first particles are splited into groups by the their micrograph, if they don't reach min_gp
    #         values their are gathered the smallest group size first criteria
    # Returns: the column '_rlnGroupNumber' is created or updated
    def particle_group(self, min_gp=None):

        # Initialization
        try:
            curr_gp = list(set(self.get_column_data('_rlnGroupNumber')))
            # print str(curr_gp)
            nparts_gp = np.zeros(shape=len(curr_gp), dtype=np.int).tolist()
            for row in range(self.get_nrows()):
                gp = self.get_element('_rlnGroupNumber', row)
                idx = curr_gp.index(gp)
                nparts_gp[idx] += 1
        except ValueError:
            try:
                mics = self.get_column_data('_rlnImageName')
                curr_gp = np.arange(len(mics)).tolist()
                nparts_gp = np.zeros(shape=len(curr_gp), dtype=np.int).tolist()
                for row in range(self.get_nrows()):
                    mic = self.get_element('_rlnImageName', row)
                    nparts_gp[mics.index(mic)] += 1
            except ValueError:
                curr_gp = [1,]
                nparts_gp = (self.get_nrows(),)
            self.add_column('_rlnGroupNumber', val=curr_gp[0])

        # Loop until the gathering is finished
        lut_gp = np.arange(0, np.asarray(curr_gp, dtype=np.int).max()+1)
        while len(set(curr_gp)) > 1:
            # Find the smallest group
            min_ids = np.argsort(np.asarray(nparts_gp, dtype=np.int))
            if (nparts_gp[min_ids[0]] > min_gp) or (len(min_ids) <= 1):
                break
            # Find the pair to gather
            for i in range(len(lut_gp)):
                if lut_gp[i] == curr_gp[min_ids[0]]:
                    lut_gp[i] = curr_gp[min_ids[1]]
            lut_gp[curr_gp[min_ids[0]]] = curr_gp[min_ids[1]]
            nparts_gp[min_ids[1]] += nparts_gp[min_ids[0]]
            # Deleting groups
            curr_gp.pop(min_ids[0])
            nparts_gp.pop(min_ids[0])

        # Setting the groups
        new_gp = np.zeros(shape=self.get_nrows(), dtype=np.int)
        for row in range(self.get_nrows()):
            old_gp = self.get_element('_rlnGroupNumber', row)
            new_gp[row] =  lut_gp[old_gp]
        self.set_column_data('_rlnGroupNumber', new_gp)

    # Check if two (self and input) are comparable, i.e. have the same particles in the same location
    def check_comparable(self, star):
        if self.get_nrows() != star.get_nrows():
            return False
        else:
            for i in range(self.get_nrows()):
                if self.get_element('_rlnImageName', i) != star.get_element('_rlnImageName', i):
                    return False
        return True

    # Returns a list of STAR file objects one for each class
    def split_class(self):

        if not self.has_column('_rlnClassNumber'):
            error_msg = 'No _rlnClassNumber column found!'
            raise PySegInputError(expr='split_class (Star)', msg=error_msg)

        else:
            classes = self.get_column_data('_rlnClassNumber')
            class_ids = set(classes)
            stars, classes = list(), np.asarray(classes, dtype=np.int)
            keys = self.get_column_keys()
            for class_id in class_ids:
                star = Star()
                for key in keys:
                    star.add_column(key)
                rows = np.where(classes == class_id)[0]
                for row in rows:
                    kwargs = dict()
                    for key in keys:
                        kwargs[key] = self.get_element(key, row)
                    star.add_row(**kwargs)
                stars.append(star)
            return stars

    # Generates another Star object with a random subset of particles
    # n: number of output rows (if greater thant the current number then just a copy is provided)
    # parse: weather parse or not (default False)
    def gen_random_subset(self, n, parse=False):

        # Initialization
        if n <= 0:
            error_msg = 'Input number of particles must be greater than zero!'
            raise PySegInputError(expr='gen_random_subset (Star)', msg=error_msg)
        elif n >= self.get_nrows():
            if parse:
                return self.get_relion_parsed_copy()
            else:
                return copy.deepcopy(self)

        # Creating the random subset
        rnd_idx = np.random.randint(0, self.get_nrows(), n)

        return self.get_subset(rnd_idx)

    # Returns a Star object with a subset of rows from an input list of rows
    def get_subset(self, rows):
        hold_star = Star()
        for key in self.get_column_keys():
            hold_star.add_column(key)
        for idx in rows:
            kwargs = dict()
            for key in hold_star.get_column_keys():
                kwargs[key] = self.get_element(key, idx)
            hold_star.add_row(**kwargs)
        return hold_star

    # Returns a copy parsed for being Relion compatible
    def get_relion_parsed_copy(self):
        hold_star = copy.deepcopy(self)
        for key in self.get_column_keys():
            if not self.is_relion_compatible(key):
                hold_star.del_column(key)
        return hold_star

    # Generates a binned copy of the current star file
    # sout_dir: output directory for storing the binned subvolumes
    # bin: binning factor (default 2)
    # res: original resolution vx/nm
    # cutoff: cutoff in nm for low pass filtering, if None not applied
    # npr: number of process, None (defaults) means they are adjusted automatically to the number of cores
    # Returns: a new STAR file pointing to the binned subvolues
    def gen_binned_copy(self, sout_dir, bin=2, res=1, cutoff=None, npr=None):

        # Parsing STAR file only particles star file can be binned
        if bin <= 0:
            error_msg = 'Input binning cannot be equal or least than zero!'
            raise PySegInputError(expr='gen_binned_copy (Star)', msg=error_msg)
        if not os.path.exists(sout_dir):
            error_msg = 'Input directory ' + str(sout_dir) + ' does not exist!'
            raise PySegInputError(expr='gen_binned_copy (Star)', msg=error_msg)
        if not self.is_column('_rlnImageName'):
            error_msg = 'This STAR file cannot be binned, it does not contain _rlnImageName column!'
            raise PySegInputError(expr='gen_binned_copy (Star)', msg=error_msg)
        try:
            part_path = self.get_element('_rlnImageName', 0)
        except ValueError:
            error_msg = 'No particles to process!'
            raise PySegInputError(expr='particles_template_matching (Star)', msg=error_msg)
        sv_shape = disperse_io.load_tomo(part_path, mmap=True).shape
        star_cp = copy.deepcopy(self)
        bin_f = float(bin)

        # Coordinates loop
        for i in xrange(star_cp.get_nrows()):
            try:
                val = star_cp.get_element('_rlnCoordinateX', i)
                star_cp.set_element('_rlnCoordinateX', i, val/bin_f)
            except ValueError:
                pass
            try:
                val = star_cp.get_element('_rlnCoordinateY', i)
                star_cp.set_element('_rlnCoordinateY', i, val/bin_f)
            except ValueError:
                pass
            try:
                val = self.get_element('_rlnCoordinateZ', i)
                self.set_element('_rlnCoordinateZ', i, val/bin_f)
            except ValueError:
                pass
            try:
                val = star_cp.get_element('_rlnOriginX', i)
                star_cp.set_element('_rlnOriginX', i, val/bin_f)
            except ValueError:
                pass
            try:
                val = star_cp.get_element('_rlnOriginY', i)
                star_cp.set_element('_rlnOriginY', i, val/bin_f)
            except ValueError:
                pass
            try:
                val = star_cp.get_element('_rlnOriginZ', i)
                star_cp.set_element('_rlnOriginZ', i, val/bin_f)
            except ValueError:
                pass

        # For temporary CSVs
        tmp_csv_dir = sout_dir + '/tmp_star_csv_dir'
        if not os.path.exists(tmp_csv_dir):
            os.makedirs(tmp_csv_dir)

        # Call to parallel subvolume processing
        if npr is None:
            npr = mp.cpu_count()
        pr_ids = list()
        # Create the list on indices to split
        npart = star_cp.get_nrows()
        sym_ids = np.arange(npart)
        spl_ids = np.array_split(range(len(sym_ids)), npr)
        if npr <= 1:
            pr_bin_star_svols(-1, spl_ids[0][0], spl_ids[0][-1], bin_f, res, sv_shape, cutoff, star_cp.__data, sout_dir,
                              tmp_csv_dir)
            pr_ids.append(-1)
        else:
            processes = list()
            for pr_id in range(npr):
                pr = mp.Process(target=pr_bin_star_svols, args=(pr_id, spl_ids[pr_id][0], spl_ids[pr_id][-1],
                                                                bin_f, res, sv_shape, cutoff, star_cp.__data, sout_dir,
                                                                tmp_csv_dir))
                pr.start()
                processes.append(pr)
                pr_ids.append(pr_id)
            pr_results = list()
            for pr in processes:
                pr.join()
                pr_results.append(pr.exitcode)
            for pr_id in range(len(processes)):
                if pr_id != pr_results[pr_id]:
                    error_msg = 'Process ' + str(pr_id) + ' exited unexpectedly!'
                    raise pexceptions.PySegInputError(expr='__class_by_exemplars (ClassStar)', msg=error_msg)
            gc.collect()

        # From CSVs to STAR
        row_id, keys = 0, star_cp.get_column_keys()
        for pr_id in pr_ids:
            tmp_csv = tmp_csv_dir + '/process_' + str(pr_id) + '.csv'
            with open(tmp_csv, 'r') as tfile:
                reader = csv.DictReader(tfile)
                for i, hold_row in enumerate(reader):
                    # if i == 0:
                    #     continue
                    star_cp.set_element('_rlnImageName', row_id, hold_row['_rlnImageName'])
                    try:
                        star_cp.set_element('_rlnCtfImage', row_id, hold_row['_rlnCtfImage'])
                    except ValueError:
                        pass
                    row_id += 1
        clean_dir(tmp_csv_dir)

        return star_cp

    # Computes template matching for particles STAR file
    # temp: template subvolume (same size as particles)
    # res: subvolues resolution (nm/vx)
    # max_shift: maximum shifting in nm)
    # mask: smooth borders mask (same size as particles), default None
    # rots|tilts|psis: sample for Euler angles (Relion format), default None
    # npr: number of process, None (defaults) means they are adjusted automatically to the number of cores
    # Returns: cross-correlation to input template scores for every particles in column '_psCCScores'
    def particles_template_matching(self, temp, res, max_shift=0, mask=None, rots=None, tilts=None, psis=None, npr=None):

        # Parsing STAR file only particles star file can be binned
        if not self.is_column('_rlnImageName'):
            error_msg = 'Template matching cannot be compute if _rlnImageName does not exist!'
            raise PySegInputError(expr='particles_template_matching (Star)', msg=error_msg)
        if res <= 0:
            error_msg = 'Input resolution cannot be equal or least than zero!'
            raise PySegInputError(expr='particles_template_matching (Star)', msg=error_msg)
        if max_shift < 0:
            error_msg = 'Input maximum shift cannot be least than zero!'
            raise PySegInputError(expr='particles_template_matching (Star)', msg=error_msg)
        try:
            part_path = self.get_element('_rlnImageName', 0)
        except ValueError:
            error_msg = 'No particles to process!'
            raise PySegInputError(expr='particles_template_matching (Star)', msg=error_msg)
        hold_part = disperse_io.load_tomo(part_path, mmap=True)
        if (len(temp.shape) != 3) or ((np.asarray(temp.shape) == np.asarray(hold_part.shape)).sum() != 3):
            error_msg = 'Input template must have particles shape: ' + str(hold_part.shape)
            raise PySegInputError(expr='particles_template_matching (Star)', msg=error_msg)
        if mask is None:
            mask = np.ones(shape=temp.shape, dtype=np.float32)
        elif (len(mask.shape) != 3) or ((np.asarray(mask.shape) == np.asarray(hold_part.shape)).sum() != 3):
            error_msg = 'Input template must have particles shape: ' + str(hold_part.shape)
            raise PySegInputError(expr='particles_template_matching (Star)', msg=error_msg)

        # Sampling arrays
        max_shift_v = math.ceil(max_shift / res)
        grid = np.arange(-max_shift_v, max_shift_v+1)
        shift_arr = np.zeros(shape=(len(grid)**2,3), dtype=np.float32)
        cont = 0
        for x, y in it.product(grid, grid):
            shift_arr[cont, :] = np.asarray((x, y, 0.), dtype=np.float32)
            cont += 1
        angs_arr = np.zeros(shape=(len(rots)*len(tilts)*len(psis),3), dtype=np.float32)
        cont = 0
        for rot, tilt, psi in it.product(rots, tilts, psis):
            angs_arr[cont, :] = np.asarray((rot, tilt, psi), dtype=np.float32)
            cont += 1


        # Call to parallel subvolume processing
        if npr is None:
            npr = mp.cpu_count()
        processes = list()
        # Create the list on indices to split
        npart = self.get_nrows()
        sym_ids = np.arange(npart)
        spl_ids = np.array_split(range(len(sym_ids)), npr)
        shared_ncc = mp.Array('f', npart)
        if npr <= 1:
            pr_star_tm(-1, spl_ids[0], temp, mask, self.__data, max_shift_v, angs_arr,
                       shared_ncc)
        else:
            for pr_id in range(npr):
                pr = mp.Process(target=pr_star_tm, args=(pr_id, spl_ids[pr_id], temp, mask, self.__data, max_shift_v, angs_arr,
                                                         shared_ncc))
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

        # Add scores column
        scores = np.frombuffer(shared_ncc.get_obj(), dtype=np.float32).reshape(npart)
        self.add_column(key='_psCCScores', val=scores)

    # Computes template matching using Z-axis averaged 2D particles
    # temp: template subvolume (same size as particles)
    # res: subvolues resolution (nm/vx)
    # max_shift: maximum shifting in nm)
    # mask: smooth borders mask (same size as particles), default None
    # npr: number of process, None (defaults) means they are adjusted automatically to the number of cores
    # Returns: cross-correlation to input template scores for every particles in column '_psCCScores'
    def particles_template_matching_za(self, temp, res, max_shift=0, mask=None, npr=None):

        # Parsing STAR file only particles star file can be binned
        if not self.is_column('_rlnImageName'):
            error_msg = 'Template matching cannot be compute if _rlnImageName does not exist!'
            raise PySegInputError(expr='particles_template_matching (Star)', msg=error_msg)
        if res <= 0:
            error_msg = 'Input resolution cannot be equal or least than zero!'
            raise PySegInputError(expr='particles_template_matching (Star)', msg=error_msg)
        if max_shift < 0:
            error_msg = 'Input maximum shift cannot be least than zero!'
            raise PySegInputError(expr='particles_template_matching (Star)', msg=error_msg)
        try:
            part_path = self.get_element('_rlnImageName', 0)
        except ValueError:
            error_msg = 'No particles to process!'
            raise PySegInputError(expr='particles_template_matching (Star)', msg=error_msg)
        hold_part = disperse_io.load_tomo(part_path, mmap=True)
        if (len(temp.shape) != 3) or ((np.asarray(temp.shape) == np.asarray(hold_part.shape)).sum() != 3):
            error_msg = 'Input template must have particles shape: ' + str(hold_part.shape)
            raise PySegInputError(expr='particles_template_matching (Star)', msg=error_msg)
        if mask is None:
            mask = np.ones(shape=temp.shape, dtype=np.float32)
        elif (len(mask.shape) != 3) or ((np.asarray(mask.shape) == np.asarray(hold_part.shape)).sum() != 3):
            error_msg = 'Input template must have particles shape: ' + str(hold_part.shape)
            raise PySegInputError(expr='particles_template_matching (Star)', msg=error_msg)

        # Sampling shifting (only normal plane to Z-axis is considered)
        max_shift_v = math.ceil(max_shift / res)
        grid = np.arange(-max_shift_v, max_shift_v+1)
        shift_arr = np.zeros(shape=(len(grid)**2,3), dtype=np.float32)
        cont = 0
        for x, y in it.product(grid, grid):
            shift_arr[cont, :] = np.asarray((x, y, 0.), dtype=np.float32)
            cont += 1

        # Call to parallel subvolume processing
        if npr is None:
            npr = mp.cpu_count()
        processes = list()
        # Create the list on indices to split
        npart = self.get_nrows()
        sym_ids = np.arange(npart)
        spl_ids = np.array_split(range(len(sym_ids)), npr)
        shared_ncc = mp.Array('f', npart)
        if npr <= 1:
            pr_star_tm_za(-1, spl_ids[0], temp, mask, self.__data, shift_arr,
                       shared_ncc)
        else:
            for pr_id in range(npr):
                pr = mp.Process(target=pr_star_tm_za, args=(pr_id, spl_ids[pr_id], temp, mask, self.__data, shift_arr,
                                                         shared_ncc))
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

        # Add scores column
        scores = np.frombuffer(shared_ncc.get_obj(), dtype=np.float32).reshape(npart)
        self.add_column(key='_psCCScores', val=scores)