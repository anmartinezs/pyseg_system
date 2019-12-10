"""
Classes for doing the spatial analysis of clouds of points embedded in a dense array.
This methodology allows to handle with irregular domains in 2D and 3D and multivariate cases.

# Author: Antonio Martinez-Sanchez (Max Planck Institute for Biochemistry)
# Date: 12.03.16
"""

__author__ = 'martinez'

import gc
from pyorg.globals import *
import Queue
import random
from variables import *
import multiprocessing as mp
import threading as mt
from scipy.signal import butter, lfilter
# import cv2
from matplotlib.pyplot import cm
from matplotlib import pyplot as plt
from abc import *
from plane import make_plane
from pyorg import pexceptions, disperse_io
# import skfmm

try:
    import cPickle as pickle
except:
    import pickle

##############################################################################################
# Global variables
#

LP_ORDER = 5
CROP_OFF = 0
ROT_TMP_TH = 0.9
K_NUM_CORR = 2.

##############################################################################################
# Helper functions
#

# Generates a 3D probability map for a binding site with normal profile
# temp: protein binary template (1-proteing volume, 0-Potential volume), 3d numpy array
# bind: binding site coordinates on the template, 1D numpy array with three coordinates
# rm: radius were maximum potential is reached (voxels)
# sg: sigma for the gaussian distribution (voxels)
# cutoff: number of sigmal for cutting-off tails (default)
def gen_pb_normal(temp, bind, sg, rm, cutoff=3):
    # Compute probality mask
    X, Y, Z = np.meshgrid(np.arange(1,temp.shape[0]+1), np.arange(1,temp.shape[1]+1), np.arange(1,temp.shape[2]+1),
                          indexing='ij')
    X, Y, Z = X-bind[0], Y-bind[1], Z-bind[2]
    X, Y, Z = X*X, Y*Y, Z*Z
    R = np.sqrt(X + Y + Z) - rm
    # Cutoff mask
    M = np.ones(shape=temp.shape, dtype=np.bool)
    M[R<=(cutoff*sg)] = False
    P = (1./np.sqrt(2.*sg*sg*np.pi)) * np.exp((-1.*(1./(2.*sg))) * R * R)
    # Applying mask
    M |= temp
    P[M] = 0.
    return P

# Sums two probability maps
# pmap1, pmap2: input probability maps to sum
# mask1, mask2: corresponding binary masks for void regions (True) in the map
def sum_pbs(pmap1, pmap2, mask1, mask2):
    mask = np.invert(mask1 * mask2)
    pmaps = np.zeros(shape=pmap1.shape, dtype=pmap1.dtype)
    pmaps[mask] = pmap1[mask] + pmap2[mask]
    return pmaps

# Returns the a subvolume of a tomogram from a center and a shape
# tomo: input tomogram
# sub_pt: subtomogram center point
# sub_shape: output subtomogram shape (all dimensions must be odd)
# copy: if True (default) a copy instead a VOI is returned
# Returns: a copy with the subvolume or a VOI, and its offset (left-up-top corner)
def get_sub_copy(tomo, sub_pt, sub_shape, copy=True):

    # Initialization
    nx, ny, nz = sub_shape[0], sub_shape[1], sub_shape[2]
    mx, my, mz = tomo.shape[0], tomo.shape[1], tomo.shape[2]
    mx1, my1, mz1 = mx-1, my-1, mz-1
    hl_x, hl_y, hl_z = int(nx*.5), int(ny*.5), int(nz*.5)
    x, y, z = sub_pt[0], sub_pt[1], sub_pt[2]

    # Compute bounding restriction
    off_l_x, off_l_y, off_l_z = x-hl_x+1, y-hl_y+1, z-hl_z+1
    off_h_x, off_h_y, off_h_z = x+hl_x+1, y+hl_y+1, z+hl_z+1
    if off_l_x < 0:
        off_l_x = 0
    if off_l_y < 0:
        off_l_y = 0
    if off_l_z < 0:
        off_l_z = 0
    if off_h_x >= mx:
        off_h_x = mx1
    if off_h_y >= my:
        off_h_y = my1
    if off_h_z >= mz:
        off_h_z = mz1
    # Update mask if no overlapping
    if copy:
        return tomo[off_l_x:off_h_x, off_l_y:off_h_y, off_l_z:off_h_z], \
               np.asarray((off_l_x, off_l_y, off_l_z), dtype=np.float32)
    else:
        hold_sv = tomo[off_l_x:off_h_x, off_l_y:off_h_y, off_l_z:off_h_z]
        return hold_sv[:], np.asarray((off_l_x, off_l_y, off_l_z), dtype=np.float32)

# Check no binary overlap between subvolume and an a tomogram
# tomo: input tomogram where the subvolume will be inserted
# point: point where sub_tomo center will be placed
# sub_tomo: sub-tomogram for being added it should be equal smaller than tomo in any dimension (all dimensions must be odd)
# Raturns: True if overlap othewise False
def check_bin_sub_tomo(tomo, point, sub_tomo):

    # Initialization
    nx, ny, nz = sub_tomo.shape[0], sub_tomo.shape[1], sub_tomo.shape[2]
    mx, my, mz = tomo.shape[0], tomo.shape[1], tomo.shape[2]
    mx1, my1, mz1 = mx-1, my-1, mz-1
    hl_x, hl_y, hl_z = int(sub_tomo.shape[0]*.5), int(sub_tomo.shape[1]*.5), int(sub_tomo.shape[2]*.5)
    x, y, z = point[0], point[1], point[2]

    # Compute bounding restriction
    off_l_x, off_l_y, off_l_z = x-hl_x+1, y-hl_y+1, z-hl_z+1
    off_h_x, off_h_y, off_h_z = x+hl_x+1, y+hl_y+1, z+hl_z+1
    dif_l_x, dif_l_y, dif_l_z = 0, 0, 0
    dif_h_x, dif_h_y, dif_h_z = nx, ny, nz
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
    # Update mask if no overlapping
    hold_sv = tomo[off_l_x:off_h_x, off_l_y:off_h_y, off_l_z:off_h_z]
    hold_sub = sub_tomo[dif_l_x:dif_h_x, dif_l_y:dif_h_y, dif_l_z:dif_h_z]
    if hold_sv[hold_sub].sum() > 0:
        return True
    else:
        return False

# Forcibly insert a sub tomogram in a bigger tomogram given a point
# tomo: input tomogram where the subvolume will be inserted
# point: point where sub_tomo center will be placed
# sub_tomo: sub-tomogram for being added it should be equal smaller than tomo in any dimension (all dimensions must be odd)
def insert_sub_tomo(tomo, point, sub_tomo):

    # Initialization
    nx, ny, nz = sub_tomo.shape[0], sub_tomo.shape[1], sub_tomo.shape[2]
    mx, my, mz = tomo.shape[0], tomo.shape[1], tomo.shape[2]
    mx1, my1, mz1 = mx-1, my-1, mz-1
    hl_x, hl_y, hl_z = int(sub_tomo.shape[0]*.5), int(sub_tomo.shape[1]*.5), int(sub_tomo.shape[2]*.5)
    x, y, z = point[0], point[1], point[2]

    # Compute bounding restriction
    off_l_x, off_l_y, off_l_z = x-hl_x+1, y-hl_y+1, z-hl_z+1
    off_h_x, off_h_y, off_h_z = x+hl_x+1, y+hl_y+1, z+hl_z+1
    dif_l_x, dif_l_y, dif_l_z = 0, 0, 0
    dif_h_x, dif_h_y, dif_h_z = nx, ny, nz
    if off_l_x < 0:
        if no_over:
            raise ValueError
        dif_l_x = abs(off_l_x)
        off_l_x = 0
    if off_l_y < 0:
        if no_over:
            raise ValueError
        dif_l_y = abs(off_l_y)
        off_l_y = 0
    if off_l_z < 0:
        if no_over:
            raise ValueError
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
    tomo[off_l_x:off_h_x, off_l_y:off_h_y, off_l_z:off_h_z] = sub_tomo[dif_l_x:dif_h_x, dif_l_y:dif_h_y, dif_l_z:dif_h_z]

# Insert a sub tomogram in a bigger tomogram given a point (center), only valid for binary tomograms
# tomo: input tomogram where the subvolume will be inserted
# point: point where sub_tomo center will be placed
# sub_tomo: sub-tomogram for being added it should be equal smaller than tomo in any dimension (all dimensions must be odd)
# no_over: if True (default) it check no overlap with between tomo and sub_tomo foregrounds (fg are non-zero
#          values)
# Raturns: a ValueError exception in it could not be inserted because overlapping
def insert_sub_tomo_bin(tomo, point, sub_tomo, no_over=True):

    # Initialization
    nx, ny, nz = sub_tomo.shape[0], sub_tomo.shape[1], sub_tomo.shape[2]
    mx, my, mz = tomo.shape[0], tomo.shape[1], tomo.shape[2]
    mx1, my1, mz1 = mx-1, my-1, mz-1
    hl_x, hl_y, hl_z = int(sub_tomo.shape[0]*.5), int(sub_tomo.shape[1]*.5), int(sub_tomo.shape[2]*.5)
    x, y, z = point[0], point[1], point[2]

    # Compute bounding restriction
    off_l_x, off_l_y, off_l_z = x-hl_x+1, y-hl_y+1, z-hl_z+1
    off_h_x, off_h_y, off_h_z = x+hl_x+1, y+hl_y+1, z+hl_z+1
    dif_l_x, dif_l_y, dif_l_z = 0, 0, 0
    dif_h_x, dif_h_y, dif_h_z = nx, ny, nz
    if off_l_x < 0:
        if no_over:
            raise ValueError
        dif_l_x = abs(off_l_x)
        off_l_x = 0
    if off_l_y < 0:
        if no_over:
            raise ValueError
        dif_l_y = abs(off_l_y)
        off_l_y = 0
    if off_l_z < 0:
        if no_over:
            raise ValueError
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
    # Update mask if no overlapping
    hold_sv = tomo[off_l_x:off_h_x, off_l_y:off_h_y, off_l_z:off_h_z]
    hold_sub = sub_tomo[dif_l_x:dif_h_x, dif_l_y:dif_h_y, dif_l_z:dif_h_z]
    if no_over:
        hold_sv[hold_sub] = True
        tomo[off_l_x:off_h_x, off_l_y:off_h_y, off_l_z:off_h_z] = hold_sv
    else:
        if hold_sv[hold_sub].sum() == 0:
            hold_sv[hold_sub] = True
            tomo[off_l_x:off_h_x, off_l_y:off_h_y, off_l_z:off_h_z] = hold_sv
        else:
            raise ValueError

# Overwrite tomogram data from a subvolume a sub tomogram in a bigger tomogram given a point (center)
# tomo: input tomogram where the subvolume will be inserted
# point: point where sub_tomo center will be placed
# sub_tomo: sub-tomogram for being added it should be equal ofr smaller than tomo in any dimension (all dimensions must be odd)
# op: numpy point-wise function for overwritting, p.e., np.min(tomo,sub_tomo)
def over_sub_tomo(tomo, point, sub_tomo, op):

    # Initialization
    nx, ny, nz = sub_tomo.shape[0], sub_tomo.shape[1], sub_tomo.shape[2]
    mx, my, mz = tomo.shape[0], tomo.shape[1], tomo.shape[2]
    mx1, my1, mz1 = mx-1, my-1, mz-1
    # hl_x, hl_y, hl_z = int(tomo.shape[1]*.5), int(tomo.shape[0]*.5), int(tomo.shape[2]*.5)
    hl_x, hl_y, hl_z = int(sub_tomo.shape[0]*.5), int(sub_tomo.shape[1]*.5), int(sub_tomo.shape[2]*.5)
    x, y, z = point[0], point[1], point[2]

    # Compute bounding restriction
    off_l_x, off_l_y, off_l_z = x-hl_x+1, y-hl_y+1, z-hl_z+1
    off_h_x, off_h_y, off_h_z = x+hl_x+1, y+hl_y+1, z+hl_z+1
    dif_l_x, dif_l_y, dif_l_z = 0, 0, 0
    dif_h_x, dif_h_y, dif_h_z = nx, ny, nz
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
    # Update tomo data
    hold_sv = tomo[off_l_x:off_h_x, off_l_y:off_h_y, off_l_z:off_h_z]
    hold_sub = sub_tomo[dif_l_x:dif_h_x, dif_l_y:dif_h_y, dif_l_z:dif_h_z]
    hold_sv = op(hold_sv, hold_sub)
    tomo[off_l_x:off_h_x, off_l_y:off_h_y, off_l_z:off_h_z] = hold_sv

# Converts a set of sparse coordinates (only 3D) into dense binary array within a mask
# coords: input coordinates array
# mask: mask, coordinates out of the mask will be deleted
# rots: (default None) additional array with equal length to coords for beigs also purged
# Returns: the dense array (False->bg, True->sample), and coords with samples out mask cleaned,
#          if an array is passed as rots then this wil be cleaned like coords
def coords_to_dense_mask(coords, mask, rots=None):

    # Initialize the dense array
    dense = np.zeros(shape=mask.shape, dtype=np.bool)

    # Set samples in the dense array
    out_coords = list()
    if rots is None:
        try:
            for coord in coords:
                x, y, z = int(round(coord[0])), int(round(coord[1])), int(round(coord[2]))
                try:
                    if mask[x, y, z]:
                        dense[x, y, z] = True
                        out_coords.append(np.asarray((x, y, z), dtype=np.float))
                except IndexError:
                    pass
        except TypeError:
            pass

        return dense, np.asarray(out_coords, dtype=np.float)

    else:
        out_rots = list()
        try:
            for coord, rot in zip(coords, rots):
                x, y, z = int(round(coord[0])), int(round(coord[1])), int(round(coord[2]))
                try:
                    if mask[x, y, z]:
                        dense[x, y, z] = True
                        out_coords.append(np.asarray((x, y, z), dtype=np.float))
                        out_rots.append(np.asarray(rot, dtype=np.float))
                except IndexError:
                    pass
        except TypeError:
            pass

        return dense, np.asarray(out_coords, dtype=np.float), np.asarray(out_rots, dtype=np.float)

# Computes Nearest Neighbour Distance of a cloud of points in a Euclidean space
# cloud: array with point coordinates
def nnde(cloud):
    dists = np.zeros(shape=cloud.shape[0], dtype=np.float)
    for i in range(len(dists)):
        hold = cloud[i] - cloud
        hold = np.sum(hold*hold, axis=1)
        hold[i] = np.inf
        dists[i] = math.sqrt(np.min(hold))
    return dists

# Computes the Crossed Nearest Neighbour Distance of a cloud of points to another
# in a Euclidean space
# cloud: array with point coordinates
# cloud_ref: reference array with point coordinates
def cnnde(cloud, cloud_ref):
    dists = np.zeros(shape=cloud.shape[0], dtype=np.float)
    for i in range(len(dists)):
        hold = cloud[i] - cloud_ref
        hold = np.sum(hold*hold, axis=1)
        dists[i] = math.sqrt(np.min(hold))
    return dists

# Compute Histogram Function from a one-dimensional array of random samples
# var: array of stochastic samples
# n: number of samples for cdf, if n is a sequence it defines the bin edges, including rightmost edge
# mx: maximum value for the histogram
# Returns: cdf values and samples respectively
def compute_hist(var, n, mx):
    bins = np.linspace(0, mx, n+1)
    hist, _ = np.histogram(var, bins=bins, density=True)
    hist[np.isnan(hist)] = 0
    return .5*bins[1] + bins[:-1], hist

# Compute Cumulative Density Function from a one-dimensional array of random samples
# var: array of stochastic samples
# n: number of samples for cdf, if n is a sequence it defines the bin edges, including rightmost edge
# mx: maximum value for the histogram
# Returns: cdf values and samples respectively
def compute_cdf(var, n, mx):
    bins = np.linspace(0, mx, n+1)
    hist, _ = np.histogram(var, bins=bins, density=True)
    hist[np.isnan(hist)] = 0
    # Compute CDF, last value is discarded because its unaccuracy and first one is set to zero
    hold_cum = np.cumsum(hist)*bins[1]
    return .5*bins[1] + bins[:-1], hold_cum

# Computes the envelope of a stochastic function
# funcs: matrix where rows every independent simulation
# per: percentile for the envelope, default is 50 (median)
# axis: Axis or axes along which the percentiles are computed (default 1).
def func_envelope(funcs, per=50,  axis=1):
    return np.percentile(funcs, per, axis=axis)

# Generates a sinusoidal random pattern in a tomogram
# n: number of points
# cycles: tuple (size 3) with the number of cycles for the sinusoidal in each dimension (X, Y, Z)
# std: standard deviation for adding Gaussian noise
# phase: 3-Tuple for phase shifting (default (0, 0, 0)) in radians
# Returns: and array with size [n, 3] with coordinates within the mask
def gen_sin_points(n, cycles, mask, std=0, phase=(0, 0, 0)):

    # Building probability field
    X, Y, Z = np.meshgrid(np.linspace(-1,1,mask.shape[1]), np.linspace(-1,1,mask.shape[0]),
                          np.linspace(-1,1,mask.shape[2]))
    sin = np.sin(cycles[0]*np.pi*X+phase[0]) + \
          np.sin(cycles[1]*np.pi*Y+phase[1]) + \
          np.sin(cycles[2]*np.pi*Z+phase[2])
    if std > 0:
        sin += np.random.normal(0, std, mask.shape)

    # Points are the n greatest field samples
    coords = list()
    ids = np.argsort(sin, axis=None)
    for idx in ids[-n:]:
        x, y, z = np.unravel_index(idx, mask.shape)
        coords.append(np.asarray((x, y, z), dtype=np.float))

    return np.asarray(coords)

def store_legend(handles, labels, out_file):
    plt.figlegend(handles, labels, loc='upper center', fontsize='xx-small')
    stem, ext = os.path.splitext(out_file)
    if os.path.splitext(out_file)[1] == '.pkl':
        pickle.dump(plt.figure(), open(out_file, 'wb'))
    else:
        plt.savefig(stem + '_legend' + ext)
    plt.close()

# Generates a completely random set of points within a mask
# n: number of points
# mask: binary mask
# temp: if None (default) then points are shapeless, otherwise it contains a 3D numpy binary array with particle template,
#       dimesion must be even
# Returns: and array with size [n, 3] with coordinates within the mask
def gen_rand_in_mask(n, mask, temp=None):

    # Input parsing
    if n <= 0:
        return None

    # Random generation loop
    ntries = 0
    mx_ntries = mask.size
    npts = 0
    coords = -1. * np.ones(shape=(n, 3), dtype=np.float32)
    if isinstance(temp, np.ndarray) and (len(temp.shape) == 3):

        quats = np.zeros(shape=(n, 4), dtype=np.float)
        hold_mask = np.copy(np.invert(mask))
        nx, ny, nz = temp.shape[0], temp.shape[1], temp.shape[2]
        mx, my, mz = mask.shape[0], mask.shape[1], mask.shape[2]
        mx1, my1, mz1 = mx-1, my-1, mz-1
        hl_x, hl_y, hl_z = int(temp.shape[1]*.5), int(temp.shape[0]*.5), int(temp.shape[2]*.5)
        while (npts < n) and (ntries < mx_ntries):
            # Coordinates
            x, y, z = random.randint(0, mask.shape[0]-1), random.randint(0, mask.shape[1]-1), \
                      random.randint(0, mask.shape[2]-1)
            # Template random rotation
            quat = rand_quat()
            mat = quat_to_rot(quat)
            eu_angs = rot_mat_eu_relion(mat)
            rot_temp = tomo_rot(temp, eu_angs, conv='relion', deg=True)
            # Compute bounding restriction
            off_l_x, off_l_y, off_l_z = x-hl_x+1, y-hl_y+1, z-hl_z+1
            off_h_x, off_h_y, off_h_z = x+hl_x+1, y+hl_y+1, z+hl_z+1
            dif_l_x, dif_l_y, dif_l_z = 0, 0, 0
            dif_h_x, dif_h_y, dif_h_z = nx, ny, nz
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
            # Update mask if no overlapping
            hold_sv = hold_mask[off_l_x:off_h_x, off_l_y:off_h_y, off_l_z:off_h_z]
            hold_rot = rot_temp[dif_l_x:dif_h_x, dif_l_y:dif_h_y, dif_l_z:dif_h_z]
            if hold_sv[hold_rot].sum() == 0:
                hold_sv[hold_rot] = True
                hold_mask[off_l_x:off_h_x, off_l_y:off_h_y, off_l_z:off_h_z] = hold_sv
                coords[npts, :] = (x, y, z)
                quats[npts, :] = quat
                npts += 1
            ntries += 1

        # print npts, ntries

        # pyseg.disperse_io.save_numpy(hold_mask, '/home/martinez/pool/pool-lucic2/antonio/workspace/stat/pyrenoid/3d_uni_temp/hold_'+str(cont)+'.mrc')

        if npts < n:
            print 'WARNING (gen_rand_in_mask): number of points simulated less than demanded: (' + \
                  str(npts) + ' of ' + str(n) + ')'

        return coords, quats

    else:
        while (npts < n) and (ntries < mx_ntries):
            x, y, z = random.randint(0, mask.shape[0]-1), random.randint(0, mask.shape[1]-1), \
                      random.randint(0, mask.shape[2]-1)
            if mask[x, y, z]:
                coords[npts, :] = (x, y, z)
                npts += 1
            ntries += 1
        return coords

# Pickle a matplolib figure but if an error ocurrs (it may depends on matplolib version) the figure lines arrays
# are pickled separatedly
def fig_pickle(fig, out_file):
    try:
        pickle.dump(fig, open(out_file, 'wb'))
    except TypeError:
        out_path, out_fname = os.path.split(out_file)
        out_stem, out_ext = os.path.splitext(out_fname)
        for i, ax in enumerate(fig.get_axes()):
            for j, line in enumerate(ax.get_lines()):
                pickle.dump(line.get_data()[0],
                            open(out_path+'/'+out_stem+'_ax'+str(i)+'_l'+str(j)+'_x'+out_ext, 'wb'))
                pickle.dump(line.get_data()[1],
                            open(out_path+'/'+out_stem+'_ax'+str(i)+'_l'+str(j)+'_y'+out_ext, 'wb'))

# So as to prevent non-sense results the arrays may be are cropped
# high: maximum allowed value for cropping the output array (default 0.95)
def compute_J(g, f, high=.95):
    mx_g, mx_f = (g < high).sum(), (f < high).sum()
    if mx_g < mx_f:
        return (1. - g[:mx_g]) / (1. - f[:mx_g])
    else:
        return (1. - g[:mx_f]) / (1. - f[:mx_f])

#######################################################################################################################
#
#   Funtions for Geodesic analysis by Fast Martching Method
#

# Computes signed distance in ndarray by Fast Marching method
# seed: input boolean ndarray where starting voxels are True
# mask: mask where False are non-valid regions
def fmm_distance(seed, mask):

    # Prepare input array
    phi = np.ones(seed.shape)
    phi[seed] = -1
    phi = np.ma.MaskedArray(phi, mask)

    # Fast Marching
    return np.ma.filled(skfmm.distance(phi), -1)

# Computes the number of points and volumes in un-regular
# nd-space for a list of geodesic distance ranges from a reference
# point
# ref_point: reference_points
# points: list of points
# dst_rgs: array [n, 2] with the distance ranges
# mask: binary array for space fg (True)
# Return: two arrays with the number of points and mask volumes
#               for every geodesice distance range
def geodesic_points_ndfold(ref_point, points, dst_rgs, mask):

    # Crop sub-volume
    max_dst, ref_point_i = int(math.ceil(dst_rgs.max())), np.round(ref_point).astype(np.int)
    sub_shape = [max_dst, max_dst, max_dst]
    svol, off = get_sub_copy(mask, ref_point_i, sub_shape, copy=True)
    tpoints = points - off
    tref_point_i = ref_point_i - off

    # Geodesic distance
    svol_points = np.zeros(shape=svol.shape, dtype=np.bool)
    svol_points[int(tref_point_i[0]), int(tref_point_i[1]), int(tref_point_i[2])] = True
    dmap = fmm_distance(svol_points, np.invert(svol))

    # Points loop
    for point in tpoints:
        pt = np.round(point).astype(np.int)
        if not ((pt[0] < 0) or (pt[1] < 1) or (pt[2] < 2) or (pt[0] >= svol.shape[0]) or (
            pt[1] >= svol.shape[1]) or (
                    pt[2] >= svol.shape[2])):
            svol_points[pt[0], pt[1], pt[2]] = True

    # Ranges loop
    vols = np.zeros(shape=len(dst_rgs), dtype=np.float32)
    npoints = np.zeros(shape=len(dst_rgs), dtype=np.float32)
    for i, dst_rg in enumerate(dst_rgs):
        hold_voi = (dmap >= dst_rg[0]) & (dmap <= dst_rg[1]) # & svol
        vols[i] = hold_voi.sum()
        npoints[i] = (svol_points & hold_voi).sum()

    return npoints, vols

######### Luis' stuff for close packing generation

def getHCPCoords(i, j, k, r):

    x = ( (2.0*i) + ((j + k)%2) )*r
    y = ( (3.0**0.5)*(j + ((1.0/3)*(k%2))) )*r
    z = ( (2.0*(6.0**0.5)/3)*k )*r

    return float(x), float(y), float(z)

def getCCPCoords(i, j, k, r):

    x = r*(2*i + j%2 + k%3)
    y = (j + float(k%3)/3)*(3**0.5)*r
    z = k*((6**0.5)/3)*2*r

    return float(x), float(y), float(z)

def generate_hcp_lattice(vol_x, vol_y, vol_z, sphere_r):

    f_sphere_r = float(sphere_r)
    rx = int(round(vol_x/sphere_r))
    ry = int(round(vol_y/sphere_r))
    rz = int(round(vol_z/sphere_r))

    point_list = []

    for i in range(rx):
        for j in range(ry):
            for k in range(rz):
                x, y, z = getHCPCoords(i, j, k, sphere_r)
                if (x<0) or (y<0) or (z<0):
                    continue
                if (x>vol_x-1) or (y>vol_y-1) or (z>vol_z-1):
                    continue

                point_list.append(np.asarray((x, y, z), dtype=np.float32))

    return point_list

def generate_ccp_lattice(vol_x, vol_y, vol_z, sphere_r):

    f_sphere_r = float(sphere_r)
    rx = int(round(vol_x/sphere_r))
    ry = int(round(vol_y/sphere_r))
    rz = int(round(vol_z/sphere_r))

    point_list = []

    for i in range(rx):
        for j in range(ry):
            for k in range(rz):

                x, y, z = getCCPCoords(i, j, k, sphere_r)
                if (x<0) or (y<0) or (z<0):
                    continue
                if (x>vol_x-1) or (y>vol_y-1) or (z>vol_z-1):
                    continue

                point_list.append(np.asarray((x-f_sphere_r, y-f_sphere_r, z-f_sphere_r),dtype=np.float32))

    return point_list


##############################################################################################
# Class for generating random coordinates in a masked tomogram
#
class RandCoordsMask(object):

    # mask_inv: inverted mask (True-bg and False-fg)
    def __init__(self, mask_inv):
        self.__shape = mask_inv.shape
        self.__ids = np.where(mask_inv == False)
        self.__init_tries = len(self.__ids[0])

    #### External functionality

    def get_initial_tries(self):
        return self.__init_tries

    # a_mask_inv: gives the possibility of an additional mask, also inverted
    # Returns: a random instance for tomogram coordinates in mask(s), or an Exception if the try failed
    def gen_rand_coords(self, a_mask_inv=None):
        idx = random.randint(0, self.__init_tries-1)
        x, y, z = self.__ids[0][idx], self.__ids[1][idx], self.__ids[2][idx]
        if (a_mask_inv is not None) and a_mask_inv[x, y, z]:
            raise Exception
        return x, y, z

    # Generates a neighbour, distance fixed, coordinates randomly
    # a_mask_inv: gives the possibility of an additional mask, also inverted
    # dst: euclideand distance to neighbour
    # sg: sigma variance for Gaussian random shifting of the distance
    # Returns: a random instance for tomogram coordinates in mask(s), or an Exception if the try failed
    def gen_neighbor_coords(self, dst, sg=0, a_mask_inv=None):
        idx = random.randint(0, self.__init_tries-1)
        x, y, z = self.__ids[0][idx], self.__ids[1][idx], self.__ids[2][idx]
        # Random neighbor rotations
        rho, tilt, _ = rot_mat_eu_relion(quat_to_rot(rand_quat()), deg=False)
        # Radius
        if sg <= 0.:
            rad = float(dst)
        else:
            rad = np.random.normal(dst, sg)
        # Neighbor coordinates
        xn, yn, zn = int(round(x+rad*math.sin(rho)*math.cos(tilt))), int(round(y+rad*math.sin(rho)*math.sin(tilt))), \
                     int(round(z+rad*math.cos(tilt)))
        if (a_mask_inv is not None) and a_mask_inv[xn, yn, zn]:
            raise Exception
        return xn, yn, zn

##############################################################################################
# Class for generating random coordinates where probabilites are set in a masked tomogram
#
class ChoiceCoordsMask(object):

    # p_map: potential map tomogram
    # l_cut: low cutoff
    # h_cut: high cutoff
    def __init__(self, p_map, l_cut, h_cut):
        self.__shape = p_map.shape
        mask = (p_map>l_cut) * (p_map<h_cut)
        hold = p_map[mask]
        self.__probs = hold / hold.sum()
        self.__ids = np.where(mask)
        self.__init_tries = len(self.__ids[0])
        self.__mx_rnd = self.__init_tries - 1

    #### External functionality

    def get_initial_tries(self):
        return self.__init_tries

    # a_mask_inv: gives the possibility of an additional mask, also inverted
    # Returns: a random instance for tomogram coordinates in mask(s), or an Exception if the try failed
    def gen_rand_coords(self, a_mask_inv=None):
        idx = np.random.choice(self.__mx_rnd, size=1, p=self.__probs)
        x, y, z = self.__ids[0][idx], self.__ids[1][idx], self.__ids[2][idx]
        return x, y, z

##############################################################################################
# Abstract class for simulators
#
class Simulator(object):

    # For Abstract Base Classes in python
    __metaclass__ = ABCMeta

    #### Set/Get methods

    #### External functionality area

    # Generates a set of simulated points within a mask
    # n: number of points
    # mask: binary mask
    # Returns: and array with size [n, 3] with coordinates within the mask
    @abstractmethod
    def gen_rand_in_mask(self, n, mask):
        raise NotImplementedError('gen_rand_in_mask() (Simulator). '
                                  'Abstract method, it requires an implementation.')

    # Generates a completely random tomogram
    # n: number of points
    # mask: binary mask
    # Returns: a tomogram with the simulated particles
    @abstractmethod
    def gen_rand_in_mask_tomo(self, n, mask):
        raise NotImplementedError('gen_rand_in_mask_tomo() (Simulator). '
                                  'Abstract method, it requires an implementation.')

##############################################################################################
# Class for CSR simulations of points
#
class CSRSimulator(Simulator):

    #### Constructor area
    def __init__(self):
        super(CSRSimulator, self).__init__()

    #### External functionality area


    def gen_rand_in_mask(self, n, mask):

        # Input parsing
        if n <= 0:
            return None
        hold_mask = np.invert(mask)
        generator = RandCoordsMask(hold_mask)

        # Random generation loop
        ntries = 0
        mx_ntries = generator.get_initial_tries()
        npts = 0
        coords = -1. * np.ones(shape=(n, 3), dtype=np.float32)
        while (npts < n) and (ntries < mx_ntries):
            # Coordinates
            try:
                x, y, z = generator.gen_rand_coords(hold_mask)
            except Exception:
                ntries += 1
                continue
            if mask[x, y, z]:
                coords[npts, :] = (x, y, z)
                npts += 1
            ntries += 1

        return coords

    def gen_rand_in_mask_tomo(self, n, mask):

        # Input parsing
        if n <= 0:
            return None
        hold_mask = np.invert(mask)
        generator = RandCoordsMask(hold_mask)

        # Random generation loop
        ntries = 0
        mx_ntries = generator.get_initial_tries()
        npts = 0
        hold_mask = np.copy(np.invert(mask))
        while (npts < n) and (ntries < mx_ntries):
            # Coordinates
            try:
                x, y, z = generator.gen_rand_coords(hold_mask)
            except Exception:
                ntries += 1
                continue
            if mask[x, y, z]:
                hold_mask[x, y, z] = True
                npts += 1
            ntries += 1

        return hold_mask

##############################################################################################
# Class for CSR simulations of particles with shape (template)
#
class CSRTSimulator(Simulator):

    #### Constructor area
    # temp: 3d binary numpy array with the template
    # nrots: number or random rotations (by default 1000)
    def __init__(self, temp, nrots=1000):
        super(CSRTSimulator, self).__init__()
        if (not isinstance(temp, np.ndarray)) or (len(temp.shape) != 3):
            error_msg = 'Input template must be a 3d numpy array'
            raise pexceptions.PySegInputError(expr='__init__ (CSRTSimulator)', msg=error_msg)
        self.__temp = np.asarray(temp, dtype=np.float32)
        self.__nrots = int(nrots)
        if self.__nrots <= 0:
            error_msg = 'The number or random rotation must be greater than zero!'
            raise pexceptions.PySegInputError(expr='__init__ (CSRTSimulator)', msg=error_msg)
        nx, ny, nz = self.__temp.shape
        self.__rnd_rots = np.zeros(shape=(nx, ny, nz, self.__nrots), dtype=np.bool)
        self.__rnd_count = 0
        self.__gen_rnd_rots()

    #### External functionality area

    def get_rnd_rots_array(self):
        return self.__rnd_rots

    def gen_rand_in_mask(self, n, mask):

        # Input parsing
        if n <= 0:
            return None
        hold_mask = np.invert(mask)
        generator = RandCoordsMask(hold_mask)

        # Random generation loop
        ntries = 0
        mx_ntries = generator.get_initial_tries()
        npts = 0
        coords = -1. * np.ones(shape=(n, 3), dtype=np.float32)

        nx, ny, nz = self.__temp.shape[0], self.__temp.shape[1], self.__temp.shape[2]
        mx, my, mz = mask.shape[0], mask.shape[1], mask.shape[2]
        mx1, my1, mz1 = mx-1, my-1, mz-1
        hl_x, hl_y, hl_z = int(self.__temp.shape[1]*.5), int(self.__temp.shape[0]*.5), int(self.__temp.shape[2]*.5)
        while (npts < n) and (ntries < mx_ntries):
            # Coordinates
            try:
                x, y, z = generator.gen_rand_coords(hold_mask)
            except Exception:
                ntries += 1
                continue
            # Template random rotation
            rot_temp = self.__get_rnd_rot()
            # Compute bounding restriction
            off_l_x, off_l_y, off_l_z = x-hl_x+1, y-hl_y+1, z-hl_z+1
            off_h_x, off_h_y, off_h_z = x+hl_x+1, y+hl_y+1, z+hl_z+1
            dif_l_x, dif_l_y, dif_l_z = 0, 0, 0
            dif_h_x, dif_h_y, dif_h_z = nx, ny, nz
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
            # Update mask if no overlapping
            hold_sv = hold_mask[off_l_x:off_h_x, off_l_y:off_h_y, off_l_z:off_h_z]
            hold_rot = rot_temp[dif_l_x:dif_h_x, dif_l_y:dif_h_y, dif_l_z:dif_h_z]
            if hold_sv[hold_rot].sum() == 0:
                hold_sv[hold_rot] = True
                hold_mask[off_l_x:off_h_x, off_l_y:off_h_y, off_l_z:off_h_z] = hold_sv
                coords[npts, :] = (x, y, z)
                npts += 1
            ntries += 1

        # print npts, ntries

        # pyseg.disperse_io.save_numpy(hold_mask, '/home/martinez/workspace/stat/pyrenoid/test_linker/hold.mrc')

        if npts < n:
            print 'WARNING (gen_rand_in_mask): number of points simulated less than demanded: (' + \
                  str(npts) + ' of ' + str(n) + ')'

        return coords

    def gen_rand_in_mask_tomo(self, n, mask):

        # Input parsing
        if n <= 0:
            return None
        hold_mask = np.invert(mask)
        generator = RandCoordsMask(hold_mask)

        # Random generation loop
        ntries = 0
        mx_ntries = generator.get_initial_tries()
        npts = 0
        hold_mask = np.invert(mask)

        nx, ny, nz = self.__temp.shape[0], self.__temp.shape[1], self.__temp.shape[2]
        mx, my, mz = mask.shape[0], mask.shape[1], mask.shape[2]
        mx1, my1, mz1 = mx-1, my-1, mz-1
        hl_x, hl_y, hl_z = int(self.__temp.shape[1]*.5), int(self.__temp.shape[0]*.5), int(self.__temp.shape[2]*.5)
        while (npts < n) and (ntries < mx_ntries):
            # Coordinates
            try:
                x, y, z = generator.gen_rand_coords(hold_mask)
            except Exception:
                ntries += 1
                continue
            # Template random rotation
            rot_temp = self.__get_rnd_rot()
            # Compute bounding restriction
            off_l_x, off_l_y, off_l_z = x-hl_x+1, y-hl_y+1, z-hl_z+1
            off_h_x, off_h_y, off_h_z = x+hl_x+1, y+hl_y+1, z+hl_z+1
            dif_l_x, dif_l_y, dif_l_z = 0, 0, 0
            dif_h_x, dif_h_y, dif_h_z = nx, ny, nz
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
            # Update mask if no overlapping
            hold_sv = hold_mask[off_l_x:off_h_x, off_l_y:off_h_y, off_l_z:off_h_z]
            hold_rot = rot_temp[dif_l_x:dif_h_x, dif_l_y:dif_h_y, dif_l_z:dif_h_z]
            # pyseg.disperse_io.save_numpy(hold_rot, '/home/martinez/workspace/stat/pyrenoid/test_linker/hold.mrc')
            if hold_sv[hold_rot].sum() == 0:
                hold_sv[hold_rot] = True
                hold_mask[off_l_x:off_h_x, off_l_y:off_h_y, off_l_z:off_h_z] = hold_sv
                npts += 1
            ntries += 1

        if npts < n:
            print 'WARNING (gen_rand_in_mask): number of points simulated less than demanded: (' + \
                  str(npts) + ' of ' + str(n) + ')'

        return hold_mask

    #### Internal functionality area

    def __get_rnd_rot(self):
        hold = self.__rnd_rots[:, :, :, self.__rnd_count]
        self.__rnd_count += 1
        if self.__rnd_count >= self.__nrots:
            self.__rnd_count = 0
        return hold


    def __gen_rnd_rots(self):
        for i in range(self.__nrots):
            quat = rand_quat()
            mat = quat_to_rot(quat)
            eu_angs = rot_mat_eu_relion(mat)
            self.__rnd_rots[:, :, :, i] = (tomo_rot(self.__temp, eu_angs, conv='relion', deg=True) > ROT_TMP_TH)

##############################################################################################
# Class for Linker model simulations
#
class LinkerSimulator(Simulator):

    #### Constructor area
    # leng: linker length (in voxels)
    # nc: number of copies of the reference template
    # temp: 3d binary numpy array with the template
    # nrots: number or random rotations (by default 1000)
    # no_over: if True (default False) linked particles can overlap
    # len_sg: sigma for normal random distribution of the linker lengths in nm
    def __init__(self, leng, nc, temp, nrots=1000, no_over=False, len_sg=0.):
        super(LinkerSimulator, self).__init__()
        self.__leng = float(leng)
        self.__nc = int(nc)
        self.__len_sg = float(len_sg)
        if (not isinstance(temp, np.ndarray)) or (len(temp.shape) != 3):
            error_msg = 'Particle distance must be a 3d numpy array!'
            raise pexceptions.PySegInputError(expr='__init__ (LinkerSimulator)', msg=error_msg)
        self.__temp = np.asarray(temp, dtype=np.float32)
        self.__nrots = int(nrots)
        if self.__nrots <= 0:
            error_msg = 'The number or random rotation must be greater than zero!'
            raise pexceptions.PySegInputError(expr='__init__ (LinkerSimulator)', msg=error_msg)
        self.__no_over = no_over
        self.__rnd_rots = None
        self.__l_temp = None
        self.__rnd_sx = None
        self.__rnd_sy = None
        self.__rnd_sz = None
        self.__rnd_count = 0
        self.__gen_rnd_rots()

    #### External functionality area

    def get_rnd_rots_array(self):
        return self.__rnd_rots

    def gen_rand_in_mask(self, n, mask):

        # Input parsing
        if n <= 0:
            return None
        hold_mask = np.invert(mask)
        generator = RandCoordsMask(hold_mask)

        # Random generation loop
        ntries = 0
        mx_ntries = generator.get_initial_tries()
        npts = 0
        coords = -1. * np.ones(shape=(n, 3), dtype=np.float32)

        hold_mask = np.copy(np.invert(mask))
        nx, ny, nz = self.__l_temp.shape[0], self.__l_temp.shape[1], self.__l_temp.shape[2]
        mx, my, mz = mask.shape[0], mask.shape[1], mask.shape[2]
        mx1, my1, mz1 = mx-1, my-1, mz-1
        hl_x, hl_y, hl_z = int(self.__l_temp.shape[1]*.5), int(self.__l_temp.shape[0]*.5), \
                           int(self.__l_temp.shape[2]*.5)
        while (npts < n) and (ntries < mx_ntries):
            # Coordinates
            try:
                x, y, z = generator.gen_rand_coords(hold_mask)
            except Exception:
                ntries += 1
                continue
            # Template random rotation
            rot_temp, sx_l, sy_l, sz_l = self.__get_rnd_rot()
            # Compute bounding restriction
            off_l_x, off_l_y, off_l_z = x-hl_x+1, y-hl_y+1, z-hl_z+1
            off_h_x, off_h_y, off_h_z = x+hl_x+1, y+hl_y+1, z+hl_z+1
            dif_l_x, dif_l_y, dif_l_z = 0, 0, 0
            dif_h_x, dif_h_y, dif_h_z = nx, ny, nz
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
            # Update mask if no overlapping
            hold_sv = hold_mask[off_l_x:off_h_x, off_l_y:off_h_y, off_l_z:off_h_z]
            hold_rot = rot_temp[dif_l_x:dif_h_x, dif_l_y:dif_h_y, dif_l_z:dif_h_z]
            if hold_sv[hold_rot].sum() == 0:
                hold_sv[hold_rot] = True
                hold_mask[off_l_x:off_h_x, off_l_y:off_h_y, off_l_z:off_h_z] = hold_sv
                for (sx, sy, sz) in zip(sx_l, sy_l, sz_l):
                    try:
                        coords[npts, :] = (x+sx, y+sy, z+sz)
                    except IndexError:
                        if npts < n:
                            raise Exception
                        break
                    npts += 1

                # pyseg.disperse_io.save_numpy(self.__l_temp, '/home/martinez/workspace/stat/pyrenoid/test_linker/temp.mrc')
                # pyseg.disperse_io.save_numpy(rot_temp, '/home/martinez/workspace/stat/pyrenoid/test_linker/temp_rot.mrc')

            ntries += 1

        # print npts, ntries
        # pyseg.disperse_io.save_numpy(hold_mask, '/home/martinez/workspace/stat/pyrenoid/test_linker/hold.mrc')

        if npts < n:
            print 'WARNING (gen_rand_in_mask_tomo): number of points simulated less than demanded: (' + \
                  str(npts) + ' of ' + str(n) + ')'

        return coords

    def gen_rand_in_mask_tomo(self, n, mask):

        # Input parsing
        if n <= 0:
            return None
        hold_mask = np.invert(mask)
        generator = RandCoordsMask(hold_mask)

        # Random generation loop
        ntries = 0
        mx_ntries = generator.get_initial_tries()
        npts = 0

        nx, ny, nz = self.__l_temp.shape[0], self.__l_temp.shape[1], self.__l_temp.shape[2]
        mx, my, mz = mask.shape[0], mask.shape[1], mask.shape[2]
        mx1, my1, mz1 = mx-1, my-1, mz-1
        hl_x, hl_y, hl_z = int(self.__l_temp.shape[1]*.5), int(self.__l_temp.shape[0]*.5), \
                           int(self.__l_temp.shape[2]*.5)
        while (npts < n) and (ntries < mx_ntries):
            # Coordinates
            try:
               x, y, z = generator.gen_rand_coords(hold_mask)
            except Exception:
                ntries += 1
                continue
            # Template random rotation
            rot_temp, _, _, _ = self.__get_rnd_rot()
            # Compute bounding restriction
            off_l_x, off_l_y, off_l_z = x-hl_x+1, y-hl_y+1, z-hl_z+1
            off_h_x, off_h_y, off_h_z = x+hl_x+1, y+hl_y+1, z+hl_z+1
            dif_l_x, dif_l_y, dif_l_z = 0, 0, 0
            dif_h_x, dif_h_y, dif_h_z = nx, ny, nz
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
            # Update mask if no overlapping
            hold_sv = hold_mask[off_l_x:off_h_x, off_l_y:off_h_y, off_l_z:off_h_z]
            hold_rot = rot_temp[dif_l_x:dif_h_x, dif_l_y:dif_h_y, dif_l_z:dif_h_z]
            if hold_sv[hold_rot].sum() == 0:
                hold_sv[hold_rot] = True
                hold_mask[off_l_x:off_h_x, off_l_y:off_h_y, off_l_z:off_h_z] = hold_sv
                npts += self.__nc
            ntries += 1

        # print npts, ntries
        # pyseg.disperse_io.save_numpy(hold_mask, '/home/martinez/pool/pool-lucic2/antonio/workspace/stat/pyrenoid/3d_uni_temp/hold_'+str(cont)+'.mrc')

        if npts < n:
            print 'WARNING (gen_rand_in_mask_tomo): number of points simulated less than demanded: (' + \
                  str(npts) + ' of ' + str(n) + ')'

        return hold_mask


    #### Internal functionality area

    # Generates a template for Linker model from an input reference template
    # Makes several copies of the input template an places them into a circle
    # temp: template volume, 3D numpy binary array, if None (default) points are considered
    def __gen_linker_temp(self):

        # Compute output dimensions
        if self.__temp is None:
            md = math.ceil(.5*(self.__leng + 2.))
        else:
            md = math.ceil(.5*(self.__leng + np.asarray(self.__temp.shape).max()))
        if (md%2) != 0:
            md += 1
        ld = 2 * md
        if self.__len_sg <= 0.:
            rad = .5 * self.__leng
        else:
            rad = .5 * np.random.normal(self.__leng, self.__len_sg)
        linker_temp = np.zeros(shape=(ld, ld, ld), dtype=np.bool)
        sx_l, sy_l, sz_l = list(), list(), list()

        # Equally spaced circle sampling
        rhos = np.linspace(0, np.pi, self.__nc)
        shifts_x, shifts_y = rad*np.cos(rhos), rad*np.sin(rhos)

        # Insert reference templates
        hl_x, hl_y, hl_z = int(self.__temp.shape[1]*.5), int(self.__temp.shape[0]*.5), \
                            int(self.__temp.shape[2]*.5)
        # Linker random rotation
        quat_l = rand_quat()
        mat_l = quat_to_rot(quat_l)
        for sx, sy in zip(shifts_x, shifts_y):
            sz = 0.
            # Rotate shifts
            v = np.mat((sx, sy, sz))
            vr = mat_l * v.transpose()
            x, y, z = int(round(md+vr[0]-.5)), int(round(md+vr[1]-.5)), int(round(md+vr[2]-.5))
            off_l_x, off_l_y, off_l_z = x-hl_x+1, y-hl_y+1, z-hl_z+1
            off_h_x, off_h_y, off_h_z = x+hl_x+1, y+hl_y+1, z+hl_z+1
            hold_linker = np.zeros(shape=(ld, ld, ld), dtype=np.bool)
            # Random template rotation
            quat = rand_quat()
            mat = quat_to_rot(quat)
            eu_angs = rot_mat_eu_relion(mat)
            rot_temp = tomo_rot(self.__temp, eu_angs, conv='relion', deg=True)
            hold_linker[off_l_x:off_h_x, off_l_y:off_h_y, off_l_z:off_h_z] = (rot_temp>ROT_TMP_TH)
            tmp_sum = linker_temp[hold_linker].sum()
            if tmp_sum > 0:
                if not self.__no_over:
                    # print 'WARNING: overlapping templates, sum = ' + str(tmp_sum)
                    raise Exception
            linker_temp[hold_linker] = True
            sx_l.append(vr[0])
            sy_l.append(vr[1])
            sz_l.append(vr[2])

            # pyseg.disperse_io.save_numpy(rot_temp, '/home/martinez/workspace/stat/pyrenoid/test_linker/hold_t.mrc')
        # pyseg.disperse_io.save_numpy(linker_temp, '/home/martinez/workspace/stat/pyrenoid/test_linker/hold_tl.mrc')

        return linker_temp, sx_l, sy_l, sz_l

    def __get_rnd_rot(self):
        hold = self.__rnd_rots[:, :, :, self.__rnd_count]
        hold_x = self.__rnd_sx[self.__rnd_count]
        hold_y = self.__rnd_sy[self.__rnd_count]
        hold_z = self.__rnd_sz[self.__rnd_count]
        self.__rnd_count += 1
        if self.__rnd_count >= self.__nrots:
            self.__rnd_count = 0
        return hold, hold_x, hold_y, hold_z


    def __gen_rnd_rots(self):
        ntries = 10 * self.__nrots
        n = 0
        for i in range(ntries):
            n += 1
            try:
                self.__l_temp, _, _, _ = self.__gen_linker_temp()
            except Exception:
                pass
            if self.__l_temp is not None:
                break
        if n >= ntries:
            print 'WARNING: overlapping templates after ' + str(n) + ' tries!!!'
            raise Exception
        nx, ny, nz = self.__l_temp.shape
        self.__rnd_rots = np.zeros(shape=(nx, ny, nz, self.__nrots), dtype=np.bool)
        self.__rnd_sx = np.zeros(shape=self.__nrots, dtype=object)
        self.__rnd_sy = np.zeros(shape=self.__nrots, dtype=object)
        self.__rnd_sz = np.zeros(shape=self.__nrots, dtype=object)
        n = 0
        nt = 0
        while (nt < self.__nrots) and (n < ntries):
            quat = rand_quat()
            mat = quat_to_rot(quat)
            eu_angs = rot_mat_eu_relion(mat)
            try:
                hold_temp, sx_c, sy_c, sz_c = self.__gen_linker_temp()
                self.__rnd_rots[:, :, :, nt] = hold_temp
                self.__rnd_sx[nt], self.__rnd_sy[nt], self.__rnd_sz[nt], = sx_c, sy_c, sz_c
                nt += 1
            except Exception:
                pass
            n += 1
        if n >= ntries:
            raise Exception


##############################################################################################
# Class for Rubisco's crystal simulator
#
class RubCSimulation(Simulator):

    #### Constructor area
    # res: pixel/nm resolution
    # temp: 3d binary numpy array with the template
    def __init__(self, res, temp):
        super(RubCSimulation, self).__init__()
        if res <= 0:
            error_msg = 'Resolution must be greater than zero!'
            raise pexceptions.PySegInputError(expr='__init__ (SinSimulator)', msg=error_msg)
        self.__res = res
        if (not isinstance(temp, np.ndarray)) or (len(temp.shape) != 3):
            error_msg = 'Particle distance must be a 3d numpy array!'
            raise pexceptions.PySegInputError(expr='__init__ (LinkerSimulator)', msg=error_msg)
        self.__temp = temp > 0
        self.__cell = None
        self.__pos_a, self.__pos_b, self.__pos_c, self.__pos_d = None, None, None, None
        self.__sx, self.__sy, self.__sz = None, None, None
        self.__gen_unit_temp()


    ###### External functionality area

    def gen_rand_in_mask(self, n, mask):

        # Input parsing
        if n <= 0:
            return None
        poss = (self.__pos_a, self.__pos_b, self.__pos_c, self.__pos_d)
        coords = list()
        hold_mask = np.invert(mask)
        # hold_tomo = np.copy(hold_mask)

        # Computing unit cell shiftings
        nx, ny, nz = self.__temp.shape[0], self.__temp.shape[1], self.__temp.shape[2]
        mx, my, mz = mask.shape[0], mask.shape[1], mask.shape[2]
        mx1, my1, mz1 = mx-1, my-1, mz-1
        hl_x, hl_y, hl_z = int(self.__temp.shape[1]*.5), int(self.__temp.shape[0]*.5), \
                           int(self.__temp.shape[2]*.5)
        xs = np.arange(0, mx, self.__sx)
        ys = np.arange(0, my, self.__sy)
        zs = np.arange(0, mz, self.__sz)

        # Tomogram loop
        for xc in xs:
            for yc in ys:
                for zc in zs:

                    # Loop for Rubisco's in a unit cell
                    for pos in poss:

                        # Rubiscos's coordinates
                        x, y, z = int(round(xc+pos[0])), int(round(yc+pos[1])), int(round(zc+pos[2]))
                        try:
                            if hold_mask[x, y, z]:
                                continue
                        except IndexError:
                            continue

                        # Compute bounding restriction
                        off_l_x, off_l_y, off_l_z = x-hl_x+1, y-hl_y+1, z-hl_z+1
                        off_h_x, off_h_y, off_h_z = x+hl_x+1, y+hl_y+1, z+hl_z+1
                        dif_l_x, dif_l_y, dif_l_z = 0, 0, 0
                        dif_h_x, dif_h_y, dif_h_z = nx, ny, nz
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
                        # Check no mask overlapping
                        hold_sv = hold_mask[off_l_x:off_h_x, off_l_y:off_h_y, off_l_z:off_h_z]
                        hold_temp = self.__temp[dif_l_x:dif_h_x, dif_l_y:dif_h_y, dif_l_z:dif_h_z]
                        if hold_sv[hold_temp].sum() == 0:
                            coords.append(np.asarray((x, y, z), dtype=np.float32))

                            # hold_sv[hold_temp] = True
                            # hold_tomo[off_l_x:off_h_x, off_l_y:off_h_y, off_l_z:off_h_z] = hold_sv

        # pyseg.disperse_io.save_numpy(hold_tomo, '/home/martinez/workspace/stat/pyrenoid/test_linker/hold.mrc')

        return np.asarray(coords, dtype=np.float32)

    def gen_rand_in_mask_tomo(self, n, mask):

        # Input parsing
        if n <= 0:
            return None
        poss = (self.__pos_a, self.__pos_b, self.__pos_c, self.__pos_d)
        hold_mask = np.invert(mask)
        hold_tomo = np.copy(hold_mask)

        # Computing unit cell shiftings
        nx, ny, nz = self.__temp.shape[0], self.__temp.shape[1], self.__temp.shape[2]
        mx, my, mz = mask.shape[0], mask.shape[1], mask.shape[2]
        mx1, my1, mz1 = mx-1, my-1, mz-1
        hl_x, hl_y, hl_z = int(self.__temp.shape[1]*.5), int(self.__temp.shape[0]*.5), \
                           int(self.__temp.shape[2]*.5)
        xs = np.arange(0, mx, self.__sx)
        ys = np.arange(0, my, self.__sy)
        zs = np.arange(0, mz, self.__sz)

        # Tomogram loop
        for xc in xs:
            for yc in ys:
                for zc in zs:

                    # Loop for Rubico's in a unit cell
                    for pos in poss:

                        # Rubiscos's coordinates
                        x, y, z = int(round(xc+pos[0])), int(round(yc+pos[1])), int(round(zc+pos[2]))
                        try:
                            if hold_mask[x, y, z]:
                                continue
                        except IndexError:
                            continue

                        # Compute bounding restriction
                        off_l_x, off_l_y, off_l_z = x-hl_x+1, y-hl_y+1, z-hl_z+1
                        off_h_x, off_h_y, off_h_z = x+hl_x+1, y+hl_y+1, z+hl_z+1
                        dif_l_x, dif_l_y, dif_l_z = 0, 0, 0
                        dif_h_x, dif_h_y, dif_h_z = nx, ny, nz
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
                        # Check no mask overlapping
                        hold_sv = hold_mask[off_l_x:off_h_x, off_l_y:off_h_y, off_l_z:off_h_z]
                        hold_temp = self.__temp[dif_l_x:dif_h_x, dif_l_y:dif_h_y, dif_l_z:dif_h_z]
                        if hold_sv[hold_temp].sum() == 0:
                            hold_sv[hold_temp] = True
                            hold_tomo[off_l_x:off_h_x, off_l_y:off_h_y, off_l_z:off_h_z] = hold_sv

                            # pyseg.disperse_io.save_numpy(hold_temp, '/home/martinez/workspace/stat/pyrenoid/test_linker/hold_t.mrc')

        return hold_tomo

    #### Internal functionality area

    # Generates a template information for the unit cell of Rubisco's crystal
    def __gen_unit_temp(self):

        # Cell unit reference data
        ref_res = 0.1368 # nm/pixel
        pos_a = np.asarray((86, 136, 150), dtype=np.float) * ref_res # nm
        pos_b = np.asarray((137, 136, 73), dtype=np.float) * ref_res
        pos_c = np.asarray((78, 85, 73), dtype=np.float) * ref_res
        pos_d = np.asarray((148, 85, 148), dtype=np.float) * ref_res
        sx = 17.1375 # nm
        sy = 14.2638
        sz = 17.3212

        # Find dimension at host resolution
        self.__pos_a = pos_a / self.__res # pixels
        self.__pos_b = pos_b / self.__res
        self.__pos_c = pos_c / self.__res
        self.__pos_d = pos_d / self.__res
        self.__sx = sx / self.__res
        self.__sy = sy / self.__res
        self.__sz = sz / self.__res

##############################################################################################
# Class for Close Packing crystal simulator
#
class CPackingSimulator(Simulator):

    #### Constructor area
    # res: pixel/nm resolution
    # sph_rad: sphere radius in nm
    # temp: 3d binary numpy array with the template
    # packing: packing mode, valid: 1 cubic close packing (default), otherwise hexagonal close packing
    def __init__(self, res, sph_rad, temp, packing=1):
        super(CPackingSimulator, self).__init__()
        if res <= 0:
            error_msg = 'Resolution must be greater than zero!'
            raise pexceptions.PySegInputError(expr='__init__ (SinSimulator)', msg=error_msg)
        self.__res = float(res)
        if (not isinstance(temp, np.ndarray)) or (len(temp.shape) != 3):
            error_msg = 'Particle distance must be a 3d numpy array!'
            raise pexceptions.PySegInputError(expr='__init__ (LinkerSimulator)', msg=error_msg)
        self.__temp = temp > 0
        self.__sph_rad = float(sph_rad)
        self.__pack = int(packing)

    ###### External functionality area

    def gen_rand_in_mask(self, n, mask):

        # Input parsing
        if n <= 0:
            return None
        sph_rad = self.__sph_rad / self.__res # voxels
        coords = list()

        if self.__pack == 1:
            coords_l = generate_ccp_lattice(mask.shape[0], mask.shape[1], mask.shape[2], sph_rad)
        else:
            coords_l = generate_hcp_lattice(mask.shape[0], mask.shape[1], mask.shape[2], sph_rad)

        # Deleting coordinates out of the mask
        for coord in coords_l:
            x, y, z = int(round(coord[0])), int(round(coord[1])), int(round(coord[2]))
            try:
                if mask[x, y, z]:
                    coords.append(np.asarray((x, y, z), dtype=np.float32))
            except IndexError:
                pass

        return np.asarray(coords, dtype=np.float32)

    def gen_rand_in_mask_tomo(self, n, mask):

        # Input parsing
        if n <= 0:
            return None
        sph_rad = self.__sph_rad / self.__res # voxels
        coords = list()
        hold_mask = np.invert(mask)
        hold_tomo = np.copy(hold_mask)

        # Computing unit cell shiftings
        nx, ny, nz = self.__temp.shape[0], self.__temp.shape[1], self.__temp.shape[2]
        mx, my, mz = mask.shape[0], mask.shape[1], mask.shape[2]
        mx1, my1, mz1 = mx-1, my-1, mz-1
        hl_x, hl_y, hl_z = int(self.__temp.shape[1]*.5), int(self.__temp.shape[0]*.5), \
                           int(self.__temp.shape[2]*.5)

        if self.__pack == 1:
            coords_l = generate_ccp_lattice(mask.shape[0], mask.shape[1], mask.shape[2], sph_rad)
        else:
            coords_l = generate_hcp_lattice(mask.shape[0], mask.shape[1], mask.shape[2], sph_rad)


        # Deleting coordinates out of the mask
        for coord in coords_l:

            x, y, z = int(round(coord[0])), int(round(coord[1])), int(round(coord[2]))

            try:

                if mask[x, y, z]:
                    coords.append(np.asarray((x, y, z), dtype=np.float32))

                    # Inserting a copy of the template in the tomogram
                    # Compute bounding restriction
                    off_l_x, off_l_y, off_l_z = x-hl_x+1, y-hl_y+1, z-hl_z+1
                    off_h_x, off_h_y, off_h_z = x+hl_x+1, y+hl_y+1, z+hl_z+1
                    dif_l_x, dif_l_y, dif_l_z = 0, 0, 0
                    dif_h_x, dif_h_y, dif_h_z = nx, ny, nz
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
                    # Check no mask overlapping
                    hold_sv = hold_mask[off_l_x:off_h_x, off_l_y:off_h_y, off_l_z:off_h_z]
                    hold_temp = self.__temp[dif_l_x:dif_h_x, dif_l_y:dif_h_y, dif_l_z:dif_h_z]
                    if hold_sv[hold_temp].sum() == 0:
                        hold_sv[hold_temp] = True
                        hold_tomo[off_l_x:off_h_x, off_l_y:off_h_y, off_l_z:off_h_z] = hold_sv

            except IndexError:
                pass


        return hold_tomo

######################################################################################################
# Class which represent a node in a protein network
#
class Node(object):

    # coords: protein coordinates
    # mx_nlikers: Maximum number of linkers
    # mx_ntries: Maximum number of tries
    def __init__(self, coords, mx_nlinkers, mx_ntries):
        self.__coords = np.asarray(coords, dtype=np.float32)
        self.__nlinkers = 0
        self.__ntries = 0
        self.__max_nlinkers = int(mx_nlinkers)
        self.__max_ntries = int(mx_ntries)

    def get_coords(self):
        return self.__coords

    def add_linker(self):
        if (self.__nlinkers >= self.__max_nlinkers) or (self.__ntries >= self.__max_ntries):
            raise ValueError
        else:
            self.__nlinkers += 1
            self.__ntries += 1

    def add_try(self):
        if (self.__nlinkers >= self.__max_nlinkers) or (self.__ntries >= self.__max_ntries):
            raise ValueError
        else:
            self.__ntries += 1

    def is_full(self):
        if (self.__nlinkers >= self.__max_nlinkers) or (self.__ntries >= self.__max_ntries):
            return True
        else:
            return False

######################################################################################################
# Class with for simulating particles agglometravie model
#   - Particle are grouped in groups with a maximum number of particles
#   - These groups are distributed randomly
#   - Ever new particle in a group is added pseudo-randomly close to already existing particles
#   - Trees grow with branches of size n particles
#   - Simulator ends if al demanded particles are placed or maximum number of tries set is reached
class AggSimulator(object):

    # Constructor area
    # res: resolution in nm/pixel
    # temp: particle template
    # l_len_l: linked neighbour closest distance
    # l_len_h: linked neighbour largest distance
    # agg_sz: maximum number of particle per conglomerate
    # t: maximum number of tries for avoiding particle overlapping
    # nrots: number or random rotations (by default 1000)
    # max_rad: maximum radius in nm for an aggregate (if None it is computed automatically)
    # rig_sd: if this not None (default) is this sigma for Gaussian random rotation in the rigid model
    def __init__(self, res, temp, l_len_l, l_len_h, agg_sz, t, nrots=1000, max_rad=None, rig_sd=None):
        # Input parsing
        self.__res = float(res)
        if (not isinstance(temp, np.ndarray)) or (len(temp.shape) != 3):
            error_msg = 'Particle distance must be a 3d numpy array!'
            raise pexceptions.PySegInputError(expr='__init__ (NetSimulator)', msg=error_msg)
        self.__temp = np.asarray(temp, dtype=np.float32)
        self.__nx, self.__ny, self.__nz = self.__temp.shape
        self.__hl_x, self.__hl_y, self.__hl_z = int(self.__temp.shape[1]*.5), int(self.__temp.shape[0]*.5), \
                                                int(self.__temp.shape[2]*.5)
        self.__t = int(t)
        self.__nrots = int(nrots)
        if self.__nrots <= 0:
            error_msg = 'The number or random rotation must be greater than zero!'
            raise pexceptions.PySegInputError(expr='__init__ (CSRTSimulator)', msg=error_msg)
        nx, ny, nz = self.__temp.shape
        self.__rnd_rots = np.zeros(shape=(nx, ny, nz, self.__nrots), dtype=np.bool)
        self.__rnd_count = 0
        self.__l_len_lv = int(round(l_len_l))
        if self.__l_len_lv <= 0:
            error_msg = 'Linker minimum length must be at least one voxel large!'
            raise pexceptions.PySegInputError(expr='__init__ (AggSimulator)', msg=error_msg)
        self.__l_len_hv = int(round(l_len_h))
        if self.__l_len_hv <= self.__l_len_lv:
            error_msg = 'Linker minimum length must be greater than minimum distance!'
            raise pexceptions.PySegInputError(expr='__init__ (AggSimulator)', msg=error_msg)
        self.__agg_sz = int(agg_sz)
        if self.__agg_sz < 0:
            error_msg = 'Maximum number of particle per aggregate must be greater than zero!'
            raise pexceptions.PySegInputError(expr='__init__ (AggSimulator)', msg=error_msg)
        self.__max_rad_v = None
        if max_rad is not None:
            self.__max_rad_v = max_rad / self.__res
        self.__rig_sd = rig_sd
        if self.__rig_sd is not None:
            self.__rig_sd = float(self.__rig_sd)
        # Create particle distance transform template
        ptemp_shape = self.__l_len_hv + np.asarray(self.__temp.shape)
        if (ptemp_shape[0]%2) != 0:
            ptemp_shape[0] += 1
        if (ptemp_shape[1]%2) != 0:
            ptemp_shape[1] += 1
        if (ptemp_shape[2]%2) != 0:
            ptemp_shape[2] += 1
        hold_ptemp = np.zeros(shape=ptemp_shape, dtype=np.bool)
        insert_sub_tomo_bin(hold_ptemp, .5*ptemp_shape-1, self.__temp>ROT_TMP_TH)
        self.__ptemp = hold_ptemp
        self.__rnd_prots = np.ones(shape=(self.__ptemp.shape[0], self.__ptemp.shape[1], self.__ptemp.shape[2],
                                          self.__nrots), dtype=np.uint16)
        self.__gen_rnd_rots()
        # Temporary variables for the aggregates
        self.__agg_mask = None
        self.__agg_pot = None
        self.__agg_pot2 = None
        self.__agg_voi = (-1.) * np.ones(shape=6, dtype=np.int)
        self.__agg_coords = None

    ######## External area

    def get_rnd_rots_array(self):
        return self.__rnd_rots

    def gen_rand_in_mask(self, n, mask):

        coords, _ = self.__gen_instance(n, mask, self.__temp)

        return np.asarray(coords, dtype=np.float32)

    def gen_rand_in_mask_tomo(self, n, mask):

        _, hold_tomo = self.__gen_instance(n, mask, self.__temp)

        return hold_tomo

    ######## Internal area

    # npart: number of particles
    # mask: tomogram mask with valid region
    # temp: sub-volume with the segmented particle template
    # Returns: a graph (graph_tool) where particles coordinates are stored in a property indexed as PART_COORDS,
    #          mask tomogram with placed particles is also returned
    def __gen_instance(self, npart, mask, temp):

        # Initialization
        hold_mask = np.invert(mask)
        hold_mask_pot = np.iinfo(np.uint16).max * np.ones(shape=hold_mask.shape, dtype=np.uint16)
        global_gen = RandCoordsMask(hold_mask)
        max_tries = global_gen.get_initial_tries()
        coords = list()

        # Loop for trees
        curr_try = 0
        while (len(coords) < npart) and (curr_try < max_tries):
            hold_np = npart-len(coords)
            if hold_np > self.__agg_sz:
                hold_np = self.__agg_sz
            hold_coords, hold_tries = self.__gen_aggregate(hold_np, max_tries-curr_try, hold_mask, hold_mask_pot,
                                                           global_gen)
            # disperse_io.save_numpy(hold_mask, '/home/martinez/workspace/stat/pyrenoid/net_high_res/hold.mrc')
            print 'AGGREGATE INSERTED WITH ' + str(len(hold_coords)) + ' PARTICLES. Left: ' + str(npart-len(coords))
            coords += hold_coords
            curr_try += hold_tries

        # hold_mask_pot = (hold_mask_pot>=self.__l_len_lv) * (hold_mask_pot<=self.__l_len_hv)
        # disperse_io.save_numpy(hold_mask_pot, '/home/martinez/workspace/stat/pyrenoid/net_high_res/hold_tomo_pot.mrc')

        # Check all demanded particles are placed
        if len(coords) < npart:
            print 'WARNING (NetSimulator): only ' + str(len(coords)) + ' of ' + str(npart) + ' has been placed after ' \
                  + str(max_tries) + ' tries.'

        return coords, hold_mask

    # Generates a tree
    # max_np    : maximum number of particles
    # max_tries: maximum number of tries
    # mask: mask with the valid volume to place particle (it can be modified)
    # mask: mask for distance transform
    # global_gen: generator for random coordinate in the global mask
    def __gen_aggregate(self, max_np, max_tries, mask, mask_pot, global_gen):

        # Initialization
        n_tries = 0

        # Generate seed particle
        pot_lock = True
        n_parts = 0
        while n_tries < max_tries:
            # Generate a new isolated particle
            n_tries += 1
            try:
                x, y, z = global_gen.gen_rand_coords(mask)

                # Template random rotation
                rot_temp, rot_ptemp = self.__get_rnd_rot()

                # Insert the new particle
                coord = np.asarray((x, y, z), dtype=np.float32)
                insert_sub_tomo_bin(mask, coord, rot_temp, no_over=False)

                # Create potential mask
                self.__init_pot_mask(coord, mask, mask_pot, rot_temp, rot_ptemp)
                pot_lock = False
                n_parts += 1
                break

            except Exception:
                pass

        # Aggregate loop
        if pot_lock:
            return None, n_tries
        nn_tries = 0
        while (nn_tries < self.__t) and (n_tries < max_tries) and (n_parts < max_np):

            # Generate new particle in the potential mask
            try:
                self.__new_pot_particle()
                n_parts += 1
            except ValueError:
                nn_tries += 1

        # Update results
        # insert_sub_tomo_bin(mask, coord, self.__agg_mask, no_over=True)
        coords = self.__get_coords()

        return coords, n_tries

    def __get_rnd_rot(self):
        hold1 = self.__rnd_rots[:, :, :, self.__rnd_count]
        hold2 = self.__rnd_prots[:, :, :, self.__rnd_count]
        self.__rnd_count += 1
        if self.__rnd_count >= self.__nrots:
            self.__rnd_count = 0
        return hold1, hold2


    def __gen_rnd_rots(self):
        for i in range(self.__nrots):
            quat = rand_quat()
            mat = quat_to_rot(quat)
            eu_angs = rot_mat_eu_relion(mat)
            hold_rot = tomo_rot(self.__temp, eu_angs, conv='relion', deg=True) > ROT_TMP_TH
            self.__rnd_rots[:, :, :, i] = hold_rot
            hold_rot = tomo_rot(self.__ptemp, eu_angs, conv='relion', deg=True) > ROT_TMP_TH
            hold_dst = sp.ndimage.morphology.distance_transform_edt(np.invert(hold_rot))
            self.__rnd_prots[:, :, :, i] = hold_dst

            # disperse_io.save_numpy(self.__temp, '/home/martinez/workspace/stat/pyrenoid/net_high_res/hold_temp.mrc')
            # disperse_io.save_numpy(self.__ptemp, '/home/martinez/workspace/stat/pyrenoid/net_high_res/hold_ptemp.mrc')
            # disperse_io.save_numpy(self.__rnd_rots[:, :, :, i], '/home/martinez/workspace/stat/pyrenoid/net_high_res/hold_rot.mrc')
            # disperse_io.save_numpy(self.__rnd_prots[:, :, :, i], '/home/martinez/workspace/stat/pyrenoid/net_high_res/hold_prot.mrc')

    def __init_pot_mask(self, coord, mask, mask_pot, rot_temp, rot_ptemp):

        # Initialization
        self.__coords = list()
        self.__agg_coords = list()


        # Compute VOI limits
        x, y, z = coord
        hold_hl = float(self.__agg_sz) * self.__l_len_hv
        if (self.__max_rad_v is not None) and (hold_hl > self.__max_rad_v):
            hl = int(math.ceil(self.__max_rad_v))
        else:
            hl = int(math.ceil(hold_hl))
        mx, my, mz = mask.shape[0], mask.shape[1], mask.shape[2]
        mx1, my1, mz1 = mx-1, my-1, mz-1
        self.__agg_voi[0], self.__agg_voi[1], self.__agg_voi[2] = x-hl+1, y-hl+1, z-hl+1
        self.__agg_voi[3], self.__agg_voi[4], self.__agg_voi[5] = x+hl+1, y+hl+1, z+hl+1
        if self.__agg_voi[0] < 0:
            self.__agg_voi[0] = 0
        if self.__agg_voi[1] < 0:
            self.__agg_voi[1] = 0
        if self.__agg_voi[2] < 0:
            self.__agg_voi[2] = 0
        if self.__agg_voi[3] >= mx:
            self.__agg_voi[3] = mx1
        if self.__agg_voi[4] >= my:
            self.__agg_voi[4] = my1
        if self.__agg_voi[5] >= mz:
            self.__agg_voi[5] = mz1

        # Copy VOI mask
        self.__agg_mask = mask[self.__agg_voi[0]:self.__agg_voi[3],
                               self.__agg_voi[1]:self.__agg_voi[4],
                               self.__agg_voi[2]:self.__agg_voi[5]]
        self.__agg_pot = mask_pot[self.__agg_voi[0]:self.__agg_voi[3],
                                  self.__agg_voi[1]:self.__agg_voi[4],
                                  self.__agg_voi[2]:self.__agg_voi[5]]

        # Add the initial particle
        xo, yo, zo = x-self.__agg_voi[0], y-self.__agg_voi[1], z-self.__agg_voi[2]
        over_sub_tomo(self.__agg_pot, (xo,yo,zo), rot_ptemp, np.minimum)
        self.__agg_pot2 = np.iinfo(np.uint16).max * np.ones(shape=self.__agg_pot.shape, dtype=np.uint16)
        over_sub_tomo(self.__agg_pot2, (xo,yo,zo), rot_ptemp, np.minimum)
        coord = np.asarray((xo, yo, zo), dtype=np.float32)
        self.__coords.append(coord)
        if self.__rig_sd is not None:
            self.__agg_coords.append(coord)

    def __new_pot_particle(self):

        # Mask for valid locations
        loc_mask = (self.__agg_pot2<self.__l_len_lv) | (self.__agg_pot2>self.__l_len_hv)
        loc_mask[self.__agg_mask] = 1.
        loc_mask[(self.__agg_pot<self.__l_len_lv) | (self.__agg_pot>self.__l_len_hv)] = 1

        # disperse_io.save_numpy(self.__agg_mask, '/home/martinez/workspace/stat/pyrenoid/net_high_res/hold_agg_mask.mrc')
        # disperse_io.save_numpy(self.__agg_pot, '/home/martinez/workspace/stat/pyrenoid/net_high_res/hold_pot_mask.mrc')
        # disperse_io.save_numpy(loc_mask, '/home/martinez/workspace/stat/pyrenoid/net_high_res/hold_loc_mask.mrc')

        # Generate the new coordinate
        if self.__rig_sd is None:
            generator = RandCoordsMask(loc_mask)
            generator.get_initial_tries()
            xo, yo, zo = generator.gen_rand_coords()
        else:
            xo, yo, zo = self.__gen_rigid_neighbour(loc_mask)
        rot_temp, rot_ptemp = self.__get_rnd_rot()

        # Add the new particle
        insert_sub_tomo_bin(self.__agg_mask, (xo,yo,zo), rot_temp, no_over=False)
        over_sub_tomo(self.__agg_pot, (xo,yo,zo), rot_ptemp, np.minimum)
        # self.__agg_pot2 = np.iinfo(np.uint16).max * np.ones(shape=self.__agg_pot.shape, dtype=np.uint16)
        over_sub_tomo(self.__agg_pot2, (xo,yo,zo), rot_ptemp, np.minimum)
        coord = np.asarray((xo, yo, zo), dtype=np.float32)
        self.__coords.append(coord)
        if self.__rig_sd is not None:
            self.__agg_coords.append(coord)

    def __get_coords(self):
        hold_coords = list()
        for coord in self.__coords:
            x, y, z = self.__agg_voi[0]+coord[0], self.__agg_voi[1]+coord[1], self.__agg_voi[2]+coord[2]
            hold_coords.append(np.asarray((x, y, z), dtype=np.float32))
        return hold_coords

    # Generates a neighbour in semi-rigid filament model for the aggregate
    def __gen_rigid_neighbour(self, loc_mask):
        if len(self.__agg_coords) >= 2:
            pt0 = self.__agg_coords[-2]
            pt1 = self.__agg_coords[-1]
            v = pt1 - pt0
            vm = math.sqrt((v*v).sum())
            if vm > 0:
                v /= vm
                verr = np.asarray((np.random.normal(0, self.__rig_sd),
                                   np.random.normal(0, self.__rig_sd),
                                   np.random.normal(0, self.__rig_sd)), dtype=np.float32)
                v += verr
                # Find the best aligned valid coordinate
                ids = np.where(loc_mask==False)
                ang = np.pi
                nx_pt = None
                for i in range(len(ids[0])):
                    idx = np.asarray((ids[0][i], ids[1][i], ids[2][i]), dtype=np.float32)
                    hold_ang = math.fabs(angle_2vec_3D(v, idx-pt1))
                    if hold_ang < ang:
                        ang = hold_ang
                        nx_pt = idx
                if nx_pt is None:
                    raise ValueError
                return nx_pt[0], nx_pt[1], nx_pt[2]
            else:
                raise ValueError
        else:
            generator = RandCoordsMask(loc_mask)
            generator.get_initial_tries()
            return generator.gen_rand_coords()

######################################################################################################
# Class with for simulating particles agglomerative model
#   - Particle are grouped in groups with a maximum number of particles
#   - These groups are distributed randomly
#   - Ever new particle in a group is added pseudo-randomly close to already existing particles
#   - Trees grow with branches of size n particles
#   - Simulator ends if al demanded particles are placed or maximum number of tries set is reached
class AggSimulator2(object):

    # Constructor area
    # res: resolution in nm/pixel
    # temp: particle template
    # binds: list with binding sites with the input coordinates
    # l_len: linker length in nm
    # l_sg: linker sigma (related with the linker flexibility) in nm
    # rs: re-start probability
    # sh_t: number of tries for trying to bind a particle
    # sh_sg: number of times maximum input template dimension region used for trying to bind a particle
    # nrots: number or random rotations (by default 1000)
    def __init__(self, res, temp, binds, l_len, l_sg, rs, sh_t, sh_sg=5, nrots=1000):
        # Input parsing
        self.__res = float(res)
        if (not isinstance(temp, np.ndarray)) or (len(temp.shape) != 3):
            error_msg = 'Particle distance must be a 3d numpy array!'
            raise pexceptions.PySegInputError(expr='__init__ (NetSimulator)', msg=error_msg)
        self.__temp = np.asarray(temp, dtype=np.float32)
        self.__rs = float(rs)
        self.__sh_t = int(sh_t)
        if self.__sh_t < 0:
            error_msg = 'Number of tries to find maximum potential must be greater than 0!'
            raise pexceptions.PySegInputError(expr='__init__ (AggSimulator)', msg=error_msg)
        if sh_sg < 0:
            error_msg = 'Subvolume for neighbourhood must be greater than zero!'
            raise pexceptions.PySegInputError(expr='__init__ (AggSimulator)', msg=error_msg)
        self.__neigh_shape = int(sh_sg) * np.asarray(self.__temp.shape, dtype=np.int)
        self.__nrots = int(nrots)
        if self.__nrots <= 0:
            error_msg = 'The number or random rotation must be greater than zero!'
            raise pexceptions.PySegInputError(expr='__init__ (CSRTSimulator)', msg=error_msg)
        nx, ny, nz = self.__temp.shape
        self.__l_len = float(l_len)
        if self.__l_len < 0:
            error_msg = 'Linker length must be greater than 0 voxels!'
            raise pexceptions.PySegInputError(expr='__init__ (AggSimulator)', msg=error_msg)
        self.__l_sg = float(l_sg)
        if self.__l_sg < 0:
            error_msg = 'Linker sigma must be greater than 0 voxels!'
            raise pexceptions.PySegInputError(expr='__init__ (AggSimulator)', msg=error_msg)
        if not hasattr(binds, '__len__'):
            error_msg = 'Input binds must be an iterable.'
            raise pexceptions.PySegInputError(expr='__init__ (AggSimulator)', msg=error_msg)
        self.__binds = list()
        for bind in binds:
            if hasattr(bind, '__len__') and (len(bind)==3):
                self.__binds.append(np.asarray(bind, dtype=np.float32))
            else:
                error_msg = 'Input binds must a 3-tuple.'
                raise pexceptions.PySegInputError(expr='__init__ (AggSimulator)', msg=error_msg)
        # Generate particles potentials and their rotations
        self.__rnd_rots = np.zeros(shape=(nx, ny, nz, self.__nrots), dtype=np.bool)
        self.__rnd_count = 0
        self.__rnd_prots = np.ones(shape=(nx, ny, nz, self.__nrots), dtype=np.float32)
        self.__pt = None
        self.__mask_nsph = None
        self.__init_pfield()
        self.__gen_rnd_rots()
        self.__gen_ns_nsph()

    ######## External area

    def get_rnd_rots_array(self):
        return self.__rnd_rots

    def gen_rand_in_mask(self, n, mask):

        coords, _ = self.__gen_instance(n, mask, self.__temp)

        return np.asarray(coords, dtype=np.float32)

    def gen_rand_in_mask_tomo(self, n, mask):

        _, hold_tomo = self.__gen_instance(n, mask, self.__temp)

        return hold_tomo

    ######## Internal area

    # npart: number of particles
    # mask: tomogram mask with valid region
    # temp: sub-volume with the segmented particle template
    # Returns: a graph (graph_tool) where particles coordinates are stored in a property indexed as PART_COORDS,
    #          mask tomogram with placed particles is also returned
    def __gen_instance(self, npart, mask, temp):

        # Initialization
        hold_mask = np.invert(mask)
        hold_mask_pot = np.zeros(shape=hold_mask.shape, dtype=np.float32)
        global_gen = RandCoordsMask(hold_mask)
        max_tries = global_gen.get_initial_tries()
        coords = list()

        # Loop for trees
        curr_try = 0
        while (len(coords) < npart) and (curr_try < max_tries):
            hold_np = npart-len(coords)
            hold_coords, hold_tries = self.__gen_aggregate(hold_np, max_tries-curr_try, hold_mask, hold_mask_pot,
                                                           global_gen)
            # disperse_io.save_numpy(hold_mask, '/home/martinez/workspace/stat/pyrenoid/net_high_res/hold.mrc')
            print 'AGGREGATE INSERTED WITH ' + str(len(hold_coords)) + ' PARTICLES. Left: ' + str(npart-len(coords))
            coords += hold_coords
            curr_try += hold_tries

        # hold_mask_pot = (hold_mask_pot>=self.__l_len_lv) * (hold_mask_pot<=self.__l_len_hv)
        # disperse_io.save_numpy(hold_mask_pot, '/home/martinez/workspace/stat/pyrenoid/net_high_res/hold_tomo_pot.mrc')

        # Check all demanded particles are placed
        if len(coords) < npart:
            print 'WARNING (NetSimulator): only ' + str(len(coords)) + ' of ' + str(npart) + ' has been placed after ' \
                  + str(max_tries) + ' tries.'

        return coords, hold_mask

    # Generates an agregate
    # max_np: maximum number of particles
    # max_tries: maximum number of tries
    # mask: mask with the valid volume to place particle (it can be modified)
    # p_field: potential field
    # global_gen: generator for random coordinate in the global mask
    def __gen_aggregate(self, max_np, max_tries, mask, p_field, global_gen):

        # Initialization
        n_tries = 0
        coords = list()

        # Generate seed particle
        while n_tries < max_tries:
            # Generate a new isolated particle
            n_tries += 1


            # Getting the seed
            try:
                x, y, z = global_gen.gen_rand_coords(mask)
            except Exception:
                continue
            seed_coord = np.asarray((x, y, z), dtype=np.float32)
            rot_temp, rot_pot = self.__get_rnd_rot()

            # Checking no particle overlapping for the seed
            if not check_bin_sub_tomo(mask, seed_coord, rot_temp):

                # Insert seed in the global mask
                insert_sub_tomo_bin(mask, seed_coord, rot_temp, no_over=True)
                hold_mask = get_sub_copy(mask, seed_coord, self.__temp.shape, copy=True)[0]
                hold_pot = get_sub_copy(p_field, seed_coord, self.__temp.shape, copy=True)[0]
                hold_add_pot = sum_pbs(hold_pot, rot_pot, hold_mask, rot_temp)
                insert_sub_tomo(p_field, seed_coord, hold_add_pot)

                # Initialize the queue with the seed
                queue = list()
                queue.append(seed_coord)
                coords.append(seed_coord)

                # Loop for adding the neighbours
                rlock = True
                while rlock and (len(queue) > 0):
                    # Extract current particle
                    part_coord = queue.pop()

                    # Get neighbourhood mask and potential
                    neigh_mask, part_off = get_sub_copy(mask, part_coord, self.__neigh_shape, copy=False)
                    neigh_pot = get_sub_copy(p_field, part_coord, self.__neigh_shape, copy=False)[0]
                    neigh_gen = RandCoordsMask(neigh_mask | self.__mask_sph)

                    # disperse_io.save_numpy(neigh_mask | self.__mask_sph, '/home/martinez/workspace/stat/pyrenoid/agg_test/hold.mrc')

                    # Adding the neighbour
                    curr_max = 0
                    curr_coord, curr_rot_temp, curr_rot_pot = None, None, None
                    for i in range(self.__sh_t):
                        # Generate a new potential neighbor
                        xo, yo, zo = neigh_gen.gen_rand_coords()
                        neigh_coord = np.asarray((xo, yo, zo), dtype=np.float32)
                        hold_rot_temp, hold_rot_pot = self.__get_rnd_rot()
                        # Check no overlap
                        if not check_bin_sub_tomo(neigh_mask, neigh_coord, hold_rot_temp):
                            # Getting its neighborhood
                            hold_mask = get_sub_copy(neigh_mask, neigh_coord, self.__temp.shape, copy=True)[0]
                            hold_pot = get_sub_copy(neigh_pot, neigh_coord, self.__temp.shape, copy=True)[0]
                            hold_add_pot = sum_pbs(hold_pot, hold_rot_pot, hold_mask, hold_rot_temp)
                            hold_add_pot_max = hold_add_pot.max()
                            if hold_add_pot_max > curr_max:
                                curr_coord = neigh_coord
                                curr_rot_temp = hold_rot_temp
                                curr_rot_pot = hold_add_pot
                                curr_max = hold_add_pot_max

                    # Update global masks, potentials and queue if a neighbor was found
                    if curr_coord is not None:
                        # Update mask and potential
                        insert_sub_tomo_bin(neigh_mask, curr_coord, curr_rot_temp, no_over=True)
                        insert_sub_tomo(neigh_pot, curr_coord, curr_rot_pot)
                        nneigh_coord = curr_coord + part_off
                        queue.append(nneigh_coord)
                        coords.append(nneigh_coord)

                        # disperse_io.save_numpy(neigh_mask, '/home/martinez/workspace/stat/pyrenoid/agg_test/hold.mrc')

                    # Re-start condition
                    if np.random.random_sample() < self.__rs:
                        break

        return coords, n_tries

    def __get_rnd_rot(self):
        hold1 = self.__rnd_rots[:, :, :, self.__rnd_count]
        hold2 = self.__rnd_prots[:, :, :, self.__rnd_count]
        self.__rnd_count += 1
        if self.__rnd_count >= self.__nrots:
            self.__rnd_count = 0
        return hold1, hold2

    def __gen_rnd_rots(self):
        for i in range(self.__nrots):
            quat = rand_quat()
            mat = quat_to_rot(quat)
            eu_angs = rot_mat_eu_relion(mat)
            hold_rot = tomo_rot(self.__temp, eu_angs, conv='relion', deg=True) > ROT_TMP_TH
            self.__rnd_rots[:, :, :, i] = hold_rot
            self.__rnd_prots[:, :, :, i] = tomo_rot(self.__pt, eu_angs, conv='relion', deg=True)

            # disperse_io.save_numpy(self.__temp, '/home/martinez/workspace/stat/pyrenoid/net_high_res/hold_temp.mrc')
            # disperse_io.save_numpy(self.__ptemp, '/home/martinez/workspace/stat/pyrenoid/net_high_res/hold_ptemp.mrc')
            # disperse_io.save_numpy(self.__rnd_rots[:, :, :, i], '/home/martinez/workspace/stat/pyrenoid/net_high_res/hold_rot.mrc')
            # disperse_io.save_numpy(self.__rnd_prots[:, :, :, i], '/home/martinez/workspace/stat/pyrenoid/net_high_res/hold_prot.mrc')

    def __init_pfield(self):
        temp_bin = self.__temp > 0
        l_sg_v, l_len_v = self.__l_sg/self.__res, self.__l_len/self.__res
        sum_pf = gen_pb_normal(temp_bin, self.__binds[0], l_sg_v, l_len_v)
        for i in range(1, len(self.__binds)):
            hold_pf = gen_pb_normal(temp_bin, self.__binds[i], l_sg_v, l_len_v)
            sum_pf = sum_pbs(hold_pf, sum_pf, temp_bin, temp_bin)
        self.__pt = sum_pf

    def __gen_ns_nsph(self):
        nx, ny, nz = self.__neigh_shape+1
        nx2, ny2, nz2 = .5 * (self.__neigh_shape + 1)
        X, Y, Z = np.meshgrid(np.arange(1,nx), np.arange(1,ny), np.arange(1,nz), indexing='ij')
        X, Y, Z = X-nx2, Y-ny2, Z-nz2
        md = np.asarray(self.__temp.shape, dtype=np.int).max()
        self.__mask_sph = np.sqrt(X*X + Y*Y + Z*Z) > (0.5*md)


##############################################################################################
# Class for univariate statistical spatial analysis
#
class UniStat(object):

    ##### Constructor area

    # coords: 2D or 3D coordinates (in voxels)
    # mask: 2D or 3D mask
    # res: voxel resolution (nm/pixel)
    # name: string with the name of the patterns (default None)
    # rots: rotation Euler angles, required if temp is not None (default None)
    # conv: Euler angles convention, valids: 'relion' (default) and 'tom'
    # mmap: default None, if not it contains a directory where heavy numpy arrays are stored as memory maps,
    #       this option saves memory but then analyis will be slower, only use if memory overload
    def __init__(self, coords, mask, res, name=None, temp=None, rots=None, conv='relion', mmap=None):
        self.__name = name
        self.__mode_2D = False
        if coords.shape[1] == 3:
            hold_coords = coords.astype(np.float)
            if hold_coords[:, 2].max() <= 1:
                self.__mode_2D = True
        elif coords.shape[1] == 2:
            hold_coords = np.zeros(shape=(coords.shape[0], 3), dtype=np.float)
            hold_coords[:, 0] = coords[:, 0]
            hold_coords[:, 1] = coords[:, 1]
            self.__mode_2D = True
        else:
            error_msg = 'Input coords must be 2D or 3D!'
            raise pexceptions.PySegInputError(expr='__init__ (UniStat)', msg=error_msg)
        if len(mask.shape) == 3:
            self.__mask = mask.astype(np.bool)
        elif len(mask.shape) == 2:
            self.__mask = np.zeros(shape=(mask.shape[0], mask.shape[1], 1), dtype=np.bool)
            self.__mask[:, :, 0] = mask
        else:
            error_msg = 'Input coords must be 2D or 3D!'
            raise pexceptions.PySegInputError(expr='__init__ (UniStat)', msg=error_msg)
        self.__z = float(res * res)
        if not self.__mode_2D:
            self.__z *= float(res)
        # Rotation variables
        self.__temp, self.__rots, self.__conv = None, None, 'relion'
        if isinstance(temp, np.ndarray) and (len(temp.shape) == 3):
            if ((temp.shape[0]%2) != 0) or ((temp.shape[1]%2) != 0) or ((temp.shape[2]%2) != 0):
                error_msg = 'Input template dimension must be even!'
                raise pexceptions.PySegInputError(expr='__init__ (UniStat)', msg=error_msg)
            if (len(rots.shape) != 2) or (rots.shape[0] != coords.shape[0]) or (rots.shape[1] != 3):
                error_msg = 'Input rots and array of 3 Euler angles with the same length as coords!'
                raise pexceptions.PySegInputError(expr='__init__ (UniStat)', msg=error_msg)
            self.__temp = temp
            self.__dense, self.__coords_d, self.__rots = coords_to_dense_mask(hold_coords, self.__mask,
                                                                              rots)
        else:
            self.__dense, self.__coords_d = coords_to_dense_mask(hold_coords, self.__mask)
        if mmap is not None:
            if not os.path.exists(mmap):
                try:
                    os.mkdir(mmap)
                except:
                    error_msg = 'Error creating hold memory map directory:' + mmap
                    raise pexceptions.PySegInputError(expr='__init__ UniStat', msg=error_msg)
        self.__n_coords = self.__coords_d.shape[0]
        self.__res = res
        self.__coords = self.__coords_d * res
        self.__mask_sum = self.__mask.sum()
        self.__lambda = self.__n_coords / (self.__mask_sum * self.__z)
        # Internal variables for computation acceleration
        sap = self.__mask.shape
        self.__Y, self.__X, self.__Z = np.meshgrid(np.arange(sap[1]).astype(np.int16),
                                                   np.arange(sap[0]).astype(np.int16),
                                                   np.arange(sap[2]).astype(np.int16),
                                                   copy=False)
        if mmap is not None:
            hold_dense = np.memmap(mmap+'/dense.dat', dtype=self.__dense.dtype,
                                   mode='w+', shape=self.__dense.shape)
            hold_dense[:] = self.__dense[:]
            del self.__dense
            self.__dense = hold_dense
            hold_mask = np.memmap(mmap+'/mask.dat', dtype=self.__mask.dtype,
                                  mode='w+', shape=self.__mask.shape)
            hold_mask[:] = self.__mask[:]
            del self.__mask
            self.__mask = hold_mask
            hold_X = np.memmap(mmap+'/X.dat', dtype=self.__X.dtype,
                               mode='w+', shape=self.__X.shape)
            hold_X[:] = self.__X[:]
            del self.__X
            self.__X = hold_X
            hold_Y = np.memmap(mmap+'/Y.dat', dtype=self.__Y.dtype,
                               mode='w+', shape=self.__Y.shape)
            hold_Y[:] = self.__Y[:]
            del self.__Y
            self.__Y = hold_Y
            hold_Z = np.memmap(mmap+'/Z.dat', dtype=self.__Z.dtype,
                               mode='w+', shape=self.__Z.shape)
            hold_Z[:] = self.__Z[:]
            del self.__Z
            self.__Z = hold_Z
        self.__n_coords = self.__coords.shape[0]
        self.__mx_b, self.__my_b, self.__mz_b = sap[0], sap[1], sap[2]
        # Setting random simulator: by default CSR but it can take into account particle shape if required
        self.__simulator = CSRSimulator()

    ##### Set/Get functionality

    def set_name(self, name):
        self.__name = name

    def set_simulator(self, simulator):
        self.__simulator = simulator

    def get_name(self):
        return self.__name

    def get_intensity(self):
        return self.__lambda

    def get_n_points(self):
        return self.__n_coords

    def get_mask(self):
        return self.__mask

    # Returns volume (of Area for 2D) in nm^3
    def get_volume(self):
        return self.__mask_sum * self.__z

    def get_resolution(self):
        return self.__res

    # in_nm: if True (default) coordinates are expresed in nm, otherwise in pixels/voxels
    def get_coords(self, in_nm=True):
        if in_nm:
            return self.__coords
        else:
            return self.__coords_d

    ##### External functionality

    def is_2D(self):
        return self.__mode_2D

    # Convert this object from 3D to 2D, not possible if points have shape (template)
    # del_coord: coordinate index to delete: 0, 1 or 2
    def to_2D(self, del_coord):

        if self.__temp is not None:
            error_msg = 'Points with 3D shape cannot be converted to 2D'
            raise pexceptions.PySegInputError(expr='to_2D (UniStat)', msg=error_msg)

        if self.is_2D():
            return

        # Compressing the mask
        self.__z = float(self.__res * self.__res)
        mask_2d = (self.__mask.sum(axis=del_coord) > 0)
        self.__mask = np.zeros(shape=(mask_2d.shape[0], mask_2d.shape[1], 1), dtype=np.bool)
        self.__mask[:, :, 0] = mask_2d
        self.__mask_sum = self.__mask.sum()

        # Compressing the coordinates
        coords_2d = make_plane(self.__coords, del_coord)
        hold_coords = np.zeros(shape=(coords_2d.shape[0], 3), dtype=np.float32)
        hold_coords[:, 0] = coords_2d[:, 0]
        hold_coords[:, 1] = coords_2d[:, 1]
        self.__dense, self.__coords_d = coords_to_dense_mask(hold_coords, self.__mask)

        # Updating class variables
        self.__n_coords = self.__coords_d.shape[0]
        self.__coords = self.__coords_d * self.__res
        self.__lambda = self.__n_coords / (self.__mask_sum * self.__z)
        # Internal variables for computation acceleration
        sap = self.__mask.shape
        self.__Y, self.__X, self.__Z = np.meshgrid(np.arange(sap[1]).astype(np.int16),
                                                   np.arange(sap[0]).astype(np.int16),
                                                   np.arange(sap[2]).astype(np.int16),
                                                   copy=False)
        self.__n_coords = self.__coords.shape[0]
        self.__mx_b, self.__my_b, self.__mz_b = sap[0], sap[1], sap[2]
        # Setting random simulator: by default CSR but it can take into account particle shape if required
        self.__simulator = CSRSimulator()

    # Store a tomograms with point pattern, the points has a shape it is taken into account
    # file_name: full path to the stored tomogram
    # mask: if True (default False) the mask is also printed
    def save_sparse(self, file_name, mask=False):
        if self.__temp is None:
            if mask:
                disperse_io.save_numpy(self.__dense, file_name)
            else:
                out_tomo = np.zeros(shape=self.__mask.shape, dtype=np.int16)
                out_tomo[out_tomo > 0] = 2
                out_tomo[self.__mask == False] = 1
                disperse_io.save_numpy(out_tomo, file_name)
        else:
            temp_f = np.asarray(self.__temp, dtype=np.float32)
            hold_mask = np.zeros(shape=self.__dense.shape, dtype=self.__temp.dtype)
            nx, ny, nz = self.__temp.shape[0], self.__temp.shape[1], self.__temp.shape[2]
            mx, my, mz = self.__dense.shape[0], self.__dense.shape[1], self.__dense.shape[2]
            mx1, my1, mz1 = mx-1, my-1, mz-1
            hl_x, hl_y, hl_z = int(self.__temp.shape[0]*.5), int(self.__temp.shape[1]*.5), int(self.__temp.shape[2]*.5)
            for coord, rot in zip(self.__coords_d, self.__rots):
                # Coordinates
                x, y, z = coord[0], coord[1], coord[2]
                # Template rotation
                rot_temp = tomo_rot(temp_f, rot, conv=self.__conv, deg=True) > ROT_TMP_TH
                # Compute bounding restriction
                off_l_x, off_l_y, off_l_z = x-hl_x+1, y-hl_y+1, z-hl_z+1
                off_h_x, off_h_y, off_h_z = x+hl_x+1, y+hl_y+1, z+hl_z+1
                dif_l_x, dif_l_y, dif_l_z = 0, 0, 0
                dif_h_x, dif_h_y, dif_h_z = nx, ny, nz
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
                # Update mask
                try:
                    hold_sv = hold_mask[off_l_x:off_h_x, off_l_y:off_h_y, off_l_z:off_h_z]
                    hold_rot = rot_temp[dif_l_x:dif_h_x, dif_l_y:dif_h_y, dif_l_z:dif_h_z]
                    hold_sv[hold_rot] = 1
                    hold_mask[off_l_x:off_h_x, off_l_y:off_h_y, off_l_z:off_h_z] = hold_sv
                except IndexError:
                    print 'Jol'
                # hold_mask[off_l_x:off_h_x, off_l_y:off_h_y, off_l_z:off_h_z] = \
                #     rot_temp[dif_l_x:dif_h_x, dif_l_y:dif_h_y, dif_l_z:dif_h_z]

                # pyseg.disperse_io.save_numpy(temp_f, '/home/martinez/workspace/stat/pyrenoid/test_linker/temp.mrc')
                # pyseg.disperse_io.save_numpy(rot_temp, '/home/martinez/workspace/stat/pyrenoid/test_linker/temp_rot.mrc')

            if mask:
                out_tomo = np.zeros(shape=self.__mask.shape, dtype=np.int16)
                out_tomo[hold_mask > 0] = 2
                out_tomo[self.__mask == False] = 1
                disperse_io.save_numpy(out_tomo, file_name)
            else:
                disperse_io.save_numpy(hold_mask, file_name)

    # Store a tomogram (dtype=int) where bg=0, mask=1, points=2
    # file_name: full path to the stored tomogram
    def save_dense(self, file_name):
        disperse_io.save_numpy(self.__mask.astype(np.int)+self.__dense.astype(np.int), file_name)

    # Stores a random instance
    # file_name: full path to the stored tomogram
    # pts: if True (default False) template coordinates is are stored instead templates
    def save_random_instance(self, file_name, pts=False):
        out_tomo = np.zeros(shape=self.__mask.shape, dtype=np.int16)
        if pts:
            out_tomo[self.__mask == False] = 1
            coords = self.__simulator.gen_rand_in_mask(self.__n_coords, self.__mask)
            for coord in coords:
                x, y, z = int(round(coord[0])), int(round(coord[1])), int(round(coord[2]))
                try:
                    out_tomo[x, y, z] = 2
                except IndexError:
                    pass
        else:
            out_tomo[self.__mask == False] = 1
            rnd = self.__simulator.gen_rand_in_mask_tomo(self.__n_coords, self.__mask)
            out_tomo[rnd > 0] = 2
        disperse_io.save_numpy(out_tomo, file_name)

    # Pickling the object state
    # fname: full path for the pickle file
    def pickle(self, fname):
        pkl_f = open(fname, 'w')
        try:
            pickle.dump(self, pkl_f)
        finally:
            pkl_f.close()

    # Computes the average of all rotation of the initial template
    # sg: sigma (nm) for Gaussian low pass filtering, if None (default) it is set to resolution
    # Returns: Gaussian filtered average or None if no template
    def temp_rot_avg(self, sg=None):

        # Initialization
        if self.__temp is None:
            return None
        if sg is None:
            s_sig = 1.
        else:
            s_sig = sg / self.__res

        # Averaging
        if callable(getattr(self.__simulator.__class__, 'get_rnd_rots_array')):
            temp_rots = self.__simulator.get_rnd_rots_array()
        else:
            return None
        temp_avg = temp_rots.sum(axis=3)

        # Low-pass filtering
        return sp.ndimage.filters.gaussian_filter(temp_avg, s_sig)


    ##### External computations

    # First order G function
    # max_d: maximum distance for computing the histograms in nm
    # bins: number of samples for output (default 50), it MUST be greater than the number
    # of pattern points
    # Returns: the pair with distance sampling and G functions are returned
    def compute_G(self, max_d, bins=50):
        # if bins > self.__n_coords:
        #     error_msg = 'Number of bins ' + str(bins) + \
        #                 ' bigger than the number of coords ' + str(self.__n_coords)
        #     raise pexceptions.PySegInputWarning(expr='compute_G (UniStat)', msg=error_msg)
        dists = nnde(self.__coords)
        sp, cdf = compute_cdf(dists, bins, max_d)
        return sp, cdf

    # First order G function
    # max_d: maximum distance for computing the histograms in nm
    # bins: number of samples for output (default 50), it MUST be greater than the number
    # of pattern points
    # f_npts: number of samples for function F, if <0 it is set as the number of input points (default)
    # Returns: the pair with distance sampling and F functions are returned
    def compute_F(self, max_d, bins=50, f_npts=-1):
        if bins > self.__n_coords:
            error_msg = 'Number of bins ' + str(bins) + \
                        ' bigger than the number of coords ' + str(self.__n_coords)
            raise pexceptions.PySegInputError(expr='compute_F (UniStat)', msg=error_msg)
        if f_npts < 0:
            f_npts = self.__coords.shape[0]
        simulator = CSRSimulator()
        r_coords = simulator.gen_rand_in_mask(f_npts, self.__mask)
        r_coords *= self.__res
        dists = cnnde(r_coords, self.__coords)
        sp, cdf = compute_cdf(dists, bins, max_d)
        return sp, cdf

    # Random simulation of G function
    # max_d: maximum distance for computing the histograms in nm
    # bins: number of samples for output (default 50), it MUST be greater than the number
    # of pattern points
    # n_sim: number of simulations (default 100)
    # p: percentile, if None (default) only 50% is computed, otherwise p, 50, 100-p are computed
    # get_sim: if True (default False) all simulations are returned in an array instead the envelopes
    # Returns: the pair with distance sampling, function G and the percentiles envelopes
    # of the simulations (50, p, 100-p)
    def simulate_G(self, max_d, bins=50, n_sim=100, p=None, get_sim=False):

        # if bins > self.__n_coords:
        #     error_msg = 'Number of bins ' + str(bins) + \
        #                 ' bigger than the number of coords ' + str(self.__n_coords)
        #     raise pexceptions.PySegInputWarning(expr='simulate_G (UniStat)', msg=error_msg)

        # Initialization
        sims = np.zeros(shape=(bins, n_sim), dtype=np.float)

        # Multi-threading
        n_th = mp.cpu_count()
        threads = list()
        # Static simulations division
        spl_ids = np.array_split(np.arange(n_sim), n_th)
        for ids in spl_ids:
            th = mt.Thread(target=self.__th_sim_function_G, args=(sims, max_d, bins, ids))
            th.start()
            threads.append(th)
        for th in threads:
            th.join()
        gc.collect()

        # Compute results
        sp, _ = self.compute_G(max_d, bins)
        if get_sim:
            return sp, np.asarray(sims, dtype=np.float)
        else:
            env_05 = func_envelope(sims, per=50)
            if p is None:
                return sp, env_05
            else:
                env_1 = func_envelope(sims, per=p)
                env_2 = func_envelope(sims, per=100-p)
                return sp, env_1, env_05, env_2

    # Random simulation of F function
    # max_d: maximum distance for computing the histograms in nm
    # bins: number of samples for output (default 50), it MUST be greater than the number
    # of pattern points
    # f_npts: number of samples for function F, if <0 it is set as the number of input points (default)
    # n_sim: number of simulations (default 100)
    # p: percentile, if None (default) only 50% is computed, otherwise p, 50, 100-p are computed
    # get_sim: if True (default False) all simulations are returned in an array instead the envelopes
    # Returns: the pair with distance sampling, function G and the percentiles envelopes
    # of the simulations (50, p, 100-p)
    def simulate_F(self, max_d, bins=50, f_npts=-1, n_sim=100, p=None, get_sim=False):

        # if bins > self.__n_coords:
        #     error_msg = 'Number of bins ' + str(bins) + \
        #                 ' bigger than the number of coords ' + str(self.__n_coords)
        #     raise pexceptions.PySegInputError(expr='simulate_F (UniStat)', msg=error_msg)

        # Initialization
        if f_npts < 0:
            f_npts = self.__coords.shape[0]
        sims = np.zeros(shape=(bins, n_sim), dtype=np.float)

        # Multi-threading
        n_th = mp.cpu_count()
        threads = list()
        # Static simulations division
        spl_ids = np.array_split(np.arange(n_sim), n_th)
        for ids in spl_ids:
            th = mt.Thread(target=self.__th_sim_function_F, args=(sims, max_d, bins, f_npts, ids))
            th.start()
            threads.append(th)
        for th in threads:
            th.join()
        gc.collect()

        # Compute results
        sp, _ = self.compute_F(max_d, bins, f_npts=f_npts)
        if get_sim:
            return sp, np.asarray(sims, dtype=np.float)
        else:
            env_05 = func_envelope(sims, per=50)
            if p is None:
                return sp, env_05
            else:
                env_1 = func_envelope(sims, per=p)
                env_2 = func_envelope(sims, per=100-p)
                return sp, env_1, env_05, env_2

    # Compute Ripley's K second order function
    # max_d: maximum distance
    # n_samp: number of samples
    # get_nd: in not False (default), K's numerator and denominators are also returned for computing densities
    # Returns: r (radius sampling) and O (Ripley's O values)
    def compute_K(self, max_d, n_samp=50, get_nd=False):

        # Initializing
        r = np.linspace(0, max_d, n_samp)
        r2 = r * r
        rc = r / self.__res
        r2c = rc * rc
        num = np.zeros(shape=r.shape, dtype=np.float)
        dem = np.zeros(shape=r.shape, dtype=np.float)

        # Multi-threading
        tlock = mt.Lock()
        n_th = mp.cpu_count()
        if n_th > self.__n_coords:
            n_th = self.__n_coords
        threads = list()
        # Static division in threads by points
        if self.__n_coords <= 0:
            if get_nd:
                return r, num,  num, self.__z*dem
            else:
                return r, num
        spl_ids = np.array_split(np.arange(self.__n_coords), n_th)
        for ids in spl_ids:
            th = mt.Thread(target=self.__th_function_K, args=(rc, r2c, num, dem, ids, tlock))
            th.start()
            threads.append(th)
        for th in threads:
            th.join()
        gc.collect()

        # Final computations
        dem[0] = 1.
        K = num / dem
        if self.__mode_2D:
            rm = np.pi * r2
        else:
            rm = (4./3.) * np.pi * r2 * r
        if get_nd:
            return r, (rm/(self.__lambda*self.__z))*K,  num, self.__z*dem
        else:
            return r, (rm/(self.__lambda*self.__z))*K

    # Compute Normalized Local Density (Shell_density/Total_density)
    # thick: shell thickness
    # max_d: maximum distance
    # n_samp: number of samples
    # gnorm: if True (default) the local density values are divided by global density for each tomogram
    # Returns: r (radius sampling) and fucntion value
    def compute_NLD(self, thick, max_d, n_samp=50, gnorm=True):

        # Initializing
        max_d_v = max_d / self.__res
        r = np.linspace(0, max_d, n_samp)
        rc = r / self.__res
        bin_s = list()
        thick_h = .5 * (float(thick)/self.__res)
        bin_s.append((rc[0], thick_h))
        for i in range(1, rc.shape[0]):
            h_l = rc[i] - thick_h
            if h_l < 0:
                h_l = 0
            bin_s.append((h_l, rc[i]+thick_h))
        num = np.zeros(shape=r.shape, dtype=np.float)
        dem = np.zeros(shape=r.shape, dtype=np.float)

        # Multi-threading
        tlock = mt.Lock()
        n_th = mp.cpu_count()
        if n_th > self.__n_coords:
            n_th = self.__n_coords
        threads = list()
        # Static division in threads by points
        if self.__n_coords <= 0:
            return r, num
        spl_ids = np.array_split(np.arange(self.__n_coords), n_th)
        for ids in spl_ids:
            th = mt.Thread(target=self.__th_function_NLD, args=(max_d_v, bin_s, num, dem, ids, tlock))
            th.start()
            threads.append(th)
        for th in threads:
            th.join()
        gc.collect()

        # Final computations
        dem[0] = 1.
        # print 'Sim r = ' + str(r)
        # print 'Sim max_d = ' + str(max_d_v)
        if gnorm:
            return r, (num/dem) / (float(self.get_n_points())/float(self.__mask_sum))
        else:
            return r, num / dem

    # Compute Redial Distribution Function (NLD without border effects)
    # thick: shell thickness (nm)
    # max_d: maximum distance (nm)
    # n_samp: number of samples
    # Returns: r (radius sampling) and fucntion value
    def compute_RBF(self, thick, max_d, n_samp=50):

        # Initializing
        rs = np.linspace(0, max_d, n_samp)
        num = np.zeros(shape=rs.shape, dtype=np.float)
        dem = np.zeros(shape=rs.shape, dtype=np.float)
        bin_s = list()
        thick_h = .5 * float(thick)
        bin_s.append((rs[0], thick_h))
        for i in range(1, rs.shape[0]):
            h_l = rs[i] - thick_h
            if h_l < 0:
                h_l = 0
            bin_s.append((h_l, rs[i]+thick_h))
            dem[i] = (4./3.) * np.pi * (bin_s[-1][1]**3-bin_s[-1][0]**3)

        # Multi-threading
        tlock = mt.Lock()
        n_th = mp.cpu_count()
        if n_th > self.__n_coords:
            n_th = self.__n_coords
        threads = list()
        # Static division in threads by points
        if self.__n_coords <= 0:
            return rs, num
        spl_ids = np.array_split(np.arange(self.__n_coords), n_th)
        for ids in spl_ids:
            th = mt.Thread(target=self.__th_function_RBF, args=(bin_s, num, ids, tlock))
            th.start()
            threads.append(th)
        for th in threads:
            th.join()
        gc.collect()

        # Final computations
        dem[0] = 1.
        num /= float(self.__n_coords)
        # print 'Sim r = ' + str(r)
        # print 'Sim max_d = ' + str(max_d_v)
        return rs, num/dem / (float(self.get_n_points())/float(self.__mask_sum))

    # Compute Ripley's O-ring second order function
    # pair_K: Ripley's K pair (sampling, K)
    # w: ring width (default 1)
    # Returns: O (Ripley's O values)
    def compute_O(self, pair_K, w=1):

        # Filtration
        r, K = pair_K
        n_s = r.shape[0]
        m = 1. / float(1.-n_s)
        c = 1. - m
        wn = m*w + c
        if wn < 0:
            wn = .0
        elif wn > 1:
            wn = 1.
        b, a = butter(LP_ORDER, wn, btype='low', analog=False)
        f_K = lfilter(b, a, K)

        # Derivation
        K_d = np.gradient(f_K, r[1]-r[0])

        # Packing output
        o_ring = np.ones(shape=K_d.shape, dtype=np.float)
        # TODO: (review) 2D-circle has perimeter 2*pi*r then 3D-shpere has surface 4*pi*r*r
        if self.__mode_2D:
            o_ring[1:] = (1./(2*np.pi*r[1:])) * K_d[1:] - 1.
        else:
            r_hold = r[1:]
            o_ring[1:] = (1./(4*np.pi*r_hold*r_hold)) * K_d[1:] - 1.

        return o_ring

    # Compute linearized Ripley's K second order function
    # pair_K: Ripley's K pair (sampling, K)
    # w: smoothing factor
    # Returns: L (Ripley's O values)
    def compute_L(self, pair_K, w=1):

        # Filtration
        r, K = pair_K
        n_s = r.shape[0]
        if w > 1:
            m = 1. / float(1.-n_s)
            c = 1. - m
            wn = m*w + c
            if wn < 0:
                wn = .0
            elif wn > 1:
                wn = 1.
            b, a = butter(LP_ORDER, wn, btype='low', analog=False)
            f_K = lfilter(b, a, K)
        else:
            f_K = K

        # Computation
        if self.__mode_2D:
            return np.sqrt(f_K/np.pi) - r
        else:
            return np.power(f_K/((4./3.)*np.pi), CSQRT_EXP) - r

    # Compute function W (aka local intensity estimation for an input size)
    # rad: radius for locality in nm
    # n_samp: number of samples
    # Returns: a 3-tuple of arrays, r (local intensity sampling), W values and image with local intensity values
    def compute_W(self, rad, n_samp=50):

        # Initializing
        ids_x, ids_y, ids_z = self.__X[self.__mask], self.__Y[self.__mask], self.__Z[self.__mask]
        ids_i = np.arange(self.__mask_sum)
        arr = np.zeros(shape=self.__mask_sum, dtype=np.float)
        img = np.zeros(shape=self.__mask.shape, dtype=np.float)

        # Multi-threading
        n_th, s_z = mp.cpu_count(), self.__mask.shape[2]
        if n_th > s_z:
            n_th = s_z
        threads = list()

        # Static division in threads by Z-axis (multi-threading is not necessary for 2D data)
        spl_i = np.array_split(ids_i, n_th)
        spl_x, spl_y, spl_z = np.array_split(ids_x, n_th), np.array_split(ids_y, n_th), np.array_split(ids_z, n_th)
        for (id_x, id_y, id_z, id_i) in zip(spl_x, spl_y, spl_z, spl_i):
            th = mt.Thread(target=self.__th_function_W, args=(rad, arr, img, id_x, id_y, id_z, id_i))
            th.start()
            threads.append(th)
        for th in threads:
            th.join()
        gc.collect()

        if self.is_2D():
            # Image resize for converting from pixel to nm
            img_r = cv2.resize(img, (int(round(img.shape[1]*self.__res)), int(round(img.shape[0]*self.__res))),
                               interpolation=cv2.INTER_CUBIC)
            img_r = np.rot90(img_r)
        else:
            img_r = img

        # Final computations (histogram)
        bins = np.linspace(0, arr.max(), n_samp+1)
        hist, _ = np.histogram(arr, bins=bins)
        return .5*bins[1] + bins[:-1], hist.astype(np.float)/float(hist.sum()), img_r

    # Radial Averaged Fourier Transform
    # max_d: maximum distance (nm)
    # tr: template minimum radius (nm), if None default it is set to resolution
    # mode: valid: 'abs' (default), 'ang', 'real' and 'imag'
    def compute_RAFT(self, max_d, tr=None, mode='abs'):

        # Initializing
        max_d_v = math.ceil(max_d / self.__res)
        rs = (np.arange(max_d_v, dtype=np.float) * self.__res) / max_d
        if (tr is None) or (tr <= 0):
            tr = self.__res
        tr = float(tr)

        # Getting dense tomogram
        dense = self.__dense.astype(np.float)
        if self.__temp is not None:
            temp = self.temp_rot_avg(tr)
            dense = sp.signal.fftconvolve(dense, temp, mode='same')

        # Force mask to be even
        off_x, off_y, off_z = self.__mask.shape
        if (off_x > 1) and (off_x%2 != 0):
            off_x -= 1
        if (off_y > 1) and (off_y%2 != 0):
            off_y -= 1
        if (off_z > 1) and (off_z%2 != 0):
            off_z -= 1
        mask = self.__mask[:off_x, :off_y, :off_z]

        # Multi-processing
        n_pr = mp.cpu_count()
        if n_pr > self.__n_coords:
            n_pr = self.__n_coords
        processes = list()
        num = np.zeros(shape=len(mpa), dtype=np.float)
        manager = mp.Manager()
        mpa = manager.list()
        for pr_id in n_pr:
            mpa.append(list())

        # Static division in process by points
        if self.__n_coords <= 0:
            return rs, num
        spl_ids = np.array_split(np.arange(self.__n_coords), n_pr)
        for ids in spl_ids:
            pr = mp.Process(target=pr_uni_RAFT, args=(self.__n_coords, ids, dense, mask, mpa))
            pr.start()
            processes.append(pr)
        for pr in processes:
            pr.join()
        gc.collect()

        # Final computations (Gathering)
        for hold_num in mpa:
            num += hold_num

        return rs, num/float(len(mpa))


    # Simulation for CSR with Ripley's K
    # max_d: maximum distance
    # n_samp: number of samples
    # sp: distance sampling in nm, floor to class resolution
    # n_sim: number of simulations (default 100)
    # p: percentile, if None (default) only 50% is computed, otherwise p, 50, 100-p are computed
    # get_sim: if True (default False) all simulations are returned in an array instead the envelopes
    # Returns: the pair with distance sampling, function K and the percentiles envelopes
    # of the simulations (50, p, 100-p) if get_sim is True, otherwise an array with simulations is returned
    def simulate_K(self, max_d, n_samp=50, n_sim=100, p=None, get_sim=False):

        # Initialization
        r = np.linspace(0, max_d, n_samp)
        r2 = r * r
        rc = r / self.__res
        r2c = rc * rc
        sims = np.zeros(shape=(r.shape[0], n_sim), dtype=np.float)
        nums = np.zeros(shape=(r.shape[0], n_sim), dtype=np.float)
        dems = np.zeros(shape=(r.shape[0], n_sim), dtype=np.float)

        # Multi-threading
        tlock = mt.Lock()
        n_th = mp.cpu_count()
        if n_th > n_sim:
            n_th = n_sim
        threads = list()
        # Loop for simulations
        for i in range(n_sim):

            # Generate a new random pattern
            hold_coords = self.__simulator.gen_rand_in_mask(self.__n_coords, self.__mask)
            dense, coords = coords_to_dense_mask(hold_coords, self.__mask)
            num = np.zeros(shape=r.shape, dtype=np.float)
            dem = np.zeros(shape=r.shape, dtype=np.float)

            # Static division in threads by points
            if len(coords) > 0:
                spl_ids = np.array_split(np.arange(coords.shape[0]), n_th)
                for ids in spl_ids:
                    th = mt.Thread(target=self.__th_sim_function_K, args=(rc, r2c, num, dem, ids, tlock,
                                                                            coords, dense))
                    th.start()
                    threads.append(th)
                for th in threads:
                    th.join()

            # Store simulation results
            if self.__mode_2D:
                rm = np.pi * r2
            else:
                rm = (4./3.) * np.pi * r2 * r
            dem[0] = coords.shape[0]
            sims[:, i] = (rm/(float(coords.shape[0])/float(self.__mask_sum))) * (num / dem)
            nums[:, i] = num
            dems[:, i] = dem

            gc.collect()

        # Compute results
        if get_sim:
            return r, np.asarray(sims, dtype=np.float), nums, dems
        else:
            env_05 = func_envelope(sims, per=50)
            if p is None:
                return r, env_05
            else:
                env_1 = func_envelope(sims, per=p)
                env_2 = func_envelope(sims, per=100-p)
                return r, env_1, env_05, env_2


    # Simulation for NLD
    # thick: shell thickness
    # max_d: maximum distance
    # n_samp: number of samples
    # sp: distance sampling in nm, floor to class resolution
    # n_sim: number of simulations (default 100)
    # p: percentile, if None (default) only 50% is computed, otherwise p, 50, 100-p are computed
    # get_sim: if True (default False) all simulations are returned in an array instead the envelopes
    # gnorm: if True (default) the local density values are divided by global density for each tomogram
    # Returns: the pair with distance sampling, function K and the percentiles envelopes
    # of the simulations (50, p, 100-p) if get_sim is True, otherwise an array with simulations is returned
    def simulate_NLD(self, thick, max_d, n_samp=50, n_sim=100, p=None, get_sim=False, gnorm=True):

        # Initialization
        max_d_v = max_d / self.__res
        r = np.linspace(0, max_d, n_samp)
        rc = r / self.__res
        bin_s = list()
        thick_h = .5 * (float(thick)/self.__res)
        bin_s.append((rc[0], thick_h))
        for i in range(1, rc.shape[0]):
            h_l = rc[i] - thick_h
            if h_l < 0:
                h_l = 0
            bin_s.append((h_l, rc[i]+thick_h))
        sims = np.zeros(shape=(r.shape[0], n_sim), dtype=np.float)

        # Multi-threading
        tlock = mt.Lock()
        n_th = mp.cpu_count()
        if n_th > n_sim:
            n_th = n_sim
        threads = list()
        # Loop for simulations
        for i in range(n_sim):

            # Generate a new random pattern
            hold_coords = self.__simulator.gen_rand_in_mask(self.__n_coords, self.__mask)
            dense, coords = coords_to_dense_mask(hold_coords, self.__mask)
            num = np.zeros(shape=r.shape, dtype=np.float)
            dem = np.zeros(shape=r.shape, dtype=np.float)

            # Static division in threads by points
            if len(coords) > 0:
                spl_ids = np.array_split(np.arange(coords.shape[0]), n_th)
                for ids in spl_ids:
                    th = mt.Thread(target=self.__th_sim_function_NLD, args=(max_d_v, bin_s, num, dem, ids, tlock,
                                                                            coords, dense))
                    th.start()
                    threads.append(th)
                for th in threads:
                    th.join()

            # Store simulation results
            dem[0] = coords.shape[0]
            if gnorm:
                sims[:, i] = (num/dem) / (float(coords.shape[0])/float(self.__mask_sum))
            else:
                sims[:, i] = num / dem

            gc.collect()

        # Compute results
        # print 'Sim r = ' + str(r)
        # print 'Sim max_d = ' + str(max_d_v)
        if get_sim:
            return r, sims
        else:
            env_05 = func_envelope(sims, per=50)
            if p is None:
                return r, env_05
            else:
                env_1 = func_envelope(sims, per=p)
                env_2 = func_envelope(sims, per=100-p)
                return r, env_1, env_05, env_2


    # Analyze local intensity distribution
    # rads: list with the radius to analyze
    # bins: number of samples for output (default 50)
    # block: if True (default) the code stops until plotted window is closed
    # out_file: path to file to store the analysis, if None (default) is not stored
    # legend: if True (default False) and out_file is not None, a legend is also stored
    # pimgs: if True (default False) images with local intensity are also plotter and/or stored
    def analyze_W(self, rads, bins=50, block=True, out_file=None, legend=False, pimgs=False):

        # Initialization
        if (not hasattr(rads, '__len__')) and (len(rads) > 0):
            error_msg = 'analyze_W() requires a list of input radius.'
            raise pexceptions.PySegInputError(expr='analyze_W (UniStat)', msg=error_msg)

        # Computations
        Ws = [list(), list()]
        imgs = list()
        if self.is_2D():
            # Multi-threading
            threads = list()
            queues = list()
            for rad in rads:
                q = Queue.Queue()
                th = mt.Thread(target=self.__compute_W_q, args=(q, rad, bins))
                th.start()
                threads.append(th)
                queues.append(q)
            for (th, q) in zip(threads, queues):
                th.join()
                Ws[0].append(q.get())
                Ws[1].append(q.get())
                imgs.append(q.get())
        else:
            # In 3D the multi-threading is data intrisic
            for rad in rads:
                rds, ints, img = self.compute_W(rad, bins)
                Ws[0].append(rds)
                Ws[1].append(ints)
                imgs.append(img)
        gc.collect()

        # Plotting
        fig = plt.figure()
        fig.suptitle('W-Function ' + str(self.__name))
        colors = cm.rainbow(np.linspace(0, 1, len(Ws[0])))
        ax = fig.gca()
        for (rd, rds, ints, c) in zip(rads, Ws[0], Ws[1], colors):
            ax.plot(rds, ints, c=c, label=str(rd)+' nm')
        ax.set_ylabel('Frequency')
        ax.set_xlabel('Local intensity (points/nm^2)')
        # Get legend data
        if legend:
            handles, labels = ax.get_legend_handles_labels()

        # Show data
        if block:
            plt.show(block=True)
        if out_file is not None:
            fig.savefig(out_file)
        # Store legend
        if legend and (out_file is not None):
            store_legend(handles, labels, out_file)
        if block:
            plt.close(fig)

        if pimgs:
            # Plotting images
            for (img, rad) in zip(imgs, rads):
                figh = plt.figure()
                plt.title('Local intensity for scale ' + str(rad) + ' nm')
                plt.imshow(img, vmin=0, vmax=img.max(), cmap='jet')
                plt.xlabel('X (nm)')
                plt.ylabel('Y (nm)')
                plt.colorbar()
                plt.show(block=block)
                f_path, f_name = os.path.split(out_file)
                f_stem, f_ext = os.path.splitext(f_name)
                if out_file is not None:
                     plt.savefig(f_path+'/'+f_stem+'_lint_'+str(rad)+f_ext)
                plt.close(figh)

    # Plot with scatter the coordinates of the points in a figure
    # block: if True (default) the code stops until plotted window is closed
    # out_file: path to file to store the figure, if None (default) is not stored and figure
    #           blocks independently of 'block' parameters value
    def plot_points(self, block=True, out_file=None):

        # Initialization
        if out_file is None:
            block = True
        fig = plt.figure()

        if self.is_2D():
            ax = fig.add_subplot(111)
            ax.scatter(self.__coords[:, 0], self.__coords[:, 1])
            ax.set_xlabel('X (nm)')
            ax.set_ylabel('Y (nm)')
            ax.set_xlim([self.__coords[:, 0].min(), self.__coords[:, 0].max()])
            ax.set_ylim([self.__coords[:, 1].min(), self.__coords[:, 1].max()])
        else:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.__coords[:, 0], self.__coords[:, 1], self.__coords[:, 2])
            ax.set_xlabel('X (nm)')
            ax.set_ylabel('Y (nm)')
            ax.set_zlabel('Z (nm)')
            ax.set_xlim([self.__coords[:, 0].min(), self.__coords[:, 0].max()])
            ax.set_ylim([self.__coords[:, 1].min(), self.__coords[:, 1].max()])
            ax.set_zlim([self.__coords[:, 2].min(), self.__coords[:, 2].max()])

        # Show data
        ax.set_aspect('equal', 'datalim')
        plt.title('Points scatter ' + str(self.__name))
        plt.show(block=block)
        if out_file is not None:
            f_ext = os.path.splitext(out_file)[1]
            if f_ext == '.vtp':
                disperse_io.save_vtp(self.to_vtp(), out_file)
            else:
                plt.savefig(out_file)
        plt.close(fig)


    # Compute homogeneity (aka local intensity) for a given scale
    # scale: scale in nm
    # Returns: a tomo with the lacal intensity
    def compute_homo(self, scale):

        # Initializing
        ids_x, ids_y, ids_z = self.__X[self.__mask], self.__Y[self.__mask], self.__Z[self.__mask]
        ids_i = np.arange(self.__mask_sum)
        arr = np.zeros(shape=self.__mask_sum, dtype=np.float)
        img = np.zeros(shape=self.__mask.shape, dtype=np.float)

        # Multi-threading
        n_th, s_z = mp.cpu_count(), self.__mask.shape[2]
        if n_th > s_z:
            n_th = s_z
        threads = list()

        # Static division in threads by Z-axis (multi-threading is not necessary for 2D data)
        spl_i = np.array_split(ids_i, n_th)
        spl_x, spl_y, spl_z = np.array_split(ids_x, n_th), np.array_split(ids_y, n_th), np.array_split(ids_z, n_th)
        for (id_x, id_y, id_z, id_i) in zip(spl_x, spl_y, spl_z, spl_i):
            th = mt.Thread(target=self.__th_function_W, args=(scale, arr, img, id_x, id_y, id_z, id_i))
            th.start()
            threads.append(th)
        for th in threads:
            th.join()
        gc.collect()

        return img

    # Generate an vtp file with the points
    def to_vtp(self):

        # Initialization
        poly = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        verts = vtk.vtkCellArray()

        # Adding geometry and topology
        for i, coord in enumerate(self.__coords_d):
            points.InsertPoint(i, coord)
            verts.InsertNextCell(1)
            verts.InsertCellPoint(i)

        # Finalizing poly
        poly.SetPoints(points)
        poly.SetVerts(verts)

        return poly

    ##### Internal computations

    # Distance transform on a coordinate point
    # point: 3D point array
    # max_d: for cropping distances (in voxels)
    # Returns: the squared cropped distance, the bounding box
    # ((x.m,x_M), (y_m,y_M), (y_m,y_M)) and the point coordinates in the subvolume in voxels
    def __point_distance_trans(self, point, max_d):

        # Computing bounding box
        max_d_v = int(math.ceil(max_d))
        max_d_v_m1, max_d_v_1 = max_d_v-CROP_OFF, max_d_v-CROP_OFF+1
        box_x_l, box_x_h = int(round(point[0]-max_d_v_m1)), int(round(point[0]+max_d_v_1))
        box_y_l, box_y_h = int(round(point[1]-max_d_v_m1)), int(round(point[1]+max_d_v_1))
        box_z_l, box_z_h = int(round(point[2]-max_d_v_m1)), int(round(point[2]+max_d_v_1))
        if box_x_l < 0:
            box_x_l = 0
        if box_x_h > self.__mx_b:
            box_x_h = self.__mx_b
        if box_y_l < 0:
            box_y_l = 0
        if box_y_h > self.__my_b:
            box_y_h = self.__my_b
        if box_z_l < 0:
            box_z_l = 0
        if box_z_h > self.__mz_b:
            box_z_h = self.__mz_b

        # Distance computation
        hold_x = self.__X[box_x_l:box_x_h, box_y_l:box_y_h, box_z_l:box_z_h] - point[0]
        hold_y = self.__Y[box_x_l:box_x_h, box_y_l:box_y_h, box_z_l:box_z_h] - point[1]
        hold_z = self.__Z[box_x_l:box_x_h, box_y_l:box_y_h, box_z_l:box_z_h] - point[2]

        return (hold_x*hold_x + hold_y*hold_y + hold_z*hold_z), \
               ((box_x_l, box_x_h), (box_y_l, box_y_h), (box_z_l, box_z_h)), \
                (point[0]-box_x_l, point[1]-box_y_l, point[2]-box_z_l)

    # Thread for function G simulation
    def __th_sim_function_G(self, sims, max_d, bins, ids):

        # Loop until all simulations are done
        for idx in ids:

            # Generate random coordinates
            r_coords = self.__simulator.gen_rand_in_mask(self.__n_coords, self.__mask)
            r_coords *= self.__res

            # Compute function G
            dists = nnde(r_coords)
            _, cdf = compute_cdf(dists, bins, max_d)

            # Store results
            sims[:, idx] = cdf

    # Thread for function G simulation
    def __th_sim_function_F(self, sims, max_d, bins, f_npts, ids):

        for idx in ids:

            # Compute function G
            simulator = CSRSimulator()
            rref_coords = simulator.gen_rand_in_mask(self.__n_coords, self.__mask)
            r1_coords = self.__simulator.gen_rand_in_mask(f_npts, self.__mask)
            r1_coords *= self.__res
            rref_coords *= self.__res
            dists = cnnde(r1_coords, rref_coords)
            _, cdf = compute_cdf(dists, bins, max_d)

            # Store results
            sims[:, idx] = cdf

    # Thread for function K (efficiency improvement of __th_funciton_K())
    def __th_function_K(self, r, r2, num, dem, ids, tlock):

        # Loop for points
        for idx in ids:

            # Distance transform
            point = self.__coords_d[idx]
            sub_dists, box, pt_s = self.__point_distance_trans(point, r[-1])
            sub_dense = np.copy(self.__dense[box[0][0]:box[0][1], box[1][0]:box[1][1], box[2][0]:box[2][1]])
            sub_mask = self.__mask[box[0][0]:box[0][1], box[1][0]:box[1][1], box[2][0]:box[2][1]]

            # Intermediate subvolumes
            if not sub_dense[int(pt_s[0]), int(pt_s[1]), int(pt_s[2])]:
                print 'WARNING (__th_function_K) : unexepected event in point ' + str(pt_s)
            sub_dense[int(pt_s[0]), int(pt_s[1]), int(pt_s[2])] = False
            dense_ids = sub_dense * sub_mask

            # Computing CDFs
            hold_dem, _ = np.histogram(sub_dists[sub_mask], bins=r2)
            hold_dem = np.cumsum(hold_dem)
            hold_num, _ = np.histogram(sub_dists[dense_ids], bins=r2)
            hold_num = np.cumsum(hold_num)

            # Update the shared arrays
            tlock.acquire()
            num[1:] += hold_num
            dem[1:] += hold_dem
            tlock.release()

    # Thread for function NLD
    def __th_function_NLD(self, max_r, bin_s, num, dem, ids, tlock):

        # Loop for points
        for idx in ids:

            # Distance transform
            point = self.__coords_d[idx]
            sub_dists, box, pt_s = self.__point_distance_trans(point, max_r)
            sub_dense = np.copy(self.__dense[box[0][0]:box[0][1], box[1][0]:box[1][1], box[2][0]:box[2][1]])
            sub_mask = self.__mask[box[0][0]:box[0][1], box[1][0]:box[1][1], box[2][0]:box[2][1]]

            # Intermediate subvolumes
            if not sub_dense[pt_s[0], pt_s[1], pt_s[2]]:
                print 'WARNING (__th_function_NSD) : unexepected event in point ' + str(pt_s)
            sub_dense[pt_s[0], pt_s[1], pt_s[2]] = False
            dense_ids = sub_dense * sub_mask

            # Count num of particles in the shell and shell volume
            point_dists = np.sqrt(sub_dists[dense_ids])
            shell_dist = np.sqrt(sub_dists[sub_mask])
            hold_num = np.zeros(shape=len(bin_s), dtype=np.float)
            hold_dem = np.zeros(shape=len(bin_s), dtype=np.float)
            for i in range(1, len(bin_s)):
                hold_num[i] = float(((point_dists>=bin_s[i][0]) & (point_dists<bin_s[i][1])).sum())
                hold_dem[i] = float(((shell_dist>=bin_s[i][0]) & (shell_dist<bin_s[i][1])).sum())

            # Update the shared arrays
            tlock.acquire()
            num[1:] += hold_num[1:]
            dem[1:] += hold_dem[1:]
            tlock.release()

    # Thread for function RBF
    def __th_function_RBF(self, bin_s, num, ids, tlock):

        # Loop for points
        for idx in ids:

            # Get current point
            point = self.__coords[idx]

            # Compute distances to neighbours
            hold = point - self.__coords
            dsts = np.sqrt((hold * hold).sum(axis=1))

            # Count num of particles in the shell and shell volume
            hold_num = np.zeros(shape=len(bin_s), dtype=np.float)
            for i in range(1, len(bin_s)):
                hold_num[i] = float(((dsts>=bin_s[i][0]) & (dsts<bin_s[i][1])).sum())

            # Update the shared arrays
            tlock.acquire()
            num[1:] += hold_num[1:]
            tlock.release()

    # Thread for function K simulation (efficiency improvement of __th_sim_function_K())
    def __th_sim_function_K(self, r, r2, num, dem, ids, tlock, coords, dense):

        # Loop for points
        for idx in ids:

            # Distance transform
            point = coords[idx]
            sub_dists, box, pt_s = self.__point_distance_trans(point, r[-1])
            sub_dense = np.copy(dense[box[0][0]:box[0][1], box[1][0]:box[1][1], box[2][0]:box[2][1]])
            sub_mask = self.__mask[box[0][0]:box[0][1], box[1][0]:box[1][1], box[2][0]:box[2][1]]

            # Intermediate subvolumes
            sub_dense[int(pt_s[0]), int(pt_s[1]), int(pt_s[2])] = False
            dense_ids = sub_dense * sub_mask
            # sub_dense_d = sub_dense * sub_dists

            # Computing CDFs
            hold_dem, _ = np.histogram(sub_dists[sub_mask], bins=r2)
            hold_dem = np.cumsum(hold_dem)
            hold_num, _ = np.histogram(sub_dists[dense_ids], bins=r2)
            hold_num = np.cumsum(hold_num)

            # Update the shared arrays
            tlock.acquire()
            num[1:] += hold_num
            dem[1:] += hold_dem
            tlock.release()

    # Thread for function K simulation (efficiency improvement of __th_sim_function_K())
    def __th_sim_function_NLD(self, max_r, bin_s, num, dem, ids, tlock, coords, dense):

        # Loop for points
        for idx in ids:

            # Distance transform
            point = coords[idx]
            sub_dists, box, pt_s = self.__point_distance_trans(point, max_r)
            sub_dense = np.copy(dense[box[0][0]:box[0][1], box[1][0]:box[1][1], box[2][0]:box[2][1]])
            sub_mask = self.__mask[box[0][0]:box[0][1], box[1][0]:box[1][1], box[2][0]:box[2][1]]

            # Intermediate subvolumes
            sub_dense[pt_s[0], pt_s[1], pt_s[2]] = False
            dense_ids = sub_dense * sub_mask
            # sub_dense_d = sub_dense * sub_dists

            # Count num of particles in the shell and shell volume
            point_dists = np.sqrt(sub_dists[dense_ids])
            shell_dist = np.sqrt(sub_dists[sub_mask])
            hold_num = np.zeros(shape=len(bin_s), dtype=np.float)
            hold_dem = np.zeros(shape=len(bin_s), dtype=np.float)
            for i in range(1, len(bin_s)):
                hold_num[i] = float(((point_dists>=bin_s[i][0]) & (point_dists<bin_s[i][1])).sum())
                hold_dem[i] = float(((shell_dist>=bin_s[i][0]) & (shell_dist<bin_s[i][1])).sum())

            # Update the shared arrays
            tlock.acquire()
            num[1:] += hold_num[1:]
            dem[1:] += hold_dem[1:]
            tlock.release()

    # Thread for function W
    def __th_function_W(self, rad, arr, img, ids_x, ids_y, ids_z, ids_i):

        rad_d = rad / self.__res
        rad_d *= rad_d

        # Loop for points
        for (x, y, z, i) in zip(ids_x, ids_y, ids_z, ids_i):

            # Distance transform
            point = np.asarray((x, y, z))
            sub_dists, box, pt_s = self.__point_distance_trans(point, rad_d)
            sub_dense = self.__dense[box[0][0]:box[0][1], box[1][0]:box[1][1], box[2][0]:box[2][1]]
            sub_mask = self.__mask[box[0][0]:box[0][1], box[1][0]:box[1][1], box[2][0]:box[2][1]]

            # # UNCOMMENT FOR DEBUGGING
            # if sub_dists[pt_s[0], pt_s[1], pt_s[2]] != 0:
            #     print 'Jol'

            region = sub_dists <= rad_d
            hold_mask = region * sub_mask
            hold_num = (sub_dense * hold_mask).sum()
            hold_dem = (hold_mask.sum() * self.__z)
            arr[i] = hold_num / hold_dem
            try:
                img[x, y, z] = arr[i]
            except IndexError:
                pass

    # Compute function W (aka local intensity estimation for an input size) and store the results in thread safe Queue
    # q: Queue were data will be stored
    # rad: radius for locality in nm
    # n_samp: number of samples
    def __compute_W_q(self, q, rad, n_samp=50):

        # Initializing
        ids_x, ids_y, ids_z = self.__X[self.__mask], self.__Y[self.__mask], self.__Z[self.__mask]
        ids_i = np.arange(self.__mask_sum)
        arr = np.zeros(shape=self.__mask_sum, dtype=np.float)
        img = np.zeros(shape=self.__mask.shape, dtype=np.float)

        # Multi-threading
        n_th, s_z = mp.cpu_count(), self.__mask.shape[2]
        if n_th > s_z:
            n_th = s_z
        threads = list()

        # Static division in threads by Z-axis (multi-threading is not necessary for 2D data)
        spl_i = np.array_split(ids_i, n_th)
        spl_x, spl_y, spl_z = np.array_split(ids_x, n_th), np.array_split(ids_y, n_th), np.array_split(ids_z, n_th)
        for (id_x, id_y, id_z, id_i) in zip(spl_x, spl_y, spl_z, spl_i):
            th = mt.Thread(target=self.__th_function_W, args=(rad, arr, img, id_x, id_y, id_z, id_i))
            th.start()
            threads.append(th)
        for th in threads:
            th.join()
        gc.collect()

        # Image resize for converting from pixel to nm
        img_r = cv2.resize(img, (int(round(img.shape[1]*self.__res)), int(round(img.shape[0]*self.__res))),
                           interpolation=cv2.INTER_CUBIC)
        img_r = np.rot90(img_r)

        # Final computations (histogram)
        bins = np.linspace(0, arr.max(), n_samp+1)
        hist, _ = np.histogram(arr, bins=bins)
        q.put(.5*bins[1] + bins[:-1])
        q.put(hist.astype(np.float)/float(hist.sum()))
        q.put(img_r)

##############################################################################################
# Class for bivariate statistical spatial analysis
#
class BiStat(object):

    ##### Constructor area

    # coords_1|2: 2D or 3D coordinates (in voxels) for patterns 1 and 2 respectively,
    #             coords_1 is the reference pattern
    # mask: 2D or 3D mask
    # res: voxel resolution (nm/pixel)
    # name: string with the name of the patterns (default None)
    def __init__(self, coords_1, coords_2, mask, res, name=None):
        self.__name = name
        self.__mode_2D = False
        if coords_1.shape[1] != coords_2.shape[1]:
            error_msg = 'Input coords must have the same dimensions'
            raise pexceptions.PySegInputError(expr='__init__ (BiStat)', msg=error_msg)
        if coords_1.shape[1] == 3:
            hold_coords_1, hold_coords_2 = coords_1.astype(np.float), coords_2.astype(np.float)
            if (hold_coords_1[:, 2].max() <= 1) and (hold_coords_2[:, 2].max() <= 1):
                self.__mode_2D = True
        elif coords_1.shape[1] == 2:
            hold_coords_1 = np.zeros(shape=(coords_1.shape[0], 3), dtype=np.float)
            hold_coords_1[:, 0] = coords_1[:, 0]
            hold_coords_1[:, 1] = coords_1[:, 1]
            hold_coords_2 = np.zeros(shape=(coords_2.shape[0], 3), dtype=np.float)
            hold_coords_2[:, 0] = coords_2[:, 0]
            hold_coords_2[:, 1] = coords_2[:, 1]
            self.__mode_2D = True
        else:
            error_msg = 'Input coords must be 2D or 3D!'
            raise pexceptions.PySegInputError(expr='__init__ (BiStat)', msg=error_msg)
        if len(mask.shape) == 3:
            self.__mask = mask.astype(np.bool)
        elif len(mask.shape) == 2:
            self.__mask = np.zeros(shape=(mask.shape[0], 3), dtype=np.bool)
            self.__mask[:, 0] = mask[:, 0]
            self.__mask[:, 1] = mask[:, 1]
        else:
            error_msg = 'Input coords must be 2D or 3D!'
            raise pexceptions.PySegInputError(expr='__init__ (BiStat)', msg=error_msg)
        self.__z = float(res * res)
        if not self.__mode_2D:
            self.__z *= float(res)
        self.__dense_1, self.__coords_d_1 = coords_to_dense_mask(hold_coords_1, self.__mask)
        self.__coords_1 = self.__coords_d_1 * res
        self.__n_coords_1 = self.__coords_1.shape[0]
        self.__dense_2, self.__coords_d_2 = coords_to_dense_mask(hold_coords_2, self.__mask)
        self.__coords_2 = self.__coords_d_2 * res
        self.__n_coords_2 = self.__coords_2.shape[0]
        self.__res = res
        self.__mask_sum = self.__mask.sum()
        hold_den = self.__mask_sum * self.__z
        self.__lambda_1 = self.__n_coords_1 / hold_den
        self.__lambda_2 = self.__n_coords_2 / hold_den
        # Internal variables for computation acceleration
        sap = self.__mask.shape
        self.__Y, self.__X, self.__Z = np.meshgrid(np.arange(sap[1]).astype(np.int16),
                                                   np.arange(sap[0]).astype(np.int16),
                                                   np.arange(sap[2]).astype(np.int16),
                                                   copy=False)
        self.__mx_b, self.__my_b, self.__mz_b = sap[0], sap[1], sap[2]
        # Setting random simulator: by default CSR but it can take into account particle shape if required
        self.__simulator = CSRSimulator()

    ##### Set/Get functionality

    def set_name(self, name):
        self.__name = name

    def get_name(self):
        return self.__name

    def get_intensities(self):
        return self.__lambda_1, self.__lambda_2

    def get_ns_points(self):
        return self.__n_coords_1, self.__n_coords_2

    ##### External functionality

    def is_2D(self):
        return self.__mode_2D

    # Store a tomogram with the specified point pattern
    # file_name: full path to the stored tomogram
    # pattern: if 1 (default) pattern 1 is stored, otherwise pattern 2
    def save_sparse(self, file_name, pattern=1):
        if pattern == 1:
            disperse_io.save_numpy(self.__dense_1, file_name)
        else:
            disperse_io.save_numpy(self.__dense_2, file_name)

    # Pickling the object state
    # fname: full path for the pickle file
    def pickle(self, fname):
        pkl_f = open(fname, 'w')
        try:
            pickle.dump(self, pkl_f)
        finally:
            pkl_f.close()

    ##### External computations

    # First order G function
    # max_d: maximum distance for computing the histograms in nm
    # bins: number of samples for output (default 50), it MUST be greater than the number
    # of pattern points
    # Returns: the pair with distance sampling and G functions are returned
    def compute_G(self, max_d, bins=50):
        if bins > self.__n_coords_2:
            error_msg = 'Number of bins ' + str(bins) + \
                        ' bigger than the number of coords ' + str(self.__n_coords_2)
            raise pexceptions.PySegInputError(expr='compute_G (BiStat)', msg=error_msg)
        dists = cnnde(self.__coords_2, self.__coords_1)
        sp, cdf = compute_cdf(dists, bins, max_d)
        return sp, cdf

    # Random simulation of G function
    # max_d: maximum distance for computing the histograms in nm
    # bins: number of samples for output (default 50), it MUST be greater than the number
    # of pattern points
    # n_sim: number of simulations (default 100)
    # p: percentile, if None (default) only 50% is computed, otherwise p, 50, 100-p are computed
    # get_sim: if True (default False) all simulations are returned in an array instead the envelopes
    # Returns: the pair with distance sampling, function G and the percentiles envelopes
    # of the simulations (50, p, 100-p)
    def simulate_G(self, max_d, bins=50, n_sim=100, p=None, get_sim=False):

        if bins > self.__n_coords_2:
            error_msg = 'Number of bins ' + str(bins) + \
                        ' bigger than the number of coords ' + str(self.__n_coords_2)
            raise pexceptions.PySegInputError(expr='simulate_G (BiStat)', msg=error_msg)

        # Initialization
        sims = np.zeros(shape=(bins, n_sim), dtype=np.float)

        # Multi-threading
        n_th = mp.cpu_count()
        threads = list()
        # Static simulations division
        spl_ids = np.array_split(np.arange(n_sim), n_th)
        for ids in spl_ids:
            th = mt.Thread(target=self.__th_sim_function_G, args=(sims, max_d, bins, ids))
            th.start()
            threads.append(th)
        for th in threads:
            th.join()
        gc.collect()

        # Compute results
        sp, _ = self.compute_G(max_d, bins)
        if get_sim:
            return sp, np.asarray(sims, dtype=np.float)
        else:
            env_05 = func_envelope(sims, per=50)
            if p is None:
                return sp, env_05
            else:
                env_1 = func_envelope(sims, per=p)
                env_2 = func_envelope(sims, per=100-p)
                return sp, env_1, env_05, env_2

    # Compute Ripley's K second order function
    # max_d: maximum distance
    # n_samp: number of samples
    # sp: distance sampling in nm, floor to class resolution
    # Returns: r (radius sampling) and O (Ripley's O values)
    def compute_K(self, max_d, n_samp=50):


        # Initializing
        r = np.linspace(0, max_d, n_samp)
        r2 = r * r
        rc = r / self.__res
        r2c = rc * rc
        num = np.zeros(shape=r.shape, dtype=np.float)
        dem = np.zeros(shape=r.shape, dtype=np.float)

        # Multi-threading
        tlock = mt.Lock()
        n_th = mp.cpu_count()
        if n_th > self.__n_coords_1:
            n_th = self.__n_coords_1
        threads = list()
        # Static division in threads by points
        spl_ids = np.array_split(np.arange(self.__n_coords_1), n_th)
        for ids in spl_ids:
            th = mt.Thread(target=self.__th_function_K_2, args=(rc, r2c, num, dem, ids, tlock))
            th.start()
            threads.append(th)
        for th in threads:
            th.join()
        gc.collect()

        # Final computations
        dem[0] = 1.
        # TODO: K_NUM_CORR is required to fit CSR unshaped simulations with theoretical CSR values, I don't know
        #       from where this values comes from
        K = (K_NUM_CORR * num) / dem
        if self.__mode_2D:
            rm = np.pi * r2
        else:
            rm = (4./3.) * np.pi * r2 * r
        return r, (rm/(self.__lambda_2*self.__z))*K

    # Compute Ripley's O-ring second order function
    # pair_K: Ripley's K pair (sampling, K)
    # w: ring width (default 1)
    # Returns: r (radius sampling) and O (Ripley's O values)
    def compute_O(self, pair_K, w=1):

        # Filtration
        r, K = pair_K
        n_s = r.shape[0]
        m = 1. / float(1.-n_s)
        c = 1. - m
        wn = m*w + c
        if wn < 0:
            wn = .0
        elif wn > 1:
            wn = 1.
        b, a = butter(LP_ORDER, wn, btype='low', analog=False)
        f_K = lfilter(b, a, K)

        # Derivation
        K_d = np.gradient(f_K, r[1]-r[0])

        # Packing output
        o_ring = np.ones(shape=K_d.shape, dtype=np.float)
        # TODO: (review) 2D-circle has perimeter 2*pi*r then 3D-sphere has surface 4*pi*r*r
        if self.__mode_2D:
            o_ring[1:] = (1./(2*np.pi*r[1:])) * K_d[1:] - 1.
        else:
            r_hold = r[1:]
            o_ring[1:] = (1./(4*np.pi*r_hold*r_hold)) * K_d[1:] - 1.

        return o_ring

    # Compute linearized Ripley's K second order function
    # pair_K: Ripley's K pair (sampling, K)
    # w: smoothing factor
    # Returns: r (radius sampling) and O (Ripley's O values)
    def compute_L(self, pair_K, w=1):

        # Filtration
        r, K = pair_K
        n_s = r.shape[0]
        if w > 1:
            m = 1. / float(1.-n_s)
            c = 1. - m
            wn = m*w + c
            if wn < 0:
                wn = .0
            elif wn > 1:
                wn = 1.
            b, a = butter(LP_ORDER, wn, btype='low', analog=False)
            f_K = lfilter(b, a, K)
        else:
            f_K = K

        # Computation
        if self.__mode_2D:
            return np.sqrt(f_K/np.pi) - r
        else:
            return np.power(f_K/((4./3.)*np.pi), CSQRT_EXP) - r

    # Simulation for CSR with Ripley's K
    # max_d: maximum distance
    # n_samp: number of samples
    # sp: distance sampling in nm, floor to class resolution
    # n_sim: number of simulations (default 100)
    # p: percentile, if None (default) only 50% is computed, otherwise p, 50, 100-p are computed
     # get_sim: if True (default False) all simulations are returned in an array instead the envelopes
    # Returns: the pair with distance sampling, function K and the percentiles envelopes
    # of the simulations (50, p, 100-p) if get_sim is True, otherwise an array with simulations is returned
    def simulate_K(self, max_d, n_samp=50, n_sim=100, p=None, get_sim=False):

        # Initialization
        r = np.linspace(0, max_d, n_samp)
        r2 = r * r
        rc = r / self.__res
        r2c = rc * rc
        sims = np.zeros(shape=(r.shape[0], n_sim), dtype=np.float)

        # Multi-threading
        tlock = mt.Lock()
        n_th = mp.cpu_count()
        if n_th > n_sim:
            n_th = n_sim
        threads = list()
        # Loop for simulations
        for i in range(n_sim):

            # Generate a new random pattern
            hold_coords = self.__simulator.gen_rand_in_mask(self.__n_coords_2, self.__mask)
            dense, coords = coords_to_dense_mask(hold_coords, self.__mask)
            num = np.zeros(shape=r.shape, dtype=np.float)
            dem = np.zeros(shape=r.shape, dtype=np.float)

            # Static division in threads by points
            spl_ids = np.array_split(np.arange(self.__coords_1.shape[0]), n_th)
            for ids in spl_ids:
                th = mt.Thread(target=self.__th_sim_function_K_2, args=(rc, r2c, num, dem, ids, tlock,
                                                                        dense))
                th.start()
                threads.append(th)
            for th in threads:
                th.join()

            # Store simulation results
            if self.__mode_2D:
                rm = np.pi * r2
            else:
                rm = (4./3.) * np.pi * r2 * r
            dem[0] = coords.shape[0]
            # TODO: K_NUM_CORR is required to fit CSR unshaped simulations with theoretical CSR values, I don't know
            #       from where this values comes from
            # sims[:, i] = (rm/(self.__lambda*self.__z)) * ((K_NUM_CORR*num) / dem)
            sims[:, i] = (rm/(float(coords.shape[0])/float(self.__mask_sum))) * ((K_NUM_CORR*num) / dem)

            gc.collect()

        # Compute results
        if get_sim:
            return r, np.asarray(sims, dtype=np.float)
        else:
            env_05 = func_envelope(sims, per=50)
            if p is None:
                return r, env_05
            else:
                env_1 = func_envelope(sims, per=p)
                env_2 = func_envelope(sims, per=100-p)
                return r, env_1, env_05, env_2

    # Plot with scatter the coordinates of the points in a figure
    # block: if True (default) the code stops until plotted window is closed
    # out_file: path to file to store the figure, if None (default) is not stored and figure
    #           blocks independently of 'block' parameters value
    def plot_points(self, block=True, out_file=None):

        # Initialization
        if out_file is None:
            block = True
        fig = plt.figure()

        if self.is_2D():
            ax = fig.add_subplot(111)
            ax.scatter(self.__coords_1[:, 0], self.__coords_1[:, 1], c='b', marker='o')
            ax.scatter(self.__coords_2[:, 0], self.__coords_2[:, 1], c='r', marker='s')
            ax.set_xlabel('X (nm)')
            ax.set_ylabel('Y (nm)')
            mn_x = np.asarray((self.__coords_1[:, 0].min(), self.__coords_2[:, 0].min())).min()
            mx_x = np.asarray((self.__coords_1[:, 0].max(), self.__coords_2[:, 0].min())).max()
            mn_y = np.asarray((self.__coords_1[:, 1].min(), self.__coords_2[:, 1].min())).min()
            mx_y = np.asarray((self.__coords_1[:, 1].max(), self.__coords_2[:, 1].min())).max()
            ax.set_xlim([mn_x, mx_x])
            ax.set_ylim([mn_y, mx_y])
        else:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.__coords_1[:, 0], self.__coords_1[:, 1], self.__coords_1[:, 2], c='b', marker='o')
            ax.scatter(self.__coords_2[:, 0], self.__coords_2[:, 1], self.__coords_2[:, 2], c='r', marker='s')
            ax.set_xlabel('X (nm)')
            ax.set_ylabel('Y (nm)')
            ax.set_zlabel('Z (nm)')
            mn_x = np.asarray((self.__coords_1[:, 0].min(), self.__coords_2[:, 0].min())).min()
            mx_x = np.asarray((self.__coords_1[:, 0].max(), self.__coords_2[:, 0].min())).max()
            mn_y = np.asarray((self.__coords_1[:, 1].min(), self.__coords_2[:, 1].min())).min()
            mx_y = np.asarray((self.__coords_1[:, 1].max(), self.__coords_2[:, 1].min())).max()
            mn_z = np.asarray((self.__coords_1[:, 2].min(), self.__coords_2[:, 2].min())).min()
            mx_z = np.asarray((self.__coords_1[:, 2].max(), self.__coords_2[:, 2].min())).max()
            ax.set_xlim([mn_x, mx_x])
            ax.set_ylim([mn_y, mx_y])
            ax.set_ylim([mn_z, mx_z])

        # Show data
        ax.set_aspect('equal', 'datalim')
        plt.title('Points scatter ' + str(self.__name))
        plt.show(block=block)
        if out_file is not None:
            plt.savefig(out_file)
        plt.close(fig)

    ##### Internal computations

    # Distance transform on a coordinate point
    # point: 3D point array
    # max_d: for cropping distances (in voxels)
    # Returns: the squared cropped distance, the bounding box
    # ((x.m,x_M), (y_m,y_M), (y_m,y_M)) and the point coordinates in the subvolume in voxels
    def __point_distance_trans(self, point, max_d):

        # Computing bounding box
        max_d_v = int(math.ceil(max_d))
        max_d_v_m1, max_d_v_1 = max_d_v-CROP_OFF, max_d_v-CROP_OFF+1
        box_x_l, box_x_h = int(round(point[0]-max_d_v_m1)), int(round(point[0]+max_d_v_1))
        box_y_l, box_y_h = int(round(point[1]-max_d_v_m1)), int(round(point[1]+max_d_v_1))
        box_z_l, box_z_h = int(round(point[2]-max_d_v_m1)), int(round(point[2]+max_d_v_1))
        if box_x_l < 0:
            box_x_l = 0
        if box_x_h > self.__mx_b:
            box_x_h = self.__mx_b
        if box_y_l < 0:
            box_y_l = 0
        if box_y_h > self.__my_b:
            box_y_h = self.__my_b
        if box_z_l < 0:
            box_z_l = 0
        if box_z_h > self.__mz_b:
            box_z_h = self.__mz_b

        # Distance computation
        hold_x = self.__X[box_x_l:box_x_h, box_y_l:box_y_h, box_z_l:box_z_h] - point[0]
        hold_y = self.__Y[box_x_l:box_x_h, box_y_l:box_y_h, box_z_l:box_z_h] - point[1]
        hold_z = self.__Z[box_x_l:box_x_h, box_y_l:box_y_h, box_z_l:box_z_h] - point[2]

        return (hold_x*hold_x + hold_y*hold_y + hold_z*hold_z), \
               ((box_x_l, box_x_h), (box_y_l, box_y_h), (box_z_l, box_z_h)), \
                (point[0]-box_x_l, point[1]-box_y_l, point[2]-box_z_l)

    # Thread for function G simulation
    def __th_sim_function_G(self, sims, max_d, bins, ids):

        # Loop until all simulations are done
        for idx in ids:

            # Generate random coordinates
            r_coords_2 = self.__simulator.gen_rand_in_mask(self.__n_coords_2, self.__mask)
            r_coords_2 *= self.__res

            # Compute function G
            dists = cnnde(r_coords_2, self.__coords_1)
            _, cdf = compute_cdf(dists, bins, max_d)

            # Store results
            sims[:, idx] = cdf

    # Thread for function K simulation
    def __th_function_K(self, r, r2, num, dem, ids, tlock):

        max_d = r[-1] * self.__res

        # Loop for points
        for idx in ids:

            # Distance transform
            point = self.__coords_d_1[idx]
            sub_dists, box, pt_s = self.__point_distance_trans(point, max_d)
            sub_dense = self.__dense_2[box[0][0]:box[0][1], box[1][0]:box[1][1], box[2][0]:box[2][1]]
            sub_mask = self.__mask[box[0][0]:box[0][1], box[1][0]:box[1][1], box[2][0]:box[2][1]]

            # # UNCOMMENT FOR DEBUGGING
            # if sub_dists[pt_s[0], pt_s[1], pt_s[2]] != 0:
            #     print 'Jol'

            # Loop for radius
            i = 1
            for (rd, rd2) in zip(r[1:], r2[1:]):
                ##### Crop again the subvolumen TODO: check if its worthy
                rd_v = int(math.ceil(rd))
                rd_v_m1, rd_v_1= rd_v-CROP_OFF, rd_v+CROP_OFF
                box_x_l, box_x_h = int(round(pt_s[0]-rd_v_m1)), int(round(pt_s[0]+rd_v_1))
                box_y_l, box_y_h = int(round(pt_s[1]-rd_v_m1)), int(round(pt_s[1]+rd_v_1))
                box_z_l, box_z_h = int(round(pt_s[2]-rd_v_m1)), int(round(pt_s[2]+rd_v_1))
                if box_x_l < 0:
                    box_x_l = 0
                if box_x_h > sub_mask.shape[0]:
                    box_x_h = sub_mask.shape[0]
                if box_y_l < 0:
                    box_y_l = 0
                if box_y_h > sub_mask.shape[1]:
                    box_y_h = sub_mask.shape[1]
                if box_z_l < 0:
                    box_z_l = 0
                if box_z_h > sub_mask.shape[2]:
                    box_z_h = sub_mask.shape[2]
                ssub_dense = sub_dense[box_x_l:box_x_h, box_y_l:box_y_h, box_z_l:box_z_h]
                ssub_mask = sub_mask[box_x_l:box_x_h, box_y_l:box_y_h, box_z_l:box_z_h]
                ssub_dists = sub_dists[box_x_l:box_x_h, box_y_l:box_y_h, box_z_l:box_z_h]

                # UNCOMMENT FOR DEBUGGING
                # pt_h = np.asarray((pt_s[0]-box_x_l, pt_s[1]-box_y_l, pt_s[2]-box_z_l))
                # if ssub_dists[pt_h[0], pt_h[1], pt_h[2]] != 0:
                #     print 'Jol'

                region = ssub_dists <= rd2
                hold_mask = region * ssub_mask
                hold_num = (ssub_dense * hold_mask).sum()
                hold_dem = (hold_mask.sum() * self.__z)
                tlock.acquire()
                num[i] += hold_num
                dem[i] += hold_dem
                tlock.release()
                i += 1

    # Thread for function K (efficiency improvement of __th_function_K())
    def __th_function_K_2(self, r, r2, num, dem, ids, tlock):

        # Loop for points
        for idx in ids:

            # Distance transform
            point = self.__coords_d_1[idx]
            sub_dists, box, pt_s = self.__point_distance_trans(point, r[-1])
            sub_dense = self.__dense_2[box[0][0]:box[0][1], box[1][0]:box[1][1], box[2][0]:box[2][1]]
            sub_mask = self.__mask[box[0][0]:box[0][1], box[1][0]:box[1][1], box[2][0]:box[2][1]]

            # Intermediate subvolumes
            sub_dense[int(pt_s[0]), int(pt_s[1]), int(pt_s[2])] = False
            dense_ids = sub_dense * sub_mask
            sub_dense_d = sub_dense * sub_dists

            # Computing CDFs
            hold_dem, _ = np.histogram(sub_dists[sub_mask], bins=r2)
            hold_dem = np.cumsum(hold_dem)
            hold_num, _ = np.histogram(sub_dense_d[dense_ids], bins=r2)
            hold_num = np.cumsum(hold_num)

            # Update the shared arrays
            tlock.acquire()
            num[1:] += hold_num
            dem[1:] += hold_dem
            tlock.release()

    # Thread for function K simulation
    def __th_sim_function_K(self, r, rc, r2, r2c, sims, ids):

        max_d = r[-1]

        # Loop for simulations
        for idx in ids:

            # Access to shared variable which control global status
            num = np.zeros(shape=r2.shape, dtype=np.float)
            dem = np.zeros(shape=r2.shape, dtype=np.float)
            dem[0] = self.__z

            # Generate a new random pattern
            hold_coords = gen_rand_in_mask(self.__n_coords_2, self.__mask)
            dense, coords = coords_to_dense_mask(hold_coords, self.__mask)
            # coords *= self.__res

            # Compute function K
            # Loop for points
            for point in self.__coords_d_1:
                sub_dists, box, pt_s = self.__point_distance_trans(point, max_d)
                sub_dense = dense[box[0][0]:box[0][1], box[1][0]:box[1][1], box[2][0]:box[2][1]]
                sub_mask = self.__mask[box[0][0]:box[0][1], box[1][0]:box[1][1], box[2][0]:box[2][1]]

                # # UNCOMMENT FOR DEBUGGING
                # if sub_dists[pt_s[0], pt_s[1], pt_s[2]] != 0:
                #     print 'Jol'

                # Loop for radius
                i = 1
                for (rd, rd2) in zip(r[1:], r2[1:]):
                    ##### Crop again the subvolumen
                    rd_v = int(math.ceil(rd))
                    rd_v_m1, rd_v_1= rd_v-CROP_OFF, rd_v+CROP_OFF
                    box_x_l, box_x_h = int(round(pt_s[0]-rd_v_m1)), int(round(pt_s[0]+rd_v_1))
                    box_y_l, box_y_h = int(round(pt_s[1]-rd_v_m1)), int(round(pt_s[1]+rd_v_1))
                    box_z_l, box_z_h = int(round(pt_s[2]-rd_v_m1)), int(round(pt_s[2]+rd_v_1))
                    if box_x_l < 0:
                        box_x_l = 0
                    if box_x_h > sub_mask.shape[0]:
                        box_x_h = sub_mask.shape[0]
                    if box_y_l < 0:
                        box_y_l = 0
                    if box_y_h > sub_mask.shape[1]:
                        box_y_h = sub_mask.shape[1]
                    if box_z_l < 0:
                        box_z_l = 0
                    if box_z_h > sub_mask.shape[2]:
                        box_z_h = sub_mask.shape[2]
                    ssub_dense = sub_dense[box_x_l:box_x_h, box_y_l:box_y_h, box_z_l:box_z_h]
                    ssub_mask = sub_mask[box_x_l:box_x_h, box_y_l:box_y_h, box_z_l:box_z_h]
                    ssub_dists = sub_dists[box_x_l:box_x_h, box_y_l:box_y_h, box_z_l:box_z_h]

                    # # UNCOMMENT FOR DEBUGGING
                    # pt_h = np.asarray((pt_s[0]-box_x_l, pt_s[1]-box_y_l, pt_s[2]-box_z_l))
                    # if ssub_dists[pt_h[0], pt_h[1], pt_h[2]] != 0:
                    #     print 'Jol'

                    region = ssub_dists <= rd2
                    hold_mask = region * ssub_mask
                    num[i] += (ssub_dense * hold_mask).sum()
                    dem[i] += (hold_mask.sum() * self.__z)
                    i += 1

            # Store results
            if self.__mode_2D:
                rm = np.pi * r2
            else:
                rm = (4./3.) * np.pi * r2 * r
            sims[:, idx] = (rm/self.__lambda_2) * (num / dem)

    # Thread for function K simulation (efficiency improvement of __th_sim_function_K())
    def __th_sim_function_K_2(self, r, r2, num, dem, ids, tlock, dense):

        # Loop for points
        for idx in ids:

            # Distance transform
            point = self.__coords_d_1[idx]
            sub_dists, box, pt_s = self.__point_distance_trans(point, r[-1])
            sub_dense = dense[box[0][0]:box[0][1], box[1][0]:box[1][1], box[2][0]:box[2][1]]
            sub_mask = self.__mask[box[0][0]:box[0][1], box[1][0]:box[1][1], box[2][0]:box[2][1]]

            # Intermediate subvolumes
            sub_dense[int(pt_s[0]), int(pt_s[1]), int(pt_s[2])] = False
            dense_ids = sub_dense * sub_mask
            sub_dense_d = sub_dense * sub_dists

            # Computing CDFs
            hold_dem, _ = np.histogram(sub_dists[sub_mask], bins=r2)
            hold_dem = np.cumsum(hold_dem)
            hold_num, _ = np.histogram(sub_dense_d[dense_ids], bins=r2)
            hold_num = np.cumsum(hold_num)

            # Update the shared arrays
            tlock.acquire()
            num[1:] += hold_num
            dem[1:] += hold_dem
            tlock.release()

##############################################################################################
# Class for plotting univaritate analysis
#
class PlotUni(object):

    # ### Constructor area

    def __init__(self, name=None):
        self.__name = name
        self.__Gs, self.__sG, self.__sims_G = None, None, None
        self.__Fs, self.__sF, self.__sims_F = None, None, None
        self.__Ks, self.__sK, self.__sims_K = None, None, None
        self.__NLDs, self.__sNLD = None, None
        self.__RBFs = None
        self.__K_nums, self.__K_dems = None, None
        self.__sims_K_num, self.__sims_K_dem = None, None
        self.__unis = list()
        self.__zs = list()

    # ### External functionality area

    def get_name(self):
        return self.__name

    # uni: a UniStat object
    # step: distance in to previous inserted UniStat object (default 1)
    def insert_uni(self, uni, step=1):
        self.__unis.append(uni)
        if len(self.__zs) == 0:
            self.__zs.append(.0)
        else:
            curr = self.__zs[-1]
            self.__zs.append(curr+step)

    # Purge already inserter UniStat objects which have a number of points lower than input
    # min_n_pt: minimum number of points
    def purge_unis(self, min_n_pt):

        # Initialization
        hold_unis, hold_zs = self.__unis, self.__zs
        self.__unis, self.__zs = list(), list()

        # Loop for deletion
        for uni, zs in zip(hold_unis, hold_zs):
            if uni.get_n_points() >= min_n_pt:
                self.__unis.append(uni)
                self.__zs.append(zs)

    # Return True if not all patterns have the same intensity
    def not_eq_intensities(self):
        if len(self.__unis) > 0:
            hold_int = self.__unis[0].get_intensity()
            for uni in self.__unis[1:]:
                if uni.get_intensity() != hold_int:
                    return True
        return False

    # Pickling the object state
    # fname: full path for the pickle file
    def pickle(self, fname):
        pkl_f = open(fname, 'w')
        try:
            pickle.dump(self, pkl_f)
        finally:
            pkl_f.close()

    # Analyze (plot in a figure) intensities
    def analyze_intensity(self, block=True, out_file=None):

        if len(self.__unis) <= 0:
            error_msg = 'No UniStat object inserted, nothing to plot.'
            raise pexceptions.PySegInputError(expr='analyze_intensity (PlotUni)', msg=error_msg)

        # Data pre-processing
        lambdas = np.zeros(shape=len(self.__unis), dtype=np.float)
        for i, uni in enumerate(self.__unis):
            lambdas[i] = uni.get_intensity()
        zs = np.asarray(self.__zs, dtype=np.float)

        # Initialization
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.title('Intensities')
        plt.xlabel('Entries')
        if self.__unis[0].is_2D():
            plt.ylabel('point/nm^2')
        else:
            plt.ylabel('point/nm^3')
        plt.xlim(zs[0]-1, zs[-1]+1)

        # Plotting
        plt.stem(zs, lambdas)

        # Show data
        plt.show(block=block)
        if out_file is not None:
            out_ext = os.path.splitext(out_file)[1]
            if out_ext == '.pkl':
                fig_pickle(fig, out_file)
            else:
                plt.savefig(out_file)
        plt.close()

    # Analyze (plot in a figure) function G
    # max_d: maximum distance for computing the histograms in nm
    # bins: number of samples for output (default 50)
    # n_sim: number of simulations for CSR (default 0), if < 1 disabled
    # p: percentile, if None (default) only 50% is computed, otherwise p, 50, 100-p are computed
    # update: if True (default False) re-computing data if forced
    # block: if True (default) the code stops until plotted window is closed
    # out_file: path to file to store the analysis, if None (default) is not stored
    # legend: if True (default False) and out_file is not None, a legend is also stored
    # gather: if not None (default) then the graphs from the different inserted UniStat objects are collapsed
    #         into one
    def analyze_G(self, max_d, bins=50, n_sim=0, p=None, update=False, block=True, out_file=None, legend=False,
                  gather=False):

        # Computations
        if update or (self.__Gs is None):
            self.__Gs, self.__sG = None, None
            self.__compute_Gs(max_d, bins, gather)
        if n_sim > 0:
            if update or (self.__sG is None):
                self.__simulate_G(max_d, bins, n_sim, p)
        colors = cm.rainbow(np.linspace(0, 1, len(self.__unis)))

        if self.not_eq_intensities():
            print 'WARNING: Function G is not a proper metric for comparing patterns with different ' \
                  'intensities'

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.title('G-Function')
        plt.xlabel('Distance (nm)')
        plt.ylabel('G')
        plt.ylim(0, 1)
        # Plot data analysis
        for (uni, g_dst, g, color) in zip(self.__unis, self.__Gs[0], self.__Gs[1], colors):
            ax.plot(g_dst, g, c=color, label=uni.get_name())
        # Get legend data
        if legend:
            handles, labels = ax.get_legend_handles_labels()
        # Plot random references
        if n_sim > 0:
            if len(self.__sG) == 4:
                ax.plot(self.__sG[0], self.__sG[1], 'k--')
                ax.plot(self.__sG[0], self.__sG[2], 'k')
                ax.plot(self.__sG[0], self.__sG[3], 'k--')
            else:
                ax.plot(self.__sG[0], self.__sG[2], 'k')

        # Show data
        plt.show(block=block)
        if out_file is not None:
            out_ext = os.path.splitext(out_file)[1]
            if out_ext == '.pkl':
                fig_pickle(fig, out_file)
            else:
                plt.savefig(out_file)
        plt.close()

        # Store legend
        if legend and (out_file is not None):
            store_legend(handles, labels, out_file)

    # Analyze (plot in a figure) function F
    # max_d: maximum distance for computing the histograms in nm
    # bins: number of samples for output (default 50)
    # f_npts: number of samples for function F, if <0 it is set as the number of input points (default)
    # n_sim: number of simulations for CSR (default 0), if < 1 disabled
    # p: percentile, if None (default) only 50% is computed, otherwise p, 50, 100-p are computed
    # update: if True (default False) re-computing data if forced
    # block: if True (default) the code stops until plotted window is closed
    # out_file: path to file to store the analysis, if None (default) is not stored
    # legend: if True (default False) and out_file is not None, a legend is also stored
    # gather: if not None (default) then the graphs from the different inserted UniStat objects are collapsed
    #         into one
    def analyze_F(self, max_d, bins=50, f_npts=-1, n_sim=0, p=None, update=False, block=True,
                  out_file=None, legend=False, gather=False):

        # Computations
        if update or (self.__Fs is None):
            self.__Fs, self.__sF = None, None
            self.__compute_Fs(max_d, bins, f_npts, gather)
        if n_sim > 0:
            if update or (self.__sF is None):
                self.__simulate_F(max_d, bins, f_npts, n_sim, p)
        colors = cm.rainbow(np.linspace(0, 1, len(self.__unis)))

        if self.not_eq_intensities():
            print 'WARNING: Function F is not a proper metric for comparing patterns with different ' \
                  'intensities'

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.title('F-Function')
        plt.xlabel('Distance (nm)')
        plt.ylabel('F')
        plt.ylim(0, 1)
        # Plot data analysis
        lines = list()
        for (uni, f_dst, f, color) in zip(self.__unis, self.__Fs[0], self.__Fs[1], colors):
            line, = plt.plot(f_dst, f, c=color, label=uni.get_name())
            lines.append(line)
        # Get legend data
        if legend:
            handles, labels = ax.get_legend_handles_labels()
        # Plot random references
        if n_sim > 0:
            if len(self.__sF) == 4:
                plt.plot(self.__sF[0], self.__sF[1], 'k--')
                plt.plot(self.__sF[0], self.__sF[2], 'k')
                plt.plot(self.__sF[0], self.__sF[3], 'k--')
            else:
                plt.plot(self.__sF[0], self.__sF[2], 'k')

        # Show data
        plt.show(block=block)
        if out_file is not None:
            out_ext = os.path.splitext(out_file)[1]
            if out_ext == '.pkl':
                fig_pickle(fig, out_file)
            else:
                plt.savefig(out_file)
        plt.close()

        # Store legend
        if legend and (out_file is not None):
            store_legend(handles, labels, out_file)

    # Analyze (plot in a figure) function J, analyze_G() and analyze_F() must be called before
    # block: if True (default) the code stops until plotted window is closed
    # out_file: path to file to store the analysis, if None (default) is not stored
    # p: percentile, if None (default) only 50% is computed, otherwise p, 50, 100-p are computed
    # legend: if True (default False) and out_file is not None, a legend is also stored
    def analyze_J(self, block=True, out_file=None, legend=False, p=None):

        # Computations
        if (self.__Gs is None) or (self.__Fs is None):
            error_msg = 'analyze_G() and analyze_F() must be called before with the same input parameters.'
            raise pexceptions.PySegInputError(expr='analyze_J (UniStat)', msg=error_msg)
        colors = cm.rainbow(np.linspace(0, 1, len(self.__unis)))

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.title('J-Function')
        plt.xlabel('Distance (nm)')
        plt.ylabel('J')
        # Plot data analysis
        lines = list()
        for (uni, g_dst, g, f, color) in zip(self.__unis, self.__Gs[0], self.__Gs[1],
                                             self.__Fs[1], colors):
            J = self.__compute_J(g, f)
            line, = plt.plot(g_dst[:J.shape[0]], J, c=color, label=uni.get_name())
            lines.append(line)
        # Get legend data
        if legend:
            handles, labels = ax.get_legend_handles_labels()
        if (self.__sG is not None) and (self.__sF is not None) and (len(self.__sG) == len(self.__sF)):
            if p is None:
                mx_j_50, j_50 = self.__compute_J_env(self.__sims_G, self.__sims_F)
                plt.plot(self.__sG[0][1:mx_j_50], j_50[1:mx_j_50], 'k')
            else:
                mx_j_p, mx_j_50, mx_j_100_p, j_p, j_50, j_100_p = self.__compute_J_env(self.__sims_G,
                                                                                       self.__sims_F, p=p)
                plt.plot(self.__sG[0][1:mx_j_50], j_50[1:mx_j_50], 'k')
                plt.plot(self.__sG[0][1:mx_j_p], j_p[1:mx_j_p], 'k--')
                plt.plot(self.__sG[0][1:mx_j_100_p], j_100_p[1:mx_j_100_p], 'k--')

        # Show data
        plt.show(block=block)
        if out_file is not None:
            out_ext = os.path.splitext(out_file)[1]
            if out_ext == '.pkl':
                fig_pickle(fig, out_file)
            else:
                plt.savefig(out_file)
        plt.close()

        # Store legend
        if legend and (out_file is not None):
            store_legend(handles, labels, out_file)

    # Analyze (plot in a figure) function K
    # max_d: maximum distance
    # n_samp: number of samples, it must be greater than 2
    # n_sim: number of simulations for CSR (default 0), if < 1 disabled
    # p: percentile, if None (default) only 50% is computed, otherwise p, 50, 100-p are computed
    # update: if True (default False) re-computing data if forced
    # block: if True (default) the code stops until plotted window is closed
    # tcsr: if True (default False) plots CSR on unbounded space
    # out_file: path to file to store the analysis, if None (default) is not stored, if the ending is .pkl, then
    #           a pickle file is assumed
    # legend: if True (default False) and out_file is not None and the format is not .pkl, a legend is also stored
    # gather: if not None (default) then the graphs from the different inserted UniStat objects are collapsed
    #         into one
    def analyze_K(self, max_d, n_samp=50, n_sim=0, p=None, update=False, block=True, tcsr=False,
                  out_file=None, legend=False, gather=False):

        if n_samp <= 2:
            error_msg = 'Number of samples must be greater than 2'
            raise pexceptions.PySegInputError(expr='analyze_K (UniStat)', msg=error_msg)

        # Computations
        if update or (self.__Ks is None):
            self.__Ks, self.__sK = None, None
            self.__K_nums, self.__K_dems = None, None
            self.__compute_Ks(max_d, n_samp)
        if n_sim > 0:
            if update or (self.__sK is None):
                self.__simulate_K(max_d, n_samp, n_sim, p)
        colors = cm.rainbow(np.linspace(0, 1, len(self.__unis)))

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.title('Ripley\'s K')
        plt.xlabel('Radius (nm)')
        plt.ylabel('K')
        # Plot data analysis
        lines = list()
        if gather:
            k_dst = self.__Ks[0][0]
            wk = np.zeros(shape=k_dst.shape[0], dtype=np.float)
            weights = list()
            for uni in self.__unis:
                weights.append(uni.get_intensity())
            weights = np.asarray(weights, dtype=np.float)
            weights /= weights.sum()
            for (w, k) in zip(weights, self.__Ks[:][1]):
                wk += (k*w)
            line, = plt.plot(k_dst[1:], wk[1:], c='b')
            lines.append(line)
        else:
            for (uni, k_dst, k, color) in zip(self.__unis, self.__Ks[:][0], self.__Ks[:][1], colors):
                line, = plt.plot(k_dst[1:], k[1:], c=color, label=uni.get_name())
                lines.append(line)
        # Get legend data
        if legend:
            handles, labels = ax.get_legend_handles_labels()
        # Plot random references
        if n_sim > 0:
            if len(self.__sK) == 4:
                plt.plot(self.__sK[0][1:], self.__sK[1][1:], 'k--')
                plt.plot(self.__sK[0][1:], self.__sK[2][1:], 'k')
                plt.plot(self.__sK[0][1:], self.__sK[3][1:], 'k--')
            else:
                plt.plot(self.__sK[0][1:], self.__sK[2][1:], 'k')
        # Plot CSR on unbounded space
        if tcsr:
            if self.__unis[0].is_2D():
                plt.plot(self.__sK[0][1:], np.pi*self.__sK[0][1:]*self.__sK[0][1:], 'k+')
            else:
                plt.plot(self.__sK[0][1:], (4./3.)*np.pi*self.__sK[0][1:]*self.__sK[0][1:]*self.__sK[0][1:], 'k+')

        # Show data
        plt.show(block=block)
        if out_file is not None:
            out_ext = os.path.splitext(out_file)[1]
            if out_ext == '.pkl':
                fig_pickle(fig, out_file)
            else:
                plt.savefig(out_file)
        plt.close()

        # Store legend
        if legend and (out_file is not None):
            store_legend(handles, labels, out_file)

    # Analyze (plot in a figure) function Normalizaed Local Density
    # shell_th: shell thickness in nm
    # max_d: maximum distance
    # n_samp: number of samples, it must be greater than 2
    # n_sim: number of simulations for CSR (default 0), if < 1 disabled
    # p: percentile, if None (default) only 50% is computed, otherwise p, 50, 100-p are computed
    # update: if True (default False) re-computing data if forced
    # block: if True (default) the code stops until plotted window is closed
    # tcsr: if True (default False) plots CSR on unbounded space
    # out_file: path to file to store the analysis, if None (default) is not stored, if the ending is .pkl, then
    #           a pickle file is assumed
    # legend: if True (default False) and out_file is not None and the format is not .pkl, a legend is also stored
    # gather: if not None (default) then the graphs from the different inserted UniStat objects are collapsed
    #         into one
    # gstd: scale dependant lines with standard deviation from different experimental data with scale, only valid
    #       when gathexier is True. (default False)
    # db: if False (default) result is provided in linear scale, if True the result (ratio) is provided in dBs
    # gnorm: if True (default) result are normalized for according to global density on each tomogram
    def analyze_NLD(self, shell_th, max_d, n_samp=50, n_sim=0, p=None, update=False, block=True,
                  out_file=None, legend=False, gather=False, gstd=False, db=False, gnorm=True):

        if n_samp <= 2:
            error_msg = 'Number of samples must be greater than 2'
            raise pexceptions.PySegInputError(expr='analyze_NLD (UniStat)', msg=error_msg)

        # Computations
        if update or (self.__NLDs is None):
            self.__NLDs, self.__sNLD = None, None
            self.__compute_NLDs(shell_th, max_d, n_samp, gnorm=gnorm)
        if n_sim > 0:
            if update or (self.__sNLD is None):
                self.__simulate_NLD(shell_th, max_d, n_samp, n_sim, p, gnorm=gnorm)
        colors = cm.rainbow(np.linspace(0, 1, len(self.__unis)))

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.title('Normalized Local Density')
        plt.xlabel('Radius (nm)')
        if db:
            plt.ylabel('Normalized Local Density (dB)')
        else:
            plt.ylabel('Normalized Local Density')
        # Plot data analysis
        lines = list()
        if gather:
            n_dst = self.__NLDs[0][0]
            wn = np.zeros(shape=n_dst.shape[0], dtype=np.float)
            weights = list()
            for uni in self.__unis:
                weights.append(uni.get_intensity())
            weights = np.asarray(weights, dtype=np.float)
            weights /= weights.sum()
            Ns = list()
            for (w, n) in zip(weights, self.__NLDs[:][1]):
                Ns.append(n)
                wn += (Ns[-1]*w)

            if gstd:
                std_l = np.std(Ns, axis=0)
                if db:
                    lines.append(plt.plot(n_dst[1:], 10.*np.log10(wn[1:]-.5*std_l[1:]), 'b--')[0])
                    lines.append(plt.plot(n_dst[1:], 10.*np.log10(wn[1:]), 'b')[0])
                    lines.append(plt.plot(n_dst[1:], 10.*np.log10(wn[1:]+.5*std_l[1:]), 'b--')[0])
                else:
                    lines.append(plt.plot(n_dst[1:], wn[1:]-.5*std_l[1:], 'b--')[0])
                    lines.append(plt.plot(n_dst[1:], wn[1:], 'b')[0])
                    lines.append(plt.plot(n_dst[1:], wn[1:]+.5*std_l[1:], 'b--')[0])
            elif p is not None:
                Ns = np.asarray(Ns, dtype=np.float32)
                env_p = func_envelope(Ns, p, axis=0)
                env_50 = func_envelope(Ns, 50, axis=0)
                env_100_p = func_envelope(Ns, 100-p, axis=0)
                if db:
                    lines.append(plt.plot(n_dst[1:], 10.*np.log10(env_p[1:]), 'b--')[0])
                    lines.append(plt.plot(n_dst[1:], 10.*np.log10(env_50[1:]), 'b')[0])
                    lines.append(plt.plot(n_dst[1:], 10.*np.log10(env_100_p[1:]), 'b--')[0])
                else:
                    lines.append(plt.plot(n_dst[1:], env_p[1:], 'b--')[0])
                    lines.append(plt.plot(n_dst[1:], env_50[1:], 'b')[0])
                    lines.append(plt.plot(n_dst[1:], env_100_p[1:], 'b--')[0])
        else:
            for (uni, n_dst, n, color) in zip(self.__unis, self.__NLDs[:][0], self.__NLDs[:][1], colors):
                if db:
                    line, = plt.plot(n_dst[1:], 10.*math.log10(n[1:]), c=color, label=uni.get_name())
                else:
                    line, = plt.plot(n_dst[1:], n[1:], c=color, label=uni.get_name())
                lines.append(line)
        # Get legend data
        if legend:
            handles, labels = ax.get_legend_handles_labels()
        # Plot random references
        if n_sim > 0:
            if len(self.__sNLD) == 4:
                if db:
                    plt.plot(self.__sNLD[0][1:], 10.*np.log10(self.__sNLD[1][1:]), 'k--')
                    plt.plot(self.__sNLD[0][1:], 10.*np.log10(self.__sNLD[2][1:]), 'k')
                    plt.plot(self.__sNLD[0][1:], 10.*np.log10(self.__sNLD[3][1:]), 'k--')
                else:
                    plt.plot(self.__sNLD[0][1:], self.__sNLD[1][1:], 'k--')
                    plt.plot(self.__sNLD[0][1:], self.__sNLD[2][1:], 'k')
                    plt.plot(self.__sNLD[0][1:], self.__sNLD[3][1:], 'k--')
            else:
                if db:
                    plt.plot(self.__sNLD[0][1:], 10.*math.log10(self.__sNLD[2][1:]), 'k')
                else:
                    plt.plot(self.__sNLD[0][1:], self.__sNLD[2][1:], 'k')

        # Show data
        plt.show(block=block)
        if out_file is not None:
            out_ext = os.path.splitext(out_file)[1]
            if out_ext == '.pkl':
                fig_pickle(fig, out_file)
            else:
                plt.savefig(out_file)
        plt.close()

        # Store legend
        if legend and (out_file is not None):
            store_legend(handles, labels, out_file)

    # Analyze (plot in a figure) function Normalizaed Local Density
    # shell_th: shell thickness in nm
    # max_d: maximum distance
    # n_samp: number of samples, it must be greater than 2
    # p: percentile, if None (default) only 50% is computed, otherwise p, 50, 100-p are computed
    # update: if True (default False) re-computing data if forced
    # block: if True (default) the code stops until plotted window is closed
    # out_file: path to file to store the analysis, if None (default) is not stored, if the ending is .pkl, then
    #           a pickle file is assumed
    # legend: if True (default False) and out_file is not None and the format is not .pkl, a legend is also stored
    # gather: if not None (default) then the graphs from the different inserted UniStat objects are collapsed
    #         into one
    # gstd: scale dependant lines with standard deviation from different experimental data with scale, only valid
    #       when gather is True. (default False)
    # db: if False (default) result is provided in linear scale, if True the result (ratio) is provided in dBs
    def analyze_RBF(self, shell_th, max_d, n_samp=50, p=None, update=False, block=True,
                  out_file=None, legend=False, gather=False, gstd=False, db=False):

        if n_samp <= 2:
            error_msg = 'Number of samples must be greater than 2'
            raise pexceptions.PySegInputError(expr='analyze_NLD (UniStat)', msg=error_msg)

        # Computations
        if update or (self.__RBFs is None):
            self.__RBFs = None
            self.__compute_RBFs(shell_th, max_d, n_samp)
        colors = cm.rainbow(np.linspace(0, 1, len(self.__unis)))

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.title('Radial Basis Function (no borders compensation)')
        plt.xlabel('Radius (nm)')
        if db:
            plt.ylabel('Normalized Local Density (dB)')
        else:
            plt.ylabel('Radial Basis Function')
        # Plot data analysis
        lines = list()
        if gather:
            n_dst = self.__RBFs[0][0]
            wr = np.zeros(shape=n_dst.shape[0], dtype=np.float)
            weights = list()
            for uni in self.__unis:
                weights.append(uni.get_intensity())
            weights = np.asarray(weights, dtype=np.float)
            weights /= weights.sum()
            Nr = list()
            for (w, n) in zip(weights, self.__RBFs[:][1]):
                Nr.append(n)
                wr += (Nr[-1]*w)

            if gstd:
                std_l = np.std(Nr, axis=0)
                if db:
                    lines.append(plt.plot(n_dst[1:], 10.*np.log10(wr[1:]-.5*std_l[1:]), 'b--')[0])
                    lines.append(plt.plot(n_dst[1:], 10.*np.log10(wr[1:]), 'b')[0])
                    lines.append(plt.plot(n_dst[1:], 10.*np.log10(wr[1:]+.5*std_l[1:]), 'b--')[0])
                else:
                    lines.append(plt.plot(n_dst[1:], wr[1:]-.5*std_l[1:], 'b--')[0])
                    lines.append(plt.plot(n_dst[1:], wr[1:], 'b')[0])
                    lines.append(plt.plot(n_dst[1:], wr[1:]+.5*std_l[1:], 'b--')[0])
            elif p is not None:
                Nr = np.asarray(Nr, dtype=np.float32)
                env_p = func_envelope(Nr, p, axis=0)
                env_50 = func_envelope(Nr, 50, axis=0)
                env_100_p = func_envelope(Nr, 100-p, axis=0)
                if db:
                    lines.append(plt.plot(n_dst[1:], 10.*np.log10(env_p[1:]), 'b--')[0])
                    lines.append(plt.plot(n_dst[1:], 10.*np.log10(env_50[1:]), 'b')[0])
                    lines.append(plt.plot(n_dst[1:], 10.*np.log10(env_100_p[1:]), 'b--')[0])
                else:
                    lines.append(plt.plot(n_dst[1:], env_p[1:], 'b--')[0])
                    lines.append(plt.plot(n_dst[1:], env_50[1:], 'b')[0])
                    lines.append(plt.plot(n_dst[1:], env_100_p[1:], 'b--')[0])
        else:
            for (uni, n_dst, n, color) in zip(self.__unis, self.__RBFs[:][0], self.__RBFs[:][1], colors):
                if db:
                    line, = plt.plot(n_dst[1:], 10.*math.log10(n[1:]), c=color, label=uni.get_name())
                else:
                    line, = plt.plot(n_dst[1:], n[1:], c=color, label=uni.get_name())
                lines.append(line)
        # Get legend data
        if legend:
            handles, labels = ax.get_legend_handles_labels()

        # Show data
        plt.show(block=block)
        if out_file is not None:
            out_ext = os.path.splitext(out_file)[1]
            if out_ext == '.pkl':
                fig_pickle(fig, out_file)
            else:
                plt.savefig(out_file)
        plt.close()

        # Store legend
        if legend and (out_file is not None):
            store_legend(handles, labels, out_file)

    # Analyze (plot in a figure) Ripley's L function, analyze_K() must be called before
    # block: if True (default) the code stops until plotted window is closed
    # out_file: path to file to store the analysis, if None (default) is not stored
    # gather: if not None (default) then the graphs from the different inserted UniStat objects are collapsed
    #         into one
    # p: percentile, if None (default) only 50% is computed, otherwise p, 50, 100-p are computed
    # legend: if True (default False) and out_file is not None, a legend is also stored
    # gstd: scale dependant lines with standard deviation from different experimental data with scale, only valid
    #       when gathexier is True. (default False)
    def analyze_L(self, block=True, out_file=None, legend=False, gather=False, p=None, gstd=False):

        # Computations
        if self.__Ks is None:
            error_msg = 'analyze_K() must be called before with the same input parameters.'
            raise pexceptions.PySegInputError(expr='analyze_L (UniStat)', msg=error_msg)
        colors = cm.rainbow(np.linspace(0, 1, len(self.__unis)))

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.title('L-Function')
        plt.xlabel('Distance (nm)')
        plt.ylabel('L')
        # Plot data analysis
        lines = list()
        mode_2D = self.__unis[0].is_2D()
        if gather:
            l_dst = self.__Ks[0][0]
            wl = np.zeros(shape=l_dst.shape[0], dtype=np.float)
            weights = list()
            for uni in self.__unis:
                weights.append(uni.get_intensity())
            weights = np.asarray(weights, dtype=np.float)
            weights /= weights.sum()
            Ls = list()
            for (w, k) in zip(weights, self.__Ks[:][1]):
                Ls.append(self.__compute_L(l_dst, k, mode_2D))
                wl += (Ls[-1]*w)

            if gstd:
                std_l = np.std(Ls, axis=0)
                lines.append(plt.plot(l_dst[1:], wl[1:]-.5*std_l[1:], 'b--')[0])
                lines.append(plt.plot(l_dst[1:], wl[1:], 'b')[0])
                lines.append(plt.plot(l_dst[1:], wl[1:]+.5*std_l[1:], 'b--')[0])
            elif p is not None:
                Ls = np.asarray(Ls, dtype=np.float32)
                env_p = func_envelope(Ls, p, axis=0)
                env_50 = func_envelope(Ls, 50, axis=0)
                env_100_p = func_envelope(Ls, 100-p, axis=0)
                lines.append(plt.plot(l_dst[1:], env_p[1:], 'b--')[0])
                lines.append(plt.plot(l_dst[1:], env_50[1:], 'b')[0])
                lines.append(plt.plot(l_dst[1:], env_100_p[1:], 'b--')[0])
        else:
            for (uni, k_dst, k, color) in zip(self.__unis, self.__Ks[0], self.__Ks[1], colors):
                line, = plt.plot(k_dst[1:], self.__compute_L(k_dst, k, mode_2D)[1:],
                                 c=color, label=uni.get_name())
                lines.append(line)
        # Get legend data
        if legend:
            handles, labels = ax.get_legend_handles_labels()
        if self.__sK is not None:
            if p is None:
                l_dst, l_50 = self.__compute_L_env(self.__sK[0], self.__sims_K, self.__unis[0].is_2D())
                plt.plot(l_dst[1:], l_50[1:], 'k')
            else:
                l_dst, l_p, l_50, l_100_p = self.__compute_L_env(self.__sK[0], self.__sims_K, self.__unis[0].is_2D(), p)
                plt.plot(l_dst[1:], l_p[1:], 'k--')
                plt.plot(l_dst[1:], l_50[1:], 'k')
                plt.plot(l_dst[1:], l_100_p[1:], 'k--')

        # Show data
        plt.show(block=block)
        if out_file is not None:
            out_ext = os.path.splitext(out_file)[1]
            if out_ext == '.pkl':
                fig_pickle(fig, out_file)
            else:
                plt.savefig(out_file)
        plt.close()

        # Store legend
        if legend and (out_file is not None):
            store_legend(handles, labels, out_file)

    # Analyze (plot in a figure) intensity over scales, analyze_K() must be called before
    # block: if True (default) the code stops until plotted window is closed
    # out_file: path to file to store the analysis, if None (default) is not stored
    # gather: if not None (default) then the graphs from the different inserted UniStat objects are collapsed
    #         into one
    # p: percentile, if None (default) only 50% is computed, otherwise p, 50, 100-p are computed
    # legend: if True (default False) and out_file is not None, a legend is also stored
    # gstd: scale dependant lines with standard deviation from different experimental data with scale, only valid
    #       when gather option is True. (default False)
    def analyze_I(self, block=True, out_file=None, legend=False, gather=False, p=None, gstd=False):

        # Computations
        if (self.__K_nums is None) or (self.__K_dems is None):
            error_msg = 'analyze_K() must be called before with the same input parameters.'
            raise pexceptions.PySegInputError(expr='analyze_I (UniStat)', msg=error_msg)
        colors = cm.rainbow(np.linspace(0, 1, len(self.__unis)))

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.title('Intensity over scales')
        plt.xlabel('Distance (nm)')
        mode_2D = self.__unis[0].is_2D()
        if mode_2D:
            plt.ylabel('Intensity (points / nm^2)')
        else:
            plt.ylabel('Intensity (points / nm^3)')
        # Plot data analysis
        lines = list()
        if gather:
            i_dst = self.__Ks[0][0]
            wi = np.zeros(shape=i_dst.shape[0], dtype=np.float)
            weights = list()
            for uni in self.__unis:
                weights.append(uni.get_intensity())
            weights = np.asarray(weights, dtype=np.float)
            weights /= weights.sum()
            Is = list()
            for (w, k_num, k_dem) in zip(weights, self.__K_nums, self.__K_dems):
                Is.append(self.__compute_I(k_num, k_dem))
                wi += (Is[-1]*w)

            if gstd:
                std_l = np.std(Is, axis=0)
                lines.append(plt.plot(i_dst[1:], wi[1:]-.5*std_l[1:], 'b--')[0])
                lines.append(plt.plot(i_dst[1:], wi[1:], 'b')[0])
                lines.append(plt.plot(i_dst[1:], wi[1:]+.5*std_l[1:], 'b--')[0])
            elif p is not None:
                Is = np.asarray(Is, dtype=np.float32)
                env_p = func_envelope(Is, p, axis=0)
                env_50 = func_envelope(Is, 50, axis=0)
                env_100_p = func_envelope(Is, 100-p, axis=0)
                lines.append(plt.plot(i_dst[1:], env_p[1:], 'b--')[0])
                lines.append(plt.plot(i_dst[1:], env_50[1:], 'b')[0])
                lines.append(plt.plot(i_dst[1:], env_100_p[1:], 'b--')[0])
        else:
            for (uni, k_dst, k_num, k_dem, color) in zip(self.__unis, self.__Ks[0], self.__K_nums, self.__K_dems, colors):
                line, = plt.plot(k_dst[1:], self.__compute_I(k_num, k_dem)[1:],
                                 c=color, label=uni.get_name())
                lines.append(line)
        # Get legend data
        if legend:
            handles, labels = ax.get_legend_handles_labels()
        if self.__sK is not None:
            if p is None:
                i_dst, i_50 = self.__compute_I_env(self.__sK[0], self.__sims_K_num, self.__sims_K_dem)
                plt.plot(i_dst[1:], i_50[1:], 'k')
            else:
                i_dst, i_p, i_50, i_100_p = self.__compute_I_env(self.__sK[0], self.__sims_K_num, self.__sims_K_dem,
                                                                 p)
                plt.plot(i_dst[1:], i_p[1:], 'k--')
                plt.plot(i_dst[1:], i_50[1:], 'k')
                plt.plot(i_dst[1:], i_100_p[1:], 'k--')

        # Show data
        plt.show(block=block)
        if out_file is not None:
            out_ext = os.path.splitext(out_file)[1]
            if out_ext == '.pkl':
                fig_pickle(fig, out_file)
            else:
                plt.savefig(out_file)
        plt.close()

        # Store legend
        if legend and (out_file is not None):
            store_legend(handles, labels, out_file)

    # Analyze (plot in a figure) Ripley's O function, analyze_O() must be called before
    # w: ring width (default 1)
    # block: if True (default) the code stops until plotted window is closed
    # out_file: path to file to store the analysis, if None (default) is not stored
    # p: percentile, if None (default) only 50% is computed, otherwise p, 50, 100-p are computed
    # legend: if True (default False) and out_file is not None, a legend is also stored
    # gstd: scale dependant lines with standard deviation from different experimental data with scale, only valid
    #       when gathexier is True. (default False)
    def analyze_O(self, w=1, block=True, out_file=None, legend=False, gather=False, p=None, gstd=False):

        # Computations
        if self.__Ks is None:
            error_msg = 'analyze_K() must be called before with the same input parameters.'
            raise pexceptions.PySegInputError(expr='analyze_O (UniStat)', msg=error_msg)
        colors = cm.rainbow(np.linspace(0, 1, len(self.__unis)))

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.title('O-ring')
        plt.xlabel('Distance (nm)')
        plt.ylabel('O')
        # Plot data analysis
        lines = list()
        mode_2D = self.__unis[0].is_2D()
        if gather:
            o_dst = self.__Ks[0][0]
            wo = np.zeros(shape=o_dst.shape[0], dtype=np.float)
            weights = list()
            for uni in self.__unis:
                weights.append(uni.get_intensity())
            weights = np.asarray(weights, dtype=np.float)
            weights /= weights.sum()
            Os = list()
            for (ww, k) in zip(weights, self.__Ks[:][1]):
                Os.append(self.__compute_O(o_dst, k, w, mode_2D))
                wo += (Os[-1]*ww)

            if gstd:
                std_l = np.std(Os, axis=0)
                lines.append(plt.plot(o_dst[1:], wo[1:]-.5*std_l[1:], 'b--')[0])
                lines.append(plt.plot(o_dst[1:], wo[1:], 'b')[0])
                lines.append(plt.plot(o_dst[1:], wo[1:]+.5*std_l[1:], 'b--')[0])
            elif p is not None:
                Os = np.asarray(Os, dtype=np.float32)
                env_p = func_envelope(Os, p, axis=0)
                env_50 = func_envelope(Os, 50, axis=0)
                env_100_p = func_envelope(Os, 100-p, axis=0)
                lines.append(plt.plot(o_dst[1:], env_p[1:], 'b--')[0])
                lines.append(plt.plot(o_dst[1:], env_50[1:], 'b')[0])
                lines.append(plt.plot(o_dst[1:], env_100_p[1:], 'b--')[0])
        else:
            for (uni, o_dst, o, color) in zip(self.__unis, self.__Ks[0], self.__Ks[1], colors):
                line, = plt.plot(o_dst[2:], self.__compute_O(o_dst, o, w, mode_2D)[2:], c=color, label=uni.get_name())
                lines.append(line)
        # Get legend data
        if legend:
            handles, labels = ax.get_legend_handles_labels()
        if self.__sK is not None:
            if p is None:
                o_dst, o_50 = self.__compute_O_env(self.__sK[0], self.__sims_K, w, mode_2D)
                plt.plot(o_dst[2:], o_50[2:], 'k')
            else:
                o_dst, o_p, o_50, o_100_p = self.__compute_O_env(self.__sK[0], self.__sims_K, w, mode_2D, p)
                plt.plot(o_dst[2:], o_p[2:], 'k--')
                plt.plot(o_dst[2:], o_50[2:], 'k')
                plt.plot(o_dst[2:], o_100_p[2:], 'k--')

        # Show data
        plt.show(block=block)
        if out_file is not None:
            out_ext = os.path.splitext(out_file)[1]
            if out_ext == '.pkl':
                fig_pickle(fig, out_file)
            else:
                plt.savefig(out_file)
        plt.close()

        # Store legend
        if legend and (out_file is not None):
            store_legend(handles, labels, out_file)

    # Analyze (plot in a figure) intensity over scales, analyze_D() must be called before
    # w: ring width (default 1)
    # block: if True (default) the code stops until plotted window is closed
    # out_file: path to file to store the analysis, if None (default) is not stored
    # gather: if not None (default) then the graphs from the different inserted UniStat objects are collapsed
    #         into one
    # p: percentile, if None (default) only 50% is computed, otherwise p, 50, 100-p are computed
    # legend: if True (default False) and out_file is not None, a legend is also stored
    # gstd: scale dependant lines with standard deviation from different experimental data with scale, only valid
    #       when gather option is True. (default False)
    def analyze_D(self, w=1, block=True, out_file=None, legend=False, gather=False, p=None, gstd=False):

        # Computations
        if (self.__K_nums is None) or (self.__K_dems is None):
            error_msg = 'analyze_K() must be called before with the same input parameters.'
            raise pexceptions.PySegInputError(expr='analyze_D (UniStat)', msg=error_msg)
        colors = cm.rainbow(np.linspace(0, 1, len(self.__unis)))

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.title('Intensity variation over scales')
        plt.xlabel('Distance (nm)')
        mode_2D = self.__unis[0].is_2D()
        plt.ylabel('Intensity derivative')
        # Plot data analysis
        lines = list()
        if gather:
            d_dst = self.__Ks[0][0]
            wd = np.zeros(shape=d_dst.shape[0], dtype=np.float)
            weights = list()
            for uni in self.__unis:
                weights.append(uni.get_intensity())
            weights = np.asarray(weights, dtype=np.float)
            weights /= weights.sum()
            Ds = list()
            for (w, k_num, k_dem) in zip(weights, self.__K_nums, self.__K_dems):
                Ds.append(self.__compute_D(d_dst, k_num, k_dem, w))
                wd += (Ds[-1]*w)

            if gstd:
                std_l = np.std(Ds, axis=0)
                lines.append(plt.plot(d_dst[1:], wd[1:]-.5*std_l[1:], 'b--')[0])
                lines.append(plt.plot(d_dst[1:], wd[1:], 'b')[0])
                lines.append(plt.plot(d_dst[1:], wd[1:]+.5*std_l[1:], 'b--')[0])
            elif p is not None:
                Ds = np.asarray(Ds, dtype=np.float32)
                env_p = func_envelope(Ds, p, axis=0)
                env_50 = func_envelope(Ds, 50, axis=0)
                env_100_p = func_envelope(Ds, 100-p, axis=0)
                lines.append(plt.plot(d_dst[1:], env_p[1:], 'b--')[0])
                lines.append(plt.plot(d_dst[1:], env_50[1:], 'b')[0])
                lines.append(plt.plot(d_dst[1:], env_100_p[1:], 'b--')[0])
        else:
            for (uni, k_dst, k_num, k_dem, color) in zip(self.__unis, self.__Ks[0], self.__K_nums, self.__K_dems, colors):
                line, = plt.plot(k_dst[1:], self.__compute_D(k_dst, k_num, k_dem, w)[1:],
                                 c=color, label=uni.get_name())
                lines.append(line)
        # Get legend data
        if legend:
            handles, labels = ax.get_legend_handles_labels()
        if self.__sK is not None:
            if p is None:
                d_dst, d_50 = self.__compute_D_env(self.__sK[0], self.__sims_K_num, self.__sims_K_dem, w)
                plt.plot(d_dst[1:], d_50[1:], 'k')
            else:
                d_dst, d_p, d_50, d_100_p = self.__compute_D_env(self.__sK[0], self.__sims_K_num, self.__sims_K_dem,
                                                                 w, p)
                plt.plot(d_dst[1:], d_p[1:], 'k--')
                plt.plot(d_dst[1:], d_50[1:], 'k')
                plt.plot(d_dst[1:], d_100_p[1:], 'k--')

        # Show data
        plt.show(block=block)
        if out_file is not None:
            out_ext = os.path.splitext(out_file)[1]
            if out_ext == '.pkl':
                fig_pickle(fig, out_file)
            else:
                plt.savefig(out_file)
        plt.close()

        # Store legend
        if legend and (out_file is not None):
            store_legend(handles, labels, out_file)

    # out_dir: output directory where the points coordinate graphs will be stored
    # to_vtp: if True (default False) results are stored as vtp file
    def save_points(self, out_dir, to_vtp=False):
        for i, uni in enumerate(self.__unis):
            name = uni.get_name()
            if name is None:
                name = 'noname_' + str(i)
            if to_vtp:
                name += '.vtp'
            uni.plot_points(block=False, out_file=out_dir+'/'+name)

    # ### Internal functionality

    def __compute_Gs(self, max_d, bins, gather=False):

        # Initialization
        dsts, Gs = list(), list()

        # Loop for computations
        if gather:
            if len(self.__unis) > 0:
                hold_dists = nnde(self.__unis[0]._UniStat__coords)
                for uni in self.__unis[1:]:
                    hold_dists = np.concatenate((hold_dists, nnde(uni._UniStat__coords)))
            try:
                dst, g = compute_cdf(hold_dists, bins, max_d)
                dsts.append(dst)
                Gs.append(g)
            except pexceptions.PySegInputWarning as e:
                print e.get_message()
                raise pexceptions.PySegInputError('__compute_Gs (PlotUni)', e.get_message())
        else:
            for uni in self.__unis:
                try:
                    dst, g = uni.compute_G(max_d, bins)
                    dsts.append(dst)
                    Gs.append(g)
                except pexceptions.PySegInputWarning as e:
                    print e.get_message()
                    raise pexceptions.PySegInputError('__compute_Gs (PlotUni)', e.get_message())

        # Updating class variable
        self.__Gs = (dsts, Gs)

    def __compute_Fs(self, max_d, bins, f_npts, gather=False):

        # Initialization
        dsts, Fs = list(), list()
        simulator = CSRSimulator()

        # Loop for computations
        if gather:
            if len(self.__unis) > 0:
                r_coords = simulator.gen_rand_in_mask(f_npts, self.__unis[0]._UniStat__mask)
                r_coords *= self.__unis[0].get_resolution()
                hold_dists = cnnde(self.__unis[0]._UniStat__coords, r_coords)
                for uni in self.__unis[1:]:
                    r_coords = simulator.gen_rand_in_mask(f_npts, uni._UniStat__mask)
                    r_coords *= uni.get_resolution()
                    hold_dists = np.concatenate((hold_dists, cnnde(uni._UniStat__coords, r_coords)))
            try:
                dst, f = compute_cdf(hold_dists, bins, max_d)
                dsts.append(dst)
                Fs.append(f)
            except pexceptions.PySegInputWarning as e:
                print e.get_message()
                raise pexceptions.PySegInputError('__compute_Gs (PlotUni)', e.get_message())
        else:
            for uni in self.__unis:
                dst, f = uni.compute_F(max_d, bins, f_npts)
                dsts.append(dst)
                Fs.append(f)

        # Updating class variable
        self.__Fs = (dsts, Fs)

    def __compute_Ks(self, max_d, n_samp):

        # Initialization
        dsts, Ks, K_nums, K_dems = list(), list(), list(), list()

        # Loop for computations
        for uni in self.__unis:
            dst, k, K_num, K_dem = uni.compute_K(max_d, n_samp, get_nd=True)
            dsts.append(dst)
            Ks.append(k)
            K_nums.append(K_num)
            K_dems.append(K_dem)

        # Updating class variable
        self.__Ks = (dsts, Ks)
        self.__K_nums = K_nums
        self.__K_dems = K_dems

    def __compute_NLDs(self, shell_th, max_d, n_samp, gnorm=True):

        # Initialization
        dsts, NLDs = list(), list()

        # Loop for computations
        for uni in self.__unis:
            dst, n = uni.compute_NLD(shell_th, max_d, n_samp, gnorm=gnorm)
            dsts.append(dst)
            NLDs.append(n)
        # Updating class variable
        self.__NLDs = (dsts, NLDs)

    def __compute_RBFs(self, shell_th, max_d, n_samp):

        # Initialization
        dsts, RBFs = list(), list()

        # Loop for computations
        for uni in self.__unis:
            dst, n = uni.compute_RBF(shell_th, max_d, n_samp)
            dsts.append(dst)
            RBFs.append(n)
        # Updating class variable
        self.__RBFs = (dsts, RBFs)

    def __compute_L(self, r, K, mode_2D):

        if mode_2D:
            return np.sqrt(K/np.pi) - r
        else:
            return np.power(K/((4./3.)*np.pi), CSQRT_EXP) - r

    def __compute_I(self, K_num, K_dem):

        return K_num / K_dem

    def __compute_L_env(self, r, Ks, mode_2D, p=None):

        # Compute L from K
        Ls = np.zeros(shape=Ks.shape, dtype=np.float)
        if mode_2D:
            for i in range(Ks.shape[1]):
                Ls[:, i] = np.sqrt(Ks[:, i]/np.pi) - r
        else:
            for i in range(Ks.shape[1]):
                Ls[:, i] = np.power(Ks[:, i]/((4./3.)*np.pi), CSQRT_EXP) - r

        # Compute envelopes
        k_50 = func_envelope(Ls, per=50)
        if p is None:
            return r, k_50
        else:
            k_p = func_envelope(Ls, per=p)
            k_100_p = func_envelope(Ls, per=100-p)
            return r, k_p, k_50, k_100_p

    def __compute_I_env(self, r, Ks_num, Ks_dem, p=None):

        # Compute L from K
        Is = np.zeros(shape=Ks_num.shape, dtype=np.float)
        for i in range(Ks_num.shape[1]):
            Is[:, i] = Ks_num[:, i] / Ks_dem[:, i]

        # Compute envelopes
        k_50 = func_envelope(Is, per=50)
        if p is None:
            return r, k_50
        else:
            k_p = func_envelope(Is, per=p)
            k_100_p = func_envelope(Is, per=100-p)
            return r, k_p, k_50, k_100_p

    def __compute_O(self, r, K, w, mode_2D):

        # Filtration
        n_s = r.shape[0]
        m = 1. / float(1.-n_s)
        c = 1. - m
        wn = m*w + c
        if wn < 0:
            wn = .0
        elif wn > 1:
            wn = 1.
        b, a = butter(LP_ORDER, wn, btype='low', analog=False)
        f_K = lfilter(b, a, K)

        # Derivation
        K_d = np.gradient(f_K, r[1]-r[0])

        # Packing output
        o_ring = np.zeros(shape=K_d.shape, dtype=np.float)
        if mode_2D:
            o_ring[1:] = (1./(2*np.pi*r[1:])) * K_d[1:] - 1.
        else:
            o_ring[1:] = (1./(4*np.pi*r[1:]*r[1:])) * K_d[1:] - 1.

        return o_ring

    def __compute_D(self, r, K_num, K_dem, w):

        # Filtration
        n_s = r.shape[0]
        m = 1. / float(1.-n_s)
        c = 1. - m
        wn = m*w + c
        if wn < 0:
            wn = .0
        elif wn > 1:
            wn = 1.
        b, a = butter(LP_ORDER, wn, btype='low', analog=False)
        I = self.__compute_I(K_num, K_dem)
        f_I = lfilter(b, a, I)

        # Derivation
        I_d = np.gradient(f_I, r[1]-r[0])

        # Packing output
        d_arr = np.zeros(shape=I_d.shape, dtype=np.float)
        d_arr[1:] = I_d[1:]

        return d_arr

    def __compute_O_env(self, r, Ks, w, mode_2D, p=None):

        # Filter initialization
        n_s = r.shape[0]
        m = 1. / float(1.-n_s)
        c = 1. - m
        wn = m*w + c
        if wn < 0:
            wn = .0
        elif wn > 1:
            wn = 1.
        b, a = butter(LP_ORDER, wn, btype='low', analog=False)
        Os = np.zeros(shape=Ks.shape, dtype=np.float)

        # Computing O from K
        for i in range(Ks.shape[1]):
            # Filtering
            f_K = lfilter(b, a, Ks[:, i])
            # Derivation
            K_d = np.gradient(f_K, r[1]-r[0])

            # Computing O
            if mode_2D:
                Os[1:, i] = (1./(2*np.pi*r[1:])) * K_d[1:] - 1.
            else:
                Os[1:, i] = (1./(4*np.pi*r[1:]*r[1:])) * K_d[1:] - 1.

        # Compute envelopes
        k_50 = func_envelope(Os, per=50)
        if p is None:
            return r, k_50
        else:
            k_p = func_envelope(Os, per=p)
            k_100_p = func_envelope(Os, per=100-p)
            return r, k_p, k_50, k_100_p

    def __compute_D_env(self, r, Ks_num, Ks_dem, w, p=None):

        # Filter initialization
        n_s = r.shape[0]
        m = 1. / float(1.-n_s)
        c = 1. - m
        wn = m*w + c
        if wn < 0:
            wn = .0
        elif wn > 1:
            wn = 1.
        b, a = butter(LP_ORDER, wn, btype='low', analog=False)
        Is = np.zeros(shape=Ks_num.shape, dtype=np.float)

        # Computing O from K
        for i in range(Ks_num.shape[1]):
            # Intensity
            I = self.__compute_I(Ks_num[:, i], Ks_dem[:, i])
            # Filtering
            f_I = lfilter(b, a, I)
            # Derivation
            I_d = np.gradient(f_I, r[1]-r[0])

            # Computing O
            Is[1:, i] = I_d[1:]

        # Compute envelopes
        k_50 = func_envelope(Is, per=50)
        if p is None:
            return r, k_50
        else:
            k_p = func_envelope(Is, per=p)
            k_100_p = func_envelope(Is, per=100-p)
            return r, k_p, k_50, k_100_p

    def __simulate_G(self, max_d, bins, n_sim, p):

        # Divide statically
        if n_sim < len(self.__unis):
            n_sim = len(self.__unis)
        spl_ids = np.array_split(np.arange(n_sim), len(self.__unis))

        # Loop for computations
        count = 0
        sims = list()
        # sims = np.zeros(shape=(bins, n_sim), dtype=np.float)
        for (uni, ids) in zip(self.__unis, spl_ids):
            h_n_sim = len(ids)
            if uni.get_n_points() >= bins:
                dst, hold_sims = uni.simulate_G(max_d, bins, h_n_sim, get_sim=True)
                for j in range(hold_sims.shape[1]):
                    # sims[:, count] = hold_sims[:, j]
                    sims.append(hold_sims[:, j])
                    count += 1
            else:
                sims.append(np.zeros(shape=bins, dtype=np.float))
        if len(sims) == 0:
            sims = np.zeros(shape=(bins, 1), dtype=np.float)
        else:
            sims = np.asarray(sims, dtype=np.float).transpose()

        # Updating class simulations variable
        self.__sims_G = sims

        # Compute envelopes
        g_50 = func_envelope(sims, per=50)
        if p is None:
            self.__sG = (dst, g_50)
        else:
            g_p = func_envelope(sims, per=p)
            g_100_p = func_envelope(sims, per=100-p)
            self.__sG = (dst, g_p, g_50, g_100_p)

    def __simulate_F(self, max_d, bins, f_npts, n_sim, p):

        # Divide statically
        if n_sim < len(self.__unis):
            n_sim = len(self.__unis)
        spl_ids = np.array_split(np.arange(n_sim), len(self.__unis))

        # Loop for computations
        count = 0
        sims = list()
        # sims = np.zeros(shape=(bins, n_sim), dtype=np.float)
        for (uni, ids) in zip(self.__unis, spl_ids):
            h_n_sim = len(ids)
            if uni.get_n_points() >= bins:
                dst, hold_sims = uni.simulate_F(max_d, bins, f_npts, h_n_sim, get_sim=True)
                for j in range(hold_sims.shape[1]):
                    # sims[:, count] = hold_sims[:, j]
                    sims.append(hold_sims[:, j])
                    count += 1
            else:
                sims.append(np.zeros(shape=bins, dtype=np.float))
        if len(sims) == 0:
            sims = np.zeros(shape=(bins, 1), dtype=np.float)
        else:
            sims = np.asarray(sims, dtype=np.float).transpose()

        # Updating class simulations variable
        self.__sims_F = sims

        # Compute envelopes
        f_50 = func_envelope(sims, per=50)
        if p is None:
            self.__sG = (dst, f_50)
        else:
            f_p = func_envelope(sims, per=p)
            f_100_p = func_envelope(sims, per=100-p)
            self.__sF = (dst, f_p, f_50, f_100_p)

    def __simulate_K(self, max_d, n_samp, n_sim, p):

        # Divide statically
        if n_sim < len(self.__unis):
            n_sim = len(self.__unis)
        spl_ids = np.array_split(np.arange(n_sim), len(self.__unis))

        # Loop for computations
        count = 0
        sims = np.zeros(shape=(n_samp, n_sim), dtype=np.float)
        sims_K_num = np.zeros(shape=(n_samp, n_sim), dtype=np.float)
        sims_K_dem = np.zeros(shape=(n_samp, n_sim), dtype=np.float)
        for (uni, ids) in zip(self.__unis, spl_ids):
            h_n_sim = len(ids)
            dst, hold_sims, hold_K_nums, hold_K_dems = uni.simulate_K(max_d, n_samp, h_n_sim, get_sim=True)
            for j in range(hold_sims.shape[1]):
                sims[:, count] = hold_sims[:, j]
                sims_K_num[:, count] = hold_K_nums[:, j]
                sims_K_dem[:, count] = hold_K_dems[:, j]
                count += 1

        # Updating class simulations variable
        self.__sims_K = sims
        self.__sims_K_num = sims_K_num
        self.__sims_K_dem = sims_K_dem

        # Compute envelopes
        k_50 = func_envelope(sims, per=50)
        if p is None:
            self.__sK = (dst, k_50)
        else:
            k_p = func_envelope(sims, per=p)
            k_100_p = func_envelope(sims, per=100-p)
            self.__sK = (dst, k_p, k_50, k_100_p)

    def __simulate_NLD(self, shell_th, max_d, n_samp, n_sim, p, gnorm=True):

        # Divide statically
        if n_sim < len(self.__unis):
            n_sim = len(self.__unis)
        spl_ids = np.array_split(np.arange(n_sim), len(self.__unis))

        # Loop for computations
        count = 0
        sims = np.zeros(shape=(n_samp, n_sim), dtype=np.float)
        for (uni, ids) in zip(self.__unis, spl_ids):
            h_n_sim = len(ids)
            dst, hold_sims = uni.simulate_NLD(shell_th, max_d, n_samp, h_n_sim, get_sim=True, gnorm=gnorm)
            for j in range(hold_sims.shape[1]):
                sims[:, count] = hold_sims[:, j]
                count += 1

        # Compute envelopes
        n_50 = func_envelope(sims, per=50)
        if p is None:
            self.__sNLD = (dst, n_50)
        else:
            n_p = func_envelope(sims, per=p)
            n_100_p = func_envelope(sims, per=100-p)
            self.__sNLD = (dst, n_p, n_50, n_100_p)

    # So as to prevent non-sense results the arrays may be are cropped
    # high: maximum allowed value for cropping the output array (default 0.95)
    def __compute_J(self, g, f, high=.98):
        mx_g, mx_f = (g < high).sum(), (f < high).sum()
        if mx_g < mx_f:
            return (1.-g[:mx_g]) / (1.-f[:mx_g])
        else:
            return (1.-g[:mx_f]) / (1.-f[:mx_f])

    def __compute_J_env(self, Gs, Fs, p=None):

        # Compute L from K
        Js = np.inf * np.ones(shape=Gs.shape, dtype=np.float)
        for i in range(Gs.shape[1]):
            hold = self.__compute_J(Gs[:, i], Fs[:, i])
            Js[:hold.shape[0], i] = hold

        # Compute envelopes
        j_50 = func_envelope(Js, per=50)
        mx_j_50 = j_50.shape[0]
        mx_ids = np.where(j_50 >= np.inf)[0]
        if len(mx_ids) > 0:
            mx_j_50 = mx_ids[0]
        if p is None:
            return mx_j_50, j_50
        else:
            j_p = func_envelope(Js, per=p)
            j_100_p = func_envelope(Js, per=100-p)
            mx_j_p, mx_j_100_p = j_p.shape[0], j_100_p.shape[0]
            mx_ids_p = np.where(j_p >= np.inf)[0]
            if len(mx_ids_p) > 0:
                mx_j_p = mx_ids_p[0]
            mx_ids_100_p = np.where(j_100_p >= np.inf)[0]
            if len(mx_ids_100_p) > 0:
                mx_j_100_p = mx_ids_100_p[0]
            return mx_j_p, mx_j_50, mx_j_100_p, j_p, j_50, j_100_p

    def __store_legend(self, handles, labels, out_file):
        plt.figlegend(handles, labels, loc='upper center')
        stem, ext = os.path.splitext(out_file)
        plt.savefig(stem + '_legend' + ext)
        plt.close()


##############################################################################################
# Class for plotting univaritate analysis
#
class PlotBi(object):

    # ### Constructor area

    def __init__(self, name=None):
        self.__name = name
        self.__Gs, self.__sG = None, None
        self.__Ks, self.__sK, self.__sims_K = None, None, None
        self.__bis = list()

    # ### External functionality area

    def insert_bi(self, bi):
        self.__bis.append(bi)

    # Return True if not all patterns have the same intensity
    def not_eq_intensities(self):
        if len(self.__bis) > 0:
            hold_int = self.__bis[0].get_intensities()[1]
            for bi in self.__bis[1:]:
                if bi.get_intensities()[1] != hold_int:
                    return True
        return False

    # Pickling the object state
    # fname: full path for the pickle file
    def pickle(self, fname):
        pkl_f = open(fname, 'w')
        try:
            pickle.dump(self, pkl_f)
        finally:
            pkl_f.close()

    # Analyze (plot in a figure) function G
    # max_d: maximum distance for computing the histograms in nm
    # bins: number of samples for output (default 50)
    # n_sim: number of simulations for CSR (default 0), if < 1 disabled
    # p: percentile, if None (default) only 50% is computed, otherwise p, 50, 100-p are computed
    # update: if True (default False) re-computing data if forced
    # block: if True (default) the code stops until plotted window is closed
    # out_file: path to file to store the analysis, if None (default) is not stored
    # legend: if True (default False) and out_file is not None, a legend is also stored
    # gather: if not None (default) then the graphs from the different inserted BiStat objects are collapsed
    #         into one
    def analyze_G(self, max_d, bins=50, n_sim=0, p=None, update=False, block=True, out_file=None, legend=False,
                  gather=False):

        # Computations
        if update or (self.__Gs is None):
            self.__Gs, self.__sG = None, None
            self.__compute_Gs(max_d, bins, gather)
        if n_sim > 0:
            if update or (self.__sG is None):
                self.__simulate_G(max_d, bins, n_sim, p)
        colors = cm.rainbow(np.linspace(0, 1, len(self.__bis)))

        if self.not_eq_intensities():
            print 'WARNING: Function G is not a proper metric for comparing patterns with different ' \
                  'intensities'

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.title('G-Function (Bivariate)')
        plt.xlabel('Distance (nm)')
        plt.ylabel('G')
        plt.ylim(0, 1)
        # Plot data analysis
        for (bi, g_dst, g, color) in zip(self.__bis, self.__Gs[:][0], self.__Gs[:][1], colors):
           ax.plot(g_dst, g, c=color, label=bi.get_name())
        # Get legend data
        if legend:
            handles, labels = ax.get_legend_handles_labels()
        # Plot random references
        if n_sim > 0:
            if len(self.__sG) == 4:
                plt.plot(self.__sG[0], self.__sG[1], 'k--')
                plt.plot(self.__sG[0], self.__sG[2], 'k')
                plt.plot(self.__sG[0], self.__sG[3], 'k--')
            else:
                plt.plot(self.__sG[0], self.__sG[2], 'k')

        # Show data
        plt.show(block=block)
        if out_file is not None:
            out_ext = os.path.splitext(out_file)[1]
            if out_ext == '.pkl':
                fig_pickle(fig, out_file)
            else:
                plt.savefig(out_file)
        plt.close()

        # Store legend
        if legend and (out_file is not None):
            store_legend(handles, labels, out_file)

    # Analyze (plot in a figure) function K
    # max_d: maximum distance
    # n_samp: number of samples, it must be greater than 2
    # n_sim: number of simulations for CSR (default 0), if < 1 disabled
    # p: percentile, if None (default) only 50% is computed, otherwise p, 50, 100-p are computed
    # update: if True (default False) re-computing data if forced
    # block: if True (default) the code stops until plotted window is closed
    # out_file: path to file to store the analysis, if None (default) is not stored
    # legend: if True (default False) and out_file is not None, a legend is also stored
    # gather: if not None (default) then the graphs from the different inserted UniStat objects are collapsed
    #         into one
    def analyze_K(self, max_d, n_samp=50, n_sim=0, p=None, update=False, block=True, out_file=None, legend=False,
                  gather=False):

        if n_samp <= 2:
            error_msg = 'Number of samples must be greater than 2'
            raise pexceptions.PySegInputError(expr='analyze_K (UniStat)', msg=error_msg)

        # Computations
        if update or (self.__Ks is None):
            self.__Ks, self.__sK = None, None
            self.__compute_Ks(max_d, n_samp)
        if n_sim > 0:
            if update or (self.__sK is None):
                self.__simulate_K(max_d, n_samp, n_sim, p)
        colors = cm.rainbow(np.linspace(0, 1, len(self.__bis)))

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.title('Ripley\'s K (Bivariate)')
        plt.xlabel('Radius (nm)')
        plt.ylabel('K')

        # Plot data analysis
        lines = list()
        if gather:
            k_dst = self.__Ks[0][0]
            wk = np.zeros(shape=k_dst.shape[0], dtype=np.float)
            weights = list()
            for bi in self.__bis:
                weights.append(bi.get_intensities()[0])
            weights = np.asarray(weights, dtype=np.float)
            weights /= weights.sum()
            for (w, k) in zip(weights, self.__Ks[:][1]):
                wk += (k*w)
            line, = plt.plot(k_dst[1:], wk[1:])
            lines.append(line)
        else:
            for (bi, k_dst, k, color) in zip(self.__bis, self.__Ks[:][0], self.__Ks[:][1], colors):
                ax.plot(k_dst[1:], k[1:], c=color, label=bi.get_name())
         # Get legend data
        if legend:
            handles, labels = ax.get_legend_handles_labels()
        # Plot random references
        if n_sim > 0:
            if len(self.__sK) == 4:
                plt.plot(self.__sK[0][1:], self.__sK[1][1:], 'k--')
                plt.plot(self.__sK[0][1:], self.__sK[2][1:], 'k')
                plt.plot(self.__sK[0][1:], self.__sK[3][1:], 'k--')
            else:
                plt.plot(self.__sK[0][1:], self.__sK[2][1:], 'k')

        # Show data
        plt.show(block=block)
        if out_file is not None:
            out_ext = os.path.splitext(out_file)[1]
            if out_ext == '.pkl':
                fig_pickle(fig, out_file)
            else:
                plt.savefig(out_file)
        plt.close()

        # Store legend
        if legend and (out_file is not None):
            store_legend(handles, labels, out_file)

    # Analyze (plot in a figure) Ripley's L function, analyze_K() must be called before
    # block: if True (default) the code stops until plotted window is closed
    # out_file: path to file to store the analysis, if None (default) is not stored
    # legend: if True (default False) and out_file is not None, a legend is also stored
    # gather: if not None (default) then the graphs from the different inserted UniStat objects are collapsed
    #         into one
    # p: percentile, if None (default) only 50% is computed, otherwise p, 50, 100-p are computed
    # gstd: scale dependant lines with standard deviation from different experimental data with scale, only valid
    #       when gathexier is True. (default False)
    def analyze_L(self, block=True, out_file=None, legend=False, gather=False, p=None, gstd=False):

        # Computations
        if self.__Ks is None:
            error_msg = 'analyze_K() must be called before with the same input parameters.'
            raise pexceptions.PySegInputError(expr='analyze_L (BiStat)', msg=error_msg)
        colors = cm.rainbow(np.linspace(0, 1, len(self.__bis)))

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.title('L-Function')
        plt.xlabel('Distance (nm)')
        plt.ylabel('L')
        # Plot data analysis
        mode_2D = self.__bis[0].is_2D()
        lines = list()
        if gather:
            l_dst = self.__Ks[0][0]
            wl = np.zeros(shape=l_dst.shape[0], dtype=np.float)
            weights = list()
            for bi in self.__bis:
                weights.append(bi.get_intensities()[0])
            weights = np.asarray(weights, dtype=np.float)
            weights /= weights.sum()
            Ls = list()
            for (w, k) in zip(weights, self.__Ks[:][1]):
                Ls.append(self.__compute_L(l_dst, k, mode_2D))
                wl += (Ls[-1]*w)

            if gstd:
                std_l = np.std(Ls, axis=0)
                lines.append(plt.plot(l_dst[1:], wl[1:]-.5*std_l[1:], 'b--')[0])
                lines.append(plt.plot(l_dst[1:], wl[1:], 'b')[0])
                lines.append(plt.plot(l_dst[1:], wl[1:]+.5*std_l[1:], 'b--')[0])
            elif p is not None:
                Ls = np.asarray(Ls, dtype=np.float32)
                env_p = func_envelope(Ls, p, axis=0)
                env_50 = func_envelope(Ls, 50, axis=0)
                env_100_p = func_envelope(Ls, 100-p, axis=0)
                lines.append(plt.plot(l_dst[1:], env_p[1:], 'b--')[0])
                lines.append(plt.plot(l_dst[1:], env_50[1:], 'b')[0])
                lines.append(plt.plot(l_dst[1:], env_100_p[1:], 'b--')[0])
        else:
            for (bi, k_dst, k, color) in zip(self.__bis, self.__Ks[0], self.__Ks[1], colors):
                ax.plot(k_dst[1:], self.__compute_L(k_dst, k, mode_2D)[1:], c=color, label=bi.get_name())
        # Get legend data
        if legend:
            handles, labels = ax.get_legend_handles_labels()
        if self.__sK is not None:
            if p is None:
                l_dst, l_50 = self.__compute_L_env(self.__sK[0], self.__sims_K, self.__bis[0].is_2D())
                plt.plot(l_dst[1:], l_50[1:], 'k')
            else:
                l_dst, l_p, l_50, l_100_p = self.__compute_L_env(self.__sK[0], self.__sims_K, self.__bis[0].is_2D(), p)
                plt.plot(l_dst[1:], l_p[1:], 'k--')
                plt.plot(l_dst[1:], l_50[1:], 'k')
                plt.plot(l_dst[1:], l_100_p[1:], 'k--')

        # Show data
        plt.show(block=block)
        if out_file is not None:
            out_ext = os.path.splitext(out_file)[1]
            if out_ext == '.pkl':
                fig_pickle(fig, out_file)
            else:
                plt.savefig(out_file)
        plt.close()

        # Store legend
        if legend and (out_file is not None):
            store_legend(handles, labels, out_file)

    # Analyze (plot in a figure) Ripley's O function, analyze_O() must be called before
    # w: ring width (default 1)
    # block: if True (default) the code stops until plotted window is closed
    # out_file: path to file to store the analysis, if None (default) is not stored
    # legend: if True (default False) and out_file is not None, a legend is also stored
    # gather: if not None (default) then the graphs from the different inserted UniStat objects are collapsed
    #         into one
    # p: percentile, if None (default) only 50% is computed, otherwise p, 50, 100-p are computed
    # gstd: scale dependant lines with standard deviation from different experimental data with scale, only valid
    #       when gathexier is True. (default False)
    def analyze_O(self, w=1, block=True, out_file=None, legend=False, gather=False, p=None, gstd=False):

        # Computations
        if self.__Ks is None:
            error_msg = 'analyze_K() must be called before with the same input parameters.'
            raise pexceptions.PySegInputError(expr='analyze_O (BiStat)', msg=error_msg)
        colors = cm.rainbow(np.linspace(0, 1, len(self.__bis)))

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.title('O-ring')
        plt.xlabel('Distance (nm)')
        plt.ylabel('O')
        # Plot data analysis
        mode_2D = self.__bis[0].is_2D()
        lines = list()
        if gather:
            o_dst = self.__Ks[0][0]
            wo = np.zeros(shape=o_dst.shape[0], dtype=np.float)
            weights = list()
            for bi in self.__bis:
                weights.append(bi.get_intensities()[0])
            weights = np.asarray(weights, dtype=np.float)
            weights /= weights.sum()
            Os = list()
            for (ww, k) in zip(weights, self.__Ks[:][1]):
                Os.append(self.__compute_O(o_dst, k, w, mode_2D))
                wo += (Os[-1]*ww)

            if gstd:
                std_l = np.std(Os, axis=0)
                lines.append(plt.plot(o_dst[1:], wo[1:]-.5*std_l[1:], 'b--')[0])
                lines.append(plt.plot(o_dst[1:], wo[1:], 'b')[0])
                lines.append(plt.plot(o_dst[1:], wo[1:]+.5*std_l[1:], 'b--')[0])
            elif p is not None:
                Os = np.asarray(Os, dtype=np.float32)
                env_p = func_envelope(Os, p, axis=0)
                env_50 = func_envelope(Os, 50, axis=0)
                env_100_p = func_envelope(Os, 100-p, axis=0)
                lines.append(plt.plot(o_dst[1:], env_p[1:], 'b--')[0])
                lines.append(plt.plot(o_dst[1:], env_50[1:], 'b')[0])
                lines.append(plt.plot(o_dst[1:], env_100_p[1:], 'b--')[0])
        else:
            for (bi, o_dst, o, color) in zip(self.__bis, self.__Ks[0], self.__Ks[1], colors):
                ax.plot(o_dst[2:], self.__compute_O(o_dst, o, w, mode_2D)[2:], c=color, label=bi.get_name())
        # Get legend data
        if legend:
            handles, labels = ax.get_legend_handles_labels()
        if self.__sK is not None:
            if p is None:
                o_dst, o_50 = self.__compute_O_env(self.__sK[0], self.__sims_K, w, mode_2D)
                plt.plot(o_dst[2:], o_50[2:], 'k')
            else:
                o_dst, o_p, o_50, o_100_p = self.__compute_O_env(self.__sK[0], self.__sims_K, w, mode_2D, p)
                plt.plot(o_dst[2:], o_p[2:], 'k--')
                plt.plot(o_dst[2:], o_50[2:], 'k')
                plt.plot(o_dst[2:], o_100_p[2:], 'k--')

        # Show data
        plt.show(block=block)
        if out_file is not None:
            out_ext = os.path.splitext(out_file)[1]
            if out_ext == '.pkl':
                fig_pickle(fig, out_file)
            else:
                plt.savefig(out_file)
        plt.close()

        # Store legend
        if legend and (out_file is not None):
            store_legend(handles, labels, out_file)

    # out_dir: output directory where the points coordinate graphs will be stored
    def save_points(self, out_dir):
        for i, bi in enumerate(self.__bis):
            name = bi.get_name()
            if name is None:
                name = 'noname_' + str(i)
            bi.plot_points(block=False, out_file=out_dir+'/'+name)

    # ### Internal functionality

    def __compute_Gs(self, max_d, bins, gather=False):

        # Initialization
        dsts, Gs = list(), list()

        # Loop for computations
        if gather:
            if len(self.__bis) > 0:
                hold_dists = cnnde(self.__bis[0]._BiStat__coords_2, self.__bis[0]._BiStat__coords_1)
                for bi in self.__bis[1:]:
                    hold_dists = np.concatenate((hold_dists, cnnde(bi._BiStat__coords_2, bi._BiStat__coords_1)))
            try:
                dst, g = compute_cdf(hold_dists, bins, max_d)
                dsts.append(dst)
                Gs.append(g)
            except pexceptions.PySegInputWarning as e:
                print e.get_message()
                raise pexceptions.PySegInputError('__compute_Gs (PlotBi)', e.get_message())
        else:
            for bi in self.__bis:
                try:
                    dst, g = bi.compute_G(max_d, bins)
                    dsts.append(dst)
                    Gs.append(g)
                except pexceptions.PySegInputWarning as e:
                    print e.get_message()
                    raise pexceptions.PySegInputError('__compute_Gs (PlotBi)', e.get_message())

        # Updating class variable
        self.__Gs = (dsts, Gs)

    def __compute_Ks(self, max_d, n_samp):

        # Initialization
        dsts, Ks = list(), list()

        # Loop for computations
        for bi in self.__bis:
            dst, k = bi.compute_K(max_d, n_samp)
            dsts.append(dst)
            Ks.append(k)

        # Updating class variable
        self.__Ks = (dsts, Ks)

    def __compute_L(self, r, K, mode_2D):

        if mode_2D:
            return np.sqrt(K/np.pi) - r
        else:
            return np.power(K/((4./3.)*np.pi), CSQRT_EXP) - r

    def __compute_L_env(self, r, Ks, mode_2D, p=None):

        # Compute L from K
        Ls = np.zeros(shape=Ks.shape, dtype=np.float)
        if mode_2D:
            for i in range(Ks.shape[1]):
                Ls[:, i] = np.sqrt(Ks[:, i]/np.pi) - r
        else:
            for i in range(Ks.shape[1]):
                Ls[:, i] = np.power(Ks[:, i]/((4./3.)*np.pi), CSQRT_EXP) - r

        # Compute envelopes
        k_50 = func_envelope(Ls, per=50)
        if p is None:
            return r, k_50
        else:
            k_p = func_envelope(Ls, per=p)
            k_100_p = func_envelope(Ls, per=100-p)
            return r, k_p, k_50, k_100_p

    def __compute_O(self, r, K, w, mode_2D):

        # Filtration
        n_s = r.shape[0]
        m = 1. / float(1.-n_s)
        c = 1. - m
        wn = m*w + c
        if wn < 0:
            wn = .0
        elif wn > 1:
            wn = 1.
        b, a = butter(LP_ORDER, wn, btype='low', analog=False)
        f_K = lfilter(b, a, K)

        # Derivation
        K_d = np.gradient(f_K, r[1]-r[0])

        # Packing output
        o_ring = np.zeros(shape=K_d.shape, dtype=np.float)
        if mode_2D:
            o_ring[1:] = (1./(2*np.pi*r[1:])) * K_d[1:] - 1.
        else:
            o_ring[1:] = (1./(4*np.pi*r[1:]*r[1:])) * K_d[1:] - 1.

        return o_ring

    def __compute_O_env(self, r, Ks, w, mode_2D, p=None):

        # Filter initialization
        n_s = r.shape[0]
        m = 1. / float(1.-n_s)
        c = 1. - m
        wn = m*w + c
        if wn < 0:
            wn = .0
        elif wn > 1:
            wn = 1.
        b, a = butter(LP_ORDER, wn, btype='low', analog=False)
        Os = np.zeros(shape=Ks.shape, dtype=np.float)

        # Computing O from K
        for i in range(Ks.shape[1]):
            # Filtering
            f_K = lfilter(b, a, Ks[:, i])
            # Derivation
            K_d = np.gradient(f_K, r[1]-r[0])

            # Computing O
            if mode_2D:
                Os[1:, i] = (1./(2*np.pi*r[1:])) * K_d[1:] - 1.
            else:
                Os[1:, i] = (1./(4*np.pi*r[1:]*r[1:])) * K_d[1:] - 1.

        # Compute envelopes
        k_50 = func_envelope(Os, per=50)
        if p is None:
            return r, k_50
        else:
            k_p = func_envelope(Os, per=p)
            k_100_p = func_envelope(Os, per=100-p)
            return r, k_p, k_50, k_100_p

    def __simulate_G(self, max_d, bins, n_sim, p):

        # Divide statically
        if n_sim < len(self.__bis):
            n_sim = len(self.__bis)
        spl_ids = np.array_split(np.arange(n_sim), len(self.__bis))

        # Loop for computations
        count = 0
        sims = np.zeros(shape=(bins, n_sim), dtype=np.float)
        for (bi, ids) in zip(self.__bis, spl_ids):
            h_n_sim = len(ids)
            dst, hold_sims = bi.simulate_G(max_d, bins, h_n_sim, get_sim=True)
            for j in range(hold_sims.shape[1]):
                sims[:, count] = hold_sims[:, j]
                count += 1

        # Updating class simulations variable
        self.__sims_G = sims

        # Compute evelopes
        g_50 = func_envelope(sims, per=50)
        if p is None:
            self.__sG = (dst, g_50)
        else:
            g_p = func_envelope(sims, per=p)
            g_100_p = func_envelope(sims, per=100-p)
            self.__sG = (dst, g_p, g_50, g_100_p)

    def __simulate_K(self, max_d, n_samp, n_sim, p):

        # Divide statically
        if n_sim < len(self.__bis):
            n_sim = len(self.__bis)
        spl_ids = np.array_split(np.arange(n_sim), len(self.__bis))

        # Loop for computations
        count = 0
        sims = np.zeros(shape=(n_samp, n_sim), dtype=np.float)
        for (bi, ids) in zip(self.__bis, spl_ids):
            h_n_sim = len(ids)
            dst, hold_sims = bi.simulate_K(max_d, n_samp, h_n_sim, get_sim=True)
            for j in range(hold_sims.shape[1]):
                sims[:, count] = hold_sims[:, j]
                count += 1

        # Updating class simulations variable
        self.__sims_K = sims

        # Compute envelopes
        k_50 = func_envelope(sims, per=50)
        if p is None:
            self.__sK = (dst, k_50)
        else:
            k_p = func_envelope(sims, per=p)
            k_100_p = func_envelope(sims, per=100-p)
            self.__sK = (dst, k_p, k_50, k_100_p)

    # So as to prevent non-sense results the arrays may be are cropped
    # high_f: maximum allowed value for cropping the output array (default 0.95)
    def __compute_J(self, g, f, high_f=.95):
        mx = (f < high_f).sum()
        return (1.-g[:mx]) / (1.-f[:mx])