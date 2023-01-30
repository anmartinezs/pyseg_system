"""
Classes for doing the spatial analysis of clouds of points in a 2D plane

# Author: Antonio Martinez-Sanchez (Max Planck Institute for Biochemistry)
# Date: 12.06.15
"""

__author__ = 'martinez'

import sys
import shutil
from pyseg.globals import *
from .variables import *
from .globals import FilVisitor2
from abc import *
import matplotlib.pyplot as plt
from skimage.morphology import convex_hull_image
from matplotlib.pyplot import cm
from scipy.signal import butter, lfilter
try:
    import pexceptions
except:
    from pyseg import pexceptions

try:
    import pickle as pickle
except:
    import pickle

##### PACKAGE VARIABLES

LP_ORDER = 5
LP_NORM_CUTOFF = 0.3

#### PACKAGE FUNCTION

# Convert a 3D bounding box into a 2D one
# bbox_3d: input list/tuple/array with the 3D bounding box [x_min, y_min, z_min, x_max, y_max, z_max]
# coord: coordinate to delete: 0, 1, 2 (default)
def make_plane_box(box_3d, coord=2):

    bbox_2d = np.zeros(shape=4, dtype=float)

    if coord == 0:
        bbox_2d[0] = box_3d[1]
        bbox_2d[1] = box_3d[2]
        bbox_2d[2] = box_3d[4]
        bbox_2d[3] = box_3d[5]
    elif coord == 1:
        bbox_2d[0] = box_3d[0]
        bbox_2d[1] = box_3d[2]
        bbox_2d[2] = box_3d[3]
        bbox_2d[3] = box_3d[5]
    else:
        bbox_2d[0] = box_3d[1]
        bbox_2d[1] = box_3d[2]
        bbox_2d[2] = box_3d[4]
        bbox_2d[3] = box_3d[5]

    return bbox_2d

# Convert a 3D cloud of points into a 2D one
# cloud_3d: input 3D array with point coordinates [n, 3]
# coord: coordinate to delete: 0, 1, 2 (default)
def make_plane(cloud_3d, coord=2):

    cloud_2d = np.zeros(shape=(cloud_3d.shape[0], 2), dtype=float)

    if coord == 0:
        cloud_2d[:, 0] = cloud_3d[:, 1]
        cloud_2d[:, 1] = cloud_3d[:, 2]
    elif coord == 1:
        cloud_2d[:, 0] = cloud_3d[:, 0]
        cloud_2d[:, 1] = cloud_3d[:, 2]
    else:
        cloud_2d[:, 0] = cloud_3d[:, 0]
        cloud_2d[:, 1] = cloud_3d[:, 1]

    return cloud_2d

# Generates a random set of points [n, 2] in a plane
# n: number of points
# box: bounding box [x_min, y_min, x_max, y_max]abc
def gen_rand_cloud(n, box):
    cloud = np.random.rand(n, 2)
    cloud[:, 0] = (box[2] - box[0]) * cloud[:, 0] + box[0]
    cloud[:, 1] = (box[3] - box[1]) * cloud[:, 1] + box[1]
    return cloud

# Computes Nearest Neighbour Distance of a cloud of points in a Euclidean space
# cloud: array with point coordinates [n, 2]
def nnde(cloud):
    dists = np.zeros(shape=cloud.shape[0], dtype=float)
    for i in range(len(dists)):
        hold = cloud[i] - cloud
        hold = np.sum(hold*hold, axis=1)
        hold[i] = np.inf
        dists[i] = math.sqrt(np.min(hold))
    return dists

# Computes the Crossed Nearest Neighbour Distance of a cloud of points to another
# in a Euclidean space
# cloud: array with point coordinates [n, 2]
# cloud_ref: reference array with point coordinates [n, 2]
def cnnde(cloud, cloud_ref):
    dists = np.zeros(shape=cloud.shape[0], dtype=float)
    for i in range(len(dists)):
        hold = cloud[i] - cloud_ref
        hold = np.sum(hold*hold, axis=1)
        dists[i] = math.sqrt(np.min(hold))
    return dists

# Compute Cumulative Density Function from a one-dimensional array of random samples
# var: array of stochastic samples
# n: number of samples for cdf, if n is a sequence it defines the bin edges, including rightmost edge
# Returns: cdf values and samples respectively
def compute_cdf(var, n):
    hist, x = np.histogram(var, bins=n+1, normed=True)
    dx = x[1] - x[0]
    # Compute CDF, last value is discarded because its unaccuracy and first one is set to zero
    hold_cum = np.cumsum(hist)*dx
    return hold_cum[:-1], x[:-2]

# Computes the envelope of a stochastic function
# funcs: matrix where rows every independent simulation
# per: percentile for the envelope, default is 50 (median)
def func_envelope(funcs, per=50):
    return np.percentile(funcs, per, axis=1)

# Delete repeated points (closer each other then eps) and leaves just one coordinate (median)
# cloud: array with coordinates
# eps: maximum precision
def purge_repeat_coords(cloud, eps):

    # Initialization
    lut_del = np.zeros(shape=cloud.shape[0], dtype=bool)
    surv = list()

    for i, point in enumerate(cloud):
        if not lut_del[i]:
            hold = cloud[i] - cloud
            hold = np.sqrt(np.sum(hold*hold, axis=1))
            ids = np.where(hold < eps)[0]
            lut_del[ids] = True
            surv.append(cloud[ids, :].mean(axis=0))

    return np.asarray(surv, dtype=float)

# Delete repeated points (coordinates and ids) (closer each other then eps) and leaves just one coordinate (median)
# cloud: array with coordinates
# eps: maximum precision
def purge_repeat_coords2(cloud, cloud_ids, eps):

    # Initialization
    lut_del = np.zeros(shape=cloud.shape[0], dtype=bool)
    surv = list()
    surv_ids = list()

    for i, point in enumerate(cloud):
        if not lut_del[i]:
            hold = cloud[i] - cloud
            hold = np.sqrt(np.sum(hold*hold, axis=1))
            ids = np.where(hold < eps)[0]
            lut_del[ids] = True
            surv.append(cloud[ids, :].mean(axis=0))
            surv_ids.append(cloud_ids[ids[0]])

    return np.asarray(surv, dtype=float), np.asarray(surv_ids)

# Merge two boxes by intersection
def merge_boxes_2D(box_a, box_b):

    box = np.zeros(shape=4, dtype=float)

    if box_a[0] > box_b[0]:
        box[0] = box_a[0]
    else:
        box[0] = box_b[0]
    if box_a[1] > box_b[1]:
        box[1] = box_a[1]
    else:
        box[1] = box_b[1]
    if box_a[2] < box_b[2]:
        box[2] = box_a[2]
    else:
        box[2] = box_b[2]
    if box_a[3] < box_b[3]:
        box[3] = box_a[3]
    else:
        box[3] = box_b[3]

    return box

# Edge compensation as Goreaud specifies [J. Vegetation Sci. 10: 433-438, 1999]
# cloud: cloud of points
# box: only points within this box are considered for k-function, the rest are only
#      considered for edge correction
# n: number of output samples
# max_d: maximum distance
# Returns: Ripley's H form and the radius samples
def ripley_goreaud(cloud, box, n, max_d):

    # Initialization
    pi_2 = 2 * np.pi
    side_a = float(box[2] - box[0])
    side_b = float(box[3] - box[1])
    if (max_d > side_a) or (max_d > side_b):
        error_msg = 'Ripley''s metric cannot be computed because max_d is greater than a cloud box dimension'
        raise pexceptions.PySegInputError(expr='__ripley (SetClouds)', msg=error_msg)
    area = side_a * side_b
    rd = np.linspace(0, max_d, n)
    N = float(cloud.shape[0])
    K = np.zeros(shape=n, dtype=float)
    if N <= 1:
        return K, rd

    # Cluster radius loop
    for k, r in enumerate(rd):

        if r == 0:
            continue

        # Points loop
        for i in range(int(N)):

            # Finding neighbours
            hold = cloud[i] - cloud
            dists = np.sqrt(np.sum(hold*hold, axis=1))
            ids = np.where((dists > 0) & (dists < r))[0]

            # Loop for neighbours
            p = cloud[i, :]
            weights = np.ones(shape=len(ids), dtype=float)
            # Distance to edges
            hold_dists = list()
            hold_dists.append(box[2] - p[0])
            hold_dists.append(p[1] - box[1])
            hold_dists.append(p[0] - box[0])
            hold_dists.append(box[3] - p[1])
            hold_dists = np.asarray(hold_dists, dtype=float)
            hold_dists = np.sqrt(hold_dists * hold_dists)
            hold_dists = np.sort(hold_dists)
            d1, d2, d3, d4 = hold_dists[0], hold_dists[1], hold_dists[2], hold_dists[3]
            for j, idx in enumerate(ids):

                # Compute distance to neighbour
                pn = cloud[idx, :]
                hold_r = p - pn
                rj = math.sqrt((hold_r * hold_r).sum())

                #### Edge compensation

                # Switch for computing angle
                if (rj > d1) and (rj <= d2) and (rj <= d3) and (rj <= d4):
                    alpha = 2 * math.acos(d1 / rj)
                elif (rj > d1) and (rj > d2) and (rj <= d3) and (rj <= d4):
                    dh = d1*d1 + d2*d2
                    r2 = rj * rj
                    if r2 <= dh:
                        alpha = 2*math.acos(d1/rj) + 2*math.acos(d2/rj)
                    else:
                        alpha = .5*np.pi + math.acos(d1/r) + math.acos(d2/r)
                elif (rj > d1) and (rj > d3) and (rj <= d2) and (rj <= d4):
                    alpha = 2*math.acos(d1/rj) + 2*math.acos(d3/rj)
                elif (rj > d1) and (rj > d2) and (rj > d3) and (rj <= d4):
                    d12 = d1*d1 + d2*d2
                    d23 = d2*d2 + d3*d3
                    r2 = rj * rj
                    if (r2 <= d12) and (r2 <= d23):
                        alpha = 2*math.acos(d1/rj) + 2*math.acos(d2/rj) + 2*math.acos(d3/rj)
                    elif (r2 <= d12) and (r2 > d23):
                        alpha = .5*np.pi + 2*math.acos(d1/rj) + math.acos(d2/rj) + math.acos(d3/rj)
                    else:
                        alpha = np.pi + math.acos(d1/rj) + math.acos(d3/rj)
                else:
                    alpha = .0

                # Correcting factor
                if alpha > pi_2:
                    weights[j] = 0.
                else:
                    weights[j] = pi_2 / (pi_2 - alpha)

            # Updating K entry
            K[k] += (weights.sum())

    # Compute the H form
    # return np.sqrt((area*K) / (np.pi*N*(N-1))) - rd, rd
    return np.sqrt((area*K) / (np.pi*N*N)) - rd, rd

###########################################################################################
# Abstract class for doing the spatial analysis
###########################################################################################

class SpA(object, metaclass=ABCMeta):

    # For Abstract Base Classes in python
    def __init__(self, n_samp, n_sim_f, p_f, n_sim_r, r_max, r_bord, p_h):
        self.__n = n_samp
        self.__nsim_f, self.__p_f = n_sim_f, p_f
        self.__nsim_r, self.__r_max, self.__p_h = n_sim_r, r_max, p_h
        self.__r_bord = 0
        if (r_bord == 0) or (r_bord == 1)  or (r_bord == 2):
            self.__r_bord = r_bord
        self.__clouds = list()
        self.__boxes = list()
        self.__dens = list()
        self.__hsim = False
        self.__g = np.zeros(shape=n_samp, dtype=float)
        self.__gx = np.linspace(0., 1., self.__n)
        self.__grl = np.zeros(shape=n_samp, dtype=float)
        self.__grm = np.zeros(shape=n_samp, dtype=float)
        self.__grh = np.zeros(shape=n_samp, dtype=float)
        self.__f = np.zeros(shape=n_samp, dtype=float)
        self.__fx = np.linspace(0., 1., self.__n)
        self.__frl = np.zeros(shape=n_samp, dtype=float)
        self.__frm = np.zeros(shape=n_samp, dtype=float)
        self.__frh = np.zeros(shape=n_samp, dtype=float)
        self.__h = list()
        self.__hx = list()
        self.__hrl = np.zeros(shape=n_samp, dtype=float)
        self.__hrm = np.zeros(shape=n_samp, dtype=float)
        self.__hrh = np.zeros(shape=n_samp, dtype=float)
        self.__cards = list()

    # Get/Set functionality

    def get_function_G(self):
        return self.__g, self.__gx, self.__grl, self.__grm, self.__grh

    def get_function_F(self):
        return self.__f, self.__fx, self.__frl, self.__frm, self.__frh

    def get_ripley_H(self):
        return self.__h, self.__hx, self.__hrl, self.__hrm, self.__hrh

    # External implemented functionality

    # Computes G-Function, F-Function and Ripley's H
    # h_sim: if True (default) random simulation for Ripleys'H is generated
    # r_acc: if True (default False) all Ripley's graphs are weighted to one
    def analyze(self, h_sim=True, verbose=False, r_acc=True):

        if verbose:
            sys.stdout.write('Progress: 0% ... ')

        # G-Function
        self.__g, self.__gx = self.__function_G(self.__n)
        if verbose:
            sys.stdout.write('17% ... ')
        self.__grl, self.__grm, self.__grh, _ = self.__rand_function_G(self.__n, self.__nsim_f,
                                                                       self.__p_f)
        if verbose:
            sys.stdout.write('33% ... ')

        # F-Function
        self.__f, self.__fx = self.__function_F(self.__n, self.__nsim_f)
        if verbose:
            sys.stdout.write('50% ... ')
        self.__frl, self.__frm, self.__frh, _ = self.__rand_function_F(self.__n, self.__nsim_f,
                                                                       self.__p_f)
        if verbose:
            sys.stdout.write('67% ... ')

        # Ripley's H
        if r_acc:
            self.__ripleys_H(self.__n, self.__r_max, self.__r_bord)
        else:
            self.__ripleys_H_test(self.__n, self.__r_max, self.__r_bord)
        if verbose:
            sys.stdout.write('83% ... ')
        if h_sim:
            self.__hsim = True
            self.__hrl, self.__hrm, self.__hrh, _ = self.__rand_ripleys_H(self.__n, self.__nsim_r,
                                                                          self.__r_max, self.__r_bord,
                                                                          self.__p_h)
        if verbose:
            print('100%')

    # Plot into figures the current analysis state
    # block: if True (default False) waits for closing windows for finishing the execution
    def plot(self, block=False):

        # Initialization
        fig_count = 0
        width = 0.35
        ind = np.arange(len(self.__dens)) - (width*.5)
        if block:
            plt.ion()

        # Plot clouds
        for i, cloud in enumerate(self.__clouds):
            fig_count += 1
            plt.figure(fig_count)
            plt.title('Cloud of points ' + str(fig_count))
            plt.xlabel('X (nm)')
            plt.ylabel('Y (nm)')
            plt.axis('scaled')
            plt.xlim(self.__boxes[i][0], self.__boxes[i][2])
            plt.ylim(self.__boxes[i][1], self.__boxes[i][3])
            if len(self.__cards) <= 0:
                plt.scatter(cloud[:, 0], cloud[:, 1])
            else:
                cax = plt.scatter(cloud[:, 0], cloud[:, 1], c=self.__cards[i], cmap=cm.jet)
                plt.colorbar(cax, orientation='horizontal')

        # Plot densities
        fig_count += 1
        plt.figure(fig_count)
        plt.title('Points density')
        plt.xlabel('Sample')
        plt.ylabel('Density (points/nm^2)')
        plt.bar(ind, np.asarray(self.__dens, dtype=float), width)

        # Plot G-Function
        fig_count += 1
        plt.figure(fig_count)
        plt.title('G-Function')
        plt.xlabel('Distance (nm)')
        plt.ylabel('G')
        plt.ylim(0, 1)
        plt.plot(self.__gx, self.__g, 'b')
        plt.plot(self.__gx, self.__grm, 'r')
        plt.plot(self.__gx, self.__grl, 'k--')
        plt.plot(self.__gx, self.__grh, 'k--')

        # Plot F-Function
        fig_count += 1
        plt.figure(fig_count)
        plt.title('F-Function')
        plt.xlabel('Distance (nm)')
        plt.ylabel('F')
        plt.ylim(0, 1)
        plt.plot(self.__fx, self.__f, 'b')
        plt.plot(self.__fx, self.__frm, 'r')
        plt.plot(self.__fx, self.__frl, 'k--')
        plt.plot(self.__fx, self.__frh, 'k--')

        # Plot Ripley's H
        fig_count += 1
        plt.figure(fig_count)
        plt.title('Ripley H')
        plt.xlabel('Radius (nm)')
        plt.ylabel('H')
        color = cm.rainbow(np.linspace(0, 1, len(self.__h)))
        idx = np.arange(len(self.__h)) + 1
        lines = list()
        for (h, hx, c, ids) in zip(self.__h, self.__hx, color, idx):
            line, = plt.plot(hx, h, c=c, label=str(ids))
            lines.append(line)
        if len(lines) > 0:
            plt.legend(handles=lines)
        # plt.plot(self.__hx, self.__h, 'b')
        if self.__hsim:
            plt.plot(self.__hx, self.__hrm, 'r')
            plt.plot(self.__hx, self.__hrl, 'k--')
            plt.plot(self.__hx, self.__hrh, 'k--')

        # Show
        plt.show(block=block)

    #### External abstract functionality

    @abstractmethod
    def insert_cloud(self, cloud, box, clsts=None, mask=None):
        self.__clouds.append(cloud)
        self.__boxes.append(box)
        area = (box[2] - box[0]) * (box[3] - box[1])
        if area > 0:
            self.__dens.append(cloud.shape[0] / area)
        else:
            self.__dens.append(0.)

    @abstractmethod
    def pickle(self, fname):
        raise NotImplementedError('pickle() (SpA). '
                                  'Abstract method, it requires an implementation.')

    #### Internal implemented functionality

    # Computes Ripley's function in H form
    # n: number of samples
    # max_d: max distance for being considered
    # border: if 0 (default) border compensation is not active, 1 points inflation mode, 2 Goreaud
    # Returns: Ripley's K values and samples respectively
    def __ripleys_H(self, n, max_d, border=0):

        # Initialization
        rips = np.zeros(shape=(n, len(self.__clouds)), dtype=float)
        rd = np.zeros(shape=n, dtype=float)

        # Ripleys K factors computation
        area = 0
        weights = np.zeros(shape=len(self.__clouds), dtype=float)
        for i, cloud in enumerate(self.__clouds):
            box = self.__boxes[i]
            area_h = float((box[2] - box[0]) * (box[3] - box[1]))
            if max_d > math.sqrt(area_h*.5):
                print(WRN_RED + 'Warning (ripleys_H): cloud area small compared with maximum distance')
            weights[i] = area_h
            area += area_h
            if border == 1:
                # Inflate point cloud
                cloud_inf = self.__inflate_2D(cloud)
                rips[:, i], rd = self.__ripley(cloud_inf, box, n, max_d)
            elif border == 2:
                rips[:, i], rd = self.__ripley_goreaud(cloud, box, n, max_d)
            else:
                rips[:, i], rd = self.__ripley(cloud, box, n, max_d)

        # Cloud weighting according to box area
        if area <= 0:
            cte = 0
        else:
            cte = 1 / area
        weights *= cte

        # Insert to object variable
        self.__h.append((weights*rips).sum(axis=1))
        self.__hx.append(rd)

    # Computes Ripley's function in H and updates the correspondent lists
    # n: number of samples
    # max_d: max distance for being considered
    # border: if 0 (default) border compensation is not active, 1 points inflation mode, 2 Goreaud
    # Returns: Ripley's K values and samples respectively
    def __ripleys_H_test(self, n, max_d, border=0):

        # Initialization
        self.__h = list()
        self.__hx = list()

        # Ripleys H computation
        for i, cloud in enumerate(self.__clouds):
            box = self.__boxes[i]
            if border == 1:
                # Inflate point cloud
                cloud_inf = self.__inflate_2D(cloud)
                hold_h, hold_x = self.__ripley(cloud_inf, box, n, max_d)
            elif border == 2:
                hold_h, hold_x = self.__ripley_goreaud(cloud, box, n, max_d)
            else:
                hold_h, hold_x = self.__ripley(cloud, box, n, max_d)
            self.__h.append(hold_h)
            self.__hx.append(hold_x)

    # Computes Ripley's function in H form for CSR
    # n: number of samples
    # m: number of simulations
    # max_d: max distance for being considered
    # border: if 0 (default) border compensation is not active, 1 points inflation mode, 2 Goreaud
    # p: percentile for computing envelopes (default 5%)
    # Returns: Ripley's K 0.05, 0.5 and 0.95 envelopes, and samples respectively
    def __rand_ripleys_H(self, n, m, max_d, border=True, p=5):

        # Generate random points
        rips = np.zeros(shape=(n, m*len(self.__clouds)), dtype=float)
        cont = 0
        rd = np.zeros(shape=n, dtype=float)
        for i in range(m):
            for j, cloud in enumerate(self.__clouds):
                box = self.__boxes[j]
                cloud_1 = gen_rand_cloud(cloud.shape[0], box)
                area_h = float((box[2] - box[0]) * (box[3] - box[1]))
                if max_d > math.sqrt(area_h*.5):
                    print(WRN_RED + 'Warning (rand_ripleys_H): cloud area small compared with maximum distance')
                if border == 1:
                    # Inflate point cloud
                    cloud_inf = self.__inflate_2D(cloud_1)
                    rips[:, i], rd = self.__ripley(cloud_inf, box, n, max_d)
                elif border == 2:
                    rips[:, i], rd = self.__ripley_goreaud(cloud_1, box, n, max_d)
                else:
                    rips[:, cont], rd = self.__ripley(cloud_1, box, n, max_d)
                cont += 1

        # Compute envelopes
        env_005 = func_envelope(rips, per=p)
        env_05 = func_envelope(rips, per=50)
        env_095 = func_envelope(rips, per=100-p)

        return env_005, env_05, env_095, rd

    def __is_not_closer_to_border(self, p, box, max_d):

        # Border distances
        hold = p[0] - box[0]
        d_1 = math.sqrt(hold * hold)
        hold = p[0] - box[2]
        d_2 = math.sqrt(hold * hold)
        hold = p[1] - box[1]
        d_3 = math.sqrt(hold * hold)
        hold = p[1] - box[3]
        d_4 = math.sqrt(hold * hold)

        if (d_1 < max_d) or (d_2 < max_d) or (d_3 < max_d) or (d_4 < max_d):
            return False
        else:
            return True

    # Inflates a 2D spatial cloud of points by adding 8 flipped versions of the original data
    # in its neighbourhood
    def __inflate_2D(self, cloud):

        # Flipping
        flip_x, flip_y = flip_cloud(cloud, 0), flip_cloud(cloud, 1)
        flip_xy = flip_cloud(flip_x, 1)

        # Computing bounding box
        min_x, min_y, max_x, max_y = cloud[:, 0].min(), cloud[:, 1].min(), \
                                     cloud[:, 0].max(), cloud[:, 1].max()

        # Adding neighbours
        c_00 = np.zeros(shape=cloud.shape, dtype=cloud.dtype)
        c_01 = np.zeros(shape=cloud.shape, dtype=cloud.dtype)
        c_02 = np.zeros(shape=cloud.shape, dtype=cloud.dtype)
        c_10 = np.zeros(shape=cloud.shape, dtype=cloud.dtype)
        c_12 = np.zeros(shape=cloud.shape, dtype=cloud.dtype)
        c_20 = np.zeros(shape=cloud.shape, dtype=cloud.dtype)
        c_21 = np.zeros(shape=cloud.shape, dtype=cloud.dtype)
        c_22 = np.zeros(shape=cloud.shape, dtype=cloud.dtype)
        c_00[:, 0], c_00[:, 1] = flip_xy[:, 0] - max_x, flip_xy[:, 1] - max_y
        c_01[:, 0], c_01[:, 1] = flip_x[:, 0] - max_x, flip_x[:, 1]
        c_02[:, 0], c_02[:, 1] = flip_xy[:, 0] - max_x, flip_xy[:, 1] + max_y
        c_10[:, 0], c_10[:, 1] = flip_y[:, 0], flip_y[:, 1] - max_y
        c_12[:, 0], c_12[:, 1] = flip_y[:, 0], flip_y[:, 1] + max_y
        c_20[:, 0], c_20[:, 1] = flip_xy[:, 0] + max_x, flip_xy[:, 1] - max_y
        c_21[:, 0], c_21[:, 1] = flip_x[:, 0] + max_x, flip_x[:, 1]
        c_22[:, 0], c_22[:, 1] = flip_xy[:, 0] + max_x, flip_xy[:, 1] + max_y

        # Concatenate result
        return np.concatenate([c_00, c_01, c_02, c_10, cloud, c_12, c_20, c_21, c_22], axis=0)

    # cloud: cloud of points
    # box: only points within this box are considered for k-function, the rest are only
    #      considered for edge correction
    # n: number of output samples
    # max_d: maximum distance
    # Returns: Ripley's H form and the radius samples
    def __ripley(self, cloud, box, n, max_d):

        # Non-edge correction points detection
        hold = (cloud[:, 0] >= box[0]) & (cloud[:, 1] >= box[1]) & \
               (cloud[:, 0] <= box[2]) & (cloud[:, 1] <= box[3])
        core_ids = np.where(hold)[0]

        # Initialization
        side_a = float(box[2] - box[0])
        side_b = float(box[3] - box[1])
        if (max_d > side_a) or (max_d > side_b):
            error_msg = 'Ripley''s metric cannot be computed because max_d is greater than a cloud box dimension'
            raise pexceptions.PySegInputError(expr='__ripley (SetClouds)', msg=error_msg)
        area = side_a * side_b
        rd = np.linspace(0, max_d, n)
        N = float(len(core_ids))
        K = np.zeros(shape=n, dtype=float)
        if N <= 1:
            return K, rd

        # Cluster radius loop
        for k, r in enumerate(rd):

            # Points loop
            for i in range(int(N)):

                # Finding neighbours
                hold = cloud[i] - cloud
                dists = np.sqrt(np.sum(hold*hold, axis=1))
                k_hold = ((dists > 0) & (dists < r)).sum()

                # Updating K entry
                K[k] += k_hold

        # Compute the H form
        # return np.sqrt((area*K) / (np.pi*N*(N-1))) - rd, rd
        return np.sqrt((area*K) / (np.pi*N*N)) - rd, rd

    # Edge compensation as Goreaud specifies [J. Vegetation Sci. 10: 433-438, 1999]
    # cloud: cloud of points
    # box: only points within this box are considered for k-function, the rest are only
    #      considered for edge correction
    # n: number of output samples
    # max_d: maximum distance
    # Returns: Ripley's H form and the radius samples
    def __ripley_goreaud(self, cloud, box, n, max_d):

        # Initialization
        pi_2 = 2 * np.pi
        side_a = float(box[2] - box[0])
        side_b = float(box[3] - box[1])
        if (max_d > side_a) or (max_d > side_b):
            error_msg = 'Ripley''s metric cannot be computed because max_d is greater than a cloud box dimension'
            raise pexceptions.PySegInputError(expr='__ripley (SetClouds)', msg=error_msg)
        area = side_a * side_b
        rd = np.linspace(0, max_d, n)
        N = float(cloud.shape[0])
        K = np.zeros(shape=n, dtype=float)
        if N <= 1:
            return K, rd

        # Cluster radius loop
        for k, r in enumerate(rd):

            if r == 0:
                continue

            # Points loop
            for i in range(int(N)):

                # Finding neighbours
                hold = cloud[i] - cloud
                dists = np.sqrt(np.sum(hold*hold, axis=1))
                ids = np.where((dists > 0) & (dists < r))[0]

                # Loop for neighbours
                p = cloud[i, :]
                weights = np.ones(shape=len(ids), dtype=float)
                # Distance to edges
                hold_dists = list()
                hold_dists.append(box[2] - p[0])
                hold_dists.append(p[1] - box[1])
                hold_dists.append(p[0] - box[0])
                hold_dists.append(box[3] - p[1])
                hold_dists = np.asarray(hold_dists, dtype=float)
                hold_dists = np.sqrt(hold_dists * hold_dists)
                hold_dists = np.sort(hold_dists)
                d1, d2, d3, d4 = hold_dists[0], hold_dists[1], hold_dists[2], hold_dists[3]
                for j, idx in enumerate(ids):

                    # Compute distance to neighbour
                    pn = cloud[idx, :]
                    hold_r = p - pn
                    rj = math.sqrt((hold_r * hold_r).sum())

                    #### Edge compensation

                    # Switch for computing angle
                    if (rj > d1) and (rj <= d2) and (rj <= d3) and (rj <= d4):
                        alpha = 2 * math.acos(d1 / rj)
                    elif (rj > d1) and (rj > d2) and (rj <= d3) and (rj <= d4):
                        dh = d1*d1 + d2*d2
                        r2 = rj * rj
                        if r2 <= dh:
                            alpha = 2*math.acos(d1/rj) + 2*math.acos(d2/rj)
                        else:
                            alpha = .5*np.pi + math.acos(d1/r) + math.acos(d2/r)
                    elif (rj > d1) and (rj > d3) and (rj <= d2) and (rj <= d4):
                        alpha = 2*math.acos(d1/rj) + 2*math.acos(d3/rj)
                    elif (rj > d1) and (rj > d2) and (rj > d3) and (rj <= d4):
                        d12 = d1*d1 + d2*d2
                        d23 = d2*d2 + d3*d3
                        r2 = rj * rj
                        if (r2 <= d12) and (r2 <= d23):
                            alpha = 2*math.acos(d1/rj) + 2*math.acos(d2/rj) + 2*math.acos(d3/rj)
                        elif (r2 <= d12) and (r2 > d23):
                            alpha = .5*np.pi + 2*math.acos(d1/rj) + math.acos(d2/rj) + math.acos(d3/rj)
                        else:
                            alpha = np.pi + math.acos(d1/rj) + math.acos(d3/rj)
                    else:
                        alpha = .0

                    # Correcting factor
                    if alpha > pi_2:
                        weights[j] = 0.
                    else:
                        weights[j] = pi_2 / (pi_2 - alpha)

                # Updating K entry
                K[k] += (weights.sum())

        # Compute the H form
        # return np.sqrt((area*K) / (np.pi*N*(N-1))) - rd, rd
        return np.sqrt((area*K) / (np.pi*N*N)) - rd, rd

    # Computes G function for the accumulated set of inserted clouds
    # n: number of samples for cdf
    # Returns: G-Function values and samples respectively
    def __function_G(self, n):

        # Compute NNDs
        dists = list()
        for cloud in self.__clouds:
            dists += nnde(cloud).tolist()
        dists = np.asarray(dists, dtype=float)

        # CDF
        return compute_cdf(dists, n)

    #### Internal abstract functionality

    # n: number of samples for cdf
    # m: number of simulations for cdf
    @abstractmethod
    def __function_F(self, n, m):
        raise NotImplementedError('__function_F() (SpA). '
                                  'Abstract method, it requires an implementation.')

    # n: number of samples for cdf
    # m: number of simulations for cdf
    # p: percentile for computing envelopes (default 5%)
    # Returns: samples, F-Function 0.05, 0.5 and 0.95 envelopes, and samples respectively
    @abstractmethod
    def __rand_function_F(self, n, m, p=5):
        raise NotImplementedError('__rand_function_F() (SpA). '
                                  'Abstract method, it requires an implementation.')

    # n: number of samples for cdf
    # m: number of simulations for cdf
    # p: percentile for computing envelopes (default 5%)
    # Returns: samples, G-Function 0.05, 0.5 and 0.95 envelopes, and samples respectively
    @abstractmethod
    def __rand_function_G(self, n, m, p=5):
        raise NotImplementedError('__rand_function_G() (SpA). '
                                  'Abstract method, it requires an implementation.')


###########################################################################################
# Class for doing a spatial analysis from several independent set of points
###########################################################################################

class SetClouds(SpA):

    # n_samp: number of samples for graphs
    # n_sim_f: number of simulations for generating F and G functions
    # p_f: confidence percentile for F and G functions
    # n_sim_r: number of simulations for Ripley's H
    # r_max: maximum distance for Ripley's H in nm
    # r_bord: if 0 (default) border compensation is not active, 1 points inflation mode and
    #         2 Goreaud
    # p_h: confidence percentile for Ripleys's H
    def __init__(self, n_samp, n_sim_f, p_f, n_sim_r, r_max, r_bord, p_h):
        super(SetClouds, self).__init__(n_samp, n_sim_f, p_f, n_sim_r, r_max, r_bord, p_h)

    #### Set/Get methods area

    #### External functionality area

    # cloud: array with point coordinates in a plane [n, 2]
    # box: bounding box [x_min, y_min, x_max, y_max]
    # cards: array with point cardinalities
    def insert_cloud(self, cloud, box, cards):
        super(SetClouds, self).insert_cloud(cloud, box)
        self._SpA__cards.append(cards)

    # Pickling the object state
    # fname: full path for the pickle file
    def pickle(self, fname):
        pkl_f = open(fname, 'w')
        try:
            pickle.dump(self, pkl_f)
        finally:
            pkl_f.close()

    #### Internal functionality area

    # Computes G function for CSR
    # n: number of samples for cdf
    # m: number of simulations for cdf
    # p: percentile for computing envelopes (default 5%)
    # Returns: G-Function 0.05, 0.5 and 0.95 envelopes, and samples respectively
    def _SpA__rand_function_G(self, n, m, p=5):

        # Generate random points
        dists = list()
        cdfs = np.zeros(shape=(n, m*len(self._SpA__clouds)), dtype=float)
        cont = 0
        for i in range(m):
            for j, cloud in enumerate(self._SpA__clouds):
                hold_dists = nnde(gen_rand_cloud(cloud.shape[0], self._SpA__boxes[j]))
                cdfs[:, cont], _ = compute_cdf(hold_dists, n)
                dists += hold_dists.tolist()
                cont += 1
        dists = np.asarray(dists, dtype=float)

        # Compute results
        gf, sp = compute_cdf(dists, n)
        env_005 = func_envelope(cdfs, per=p)
        env_05 = func_envelope(cdfs, per=50)
        env_095 = func_envelope(cdfs, per=100-p)

        return env_005, env_05, env_095, sp

    # Computes F function for CSR
    # n: number of samples for cdf
    # m: number of simulations for cdf
    # p: percentile for computing envelopes (default 5%)
    # Returns: F-Function 0.05, 0.5 and 0.95 envelopes, and samples respectively
    def _SpA__rand_function_F(self, n, m, p=5):

        # Generate random points
        dists = list()
        cdfs = np.zeros(shape=(n, m*len(self._SpA__clouds)), dtype=float)
        cont = 0
        for i in range(m):
            for j, cloud in enumerate(self._SpA__clouds):
                cloud_1 = gen_rand_cloud(cloud.shape[0], self._SpA__boxes[j])
                cloud_2 = gen_rand_cloud(cloud.shape[0], self._SpA__boxes[j])
                hold_dists = cnnde(cloud_1, cloud_2)
                cdfs[:, cont], _ = compute_cdf(hold_dists, n)
                dists += hold_dists.tolist()
                cont += 1
        dists = np.asarray(dists, dtype=float)

        # Compute results
        gf, sp = compute_cdf(dists, n)
        env_005 = func_envelope(cdfs, per=p)
        env_05 = func_envelope(cdfs, per=50)
        env_095 = func_envelope(cdfs, per=100-p)

        return env_005, env_05, env_095, sp

    # Computes F function for the accumulated set of inserted clouds
    # n: number of samples for cdf
    # m: number of random simulations
    # Returns: F-Function values and samples respectively
    def _SpA__function_F(self, n, m):

        # Generate random points
        dists = list()
        for i in range(m):
            for j, cloud in enumerate(self._SpA__clouds):
                dists += cnnde(cloud, gen_rand_cloud(cloud.shape[0], self._SpA__boxes[j])).tolist()
        dists = np.asarray(dists, dtype=float)

        # CDF
        return compute_cdf(dists, n)

###########################################################################################
# Class for doing a spatial analysis from cluster of points
###########################################################################################

class SetClusters(SpA):

    # n_samp: number of samples for graphs
    # n_sim_f: number of simulations for generating F and G functions
    # p_f: confidence percentile for F and G functions
    # n_sim_r: number of simulations for Ripley's H
    # r_max: maximum distance for Ripley's H in nm
    # r_bord: if 0 (default) border compensation is not active, 1 points inflation mode and
    #         2 Goreaud
    # p_h: confidence percentile for Ripleys's H
        # r_t: number of tries for random clusters generation
    def __init__(self, n_samp, n_sim_f, p_f, n_sim_r, r_max, r_bord, p_h, r_t=50):
        super(SetClusters, self).__init__(n_samp, n_sim_f, p_f, n_sim_r, r_max, r_bord, p_h)
        self.__clsts_l = list()
        self.__masks_l = list()
        self.__r_t = r_t

    #### External functionality area

    # cloud: array with point coordinates of clusters centers of gravity in a plane [n, 2]
    # box: bounding box [x_min, y_min, x_max, y_max]
    # clsts: ordered list with clusters, each clusters is an array of points
    # mask: mask where False-values mark invalid regions
    def insert_cloud(self, cloud_cg, box, clsts, mask):
        super(SetClusters, self).insert_cloud(cloud_cg, box)
        self.__clsts_l.append(clsts)
        self.__masks_l.append(mask)

    # Pickling the object state
    # fname: full path for the pickle file
    def pickle(self, fname):
        pkl_f = open(fname, 'w')
        try:
            pickle.dump(self, pkl_f)
        finally:
            pkl_f.close()

    #### Internal functionality area

    # Generates a random distribution of the internal clusters
    # clsts: list of clusters
    # box: bounding box
    # mask: binary mask where False valued regions are invalids
    # tries: number of tries for getting the less overlapped location for every cluster
    # Returns: an array with new centroids
    def __get_rand_clsts(self, clsts, box, mask):

        # Initialization
        n_cgs = np.zeros(shape=(len(clsts), 2), dtype=float)

        # Loop for clusters
        mask_h = np.copy(mask)
        for i, c_cloud in enumerate(clsts):

            # Translate to base coordinates and computes minimum distance to center of gravity
            cg = c_cloud.mean(axis=0)
            f_cloud = c_cloud - cg
            # Compute valid search areas
            dst_t = sp.ndimage.morphology.distance_transform_edt(mask_h)
            mask_dst = np.zeros(shape=mask_h.shape, dtype=mask_h.dtype)
            mask_dst[dst_t > 0] = True
            if (dst_t > 0).sum() <= 0:
                error_msg = 'Mask fully overlapped.'
                raise pexceptions.PySegTransitionError(expr='__get_rand_clsts (SetClusters)',
                                                       msg=error_msg)
            # Keep the best try (lower overlapping)
            min_ov = MAX_FLOAT
            h_cg = None
            h_chull = np.zeros(shape=mask_h.shape, dtype=mask_h.dtype)
            for c_try in range(self.__r_t):
                # Random selection for the new centroid from valid areas
                m_ids = np.where(mask_dst)
                r_x, r_y = np.random.randint(0, len(m_ids[0])), np.random.randint(0, len(m_ids[1]))
                cg_x, cg_y = m_ids[0][r_x], m_ids[1][r_y]
                # Rotate randomly against base center [0, 0]
                rho = np.random.rand() * (2*np.pi)
                sinr, cosr = math.sin(rho), math.cos(rho)
                r_cloud = np.zeros(shape=f_cloud.shape, dtype=f_cloud.dtype)
                r_cloud[:, 0] = f_cloud[:, 0]*cosr - f_cloud[:, 1]*sinr
                r_cloud[:, 1] = f_cloud[:, 0]*sinr + f_cloud[:, 1]*cosr
                # Translation to randomly already selected center
                n_cg = np.asarray((cg_x, cg_y) , dtype=float)
                # v = n_cg - cg
                t_cloud = r_cloud + n_cg
                chull, _ = self.__compute_chull_no_bound(t_cloud, box)
                # Update minimum overlap
                ov = chull.sum() - (chull * mask_h).sum()
                if ov < min_ov:
                    min_ov = ov
                    h_cg = n_cg
                    h_chull = chull
                    if ov == 0:
                        break
                else:
                    if h_cg is None:
                        h_cg = n_cg

            # Update mask
            mask_h[h_chull] = False
            # Get new center transposed
            n_cgs[i, 0] = h_cg[1]
            n_cgs[i, 1] = h_cg[0]

        # plt.ion()
        # plt.figure()
        # plt.title('Test')
        # plt.imshow(mask)
        # plt.show()
        # plt.figure()
        # plt.title('Test 2')
        # plt.scatter(n_cgs[:, 0], n_cgs[:, 1])
        # plt.show()

        return n_cgs

    # Returns convex hull and discard points out of bounds are discarded and no exception is
    #           raised, instead in a second variable a true is returned
    def __compute_chull_no_bound(self, c_cloud, box):

        # Create holding image
        off_x = math.floor(box[1])
        off_y = math.floor(box[0])
        m, n = math.ceil(box[3]) - off_x + 1, math.ceil(box[2]) - off_y + 1
        img = np.zeros(shape=(m, n), dtype=bool)

        # Filling holding image
        hold = np.asarray(np.round(c_cloud), dtype=int)
        hold[:, 0] -= off_y
        hold[:, 1] -= off_x
        excep = False
        p_count = 0
        for p in hold:
            try:
                img[p[0], p[1]] = True
            except IndexError:
                excep = True
                continue
            p_count += 1

        # Computing the convex hull
        if p_count > 0:
            chull = np.asarray(convex_hull_image(img), dtype=bool)
        else:
            chull = img

        return chull, excep

    # Computes G function for CSR
    # n: number of samples for cdf
    # m: number of simulations for cdf
    # p: percentile for computing envelopes (default 5%)
    # Returns: G-Function 0.05, 0.5 and 0.95 envelopes, and samples respectively
    def _SpA__rand_function_G(self, n, m, p=5):

        # Generate random points
        dists = list()
        cdfs = np.zeros(shape=(n, m*len(self.__clsts_l)), dtype=float)
        cont = 0
        for i in range(m):
            for j, clsts in enumerate(self.__clsts_l):
                hold_dists = nnde(self.__get_rand_clsts(clsts, self._SpA__boxes[j],
                                                        self.__masks_l[j]))
                cdfs[:, cont], _ = compute_cdf(hold_dists, n)
                dists += hold_dists.tolist()
                cont += 1
        dists = np.asarray(dists, dtype=float)

        # Compute results
        gf, sp = compute_cdf(dists, n)
        env_005 = func_envelope(cdfs, per=p)
        env_05 = func_envelope(cdfs, per=50)
        env_095 = func_envelope(cdfs, per=100-p)

        return env_005, env_05, env_095, sp

    # Computes F function for CSR
    # n: number of samples for cdf
    # m: number of simulations for cdf
    # p: percentile for computing envelopes (default 5%)
    # Returns: F-Function 0.05, 0.5 and 0.95 envelopes, and samples respectively
    def _SpA__rand_function_F(self, n, m, p=5):

        # Generate random points
        dists = list()
        cdfs = np.zeros(shape=(n, m*len(self.__clsts_l)), dtype=float)
        cont = 0
        for i in range(m):
            for j, clsts in enumerate(self.__clsts_l):
                cloud_1 = self.__get_rand_clsts(clsts, self._SpA__boxes[j], self.__masks_l[j])
                cloud_2 = self.__get_rand_clsts(clsts, self._SpA__boxes[j], self.__masks_l[j])
                hold_dists = cnnde(cloud_1, cloud_2)
                cdfs[:, cont], _ = compute_cdf(hold_dists, n)
                dists += hold_dists.tolist()
                cont += 1
        dists = np.asarray(dists, dtype=float)

        # Compute results
        gf, sp = compute_cdf(dists, n)
        env_005 = func_envelope(cdfs, per=p)
        env_05 = func_envelope(cdfs, per=50)
        env_095 = func_envelope(cdfs, per=100-p)

        return env_005, env_05, env_095, sp

    # Computes F function for the accumulated set of inserted clouds
    # n: number of samples for cdf
    # m: number of random simulations
    # Returns: F-Function values and samples respectively
    def _SpA__function_F(self, n, m):

        # Generate random points
        dists = list()
        for i in range(m):
            for j, cloud in enumerate(self._SpA__clouds):
                dists += cnnde(cloud, self.__get_rand_clsts(self.__clsts_l[j],
                                                            self._SpA__boxes[j],
                                                            self.__masks_l[j])).tolist()
        dists = np.asarray(dists, dtype=float)

        # CDF
        return compute_cdf(dists, n)

###########################################################################################
# Abstract class for doing the spatial analysis of a set of slices
###########################################################################################

class SlA(object, metaclass=ABCMeta):

    # For Abstract Base Classes in python
    def __init__(self, box, n_samp, n_sim_f, r_max, r_bord, p_f=None):
        self.__box = box
        self.__n = n_samp
        self.__nsim_f = n_sim_f
        self.__r_max = r_max
        self.__r_bord = 0
        self.__p_f = p_f
        if (r_bord == 0) or (r_bord == 1)  or (r_bord == 2):
            self.__r_bord = r_bord
        self.__clouds = list()
        self.__dens = list()
        self.__g = list()
        self.__gx = list()
        self.__grm = np.zeros(shape=n_samp, dtype=float)
        self.__grm1 = np.zeros(shape=n_samp, dtype=float)
        self.__grm2 = np.zeros(shape=n_samp, dtype=float)
        self.__grmx = np.zeros(shape=n_samp, dtype=float)
        self.__f = list()
        self.__fx = list()
        self.__frm = np.zeros(shape=n_samp, dtype=float)
        self.__frm1 = np.zeros(shape=n_samp, dtype=float)
        self.__frm2 = np.zeros(shape=n_samp, dtype=float)
        self.__frmx = np.zeros(shape=n_samp, dtype=float)
        self.__h = list()
        self.__hx = list()
        self.__l = list()
        self.__lx = list()
        self.__hp = list()
        self.__hpx = list()
        self.__lp = list()
        self.__lpx = list()
        self.__cards = list()
        self.__srs = list()
        # Low pass filter for differentials
        b, a = butter(LP_ORDER, LP_NORM_CUTOFF, btype='low', analog=False)
        self.__lpf = (b, a)

    # Get/Set functionality

    def get_box(self):
        return self.__box

    # Return a cloud coordinates by passing a name (sr string), no if this name is not valid
    def get_cloud_by_name(self, name):
        try:
            idx = self.__srs.index(name)
            return self.__clouds[idx]
        except ValueError:
            return None

    def get_clouds_list(self):
        return self.__clouds

    def get_densities(self):
        return np.asarray(self.__dens, dtype=float)

    def get_function_G(self):
        return self.__g, self.__gx, self.__grm

    def get_function_F(self):
        return self.__f, self.__fx, self.__frm

    def get_ripley_H(self):
        return self.__h, self.__hx

    def get_ripley_Hp(self):
        return self.__hp, self.__hpx

    def get_ripley_L(self):
        return self.__l, self.__lx

    def get_ripley_Lp(self):
        return self.__lp, self.__lpx

    def get_slice_ranges(self):
        return self.__srs

    # External implemented functionality

    # Computes G-Function, F-Function and Ripley's H
    def analyze(self, verbose=False):

        if verbose:
            sys.stdout.write('Progress: 0% ... ')

        # G-Function
        self.__function_G(self.__n)
        if verbose:
            sys.stdout.write('20% ... ')
        if self.__nsim_f > 0:
            self.__grmx, self.__grm, self.__grm1, self.__grm2 = self.__rand_function_G(self.__n, self.__nsim_f,
                                                                                       self.__p_f)
        if verbose:
            sys.stdout.write('40% ... ')

        # F-Function
        self.__function_F(self.__n, self.__nsim_f)
        if verbose:
            sys.stdout.write('60% ... ')
        if self.__nsim_f > 0:
            self.__frmx, self.__frm, self.__frm1, self.__frm2 = self.__rand_function_F(self.__n, self.__nsim_f,
                                                                                       self.__p_f)
        if verbose:
            sys.stdout.write('80% ... ')

        # Ripley's metrics
        self.__ripleys_H_test(self.__n, self.__r_max, self.__r_bord)
        self.__ripleys_L()
        self.__ripleys_Hp()
        self.__ripleys_Lp()
        if verbose:
            print('100%')

    # Plot into figures the current analysis state
    # block: if True (default False) waits for closing windows for finishing the execution
    # cloud_over: if True (default) all clouds are plot in the same figure
    # fourier: it True (default) the fourier analysis is also plotted
    # l_metric: it True (default False) the Ripley's L metric is computed
    # r_stat: it True (default False) Ripley's H statistics are measured
    def plot(self, block=False, cloud_over=True, fourier=True, l_metric=False, r_stat=False):

        # Initialization
        fig_count = 0
        if block:
            plt.ion()
        labels = self.__srs
        ind = np.arange(1, len(labels)+1)
        color = cm.rainbow(np.linspace(0, 1, len(self.__srs)))

        # Plot clouds
        if cloud_over:
            fig_count += 1
            plt.figure(fig_count)
            plt.title('Clouds of points')
            plt.xlabel('X (nm)')
            plt.ylabel('Y (nm)')
            plt.axis('scaled')
            plt.xlim(self.__box[0], self.__box[2])
            plt.ylim(self.__box[1], self.__box[3])
            for i, cloud in enumerate(self.__clouds):
                if cloud.shape[0] > 0:
                    if (len(self.__cards) <= 0) or (self.__cards[i] is None):
                        plt.scatter(cloud[:, 0], cloud[:, 1], c=color[i])
                    else:
                        cax = plt.scatter(cloud[:, 0], cloud[:, 1], c=self.__cards[i], cmap=cm.jet)
                        plt.colorbar(cax, orientation='horizontal')
        else:
            for i, cloud in enumerate(self.__clouds):
                fig_count += 1
                plt.figure(fig_count)
                plt.title('Clouds of points ' + labels[i])
                plt.xlabel('X (nm)')
                plt.ylabel('Y (nm)')
                plt.axis('scaled')
                plt.xlim(self.__box[0], self.__box[2])
                plt.ylim(self.__box[1], self.__box[3])
                if cloud.shape[0] > 0:
                    if (len(self.__cards) <= 0) or (self.__cards[i] is None):
                        plt.scatter(cloud[:, 0], cloud[:, 1])
                    else:
                        cax = plt.scatter(cloud[:, 0], cloud[:, 1], c=self.__cards[i], cmap=cm.jet)
                        plt.colorbar(cax, orientation='horizontal')

        # Plot densities
        fig_count += 1
        plt.figure(fig_count)
        plt.title('Points density')
        plt.xlabel('Sample')
        plt.ylabel('Density (points/nm^2)')
        plt.xlim(ind[0]-1, ind[-1]+1)
        plt.stem(ind, np.asarray(self.__dens, dtype=float))

        # Plot G-Function
        fig_count += 1
        plt.figure(fig_count)
        plt.title('G-Function')
        plt.xlabel('Distance (nm)')
        plt.ylabel('G')
        plt.ylim(0, 1)
        if self.__nsim_f > 0:
            plt.plot(self.__grmx, self.__grm, 'k')
            if self.__p_f is not None:
                plt.plot(self.__grmx, self.__grm1, 'k--')
                plt.plot(self.__grmx, self.__grm2, 'k--')
        lines = list()
        for (g, gx, c, lbl) in zip(self.__g, self.__gx, color, labels):
            line, = plt.plot(gx, g, c=c, label=lbl)
            lines.append(line)
        if len(lines) > 0:
            plt.legend(handles=lines)

        # Plot F-Function
        fig_count += 1
        plt.figure(fig_count)
        plt.title('F-Function')
        plt.xlabel('Distance (nm)')
        plt.ylabel('F')
        plt.ylim(0, 1)
        if self.__nsim_f > 0:
            plt.plot(self.__frmx, self.__frm, 'k')
            if self.__p_f is not None:
                plt.plot(self.__frmx, self.__frm1, 'k--')
                plt.plot(self.__frmx, self.__frm2, 'k--')
        lines = list()
        for (f, fx, c, lbl) in zip(self.__f, self.__fx, color, labels):
            line, = plt.plot(fx, f, c=c, label=lbl)
            lines.append(line)
        if len(lines) > 0:
            plt.legend(handles=lines)

        # Plot Ripley's H
        fig_count += 1
        plt.figure(fig_count)
        plt.title('Ripley H')
        plt.xlabel('Radius (nm)')
        plt.ylabel('H')
        for (h, hx, c, lbl) in zip(self.__h, self.__hx, color, labels):
            plt.plot(hx, h, c=c, label=lbl)
        if len(self.__h) > 0:
            plt.plot(self.__hx[0], np.zeros(shape=len(self.__h[0])), 'k--')

        # Plot Ripley's H Fourier analysis
        if fourier:
            # Compute FFT
            hfs = list()
            freqs = list()
            for (h, hx) in zip(self.__h, self.__hx):
                freqs.append(np.fft.fftshift(np.fft.fftfreq(len(h), hx[1] - hx[0])))
                hfs.append(np.fft.fftshift(np.fft.fft(h)))
            # Figures
            fig_count += 1
            plt.figure(str(fig_count) + '- Ripley H Fourier analysis')
            plt.subplot(2, 2, 1)
            plt.xlabel('Freq')
            plt.ylabel('Real')
            for (hf, f, c, lbl) in zip(hfs, freqs, color, labels):
                plt.plot(f, np.real(hf), c=c, label=lbl)
            plt.subplot(2, 2, 2)
            plt.xlabel('Freq')
            plt.ylabel('Imag')
            for (hf, f, c, lbl) in zip(hfs, freqs, color, labels):
                plt.plot(f, np.imag(hf), c=c, label=lbl)
            plt.subplot(2, 2, 3)
            plt.xlabel('Freq')
            plt.ylabel('Abs')
            for (hf, f, c, lbl) in zip(hfs, freqs, color, labels):
                plt.plot(f, np.abs(hf), c=c, label=lbl)
            plt.subplot(2, 2, 4)
            plt.xlabel('Freq')
            plt.ylabel('Angle')
            for (hf, f, c, lbl) in zip(hfs, freqs, color, labels):
                plt.plot(f, np.angle(hf), c=c, label=lbl)

        # Plot Ripley's L
        if l_metric:
            fig_count += 1
            plt.figure(fig_count)
            plt.title('Ripley L')
            plt.xlabel('Radius (nm)')
            plt.ylabel('L')
            for (l, lx, c, lbl) in zip(self.__l, self.__lx, color, labels):
                plt.plot(lx, l, c=c, label=lbl)
            if len(self.__l) > 0:
                plt.plot(self.__lx[0], self.__lx[0], 'k--')

            # Plot Ripley's L Fourier analysis
            if fourier:
                # Compute FFT
                lfs = list()
                freqs = list()
                for (l, lx) in zip(self.__l, self.__lx):
                    freqs.append(np.fft.fftshift(np.fft.fftfreq(len(l), lx[1] - lx[0])))
                    lfs.append(np.fft.fftshift(np.fft.fft(l)))
                # Figures
                fig_count += 1
                plt.figure(str(fig_count) + '- Ripley L Fourier analysis')
                plt.subplot(2, 2, 1)
                plt.xlabel('Freq')
                plt.ylabel('Real')
                for (lf, f, c, lbl) in zip(lfs, freqs, color, labels):
                    plt.plot(f, np.real(lf), c=c, label=lbl)
                plt.subplot(2, 2, 2)
                plt.xlabel('Freq')
                plt.ylabel('Imag')
                for (lf, f, c, lbl) in zip(lfs, freqs, color, labels):
                    plt.plot(f, np.imag(lf), c=c, label=lbl)
                plt.subplot(2, 2, 3)
                plt.xlabel('Freq')
                plt.ylabel('Abs')
                for (lf, f, c, lbl) in zip(lfs, freqs, color, labels):
                    plt.plot(f, np.abs(lf), c=c, label=lbl)
                plt.subplot(2, 2, 4)
                plt.xlabel('Freq')
                plt.ylabel('Angle')
                for (lf, f, c, lbl) in zip(lfs, freqs, color, labels):
                    plt.plot(f, np.angle(lf), c=c, label=lbl)

        # Plot Ripley's H'
        fig_count += 1
        plt.figure(fig_count)
        plt.title('Ripley H first derivative')
        plt.xlabel('Radius (nm)')
        plt.ylabel('H\'')
        for (hp, hpx, c, lbl) in zip(self.__hp, self.__hpx, color, labels):
            plt.plot(hpx, hp, c=c, label=lbl)
        if len(self.__hp) > 0:
            plt.plot(self.__hpx[0], np.zeros(shape=len(self.__hp[0])), 'k--')

        # Plot Ripley's H' Fourier analysis
        if fourier:
            # Compute FFT
            hpfs = list()
            freqs = list()
            for (hp, hpx) in zip(self.__hp, self.__hpx):
                freqs.append(np.fft.fftshift(np.fft.fftfreq(len(hp), hpx[1] - hpx[0])))
                hpfs.append(np.fft.fftshift(np.fft.fft(hp)))
            # Figures
            fig_count += 1
            plt.figure(str(fig_count) + '- Ripley H\' Fourier analysis')
            plt.subplot(2, 2, 1)
            plt.xlabel('Freq')
            plt.ylabel('Real')
            for (hpf, f, c, lbl) in zip(hpfs, freqs, color, labels):
                plt.plot(f, np.real(hpf), c=c, label=lbl)
            plt.subplot(2, 2, 2)
            plt.xlabel('Freq')
            plt.ylabel('Imag')
            for (hpf, f, c, lbl) in zip(hpfs, freqs, color, labels):
                plt.plot(f, np.imag(hpf), c=c, label=lbl)
            plt.subplot(2, 2, 3)
            plt.xlabel('Freq')
            plt.ylabel('Abs')
            for (hpf, f, c, lbl) in zip(hpfs, freqs, color, labels):
                plt.plot(f, np.abs(hpf), c=c, label=lbl)
            plt.subplot(2, 2, 4)
            plt.xlabel('Freq')
            plt.ylabel('Angle')
            for (hpf, f, c, lbl) in zip(hpfs, freqs, color, labels):
                plt.plot(f, np.angle(hpf), c=c, label=lbl)

        # Plot Ripley's L\'
        if l_metric:
            fig_count += 1
            plt.figure(fig_count)
            plt.title('Ripley L first derivative')
            plt.xlabel('Radius (nm)')
            plt.ylabel('L\'')
            for (lp, lpx, c, lbl) in zip(self.__lp, self.__lpx, color, labels):
                plt.plot(lpx, lp, c=c, label=lbl)
            if len(self.__lp) > 0:
                plt.plot(self.__lpx[0], np.ones(shape=len(self.__lp[0])), 'k--')

            # Plot Ripley's L' Fourier analysis
            if fourier:
                # Compute FFT
                lpfs = list()
                freqs = list()
                for (lp, lpx) in zip(self.__lp, self.__lpx):
                    freqs.append(np.fft.fftshift(np.fft.fftfreq(len(lp), lpx[1] - lpx[0])))
                    lpfs.append(np.fft.fftshift(np.fft.fft(lp)))
                # Figures
                fig_count += 1
                plt.figure(str(fig_count) + '- Ripley L\' Fourier analysis')
                plt.subplot(2, 2, 1)
                plt.xlabel('Freq')
                plt.ylabel('Real')
                for (lpf, f, c, lbl) in zip(lpfs, freqs, color, labels):
                    plt.plot(f, np.real(lpf), c=c, label=lbl)
                plt.subplot(2, 2, 2)
                plt.xlabel('Freq')
                plt.ylabel('Imag')
                for (lpf, f, c, lbl) in zip(lpfs, freqs, color, labels):
                    plt.plot(f, np.imag(lpf), c=c, label=lbl)
                plt.subplot(2, 2, 3)
                plt.xlabel('Freq')
                plt.ylabel('Abs')
                for (lpf, f, c, lbl) in zip(lpfs, freqs, color, labels):
                    plt.plot(f, np.abs(lpf), c=c, label=lbl)
                plt.subplot(2, 2, 4)
                plt.xlabel('Freq')
                plt.ylabel('Angle')
                for (lpf, f, c, lbl) in zip(lpfs, freqs, color, labels):
                    plt.plot(f, np.angle(lpf), c=c, label=lbl)

        # Plot Riley's H statistics
        if r_stat:
            # Compute stats
            maxs = list()
            medians = list()
            stds = list()
            for h in self.__h:
                maxs.append(h.max())
                medians.append(np.median(h))
                stds.append(h.std())
            nsam = np.arange(len(maxs))
            # Plotting
            fig_count += 1
            plt.figure(fig_count)
            plt.title('Ripley\'s H maximum')
            plt.xlabel('Sample')
            plt.ylabel('H maximum')
            plt.xlim(nsam[0]-1, nsam[-1]+1)
            plt.stem(nsam, np.asarray(maxs, dtype=float))
            fig_count += 1
            plt.figure(fig_count)
            plt.title('Ripley\'s H medians')
            plt.xlabel('Sample')
            plt.ylabel('H medians')
            plt.xlim(nsam[0]-1, nsam[-1]+1)
            plt.stem(nsam, np.asarray(medians, dtype=float))
            fig_count += 1
            plt.figure(fig_count)
            plt.title('Ripley\'s H standard deviations')
            plt.xlabel('Sample')
            plt.ylabel('H deviations')
            plt.xlim(nsam[0]-1, nsam[-1]+1)
            plt.stem(nsam, np.asarray(stds, dtype=float))

        # Show
        plt.show(block=block)

    # Plot into figures the current analysis state
    # path: path to the folder where figures will be stored
    # cloud_over: if True (default) all clouds are plot in the same figure
    # fourier: it True (default) the fourier analysis is also plotted
    def store_figs(self, path, cloud_over=True, fourier=True):

        # Initialization
        fig_count = 0
        labels = self.__srs
        ind = np.arange(1, len(labels)+1)
        color = cm.rainbow(np.linspace(0, 1, len(self.__srs)))

        # Plot clouds
        if cloud_over:
            fig_count += 1
            plt.figure(fig_count)
            plt.title('Clouds of points')
            plt.xlabel('X (nm)')
            plt.ylabel('Y (nm)')
            plt.axis('scaled')
            plt.xlim(self.__box[0], self.__box[2])
            plt.ylim(self.__box[1], self.__box[3])
            for i, cloud in enumerate(self.__clouds):
                if cloud.shape[0] > 0:
                    if (len(self.__cards) <= 0) or (self.__cards[i] is None):
                        plt.scatter(cloud[:, 0], cloud[:, 1], c=color[i])
                    else:
                        cax = plt.scatter(cloud[:, 0], cloud[:, 1], c=self.__cards[i], cmap=cm.jet)
                        plt.colorbar(cax, orientation='horizontal')
            plt.savefig(path + '/clouds.png')
            plt.close()
        else:
            for i, cloud in enumerate(self.__clouds):
                fig_count += 1
                plt.figure(fig_count)
                plt.title('Clouds of points ' + labels[i])
                plt.xlabel('X (nm)')
                plt.ylabel('Y (nm)')
                plt.axis('scaled')
                plt.xlim(self.__box[0], self.__box[2])
                plt.ylim(self.__box[1], self.__box[3])
                if cloud.shape[0] > 0:
                    if (len(self.__cards) <= 0) or (self.__cards[i] is None):
                        plt.scatter(cloud[:, 0], cloud[:, 1])
                    else:
                        cax = plt.scatter(cloud[:, 0], cloud[:, 1], c=self.__cards[i], cmap=cm.jet)
                        plt.colorbar(cax, orientation='horizontal')
                    plt.savefig(path + '/cloud_' + labels[i] + '.png')
                    plt.close()

        # Plot densities
        fig_count += 1
        plt.figure(fig_count)
        plt.title('Points density')
        plt.xlabel('Sample')
        plt.ylabel('Density (points/nm^2)')
        plt.xlim(ind[0]-1, ind[-1]+1)
        plt.stem(ind, np.asarray(self.__dens, dtype=float))
        plt.savefig(path + '/dens.png')
        plt.close()

        # Plot G-Function
        fig_count += 1
        plt.figure(fig_count)
        plt.title('G-Function')
        plt.xlabel('Distance (nm)')
        plt.ylabel('G')
        plt.ylim(0, 1)
        if self.__nsim_f > 0:
            plt.plot(self.__grmx, self.__grm, 'k')
            if self.__p_f is not None:
                plt.plot(self.__grmx, self.__grm1, 'k--')
                plt.plot(self.__grmx, self.__grm2, 'k--')
        lines = list()
        for (g, gx, c, lbl) in zip(self.__g, self.__gx, color, labels):
            line, = plt.plot(gx, g, c=c, label=lbl)
            lines.append(line)
        if len(lines) > 0:
            plt.legend(handles=lines)
        plt.savefig(path + '/g.png')
        plt.close()

        # Plot F-Function
        fig_count += 1
        plt.figure(fig_count)
        plt.title('F-Function')
        plt.xlabel('Distance (nm)')
        plt.ylabel('F')
        plt.ylim(0, 1)
        if self.__nsim_f > 0:
            plt.plot(self.__frmx, self.__frm, 'k')
            if self.__p_f is not None:
                plt.plot(self.__frmx, self.__frm1, 'k--')
                plt.plot(self.__frmx, self.__frm2, 'k--')
        lines = list()
        for (f, fx, c, lbl) in zip(self.__f, self.__fx, color, labels):
            line, = plt.plot(fx, f, c=c, label=lbl)
            lines.append(line)
        if len(lines) > 0:
            plt.legend(handles=lines)
        plt.savefig(path + '/f.png')
        plt.close()

        # Plot Ripley's H
        fig_count += 1
        plt.figure(fig_count)
        plt.title('Ripley H')
        plt.xlabel('Radius (nm)')
        plt.ylabel('H')
        for (h, hx, c, lbl) in zip(self.__h, self.__hx, color, labels):
            plt.plot(hx, h, c=c, label=lbl)
        if len(self.__h) > 0:
            plt.plot(self.__hx[0], np.zeros(shape=len(self.__h[0])), 'k--')
        plt.savefig(path + '/h.png')
        plt.close()

        # Plot Ripley's H Fourier analysis
        if fourier:
            # Compute FFT
            hfs = list()
            freqs = list()
            for (h, hx) in zip(self.__h, self.__hx):
                freqs.append(np.fft.fftshift(np.fft.fftfreq(len(h), hx[1] - hx[0])))
                hfs.append(np.fft.fftshift(np.fft.fft(h)))
            # Figures
            fig_count += 1
            plt.figure(str(fig_count) + '- Ripley H Fourier analysis')
            plt.subplot(2, 2, 1)
            plt.xlabel('Freq')
            plt.ylabel('Real')
            for (hf, f, c, lbl) in zip(hfs, freqs, color, labels):
                plt.plot(f, np.real(hf), c=c, label=lbl)
            plt.subplot(2, 2, 2)
            plt.xlabel('Freq')
            plt.ylabel('Imag')
            for (hf, f, c, lbl) in zip(hfs, freqs, color, labels):
                plt.plot(f, np.imag(hf), c=c, label=lbl)
            plt.subplot(2, 2, 3)
            plt.xlabel('Freq')
            plt.ylabel('Abs')
            for (hf, f, c, lbl) in zip(hfs, freqs, color, labels):
                plt.plot(f, np.abs(hf), c=c, label=lbl)
            plt.subplot(2, 2, 4)
            plt.xlabel('Freq')
            plt.ylabel('Angle')
            for (hf, f, c, lbl) in zip(hfs, freqs, color, labels):
                plt.plot(f, np.angle(hf), c=c, label=lbl)
            plt.savefig(path + '/h_f.png')
            plt.close()

        # Plot Ripley's L
        fig_count += 1
        plt.figure(fig_count)
        plt.title('Ripley L')
        plt.xlabel('Radius (nm)')
        plt.ylabel('L')
        for (l, lx, c, lbl) in zip(self.__l, self.__lx, color, labels):
            plt.plot(lx, l, c=c, label=lbl)
        if len(self.__l) > 0:
            plt.plot(self.__lx[0], self.__lx[0], 'k--')
        plt.savefig(path + '/l.png')
        plt.close()

        # Plot Ripley's L Fourier analysis
        if fourier:
            # Compute FFT
            lfs = list()
            freqs = list()
            for (l, lx) in zip(self.__l, self.__lx):
                freqs.append(np.fft.fftshift(np.fft.fftfreq(len(l), lx[1] - lx[0])))
                lfs.append(np.fft.fftshift(np.fft.fft(l)))
            # Figures
            fig_count += 1
            plt.figure(str(fig_count) + '- Ripley L Fourier analysis')
            plt.subplot(2, 2, 1)
            plt.xlabel('Freq')
            plt.ylabel('Real')
            for (lf, f, c, lbl) in zip(lfs, freqs, color, labels):
                plt.plot(f, np.real(lf), c=c, label=lbl)
            plt.subplot(2, 2, 2)
            plt.xlabel('Freq')
            plt.ylabel('Imag')
            for (lf, f, c, lbl) in zip(lfs, freqs, color, labels):
                plt.plot(f, np.imag(lf), c=c, label=lbl)
            plt.subplot(2, 2, 3)
            plt.xlabel('Freq')
            plt.ylabel('Abs')
            for (lf, f, c, lbl) in zip(lfs, freqs, color, labels):
                plt.plot(f, np.abs(lf), c=c, label=lbl)
            plt.subplot(2, 2, 4)
            plt.xlabel('Freq')
            plt.ylabel('Angle')
            for (lf, f, c, lbl) in zip(lfs, freqs, color, labels):
                plt.plot(f, np.angle(lf), c=c, label=lbl)
            plt.savefig(path + '/l_f.png')
            plt.close()

        # Plot Ripley's H'
        fig_count += 1
        plt.figure(fig_count)
        plt.title('Ripley H first derivative')
        plt.xlabel('Radius (nm)')
        plt.ylabel('H\'')
        for (hp, hpx, c, lbl) in zip(self.__hp, self.__hpx, color, labels):
            plt.plot(hpx, hp, c=c, label=lbl)
        if len(self.__hp) > 0:
            plt.plot(self.__hpx[0], np.zeros(shape=len(self.__hp[0])), 'k--')
        plt.savefig(path + '/hp.png')
        plt.close()

        # Plot Ripley's H' Fourier analysis
        if fourier:
            # Compute FFT
            hpfs = list()
            freqs = list()
            for (hp, hpx) in zip(self.__hp, self.__hpx):
                freqs.append(np.fft.fftshift(np.fft.fftfreq(len(hp), hpx[1] - hpx[0])))
                hpfs.append(np.fft.fftshift(np.fft.fft(hp)))
            # Figures
            fig_count += 1
            plt.figure(str(fig_count) + '- Ripley H\' Fourier analysis')
            plt.subplot(2, 2, 1)
            plt.xlabel('Freq')
            plt.ylabel('Real')
            for (hpf, f, c, lbl) in zip(hpfs, freqs, color, labels):
                plt.plot(f, np.real(hpf), c=c, label=lbl)
            plt.subplot(2, 2, 2)
            plt.xlabel('Freq')
            plt.ylabel('Imag')
            for (hpf, f, c, lbl) in zip(hpfs, freqs, color, labels):
                plt.plot(f, np.imag(hpf), c=c, label=lbl)
            plt.subplot(2, 2, 3)
            plt.xlabel('Freq')
            plt.ylabel('Abs')
            for (hpf, f, c, lbl) in zip(hpfs, freqs, color, labels):
                plt.plot(f, np.abs(hpf), c=c, label=lbl)
            plt.subplot(2, 2, 4)
            plt.xlabel('Freq')
            plt.ylabel('Angle')
            for (hpf, f, c, lbl) in zip(hpfs, freqs, color, labels):
                plt.plot(f, np.angle(hpf), c=c, label=lbl)
            plt.savefig(path + '/hp_f.png')
            plt.close()

        # Plot Ripley's L
        fig_count += 1
        plt.figure(fig_count)
        plt.title('Ripley L first derivative')
        plt.xlabel('Radius (nm)')
        plt.ylabel('L\'')
        for (lp, lpx, c, lbl) in zip(self.__lp, self.__lpx, color, labels):
            plt.plot(lpx, lp, c=c, label=lbl)
        if len(self.__lp) > 0:
            plt.plot(self.__lpx[0], np.ones(shape=len(self.__lp[0])), 'k--')
        plt.savefig(path + '/lp.png')
        plt.close()

        # Plot Ripley's L' Fourier analysis
        if fourier:
            # Compute FFT
            lpfs = list()
            freqs = list()
            for (lp, lpx) in zip(self.__lp, self.__lpx):
                freqs.append(np.fft.fftshift(np.fft.fftfreq(len(lp), lpx[1] - lpx[0])))
                lpfs.append(np.fft.fftshift(np.fft.fft(lp)))
            # Figures
            fig_count += 1
            plt.figure(str(fig_count) + '- Ripley L\' Fourier analysis')
            plt.subplot(2, 2, 1)
            plt.xlabel('Freq')
            plt.ylabel('Real')
            for (lpf, f, c, lbl) in zip(lpfs, freqs, color, labels):
                plt.plot(f, np.real(lpf), c=c, label=lbl)
            plt.subplot(2, 2, 2)
            plt.xlabel('Freq')
            plt.ylabel('Imag')
            for (lpf, f, c, lbl) in zip(lpfs, freqs, color, labels):
                plt.plot(f, np.imag(lpf), c=c, label=lbl)
            plt.subplot(2, 2, 3)
            plt.xlabel('Freq')
            plt.ylabel('Abs')
            for (lpf, f, c, lbl) in zip(lpfs, freqs, color, labels):
                plt.plot(f, np.abs(lpf), c=c, label=lbl)
            plt.subplot(2, 2, 4)
            plt.xlabel('Freq')
            plt.ylabel('Angle')
            for (lpf, f, c, lbl) in zip(lpfs, freqs, color, labels):
                plt.plot(f, np.angle(lpf), c=c, label=lbl)
            plt.savefig(path + '/lp_f.png')
            plt.close()

    #### External abstract functionality

    @abstractmethod
    def insert_cloud(self, cloud, sr, clsts=None, mask=None):
        self.__clouds.append(cloud)
        self.__srs.append(sr)
        area = (self.__box[2] - self.__box[0]) * (self.__box[3] - self.__box[1])
        if area > 0:
            self.__dens.append(cloud.shape[0] / area)
        else:
            self.__dens.append(0.)

    @abstractmethod
    def pickle(self, fname):
        raise NotImplementedError('pickle() (SpA). '
                                  'Abstract method, it requires an implementation.')

    #### Internal implemented functionality

    # Computes Ripley's function in H and updates the correspondent lists
    # n: number of samples
    # max_d: max distance for being considered
    # border: if 0 (default) border compensation is not active, 1 points inflation mode, 2 Goreaud
    # Returns: Ripley's K values and samples respectively
    def __ripleys_H_test(self, n, max_d, border=0):

        # Initialization
        self.__h = list()
        self.__hx = list()
        box = self.__box

        # Ripleys H computation
        for i, cloud in enumerate(self.__clouds):
            if border == 1:
                # Inflate point cloud
                cloud_inf = self.__inflate_2D(cloud)
                hold_h, hold_x = self.__ripley(cloud_inf, box, n, max_d)
            elif border == 2:
                hold_h, hold_x = self.__ripley_goreaud(cloud, box, n, max_d)
            else:
                hold_h, hold_x = self.__ripley(cloud, box, n, max_d)
            # Low pass filtering
            self.__h.append(lfilter(self.__lpf[0], self.__lpf[1], hold_h))
            # self.__h.append(hold_h)
            self.__hx.append(hold_x)

    # Computes Ripley's L form from H
    def __ripleys_L(self):
        for (hx, h) in zip(self.__hx, self.__h):
            self.__lx.append(hx)
            # Low pass filtering
            self.__l.append(lfilter(self.__lpf[0], self.__lpf[1], h+hx))

    # Computes Ripley's H first derivative
    def __ripleys_Hp(self):
        for (hx, h) in zip(self.__hx, self.__h):
            self.__hpx.append(hx)
            if hx.shape[0] > 1:
                # Equally spaced differential
                self.__hp.append(np.gradient(h, hx[1] - hx[0]))
            else:
                self.__hp.append(np.asarray(.0))

    # Computes Ripley's L first derivative
    def __ripleys_Lp(self):
        for (lx, l) in zip(self.__lx, self.__l):
            self.__lpx.append(lx)
            if lx.shape[0] > 1:
                # Equally spaced differential
                self.__lp.append(np.gradient(l, lx[1] - lx[0]))
            else:
                self.__lp.append(np.asarray(.0))

    def __is_not_closer_to_border(self, p, box, max_d):

        # Border distances
        hold = p[0] - box[0]
        d_1 = math.sqrt(hold * hold)
        hold = p[0] - box[2]
        d_2 = math.sqrt(hold * hold)
        hold = p[1] - box[1]
        d_3 = math.sqrt(hold * hold)
        hold = p[1] - box[3]
        d_4 = math.sqrt(hold * hold)

        if (d_1 < max_d) or (d_2 < max_d) or (d_3 < max_d) or (d_4 < max_d):
            return False
        else:
            return True

    # Inflates a 2D spatial cloud of points by adding 8 flipped versions of the original data
    # in its neighbourhood
    def __inflate_2D(self, cloud):

        # Flipping
        flip_x, flip_y = flip_cloud(cloud, 0), flip_cloud(cloud, 1)
        flip_xy = flip_cloud(flip_x, 1)

        # Computing bounding box
        min_x, min_y, max_x, max_y = cloud[:, 0].min(), cloud[:, 1].min(), \
                                     cloud[:, 0].max(), cloud[:, 1].max()

        # Adding neighbours
        c_00 = np.zeros(shape=cloud.shape, dtype=cloud.dtype)
        c_01 = np.zeros(shape=cloud.shape, dtype=cloud.dtype)
        c_02 = np.zeros(shape=cloud.shape, dtype=cloud.dtype)
        c_10 = np.zeros(shape=cloud.shape, dtype=cloud.dtype)
        c_12 = np.zeros(shape=cloud.shape, dtype=cloud.dtype)
        c_20 = np.zeros(shape=cloud.shape, dtype=cloud.dtype)
        c_21 = np.zeros(shape=cloud.shape, dtype=cloud.dtype)
        c_22 = np.zeros(shape=cloud.shape, dtype=cloud.dtype)
        c_00[:, 0], c_00[:, 1] = flip_xy[:, 0] - max_x, flip_xy[:, 1] - max_y
        c_01[:, 0], c_01[:, 1] = flip_x[:, 0] - max_x, flip_x[:, 1]
        c_02[:, 0], c_02[:, 1] = flip_xy[:, 0] - max_x, flip_xy[:, 1] + max_y
        c_10[:, 0], c_10[:, 1] = flip_y[:, 0], flip_y[:, 1] - max_y
        c_12[:, 0], c_12[:, 1] = flip_y[:, 0], flip_y[:, 1] + max_y
        c_20[:, 0], c_20[:, 1] = flip_xy[:, 0] + max_x, flip_xy[:, 1] - max_y
        c_21[:, 0], c_21[:, 1] = flip_x[:, 0] + max_x, flip_x[:, 1]
        c_22[:, 0], c_22[:, 1] = flip_xy[:, 0] + max_x, flip_xy[:, 1] + max_y

        # Concatenate result
        return np.concatenate([c_00, c_01, c_02, c_10, cloud, c_12, c_20, c_21, c_22], axis=0)

    # cloud: cloud of points
    # box: only points within this box are considered for k-function, the rest are only
    #      considered for edge correction
    # n: number of output samples
    # max_d: maximum distance
    # Returns: Ripley's H form and the radius samples
    def __ripley(self, cloud, box, n, max_d):

        # Non-edge correction points detection
        hold = (cloud[:, 0] >= box[0]) & (cloud[:, 1] >= box[1]) & \
               (cloud[:, 0] <= box[2]) & (cloud[:, 1] <= box[3])
        core_ids = np.where(hold)[0]

        # Initialization
        side_a = float(box[2] - box[0])
        side_b = float(box[3] - box[1])
        if (max_d > side_a) or (max_d > side_b):
            error_msg = 'Ripley''s metric cannot be computed because max_d is greater than a cloud box dimension'
            raise pexceptions.PySegInputError(expr='__ripley (SetClouds)', msg=error_msg)
        area = side_a * side_b
        rd = np.linspace(0, max_d, n)
        N = float(len(core_ids))
        K = np.zeros(shape=n, dtype=float)
        if N <= 1:
            return K, rd

        # Cluster radius loop
        for k, r in enumerate(rd):

            # Points loop
            for i in range(int(N)):

                # Finding neighbours
                hold = cloud[i] - cloud
                dists = np.sqrt(np.sum(hold*hold, axis=1))
                k_hold = ((dists > 0) & (dists < r)).sum()

                # Updating K entry
                K[k] += k_hold

        # Compute the H form
        # return np.sqrt((area*K) / (np.pi*N*(N-1))) - rd, rd
        return np.sqrt((area*K) / (np.pi*N*N)) - rd, rd

    # Edge compensation as Goreaud specifies [J. Vegetation Sci. 10: 433-438, 1999]
    # cloud: cloud of points
    # box: only points within this box are considered for k-function, the rest are only
    #      considered for edge correction
    # n: number of output samples
    # max_d: maximum distance
    # Returns: Ripley's H form and the radius samples
    def __ripley_goreaud(self, cloud, box, n, max_d):

        # Initialization
        pi_2 = 2 * np.pi
        side_a = float(box[2] - box[0])
        side_b = float(box[3] - box[1])
        if (max_d > side_a) or (max_d > side_b):
            error_msg = 'Ripley''s metric cannot be computed because max_d is greater than a cloud box dimension'
            raise pexceptions.PySegInputError(expr='__ripley (SetClouds)', msg=error_msg)
        area = side_a * side_b
        rd = np.linspace(0, max_d, n)
        N = float(cloud.shape[0])
        K = np.zeros(shape=n, dtype=float)
        if N <= 1:
            return K, rd

        # Cluster radius loop
        for k, r in enumerate(rd):

            if r == 0:
                continue

            # Points loop
            for i in range(int(N)):

                # Finding neighbours
                hold = cloud[i] - cloud
                dists = np.sqrt(np.sum(hold*hold, axis=1))
                ids = np.where((dists > 0) & (dists < r))[0]

                # Loop for neighbours
                p = cloud[i, :]
                weights = np.ones(shape=len(ids), dtype=float)
                # Distance to edges
                hold_dists = list()
                hold_dists.append(box[2] - p[0])
                hold_dists.append(p[1] - box[1])
                hold_dists.append(p[0] - box[0])
                hold_dists.append(box[3] - p[1])
                hold_dists = np.asarray(hold_dists, dtype=float)
                hold_dists = np.sqrt(hold_dists * hold_dists)
                hold_dists = np.sort(hold_dists)
                d1, d2, d3, d4 = hold_dists[0], hold_dists[1], hold_dists[2], hold_dists[3]
                for j, idx in enumerate(ids):

                    # Compute distance to neighbour
                    pn = cloud[idx, :]
                    hold_r = p - pn
                    rj = math.sqrt((hold_r * hold_r).sum())

                    #### Edge compensation

                    # Switch for computing angle
                    if (rj > d1) and (rj <= d2) and (rj <= d3) and (rj <= d4):
                        alpha = 2 * math.acos(d1 / rj)
                    elif (rj > d1) and (rj > d2) and (rj <= d3) and (rj <= d4):
                        dh = d1*d1 + d2*d2
                        r2 = rj * rj
                        if r2 <= dh:
                            alpha = 2*math.acos(d1/rj) + 2*math.acos(d2/rj)
                        else:
                            alpha = .5*np.pi + math.acos(d1/r) + math.acos(d2/r)
                    elif (rj > d1) and (rj > d3) and (rj <= d2) and (rj <= d4):
                        alpha = 2*math.acos(d1/rj) + 2*math.acos(d3/rj)
                    elif (rj > d1) and (rj > d2) and (rj > d3) and (rj <= d4):
                        d12 = d1*d1 + d2*d2
                        d23 = d2*d2 + d3*d3
                        r2 = rj * rj
                        if (r2 <= d12) and (r2 <= d23):
                            alpha = 2*math.acos(d1/rj) + 2*math.acos(d2/rj) + 2*math.acos(d3/rj)
                        elif (r2 <= d12) and (r2 > d23):
                            alpha = .5*np.pi + 2*math.acos(d1/rj) + math.acos(d2/rj) + math.acos(d3/rj)
                        else:
                            alpha = np.pi + math.acos(d1/rj) + math.acos(d3/rj)
                    else:
                        alpha = .0

                    # Correcting factor
                    if alpha > pi_2:
                        weights[j] = 0.
                    else:
                        weights[j] = pi_2 / (pi_2 - alpha)

                # Updating K entry
                K[k] += (weights.sum())

        # Compute the H form
        # return np.sqrt((area*K) / (np.pi*N*(N-1))) - rd, rd
        return np.sqrt((area*K) / (np.pi*N*N)) - rd, rd

    # Computes G function for every slice
    # n: number of samples for cdf
    def __function_G(self, n):

        # Generate random points
        for i, cloud in enumerate(self.__clouds):
            dists = nnde(cloud)
            hold_g, hold_gx = compute_cdf(dists, n)
            self.__g.append(hold_g)
            self.__gx.append(hold_gx)

    #### Internal abstract functionality

    # n: number of samples for cdf
    # m: number of simulations for cdf
    @abstractmethod
    def __function_F(self, n, m):
        raise NotImplementedError('__function_F() (SpA). '
                                  'Abstract method, it requires an implementation.')

    # n: number of samples for cdf
    # m: number of simulations for cdf
    # p: percentile for computing envelopes, inf None (default) only median is computed
    # Returns: samples, F-Function, median and percentiles p and 100-p, if p is None the last two
    # are zero arrays
    @abstractmethod
    def __rand_function_F(self, n, m, p=None):
        raise NotImplementedError('__rand_function_F() (SpA). '
                                  'Abstract method, it requires an implementation.')

    # n: number of samples for cdf
    # m: number of simulations for cdf
    # p: percentile for computing envelopes, inf None (default) only median is computed
    # Returns: samples, G-Function, median and percentiles p and 100-p, if p is None the last two
    # are zero arrays
    @abstractmethod
    def __rand_function_G(self, n, m, p=None):
        raise NotImplementedError('__rand_function_G() (SpA). '
                                  'Abstract method, it requires an implementation.')

###########################################################################################
# Class for doing a spatial analysis from several independent set of points in a slice
###########################################################################################

class SetCloudsP(SlA):

    # box: unique bounding box
    # n_samp: number of samples for graphs
    # n_sim_f: number of simulations for generating F and G functions
    # r_max: maximum distance for Ripley's H in nm
    # r_bord: if 0 (default) border compensation is not active, 1 points inflation mode and
    #         2 Goreaud
    # p_f: confidence percentile for F and G functions
    def __init__(self, box, n_samp, n_sim_f, r_max, r_bord, p_f):
        super(SetCloudsP, self).__init__(box, n_samp, n_sim_f, r_max, r_bord, p_f)

    #### Set/Get methods area

    #### External functionality area

    # cloud: array with point coordinates in a plane [n, 2]
    # box: bounding box [x_min, y_min, x_max, y_max]
    # sr: sample range [low, high]
    # cards: array with point cardinalities
    def insert_cloud(self, cloud, sr, cards=None):
        super(SetCloudsP, self).insert_cloud(cloud, sr)
        self._SlA__cards.append(cards)

    # Pickling the object state
    # fname: full path for the pickle file
    def pickle(self, fname):
        pkl_f = open(fname, 'w')
        try:
            pickle.dump(self, pkl_f)
        finally:
            pkl_f.close()

    #### Internal functionality area

    def _SlA__rand_function_G(self, n, m, p=None):

        # Generate random points
        dists = list()
        cdfs = np.zeros(shape=(n, m*len(self._SlA__clouds)), dtype=float)
        cont = 0
        # Random simulation
        for i in range(m):
            for j, cloud in enumerate(self._SlA__clouds):
                rand_dists = nnde(gen_rand_cloud(cloud.shape[0], self._SlA__box))
                cdfs[:, cont], _ = compute_cdf(rand_dists, n)
                cont += 1
        # Real data
        for cloud in self._SlA__clouds:
            hold_dists = nnde(cloud)
            dists += hold_dists.tolist()
        dists = np.asarray(dists, dtype=float)

        # Compute results
        gf, sp = compute_cdf(dists, n)
        env_05 = func_envelope(cdfs, per=50)
        if p is None:
            return sp, env_05, \
                   np.zeros(shape=len(sp), dtype=float), np.zeros(shape=len(sp), dtype=float)
        else:
            env_1 = func_envelope(cdfs, per=p)
            env_2 = func_envelope(cdfs, per=100-p)
            return sp, env_05, env_1, env_2

    def _SlA__rand_function_F(self, n, m, p=None):

        # Generate random points
        dists = list()
        cdfs = np.zeros(shape=(n, m*len(self._SlA__clouds)), dtype=float)
        cont = 0
        # Random simulation and real data
        for i in range(m):
            for j, cloud in enumerate(self._SlA__clouds):
                cloud_1 = gen_rand_cloud(cloud.shape[0], self._SlA__box)
                cloud_2 = gen_rand_cloud(cloud.shape[0], self._SlA__box)
                rand_dists = cnnde(cloud_1, cloud_2)
                cdfs[:, cont], _ = compute_cdf(rand_dists, n)
                hold_dists = cnnde(cloud_1, cloud)
                dists += hold_dists.tolist()
                cont += 1
        dists = np.asarray(dists, dtype=float)

        # Compute results
        gf, sp = compute_cdf(dists, n)
        env_05 = func_envelope(cdfs, per=50)
        if p is None:
            return sp, env_05, \
                   np.zeros(shape=len(sp), dtype=float), np.zeros(shape=len(sp), dtype=float)
        else:
            env_1 = func_envelope(cdfs, per=p)
            env_2 = func_envelope(cdfs, per=100-p)
            return sp, env_05, env_1, env_2

    # Computes F function for every slice
    # n: number of samples for cdf
    # m: number of random simulations
    # Returns: F-Function values and samples respectively
    def _SlA__function_F(self, n, m):

        # Generate random points
        for i, cloud in enumerate(self._SlA__clouds):
            dists = list()
            for j in range(m):
                dists += cnnde(cloud, gen_rand_cloud(cloud.shape[0], self._SlA__box)).tolist()
            dists = np.asarray(dists, dtype=float)
            hold_f, hold_fx = compute_cdf(dists, n)
            self._SlA__f.append(hold_f)
            self._SlA__fx.append(hold_fx)

###########################################################################################
# Class for doing a spatial analysis from cluster of points in a slice
###########################################################################################

class SetClustersP(SlA):

    # box: unique bounding box
    # n_samp: number of samples for graphs
    # n_sim_f: number of simulations for generating F and G functions
    # r_max: maximum distance for Ripley's H in nm
    # r_bord: if 0 (default) border compensation is not active, 1 points inflation mode and
    #         2 Goreaud
    # p_f: confidence percentile for F and G functions
    # r_t: number of tries for random clusters generation
    def __init__(self, box, n_samp, n_sim_f, r_max, r_bord, p_f, r_t=50):
        super(SetClustersP, self).__init__(box, n_samp, n_sim_f, r_max, r_bord, p_f)
        self.__clsts_l = list()
        self.__masks_l = list()
        self.__r_t = r_t

    #### External functionality area

    # cloud: array with point coordinates of clusters centers of gravity in a plane [n, 2]
    # sr: sample range [low, high]
    # clsts: ordered list with clusters, each clusters is an array of points
    # mask: mask where False-values mark invalid regions
    def insert_cloud(self, cloud_cg, sr, clsts, mask):
        super(SetClustersP, self).insert_cloud(cloud_cg, sr)
        self.__clsts_l.append(clsts)
        self.__masks_l.append(mask)

    # Pickling the object state
    # fname: full path for the pickle file
    def pickle(self, fname):
        pkl_f = open(fname, 'w')
        try:
            pickle.dump(self, pkl_f)
        finally:
            pkl_f.close()

    #### Internal functionality area

    # Generates a random distribution of the internal clusters
    # clsts: list of clusters
    # box: bounding box
    # mask: binary mask where False valued regions are invalids
    # tries: number of tries for getting the less overlapped location for every cluster
    # Returns: an array with new centroids
    def __get_rand_clsts(self, clsts, box, mask):

        # Initialization
        n_cgs = np.zeros(shape=(len(clsts), 2), dtype=float)

        # Loop for clusters
        mask_h = np.copy(mask)
        for i, c_cloud in enumerate(clsts):

            # Translate to base coordinates and computes minimum distance to center of gravity
            cg = c_cloud.mean(axis=0)
            f_cloud = c_cloud - cg
            # Compute valid search areas
            dst_t = sp.ndimage.morphology.distance_transform_edt(mask_h)
            mask_dst = np.zeros(shape=mask_h.shape, dtype=mask_h.dtype)
            mask_dst[dst_t > 0] = True
            if (dst_t > 0).sum() <= 0:
                error_msg = 'Mask fully overlapped.'
                raise pexceptions.PySegTransitionError(expr='__get_rand_clsts (SetClustersP)',
                                                       msg=error_msg)
            # Keep the best try (lower overlapping)
            min_ov = MAX_FLOAT
            h_cg = None
            h_chull = np.zeros(shape=mask_h.shape, dtype=mask_h.dtype)
            for c_try in range(self.__r_t):
                # Random selection for the new centroid from valid areas
                m_ids = np.where(mask_dst)
                r_x, r_y = np.random.randint(0, len(m_ids[0])), np.random.randint(0, len(m_ids[1]))
                cg_x, cg_y = m_ids[0][r_x], m_ids[1][r_y]
                # Rotate randomly against base center [0, 0]
                rho = np.random.rand() * (2*np.pi)
                sinr, cosr = math.sin(rho), math.cos(rho)
                r_cloud = np.zeros(shape=f_cloud.shape, dtype=f_cloud.dtype)
                r_cloud[:, 0] = f_cloud[:, 0]*cosr - f_cloud[:, 1]*sinr
                r_cloud[:, 1] = f_cloud[:, 0]*sinr + f_cloud[:, 1]*cosr
                # Translation to randomly already selected center
                n_cg = np.asarray((cg_x, cg_y) , dtype=float)
                # v = n_cg - cg
                t_cloud = r_cloud + n_cg
                chull, _ = self.__compute_chull_no_bound(t_cloud, box)
                # Update minimum overlap
                ov = chull.sum() - (chull * mask_h).sum()
                if ov < min_ov:
                    min_ov = ov
                    h_cg = n_cg
                    h_chull = chull
                    if ov == 0:
                        break
                else:
                    if h_cg is None:
                        h_cg = n_cg

            # Update mask
            mask_h[h_chull] = False
            # Get new center transposed
            n_cgs[i, 0] = h_cg[1]
            n_cgs[i, 1] = h_cg[0]

        return n_cgs

    # Returns convex hull and discard points out of bounds are discarded and no exception is
    #           raised, instead in a second variable a true is returned
    def __compute_chull_no_bound(self, c_cloud, box):

        # Create holding image
        off_x = math.floor(box[1])
        off_y = math.floor(box[0])
        m, n = math.ceil(box[3]) - off_x + 1, math.ceil(box[2]) - off_y + 1
        img = np.zeros(shape=(m, n), dtype=bool)

        # Filling holding image
        hold = np.asarray(np.round(c_cloud), dtype=int)
        hold[:, 0] -= off_y
        hold[:, 1] -= off_x
        excep = False
        p_count = 0
        for p in hold:
            try:
                img[p[0], p[1]] = True
            except IndexError:
                excep = True
                continue
            p_count += 1

        # Computing the convex hull
        if p_count > 0:
            chull = np.asarray(convex_hull_image(img), dtype=bool)
        else:
            chull = img

        return chull, excep

    def _SlA__rand_function_G(self, n, m, p=None):

        # Generate random points
        dists = list()
        cdfs = np.zeros(shape=(n, m*len(self.__clsts_l)), dtype=float)
        cont = 0
        for i in range(m):
            for j, clsts in enumerate(self.__clsts_l):
                hold_dists = nnde(self.__get_rand_clsts(clsts, self._SlA__box,
                                                        self.__masks_l[j]))
                cdfs[:, cont], _ = compute_cdf(hold_dists, n)
                dists += hold_dists.tolist()
                cont += 1
        dists = np.asarray(dists, dtype=float)

        # Compute results
        gf, sp = compute_cdf(dists, n)
        env_05 = func_envelope(cdfs, per=50)
        if p is None:
            return sp, env_05, \
                   np.zeros(shape=len(sp), dtype=float), np.zeros(shape=len(sp), dtype=float)
        else:
            env_1 = func_envelope(cdfs, per=p)
            env_2 = func_envelope(cdfs, per=100-p)
            return sp, env_05, env_1, env_2

    def _SlA__rand_function_F(self, n, m, p=None):

        # Generate random points
        dists = list()
        cdfs = np.zeros(shape=(n, m*len(self.__clsts_l)), dtype=float)
        cont = 0
        for i in range(m):
            for j, clsts in enumerate(self.__clsts_l):
                cloud_1 = self.__get_rand_clsts(clsts, self._SlA__box, self.__masks_l[j])
                cloud_2 = self.__get_rand_clsts(clsts, self._SlA__box, self.__masks_l[j])
                hold_dists = cnnde(cloud_1, cloud_2)
                cdfs[:, cont], _ = compute_cdf(hold_dists, n)
                dists += hold_dists.tolist()
                cont += 1
        dists = np.asarray(dists, dtype=float)

        # Compute results
        gf, sp = compute_cdf(dists, n)
        env_05 = func_envelope(cdfs, per=50)
        if p is None:
            return sp, env_05, \
                   np.zeros(shape=len(sp), dtype=float), np.zeros(shape=len(sp), dtype=float)
        else:
            env_1 = func_envelope(cdfs, per=p)
            env_2 = func_envelope(cdfs, per=100-p)
            return sp, env_05, env_1, env_2

    # Computes F function for every slice
    # n: number of samples for cdf
    # m: number of random simulations
    def _SlA__function_F(self, n, m):

        # Generate random points
        for i, cloud in enumerate(self._SlA__clouds):
            dists = list()
            for j in range(m):
                dists += cnnde(cloud, self.__get_rand_clsts(self.__clsts_l[i],
                                                            self._SlA__box,
                                                            self.__masks_l[i])).tolist()
            dists = np.asarray(dists, dtype=float)
            hold_f, hold_fx = compute_cdf(dists, n)
            self._SlA__f.append(hold_f)
            self._SlA__fx.append(hold_fx)

###########################################################################################
# Class for doing a spatial analysis from a pair of point clouds
# VERY IMPORTANT: only valid for 2D data
###########################################################################################

class PairClouds(object):

    # cloud_a/b: the pair of clouds (2D)
    # box: bounding box [x_min, y_min, x_max, y_max] or the enclosing euclidean space
    def __init__(self, cloud_a, cloud_b, box):
        self.__cloud_a = cloud_a
        self.__cloud_b = cloud_b
        self.__box = box
        # For image indexing
        self.__ox = int(math.floor(self.__box[1]))
        self.__oy = int(math.floor(self.__box[0]))
        self.__m = int(math.ceil(self.__box[3]) - self.__ox + 1)
        self.__n = int(math.ceil(self.__box[2]) - self.__oy + 1)

    #### Get/Set methods

    #### External functionality area

    # Classifies the euclidean space according kNN classifier (Brute Force)
    # k: number of neighbours (it should be odd)
    # mask: if not None (default), image with bounding box dimensions where 0 values
    # sets the background
    # Return: image with box dimensions, pixel value: 1 side A, 2 side B and 0 bg
    def knn(self, k, mask=None):

        # Initialization
        img = np.zeros(shape=(self.__m, self.__n), dtype=np.uint8)
        cloud_a = np.asarray(np.round(self.__cloud_a), dtype=int)
        cloud_b = np.asarray(np.round(self.__cloud_b), dtype=int)
        cloud = np.concatenate((cloud_a, cloud_b), axis=0)
        cloud[:, 0] -= self.__oy
        cloud[:, 1] -= self.__ox
        border = cloud_a.shape[0]

        # Applying kNN criteria
        for y in range(self.__m):
            for x in range(self.__n):
                # Pixel distance to all cloud points
                pix = np.asarray((x, y))
                hold = pix - cloud
                # Getting the k-neighbours
                idx = np.argsort(np.sum(hold*hold, axis=1))[0:k]
                # kNN discriminant
                s_a = np.sum(idx < border)
                s_b = len(idx) - s_a
                if s_a > s_b:
                    img[y, x] = 1
                else:
                    img[y, x] = 2

        # Masking
        if mask is not None:
            img[mask == 0] = 0

        return img

    # Customized kNN classifier
    # k: number of neighbours
    # mask: if not None (default), image with bounding box dimensions where 0 values
    # sets the background
    # max_dist: if closest neighbour is farther than it, it is considered bg, default (MAX_FLOAT).
    #           This distance is meausred in pixels
    # Return: image with box dimensions, pixel value: 1 side A, 2 side B, 3 mix and 0 bg
    def knnc(self, k, mask=None, max_dist=MAX_FLOAT):

        # Initialization
        img = np.zeros(shape=(self.__m, self.__n), dtype=np.uint8)
        cloud_a = np.asarray(np.round(self.__cloud_a), dtype=int)
        cloud_b = np.asarray(np.round(self.__cloud_b), dtype=int)
        cloud = np.concatenate((cloud_a, cloud_b), axis=0)
        cloud[:, 0] -= self.__oy
        cloud[:, 1] -= self.__ox
        border = cloud_a.shape[0]

        # Applying kNN criteria
        for y in range(self.__m):
            for x in range(self.__n):
                # Pixel distance to all cloud points
                pix = np.asarray((x, y))
                hold = pix - cloud
                # Getting the k-neighbours
                dists = np.sum(hold*hold, axis=1)
                idx = np.argsort(dists)[0:k]
                f_neigh = math.sqrt(dists[idx[0]])
                # kNN discriminant
                if f_neigh < max_dist:
                    lidx = len(idx)
                    s_a = np.sum(idx < border)
                    s_b = lidx - s_a
                    if s_a == lidx:
                        img[y, x] = 1
                    elif s_b == lidx:
                        img[y, x] = 2
                    else:
                        img[y, x] = 3

        # Masking
        if mask is not None:
            img[mask == 0] = 0

        return img

###########################################################################################
# Class for doing a spatial analysis from pairs of independent set of points in a slice
###########################################################################################

class SetPairClouds(object):

    # box_x: 2D boxes of every element of the pair, they should overlap
    # n_samp: number of samples for graphs
    # n_sim_f: number of simulations for generating G function
    # r_max: maximum distance for Ripley's H in nm
    # p_f: confidence percentile for F and G functions
    # fwd: if True (default) forward cross Ripley's H is computed, backward if False
    def __init__(self, box_a, box_b, n_samp, n_sim_f, r_max, p_f, fwd=True):
        self.__n = n_samp
        self.__nsim_f = n_sim_f
        self.__r_max = r_max
        self.__r_bord = 0
        self.__p_f = p_f
        self.__clouds_a = list()
        self.__clouds_b = list()
        self.__srs = list()
        self.__g = list()
        self.__gx = list()
        self.__grm = np.zeros(shape=n_samp, dtype=float)
        self.__grm1 = np.zeros(shape=n_samp, dtype=float)
        self.__grm2 = np.zeros(shape=n_samp, dtype=float)
        self.__grmx = np.zeros(shape=n_samp, dtype=float)
        self.__hcr = list()
        self.__hcp = list()
        self.__hx = list()
        self.__box = merge_boxes_2D(box_a, box_b)
        l1 = self.__box[0] * self.__box[2]
        l2 = self.__box[1] * self.__box[3]
        if (l1 < self.__r_max) or (l2 < self.__r_max):
            error_msg = 'Ripley\'s H range bigger than overlapped box.'
            raise pexceptions.PySegInputWarning(expr='__init__ (SetPairClouds)',
                                                msg=error_msg)
        # Low pass filter for differentials
        b, a = butter(5, .3, btype='low', analog=False)
        self.__lpf = (b, a)
        self.__fwd = fwd

    ###### Set/Get functionality

    def get_slice_ranges(self):
        return self.__srs

    def get_cross_ripley_H(self):
        return self.__hcr, self.__hx

    def get_comp_ripley_H(self):
        return self.__hcp, self.__hx

    def get_cross_G(self):
        return self.__g, self.__gx

    ###### External functionality area

    def insert_pair(self, cloud_a, cloud_b, sr):
        self.__clouds_a.append(cloud_a)
        self.__clouds_b.append(cloud_b)
        self.__srs.append(sr)

    # Computes crossed F-Function and Ripley's H, and complemented Ripley's H
    def analyze(self, verbose=False):

        if verbose:
            sys.stdout.write('Progress: 0% ... ')

        # G-Function
        self.__function_cross_G(self.__n)
        if verbose:
            sys.stdout.write('20% ... ')
        if self.__nsim_f > 0:
            self.__grmx, self.__grm, self.__grm1, self.__grm2 = self.__rand_function_cross_G(self.__n,
                                                                                             self.__nsim_f,
                                                                                             self.__p_f)
        if verbose:
            sys.stdout.write('40% ... ')
            self.__ripleys_H_cross(self.__n, self.__r_max)
        if verbose:
            sys.stdout.write('80% ... ')
            self.__ripleys_H_comp(self.__n, self.__r_max)

        # Ripley's crossed metrics

        if verbose:
            print('100%')

    # Plot into figures the current analysis state
    # block: if True (default False) waits for closing windows for finishing the execution
    # cloud_over: if True (default) all clouds are plot in the same figure
    # fourier: it True (default) the fourier analysis is also plotted
    # l_metric: it True (default False) the Ripley's L metric is computed
    # r_stat: it True (default False) Ripley's H statistics are measured
    def plot(self, block=False, cloud_over=True, r_stat=False):

        # Initialization
        fig_count = 0
        if block:
            plt.ion()
        labels = self.__srs
        ind = np.arange(1, len(labels)+1)
        color = cm.rainbow(np.linspace(0, 1, len(self.__srs)))

        # Plot clouds
        if cloud_over:
            fig_count += 1
            plt.figure(fig_count)
            plt.title('Clouds of points')
            plt.xlabel('X (nm)')
            plt.ylabel('Y (nm)')
            plt.axis('scaled')
            plt.xlim(self.__box[0], self.__box[2])
            plt.ylim(self.__box[1], self.__box[3])
            for i, cloud_a in enumerate(self.__clouds_a):
                if cloud_a.shape[0] > 0:
                    plt.scatter(cloud_a[:, 0], cloud_a[:, 1], c=color[i], marker='.')
                if self.__clouds_b[i].shape[0] > 0:
                    plt.scatter(self.__clouds_b[i][:, 0], self.__clouds_b[i][:, 1], c=color[i], marker='x')
        else:
            for i, cloud_a in enumerate(self.__clouds_a):
                fig_count += 1
                plt.figure(fig_count)
                plt.title('Clouds of points ' + labels[i])
                plt.xlabel('X (nm)')
                plt.ylabel('Y (nm)')
                plt.axis('scaled')
                plt.xlim(self.__box[0], self.__box[2])
                plt.ylim(self.__box[1], self.__box[3])
                if cloud_a.shape[0] > 0:
                    plt.scatter(cloud_a[:, 0], cloud_a[:, 1], marker='.')
                if self.__clouds_b[i].shape[0] > 0:
                    plt.scatter(self.__clouds_b[i][:, 0], self.__clouds_b[i][:, 1], marker='x')

        # Plot crossed G-Function
        fig_count += 1
        plt.figure(fig_count)
        plt.title('Crossed G-Function')
        plt.xlabel('Distance (nm)')
        plt.ylabel('G')
        plt.ylim(0, 1)
        if self.__nsim_f > 0:
            plt.plot(self.__grmx, self.__grm, 'k')
            if self.__p_f is not None:
                plt.plot(self.__grmx, self.__grm1, 'k--')
                plt.plot(self.__grmx, self.__grm2, 'k--')
        lines = list()
        for (g, gx, c, lbl) in zip(self.__g, self.__gx, color, labels):
            line, = plt.plot(gx, g, c=c, label=lbl)
            lines.append(line)
        if len(lines) > 0:
            plt.legend(handles=lines)

        # Plot crossed Ripley's H
        fig_count += 1
        plt.figure(fig_count)
        plt.title('Crossed Ripley H')
        plt.xlabel('Radius (nm)')
        plt.ylabel('H')
        for (h, hx, c, lbl) in zip(self.__hcr, self.__hx, color, labels):
            plt.plot(hx, h, c=c, label=lbl)
        if len(self.__hcr) > 0:
            plt.plot(self.__hx[0], np.zeros(shape=len(self.__hcr[0])), 'k--')

        # Plot complemented Ripley's H
        fig_count += 1
        plt.figure(fig_count)
        plt.title('Complemented Ripley H')
        plt.xlabel('Radius (nm)')
        plt.ylabel('H')
        for (h, hx, c, lbl) in zip(self.__hcp, self.__hx, color, labels):
            plt.plot(hx, h, c=c, label=lbl)
        if len(self.__hcp) > 0:
            plt.plot(self.__hx[0], np.zeros(shape=len(self.__hcp[0])), 'k--')

        # Plot Riley's H statistics
        if r_stat:
            # Compute stats
            maxs = list()
            medians = list()
            stds = list()
            for h in self.__h:
                maxs.append(h.max())
                medians.append(np.median(h))
                stds.append(h.std())
            nsam = np.arange(len(maxs))
            # Plotting
            fig_count += 1
            plt.figure(fig_count)
            plt.title('Ripley\'s H maximum')
            plt.xlabel('Sample')
            plt.ylabel('H maximum')
            plt.xlim(nsam[0]-1, nsam[-1]+1)
            plt.stem(nsam, np.asarray(maxs, dtype=float))
            fig_count += 1
            plt.figure(fig_count)
            plt.title('Ripley\'s H medians')
            plt.xlabel('Sample')
            plt.ylabel('H medians')
            plt.xlim(nsam[0]-1, nsam[-1]+1)
            plt.stem(nsam, np.asarray(medians, dtype=float))
            fig_count += 1
            plt.figure(fig_count)
            plt.title('Ripley\'s H standard deviations')
            plt.xlabel('Sample')
            plt.ylabel('H deviations')
            plt.xlim(nsam[0]-1, nsam[-1]+1)
            plt.stem(nsam, np.asarray(stds, dtype=float))

        # Show
        plt.show(block=block)

    # Plot into figures the current analysis state
    # path: path to the folder where figures will be stored
    # fourier: it True (default) the fourier analysis is also plotted
    def store_figs(self, path, cloud_over=True):

        # Initialization
        fig_count = 0
        labels = self.__srs
        color = cm.rainbow(np.linspace(0, 1, len(self.__srs)))

        # Plot clouds
        if cloud_over:
            fig_count += 1
            plt.figure(fig_count)
            plt.title('Clouds of points')
            plt.xlabel('X (nm)')
            plt.ylabel('Y (nm)')
            plt.axis('scaled')
            plt.xlim(self.__box[0], self.__box[2])
            plt.ylim(self.__box[1], self.__box[3])
            for i, cloud_a in enumerate(self.__clouds_a):
                if cloud_a.shape[0] > 0:
                    plt.scatter(cloud_a[:, 0], cloud_a[:, 1], c=color[i], marker='.')
                if self.__clouds_b[i].shape[0] > 0:
                    plt.scatter(self.__clouds_b[i][:, 0], self.__clouds_b[i][:, 1], c=color[i], marker='x')
            plt.savefig(path + '/clouds.png')
            plt.close()
        else:
            for i, cloud_a in enumerate(self.__clouds_a):
                fig_count += 1
                plt.figure(fig_count)
                plt.title('Clouds of points ' + labels[i])
                plt.xlabel('X (nm)')
                plt.ylabel('Y (nm)')
                plt.axis('scaled')
                plt.xlim(self.__box[0], self.__box[2])
                plt.ylim(self.__box[1], self.__box[3])
                if cloud_a.shape[0] > 0:
                    plt.scatter(cloud_a[:, 0], cloud_a[:, 1], marker='.')
                if self.__clouds_b[i].shape[0] > 0:
                    plt.scatter(self.__clouds_b[i][:, 0], self.__clouds_b[i][:, 1], marker='x')
                plt.savefig(path + '/cloud_' + labels[i] + '.png')
                plt.close()

        # Plot crossed G-Function
        fig_count += 1
        plt.figure(fig_count)
        plt.title('Crossed G-Function')
        plt.xlabel('Distance (nm)')
        plt.ylabel('G')
        plt.ylim(0, 1)
        if self.__nsim_f > 0:
            plt.plot(self.__grmx, self.__grm, 'k')
            if self.__p_f is not None:
                plt.plot(self.__grmx, self.__grm1, 'k--')
                plt.plot(self.__grmx, self.__grm2, 'k--')
        lines = list()
        for (g, gx, c, lbl) in zip(self.__g, self.__gx, color, labels):
            line, = plt.plot(gx, g, c=c, label=lbl)
            lines.append(line)
        if len(lines) > 0:
            plt.legend(handles=lines)
        plt.savefig(path + '/g_cr.png')
        plt.close()

        # Plot crossed Ripley's H
        fig_count += 1
        plt.figure(fig_count)
        plt.title('Crossed Ripley H')
        plt.xlabel('Radius (nm)')
        plt.ylabel('H')
        for (h, hx, c, lbl) in zip(self.__hcr, self.__hx, color, labels):
            plt.plot(hx, h, c=c, label=lbl)
        if len(self.__hcr) > 0:
            plt.plot(self.__hx[0], np.zeros(shape=len(self.__hcr[0])), 'k--')
        plt.savefig(path + '/h_cr.png')
        plt.close()

        # Plot complemented Ripley's H
        fig_count += 1
        plt.figure(fig_count)
        plt.title('Complemented Ripley H')
        plt.xlabel('Radius (nm)')
        plt.ylabel('H')
        for (h, hx, c, lbl) in zip(self.__hcp, self.__hx, color, labels):
            plt.plot(hx, h, c=c, label=lbl)
        if len(self.__hcp) > 0:
            plt.plot(self.__hx[0], np.zeros(shape=len(self.__hcp[0])), 'k--')
        plt.savefig(path + '/h_cp.png')
        plt.close()

    # Pickling the object state
    # fname: full path for the pickle file
    def pickle(self, fname):
        pkl_f = open(fname, 'w')
        try:
            pickle.dump(self, pkl_f)
        finally:
            pkl_f.close()

    ##### Internal functionality area

    # Computes crossed G function for a pair of slices
    # n: number of samples for cdf
    def __function_cross_G(self, n):

        # Generate random points
        for cloud_a, cloud_b in zip(self.__clouds_a, self.__clouds_b):
            dists = cnnde(cloud_b, cloud_a)
            hold_g, hold_gx = compute_cdf(dists, n)
            self.__g.append(hold_g)
            self.__gx.append(hold_gx)

    def __rand_function_cross_G(self, n, m, p=None):

        # Generate random points
        dists = list()
        cdfs = np.zeros(shape=(n, m*len(self.__clouds_b)), dtype=float)
        cont = 0
        for i in range(m):
            for j, cloud_b in enumerate(self.__clouds_b):
                rand_cloud_b = gen_rand_cloud(cloud_b.shape[0], self.__box)
                rand_cloud_a = gen_rand_cloud(self.__clouds_a[j].shape[0], self.__box)
                hold_dists = cnnde(rand_cloud_b, self.__clouds_a[j])
                rand_dists = cnnde(rand_cloud_b, rand_cloud_a)
                cdfs[:, cont], _ = compute_cdf(rand_dists, n)
                dists += hold_dists.tolist()
                cont += 1
        dists = np.asarray(dists, dtype=float)

        # Compute results
        gf, sp = compute_cdf(dists, n)
        env_05 = func_envelope(cdfs, per=50)
        if p is None:
            return sp, env_05, \
                   np.zeros(shape=len(sp), dtype=float), np.zeros(shape=len(sp), dtype=float)
        else:
            env_1 = func_envelope(cdfs, per=p)
            env_2 = func_envelope(cdfs, per=100-p)
            return sp, env_05, env_1, env_2

    # Computes Ripley's crossed function in H and updates the correspondent lists, only Goreaud border
    # compensation is allowed
    # n: number of samples
    # max_d: max distance for being considered
    # Returns: Ripley's K values and samples respectively
    def __ripleys_H_cross(self, n, max_d):

        # Initialization
        self.__h = list()
        self.__hx = list()
        box = self.__box

        # Ripleys H computation
        for cloud_a, cloud_b in zip(self.__clouds_a, self.__clouds_b):
            if self.__fwd:
                hold_h, hold_x = self.__ripley_cross_goreaud(cloud_a, cloud_b, box, n, max_d)
            else:
                hold_h, hold_x = self.__ripley_cross_goreaud(cloud_b, cloud_a, box, n, max_d)
            # Low pass filtering
            self.__hcr.append(lfilter(self.__lpf[0], self.__lpf[1], hold_h))
            # self.__h.append(hold_h)
            self.__hx.append(hold_x)

    # Computes Ripley's function in H of the union of the pairs
    # n: number of samples
    # max_d: max distance for being considered
    # Returns: Ripley's K values and samples respectively
    def __ripleys_H_comp(self, n, max_d, border=0):

        # Initialization
        self.__h = list()
        self.__hx = list()
        box = self.__box

        # Ripleys H computation
        for cloud_a, cloud_b in zip(self.__clouds_a, self.__clouds_b):
            hold_h, hold_x = self.__ripley_goreaud(np.concatenate((cloud_a, cloud_b), axis=0), box, n, max_d)
            # Low pass filtering
            self.__hcp.append(lfilter(self.__lpf[0], self.__lpf[1], hold_h))
            # self.__h.append(hold_h)
            self.__hx.append(hold_x)

    # Edge compensation as Goreaud specifies [J. Vegetation Sci. 10: 433-438, 1999]
    # cloud: cloud of points
    # box: only points within this box are considered for k-function, the rest are only
    #      considered for edge correction
    # n: number of output samples
    # max_d: maximum distance
    # Returns: Ripley's H form and the radius samples
    def __ripley_goreaud(self, cloud, box, n, max_d):

        # Initialization
        pi_2 = 2 * np.pi
        side_a = float(box[2] - box[0])
        side_b = float(box[3] - box[1])
        if (max_d > side_a) or (max_d > side_b):
            error_msg = 'Ripley''s metric cannot be computed because max_d is greater than a cloud box dimension'
            raise pexceptions.PySegInputError(expr='__ripley (SetClouds)', msg=error_msg)
        area = side_a * side_b
        rd = np.linspace(0, max_d, n)
        N = float(cloud.shape[0])
        K = np.zeros(shape=n, dtype=float)
        if N <= 1:
            return K, rd

        # Cluster radius loop
        for k, r in enumerate(rd):

            if r == 0:
                continue

            # Points loop
            for i in range(int(N)):

                # Finding neighbours
                hold = cloud[i] - cloud
                dists = np.sqrt(np.sum(hold*hold, axis=1))
                ids = np.where((dists > 0) & (dists < r))[0]

                # Loop for neighbours
                p = cloud[i, :]
                weights = np.ones(shape=len(ids), dtype=float)
                # Distance to edges
                hold_dists = list()
                hold_dists.append(box[2] - p[0])
                hold_dists.append(p[1] - box[1])
                hold_dists.append(p[0] - box[0])
                hold_dists.append(box[3] - p[1])
                hold_dists = np.asarray(hold_dists, dtype=float)
                hold_dists = np.sqrt(hold_dists * hold_dists)
                hold_dists = np.sort(hold_dists)
                d1, d2, d3, d4 = hold_dists[0], hold_dists[1], hold_dists[2], hold_dists[3]
                for j, idx in enumerate(ids):

                    # Compute distance to neighbour
                    pn = cloud[idx, :]
                    hold_r = p - pn
                    rj = math.sqrt((hold_r * hold_r).sum())

                    #### Edge compensation

                    # Switch for computing angle
                    if (rj > d1) and (rj <= d2) and (rj <= d3) and (rj <= d4):
                        alpha = 2 * math.acos(d1 / rj)
                    elif (rj > d1) and (rj > d2) and (rj <= d3) and (rj <= d4):
                        dh = d1*d1 + d2*d2
                        r2 = rj * rj
                        if r2 <= dh:
                            alpha = 2*math.acos(d1/rj) + 2*math.acos(d2/rj)
                        else:
                            alpha = .5*np.pi + math.acos(d1/r) + math.acos(d2/r)
                    elif (rj > d1) and (rj > d3) and (rj <= d2) and (rj <= d4):
                        alpha = 2*math.acos(d1/rj) + 2*math.acos(d3/rj)
                    elif (rj > d1) and (rj > d2) and (rj > d3) and (rj <= d4):
                        d12 = d1*d1 + d2*d2
                        d23 = d2*d2 + d3*d3
                        r2 = rj * rj
                        if (r2 <= d12) and (r2 <= d23):
                            alpha = 2*math.acos(d1/rj) + 2*math.acos(d2/rj) + 2*math.acos(d3/rj)
                        elif (r2 <= d12) and (r2 > d23):
                            alpha = .5*np.pi + 2*math.acos(d1/rj) + math.acos(d2/rj) + math.acos(d3/rj)
                        else:
                            alpha = np.pi + math.acos(d1/rj) + math.acos(d3/rj)
                    else:
                        alpha = .0

                    # Correcting factor
                    if alpha > pi_2:
                        weights[j] = 0.
                    else:
                        weights[j] = pi_2 / (pi_2 - alpha)

                # Updating K entry
                K[k] += (weights.sum())

        # Compute the H form
        # return np.sqrt((area*K) / (np.pi*N*(N-1))) - rd, rd
        return np.sqrt((area*K) / (np.pi*N*N)) - rd, rd

    # Crossed Ripley's H form coputation with edge compensation as Goreaud specifies
    # [J. Vegetation Sci. 10: 433-438, 1999]
    # cloud_a: cloud of points for taking the measures
    # cloud_b: cloud of points working as neighbours
    # box: only points within this box are considered for k-function, the rest are only
    #      considered for edge correction
    # n: number of output samples
    # max_d: maximum distance
    # Returns: Ripley's H form and the radius samples
    def __ripley_cross_goreaud(self, cloud_a, cloud_b, box, n, max_d):

        # Initialization
        pi_2 = 2 * np.pi
        side_a = float(box[2] - box[0])
        side_b = float(box[3] - box[1])
        if (max_d > side_a) or (max_d > side_b):
            error_msg = 'Ripley''s metric cannot be computed because max_d is greater than a cloud box dimension'
            raise pexceptions.PySegInputError(expr='__ripley_cross_goreaud (PairSetClouds)', msg=error_msg)
        area = side_a * side_b
        rd = np.linspace(0, max_d, n)
        N = float(cloud_a.shape[0])
        K = np.zeros(shape=n, dtype=float)
        if N <= 1:
            return K, rd

        # Cluster radius loop
        for k, r in enumerate(rd):

            if r == 0:
                continue

            # Points loop
            for i in range(int(N)):

                # Finding neighbours
                hold = cloud_a[i] - cloud_b
                dists = np.sqrt(np.sum(hold*hold, axis=1))
                ids = np.where((dists > 0) & (dists < r))[0]

                # Loop for neighbours
                p = cloud_a[i, :]
                weights = np.ones(shape=len(ids), dtype=float)
                # Distance to edges
                hold_dists = list()
                hold_dists.append(box[2] - p[0])
                hold_dists.append(p[1] - box[1])
                hold_dists.append(p[0] - box[0])
                hold_dists.append(box[3] - p[1])
                hold_dists = np.asarray(hold_dists, dtype=float)
                hold_dists = np.sqrt(hold_dists * hold_dists)
                hold_dists = np.sort(hold_dists)
                d1, d2, d3, d4 = hold_dists[0], hold_dists[1], hold_dists[2], hold_dists[3]
                for j, idx in enumerate(ids):

                    # Compute distance to neighbour
                    pn = cloud_b[idx, :]
                    hold_r = p - pn
                    rj = math.sqrt((hold_r * hold_r).sum())

                    #### Edge compensation

                    # Switch for computing angle
                    if (rj > d1) and (rj <= d2) and (rj <= d3) and (rj <= d4):
                        alpha = 2 * math.acos(d1 / rj)
                    elif (rj > d1) and (rj > d2) and (rj <= d3) and (rj <= d4):
                        dh = d1*d1 + d2*d2
                        r2 = rj * rj
                        if r2 <= dh:
                            alpha = 2*math.acos(d1/rj) + 2*math.acos(d2/rj)
                        else:
                            alpha = .5*np.pi + math.acos(d1/r) + math.acos(d2/r)
                    elif (rj > d1) and (rj > d3) and (rj <= d2) and (rj <= d4):
                        alpha = 2*math.acos(d1/rj) + 2*math.acos(d3/rj)
                    elif (rj > d1) and (rj > d2) and (rj > d3) and (rj <= d4):
                        d12 = d1*d1 + d2*d2
                        d23 = d2*d2 + d3*d3
                        r2 = rj * rj
                        if (r2 <= d12) and (r2 <= d23):
                            alpha = 2*math.acos(d1/rj) + 2*math.acos(d2/rj) + 2*math.acos(d3/rj)
                        elif (r2 <= d12) and (r2 > d23):
                            alpha = .5*np.pi + 2*math.acos(d1/rj) + math.acos(d2/rj) + math.acos(d3/rj)
                        else:
                            alpha = np.pi + math.acos(d1/rj) + math.acos(d3/rj)
                    else:
                        alpha = .0

                    # Correcting factor
                    if alpha > pi_2:
                        weights[j] = 0.
                    else:
                        weights[j] = pi_2 / (pi_2 - alpha)

                # Updating K entry
                K[k] += (weights.sum())

        # Compute the H form
        return np.sqrt((area*K) / (np.pi*N*N)) - rd, rd

###########################################################################################
# Class for finding and analyzing filaments in cloud of points on a plane
# VERY IMPORTANT: only valid for 2D data
###########################################################################################

class NetFilCloud(object):

    # cloud: cloud of points (2D array with n points)
    # res: resolution in nm
    # k: number of nearest neighbours (default 1) for building the graph through knn
    # e_len: edge maximum length
    # min_len: minimum length for the Filaments (default 0)
    # max_len: maximum length for the Filaments (default MAX_FLOAT)
    def __init__(self, cloud, res, k=1, e_len=MAX_FLOAT, min_len=0, max_len=MAX_FLOAT):
        self.__cloud = cloud
        self.__res = res
        self.__e_len = e_len
        self.__graph = self.__build_graph(cloud, k, e_len)
        self.__min_len = min_len
        self.__max_len = max_len
        self.__fils = list()
        self.__find_fils()

    #### Get/Set methods

    def get_filaments(self):
        return self.__fils

    # Returns the number of different vertices which compound the filament network
    def get_num_fil_vertices(self):
        cont = 0
        lut = np.ones(shape=self.__graph.num_vertices(), dtype=bool)
        for fil in self.__fils:
            for v in fil.get_vertices():
                if lut[int(v)]:
                    cont += 1
        return cont

    # Returns the number of different edges which compound the filament network
    def get_num_fil_edges(self):
        cont = 0
        n_verts = self.__graph.num_vertices()
        lut = np.ones(shape=(n_verts, n_verts), dtype=bool)
        for fil in self.__fils:
            for e in fil.get_edges():
                s, t = int(e.source()), (e.target())
                if lut[s, t] and lut[t, s]:
                    cont += 1
                    lut[s, t], lut[t, s] = True, True

        return cont

    def get_graph_vtp(self):

        # Initialization
        point_id = 0
        cell_id = 0
        points = vtk.vtkPoints()
        verts = vtk.vtkCellArray()
        lines = vtk.vtkCellArray()
        cell_data = vtk.vtkIntArray()
        cell_data.SetNumberOfComponents(1)
        cell_data.SetName(STR_CELL)
        len_data = vtk.vtkFloatArray()
        len_data.SetNumberOfComponents(1)
        len_data.SetName(STR_2GT_EL)

        # Write vertices
        for p in self.__cloud:
            verts.InsertNextCell(1)
            points.InsertPoint(point_id, p[0], p[1], 0)
            verts.InsertCellPoint(point_id)
            point_id += 1
            cell_id += 1
            cell_data.InsertNextTuple((cell_id,))
            len_data.InsertNextTuple((NO_CONNECTION,))

        # Write edges
        for e in self.__graph.edges():

            lines.InsertNextCell(2)
            s = self.__cloud[int(e.source())]
            t = self.__cloud[int(e.target())]
            points.InsertPoint(point_id, s[0], s[1], 0)
            lines.InsertCellPoint(point_id)
            point_id += 1
            points.InsertPoint(point_id, t[0], t[1], 0)
            lines.InsertCellPoint(point_id)
            point_id += 1
            cell_id += 1
            cell_data.InsertNextTuple((cell_id,))
            length = s - t
            len_data.InsertNextTuple((math.sqrt((length*length).sum()),))

        # Poly building
        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        poly.SetVerts(verts)
        poly.SetLines(lines)
        poly.GetCellData().AddArray(cell_data)
        poly.GetCellData().AddArray(len_data)

        return poly

    def get_fils_vtp(self):

        # Initialization
        point_id = 0
        cell_id = 0
        points = vtk.vtkPoints()
        verts = vtk.vtkCellArray()
        lines = vtk.vtkCellArray()
        cell_data = vtk.vtkIntArray()
        cell_data.SetNumberOfComponents(1)
        cell_data.SetName(STR_CELL)
        len_data = vtk.vtkFloatArray()
        len_data.SetNumberOfComponents(1)
        len_data.SetName(STR_2FIL_LEN)
        ct_data = vtk.vtkFloatArray()
        ct_data.SetNumberOfComponents(1)
        ct_data.SetName(STR_2FIL_CT)
        sin_data = vtk.vtkFloatArray()
        sin_data.SetNumberOfComponents(1)
        sin_data.SetName(STR_2FIL_SIN)
        smo_data = vtk.vtkFloatArray()
        smo_data.SetNumberOfComponents(1)
        smo_data.SetName(STR_2FIL_SMO)
        mc_data = vtk.vtkFloatArray()
        mc_data.SetNumberOfComponents(1)
        mc_data.SetName(STR_2FIL_MC)

        # Write vertices
        for p in self.__cloud:
            verts.InsertNextCell(1)
            points.InsertPoint(point_id, p[0], p[1], 0)
            verts.InsertCellPoint(point_id)
            point_id += 1
            cell_id += 1
            cell_data.InsertNextTuple((cell_id,))
            len_data.InsertNextTuple((-1,))
            ct_data.InsertNextTuple((-1,))
            sin_data.InsertNextTuple((-1,))
            smo_data.InsertNextTuple((-1,))
            mc_data.InsertNextTuple((-1,))

        # Write lines
        for i, f in enumerate(self.__fils):

            # Getting children if demanded
            coords = f.get_coords()
            lines.InsertNextCell(coords.shape[0])
            for c in coords:
                points.InsertPoint(point_id, c[0], c[1], 0)
                lines.InsertCellPoint(point_id)
                point_id += 1
            cell_id += 1
            cell_data.InsertNextTuple((cell_id,))
            len_data.InsertNextTuple((f.get_length(),))
            ct_data.InsertNextTuple((f.get_total_curvature(),))
            sin_data.InsertNextTuple((f.get_sinuosity(),))
            smo_data.InsertNextTuple((f.get_smoothness(),))
            mc_data.InsertNextTuple((f.get_max_curvature(),))

        # Poly building
        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        poly.SetVerts(verts)
        poly.SetLines(lines)
        poly.GetCellData().AddArray(cell_data)
        poly.GetCellData().AddArray(len_data)
        poly.GetCellData().AddArray(ct_data)
        poly.GetCellData().AddArray(sin_data)
        poly.GetCellData().AddArray(smo_data)
        poly.GetCellData().AddArray(mc_data)

        return poly

    # Generates a window where the network is rendered
    # mode= If 1 (default) the graph is render, otherwise the filament network
    def render(self, mode=1):

        # create a rendering window and renderer
        ren = vtk.vtkRenderer()
        renWin = vtk.vtkRenderWindow()
        renWin.AddRenderer(ren)

        # create a renderwindowinteractor
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)

        # Get source
        if mode == 1:
            source = self.get_graph_vtp()
        else:
            source = self.get_fils_vtp()

        # actor for vertices
        verts = vtk.vtkPolyData()
        verts.SetPoints(source.GetPoints())
        verts.SetVerts(source.GetVerts())
        mapper_v = vtk.vtkPolyDataMapper()
        mapper_v.SetInputData(verts)
        actor_v = vtk.vtkActor()
        actor_v.SetMapper(mapper_v)
        actor_v.GetProperty().SetColor(1,0,0) # (R,G,B)

        # actor for lines
        lines = vtk.vtkPolyData()
        lines.SetPoints(source.GetPoints())
        lines.SetLines(source.GetLines())
        mapper_l = vtk.vtkPolyDataMapper()
        mapper_l.SetInputData(lines)
        actor_l = vtk.vtkActor()
        actor_l.SetMapper(mapper_l)
        actor_l.GetProperty().SetColor(0,1,0) # (R,G,B)

        # assign actors to the renderer
        ren.AddActor(actor_v)
        ren.AddActor(actor_l)

        # enable user interface interactor
        iren.Initialize()
        renWin.Render()
        iren.Start()

    #### External functionality area

    # th_*: threshold objects (logical and is applied to all thresholds)
    def threshold_and_fils(self, th_len=None, th_ct=None, th_sin=None, th_smo=None, th_mc=None):

        # LUT for marking filaments to delete
        del_lut = np.ones(shape=len(self.__fils), dtype=bool)

        # Loop for filaments
        for i, fil in enumerate(self.__fils):

            if th_len is not None:
                if not th_len.test(fil.get_length()):
                    continue
            if th_ct is not None:
                if not th_ct.test(fil.get_total_curvature()):
                    continue
            if th_sin is not None:
                if not th_sin.test(fil.get_sinuosity()):
                    continue
            if th_smo is not None:
                if not th_smo.test(fil.get_smoothness()):
                    continue
            if th_mc is not None:
                if not th_mc.test(fil.get_max_curvature()):
                    continue

            del_lut[i] = False

        # Copy just filaments passed all tests
        hold_fils = self.__fils
        self.__fils = list()
        for i, fil in enumerate(hold_fils):
            if not del_lut[i]:
                self.__fils.append(fil)

    #### Internal functionality area

    # Build a GraphGT by knn criterion
    def __build_graph(self, cloud, k, e_len=MAX_FLOAT):

        if cloud.shape[0] < (k+1):
            error_msg = 'The number of point of the cloud must be greater than k =' + str(k)
            raise pexceptions.PySegInputError(expr='__build_graph (NetFilCloud)', msg=error_msg)

        # Graph initialization
        graph = gt.Graph(directed=False)
        graph.add_vertex(cloud.shape[0])
        lengths = list()

        # Applying kNN criteria
        for i, p in enumerate(cloud):
            # point distance to all cloud points
            hold = p - cloud
            # Getting the k-neighbours
            dists = np.sqrt(np.sum(hold * hold, axis=1))
            ids = np.argsort(dists)[1:k+1]
            # Add the edges
            for idx in ids:
                dist = dists[idx]
                if dist > e_len:
                    break
                lengths.append(dists[idx])
                graph.add_edge(i, idx)

        # Set edge length as the euclidean distance
        graph.edge_properties[STR_2GT_EL] = graph.new_edge_property('float')
        graph.edge_properties[STR_2GT_EL].get_array()[:] = np.asarray(lengths, dtype=float)

        return graph

    def __find_fils(self):

        # Visiting procedure initialization
        n_vertices = self.__graph.num_vertices()
        connt = np.zeros(shape=(n_vertices, n_vertices), dtype=bool)
        prop_con = self.__graph.edge_properties[STR_2GT_EL]

        # Main loop for finding filaments at every vertex
        for source in self.__graph.vertices():

            # An isolated vertex cannot be a Filament
            if sum(1 for _ in source.all_edges()) <= 0:
                continue

            # Search filaments in source neighbourhood
            visitor = FilVisitor2(self.__graph, source, self.__min_len, self.__max_len)
            gt.dijkstra_search(self.__graph, source, prop_con, visitor)
            hold_v_paths, hold_e_paths = visitor.get_paths()

            # Build the filaments
            for i, v_path in enumerate(hold_v_paths):
                head, tail = v_path[0], v_path[-1]
                head_i, tail_i = int(head), int(v_path[-1])
                if not(connt[head_i, tail_i]) and not(connt[tail_i, head_i]):
                    v_list = list()
                    e_list = list()
                    e_path = hold_e_paths[i]
                    for j in range(len(v_path) - 1):
                        v_list.append(v_path[j])
                        e_list.append(e_path[j])
                    v_list.append(v_path[-1])
                    # Building a filament
                    self.__fils.append(FilamentU(self.__graph, self.__cloud,
                                                 v_list, e_list, self.__res))
                    # Set as unconnected already processed pair of vertices
                    connt[head_i, tail_i] = True
                    connt[tail_i, head_i] = True


###########################################################################################
# Class for modelling a filament (unoriented curve in a plane) (input graph is GraphGT)
###########################################################################################

class FilamentU(object):

    # graph: parent GraphGT
    # vertices: list of ordered vertices for the whole GraphGT
    # coords: list of vertex coordinates
    # edge: list of ordered edges, v{i} -> e{i} -> v{i+1}
    # res: resolution in nm
    def __init__(self, graph, coords, vertices, edges, res):
        self.__graph = graph
        self.__vertices = vertices
        self.__coords = self.__get_path_coords(coords)
        self.__edges = edges
        self.__res = res
    #### Set/Get methods area

    def get_edges(self):
        return self.__edges

    def get_vertices(self):
        return self.__vertices

    def get_num_vertices(self):
        return len(self.__vertices)

    def get_head(self):
        return self.__vertices[0]

    def get_tail(self):
        return self.__vertices[-1]

    # Return filament path coordinates order from head to tail
    def get_coords(self):
        return self.__coords

    def get_length(self):
        length = 0.
        coords = self.get_coords()
        for i in range(coords.shape[0] - 1):
            x1, y1 = coords[i, 0], coords[i, 1]
            x2, y2 = coords[i+1, 0], coords[i+1, 1]
            hold = np.asarray((x1-x2, y1-y2), dtype=float)
            length += math.sqrt(np.sum(hold*hold))
        return length * self.__res

    def get_head_tail_dist(self):
        hold = np.asarray((self.__coords[0][0]-self.__coords[-1][0],
                           self.__coords[0][1]-self.__coords[-1][1],), dtype=float)
        return math.sqrt(np.sum(hold*hold)) * self.__res

    # Computes total curvature
    def get_total_curvature(self):

        # Getting curve coordinates in space
        curve = self.__coords * self.__res

        # Computing curvatures
        curvatures = compute_plane_k(curve)

        # Curvature integral
        total_k = 0.
        for i in range(1, curve.shape[0]-1):
            v_i_l1, v_i, v_i_p1 = curve[i-1, :], curve[i, :], curve[i+1, :]
            h_1 = v_i_p1 - v_i
            h_2 = v_i-v_i_l1
            h_1 = math.sqrt(h_1[0]*h_1[0] + h_1[1]*h_1[1])
            h_2 = math.sqrt(h_2[0]*h_2[0] + h_2[1]*h_2[1])
            total_k += (0.5 * (h_1 + h_2) * curvatures[i-1])

        return total_k

    # Computes a smoothness metric based on total curvature
    def get_smoothness(self):

        # Getting curve coordinates in space
        curve = self.__coords * self.__res

        # Computing curvatures
        curvatures = compute_plane_k(curve)

        # Square for avoiding orientation information
        curvatures *= curvatures

        # Curvature integral
        total_k = 0.
        for i in range(1, curve.shape[0]-1):
            v_i_l1, v_i, v_i_p1 = curve[i-1, :], curve[i, :], curve[i+1, :]
            h_1 = v_i_p1 - v_i
            h_2 = v_i-v_i_l1
            h_1 = math.sqrt(h_1[0]*h_1[0] + h_1[1]*h_1[1])
            h_2 = math.sqrt(h_2[0]*h_2[0] + h_2[1]*h_2[1])
            hold = (0.5 * (h_1 + h_2) * curvatures[i-1])
            total_k += (hold * hold)

        return total_k

    # Computes maximum local curvature along the whole network
    def get_max_curvature(self):

        # Getting curve coordinates in space
        curve = self.__coords * self.__res

        # Computing curvatures
        curvatures = np.absolute(compute_plane_k(curve))

        # Maximum
        return curvatures.max()

    # Computes curve sinuosity (ratio between length and distance between extremes)
    def get_sinuosity(self):
        length = self.get_length()
        if length == 0:
            return 0
        dst = self.get_head_tail_dist()
        if dst == 0:
            return 0
        else:
            return length / dst

##### Internal functionality area

    def __get_path_coords(self, cloud):
        n_v = len(self.__vertices)
        coords = np.zeros(shape=(n_v, 2), dtype=float)
        for i in range(n_v):
            coords[i, :] = cloud[int(self.__vertices[i]), :]
        return coords

###########################################################################################
# Class for analyzing groups of clouds
###########################################################################################

class GroupClouds(object):

    # box: unique bounding box
    # n_samp: number of samples for graphs
    # n_sim_f: number of simulations for generating F and G functions
    # max_d: maximum distance for Ripley's metrics
    # p_f: percentile for F and G simulations test, if None (default) tests are not done
    def __init__(self, n_samp, n_sim_f, max_d, p_f=None):
        self.__n = n_samp
        self.__nsim_f = n_sim_f
        self.__max_d = max_d
        self.__p_f = p_f
        self.__groups_cloud = list()
        self.__groups_boxes = list()
        self.__names = list()
        self.__groups_g = list()
        self.__groups_f = list()
        self.__groups_h = list()
        self.__groups_hp = list()
        self.__groups_wh = list()
        self.__groups_whp = list()
        self.__gs = None
        self.__gsx = None
        self.__gsl = None
        self.__gsh = None
        self.__fs = None
        self.__fsx = None
        self.__fsl = None
        self.__fsh = None
        self.__cc_lbls = list()
        self.__cch = None
        self.__cchp = None
        # Low pass filter for differentials
        b, a = butter(LP_ORDER, LP_NORM_CUTOFF, btype='low', analog=False)
        self.__lpf = (b, a)

    # Get/Set functionality

    # External functionality

    # clouds: list of clouds
    # boxes: list of boxes
    # name: string with group name
    def insert_group(self, clouds, boxes, name):
        self.__groups_cloud.append(clouds)
        self.__groups_boxes.append(boxes)
        self.__names.append(name)

    # Pickling the object state
    # fname: full path for the pickle file
    def pickle(self, fname):
        pkl_f = open(fname, 'w')
        try:
            pickle.dump(self, pkl_f)
        finally:
            pkl_f.close()

    def analyze_1(self, verbose=False):

        if verbose:
            sys.stdout.write('Progress analysis level 1: 0% ... ')
        for group in self.__groups_cloud:
            g, gx = self.__group_function_G(group, self.__n)
            self.__groups_g.append((g, gx))

        # Making plane the list of groups
        p_groups = list()
        p_boxes = list()
        for (group, boxes) in zip(self.__groups_cloud, self.__groups_boxes):
            for (cloud, box) in zip(group, boxes):
                p_groups.append(cloud)
                p_boxes.append(box)

        if verbose:
            sys.stdout.write('25% ... ')
        if self.__p_f is not None:
            self.__gsx, self.__gsl, self.__gsm, self.__gsh = self.__rand_group_function_G(p_groups,
                                                                                          p_boxes,
                                                                                          self.__n,
                                                                                          self.__nsim_f,
                                                                                          self.__p_f)

        if verbose:
            sys.stdout.write('50% ... ')
        for (group, boxes) in zip(self.__groups_cloud, self.__groups_boxes):
            f, fx = self.__group_function_F(group, boxes, self.__n, self.__nsim_f)
            self.__groups_f.append((f, fx))

        if verbose:
            sys.stdout.write('75% ... ')
        if self.__p_f is not None:
            self.__fsx, self.__fsl, self.__fsm, self.__fsh = self.__rand_group_function_F(p_groups,
                                                                                          p_boxes,
                                                                                          self.__n,
                                                                                          self.__nsim_f,
                                                                                          self.__p_f)

        if verbose:
            print('100%')

    def analyze_2(self, verbose=False):

        if verbose:
            sys.stdout.write('Progress analysis level 2: 0% ... ')

        tot = 1
        for boxes in self.__groups_boxes:
            tot += len(boxes)

        # Compute Ripleys H for every cloud
        cont = 0
        for (clouds, boxes) in zip(self.__groups_cloud, self.__groups_boxes):
            group_h = list()
            group_hp = list()
            for (cloud, box) in zip(clouds, boxes):
                h, hx = ripley_goreaud(cloud, box, self.__n, self.__max_d)
                h_f = lfilter(self.__lpf[0], self.__lpf[1], h)
                group_h.append((h_f, hx))
                group_hp.append((np.gradient(h_f, hx[1] - hx[0]), hx))
                cont += 1
                pct = 100. * (float(cont) / float(tot))
                sys.stdout.write(str(round(pct, 1)) + '% ... ')
            self.__groups_h.append(group_h)
            self.__groups_hp.append(group_hp)

        # Compute Ripleys for every group
        for (h_pairs, clouds) in zip(self.__groups_h, self.__groups_cloud):
            # Compute weights
            weights = np.zeros(shape=len(clouds), dtype=float)
            for i, cloud in enumerate(clouds):
                weights[i] = cloud.shape[0]
            weights /= weights.sum()
            # Compute averages
            ha = np.zeros(shape=h_pairs[0][0].shape, dtype=float)
            hpa = np.zeros(shape=h_pairs[0][0].shape, dtype=float)
            for i in range(len(h_pairs)):
                ha += (weights[i] * h_pairs[i][0])
                hpa += (weights[i] * np.gradient(h_pairs[i][0], h_pairs[i][1][1] - h_pairs[i][1][0]))
            ha_x = h_pairs[0][1]
            self.__groups_wh.append((ha, ha_x))
            self.__groups_whp.append((hpa, ha_x))

        # Computing cross-correlation coefficients
        h_mat = np.zeros(shape=(tot-1, self.__n), dtype=float)
        hp_mat = np.zeros(shape=(tot-1, self.__n), dtype=float)
        cont = 0
        for i in range(len(self.__groups_h)):
            for j in range(len(self.__groups_h[i])):
                h_mat[cont, :] = self.__groups_h[i][j][0]
                hp_mat[cont, :] = self.__groups_hp[i][j][0]
                self.__cc_lbls.append(i)
                cont += 1
        self.__cch = np.corrcoef(h_mat)
        self.__cchp = np.corrcoef(hp_mat)

        if verbose:
            print('100%')



    # Plot into figures the current analysis level 1 state
    # block: if True (default False) waits for closing windows for finishing the execution
    def plot_1(self, block=False):

        if len(self.__groups_g) == 0:
            if len(self.__names) == 0:
                print('WARNING: no groups added, run insert_group() and analyze_1() first!')
            else:
                print('WARNING: run analyze_1() first!')

        # Initialization
        fig_count = 0
        if block:
            plt.ion()
        width = 0.35
        ind = np.arange(len(self.__names))
        color = cm.rainbow(np.linspace(0, 1, len(self.__names)))

        # Plot densities
        fig_count += 1
        ax = plt.figure(fig_count).add_subplot(111)
        plt.title('Averaged Nearest Neighbour Distance')
        plt.xlabel('Group')
        plt.ylabel('ANN (nm)')
        means = list()
        stds = list()
        for (group, lbl) in zip(self.__groups_g, self.__names):
            means.append(np.mean(group[0]))
            stds.append(np.std(group[0]))
        bars1 = plt.bar(ind, np.asarray(means, dtype=float), width, color='b')
        bars2 = plt.bar(ind+width, np.asarray(stds, dtype=float), width, color='r')
        ax.set_xticks(ind + width)
        ax.set_xticklabels(self.__names)
        ax.legend((bars1[0], bars2[0]), ('Mean', 'Std'))

        # Plot G-Function
        fig_count += 1
        plt.figure(fig_count)
        plt.title('G-Function')
        plt.xlabel('Distance (nm)')
        plt.ylabel('G')
        plt.ylim(0, 1)
        if self.__p_f is not None:
            plt.plot(self.__gsx, self.__gsm, 'k')
            plt.plot(self.__gsx, self.__gsl, 'k--')
            plt.plot(self.__gsx, self.__gsh, 'k--')
        lines = list()
        for (group, lbl, c) in zip(self.__groups_g, self.__names, color):
            line, = plt.plot(group[1], group[0], c=c, label=lbl)
            lines.append(line)
        if len(lines) > 0:
            plt.legend(handles=lines)

        # Plot F-Function
        fig_count += 1
        plt.figure(fig_count)
        plt.title('F-Function')
        plt.xlabel('Distance (nm)')
        plt.ylabel('F')
        plt.ylim(0, 1)
        if self.__p_f is not None:
            plt.plot(self.__fsx, self.__fsm, 'k')
            plt.plot(self.__fsx, self.__fsl, 'k--')
            plt.plot(self.__fsx, self.__fsh, 'k--')
        lines = list()
        for (group, lbl, c) in zip(self.__groups_f, self.__names, color):
            line, = plt.plot(group[1], group[0], c=c, label=lbl)
            lines.append(line)
        if len(lines) > 0:
            plt.legend(handles=lines)

        # Show
        plt.show(block=block)

    # Plot into figures the current analysis level 1 state
    # path: path to the folder where figures will be stored
    def store_figs_1(self, path):

        if len(self.__groups_g) == 0:
            if len(self.__names) == 0:
                print('WARNING: no groups added, run insert_group() and analyze_1() first!')
            else:
                print('WARNING: run analyze_1() first!')

        # Initialization
        fig_count = 0
        width = 0.35
        ind = np.arange(len(self.__names))
        color = cm.rainbow(np.linspace(0, 1, len(self.__names)))

        # Plot densities
        fig_count += 1
        ax = plt.figure(fig_count).add_subplot(111)
        plt.title('Averaged Nearest Neighbour Distance')
        plt.xlabel('Group')
        plt.ylabel('ANN (nm)')
        means = list()
        stds = list()
        for (group, lbl) in zip(self.__groups_g, self.__names):
            means.append(np.mean(group[0]))
            stds.append(np.std(group[0]))
        bars1 = plt.bar(ind, np.asarray(means, dtype=float), width, color='b')
        bars2 = plt.bar(ind+width, np.asarray(stds, dtype=float), width, color='r')
        ax.set_xticks(ind + width)
        ax.set_xticklabels(self.__names)
        ax.legend((bars1[0], bars2[0]), ('Mean', 'Std'))
        plt.savefig(path + '/annd.png')
        plt.close()

        # Plot G-Function
        fig_count += 1
        plt.figure(fig_count)
        plt.title('G-Function')
        plt.xlabel('Distance (nm)')
        plt.ylabel('G')
        plt.ylim(0, 1)
        if self.__p_f is not None:
            plt.plot(self.__gsx, self.__gsm, 'k')
            plt.plot(self.__gsx, self.__gsl, 'k--')
            plt.plot(self.__gsx, self.__gsh, 'k--')
        lines = list()
        for (group, lbl, c) in zip(self.__groups_g, self.__names, color):
            line, = plt.plot(group[1], group[0], c=c, label=lbl)
            lines.append(line)
        if len(lines) > 0:
            plt.legend(handles=lines)
        plt.savefig(path + '/g.png')
        plt.close()

        # Plot F-Function
        fig_count += 1
        plt.figure(fig_count)
        plt.title('F-Function')
        plt.xlabel('Distance (nm)')
        plt.ylabel('F')
        plt.ylim(0, 1)
        if self.__p_f is not None:
            plt.plot(self.__fsx, self.__fsm, 'k')
            plt.plot(self.__fsx, self.__fsl, 'k--')
            plt.plot(self.__fsx, self.__fsh, 'k--')
        lines = list()
        for (group, lbl, c) in zip(self.__groups_f, self.__names, color):
            line, = plt.plot(group[1], group[0], c=c, label=lbl)
            lines.append(line)
        if len(lines) > 0:
            plt.legend(handles=lines)
        plt.savefig(path + '/f.png')
        plt.close()

        # Plot into figures the current analysis level 1 state
    # block: if True (default False) waits for closing windows for finishing the execution
    def plot_2(self, block=False):

        if len(self.__groups_h) == 0:
            if len(self.__names) == 0:
                print('WARNING: no groups added, run insert_group() and analyze_2() first!')
            else:
                print('WARNING: run analyze_2() first!')

        # Initialization
        fig_count = 0
        if block:
            plt.ion()

        # Plot individual Ripley's H
        for (h_pairs, name) in zip(self.__groups_h, self.__names):
            # Plot G-Function
            fig_count += 1
            plt.figure(fig_count)
            plt.title('Ripley\'s H for group ' + str(name))
            plt.xlabel('Radius (nm)')
            plt.ylabel('H')
            cont = 1
            lines = list()
            color = cm.rainbow(np.linspace(0, 1, len(h_pairs)))
            for (h_pair, c) in zip(h_pairs, color):
                line, = plt.plot(h_pair[1], h_pair[0], c=c, label=str(cont))
                lines.append(line)
                cont += 1
            plt.legend(handles=lines)

        # Plot individual Ripley's H'
        for (hp_pairs, name) in zip(self.__groups_hp, self.__names):
            # Plot G-Function
            fig_count += 1
            plt.figure(fig_count)
            plt.title('Ripley\'s H\' for group ' + str(name))
            plt.xlabel('Radius (nm)')
            plt.ylabel('H\'')
            cont = 1
            lines = list()
            color = cm.rainbow(np.linspace(0, 1, len(h_pairs)))
            for (hp_pair, c) in zip(hp_pairs, color):
                line, = plt.plot(hp_pair[1], hp_pair[0], c=c, label=str(cont))
                lines.append(line)
                cont += 1
            plt.legend(handles=lines)

        # Plot weighted Ripleys'H
        fig_count += 1
        plt.figure(fig_count)
        plt.title('Weighted Ripley\'s H')
        plt.xlabel('Radius (nm)')
        plt.ylabel('H')
        lines = list()
        color = cm.rainbow(np.linspace(0, 1, len(self.__groups_wh)))
        for (wh_pair, name, c) in zip(self.__groups_wh, self.__names, color):
            line, = plt.plot(wh_pair[1], wh_pair[0], c=c, label=name)
            lines.append(line)
        plt.legend(handles=lines)

        # Plot weighted Ripleys'H
        fig_count += 1
        plt.figure(fig_count)
        plt.title('Weighted Ripley\'s H\'')
        plt.xlabel('Radius (nm)')
        plt.ylabel('H\'')
        lines = list()
        for (whp_pair, name, c) in zip(self.__groups_whp, self.__names, color):
            line, = plt.plot(whp_pair[1], whp_pair[0], c=c, label=name)
            lines.append(line)
        plt.legend(handles=lines)

        # Plot cross-correlation for Ripley's H
        fig_count += 1
        plt.figure(fig_count)
        plt.title('Cross-correlation matrix for Ripley\'s H')
        plt.xlim(0, self.__cch.shape[0])
        plt.ylim(0, self.__cch.shape[1])
        plt.pcolor(self.__cch, cmap='jet', vmin=-1, vmax=1)
        plt.colorbar()
        plt.xticks(np.arange(.5, self.__cch.shape[0]+.5), self.__cc_lbls)
        plt.yticks(np.arange(.5, self.__cch.shape[1]+.5), self.__cc_lbls)

        # Plot cross-correlation for Ripley's H'
        fig_count += 1
        plt.figure(fig_count)
        plt.title('Cross-correlation matrix for Ripley\'s H\'')
        plt.xlim(0, self.__cchp.shape[0])
        plt.ylim(0, self.__cchp.shape[1])
        plt.pcolor(self.__cchp, cmap='jet', vmin=-1, vmax=1)
        plt.colorbar()
        plt.xticks(np.arange(.5, self.__cchp.shape[0]+.5), self.__cc_lbls)
        plt.yticks(np.arange(.5, self.__cchp.shape[1]+.5), self.__cc_lbls)

        # Show
        plt.show(block=block)

    # Store figures the current analysis level 2 state
    # path: path to the folder where figures will be stored
    # plt_cl: if True (default False) clouds coordinates are stored
    def store_figs_2(self, path, plt_cl=True):

        if len(self.__groups_h) == 0:
            if len(self.__names) == 0:
                print('WARNING: no groups added, run insert_group() and analyze_1() first!')
            else:
                print('WARNING: run analyze_2() first!')

        # Initialization
        fig_count = 0

        # Plot individual Ripley's H
        for (h_pairs, name) in zip(self.__groups_h, self.__names):
            # Plot G-Function
            fig_count += 1
            plt.figure(fig_count)
            plt.title('Ripley\'s H for group ' + name)
            plt.xlabel('Radius (nm)')
            plt.ylabel('H')
            cont = 1
            lines = list()
            color = cm.rainbow(np.linspace(0, 1, len(h_pairs)))
            for (h_pair, c) in zip(h_pairs, color):
                line, = plt.plot(h_pair[1], h_pair[0], c=c, label=str(cont))
                lines.append(line)
                cont += 1
            plt.legend(handles=lines)
            plt.savefig(path + '/' + name + '_h.png')
            plt.close()

        # Plot individual Ripley's H'
        for (hp_pairs, name) in zip(self.__groups_hp, self.__names):
            # Plot G-Function
            fig_count += 1
            plt.figure(fig_count)
            plt.title('Ripley\'s H\' for group ' + str(name))
            plt.xlabel('Radius (nm)')
            plt.ylabel('H\'')
            cont = 1
            lines = list()
            color = cm.rainbow(np.linspace(0, 1, len(h_pairs)))
            for (hp_pair, c) in zip(hp_pairs, color):
                line, = plt.plot(hp_pair[1], hp_pair[0], c=c, label=str(cont))
                lines.append(line)
                cont += 1
            plt.legend(handles=lines)
            plt.savefig(path + '/' + name + '_hp.png')
            plt.close()

        # Plot weighted Ripleys'H
        fig_count += 1
        plt.figure(fig_count)
        plt.title('Weighted Ripley\'s H')
        plt.xlabel('Radius (nm)')
        plt.ylabel('H')
        lines = list()
        color = cm.rainbow(np.linspace(0, 1, len(self.__groups_wh)))
        for (wh_pair, name, c) in zip(self.__groups_wh, self.__names, color):
            line, = plt.plot(wh_pair[1], wh_pair[0], c=c, label=name)
            lines.append(line)
        plt.legend(handles=lines)
        plt.savefig(path + '/wh.png')
        plt.close()

        # Plot weighted Ripleys'H
        fig_count += 1
        plt.figure(fig_count)
        plt.title('Weighted Ripley\'s H\'')
        plt.xlabel('Radius (nm)')
        plt.ylabel('H\'')
        lines = list()
        for (whp_pair, name, c) in zip(self.__groups_whp, self.__names, color):
            line, = plt.plot(whp_pair[1], whp_pair[0], c=c, label=name)
            lines.append(line)
        plt.legend(handles=lines)
        plt.savefig(path + '/whp.png')
        plt.close()

        # Plot cross-correlation for Ripley's H
        fig_count += 1
        plt.figure(fig_count)
        plt.title('Cross-correlation matrix for Ripley\'s H')
        plt.xlim(0, self.__cch.shape[0])
        plt.ylim(0, self.__cch.shape[1])
        plt.pcolor(self.__cch, cmap='jet', vmin=-1, vmax=1)
        plt.colorbar()
        plt.xticks(np.arange(.5, self.__cch.shape[0]+.5), self.__cc_lbls)
        plt.yticks(np.arange(.5, self.__cch.shape[1]+.5), self.__cc_lbls)
        plt.savefig(path + '/cch.png')
        plt.close()

        # Plot cross-correlation for Ripley's H'
        fig_count += 1
        plt.figure(fig_count)
        plt.title('Cross-correlation matrix for Ripley\'s H\'')
        plt.xlim(0, self.__cchp.shape[0])
        plt.ylim(0, self.__cchp.shape[1])
        plt.pcolor(self.__cchp, cmap='jet', vmin=-1, vmax=1)
        plt.colorbar()
        plt.xticks(np.arange(.5, self.__cchp.shape[0]+.5), self.__cc_lbls)
        plt.yticks(np.arange(.5, self.__cchp.shape[1]+.5), self.__cc_lbls)
        plt.savefig(path + '/cchp.png')
        plt.close()

        # Plot clouds
        if plt_cl:
            for (clouds, boxes, name) in zip(self.__groups_cloud, self.__groups_boxes, self.__names):
                figs_dir = path + '/' + name + '_clouds'
                if os.path.isdir(figs_dir):
                    shutil.rmtree(figs_dir)
                os.makedirs(figs_dir)
                cont = 1
                for (cloud, box) in zip(clouds, boxes):
                    fig_count += 1
                    plt.figure(fig_count)
                    plt.title('Clouds of points group ' + name + ' entry ' + str(cont))
                    plt.xlabel('X (nm)')
                    plt.ylabel('Y (nm)')
                    plt.axis('scaled')
                    plt.xlim(box[0], box[2])
                    plt.ylim(box[1], box[3])
                    plt.scatter(cloud[:, 0], cloud[:, 1])
                    plt.savefig(figs_dir + '/' + str(cont) + '.png')
                    plt.close()
                    cont += 1

    ##### Internal functionality area

    # Computes G function from a list of clouds
    # group: list of clouds
    # boxes: list of boxes
    # n: number of samples for cdf
    # Returns: averaged function g and samples
    def __group_function_G(self, group, n):

        # Computing Nearest Neighbour distances
        dists = list()
        for cloud in group:
            dists += nnde(cloud).tolist()

        # Computing Cumulative Probability Distribution
        return compute_cdf(np.asarray(dists, dtype=float), n)

    # Computes F function from a list of clouds
    # group: list of clouds
    # boxes: list of boxes
    # n: number of samples for cdf
    # m: number of random simulations
    # Returns: averaged function F and samples
    def __group_function_F(self, group, boxes, n, m):

        l_group = len(group)
        if m < l_group:
            error_msg = 'The number of simulations (' + str(m) + ') must be equal or greather than the length ' \
                        'of clouds (' + str(l_group) + ')'
            raise pexceptions.PySegInputError(expr='__group_function_F (GroupClouds)', msg=error_msg)

        # Computing Nearest Neighbour distances
        dists = list()
        for i in range(m):
            c_id = i % l_group
            dists += cnnde(gen_rand_cloud(group[c_id].shape[0], boxes[c_id]), group[c_id]).tolist()

        # Computing Cumulative Probability Distribution
        dists = np.asarray(dists, dtype=float)
        return compute_cdf(dists, n)

    # Simulates G-Function for a the random case with a number of simulations
    # group: list of clouds for reference
    # boxes: list of boxes
    # n: number of samples
    # m: number of random simulations
    # p: percentile for envelopes
    # Returns: samples, >p envelope, median, <100-p envelope
    def __rand_group_function_G(self, group, boxes, n, m, p):

        l_group = len(group)
        if m < l_group:
            error_msg = 'The number of simulations (' + str(m) + ') must be equal or greather than the length ' \
                        'of clouds (' + str(l_group) + ')'
            raise pexceptions.PySegInputError(expr='_rand_group_function_G (GroupClouds)', msg=error_msg)

        # Generate random points
        cont = 0
        cdfs = np.zeros(shape=(n, m), dtype=float)
        # Random simulation
        for i in range(m):
            c_id = i % l_group
            rand_dists = nnde(gen_rand_cloud(group[c_id].shape[0], boxes[c_id]))
            cdfs[:, cont], sp = compute_cdf(rand_dists, n)
            cont += 1

        # Compute envelopes
        env_1 = func_envelope(cdfs, per=p)
        env_2 = func_envelope(cdfs, per=50)
        env_3 = func_envelope(cdfs, per=100-p)

        return sp, env_1, env_2, env_3

    # Simulates F-Function for a the random case with a number of simulations
    # group: list of clouds for reference
    # boxes: list of boxes
    # n: number of samples
    # m: number of random simulations
    # p: percentile for envelopes
    # Returns: samples, >p envelope, median, <100-p envelope
    def __rand_group_function_F(self, group, boxes, n, m, p):

        l_group = len(group)
        if m < l_group:
            error_msg = 'The number of simulations (' + str(m) + ') must be equal or greather than the length ' \
                        'of clouds (' + str(l_group) + ')'
            raise pexceptions.PySegInputError(expr='_rand_group_function_G (GroupClouds)', msg=error_msg)

        # Generate random points
        cont = 0
        cdfs = np.zeros(shape=(n, m), dtype=float)
        # Random simulation
        for i in range(m):
            c_id = i % l_group
            rand_dists = cnnde(gen_rand_cloud(group[c_id].shape[0], boxes[c_id]),
                               gen_rand_cloud(group[c_id].shape[0], boxes[c_id]))
            cdfs[:, cont], sp = compute_cdf(rand_dists, n)
            cont += 1

        # Compute envelopes
        env_1 = func_envelope(cdfs, per=p)
        env_2 = func_envelope(cdfs, per=50)
        env_3 = func_envelope(cdfs, per=100-p)

        return sp, env_1, env_2, env_3

###########################################################################################
# Class for plotting overlapped clouds from different membrane slices
###########################################################################################

class GroupPlotter(object):

    def __init__(self, name):
        self.__name = name
        self.__clouds = list()
        self.__names = list()
        self.__markers = list()
        self.__box = None

    # Get/Set functionality

    # External functionality

    # clouds: input cloud
    # box: input box
    # name: name for the cloud
    # marker: if None (default) maker 'o' (circles) is inserted
    def insert_cloud(self, cloud, box, name, marker=None):

        # Check for valid box
        if self.__box is None:
            self.__box = box
        else:
            if box[0] < self.__box[0]:
                self.__box[0] = box[0]
            if box[1] < self.__box[1]:
                self.__box[1] = box[1]
            if box[2] > self.__box[0]:
                self.__box[2] = box[2]
            if box[3] > self.__box[3]:
                self.__box[3] = box[3]

        # Insert cloud
        self.__clouds.append(cloud)
        self.__names.append(name)
        if marker is None:
            self.__markers.append('o')
        else:
            self.__markers.append(marker)

    # Pickling the object state
    # fname: full path for the pickle file
    def pickle(self, fname):
        pkl_f = open(fname, 'w')
        try:
            pickle.dump(self, pkl_f)
        finally:
            pkl_f.close()

    # Plot figures
    # block: if True (default False) waits for closing windows for finishing the execution
    def plot(self, block=False):

        if len(self.__names) == 0:
            print('WARNING: no groups added, call insert_group() first!')

        # Initialization
        fig_count = 0
        if block:
            plt.ion()
        color = cm.rainbow(np.linspace(0, 1, len(self.__names)))

        # Plot with legend
        fig_count += 1
        plt.figure(fig_count)
        plt.title('Overlapped clouds for ' + self.__name +' (legend)')
        plt.xlabel('X (nm)')
        plt.ylabel('Y (nm)')
        plt.xlim(self.__box[0], self.__box[2])
        plt.ylim(self.__box[1], self.__box[3])
        lines = list()
        for (cloud, mark, c) in zip(self.__clouds, self.__markers, color):
            line = plt.scatter(cloud[:, 0], cloud[:, 1], c=c, marker=mark)
            lines.append(line)
        if len(lines) > 0:
            plt.legend(lines, self.__names)

        # Plot with legend
        fig_count += 1
        plt.figure(fig_count)
        plt.title('Overlapped clouds for ' + self.__name)
        plt.xlabel('X (nm)')
        plt.ylabel('Y (nm)')
        plt.xlim(self.__box[0], self.__box[2])
        plt.ylim(self.__box[1], self.__box[3])
        for (cloud, mark, c) in zip(self.__clouds, self.__markers, color):
            plt.scatter(cloud[:, 0], cloud[:, 1], c=c, marker=mark)

        # Show
        plt.show(block=block)

    # Stores figures
    # path: path to the folder where figures will be stored
    def store_figs(self, path):

        if len(self.__names) == 0:
            print('WARNING: no groups added, call insert_group() first!')

        # Initialization
        fig_count = 0
        color = cm.rainbow(np.linspace(0, 1, len(self.__names)))

        # Plot with legend
        fig_count += 1
        plt.figure(fig_count)
        plt.title('Overlapped clouds for ' + self.__name +' (legend)')
        plt.xlabel('X (nm)')
        plt.ylabel('Y (nm)')
        plt.xlim(self.__box[0], self.__box[2])
        plt.ylim(self.__box[1], self.__box[3])
        lines = list()
        for (cloud, mark, c) in zip(self.__clouds, self.__markers, color):
            line = plt.scatter(cloud[:, 0], cloud[:, 1], c=c, marker=mark)
            lines.append(line)
        if len(lines) > 0:
            plt.legend(lines, self.__names)
        plt.savefig(path + '/' + self.__name + '_ov_lg.png')
        plt.close()

        # Plot with legend
        fig_count += 1
        plt.figure(fig_count)
        plt.title('Overlapped clouds for ' + self.__name)
        plt.xlabel('X (nm)')
        plt.ylabel('Y (nm)')
        plt.xlim(self.__box[0], self.__box[2])
        plt.ylim(self.__box[1], self.__box[3])
        for (cloud, mark, c) in zip(self.__clouds, self.__markers, color):
            plt.scatter(cloud[:, 0], cloud[:, 1], c=c, marker=mark)
        plt.savefig(path + '/' + self.__name + '_ov.png')
        plt.close()