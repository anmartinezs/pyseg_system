"""
Classes with additional functionality for applying the spatial analysis on stacks of images

# Author: Antonio Martinez-Sanchez (Max Planck Institute for Biochemistry)
# Date: 12.04.16
"""

__author__ = 'martinez'

import os
# import cv2
import math
import numpy as np
import pyseg as ps
from pyseg.spatial.sparse import PlotUni, UniStat
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from .variables import *

try:
    import pickle as pickle
except:
    import pickle

##############################################################################################
# Global variables
#

PST_LBL = 2
MSK_LBL = 1

##############################################################################################
# Helper functions
#


# Reduce tomo size in axis
# tomo: input 3D tomogram with the stack
# mask: if not None (default) the input mask is also purged
# axis: stack axis, X-0, Y-1 and Z-2 (default)
# ratio: x:1 ratio of deletion
# Returns: a 3-tuple with tomo and mask (None if no input mask) purged, and the spacing array
def purge_stack(tomo, mask=None, axis=1, ratio=1):

    # Initialization
    if (not isinstance(tomo, np.ndarray)) or (len(tomo.shape) != 3):
        error_msg = 'Input tomogram must be a ndarray with tree dimensions.'
        raise ps.pexceptions.PySegInputError(expr='purge_stack', msg=error_msg)
    if mask is not None:
        if (not isinstance(mask, np.ndarray)) or (tomo.shape != mask.shape):
            error_msg = 'Input tomogram and mask must be the same class and have the same dimensions.'
            raise ps.pexceptions.PySegInputError(expr='purge_stack', msg=error_msg)
    if ratio <= 1:
        return tomo, mask, np.ones(shape=tomo.shape[axis], dtype=float)

    # Index array
    ids = np.arange(0, tomo.shape[axis], step=ratio)
    p = len(ids)
    ids_p = np.zeros(shape=p, dtype=float)

    # Purging loop
    if axis == 0:
        m, n = tomo.shape[1], tomo.shape[2]
        stack = np.zeros(shape=(m, n, p), dtype=tomo.dtype)
        if mask is not None:
            stack_m = np.zeros(shape=(m, n, p), dtype=tomo.dtype)
        for i, idx in enumerate(ids):
            ids_p[i] = idx
            stack[:, :, i] = tomo[idx, :, :].reshape(m, n)
            if mask is None:
                stack_m[:, :, i] = mask[idx, :, :].reshape(m, n)
    if axis == 1:
        m, n = tomo.shape[0], tomo.shape[2]
        stack = np.zeros(shape=(m, n, p), dtype=tomo.dtype)
        if mask is not None:
            stack_m = np.zeros(shape=(m, n, p), dtype=tomo.dtype)
        for i, idx in enumerate(ids):
            ids_p[i] = idx
            stack[:, :, i] = tomo[:, idx, :].reshape(m, n)
            if mask is None:
                stack_m[:, :, i] = mask[:, idx, :].reshape(m, n)
    else:
        m, n = tomo.shape[0], tomo.shape[1]
        stack = np.zeros(shape=(m, n, p), dtype=tomo.dtype)
        if mask is not None:
            stack_m = np.zeros(shape=(m, n, p), dtype=tomo.dtype)
        for i, idx in enumerate(ids):
            ids_p[i] = idx
            stack[:, :, i] = tomo[:, :, idx].reshape(m, n)
            if mask is None:
                stack_m[:, :, i] = mask[:, :, idx].reshape(m, n)

    return stack, stack_m, ids_p

##############################################################################################
# Class for univariate statistical spatial analysis o the stack contained by a tomogram
# (here it is seen as set of 2D images stacked along a specified axis)
#
class TomoUni(object):

    ##### Constructor area

    # IT CAN RAISE A PySegInputWarning IF NO POINT FOUND IN THE STACK
    # tomo: input tomogram (3D array) where 0-bg and >0-fg
    # mask: input mask, tomogram with the same size as tomo
    # res: voxel resolution (nm/pixel) (default 1)
    # spacing: distance among images in the stack, it can be an array with an entry for every image
    #          in the stack, or real when the space is regular, if None (default) it is set equal to 'res'
    # axis: axis for staking, 0-X, 1-Y and 2-Z (default)
    # area_min: minimum and maximum areas for consider an object in nm (default [5, 20])
    # ec: maximum eccentricity for consider an object (default .5)
    # pre: if True (default False) input tomogram is preprocessed with contour analysis
    # name: string with the name of the stack (default None)
    def __init__(self, tomo, mask=None, res=1., spacing=None, axis=2, name=None, area=(4, 20), ec=.9, pre=False):

        # Parsing axis
        if axis == 0:
            self.__axis = axis
        elif axis == 1:
            self.__axis = axis
        elif axis == 2:
            self.__axis = axis
        else:
            error_msg = 'Value ' + str(axis) + ' is not a valid axis (0-X, 1-Y and 2-Z).'
            raise ps.pexceptions.PySegInputError(expr='__init__ (TomoUni)', msg=error_msg)

        # Parsing tomo and mask
        if (not isinstance(tomo, np.ndarray)) or (len(tomo.shape) != 3):
            error_msg = 'Input tomogram must be a ndarray with tree dimensions.'
            raise ps.pexceptions.PySegInputError(expr='__init__ (TomoUni)', msg=error_msg)
        if mask is None:
            mask = np.ones(shape=tomo.shape, dtype=bool)
        elif (not isinstance(mask, np.ndarray)) or (mask.shape[0] != mask.shape[0]) or \
                (mask.shape[1] != mask.shape[1]) or (mask.shape[2] != mask.shape[2]):
            error_msg = 'Input mas must have the same dimension of the tomogram.'
            raise ps.pexceptions.PySegInputError(expr='__init__ (TomoUni)', msg=error_msg)
        self.__stack, self.__stack_m = list(), list()
        if self.__axis == 0:
            m, n = tomo.shape[1], tomo.shape[2]
            for i in range(tomo.shape[0]):
                self.__stack.append((tomo[i, :, :].reshape(m, n) > 0).astype(np.uint8))
                self.__stack_m.append(mask[i, :, :].reshape(m, n) > 0)
        if self.__axis == 1:
            m, n = tomo.shape[0], tomo.shape[2]
            for i in range(tomo.shape[1]):
                self.__stack.append((tomo[:, i, :].reshape(m, n) > 0).astype(np.uint8))
                self.__stack_m.append(mask[:, i, :].reshape(m, n) > 0)
        else:
            m, n = tomo.shape[0], tomo.shape[1]
            for i in range(tomo.shape[2]):
                self.__stack.append((tomo[:, :, i] > 0).astype(np.uint8))
                self.__stack_m.append(mask[:, :, i] > 0)
        self.__m, self.__n = m, n

        # Parsing res an spacing
        self.__res = float(res)
        if spacing is None:
            self.__spacing = np.arange(0, (len(self.__stack)-1)*self.__res, self.__res)
        elif isinstance(spacing, np.ndarray):
            if len(spacing) != len(self.__stack):
                error_msg = 'Input spacing is an array but does not fit the stack size.'
                raise ps.pexceptions.PySegInputError(expr='__init__ (TomoUni)', msg=error_msg)
            else:
                self.__spacing = spacing
        else:
            hold_sp = float(spacing)
            self.__spacing = np.arange(0, (len(self.__stack)-1)*hold_sp, hold_sp)
        self.__spacing = list(self.__spacing)

        # Parsing shape thresholds
        if (not hasattr(area, '__len__')) or len(area) != 2:
            error_msg = 'Input thresholds for area must be 2-tuple.'
            raise ps.pexceptions.PySegInputError(expr='__init__ (TomoUni)', msg=error_msg)
        self.__area = tuple(area)
        self.__ec = float(ec)

        # Parsing name
        self.__name = str(name)

        # Find coordinates
        self.__stack_c = list()
        self.__find_coords(pre)

    ###### Set/Get functionality

    # For updating mask
    def set_mask(self, mask):
        if (not isinstance(mask, np.ndarray)) or (len(mask.shape) != 3):
            error_msg = 'Input mask must be a 3D numpy array.'
            raise ps.pexceptions.PySegInputError(expr='set_mask (TomoUni)', msg=error_msg)
        m_m, m_n, m_z = mask.shape[0], mask.shape[1], mask.shape[2]
        if (m_m != self.__m) or (m_n != self.__n) or (m_z != len(self.__stack_m)):
            error_msg = 'Input mask must fit stack dimensions (' \
                        + str(self.__m) + ', ' + str(self.__n) + ', ' + str(len(self.__stack_m)) + ')'
            raise ps.pexceptions.PySegInputError(expr='set_mask (TomoUni)', msg=error_msg)
        self.__stack_m = list()
        for i in range(m_z):
            self.__stack_m.append(mask[:, :, i] > 0)

    # Creates a 3D numpy array for holding the mask
    def get_mask_stack(self):
        mask = np.zeros(shape=(self.__m, self.__n, len(self.__stack_m)), dtype=bool)
        for i in range(mask.shape[2]):
            mask[:, :, i] = self.__stack_m[i]
        return mask

    # Retruns an array or coordinates in 3D coodinates (Z axis is included)
    def get_coords_3d(self):
        coords = list()
        for i, s_coords in enumerate(self.__stack_c):
            for coord in s_coords:
                coords.append(np.asarray((coord[0], coord[1], i), dtype=float))
        return np.asarray(coords)

    ###### External functionality

    # Build a PlotUni object from the images and coordinates patter of the stack
    def generate_PlotUni(self):
        puni = StackPlotUni(name=self.__name)
        for (coords, mask, st) in zip(self.__stack_c, self.__stack_m, self.__spacing):
            uni = UniStat(coords, mask, self.__res, name=str(round(st, 3))+'nm')
            puni.insert_uni(uni, step=st)
        return puni

    # Generates a tomogram which represents the stack (0-bg, MSK_LBL-fg, 2-PTS_LBL)
    def generate_tomo_stack(self):

        # Initialization
        if (len(self.__stack) == 0) or (self.__m <= 0) or (self.__n <=0):
            error_msg = 'The stack contains no images.'
            raise ps.pexceptions.PySegInputError(expr='generate_tomo_stack (TomoUni)', msg=error_msg)
        tomo = np.zeros(shape=(self.__m, self.__n, len(self.__stack)), dtype=self.__stack[0].dtype)

        # Main loop
        for (i, img, mask, coords) in zip(list(range(len(self.__stack))), self.__stack, self.__stack_m, self.__stack_c):
            hold_img = np.zeros(shape=(self.__m, self.__n), dtype=tomo.dtype)
            img_ids = (img * mask) > 0
            hold_img[img_ids] = 1
            for coord in coords:
                px, py = int(coord[0]), int(coord[1])
                try:
                    if mask[px, py]:
                        hold_img[px, py] = 2
                except IndexError:
                    pass
            tomo[:, :, i] = hold_img

        return np.abs(tomo)

    # Generate homogeneity stack
    # freq: frequency for the low pass filter
    # mode_3d: if default the homogeneity test is carried out in 3D, otherwise in 2D
    def homo_stack(self, freq, mode_3d=True):

        # Getting stack with points
        tomo = self.generate_tomo_stack()
        tomo = (tomo == 2).astype(float)

        if mode_3d:
            homo = ps.globals.tomo_butter_low(tomo, freq, n=5)
        else:
            homo = np.zeros(shape=tomo.shape, dtype=float)
            for z in range(tomo.shape[2]):
                hold_tomo = np.zeros(shape=(tomo.shape[0], tomo.shape[1], 1), dtype=float)
                hold_tomo[:, :, 0] = tomo[:, :, z]
                flt = ps.globals.tomo_butter_low(hold_tomo, freq, n=5)
                # ps.disperse_io.save_numpy(flt, '/home/martinez/workspace/disperse/data/marion/cluster/out/flt.mrc')
                homo[:,:, z] = flt[:, :, 0]

        return homo

    # Returns a list with the number of points per image in the stack
    def list_n_points(self):
        pts = list()
        for coords in self.__stack_c:
            pts.append(len(coords))
        return pts

    ###### Internal functionality

    def __find_coords(self, pre=False):

        # Initialization
        hold_stack, hold_spacing, hold_masks = self.__stack, self.__spacing, self.__stack_m
        self.__stack, self.__spacing, self.__stack_m = list(), list(), list()
        res2 = self.__res * self.__res

        # Loop for finding coordinates (discard empty images)
        for (img, sp, mask) in zip(hold_stack, hold_spacing, hold_masks):
            hold_st_c = list()
            if pre:
                contours, _ = cv2.findContours(img, cv2.cv.CV_RETR_LIST, cv2.cv.CV_CHAIN_APPROX_NONE)
                for cnt in contours:
                    M = cv2.moments(cnt)
                    m00, m01, m10, m11, m02, m20 = M['m00'], M['m01'], M['m10'], M['m11'], M['m02'], M['m20']
                    if m00 == 0:
                        continue
                    area = m00 * res2
                    cx, cy = m01/m00, m10/m00
                    # Filter by area
                    if (area >= self.__area[0]) and (area <= self.__area[1]):
                        # Filter by eccentricity
                        if self.__eccentricy(m00, m01, m10, m11, m02, m20) < self.__ec:

                            try:
                                if mask[int(cx), int(cy)]:
                                    hold_st_c.append(np.asarray((cx, cy)))
                            except IndexError:
                                pass
            else:
                ids = np.where(img > 0)
                for (idx, idy) in zip(ids[0], ids[1]):
                    hold_st_c.append(np.asarray((idx, idy), dtype=np.float32))
            if len(hold_st_c) > 0:
                self.__stack.append(img)
                self.__stack_c.append(np.asarray(hold_st_c, dtype=int))
                self.__spacing.append(sp)
                self.__stack_m.append(mask)

        # Raise a warning in no img survived to the purging
        if len(self.__stack) == 0:
            error_msg = 'No point found in the stack.'
            raise ps.pexceptions.PySegInputWarning(expr='__find_coords (TomoUni)', msg=error_msg)

    # Compute eccentricity from shape moment
    def __eccentricy(self, m00, m01, m10, m11, m02, m20):

        # Second order central moments (covariance matrix)
        cx, cy = m10/m00, m01/m00
        mp11 = m11/m00 - cx*cy
        mp20 = m20/m00 - cx*cx
        mp02 = m02/m00 - cy*cy

        # Eigenvalues
        hold_a = .5 * (mp20 + mp02)
        hold_c = mp20 - mp02
        hold_b = .5 * math.sqrt(4*mp11*mp11 + hold_c*hold_c)
        l1, l2 = math.fabs(hold_a+hold_b), math.fabs(hold_a-hold_b)

        # Eccentricity computation
        if l1 > l2:
            return math.sqrt(1. - l2/l1)
        else:
            if l2 == .0:
                return 1.
            else:
                return math.sqrt(1. - l1/l2)

##############################################################################################
# Class for extending PlotUni functionality for dealing with Stacks
#
class StackPlotUni(PlotUni):

    # Analyze (plot a surface o image in a figure) function G
    # max_d: maximum distance for computing the histograms in nm
    # bins: number of samples for output (default 50)
    # update: if True (default False) re-computing data if forced
    # block: if True (default) the code stops until plotted window is closed
    # out_file: path to file to store the analysis, if None (default) is not stored
    # bar: if True (default False) and out_file is not None, a colorbar is also stored
    # mode: if '3d' (default) the result are ploted in a surface, otherwise in a filled contour
    def analyze_stack_G(self, max_d, bins=50, update=False, block=True, out_file=None, bar=False, mode='3d'):

        # Computations
        if update or (self._PlotUni__Gs is None):
            self._PlotUni__Gs = None
            self._PlotUni__compute_Gs(max_d, bins)
        cmap = cm.jet

        if self.not_eq_intensities():
            print('WARNING: Function G is not a proper metric for comparing patterns with different ' \
                  'intensities')

        # Preparing data
        X, Y, Z = self.__gen_surface(self._PlotUni__Gs, self._PlotUni__zs, off=0)

        # Plotting
        fig = plt.figure()
        plt.title('G-Function')
        if mode == '3d':
            ax = fig.gca(projection='3d')
            im = ax.plot_surface(X, Y, Z, cmap=cmap)
            ax.set_zlabel('G')
            ax.set_zlim(0, 1)
        else:
            ax = fig.gca()
            im = ax.contourf(X, Y, Z, cmap=cmap)
        ax.set_ylabel('Dst (nm)')
        ax.set_xlabel('Stk (nm)')

        # Show data
        plt.show(block=block)
        if out_file is not None:
            plt.savefig(out_file)

        # Store the colorbar
        if bar and (out_file is not None):
            self.__store_colorbar(im, ax, out_file)

        plt.close()

    # Analyze (plot a surface o image in a figure) function K
    # max_d: maximum distance for computing the histograms in nm
    # bins: number of samples for output (default 50)
    # update: if True (default False) re-computing data if forced
    # block: if True (default) the code stops until plotted window is closed
    # out_file: path to file to store the analysis, if None (default) is not stored
    # bar: if True (default False) and out_file is not None, a colorbar is also stored
    # mode: if '3d' (default) the result are ploted in a surface, otherwise in a filled contour
    def analyze_stack_K(self, max_d, bins=50, update=False, block=True, out_file=None, bar=False, mode='3d'):

        # Computations
        if update or (self._PlotUni__Ks is None):
            self._PlotUni__Ks = None
            self._PlotUni__compute_Ks(max_d, bins)
        cmap = cm.jet

        # Preparing data
        X, Y, Z = self.__gen_surface(self._PlotUni__Ks, self._PlotUni__zs, off=0)

        # Plotting
        fig = plt.figure()
        plt.title('K-Function')
        if mode == '3d':
            ax = fig.gca(projection='3d')
            im = ax.plot_surface(X, Y, Z, cmap=cmap)
            ax.set_zlabel('K')
        else:
            ax = fig.gca()
            im = ax.contourf(X, Y, Z, cmap=cmap)
        ax.set_ylabel('Dst (nm)')
        ax.set_xlabel('Stk (nm)')

        # Show data
        plt.show(block=block)
        if out_file is not None:
            plt.savefig(out_file)

        # Store the colorbar
        if bar and (out_file is not None):
            self.__store_colorbar(im, ax, out_file)

        plt.close()


    # Analyze (plot a surface o image in a figure) function L (requires the previous computation of K,
    # see analyze_K() or analyze_stack_K())
    # block: if True (default) the code stops until plotted window is closed
    # out_file: path to file to store the analysis, if None (default) is not stored
    # bar: if True (default False) and out_file is not None, a colorbar is also stored
    # mode: if '3d' (default) the result are ploted in a surface, otherwise in a filled contour
    def analyze_stack_L(self, block=True, out_file=None, bar=False, mode='3d'):

        # Computations
        if self._PlotUni__Ks is None:
            error_msg = 'analyze_K() or analyze_stack_K() must be called before with the same input parameters.'
            raise ps.pexceptions.PySegInputError(expr='analyze_stack_J (StackPlotUni)', msg=error_msg)
        cmap = cm.jet

        # Compute Ls for the full stack
        dsts, Ls = list(), list()
        for (uni, r, K) in zip(self._PlotUni__unis, self._PlotUni__Ks[0], self._PlotUni__Ks[1]):
            L = self._PlotUni__compute_L(r, K, uni.is_2D())
            dsts.append(r)
            Ls.append(L)
        Ls = (dsts, Ls)

        # Preparing data
        X, Y, Z = self.__gen_surface(Ls, self._PlotUni__zs, off=0)

        # Plotting
        fig = plt.figure()
        plt.title('L-Function')
        if mode == '3d':
            ax = fig.gca(projection='3d')
            im = ax.plot_surface(X, Y, Z, cmap=cmap)
            ax.set_zlabel('O')
            ax.set_ylabel('Dst (nm)')
            ax.set_xlabel('Stk (nm)')
        elif mode == 'img':
            ax = fig.gca()
            z_res = float(self._PlotUni__zs[-1]-self._PlotUni__zs[0])/float(len(self._PlotUni__zs))
            o_res = float(r[-1]-r[0])/float(len(r))
            img_r = cv2.resize(Z, (int(round(Z.shape[1]*o_res)), int(round(Z.shape[0]*z_res))),
                               interpolation=cv2.INTER_CUBIC)
            im = ax.imshow(img_r, cmap=cmap)
            ax.set_xlabel('Dst (nm)')
            ax.set_ylabel('Stk (nm)')
        else:
            ax = fig.gca()
            im = ax.contourf(X, Y, Z, cmap=cmap)

        # Show data
        plt.show(block=block)
        if out_file is not None:
            plt.savefig(out_file)

        # Store the colorbar
        if bar and (out_file is not None):
            self.__store_colorbar(im, ax, out_file)

        plt.close()

    # Analyze (plot a surface o image in a figure) function O (requires the previous computation of K,
    # see analyze_K() or analyze_stack_K())
    # w: ring width (default 1)
    # block: if True (default) the code stops until plotted window is closed
    # out_file: path to file to store the analysis, if None (default) is not stored
    # bar: if True (default False) and out_file is not None, a colorbar is also stored
    # mode: if '3d' (default) the result are plotted in a surface, else if 'img' and image with fitted resolution
    #       and otherwise a contourmap
    def analyze_stack_O(self, w=1, block=True, out_file=None, bar=False, mode='3d'):

        # Computations
        if self._PlotUni__Ks is None:
            error_msg = 'analyze_K() or analyze_stack_K() must be called before with the same input parameters.'
            raise ps.pexceptions.PySegInputError(expr='analyze_stack_J (StackPlotUni)', msg=error_msg)
        cmap = cm.jet

        # Compute Ls for the full stack
        dsts, Os = list(), list()
        for (uni, r, K) in zip(self._PlotUni__unis, self._PlotUni__Ks[0], self._PlotUni__Ks[1]):
            O = self._PlotUni__compute_O(r, K, w, uni.is_2D())
            dsts.append(r)
            Os.append(O)
        Os = (dsts, Os)

        # Preparing data
        X, Y, Z = self.__gen_surface(Os, self._PlotUni__zs, off=0)

        # Plotting
        fig = plt.figure()
        plt.title('O-Function')
        if mode == '3d':
            ax = fig.gca(projection='3d')
            im = ax.plot_surface(X, Y, Z, cmap=cmap)
            ax.set_zlabel('O')
            ax.set_ylabel('Dst (nm)')
            ax.set_xlabel('Stk (nm)')
        elif mode == 'img':
            ax = fig.gca()
            z_res = float(self._PlotUni__zs[-1]-self._PlotUni__zs[0])/float(len(self._PlotUni__zs))
            o_res = float(r[-1]-r[0])/float(len(r))
            img_r = cv2.resize(Z, (int(round(Z.shape[1]*o_res)), int(round(Z.shape[0]*z_res))),
                               interpolation=cv2.INTER_CUBIC)
            im = ax.imshow(img_r, cmap=cmap)
            ax.set_xlabel('Dst (nm)')
            ax.set_ylabel('Stk (nm)')
        else:
            ax = fig.gca()
            im = ax.contourf(X, Y, Z, cmap=cmap)

        # Show data
        plt.show(block=block)
        if out_file is not None:
            plt.savefig(out_file)

        # Store the colorbar
        if bar and (out_file is not None):
            self.__store_colorbar(im, ax, out_file)

        plt.close()


    ##### Internal functionality area

    def __gen_surface(self, dat, y_arr, off=0):
        x_arr = dat[0][0]
        Y, X = np.meshgrid(x_arr[off:], y_arr)
        Z = np.zeros(shape=(X.shape[0], X.shape[1]), dtype=x_arr.dtype)
        for i, d in enumerate(dat[1]):
            Z[i, :] = d[off:]
        return X, Y, Z

    def __store_colorbar(self, im, ax, out_file):
        plt.colorbar(im, ax=ax)
        stem, ext = os.path.splitext(out_file)
        plt.savefig(stem + '_colorbar' + ext)
        plt.close()