"""
Classes for generating synthetic phantoms for validation

# Author: Antonio Martinez-Sanchez (Max Planck Institute for Biochemistry)
# Date: 04.03.15
"""

__author__ = 'martinez'

import pyseg.globals
# import numpy as np
# from pyseg.pexceptions import *
from abc import *
from pyseg.disperse_io import *
import vtk
from scipy import signal
import scipy as sp

####### Package global variables
# off_xy = 0.3
off_xy = 0.4
off_z = 1.5
pi2 = 2 * np.pi
STR_TO_SCALES = 'scales'
STR_GR_FDEG = 'feat_degree'

try:
    import pickle as pickle
except:
    import pickle

#####################################################################################################
#   Abstract class for being base class for all phantoms
#
#
class Phantom(object, metaclass=ABCMeta):

    # For Abstract Base Classes in python
    def __init__(self):
        self.__gtruth = None
        self.__tomo = None
        self.__mask = None
        self.__resolution = 1
        self.__snr = np.finfo(float).max

    #### Set/Get functions area

    # In nm/voxel, by default 1
    def set_resolution(self, resolution=1):
        self.__resolution = resolution

    def get_snr(self):
        return self.__snr

    #### Functionality area

    # Add pure random noise to the ground truth
    # dist: statistical distribution of the noise, valid: normal (default) and poisson
    # p: noise distribution parameter, for normal and poisson correspond with their standard
    # deviation
    def add_noise(self, dist='normal', p=1):

        if self.__tomo is None:
            error_msg = 'Ground truth has not been generated yet, call build() before.'
            raise pexceptions.PySegInputError(expr='add_noise (Phantom)', msg=error_msg)
        if p <= 0:
            error_msg = ' ''p'' must be >= than 0, given: ' + str(p)
            raise pexceptions.PySegInputError(expr='add_noise (Phantom)', msg=error_msg)

        # Creating the noise
        if dist == 'normal':
            noise = np.random.normal(0, p, size=self.__gtruth.shape)
        elif dist == 'poisson':
            noise = np.random.poisson(p, size=self.__gtruth.shape)
        else:
            error_msg = 'Non valid distribution type: ' + dist
            raise pexceptions.PySegInputError(expr='add_noise (Phantom)', msg=error_msg)

        # Update SNR measure
        self.__snr = self.__measure_snr(self.__tomo, noise, 'linear')

        #Adding the noise
        self.__tomo = self.__tomo + noise

        # Adjust density range
        self.__tomo = lin_map(self.__tomo, lb=0, ub=1)

    # Add missing wedge to distortion
    # In this version only tilt axes perpendicular to XY are considered
    # wr_ang: wedge rotation angle in degrees [-90, 90]
    # tilt_ang: maximum tilt angle in degrees [0, 90]
    def add_mw(self, tilt_ang, wr_ang=0):
        self.__tomo = add_mw(tomo=self.__tomo, wr_ang=wr_ang, tilt_ang=tilt_ang)

    # Store Ground truth
    # file_name: full path and file name, the ending determines the format
    def save_gtruth(self, file_name):
        if self.__gtruth is None:
            error_msg = 'Ground truth has not been generated yet, call build() before.'
            raise pexceptions.PySegInputError(expr='save_gtruth (Phantom)', msg=error_msg)
        disperse_io.save_numpy(self.__gtruth.transpose(), file_name)

    # Store the synthetic tomogram
    # file_name: full path and file name, the ending determines the format
    def save_tomo(self, file_name):
        if self.__tomo is None:
            error_msg = 'The tomogram has not been generated yet, call build() before.'
            raise pexceptions.PySegInputError(expr='save_tomo (Phantom)', msg=error_msg)
        disperse_io.save_numpy(self.__tomo.transpose(), file_name)

    # Store the mask for DisPerSe (0-fg and 1-bg)
    # file_name: full path and file name, the ending determines the format
    def save_mask(self, file_name):
        if self.__mask is None:
            error_msg = 'The mask has not been generated yet, call build() before.'
            raise pexceptions.PySegInputError(expr='save_mask (Phantom)', msg=error_msg)
        disperse_io.save_numpy(self.__mask.transpose(), file_name)

    # After the call to this method the tomogram will be created
    @abstractmethod
    def build(self):
        raise NotImplementedError('build() (Phantom). '
                                  'Abstract method, it requires an implementation.')

    # Save feature points in VTK format
    @abstractmethod
    def save_vtp(self):
        raise NotImplementedError('build() (Phantom). '
                                  'Abstract method, it requires an implementation.')

    # For pickling the phantom
    @abstractmethod
    def pickle(self, out_pkl):
        raise NotImplementedError('build() (Phantom). '
                                  'Abstract method, it requires an implementation.')

    ######### Internal methods area

    # By default measures signal to noise ratio (does not use if missing wedge has been applied)
    # according to 'R.C. Gonzalez and R.E. Woods, "Digital Image Processing," Third Edition, Prentice Hall2008. ISBN: 013168728'
    # If signal (before noise) and noise data are passed then the SNR is computed with
    # the ratio of their variances
    # Returns: SNR linear (default) or dB if mode not equal to 'linear'
    def __measure_snr(self, signal=None, noise=None, mode='linear'):

        if (signal is None) or (noise is None):
            num = np.sum(self.__tomo * self.__tomo)
            dem = np.sum(np.square(self.__gtruth - self.__tomo))
        else:
            num = np.var(signal)
            dem = np.var(noise)

        if dem != 0:
            ratio = num / dem
        else:
            ratio = np.finfo(float).max

        if mode == 'linear':
            return ratio
        else:
            return 10 * math.log10(ratio)


#####################################################################################################
#   Abstract class for creating a torus made up by randomly distributed multiscale features
#
#
class Torus(Phantom, metaclass=ABCMeta):

    # For Abstract Base Classes in python
    def __init__(self):
        super(Torus, self).__init__()
        self.__R = 50
        self.__r = 5
        self.__n_feat = 50
        self.__rand_f = 0.2
        self.__range_s = (2, 6)
        self.__points = None
        self.__scales = None

    #### Set/Get functions area

    # Set input configuration parameter for this phantom
    # R: nominal circle radius in nm (default 50)
    # r: nominal tube radius in nm (default 5)
    # n_feat: number of features (default 50)
    # rand_f: strength factor for randomly distorting nominal tube radius [0,1] (default 0.2)
    # range_s: range for scale the features (default [2, 6])
    def set_parameters(self, R=50, r=5, n_feat=50, rand_f=0.2, range_s=(2, 6)):
        self.__R = R
        self.__r = r
        self.__n_feat = n_feat
        self.__rand_f = rand_f
        self.__range_s = range_s

    def get_num_features(self):
        return self.__points.shape[0]

    #### Functionality area

    # After the call to this method the tomogram will be created
    def build(self):

        # Initialization
        dim_xy = np.ceil((self.__R+self.__r+self.__range_s[1]) * (1+off_xy))
        dim_z = np.ceil((self.__r+self.__range_s[1]) * (1+off_z))
        (X, Y, Z) = np.meshgrid(np.arange(-dim_xy, dim_xy, 1),
                                np.arange(-dim_xy, dim_xy, 1),
                                np.arange(-dim_z, dim_z, 1))
        t = pi2 * np.random.rand(self.__n_feat)
        cos_t = np.cos(t)
        sin_t = np.sin(t)
        ff = pi2 / self.__n_feat
        p = np.linspace(0, pi2-ff, self.__n_feat)
        cos_p = np.cos(p)
        sin_p = np.sin(p)
        self.__scales = (self.__range_s[1]-self.__range_s[0]) * np.random.rand(self.__n_feat) \
                        + self.__range_s[0]
        # c = math.pow(pi2, 2.0/3.0)
        self._Phantom__gtruth = np.zeros(shape=X.shape, dtype=np.float32)

        # Building the mask
        self._Phantom__mask = np.ones(shape=X.shape, dtype=int)
        offset_xy = dim_xy - (self.__R+self.__r+self.__range_s[1])
        offset_xy = int(math.ceil(offset_xy * 0.25))
        if offset_xy < 1:
            offset_xy = int(1)
        offset_z = dim_z - (self.__r+self.__range_s[1])
        offset_z = int(math.ceil(offset_z * 0.25))
        if offset_z < 1:
            offset_z = int(1)
        self._Phantom__mask[offset_xy:X.shape[0]-offset_xy, offset_xy:X.shape[1]-offset_xy,
                            offset_z:X.shape[2]-offset_z] = 0

        # Feature coordinates (by using Torus equation)
        self.__points = np.zeros(shape=(self.__n_feat, 3), dtype=float)
        self.__points[:, 0] = (self.__R + self.__r*cos_t) * cos_p
        self.__points[:, 1] = (self.__R + self.__r*cos_t) * sin_p
        self.__points[:, 2] = self.__r * sin_t

        # Loop for adding the features in a randomly distorted torus surface
        for i in range(self.__n_feat):
            s = self.__scales[i]
            Xh = X - self.__points[i, 0]
            Yh = Y - self.__points[i, 1]
            Zh = Z - self.__points[i, 2]
            Gh = (-1) * (Xh*Xh + Yh*Yh + Zh*Zh)
            # G = (1.0/(math.pow(s, 3)*c)) * np.exp(Gh * (1.0/(2*math.pow(s, 2))))
            G = np.exp(Gh * (1.0/(2*math.pow(s, 2))))
            self._Phantom__gtruth += G

        # X and Y coordinates must be permuted
        hold = np.copy(self.__points)
        self.__points[:, 0] = hold[:, 1]
        self.__points[:, 1] = hold[:, 0]

        # Invert to create the density map
        self._Phantom__gtruth = lin_map(self._Phantom__gtruth, lb=1, ub=0)
        # disperse_io.save_numpy(self._Phantom__gtruth, './out/hold.mrc')

        self._Phantom__tomo = self._Phantom__gtruth


    # Check the number of features correctly localized
    # points: input coordinates ([n_points, 3]) for comparing with the ground truth
    # eps: greater distance between an input point and the closest ground truth point will be
    # consider a true negative, if there is no ground truth feature within this distance then
    # is consider false positive. In nm (default 1 nm)
    # Returns: two array making which ground truth points are true negatives and which input
    # points are false positives
    def check_feat_localization(self, points, eps=1):

        if self.__points is None:
            error_msg = 'Ground truth has not been generated yet, call build() before.'
            raise pexceptions.PySegInputError(expr='check_feat_localization (Phantom)',
                                              msg=error_msg)
        true_negatives = np.zeros(shape=self.__n_feat, dtype=bool)
        false_positives = np.zeros(shape=points.shape[0], dtype=bool)
        # If no features
        if (self.__points.shape[0]) <= 0 or (points.shape[0] <= 0):
            return true_negatives, false_positives

        # Correcting the offset in the input coordinates
        corr_points = np.zeros(shape=points.shape, dtype=points.dtype)
        corr_points[:, 0] = points[:, 0] - self._Phantom__tomo.shape[0]*0.5
        corr_points[:, 1] = points[:, 1] - self._Phantom__tomo.shape[1]*0.5
        corr_points[:, 2] = points[:, 2] - self._Phantom__tomo.shape[2]*0.5

        # True negatives loop
        for i in range(self.__n_feat):
            dists = self.__points[i, :] - corr_points
            dists = np.sqrt(np.sum(dists*dists, axis=1)) * self._Phantom__resolution
            if dists.min() > eps:
                true_negatives[i] = True

        # False positives loop
        for i in range(corr_points.shape[0]):
            dists = corr_points[i, :] - self.__points
            dists = np.sqrt(np.sum(dists*dists, axis=1)) * self._Phantom__resolution
            if dists.min() > eps:
                false_positives[i] = True

        return true_negatives, false_positives

    # Store a set of points with the features center and the scale as attributes
    # file_name: full path and file name, the ending determines the format (.vtp)
    def save_vtp(self, file_name):

        if self.__points is None:
            error_msg = 'The mask has not been generated yet, call build() before.'
            raise pexceptions.PySegInputError(expr='save_vtp (Torus)', msg=error_msg)

        # Intialization
        poly = vtk.vtkPolyData()
        pts = vtk.vtkPoints()
        verts = vtk.vtkCellArray()
        scales = vtk.vtkFloatArray()
        scales.SetNumberOfComponents(1)
        scales.SetName(STR_TO_SCALES)

        # Geometry, topology and attributes
        nx2 = self._Phantom__tomo.shape[0] * 0.5
        ny2 = self._Phantom__tomo.shape[1] * 0.5
        nz2 = self._Phantom__tomo.shape[2] * 0.5
        for i in range(self.__points.shape[0]):
            x = self.__points[i, 0] + nx2
            y = self.__points[i, 1] + ny2
            z = self.__points[i, 2] + nz2
            pts.InsertPoint(i, x, y, z)
            verts.InsertNextCell(1)
            verts.InsertCellPoint(i)
            scales.InsertTuple1(i, self.__scales[i])

        # Build the poly
        poly.SetPoints(pts)
        poly.SetVerts(verts)
        poly.GetCellData().AddArray(scales)

        # Store the poly
        disperse_io.save_vtp(poly, file_name)


    # fname: file name ended with .pkl
    def pickle(self, fname):

        pkl_f = open(fname, 'w')
        try:
            pickle.dump(self, pkl_f)
        finally:
            pkl_f.close()

#####################################################################################################
#   Abstract class for creating a 3D grid with multiscale features
#
#
class Grid3D(Phantom, metaclass=ABCMeta):

    # For Abstract Base Classes in python
    def __init__(self):
        super(Grid3D, self).__init__()
        self.__L = (4, 4, 3)
        self.__spacing = 10
        self.__thick = (1, 3)
        self.__points = None
        self.__degree = None
        self.__skel = None
        self.__ev_ratio = 1. / 3.

    #### Set/Get functions area

    # Set input configuration parameter for this phantom
    # L: 3-tuple with the number of rows at every dimension
    # sp: spacing between rows
    # thick: 2-tuple with thickness range for the rows
    # ve_ratio: density (inverted) ratio between edges and vertices, it must be in range (0, 1)
    def set_parameters(self, L=(4, 4, 3), sp=10, thick=(1, 3), ev_ratio=1./3.):
        self.__L = L
        self.__spacing = sp
        self.__thick = thick
        if (ev_ratio <= 0) or (ev_ratio >= 1):
            raise ValueError
        self.__ev_ratio = float(ev_ratio)

    def get_num_features(self):
        return self.__points.shape[0] * self.__points.shape[1] * self.__points.shape[2]

    def get_shape(self):
        return self._Phantom__gtruth.shape

    # TODO: equation estimated by me for the number of edges in 3D regular degree
    def get_num_edges(self):
        m, n, p = self.__L
        return (p-1)*m*n + p*(n*(m-1) + m*(n-1))

    #### Functionality area

    # After the call to this method the tomogram will be created
    def build(self):

        # Initialization
        npx = self.__spacing*(self.__L[0]-1) + self.__thick[1]
        npy = self.__spacing*(self.__L[1]-1) + self.__thick[1]
        npz = self.__spacing*(self.__L[2]-1) + self.__thick[1]
        off = np.max((npx, npy, npz)) * off_xy
        if off < 10:
            off = 10
        nx, ny, nz = int(math.ceil(npx + 2*off)), int(math.ceil(npy + 2*off)), \
                     int(math.ceil(npz + 2*off))
        self._Phantom__gtruth = np.zeros(shape=(nx, ny, nz), dtype=float)
        self.__skel = np.zeros(shape=(nx, ny, nz), dtype=bool)
        offx, offy, offz = off_xy * npx, off_xy * npy, off_xy * npz
        self.__points = np.empty(shape=(self.__L[0], self.__L[1], self.__L[2]),
                                 dtype=object)
        self.__degree = np.zeros(shape=self.__points.shape, dtype=int)
        # n_feat = self.__points.shape[0] * self.__points.shape[1] * self.__points.shape[2]
        rx_arr = np.linspace(self.__thick[0], self.__thick[1], self.__L[0])
        ry_arr = np.linspace(self.__thick[0], self.__thick[1], self.__L[1])

        # Creating the feature points
        # sp2 = self.__spacing * off_xy
        # sp2 = self.__spacing * 0.25
        sp2 = 0
        for rx in range(self.__L[0]):
            for ry in range(self.__L[1]):
                for rz in range(self.__L[2]):
                    hold = np.zeros(shape=3, dtype=float)
                    hold[0] = off + sp2 + rx*self.__spacing
                    hold[1] = off + sp2 + ry*self.__spacing
                    hold[2] = off + sp2 + rz*self.__spacing
                    self.__points[rx, ry, rz] = hold

        # self.__add_line_skel(self.__points[0, 0, :], rx_arr[0], axis=2)
        # Adding vertical (Z-axis) lines which crosses feature points for creating the grid
        for rx in range(self.__L[0]):
            for ry in range(self.__L[1]):
                row = self.__points[rx, ry, :]
                # self.__add_line_skel(row, rx_arr[rx], axis=2)
                self.__add_line_skel_2(row, rx_arr[rx], axis=2)
                # self.__add_line_skel_3(row, rx_arr[rx], axis=2)
                self.__degree[rx, ry, :] += 1

        # Adding horizontal (Y-axis) lines which crosses feature points for creating the grid
        for rx in range(self.__L[0]):
            for rz in range(self.__L[2]):
                row = self.__points[rx, :, rz]
                # self.__add_line_skel(row, rx_arr[rx], axis=1)
                self.__add_line_skel_2(row, rx_arr[rx], axis=1)
                # self.__add_line_skel_3(row, rx_arr[rx], axis=1)
                self.__degree[rx, :, rz] += 1

        # Adding horizontal (X-axis) lines which crosses feature points for creating the grid
        for ry in range(self.__L[1]):
            for rz in range(self.__L[2]):
                row = self.__points[:, ry, rz]
                # self.__add_line_skel(row, ry_arr[ry], axis=0)
                self.__add_line_skel_2(row, ry_arr[ry], axis=0)
                # self.__add_line_skel_3(row, ry_arr[ry], axis=0)
                self.__degree[:, ry, rz] += 1

        # Update the degree
        for i in range(1, self.__L[0]-1):
            self.__degree[i, :, :] += 1
        for i in range(1, self.__L[1]-1):
            self.__degree[:, i, :] += 1
        for i in range(1, self.__L[2]-1):
            self.__degree[:, :, i] += 1

        # TODO: for testing
        # self._Phantom__gtruth[self._Phantom__gtruth > 0] = 1


        # Building the mask
        self._Phantom__mask = np.ones(shape=self._Phantom__gtruth.shape, dtype=int)
        off_x = int(math.floor(nx - npx) * 0.15)
        if off_x < 1:
            off_x = int(1)
        off_y = int(math.floor(ny - npy) * 0.15)
        if off_y < 1:
            off_y = int(1)
        off_z = int(math.floor(nz - npz) * 0.15)
        if off_z < 1:
            off_z = int(1)
        self._Phantom__mask[off_x:nx-off_x, off_y:ny-off_y, off_z:nz-off_z] = 0

        # Applyting edges/vertices ratio
        self._Phantom__gtruth[self._Phantom__gtruth==1] = self.__ev_ratio
        self._Phantom__gtruth[self._Phantom__gtruth==3] = 1

        self._Phantom__gtruth = lin_map(self._Phantom__gtruth, lb=1, ub=0)
        self._Phantom__tomo = self._Phantom__gtruth

    # Store a set of points with the features center and the scale as attributes
    # file_name: full path and file name, the ending determines the format (.vtp)
    def save_vtp(self, file_name):

        if self.__points is None:
            error_msg = 'The mask has not been generated yet, call build() before.'
            raise pexceptions.PySegInputError(expr='save_vtp (Torus)', msg=error_msg)

        # Initialization
        poly = vtk.vtkPolyData()
        pts = vtk.vtkPoints()
        verts = vtk.vtkCellArray()
        degree = vtk.vtkIntArray()
        degree.SetNumberOfComponents(1)
        degree.SetName(STR_GR_FDEG)

        # Geometry, topology and attributes
        count = 0
        for rx in range(self.__points.shape[0]):
            for ry in range(self.__points.shape[1]):
                for rz in range(self.__points.shape[2]):
                    point = self.__points[rx, ry, rz]
                    pts.InsertPoint(count, point[0], point[1], point[2])
                    verts.InsertNextCell(1)
                    verts.InsertCellPoint(count)
                    degree.InsertTuple1(count, self.__degree[rx, ry, rz])
                    count += 1

        # Build the poly
        poly.SetPoints(pts)
        poly.SetVerts(verts)
        poly.GetCellData().AddArray(degree)

        # Store the poly
        disperse_io.save_vtp(poly, file_name)

    # fname: file name ended with .pkl
    def pickle(self, fname):

        pkl_f = open(fname, 'wb')
        try:
            pickle.dump(self, pkl_f)
        finally:
            pkl_f.close()

    # Hard test which checks if features are correctly localized and have the correct degree
    # points: array of coordinates ([n_points, 3]) for comparing with the ground truth
    # degrees: array with degrees ([n_points]) for comparing with the ground truth
    # eps: greater distance between an input point and the closest ground truth point will be
    # consider a true negative, if there is no ground truth feature within this distance then
    # is consider false positive. In nm (default 1 nm)
    # Returns: three values, fraction of true negatives, fraction of true positives and
    # the error degree fraction. None if something was not correct
    def hard_test(self, points, degrees, eps=1):

        if self.__points is None:
            error_msg = 'Ground truth has not been generated yet, call build() before.'
            raise pexceptions.PySegInputError(expr='hard_test (Grid3D)', msg=error_msg)

        n_feat = self.get_num_features()
        true_negatives = 0
        false_positives = 0
        error_degree = 0
        # If no features
        if (n_feat <= 0) or (points.shape[0] <= 0) or (points.shape[0] != degrees.shape[0]):
            return None

        # Reshape feature points array
        count = 0
        feat_points = np.zeros(shape=(n_feat, 3), dtype=float)
        feat_deg = np.zeros(shape=n_feat, dtype=int)
        for x in range(self.__points.shape[0]):
            for y in range(self.__points.shape[1]):
                for z in range(self.__points.shape[2]):
                    feat_points[count, :] = self.__points[x, y, z]
                    feat_deg[count] = self.__degree[x, y, z]
                    count += 1

        # True negatives loop
        for i in range(n_feat):
            dists = feat_points[i, :] - points
            dists = np.sqrt(np.sum(dists*dists, axis=1)) * self._Phantom__resolution
            amin = dists.argmin()
            if dists[amin] > eps:
                true_negatives += 1
            # if degrees[amin] != feat_deg[i]:
            elif degrees[amin] != feat_deg[i]:
                error_degree += 1

        # False positives loop
        for i in range(points.shape[0]):
            dists = points[i, :] - feat_points
            dists = np.sqrt(np.sum(dists*dists, axis=1)) * self._Phantom__resolution
            if dists.min() > eps:
                false_positives += 1

        return float(true_negatives) / n_feat, \
               float(false_positives) / points.shape[0], \
               float(error_degree) / n_feat

    # Carry out a soft test by measuring the average distances from an input skeleton and the
    # grid skeleton
    # skel: input skeleton as vtkPolyData
    # Return: two values, the average distance of input points to ground truth skeleton and
    # the average distance to ground truth skeleton to input skeleton points
    def soft_test(self, skel):

        # Create a tomogram with the input skeleton points
        tomo_in = np.zeros(shape=self.__skel.shape, dtype=bool)
        for i in range(skel.GetNumberOfPoints()):
            x, y, z = skel.GetPoint(i)
            tomo_in[int(round(x)), int(round(y)), int(round(z))] = True

        # Distance transform
        d_in = sp.ndimage.morphology.distance_transform_edt(tomo_in)
        d_sk = sp.ndimage.morphology.distance_transform_edt(self.__skel)

        # Measuring average distances
        av1 = 0
        idx = np.where(tomo_in)
        if len(idx) < 3:
            return None
        for i in range(len(idx[0])):
            av1 += d_sk[idx[0][i], idx[1][i], idx[2][i]]


        av1 /= (len(idx[0]) + len(idx[1]) + len(idx[2]))
        av2 = 0
        idx = np.where(self.__skel)
        for i in range(len(idx[0])):
            av2 += d_in[idx[0][i], idx[1][i], idx[2][i]]
        av2 /= (len(idx[0]) + len(idx[1]) + len(idx[2]))

        return av1, av2

    # Returns three arrays with grid points coordinates X, Y, S
    def get_grid_points(self):

        gxs = np.zeros(shape=self.__L, dtype=np.float32)
        gys = np.zeros(shape=self.__L, dtype=np.float32)
        gzs = np.zeros(shape=self.__L, dtype=np.float32)
        for x in range(self.__L[0]):
            for y in range(self.__L[1]):
                for z in range(self.__L[2]):
                    gxs[x, y, z], gys[x, y, z], gzs[x, y, z] = self.__points[x, y, z]

        return gxs, gys, gzs

    # point: points coordinates to eps
    # eps: maximum distance to feature center
    # Returns True if the coordinate specified is within the feature
    def in_feature(self, point, eps):
        for i in range(self.__L[0]):
            for j in range(self.__L[1]):
                for k in range(self.__L[2]):
                    hold_point = self.__points[i, j, k]
                    dst = hold_point - point
                    dst = math.sqrt((dst*dst).sum())
                    if dst <= eps:
                        return True
        return False

    # point: points coordinates to eps
    # eps: maximum distance to feature center
    # Returns True if the coordinate specified is within the feature
    def in_grid(self, point):
        int_point = np.round(point).astype(int)
        try:
            if self._Phantom__gtruth[int_point[0], int_point[1], int_point[2]] != 1:
                return True
        except IndexError:
            pass
        return False


    ####### Area for internal functionality

    # row: array with the row points
    # r: thickness radius
    # axis: axis (1-X, 2-Y, otherwise-Z)
    def __add_line_skel(self, row, r, axis):

        # Initialization
        nx, ny, nz = self._Phantom__gtruth.shape
        nx2, ny2, nz2 = nx * 0.5, ny * 0.5, nz * 0.5
        hold = np.zeros(shape=(nx, ny, nz), dtype=self._Phantom__gtruth.dtype)
        p1, p2 = np.round(row[0]), np.round(row[-1])
        p1 = p1.astype(int)
        p2 = p2.astype(int)

        # Line skeleton
        if axis == 0:
            hold[p1[0]:(p2[0]+1), p1[1], p1[2]] = 1
        elif axis == 1:
            hold[p1[0], p1[1]:(p2[1]+1), p1[2]] = 1
        else:
            hold[p1[0], p1[1], p1[2]:(p2[2]+1)] = 1
        self.__skel[hold == 1] = True

        # Giving thickness to the line
        X, Y, Z = np.meshgrid(np.linspace(-nx2, nx2, nx), np.linspace(-ny2, ny2, ny),
                              np.linspace(-nz2, nz2, nz))
        G = np.exp((-1) * (X*X + Y*Y + Z*Z) * (1.0/(2*math.pow(r, 2))))
        self._Phantom__gtruth += signal.fftconvolve(hold, G, 'same')

    # row: array with the row points
    # r: thickness radius
    # axis: axis (1-X, 2-Y, otherwise-Z)
    def __add_line_skel_2(self, row, r, axis):

        # Initialization
        nx, ny, nz = self._Phantom__gtruth.shape
        hold = np.zeros(shape=(nx, ny, nz), dtype=self._Phantom__gtruth.dtype)
        p1, p2 = np.round(row[0]), np.round(row[-1])
        p1 = p1.astype(int)
        p2 = p2.astype(int)
        thick = int(math.ceil(r))

        # Line skeleton
        if axis == 0:
            hold[(p1[0]-thick):(p2[0]+thick), (p1[1]-thick):(p1[1]+thick),
                 (p1[2]-thick):(p1[2]+thick)] = 1
        elif axis == 1:
            hold[(p1[0]-thick):(p1[0]+thick), (p1[1]-thick):(p2[1]+thick),
                 (p1[2]-thick):(p1[2]+thick)] = 1
        else:
            hold[(p1[0]-thick):(p1[0]+thick), (p1[1]-thick):(p1[1]+thick),
                 (p1[2]-thick):(p2[2]+thick)] = 1
        self.__skel[hold == 1] = True

        # Adding the line to the ground truth
        self._Phantom__gtruth += hold

    # For generating odd lines
    # row: array with the row points
    # r: thickness radius
    # axis: axis (1-X, 2-Y, otherwise-Z)
    def __add_line_skel_3(self, row, r, axis):

        # Initialization
        nx, ny, nz = self._Phantom__gtruth.shape
        hold = np.zeros(shape=(nx, ny, nz), dtype=self._Phantom__gtruth.dtype)
        p1, p2 = np.round(row[0]), np.round(row[-1])
        p1 = p1.astype(int)
        p2 = p2.astype(int)
        # thick = int(math.ceil(r))
        thick = 1

        # Line skeleton
        if axis == 0:
            hold[(p1[0]-thick):(p2[0]+thick-1), (p1[1]-thick):(p1[1]+thick-1),
                 (p1[2]-thick):(p1[2]+thick-1)] = 1
        elif axis == 1:
            hold[(p1[0]-thick):(p1[0]+thick-1), (p1[1]-thick):(p2[1]+thick-1),
                 (p1[2]-thick):(p1[2]+thick-1)] = 1
        else:
            hold[(p1[0]-thick):(p1[0]+thick-1), (p1[1]-thick):(p1[1]+thick-1),
                 (p1[2]-thick):(p2[2]+thick-1)] = 1
        self.__skel[hold == 1] = True

        # Adding the line to the ground truth
        self._Phantom__gtruth += hold