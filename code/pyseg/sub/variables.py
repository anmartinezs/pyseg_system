"""

    Global variables and classes for the package

"""

__author__ = 'martinez'

import numpy as np

STR_PEAK_ID = 'peak_id'
MS_CLST_LABELS = 'ms_clst_lables'

# Structs
PK_COORDS = 'pk_coords'

###########################################################################################
# Classes
###########################################################################################


#########################################################################################
# Class for the radial average of a 3D volume
#
class RadialAvg3D(object):

    # shape: subvolume shape (3-tuple)
    # axis: axis used for the radial average, valid: 'x', 'y' and 'z' (default)
    def  __init__(self, shape, axis='z'):

        # Input parsing
        if (not hasattr(shape, '__len__')) or (len(shape) != 3):
            print 'ERROR msg'
        if axis == 'x':
            self.__axis_i = 0
        elif axis == 'y':
            self.__axis_i = 1
        elif axis == 'z':
            self.__axis_i = 2
        else:
            print 'ERROR msg'

        # Precompute kernels for the radial averaging
        img_c = np.round(.5 * np.asarray(shape)).astype(np.int)
        self.__n_samp_h = shape[self.__axis_i]
        self.__n_samp_r = self.__n_samp_h - img_c[self.__axis_i]
        if axis == 'x':
            Y, Z = np.meshgrid(np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
            self.__kernels = np.zeros(shape=(shape[1], shape[2], self.__n_samp_r), dtype=np.float32)
        elif axis == 'y':
            X, Z = np.meshgrid(np.arange(shape[0]), np.arange(shape[2]), indexing='ij')
            self.__kernels = np.zeros(shape=(shape[0], shape[2], self.__n_samp_r), dtype=np.float32)
        else:
            X, Y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
            self.__kernels = np.zeros(shape=(shape[0], shape[1], self.__n_samp_r), dtype=np.float32)
        # General case
        for i in range(self.__n_samp_r):
            R = -1. * np.abs(np.sqrt((X - img_c[0]-0.5)**2 + (Y - img_c[1]-0.5)**2) - i)
            hold = R + 2
            hold_mask = hold < 0
            hold[hold_mask] = 0
            self.__kernels[:, :, i] = (1./2.) * hold

    #### External functionality area

    # Returns a subvolume with the kernels stack along the third axis for doing the averages
    def get_kernels(self):
        return self.__kernels

    # Return the dimensions of the output 2D image with the radial average
    def get_output_dim(self):
        return self.__n_samp_h, self.__n_samp_r

    # Returns a centered circular mask that can be used for the averaged particles
    # rad: radius in voxels
    def get_centered_mask(self, rad):
        X, Y = np.meshgrid(np.arange(self.__n_samp_h), np.arange(self.__n_samp_r), indexing='ij')
        R = (X-self.__n_samp_r-0.5)**2 + (Y-0.5)**2
        return R <= (rad*rad)

    def avg_vol(self, vol, mask=None, rg_h=None, rg_r=None):
        '''
        Average an input volume
        :param vol: 3D numpy array with the proper shape
        :param mask: mask with the valid region in vol (default None)
        :param rg_h: range for making focused computations in height (default None)
        :param rg_r: range for making focused computations in raidus (default None)
        :return: a 2D image with average
        '''

        # Input parsing
        if not isinstance(vol, np.ndarray):
            print 'ERROR msg'
        if mask is None:
            mask = np.ones(shape=vol.shape, dtype=vol.dtype)
        elif (vol.shape[0] != mask.shape[0]) or (vol.shape[1] != mask.shape[1]) or (vol.shape[2] != mask.shape[2]):
            print 'ERROR msg'
        if rg_h is None:
            if self.__axis_i == 0:
                ids_h = np.arange(vol.shape[0])
            elif self.__axis_i == 1:
                ids_h = np.arange(vol.shape[1])
            elif self.__axis_i == 2:
                ids_h = np.arange(vol.shape[2])
        else:
            h_l = rg_h[0]
            if h_l < 0:
                h_l = 0
            h_h = rg_h[1]
            if self.__axis_i == 0:
                if h_h >= vol.shape[0]:
                    h_h = vol.shape[0] - 1
            elif self.__axis_i == 1:
                if h_h >= vol.shape[1]:
                    h_h = vol.shape[1] - 1
            elif self.__axis_i == 2:
                if h_h >= vol.shape[2]:
                    h_h = vol.shape[2] - 1
            ids_h = np.arange(h_l, h_h+1)
        if rg_r is None:
            ids_r = np.arange(self.__kernels.shape[2])
        else:
            r_l = rg_r[0]
            if r_l < 0:
                r_l = 0
            r_h = rg_r[1]
            if r_h >= self.__kernels.shape[2]:
                r_h = self.__kernels.shape[2] - 1
            ids_r = np.arange(r_l, r_h+1)
        # Average
        avg = np.zeros(shape=self.get_output_dim(), dtype=np.float32)
        if self.__axis_i == 0:
            for i in ids_h:
                for j in ids_r:
                    hold = vol[i, :, :].reshape(vol.shape[1], vol.shape[2])
                    hold_mask = mask[i, :, :].reshape(vol.shape[1], vol.shape[2])
                    hold_mask *= self.__kernels[:, :, j]
                    hold_mask_sum = hold_mask.sum()
                    if hold_mask_sum > 0:
                        avg[i, j] = (hold * hold_mask) / hold_mask.sum()
        elif self.__axis_i == 1:
            for i in ids_h:
                for j in ids_r:
                    hold = vol[:, i, :].reshape(vol.shape[0], vol.shape[2])
                    hold_mask = mask[:, i, :].reshape(vol.shape[1], vol.shape[2])
                    hold_mask *= self.__kernels[:, :, j]
                    hold_mask_sum = hold_mask.sum()
                    if hold_mask_sum > 0:
                        avg[i, j] = (hold * hold_mask) / hold_mask.sum()
        elif self.__axis_i == 2:
            for i in ids_h:
                hold = vol[:, :, i]
                hold_mask = mask[:, :, i]
                for j in ids_r:
                    kernel = hold_mask * self.__kernels[:, :, j]
                    kernel_sum = kernel.sum()
                    if kernel_sum > 0:
                        avg[i, j] = (hold * kernel).sum() / kernel_sum

        return avg