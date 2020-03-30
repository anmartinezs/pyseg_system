'''
Classes for representing tomograms with segmentations

'''


import os
import copy
import numpy as np
import scipy as sp
from shutil import rmtree
from utils import *
from pyorg import pexceptions, sub, disperse_io
from pyorg.globals.utils import unpickle_obj
from pyorg import globals as gl
try:
    import cPickle as pickle
except:
    import pickle

__author__ = 'Antonio Martinez-Sanchez'

##### Global variables


# GLOBAL FUNCTIONS

# Clean an directory contents (directory is preserved)
# dir: directory path
def clean_dir(dir):
    for root, dirs, files in os.walk(dir):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            rmtree(os.path.join(root, d))

# PARALLEL PROCESSES


# CLASSES

############################################################################
# Class for a Segmentation: set of voxel in a tomogram
#
class Segmentation(object):

    def __init__(self, tomo, lbl):
        """
        :param tomo: tomogram which contains the segmentation
        :param lbl: label for the segmentation
        """

        # Input parsing
        self.__ids = np.where(tomo == lbl)
        self.__vcount = len(self.__ids[0])
        assert self.__vcount > 0
        self.__lbl = lbl

        # Pre-compute bounds for accelerate computations
        self.__bounds = np.zeros(shape=6, dtype=np.float32)
        self.__update_bounds()

    #### Set/Get functionality

    def get_ids(self):
        return self.__ids

    def get_label(self):
        return self.__lbl

    def get_voxels_count(self):
        """
        :return: the number of voxel in the segmentation
        """
        return self.__vcount

    def get_bounds(self):
        """
        :return: surface bounds (x_min, x_max, y_min, y_max, z_min, z_max) as array
        """
        return self.__bounds

    #### External functionality

    def bound_in_bounds(self, bounds):
        """
        Check if the object's bound are at least partially in another bound
        :param bounds: input bound
        :return:
        """
        x_over, y_over, z_over = True, True, True
        if (self.__bounds[0] > bounds[1]) or (self.__bounds[1] < bounds[0]):
            x_over = False
        if (self.__bounds[2] > bounds[3]) or (self.__bounds[3] < bounds[2]):
            y_over = False
        if (self.__bounds[4] > bounds[5]) or (self.__bounds[5] < bounds[4]):
            y_over = False
        return x_over and y_over and z_over

    def point_in_bounds(self, point):
        """
        Check if a point within filament's bounds
        :param point: point to check
        :return:
        """
        x_over, y_over, z_over = True, True, True
        if (self.__bounds[0] > point[0]) or (self.__bounds[1] < point[0]):
            x_over = False
        if (self.__bounds[2] > point[1]) or (self.__bounds[3] < point[1]):
            y_over = False
        if (self.__bounds[4] > point[2]) or (self.__bounds[5] < point[2]):
            y_over = False
        return x_over and y_over and z_over

    def pickle(self, fname):
        """
        VTK attributes requires a special treatment during pickling
        :param fname: file name ended with .pkl
        :return:
        """

        # Dump pickable objects and store the file names of the unpickable objects
        stem, ext = os.path.splitext(fname)
        self.__vtp_fname = stem + '_curve.vtp'
        pkl_f = open(fname, 'w')
        try:
            pickle.dump(self, pkl_f)
        finally:
            pkl_f.close()

    # INTERNAL FUNCTIONALITY AREA

    def __update_bounds(self):
        self.__bounds[0], self.__bounds[1] = self.__ids[0].min(), self.__ids[0].max()
        self.__bounds[2], self.__bounds[3] = self.__ids[1].min(), self.__ids[1].max()
        self.__bounds[4], self.__bounds[5] = self.__ids[2].min(), self.__ids[2].max()


############################################################################
# Class for a OMSegmentation: oriented membrane segmentation.
# Contains two segmentations, one with the membrane the other with the lumen
#
class OMSegmentation(object):

    def __init__(self, tomo_mb, tomo_lm, lbl):
        """
        :param tomo_mb: tomogram with the membrane (None is allowed)
        :param tomo_lm: tomogram with the lumen
        :param lbl:
        """

        # Input parsing
        self.__ids_lm = np.where(tomo_lm == lbl)
        self.__vcount_lm = len(self.__ids_lm[0])
        self.__lbl = lbl
        assert self.__vcount_lm > 0
        if tomo_mb is None:
            self.__ids_mb = self.__ids_lm
            self.__vcount_mb = self.__vcount_lm
        else:
            self.__ids_mb = np.where(tomo_mb == lbl)
            self.__vcount_mb = len(self.__ids_mb[0])
        assert self.__vcount_mb > 0

        # Pre-compute bounds for accelerate computations
        self.__bounds = np.zeros(shape=6, dtype=np.float32)
        self.__update_bounds()

    #### Set/Get functionality

    def get_label(self):
        """
        :return: an integer label
        """
        return self.__lbl

    def get_ids(self, mode='lm'):
        """
        :param mode: 'mb' or 'lumen' ids
        :return: segmented voxel indices an array with 4 dimension (N,X,Y,Z)
        """
        assert (mode == 'mb') or (mode == 'lm')
        if mode == 'mb':
            return self.__ids_mb
        elif mode == 'lm':
            return self.__ids_lm

    def get_voxels_count(self, mode='mb'):
        """
        :param mode: to count 'mb' or 'lumen' voxels
        :return: the number of voxel in the segmentation
        """
        assert (mode == 'mb') or (mode == 'lm')
        if mode == 'mb':
            return self.__vcount_mb
        elif mode == 'lm':
            return self.__vcount_lm

    def get_bounds(self):
        """
        :return: surface bounds (x_min, x_max, y_min, y_max, z_min, z_max) as array
        """
        return self.__bounds

    #### External functionality

    def bound_in_bounds(self, bounds):
        """
        Check if the object's bound are at least partially in another bound
        :param bounds: input bound
        :return:
        """
        hold_bounds = self.__bounds
        x_over, y_over, z_over = True, True, True
        if (hold_bounds[0] > bounds[1]) or (hold_bounds[1] < bounds[0]):
            x_over = False
        if (hold_bounds[2] > bounds[3]) or (hold_bounds[3] < bounds[2]):
            y_over = False
        if (hold_bounds[4] > bounds[5]) or (hold_bounds[5] < bounds[4]):
            y_over = False
        return x_over and y_over and z_over

    def point_in_bounds(self, point):
        """
        Check if a point within filament's bounds
        :param point: point to check
        :return:
        """
        hold_bounds = self.__bounds
        x_over, y_over, z_over = True, True, True
        if (hold_bounds[0] > point[0]) or (hold_bounds[1] < point[0]):
            x_over = False
        if (hold_bounds[2] > point[1]) or (hold_bounds[3] < point[1]):
            y_over = False
        if (hold_bounds[4] > point[2]) or (hold_bounds[5] < point[2]):
            y_over = False
        return x_over and y_over and z_over

    def pickle(self, fname):
        """
        VTK attributes requires a special treatment during pickling
        :param fname: file name ended with .pkl
        :return:
        """

        # Dump pickable objects and store the file names of the unpickable objects
        stem, ext = os.path.splitext(fname)
        self.__vtp_fname = stem + '_curve.vtp'
        pkl_f = open(fname, 'w')
        try:
            pickle.dump(self, pkl_f)
        finally:
            pkl_f.close()

    # INTERNAL FUNCTIONALITY AREA

    def __update_bounds(self):
        bounds_mb, bounds_lm = np.zeros(shape=6, dtype=np.float32), np.zeros(shape=6, dtype=np.float32)
        bounds_mb[0], bounds_mb[1] = self.__ids_mb[0].min(), self.__ids_mb[0].max()
        bounds_mb[2], bounds_mb[3] = self.__ids_mb[1].min(), self.__ids_mb[1].max()
        bounds_mb[4], bounds_mb[5] = self.__ids_mb[2].min(), self.__ids_mb[2].max()
        bounds_lm[0], bounds_lm[1] = self.__ids_lm[0].min(), self.__ids_lm[0].max()
        bounds_lm[2], bounds_lm[3] = self.__ids_lm[1].min(), self.__ids_lm[1].max()
        bounds_lm[4], bounds_lm[5] = self.__ids_lm[2].min(), self.__ids_lm[2].max()
        if bounds_mb[0] < bounds_lm[0]:
            self.__bounds[0] = bounds_mb[0]
        else:
            self.__bounds[0] = bounds_lm[0]
        if bounds_mb[1] < bounds_lm[1]:
            self.__bounds[1] = bounds_mb[1]
        else:
            self.__bounds[1] = bounds_lm[1]
        if bounds_mb[2] < bounds_lm[2]:
            self.__bounds[2] = bounds_mb[2]
        else:
            self.__bounds[2] = bounds_lm[2]
        if bounds_mb[3] < bounds_lm[3]:
            self.__bounds[3] = bounds_mb[3]
        else:
            self.__bounds[3] = bounds_lm[3]
        if bounds_mb[4] < bounds_lm[4]:
            self.__bounds[4] = bounds_mb[4]
        else:
            self.__bounds[4] = bounds_lm[4]
        if bounds_mb[5] < bounds_lm[5]:
            self.__bounds[5] = bounds_mb[5]
        else:
            self.__bounds[5] = bounds_lm[5]


############################################################################
# Class for tomograms with oriented membrane segmentations
#
class TomoOMSegmentations(object):

    def __init__(self, name, voi_mb=None, voi_lm=None, max_dst=0):
        """
        :param name: name to identify the tomogram
        :param voi_mb: if None (default) the membrane tomogram is loaded from tomo_fname, otherwise this is actually
        the input tomogram
        :param voi_lm: if None (default) the lumen tomogram is loaded from tomo_fname, otherwise this is actually
        the input tomogram
        :param max_dst: maximum distance to lumen border for membrane segmentation (in segmentation pixels)
        """

        # Input parsing
        if not isinstance(name, str):
            error_msg = 'Input is not a string.'
            raise pexceptions.PySegInputError(expr='__init__ (TomoOMSegmentations)', msg=error_msg)
        if (voi_mb is not None) and (not isinstance(voi_mb, np.ndarray)):
            error_msg = 'Input VOI for membranes must be an numpy.ndarray.'
            raise pexceptions.PySegInputError(expr='__init__ (TomoOMSegmentations)', msg=error_msg)
        if (voi_lm is not None) and (not isinstance(voi_lm, np.ndarray)):
            error_msg = 'Input VOI for lumen must be an numpy.ndarray.'
            raise pexceptions.PySegInputError(expr='__init__ (TomoOMSegmentations)', msg=error_msg)
        self.__name = name
        self.__segs = list()

        # Create the lumen's label field
        if voi_mb.shape != voi_lm.shape:
            error_msg = 'Input tomograms for membranes and lumen must have the same sizes.'
            raise pexceptions.PySegInputError(expr='__init__ (TomoOMSegmentations)', msg=error_msg)
        self.__lbl_voi_lm, nlbls = sp.ndimage.label(voi_lm, structure=np.ones(shape=(3, 3, 3)))
        lbls_lm = range(1, nlbls+1)
        # disperse_io.save_numpy(self.__lbl_voi_lm, '/fs/pool/pool-ruben/antonio/filaments/ltomos_omsegs/test/hold_lm.mrc')
        hold_lm = sp.ndimage.morphology.binary_dilation(voi_lm > 0)
        dst_field_lm, dst_ids_lm = sp.ndimage.morphology.distance_transform_edt(hold_lm, return_distances=True,
                                                                                return_indices=True)
        # hold_lm = sp.ndimage.morphology.binary_dilation(voi_lm == 0)
        # dst_field_inv_lm, dst_ids_inv_lm = sp.ndimage.morphology.distance_transform_edt(hold_lm, return_distances=True,
        #                                                                                return_indices=True)

        # Set lumen labels to membrane segmentation
        mb_ids = np.where(voi_mb)
        self.__lbl_voi_mb = np.zeros(shape=voi_mb.shape, dtype=np.int32)
        for x, y, z in zip(mb_ids[0], mb_ids[1], mb_ids[2]):
            hold_dst = dst_field_lm[x, y, z]
            if (hold_dst > 0) and (hold_dst <= max_dst):
                x_idx, y_idx, z_idx = dst_ids_lm[:, x, y, z]
                x_l = x_idx - 2
                if x_l <= 0:
                    x_l = 0
                x_h = x_idx + 3
                if x_h >= self.__lbl_voi_mb.shape[0]:
                    x_h = self.__lbl_voi_mb.shape[0]
                y_l = y_idx - 2
                if y_l <= 0:
                    y_l = 0
                y_h = y_idx + 3
                if y_h >= self.__lbl_voi_mb.shape[1]:
                    y_h = self.__lbl_voi_mb.shape[1]
                z_l = z_idx - 2
                if z_l <= 0:
                    z_l = 0
                z_h = z_idx + 3
                if z_h >= self.__lbl_voi_mb.shape[2]:
                    z_h = self.__lbl_voi_mb.shape[2]
                hold_lbls_lm = self.__lbl_voi_lm[x_l:x_h, y_l:y_h, z_l:z_h]
                try:
                    hold_lbl_lm = np.argmax(np.bincount(hold_lbls_lm[hold_lbls_lm > 0]))
                    self.__lbl_voi_mb[x, y, z] = hold_lbl_lm
                except ValueError:
                    pass # print 'jol 1'
            # else:
            #     hold_dst_inv = dst_field_inv_lm[x, y, z]
            #     if (hold_dst_inv > 0) and (hold_dst_inv <= max_dst):
            #         x_idx, y_idx, z_idx = dst_ids_inv_lm[:, x, y, z]
            #         x_l = x_idx - 2
            #         if x_l <= 0:
            #             x_l = 0
            #         x_h = x_idx + 3
            #         if x_h >= self.__lbl_voi_mb.shape[0]:
            #             x_h = self.__lbl_voi_mb.shape[0]
            #         y_l = y_idx - 2
            #         if y_l <= 0:
            #             y_l = 0
            #         y_h = y_idx + 3
            #         if y_h >= self.__lbl_voi_mb.shape[1]:
            #             y_h = self.__lbl_voi_mb.shape[1]
            #         z_l = z_idx - 2
            #         if z_l <= 0:
            #             z_l = 0
            #         z_h = z_idx + 3
            #         if z_h >= self.__lbl_voi_mb.shape[2]:
            #             z_h = self.__lbl_voi_mb.shape[2]
            #         hold_lbls_lm = self.__lbl_voi_lm[x_l:x_h, y_l:y_h, z_l:z_h]
            #         try:
            #             hold_lbl_lm = np.argmax(np.bincount(hold_lbls_lm[hold_lbls_lm > 0]))
            #             self.__lbl_voi_mb[x, y, z] = hold_lbl_lm
            #         except ValueError:
            #             pass # print 'Jol 2'
            #     else:
            #         pass # print 'Jol 3'

        # Create the segmentations
        for lbl_lm in lbls_lm:
            try:
                self.__segs.append(OMSegmentation(self.__lbl_voi_mb, self.__lbl_voi_lm, lbl_lm))
            except AssertionError:
                continue

    # GET/SET AREA

    def get_voi(self, mode='mb'):
        """
        Get the tomograms with the segmentations VOI
        :param mode: 'mb' membrane, 'lm' lumen, 'mb-lm' membrane and lumen fused
        :return: a binary ndarray
        """
        if mode == 'mb':
            return self.__lbl_voi_mb > 0
        elif mode == 'lm':
            return self.__lbl_voi_lm > 0
        elif mode == 'mb-lm':
            return (self.__voi_mb + self.__voi_lm) > 0
        else:
            error_msg = 'Input mode not valid: ' + str(mode)
            raise pexceptions.PySegInputError(expr='get_voi (TomoOMSegmentations)', msg=error_msg)

    def get_lbl_voi(self, mode='mb'):
        """
        Get the labeled tomograms with the segmentations
        :param mode: 'mb' membrane, 'lm' lumen
        :return: an ndarray with the segmentations labeled
        """
        if mode == 'mb':
            return self.__lbl_voi_mb
        elif mode == 'lm':
            return self.__lbl_voi_lm
        else:
            error_msg = 'Input mode not valid: ' + str(mode)
            raise pexceptions.PySegInputError(expr='get_voi (TomoOMSegmentations)', msg=error_msg)

    def get_tomo_name(self):
        return self.__name

    def get_segmentations(self):
        return self.__segs

    def get_num_segmentations(self):
        return len(self.__segs)

    # EXTERNAL FUNCTIONALITY AREA

    def delete_segmentation(self, seg_ids):
        """
        Remove  segmentations from the list
        :param seg_ids: integer ids of the segmentations (their position in the current list of segmentations)
        :return: None
        """
        # Loop for keeping survivors
        hold_segs = list()
        for i in range(len(self.__segs)):
            if not i in seg_ids:
                hold_segs.append(self.__segs[i])

        # Updating the list of segmentations
        self.__segs = hold_segs

    def pickle(self, fname):
        """
        :param fname: file name ended with .pkl
        :return:
        """
        pkl_f = open(fname, 'w')
        try:
            pickle.dump(self, pkl_f)
        finally:
            pkl_f.close()

    def compute_voi_volume(self, mode='mb'):
        """
        Compute voi volume in voxels
        :param mode: 'mb' membrane, 'lm' lumen, 'mb-lm' membrane and lumen fused
        :return:
        """
        if mode == 'mb':
            return self.__voi_mb.sum()
        elif mode == 'lm':
            return self.__voi_lm.sum()
        elif mode == 'mb-lm':
            return self.__voi_mb.sum() + self.__voi_lm.sum()
        else:
            error_msg = 'Input mode not valid: ' + str(mode)
            raise pexceptions.PySegInputError(expr='compute_voi_volume (TomoOMSegmentations)', msg=error_msg)

    def compute_om_seg_dsts(self):
        """
        Computes the distance among the different oriented membrane segmentations
        :return: a 3D array (tomogram segmemntation) where each voxels encodes the distance to the closes membrane
                 segmentation, background pixels are set to zero.
        """

        # Initialization
        dsts_field = np.zeros(shape=self.__lbl_voi_lm.shape, dtype=np.float32)

        # Loop for segmentations
        for seg in self.__segs:
            hold_lbl_voi = (self.__lbl_voi_lm == seg.get_label()) + (self.__lbl_voi_lm == 0)
            hold_dsts = sp.ndimage.morphology.distance_transform_edt(hold_lbl_voi, return_distances=True,
                                                                     return_indices=False)
            mb_ids = self.__lbl_voi_mb == seg.get_label()
            dsts_field[mb_ids] = hold_dsts[mb_ids]

        return dsts_field

    # INTERNAL FUNCTIONALITY AREA


############################################################################
# Class for a list of tomograms with oriented membrane segmentations
#
class ListTomoOMSegmentations(object):

    def __init__(self):
        self.__tomos = dict()
        # For pickling
        self.__pkls = None

    # EXTERNAL FUNCTIONALITY

    def get_tomos(self):
        return self.__tomos

    def get_num_segmentations(self):
        total = 0
        for tomo in self.__tomos.itervalues():
            total += tomo.get_num_filaments()
        return total

    def get_num_segmentations_dict(self):
        nsegs = dict()
        for key, tomo in zip(self.__tomos.iterkeys(), self.__tomos.itervalues()):
            nsegs[key] = tomo.get_num_segmentations()
        return nsegs

    def get_volumes_dict(self):
        vols = dict()
        for key, tomo in zip(self.__tomos.iterkeys(), self.__tomos.itervalues()):
            vols[key] = tomo.compute_voi_volume()
        return vols

    def get_tomo_name_list(self):
        return self.__tomos.keys()

    def get_tomo_list(self):
        return self.__tomos.values()

    def get_tomo_by_key(self, key):
        return self.__tomos[key]

    def add_tomo(self, tomo_fils):
        """
        :param tomo_fils: TomoFilaments object to add to the list
        :return:
        """
        # Input parsing
        tomo_fname = tomo_fils.get_tomo_fname()
        if tomo_fname in self.get_tomo_name_list():
            print 'WARNING: tomo_surf (ListTomoOMSegmentations): tomogram ' + tomo_fname + ' was already inserted.'
            return
        # Adding the tomogram to the list and dictionaries
        self.__tomos[tomo_fname] = tomo_fils

    def del_tomo(self, tomo_key):
        """
        Delete a TomoSurface entry in the list
        :param tomo_key: TomoFilament key
        :return:
        """
        del self.__tomos[tomo_key]

    def insert_tomo(self, name, voi_mb=None, voi_lm=None, max_dst=0):
        """
        :param name: name to identify the tomogram
        :param voi_mb: if None (default) the membrane tomogram is loaded from tomo_fname, otherwise this is actually
        the input tomogram
        :param voi_lm: if None (default) the lumen tomogram is loaded from tomo_fname, otherwise this is actually
        the input tomogram
        :param max_dst: maximum distance for membrane pixles to border inside the lumen (in segmentation pixels)
        :param lbls_lm: lists of labels with the lumen segmentations
        """
        self.__tomos[name] = TomoOMSegmentations(name, voi_mb=voi_mb, voi_lm=voi_lm, max_dst=max_dst)

    def store_stars(self, out_stem, out_dir):
        """
        Store the list of tomograms in STAR file and also the STAR files for every tomogram
        :param out_stem: string stem used for storing the STAR file for TomoOMSegmentation objects
        :param out_dir: output directory
        :return:
        """

        # STAR file for the tomograms
        tomos_star = sub.Star()
        tomos_star.add_column('_omTomoSegmentation')

        # Tomograms loop
        for i, tomo_fname in enumerate(self.__tomos.keys()):

            # Pickling the tomogram
            tomo_dir = out_dir + '/tomo_' + str(i)
            if not os.path.exists(tomo_dir):
                os.makedirs(tomo_dir)
            tomo_stem = os.path.splitext(os.path.split(tomo_fname)[1])[0]
            pkl_file = tomo_dir + '/' + tomo_stem + '_tom.pkl'
            self.__tomos[tomo_fname].pickle(pkl_file)

            # Adding a new enter
            kwargs = {'_omTomoSegmentation': pkl_file}
            tomos_star.add_row(**kwargs)

        # Storing the tomogram STAR file
        tomos_star.store(out_dir + '/' + out_stem + '_toml.star')

    def store_segmentations(self, out_dir, mode='mb'):
        """
        Store the filaments in a vtkPolyData per tomogram
        :param out_dir: output directory
        :param mode: 'mb' membrane, 'lm' lumen, 'mb-lm' membrane and lumen fused
        :return:
        """
        # Tomograms loop
        for i, tomo_fname in enumerate(self.__tomos.keys()):

            tomo_stem = os.path.splitext(os.path.split(tomo_fname)[1])[0]

            # Storing
            disperse_io.save_numpy(self.__tomos[tomo_fname].gen_seg_tomo(mode=mode),
                                   out_dir + '/' + tomo_stem + '_oms.mrc')


    # fname: file name ended with .pkl
    def pickle(self, fname):
        # Pickling
        pkl_f = open(fname, 'w')
        try:
            pickle.dump(self, pkl_f)
        finally:
            pkl_f.close()

    def filter_by_segmentations_num(self, min_num_seg=1):
        """
        Delete for list the tomogram with low segmentations
        :param min_num_segs: minimum number of particles, below the tomogram is deleted (default 1)
        :return:
        """
        hold_dict = dict()
        for key, tomo in zip(self.__tomos.iterkeys(), self.__tomos.itervalues()):
            # print key + ': ' + str(tomo.get_num_filaments())
            if tomo.get_num_segmentations() >= min_num_seg:
                hold_dict[key] = tomo
        self.__tomos = hold_dict

    def clean_low_pouplated_tomos(self, n_keep):
        """
        Clean tomogram with a low amount of particles
        :param n_keep: number of tomograms to, the one with the highest amount of particles
        :return:
        """
        if (n_keep is None) or (n_keep < 0) or (n_keep >= len(self.__tomos.keys())):
            return
        # Sorting loop
        n_segs = dict()
        for key, tomo in zip(self.__tomos.iterkeys(), self.__tomos.itervalues()):
            n_segs[key] = tomo.get_num_segmentations()
        pargs_sort = np.argsort(np.asarray(n_segs.values()))[::-1]
        keys = n_segs.keys()
        # Cleaning loop
        hold_dict = dict()
        for parg in pargs_sort[:n_keep]:
            key = keys[parg]
            hold_dict[key] = self.__tomos[key]
        self.__tomos = hold_dict

    def segmentations_by_tomos(self):
        """
        :return: return a dictionary with the num of filaments by tomos
        """
        keys = self.get_tomo_fname_list()
        segs = dict.fromkeys(keys)
        for key in keys:
            segs[key] = self.__tomos[key].get_num_segmentations()
        return segs

    def to_tomos_star(self, out_dir):
        """
        Generates a STAR file with TomoOMSegmentations pickles
        :param out_dir:  output directory for the pickles
        :return: a STAR file
        """

        # Initialization
        star_tomo = sub.Star()
        star_tomo.add_column('_psPickleFile')

        # Tomograms loop
        for tomo in self.get_tomo_list():
            tkey = os.path.splitext(os.path.split(tomo.get_tomo_fname())[1])[0]
            out_pkl = out_dir + '/' + tkey + '_tp.pkl'
            tomo.pickle(out_pkl)
            # Insert the pickled tomo into the star file
            star_tomo.set_column_data('_psPickleFile', out_pkl)

        return star_tomo

    # INTERNAL FUNCTIONALITY AREA

############################################################################
# Class for a set of list tomograms with embedded filaments
#
class SetListTomoOMSegmentations(object):

    def __init__(self):
        self.__lists = dict()

    # EXTERNAL FUNCTIONALITY

    def get_lists(self):
        return self.__lists

    def get_lists_by_key(self, key):
        return self.__lists[key]

    def get_key_from_short_key(self, short_key):
        for lkey in self.__lists.iterkeys():
            fkey = os.path.split(lkey)[1]
            hold_key_idx = fkey.index('_')
            hold_key = fkey[:hold_key_idx]
            if short_key == hold_key:
                return lkey

    def get_lists_by_short_key(self, key):
        for lkey, list in zip(self.__lists.iterkeys(), self.__lists.itervalues()):
            if self.get_key_from_short_key(lkey) == key:
                return  list

    def get_tomos_name(self):
        hold_list = list()
        for ltomos in self.get_lists().values():
            hold_list += ltomos.get_tomo_name_list()
        return list(set(hold_list))

    def add_list_tomos(self, ltomos, list_name):
        """
        Add a new ListTomoOMSegmentations to the set
        :param ltomos: input ListTomoOMSegmentations object
        :param list_name: string for naming the list
        :return:
        """
        # Input parsing (compatible with older versions)
        if ltomos.__class__.__name__ != 'ListTomoOMSegmentations':
            error_msg = 'WARNING: add_tomo (SetListTomoOMSegmentations): ltomos input must be ListTomoParticles object.'
            raise pexceptions.PySegInputError(expr='add_tomo (SetListTomoOMSegmentations)', msg=error_msg)
        # Adding the list to the dictionary
        self.__lists[str(list_name)] = ltomos

    def get_set_tomos(self):
        """
        :return: return a set with all tomos in all list
        """
        tomos_l = list()
        for key, ltomos in zip(self.__lists.iterkeys(), self.__lists.itervalues()):
            for tomo in ltomos.get_tomo_list():
                tomos_l.append(tomo.get_tomo_fname())
        return set(tomos_l)

    def segmentations_by_list(self):
        """
        :return: Return a dictionary with the number of filaments by list
        """
        segs = dict()
        for key, ltomos in zip(self.__lists.iterkeys(), self.__lists.itervalues()):
            segs[key] = ltomos.get_num_segmentations()
        return segs

    def segmentations_by_tomos(self):
        """
        :return: Return a dictionary with the number of filaments by tomogram
        """
        segs = dict()
        for key_l, ltomos in zip(self.__lists.iterkeys(), self.__lists.itervalues()):
            for tomo in ltomos.get_tomo_list():
                try:
                    segs[tomo.get_tomo_fname()] += tomo.get_num_segmentations()
                except KeyError:
                    segs[tomo.get_tomo_fname()] = tomo.get_num_segmentations()
        return segs

    def proportions_by_tomos(self):
        """
        :return: Return a dictionary with the proportions for every tomogram
        """
        segs = dict()
        for key_l, ltomos in zip(self.__lists.iterkeys(), self.__lists.itervalues()):
            for tomo in ltomos.get_tomo_list():
                key_t = tomo.get_tomo_name()
                try:
                    segs[key_t].append(tomo.get_num_segmentations())
                except KeyError:
                    segs[key_t] = list()
                    segs[key_t].append(tomo.get_num_segmentations())
        return segs

    def proportions_by_list(self):
        """
        :return: Return a dictionary with the proportions for every tomogram
        """
        # Initialization
        segs_list, segs_tomo = dict.fromkeys(self.__lists.keys()), dict.fromkeys(self.get_set_tomos())
        for key_t in segs_tomo.iterkeys():
            segs_tomo[key_t] = 0
        for key_l in segs_list.iterkeys():
            segs_list[key_l] = np.zeros(shape=len(segs_tomo.keys()))
        # Particles loop
        for key_l, ltomos in zip(self.__lists.iterkeys(), self.__lists.itervalues()):
            for i, key_t in enumerate(segs_tomo.keys()):
                tomo = ltomos.get_tomo_by_key(key_t)
                hold_fils = tomo.get_num_segmentations()
                segs_tomo[key_t] += hold_fils
                segs_list[key_l][i] += hold_fils
        # Proportions loop
        for key_l in segs_list.iterkeys():
            for i, tomo_nfils in enumerate(segs_tomo.values()):
                if tomo_nfils > 0:
                    segs_list[key_l][i] /= tomo_nfils
                else:
                    segs_list[key_l][i] = 0.
        return segs_list

    def pickle_tomo_star(self, out_star, out_dir_pkl):
        """
        Generates a STAR file with the ListTomoOMSegmentations and pickes their TomoOMSegmentations
        :param out_star: output STAR file
        :param out_dir_pkl: output directory for the pickles
        :return: a STAR file
        """

        # Initialization
        star_list = sub.Star()
        star_list.add_column('_psPickleFile')

        # Tomograms loop
        for lname, ltomo in zip(self.__lists.iterkeys(), self.__lists.itervalues()):
            lkey = os.path.splitext(os.path.split(lname)[1])[0]
            out_star, out_list_dir = out_dir_pkl + '/' + lkey + '_tf.star', out_dir_pkl + '/' + lkey + '_tf'
            clean_dir(out_list_dir)
            list_star = ltomo.to_tomo_star(out_list_dir)
            list_star.store(out_star)
            # Insert the pickled tomo into the star file
            star_list.set_column_data('_psPickleFile', out_star)

        star_list.store(out_star)

    def filter_by_segmentations_num_tomos(self, min_num_segs=1):
        """
        Delete those tomograms with a number of filaments lower than an input value for any list
        :param min_num_segmentations: a number or a dict, the allows to specify different minimum number for each layer
        :return:
        """

        # Computing total particles per tomogram loop
        if isinstance(min_num_segs, dict):
            tomos_dict = dict().fromkeys(self.get_set_tomos(), 0)
            for lkey, ltomos in zip(self.__lists.iterkeys(), self.__lists.itervalues()):
                hold_min = min_num_segs[lkey]
                for tomo in ltomos.get_tomo_list():
                    if tomo.get_num_segmentations() >= hold_min:
                        tomos_dict[tomo.get_tomo_fname()] += 1
            tomos_del = dict().fromkeys(self.get_set_tomos(), False)
            for key in tomos_dict.iterkeys():
                if tomos_dict[key] < len(min_num_segs.keys()):
                    tomos_del[key] = True
        else:
            tomos_del = dict().fromkeys(self.get_set_tomos(), False)
            for tkey in tomos_del.keys():
                for ltomos in self.__lists.itervalues():
                    try:
                        tomo = ltomos.get_tomo_by_key(tkey)
                    except KeyError:
                        continue
                    if tomo.get_num_segmentations() < min_num_segs:
                        tomos_del[tkey] = True

        # Deleting loop
        for ltomos in self.__lists.itervalues():
            for tkey in tomos_del.keys():
                if tomos_del[tkey]:
                    try:
                        ltomos.del_tomo(tkey)
                    except KeyError:
                        continue

    def filter_by_segmentations_num(self, min_num_segs=1):
        """
        Delete for list the tomogram with low particles (considering all lists)
        :param min_num_segs: minimum number of particles, below that the tomogram is deleted (default 1)
        :return:
        """

        # Computing total particles per tomogram loop
        hold_dict = dict()
        for ltomos in self.__lists.itervalues():
            for tomo in ltomos.get_tomo_list():
                tkey, n_parts = tomo.get_tomo_fname(), tomo.get_num_segmentations()
                try:
                    hold_dict[tkey] += n_parts
                except KeyError:
                    hold_dict[tkey] = n_parts

        # Deleting loop
        for tkey, n_parts in zip(hold_dict.iterkeys(), hold_dict.itervalues()):
            if n_parts < min_num_segs:
                for ltomos in self.__lists.itervalues():
                    ltomos.del_tomo(tkey)
