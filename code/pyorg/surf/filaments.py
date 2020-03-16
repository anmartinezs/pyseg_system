'''
Classes for representing a tomograms with filaments

'''

import gc
import os
import time
import random
from shutil import rmtree
from pyorg import pexceptions, sub
from utils import *
from pyorg.spatial.sparse import nnde, cnnde
from pyorg.globals.utils import unpickle_obj
from sklearn.cluster import AffinityPropagation
import numpy as np
import scipy as sp
import multiprocessing as mp
try:
    import cPickle as pickle
except:
    import pickle

__author__ = 'Antonio Martinez-Sanchez'

##### Global variables

FIL_ID = 'fil_id'

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
# Class for tomograms with filaments
#
class TomoFilaments(object):

    def __init__(self, tomo_fname, lbl, voi=None, sg=None):
        """
        :param tomo_fname: full path to a segmented tomogram
        :param lbl: label which marks the VOI (Volume Of Interest)
        :param voi: if None (default) unused, otherwise VOI is already available so 'lbl' will not be considered
        :param sg: sigma Gaussian smoothing to reduce surface stepping, default None. Only valid if VOI is None
        """

        # Input parsing
        if not isinstance(tomo_fname, str):
            error_msg = 'Input is not a path.'
            raise pexceptions.PySegInputError(expr='__init__ (TomoParticles)', msg=error_msg)
        if (voi is not None) and not(isinstance(voi, vtk.vtkPolyData) or isinstance(voi, np.ndarray)):
            error_msg = 'Input VOI must an vtkPolyData object.'
            raise pexceptions.PySegInputError(expr='__init__ (TomoParticles)', msg=error_msg)
        self.__fils = list()
        self.__fname = tomo_fname
        self.__df_voi, self.__df_dst_ids = None, None

        # Getting VOI
        self.__voi_selector = None
        if isinstance(voi, np.ndarray):
            self.__voi = voi > 0
            self.__voi_dst_ids = sp.ndimage.morphology.distance_transform_edt(np.invert(self.__voi),
                                                                              return_distances=False,
                                                                              return_indices=True)
            self.__bounds = np.zeros(shape=6, dtype=np.float32)
            self.__update_bounds()
        else:
            self.__voi = voi
            self.__voi_dst_ids = None
            # Selector
            self.__voi_selector = vtk.vtkSelectEnclosedPoints()
            self.__voi_selector.Initialize(self.__voi)
            # Bounds
            self.__bounds = np.zeros(shape=6, dtype=np.float32)
            self.__update_bounds()
        if self.__voi is None:
            tomo = disperse_io.load_tomo(tomo_fname, mmap=False)
            tomo = (tomo == lbl).astype(np.float32)
            if (sg is not None) and (sg > 0):
                tomo = sp.ndimage.filters.gaussian_filter(tomo, sg)
            self.__voi = iso_surface(tomo, 0.5, closed=True, normals='outwards')
            self.__voi_selector = vtk.vtkSelectEnclosedPoints()
            self.__voi_selector.Initialize(self.__voi)
            del tomo
            # Bounds
            self.__bounds = np.zeros(shape=6, dtype=np.float32)
            self.__update_bounds()

        # Intermediate file for pickling assisting
        self.__voi_fname = None
        self.__voi_ids_fname = None
        self.__fils_fname = None

    # GET/SET AREA

    # Decimate the VOI
    # dec: decimation factor (default None) for VOI surface
    def decimate_voi(self, dec):
        self.__voi = poly_decimate(self.__voi, dec)

    def get_voi(self):
        try:
            if isinstance(self.__df_voi, str):
                return np.load(self.__df_voi)
            else:
                return self.__voi
        except AttributeError:
            # For compatibility with older versions
            return self.__voi

    # Returns: tomogram full path on disk
    def get_tomo_fname(self):
        return self.__fname

    def get_filaments(self):
        return self.__fils

    def get_num_filaments(self):
        return len(self.__parts)


    # EXTERNAL FUNCTIONALITY AREA

    # Particle bounds are extended distance within the valid range of the tomogram
    # Returns: the extended surface
    def get_extended_bounds(self, part, ex_dst):

        ex_f, hold_bounds = float(ex_dst), part.get_bounds()

        hold_bounds[0] -= ex_f
        if hold_bounds[0] < self.__bounds[0]:
            hold_bounds[0] = self.__bounds[0]
        hold_bounds[2] -= ex_f
        if hold_bounds[2] < self.__bounds[2]:
            hold_bounds[2] = self.__bounds[2]
        hold_bounds[4] -= ex_f
        if hold_bounds[4] < self.__bounds[4]:
            hold_bounds[4] = self.__bounds[4]
        hold_bounds[1] += ex_f
        if hold_bounds[1] < self.__bounds[1]:
            hold_bounds[1] = self.__bounds[1]
        hold_bounds[3] += ex_f
        if hold_bounds[3] < self.__bounds[3]:
            hold_bounds[3] = self.__bounds[3]
        hold_bounds[5] += ex_f
        if hold_bounds[5] < self.__bounds[5]:
            hold_bounds[5] = self.__bounds[5]

        return hold_bounds

    def make_voi_surf(self, iso_th=.5, dec=.9):
        """
        Converts the VOI, in case it is a ndarray, into a vtkPolyData
        :return:
        """

        if isinstance(self.__voi, np.ndarray):
            shape = self.__voi.shape
            seg = np.zeros(shape=np.asarray(shape) + 2, dtype=np.float32)
            seg[1:shape[0] + 1, 1:shape[1] + 1, 1:shape[2] + 1] = self.__voi
            voi = iso_surface(seg, iso_th, closed=True, normals='outwards')
            self.__voi_dst_ids = None
            self.__voi = poly_decimate(voi, dec)
            self.__voi_selector = vtk.vtkSelectEnclosedPoints()
            self.__voi_selector.Initialize(self.__voi)
            # Bounds
            self.__bounds = np.zeros(shape=6, dtype=np.float32)
            self.__update_bounds()

    def has_voi_vtp(self):
        """
        :return: Only return true if the VOI is an vtkPolyData object
        """
        if isinstance(self.__voi, vtk.vtkPolyData):
            return True
        return False

    # Check if a surface is embedded in the tomogram
    # surf: the surface to check
    # mode: embedding mode, valid: 'full' the whole surface must be enclosed in the valid tomogram
    #       region, 'center' surface center,  'box' surface box
    def is_embedded(self, surf, mode='full'):

        # Input parsing
        if (mode != 'full') and (mode != 'center') and (mode != 'box'):
            error_msg = 'Input mode ' + str(mode) + ' is not valid!'
            raise pexceptions.PySegInputError(expr='is_embedded (TomoParticles)', msg=error_msg)

        if mode == 'center':
            # Check if the center point is fully embedded
            center = surf.get_center()
            if (center[0] < self.__bounds[0]) or (center[0] > self.__bounds[1]) or \
                    (center[1] < self.__bounds[2]) or (center[1] > self.__bounds[3]) or \
                    (center[2] < self.__bounds[4]) or (center[2] > self.__bounds[5]):
                return False
            x, y, z = int(round(center[0])), int(round(center[1])), int(round(center[2]))
            if self.__voi is not None:
                if isinstance(self.__voi, np.ndarray):
                    if (x >= 0) and (y >= 0) and (z >= 0) and (x < self.__voi.shape[0]) and (y < self.__voi.shape[1]) \
                            and (z < self.__voi.shape[2]):
                        if self.__voi[x, y, z] > 0:
                            return True
                else:
                    if self.__voi_selector.IsInsideSurface(x, y, z) > 0:
                        return True
                    # print 'ERROR: is embedded with mode=center for surfaces not implemented yet!'
                    # raise NotImplementedError
            return False


        # Check the possibility of embedding
        surf_bounds = surf.get_bounds()
        if (surf_bounds[1] < self.__bounds[0]) and (surf_bounds[0] > self.__bounds[1]) and \
                (surf_bounds[3] < self.__bounds[2]) and (surf_bounds[2] > self.__bounds[3]) and \
                (surf_bounds[5] < self.__bounds[4]) and (surf_bounds[4] > self.__bounds[5]):
            return False

        if mode == 'full':
            # Check if the box is fully embedded
            if (surf_bounds[0] < self.__bounds[0]) or (surf_bounds[1] > self.__bounds[1]) or \
                    (surf_bounds[2] < self.__bounds[2]) or (surf_bounds[3] > self.__bounds[3]) or \
                    (surf_bounds[4] < self.__bounds[4]) or (surf_bounds[5] > self.__bounds[5]):
                return False
            # Check if surface intersect the mask
            if not is_2_polys_intersect(surf.get_vtp(mode='surface'), self.__voi):
                    return False

        return True

    def insert_filament(self, fil, check_bounds=True, check_inter=None):
        """

        :param fil: particle to insert in the tomogram, it must be fully embedded by the tomogram
        :param check_bounds: if True (default) checks that all input filament points are embedded
                             within the tomogram bounds
        :param check_inter: if a value (default None) check that filament points are further than these value to another
                            already inserted filament
        :return:
        """

        # Input parsing
        if not isinstance(fil, Filament):
            error_msg = 'Input object must be a Filament instance.'
            raise pexceptions.PySegInputError(expr='insert_filament (TomoFilaments)', msg=error_msg)
        if (check_inter is not None) and self.check_filaments_itersection(fil, check_inter):
            error_msg = 'This particle intersect with another already inserted one.'
            raise pexceptions.PySegInputError(expr='insert_surface (TomoFilaments)', msg=error_msg)

        # Insert to list
        if check_bounds and (not self.is_embedded(fil)):
            error_msg = 'Input Filament is not fully embedded in the reference tomogram.'
            raise pexceptions.PySegInputError(expr='insert_filament (TomoFilaments)', msg=error_msg)
        self.__fils.append(fil)

    def delete_filaments(self, fil_ids):
        """
        Remove  filaments from the list
        :param fil_ids: integer ids of the filaments (their position in the current list of filaments)
        :return: None
        """
        # Loop for keeping survivors
        hold_fils = list()
        for i in range(len(self.__fils)):
            if not i in fil_ids:
                hold_fils.append(self.__fils[i])

        # Updating the list of filaments
        self.__fils = hold_fils

    def pickle(self, fname):
        """
        VTK attributes requires a special treatment during pickling
        :param fname: file name ended with .pkl
        :return:
        """

        # Dump pickable objects and store the file names of the unpickable objects
        stem_path = os.path.split(fname)[0]
        stem, ext = os.path.splitext(fname)
        stem_ = stem.replace('/', '_')
        is_voi_vtp = isinstance(self.__voi, vtk.vtkPolyData)
        if is_voi_vtp:
            self.__voi_fname = stem_path + '/' + stem_ + '_surf.vtp'
        else:
            self.__voi_fname = stem_path + '/' + stem_ + '_mask.npy'
            self.__voi_ids_fname = stem_path + '/' + stem_ + '_mask_ids.npy'
        # self.__parts_fname = stem_path + '/' + os.path.splitext(stem_)[0]+'_parts.star'
        self.__fils_fname = stem_path + '/' + stem_ + '_fils.star'
        pkl_f = open(fname, 'w')
        try:
            pickle.dump(self, pkl_f)
        finally:
            pkl_f.close()

        # Store unpickable objects
        if is_voi_vtp:
            writer = vtk.vtkXMLPolyDataWriter()
            writer.SetFileName(self.__voi_fname)
            vtk_ver = vtk.vtkVersion().GetVTKVersion()
            if int(vtk_ver[0]) < 6:
                writer.SetInput(self.__voi)
            else:
                writer.SetInputData(self.__voi)
            if writer.Write() != 1:
                error_msg = 'Error writing %s.' % self.__voi_fname
                raise pexceptions.PySegInputError(expr='pickle (TomoFilaments)', msg=error_msg)
        else:
            np.save(self.__voi_fname, self.__voi)
            np.save(self.__voi_ids_fname, self.__voi_dst_ids)

        # Store particles list
        self.filaments_pickle(self.__fils_fname)

    def filaments_pickle(self, out_fname):
        """
        Pickle the list of filaments and stores a STAR file for listing them
        :param out_fname: file name for the ouput STAR file
        :return:
        """

        # STAR file for the particles
        star = sub.Star()
        star.add_column('_fbCurve')

        # Create the directory for particles
        out_dir = os.path.splitext(out_fname)[0]
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Tomograms loop
        for i, part in enumerate(self.__parts):

            # Pickling the particle
            pkl_file = out_dir + '/filament_' + str(i) + '.pkl'
            part.pickle(pkl_file)

            # Adding a new enter
            kwargs = {'_fbCurve': pkl_file}
            star.add_row(**kwargs)

        # Storing the tomogram STAR file
        star.store(out_fname)

    def load_filaments_pickle(self, in_fname):
        """
        Load pickled Filaments listed by a STAR file
        :param in_fname: file name for the input STAR file
        :return:
        """

        # Initialization
        self.__fils = list()

        # STAR file for the particles
        star = sub.Star()
        star.load(in_fname)

        # Rows loop
        for row in range(star.get_nrows()):

            # Loading the particles
            self.__parts.append(unpickle_obj(star.get_element('_suLVtp', row)))

    def gen_filaments_vtp(self):
        """
        Generates a vtkPolyData object with all filaments
        :return: an VTP object with the filaments
        """

        # Initialization
        appender = vtk.vtkAppendPolyData()

        for i, fil in enumerate(self.get_filaments()):

            hold_vtp = fil.get_vtp()

            # print mode, str(hold_vtp.GetNumberOfCells())

            # Add particle id property
            fil_id = vtk.vtkIntArray()
            fil_id.SetName(FIL_ID)
            fil_id.SetNumberOfComponents(1)
            fil_id.SetNumberOfTuples(hold_vtp.GetNumberOfCells())
            fil_id.FillComponent(0, i)
            hold_vtp.GetCellData().AddArray(fil_id)

            # Append current particles
            appender.AddInputData(hold_vtp)

        # Fuse to one vtkPolyData
        appender.Update()

        return appender.GetOutput()

    def gen_model_instance(self, model_type, rad=0):
        """
        Generates and instance of the current tomogram from a model
        :param model_type: model template for generate the new instance
        :param rad: filament radius
        :return:
        """
        model = model_type(self.__voi, self.__fils[0])
        return model.gen_instance(len(self.__fils), self.__fname, rad=0)

    # Estimates VOI volume
    def compute_voi_volume(self):
        if self.__voi is None:
            vol = (self.__bounds[1]-self.__bounds[0]) * (self.__bounds[3]-self.__bounds[2]) * \
                  (self.__bounds[5]-self.__bounds[4])
        elif isinstance(self.__voi, vtk.vtkPolyData):
            masser = vtk.vtkMassProperties()
            masser.SetInputData(self.__voi)
            vol = masser.GetVolume()
        else:
            vol = float(self.__voi.sum())
        return vol

    # Estimates VOI volume
    def compute_voi_surface(self):
        if self.__voi is None:
            surf = 2. * (self.__bounds[1]-self.__bounds[0]) * (self.__bounds[3]-self.__bounds[2]) + \
                  2. * (self.__bounds[1]-self.__bounds[0]) * (self.__bounds[5]-self.__bounds[4]) + \
                  2. * (self.__bounds[3]-self.__bounds[2]) * (self.__bounds[5]-self.__bounds[4])
        else:
            surfacer = vtk.vtkMassProperties()
            surfacer.SetInputData(self.__voi)
            surf = surfacer.GetSurfaceArea()
        return surf

    # INTERNAL FUNCTIONALITY AREA

    def __update_bounds(self):
        if isinstance(self.__voi, vtk.vtkPolyData):
            arr = self.__voi.GetPoints().GetData()
            self.__bounds[0], self.__bounds[1] = arr.GetRange(0)
            self.__bounds[2], self.__bounds[3] = arr.GetRange(1)
            self.__bounds[4], self.__bounds[5] = arr.GetRange(2)
        else:
            hold_ids = np.where(self.__voi)
            self.__bounds[0], self.__bounds[1] = hold_ids[0].min(), hold_ids[0].max()
            self.__bounds[2], self.__bounds[3] = hold_ids[1].min(), hold_ids[1].max()
            self.__bounds[4], self.__bounds[5] = hold_ids[2].min(), hold_ids[2].max()

    # Restore previous state
    def __setstate__(self, state):
        self.__dict__.update(state)
        # Restore unpickable objects
        self.__voi_selector, self.__voi_dst_ids = None, None
        if os.path.splitext(self.__voi_fname)[1] == '.vtp':
            reader = vtk.vtkXMLPolyDataReader()
            reader.SetFileName(self.__voi_fname)
            reader.Update()
            self.__voi = reader.GetOutput()
            self.__voi_selector = vtk.vtkSelectEnclosedPoints()
            self.__voi_selector.Initialize(self.__voi)
        else:
            self.__voi, self.__voi_dst_ids = None, None
            self.__voi = np.load(self.__voi_fname, mmap_mode='r')
            self.__voi_dst_ids = np.load(self.__voi_ids_fname, mmap_mode='r')
        # Load particles from STAR file
        self.load_filaments_pickle(self.__fils_fname)

    # Copy the object's state from self.__dict__ which contains all instance attributes.
    # Afterwards remove unpickable objects
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_TomoFilaments__voi']
        del state['_TomoFilaments__voi_selector']
        del state['_TomoFilaments__voi_dst_ids']
        del state['_TomoFilaments__parts']
        return state

