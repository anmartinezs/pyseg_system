'''
Classes for representing tomograms with filaments

'''


import os
import copy
import numpy as np
import scipy as sp
from shutil import rmtree
from utils import *
from pyorg import pexceptions, sub
from pyorg.globals.utils import unpickle_obj
from pyorg import globals as gl
from pyorg.diff_geom import SpaceCurve
try:
    import cPickle as pickle
except:
    import pickle

__author__ = 'Antonio Martinez-Sanchez'

##### Global variables

FIL_ID = 'fil_id'
FIL_MODE_FAST = 2
FIL_MODE_PRECISE = 1

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
# Class for a Filament: curve in the space
#
class Filament(object):

    def __init__(self, coords, dst_samp=None):
        """
        :param coords: sequence of coordinates
        :param dst_samp: if not None (default) ensures uniform sampling, if the curve length is smaller than the
        distance sample a ValueError is raised
        """

        # Ensure curve uniform sampling
        if dst_samp is not None:
            hold_curve = SpaceCurve(coords, mode=FIL_MODE_FAST, do_geom=True)
            hold_coords = hold_curve.gen_uni_sampled_coords(dst_samp)
        else:
            hold_coords = coords

        # Input parsing
        self.__curve = SpaceCurve(hold_coords, mode=FIL_MODE_PRECISE, do_geom=True)
        self.__vtp = self.__curve.get_vtp()

        # Pre-compute bounds for accelerate computations
        self.__bounds = np.zeros(shape=6, dtype=np.float32)
        self.__update_bounds()

    #### Set/Get functionality

    def get_coords(self):
        """
        :return: return curve coordinates
        """
        return self.__curve.get_samples()

    def get_vtp(self):
        """
        :return: return a vtkPolyData object
        """
        return self.__vtp

    def get_bounds(self):
        """
        :return: surface bounds (x_min, x_max, y_min, y_max, z_min, z_max) as array
        """
        return self.__bounds

    #### External functionality

    def add_vtp_global_attribute(self, name, vtk_type, value):
        """
        Add and attribute with the same value for all cells
        :param name: attribute name
        :param vtk_type: a child vtkAbstractArray for data type
        :param value: tuple with the value
        :return:
        """

        # Input parsing
        if not issubclass(vtk_type, vtk.vtkAbstractArray):
            error_msg = 'Input vtk_type must be child class of vtkAbstractArray!'
            raise pexceptions.PySegInputError(expr='add_vtp_global_attribute (Particle)', msg=error_msg)
        if isinstance(value, str):
            t_value, n_comp = value, 1
        elif not isinstance(value, tuple):
            t_value, n_comp = (value,), 1
        else:
            t_value, n_comp = value, len(value)

        # Initialization
        prop = vtk_type()
        prop.SetNumberOfComponents(n_comp)
        prop.SetName(str(name))

        # Array construction
        n_cells = self.__vtp.GetNumberOfCells()
        prop.SetNumberOfTuples(n_cells)
        for i in range(n_cells):
            # prop.SetValue(i, t_value)
            prop.SetTuple(i, t_value)

        # Adding the data
        self.__vtp.GetCellData().AddArray(prop)

    def translate(self, shift_x, shift_y, shift_z):
        """
        Rigid translation
        :param shift_i: translations in X, Y, Z axes respectively
        :return:
        """

        # Transformation on the PolyData
        box_tr = vtk.vtkTransform()
        box_tr.Translate(shift_x, shift_y, shift_z)
        tr_box = vtk.vtkTransformPolyDataFilter()
        tr_box.SetInputData(self.__vtp)
        tr_box.SetTransform(box_tr)
        tr_box.Update()
        self.__vtp = tr_box.GetOutput()

        # Update the curve point from PolyData
        self.__curve = polyline_to_points(self.__vtp)

        # Update the bounds
        self.__update_bounds()

    def rotate(self, rot, tilt, psi, active=True):
        """
        Rigid rotation
        :param rot, tilt, psi: rotation angles in Relion format in degrees
        :param active: if True the rotation is active, otherwise it is possible
        :return:
        """

        # Transformation on the PolyData
        M = gl.rot_mat_relion(rot, tilt, psi, deg=True)
        if not active:
            M = M.T
        mat_rot = vtk.vtkMatrix4x4()
        for i in range(3):
            for j in range(3):
                mat_rot.SetElement(i, j, M[i, j])
        rot_tr = vtk.vtkTransform()
        rot_tr.SetMatrix(mat_rot)
        tr_rot = vtk.vtkTransformPolyDataFilter()
        tr_rot.SetInputData(self.__vtp)
        tr_rot.SetTransform(rot_tr)
        tr_rot.Update()
        self.__vtp = tr_rot.GetOutput()

        # Update the curve point from PolyData
        self.__curve = polyline_to_points(self.__vtp)

        # Update the bounds
        self.__update_bounds()

    def swap_xy(self):
        """
        Swap the coordinates of XY axes
        :return:
        """
        # Swap surface points
        self.__vtp = poly_swapxy(self.__vtp)
        # Swap the centers
        self.__center = poly_swapxy(self.__center)
        # Swap the normal vector
        hold_normal = self.get_normal()
        self.set_normal((hold_normal[1], hold_normal[0], hold_normal[2]))
        # Swap the bounds by updating
        self.__update_bounds()

    def store_vtp(self, fname):
        """
        Saves the object poly in disk as a *.vtp file
        :param fname: full path of the output file
        :param mode: kind of poly to store, valid: 'surface' (default) or 'center'
        :return:
        """
        # Input parsing
        disperse_io.save_vtp(self.__center, fname)

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

        # Store unpickable objects
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(self.__vtp_fname)
        vtk_ver = vtk.vtkVersion().GetVTKVersion()
        if int(vtk_ver[0]) < 6:
            writer.SetInput(self.__vtp)
        else:
            writer.SetInputData(self.__vtp)
        if writer.Write() != 1:
            error_msg = 'Error writing %s.' % self.__vtp_fname
            raise pexceptions.PySegInputError(expr='pickle (Filament)', msg=error_msg)

    # INTERNAL FUNCTIONALITY AREA

    def __update_bounds(self):
        arr = self.__vtp.GetBounds() # self.__vtp.GetPoints().GetData()
        self.__bounds[0], self.__bounds[1] = arr[0], arr[1] # arr.GetRange(0)
        self.__bounds[2], self.__bounds[3] = arr[2], arr[3] # arr.GetRange(1)
        self.__bounds[4], self.__bounds[5] = arr[4], arr[5] # arr.GetRange(2)

    # Restore previous state
    def __setstate__(self, state):
        self.__dict__.update(state)
        # Restore unpickable objects
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(self.__vtp_fname)
        reader.Update()
        self.__vtp = reader.GetOutput()

    # Copy the object's state from self.__dict__ which contains all instance attributes.
    # Afterwards remove unpickable objects
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_Filament__vtp']
        return state

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
        return len(self.__fils)


    # EXTERNAL FUNCTIONALITY AREA

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

    def is_embedded(self, fil, mode='full'):
        """
        Check if a surface is embedded in the tomogram
        :param surf: the filament to check
        :param mode:  embedding mode, valid: 'full' the whole filament must be enclosed in the valid tomogram
    #       region, 'box' surface box
        :return:
        """

        # Input parsing
        if (mode != 'full') and (mode != 'box'):
            error_msg = 'Input mode ' + str(mode) + ' is not valid!'
            raise pexceptions.PySegInputError(expr='is_embedded (TomoFilaments)', msg=error_msg)


        # Check the possibility of embedding
        fil_bounds = fil.get_bounds()
        if (fil_bounds[1] < self.__bounds[0]) and (fil_bounds[0] > self.__bounds[1]) and \
                (fil_bounds[3] < self.__bounds[2]) and (fil_bounds[2] > self.__bounds[3]) and \
                (fil_bounds[5] < self.__bounds[4]) and (fil_bounds[4] > self.__bounds[5]):
            return False

        if mode == 'full':
            # Check if the box is fully embedded
            if (fil_bounds[0] < self.__bounds[0]) or (fil_bounds[1] > self.__bounds[1]) or \
                    (fil_bounds[2] < self.__bounds[2]) or (fil_bounds[3] > self.__bounds[3]) or \
                    (fil_bounds[4] < self.__bounds[4]) or (fil_bounds[5] > self.__bounds[5]):
                return False
            # Check if surface intersect the VOI
            if isinstance(self.__voi, np.ndarray):
                if not is_2_polys_intersect(fil.get_vtp(), self.__voi):
                    return False
            else:
                coords = fil.get_coords()
                for coord in coords:
                    x, y, z = int(round(coord[0])), int(round(coord[1])), int(round(coord[2]))
                    if (x>=0) and (y>=0) and (z>=0) and \
                        (x<self.__voi.shape[0]) and (y<self.__voi.shape[1]) and (z<self.__voi.shape[2]):
                        if self.__voi[x, y, z] == 0:
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
        if (check_inter is not None) and self.check_filament_intersection(fil, check_inter):
            error_msg = 'This particle intersect with another already inserted one.'
            raise pexceptions.PySegInputError(expr='insert_surface (TomoFilaments)', msg=error_msg)

        # Insert to list
        if check_bounds and (not self.is_embedded(fil)):
            error_msg = 'Input Filament is not fully embedded in the reference tomogram.'
            raise pexceptions.PySegInputError(expr='insert_filament (TomoFilaments)', msg=error_msg)
        self.__fils.append(fil)

    def check_filament_intersection(self, fil, rad):
        """
        Determines if a filament intersect with an already inserted filament
        :param fil: input filament
        :param rad: filament radius
        :return:
        """
        fil_vtp = fil.get_vtp()
        for nfil in self.get_filaments():
            dst = vtp_to_vtp_closest_point(fil_vtp, nfil.get_vtp())
            if dst < rad:
                return True
        return False

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
        for i, part in enumerate(self.__fils):

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
            self.__fils.append(unpickle_obj(star.get_element('_fbCurve', row)))

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
        del state['_TomoFilaments__fils']
        return state

############################################################################
# Class for a list of tomograms with embedded filaments
#
class ListTomoFilaments(object):

    def __init__(self):
        self.__tomos = dict()
        # For pickling
        self.__pkls = None

    # EXTERNAL FUNCTIONALITY

    def get_tomos(self):
        return self.__tomos

    def get_num_filaments(self):
        total = 0
        for tomo in self.__tomos.itervalues():
            total += tomo.get_num_filaments()
        return total

    def get_num_filaments_dict(self):
        nfils = dict()
        for key, tomo in zip(self.__tomos.iterkeys(), self.__tomos.itervalues()):
            nfils[key] = tomo.get_num_filaments()
        return nfils

    def get_volumes_dict(self):
        vols = dict()
        for key, tomo in zip(self.__tomos.iterkeys(), self.__tomos.itervalues()):
            vols[key] = tomo.compute_voi_volume()
        return vols

    def get_tomo_fname_list(self):
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
        if tomo_fname in self.get_tomo_fname_list():
            print 'WARNING: tomo_surf (ListTomoFilaments): tomogram ' + tomo_fname + ' was already inserted.'
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

    def insert_filament(self, fil, tomo_fname, check_bounds=True, check_inter=None):
        """

        :param fil: particle to insert in the tomogram, it must be fully embedded by the tomogram
        :param tomo_fname: path to the tomogram
        :param check_bounds: if True (default) checks that all input filament points are embedded
                             within the tomogram bounds
        :param check_inter: if a value (default None) check that filament points are further than these value to another
                            already inserted filament
        :return:
        """
        try:
            self.__tomos[tomo_fname].insert_filament(fil, check_bounds, check_inter=check_inter)
        except KeyError:
            error_msg = 'Tomogram ' + tomo_fname + ' is not added to list!'
            raise pexceptions.PySegInputError(expr='insert_particle (ListTomoFilaments)', msg=error_msg)

    def store_stars(self, out_stem, out_dir):
        """
        Store the list of tomograms in STAR file and also the STAR files for every tomogram
        :param out_stem: string stem used for storing the STAR file for TomoFilaments objects
        :param out_dir: output directory
        :return:
        """

        # STAR file for the tomograms
        tomos_star = sub.Star()
        tomos_star.add_column('_fbTomoFilaments')

        # Tomograms loop
        for i, tomo_fname in enumerate(self.__tomos.keys()):

            # Pickling the tomogram
            tomo_dir = out_dir + '/tomo_' + str(i)
            if not os.path.exists(tomo_dir):
                os.makedirs(tomo_dir)
            tomo_stem = os.path.splitext(os.path.split(tomo_fname)[1])[0]
            pkl_file = tomo_dir + '/' + tomo_stem + '_tf.pkl'
            self.__tomos[tomo_fname].pickle(pkl_file)

            # Adding a new enter
            kwargs = {'_fbTomoFilaments': pkl_file}
            tomos_star.add_row(**kwargs)

        # Storing the tomogram STAR file
        tomos_star.store(out_dir + '/' + out_stem + '_tfl.star')

    def store_filaments(self, out_dir):
        """
        Store the filaments in a vtkPolyData per tomogram
        :param out_dir: output directory
        :param mode:
        :return:
        """
        # Tomograms loop
        for i, tomo_fname in enumerate(self.__tomos.keys()):

            tomo_stem = os.path.splitext(os.path.split(tomo_fname)[1])[0]

            # Storing
            disperse_io.save_vtp(self.__tomos[tomo_fname].gen_filaments_vtp(), out_dir + '/' + tomo_stem + '_fils.vtp')


    # fname: file name ended with .pkl
    def pickle(self, fname):

        # Store unpickable objects and create the table to find them
        f_path, f_name = os.path.split(fname)
        f_stem = os.path.splitext(f_name)[0]
        tomos_dir = f_path + '/' + f_stem
        if not os.path.exists(tomos_dir):
            os.makedirs(tomos_dir)

        # Dump pickable objects and store the file names of the unpickable objects
        count = 0
        self.__pkls = dict()
        for key, tomo in self.__tomos.iteritems():
            key_stem = os.path.splitext(os.path.split(key)[1])[0]
            hold_file = tomos_dir + '/' + key_stem + '.pkl'
            try:
                tomo.pickle(hold_file)
            except IOError:
                hold_file = tomos_dir + '/_' + str(count) + '.pkl'
                tomo.pickle(hold_file)
            self.__pkls[key] = hold_file
            count += 1

        # Pickling
        pkl_f = open(fname, 'w')
        try:
            pickle.dump(self, pkl_f)
        finally:
            pkl_f.close()

    def gen_model_instance(self, model, rad=0):
        """
        Generates and instance of the current tomogram from a model
        :param model: class for modeling TomoParticles
        :param rad: filament radius
        :return:
        """
        tomos = ListTomoFilaments()
        for i, tomo in enumerate(self.__tomos.itervalues()):
            model_in = model(tomo.get_voi(), rad=0)
            tomo_name = 'tomo_model_' + model_in.get_type_name() + '_' + str(i)
            tomos.add_tomo(model.gen_instance(model_in.get_num_filaments(), tomo_name))
        return tomos

    def filter_by_filaments_num(self, min_num_filaments=1):
        """
        Delete for list the tomogram with low filaments
        :param min_num_filaments: minimum number of particles, below the tomogram is deleted (default 1)
        :return:
        """
        hold_dict = dict()
        for key, tomo in zip(self.__tomos.iterkeys(), self.__tomos.itervalues()):
            # print key + ': ' + str(tomo.get_num_filaments())
            if tomo.get_num_filaments() >= min_num_filaments:
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
        n_fils = dict()
        for key, tomo in zip(self.__tomos.iterkeys(), self.__tomos.itervalues()):
            n_fils[key] = tomo.get_num_filaments()
        pargs_sort = np.argsort(np.asarray(n_fils.values()))[::-1]
        keys = n_fils.keys()
        # Cleaning loop
        hold_dict = dict()
        for parg in pargs_sort[:n_keep]:
            key = keys[parg]
            hold_dict[key] = self.__tomos[key]
        self.__tomos = hold_dict

    def filaments_by_tomos(self):
        """
        :return: return a dictionary with the num of filaments by tomos
        """
        keys = self.get_tomo_fname_list()
        fil = dict.fromkeys(keys)
        for key in keys:
            fil[key] = self.__tomos[key].get_num_filaments()
        return fil

    def to_tomos_star(self, out_dir):
        """
        Generates a STAR file with TomoFilaments pickles
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

    def __setstate__(self, state):
        """
        Restore previous state
        :param state:
        :return:
        """
        self.__dict__.update(state)
        # Restore unpickable objects
        self.__tomos = dict()
        for key, pkl in self.__pkls.iteritems():
            self.__tomos[key] = unpickle_obj(pkl)

    def __getstate__(self):
        """
        Copy the object's state from self.__dict__ which contains all instance attributes.
        Afterwards remove unpickable objects
        :return:
        """
        state = self.__dict__.copy()
        del state['_ListTomoFilaments__tomos']
        return state

############################################################################
# Class for a set of list tomograms with embedded filaments
#
class SetListTomoFilaments(object):

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

    def get_tomos_fname(self):
        hold_list = list()
        for ltomos in self.get_lists().values():
            hold_list += ltomos.get_tomo_fname_list()
        return list(set(hold_list))

    def add_list_tomos(self, ltomos, list_name):
        """
        Add a new ListTomoFilaments to the set
        :param ltomos: input ListTomoFilaments object
        :param list_name: string for naming the list
        :return:
        """
        # Input parsing (compatible with older versions)
        if ltomos.__class__.__name__ != 'ListTomoFilaments':
            error_msg = 'WARNING: add_tomo (SetListTomoFilaments): ltomos input must be ListTomoParticles object.'
            raise pexceptions.PySegInputError(expr='add_tomo (SetListTomoFilaments)', msg=error_msg)
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

    def merge_to_one_list(self, list_names=None):
        """
        Merge the particles lists into one
        :param list_names: a list with name of the list to merge, if None (default) then all are merged
        :return: a ListTomoFilaments object
        """

        # Input parsing
        if list_names is None:
            list_names = self.__lists.keys()
        out_list = ListTomoFilaments()

        # Getting the list of all tomograms
        tomo_names, vois = list(), dict()
        for list_name in list_names:
            hold_list = self.__lists[list_name]
            for tomo_name in hold_list.get_tomo_fname_list():
                tomo_names.append(tomo_name)
                if not(tomo_name in vois.keys()):
                    hold_tomo = hold_list.get_tomo_by_key(tomo_name)
                    vois[tomo_name] = hold_tomo.get_voi()
        tomo_names = list(set(tomo_names))

        # Loop for filling the tomograms
        hold_tomos = list()
        for tomo_name in tomo_names:
            hold_tomo = TomoFilaments(tomo_name, 1, voi=vois[tomo_name])
            out_list.add_tomo(hold_tomo)
            for list_name in list_names:
                hold_list = self.__lists[list_name]
                hold_ltomo = hold_list.get_tomo_by_key(tomo_name)
                for hold_fil in hold_ltomo.get_filaments():
                    out_list.insert_filament(copy.deepcopy(hold_fil), tomo_name, check_bounds=False, check_inter=False)

        return out_list

    def filaments_by_list(self):
        """
        :return: Return a dictionary with the number of filaments by list
        """
        fils = dict()
        for key, ltomos in zip(self.__lists.iterkeys(), self.__lists.itervalues()):
            fils[key] = ltomos.get_num_filaments()
        return fils

    def filaments_by_tomos(self):
        """
        :return: Return a dictionary with the number of filaments by tomogram
        """
        fils = dict()
        for key_l, ltomos in zip(self.__lists.iterkeys(), self.__lists.itervalues()):
            for tomo in ltomos.get_tomo_list():
                try:
                    fils[tomo.get_tomo_fname()] += tomo.get_num_filaments()
                except KeyError:
                    fils[tomo.get_tomo_fname()] = tomo.get_num_filaments()
        return fils

    def proportions_by_tomos(self):
        """
        :return: Return a dictionary with the proportions for every tomogram
        """
        parts = dict()
        for key_l, ltomos in zip(self.__lists.iterkeys(), self.__lists.itervalues()):
            for tomo in ltomos.get_tomo_list():
                key_t = tomo.get_tomo_fname()
                try:
                    parts[key_t].append(tomo.get_num_filaments())
                except KeyError:
                    parts[key_t] = list()
                    parts[key_t].append(tomo.get_num_filaments())
        return parts

    def proportions_by_list(self):
        """
        :return: Return a dictionary with the proportions for every tomogram
        """
        # Initialization
        fils_list, fils_tomo = dict.fromkeys(self.__lists.keys()), dict.fromkeys(self.get_set_tomos())
        for key_t in fils_tomo.iterkeys():
            fils_tomo[key_t] = 0
        for key_l in fils_list.iterkeys():
            fils_list[key_l] = np.zeros(shape=len(fils_tomo.keys()))
        # Particles loop
        for key_l, ltomos in zip(self.__lists.iterkeys(), self.__lists.itervalues()):
            for i, key_t in enumerate(fils_tomo.keys()):
                tomo = ltomos.get_tomo_by_key(key_t)
                hold_fils = tomo.get_num_filaments()
                fils_tomo[key_t] += hold_fils
                fils_list[key_l][i] += hold_fils
        # Proportions loop
        for key_l in fils_list.iterkeys():
            for i, tomo_nfils in enumerate(fils_tomo.values()):
                if tomo_nfils > 0:
                    fils_list[key_l][i] /= tomo_nfils
                else:
                    fils_list[key_l][i] = 0.
        return fils_list

    def pickle_tomo_star(self, out_star, out_dir_pkl):
        """
        Generates a STAR file with the ListTomoFilaments and pickes their TomoFilaments
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

    def filter_by_filaments_num_tomos(self, min_num_filaments=1):
        """
        Delete those tomograms with a number of filaments lower than an input value for any list
        :param min_num_filaments: a number or a dict, the allows to specify different minimum number for each layer
        :return:
        """

        # Computing total particles per tomogram loop
        if isinstance(min_num_filaments, dict):
            tomos_dict = dict().fromkeys(self.get_set_tomos(), 0)
            for lkey, ltomos in zip(self.__lists.iterkeys(), self.__lists.itervalues()):
                hold_min = min_num_filaments[lkey]
                for tomo in ltomos.get_tomo_list():
                    if tomo.get_num_filaments() >= hold_min:
                        tomos_dict[tomo.get_tomo_fname()] += 1
            tomos_del = dict().fromkeys(self.get_set_tomos(), False)
            for key in tomos_dict.iterkeys():
                if tomos_dict[key] < len(min_num_filaments.keys()):
                    tomos_del[key] = True
        else:
            tomos_del = dict().fromkeys(self.get_set_tomos(), False)
            for tkey in tomos_del.keys():
                for ltomos in self.__lists.itervalues():
                    try:
                        tomo = ltomos.get_tomo_by_key(tkey)
                    except KeyError:
                        continue
                    if tomo.get_num_filaments() < min_num_filaments:
                        tomos_del[tkey] = True

        # Deleting loop
        for ltomos in self.__lists.itervalues():
            for tkey in tomos_del.keys():
                if tomos_del[tkey]:
                    try:
                        ltomos.del_tomo(tkey)
                    except KeyError:
                        continue

    def filter_by_filaments_num(self, min_num_filaments=1):
        """
        Delete for list the tomogram with low particles (considering all lists)
        :param min_num_filaments: minimum number of particles, below that the tomogram is deleted (default 1)
        :return:
        """

        # Computing total particles per tomogram loop
        hold_dict = dict()
        for ltomos in self.__lists.itervalues():
            for tomo in ltomos.get_tomo_list():
                tkey, n_parts = tomo.get_tomo_fname(), tomo.get_num_filaments()
                try:
                    hold_dict[tkey] += n_parts
                except KeyError:
                    hold_dict[tkey] = n_parts

        # Deleting loop
        for tkey, n_parts in zip(hold_dict.iterkeys(), hold_dict.itervalues()):
            if n_parts < min_num_filaments:
                for ltomos in self.__lists.itervalues():
                    ltomos.del_tomo(tkey)
