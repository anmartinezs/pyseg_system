'''
Classes for generating organization models of particles in tomograms

'''

import os
import gc
import sys
import vtk
import abc
import math
import time
import random
import shutil
import numpy as np
import ctypes
import multiprocessing as mp
from pyorg import pexceptions
from pyorg import disperse_io
from pyorg.globals import unpickle_obj, clean_dir, lin_map
from surface import ListTomoParticles, TomoParticles, Particle, ParticleL
try:
    import cPickle as pickle
except:
    import pickle

__author__ = 'Antonio Martinez-Sanchez'

# GLOBAL VARIABLES

VTK_RAY_TOLERANCE = 0.000001
MAX_ITER_CONV = 1000

# GLOBAL FUNCTIONS

# # Clean an directory contents (directory is preserved)
# # dir: directory path
# def clean_dir(dir):
#     for root, dirs, files in os.walk(dir):
#         for f in files:
#             os.unlink(os.path.join(root, f))
#         for d in dirs:
#             shutil.rmtree(os.path.join(root, d))

# PARALLEL PROCESS

def pr_gen_tlist(pr_id, q_coords, q_angs, voi_shared, voi_shape, n_tomos, n_part_tomo, tomo_fname, mode_emb, voi_fname,
                 part_fname, model_obj):

    # Pre-process the VOI
    if voi_fname is None:
        hold_voi = np.frombuffer(voi_shared, dtype=np.uint8).view()
        hold_voi.shape = voi_shape
    else:
        hold_voi = disperse_io.load_poly(voi_fname)
    part_vtp = disperse_io.load_poly(part_fname)

    for i in range(n_tomos):

        # Generate instance
        model_obj.set_voi(hold_voi)
        model_obj.set_part(part_vtp)
        coords, angs = model_obj.gen_instance(n_part_tomo, tomo_fname, mode=mode_emb, coords=True)

        # Enqueue the result
        q_coords.put(coords)
        q_angs.put(angs)

    if pr_id < 0:
        return
    else:
        sys.exit(pr_id)

def pr_gen_tlist2(pr_id, n_part_tomo, tomo_fname, mode_emb, tmp_folder):

    # Load the model
    model_class = unpickle_obj(tmp_folder + '/shared_model.pkl')

    # Generate instance
    tomo_1, tomo_2 = model_class.gen_instance(n_part_tomo, tomo_fname, mode=mode_emb)

    # Pickle the result
    tomo_1.pickle(tmp_folder + '/model1_pr_'+str(pr_id)+'.pkl')
    tomo_2.pickle(tmp_folder + '/model2_pr_'+str(pr_id)+'.pkl')

    # Exit
    sys.exit(pr_id)


def gen_tlist(n_tomos, n_part_tomo, model_obj, voi, part_fname, mode_emb='full', npr=None, tmp_folder=None,
              in_coords=None):
    """
    Generates an instance of a ListTomoParticles for testing
    :param n_tomos: number of tomograms
    :param n_part_tomo: number of particles per tomogram
    :param model_obj: model obj for generating particle localizations
    :param voi: input VOI
    :param part_fname: particle file name with the shape
    :param mode_emb: model for embedding, valid: 'full' (default) and 'center'
    :param voi_surf: then VOIs are forcibly converted to vtkPolyData
    :param npr: number of parrallel processors (default None - Auto)
    :param tmp_folder: temporary folder only used for multiprocessing, required if VOIs are vtkDataObject
    :param in_coords: input coordinates if they are required, i.e. for ModelRR
    :return: a ListTomoParticles with the TomoParticles generated
    """

    # Input parsing
    if n_tomos <= 0:
        error_msg = 'Input number of tomograms must be greater than zero'
        raise pexceptions.PySegInputError(expr='gen_test_instance', msg=error_msg)
    if n_part_tomo <= 0:
        error_msg = 'Input number of particles per tomograms must be greater than zero'
        raise pexceptions.PySegInputError(expr='gen_test_instance', msg=error_msg)
    if not isinstance(model_obj, Model):
        error_msg = 'Input model_class must be a subclass of Model'
        raise pexceptions.PySegInputError(expr='gen_tlist', msg=error_msg)
    voi_fname, voi_shape, voi_shared = None, None, None
    if isinstance(voi, vtk.vtkPolyData):
        if tmp_folder is None:
            error_msg = 'When input VOI is a vtkPolyData object tmp_folder is compulsory.'
            raise pexceptions.PySegInputError(expr='gen_tlist', msg=error_msg)
        voi_fname = tmp_folder + '/gen_tlist_voi.vtp'
        disperse_io.save_vtp(voi, voi_fname)
    elif isinstance(voi, np.ndarray):
        voi_shape = voi.shape
        voi_len = np.array(voi_shape).prod()
        # voi_shared = mp.RawArray(ctypes.c_bool, voi.reshape(np.array(voi_shape).prod()))
        voi_shared_raw = mp.RawArray(np.ctypeslib.ctypes.c_uint8, voi_len)
        voi_shared = np.ctypeslib.as_array(voi_shared_raw)
        voi_shared[:] = voi.reshape(voi_len).astype(np.uint8)
    else:
        error_msg = 'Input VOI must be either a vtkPolyData or a ndarray.'
        raise pexceptions.PySegInputError(expr='gen_tlist', msg=error_msg)

    # Initialization
    ltomos = ListTomoParticles()

    # Multiprocessing initialization
    q_coords, q_angs = mp.Queue(), mp.Queue()
    if npr is None:
        npr = mp.cpu_count()
    if npr > n_tomos:
        npr = n_tomos
    spl_ids = np.array_split(np.arange(n_tomos), npr)

    # Multiprocessing loop
    processes = list()
    for pr_id, ids in enumerate(spl_ids):
        hold_fname = 'tomo_test_' + str(pr_id)
        if in_coords is not None:
            hold_vals = in_coords
        else:
            hold_vals = n_part_tomo
        pr = mp.Process(target=pr_gen_tlist, args=(pr_id, q_coords, q_angs, voi_shared, voi_shape, len(ids), hold_vals,
                                                   hold_fname, mode_emb, voi_fname, part_fname, model_obj))
        pr.start()
    pr_results = list()
    for pr in processes:
        pr.join()
        pr_results.append(pr.exitcode)
    for pr_id in range(len(processes)):
        if pr_id != pr_results[pr_id]:
            error_msg = 'Process ' + str(pr_id) + ' exited unexpectedly!'
            raise pexceptions.PySegInputError(expr='compute_uni_2nd_order (TomoParticles)', msg=error_msg)
    gc.collect()

    # Loop for gathering the results
    t_count = 0
    part_vtp = disperse_io.load_poly(part_fname)
    ref_surf_fname = model_obj.get_surf_fname()
    while t_count < n_tomos:
        hold_coords = q_coords.get()
        hold_angs = q_angs.get()
        hold_fname = 'tomo_test_' + str(t_count)
        hold_tomo = TomoParticles(hold_fname, 1, voi)
        # Loop for particles
        for i, coord in enumerate(hold_coords):
            angs = hold_angs[i]
            if ref_surf_fname is None:
                hold_part = Particle(part_vtp, center=(0, 0, 0))
                hold_part.rotation(float(angs[0]), float(angs[1]), float(angs[2]))
                hold_part.translation(float(coord[0]), float(coord[1]), float(coord[2]))
            else:
                hold_part = ParticleL(ref_surf_fname, center=(float(coord[0]), float(coord[1]), float(coord[2])),
                                      eu_angs=(float(angs[0]), float(angs[1]), float(angs[2])))
            hold_tomo.insert_particle(hold_part, check_bounds=False, check_inter=False)
        t_count += 1
        ltomos.add_tomo(hold_tomo)

    return ltomos

# Generates an instance of a ListTomoParticles from a model and another ListTomoParticles testing
# ltomos: input list of particles used as reference
# part_vtp: particle vtkPolyData
# n_part_tomos: number of tomograms per particle
# model_class: class with the model for generating particle localizations
# mode_emb: model for embedding, valid: 'full' (default) and 'center'
# npr: number of parrallel processors (default None - Auto)
# tmp_folder: temporary folder only used for multiprocessing (default './tmp_gen_list')
def gen_tlist_from_tlist(ltomos, part_vtp, model_class, mode_emb='full', npr=None, tmp_folder='./tmp_gen_list'):

    # Input parsing
    n_tomos = len(ltomos.get_tomo_list())
    if n_tomos <= 0:
        error_msg = 'Input number of tomograms must be greater than zero'
        raise pexceptions.PySegInputError(expr='gen_test_instance', msg=error_msg)
    if  ltomos.get_num_particles() <= 0:
        error_msg = 'Input number of particles per tomograms must be greater than zero'
        raise pexceptions.PySegInputError(expr='gen_test_instance', msg=error_msg)
    if not issubclass(model_class, Model):
        error_msg = 'Input model_class must be as subclass of Model'
        raise pexceptions.PySegInputError(expr='gen_tlist', msg=error_msg)

    # Initialization
    ltomos_out = ListTomoParticles()

    # Tomograms loop
    if npr is None:
        npr = mp.cpu_count()
    if npr == 1:
        # Serial instance generation
        for tomo in ltomos.get_tomo_list():
            # Generate data from model
            if tomo.get_num_particles() > 0:
                model_obj = model_class(tomo.get_voi(), part_vtp)
                ltomos_out.add_tomo(model_obj.gen_instance(tomo.get_num_particles(), tomo.get_tomo_fname(), mode=mode_emb))
    else:

        # Temporary directory
        if not os.path.exists(tmp_folder):
            os.makedirs(tmp_folder)
        else:
            clean_dir(tmp_folder)

        # Pickle the initial model
        pkls, fnames, n_parts = list(), list(), list()
        h_n_tomos = 0
        for tomo in ltomos.get_tomo_list():
            model_obj = model_class(tomo.get_voi(), part_vtp)
            fname = tomo.get_tomo_fname()
            tomo_stem = os.path.splitext(os.path.split(fname)[1])[0]
            pkl = 'shared_model_' + str(tomo_stem) + '.pkl'
            model_obj.pickle(tmp_folder + '/' + pkl)
            pkls.append(pkl)
            fnames.append(fname)
            n_parts.append(tomo.get_num_particles())
            h_n_tomos += 1
        # print str(n_tomos), str(h_n_tomos)

        # Parallel instance generation
        # npr = 1
        spl_ids = np.array_split(range(n_tomos), math.ceil(float(n_tomos) / npr))
        processes = list()
        for bunch in spl_ids:
            for pr_id in bunch:
                hold_fname = 'tomo_test_' + str(pr_id)
                pr = mp.Process(target=pr_gen_tlist, args=(pr_id, n_parts[pr_id], fnames[pr_id], mode_emb,
                                                           tmp_folder, pkls[pr_id]))
                pr.start()
                processes.append(pr)
                pr_results = list()
            for pr in processes:
                pr.join()
                pr_results.append(pr.exitcode)

        # Un pickle all generated models
        for bunch in spl_ids:
            for pr_id in bunch:
                if pr_id != pr_results[pr_id]:
                    error_msg = 'Process ' + str(pr_id) + ' exited unexpectedly!'
                    raise pexceptions.PySegInputError(expr='gen_tlist', msg=error_msg)
                try:
                    hold_tomo = unpickle_obj(tmp_folder + '/model_pr_' + str(pr_id) + '.pkl')
                except IOError:
                    error_msg = 'Process ' + str(pr_id) + ' did not store its instance!'
                    raise pexceptions.PySegInputError(expr='gen_tlist', msg=error_msg)
                ltomos_out.add_tomo(hold_tomo)
        gc.collect()

        # Clean up temporary folders
        shutil.rmtree(tmp_folder, ignore_errors=False, onerror=None)

    return ltomos_out


# Generates tow instances of a ListTomoParticles for testing Bi-variate analysis
# n_tomos: number of tomograms
# n_part_tomos: number of tomograms per particle
# model_class: class with the model for generating particle localizations
# mode_emb: model for embedding, valid: 'full' (default) and 'center'
# npr: number of parrallel processors (default None - Auto)
# tmp_folder: temporary folder only used for multiprocessing (default './tmp_gen_list')
def gen_tlist2(n_tomos, n_part_tomo, model_class, mode_emb='full', npr=None, tmp_folder='./tmp_gen_list'):

    # Input parsing
    if n_tomos <= 0:
        error_msg = 'Input number of tomograms must be greater than zero'
        raise pexceptions.PySegInputError(expr='gen_test_instance', msg=error_msg)
    if n_tomos <= 0:
        error_msg = 'Input number of particles per tomograms must be greater than zero'
        raise pexceptions.PySegInputError(expr='gen_test_instance', msg=error_msg)
    if not isinstance(model_class, Model):
        error_msg = 'Input model_class must be as subclass of Model'
        raise pexceptions.PySegInputError(expr='gen_tlist', msg=error_msg)

    # Initialization
    ltomos_1, ltomos_2 = ListTomoParticles(), ListTomoParticles()

    # Tomograms loop
    if npr is None:
        npr = mp.cpu_count()
    if npr == 1:
        # Serial instance generation
        for i in range(n_tomos):
            # Generate data from model
            tomo_1, tomo_2 = model_class.gen_instance(n_part_tomo, 'tomo_test_'+str(i), mode=mode_emb)
            ltomos_1.add_tomo(tomo_1)
            ltomos_2.add_tomo(tomo_2)
    else:

        # Pickle the initial model
        if not os.path.exists(tmp_folder):
            os.makedirs(tmp_folder)
        model_class.pickle(tmp_folder + '/shared_model.pkl')

        # Parallel instance generation
        # npr = 1
        spl_ids = np.array_split(range(n_tomos), math.ceil(float(n_tomos)/npr))
        processes = list()
        for bunch in spl_ids:
            for pr_id in bunch:
                hold_fname = 'tomo_test_'+str(pr_id)
                pr = mp.Process(target=pr_gen_tlist2, args=(pr_id, n_part_tomo, hold_fname, mode_emb,
                                                            tmp_folder))
                pr.start()
                processes.append(pr)
                pr_results = list()
            for pr in processes:
                pr.join()
                pr_results.append(pr.exitcode)

        # Un pickle all generated models
        for bunch in spl_ids:
            for pr_id in bunch:
                if pr_id != pr_results[pr_id]:
                    error_msg = 'Process ' + str(pr_id) + ' exited unexpectedly!'
                    raise pexceptions.PySegInputError(expr='gen_tlist2', msg=error_msg)
                try:
                    hold_tomo_1 = unpickle_obj(tmp_folder + '/model1_pr_'+str(pr_id)+'.pkl')
                    hold_tomo_2 = unpickle_obj(tmp_folder + '/model2_pr_'+str(pr_id)+'.pkl')
                except IOError:
                    error_msg = 'Process ' + str(pr_id) + ' did not store its instance!'
                    raise pexceptions.PySegInputError(expr='gen_tlist2', msg=error_msg)
                ltomos_1.add_tomo(hold_tomo_1)
                ltomos_2.add_tomo(hold_tomo_2)
        gc.collect()

        # Clean up temporary folders
        shutil.rmtree(tmp_folder, ignore_errors=False, onerror=None)

    return ltomos_1, ltomos_2

############################################################################
# Abstract class for particles generator within a VOI
#
class Model(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, voi, part, vect=(0, 0, 1)):
        self.set_voi(voi)
        self.set_part(part, vect)
        self.__type_name = None
        # Pickling variables
        self.__part_fname = None
        self.__voi_fname = None
        self.__ref_surf_fname = None

    def set_voi(self, voi):
        """
        Set the VOI for model, required to generate a model
        :param voi: input VOI, valid: a vtkPolyData object or a numpy ndarray
        :param vect: particle reference vector, default (0, 0, 1)
        """
        # Input parsing
        self.__voi = None
        if voi is None:
            return
        if (not isinstance(voi, vtk.vtkPolyData)) and (not isinstance(voi, np.ndarray)):
            error_msg = 'Invalid VOI type, it must be vtkPolyData or ndarray!'
            raise pexceptions.PySegInputError(expr='__init__ (Model)', msg=error_msg)
        self.__voi = voi
        self.__voi_ids = None
        if isinstance(self.__voi, vtk.vtkPolyData):
            self.__selector = vtk.vtkSelectEnclosedPoints()
            self.__selector.Initialize(self.__voi)
            self.__selector.SetTolerance(VTK_RAY_TOLERANCE)
            arr = self.__voi.GetPoints().GetData()
            self.__x_min, self.__x_max = arr.GetRange(0)
            self.__y_min, self.__y_max = arr.GetRange(1)
            self.__z_min, self.__z_max = arr.GetRange(2)
            self.__n_tries = (self.__x_max - self.__x_min) * (self.__y_max - self.__y_min) * \
                             (self.__z_max - self.__z_min)
        else:
            self.__selector = None
            self.__voi_ids = np.where(self.__voi)
            self.__x_min, self.__x_max = self.__voi_ids[0].min(), self.__voi_ids[0].max()
            self.__y_min, self.__y_max = self.__voi_ids[1].min(), self.__voi_ids[1].max()
            self.__z_min, self.__z_max = self.__voi_ids[2].min(), self.__voi_ids[2].max()
            self.__n_tries = self.__voi.sum()
        self.__n_tries = int(self.__n_tries)
        if self.__n_tries <= 0:
            error_msg = 'Input VOI seems to have null volume!'
            raise pexceptions.PySegInputError(expr='__init__ (Model)', msg=error_msg)

    def set_part(self, part, vect=(0, 0, 1)):
        """
        Set the input particle shape, a vtkPolyData object, required to generate instances
        :param part: the input vtkPolyData object
        :param vect: particle reference vector, default (0, 0, 1)
        :return:
        """
        self.__part = None
        if part is None:
            return
        if not isinstance(part, vtk.vtkPolyData):
            error_msg = 'Invalid particle type, it must be vtkPolyData!'
            raise pexceptions.PySegInputError(expr='__init__ (Model)', msg=error_msg)
        self.__part = part
        self.__vect = None
        if vect is None:
            return
        if (not hasattr(vect, '__len__')) or ((len(vect) != 3) and (len(vect) != 4)):
            error_msg = 'Invalid vector must be 3-tuple!'
            raise pexceptions.PySegInputError(expr='__init__ (Model)', msg=error_msg)
        self.__vect = np.asarray(vect, dtype=np.float)

    def set_ParticleL_ref(self, ref_surf_fname):
        """
        Set the reference surface, it enables to generate particles as ParticleL instances instead of Particle
        :param ref_surf_fname: path to the reference filename
        :return:
        """
        self.__ref_surf_fname = ref_surf_fname

    def get_surf_fname(self):
        return self.__ref_surf_fname

    def get_type_name(self):
        return self.__type_name

    # n_parts: number of particles to generate
    # tomo_fname: tomogram file name
    # mode: mode for embedding, valid: 'full' and 'center'
    # Returns: an TomoParticles object with simulated instance
    @abc.abstractmethod
    def gen_instance(self, n_parts, tomo_fname, mode='full'):
        if (mode != 'full') or (mode != 'center'):
            error_msg = 'Only modes \'full\' and \'center\' are valid for embedding, current \'' + \
                        str(mode) + '\' is not valid!'
            raise pexceptions.PySegInputError(expr='is_embedded (TomoParticles)', msg=error_msg)
        raise NotImplementedError

    # VTK attributes requires a special treatment during pickling
    # fname: file name ended with .pkl
    def pickle(self, fname):

        # Dump pickable objects and store the file names of the unpickable objects
        stem, ext = os.path.splitext(fname)
        self.__part_fname = stem + '_part.vtp'
        if isinstance(self.__voi, vtk.vtkPolyData):
            self.__voi_fname = stem + '_voi.vtp'
        else:
            self.__voi_fname = stem + '_voi.npy'
        pkl_f = open(fname, 'w')
        try:
            pickle.dump(self, pkl_f)
        finally:
            pkl_f.close()

        # Store unpickable objects
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(self.__part_fname)
        vtk_ver = vtk.vtkVersion().GetVTKVersion()
        if int(vtk_ver[0]) < 6:
            writer.SetInput(self.__part)
        else:
            writer.SetInputData(self.__part)
        if writer.Write() != 1:
            error_msg = 'Error writing %s.' % self.__part_fname
            raise pexceptions.PySegInputError(expr='pickle (Particle)', msg=error_msg)
        if isinstance(self.__voi, vtk.vtkPolyData):
            writer = vtk.vtkXMLPolyDataWriter()
            writer.SetFileName(self.__voi_fname)
            if int(vtk_ver[0]) < 6:
                writer.SetInput(self.__voi)
            else:
                writer.SetInputData(self.__voi)
            if writer.Write() != 1:
                error_msg = 'Error writing %s.' % self.__voi_fname
                raise pexceptions.PySegInputError(expr='pickle (Particle)', msg=error_msg)
        else:
            np.save(self.__voi_fname, self.__voi)

    # INTERNAL FUNCTIONALITY AREA

    # Restore previous state
    def __setstate__(self, state):
        self.__dict__.update(state)
        # Restore unpickable objects
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(self.__part_fname)
        reader.Update()
        self.__part = reader.GetOutput()
        if os.path.splitext(self.__voi_fname)[1] == '.vtp':
            reader = vtk.vtkXMLPolyDataReader()
            reader.SetFileName(self.__voi_fname)
            reader.Update()
            self.__voi = reader.GetOutput()
            self.__selector = vtk.vtkSelectEnclosedPoints()
            self.__selector.Initialize(self.__voi)
        else:
            self.__selector = None
            self.__voi = np.load(self.__voi_fname)

    # Copy the object's state from self.__dict__ which contains all instance attributes.
    # Afterwards remove unpickable objects
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_Model__part']
        del state['_Model__voi']
        del state['_Model__selector']
        return state


############################################################################
# Class for Completely Spatial Randomness with Volume exclusion (CSRV)
#
class ModelCSRV(Model):

    # voi: VOI surface
    # part: particle surface
    def __init__(self, voi=None, part=None):
        super(ModelCSRV, self).__init__(voi, part)
        self._Model__type_name = 'CSRV'

    def gen_instance(self, n_parts, tomo_fname, mode='full', coords=False, max_ntries_factor=10):
        """
        Generates a TomoParticles with this model
        :param n_parts: number of particles to generate
        :param tomo_fname: tomogram file name
        :param mode: mode for embedding, valid: 'full' and 'center'
        :param coords: if False (default) then a TomoParticles object is return, otherwise just a 2-tuple with a list
        with the coordinates and other with the rotations
        :param max_ntries_factor: if not None then it set the maximum number of tries to n_parts*max_ntries_factor
        :return: a TomoParticles object with simulated instance
        """

        # Seeding random generator needed for multiprocessing
        timestamp = time.time()
        np.random.seed(seed=int(str(math.fabs(int(timestamp)-timestamp))[2:9])) # getting second decimals

        # Initialization
        tomo = TomoParticles(tomo_fname, -1, self._Model__voi)
        n_tries = self._Model__n_tries
        ids_rnd = np.random.randint(0, n_tries-1, n_tries)

        # Generations loop
        count, count_it, out_coords, out_rots = 0, 0, list(), list()
        for i in range(n_tries):

            # Generate random center and rotation
            if isinstance(self._Model__voi, vtk.vtkPolyData):
                x_rnd = random.uniform(self._Model__x_min, self._Model__x_max)
                y_rnd = random.uniform(self._Model__y_min, self._Model__y_max)
                z_rnd = random.uniform(self._Model__z_min, self._Model__z_max)
                if self._Model__selector.IsInsideSurface(x_rnd, y_rnd, z_rnd) == 0:
                    continue
            elif self._Model__voi_ids is not None:
                id_rnd = ids_rnd[i]
                x_rnd, y_rnd, z_rnd = self._Model__voi_ids[0][id_rnd], self._Model__voi_ids[1][id_rnd], \
                                      self._Model__voi_ids[2][id_rnd]
            else:
                if self._Model__voi_ids is not None:
                    id_rnd = ids_rnd[i]
                    x_rnd, y_rnd, z_rnd = self._Model__voi_ids[0][id_rnd], self._Model__voi_ids[1][id_rnd], \
                                          self._Model__voi_ids[2][id_rnd]
                else:
                    x_rnd = random.uniform(self._Model__x_min, self._Model__x_max)
                    y_rnd = random.uniform(self._Model__y_min, self._Model__y_max)
                    z_rnd = random.uniform(self._Model__z_min, self._Model__z_max)
                x_rnd_i, y_rnd_i, z_rnd_i = int(round(x_rnd)), int(round(y_rnd)), int(round(z_rnd))
                if (x_rnd_i < 0) or (y_rnd_i < 0) or (z_rnd_i < 0) or (x_rnd_i > self._Model__voi.shape[0]) or \
                        (y_rnd_i > self._Model__voi.shape[1]) or (z_rnd_i > self._Model__voi.shape[2]) or \
                        (not self._Model__voi[x_rnd_i, y_rnd_i, z_rnd_i]):
                    continue
            rot_rnd = random.uniform(0., 360.)
            tilt_rnd = random.uniform(0., 360.)
            psi_rnd = random.uniform(0., 360.)

            # Particle construction
            if self._Model__ref_surf_fname is not None:
                try:
                    hold_part = ParticleL(self._Model__ref_surf_fname, center=(x_rnd, y_rnd, z_rnd),
                                          eu_angs=(rot_rnd, tilt_rnd, psi_rnd))
                except pexceptions.PySegInputError:
                    continue
            else:
                try:
                    hold_part = Particle(self._Model__part, center=(0, 0, 0), normal=self._Model__vect)
                except pexceptions.PySegInputError:
                    continue
                # Random rigid body transformation
                hold_part.rotation(rot_rnd, tilt_rnd, psi_rnd)
                hold_part.translation(x_rnd, y_rnd, z_rnd)

            # Checking embedding and no overlapping
            try:
                tomo.insert_particle(hold_part, check_bounds=True, mode=mode, check_inter=True)
            except pexceptions.PySegInputError:
                if count_it > MAX_ITER_CONV:
                    break
                count_it += 1
                continue
            out_coords.append((x_rnd, y_rnd, z_rnd))
            out_rots.append((rot_rnd, tilt_rnd, psi_rnd))
            count_it = 0

            # A new particle has been inserted
            count += 1
            if count >= n_parts:
                break

        # Check all particles has been placed
        if count < n_parts:
            print 'WARNING (ModelCSRV:gen_instance): TomoParticles generated with less particles, ' + \
                  str(count) + ', than demanded, ' + str(n_parts) + ' (' + str(100.*count/float(n_parts)) + '%).'

        if coords:
            return out_coords, out_rots
        else:
            return tomo


############################################################################
# Class for Sinusoidal Random Pattern Volume exclusion (SRPV)
#
class ModelSRPV(Model):

    # voi: VOI surface
    # part: particle surface
    # n_cycles: 3-tuple with the number of cycles for the sinusoidal in each dimension (X, Y, Z)
    #           (default (1,1,1))
    # sin_t: sinus threshold (default 0.5)
    # phase: 3-Tuple for phase shifting (default (0, 0, 0)) in radians
    def __init__(self, voi=None, part=None, vect=(0,0,1), n_cycles=(1,1,1), sin_t=0.5, phase=(0,0,0)):
        super(ModelSRPV, self).__init__(voi, part, vect)
        if (not hasattr(n_cycles, '__len__')) or (len(n_cycles) != 3):
            error_msg = 'Invalid vector must be 3-tuple!'
            raise pexceptions.PySegInputError(expr='__init__ (ModelSRPV)', msg=error_msg)
        if (not hasattr(phase, '__len__')) or (len(phase) != 3):
            error_msg = 'Invalid vector must be 3-tuple!'
            raise pexceptions.PySegInputError(expr='__init__ (ModelSRPV)', msg=error_msg)
        if (sin_t < 0) or (sin_t > 1):
            error_msg = 'Input sin_t must in range [0, 1]!'
            raise pexceptions.PySegInputError(expr='__init__ (ModelSRPV)', msg=error_msg)
        self._Model__type_name = 'SRPV'
        self.__n_cycles = np.asarray(n_cycles, dtype=np.int)
        self.__sin_t = float(sin_t)
        self.__phase = np.asarray(phase, dtype=float)

    def gen_instance(self, n_parts, tomo_fname, mode='full', coords=False):
        """
        Generates a TomoParticles with this model
        :param n_parts: number of particles to generate
        :param tomo_fname: tomogram file name
        :param mode: mode for embedding, valid: 'full' and 'center'
        :param coords: if False (default) then a TomoParticles object is return, otherwise just a 2-tuple with a list
        with the coordinates and other with the rotations
        :return: a TomoParticles object with simulated instance
        """

        # Initialization
        tomo = TomoParticles(tomo_fname, -1, self._Model__voi)

        # Building probability field
        if isinstance(self._Model__voi, vtk.vtkPolyData):
            voi_shape = (int(self._Model__x_max - self._Model__x_min + 1),
                         int(self._Model__y_max - self._Model__y_min + 1),
                         int(self._Model__z_max - self._Model__z_min + 1))
        else:
            voi_shape = self._Model__voi.shape
        X, Y, Z = np.meshgrid(np.linspace(-1., 1., voi_shape[1]),
                              np.linspace(-1., 1., voi_shape[0]),
                              np.linspace(-1., 1., voi_shape[2]))
        sin = np.sin(self.__n_cycles[0] * np.pi * X + self.__phase[0]) + \
              np.sin(self.__n_cycles[1] * np.pi * Y + self.__phase[1]) + \
              np.sin(self.__n_cycles[2] * np.pi * Z + self.__phase[2])
        # sin = lin_map(sin, 0, 1)
        sin *= (1. / 3.)

        # Generations loop
        # Points are the n greatest field samples
        ids_x, ids_y, ids_z = np.where(sin > self.__sin_t)
        ids_rnd = np.arange(len(ids_x))
        np.random.shuffle(ids_rnd)
        count, out_coords, out_rots = 0, list(), list()
        for id_rnd in ids_rnd:

            # Get center and generated random uniform rotation
            coord_rnd = np.asarray((ids_x[id_rnd], ids_y[id_rnd], ids_z[id_rnd]), dtype=float)
            if isinstance(self._Model__voi, vtk.vtkPolyData):
                if self._Model__selector.IsInsideSurface(coord_rnd[0], coord_rnd[1], coord_rnd[2]) == 0:
                    continue
            else:
                rcoord = np.round(coord_rnd).astype(np.int)
                if (rcoord[0] >= 0) and (rcoord[1] >= 0) and (rcoord[2] >= 0) and (rcoord[0] < voi_shape[0]) and \
                        (rcoord[1] < voi_shape[1]) and (rcoord[2] < voi_shape[2]):
                    if not self._Model__voi[rcoord[0], rcoord[1], rcoord[2]]:
                        continue
                else:
                    continue
            rot_rnd = random.uniform(0., 360.)
            tilt_rnd = random.uniform(0., 360.)
            psi_rnd = random.uniform(0., 360.)

            # Particle construction
            if self._Model__ref_surf_fname is not None:
                try:
                    hold_part = ParticleL(self._Model__ref_surf_fname, center=(coord_rnd[0], coord_rnd[1], coord_rnd[2]),
                                          eu_angs=(rot_rnd, tilt_rnd, psi_rnd))
                except pexceptions.PySegInputError:
                    continue
            else:
                # Particle construction
                try:
                    hold_part = Particle(self._Model__part, center=(0, 0, 0), normal=self._Model__vect)
                except pexceptions.PySegInputError:
                    continue
                # Random rigid body transformation
                hold_part.rotation(rot_rnd, tilt_rnd, psi_rnd)
                hold_part.translation(coord_rnd[0], coord_rnd[1], coord_rnd[2])

            # Checking embedding and no overlapping
            try:
                tomo.insert_particle(hold_part, check_bounds=True, mode=mode, check_inter=True)
            except pexceptions.PySegInputError:
                continue
            out_coords.append((coord_rnd[0], coord_rnd[1], coord_rnd[2]))
            out_rots.append((rot_rnd, tilt_rnd, psi_rnd))

            # A new particle has been inserted
            count += 1
            if count >= n_parts:
                break

        # Check all particles has been placed
        if count < n_parts:
            print 'WARNING (ModelSRPV:gen_instance): TomoParticles generated with less particles, ' + \
                  str(count) + ', than demanded, ' + str(n_parts) + '.'

        if coords:
            return out_coords, out_rots
        else:
            return tomo


############################################################################
# Class for 2 Co-localized SRPV
#
class Model2CCSRV(Model):

    # voi: VOI surface
    # part: particle surface
    # dst: characteristic interparticle averaged distance (default 0)
    # std: standard deviation for modeling Gaussian co-localization, if sg is None (default)
    #     two output patterns are un-correlated
    def __init__(self, voi=None, part=None, dst=0, std=None):
        super(Model2CCSRV, self).__init__(voi, part)
        self._Model__type_name = '2CCSRV'
        self.__dst = float(dst)
        self.__std = std

    def gen_instance(self, n_parts, tomo_fname, mode='full'):
        """
        Generates a TomoParticles with this model
        :param n_parts: number of particles to generate
        :param tomo_fname: tomogram file name
        :param mode: mode for embedding, valid: 'full' and 'center'
        :return: a TomoParticles object with simulated instance
        """

        # Initialization
        tomo1, tomo2 = TomoParticles(tomo_fname, -1, self._Model__voi), \
                       TomoParticles(tomo_fname, -1, self._Model__voi)
        if self._Model__voi_ids is not None:
            ids_rnd = np.random.randint(0, self._Model__n_tries-1, self._Model__n_tries)

        # Generations loop
        count = 0
        for i in range(self._Model__n_tries):

            # Generate random center and rotation for particle 1
            if isinstance(self._Model__voi, vtk.vtkPolyData):
                x_rnd = random.uniform(self._Model__x_min, self._Model__x_max)
                y_rnd = random.uniform(self._Model__y_min, self._Model__y_max)
                z_rnd = random.uniform(self._Model__z_min, self._Model__z_max)
                if self._Model__selector.IsInsideSurface(x_rnd, y_rnd, z_rnd) == 0:
                    continue
            elif self._Model__voi_ids is not None:
                id_rnd = ids_rnd[i]
                x_rnd, y_rnd, z_rnd = self._Model__voi_ids[0][id_rnd], self._Model__voi_ids[1][id_rnd], \
                                      self._Model__voi_ids[2][id_rnd]
            else:
                x_rnd = random.uniform(self._Model__x_min, self._Model__x_max)
                y_rnd = random.uniform(self._Model__y_min, self._Model__y_max)
                z_rnd = random.uniform(self._Model__z_min, self._Model__z_max)
                try:
                    x_rnd_i, y_rnd_i, z_rnd_i = int(round(x_rnd)), int(round(y_rnd)), int(round(z_rnd))
                    if not self._Model__voi[x_rnd_i, y_rnd_i, z_rnd_i]:
                        continue
                except IndexError:
                    continue

            rot_rnd = random.uniform(self.__dst, 360.)
            tilt_rnd = random.uniform(self.__dst, 360.)
            psi_rnd = random.uniform(self.__dst, 360.)

            if self._Model__ref_surf_fname is not None:
                try:
                    hold_part = ParticleL(self._Model__ref_surf_fname, center=(x_rnd, y_rnd, z_rnd),
                                          eu_angs=(rot_rnd, tilt_rnd, psi_rnd))
                except pexceptions.PySegInputError:
                    continue
            else:
                # Particle 1 construction and rigid body transformations
                try:
                    hold_part = Particle(self._Model__part, center=(0, 0, 0), normal=self._Model__vect)
                except pexceptions.PySegInputError:
                    continue
                hold_part.rotation(rot_rnd, tilt_rnd, psi_rnd)
                hold_part.translation(x_rnd, y_rnd, z_rnd)

            # Generate co-localized counterpart, particle 2
            if self.__std is None:
                x2_rnd = random.uniform(self._Model__x_min, self._Model__x_max)
                y2_rnd = random.uniform(self._Model__y_min, self._Model__y_max)
                z2_rnd = random.uniform(self._Model__z_min, self._Model__z_max)
            else:
                v_rnd = np.random.rand(3)
                norm = math.sqrt((v_rnd*v_rnd).sum())
                if norm <= 0:
                    v_rnd = np.asarray((0, 0, 1), dtype=float)
                v_rnd /= norm
                v_rnd *= (self.__std*np.random.randn(1) + self.__dst)[0]
                x2_rnd, y2_rnd, z2_rnd = x_rnd + v_rnd[0], y_rnd + v_rnd[1], z_rnd + v_rnd[2]
            if isinstance(self._Model__voi, vtk.vtkPolyData):
                if self._Model__selector.IsInsideSurface(x2_rnd, y2_rnd, z2_rnd) == 0:
                    continue
            else:
                x2_rnd, y2_rnd, z2_rnd = int(round(x2_rnd)), int(round(y2_rnd)), int(round(z2_rnd))
                try:
                    if not self._Model__voi[x2_rnd, y2_rnd, z2_rnd]:
                        continue
                except IndexError:
                    continue
            rot2_rnd = random.uniform(0., 360.)
            tilt2_rnd = random.uniform(0., 360.)
            psi2_rnd = random.uniform(0., 360.)

            # Particle 2 construction and rigid body transformations
            if self._Model__ref_surf_fname is not None:
                try:
                    hold_part2 = ParticleL(self._Model__ref_surf_fname, center=(x2_rnd, y2_rnd, z2_rnd),
                                           eu_angs=(rot2_rnd, tilt2_rnd, psi2_rnd))
                except pexceptions.PySegInputError:
                    continue
            else:
                try:
                    hold_part2 = Particle(self._Model__part, center=(0, 0, 0), normal=self._Model__vect)
                except pexceptions.PySegInputError:
                    continue
                hold_part2.rotation(rot2_rnd, tilt2_rnd, psi2_rnd)
                hold_part2.translation(x2_rnd, y2_rnd, z2_rnd)

            # Checking embedding and no overlapping for the pair of pair particles
            try:
                tomo1.insert_particle(hold_part, check_bounds=True, mode=mode, check_inter=True)
                tomo2.insert_particle(hold_part2, check_bounds=True, mode=mode, check_inter=True)
            except pexceptions.PySegInputError:
                continue

            # A new particle has been inserted
            count += 1
            if count >= n_parts:
                break

        # Check all particles has been placed
        if count < n_parts:
            print 'WARNING (Model2CCSRV:gen_instance): TomoParticles generated with less particles, ' + \
                  str(count) + ', than demanded, ' + str(n_parts) + '.'

        return tomo1, tomo2

############################################################################
# Class for Rotational Randomness (RR)
#
class ModelRR(Model):

    # voi: VOI surface
    # part: particle surface
    def __init__(self, voi=None, part=None):
        super(ModelRR, self).__init__(voi, part)
        self._Model__type_name = 'RR'

    def gen_instance(self, in_coords, tomo_fname, mode='center', coords=False, nrot_tries=10):
        """
        Generates a TomoParticles with this model
        :param in_coords: fixed input particles coordinates
        :param tomo_fname: tomogram file name
        :param mode: mode for embedding, valid: 'full' and 'center' (default)
        :param coords: if False (default) then a TomoParticles object is return, otherwise just a 2-tuple with a list
        :param ntries: number of tries for fitting without overlap a rotated particle
        with the coordinates and other with the rotations
        :return: a TomoParticles object with simulated instance
        """

        # Seeding random generator needed for multiprocessing
        timestamp = time.time()
        np.random.seed(seed=int(str(math.fabs(int(timestamp)-timestamp))[2:9])) # getting second decimals

        # Initialization
        n_parts = len(in_coords)
        tomo = TomoParticles(tomo_fname, -1, self._Model__voi)
        if self._Model__voi_ids is not None:
            ids_rnd = np.random.randint(0, self._Model__n_tries-1, self._Model__n_tries)

        # Generations loop
        count, out_coords, out_rots = 0, list(), list()
        for in_coord in in_coords:

            for j in range(nrot_tries):

                # Generate random rotations
                rot_rnd = random.uniform(0., 360.)
                tilt_rnd = random.uniform(0., 360.)
                psi_rnd = random.uniform(0., 360.)

                # Particle construction
                if self._Model__ref_surf_fname is not None:
                    try:
                        hold_part = ParticleL(self._Model__ref_surf_fname, center=(in_coord[0], in_coord[1], in_coord[2]),
                                              eu_angs=(rot_rnd, tilt_rnd, psi_rnd))
                    except pexceptions.PySegInputError:
                        continue
                else:
                    try:
                        hold_part = Particle(self._Model__part, center=(0, 0, 0), normal=self._Model__vect)
                    except pexceptions.PySegInputError:
                        continue
                    # Random rigid body transformation
                    hold_part.rotation(rot_rnd, tilt_rnd, psi_rnd)
                    hold_part.translation(in_coord[0], in_coord[1], in_coord[2])

                # Checking embedding and no overlapping
                try:
                    tomo.insert_particle(hold_part, check_bounds=False, mode=mode, check_inter=False)
                    out_coords.append((in_coord[0], in_coord[1], in_coord[2]))
                    out_rots.append((rot_rnd, tilt_rnd, psi_rnd))
                    count += 1
                    break
                except pexceptions.PySegInputError:
                    continue


        # Check all particles has been placed
        if count < n_parts:
            print 'WARNING (ModelRR:gen_instance): TomoParticles generated with less particles, ' + \
                    str(count) + ', than demanded, ' + str(n_parts) + ' (' + str(100. * count / float(n_parts)) + '%).'

        if coords:
            return out_coords, out_rots
        else:
            return tomo
