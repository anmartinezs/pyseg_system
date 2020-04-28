'''
Classes for representing a tomograms with particles

'''

import gc
import os
import sys
import math
import time
import copy
import random
from shutil import rmtree
from pyorg import pexceptions, sub
from pyorg import globals as gl
# from pyorg.surf import gen_tlist, Model
from utils import *
from pyorg.spatial.sparse import nnde, cnnde
from pyorg.globals.utils import unpickle_obj
from sklearn.cluster import AffinityPropagation
import numpy as np
import scipy as sp
import ctypes
import multiprocessing as mp
import abc
import skfmm
from vtk.util import numpy_support
try:
    import cPickle as pickle
except:
    import pickle

__author__ = 'Antonio Martinez-Sanchez'

##### Global variables

SF_LOC_V, SF_GL_V = 'sf_local_vector', 'sf_global_vector'
SF_LOC_D, SF_GL_D = 'sf_local_distance', 'sf_global_distance'
SF_CC = 'sf_cell_centers'
SF_DEF_VAL = 0
OBJ_TYPE = 'sf_type'
OBJ_PART = 1
PT_ID = 'part_id'
MIN_THICK = 1e-6
SPH_CTE = 4. / 3.
ONE_THIRD = 1. / 3.
VTK_RAY_TOLERANCE = 0.001 # 0.000001
OVER_TOLERANCE = 0.01 # .02
MP_NUM_ATTEMPTS, MP_MIN_TIME, MP_JOIN_TIME = 10, 0.5, 10
MAX_TRIES_FACTOR = 10 # 100
PARTS_LBL_FIELD = 'parts_field'
TRY_FMM = False
CODE_POSITIVE_DST = 1
CODE_NEGATIVE_DST = 2
CODE_BORDER = 3
CODE_STR = 'CODE'

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

def pr_2nd_tomo(pr_id, part_ids, part_centers_1, part_centers_2, distances, thick, border,
                    conv_iter, max_iter, fmm, bi_mode,
                    voi_fname, voi_shape, voi_shared,
                    shared_mat_num, shared_mat_vol):
    """
    Parallel univariate particle local density for one tomogram in a list of neighborhoods
    :param pr_id: process identifier
    :param part_ids: ids for identify tomogram particles
    :param part_centers_1: coordinates for sampling
    :param part_centers_2: coordinates for the neighbours particles
    :param distances: distances array for the nhoods
    :param thick: thickness for the nhoods (if None the spherical nhood_vol, otherwise shell one)
    :param border: if True the border compensation is done
    :param conv_iter: number of iterations for convergence in volume estimation
    :param max_iter: maximum number of iterations in volume estimation
    :param fmm: if True Fast Marching Method active (it requires conv_iter=None and max_iter=None)
    :param bi_mode: if True then bivariate mode active (no central particle substraction)
    :param voi_shape: shared array for VOI as ndarray
    :param voi_shared: 3-tuple with VOI shape when it is a ndarray
    :param shared_mat_num-shared_mat_vol: output shared matrices (number of points, local volumes)
    :return:
    """

    # print 'Process ' + str(pr_id) + ' started: ' + time.strftime("%c")

    # Intialize VOI selector
    voi_surf, selector_voi = None, None
    if border:
        if voi_fname is None:
            voi = np.frombuffer(voi_shared, dtype=np.bool).view()
            voi.shape = voi_shape
            part_centers_1_int = np.round(part_centers_1).astype(np.int)
            part_centers_2_int = np.round(part_centers_2).astype(np.int)
        else:
            voi = disperse_io.load_poly(voi_fname)
            selector_voi = vtk.vtkSelectEnclosedPoints()
            selector_voi.SetTolerance(VTK_RAY_TOLERANCE)
            selector_voi.Initialize(voi)
    if thick is None:
        thick_f = float(distances[1] - distances[0])
    else:
        thick_f = float(thick)
    d_max = int(math.ceil(distances.max()))


    # Loop for particles
    n_parts, n_parts_id = len(part_centers_1), len(part_ids)
    n_dsts, n_parts_f = len(distances), float(n_parts)
    for i, p_id in enumerate(part_ids):

        # Get the subvolume of VOI
        if isinstance(voi, np.ndarray):
            particle = part_centers_1_int[p_id, :]
            particle_low, particle_high = particle - d_max, particle + d_max
            if particle_low[0] < 0:
                particle_low[0] = 0
            if particle_low[1] < 0:
                particle_low[1] = 0
            if particle_low[2] < 0:
                particle_low[2] = 0
            if particle_high[0] >= voi.shape[0]:
                particle_high[0] = voi.shape[0]
            if particle_high[1] >= voi.shape[1]:
                particle_high[1] = voi.shape[1]
            if particle_high[2] >= voi.shape[2]:
                particle_high[2] = voi.shape[2]
            voi_surf = np.copy(voi[int(particle_low[0]):int(particle_high[0]),
                                   int(particle_low[1]):int(particle_high[1]),
                                   int(particle_low[2]):int(particle_high[2])]).astype(np.bool)
            particle -= particle_low
            particles_2 = part_centers_2_int - particle_low
        else:
            particle = part_centers_1[p_id, :]
            particles_2 = part_centers_2
            voi_surf = voi
        try:
            if thick is None:
                nhoods = ListSphereNhood(particle, distances,
                                         voi=voi_surf, selector_voi=selector_voi,
                                         conv_iter=conv_iter, max_iter=max_iter, fmm=fmm)
            else:
                nhoods = ListShellNhood(particle, distances, thick=thick_f,
                                        voi=voi_surf, selector_voi=selector_voi,
                                        conv_iter=conv_iter, max_iter=max_iter, fmm=fmm)
        except ValueError:
            print 'WARNING: process ' + str(pr_id) + ' failed to process the particle ' + str(p_id) + ', skipping...'
            raise ValueError
            # continue

        # Get spheres mask
        rgs = nhoods.get_rad_ranges()
        sph_mask = rgs[:, 0] <= 0

        ### Compute nhoods densities
        # num = np.array(nhoods.get_nums_embedded(particles_2), dtype=float)
        # num = np.array(nhoods.get_nums_embedded(part_centers_2), dtype=float)
        # dem = np.array(nhoods.get_volumes(), dtype=float)

        ### Compute number of embedded and volumes at once
        num, dem = nhoods.analyse(particles_2)

        # If thick is None the Ripley's L metric is assumed so center particle contribution must be cancel
        if not bi_mode:
            num_mask = num > 0
            num[num_mask & sph_mask] -= 1

        # Update shared data
        li = p_id * n_dsts
        ui = li + n_dsts
        shared_mat_num[li:ui] = num
        shared_mat_vol[li:ui] = dem

        # print 'Process ' + str(pr_id) + ' progress: ' + str(i + 1) + ' particles processed of ' \
        #       + str(len(part_ids)) + '.'

    # print 'Process ' + str(pr_id) + ' finished: ' + time.strftime("%c")

    if pr_id < 0:
        return
    else:
        sys.exit(pr_id)


def pr_sim_2nd_tomo(pr_id, sim_ids, part_centers_1, part_centers_2, distances, thick, border,
                    conv_iter, max_iter, fmm, bi_mode,
                    voi_fname, tem_model, part_vtp, mode_emb, voi_shape, voi_shared,
                    shared_mat_num, shared_mat_vol, shared_mat_spart):
    """
    Parallel univariate particle local density for one tomogram in a list of neighborhoods
    :param pr_id: process identifier
    :param sim_ids: simulation ids
    :param part_centers_1: coordinates for sampling
    :param part_centers_2: coordinates for the neighbours particles
    :param distances: distances array for the nhoods
    :param thick: thickness for the nhoods (if None the spherical nhood_vol, otherwise shell one)
    :param border: if True the border compensation is done
    :param conv_iter: number of iterations for convergence in volume estimation
    :param max_iter: maximum number of iterations in volume estimation
    :param fmm: if True Fast Marching Method active (it requires conv_iter=None and max_iter=None)
    :param bi_mode: if True then bivariate mode active (no central particle substraction)
    :param tem_model: class template for model simulations (set it as not for not to use simulation)
    :param part_vtp: particle vtkPolyData shape
    :param mode_emb: mode for particle embedding
    :param voi_shape: 3-tuple with VOI shape when it is a ndarray
    :param voi_shared: shared array for VOI as ndarray
    :param shared_mat_*: output shared matrices (number of points, local volumes, number of simulated particles)
    :return:
    """

    # print 'Process ' + str(pr_id) + ' started: ' + time.strftime("%c")

    # Intialize VOI selector
    voi_surf, selector_voi = None, None
    if border:
        if voi_fname is None:
            voi = np.frombuffer(voi_shared, dtype=np.uint8).view()
            voi.shape = voi_shape
        else:
            voi = disperse_io.load_poly(voi_fname)
            selector_voi = vtk.vtkSelectEnclosedPoints()
            selector_voi.SetTolerance(VTK_RAY_TOLERANCE)
            selector_voi.Initialize(voi)
    if thick is None:
        thick_f = float(distances[1] - distances[0])
    else:
        thick_f = float(thick)
    d_max = int(math.ceil(distances.max()))

    # Create the model
    tem_model.set_voi(voi)
    tem_model.set_part(part_vtp)

    # Loop for simulations
    for i, sim_id in enumerate(sim_ids):

        # Simulate the model
        if bi_mode:
            sim_tomo = tem_model.gen_instance(len(part_centers_2), 'tomo_sim_' + str(i), mode=mode_emb)
            part_centers_2 = sim_tomo.get_particle_coords()
        else:
            sim_tomo = tem_model.gen_instance(len(part_centers_1), 'tomo_sim_' + str(i), mode=mode_emb)
            part_centers_1 = sim_tomo.get_particle_coords()
            part_centers_2 = part_centers_1
        if voi_fname is None:
            part_centers_1_int = np.round(part_centers_1).astype(np.int)
            part_centers_2_int = np.round(part_centers_2).astype(np.int)

        # Loop for particles
        n_parts_1, n_parts_2 = len(part_centers_1), len(part_centers_2)
        n_dsts, n_parts_1_f = len(distances), float(n_parts_1)
        sim_off = sim_id * n_parts_1 * n_dsts
        for j in range(n_parts_1):

            # Get the subvolume of VOI
            if isinstance(voi, np.ndarray):
                particle = part_centers_1_int[j, :]
                particle_low, particle_high = particle - d_max, particle + d_max
                if particle_low[0] < 0:
                    particle_low[0] = 0
                if particle_low[1] < 0:
                    particle_low[1] = 0
                if particle_low[2] < 0:
                    particle_low[2] = 0
                if particle_high[0] >= voi.shape[0]:
                    particle_high[0] = voi.shape[0]
                if particle_high[1] >= voi.shape[1]:
                    particle_high[1] = voi.shape[1]
                if particle_high[2] >= voi.shape[2]:
                    particle_high[2] = voi.shape[2]
                voi_surf = np.copy(voi[int(particle_low[0]):int(particle_high[0]),
                                   int(particle_low[1]):int(particle_high[1]),
                                   int(particle_low[2]):int(particle_high[2])]).astype(dtype=np.bool)
                particle -= particle_low
                particles_2 = part_centers_2_int - particle_low
            else:
                particle = part_centers_1[j, :]
                particles_2 = part_centers_2
                voi_surf = voi

            try:
                if thick is None:
                    nhoods = ListSphereNhood(particle, distances,
                                             voi=voi_surf, selector_voi=selector_voi,
                                             conv_iter=conv_iter, max_iter=max_iter, fmm=fmm)
                else:
                    nhoods = ListShellNhood(particle, distances, thick=thick_f,
                                            voi=voi_surf, selector_voi=selector_voi,
                                            conv_iter=conv_iter, max_iter=max_iter, fmm=fmm)
            except ValueError:
                print 'WARNING: process ' + str(pr_id) + ' failed to process the particle ' + str(j) + ', skipping...'
                raise ValueError
                # continue

            # Get spheres mask
            rgs = nhoods.get_rad_ranges()
            sph_mask = rgs[:, 0] <= 0

            ### Compute number of embedded and volumes at once
            num, dem = nhoods.analyse(particles_2)

            # If thick is None the Ripley's L metric is assumed so center particle contribution must be cancel
            if not bi_mode:
                num_mask = num > 0
                num[num_mask & sph_mask] -= 1

            # Update shared data
            li = sim_off + j * n_dsts
            ui = li + n_dsts
            shared_mat_num[li:ui] = num
            shared_mat_vol[li:ui] = dem
        shared_mat_spart[sim_id] = float(n_parts_2)

        # print 'Process ' + str(pr_id) + ' progress: ' + str(i) + ' simulations of ' + str(n_sims) + '.'

    # print 'Process ' + str(pr_id) + ' finished: ' + time.strftime("%c")

    if pr_id < 0:
        return
    else:
        sys.exit(pr_id)

# GLOBAL FUNCTIONS

# Computes the amount of samples to achieve some arc resolution on a circle
# res: arc resolution (length of an arc of the circle perimeter)
# rad: circle radius
# min_samp: minimmun number of samples (default 4)
def samples_arc_res(res, rad, min_samp=4):
    per = 2. * np.pi * rad
    n_samp = int(math.ceil(per / res))
    if n_samp < min_samp:
        return min_samp
    else:
        return n_samp

############################################################################
# Class for holding the parameters and results (matrix) of a simulation
#
class Simulation(object):

    # rads: array with the neighbourhood radius
    # thick: neighbourhood shell thickness, if None then SphereNhood is assumed
    def __init__(self, rads, thick, sim_mat):

        # Input parsing
        if (not isinstance(rads, np.ndarray)) or (not isinstance(rads, np.ndarray)):
            error_msg = 'Input rads and sim_mat must be a ndarray.'
            raise pexceptions.PySegInputError(expr='__init__ (Simulation)', msg=error_msg)
        if (len(rads.shape) != 1) or (len(sim_mat.shape) != 2):
            error_msg = 'Input rads must be a one dimensional array and sim_mat a matrix.'
            raise pexceptions.PySegInputError(expr='__init__ (Simulation)', msg=error_msg)
        if rads.shape[0] != sim_mat.shape[1]:
            error_msg = 'Input rads and sim_mat.shape[1] does not agree.'
            raise pexceptions.PySegInputError(expr='__init__ (Simulation)', msg=error_msg)
        self.__thick = None
        if thick is not None:
            self.__thick = float(thick)
        self.__rads = rads
        self.__sim_mat = sim_mat

    # To select a specific Nhood range
    # sub_range: [low, high] range values, if one out of the initial range an exception is raised
    def set_sub_range(self, sub_range):
        if (sub_range[0] < self.__rads[0]) or (sub_range[1] > self.__rads[-1]):
            error_msg = 'Input sub range must be included in the initial range: [' + str(self.__in_rads[0]) + ', ' + \
                        str(self.__in_rads[-1]) + ']'
            raise pexceptions.PySegInputError(expr='__init__ (Simulation)', msg=error_msg)
        ids = np.where((self.__rads >= sub_range[0]) & (self.__rads <= sub_range[1]))[0]
        hold_rads, hold_sim_mat = self.__rads, self.__sim_mat
        self.__rads, self.__sim_mat = np.zeros(shape=len(ids), dtype=np.float32), \
                                      np.zeros(shape=(self.__sim_mat.shape[0], len(ids)), dtype=np.float32)
        for i, idx in enumerate(ids):
            self.__rads[i] = hold_rads[idx]
            self.__sim_mat[:, i] = hold_sim_mat[:, idx]

    def get_thick(self):
        return self.__thick

    def get_rads(self):
        return self.__rads

    def get_sim_mat(self):
        return self.__sim_mat

    def add_sim_mat(self, sim_mat):
        self.__sim_mat = np.concatenate([self.__sim_mat, sim_mat], axis=0)

    # Compares if another Simulation is compatible
    def is_compatible(self, sim):
        hold_rads, sim_rads = self.get_rads(), sim.get_rads()
        if self.get_thick() != sim.get_thick():
            return False
        return np.array_equal(self.get_rads(), sim.get_rads())

    # Computes the p-values (high or low sides) for experimental data
    # exp_arr: array with the experimental data
    # side: distribution side either 'high' (Default) or 'low'
    def compute_pvals(self, exp_arr, side='high'):
        n_samples = float(self.__sim_mat.shape[0])
        p_vals = np.ones(shape=len(exp_arr), dtype=np.float32)
        if n_samples > 0:
            if side == 'low':
                for i, exp in enumerate(exp_arr):
                    p_vals[i] = float((self.__sim_mat[:, i] >= exp).sum()) / n_samples
            else:
                for i, exp in enumerate(exp_arr):
                    p_vals[i] = float((self.__sim_mat[:, i] <= exp).sum()) / n_samples
        return p_vals


############################################################################
# Container a list of simulations
#
class ListSimulations(object):

    def __init__(self):
        self.__sim_tomos = dict()

    def get_simulation(self, key):
        return self.__sim_tomos[key]

    # key: string key to index this simulation
    # sim: Simulation object to insert
    def insert_simulation(self, key, sim):

        # Input parsing
        if not isinstance(sim, Simulation):
            error_msg = 'Input sim must be a Simulation object.'
            raise pexceptions.PySegInputError(expr='insert_simulation (ListSimulations)', msg=error_msg)
        if not isinstance(key, str):
            error_msg = 'Input key must be a string.'
            raise pexceptions.PySegInputError(expr='insert_simulation (ListSimulations)', msg=error_msg)
        if len(self.__sim_tomos.values()) > 0:
            first_sim = self.__sim_tomos.values()[0]
            if not first_sim.is_compatible(sim):
                error_msg = 'Input simulation is not compatible so it cannot be inserted.'
                raise pexceptions.PySegInputError(expr='insert_simulation (ListSimulations)', msg=error_msg)

        # Inserting
        self.__sim_tomos[key] = sim


############################################################################
# Container for a set of lists of simulations
#
class SetListSimulations(object):

    def __init__(self):
        self.__sim_lists = dict()

    def get_list_simulations(self, key):
        return self.__sim_lists[key]

    # key: string key to index this list of simulations
    # sims: ListSimulations object to insert
    def insert_list_simulations(self, key, sims):

        # Input parsing
        if not isinstance(sims, ListSimulations):
            error_msg = 'Input sims must be a Simulation object.'
            raise pexceptions.PySegInputError(expr='insert_lists_simulations (SetListSimulations)', msg=error_msg)
        if not isinstance(key, str):
            error_msg = 'Input key must be a string.'
            raise pexceptions.PySegInputError(expr='insert_lists_simulations (SetListSimulations)', msg=error_msg)

        # Inserting
        self.__sim_lists[key] = sims

############################################################################
# Class for holding particle information and saving memory
#
class ParticleL(object):

    def __init__(self, vtp_file, center=(0., 0., 0.), eu_angs=(0., 0., 0.)):
        """

        :param vtp_file: the full path to the reference vtkPolyData (closed surface)
        :param center: coordinates for the particle center (default (0., 0., 0.))
        :param eu_angs: Euler Angles in Relion format
        :param img_name: path to a sub-volume (default None)
        """

        # Input parsing
        if (not hasattr(center, '__len__')) or (len(center) != 3):
            error_msg = 'Input center must be a 3-tuple!'
            raise pexceptions.PySegInputError(expr='__init__ (ParticleL)', msg=error_msg)
        if (not hasattr(eu_angs, '__len__')) or (len(eu_angs) != 3):
            error_msg = 'Input Euler angles must be a 3-tuple!'
            raise pexceptions.PySegInputError(expr='__init__ (ParticleL)', msg=error_msg)
        try:
            hold_vtp = disperse_io.load_poly(vtp_file)
        except pexceptions.PySegInputError:
            error_msg = 'File ' + vtp_file + ' could not be read!'
            raise pexceptions.PySegInputError(expr='__init__ (ParticleL)', msg=error_msg)
        if not isinstance(hold_vtp, vtk.vtkPolyData):
            error_msg = 'File ' + vtp_file + ' must be a vtkPolyData object'
            raise pexceptions.PySegInputError(expr='__init__ (ParticleL)', msg=error_msg)
        del hold_vtp
        self.__vtp_fname = vtp_file
        self.__center = np.asarray(center, dtype=np.float32)
        self.__eu_angs = np.asarray(eu_angs, dtype=np.float32)
        self.__props = dict()
        self.__meta = dict()

        # Compute bounds
        self.__bounds = np.zeros(shape=6, dtype=np.float32)
        hold_vtp = self.get_vtp()
        arr = hold_vtp.GetPoints().GetData()
        self.__bounds[0], self.__bounds[1] = arr.GetRange(0)
        self.__bounds[2], self.__bounds[3] = arr.GetRange(1)
        self.__bounds[4], self.__bounds[5] = arr.GetRange(2)
        # if not self.point_in_bounds(self.__center):
        #     error_msg = 'Input center must be within surface bounds!'
        #     raise pexceptions.PySegInputError(expr='__init__ (Particle)', msg=error_msg)

    # EXTERNAL FUNCTIONLITY

    def get_vtp_fname(self):
        return self.__vtp_fname

    def get_meta(self):
        return self.__meta

    def get_bounds(self):
        return self.__bounds

    def get_center(self):
        return self.__center

    def get_eu_angs(self):
        return self.__eu_angs

    def get_vtp(self):
        hold_vtp = self.__load_ref()
        hold_vtp = self.__add_props_vtp(hold_vtp)
        return self.__transform(hold_vtp)

    def get_surf_point(self, point_id, ref_vtp):
        """
        Returns the coordinates of a point in the Particle surface
        :param point_id: VTK point ID in the reference vtkPolyData object
        :param ref_vtp: if used (default None) then the reference surface does not need to be loaded from disk,
        useful for accelerating repeated calls.
        :return:
        """

        # Initialization
        if ref_vtp is None:
            ref_vtp = disperse_io.load_poly(self.__vtp_fname)
        point = np.asarray(ref_vtp.GetPoint(point_id))
        point_vtp = point_to_poly(point)

        # Rotation
        rot, tilt, psi = self.__eu_angs
        M = gl.rot_mat_relion(rot, tilt, psi, deg=True)
        M = M.T
        mat_rot = vtk.vtkMatrix4x4()
        for i in range(3):
            for j in range(3):
                mat_rot.SetElement(i, j, M[i, j])
        rot_tr = vtk.vtkTransform()
        rot_tr.SetMatrix(mat_rot)
        tr_rot = vtk.vtkTransformPolyDataFilter()
        tr_rot.SetInputData(point_vtp)
        tr_rot.SetTransform(rot_tr)
        tr_rot.Update()
        point_vtp = tr_rot.GetOutput()

        # Translation
        shift_x, shift_y, shift_z = self.__center
        box_tr = vtk.vtkTransform()
        box_tr.Translate(shift_x, shift_y, shift_z)
        tr_box = vtk.vtkTransformPolyDataFilter()
        tr_box.SetInputData(point_vtp)
        tr_box.SetTransform(box_tr)
        tr_box.Update()
        point_vtp = tr_box.GetOutput()

        return point_vtp.GetPoints().GetPoint(0)

    def add_meta(self, meta):
        for key, val in zip(meta.iterkeys(), meta.itervalues()):
            self.__meta[key] = val

    def add_prop(self, key, vtk_type, val):
        # Input parsing
        if not issubclass(vtk_type, vtk.vtkAbstractArray):
            error_msg = 'Input vtk_type must be child class of vtkAbstractArray!'
            raise pexceptions.PySegInputError(expr='add_vtp_global_attribute (ParticleL)', msg=error_msg)
        self.__props[key] = (vtk_type, val)

    # Check if a point with in the particle bounds
    # point: point to check
    def point_in_bounds(self, point):
        x_over, y_over, z_over = True, True, True
        if (self.__bounds[0] > point[0]) or (self.__bounds[1] < point[0]):
            x_over = False
        if (self.__bounds[2] > point[1]) or (self.__bounds[3] < point[1]):
            y_over = False
        if (self.__bounds[4] > point[2]) or (self.__bounds[5] < point[2]):
            y_over = False
        return x_over and y_over and z_over

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

    def to_Particle(self):
        """
        Get a copy class Particle instance equivalent to self but
        :return:
        """
        part_vtp = disperse_io.load_poly(self.__vtp_fname)
        part = Particle(part_vtp,center=(0,0,0))
        part.rotation(self.__eu_angs[0], self.__eu_angs[1], self.__eu_angs[2])
        part.translation(self.__center[0], self.__center[1], self.__center[2])
        return part

    # fname: file name ended with .pkl
    def pickle(self, fname):
        pkl_f = open(fname, 'w')
        try:
            pickle.dump(self, pkl_f)
        finally:
            pkl_f.close()

    # INTERNAL FUNCTIONALITY

    # Load ref vtkPolyData file form disk
    def __load_ref(self):
        return disperse_io.load_poly(self.__vtp_fname)

    # Applies rigid tranformations
    def __transform(self, vtp):

        # Rotation
        M = gl.rot_mat_relion(self.__eu_angs[0], self.__eu_angs[1], self.__eu_angs[2], deg=True)
        M = M.T
        mat_rot = vtk.vtkMatrix4x4()
        for i in range(3):
            for j in range(3):
                mat_rot.SetElement(i, j, M[i, j])
        rot_tr = vtk.vtkTransform()
        rot_tr.SetMatrix(mat_rot)
        tr_rot = vtk.vtkTransformPolyDataFilter()
        tr_rot.SetInputData(vtp)
        tr_rot.SetTransform(rot_tr)
        tr_rot.Update()
        vtp = tr_rot.GetOutput()

        # Translation
        box_tr = vtk.vtkTransform()
        box_tr.Translate(self.__center[0], self.__center[1], self.__center[2])
        tr_box = vtk.vtkTransformPolyDataFilter()
        tr_box.SetInputData(vtp)
        tr_box.SetTransform(box_tr)
        tr_box.Update()
        return tr_box.GetOutput()

    # Add properties
    def __add_props_vtp(self, vtp):

        for key, propt in zip(self.__props.iterkeys(), self.__props.itervalues()):

            # Initialization
            if isinstance(propt[1], str):
                t_value, n_comp = propt[1], 1
            elif not isinstance(propt[1], tuple):
                t_value, n_comp = (propt[1],), 1
            else:
                t_value, n_comp = propt[1], len(propt[1])
            prop = propt[0]()
            prop.SetNumberOfComponents(n_comp)
            prop.SetName(str(key))

            # Array construction
            n_cells = vtp.GetNumberOfCells()
            prop.SetNumberOfTuples(n_cells)
            for i in range(n_cells):
                # prop.SetValue(i, t_value)
                prop.SetTuple(i, t_value)

            # Adding the data
            vtp.GetCellData().AddArray(prop)

        return vtp


############################################################################
# Class for a Particle: surface, center and normals
#
class Particle(object):

    def __init__(self, vtp, center, normal=(0., 0., 0.)):
        """
        :param vtp: coordinates for the particle center
        :param center: vtkPolyData object with the surface or the full path to a file with a Surface object stored
        :param normal: normal vector (default (0, 0, 0))
        :param img_name: path to a sub-volume (default None)
        """

        # Input parsing
        if isinstance(vtp, str):
            self.__vtp = disperse_io.load_vtp(vtp)
        else:
            self.__vtp = vtk.vtkPolyData()
            self.__vtp.DeepCopy(vtp)
        if not isinstance(self.__vtp, vtk.vtkPolyData):
            error_msg = 'Input vtp is not a vtkPolyData (or a path to it)'
            raise pexceptions.PySegInputError(expr='__init__ (Particle)', msg=error_msg)
        if not is_closed_surface(self.__vtp):
            error_msg = 'Input vtp must be a closed surface!'
            raise pexceptions.PySegInputError(expr='__init__ (Particle)', msg=error_msg)
        if (not hasattr(center, '__len__')) or (len(center) != 3):
            error_msg = 'Input center must be a 3-tuple!'
            raise pexceptions.PySegInputError(expr='__init__ (Particle)', msg=error_msg)
        if (normal is not None) and ((not hasattr(center, '__len__')) or (len(center) != 3)):
            error_msg = 'Input normal must be a 3-tuple!'
            raise pexceptions.PySegInputError(expr='__init__ (Particle)', msg=error_msg)
        self.__fname = None
        self.__vtp_fname = None
        self.__center = point_to_poly(center, normal=normal)
        self.__center_fname = None
        self.__meta = dict()

        # Pre-compute bounds for accelerate computations
        self.__bounds = np.zeros(shape=6, dtype=np.float32)
        self.__update_bounds()
        # if not self.point_in_bounds(self.get_center()):
        #     error_msg = 'Input center must be within surface bounds!'
        #     raise pexceptions.PySegInputError(expr='__init__ (Particle)', msg=error_msg)

        # Create distance properties
        self.__obj_type = None
        self.__loc_vect, self.__gl_vect = None, None
        self.__loc_dst, self.__gl_dst = None, None
        self.__ref_loc_p, self.__ref_loc_v = None, None
        self.__centers = None
        self.clean_properties()

    #### Set/Get functionality

    # normal: normal vector (default (0, 0, 0))
    def set_normal(self, normal=(0., 0., 0.)):
        if (not hasattr(normal, '__len__')) or (len(normal) != 3):
            error_msg = 'Input normal must be 3-tuple'
            raise pexceptions.PySegInputError(expr='set_normal (Particle)', msg=error_msg)
        prop_n = self.__center.GetPointData().GetArray(0)
        prop_n.FillComponent(0, normal[0])
        prop_n.FillComponent(1, normal[1])
        prop_n.FillComponent(2, normal[2])

    def get_meta(self):
        return self.__meta

    # mode: kind of poly to store, valid: 'surface' (default) or 'center'
    def get_vtp(self, mode='surface'):
        # Input parsing
        if mode == 'surface':
            return self.__vtp
        elif mode == 'center':
            return self.__center
        else:
            error_msg = 'Input mode ' + str(mode) + ' is invalid!'
            raise pexceptions.PySegInputError(expr='get_vtp (Particle)', msg=error_msg)

    def get_fname(self):
        return self.__fname

    # Returns: surface bounds (x_min, x_max, y_min, y_max, z_min, z_max) as array
    def get_bounds(self):
        return self.__bounds

    def get_center(self):
        return self.__center.GetPoints().GetPoint(0)

    def get_normal(self):
        return self.__center.GetPointData().GetArray(0).GetTuple(0)

    def get_surf_points(self):
        """
        Get the coordinates of the surface points
        :return: an array [n, 3] where n is the number of points
        """
        points_arr = np.zeros(shape=(self.__vtp.GetNumberOfPoints(), 3), dtype=np.float32)
        for i in range(self.__vtp.GetNumberOfPoints()):
            points_arr[i, :] = self.__vtp.GetPoint(i)
        return points_arr

    #### External functionality

    def add_meta(self, meta):
        for key, val in zip(meta.iterkeys(), meta.itervalues()):
            self.__meta[key] = val

    # Add and attribute with the same value for all cells
    # name: attribute name
    # vtk_type: a child vtkAbstractArray for data type
    # value: tuple with the value
    def add_vtp_global_attribute(self, name, vtk_type, value):

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

    # Surface mesh decimation
    # dec: Specify the desired reduction in the total number of polygons, default None (not applied)
    #      (e.g., if TargetReduction is set to 0.9,
    #      this filter will try to reduce the data set to 10% of its original size).
    def decimation(self, dec):
        self.__vtp = poly_decimate(self.__vtp, dec)

    def scale(self, scale_x, scale_y, scale_z):
        """
        Scaling tranformation
        :param scale_x|y|z: scaling factor in X|Y|Z
        :return:
        """

        # Transformation on the PolyData
        box_tr = vtk.vtkTransform()
        box_tr.Scale(scale_x, scale_y, scale_z)
        tr_box = vtk.vtkTransformPolyDataFilter()
        tr_box.SetInputData(self.__vtp)
        tr_box.SetTransform(box_tr)
        tr_box.Update()
        self.__vtp = tr_box.GetOutput()

        # Update center (does not apply to normals)
        box_tr = vtk.vtkTransform()
        box_tr.Translate(scale_x, scale_y, scale_z)
        tr_box = vtk.vtkTransformPolyDataFilter()
        tr_box.SetInputData(self.__center)
        tr_box.SetTransform(box_tr)
        tr_box.Update()
        self.__center = tr_box.GetOutput()

        # Update the bounds
        self.__update_bounds()

    # Rigid translation
    # shift_i: translations in X, Y, Z axes respectively
    def translation(self, shift_x, shift_y, shift_z):

        # Transformation on the PolyData
        box_tr = vtk.vtkTransform()
        box_tr.Translate(shift_x, shift_y, shift_z)
        tr_box = vtk.vtkTransformPolyDataFilter()
        tr_box.SetInputData(self.__vtp)
        tr_box.SetTransform(box_tr)
        tr_box.Update()
        self.__vtp = tr_box.GetOutput()

        # Update center (does not apply to normals)
        box_tr = vtk.vtkTransform()
        box_tr.Translate(shift_x, shift_y, shift_z)
        tr_box = vtk.vtkTransformPolyDataFilter()
        tr_box.SetInputData(self.__center)
        tr_box.SetTransform(box_tr)
        tr_box.Update()
        self.__center = tr_box.GetOutput()

        # Update the bounds
        self.__update_bounds()

    # Rigid rotation
    # rot, tilt, psi: rotation angles in Relion format in degrees
    # active: if True the rotation is active, otherwise it is possible
    def rotation(self, rot, tilt, psi, active=True):

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

        # Update center
        hold_center = np.asarray(self.get_center(), dtype=np.float)
        rot_tr = vtk.vtkTransform()
        rot_tr.SetMatrix(mat_rot)
        tr_rot = vtk.vtkTransformPolyDataFilter()
        tr_rot.SetInputData(self.__center)
        tr_rot.SetTransform(rot_tr)
        tr_rot.Update()
        self.__center = tr_rot.GetOutput()

        # Normal rotation
        hold_normal = np.asarray(self.get_normal(), dtype=np.float) + hold_center
        rot_center, rot_normal = M*hold_center.reshape(3, 1), M*hold_normal.reshape(3, 1)
        hold_normal = np.asarray((rot_normal - rot_center), dtype=np.float).reshape(3)
        # Re-normalization
        norm_normal = math.sqrt((hold_normal*hold_normal).sum())
        if norm_normal > 0:
            self.set_normal(hold_normal / norm_normal)

        # Update the bounds
        self.__update_bounds()

    # Swap the coordinates of XY axes
    def swap_xy(self):
        # Swap surface points
        self.__vtp = poly_swapxy(self.__vtp)
        # Swap the centers
        self.__center = poly_swapxy(self.__center)
        # Swap the normal vector
        hold_normal = self.get_normal()
        self.set_normal((hold_normal[1], hold_normal[0], hold_normal[2]))
        # Swap the bounds by updating
        self.__update_bounds()

    # Saves the object poly in disk as a *.vtp file
    # fname: full path of the output file
    # mode: kind of poly to store, valid: 'surface' (default) or 'center'
    def store_vtp(self, fname, mode='surface'):

        # Input parsing
        if mode == 'surface':
            disperse_io.save_vtp(self.__vtp, fname)
            self.__fname = fname
        elif mode == 'center':
            disperse_io.save_vtp(self.__center, fname)
        else:
            error_msg = 'Input mode ' + str(fname) + ' is invalid!'
            raise pexceptions.PySegInputError(expr='store_vtp (Particle)', msg=error_msg)

    # Check if the object's bound are in a sphere with some radius
    # center: sphere center coordinates
    # radius: sphere radius
    def bound_in_radius(self, center, radius):

        # Computing min corner distance
        hold_x, hold_y, hold_z = float(center[0])-self.__bounds[0], float(center[1])-self.__bounds[2], \
                                 float(center[2])-self.__bounds[4]
        dst_min = math.sqrt(hold_x*hold_x + hold_y*hold_y + hold_z*hold_z)

        # Computing max corner distance
        hold_x, hold_y, hold_z = float(center[0]) - self.__bounds[1], float(center[1]) - self.__bounds[3], \
                                 float(center[2]) - self.__bounds[5]
        dst_max = math.sqrt(hold_x * hold_x + hold_y * hold_y + hold_z * hold_z)

        # Check if at least one side is in the sphere
        if (radius <= dst_min) or (radius <= dst_max):
            return True
        else:
            return False

    # Check if the object's bound are at least partially in another bound
    # bounds: input bound
    def bound_in_bounds(self, bounds):
        x_over, y_over, z_over = True, True, True
        if (self.__bounds[0] > bounds[1]) or (self.__bounds[1] < bounds[0]):
            x_over = False
        if (self.__bounds[2] > bounds[3]) or (self.__bounds[3] < bounds[2]):
            y_over = False
        if (self.__bounds[4] > bounds[5]) or (self.__bounds[5] < bounds[4]):
            y_over = False
        return x_over and y_over and z_over

    # Check if a point with in the particle bounds
    # point: point to check
    def point_in_bounds(self, point):
        x_over, y_over, z_over = True, True, True
        if (self.__bounds[0] > point[0]) or (self.__bounds[1] < point[0]):
            x_over = False
        if (self.__bounds[2] > point[1]) or (self.__bounds[3] < point[1]):
            y_over = False
        if (self.__bounds[4] > point[2]) or (self.__bounds[5] < point[2]):
            y_over = False
        return x_over and y_over and z_over

    # Clean vector and distance properties
    def clean_properties(self):
        self.__obj_type = vtk.vtkIntArray()
        self.__obj_type.SetName(OBJ_TYPE)
        self.__obj_type.SetNumberOfComponents(1)
        self.__loc_vect = vtk.vtkFloatArray()
        self.__loc_vect.SetName(SF_LOC_V)
        self.__loc_vect.SetNumberOfComponents(3)
        self.__gl_vect = vtk.vtkFloatArray()
        self.__gl_vect.SetName(SF_GL_V)
        self.__gl_vect.SetNumberOfComponents(3)
        self.__centers = vtk.vtkFloatArray()
        self.__centers.SetName(SF_CC)
        self.__centers.SetNumberOfComponents(3)
        self.__loc_dst = vtk.vtkFloatArray()
        self.__loc_dst.SetName(SF_LOC_D)
        self.__loc_dst.SetNumberOfComponents(1)
        self.__gl_dst = vtk.vtkFloatArray()
        self.__gl_dst.SetName(SF_GL_D)
        self.__gl_dst.SetNumberOfComponents(1)
        for i in range(self.__vtp.GetNumberOfCells()):
            self.__obj_type.InsertTuple(i, (OBJ_PART,))
            self.__loc_vect.InsertTuple(i, (SF_DEF_VAL, SF_DEF_VAL, SF_DEF_VAL))
            self.__gl_vect.InsertTuple(i, (SF_DEF_VAL, SF_DEF_VAL, SF_DEF_VAL))
            self.__centers.InsertTuple(i, (SF_DEF_VAL, SF_DEF_VAL, SF_DEF_VAL))
            self.__loc_dst.InsertTuple(i, (SF_DEF_VAL,))
            self.__gl_dst.InsertTuple(i, (SF_DEF_VAL,))
        self.__vtp.GetCellData().AddArray(self.__obj_type)
        self.__vtp.GetCellData().AddArray(self.__centers)
        self.__vtp.GetCellData().AddArray(self.__gl_vect)
        self.__vtp.GetCellData().AddArray(self.__loc_vect)
        self.__vtp.GetCellData().AddArray(self.__gl_dst)
        self.__vtp.GetCellData().AddArray(self.__gl_dst)
        self.__ref_loc_p, self.__ref_loc_v = None, None

    # Compute cell's centers
    def compute_centers(self):
        points, cells = self.__vtp.GetPoints(), self.__vtp.GetPolys()
        for i in range(cells.GetNumberOfCells()):
            id_l = vtk.vtkIdList()
            cells.GetCell(i, id_l)
            center, n_ids = np.zeros(shape=3, dtype=np.float32), float(id_l.GetNumberOfIds())
            for j in range(n_ids):
                hold_point = points.GetPoint(id_l.GetId(j))
                center += np.asarray(hold_point, dtype=np.float32)
            center /= n_ids
            self.__centers.InsertTuple(i, center)

    # Update vector properties against an input surface
    # surf: input Surface
    # max_dst: maximum distance to look at
    def update_vectors(self, surf, max_dst):

        # Initialization
        locator = vtk.vtkCellLocator()
        locator.SetDataSet(surf)

        # Loop for computing local vectors
        min_dst, min_loc_p, min_loc_v = np.inf, None, None
        points, cells = self.__vtp.GetPoints(), self.__vtp.GetPolys()
        for i in range(cells.GetNumberOfCells()):

            # Finding cell's center
            center = np.asarray(self.__centers.GetTuple(i), dtype=np.float32)

            # Finding the closest distance
            c_dst, c_point = np.inf, np.inf*np.ones(shape=3, dtype=np.float)
            locator.FindClosestPointWithinRadius(center, max_dst, c_point, None, None, c_dst)
            if c_dst < np.inf:
                loc_vect = c_point - center
                if c_dst < min_dst:
                    min_dst = c_dst
                    min_loc_p, min_loc_v = center, loc_vect
                # Set the surface property
                self.__loc_vect.SetTuple(i, loc_vect)

        # Cell in valid radius
        if min_loc_v is None:
            return
        self.__ref_loc_p, self.__ref_loc_v = min_loc_p, min_loc_v

        # Loop for computing global vectors
        for i in range(cells.GetNumberOfCells()):

            # Finding cell's center
            center = np.asarray(self.__centers.GetTuple(i), dtype=np.float32)

            # Computing the global vector
            gl_vect = self.__ref_loc_p - center
            gl_vect += self.__ref_loc_v

            # Set the surface property
            self.__gl_vect.SetTuple(i, gl_vect)

    # Update distance properties from current vectors properties
    def update_distances(self):

        # Loop for cells
        points, cells = self.__vtp.GetPoints(), self.__vtp.GetPolys()
        for i in range(cells.GetNumberOfCells()):

            # Getting the vectors
            loc_vect = np.asarray(self.__loc_vect.GetTuple(i), dtype=np.float32)
            gl_vect = np.asarray(self.__gl_vect.GetTuple(i), dtype=np.float32)

            # Computing distances
            loc_dst = math.sqrt((loc_vect * loc_vect).sum())
            gl_dst = math.sqrt((gl_vect * gl_vect).sum())

            # Set the surface property
            self.__loc_dst.SetTuple1(i, loc_dst)
            self.__gl_dst.SetTuple1(i, gl_dst)

    # VTK attributes requires a special treatment during pickling
    # fname: file name ended with .pkl
    def pickle(self, fname):

        # Dump pickable objects and store the file names of the unpickable objects
        stem, ext = os.path.splitext(fname)
        self.__vtp_fname = stem + '_surf.vtp'
        self.__center_fname = stem + '_center.vtp'
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
            raise pexceptions.PySegInputError(expr='pickle (Particle)', msg=error_msg)
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(self.__center_fname)
        vtk_ver = vtk.vtkVersion().GetVTKVersion()
        if int(vtk_ver[0]) < 6:
            writer.SetInput(self.__center)
        else:
            writer.SetInputData(self.__center)
        if writer.Write() != 1:
            error_msg = 'Error writing %s.' % self.__center_fname
            raise pexceptions.PySegInputError(expr='pickle (Particle)', msg=error_msg)

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
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(self.__center_fname)
        reader.Update()
        self.__center = reader.GetOutput()
        self.__obj_type = self.__vtp.GetCellData().GetArray(OBJ_TYPE)
        self.__loc_vect = self.__vtp.GetCellData().GetArray(SF_LOC_V)
        self.__gl_vect = self.__vtp.GetCellData().GetArray(SF_GL_V)
        self.__loc_dst = self.__vtp.GetCellData().GetArray(SF_LOC_D)
        self.__gl_dst = self.__vtp.GetCellData().GetArray(SF_GL_D)
        self.__centers = self.__vtp.GetCellData().GetArray(SF_CC)

    # Copy the object's state from self.__dict__ which contains all instance attributes.
    # Afterwards remove unpickable objects
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_Particle__vtp']
        del state['_Particle__center']
        del state['_Particle__obj_type']
        del state['_Particle__loc_vect']
        del state['_Particle__gl_vect']
        del state['_Particle__loc_dst']
        del state['_Particle__gl_dst']
        del state['_Particle__centers']
        return state


############################################################################
# Class for a set of Particles embedded in the same Tomogram (Euclidean 3D space)
#
class TomoParticles(object):

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
        self.__parts = list()
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
        self.__parts_fname = None

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

    def get_particles(self):
        return self.__parts

    def get_particle_coords(self):
        if len(self.__parts) <= 0:
            return None
        hold_parts = np.zeros(shape=(self.get_num_particles(), len(self.__parts[0].get_center())), dtype=np.float32)
        for i, part in enumerate(self.__parts):
            hold_parts[i, :] = part.get_center()
        return hold_parts

    def get_num_particles(self):
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

    # Checks intersection with other already inserted particles
    # part: input particle
    def check_particles_itersection(self, part):

        # Initialization
        selector = vtk.vtkSelectEnclosedPoints()
        selector.SetTolerance(VTK_RAY_TOLERANCE)
        # print(selector.GetTolerance())
        selector.Initialize(part.get_vtp())

        # # Particles loop, for the shake of speed first check bounds ovelapping
        # # Checking by points
        # for host_part in self.__parts:
        #     if part.bound_in_bounds(host_part.get_bounds()):
        #         poly_b = host_part.get_vtp(mode='surface')
        #         for i in range(poly_b.GetNumberOfPoints()):
        #             if selector.IsInsideSurface(poly_b.GetPoint(i)) > 0:
        #                 return True

        # Particles loop, for the shake of speed first check bounds ovelapping
        # Checking by cells
        over = 0
        for host_part in self.__parts:
            if part.bound_in_bounds(host_part.get_bounds()):
                poly_b = host_part.get_vtp()
                count, n_points = 0., poly_b.GetNumberOfPoints()
                n_points_f = float(n_points)
                for i in range(n_points):
                    if selector.IsInsideSurface(poly_b.GetPoint(i)) > 0:
                        count += 1
                        over = count / n_points_f
                        if over > OVER_TOLERANCE:
                            # print('Overlapping: ' + str(100. * over) + ' %')
                            return True

        # print('Overlapping: ' + str(100. * over) + ' %')
        return False

    def insert_particle(self, part, check_bounds=True, mode='full', check_inter=False, voi_pj=False,
                        voi_to_pj=None, meta=None):
        """

        :param part: particle to insert in the tomogram, it must be fully embedded by the tomogram
        :param check_bounds: exception if input surface is not fully embedded
        :param mode: to disable bounds checking (default True)
        :param check_inter: embedding mode, only applies if check_bounds is True (see is_embedded)
        :param voi_pj: if True (default False) particle center is projected on tomogram voi thourgh the normal to its
        closest point.
        :param voi_to_pj: if not None (default) then the VOI where the particle is projected, only
            valid if voi_pj=True and used to set a VOI different from self's VOI. The only valid
            VOI is a vtkPolyData
        :param meta: a dictionary with meta information for the particle (default None)
        :return:
        """

        # Input parsing
        if (not isinstance(part, Particle)) and (not isinstance(part, ParticleL)):
            error_msg = 'Input object must be a Surface instance.'
            raise pexceptions.PySegInputError(expr='insert_surface (TomoParticles)', msg=error_msg)
        if check_inter and self.check_particles_itersection(part):
            error_msg = 'This particle intersect with another already inserted one.'
            raise pexceptions.PySegInputError(expr='insert_surface (TomoParticles)', msg=error_msg)

        # VOI projection?
        part_cent = part.get_center()
        if voi_pj:
            if voi_to_pj is None:
                if isinstance(self.__voi, vtk.vtkPolyData):
                    voi_cent = vtp_closest_point(self.__voi, part_cent)
                else:
                    x, y, z = int(round(part_cent[0])), int(round(part_cent[1])), int(round(part_cent[2]))
                    try:
                        voi_cent = self.__voi_dst_ids[:, x, y, z]
                    except IndexError:
                        error_msg = 'Particle out of numpy mask.'
                        raise pexceptions.PySegInputError(expr='insert_surface (TomoParticles)', msg=error_msg)
            else:
                if isinstance(voi_to_pj, vtk.vtkPolyData):
                    voi_cent = vtp_closest_point(voi_to_pj, part_cent)
                else:
                    error_msg = 'Projection VOI must be a vtkPolyData.'
                    raise pexceptions.PySegInputError(expr='insert_surface (TomoParticles)', msg=error_msg)
            if isinstance(part, Particle):
                shifts = np.asarray(voi_cent, dtype=float) - np.asarray(part_cent, dtype=float)
                part.translation(shifts[0], shifts[1], shifts[2])
            else:
                part = ParticleL(part.get_vtp_fname(), voi_cent, part.get_eu_angs())
        if meta is not None:
            part.add_meta(meta)

        # Insert to list
        if check_bounds and (not self.is_embedded(part, mode)):
            error_msg = 'Input surface is not fully embedded in the reference tomogram.'
            raise pexceptions.PySegInputError(expr='insert_surface (TomoParticles)', msg=error_msg)
        self.__parts.append(part)

    def delete_particles(self, part_ids):
        """
        Remove  particles from the list and delete them
        :param part_ids: integer ids of the particles (their position in the current list of particles)
        :return: None
        """
        # Loop for keeping survivors
        hold_parts = list()
        for i in range(len(self.__parts)):
            if not i in part_ids:
                hold_parts.append(self.__parts[i])

        # Updating the list of particles
        self.__parts = hold_parts

    # Compute the vectors and distance properties of an input Surface respect tomogram's surfaces
    # part: input part to compute its surface properties
    # max_dst: maximum distance to search for neighbor surface, if None it is set automatically
    #          to reference tomogram maximum diagonal
    # ex_l: exclusion list with the indices of Surfaces to exclude (default None)
    # Returns: properties of input Surface object are (re-)computed
    def compute_surface_props(self, part, max_dst=None, ex_l=None):

        # Initialization
        if ex_l is None:
            surfs_l = list(self.__parts)
        else:
            lut_ex = np.ones(shape=len(self.__parts), dtype=np.bool)
            for ex_id in ex_l:
                try:
                    lut_ex[ex_id] = False
                except IndexError:
                    print 'WARNING: compute_surface_props (TomoSurfaces): exclusion index not present!'
            surfs_l = list()
            for i, hold_surf in enumerate(self.__parts):
                if lut_ex[i]:
                    surfs_l.append(hold_surf)
        ex_bounds = self.get_extended_bounds(part, max_dst)

        # Vector properties loop
        for hold_surf in surfs_l:
            if hold_surf.is_in_bounds(ex_bounds):
                part.update_vectors(hold_surf, max_dst)

        # Update distances properties
        part.update_distances()

    # VTK attributes requires a special treatment during pickling
    # fname: file name ended with .pkl
    def pickle(self, fname):

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
        self.__parts_fname = stem_path + '/' + stem_ + '_parts.star'
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
                raise pexceptions.PySegInputError(expr='pickle (TomoParticles)', msg=error_msg)
        else:
            np.save(self.__voi_fname, self.__voi)
            np.save(self.__voi_ids_fname, self.__voi_dst_ids)

        # Store particles list
        self.particles_pickle(self.__parts_fname)

    # Pickle the list of particles and stores a STAR file for listing them
    # out_fname: file name for the ouput STAR file
    def particles_pickle(self, out_fname):

        # STAR file for the particles
        star = sub.Star()
        star.add_column('_suSurfaceVtp')

        # Create the directory for particles
        out_dir = os.path.splitext(out_fname)[0]
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Tomograms loop
        for i, part in enumerate(self.__parts):

            # Pickling the particle
            pkl_file = out_dir + '/particle_' + str(i) + '.pkl'
            part.pickle(pkl_file)

            # Adding a new enter
            kwargs = {'_suSurfaceVtp': pkl_file}
            star.add_row(**kwargs)

        # Storing the tomogram STAR file
        star.store(out_fname)

    # Load pickled Particles listed by a STAR file
    # in_fname: file name for the input STAR file
    def load_particles_pickle(self, in_fname):

        # Initialization
        self.__parts = list()

        # STAR file for the particles
        star = sub.Star()
        star.load(in_fname)

        # Rows loop
        for row in range(star.get_nrows()):

            # Loading the particles
            self.__parts.append(unpickle_obj(star.get_element('_suSurfaceVtp', row)))

    # Generates a vtkPolyData object with all particles
    # mode: kind of poly to store, valid: 'surface' (default) or 'center'
    def gen_particles_vtp(self, mode='surface'):

        # Initialization
        appender = vtk.vtkAppendPolyData()

        for i, part in enumerate(self.get_particles()):

            if isinstance(part, ParticleL):
                hold_vtp = part.get_vtp()
            else:
                hold_vtp = part.get_vtp(mode=mode)

            # print mode, str(hold_vtp.GetNumberOfCells())

            # Add particle id property
            part_id = vtk.vtkIntArray()
            part_id.SetName(PT_ID)
            part_id.SetNumberOfComponents(1)
            part_id.SetNumberOfTuples(hold_vtp.GetNumberOfCells())
            part_id.FillComponent(0, i)
            hold_vtp.GetCellData().AddArray(part_id)

            # Append current particles
            appender.AddInputData(hold_vtp)

        # Fuse to one vtkPolyData
        appender.Update()

        return appender.GetOutput()

    # Generates and instance of the current tomogram from a model
    # model_type: model template for generate the new instance
    # mode: mode for embedding, valid: 'full' (default) and 'center'
    def gen_model_instance(self, model_type, mode='full'):
        model = model_type(self.__voi, self.__parts[0])
        return model.gen_instance(len(self.__parts), self.__fname, mode=mode)

    def compute_uni_2nd_order(self, distances, thick=None, border=True, conv_iter=None, max_iter=None, fmm=False,
                              out_sep=0, dimen=3, rdf=False, npr=None, tmp_folder=None):
        """
        Compute Univariate 2nd order statistics (Ripley's L for spherical nhood_vol and O for shell)
        :param distances: range of distances (considered as and iterable of floats)
        :param thick: nhood_vol thickes if None (default) then 'sphere', otherwise 'shell'
        :param border: if True (default) border compensation is activated
        :param conv_iter: number of iterations for convergence in volume estimation, if None then DSA computations active
        :param max_iter: maximum number of iterations in volume estimation, if None then DSA computations active
        :param fmm: if DSA computation active if False (default) then distance transform is used, otherwise Fast Martching Method
        :param out_sep: if 0 (default) then averaged Ripley's metric is provided
               elif 1 then two arrays with the averaged number of neighbors and volues are provided
               else then two matrices (number of neighbors and local volumes) with an entry for every point are provides
        :param dimen: number of space dimensions, valid: 3 (default) and 2
        :param rbf: if True (default False) then RDF is computed, only considered if thick is not None
        :param npr: number of process, None (defaults) means they are adjusted automatically to the number of cores
        :param tmp_folder: folder for temporarily store VOI, it its only required if the VOI is a vtkPolyData objects
        :return: an array with the statistical metric for every input distance, or two arrays if out_sep is True
        """

        # Input parsing
        if self.get_num_particles() <= 0:
            if out_sep:
                return None, None
            else:
                return None
        dens = self.compute_global_density()
        if dens <= 0:
            if out_sep:
                return None, None
            else:
                return None

        # MULTIPROCESSING
        if npr is None:
            npr = mp.cpu_count()
        if npr > self.get_num_particles():
            npr = self.get_num_particles()
        processes = list()
        # Create the list on indices to split
        spl_ids = np.array_split(range(self.get_num_particles()), npr)
        # Shared arrays
        voi_fname, voi_shape, voi_shared = None, None, None
        hold_voi = self.get_voi()
        if isinstance(hold_voi, vtk.vtkPolyData):
            if tmp_folder is None:
                error_msg = 'When input VOI is a vtkPolyData object tmp_folder is compulsory.'
                raise pexceptions.PySegInputError(expr='compute_uni_2nd_order', msg=error_msg)
            voi_fname = tmp_folder + '/gen_tlist_voi.vtp'
            disperse_io.save_vtp(hold_voi, voi_fname)
        else:
            hold_voi = np.asarray(hold_voi, dtype=np.bool)
            voi_shape = hold_voi.shape
            # voi_shared = mp.RawArray(ctypes.c_bool, hold_voi.reshape(np.array(voi_shape).prod()))
            voi_len = np.array(voi_shape).prod()
            voi_shared_raw = mp.RawArray(np.ctypeslib.ctypes.c_uint8, voi_len)
            voi_shared = np.ctypeslib.as_array(voi_shared_raw)
            voi_shared[:] = hold_voi.reshape(voi_len).astype(np.uint8)
        shared_mat_1 = mp.Array('f', self.get_num_particles()*len(distances))
        shared_mat_2 = mp.Array('f', self.get_num_particles()*len(distances))

        # Get particle points
        parts_centers_1 = np.zeros(shape=(self.get_num_particles(), len(self.__parts[0].get_center())),
                                   dtype=np.float32)
        for i in range(self.get_num_particles()):
            parts_centers_1[i] = self.__parts[i].get_center()
        parts_centers_2 = parts_centers_1

        # Particles loop (Parallel)
        if npr <= 1:
            pr_2nd_tomo(-1, spl_ids[0], parts_centers_1, parts_centers_2, distances, thick, border,
                            conv_iter, max_iter, fmm, False,
                            voi_fname, voi_shape, voi_shared,
                            shared_mat_1, shared_mat_2)
        else:
            for pr_id in range(npr):
                pr = mp.Process(target=pr_2nd_tomo, args=(pr_id, spl_ids[pr_id], parts_centers_1, parts_centers_2, distances, thick, border,
                            conv_iter, max_iter, fmm, False,
                            voi_fname, voi_shape, voi_shared,
                            shared_mat_1, shared_mat_2))
                pr.start()
                processes.append(pr)
            pr_results = list()
            for pr in processes:
                pr.join()
                pr_results.append(pr.exitcode)
            for pr_id in range(len(processes)):
                if pr_id != pr_results[pr_id]:
                    error_msg = 'Process ' + str(pr_id) + ' exited unexpectedly!'
                    raise pexceptions.PySegInputError(expr='compute_uni_2nd_order (TomoParticles)',
                                                      msg=error_msg)
            gc.collect()

        # Recover parallel processed results
        hold_buffer_1 = np.frombuffer(shared_mat_1.get_obj(),
                                    dtype=np.float32).reshape(self.get_num_particles(), len(distances))
        hold_buffer_2 = np.frombuffer(shared_mat_2.get_obj(),
                                    dtype=np.float32).reshape(self.get_num_particles(), len(distances))

        # np.save('/fs/home/martinez/workspace/python/pyorg/surf/test/out/uni_2nd/hold_1_mca_2.npy', hold_buffer_1)
        # np.save('/fs/home/martinez/workspace/python/pyorg/surf/test/out/uni_2nd/hold_2_mca_2.npy', hold_buffer_2)

        # Metric computation
        if out_sep == 0:
            if thick is None:
                if dimen == 2:
                    return distances * (np.sqrt((1. / dens) * (hold_buffer_1.sum(axis=0) / hold_buffer_2.sum(axis=0))) - 1.)
                else:
                    return distances * (np.cbrt((1./dens) * (hold_buffer_1.sum(axis=0) / hold_buffer_2.sum(axis=0))) - 1.)
            else:
                if rdf:
                    return (1. / dens) * (hold_buffer_1.sum(axis=0) / hold_buffer_2.sum(axis=0))
                else:
                    return hold_buffer_1.sum(axis=0) / hold_buffer_2.sum(axis=0)
        elif out_sep == 1:
            return hold_buffer_1.sum(axis=0), hold_buffer_2.sum(axis=0)
        else:
            return hold_buffer_1, hold_buffer_2

    def filter_uni_2nd_order(self, distances, th, mode='ripley', side='high', neigh_min=1, th_nb=None, sims=None, thick=None, border=True,
                             conv_iter=None, max_iter=None, fmm=False, dimen=3, rdf=True, npr=None, tmp_folder=None):
        """
        Filter tomo particles by comparing the Univariate 2nd order statistics with a model
        :param distances:
        :param th: threshold
        :param mode: thresholding mode, either 'ripley' (Default) or 'pval' (by pvalue)
        :param side: side to apply the thesholding (default 'high')
        :param n_neigh: minimum number of neighbours
        :param th_nb: threhold for the number of neighbours
        :param sims: TomoSimulations object, only required if mode='pval' (Default None)
        :param thick, border, conv_iter, max_iter, fmm, dimen, npr: like compute_uni_2nd_order()
        :param tmp_folder: folder for temporarily store VOI, it its only required if the VOI is a vtkPolyData objects
        :return: a filtered copy of the TomoParticles object, estimated univarite result is added as particle property 'uni_2d_max|min'
        """

        # Input parsing
        if self.get_num_particles() <= 0:
            return None
        dens = self.compute_global_density()
        if dens <= 0:
            return None
        if (mode == 'pval') and (sims is None):
            error_msg = 'Simulations are required when mode is pval!'
            raise pexceptions.PySegInputError(expr='filter_uni_2nd_order (TomoParticles)',
                                              msg=error_msg)
        mode_rip = True
        if mode == 'pval':
            mode_rip = False
        side_high = True
        if side == 'low':
            side_high = False

        # MULTIPROCESSING
        if npr is None:
            npr = mp.cpu_count()
        if npr > self.get_num_particles():
            npr = self.get_num_particles()
        processes = list()
        # Create the list on indices to split
        spl_ids = np.array_split(range(self.get_num_particles()), npr)
        # Shared arrays
        voi_fname, voi_shape, voi_shared = None, None, None
        hold_voi = self.get_voi()
        if isinstance(hold_voi, vtk.vtkPolyData):
            if tmp_folder is None:
                error_msg = 'When input VOI is a vtkPolyData object tmp_folder is compulsory.'
                raise pexceptions.PySegInputError(expr='gen_tlist', msg=error_msg)
            voi_fname = tmp_folder + '/gen_tlist_voi.vtp'
            disperse_io.save_vtp(hold_voi, voi_fname)
        else:
            hold_voi = np.asarray(hold_voi, dtype=np.bool)
            voi_shape = hold_voi.shape
            # voi_shared = mp.RawArray(ctypes.c_bool, hold_voi.reshape(np.array(voi_shape).prod()))
            voi_len = np.array(voi_shape).prod()
            voi_shared_raw = mp.RawArray(np.ctypeslib.ctypes.c_uint8, voi_len)
            voi_shared = np.ctypeslib.as_array(voi_shared_raw)
            voi_shared[:] = hold_voi.reshape(voi_len).astype(np.uint8)
        shared_mat_1 = mp.Array('f', self.get_num_particles() * len(distances))
        shared_mat_2 = mp.Array('f', self.get_num_particles() * len(distances))

        # Get particle points
        parts_centers_1 = np.zeros(shape=(self.get_num_particles(), len(self.__parts[0].get_center())),
                                   dtype=np.float32)
        for i in range(self.get_num_particles()):
            parts_centers_1[i] = self.__parts[i].get_center()
        parts_centers_2 = parts_centers_1

        # Particles loop (Parallel)
        if npr <= 1:
            pr_2nd_tomo(-1, spl_ids[0], parts_centers_1, parts_centers_2,
                                                          distances, thick, border,
                                                          conv_iter, max_iter, fmm, False, False,
                                                          voi_fname, voi_shape, voi_shared,
                                                          shared_mat_1, shared_mat_2)
        else:
            for pr_id in range(npr):
                pr = mp.Process(target=pr_2nd_tomo, args=(pr_id, spl_ids[pr_id], parts_centers_1, parts_centers_2,
                                                          distances, thick, border,
                                                          conv_iter, max_iter, fmm, False, False,
                                                          voi_fname, voi_shape, voi_shared,
                                                          shared_mat_1, shared_mat_2))
                pr.start()
                processes.append(pr)
            pr_results = list()
            for pr in processes:
                pr.join()
                pr_results.append(pr.exitcode)
            for pr_id in range(len(processes)):
                if pr_id != pr_results[pr_id]:
                    error_msg = 'Process ' + str(pr_id) + ' exited unexpectedly!'
                    raise pexceptions.PySegInputError(expr='compute_uni_2nd_order (TomoParticles)',
                                                      msg=error_msg)
            gc.collect()

        # Recover parallel processed results
        hold_buffer_1 = np.frombuffer(shared_mat_1.get_obj(),
                                    dtype=np.float32).reshape(self.get_num_particles(), len(distances))
        hold_buffer_2 = np.frombuffer(shared_mat_2.get_obj(),
                                    dtype=np.float32).reshape(self.get_num_particles(), len(distances))

        # Metric computation
        if thick is None:
            if dimen == 2:
                uni_2nd = distances * (np.sqrt((1. / dens) * (hold_buffer_1 / hold_buffer_2)) - 1.)
            else:
                uni_2nd = distances * (np.cbrt((1. / dens) * (hold_buffer_1 / hold_buffer_2)) - 1.)
        else:
            if rdf:
                uni_2nd = (1. / dens) * (hold_buffer_1 / hold_buffer_2)
            else:
                uni_2nd = hold_buffer_1 / hold_buffer_2

        # Filtering by particles
        hold_ltomos = TomoParticles(self.get_tomo_fname(), -1, self.get_voi())
        for i, part in enumerate(self.__parts):
            if mode_rip:
                vals, neighs = uni_2nd[i, :], hold_buffer_1[i, :]
            else:
                vals, neighs = sims.compute_pvals(uni_2nd[i, :], side), hold_buffer_1[i, :]
            val_max, neigh_max = vals.max(), neighs.max()
            # part.add_vtp_global_attribute('uni_2nd', vtk.vtkFloatArray, val_max)
            # part.add_vtp_global_attribute('neighs', vtk.vtkFloatArray, neigh_max)
            if side_high:
                if (th_nb is None) or (neigh_max >= th_nb):
                    if val_max >= th:
                        hold_ltomos.insert_particle(part, check_bounds=False)
            else:
                if (th_nb is None) or (neigh_min <= th_nb):
                    if val_max <= th:
                        hold_ltomos.insert_particle(part, check_bounds=False)

        return hold_ltomos

    def compute_bi_2nd_order(self, tomo, distances, thick=None, border=True, conv_iter=None, max_iter=None, fmm=False,
                             out_sep=0, dimen=3, rdf=False, npr=None, tmp_folder=None):
        """
        Compute Bivariate 2nd order statistics (Ripley's L for spherical nhood_vol and O for shell)
        :param tomo:
        :param rest_of_parameters: like compute_uni_2nd_order
        :return: an array with the statistical metric for every input distance, or two array if out_sep is True
        """

        # Input parsing
        if self.get_num_particles() <= 0:
            if out_sep:
                return None, None
            else:
                return None
        dens = tomo.compute_global_density()
        if dens <= 0:
            if out_sep:
                return None, None
            else:
                return None

        # MULTIPROCESSING
        if npr is None:
            npr = mp.cpu_count()
        processes, pr_ids = list(), list()
        # Create the list on indices to split
        spl_ids = np.array_split(range(self.get_num_particles()), npr)
        # Shared arrays
        voi_fname, voi_shape, voi_shared = None, None, None
        hold_voi = self.get_voi()
        if isinstance(hold_voi, vtk.vtkPolyData):
            if tmp_folder is None:
                error_msg = 'When input VOI is a vtkPolyData object tmp_folder is compulsory.'
                raise pexceptions.PySegInputError(expr='gen_tlist', msg=error_msg)
            voi_fname = tmp_folder + '/gen_tlist_voi.vtp'
            disperse_io.save_vtp(hold_voi, voi_fname)
        else:
            hold_voi = np.asarray(hold_voi, dtype=np.bool)
            voi_shape = hold_voi.shape
            # voi_shared = mp.RawArray(ctypes.c_bool, hold_voi.reshape(np.array(voi_shape).prod()))
            voi_len = np.array(voi_shape).prod()
            voi_shared_raw = mp.RawArray(np.ctypeslib.ctypes.c_uint8, voi_len)
            voi_shared = np.ctypeslib.as_array(voi_shared_raw)
            voi_shared[:] = hold_voi.reshape(voi_len).astype(np.uint8)
        shared_mat_1 = mp.Array('f', self.get_num_particles() * len(distances))
        shared_mat_2 = mp.Array('f', self.get_num_particles() * len(distances))

        # Get particle points
        parts_centers_1 = np.zeros(shape=(self.get_num_particles(), len(self.__parts[0].get_center())),
                                   dtype=np.float32)
        for i in range(self.get_num_particles()):
            parts_centers_1[i] = self.__parts[i].get_center()
        parts_centers_2 = np.zeros(shape=(tomo.get_num_particles(), len(tomo.__parts[0].get_center())), dtype=np.float32)
        for i in range(tomo.get_num_particles()):
            parts_centers_2[i] = tomo.__parts[i].get_center()

        # Particles loop (Parallel)
        if npr <= 1:
            pr_2nd_tomo(-1, spl_ids[0], parts_centers_1, parts_centers_2,
                                                          distances, thick, border,
                                                          conv_iter, max_iter, fmm, True,
                                                          voi_fname, voi_shape, voi_shared,
                                                          shared_mat_1, shared_mat_2)
        else:
            for pr_id in range(npr):
                pr = mp.Process(target=pr_2nd_tomo, args=(pr_id, spl_ids[pr_id], parts_centers_1, parts_centers_2,
                                                          distances, thick, border,
                                                          conv_iter, max_iter, fmm, True,
                                                          voi_fname, voi_shape, voi_shared,
                                                          shared_mat_1, shared_mat_2))
                pr.start()
                processes.append(pr)
            pr_results = list()
            for pr in processes:
                pr.join()
                pr_results.append(pr.exitcode)
            for pr_id in range(len(processes)):
                if pr_id != pr_results[pr_id]:
                    error_msg = 'Process ' + str(pr_id) + ' exited unexpectedly!'
                    raise pexceptions.PySegInputError(expr='compute_uni_2nd_order (TomoParticles)',
                                                          msg=error_msg)
            gc.collect()

        # Recover parallel processed results
        hold_buffer_1 = np.frombuffer(shared_mat_1.get_obj(),
                                    dtype=np.float32).reshape(self.get_num_particles(), len(distances))
        hold_buffer_2 = np.frombuffer(shared_mat_2.get_obj(),
                                    dtype=np.float32).reshape(self.get_num_particles(), len(distances))

        # Metric computation
        if out_sep == 0:
            if thick is None:
                if dimen == 2:
                    return distances * (np.sqrt((1. / dens) * (hold_buffer_1.sum(axis=0) / hold_buffer_2.sum(axis=0))) - 1.)
                else:
                    return distances * (np.cbrt((1. / dens) * (hold_buffer_1.sum(axis=0) / hold_buffer_2.sum(axis=0))) - 1.)
            else:
                if rdf:
                    return (1. / dens) * (hold_buffer_1.sum(axis=0) / hold_buffer_2.sum(axis=0))
                else:
                    return hold_buffer_1.sum(axis=0) / hold_buffer_2.sum(axis=0)
        elif out_sep == 1:
            return hold_buffer_1.sum(axis=0), hold_buffer_2.sum(axis=0)
        else:
            if (len(hold_buffer_1) > 0) and (len(hold_buffer_2) > 0):
                return hold_buffer_1, hold_buffer_2
            else:
                return None, None

    def simulate_uni_2nd_order(self, n_sims, temp_model, part_vtp, mode_emb,
                               distances, thick=None, border=True,
                               conv_iter=100, max_iter=100000, fmm=False, dimen=3, rdf=False,
                               npr=None, tmp_folder=None):
        """
        Simulate instances of Univariate 2nd order statistics of an input model
        :param n_sims: number of of instances for the simulation
        :param temp_model: model template class (child of Model class) for doing the simulations
        :param part_vtp: vtkPolyData used for as reference for the particle surfaces
        :param mode_emb: mode for particle embedding (see Model.gen_instance())
        :param distances, thick, border, conv_iter, max_iter, fmm: see compute_uni_2nd_order() method
        :param dimen: number of space dimensions, valid: 3 (default) and 2
        :param npr: number of parallel process
        :param tmp_folder: directory to store the temporary dmaps, required to parallelize the simulations,
        if None (default) then only the metric computation is parallelized
        :return: a matrix with n rows for every particle simulated and r columns for the distances
        """

        # Input parsing
        n_parts_tomo, tomo_fname = self.get_num_particles(), self.get_tomo_fname()
        if (n_sims <= 0) or (n_parts_tomo <= 0):
            return None
        from pyorg.surf import Model
        if not isinstance(temp_model, Model):
            error_msg = 'Input template model must be a subclass of Model.'
            raise pexceptions.PySegInputError(expr='simulate_uni_2nd_order_matrix (TomoParticles)',
                                                  msg=error_msg)
        if not isinstance(part_vtp, vtk.vtkPolyData):
            error_msg = 'Input particle surface must be a vtkPolyData.'
            raise pexceptions.PySegInputError(expr='simulate_uni_2nd_order_matrix (TomoParticles)',
                                                  msg=error_msg)
        if not hasattr(distances, '__len__'):
            error_msg = 'Input distances range must be an iterable of floats.'
            raise pexceptions.PySegInputError(expr='simulate_uni_2nd_order (TomoParticles)',
                                              msg=error_msg)

        # MULTIPROCESSING
        if npr is None:
            npr = mp.cpu_count()
        if npr > n_sims:
            npr = n_sims
        processes = list()
        # Create the list on indices to split
        spl_ids = np.array_split(range(n_sims), npr)
        # Shared arrays
        voi_fname, voi_shape, voi_shared = None, None, None
        hold_voi = self.get_voi()
        if isinstance(hold_voi, vtk.vtkPolyData):
            if tmp_folder is None:
                error_msg = 'When input VOI is a vtkPolyData object tmp_folder is compulsory.'
                raise pexceptions.PySegInputError(expr='gen_tlist', msg=error_msg)
            voi_fname = tmp_folder + '/gen_tlist_voi.vtp'
            disperse_io.save_vtp(hold_voi, voi_fname)
        else:
            hold_voi = np.asarray(hold_voi, dtype=np.bool)
            voi_shape = hold_voi.shape
            # voi_shared = mp.RawArray(ctypes.c_bool, hold_voi.reshape(np.array(voi_shape).prod()))
            voi_len = np.array(voi_shape).prod()
            voi_shared_raw = mp.RawArray(np.ctypeslib.ctypes.c_uint8, voi_len)
            voi_shared = np.ctypeslib.as_array(voi_shared_raw)
            voi_shared[:] = hold_voi.reshape(voi_len).astype(np.uint8)
        n_dsts, n_parts = len(distances), self.get_num_particles()
        shared_mat_1 = mp.Array('f', n_parts * n_dsts * n_sims)
        shared_mat_2 = mp.Array('f', n_parts * n_dsts * n_sims)
        shared_mat_3 = mp.Array('f', n_sims)

        # Get particle points
        parts_centers_1 = np.zeros(shape=(n_parts, len(self.__parts[0].get_center())), dtype=np.float32)
        for i in range(self.get_num_particles()):
            parts_centers_1[i] = self.__parts[i].get_center()
        parts_centers_2 = parts_centers_1

        # Particles loop (Parallel)
        if npr <= 1:
            pr_sim_2nd_tomo(-1, spl_ids[0], parts_centers_1, parts_centers_2, distances, thick, border,
                            conv_iter, max_iter, fmm, False,
                            voi_fname, temp_model, part_vtp, mode_emb, voi_shape, voi_shared,
                            shared_mat_1, shared_mat_2, shared_mat_3)
        else:
            for pr_id in range(npr):
                pr = mp.Process(target=pr_sim_2nd_tomo, args=(pr_id, spl_ids[pr_id], parts_centers_1, parts_centers_2,
                                                              distances, thick, border,
                                                              conv_iter, max_iter, fmm, False,
                                                              voi_fname, temp_model, part_vtp, mode_emb, voi_shape, voi_shared,
                                                              shared_mat_1, shared_mat_2, shared_mat_3))
                pr.start()
                processes.append(pr)
            pr_results = list()
            for pr in processes:
                pr.join()
                pr_results.append(pr.exitcode)
            for pr_id in range(len(processes)):
                if pr_id != pr_results[pr_id]:
                    error_msg = 'Process ' + str(pr_id) + ' exited unexpectedly!'
                    raise pexceptions.PySegInputError(expr='compute_uni_2nd_order (TomoParticles)', msg=error_msg)
            gc.collect()

        # Recover parallel processed results
        hold_buffer_1 = np.frombuffer(shared_mat_1.get_obj(), dtype=np.float32).reshape(n_parts * n_sims, n_dsts)
        hold_buffer_2 = np.frombuffer(shared_mat_2.get_obj(), dtype=np.float32).reshape(n_parts * n_sims, n_dsts)
        hold_buffer_3 = np.frombuffer(shared_mat_3.get_obj(), dtype=np.float32) / float(self.compute_voi_volume())

        # Metric computation
        mat = np.zeros(shape=(n_sims, len(distances)), dtype=float)
        if thick is None:
            for i in range(n_sims):
                li = i * n_parts
                ui = li + n_parts
                if dimen == 2:
                    mat[i, :] = distances * (np.sqrt((1. / hold_buffer_3[i]) *
                                                     (hold_buffer_1[li:ui, :].sum(axis=0) / hold_buffer_2[li:ui, :].sum(
                                                         axis=0))) - 1.)
                else:
                    mat[i, :] = distances * (np.cbrt((1. / hold_buffer_3[i]) *
                                             (hold_buffer_1[li:ui, :].sum(axis=0) / hold_buffer_2[li:ui, :].sum(axis=0))) - 1.)
        else:
            for i in range(n_sims):
                li = i * n_parts
                ui = li + n_parts
                if rdf:
                    mat[i, :] = (1. / hold_buffer_3[i]) * \
                                (hold_buffer_1[li:ui, :].sum(axis=0) / hold_buffer_2[li:ui, :].sum(axis=0))
                else:
                    mat[i, :] = hold_buffer_1[li:ui, :].sum(axis=0) / hold_buffer_2[li:ui, :].sum(axis=0)

        return mat

    def simulate_bi_2nd_order(self, tomo, n_sims, temp_model, part_vtp, mode_emb,
                               distances, thick=None, border=True,
                               conv_iter=100, max_iter=100000, fmm=False, switched=False, dimen=3, rdf=False,
                               npr=None, tmp_folder=None):
        """
        Simulate instances of Bivariate 2nd order statistics of an input model
        :param tomo: tomogram with particles to compare
        :param n_sims: number of of instances for the simulation
        :param switched: if True (default) then the analysis is self->tomo, otherwise tomo->self
        :param the-rest: same as simulate_uni_2nd_order()
        :return: a matrix with n rows for every particle simulated and r columns for the distances
        """

        # Input parsing
        from pyorg.surf import Model
        if switched:
            n_parts_tomo, tomo_fname = self.get_num_particles(), self.get_tomo_fname()
        else:
            n_parts_tomo, tomo_fname = tomo.get_num_particles(), tomo.get_tomo_fname()
        if (n_sims <= 0) or (n_parts_tomo <= 0):
            return None
        from pyorg.surf import Model
        if not isinstance(temp_model, Model):
            error_msg = 'Input template model must be a subclass of Model.'
            raise pexceptions.PySegInputError(expr='simulate_uni_2nd_order_matrix (TomoParticles)',
                                              msg=error_msg)
        if not isinstance(part_vtp, vtk.vtkPolyData):
            error_msg = 'Input particle surface must be a vtkPolyData.'
            raise pexceptions.PySegInputError(expr='simulate_uni_2nd_order_matrix (TomoParticles)',
                                              msg=error_msg)
        if not hasattr(distances, '__len__'):
            error_msg = 'Input distances range must be an iterable of floats.'
            raise pexceptions.PySegInputError(expr='simulate_uni_2nd_order (TomoParticles)',
                                              msg=error_msg)

        # MULTIPROCESSING
        if npr is None:
            npr = mp.cpu_count()
        if npr > n_sims:
            npr = n_sims
        processes = list()
        # Create the list on indices to split
        spl_ids = np.array_split(range(n_sims), npr)
        # Shared arrays
        voi_fname, voi_shape, voi_shared = None, None, None
        if switched:
            hold_voi = self.get_voi()
        else:
            hold_voi = tomo.get_voi()
        if isinstance(hold_voi, vtk.vtkPolyData):
            if tmp_folder is None:
                error_msg = 'When input VOI is a vtkPolyData object tmp_folder is compulsory.'
                raise pexceptions.PySegInputError(expr='gen_tlist', msg=error_msg)
            voi_fname = tmp_folder + '/gen_tlist_voi.vtp'
            disperse_io.save_vtp(hold_voi, voi_fname)
        else:
            hold_voi = np.asarray(hold_voi, dtype=np.bool)
            voi_shape = hold_voi.shape
            # voi_shared = mp.RawArray(ctypes.c_bool, hold_voi.reshape(np.array(voi_shape).prod()))
            voi_len = np.array(voi_shape).prod()
            voi_shared_raw = mp.RawArray(np.ctypeslib.ctypes.c_uint8, voi_len)
            voi_shared = np.ctypeslib.as_array(voi_shared_raw)
            voi_shared[:] = hold_voi.reshape(voi_len).astype(np.uint8)
        if switched:
            n_parts = tomo.get_num_particles()
        else:
            n_parts = self.get_num_particles()
        n_dsts = len(distances)
        shared_mat_1 = mp.Array('f', n_parts * n_dsts * n_sims)
        shared_mat_2 = mp.Array('f', n_parts * n_dsts * n_sims)
        shared_mat_3 = mp.Array('f', n_sims)

        # Get particle points
        if switched:
            parts_centers_1 = np.zeros(shape=(n_parts, len(tomo.__parts[0].get_center())),
                                       dtype=np.float32)
            parts_centers_2 = np.zeros(shape=(self.get_num_particles(), len(self.__parts[0].get_center())),
                                       dtype=np.float32)
            for i in range(n_parts):
                parts_centers_1[i] = tomo.__parts[i].get_center()
            for i in range(self.get_num_particles()):
                parts_centers_2[i] = self.__parts[i].get_center()
        else:
            parts_centers_1 = np.zeros(shape=(n_parts, len(self.__parts[0].get_center())), dtype=np.float32)
            parts_centers_2 = np.zeros(shape=(tomo.get_num_particles(), len(tomo.__parts[0].get_center())),
                                       dtype=np.float32)
            for i in range(n_parts):
                parts_centers_1[i] = self.__parts[i].get_center()
            for i in range(tomo.get_num_particles()):
                parts_centers_2[i] = tomo.__parts[i].get_center()

        # Particles loop (Parallel)
        if npr <= 1:
            pr_sim_2nd_tomo(-1, spl_ids[0], parts_centers_1, parts_centers_2, distances, thick, border,
                            conv_iter, max_iter, fmm, True,
                            voi_fname, temp_model, part_vtp, mode_emb, voi_shape, voi_shared,
                            shared_mat_1, shared_mat_2, shared_mat_3)
        else:
            for pr_id in range(npr):
                pr = mp.Process(target=pr_sim_2nd_tomo, args=(pr_id, spl_ids[pr_id], parts_centers_1, parts_centers_2,
                                                              distances, thick, border,
                                                              conv_iter, max_iter, fmm, True,
                                                              voi_fname, temp_model, part_vtp, mode_emb, voi_shape,
                                                              voi_shared,
                                                              shared_mat_1, shared_mat_2, shared_mat_3))
                pr.start()
                processes.append(pr)
            pr_results = list()
            for pr in processes:
                pr.join()
                pr_results.append(pr.exitcode)
            for pr_id in range(len(processes)):
                if pr_id != pr_results[pr_id]:
                    error_msg = 'Process ' + str(pr_id) + ' exited unexpectedly!'
                    raise pexceptions.PySegInputError(expr='compute_uni_2nd_order (TomoParticles)', msg=error_msg)
            gc.collect()

        # Recover parallel processed results
        hold_buffer_1 = np.frombuffer(shared_mat_1.get_obj(),
                                      dtype=np.float32).reshape(n_parts * n_sims, n_dsts)
        hold_buffer_2 = np.frombuffer(shared_mat_2.get_obj(),
                                      dtype=np.float32).reshape(n_parts * n_sims, n_dsts)
        hold_buffer_3 = np.frombuffer(shared_mat_3.get_obj(), dtype=np.float32) / float(self.compute_voi_volume())

        # Metric computation
        mat = np.zeros(shape=(n_sims, len(distances)), dtype=float)
        if thick is None:
            for i in range(n_sims):
                li = i * n_parts
                ui = li + n_parts
                if dimen == 2:
                    mat[i, :] = distances * (np.sqrt((1. / hold_buffer_3[i]) *
                                                     (hold_buffer_1[li:ui, :].sum(axis=0) / hold_buffer_2[li:ui, :].sum(
                                                         axis=0))) - 1.)
                else:
                    mat[i, :] = distances * (np.cbrt((1. / hold_buffer_3[i]) *
                                                     (hold_buffer_1[li:ui, :].sum(axis=0) / hold_buffer_2[li:ui, :].sum(
                                                     axis=0))) - 1.)
        else:
            for i in range(n_sims):
                li = i * n_parts
                ui = li + n_parts
                if rdf:
                    mat[i, :] = (1. / hold_buffer_3[i]) * \
                                (hold_buffer_1[li:ui, :].sum(axis=0) / hold_buffer_2[li:ui, :].sum(axis=0))
                else:
                    mat[i, :] = hold_buffer_1[li:ui, :].sum(axis=0) / hold_buffer_2[li:ui, :].sum(axis=0)

        return mat

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

    # Estimates global density
    # surface: if True (default False) then density is estimated by surface intead of volume
    def compute_global_density(self, surface=False):

        # Compute VOI volume
        if surface:
            vol = self.compute_voi_surface()
        else:
            vol = self.compute_voi_volume()

        if vol <= 0:
            return 0.
        else:
            return float(self.get_num_particles()) / vol

    # Append all particle surfaces in one vtkPolyData
    # mode: (default 'surface') mode for generating the vtkPolyData for every particle,
    #       see Particle.get_vtp()
    # lbl: if not None (default) it is added as integer attribute for labeling
    # Returns: a single vtkPolyData
    def append_particles_vtp(self, mode='surface', lbl=None):
        is_empty = True
        appender = vtk.vtkAppendPolyData()
        for part in self.__parts:
            if lbl is not None:
                if isinstance(part, Particle):
                    part.add_vtp_global_attribute('list', vtk.vtkIntArray, int(lbl))
                else:
                    part.add_prop('list', vtk.vtkIntArray, int(lbl))
            copy_vtp = vtk.vtkPolyData()
            if isinstance(part, Particle):
                copy_vtp.DeepCopy(part.get_vtp(mode=mode))
            else:
                copy_vtp.DeepCopy(part.get_vtp())
            appender.AddInputData(copy_vtp)
            is_empty = False
        if is_empty:
            return None
        else:
            appender.Update()
            return appender.GetOutput()

    def gen_csr_coords(self, n_coords):
        """
        Generates an array of coordinates randomly distibuited on VOI (CSR pattern)
        :param n_coords: number of coordinates to simulate
        :return: a [n_coords, 3] array
        """
        count = 0
        voi = self.get_voi()
        x_min, x_max, y_min, y_max, z_min, z_max = self.__bounds
        n_tries = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
        if n_tries < 10 * n_coords:
            n_tries = 10 * n_coords
        hold_coords = np.zeros(shape=(n_coords, 3), dtype=np.float32)
        for i in range(n_tries):
            if count < n_coords:
                x_rnd, y_rnd, z_rnd = random.randint(x_min, x_max), random.randint(y_min, y_max), \
                                      random.randint(z_min, z_max)
                if self.__voi_selector is not None:
                    if self.__voi_selector.IsInsideSurface((x_rnd, y_rnd, z_rnd)) > 0:
                        hold_coords[count, :] = x_rnd, y_rnd, z_rnd
                        count += 1
                else:
                    x_rnd_i, y_rnd_i, z_rnd_i = int(round(x_rnd)), int(round(y_rnd)), int(round(z_rnd))
                    if voi[x_rnd_i, y_rnd_i, z_rnd_i]:
                        hold_coords[count, :] = x_rnd_i, y_rnd_i, z_rnd_i
                        count += 1
            else:
                break
        return hold_coords

    # Computes first order metrics
    # f_nsamp: if 0<= (Default) then inter partices nearest distance is computed,otherwise is the number or CSR samples
    #          then sample-particle nearest distance is computed
    # Returns: an array with the n (number of particles or samples) nearest distances
    def compute_uni_1st_order_dsts(self, f_nsamp=0):
        if self.get_num_particles() <= 1:
            return None
        # Function-G mode
        if f_nsamp <= 0:
            return nnde(self.get_particle_coords())
        else:
            return cnnde(self.gen_csr_coords(f_nsamp), self.get_particle_coords())

    # Computes first order metrics from a reference set of coordintaes
    # coords_ref: reference coordinates
    # Returns: an array with the n (number of input coordinates) nearest distances to tomogram particles
    def compute_bi_1st_order_dsts(self, coords_ref):
        # Function-G bi-variate mode
        return cnnde(coords_ref, self.get_particle_coords())

    # Purge particles by scale suppresion
    # ss: scale for supression
    # ext_coords: external list of reference coordinates (default None)
    def scale_suppression(self, ss, ext_coords=None):
        # Mark particles to delete
        is_ext = True
        coords = self.get_particle_coords()
        if coords is None:
            return
        if ext_coords is None:
            ext_coords = coords
            is_ext = False
        lut = np.ones(shape=len(self.__parts), dtype=np.bool)
        for i, coord in enumerate(ext_coords):
            if is_ext or lut[i]:
                hold = coords - coord
                dsts = np.sqrt((hold * hold).sum(axis=1))
                ids = np.where(dsts < ss)[0]
                for idx in ids:
                    if is_ext:
                        lut[idx] = False
                    elif idx != i:
                        lut[idx] = False

        # Delete the marked particles
        hold_parts = self.__parts
        self.__parts = list()
        for i, part in enumerate(hold_parts):
            if lut[i]:
                self.__parts.append(part)

    def gen_newtomo_ap(self, ap_pref=None, ap_damp=.5, ap_max_iter=20000, ap_conv_iter=15, pj=False,
                       c_rad=5):
        """
        Generates a new TomoParticles but now particles coordinates are at Affinity Propagation
        cluster centroids. Centroids can only be spheres.
        :param ap_pref: AP preference value
        :param ap_damp: AP damping parameter (default .5)
        :param ap_max_iter: AP maximum number of iterations (default 20000)
        :param ap_conv_iter: AP iterations for convergence (default 15)
        :param pj=True: project the generated centroids (default False)
        :param c_rad=5: centroid radius (centroids can only have spherical shape)
        :return: The newly generated TomoParticles object where particles are cluster's centroids
        """

        # Initialization
        self.get_tomo_fname()

        # Getting current coordinates
        coords = self.get_particle_coords()

        # AP
        aff = AffinityPropagation(damping=ap_damp,
                                  convergence_iter=ap_conv_iter,
                                  max_iter=ap_max_iter,
                                  preference=ap_pref)
        aff.fit(coords)

        # Creating the new TomoParticles objects
        hold_tomo = TomoParticles(self.get_tomo_fname(), 1, voi=self.get_voi())
        if aff.cluster_centers_indices_ is None:
            hold_coords = coords
        else:
            hold_coords = aff.cluster_centers_
        for cg in hold_coords:
            vtp = vtk.vtkSphereSource()
            vtp.SetCenter(0., 0., 0.)
            vtp.SetRadius(float(c_rad))
            vtp.Update()
            part = Particle(vtp.GetOutput(), center=(0, 0, 0), normal=(0, 0, 0))
            try:
                part.translation(cg[0], cg[1], cg[2])
            except IndexError:
                    print 'WARNING(TomoParticles.gen_newtomo_ap()): centroid ' + str(cg) + ' failed to be inserted!'
            hold_tomo.insert_particle(part, check_bounds=None, voi_pj=pj)

        return hold_tomo

    def compute_shortest_distances_matrix(self, in_pids, out_pids, rg_dsts=None, inter_ssize=None,
                                          part_surf_fname=None, max_conn=1, del_border=0, min_neg_dst=0):
        """
        Computes the shortest inter-particles distance considering their surface
        :param start_pids: list of point IDs on the particle reference for distance starting points
        :param end_pids: list of point IDs on the particle reference for distance ending points
        :param rg_dsts: 2-tuple with maximum and minimum distance to search, if None (default) then [0, TOMO_MAX_SIZE]
        is set
        :param inter_ssize: step size for checking intersection, if None (default) then no intersection is checked
        :param part_surf_fname: external reference surface path, required if the particles of self are not ParticleL
        :param max_conn: it limits the maximum number of connection for an input point
        :param del_border: if >0 (default 0) the it defines the distance from points to segmentation border to be
        discarded
        :param min_neg_dst: threshold to consider a negative distance
        :return: A [P, S] matrix with the distances (a list per each matrix entry) obtained, P is the number of
        particles and S the number of starting points, an empty list means no value found. A second matrix with a
        code for each distance found: 1->'positive' interdistance, 2->'negative' interdistanc and 3->border point
        Three vtkPolyData objects with the
        connection points, the connection lines and particles' surface (if inter_ssize is not None).
        """

        # Input parsing
        n_parts = self.get_num_particles()
        if n_parts <= 1:
            return np.zeros(shape=(0,0), dtype=np.float32)
        if (not isinstance(self.__parts[0], ParticleL)) and (part_surf_fname is None):
            error_msg = 'Funtion-NNS only works for ParticleL objects and exteranl surface was introduced.'
            raise pexceptions.PySegInputError(expr='compute_shortest_distances_matrix', msg=error_msg)
        if rg_dsts is None:
            max_x = self.__bounds[1] - self.__bounds[0]
            max_y = self.__bounds[3] - self.__bounds[2]
            max_z = self.__bounds[5] - self.__bounds[4]
            rg_dsts = (0, math.sqrt(max_x*max_x + max_y*max_y + max_z*max_z))
        n_points_in, n_points_out = len(in_pids), len(out_pids)
        # mat_out = (-1) * np.ones(shape=(n_parts, n_points_in), dtype=np.float32)
        # pairs = np.zeros(shape=(n_parts, n_points_in), dtype=object)
        pairs = dict().fromkeys(np.arange(n_parts))
        pairs_dsts = dict().fromkeys(pairs.keys())
        mat_out = np.ones(shape=(n_parts, n_points_in), dtype=object)
        mat_codes = np.ones(shape=(n_parts, n_points_in), dtype=object)
        for i in range(mat_out.shape[0]):
            for j in range(mat_out.shape[1]):
                mat_out[i, j], mat_codes[i, j] = list(), list()
        part_centers = self.get_particle_coords()
        if inter_ssize is not None:
            mask_surf = self.gen_particles_vtp()
            selector_mask = vtk.vtkSelectEnclosedPoints()
            selector_mask.SetTolerance(VTK_RAY_TOLERANCE)
            selector_mask.Initialize(mask_surf)

        # Compute the distance map to segmentation
        dst_map = None
        if (del_border > 0) and isinstance(self.__voi, np.ndarray):
            dst_map = sp.ndimage.morphology.distance_transform_edt(self.__voi)

        # Loop for particles
        if part_surf_fname is None:
            part_surf_fname = self.__parts[0].get_vtp_fname()
        part_surf = disperse_io.load_poly(part_surf_fname)
        for i in range(n_parts):
            part_1 = self.__parts[i]
            part_1_center = part_1.get_center()
            points_1 = list()
            for ii in range(n_points_in):
                hold_point = part_1.get_surf_point(in_pids[ii], ref_vtp=part_surf)
                points_1.append(np.asarray(hold_point, dtype=np.float32))
            selector_part = vtk.vtkSelectEnclosedPoints()
            selector_part.SetTolerance(VTK_RAY_TOLERANCE)
            # print(selector.GetTolerance())
            part_1_surf = part_1.get_vtp()
            selector_part.Initialize(part_1_surf)

            # Sort the neighbour particles by closeness to particle 1
            hold = part_centers - part_1_center
            dsts = np.sqrt((hold * hold).sum(axis=1))
            sort_ids = np.argsort(dsts)
            sort_dsts = dsts[sort_ids]
            l_id, h_id = np.searchsorted(sort_dsts, rg_dsts[0]), np.searchsorted(sort_dsts, rg_dsts[1])
            if l_id == 0:
                l_id += 1
                if l_id > h_id:
                    l_id = h_id

            # Compute the intersection mask and create and VTK manager
            parts_2 = list()
            for j in sort_ids[l_id:h_id]:
                part_2 = self.__parts[j]
                parts_2.append(part_2)

            # Loop for neighbourhod particles
            points_2 = list()
            for j, part_2 in enumerate(parts_2):
                for jj in range(n_points_out):
                    hold_point = part_2.get_surf_point(out_pids[jj], ref_vtp=part_surf)
                    points_2.append(np.asarray(hold_point, dtype=np.float32))

            # Loop for looking for the closest distances between part_1 and part_2
            hold_iis = list()
            for ii, point_1 in enumerate(points_1):
                hold_pairs, hold_dsts, hold_codes = list(), list(), list()

                # min_dst, min_id = np.finfo(np.float).max, -1
                for jj, point_2 in enumerate(points_2):
                    hold = (point_1 - point_2).astype(np.double)
                    dst = np.sqrt((hold * hold).sum())
                    # if dst < min_dst:
                    if inter_ssize is not None:
                        is_valid = True
                        line_coords = line_2_pts(point_1, point_2, inter_ssize)
                        for k, point_l in enumerate(line_coords[1:-1]):
                            if selector_mask.IsInsideSurface(point_l) > 0:
                                is_valid = False
                                break
                        if not is_valid:
                            continue
                    if (dst >= rg_dsts[0]) and (dst < rg_dsts[1]):
                        hold_pairs.append((jj, point_1, point_2, ii))
                        hold_dsts.append(dst)
                        #### Get the code for this distance
                        # Check if segmentation border is closer than closest inter-particles distance,
                        # if True the measurements for these particle are discarded
                        hold_code = CODE_POSITIVE_DST
                        if dst_map is not None:
                            x, y, z = int(round(point_1[0])), int(round(point_1[1])), int(round(point_1[2]))
                            try:
                                dst = dst_map[x, y, z]
                                if dst < del_border:
                                    hold_code = CODE_BORDER
                            except IndexError:
                                print 'WARNING (TomoParticles:compute_shortest_distances_matrix): incorrect indexing ' \
                                      'for the distance map ' + str(time.time())
                                hold_code = CODE_BORDER
                        if hold_code == CODE_POSITIVE_DST:
                            # Check if point_2 is inside the reference particle ('negative' distance)
                            if is_point_inside_surf(point_2, selector_part, 2, 10):
                                if min_neg_dst > 0:
                                    surf_pt_1 = vtp_closest_point(part_1_surf, point_2)
                                    hold_pt_1_dst = point_2 - surf_pt_1
                                    hold_pt_1_dst = np.sqrt((hold_pt_1_dst * hold_pt_1_dst).sum())
                                    if hold_pt_1_dst > min_neg_dst:
                                        hold_code = CODE_NEGATIVE_DST
                                else:
                                    hold_code = CODE_NEGATIVE_DST
                        hold_codes.append(hold_code)

                hold_max_conn, len_hold_pairs = max_conn, len(hold_pairs)
                if len_hold_pairs > 0:
                    if hold_max_conn > len(hold_pairs):
                        hold_max_conn = len(hold_pairs)
                    sort_ids2 = np.argsort(hold_dsts)
                    sorted_hold_pairs, sorted_dsts, sorted_codes = list(), list(), list()
                    for k in range(hold_max_conn):
                        sorted_hold_pairs.append(hold_pairs[sort_ids2[k]])
                        sorted_dsts.append(hold_dsts[sort_ids2[k]])
                        sorted_codes.append(hold_codes[sort_ids2[k]])
                    if pairs[i] is None:
                        pairs[i], pairs_dsts[i] = list(), list()
                    # pairs[i] += sorted_hold_pairs
                    # pairs_dsts[i] += sorted_dsts
                    # mat_out[i, ii] += sorted_dsts
                    # mat_codes[i, ii] += sorted_codes
                    hold_iis.append(ii)
                    for hold_pair, hold_dst, hold_code in zip(sorted_hold_pairs, sorted_dsts, sorted_codes):
                        pairs[i].append(hold_pair)
                        if hold_code == 2:
                            pairs_dsts[i].append(hold_dst)
                            mat_out[i, ii].append(hold_dst)
                            hold_code = 1
                        else:
                            pairs_dsts[i].append(hold_dst)
                            mat_out[i, ii].append(hold_dst)
                        mat_codes[i, ii].append(hold_code)

        # Create the particles vtkPolyData and add border property to particles
        appender = vtk.vtkAppendPolyData()
        for i in range(n_parts):

            # Check particle code
            hold_code = CODE_POSITIVE_DST
            for ii in range(len(mat_codes[i, :])):
                if mat_codes[i, ii][0] == CODE_BORDER:
                    hold_code = CODE_BORDER
                    break

            # Add particle id property
            hold_vtp = self.__parts[i].get_vtp()
            part_id, part_code = vtk.vtkIntArray(), vtk.vtkIntArray()
            part_id.SetName(PT_ID)
            part_id.SetNumberOfComponents(1)
            part_id.SetNumberOfTuples(hold_vtp.GetNumberOfCells())
            part_id.FillComponent(0, i)
            part_code.SetName(CODE_STR)
            part_code.SetNumberOfComponents(1)
            part_code.SetNumberOfTuples(hold_vtp.GetNumberOfCells())
            part_code.FillComponent(0, hold_code)
            hold_vtp.GetCellData().AddArray(part_id)
            hold_vtp.GetCellData().AddArray(part_code)

            # Append current particle
            appender.AddInputData(hold_vtp)

            # Fuse to one vtkPolyData
            appender.Update()

        poly_parts = appender.GetOutput()

        # Loop for creating the connections paths VTKs
        poly_p, points_p = vtk.vtkPolyData(), vtk.vtkPoints()
        poly_l, points_l = vtk.vtkPolyData(), vtk.vtkPoints()
        cells_p, cells_l = vtk.vtkCellArray(), vtk.vtkCellArray()
        arr_type_p, arr_pid_p, arr_id_p, arr_code_p = vtk.vtkIntArray(), vtk.vtkIntArray(), vtk.vtkIntArray(), \
                                                     vtk.vtkIntArray()
        arr_type_p.SetName('TYPE')
        arr_type_p.SetNumberOfComponents(1)
        arr_pid_p.SetName('POINT_ID')
        arr_pid_p.SetNumberOfComponents(1)
        arr_id_p.SetName('PART_ID')
        arr_id_p.SetNumberOfComponents(1)
        arr_code_p.SetName('CODE')
        arr_code_p.SetNumberOfComponents(1)
        arr_pid_l, arr_id_l, arr_len_l, arr_code_l = vtk.vtkIntArray(), vtk.vtkIntArray(), vtk.vtkFloatArray(), \
                                                     vtk.vtkIntArray()
        arr_pid_l.SetName('POINT_ID')
        arr_pid_l.SetNumberOfComponents(1)
        arr_id_l.SetName('PART_ID')
        arr_id_l.SetNumberOfComponents(1)
        arr_len_l.SetName('LENGTH')
        arr_len_l.SetNumberOfComponents(1)
        arr_code_l.SetName('CODE')
        arr_code_l.SetNumberOfComponents(1)
        count_p = 0
        # Loop for points
        for i in range(n_parts):
            # Check particle code
            hold_code = CODE_POSITIVE_DST
            for ii in range(len(mat_codes[i, :])):
                if mat_codes[i, ii][0] == CODE_BORDER:
                    hold_code = CODE_BORDER
                    break
            part = self.__parts[i]
            for ii in range(n_points_in):
                hold_point = part.get_surf_point(in_pids[ii], ref_vtp=part_surf)
                points_p.InsertNextPoint(hold_point)
                cells_p.InsertNextCell(1)
                cells_p.InsertCellPoint(count_p)
                arr_type_p.InsertTuple(count_p, (1,))
                arr_pid_p.InsertTuple(count_p, (ii,))
                arr_id_p.InsertTuple(count_p, (i,))
                arr_code_p.InsertTuple(count_p, (hold_code,))
                count_p += 1
            for ii in range(n_points_out):
                hold_point = part.get_surf_point(out_pids[ii], ref_vtp=part_surf)
                points_p.InsertNextPoint(hold_point)
                cells_p.InsertNextCell(1)
                cells_p.InsertCellPoint(count_p)
                arr_type_p.InsertTuple(count_p, (2,))
                arr_pid_p.InsertTuple(count_p, (ii,))
                arr_id_p.InsertTuple(count_p, (i,))
                arr_code_p.InsertTuple(count_p, (hold_code,))
                count_p += 1
        # Loop for connections
        count_l, count_p = 0, 0
        # for i in range(n_parts):
        for i in pairs.iterkeys():
            # Check particle code
            hold_code = CODE_POSITIVE_DST
            for ii in range(len(mat_codes[i, :])):
                if mat_codes[i, ii][0] == CODE_BORDER:
                    hold_code = CODE_BORDER
                    break
            hold_pairs = pairs[i]
            if hold_pairs is None:
                continue
            # for ii in range(n_points_in):
            for hold_t in hold_pairs:
                # try:
                #     point_1, point_2 = pairs[i, ii]
                # except TypeError:
                #     continue
                jj, point_1, point_2, ii = hold_t[0], hold_t[1], hold_t[2], hold_t[3]
                cells_l.InsertNextCell(2)
                points_l.InsertNextPoint(point_1)
                cells_l.InsertCellPoint(count_p)
                count_p += 1
                points_l.InsertNextPoint(point_2)
                cells_l.InsertCellPoint(count_p)
                count_p += 1
                arr_pid_l.InsertTuple(count_l, (jj,))
                arr_id_l.InsertTuple(count_l, (i,))
                hold = point_2 - point_1
                hold_dst = math.sqrt((hold * hold).sum())
                arr_len_l.InsertTuple(count_l, (hold_dst,))
                arr_code_l.InsertTuple(count_l, (hold_code,))
                # mat_out[i, ii] = dst
                count_l += 1
        poly_p.SetPoints(points_p)
        poly_p.SetVerts(cells_p)
        poly_l.SetPoints(points_l)
        poly_l.SetLines(cells_l)
        poly_p.GetCellData().AddArray(arr_type_p)
        poly_p.GetCellData().AddArray(arr_pid_p)
        poly_p.GetCellData().AddArray(arr_id_p)
        poly_p.GetCellData().AddArray(arr_code_p)
        poly_l.GetCellData().AddArray(arr_pid_l)
        poly_l.GetCellData().AddArray(arr_id_l)
        poly_l.GetCellData().AddArray(arr_len_l)
        poly_l.GetCellData().AddArray(arr_code_l)

        return mat_out, mat_codes, poly_p, poly_l, poly_parts

    def ensure_embedding(self):
        """
        Ensure that all inserted particles are embedded (actually only computed for VOI as array)
        :return: True if all are embedded
        """
        if (self.__voi is not None) or isinstance(self.__voi, np.ndarray):
            coords = self.get_particle_coords()
            for coord in coords:
                if not self.__voi[int(round(coord[0])), int(round(coord[1])), int(round(coord[2]))]:
                    return False
        return True

    def resize(self, factor):
        """
        Resize the tomogram, particle coordinates and VOI.
        :param factor: resizing factor (default 1.)
        :return:
        """
        # Resize the particles
        for part in self.__parts:
            # part.scale(factor, factor, factor)
            v_t = (factor-1.) * np.asarray(part.get_center())
            part.translation(v_t[0], v_t[1], v_t[2])
        # Resize the VOI
        if isinstance(self.__voi, np.ndarray):
            self.__voi = sp.ndimage.zoom(self.__voi, factor, order=0)
            self.__voi_dst_ids = sp.ndimage.zoom(self.__voi, factor, order=0)
        else:
            tr, tr_flt = vtk.vtkTransform(), vtk.vktTransformFilter()
            tr.Scale(factor, factor, factor)
            tr_flt.SetInputConnection(self.__voi)
            tr_flt.SetTransform(tr)
            self.__voi = tr_flt.GetOutputPort()
            self.__voi_selector = vtk.vtkSelectEnclosedPoints()
            self.__voi_selector.Initialize(self.__voi)
        self.__update_bounds()

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
        self.load_particles_pickle(self.__parts_fname)

    # Copy the object's state from self.__dict__ which contains all instance attributes.
    # Afterwards remove unpickable objects
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_TomoParticles__voi']
        del state['_TomoParticles__voi_selector']
        del state['_TomoParticles__voi_dst_ids']
        del state['_TomoParticles__parts']
        return state


############################################################################
# Class for a set of tomograms with embedded particles
#
class ListTomoParticles(object):

    def __init__(self):
        self.__tomos = dict()
        # For pickling
        self.__pkls = None

    # EXTERNAL FUNCTIONALITY

    def get_tomos(self):
        return self.__tomos

    def get_num_particles(self):
        total = 0
        for tomo in self.__tomos.itervalues():
            total += tomo.get_num_particles()
        return total

    def get_num_particles_dict(self):
        nparts = dict()
        for key, tomo in zip(self.__tomos.iterkeys(), self.__tomos.itervalues()):
            nparts[key] = tomo.get_num_particles()
        return nparts

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

    # tomo_surf: TomoSurface object to add to the list
    def add_tomo(self, tomo_surf):
        # Input parsing
        tomo_fname = tomo_surf.get_tomo_fname()
        if tomo_fname in self.get_tomo_fname_list():
            print 'WARNING: tomo_surf (ListTomoParticles): tomogram ' + tomo_fname + ' was already inserted.'
            return
        # Adding the tomogram to the list and dictionaries
        self.__tomos[tomo_fname] = tomo_surf

    # Delete a TomoSurface entry in the list
    # tomo_key: TomoSurface key
    # Return: raise a KeyError if these tomogram were not present in the list
    def del_tomo(self, tomo_key):
        del self.__tomos[tomo_key]

    def insert_particle(self, part, tomo_fname, check_bounds=False, mode='full', voi_pj=True,
                        voi_to_pj=None, meta=None):
        """
        Insert a new Particle object in the correspondent tomogram
        :param part: Particle object to insert
        :param tomo_fname: path to the tomogram
        :param check_bounds: to disable bounds checking (default True)
        :param mode: embedding mode, only applies if check_bounds is True (see is_embedded)
        :param voi_pj: if True (default False) particle center is projected on tomogram voi thourgh the normal to its closest
    #         point.
        :param voi_to_pj: if not None (default) then the VOI where the particle is projected, only
            valid if voi_pj=True and used to set a VOI different from self's VOI. The only valid
            VOI is a vtkPolyData
        :param meta: a dictionary with meta information for the particle (default None)
        :return:
        """
        try:
            self.__tomos[tomo_fname].insert_particle(part, check_bounds, mode=mode, voi_pj=voi_pj,
                                                     voi_to_pj=voi_to_pj, meta=meta)
        except KeyError:
            error_msg = 'Tomogram ' + tomo_fname + ' is not added to list!'
            raise pexceptions.PySegInputError(expr='insert_particle (ListTomoParticles)', msg=error_msg)

    # Store the list of tomograms in STAR file and also the STAR files for every tomogram
    # out_stem: string stem used for storing the STAR file for TomoParticles objects
    # out_dir: output directory
    def store_stars(self, out_stem, out_dir):

        # STAR file for the tomograms
        tomos_star = sub.Star()
        tomos_star.add_column('_suTomoParticles')

        # Tomograms loop
        for i, tomo_fname in enumerate(self.__tomos.keys()):

            # Pickling the tomogram
            tomo_dir = out_dir + '/tomo_' + str(i)
            if not os.path.exists(tomo_dir):
                os.makedirs(tomo_dir)
            tomo_stem = os.path.splitext(os.path.split(tomo_fname)[1])[0]
            pkl_file = tomo_dir + '/' + tomo_stem + '_tp.pkl'
            self.__tomos[tomo_fname].pickle(pkl_file)

            # Adding a new enter
            kwargs = {'_suTomoParticles': pkl_file}
            tomos_star.add_row(**kwargs)

        # Storing the tomogram STAR file
        tomos_star.store(out_dir + '/' + out_stem + '_tpl.star')

    # Store the particles in every vtkPolyData per tomogram
    # out_dir: output directory
    # mode: kind of poly to store, valid: 'surface' (default) or 'center'
    def store_particles(self, out_dir, mode='surface'):

        # Tomograms loop
        for i, tomo_fname in enumerate(self.__tomos.keys()):

            tomo_stem = os.path.splitext(os.path.split(tomo_fname)[1])[0]

            # Storing
            if mode == 'surface':
                disperse_io.save_vtp(self.__tomos[tomo_fname].gen_particles_vtp(mode='surface'),
                                     out_dir + '/' + tomo_stem + '_parts_surf.vtp')
            elif mode == 'center':
                disperse_io.save_vtp(self.__tomos[tomo_fname].gen_particles_vtp(mode='center'),
                                     out_dir + '/' + tomo_stem + '_parts_center.vtp')
            else:
                error_msg = 'Input mode ' + str(tomo_fname) + ' is invalid!'
                raise pexceptions.PySegInputError(expr='store_particles (ListTomoParticles)',
                                                  msg=error_msg)


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

    # Generate another instance based on self and the input Model
    # model: class for modeling TomoParticles
    # part_suf: particle surface
    # mode: mode for embedding, valid: 'full' and 'center'
    def gen_model_instance(self, model, part_surf, mode='full'):
        tomos = ListTomoParticles()
        for i, tomo in enumerate(self.__tomos.itervalues()):
            model_in = model(tomo.get_voi(), part_surf)
            tomo_name = 'tomo_model_' + model_in.get_type_name() + '_' + str(i)
            tomos.add_tomo(model.gen_instance(model_in.get_num_particles(), tomo_name, mode=mode))
        return tomos

    # # Compute Univariate 2nd order statistics
    # # distances: range of distances (considered as and iterable of floats)
    # # res: nhood_vol arc resolution
    # # thick: nhood_vol thickes if None (default) then 'sphere', otherwise 'shell'
    # # border: if True (default) border compensation is activated
    # # area: if True (default False) then area of the 2-manifold homolog is estimated
    # # vol_iter: number of points for estimating the volume (default 1000)
    # # npr: number of process, None (defaults) means they are adjusted automatically to the number of cores
    # # verbose: if True then some progress text is provided (default False)
    # # Returns: two matricies (number of points, volumes) (MxN) with particle (M) values for each distance (N), and an array for every
    # #          particle (M) with the tomogram indices starting from zero,
    # def compute_uni_2nd_order(self, distances, res, thick=None, border=True, area=False,
    #                           vol_iter=1000, npr=None, verbose=False):
    #
    #     # Input parsing
    #     if not hasattr(distances, '__len__'):
    #         error_msg = 'Input distances range must be an iterable of floats.'
    #         raise pexceptions.PySegInputError(expr='compute_uni_2nd_order (ListTomoParticles)',
    #                                           msg=error_msg)
    #
    #     # Initialization
    #     dsts = np.asarray(distances, dtype=np.float)
    #     mat_nums = np.zeros(shape=(self.get_num_particles(), len(dsts)), dtype=np.float)
    #     mat_vols = np.zeros(shape=(self.get_num_particles(), len(dsts)), dtype=np.float)
    #
    #     # Call the counter within every tomogram
    #     lb, ub = 0, 0
    #     for i, tomo in enumerate(self.__tomos.itervalues()):
    #         n_parts_tomo = tomo.get_num_particles()
    #         if n_parts_tomo > 0:
    #             ub = lb + n_parts_tomo
    #             mat_nums[lb:ub], mat_vols[lb:ub] = tomo.compute_uni_2nd_order(dsts, res, thick, border, vol_iter,
    #                                                                           area=area,
    #                                                                           npr=npr, verbose=verbose)
    #
    #             lb = ub
    #
    #     return mat_nums[:ub], mat_vols[:ub]

    # Compute Univariate 2nd order statistics by tomograms
    # distances: range of distances (considered as and iterable of floats)
    # thick: nhood_vol thickes if None (default) then 'sphere', otherwise 'shell'
    # border: if True (default) border compensation is activated
    # border: if True the border compensation is done
    # conv_iter: number of iterations for convergence in volume estimation
    # max_iter: maximum number of iterations in volume estimation
    # npr: number of process, None (defaults) means they are adjusted automatically to the number of cores
    # verbose: if True then some progress text is provided (default False)
    # Returns: a dictionay with statistic metric array indexed by tomogram key
    def compute_uni_2nd_order_by_tomos(self, distances, thick=None, border=True,
                                       conv_iter=100, max_iter=100000, npr=None, verbose=False):

        # Input parsing
        if not hasattr(distances, '__len__'):
            error_msg = 'Input distances range must be an iterable of floats.'
            raise pexceptions.PySegInputError(expr='compute_uni_2nd_order (ListTomoParticles)',
                                              msg=error_msg)

        # Initialization
        dsts = np.asarray(distances, dtype=np.float)
        mat = dict()

        # Call the counter within every tomogram
        for i, tomo in enumerate(self.__tomos.itervalues()):
            n_parts_tomo = tomo.get_num_particles()
            if verbose:
                print '\t\t\t+ Processing tomo ' + tomo.get_tomo_fname()
            if n_parts_tomo > 0:
                hold_arr = tomo.compute_uni_2nd_order(dsts, thick, border, conv_iter, max_iter, npr=npr,
                                                      verbose=False)
                if hold_arr is not None:
                    mat[tomo.get_tomo_fname()] = hold_arr

        return mat

    # Compute Univariate 2nd order statistics globally
    # distances: range of distances (considered as and iterable of floats)
    # thick: nhood_vol thickes if None (default) then 'sphere', otherwise 'shell'
    # border: if True (default) border compensation is activated
    # border: if True the border compensation is done
    # conv_iter: number of iterations for convergence in volume estimation
    # max_iter: maximum number of iterations in volume estimation
    # dens_gl: if True (default) then density is estimated globally, otherwise locally
    # npr: number of process, None (defaults) means they are adjusted automatically to the number of cores
    # verbose: if True then some progress text is provided (default False)
    # Returns: a dictionay with statistic metric array indexed by tomogram key
    def compute_uni_2nd_order(self, distances, thick=None, border=True, conv_iter=100, max_iter=100000,
                              dens_gl=True, npr=None, verbose=False):

        # Input parsing
        if not hasattr(distances, '__len__'):
            error_msg = 'Input distances range must be an iterable of floats.'
            raise pexceptions.PySegInputError(expr='compute_uni_2nd_order (ListTomoParticles)', msg=error_msg)

        # Initialization
        dsts = np.asarray(distances, dtype=np.float)
        if dens_gl:
            mat_num, mat_dem = np.zeros(shape=dsts.shape[0], dtype=np.float), \
                               np.zeros(shape=dsts.shape[0], dtype=np.float)
            dens = self.compute_global_density()
        else:
            mat_arr = np.zeros(shape=dsts.shape[0], dtype=np.float)
            t_nparts = 0

        # Call the counter within every tomogram
        for i, tomo in enumerate(self.__tomos.itervalues()):
            n_parts_tomo = tomo.get_num_particles()
            if verbose:
                print '\t\t\t+ Processing tomo ' + tomo.get_tomo_fname()
            if n_parts_tomo > 0:
                hold_arr_num, hold_arr_dem = tomo.compute_uni_2nd_order(dsts, thick, border,
                                                                        conv_iter, max_iter, npr=npr,
                                                                        out_sep=True, verbose=False)
                if (hold_arr_num is not None) and (hold_arr_dem is not None):
                    if dens_gl:
                        mat_num += hold_arr_num
                        mat_dem += hold_arr_dem
                    else:
                        dens = float(n_parts_tomo) / tomo.compute_voi_volume()
                        if thick is None:
                            hold_arr = distances * (np.cbrt((1. / dens) * (hold_arr_num / hold_arr_dem)) - 1.)
                        else:
                            hold_arr = (1. / dens) * (hold_arr_num / hold_arr_dem) - 1.
                        hold_arr *= n_parts_tomo
                        mat_arr += hold_arr
                        t_nparts += n_parts_tomo

        # Compute metrics
        if dens_gl:
            if thick is None:
                return distances * (np.cbrt((1. / dens) * (mat_num / mat_dem)) - 1.)
            else:
                # return (1. / dens) * (mat_num / mat_dem) - 1.
                return mat_num / mat_dem
        else:
            return mat_arr / float(t_nparts)

    def simulate_uni_2nd_order_by_tomos(self, n_sims, temp_model, part_vtp,
                               distances, thick=None, border=True,
                               conv_iter=100, max_iter=100000, npr=None, temp_dmaps=None, verbose=False):
        """
        Simulate instances of Univariate 2nd order statistics of an input model
        :param n_sims: number of of instances for the simulation
        :param temp_model: model template class (child of Model class) for doing the simulations
        :param part_vtp: vtkPolyData used for as reference for the particle surfaces
        :param distances, thick, border, conv_iter, max_iter, npr, verbose: same as TomoParticles.compute_uni_2nd_order()
        :return:
        """

        # Input parsing
        from pyorg.surf import Model
        if n_sims <= 0:
            return None
        if not issubclass(temp_model, Model):
            error_msg = 'Input template model must be a subclass of Model.'
            raise pexceptions.PySegInputError(expr='simulate_uni_2nd_order (ListTomoParticles)',
                                              msg=error_msg)
        if not isinstance(part_vtp, vtk.vtkPolyData):
            error_msg = 'Input particle surface must be a vtkPolyData.'
            raise pexceptions.PySegInputError(expr='simulate_uni_2nd_order (ListTomoParticles)',
                                              msg=error_msg)
        if not hasattr(distances, '__len__'):
            error_msg = 'Input distances range must be an iterable of floats.'
            raise pexceptions.PySegInputError(expr='simulate_uni_2nd_order (ListTomoParticles)',
                                              msg=error_msg)

        # Initialization
        dsts = np.asarray(distances, dtype=np.float)
        mat = dict()

        # Loop for tomograms
        for key, tomo in zip(self.__tomos.iterkeys(), self.__tomos.itervalues()):
            if tomo.get_num_particles() > 0:
                if verbose:
                    print '\t\t\t+ Processing tomo ' + tomo.get_tomo_fname()
                # Loop for simulations
                sim_mat, n_real_sims = list(), 0
                for i in range(n_sims):
                    model = temp_model(tomo.get_voi(), part_vtp)
                    sim_tomo = model.gen_instance(tomo.get_num_particles(), tomo.get_tomo_fname(), mode='center')
                    n_parts_sim = sim_tomo.get_num_particles()
                    if n_parts_sim > 0:
                        hold_arr = sim_tomo.compute_uni_2nd_order(dsts, thick, border, conv_iter, max_iter,
                                                                    npr=npr, verbose=False)
                        if hold_arr is not None:
                            sim_mat.append(hold_arr)
                            n_real_sims += 1
                # Average simulations
                if n_real_sims > 0:
                    mat[tomo.get_tomo_fname()] = (1./n_real_sims) * np.asarray(sim_mat, dtype=np.float).sum(axis=0)

        return mat

    # Simulate instances of Univariate 2nd order statistics of an input model
    # n_sims: number of of instances for the simulation
    # temp_model: model template class (child of Model class) for doing the simulations
    # part_vtp: vtkPolyData used for as reference for the particle surfaces
    # dens_gl: if True (default) then density is estimated globally, otherwise locally
    # pointwise: if True (default) the matrics are computed for every point independently, otherwise output metric
    #            for every point is the average of all contained in the tomogram similarly to compute_uni_2nd_order()
    # rest of input parameters: equal to compute_uni_2nd_order
    # Returns: a matrix with n rows for every particle simulated and r columns for the distances
    def simulate_uni_2nd_order_matrix(self, n_sims, temp_model, part_vtp,
                               distances, thick=None, border=True, dens_gl=True, pointwise=True,
                               conv_iter=100, max_iter=100000, npr=None, verbose=False):

        # Input parsing
        from pyorg.surf import Model
        n_parts_list = self.get_num_particles()
        if (n_sims <= 0) or ():
            return None
        if not issubclass(temp_model, Model):
            error_msg = 'Input template model must be a subclass of Model.'
            raise pexceptions.PySegInputError(expr='simulate_uni_2nd_order (ListTomoParticles)',
                                              msg=error_msg)
        if not isinstance(part_vtp, vtk.vtkPolyData):
            error_msg = 'Input particle surface must be a vtkPolyData.'
            raise pexceptions.PySegInputError(expr='simulate_uni_2nd_order (ListTomoParticles)',
                                              msg=error_msg)
        if not hasattr(distances, '__len__'):
            error_msg = 'Input distances range must be an iterable of floats.'
            raise pexceptions.PySegInputError(expr='simulate_uni_2nd_order (ListTomoParticles)',
                                              msg=error_msg)

        # Initialization
        dsts = np.asarray(distances, dtype=np.float)
        mat = np.zeros(shape=(n_sims*n_parts_list, len(distances)), dtype=np.float32)
        hold_dens = None
        if dens_gl:
            hold_dens = self.compute_global_density()

        # Loop for tomograms
        part_id = 0
        for key, tomo in zip(self.__tomos.iterkeys(), self.__tomos.itervalues()):
            if tomo.get_num_particles() > 0:
                if verbose:
                    print '\t\t\t+ Processing tomo ' + tomo.get_tomo_fname()
                # Loop for simulations
                for i in range(n_sims):
                    model = temp_model(tomo.get_voi(), part_vtp)
                    sim_tomo = model.gen_instance(tomo.get_num_particles(), tomo.get_tomo_fname(), mode='center')
                    n_parts_sim = sim_tomo.get_num_particles()
                    if n_parts_sim > 0:
                        hold_mat = sim_tomo.compute_uni_2nd_order_matrix(dsts, thick, border,
                                                                         conv_iter, max_iter,
                                                                         dens_gl=hold_dens, pointwise=pointwise,
                                                                         npr=npr, verbose=False)
                        if hold_mat is not None:
                            for i in xrange(hold_mat.shape[0]):
                                mat[part_id, :] = hold_mat[i, :]
                                part_id +=1

        return mat[:part_id]

    # Compute Bivariate 2nd order statistics by tomograms
    # ltomos: input ListTomoParticles to compare
    # distances: range of distances (considered as and iterable of floats)
    # thick: nhood_vol thickess if None (default) then 'sphere', otherwise 'shell'
    # border: if True (default) border compensation is activated
    # conv_iter: number of iterations for convergence in volume estimation
    # max_iter: maximum number of iterations in volume estimation
    # npr: number of process, None (defaults) means they are adjusted automatically to the number of cores
    # verbose: if True then some progress text is provided (default False)
    # Returns: a dictionary with statistic metric array indexed by tomogram key
    def compute_bi_2nd_order_by_tomos(self, ltomos, distances, thick=None, border=True,
                             conv_iter=100, max_iter=100000, npr=None, verbose=False):

        # Input parsing
        if not isinstance(ltomos, ListTomoParticles):
            error_msg = 'Input ltomos must be a ListTomoParticles.'
            raise pexceptions.PySegInputError(expr='compute_bi_2nd_order (ListTomoParticles)',
                                              msg=error_msg)
        if not hasattr(distances, '__len__'):
            error_msg = 'Input distances range must be an iterable of floats.'
            raise pexceptions.PySegInputError(expr='compute_bi_2nd_order (ListTomoParticles)',
                                              msg=error_msg)

        # Initialization
        dsts = np.asarray(distances, dtype=np.float)
        mat = dict()

        # Call the counter within every tomogram
        for key, tomo_1 in zip(self.__tomos.iterkeys(), self.__tomos.itervalues()):
            try:
                tomo_2 = ltomos.__tomos[key]
                if (tomo_1.get_num_particles() > 0) and (tomo_2.get_num_particles() > 0):
                    if verbose:
                        print '\t\t\t+ Processing tomo ' + tomo_2.get_tomo_fname()
                    for attempt in range(MP_NUM_ATTEMPTS):
                        hold_arr = None
                        try:
                            hold_arr = tomo_1.compute_bi_2nd_order(tomo_2, dsts, thick, border,
                                                                               conv_iter, max_iter,
                                                                               npr=npr, verbose=False)
                        except pexceptions.PySegInputError:
                            continue
                        if hold_arr is not None:
                            break
                    if hold_arr is not None:
                        mat[tomo_1.get_tomo_fname()] = hold_arr
                    else:
                        error_msg = 'Number of unsuccessfully attempts for run ' \
                                    'TomoParticles.compute_bi_2nd_order reached: ' + str(MP_NUM_ATTEMPTS)
                        raise pexceptions.PySegInputError(expr='compute_bi_2nd_order (ListTomoParticles)',
                                                          msg=error_msg)
            except KeyError:
                print 'WARNING compute_bi_2nd_order (ListTomoParticles): tomogram ' + str(key) + ' did ' + \
                    'not have counterpart on input list of tomograms'
                continue

        return mat

    # Compute Bivariate 2nd order statistics by tomograms
    # ltomos: input ListTomoParticles to compare
    # distances: range of distances (considered as and iterable of floats)
    # thick: nhood_vol thickess if None (default) then 'sphere', otherwise 'shell'
    # border: if True (default) border compensation is activated
    # conv_iter: number of iterations for convergence in volume estimation
    # max_iter: maximum number of iterations in volume estimation
    # dens_gl: if True (default) then density is estimated globally, otherwise locally
    # npr: number of process, None (defaults) means they are adjusted automatically to the number of cores
    # verbose: if True then some progress text is provided (default False)
    # Returns: a dictionary with statistic metric array indexed by tomogram key
    def compute_bi_2nd_order(self, ltomos, distances, thick=None, border=True, dens_gl=True,
                             conv_iter=100, max_iter=100000, npr=None, verbose=False):

        # Input parsing
        if not isinstance(ltomos, ListTomoParticles):
            error_msg = 'Input ltomos must be a ListTomoParticles.'
            raise pexceptions.PySegInputError(expr='compute_bi_2nd_order (ListTomoParticles)',
                                                  msg=error_msg)
        if not hasattr(distances, '__len__'):
            error_msg = 'Input distances range must be an iterable of floats.'
            raise pexceptions.PySegInputError(expr='compute_bi_2nd_order (ListTomoParticles)',
                                                  msg=error_msg)

        # Initialization
        dsts = np.asarray(distances, dtype=np.float)
        n_parts_tomo_2_t = 0
        if dens_gl:
            mat_num, mat_dem = np.zeros(shape=dsts.shape[0], dtype=np.float), \
                               np.zeros(shape=dsts.shape[0], dtype=np.float)
            dens = self.compute_global_density()
        else:
            mat_arr = np.zeros(shape=dsts.shape[0], dtype=np.float)
            t_nparts = 0

        # Call the counter within every tomogram
        for key, tomo_1 in zip(self.__tomos.iterkeys(), self.__tomos.itervalues()):
            try:
                tomo_2 = ltomos.__tomos[key]
                n_parts_tomo_1, n_parts_tomo_2 = tomo_1.get_num_particles(), tomo_2.get_num_particles()
                if (tomo_1.get_num_particles() > 0) and (tomo_2.get_num_particles() > 0):
                    if verbose:
                        print '\t\t\t+ Processing tomo ' + tomo_2.get_tomo_fname()
                    hold_arr_num, hold_arr_dem = tomo_1.compute_bi_2nd_order(tomo_2, dsts, thick,
                                                                             border, conv_iter, max_iter,
                                                                             out_sep=True,
                                                                             npr=npr, verbose=False)
                    if (hold_arr_num is not None) and (hold_arr_dem is not None):
                        n_parts_tomo_2_t += n_parts_tomo_2
                        if dens_gl:
                            mat_num += hold_arr_num
                            mat_dem += hold_arr_dem
                        else:
                            dens = float(n_parts_tomo_2) / tomo_2.compute_voi_volume()
                            if thick is None:
                                hold_arr = distances * (np.cbrt((1. / dens) * (hold_arr_num / hold_arr_dem)) - 1.)
                            else:
                                hold_arr = (1. / dens) * (hold_arr_num / hold_arr_dem) - 1.
                            hold_arr *= n_parts_tomo_1
                            mat_arr += hold_arr
                            t_nparts += n_parts_tomo_1
                    else:
                        error_msg = 'Number of unsuccessfully attempts for run ' \
                                    'TomoParticles.compute_bi_2nd_order reached: ' + str(MP_NUM_ATTEMPTS)
                        raise pexceptions.PySegInputError(expr='compute_bi_2nd_order (ListTomoParticles)',
                                                          msg=error_msg)
            except KeyError:
                print 'WARNING compute_bi_2nd_order (ListTomoParticles): tomogram ' + str(key) + ' did ' + \
                      'not have counterpart on input list of tomograms'
                continue

        # Compute metrics
        if t_nparts == 0:
            return mat_num
        elif dens_gl:
            dens = ltomos.compute_global_density()
            if thick is None:
                return distances * (np.cbrt((1. / dens) * (mat_num / mat_dem)) - 1.)
            else:
                # return (1. / dens) * (mat_num / mat_dem) - 1.
                return mat_num / mat_dem
        else:
            if n_parts_tomo_2_t <= 0:
                return mat_arr
            else:
                return mat_arr / float(t_nparts)

    # Simulate instances of Bivariate 2nd order statistics of an input model
    # ltomos: input ListTomoParticles to compare
    # n_sims: number of of instances for the simulation
    # temp_model: model template class (child of Model class) for doing the simulations
    # part_vtp: vtkPolyData used for as reference for the particle surfaces
    # rest of input parameters: equal to compute_uni_2nd_order
    # Returns: a dictionary with statistic metric array indexed by tomogram key
    def simulate_bi_2nd_order_by_tomos(self, ltomos, n_sims, temp_model, part_vtp, distances,
                                       thick=None, border=True,
                                       conv_iter=100, max_iter=100000, npr=None, verbose=False):

        # Input parsing
        from pyorg.surf import Model
        if n_sims <= 0:
            return None
        if not issubclass(temp_model, Model):
            error_msg = 'Input template model must be a subclass of Model.'
            raise pexceptions.PySegInputError(expr='simulate_uni_2nd_order (ListTomoParticles)',
                                              msg=error_msg)
        if not isinstance(part_vtp, vtk.vtkPolyData):
            error_msg = 'Input particle surface must be a vtkPolyData.'
            raise pexceptions.PySegInputError(expr='simulate_uni_2nd_order (ListTomoParticles)',
                                              msg=error_msg)
        if not hasattr(distances, '__len__'):
            error_msg = 'Input distances range must be an iterable of floats.'
            raise pexceptions.PySegInputError(expr='simulate_uni_2nd_order (ListTomoParticles)',
                                              msg=error_msg)

        # Initialization
        dsts = np.asarray(distances, dtype=np.float)
        mat = dict()

        # Loop for tomograms
        for key, tomo_1 in zip(self.__tomos.iterkeys(), self.__tomos.itervalues()):
            try:
                tomo_2 = ltomos.__tomos[key]
                n_parts_tomo_1, n_parts_tomo_2 = tomo_1.get_num_particles(), tomo_2.get_num_particles()
                if (n_parts_tomo_1 > 0) and (n_parts_tomo_2 > 0):
                    if verbose:
                        print '\t\t\t+ Processing tomo ' + tomo_2.get_tomo_fname()
                    # Loop for simulations
                    sim_mat, n_real_sims = list(), 0
                    for i in range(n_sims):
                        n_real_sims = 0
                        model = temp_model(tomo_2.get_voi(), part_vtp)
                        sim_tomo = model.gen_instance(n_parts_tomo_2, tomo_2.get_tomo_fname(), mode='center')
                        n_parts_sim = float(sim_tomo.get_num_particles())
                        if n_parts_sim > 0:
                            hold_arr = tomo_1.compute_bi_2nd_order(sim_tomo, dsts, thick, border, conv_iter, max_iter,
                                                                       npr=npr, verbose=verbose)
                            if hold_arr is not None:
                                sim_mat.append(hold_arr)
                                n_real_sims += 1
                    # Average simulations
                    if n_real_sims > 0:
                        mat[tomo_1.get_tomo_fname()] = (1. / n_real_sims) * \
                                                           np.asarray(sim_mat, dtype=np.float).sum(axis=0)
            except KeyError:
                print 'WARNING simulate_bi_2nd_order (ListTomoParticles): tomogram ' + str(key) + ' did ' + \
                            'not have counterpart on input list of tomograms'
                continue

        return mat

    # Simulate instances of Bivariate 2nd order statistics of an input model
    # ltomos: input ListTomoParticles to compare
    # n_sims: number of of instances for the simulation
    # temp_model: model template class (child of Model class) for doing the simulations
    # part_vtp: vtkPolyData used for as reference for the particle surfaces
    # dens_gl: if True (default) then density is estimated globally, otherwise locally
    # pointwise: if True (default) the matrics are computed for every point independently, otherwise output metric
    #            for every point is the average of all contained in the tomogram similarly to compute_uni_2nd_orde
    # rest of input parameters: equal to compute_uni_2nd_order
    # Returns: a matrix with n rows for every particle simulated and r columns for the distances
    def simulate_bi_2nd_order_matrix(self, ltomos, n_sims, temp_model, part_vtp,
                               distances, thick=None, border=True, dens_gl=True, pointwise=True,
                               conv_iter=100, max_iter=100000, npr=None, verbose=False):

        # Input parsing
        from pyorg.surf import Model
        n_parts_list = self.get_num_particles()
        if (n_sims <= 0) or ():
            return None
        if not issubclass(temp_model, Model):
            error_msg = 'Input template model must be a subclass of Model.'
            raise pexceptions.PySegInputError(expr='simulate_uni_2nd_order (ListTomoParticles)',
                                              msg=error_msg)
        if not isinstance(part_vtp, vtk.vtkPolyData):
            error_msg = 'Input particle surface must be a vtkPolyData.'
            raise pexceptions.PySegInputError(expr='simulate_uni_2nd_order (ListTomoParticles)',
                                              msg=error_msg)
        if not hasattr(distances, '__len__'):
            error_msg = 'Input distances range must be an iterable of floats.'
            raise pexceptions.PySegInputError(expr='simulate_uni_2nd_order (ListTomoParticles)',
                                              msg=error_msg)

        # Initialization
        dsts = np.asarray(distances, dtype=np.float)
        mat = np.zeros(shape=(n_sims * n_parts_list, len(distances)), dtype=np.float32)
        hold_dens = None
        if dens_gl:
            hold_dens = ltomos.compute_global_density()

        # Loop for tomograms
        part_id = 0
        for key, tomo_1 in zip(self.__tomos.iterkeys(), self.__tomos.itervalues()):
            try:
                tomo_2 = ltomos.__tomos[key]
                n_parts_tomo_1, n_parts_tomo_2 = tomo_1.get_num_particles(), tomo_2.get_num_particles()
                if (n_parts_tomo_1 > 0) and (n_parts_tomo_2 > 0):
                    if verbose:
                        print '\t\t\t+ Processing tomo ' + tomo_2.get_tomo_fname()
                    # Loop for simulations
                    for i in range(n_sims):
                        n_real_sims = 0
                        model = temp_model(tomo_2.get_voi(), part_vtp)
                        sim_tomo = model.gen_instance(n_parts_tomo_2, tomo_2.get_tomo_fname(), mode='center')
                        n_parts_sim = float(sim_tomo.get_num_particles())
                        if n_parts_sim > 0:
                            hold_mat = tomo_1.compute_bi_2nd_order_matrix(sim_tomo, dsts, thick, border,
                                                                         conv_iter, max_iter,
                                                                         dens_gl=hold_dens, pointwise=True,
                                                                         npr=npr, verbose=False)
                            if hold_mat is not None:
                                for i in xrange(hold_mat.shape[0]):
                                    mat[part_id, :] = hold_mat[i, :]
                                    part_id += 1
            except KeyError:
                # print 'WARNING simulate_bi_2nd_order (ListTomoParticles): tomogram ' + str(key) + ' did ' + \
                #         'not have counterpart on input list of tomograms'
                continue

        return mat[:part_id]

    # Estimates global densities for every tomogram and return them in a dictionary
    # surface: if False (default) then densities are estimated by volume, otherwise by surface
    def compute_global_densities(self, surface=False):
        dens = dict()
        for key, tomo in zip(self.__tomos.iterkeys(), self.__tomos.itervalues()):
            dens[key] = tomo.compute_global_density(surface=surface)
        return dens

    # Estimates global density
    # surface: if False (default) then densities are estimated by volume, otherwise by surface
    def compute_global_density(self, surface=False):
        n_parts, vol = 0, 0.
        for tomo in self.__tomos.itervalues():
            n_parts += tomo.get_num_particles()
            if surface:
                vol += tomo.compute_voi_surface()
            else:
                vol += tomo.compute_voi_volume()
        if vol <= 0:
            return 0.
        else:
            return float(n_parts) / vol

    # For every tomogram append all particles into one vtkPolyData and store it in the specified
    # folder
    # out_dir: output directory to store the output tomograms
    # out_stem: (default '') output stem string added to the stored files
    def store_appended_tomos(self, out_dir, out_stem='', mode='surface'):
        for key, tomo in zip(self.__tomos.iterkeys(), self.__tomos.itervalues()):
            disperse_io.save_vtp(tomo.append_particles_vtp(mode=mode),
                                 out_dir+'/'+out_stem+key+'_append.vtp')

    # Delete for list the tomogram with low particles
    # min_num_particles: minimum number of particles, below that the tomogram is deleted (default 1)
    def filter_by_particles_num(self, min_num_particles=1):
        hold_dict = dict()
        for key, tomo in zip(self.__tomos.iterkeys(), self.__tomos.itervalues()):
            # print key + ': ' + str(tomo.get_num_particles())
            if tomo.get_num_particles() >= min_num_particles:
                hold_dict[key] = tomo
        self.__tomos = hold_dict

    # Clean tomogram with a low amount of particles
    # n_keep: number of tomograms to, the one with the highest amount of particles
    def clean_low_pouplated_tomos(self, n_keep):
        if (n_keep is None) or (n_keep < 0) or (n_keep >= len(self.__tomos.keys())):
            return
        # Sorting loop
        n_parts = dict()
        for key, tomo in zip(self.__tomos.iterkeys(), self.__tomos.itervalues()):
            n_parts[key] = tomo.get_num_particles()
        pargs_sort = np.argsort(np.asarray(n_parts.values()))[::-1]
        keys = n_parts.keys()
        # Cleaning loop
        hold_dict = dict()
        for parg in pargs_sort[:n_keep]:
            key = keys[parg]
            hold_dict[key] = self.__tomos[key]
        self.__tomos = hold_dict

    # Returns a dictionary with the num particles by tomos
    def particles_by_tomos(self):
        keys = self.get_tomo_fname_list()
        part = dict.fromkeys(keys)
        for key in keys:
            part[key] = self.__tomos[key].get_num_particles()
        return part

    # Returns a dictionary with the num particles by tomos
    # surface: if True (default False) then density is estimated by surface instead of volume
    def densities_by_tomos(self, surface=False):
        keys = self.get_tomo_fname_list()
        dens = dict.fromkeys(keys)
        for key in keys:
            dens[key] = self.__tomos[key].compute_global_density(surface=surface)
        return dens

    # Compute total density combining al tomograms
    def total_density(self):
        num, vols = 0., 0.
        for tomo in self.__tomos.itervalues():
            num += tomo.get_num_particles()
            vols += tomo.compute_global_density()
        if vols <= 0:
            return 0.
        else:
            return float(num) / vols

    # Generates a particles STAR file without rlnImageName
    # Return: a STAR file
    def to_particles_star(self):

        # Initialization
        star_part = sub.Star()
        hold_part = None
        for tomo in self.get_tomo_list():
            for part in tomo.get_particles():
                hold_part = part
                if hold_part is not None:
                    break
        #     if hold_part is None:
        #         break
        # if hold_part is None:
        #     return None
        meta_dic = hold_part.get_meta()
        for key in meta_dic.iterkeys():
            star_part.add_column(key)
        star_part.add_column('_rlnGroupNumber')

        # Create micrographs dictionary
        mic_dic = dict()
        for i, tomo in enumerate(self.get_tomo_list()):
            mic_dic[tomo.get_tomo_fname()] = i

        # Tomograms loop
        for tomo in self.get_tomo_list():
            tomo_fname = tomo.get_tomo_fname()
            for part in tomo.get_particles():
                kwargs, meta_dic = dict(), part.get_meta()
                for key, val in zip(meta_dic.iterkeys(), meta_dic.itervalues()):
                    kwargs[key] = val
                kwargs['_rlnGroupNumber'] = mic_dic[tomo_fname]
                # Insert the particles into the star file
                star_part.add_row(**kwargs)

        return star_part

    # Generates a STAR file with TomoParicles pickles
    # out_dir: output directory for the pickles
    # Return: a STAR file
    def to_tomos_star(self, out_dir):

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

    # Purge particles by scale suppresion
    # ss: scale for supression
    def scale_suppression(self, ss):
        for tomo in self.__tomos.itervalues():
            if tomo.get_num_particles() > 0:
                tomo.scale_suppression(ss)

    def ensure_embedding(self):
        """
        Ensure that all inserted particles are embedded (actually only computed for VOI as array)
        :return: True if all are embedded
        """
        for tomo in self.__tomos.itervalues():
            if not tomo.ensure_embedding():
                return False
        return True

    # INTERNAL FUNCTIONALITY AREA

    # Restore previous state
    def __setstate__(self, state):
        self.__dict__.update(state)
        # Restore unpickable objects
        self.__tomos = dict()
        for key, pkl in self.__pkls.iteritems():
            self.__tomos[key] = unpickle_obj(pkl)

    # Copy the object's state from self.__dict__ which contains all instance attributes.
    # Afterwards remove unpickable objects
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_ListTomoParticles__tomos']
        return state


############################################################################
# Class for a set of list tomograms with embedded particles
#
class SetListTomoParticles(object):

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

    # Add a new ListTomoParticles to the set
    # ltomos: input ListTomoParticles object
    # list_name: string for naming the list
    def add_list_tomos(self, ltomos, list_name):
        # Input parsing (compatible with older versions)
        if ltomos.__class__.__name__ != 'ListTomoParticles':
            error_msg = 'WARNING: add_tomo (ListTomoParticles): ltomos input must be ListTomoParticles object.'
            raise pexceptions.PySegInputError(expr='add_tomo (ListTomoParticles)', msg=error_msg)
        # Adding the list to the dictionary
        self.__lists[str(list_name)] = ltomos

    # Returns: retuns a set with all tomos in all list
    def get_set_tomos(self):
        tomos_l = list()
        for key, ltomos in zip(self.__lists.iterkeys(), self.__lists.itervalues()):
            for tomo in ltomos.get_tomo_list():
                tomos_l.append(tomo.get_tomo_fname())
        return set(tomos_l)

    def merge_to_one_list(self, list_names=None):
        """
        Merge the particles lists into one
        :param list_names: a list with name of the list to merge, if None (default) then all are merged
        :return: a ListTomoParticles object
        """

        # Input parsing
        if list_names is None:
            list_names = self.__lists.keys()
        out_list = ListTomoParticles()

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
            hold_tomo = TomoParticles(tomo_name, 1, voi=vois[tomo_name])
            out_list.add_tomo(hold_tomo)
            for list_name in list_names:
                hold_list = self.__lists[list_name]
                hold_ltomo = hold_list.get_tomo_by_key(tomo_name)
                for hold_part in hold_ltomo.get_particles():
                    out_list.insert_particle(copy.deepcopy(hold_part), tomo_name, check_bounds=False,
                                              mode='full', voi_pj=False, voi_to_pj=None)

        return out_list

    # Returns a dictionary with the number of particles by list
    def particles_by_list(self):
        parts = dict()
        for key, ltomos in zip(self.__lists.iterkeys(), self.__lists.itervalues()):
            parts[key] = ltomos.get_num_particles()
        return parts

    # Returns a dictionary with the number of particles by tomogram
    def particles_by_tomos(self):
        parts = dict()
        for key_l, ltomos in zip(self.__lists.iterkeys(), self.__lists.itervalues()):
            for tomo in ltomos.get_tomo_list():
                try:
                    parts[tomo.get_tomo_fname()] += tomo.get_num_particles()
                except KeyError:
                    parts[tomo.get_tomo_fname()] = tomo.get_num_particles()
        return parts

    # Returns a dictionary with the proportions for every tomogram
    def proportions_by_tomos(self):
        parts = dict()
        for key_l, ltomos in zip(self.__lists.iterkeys(), self.__lists.itervalues()):
            for tomo in ltomos.get_tomo_list():
                key_t = tomo.get_tomo_fname()
                try:
                    parts[key_t].append(tomo.get_num_particles())
                except KeyError:
                    parts[key_t] = list()
                    parts[key_t].append(tomo.get_num_particles())
        return parts

    # Returns a dictionary with the proportions for every tomogram
    def proportions_by_list(self):
        # Initialization
        parts_list, parts_tomo = dict.fromkeys(self.__lists.keys()), dict.fromkeys(self.get_set_tomos())
        for key_t in parts_tomo.iterkeys():
            parts_tomo[key_t] = 0
        for key_l in parts_list.iterkeys():
            parts_list[key_l] = np.zeros(shape=len(parts_tomo.keys()))
        # Particles loop
        for key_l, ltomos in zip(self.__lists.iterkeys(), self.__lists.itervalues()):
            for i, key_t in enumerate(parts_tomo.keys()):
                tomo = ltomos.get_tomo_by_key(key_t)
                hold_parts = tomo.get_num_particles()
                parts_tomo[key_t] += hold_parts
                parts_list[key_l][i] += hold_parts
        # Proportions loop
        for key_l in parts_list.iterkeys():
            for i, tomo_nparts in enumerate(parts_tomo.values()):
                if tomo_nparts > 0:
                    parts_list[key_l][i] /= tomo_nparts
                else:
                    parts_list[key_l][i] = 0.
        return parts_list

    # Returns a dictionary with the number of particles by list
    def density_by_list(self, surface=False):
        dens = dict()
        for key, ltomos in zip(self.__lists.iterkeys(), self.__lists.itervalues()):
            dens[key] = ltomos.compute_global_density(surface=surface)
        return dens

    # Returns a dictionary with the number of particles by tomograms
    def density_by_tomos(self, surface=False):
        dens, n_dens = dict(), dict()
        for key_l, ltomos in zip(self.__lists.iterkeys(), self.__lists.itervalues()):
            for tomo in ltomos.get_tomo_list():
                try:
                    dens[tomo.get_tomo_fname()] += tomo.compute_global_density(surface=surface)
                except KeyError:
                    dens[tomo.get_tomo_fname()] = tomo.compute_global_density(surface=surface)
                try:
                    n_dens[tomo.get_tomo_fname()] += 1.
                except KeyError:
                    n_dens[tomo.get_tomo_fname()] = 1.
        for key in dens.iterkeys():
            dens[key] /= n_dens[key]
        return dens

    # Compute Univariate 2nd order statistics by list
    # distances: range of distances (considered as and iterable of floats)
    # thick: nhood_vol thickes if None (default) then 'sphere', otherwise 'shell'
    # border: if True (default) border compensation is activated
    # conv_iter: number of iterations for convergence in volume estimation
    # max_iter: maximum number of iterations in volume estimation
    # dens_gl: if True (default) then density is estimated globally, otherwise locally
    # npr: number of process, None (defaults) means they are adjusted automatically to the number of cores
    # verbose: if True then some progress text is provided (default False)
    # Returns: a dictionary indexed by ListTomoParticles names with tomograms dictionaries
    def compute_uni_2nd_order_by_list(self, distances, thick=None, border=True,
                                      conv_iter=100, max_iter=100000, dens_gl=True,
                                      npr=None, verbose=False):
        mats = dict()
        count = 1
        for key_l, ltomos in zip(self.__lists.iterkeys(), self.__lists.itervalues()):
            if verbose:
                print '\t\t+ Processing list ' + str(count) + ' of ' + str(len(self.__lists.keys()))
                count += 1
            mats[key_l] = ltomos.compute_uni_2nd_order(distances=distances, thick=thick, border=border,
                                                       conv_iter=conv_iter, max_iter=max_iter, dens_gl=dens_gl,
                                                       npr=npr, verbose=False)
        return mats

    # Simulate Univariate 2nd order statistics matrix
    # n_sims: number of of instances for the simulation
    # temp_model: model template class (child of Model class) for doing the simulations
    # part_vtp: vtkPolyData used for as reference for the particle surfaces
    # distances: range of distances (considered as and iterable of floats)
    # thick: nhood_vol thickes if None (default) then 'sphere', otherwise 'shell'
    # border: if True (default) border compensation is activated
    # conv_iter: number of iterations for convergence in volume estimation
    # max_iter: maximum number of iterations in volume estimation
    # dens_gl: if True (default) then density is estimated globally, otherwise locally
    # pointwise: if True (default) the matrics are computed for every point independently, otherwise output metric
    #            for every point is the average of all contained in the tomogram similarly to compute_uni_2nd_order()
    # npr: number of process, None (defaults) means they are adjusted automatically to the number of cores
    # verbose: if True then some progress text is provided (default False)
    # Returns: a dictionary indexed by ListTomoParticles names with simulation matrices
    def simulate_uni_2nd_order_matrix(self, n_sims, temp_model, part_vtp,
                                      distances, thick=None, border=True,
                                      conv_iter=100, max_iter=100000, dens_gl=True, pointwise=True, keep=None,
                                      npr=None, verbose=False):
        mats = dict()
        count = 1
        for key_l, ltomos in zip(self.__lists.iterkeys(), self.__lists.itervalues()):
            if verbose:
                print '\t\t+ Processing list ' + str(count) + ' of ' + str(len(self.__lists.keys()))
                count += 1
            mats[key_l] =ltomos.simulate_uni_2nd_order_matrix(n_sims, temp_model, part_vtp,
                                                              distances=distances, thick=thick, border=border,
                                                              conv_iter=conv_iter, max_iter=max_iter,
                                                              dens_gl=dens_gl, pointwise=pointwise,
                                                              npr=npr, verbose=verbose)
        return mats

    # Compute Bivariate 2nd order statistics
    # stomos: input SetListTomoParticles to compare
    # distances: range of distances (considered as and iterable of floats)
    # thick: nhood_vol thickes if None (default) then 'sphere', otherwise 'shell'
    # border: if True (default) border compensation is activated
    # conv_iter: number of iterations for convergence in volume estimation
    # max_iter: maximum number of iterations in volume estimation
    # no_eq_name: not compare ListTomoParticles with the same name inserted (default True)
    # dens_gl: if True (default) then density is estimated globally, otherwise locally
    # keep: if None (default) all tomograms in a list are used, otherwise, as minimum, a only those 'keep' with
    #       highest amount or particles are used for computations
    # npr: number of process, None (defaults) means they are adjusted automatically to the number of cores
    # verbose: if True then some progress text is provided (default False)
    # Returns: a dictionary (density ratios) indexed by ListTomoParticles names with the computed matrices
    def compute_bi_2nd_order_by_list(self, stomos, distances, thick=None, border=True,
                                     conv_iter=100, max_iter=100000, no_eq_name=True, dens_gl=True, keep=None,
                                     npr=None, verbose=False):
        slists = stomos.get_lists()
        mats_o = dict()
        count = 1
        for key_o, ltomos_o in zip(self.__lists.iterkeys(), self.__lists.itervalues()):
            mats_i = dict()
            for key_i, ltomos_i in zip(slists.iterkeys(), slists.itervalues()):
                if no_eq_name and (key_o == key_i):
                    continue
                if verbose:
                    print '\t\t+ Processing list ' + key_o + ' vs  list' + key_i
                    count += 1
                if keep is not None:
                    ltomos_o.clean_low_pouplated_tomos(keep)
                mats_i[key_i] = ltomos_o.compute_bi_2nd_order(ltomos_i, distances=distances,
                                                              thick=thick, border=border, dens_gl=dens_gl,
                                                              conv_iter=conv_iter, max_iter= max_iter,
                                                              npr=npr, verbose=verbose)
            mats_o[key_o] = mats_i
        return mats_o

    # Simulate Bivariate 2nd order statistics matrix
    # stomos: input SetListTomoParticles to compare
    # n_sims: number of of instances for the simulation
    # temp_model: model template class (child of Model class) for doing the simulations
    # part_vtp: vtkPolyData used for as reference for the particle surfaces
    # distances: range of distances (considered as and iterable of floats)
    # thick: nhood_vol thickes if None (default) then 'sphere', otherwise 'shell'
    # border: if True (default) border compensation is activated
    # conv_iter: number of iterations for convergence in volume estimation
    # max_iter: maximum number of iterations in volume estimation
    # no_eq_name: not compare ListTomoParticles with the same name inserted (default True)
    # dens_gl: if True (default) then density is estimated globally, otherwise locally
    # pointwise: if True (default) the matrics are computed for every point independently, otherwise output metric
    #            for every point is the average of all contained in the tomogram similarly to compute_uni_2nd_order()
    # keep: if None (default) all tomograms in a list are used, otherwise, as minimum, a only those 'keep' with
    #       highest amount or particles are used for computations
    # npr: number of process, None (defaults) means they are adjusted automatically to the number of cores
    # verbose: if True then some progress text is provided (default False)
    # Returns: a dictionary indexed by ListTomoParticles names with simulation matrices
    def simulate_bi_2nd_order_matrix(self, stomos, n_sims, temp_model, part_vtp,
                                     distances, thick=None, border=True,
                                     conv_iter=100, max_iter=100000, no_eq_name=True, dens_gl=True,
                                     pointwise=True, keep=None, npr=None, verbose=False):

        slists = stomos.get_lists()
        mats_o = dict()
        count = 1
        for key_o, ltomos_o in zip(self.__lists.iterkeys(), self.__lists.itervalues()):
            mats_i = dict()
            for key_i, ltomos_i in zip(slists.iterkeys(), slists.itervalues()):
                if no_eq_name and (key_o == key_i):
                    continue
                if verbose:
                    print '\t\t+ Processing list ' + key_o + ' vs  list' + key_i
                    count += 1
                if keep is not None:
                    ltomos_o.clean_low_pouplated_tomos(keep)
                mats_i[key_i] = ltomos_o.simulate_bi_2nd_order_matrix(ltomos_i, n_sims, temp_model, part_vtp,
                                                                  distances=distances, thick=thick, border=border,
                                                                  conv_iter=conv_iter, max_iter=max_iter,
                                                                  dens_gl=dens_gl, pointwise=pointwise,
                                                                  npr=npr, verbose=verbose)
            mats_o[key_o] = mats_i
        return mats_o

    # Generates a dictionary of vtkPolyData files for every tomogram by gathering all lists
    # mode: (default 'surface') mode for generating the vtkPolyData for every particle,
    #       see Particle.get_vtp()
    def tomos_to_vtp(self, mode='surface'):

        # Loop for initialization
        appenders = dict()
        for key, ltomo in zip(self.__lists.iterkeys(), self.__lists.itervalues()):
            # Loop for tomos
            for tomo in ltomo.get_tomo_list():
                tomo_fname = tomo.get_tomo_fname()
                if not (tomo_fname in appenders.keys()):
                    appenders[tomo_fname] = None
                # Add a key only if ic can be converted to an integer
                l_stem = os.path.splitext(os.path.split(key)[1])[0]
                try:
                    idx = l_stem.index('_')
                    hold_vtp = tomo.append_particles_vtp(mode=mode, lbl=int(l_stem[:idx]))
                except (IndexError, ValueError):
                    hold_vtp = tomo.append_particles_vtp(mode=mode, lbl=None)
                if (hold_vtp is not None) and (hold_vtp.GetNumberOfCells() > 0):
                    if appenders[tomo_fname] is None:
                        appenders[tomo_fname] = vtk.vtkAppendPolyData()
                    copy_vtp = vtk.vtkPolyData()
                    copy_vtp.DeepCopy(hold_vtp)
                    appenders[tomo_fname].AddInputData(copy_vtp)

        # Loop for appending
        polys = dict()
        for key, appender in zip(appenders.iterkeys(), appenders.itervalues()):
            if appender is not None:
                appender.Update()
                polys[key] = appender.GetOutput()

        return polys

    # Generates a particles STAR file without rlnImageName
    # Return: a STAR file, every list is stored as a different class (_rlnClassNumber)
    def to_particles_star(self):

        # Initialization
        star_part = sub.Star()
        hold_part = None
        for i, llist in enumerate(self.get_lists().values()):
            for tomo in llist.get_tomo_list():
                for part in tomo.get_particles():
                    hold_part = part
                    if hold_part is None:
                        break
                if hold_part is None:
                    break
            if hold_part is None:
                break
        if hold_part is None:
            return None
        meta_dic = hold_part.get_meta()
        for key in meta_dic.iterkeys():
            star_part.add_column(key)
        star_part.add_column('_rlnGroupNumber')
        star_part.add_column('_rlnClassNumber')

        # Create micrographs dictionary
        mic_dic = dict()
        for i, tomo_fname in enumerate(self.get_tomos_fname()):
            mic_dic[tomo_fname] = i

        # Lists loop
        for i, lkey, list in zip(range(len(self.__lists)), self.__lists.iterkeys(), self.__lists.itervalues()):
            # Tomograms loop
            for tomo in list.get_tomo_list():
                tomo_fname = tomo.get_tomo_fname()
                for part in tomo.get_particles():
                    kwargs, meta_dic = dict(), part.get_meta()
                    for key, val in zip(meta_dic.iterkeys(), meta_dic.itervalues()):
                        kwargs[key] = val
                    kwargs['_rlnGroupNumber'] = mic_dic[tomo_fname]
                    try:
                        klass = int(os.path.split(lkey)[1].split('_')[0])
                    except ValueError:
                        klass = -1
                    kwargs['_rlnClassNumber'] = klass
                    # Insert the particles into the star file
                    star_part.add_row(**kwargs)

        return star_part

    # Generates a STAR file with the ListTomoParicles and pickes their TomoParticles
    # out_star: output STAR file
    # out_dir_pkl: output directory for the pickles
    # Return: a STAR file
    def pickle_tomo_star(self, out_star, out_dir_pkl):

        # Initialization
        star_list = sub.Star()
        star_list.add_column('_psPickleFile')

        # Tomograms loop
        for lname, ltomo in zip(self.__lists.iterkeys(), self.__lists.itervalues()):
            lkey = os.path.splitext(os.path.split(lname)[1])[0]
            out_star, out_list_dir = out_dir_pkl + '/' + lkey + '_tp.star', out_dir_pkl + '/' + lkey + '_tp'
            clean_dir(out_list_dir)
            list_star = ltomo.to_tomo_star(out_list_dir)
            list_star.store(out_star)
            # Insert the pickled tomo into the star file
            star_list.set_column_data('_psPickleFile', out_star)

        star_list.store(out_star)

    # Purge particles by scale suppresion
    # ss: scale for supression
    # ref_list: reference list key for crossed scale suppression
    def scale_suppression(self, ss, ref_list=None):
        if ref_list is None:
            for tlist in self.__lists.itervalues():
                tlist.scale_suppression(ss)
        else:
            try:
                rlist = self.__lists[ref_list]
                sk_rlist = ref_list
            except KeyError:
                sk_rlist = self.get_key_from_short_key(ref_list)
                rlist = self.__lists[sk_rlist]
            for tomo in rlist.get_tomo_list():
                tomo.scale_suppression(ss)
            for key, hlist in zip(self.__lists.iterkeys(), self.__lists.itervalues()):
                if key != sk_rlist:
                    for tomo in hlist.get_tomo_list():
                        try:
                            rtomo = rlist.get_tomo_by_key(tomo.get_tomo_fname())
                        except KeyError:
                            continue
                        ref_coords = rtomo.get_particle_coords()
                        tomo.scale_suppression(ss, ext_coords=ref_coords)

    #
    def filter_by_particles_num_tomos(self, min_num_particles=1):
        """
        Delete those tomograms with a number of particles lower than an input value for any list
        :param min_num_particles: a number or a dict, the allows to specify different minimum number for each layer
        :return:
        """

        # Computing total particles per tomogram loop
        if isinstance(min_num_particles, dict):
            tomos_dict = dict().fromkeys(self.get_set_tomos(), 0)
            for lkey, ltomos in zip(self.__lists.iterkeys(), self.__lists.itervalues()):
                hold_min = min_num_particles[lkey]
                for tomo in ltomos.get_tomo_list():
                    if tomo.get_num_particles() >= hold_min:
                        tomos_dict[tomo.get_tomo_fname()] += 1
            tomos_del = dict().fromkeys(self.get_set_tomos(), False)
            for key in tomos_dict.iterkeys():
                if tomos_dict[key] < len(min_num_particles.keys()):
                    tomos_del[key] = True
        else:
            tomos_del = dict().fromkeys(self.get_set_tomos(), False)
            for tkey in tomos_del.keys():
                for ltomos in self.__lists.itervalues():
                    try:
                        tomo = ltomos.get_tomo_by_key(tkey)
                    except KeyError:
                        continue
                    if tomo.get_num_particles() < min_num_particles:
                        tomos_del[tkey] = True

        # Deleting loop
        for ltomos in self.__lists.itervalues():
            for tkey in tomos_del.keys():
                if tomos_del[tkey]:
                    try:
                        ltomos.del_tomo(tkey)
                    except KeyError:
                        continue

    # Delete for list the tomogram with low particles (considering all lists)
    # min_num_particles: minimum number of particles, below that the tomogram is deleted (default 1)
    def filter_by_particles_num(self, min_num_particles=1):

        # Computing total particles per tomogram loop
        hold_dict = dict()
        for ltomos in self.__lists.itervalues():
            for tomo in ltomos.get_tomo_list():
                tkey, n_parts = tomo.get_tomo_fname(), tomo.get_num_particles()
                try:
                    hold_dict[tkey] += n_parts
                except KeyError:
                    hold_dict[tkey] = n_parts

        # Deleting loop
        for tkey, n_parts in zip(hold_dict.iterkeys(), hold_dict.itervalues()):
            if n_parts < min_num_particles:
                for ltomos in self.__lists.itervalues():
                    ltomos.del_tomo(tkey)

    def print_tomo_particles(self, out_dir, shapes, seg_mic_dir=None, over=False, sg=None):
        '''
        Print particles in each corresponding tomogram and store it in a directory
        :param out_dir: output directory to store the tomograms
        :param shapes: dictionary pairing particles list with its corresponding path to shape model subvolume
               VERY IMPORTANT: so far only binary shapes are considered and not rotations are considered,
                               so results are guaranteed only for spherical and binary shapes
        :param seg_mic_dir: dictionary for pairing tomograms segmentation parth with the orginal density map
                            (optional)
        :param over: (default False) control if printed particles are overlapped in the same label field or not
        :param sg: sigma for Gaussian filtering the density scalar field (default None: not applied)
        :return: Stores in output a vtkImageData file for each tomograms with the particles printed
        '''

        # Initialization
        shapes_tomo = dict()

        # Loading particle shapes
        for ltomos_key, shape_path in zip(shapes.iterkeys(), shapes.itervalues()):
            shapes_tomo[ltomos_key] = disperse_io.load_tomo(shape_path)

        # Lists loop
        # lbl = int(0)
        for ltomos_key, ltomos in zip(self.__lists.iterkeys(), self.__lists.itervalues()):

            tomos = ltomos.get_tomo_list()
            if over:
                lbl = int(ltomos_key.split('_')[0])
            else:
                lbl = ltomos_key.split('_')[0]

            # Tomograms loop
            for tomo in tomos:

                # Storing/Loading the output tomogram
                if seg_mic_dir is None:
                    tomo_fname = tomo.get_tomo_fname()
                else:
                    tomo_fname = seg_mic_dir[tomo.get_tomo_fname()]
                shape = shapes_tomo[ltomos_key]
                off_svol = np.ceil(.5 * np.asarray(shape.shape, dtype=np.float32)).astype(np.int)
                out_stem = os.path.split(tomo_fname)[0].replace('/', '_')
                out_path_tomo = out_dir + '/' + out_stem + '.vti'
                if os.path.exists(out_path_tomo):
                    tomo_vti = disperse_io.load_vti(out_stem + '.vti', out_dir)
                    if over:
                        hold_field = tomo_vti.GetPointData().GetArray(PARTS_LBL_FIELD)
                    else:
                        hold_field = numpy_support.numpy_to_vtk(
                            num_array=(-1) * np.ones(shape=np.prod(hold_tomo.shape), dtype=np.int),
                            deep=True, array_type=vtk.VTK_INT)
                        hold_field.SetName(PARTS_LBL_FIELD + '_' + lbl)
                        tomo_vti.GetPointData().AddArray(hold_field)
                else:
                    hold_tomo = disperse_io.load_tomo(tomo_fname)
                    if sg is not None:
                        hold_tomo = sp.ndimage.filters.gaussian_filter(hold_tomo, sg)
                    # Create the new vtkImageData
                    tomo_vti = vtk.vtkImageData()
                    tomo_vti.SetSpacing([1, 1, 1])
                    tomo_vti.SetDimensions(hold_tomo.shape)
                    tomo_vti.AllocateScalars(vtk.VTK_FLOAT, 1)
                    hold_field = numpy_support.numpy_to_vtk(num_array=hold_tomo.ravel(order='F'),
                                                            deep=True, array_type=vtk.VTK_FLOAT)
                    tomo_vti.GetPointData().SetScalars(hold_field)
                    hold_field = numpy_support.numpy_to_vtk(num_array=(-1)*np.ones(shape=np.prod(hold_tomo.shape), dtype=np.int),
                                                            deep=True, array_type=vtk.VTK_INT)
                    if over:
                        hold_field.SetName(PARTS_LBL_FIELD)
                    else:
                        hold_field.SetName(PARTS_LBL_FIELD + '_' + lbl)
                    tomo_vti.GetPointData().AddArray(hold_field)

                # Printing the current list of particles
                nx, ny, nz = tomo_vti.GetDimensions()
                shape_coords = np.where(shape != 0)
                xs, ys, zs = shape_coords[0], shape_coords[1], shape_coords[2]
                # Particles loop
                coords = tomo.get_particle_coords()
                if coords is not None:
                    for cent_coord in coords:
                        # Loop for printing the shape
                        for i in range(len(xs)):
                            sh_coord = np.asarray([xs[i], ys[i], zs[i]])
                            coord = cent_coord + sh_coord - off_svol
                            if (coord[0] >= 0) and (coord[0] < nx) and \
                                (coord[1] >= 0) and (coord[1] < ny) and \
                                (coord[2] >= 0) and (coord[2] < nz):
                                p_id = tomo_vti.ComputePointId(coord.astype(np.int))
                                if over:
                                    hold_field.InsertTuple(p_id, (lbl,))
                                else:
                                    hold_field.InsertTuple(p_id, (1,))
                disperse_io.save_vti(tomo_vti, out_stem + '.vti', out_dir)


############################################################################
# Abstract class for defining neighbourhoods
#
class Nhood(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, center, rad, voi, conv_iter=None, max_iter=None, dst_field=None):
        """
        Constructor
        :param center: center point (3-tuple)
        :param rad: radius
        :param voi: a vtkPolyData with the VOI
        :param conv_iter: number of iterations for convergence in Monte Carlo method
        :param max_iter: maximum number of iterations in Monte Carlo method
        :param dst_field: distance field used for Direct Sum volume computation method
        """

        # Input parsing
        if (not hasattr(center, '__len__')) or (len(center) != 3):
            error_msg = 'Center must be 3-tuple.'
            raise pexceptions.PySegInputError(expr='__init__ (Nhood)', msg=error_msg)
        if rad <= 0:
            error_msg = 'Input radius must be greater than zero, current ' + str(rad)
            raise pexceptions.PySegInputError(expr='__init__ (Nhood)', msg=error_msg)
        if (voi is not None) and not(isinstance(voi, vtk.vtkPolyData) or isinstance(voi, np.ndarray)):
            error_msg = 'Input VOI must be an vtkPolyData object or a numpy array.'
            raise pexceptions.PySegInputError(expr='__init__ (Nhood)', msg=error_msg)
        self.__center = np.asarray(center, dtype=np.float)
        if rad < 0:
            self.__rad = 0.
        else:
            self.__rad = float(rad)
        self.__voi = voi
        self.__dst_field = dst_field
        self.__conv_iter = None
        self.__max_iter = None
        if (conv_iter is not None) and (max_iter is not None):
            conv_iter_i, max_iter_i = int(conv_iter), int(max_iter)
            if conv_iter_i > max_iter_i:
                conv_iter_i = max_iter_i
            self.__conv_iter = conv_iter_i
            self.__max_iter = max_iter_i

    # EXTERNAL FUNCTIONALITY

    def compute_density(self, coords):
        vol = self.compute_volume()
        if vol > 0:
            return float(self.get_num_embedded(coords)) / vol
        else:
            return 0.

    @abc.abstractmethod
    def get_num_embedded(self, coords):
        raise NotImplementedError()

    # Return radius range
    @abc.abstractmethod
    def get_rad_range(self):
        raise NotImplementedError()

    # Compute and returns nominal volume, without VOI intersection
    @abc.abstractmethod
    def compute_volume_nominal(self):
        raise NotImplementedError()

    # Generates a coordiante from uniformly random distribution within Nhood shape without VOI intersection
    @abc.abstractmethod
    def gen_uni_sample(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def compute_volume(self, coord):
        raise NotImplementedError

    @abc.abstractmethod
    def analyse(self, coord):
        raise NotImplementedError

    # INTERNAL FUNCTIONALITY


############################################################################
# Class for Sphere Nhood
#
class SphereNhood(Nhood):

    def __init__(self, center, rad, voi=None, selector_voi=None, conv_iter=100, max_iter=10000, dst_field=None):
        """
        Constructor
        :param center: center point (3-tuple)
        :param rad: radius
        :param voi: a vtkPolyData with the VOI
        :param selector_voi: a vtkSelectEnclosedPoints initialized for the VOI surface, otherwise it is generated
        :param conv_iter: number of iteration until convergence for volume estimation (default 100)
        :param max_iter: maximum number of iterations for volume estimation (default 10000)
        :param dst_field: distance field required for Direct Sum VOI volume estimation
        """

        # Call base constructor
        super(SphereNhood, self).__init__(center, rad, voi, conv_iter, max_iter, dst_field)

        # Setting variables for compute volume in VOI
        self.__selector_voi = None
        if self._Nhood__voi is not None:
            if isinstance(self._Nhood__voi, vtk.vtkPolyData):
                if selector_voi is None:
                    self.__selector_voi = vtk.vtkSelectEnclosedPoints()
                    self.__selector_voi.SetTolerance(VTK_RAY_TOLERANCE)
                    selector_voi.Initialize(voi)
                else:
                    self.__selector_voi = selector_voi

    def get_rad_range(self):
        return [0, self._Nhood__rad]

    # Compute and returns nominal volume, without VOI intersection
    def compute_volume_nominal(self):
        return SPH_CTE * np.pi * self._Nhood__rad * self._Nhood__rad * self._Nhood__rad

    # Generates a coordiante from uniformly random distribution within Nhood shape without VOI intersection
    # Box-Muller algorithm adaptation for a S^3 volume with generic radius
    def gen_uni_sample(self):
        u = np.random.rand(1)[0]
        u = self._Nhood__rad * np.cbrt(u)
        X = np.random.randn(1, 3)[0]
        norm = u / np.linalg.norm(X)
        X *= norm
        return X + self._Nhood__center

    def get_num_embedded(self, coords, voi=None):
        """
        Determine the number of embedded coordinates
        :param coords: array [n, 3] with the n coordinates to test
        :param voi: to pass a pre-computed VOI, otherwise (default None) it is computed internally
        :return: the number of embedded coordinates
        """
        count = 0
        if self._Nhood__dst_field is None:
            for coord in coords:
                hold = self._Nhood__center - coord
                if math.sqrt((hold*hold).sum()) <= self._Nhood__rad:
                    count += 1
        else:
            if voi is None:
                voi = (self._Nhood__dst_field >= 0) & (self._Nhood__dst_field <= self._Nhood__rad)
            for coord in coords:
                x, y, z = np.round(coord).astype(np.int)
                if (x >= 0) and (y >= 0) and (z >= 0) and (x < self._Nhood__voi.shape[0]) \
                        and (y < self._Nhood__voi.shape[1]) and (z < self._Nhood__voi.shape[2]):
                    if voi[x, y, z]:
                        count += 1
        return count


    def compute_volume(self):
        if self._Nhood__voi is None:
            return self.compute_volume_nominal()
        elif isinstance(self._Nhood__voi, vtk.vtkPolyData):
            return self.__compute_volume_mcs()
        elif self._Nhood__dst_field is not None:
            return self.__compute_volume_dsa()
        else:
            return self.__compute_volume_mca()

    def analyse(self, coords):
        """
        It computes the Nhood volume and the number of embedded coordiantes at once, so it is more efficient than
        call to get_num_embedded() and compute_volume() seperatedly for DSA and FMM
        :param coords: array [n, 3] with the n coordinates to test
        :return: 2-tuple with the number of embedded coords and volumes
        """

        # Get the number of embedded coordiantes in the pre-computed VOI
        count = 0
        if self._Nhood__dst_field is None:
            # for coord in coords:
            #     hold = self._Nhood__center - coord
            #     if math.sqrt((hold * hold).sum()) <= self._Nhood__rad:
            #         if self.__selector_voi.IsInsideSurface(coord) == 1:
            #            count += 1
            if len(coords) > 0:
                hold = self._Nhood__center - coords
                count = (np.sqrt((hold * hold).sum(axis=1)) <= self._Nhood__rad).sum()
        else:
            # Compute the VOI
            voi = (self._Nhood__dst_field >= 0) & (self._Nhood__dst_field <= self._Nhood__rad)
            for coord in coords:
                x, y, z = coord
                if (x >= 0) and (y >= 0) and (z >= 0) and (x < voi.shape[0]) and (y < voi.shape[1]) and \
                        (z < voi.shape[2]):
                    if voi[x, y, z]:
                        count += 1

        # Compute VOI volume
        if self._Nhood__voi is None:
            vol = self.compute_volume_nominal()
        elif isinstance(self._Nhood__voi, vtk.vtkPolyData):
            vol = self.__compute_volume_mcs()
        elif self._Nhood__dst_field is not None:
            vol = voi.sum()
        else:
            vol = self.__compute_volume_mca()

        return count, vol

    # INTERNAL FUNCTIONALITY

    # Compute volume by the Monte Carlo and Ray Tracing (VOI as a closed surface) method
    def __compute_volume_mcs(self):
        n_iter, n_hits = 0, 0
        while (n_hits < self._Nhood__conv_iter) and (n_iter < self._Nhood__max_iter):
            n_iter += 1
            # Generating random point
            coord = self.gen_uni_sample()
            # Checking inside
            if self.__selector_voi.IsInsideSurface(coord) > 0:
                n_hits += 1
        voi_factor = float(n_hits) / float(n_iter)
        return self.compute_volume_nominal() * voi_factor

    # Compute volume by the Monte Carlo (VOI as a boolean 3D array) method
    def __compute_volume_mca(self):
        n_iter, n_hits = 0, 0
        while (n_hits < self._Nhood__conv_iter) and (n_iter < self._Nhood__max_iter):
            n_iter += 1
            # Generating random point
            coord = self.gen_uni_sample()
            x, y, z = int(round(coord[0])), int(round(coord[1])), int(round(coord[2]))
            # y, x, z = int(round(coord[0])), int(round(coord[1])), int(round(coord[2]))
            # Checking inside
            if (x >= 0) and (y >= 0) and (z >= 0) and (x < self._Nhood__voi.shape[0]) \
                and (y < self._Nhood__voi.shape[1]) and (z < self._Nhood__voi.shape[2]):
                if self._Nhood__voi[x, y, z]:
                    n_hits += 1
        voi_factor = float(n_hits) / float(n_iter)
        return self.compute_volume_nominal() * voi_factor

    # Compute volume by the Direct Sum (VOI as a boolean 3D array)
    def __compute_volume_dsa(self):
        return ((self._Nhood__dst_field >= 0) &
                (self._Nhood__dst_field <= self._Nhood__rad)).sum()


############################################################################
# Class for Shell Nhood
#
class ShellNhood(Nhood):

    def __init__(self, center, rad, voi=None, selector_voi=None, thick=MIN_THICK,
                 conv_iter=100, max_iter=10000, dst_field=None):
        """
        Constructor
        :param center: center point (3-tuple)
        :param rad: radius
        :param voi: a vtkPolyData with the VOI
        :param selector_voi: a vtkSelectEnclosedPoints initialized for the VOI surface, otherwise it is generated
        :param thick: shell thickness
        :param conv_iter: number of iteration until convergence for volume estimation (default 100)
        :param max_iter: maximum number of iterations for volume estimation (default 10000)
        :param dst_field_sq: distance field required for Direct Sum VOI volume estimation
        """

        # Call base constructor
        super(ShellNhood, self).__init__(center, rad, voi, conv_iter, max_iter, dst_field)

        # Input parsing
        if thick <= 0:
            error_msg = 'Input thickness must be greater than zero, current ' + str(thick)
            raise pexceptions.PySegInputError(expr='__init__ (ShellNhood)', msg=error_msg)

        # Setting radii
        self.__thick = float(thick)
        thick_2 = .5 * self.__thick
        self.__rad_1 = self._Nhood__rad + thick_2
        self.__rad_2 = self._Nhood__rad - thick_2
        if self.__rad_2 < 0:
            self.__rad_2 = 0

        # Setting variables for compute volume in VOI
        self.__selector_voi = None
        if self._Nhood__voi is not None:
            if isinstance(self._Nhood__voi, vtk.vtkPolyData):
                if selector_voi is None:
                    self.__selector_voi = vtk.vtkSelectEnclosedPoints()
                    self.__selector_voi.SetTolerance(VTK_RAY_TOLERANCE)
                    selector_voi.Initialize(voi)
                else:
                    self.__selector_voi = selector_voi

    def get_rad_range(self):
        return [self.__rad_2, self.__rad_1]

    # Compute and returns nominal volume, without VOI intersection
    def compute_volume_nominal(self):
        return SPH_CTE * np.pi * ((self.__rad_1 * self.__rad_1 * self.__rad_1) -
                                    (self.__rad_2 * self.__rad_2 * self.__rad_2))

    def get_num_embedded(self, coords):
        """
        Determine the number of embedded coordinates
        :param coords: array [n, 3] with the n coordinates to test
        :return: the number of embedded coordinates
        """
        count = 0
        if self._Nhood__dst_field is None:
            for coord in coords:
                hold = self._Nhood__center - coord
                hold = math.sqrt((hold * hold).sum())
                if (hold >= self.__rad_2) and (hold <= self.__rad_1):
                    count += 1
        else:
            for coord in coords:
                x, y, z = np.round(coord).astype(np.int)
                if (x >= 0) and (y >= 0) and (z >= 0) and (x < self._Nhood__voi.shape[0]) \
                        and (y < self._Nhood__voi.shape[1]) and (z < self._Nhood__voi.shape[2]):
                    if self._Nhood__voi[x, y, z]:
                        count += 1
        return count

    # Generates a coordiante from uniformly random distribution within Nhood shape without VOI intersection
    # Box-Muller algorithm adaptation for a S^2 surface with generic radius
    def gen_uni_sample(self):
        X = np.random.randn(1, 3)[0]
        norm = self._Nhood__rad / np.linalg.norm(X)
        X *= norm
        return X + self._Nhood__center

    def compute_volume(self):
        if self._Nhood__voi is None:
            return self.compute_volume_nominal()
        elif isinstance(self._Nhood__voi, vtk.vtkPolyData):
            return self.__compute_volume_mcs()
        elif self._Nhood__dst_field is not None:
            return self.__compute_volume_dsa()
        else:
            return self.__compute_volume_mca()

    def analyse(self, coords):
        """
        It computes the Nhood volume and the number of embedded coordiantes at once, so it is more efficient than
        call to get_num_embedded() and compute_volume() seperatedly for DSA and FMM
        :param coords: array [n, 3] with the n coordinates to test
        :return: 2-tuple with the number of embedded coords and volumes
        """

        # Get the number of embedded coordiantes in the pre-computed VOI
        count = 0
        if self._Nhood__dst_field is None:
            if len(coords) > 0:
                hold = self._Nhood__center - coords
                dsts = np.sqrt((hold * hold).sum(axis=1))
                count = ((dsts >= self.__rad_2) & (dsts <= self.__rad_1)).sum()
        else:
            # Compute the VOI
            voi = (self._Nhood__dst_field >= self.__rad_2) & (self._Nhood__dst_field <= self.__rad_1)
            for coord in coords:
                x, y, z = coord
                if (x >= 0) and (y >= 0) and (z >= 0) and (x < voi.shape[0]) and (y < voi.shape[1]) and \
                        (z < voi.shape[2]):
                    if voi[x, y, z]:
                        count += 1

        # Compute VOI volume
        if self._Nhood__voi is None:
            vol = self.compute_volume_nominal()
        elif isinstance(self._Nhood__voi, vtk.vtkPolyData):
            vol = self.__compute_volume_mcs()
        elif self._Nhood__dst_field is not None:
            vol = voi.sum()
        else:
            vol = self.__compute_volume_mca()

        return count, vol

    # INTERNAL FUNCTIONALITY

    # Compute volume by the Monte Carlo and Ray Tracing (VOI as a closed surface) method
    def __compute_volume_mcs(self):
        n_iter, n_hits = 0, 0
        while (n_hits < self._Nhood__conv_iter) and (n_iter < self._Nhood__max_iter):
            n_iter += 1
            # Generating random point
            coord = self.gen_uni_sample()
            # Checking inside
            if self.__selector_voi.IsInsideSurface(coord) > 0:
                n_hits += 1
        voi_factor = float(n_hits) / float(n_iter)
        return self.compute_volume_nominal() * voi_factor

    # Compute volume by the Monte Carlo (VOI as a boolean 3D array) method
    def __compute_volume_mca(self):
        n_iter, n_hits = 0, 0
        while (n_hits < self._Nhood__conv_iter) and (n_iter < self._Nhood__max_iter):
            n_iter += 1
            # Generating random point
            coord = self.gen_uni_sample()
            x, y, z = int(round(coord[0])), int(round(coord[1])), int(round(coord[2]))
            # Checking inside
            if (x >= 0) and (y >= 0) and (z >= 0) and (x < self._Nhood__voi.shape[0]) \
                and (y < self._Nhood__voi.shape[1]) and (z < self._Nhood__voi.shape[2]):
                if self._Nhood__voi[x, y, z]:
                    n_hits += 1
        voi_factor = float(n_hits) / float(n_iter)
        return self.compute_volume_nominal() * voi_factor

    # Compute volume by the Direct Sum (VOI as a boolean 3D array)
    def __compute_volume_dsa(self):
        return ((self._Nhood__dst_field >= self.__rad_2) &
                (self._Nhood__dst_field <= self.__rad_1)).sum()


############################################################################
# Abstract for a List of Nhoods
#
class ListNhood(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, center, radius_rg, voi=None, conv_iter=None, max_iter=None, fmm=False):
        """
        Constructor
        :param center: 3-tuples with the coordinates of the center
        :param radius_rg: array with the range of radius
        :param voi: input VOI
        :param conv_iter: number of iterations for convergence in Monte Carlo methods
        :param max_iter: maximum number of iterations in Monte Carlo methods
        :param fmm: if True Fast Marching Method is used for computing the distances, otherwise the Distance Transform
        """

        # import itertools

        # Normal construction
        # Input parsing
        self.__range = np.asarray(radius_rg, dtype=np.float32)
        if len(self.__range) <= 0:
            error_msg = 'Invalid input range values!'
            raise pexceptions.PySegInputError(expr='__init__ (ListNhood)', msg=error_msg)

        # Parent variables
        self.__center = center
        self.__voi = voi
        self.__conv_iter = conv_iter
        self.__max_iter = max_iter

        # Computing distance field (if required)
        self.__dst_field = None
        if isinstance(voi, np.ndarray) and ((conv_iter is None) or (max_iter is None)):
            if fmm:
                dst_field = np.ones(shape=self.__voi.shape, dtype=np.float32)
                x, y, z = self.__center
                field_found = False
                if self.__voi[x, y, z]:
                    dst_field[x, y, z] = -1
                    self.__dst_field = skfmm.travel_time(dst_field, self.__voi.astype(np.float32), dx=1)
                    field_found = True
                # else:
                #     # Look in the 3x3x3 neighborhood to ensure we are inside of the VOI
                #     dst_field[x, y, z] = 1
                #     for off_x, off_y, off_z in itertools.product([-1, 0, 1], repeat=3):
                #         dst_field[x + off_x, y + off_y, z + off_z] = -1
                #         try:
                #             self.__dst_field = skfmm.travel_time(dst_field, self.__voi.astype(np.float32), dx=1)
                #             field_found = True
                #             break
                #         except ValueError:
                #             dst_field[x + off_x, y + off_y, z + off_z] = 1
                if field_found:
                    self.__dst_field = self.__dst_field.data
                    self.__dst_field += .5
                    self.__dst_field[x, y, z] = 0
                    # disperse_io.save_numpy(self.__dst_field, '/fs/home/martinez/workspace/python/pyorg/surf/test/out/uni_2nd/dst_field_1.mrc')
                    self.__dst_field[np.invert(self.__voi)] = -1
                    # disperse_io.save_numpy(self.__dst_field, '/fs/home/martinez/workspace/python/pyorg/surf/test/out/uni_2nd/dst_field_2.mrc')
                    # [Y, X, Z] = np.meshgrid(np.arange(self.__voi.shape[1]), np.arange(self.__voi.shape[0]),
                    #                         np.arange(self.__voi.shape[2]), indexing='xy')
                    # hold_x, hold_y, hold_z = X - center[0], Y - center[1], Z - center[2]
                    # dst_field = np.sqrt(hold_x * hold_x + hold_y * hold_y + hold_z * hold_z)
                    # dst_field[np.invert(self.__voi)] = -1
                    # disperse_io.save_numpy(dst_field, '/fs/home/martinez/workspace/python/pyorg/surf/test/out/uni_2nd/dst_field_dt.mrc')
                else:
                    raise ValueError
            else:
                [Y, X, Z] = np.meshgrid(np.arange(self.__voi.shape[1]), np.arange(self.__voi.shape[0]),
                                        np.arange(self.__voi.shape[2]), indexing='xy')
                hold_x, hold_y, hold_z = X-center[0], Y-center[1], Z-center[2]
                self.__dst_field = np.sqrt(hold_x*hold_x + hold_y*hold_y + hold_z*hold_z)
                self.__dst_field[np.invert(self.__voi)] = -1
                # disperse_io.save_numpy(self.__dst_field, '/fs/home/martinez/workspace/python/pyorg/surf/test/out/uni_2nd/dst_field_dt.mrc')

        # Nhoods must be build in the child classes
        self.__nhoods = list()

    # EXTERNAL FUNCTIONALITY

    # Returns an array with the radii ranges
    def get_rad_ranges(self):
        rgs = np.zeros(shape=[len(self.__nhoods), 2], dtype=np.float32)
        for i, nhood in enumerate(self.__nhoods):
            rgs[i, :] = nhood.get_rad_range()
        return rgs

    # Get the volumes of each nhood_vol in a list
    def get_volumes(self):
        vols = list()
        for nhood in self.__nhoods:
            vols.append(nhood.compute_volume())
        return vols

    def get_nums_embedded(self, coords):
        nums, is_empty = list(), False
        for nhood in self.__nhoods:
            nums.append(nhood.get_num_embedded(coords))
        return nums

    # Get the densities of each nhood_vol in a list
    # one_less: if True (default False) then one particle is discarded (sphere univariate)
    # area: if True (default False) then area of the 2-manifold homolog is estimated
    def get_densities(self, coords, one_less=True, area=False):
        dens = list()
        for nhood in self.__nhoods:
            dens.append(nhood.compute_density(coords, one_less=one_less, area=area))
        return dens

    def analyse(self, coords):
        """
        It computes the ListNhood volumes and the number of embedded coordinates at once, so it is more efficient than
        call to get_nums_embedded() and compute_volumes() separatedly for DSA and FMM
        :param coords: array [n, 3] with the n coordinates to test
        :return: 2-tuple with two arrays with the number of embedded coords and volumes
        """
        n_hoods = len(self.__nhoods)
        nums, vols = np.zeros(shape=n_hoods, dtype=float), np.zeros(shape=n_hoods, dtype=float)
        for i in xrange(n_hoods):
            hold = self.__nhoods[i].analyse(coords)
            nums[i], vols[i] = hold[0], hold[1]
        return nums, vols

    # Returns a list with every Nhood as a vtkPolyData
    def get_vtps(self):
        vtps = list()
        for nhood in self.__nhoods:
            vtps.append(nhood.get_vtp(clip_voi=False))
        return vtps

    # Counts points in nhoods
    # points: points coordinates
    # no_voi: if True (default False) then no VOI is not considered
    def count_points(self, points, no_voi=False):
        counts = list()
        for nhood in self.__nhoods:
            counts.append(nhood.get_num_embedded(points))
        return counts


############################################################################
# Class for a list of shpere nhoods
#
class ListSphereNhood(ListNhood):

    def __init__(self, center, radius_rg, voi=None, selector_voi=None, conv_iter=100, max_iter=10000, fmm=False):
        """
        Constructor
        :param center: 3-tuples with the coordinates of the center
        :param radius_rg: array with the range of radius
        :param voi: input VOI
        :param conv_iter: number of iterations for convergence in Monte Carlo methods
        :param max_iter: maximum number of iterations in Monte Carlo methods
        :param fmm: if True Fast Marching Method is used for computing the distances, otherwise the Distance Transform
        """

        # Call to base constructor
        super(ListSphereNhood, self).__init__(center, radius_rg, voi, conv_iter, max_iter, fmm)

        # Input parsing
        if (voi is not None) and (isinstance(voi, vtk.vtkPolyData)) and (selector_voi is None):
            selector_voi = vtk.vtkSelectEnclosedPoints()
            selector_voi.SetTolerance(VTK_RAY_TOLERANCE)
            selector_voi.Initialize(voi)

        # Generate the spheres
        for rad in self._ListNhood__range:
            self._ListNhood__nhoods.append(SphereNhood(center=self._ListNhood__center, rad=rad,
                                                       voi=self._ListNhood__voi, selector_voi=selector_voi,
                                                       conv_iter=self._ListNhood__conv_iter,
                                                       max_iter=self._ListNhood__max_iter,
                                                       dst_field=self._ListNhood__dst_field))


############################################################################
# Class for a list of shell nhoods
#
class ListShellNhood(ListNhood):

    def __init__(self, center, radius_rg, voi=None, selector_voi=None, thick=MIN_THICK,
                 conv_iter=100, max_iter=10000, fmm=False):
        """
        Constructor
        :param center: 3-tuples with the coordinates of the center
        :param radius_rg: array with the range of radius
        :param voi: input VOI
        :param selector_voi: a vtkSelectEnclosedPoints initialized for the VOI surface, otherwise it is generated
        :param thick: shell thickness
        :param conv_iter: number of iterations for convergence in Monte Carlo methods
        :param max_iter: maximum number of iterations in Monte Carlo methods
        :param fmm: if True Fast Marching Method is used for computing the distances, otherwise the Distance Transform
        """

        # Call to base constructor
        super(ListShellNhood, self).__init__(center, radius_rg, voi, conv_iter, max_iter, fmm)

        # Input parsing
        thick_f = float(thick)
        if thick_f < MIN_THICK:
            thick_f = MIN_THICK
        if (voi is not None) and (isinstance(voi, vtk.vtkPolyData)) and (selector_voi is None):
            selector_voi = vtk.vtkSelectEnclosedPoints()
            selector_voi.SetTolerance(VTK_RAY_TOLERANCE)
            selector_voi.Initialize(voi)

        # Generate the spheres
        for rad in self._ListNhood__range:
            self._ListNhood__nhoods.append(ShellNhood(center=self._ListNhood__center, rad=rad, thick=thick_f,
                                                      voi=self._ListNhood__voi, selector_voi=selector_voi,
                                                      conv_iter=self._ListNhood__conv_iter,
                                                      max_iter=self._ListNhood__max_iter,
                                                      dst_field=self._ListNhood__dst_field))
