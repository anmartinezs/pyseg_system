"""
Set of classes for analyzing Synaptic Structural Columns

# Author: Antonio Martinez-Sanchez (Max Planck Institute for Biochemistry)
# Date: 1.09.18
"""

__author__ = 'Antonio Martinez-Sanchez'

import vtk
import copy
import math
import numpy as np
import scipy as sp
from pyorg.surf import TomoParticles, Particle
from pyorg.pexceptions import PySegInputError
from pyorg.sub import Star
from pyorg import disperse_io
from sklearn.cluster import AffinityPropagation

##### Global variables

COL_ID = 'col_id'
COL_LAYER = 'col_layer'

###########################################################################################
# Global functionality
###########################################################################################

def gen_layer_model(lyr, part_vtp, model_temp, mode_emb='center'):
    """
    Generates a random instance of a layer
    :param lyr: TomoParticles object with the layer used a reference
    :param part_vtp: vtkPolyData object with the particle shape
    :param model_temp: model template class (child of Model class) for doing the simulations
    :param mode_emb: particle embedding mode (default 'center')
    :return: a simulated instance of TomoParticles
    """

    # The trivial case
    if lyr.get_num_particles() <= 0:
        return None

    # Instance simulation
    model = model_temp(lyr.get_voi(), part_vtp)
    # disperse_io.save_vtp(lyr.get_particles()[0].get_vtp(), '/fs/pool/pool-lucic2/antonio/workspace/psd_an/ex/syn/sub/relion/fils/pst/ampar_vs_nmdar/org/col_pre_v2/col_all_az/col_scol_sim_5_maxd_15_nn1_2_nn2_2_nn3_2_test/tomos/hold_voi.vtp')
    tomo_sim = model.gen_instance(lyr.get_num_particles(), lyr.get_tomo_fname(), mode=mode_emb)

    return tomo_sim

###########################################################################################
# Class to find columns in within a tomogram
###########################################################################################

class Cluster(object):

    def __init__(self, ids, coords):
        if hasattr(coords, '__len__'):
            self.__coords = coords
        else:
            self.__coords = [coords, ]
        if hasattr(ids, '__len__'):
            self.__ids = ids
        else:
            self.__ids = [ids, ]
        assert len(self.__ids) == len(self.__coords)

    def get_coords(self):
        """
        :return: return the iterable with the coordinates that form the cluster
        """
        return self.__coords

    def get_num_coords(self):
        """
        :return: return the number of coordinates within the cluster
        """
        return len(self.__coords)

    def get_ids(self):
        """
        :return: return the particle IDs for the cluster
        """
        return self.__ids

    def add_points(self, ids, coords):
        """
        Add points to the cluster
        :param ids: list point ids
        :param coords: list with point coordinates
        :return:
        """
        if not hasattr(ids, '__len__'):
            ids = [ids, ]
        if not hasattr(coords, '__len__'):
            coords = [coords, ]
        assert len(ids) == len(coords)
        for i, idx in enumerate(ids):
            if not(idx in self.__ids):
                self.__ids.append(idx)
                self.__coords.append(coords[i, :])

    def find_num_aligned(self, clst, dst):
        """
        Finds the number of aligned particles between self and input clusters
        :param clst: input cluster
        :param dst: distance criterium for the alignment
        :return: the number of aligned particles
        """
        count_aln = 0
        coords_2 = clst.get_coords()
        for coord_1 in self.__coords:
            for coord_2 in coords_2:
                hold = np.asarray(coord_1, dtype=np.float) - np.asarray(coord_2, dtype=np.float)
                hold_dst = math.sqrt((hold * hold).sum())
                if hold_dst <= dst:
                    count_aln += 1
        return count_aln

class ColumnsFinder(object):

    def __init__(self, tl1, tl2, tl3, clst_rad, col_rad, clst_method='HC'):
        """
        :param tl1: TomoParticles object for layer 1
        :param tl2: TomoParticles object for layer 2
        :param tl3: TomoParticles object for layer 3
        """
        if (tl1.__class__.__name__ != 'TomoParticles') or (tl2.__class__.__name__ != 'TomoParticles') or \
                (tl3.__class__.__name__ != 'TomoParticles'):
            error_msg = 'All input layer must be a TomoParticles object.'
            raise PySegInputError(expr='__init__ (ColumnsFinder)', msg=error_msg)
        self.__tl1, self.__tl2, self.__tl3 = tl1, tl2, tl3
        self.__cols = dict()
        self.__scols = dict()
        self.__scols_props = dict()
        self.__cols_props = dict()
        self.build_clusters(0, method=clst_method, scols=None)
        self.build_sub_columns_v3(col_rad)
        self.build_clusters(clst_rad, method=clst_method, scols=col_rad)

    # Get methods

    def get_layers(self, lyr=0):
        """
        Get the TomoParticles object for each layer
        :param lyr: which layer to is return: 1, 2 and 3-> layer 1, 2 and 3, otherwise (default) the three layers
        :return: the demanded layer TomoParticles object(s)
        """
        if lyr == 1:
            return self.__tl1
        elif lyr == 2:
            return self.__tl2
        elif lyr == 3:
            return self.__tl3
        else:
            return self.__tl1, self.__tl2, self.__tl3

    def get_num_columns(self):
        """
        Get the number of columns found
        :return: the number of columns found
        """
        count = 0
        for col in self.__cols.itervalues():
            if col is not None:
                lyr_1, lyr_2, lyr_3 = col[0], col[1], col[2]
                if (len(lyr_1) > 0) and (len(lyr_2) > 0) and (len(lyr_3) > 0):
                    count += 1
        return count

    def get_num_sub_columns(self):
        """
        Get the number of columns found
        :return: the number of columns found
        """
        count = 0
        for scol in self.__scols.itervalues():
            if scol is not None:
                lyr_1, lyr_2, lyr_3 = scol[0], scol[1], scol[2]
                if (len(lyr_1) > 0) and (len(lyr_2) > 0) and (len(lyr_3) > 0):
                    count += 1
        return count

    def get_den_columns(self):
        """
        Get the density of columns within the segmented volumen for layer 1
        :return: the density of columns by segmented volume
        """
        return self.get_num_columns() / self.__tl1.compute_voi_volume()

    def get_num_particles(self, lyr=[1, 2, 3]):
        """
        Get the number of particles
        :param lyr: list with the layers to consider
        :return: the number of particles found
        """

        # Input parsing
        if not hasattr(lyr, '__len__'):
            lyr_p = [lyr,]
        else:
            lyr_p = lyr

        # Creating the lut to avoid double-counting
        lut_p = dict()
        for lyr_id in lyr_p:
            lut = list()
            for scol in self.__scols.itervalues():
                for clst in scol[lyr_id-1]:
                    lut += clst.get_ids()
            lut_p[lyr_id] = list(set(lut))

        # Counting loop
        count = 0
        for lyr_id in lyr_p:
            count += len(lut_p[lyr_id])

        return count

    def get_area_tomo(self):
        """
        Computes the area of the reference layer by approximating it to a plane
        :return: The area value
        """
        voi = self.__tl1.get_voi()
        if isinstance(voi, vtk.vtkPolyData):
            masser = vtk.vtkMassProperties()
            masser.SetInputData(voi)
            return .5 * masser.GetAreaSurface()
        elif isinstance(voi, np.ndarray):
            # Flattern the VOI
            voi_ids = np.where(voi)
            min_x, max_x = voi_ids[0].min(), voi_ids[0].max()
            min_y, max_y = voi_ids[1].min(), voi_ids[1].max()
            min_z, max_z = voi_ids[2].min(), voi_ids[2].max()
            lx, ly, lz = max_x - min_x, max_y - min_y, max_z - min_z
            sorted = np.sort(np.asarray((lx, ly, lz), dtype=np.int))[::-1]
            return sorted[0] * sorted[1]
        else:
            return .0

    def get_area_columns(self, mode='cyl', rad=15, layers=[1, 2, 3]):
        """
        Return a areas of all columns found, its approximated using convexhull in 2D
        :param mode: mode for computing the areas, 'cyl'-> the union of 2D projected spheres with radius 'rad' centered
        a each particle in the reference layer, 'hull' 2D convex hull of all column particles
        :param rad: sphere raidus for 'cyl' mode
        :param layers: list with the layers to consider (default [1, 2, 3])
        :return: a list with area for each column found
        """

        # Input parsing
        if not ((mode == 'cyl') or (mode == 'hull')):
            error_msg = 'Non valid input mode for area computation: ' + str(mode)
            raise PySegInputError(expr='get_area_columns (ColumnsFinder)', msg=error_msg)
        areas = list()

        voi = self.__tl1.get_voi()
        if isinstance(voi, vtk.vtkPolyData):
            min_x, max_x, min_y, max_y, min_z, max_z = voi.GetBounds()
        elif isinstance(voi, np.ndarray):
            # Flattern the VOI
            voi_ids = np.where(voi)
            min_x, max_x = voi_ids[0].min(), voi_ids[0].max()
            min_y, max_y = voi_ids[1].min(), voi_ids[1].max()
            min_z, max_z = voi_ids[2].min(), voi_ids[2].max()
            lx, ly, lz = max_x - min_x, max_y - min_y, max_z - min_z
            ids_sorted = np.argsort(np.asarray((lx, ly, lz), dtype=np.int))
        else:
            error_msg = 'Non-valid VOI type: ' + str(voi.__class__)
            raise PySegInputError(expr='get_area_columns (ColumnsFinder)', msg=error_msg)

        # Load the coordinates
        for col_id, col in zip(self.__cols.iterkeys(), self.__cols.itervalues()):
            if col is not None:
                col_coords = list()
                for lyr_id in layers:
                    for clst in col[lyr_id-1]:
                        for coord in clst.get_coords():
                            if ids_sorted[0] == 0:
                                col_coords.append((coord[1], coord[2]))
                            elif ids_sorted[0] == 1:
                                col_coords.append((coord[0], coord[2]))
                            else:
                                col_coords.append((coord[0], coord[1]))

                # Compute the areas by computing the convex hull
                if mode == 'hull':
                    coords = np.asarray(col_coords, dtype=np.float32)
                    areas.append(sp.spatial.ConvexHull(coords).area)

                # Compute areas by adding cylinders to the coordinate particles
                elif mode == 'cyl':
                    coords = np.asarray(col_coords).round().astype(np.int)
                    if ids_sorted[0] == 0:
                        off = np.asarray((min_y, min_z), dtype=np.int)
                        coords -= off
                        flat_voi = np.ones(shape=(int(math.ceil(max_y-min_y))+1, int(math.ceil(max_z-min_z))+1), dtype=np.bool)
                    elif ids_sorted[0] == 1:
                        off = np.asarray((min_x, min_z), dtype=np.int)
                        coords -= off
                        flat_voi = np.ones(shape=(int(math.floor(max_x - min_x))+1, int(math.ceil(max_z - min_z))+1),
                                               dtype=np.bool)
                    else:
                        off = np.asarray((min_x, min_y), dtype=np.int)
                        coords -= off
                        flat_voi = np.ones(shape=(int(math.floor(max_x - min_x))+1, int(math.ceil(max_y - min_y))+1),
                                               dtype=np.bool)
                    for coord in coords:
                        try:
                            flat_voi[coord[0], coord[1]] = False
                        except IndexError:
                            print 'WARNING: get_area_columns() point out of flattered VOI!'
                    dst_field = sp.ndimage.morphology.distance_transform_edt(flat_voi, return_distances=True,
                                                                                 return_indices=False)
                    areas.append((dst_field <= rad).sum())

        return areas

    def get_occupancy(self, mode='cyl', rad=15, layers=[0, 1, 2], area=False, scols=False):
        """
        Return the proportion of suface area occupied by columns
        :param mode, rad: same as get_area_columns()
        :param layers: list with the layers to include
        :param area: if True (default False) then area, instead of occupancy, is computed considering overlappings
        :param scols: if True (default False) then area correspond with subcolumns instead of columns
        :return: get the occupancy value found
        """

        # Input parsing
        if not ((mode == 'cyl') or (mode == 'hull')):
            error_msg = 'Non valid input mode for area computation: ' + str(mode)
            raise PySegInputError(expr='get_occupancy (ColumnsFinder)', msg=error_msg)
        areas = list()

        voi = self.__tl1.get_voi()
        if isinstance(voi, vtk.vtkPolyData):
            return np.asarray(self.get_area_columns(mode=mode, rad=rad), dtype=np.float).sum() \
                   / float(self.get_area_tomo())
        elif isinstance(voi, np.ndarray):
            # Flattern the VOI
            voi_ids = np.where(voi)
            min_x, max_x = voi_ids[0].min(), voi_ids[0].max()
            min_y, max_y = voi_ids[1].min(), voi_ids[1].max()
            min_z, max_z = voi_ids[2].min(), voi_ids[2].max()
            lx, ly, lz = max_x - min_x, max_y - min_y, max_z - min_z
            ids_sorted = np.argsort(np.asarray((lx, ly, lz), dtype=np.int))
        else:
            error_msg = 'Non-valid VOI type: ' + str(voi.__class__)
            raise PySegInputError(expr='get_area_columns (ColumnsFinder)', msg=error_msg)

        # Compute the areas
        if mode == 'cyl':
            col_coords = list()

        if scols:
            hold_cols = self.__scols
        else:
            hold_cols = self.__cols

        # Load the coordinates
        for col_id, col in zip(hold_cols.iterkeys(), hold_cols.itervalues()):
            if col is not None:
                if mode == 'hull':
                    col_coords = list()
                for lyr_id in layers:
                    for clst in col[lyr_id]:
                        for coord in clst.get_coords():
                            if ids_sorted[0] == 0:
                                col_coords.append((coord[1], coord[2]))
                            elif ids_sorted[0] == 1:
                                col_coords.append((coord[0], coord[2]))
                            else:
                                col_coords.append((coord[0], coord[1]))

                if mode == 'hull':
                    coords = np.asarray(col_coords, dtype=np.float32)
                    # Compute the areas by computing the convex hull
                    areas.append(sp.spatial.ConvexHull(coords).area)

        if mode == 'cyl':
            if len(col_coords) > 0:
                coords = np.asarray(col_coords).round().astype(np.int)
                if ids_sorted[0] == 0:
                    off = np.asarray((min_y, min_z), dtype=np.int)
                    coords -= off
                    flat_voi = np.ones(shape=(int(math.ceil(max_y-min_y))+1, int(math.ceil(max_z-min_z))+1), dtype=np.bool)
                elif ids_sorted[0] == 1:
                    off = np.asarray((min_x, min_z), dtype=np.int)
                    coords -= off
                    flat_voi = np.ones(shape=(int(math.floor(max_x - min_x))+1, int(math.ceil(max_z - min_z))+1), dtype=np.bool)
                else:
                    off = np.asarray((min_x, min_y), dtype=np.int)
                    coords -= off
                    flat_voi = np.ones(shape=(int(math.floor(max_x - min_x))+1, int(math.ceil(max_y - min_y))+1), dtype=np.bool)
                for coord in coords:
                    try:
                        flat_voi[coord[0], coord[1]] = False
                    except IndexError:
                        print 'WARNING: get_occupancy() point out of flattered VOI!'
            else:
                return .0

        if mode == 'hull':
            if area:
                return np.asarray(areas, dtype=np.float).sum()
            else:
                return np.asarray(areas, dtype=np.float).sum() / float(self.get_area_tomo())
        elif mode == 'cyl':
            dst_field = sp.ndimage.morphology.distance_transform_edt(flat_voi, return_distances=True, return_indices=False)
            if area:
                return (dst_field < rad).sum()
            else:
                return (dst_field < rad).sum() / float(self.get_area_tomo())
        else:
            raise ValueError

    # Functionality

    def build_clusters(self, dst=0, method='HC', scols=None):
        """
        Build the clusters for each layer, if rad <= 0 then eacha particle is a cluster
        :param dst: distance parameter for clustering alogorithm (HC-> inter particle distance, MS-> meanshift)
        :param nn_x: minimum number of particles per cluster
        :param method: clustering method: 'AC' -> Agglomerative clustering (default), 'HC'-> hierarchical clustering,
        'MS'-> meanshift
        :param scols: if None (default) then subcolumns alignment is considered, otherwise it defines the sub-columns
                      cluster minimum distance among particles
        :return: None, cluster are stored in internal variables
        """

        # Each cluster is a input coordinate
        if dst <= 0:
            self.__cool_clst = (list(), list(), list())
            for i, lyr in enumerate([self.__tl1, self.__tl2, self.__tl3]):
                for j, part in enumerate(lyr.get_particles()):
                    self.__cool_clst[i].append(Cluster([j,], [part.get_center(),]))
        else:
            if method == 'HC':
                if scols is None:
                    hold_col = self.__build_clusters_HC_scols(dst)
                else:
                    # hold_col = self.__build_clusters_HC_m2(dst, scols)
                    hold_col = self.__build_clusters_HC_m1(dst, scols)
            elif method == 'AP':
                if scols:
                    raise NotImplementedError
                else:
                    raise NotImplementedError
            elif method == 'MS':
                raise NotImplementedError
            else:
                raise NotImplementedError

            # Update clusters ids
            self.__cols = hold_col
            self.__cool_clst = (list(), list(), list())
            for key, col in zip(self.__cols.iterkeys(), self.__cols.itervalues()):
                for id_lyr, lyr in enumerate(col):
                    for clst in lyr:
                        self.__cool_clst[id_lyr].append(clst)
                self.__cols_props[key] = dict()

    def build_sub_columns(self, col_rad=0):
        """
        Procedure to find the sub-columns among the three layers
        :param col_rad: sub-column radius
        :return: None, the results are store in class variable as a list of sub-columns (a tuple of three lists with
                the coordinates for each layer)
        """

        # Initialization
        lyr1, lyr2, lyr3 = self.__cool_clst[0], self.__cool_clst[1], self.__cool_clst[2]

        # Build the LUT with the number of aligned particles respect the reference layer
        lut_clsts_2 = np.zeros(shape=(len(lyr1), len(lyr2)), dtype=np.int)
        lut_clsts_3 = np.zeros(shape=(len(lyr1), len(lyr3)), dtype=np.int)
        for i, clst_i in enumerate(lyr1):
            for j, clst_j in enumerate(lyr2):
                lut_clsts_2[i, j] = clst_i.find_num_aligned(clst_j, col_rad)
            for k, clst_k in enumerate(lyr3):
                lut_clsts_3[i, k] = clst_i.find_num_aligned(clst_k, col_rad)

        # Generating the provisional columns
        hold_cols = dict()
        for i, clst_i in enumerate(lyr1):
            hold_cols[i] = (list(), list(), list())
            hold_cols[i][0].append(clst_i)
        for j, clst_j in enumerate(lyr2):
            i_id = np.argmax(lut_clsts_2[:, j])
            if lut_clsts_2[i_id, j] > 0:
                hold_cols[i_id][1].append(clst_j)
        for k, clst_k in enumerate(lyr3):
            i_id = np.argmax(lut_clsts_3[:, k])
            if lut_clsts_3[i_id, k] > 0:
                hold_cols[i_id][2].append(clst_k)

        # Getting definintive columnns by removing those which are not aligned for the three layers
        col_id, self.__scols = 0, dict()
        for hold_col in hold_cols.itervalues():
            if (len(hold_col[0]) > 0) and (len(hold_col[1]) > 0) and (len(hold_col[2]) > 0):
                self.__scols[col_id] = (list(), list(), list())
                for i in range(len(hold_col[0])):
                    self.__scols[col_id][0].append(hold_col[0][i])
                for j in range(len(hold_col[1])):
                    self.__scols[col_id][1].append(hold_col[1][j])
                for k in range(len(hold_col[2])):
                    self.__scols[col_id][2].append(hold_col[2][k])
                col_id += 1

    def build_sub_columns_v3(self, col_rad=0):
        """
        Procedure to find the sub-columns among the three layers
        :param col_rad: sub-column radius
        :return: None, the results are store in class variable as a list of sub-columns (a tuple of three lists with
                the coordinates for each layer)
        """

        # Initialization
        lyr1, lyr2, lyr3 = self.__cool_clst[0], self.__cool_clst[1], self.__cool_clst[2]

        # Build the LUT with the number of aligned particles respect the reference layer
        lut_clsts_2 = np.zeros(shape=(len(lyr1), len(lyr2)), dtype=np.int)
        lut_clsts_3 = np.zeros(shape=(len(lyr1), len(lyr3)), dtype=np.int)
        for i, clst_i in enumerate(lyr1):
            for j, clst_j in enumerate(lyr2):
                lut_clsts_2[i, j] = clst_i.find_num_aligned(clst_j, col_rad)
            for k, clst_k in enumerate(lyr3):
                lut_clsts_3[i, k] = clst_i.find_num_aligned(clst_k, col_rad)

        # Generating the provisional subcolumns
        col_id, self.__scols, self.__scols_props = 0, dict(), dict()
        for i, clst_i in enumerate(lyr1):
            hold_col = (list(), list(), list())
            hold_col[0].append(clst_i)
            for j, clst_j in enumerate(lyr2):
                if lut_clsts_2[i, j] > 0:
                    hold_col[1].append(clst_j)
            for k, clst_k in enumerate(lyr3):
                if lut_clsts_3[i, k] > 0:
                    hold_col[2].append(clst_k)
            if (len(hold_col[0]) > 0) and (len(hold_col[1]) > 0) and (len(hold_col[2]) > 0):
                self.__scols[col_id] = hold_col
                self.__scols_props[col_id] = dict()
            col_id += 1

        # # Inserting points in the surroundings of sub-columns for layer 1
        # l1_coords = self.__tl1.get_particle_coords()
        # lut_points_1 = -1 * np.ones(shape=l1_coords.shape[0], dtype=np.int)
        # for key, scol in zip(self.__scols.iterkeys(), self.__scols.itervalues()):
        #     for clst in scol[0]:
        #         for idx in clst.get_ids():
        #             lut_points_1[idx] = key
        # pts_toadd = dict()
        # for i in range(len(lut_points_1)):
        #     if lut_points_1[i] < 0:
        #         pt_coord = l1_coords[i, :]
        #         min_dst, min_key = np.finfo(np.float).max, None
        #         for key, scol in zip(self.__scols.iterkeys(), self.__scols.itervalues()):
        #             for clst in scol[0]:
        #                 for coord in clst.get_coords():
        #                     hold = pt_coord - coord
        #                     dst = np.sqrt((hold * hold).sum())
        #                     if dst < min_dst:
        #                         min_dst, min_key = dst, key
        #         if (min_key is not None) and (min_dst <= col_rad):
        #             pts_toadd[i] = min_key
        # for pt_id, scol_key in zip(pts_toadd.iterkeys(), pts_toadd.itervalues()):
        #     self.__scols[scol_key][0].append(Cluster([pt_id, ], [l1_coords[pt_id, :], ]))

    def build_columns(self, col_rad=0):
        """
        Procedure to find the columns among the three layers
        :param col_rad: column radius in nm (if <= 0 then no clustering is applied then each cordiante is a cluster)
        :return: None, the results are store in class variable as a list of columns (a tuple of three lists with
                the coordinates for each layer)
        """

        # Initialization
        lyr1, lyr2, lyr3 = self.__cool_clst[0], self.__cool_clst[1], self.__cool_clst[2]

        # Build the LUT with the number of aligned particles respect the reference layer
        lut_clsts_2 = np.zeros(shape=(len(lyr1), len(lyr2)), dtype=np.int)
        lut_clsts_3 = np.zeros(shape=(len(lyr1), len(lyr3)), dtype=np.int)
        for i, clst_i in enumerate(lyr1):
            for j, clst_j in enumerate(lyr2):
                lut_clsts_2[i, j] = clst_i.find_num_aligned(clst_j, col_rad)
            for k, clst_k in enumerate(lyr3):
                lut_clsts_3[i, k] = clst_i.find_num_aligned(clst_k, col_rad)

        # Generating the provisional columns
        hold_cols = dict()
        for i, clst_i in enumerate(lyr1):
            hold_cols[i] = (list(), list(), list())
            hold_cols[i][0].append(clst_i)
        for j, clst_j in enumerate(lyr2):
            i_id = np.argmax(lut_clsts_2[:, j])
            if lut_clsts_2[i_id, j] > 0:
                hold_cols[i_id][1].append(clst_j)
        for k, clst_k in enumerate(lyr3):
            i_id = np.argmax(lut_clsts_3[:, k])
            if lut_clsts_3[i_id, k] > 0:
                hold_cols[i_id][2].append(clst_k)

        # Getting definintive columnns by removing those which are not aligned for the three layers
        col_id, self.__cols = 0, dict()
        for hold_col in hold_cols.itervalues():
            if (len(hold_col[0]) > 0) and (len(hold_col[1]) > 0) and (len(hold_col[2]) > 0):
                self.__cols[col_id] = (list(), list(), list())
                for i in range(len(hold_col[0])):
                    self.__cols[col_id][0].append(hold_col[0][i])
                for j in range(len(hold_col[1])):
                    self.__cols[col_id][1].append(hold_col[1][j])
                for k in range(len(hold_col[2])):
                    self.__cols[col_id][2].append(hold_col[2][k])
                col_id += 1

    def filter_columns(self):
        """
        Filter those clusters which are not aligned
        :return:
        """

        # Initialization
        np_lyr1, np_lyr2, np_lyr3 = self.__tl1.get_num_particles(), self.__tl2.get_num_particles(), \
                                    self.__tl3.get_num_particles()
        del_ids_1, del_ids_2, del_ids_3 = np.ones(shape=np_lyr1, dtype=np.bool), np.ones(shape=np_lyr2, dtype=np.bool), \
                                          np.ones(shape=np_lyr3, dtype=np.bool)

        # Loop for finding the particle to filter
        for col in self.__cols.itervalues():
            if col is not None:
                lyr_1, lyr_2, lyr_3 = col[0], col[1], col[2]
                for clst in lyr_1:
                    for idx in clst.get_ids():
                        del_ids_1[idx] = False
                for clst in lyr_2:
                    for idx in clst.get_ids():
                        del_ids_2[idx] = False
                for clst in lyr_3:
                    for idx in clst.get_ids():
                        del_ids_3[idx] = False

        # Deleting particles loop
        del_lyr1_ids, del_lyr2_ids, del_lyr3_ids = np.where(del_ids_1)[0], np.where(del_ids_2)[0], np.where(del_ids_3)[0]
        if len(del_lyr1_ids):
            self.__tl1.delete_particles(del_lyr1_ids)
        if len(del_lyr2_ids) > 0:
            self.__tl2.delete_particles(del_lyr2_ids)
        if len(del_lyr3_ids) > 0:
            self.__tl3.delete_particles(del_lyr3_ids)

    def gen_columns_vtp(self):
        """
        Generates a VTK object to display columns
        :return: a vtkPolyData object for columns
        """

        # Initialization
        poly = vtk.vtkPolyData()
        p_points = vtk.vtkPoints()
        p_cells = vtk.vtkCellArray()
        p_ids = vtk.vtkIntArray()
        p_ids.SetName('COL')
        p_ids.SetNumberOfComponents(1)
        p_lyr = vtk.vtkIntArray()
        p_lyr.SetName('LAYER')
        p_lyr.SetNumberOfComponents(1)
        lyr1_coords, lyr2_coords, lyr3_coords = self.__tl1.get_particle_coords(), self.__tl2.get_particle_coords(), \
                                                self.__tl3.get_particle_coords()


        # Insterting coordinates
        count = 0
        # for col_id, col in enumerate(self.__cool_coords):
        for col_id, col in zip(self.__cols.iterkeys(), self.__cols.itervalues()):
            if col is not None:
                lyr_1, lyr_2, lyr_3 = col[0], col[1], col[2]
                for clst_1 in lyr_1:
                    for idx in clst_1.get_ids():
                        coord_1 = lyr1_coords[idx]
                        p_points.InsertNextPoint(coord_1)
                        p_cells.InsertNextCell(1)
                        p_cells.InsertCellPoint(count)
                        p_ids.InsertTuple(count, (col_id,))
                        p_lyr.InsertTuple(count, (1,))
                        count += 1
                for clst_2 in lyr_2:
                    for idx in clst_2.get_ids():
                        coord_2 = lyr2_coords[idx]
                        p_points.InsertNextPoint(coord_2)
                        p_cells.InsertNextCell(1)
                        p_cells.InsertCellPoint(count)
                        p_ids.InsertTuple(count, (col_id,))
                        p_lyr.InsertTuple(count, (2,))
                        count += 1
                for clst_3 in lyr_3:
                    for idx in clst_3.get_ids():
                        coord_3 = lyr3_coords[idx]
                        p_points.InsertNextPoint(coord_3)
                        p_cells.InsertNextCell(1)
                        p_cells.InsertCellPoint(count)
                        p_ids.InsertTuple(count, (col_id,))
                        p_lyr.InsertTuple(count, (3,))
                        count += 1

        # Generate the poly
        poly.SetPoints(p_points)
        poly.SetVerts(p_cells)
        poly.GetPointData().AddArray(p_ids)
        poly.GetPointData().AddArray(p_lyr)
        return poly

    def gen_subcolumns_vtp(self):
        """
        Generates a VTK object to display subcolumns
        :return: a vtkPolyData object for subcolumns
        """

        # Initialization
        poly = vtk.vtkPolyData()
        p_points = vtk.vtkPoints()
        p_cells = vtk.vtkCellArray()
        p_ids = vtk.vtkIntArray()
        p_ids.SetName('SCOL')
        p_ids.SetNumberOfComponents(1)
        p_lyr = vtk.vtkIntArray()
        p_lyr.SetName('LAYER')
        p_lyr.SetNumberOfComponents(1)
        lyr1_coords, lyr2_coords, lyr3_coords = self.__tl1.get_particle_coords(), self.__tl2.get_particle_coords(), \
                                                self.__tl3.get_particle_coords()


        # Insterting coordinates
        count = 0
        for scol_id, scol in zip(self.__scols.iterkeys(), self.__scols.itervalues()):
            if scol is not None:
                lyr_1, lyr_2, lyr_3 = scol[0], scol[1], scol[2]
                for sclst_1 in lyr_1:
                    for idx in sclst_1.get_ids():
                        coord_1 = lyr1_coords[idx]
                        p_points.InsertNextPoint(coord_1)
                        p_cells.InsertNextCell(1)
                        p_cells.InsertCellPoint(count)
                        p_ids.InsertTuple(count, (scol_id,))
                        p_lyr.InsertTuple(count, (1,))
                        count += 1
                for sclst_2 in lyr_2:
                    for idx in sclst_2.get_ids():
                        coord_2 = lyr2_coords[idx]
                        p_points.InsertNextPoint(coord_2)
                        p_cells.InsertNextCell(1)
                        p_cells.InsertCellPoint(count)
                        p_ids.InsertTuple(count, (scol_id,))
                        p_lyr.InsertTuple(count, (2,))
                        count += 1
                for sclst_3 in lyr_3:
                    for idx in sclst_3.get_ids():
                        coord_3 = lyr3_coords[idx]
                        p_points.InsertNextPoint(coord_3)
                        p_cells.InsertNextCell(1)
                        p_cells.InsertCellPoint(count)
                        p_ids.InsertTuple(count, (scol_id,))
                        p_lyr.InsertTuple(count, (3,))
                        count += 1

        # Generate the poly
        poly.SetPoints(p_points)
        poly.SetVerts(p_cells)
        poly.GetPointData().AddArray(p_ids)
        poly.GetPointData().AddArray(p_lyr)
        return poly

    def gen_col_tomo_particles(self, tomo_fname, part_vtp, voi):
        """
        Creates a TomoParticles object with the columns
        :param tomo_fname: tomogram file name
        :param part_vtp: input particle shape
        :param voi: input VOI
        :return: a TomoParticles object
        """
        tomo = TomoParticles(tomo_fname, -1, voi=voi)
        for col in self.__cool_coords:
            for lyr in col:
                for coord in lyr:
                    # Particle construction
                    try:
                        hold_part = Particle(part_vtp, center=(0, 0, 0))
                    except PySegInputError:
                        continue
                    # Random rigid body transformation
                    hold_part.translation(coord[0], coord[1], coord[2])
                    # Checking embedding and no overlapping
                    try:
                        tomo.insert_particle(hold_part, check_bounds=True, mode='center', check_inter=False)
                    except PySegInputError:
                        continue
        return tomo

    def add_columns_to_star(self, star, mic_name, layer=None, mode='cols'):
        '''
        Add the corresponding rows for columns to a particles STAR file (Star object)
        :param star: input Star object
        :param mic: _rlnMicrographName entry
        :param layer: and integer
        :param mode: valid; 'cols' (default) column particles, 'scols' sub-column particles, otherwise all particles
        '''

        # Parsing the input coordinates
        hold_coords = list()
        lyr1_coords, lyr2_coords, lyr3_coords = self.__tl1.get_particle_coords(), self.__tl2.get_particle_coords(), \
                                                self.__tl3.get_particle_coords()
        if mode == 'scols':
            for hscols in self.__scols.itervalues():
                scol_1, scol_2, scol_3 = hscols[0], hscols[1], hscols[2]
                if (layer != 2) and (layer != 3):
                    for clst in scol_1:
                        for idx in clst.get_ids():
                            hold_coords.append(lyr1_coords[idx])
                if (layer != 1) and (layer != 3):
                    for clst in scol_2:
                        for idx in clst.get_ids():
                            hold_coords.append(lyr2_coords[idx])
                if (layer != 1) and (layer != 2):
                    for clst in scol_3:
                        for idx in clst.get_ids():
                            hold_coords.append(lyr3_coords[idx])
        elif mode == 'cols':
            for col in self.__cols.itervalues():
                if col is not None:
                    col_1, col_2, col_3 = col[0], col[1], col[2]
                    if (layer != 2) and (layer != 3):
                        for clst in col_1:
                            for idx in clst.get_ids():
                                hold_coords.append(lyr1_coords[idx])
                    if (layer != 1) and (layer != 3):
                        for clst in col_2:
                            for idx in clst.get_ids():
                                hold_coords.append(lyr2_coords[idx])
                    if (layer != 1) and (layer != 2):
                        for clst in col_3:
                            for idx in clst.get_ids():
                                hold_coords.append(lyr3_coords[idx])
        else:
            if (layer != 2) and (layer != 3):
                hold_coords += list(lyr1_coords)
            if (layer != 1) and (layer != 3):
                hold_coords += list(lyr2_coords)
            if (layer != 1) and (layer != 2):
                hold_coords += list(lyr3_coords)

        # Parsing the input tomogram
        g_name = 0
        if star.get_nrows() <= 0:
            star.add_column('_rlnMicrographName')
            star.add_column('_rlnGroupNumber')
            star.add_column('_rlnCoordinateX')
            star.add_column('_rlnCoordinateY')
            star.add_column('_rlnCoordinateZ')
        else:
            # Look if the micrograph already exist
            mics = star.get_column_data('_rlnMicrographName')
            try:
                mic_id = mics.index(mic_name)
                g_name = star.get_element('_rlnGroupNumber', mic_id)
            except ValueError:
                g_name = max(star.get_column_data('_rlnGroupNumber')) + 1

        # Adding the rows
        for coord in hold_coords:
            hold_dict = {'_rlnMicrographName': mic_name,
                         '_rlnGroupNumber': g_name,
                         '_rlnCoordinateX': coord[1],
                         '_rlnCoordinateY': coord[0],
                         '_rlnCoordinateZ': coord[2]}
            star.add_row(**hold_dict)

    def get_coloc_columns(self, coords, dst, layers=[1, 2, 3]):
        """
        Count the number of particles aligned with columns of an external pattern
        :param coords: an iterable with the coordinates for the input pattern
        :param dst: alignment distance
        :param layers: list with the layers of the columns to consider
        :return: An integer that counts with number of input particles aligned with a column
        """

        # Loop for coordinates
        layers_ids, count = np.asarray(layers) - 1, 0
        for coord in coords:
            is_aln = False
            for col in self.__cols.itervalues():
                for lyr_id in layers_ids:
                    for clst in col[lyr_id]:
                        if clst.find_num_aligned(Cluster([0, ], [coord, ]), dst) > 0:
                            count += 1
                            is_aln = True
                        if is_aln:
                            break
                    if is_aln:
                        break
                if is_aln:
                    break

        return count

    def find_subcols_overlap(self, cfinder, dst, prop_key, layers=[1, ]):
        """
        Find the number of sub-columns overlapped with other input sub-columns
        :param cfinder: input ColumnsFinder with the sub-columns to find the overlapping
        :param dst: overlapping distance
        :param prop_key: key for the property generated
        :param layers: list with the layers to include the overlapping (default [1, ])
        :return: a new binary sub-column property is generated where True mark the overlapped columns
        """
        for key, scol in zip(self.__scols.iterkeys(), self.__scols.itervalues()):
            self.__scols_props[key][prop_key] = False
            ids, coords = list(), list()
            for lyr in layers:
                lyr_id = lyr - 1
                for hold_clst in scol[lyr_id]:
                    ids += hold_clst.get_ids()
                    coords += hold_clst.get_coords()
            clst = Cluster(ids, coords)
            for scol_in in cfinder.__scols.itervalues():
                ids_in, coords_in = list(), list()
                lyr_id = lyr - 1
                for hold_clst in scol_in[lyr_id]:
                    ids_in += hold_clst.get_ids()
                    coords_in += hold_clst.get_coords()
                clst_in = Cluster(ids_in, coords_in)
                if clst.find_num_aligned(clst_in, dst) > 0:
                    self.__scols_props[key][prop_key] = True
                    break

    def find_cols_overlap(self, cfinder, dst, prop_key, layers=[1, ]):
        """
        Find the number of columns overlapped with other input columns
        :param cfinder: input ColumnsFinder with the columns to find the overlapping
        :param dst: overlapping distance
        :param prop_key: key for the property generated
        :param layers: list with the layers to include the overlapping (default [1, ])
        :return: a new binary column property is generated where True mark the overlapped columns
        """
        for key, col in zip(self.__cols.iterkeys(), self.__cols.itervalues()):
            self.__cols_props[key][prop_key] = False
            ids, coords = list(), list()
            for lyr in layers:
                lyr_id = lyr - 1
                for hold_clst in col[lyr_id]:
                    ids += hold_clst.get_ids()
                    coords += hold_clst.get_coords()
            clst = Cluster(ids, coords)
            for col_in in cfinder.__cols.itervalues():
                ids_in, coords_in = list(), list()
                lyr_id = lyr - 1
                for hold_clst in col_in[lyr_id]:
                    ids_in += hold_clst.get_ids()
                    coords_in += hold_clst.get_coords()
                clst_in = Cluster(ids_in, coords_in)
                if clst.find_num_aligned(clst_in, dst) > 0:
                    self.__cols_props[key][prop_key] = True
                    break

    def count_scols_prop(self, prop_key, val, oper):
        """
        Count the number of sub-columns that fulfill some condition
        :param prop_key: property key
        :param val: condition value
        :param oper: condition operation
        :return: The number of sub-columns that fulfill the input condition
        """
        count = 0
        for scol in self.__scols_props.itervalues():
            if oper(scol[prop_key], val):
                count += 1
        return count

    def count_cols_prop(self, prop_key, val, oper):
        """
        Count the number of columns that fulfill some condition
        :param prop_key: property key
        :param val: condition value
        :param oper: condition operation
        :return: The number of columns that fulfill the input condition
        """
        count = 0
        for scol in self.__cols_props.itervalues():
            if oper(scol[prop_key], val):
                count += 1
        return count

    ## INTERNAL FUNCTIONALITY

    def __build_clusters_HC_scols(self, dst):
        """
        Private method for generating clusters within layers by applying Hirarchical Clustering for sub-columns
        :param dst: distance threshold
        :return:
        """

        # Initialization
        hold_cols = dict()

        # Loop for layers
        hold_clsts = dict()
        for id_lyr, lyr in enumerate([self.__tl1, self.__tl2, self.__tl3]):
            X = np.zeros(shape=(lyr.get_num_particles(), 3), dtype=np.float32)
            for i, part in enumerate(lyr.get_particles()):
                X[i, :] = part.get_center()

            # Clustering algorithm
            hold_clsts[id_lyr] = list()
            if len(X) <= 0:
                continue
            elif len(X) == 1:
                hold_clsts[id_lyr].append(Cluster([0, ], [X[0, :], ]))
                continue
            Y = sp.spatial.distance.pdist(X)
            # Z = sp.cluster.hierarchy.single(Y)
            Z = sp.cluster.hierarchy.linkage(Y, method='single', metric='euclidean')
            # lbls = sp.cluster.hierarchy.fcluster(Z, dst, criterion='distance')
            lbls = sp.cluster.hierarchy.fcluster(Z, dst, criterion='distance')
            class_ids = list(set(lbls))

            # Creating the clustering
            for idx in class_ids:
                if idx == -1:
                    continue
                c_ids, coords = list(), list()
                for i, lbl in enumerate(lbls):
                    if lbl == idx:
                        c_ids.append(i)
                        coords.append(X[i, :])
                if len(c_ids) > 0:
                    hold_clsts[id_lyr].append(Cluster(c_ids, coords))

        # Generating the sub-columns by checking clustering alignment
        count_col = 0
        for i, clst_1 in enumerate(hold_clsts[0]):
            hold_col = (list(), list(), list())
            hold_col[0].append(clst_1)
            for clst_2 in hold_clsts[1]:
                if clst_1.find_num_aligned(clst_2, dst) > 0:
                    hold_col[1].append(clst_2)
            for clst_3 in hold_clsts[2]:
                if clst_1.find_num_aligned(clst_3, dst) > 0:
                    hold_col[2].append(clst_3)
            if (len(hold_col[0])> 0) and (len(hold_col[1])> 0) and (len(hold_col[2])> 0):
                hold_cols[count_col] = hold_col
                count_col += 1

        return hold_cols

    def __build_clusters_HC_m1(self, dst_clst, dst_aln):
        """
        Private method for generating clusters within layers by applying Hierarchical Clustering
        :param dst_clst: cluster distance threshold
        :param dst_aln: distance alignment threshold
        :return:
        """

        # Initialization
        hold_col= dict()

        # Loop for particles within subcolumns
        X, lut_lyr, lut_cid = list(), list(), list()
        lut1_cid = np.zeros(shape=self.__tl1.get_num_particles(), dtype=np.bool)
        for scol in self.__scols.itervalues():
            for clst in scol[0]:
                c_ids, coords = clst.get_ids(), clst.get_coords()
                for c_id, coord in zip(c_ids, coords):
                    if not lut1_cid[c_id]:
                        X.append(coord)
                        lut_lyr.append(0)
                        lut_cid.append(c_id)
                        lut1_cid[c_id] = True
        X, lut_lyr, lut_cid = np.asarray(X, dtype=np.float32), np.asarray(lut_lyr, dtype=np.int), \
                              np.asarray(lut_cid, dtype=np.int)

        # Clustering algorithm just for layer 1
        hold_clsts_1, hold_cols_2, hold_cols_3, hold_cols = list(), list(), list(), dict()
        if len(X) <= 0:
            return hold_cols
        elif len(X) == 1:
            lbls = [1, ]
            class_ids = [1, ]
        else:
            Y = sp.spatial.distance.pdist(X)
            # Z = sp.cluster.hierarchy.single(Y)
            Z = sp.cluster.hierarchy.linkage(Y, method='single', metric='euclidean')
            lbls = sp.cluster.hierarchy.fcluster(Z, dst_clst, criterion='distance')
            class_ids = list(set(lbls))

        # Creating the cluster for the first layer
        for idx in class_ids:
            if idx == -1:
                continue
            c_ids, coords = list(), list()
            for i, lbl in enumerate(lbls):
                if lbl == idx:
                    c_ids.append(lut_cid[i])
                    coords.append(X[i, :])
            if len(c_ids) > 0:
                hold_clsts_1.append(Cluster(c_ids, coords))
        lut2_cid = np.zeros(shape=self.__tl2.get_num_particles(), dtype=np.bool)
        lut3_cid = np.zeros(shape=self.__tl3.get_num_particles(), dtype=np.bool)
        for scol in self.__scols.itervalues():
            for clst in scol[1]:
                c_ids, c_coords = clst.get_ids(), clst.get_coords()
                for c_id, c_coord in zip(c_ids, c_coords):
                    if not lut2_cid[c_id]:
                        hold_cols_2.append(Cluster([c_id, ], [c_coord, ]))
                        lut2_cid[c_id] = True
            for clst in scol[2]:
                c_ids, c_coords = clst.get_ids(), clst.get_coords()
                for c_id, c_coord in zip(c_ids, c_coords):
                    if not lut3_cid[c_id]:
                        hold_cols_3.append(Cluster([c_id, ], [c_coord, ]))
                        lut3_cid[c_id] = True

        # Loop for finding the closest cluster for each subcolumn cluster in layer 2
        lut2_cid, hold_cols_lyr2 = dict().fromkeys(self.__scols.keys(), -1), dict()
        for c2_id, scol_clst in enumerate(hold_cols_2):
            n_algn, c1_id = 0, -1
            for hold_c_id, clsts_1 in enumerate(hold_clsts_1):
                hold_n_algn = clsts_1.find_num_aligned(scol_clst, dst_aln)
                if hold_n_algn > n_algn:
                    c1_id = hold_c_id
                    n_algn = hold_n_algn
            if c1_id >= 0:
                lut2_cid[c2_id] = c1_id
        for c2_id, c1_id in zip(lut2_cid.iterkeys(), lut2_cid.itervalues()):
            if c1_id >= 0:
                try:
                    hold_cols_lyr2[c1_id].append(hold_cols_2[c2_id])
                except KeyError:
                    hold_cols_lyr2[c1_id] = list()
                    hold_cols_lyr2[c1_id].append(hold_cols_2[c2_id])

        # Loop for finding the closest cluster for each subcolumn cluster in layer 3
        lut3_cid, hold_cols_lyr3 = dict().fromkeys(self.__scols.keys(), -1), dict()
        for c3_id, scol_clst in enumerate(hold_cols_3):
            n_algn, c1_id = 0, -1
            for hold_c_id, clsts_1 in enumerate(hold_clsts_1):
                hold_n_algn = clsts_1.find_num_aligned(scol_clst, dst_aln)
                if hold_n_algn > n_algn:
                    c1_id = hold_c_id
                    n_algn = hold_n_algn
            if c1_id >= 0:
                lut3_cid[c3_id] = c1_id
        for c3_id, c1_id in zip(lut3_cid.iterkeys(), lut3_cid.itervalues()):
            if c1_id >= 0:
                try:
                    hold_cols_lyr3[c1_id].append(hold_cols_3[c3_id])
                except KeyError:
                    hold_cols_lyr3[c1_id] = list()
                    hold_cols_lyr3[c1_id].append(hold_cols_3[c3_id])

        # Creating the columns
        count_col = 0
        for c1_id, clst_1 in enumerate(hold_clsts_1):
            if (c1_id in hold_cols_lyr2.keys()) and (c1_id in hold_cols_lyr3.keys()):
                hold_cols[count_col] = (list(), list(), list())
                hold_cols[count_col][0].append(clst_1)
                for clst_2 in hold_cols_lyr2[c1_id]:
                    hold_cols[count_col][1].append(clst_2)
                for clst_3 in hold_cols_lyr3[c1_id]:
                    hold_cols[count_col][2].append(clst_3)
                count_col += 1

        return hold_cols

    def __build_clusters_HC_m2(self, dst_clst, dst_aln):
        """
        Private method for generating clusters within layers by applying Hierarchical Clustering for columns
        :param dst_clst: cluster distance threshold
        :param dst_aln: distance alignment threshold
        :return:
        """

        # Initialization
        hold_cols = dict()

        # Loop for layers
        hold_clsts = dict()
        for id_lyr, lyr in enumerate([self.__tl1, self.__tl2, self.__tl3]):
            X = np.zeros(shape=(lyr.get_num_particles(), 3), dtype=np.float32)
            for i, part in enumerate(lyr.get_particles()):
                X[i, :] = part.get_center()

            # Clustering algorithm
            hold_clsts[id_lyr] = list()
            if len(X) <= 0:
                continue
            elif len(X) == 1:
                hold_clsts[id_lyr].append(Cluster([0, ], [X[0, :], ]))
                continue
            Y = sp.spatial.distance.pdist(X)
            # Z = sp.cluster.hierarchy.single(Y)
            Z = sp.cluster.hierarchy.linkage(Y, method='single', metric='euclidean')
            # lbls = sp.cluster.hierarchy.fcluster(Z, dst, criterion='distance')
            lbls = sp.cluster.hierarchy.fcluster(Z, dst_clst, criterion='distance')
            class_ids = list(set(lbls))

            # Creating the clustering
            for idx in class_ids:
                if idx == -1:
                    continue
                c_ids, coords = list(), list()
                for i, lbl in enumerate(lbls):
                    if lbl == idx:
                        c_ids.append(i)
                        coords.append(X[i, :])
                if len(c_ids) > 0:
                    hold_clsts[id_lyr].append(Cluster(c_ids, coords))

        # Getting the ids of all the particles within sub-columns
        scol_ids_1, scol_ids_2, scol_ids_3 = list(), list(), list()
        for scol in self.__scols.itervalues():
            for clst_1 in scol[0]:
                scol_ids_1 += clst_1.get_ids()
            for clst_2 in scol[1]:
                scol_ids_2 += clst_2.get_ids()
            for clst_3 in scol[2]:
                scol_ids_3 += clst_3.get_ids()

        # Generating the Columns by checking clustering alignment by guaranteeing that it contains a sub-column
        count_col = 0
        for i, clst_1 in enumerate(hold_clsts[0]):
            hold_col = (list(), list(), list())
            hold_col[0].append(clst_1)
            for clst_2 in hold_clsts[1]:
                if clst_1.find_num_aligned(clst_2, dst_aln) > 0:
                    hold_col[1].append(clst_2)
            for clst_3 in hold_clsts[2]:
                if clst_1.find_num_aligned(clst_3, dst_aln) > 0:
                    hold_col[2].append(clst_3)
            # Check clusters column alignment
            if (len(hold_col[0]) > 0) and (len(hold_col[1]) > 0) and (len(hold_col[2]) > 0):
                # Check that within the cluster there are at least a sub-column
                is_scol_1, is_scol_2, is_scol_3 = False, False, False
                for clst_1 in hold_col[0]:
                    for c_id in clst_1.get_ids():
                        if c_id in scol_ids_1:
                            is_scol_1 = True
                            break
                    if is_scol_1:
                        break
                for clst_2 in hold_col[1]:
                    for c_id in clst_2.get_ids():
                        if c_id in scol_ids_2:
                            is_scol_2 = True
                            break
                    if is_scol_2:
                        break
                for clst_3 in hold_col[2]:
                    for c_id in clst_3.get_ids():
                        if c_id in scol_ids_3:
                            is_scol_3 = True
                            break
                    if is_scol_3:
                        break
                if is_scol_1 and is_scol_2 and is_scol_3:
                    hold_cols[count_col] = hold_col
                    count_col += 1
                # hold_cols[count_col] = hold_col
                # count_col += 1
        return hold_cols
