"""
Classes for reducing the size of a SkelGraph but keeping its Morse-Smale complex structure

# Author: Antonio Martinez-Sanchez (Max Planck Institute for Biochemistry)
# Date: 18.11.14
"""

__author__ = 'martinez'

import gc
import operator
from core import *
import warnings
from scipy import sparse
try:
    import pexceptions
except:
    from pyseg import pexceptions
# import math
import copy

try:
    import disperse_io
except:
    from pyseg import disperse_io
from pyseg.factory import SubGraphVisitor
from pyseg.filament import SetSpaceCurve, FilPerVisitor
from pyseg.sub import TomoPeaks
# import graph_tool.all as gt
from pyseg.globals import *
from .gt import GraphGT
from pyseg import diff_geom
import multiprocessing as mp

try:
    import cPickle as pickle
except:
    import pickle

###### Global definitions

MAX_NPROPS = 100
CROP_OFF = 0

####### Helper functions

# Thread for finding max persistence length for filaments
        # Shared arrays idex: 0: persistence length, 1: (unsigned) third curavature, 2: sinuosity, 3: apex length, 4: unsigned
        #                        curvature, 5: unsigned torsion
def th_find_max_per(th_id, graph_mcf, vertices, v_ids, samp_len, mx_ktt, mn_len, mx_len, shared_arrs, fils):

    # Initialization
    ukt_arr, len_arr = shared_arrs[0], shared_arrs[1]
    sin_arr, apl_arr = shared_arrs[2], shared_arrs[3]
    unk_arr, unt_arr = shared_arrs[4], shared_arrs[5]

    # Vertices loop
    hold_fils = fils[th_id]
    nv = len(v_ids)
    # count = 1
    for i in range(nv):

        # print 'Process ' + str(th_id) + ': ' + str(count) + ' of ' + str(nv)
        # count += 1

        # Find maximum length filament before stopping condition (third curvature limit)
        v = vertices[v_ids[i]]
        v_id = v.get_id()
        finder = FilPerVisitor(graph_mcf, v_id, samp_len, mx_ktt, mn_len, mx_len)
        if fils is not None:
            perx, film, fil = finder.find_max_per_filament(gen_fil=True)
            if fil is not None:
                hold_fils.append(fil)
        else:
            perx, film = finder.find_max_per_filament(gen_fil=False)

        # Synchronize shared data
        if film is not None:
            for v in film.get_vertices():
                v_id = v.get_id()
                if (len_arr[v_id] < 0) or (len_arr[v_id] < fil.get_length()):
                    ukt_arr[v_id], len_arr[v_id] = fil.get_total_ukt(), fil.get_length()
                    sin_arr[v_id], apl_arr[v_id] = fil.get_sinuosity(), fil.get_apex_length(update=False)
                    unk_arr[v_id], unt_arr[v_id] = fil.get_total_ut(), fil.get_total_ut()
                if fils is not None:
                    fils[th_id] = hold_fils
            for e in film.get_edges():
                e_id = e.get_id()
                if (len_arr[e_id] < 0) or (len_arr[e_id] > fil.get_length()):
                    ukt_arr[e_id], len_arr[e_id] = fil.get_total_ukt(), fil.get_length()
                    sin_arr[e_id], apl_arr[e_id] = fil.get_sinuosity(), fil.get_apex_length(update=False)
                    unk_arr[e_id], unt_arr[e_id] = fil.get_total_ut(), fil.get_total_ut()

##### Class for fast computation of the distances tranform on subvolumes
class SubVolDtrans(object):

    # mask: input binary mask
    def __init__(self, mask):
        self.__mask = mask.astype(np.bool)
        sap = self.__mask.shape
        self.__mx_b, self.__my_b, self.__mz_b = sap[0], sap[1], sap[2]
        self.__Y, self.__X, self.__Z = np.meshgrid(np.arange(sap[1]).astype(np.int16),
                                                   np.arange(sap[0]).astype(np.int16),
                                                   np.arange(sap[2]).astype(np.int16),
                                                   copy=False)

    def get_subvol(self, box):
        return self.__mask[box[0][0]:box[0][1], box[1][0]:box[1][1], box[2][0]:box[2][1]]

    # Distance transform on a coordinate point
    # point: 3D point array
    # max_d: for cropping distances (in voxels)
    # Returns: the squared cropped distance, the bounding box
    # ((x.m,x_M), (y_m,y_M), (y_m,y_M)) and the point coordinates in the subvolume in voxels
    def point_distance_trans(self, point, max_d):

        # Computing bounding box
        max_d_v = int(math.ceil(max_d))
        max_d_v_m1, max_d_v_1 = max_d_v-CROP_OFF, max_d_v-CROP_OFF+1
        box_x_l, box_x_h = int(round(point[0]-max_d_v_m1)), int(round(point[0]+max_d_v_1))
        box_y_l, box_y_h = int(round(point[1]-max_d_v_m1)), int(round(point[1]+max_d_v_1))
        box_z_l, box_z_h = int(round(point[2]-max_d_v_m1)), int(round(point[2]+max_d_v_1))
        if box_x_l < 0:
            box_x_l = 0
        if box_x_h > self.__mx_b:
            box_x_h = self.__mx_b
        if box_y_l < 0:
            box_y_l = 0
        if box_y_h > self.__my_b:
            box_y_h = self.__my_b
        if box_z_l < 0:
            box_z_l = 0
        if box_z_h > self.__mz_b:
            box_z_h = self.__mz_b

        # Distance computation
        hold_x = self.__X[box_x_l:box_x_h, box_y_l:box_y_h, box_z_l:box_z_h] - point[0]
        hold_y = self.__Y[box_x_l:box_x_h, box_y_l:box_y_h, box_z_l:box_z_h] - point[1]
        hold_z = self.__Z[box_x_l:box_x_h, box_y_l:box_y_h, box_z_l:box_z_h] - point[2]

        return (hold_x*hold_x + hold_y*hold_y + hold_z*hold_z), \
               ((box_x_l, box_x_h), (box_y_l, box_y_h), (box_z_l, box_z_h)), \
                (point[0]-box_x_l, point[1]-box_y_l, point[2]-box_z_l)

# Process for parallel computation
def pr_graph_rdf(pr_id, max_r, bin_s, ids, coords, dsts_vec, mask, res, verts_rdf_mpa):

    res3 = res**3
    len_b = len(bin_s)

    # Loop for vertices
    for i, idx in enumerate(ids):

        # Distance transform
        coord = coords[i]
        sub_dists, box, pt_s = mask.point_distance_trans(coord, max_r)
        sub_mask = mask.get_subvol(box)

        # Count num of particles in the shell and shell volume
        point_dists = dsts_vec[i]
        shell_dist = np.sqrt(sub_dists[sub_mask])
        hold_num = np.zeros(shape=len_b, dtype=np.float)
        hold_dem = np.zeros(shape=len_b, dtype=np.float)
        for j in range(1, len(bin_s)):
            hold_num[j] = float(((point_dists>=bin_s[j][0]) & (point_dists<bin_s[j][1])).sum())
            hold_dem[j] = float(((shell_dist>=bin_s[j][0]) & (shell_dist<bin_s[j][1])).sum())
        hold_dem[0] = 1.
        hold_dem *= res3

        # Update the shared arrays
        for j, idx in enumerate(np.arange(idx, idx+len_b)):
            verts_rdf_mpa[idx] = hold_num[j] / hold_dem[j]

        # print 'Thread ' + str(pr_id) + ': processing state vertex ' + str(idx) + ' of ' + str(len(ids))
    print 'Thread ' + str(pr_id) + ': finished!'
    return

# ####################################################################################################
# Class which represent an arc between a minima and a saddle point
#
#
class ArcMCF(object):
    #### Constructor Area

    # id: arc_id in the skeleton
    # ids: ordered point ids of the arc, head is minima and tail is saddle
    def __init__(self, id, ids):
        if len(ids) < 2:
            error_msg = 'An arc must comprise 2 or more points.'
            raise pexceptions.PySegInputError(expr='(ArcMCF)', msg=error_msg)
        self.__id = id
        self.__ids = ids

    #### Set/Get methods area

    def get_ids(self):
        return self.__ids

    # mode: set if cell or point id is returned
    def get_min_id(self):
        return self.__ids[0]

    # mode: set if cell or point id is returned
    def get_sad_id(self):
        return self.__ids[-1]

    def get_id(self):
        return self.__id

    def get_npoints(self):
        return len(self.__ids)

    def get_point_id(self, idx):
        return self.__ids[idx]

    def get_length(self, skel):

        length = 0
        x1, y1, z1 = skel.GetPoint(self.__ids[0])
        for i in range(1, len(self.__ids)):
            x2, y2, z2 = skel.GetPoint(self.__ids[i])
            xh, yh, zh = x1 - x2, y1 - y2, z1 - z2
            length += math.sqrt(xh * xh + yh * yh + zh * zh)
            x1, y1, z1 = x2, y2, z2

        return length

    # Weighted edge length (similar to Mahalanobis distance)
    # w_x|y|z: weighting for each dimension
    def get_length_2(self, skel, w_x, w_y, w_z):

        length = 0
        x1, y1, z1 = skel.GetPoint(self.__ids[0])
        for i in range(1, len(self.__ids)):
            x2, y2, z2 = skel.GetPoint(self.__ids[i])
            xh, yh, zh = x1 - x2, y1 - y2, z1 - z2
            length += math.sqrt(w_x*xh*xh + w_y*yh*yh + w_z*zh*zh)
            x1, y1, z1 = x2, y2, z2

        return length

    # Extend the arc with another at the end, the extension must share the an extremum with
    # the current arc, otherwise the function raises an error
    # side: if 'sad' (default) if the new arc id list is added at the end (saddle point side),
    # otherwise it is added at the begining (minimum point side)
    def extend(self, arc, side='sad'):
        if side == 'sad':
            if self.__ids[-1] == arc.__ids[0]:
                for i in range(1, len(arc.__ids)):
                    self.__ids.append(arc.__ids[i])
            elif self.__ids[-1] == arc.__ids[-1]:
                rev_list = arc.__ids[::-1]
                for i in range(1, len(rev_list)):
                    self.__ids.append(rev_list[i])
            else:
                error_msg = 'Arc for extesion must share an extremum with current arc.'
                raise pexceptions.PySegInputError(expr='(ArcMCF)', msg=error_msg)
        else:
            if self.__ids[0] == arc.__ids[0]:
                for i in range(1, len(arc.__ids)):
                    self.__ids.insert(0, arc.__ids[i])
            elif self.__ids[0] == arc.__ids[-1]:
                rev_list = arc.__ids[::-1]
                for i in range(1, len(rev_list)):
                    self.__ids.insert(0, rev_list[i])
            else:
                error_msg = 'Arc for extesion must share an extremum with current arc.'
                raise pexceptions.PySegInputError(expr='(ArcMCF)', msg=error_msg)



#####################################################################################################
# Class which represent a filamentary complex in Morse theory MFC (arcs between minima
# and saddle points)
#
class VertexMCF(object):
    #### Constructor Area

    # id: point id of the minima
    # arcs_id: list with the point id of the arcs (ArcMFC) which compound this MCF
    def __init__(self, id, arcs=None):
        self.__id = id
        if arcs is None:
            self.__arcs = list()
        else:
            self.__arcs = arcs
        # self.__parse_topology()
        self.__geometry = None

    #### Set/Get functionlity

    def get_id(self):
        return self.__id

    # skel: disperse skeleton
    def get_coordinates(self, skel):
        return skel.GetPoint(self.__id)

    def get_geometry(self):
        return self.__geometry

    def get_arcs(self):
        return self.__arcs

    #### Funtionality area

    def del_arc(self, arc):
        try:
            self.__arcs.remove(arc)
        except:
            return

    def add_arc(self, arc):
        self.__arcs.append(arc)

    def add_geometry(self, geometry):
        self.__geometry = geometry

    # # skel: disperse skeleton
    # # manifold: numpy array with the labels for the manifolds
    # # density: numpy array with the density map
    # def add_geometry(self, skel, manifold, density):
    #
    #     # Checks that booth tomograms have the same size
    #     if manifold.shape != density.shape:
    #         error_msg = 'Manifold and Density tomograms have different size.'
    #         raise pexceptions.PySegInputError(expr='add_geometry (VertexMCF)', msg=error_msg)
    #
    #     # Creates geometry
    #     self.__geometry = geometry.PointGeometry(self.get_coordinates(skel), manifold, density)

    #### Internal area function

    def __parse_topology(self):

        # Check there are arcs
        if len(self.__arcs) < 1:
            error_msg = 'A VertexMCF must have at least one ArcMFC.'
            raise pexceptions.PySegInputError(expr='__parse_topology (VertexMCF)', msg=error_msg)

        # Check that all arc heads are the same minima
        for a in self.__arcs:
            if a.get_head_id() != self.__id:
                error_msg = 'The ArcMCF must have the same minima.'
                raise pexceptions.PySegInputError(expr='__parse_topology (VertexMCF)', msg=error_msg)


##################################################################################################
# Class which represent Edges, connections between the same saddle point shared by two VertexMCF
#
#
class EdgeMCF(object):
    #### Constructor Area

    # id: point id of the saddle point
    # v_id_s, v_id_t: id of the source and target vertices
    def __init__(self, id, v_id_s, v_id_t):
        self.__id = id
        self.__v_id_s = v_id_s
        self.__v_id_t = v_id_t

    #### Set/Get functionality

    def get_id(self):
        return self.__id

    # Return id of the source vertex
    def get_source_id(self):
        return self.__v_id_s

    # Return id of the target vertex
    def get_target_id(self):
        return self.__v_id_t


#########################################################################################################
# Class for holding the information and value of the properties
# TODO: deprecated, substitution version under development
#
class TableProps(object):
    ###### Constructor Area

    # entries -> number of entries from the beginning (default 0), this is an static value
    # key-> string which identifies the property
    # type-> only 'int' and 'double' are currently accepted
    # ncomp-> number of components
    def __init__(self, entries=0):
        self.__key = list()
        self.__type = list()
        self.__ncomp = list()
        self.__values = np.empty(shape=entries, dtype=list)
        for i in range(len(self.__values)):
            self.__values[i] = list()

    ##### Get Area

    def get_nentries(self):
        return len(self.__values)

    def get_nprops(self):
        return len(self.__key)

    def get_key(self, index=-1):
        return self.__key[index]

    def get_keys(self):
        return self.__key

    def get_type(self, index=-1, key=None):
        if key is None:
            return self.__type[index]
        else:
            try:
                idx = self.__key.index(key)
            except:
                error_msg = "No property found with the key '%s'." % key
                raise pexceptions.PySegInputWarning(expr='get_type (Properties)', msg=error_msg)
            return self.__type[idx]

    def get_ncomp(self, index=-1, key=None):
        if key is None:
            return self.__ncomp[index]
        else:
            try:
                idx = self.__key.index(key)
            except:
                error_msg = "No property found with the key '%s'." % key
                raise pexceptions.PySegInputWarning(expr='get_ncomp (Properties)', msg=error_msg)
            return self.__ncomp[idx]

    # Set a prop value in all entries
    def set_prop(self, key, value):
        idx = self.is_already(key)
        if idx is None:
            error_msg = "No property found with the key '%s'." % key
            raise pexceptions.PySegInputWarning(expr='set_prop (Properties)', msg=error_msg)
        if len(value) != self.__ncomp[idx]:
            error_msg = "This property has %d components instead %s." % \
                        (self.__ncomp[idx], len(value))
            raise pexceptions.PySegInputWarning(expr='set_prop (Properties)', msg=error_msg)
        for v in self.__values:
            v[idx] = value

    # Set the property value of an entry
    def set_prop_entry(self, key, value, id_entry):
        idx = self.is_already(key)
        if idx is None:
            error_msg = "No property found with the key '%s'." % key
            raise pexceptions.PySegInputWarning(expr='set_prop (Properties)', msg=error_msg)
        try:
            if len(value) != self.__ncomp[idx]:
                error_msg = "This property has %d components instead %d." % \
                            (self.__ncomp[idx], len(value))
                raise pexceptions.PySegInputWarning(expr='set_prop (Properties)', msg=error_msg)
        except TypeError:
            if self.__ncomp[idx] != 1:
                error_msg = "This property has %d components instead 1." % self.__ncomp[idx]
                raise pexceptions.PySegInputWarning(expr='set_prop (Properties)', msg=error_msg)
        ent = self.__values[int(id_entry)]
        if isinstance(value, tuple):
            ent[idx] = value
        else:
            try:
                ent[idx] = tuple(value)
            except TypeError:
                ent[idx] = (value,)

    # Get a prop value in all entries in a list
    def get_prop(self, key):
        idx = self.is_already(key)
        if idx is None:
            error_msg = "No property found with the key '%s'." % key
            raise pexceptions.PySegInputWarning(expr='get_prop (Properties)', msg=error_msg)
        prop = list()
        for v in self.__values:
            prop.append(v[idx])

    # Get a prop max
    def get_prop_max(self, key):
        idx = self.is_already(key)
        if idx is None:
            error_msg = "No property found with the key '%s'." % key
            raise pexceptions.PySegInputWarning(expr='get_prop_max (Properties)', msg=error_msg)
        mx = float("-inf")
        if self.__ncomp[idx] > 1:
            for v in self.__values:
                hold = math.sqrt(sum(v[idx] * v[idx]))
                if hold > mx:
                    mx = hold
        else:
            for v in self.__values:
                hold = v[idx][0]
                if hold > mx:
                    mx = hold
        return mx

    # Get a prop min
    def get_prop_min(self, key):
        idx = self.is_already(key)
        if idx is None:
            error_msg = "No property found with the key '%s'." % key
            raise pexceptions.PySegInputWarning(expr='get_prop_min (Properties)', msg=error_msg)
        mn = float("inf")
        if self.__ncomp[idx] > 1:
            for v in self.__values:
                hold = math.sqrt(sum(v[idx] * v[idx]))
                if hold < mn:
                    mn = hold
        else:
            for v in self.__values:
                hold = v[idx][0]
                if hold < mn:
                    mn = hold
        return mn

    # Get the property value of an entry
    def get_prop_entry(self, key, id_entry):
        idx = self.is_already(key)
        if idx is None:
            error_msg = "No property found with the key '%s'." % key
            raise pexceptions.PySegInputWarning(expr='get_prop_entry (Properties)', msg=error_msg)
        return self.__values[id_entry][idx]

    ##### Functionality area

    # If this already exists return its index, otherwise None
    def is_already(self, key):
        try:
            idx = self.__key.index(key)
        except:
            return None
        return idx

    # If this property already exists it is overwritten
    def add_prop(self, key, type, ncomp, def_val=-1):
        idx = self.is_already(key)
        if idx is None:
            self.__key.append(key)
            self.__type.append(type)
            self.__ncomp.append(ncomp)
            for v in self.__values:
                val = list()
                for i in range(ncomp):
                    val.append(def_val)
                v.append(tuple(val))
        else:
            for v in self.__values:
                val = list()
                for i in range(ncomp):
                    val.append(def_val)
                v[idx] = tuple(val)

    # Remove by index or key (if key parameter is different from 0)
    def remove_prop(self, idx=-1, key=None):

        if key is None:
            if idx < 0:
                self.__key.pop()
                self.__type.pop()
                self.__ncomp.pop()
                for v in self.__values:
                    v.pop()
            else:
                self.__key.remove(idx)
                self.__type.remove(idx)
                self.__ncomp.remove(idx)
                for v in self.__values:
                    v.remove(idx)
        else:
            idx = self.is_already(key)
            if idx is not None:
                self.__key.remove(idx)
                self.__type.remove(idx)
                self.__ncomp.remove(idx)
                for v in self.__values:
                    v.remove(idx)

    # Creates a new instance and copies in it the current state
    # TODO: DEPRECATED, USE COPY PACKAGE INSTEAD
    def copy(self):

        copy_table = TableProps(self.get_nentries())
        for i in range(self.get_nprops()):
            copy_table.__key.append(self.__key[i])
            copy_table.__type.append(self.__type[i])
            copy_table.__ncomp.append(self.__ncomp[i])
        for i in range(self.get_nentries()):
            for e in self.__values[i]:
                copy_table.__values[i].append(e)

        return copy_table

#########################################################################################################
# Class for holding the information and value of the properties based on a sparse matrix
# All entries will be casted to np.float for being stored, so all bigger formats will be
# truncated
#
class TablePropsTest(object):
    ###### Constructor Area

    # entries -> number of entries from the beginning (default 0), this is an static value
    # key-> string which identifies the property
    # type-> only 'int' and 'double' are currently accepted
    # ncomp-> number of components
    # nmax_props-> maximum number of props (default MAX_NPROPS), this is due to the sparse matrix
    # must be created statically. I a property has n components it is counted n props
    def __init__(self, entries=0, nmax_props=MAX_NPROPS):
        self.__key = np.empty(shape=nmax_props, dtype=object)
        self.__type = np.empty(shape=nmax_props, dtype=object)
        self.__ncomp = np.empty(shape=nmax_props, dtype=np.int8)
        self.__values = sparse.lil_matrix((nmax_props, entries), dtype=np.float)
        self.__entries = sparse.lil_matrix((1, entries), dtype=np.bool)
        # Properties counter
        self.__props_count = 0

    ##### Get Area

    def get_nentries(self):
        return self.__values.shape[1]

    def get_nprops(self):
        prop = list()
        for i in range(self.__props_count):
            if self.__key[i] not in prop:
                prop.append(self.__key[i])
        return len(prop)
        #return self.__props_count

    def get_key(self, index=-1):
        return self.__key[index]

    # Return the keys as a list
    def get_keys(self):
        prop = list()
        for i in range(self.__props_count):
            if self.__key[i] not in prop:
                prop.append(self.__key[i])
        return prop
        # idx = 0
        # count = 0
        # l_keys = list()
        #
        # while count < self.__props_count:
        #     l_keys.append(self.__key[idx])
        #     # Compute next index
        #     idx += self.__ncomp[idx]
        #     count += 1
        #
        # return l_keys

    def get_type(self, index=-1, key=None):
        if key is None:
            return self.__type[index]
        else:
            try:
                idx = self.__key.index(key)
            except:
                error_msg = "No property found with the key '%s'." % key
                raise pexceptions.PySegInputWarning(expr='get_type (TableProps)', msg=error_msg)
            return self.__type[idx]

    def get_ncomp(self, index=-1, key=None):
        if key is None:
            return self.__ncomp[index]
        else:
            idx = self.is_already(key)
            if idx is None:
                error_msg = "No" \
                            " property found with the key '%s'." % key
                raise pexceptions.PySegInputWarning(expr='get_ncomp (TableProps)', msg=error_msg)
            return self.__ncomp[idx]

    # Set a prop value in all entries
    def set_prop(self, key, value):
        idx = self.is_already(key)
        if idx is None:
            error_msg = "No property found with the key '%s'." % key
            raise pexceptions.PySegInputWarning(expr='set_prop (TableProps)', msg=error_msg)
        try:
            n_comp = len(value)
        except TypeError:
            n_comp = 1
        if n_comp != self.__ncomp[idx]:
            error_msg = "This property has %d components instead of %d." % \
                        (self.__ncomp[idx], len(value))
            raise pexceptions.PySegInputWarning(expr='set_prop (TableProps)', msg=error_msg)
        nz_idx = self.__entries.nonzero()
        if n_comp == 1:
            for i in range(len(nz_idx[1])):
                self.__values[idx, nz_idx[1][i]] = value
        else:
            for i in range(n_comp):
                for j in range(len(nz_idx[1])):
                    self.__values[idx+i, nz_idx[1][j]] = value[i]

    # Set the property value of an entry
    def set_prop_entry(self, key, value, id_entry):
        idx = self.is_already(key)
        if idx is None:
            error_msg = "No property found with the key '%s'." % key
            raise pexceptions.PySegInputWarning(expr='set_prop_entry (TableProps)',
                                                msg=error_msg)
        try:
            n_comp = len(value)
        except TypeError:
            n_comp = 1
        if n_comp != self.__ncomp[idx]:
                error_msg = "This property has %d components instead of %d." % \
                            (self.__ncomp[idx], len(value))
                raise pexceptions.PySegInputWarning(expr='set_prop_entry (TableProps)',
                                                    msg=error_msg)
        self.__entries[0, id_entry] = True
        for i in range(n_comp):
            self.__values[idx+i, id_entry] = value[i]

    # Set the property value of an entry faster than set_prop_entry()
    def set_prop_entry_fast(self, key_id, value, id_entry, n_comp):
        self.__entries[0, id_entry] = True
        for i in range(n_comp):
            self.__values[key_id+i, id_entry] = value[i]

    # Get a prop value in all entries in a list
    def get_prop(self, key):
        idx = self.is_already(key)
        if idx is None:
            error_msg = "No property found with the key '%s'." % key
            raise pexceptions.PySegInputWarning(expr='get_prop (TableProps)', msg=error_msg)
        n_comp = self.get_ncomp(idx)
        if n_comp != self.__ncomp[idx]:
            error_msg = "This property has %d components instead of %d." % \
                        (self.__ncomp[idx], n_comp)
            raise pexceptions.PySegInputWarning(expr='get_prop (TableProps)', msg=error_msg)
        prop = list()
        nz_idx = self.__entries.nonzero()
        if n_comp == 1:
            for i in range(len(nz_idx[1])):
                prop.append(self.__values[idx, nz_idx[1][i]])
        else:
            for i in range(len(nz_idx[1])):
                entry = list()
                for j in range(n_comp):
                    entry.append(self.__values[idx+j, nz_idx[1][i]])
                prop.append(tuple(entry))

    # Get a prop max, if the prop is a vector (number of components greater than 1) the
    # euclidean norm is taken
    def get_prop_max(self, key):
        idx = self.is_already(key)
        if idx is None:
            error_msg = "No property found with the key '%s'." % key
            raise pexceptions.PySegInputWarning(expr='get_prop_max (TableProps)', msg=error_msg)
        data_type = disperse_io.TypesConverter().gt_to_numpy(self.get_type(index=idx))
        try:
            mx = np.finfo(data_type).min
        except:
            mx = np.iinfo(data_type).min
        n_comp = self.get_ncomp(index=idx)
        nz_idx = self.__entries.nonzero()
        if n_comp > 1:
            for i in range(len(nz_idx[1])):
                hold = .0
                for j in range(n_comp):
                    hold2 = self.__values[idx+j, nz_idx[1][i]]
                    hold += (hold2 * hold2)
                hold = math.sqrt(hold)
                if hold > mx:
                    mx = hold
        else:
            for i in range(len(nz_idx[1])):
                hold = self.__values[idx, nz_idx[1][i]]
                if hold > mx:
                    mx = hold

        return data_type(mx)

    # Get a prop min, if the prop is a vector (number of components greater than 1) the
    # euclidean norm is taken
    def get_prop_min(self, key):
        idx = self.is_already(key)
        if idx is None:
            error_msg = "No property found with the key '%s'." % key
            raise pexceptions.PySegInputWarning(expr='get_prop_min (TableProps)', msg=error_msg)
        data_type = disperse_io.TypesConverter().gt_to_numpy(self.get_type(index=idx))
        try:
            mn = np.finfo(data_type).max
        except:
            mn = np.iinfo(data_type).max
        n_comp = self.get_ncomp(index=idx)
        nz_idx = self.__entries.nonzero()
        if n_comp > 1:
            for i in range(len(nz_idx[1])):
                hold = .0
                for j in range(n_comp):
                    hold2 = self.__values[idx+j, nz_idx[1][i]]
                    hold += (hold2 * hold2)
                hold = math.sqrt(hold)
                if hold < mn:
                    mn = hold
        else:
            for i in range(len(nz_idx[1])):
                hold = self.__values[idx, nz_idx[1][i]]
                if hold < mn:
                    mn = hold

        return data_type(mn)

    def invert_prop(self, key_old, key_new):
        idx_old = self.is_already(key_old)
        if idx_old is None:
            error_msg = "No property found with the key '%s'." % key_old
            raise pexceptions.PySegInputWarning(expr='invert_prop (TableProps)', msg=error_msg)
        data_type = disperse_io.TypesConverter().gt_to_numpy(self.get_type(index=idx_old))

        # Max and min computation
        try:
            mx = np.finfo(data_type).min
        except:
            mx = np.iinfo(data_type).min
        try:
            mn = np.finfo(data_type).max
        except:
            mn = np.iinfo(data_type).max
        n_comp = self.get_ncomp(index=idx_old)
        nz_idx = self.__entries.nonzero()
        if n_comp > 1:
            error_msg = "Only properties with one component can be inverted."
            raise pexceptions.PySegInputWarning(expr='invert_prop (TableProps)', msg=error_msg)
        if len(nz_idx[1]) == 0:
            return
        for i in range(len(nz_idx[1])):
            hold = self.__values[idx_old, nz_idx[1][i]]
            if hold > mx:
                mx = hold
            if hold < mn:
                mn = hold

        # Remapping
        hold = mn - mx
        if hold == 0:
            m = 0
        else:
            m = (mx-mn) / hold
        c = mx - m*mn
        idx_new = self.add_prop(key_new, self.get_type(index=idx_old), n_comp, def_val=0)
        for i in range(len(nz_idx[1])):
            entry = nz_idx[1][i]
            self.__values[idx_new, entry] = self.__values[idx_old, entry]*m + c

        return idx_new

    # Get the property value of an entry, if key_id and n_comp is provided then it works faster,
    # if not this information must be inferred from key
    # key and key_id cannot be None at the same time
    def get_prop_entry(self, key=None, id_entry=0, key_id=None, n_comp=None, data_type=None):
        if key_id is None:
            key_id = self.is_already(key)
        if key_id is None:
            error_msg = "No property found with the key '%s' or key_id '%d'." % (key, key_id)
            raise pexceptions.PySegInputWarning(expr='get_prop_entry (TableProps)',
                                                msg=error_msg)
        if n_comp is None:
            n_comp = self.get_ncomp(index=key_id)
        if data_type is None:
            data_type = disperse_io.TypesConverter().gt_to_numpy(self.get_type(index=key_id))
        prop = list()
        for i in range(n_comp):
            prop.append(data_type(self.__values[key_id+i, id_entry]))
        return tuple(prop)

    # Set the property value of an entry faster than set_prop_entry()
    def get_prop_entry_fast(self, key_id, id_entry, n_comp, data_type):
        prop = list()
        for i in range(n_comp):
            prop.append(data_type(self.__values[key_id+i, id_entry]))
        return tuple(prop)

    ##### Functionality area

    # If this already exists return its index, otherwise None
    def is_already(self, key):
        try:
            hold = np.where(self.__key == key)
            # Get first occurrence
            idx = hold[0][0]
        except:
            return None
        return idx

    # If this property already exists it is overwritten
    # def_val: if def_val is 0 (default -1), it works faster
    def add_prop(self, key, type, ncomp, def_val=-1):
        idx = self.is_already(key)
        if idx is None:
            if self.__props_count >= MAX_NPROPS:
                error_msg = "The number of properties cannot be greater than '%d'." % \
                            self.__values.shape[0]
                raise pexceptions.PySegInputError(expr='add_prop (TableProps)', msg=error_msg)
            self.__props_count += ncomp
            idx = self.__props_count - ncomp
        else:
            ncomp = self.get_ncomp(index=idx)
        for i in range(idx, idx+ncomp):
            self.__key[i] = key
            self.__type[i] = type
            self.__ncomp[i] = ncomp
        if def_val != 0:
            nz_idx = self.__entries.nonzero()
            for i in range(ncomp):
                for j in range(len(nz_idx[1])):
                    self.__values[idx+i, nz_idx[1][j]] = def_val

        return idx

    def remove_entry(self, id_entry):
        self.__entries[0, id_entry] = False

    def remove_prop(self, key, idx=None):
        if self.__props_count == 0:
            error_msg = "No properties for removing."
            raise pexceptions.PySegInputWarning(expr='remove_prop (TableProps)', msg=error_msg)
        if idx is None:
            idx = self.is_already(key)
        if idx is None:
            error_msg = "No property found with the key '%s' or key_id '%d'." % (key, idx)
            raise pexceptions.PySegInputWarning(expr='remove_prop (TableProps)', msg=error_msg)
        for i in range(idx, self.__props_count):
            hold = self.__key[i+1]
            self.__key[i] = hold
            hold = self.__ncomp[i+1]
            self.__ncomp[i] = hold
            hold = self.__type[i+1]
            self.__type[i] = hold
            hold = self.__values.getrow(i+1)
            self.__values[i, :] = hold
            # hold = self.__values.getrow(i+i)
            # self.__values[idx, :] = hold
        self.__props_count -= 1

##################################################################################################
# Class for a subgraph contained by a parent GraphMCF. This subgraph will represent an
# independent subgraph of the parent, but vertices and edge properties will be consulted
# from the parent
#
class SubGraphMCF(object):

    #### Constructor Area

    # graph_mcf: Parent GraphMCF
    # v_ids: list of the vertices ids which form the subgraph
    # e_ids: list of the edges ids which form the subrgraph
    def __init__(self, graph_mcf, v_ids, e_ids):
        self.__graph_mcf = graph_mcf
        self.__graph = gt.Graph(directed=False)
        self.__build(v_ids, e_ids)

    def get_num_vertices(self):
        return self.__graph.num_vertices()

    def get_num_edges(self):
        return self.__graph.num_edges()

    # Return the volume accumulated by all vertex geometries
    def get_volume(self):
        vol = 0
        v_prop = self.__graph.vertex_properties[STR_SGM_VID]
        for v in self.__graph.vertices():
            geom = self.__graph_mcf.get_vertex(v_prop[v]).get_geometry()
            vol += geom.get_volume()
        return vol

    #### Internal functionality area

    def __build(self, v_ids, e_ids):

        # Initialization LUT
        lut = np.zeros(shape=self.__graph_mcf.get_nid(), dtype=object)

        # Adding vertices
        for v_id in v_ids:
            lut[v_id] = self.__graph.add_vertex()
        self.__graph.vertex_properties[STR_SGM_VID] = self.__graph.new_vertex_property('int')
        self.__graph.vertex_properties[STR_SGM_VID].get_array()[:] = np.asarray(v_ids,
                                                                                dtype=np.int)

        # Adding edges
        for e_id in e_ids:
            s_id = self.__graph_mcf.get_edge(e_id).get_source_id()
            t_id = self.__graph_mcf.get_edge(e_id).get_target_id()
            self.__graph.add_edge(lut[s_id], lut[t_id])
        self.__graph.edge_properties[STR_SGM_EID] = self.__graph.new_edge_property('int')
        self.__graph.edge_properties[STR_SGM_EID].get_array()[:] = np.asarray(e_ids,
                                                                              dtype=np.int)

##################################################################################################
# Class for a graph of MCFs (now vertices, edges and arcs are indexed by cell_id)
# IMPORTANT: point id and cell id of the input skeleton for Vertex like cell must be equal
#
class GraphMCF(object):

    #### Constructor Area

    # skel: DisPerSe skeleton
    # manifolds: DisPerSe manifolds
    # density: image density map
    # table_props: if the table of props has already been created outside
    # def __init__(self, skel, manifolds, density, table_props=None):
    def __init__(self, skel, manifolds, density):
        self.__skel = skel
        self.__manifolds = manifolds
        self.__density = density
        nverts = self.__skel.GetVerts().GetNumberOfCells()
        ncells = self.__skel.GetNumberOfCells()
        self.__vertices = np.empty(shape=nverts, dtype=VertexMCF)
        self.__edges = np.empty(shape=nverts, dtype=EdgeMCF)
        self.__props_info = TablePropsTest(ncells)
        self.__resolution = 1
        self.__graph_gt = None
        # For pickling VTK objects
        self.__skel_fname = None
        # Only for topological simplification
        self.__pair_prop_key = STR_FIELD_VALUE
        self.__v_lst = None
        self.__per_lst = None

    #### Get/Set

    # This function allows to modify property for topological simplification (default STR_FIELD_VALUE)
    def set_pair_prop(self, prop_key=STR_FIELD_VALUE):
        self.__pair_prop_key = prop_key

    # resolution: nm per voxel width
    def set_resolution(self, resolution):
        self.__resolution = resolution

    # Get a lighted copy with the minimum topological information of the GraphMCF
    def get_light_copy(self):
        hold_skel = vtk.vtkPolyData()
        hold_skel.DeepCopy(self.get_skel())
        graph_l = GraphMCF(hold_skel, None, None)
        graph_l.__vertices = copy.deepcopy(self.__vertices)
        graph_l.__edges = copy.deepcopy(self.__edges)
        graph_l.__props_info = copy.deepcopy(self.__props_info)
        graph_l.__resolution = copy.deepcopy(self.__resolution)
        return graph_l

    def get_resolution(self):
        return self.__resolution

    def get_skel(self):
        return self.__skel

    def get_density(self):
        return self.__density

    def get_vertex(self, id):
        return self.__vertices[id]

    # Return two lists one with the vertices directly connected to the one with id, the second
    # list contains the edges which make the connections in the same order
    # Self-edges are not inserted
    # If no neighbours then it returns the lists empty
    def get_vertex_neighbours(self, id):
        v = self.__vertices[id]
        neighs = list()
        edges = list()
        for a in v.get_arcs():
            e = self.get_edge(a.get_sad_id())
            if e is not None:
                e_id = e.get_source_id()
                if e_id == id:
                    e_id = e.get_target_id()
                if e_id != id:
                    neighs.append(self.__vertices[e_id])
                    edges.append(e)
        return neighs, edges

    def get_vertex_coords(self, v):
        return self.__skel.GetPoint(v.get_id())

    # From an iterable of vertex ids returns a numpy array (n,3) with their coordinates
    # vids: list with vertices ids if vids is None (default) then the coordines of all vertices are provided
    def get_vertices_coords(self, v_ids=None):
        if v_ids is None:
            vertices = self.get_vertices_list()
            coords = np.zeros(shape=(len(vertices), 3), dtype=np.float32)
            for i, vertex in enumerate(vertices):
                coords[i, :] = self.__skel.GetPoint(vertex.get_id())
            return coords
        else:
            coords = np.zeros(shape=(len(v_ids), 3), dtype=np.float32)
            for i, v_id in enumerate(v_ids):
                coords[i, :] = self.__skel.GetPoint(v_id)
            return coords

    # Saddle point coordinates
    def get_edge_coords(self, e):
        return self.__skel.GetPoint(e.get_id())

    def get_edge(self, id):
        return self.__edges[id]

    def get_edge_arcs(self, e):
        arc_s = None
        arc_t = None
        s = self.get_vertex(e.get_source_id())
        t = self.get_vertex(e.get_target_id())
        e_id = e.get_id()
        # Finding the arcs which contain the connector
        for a in s.get_arcs():
            if a.get_sad_id() == e_id:
                arc_s = a
                break
        for a in t.get_arcs():
            if a.get_sad_id() == e_id:
                arc_t = a
                break
        return arc_s, arc_t

    # no_repeat: if True (default False) consecutive repeated points are ereased
    def get_edge_ids(self, e, no_repeat=False):
        arc_s = None
        arc_t = None
        s = self.get_vertex(e.get_source_id())
        t = self.get_vertex(e.get_target_id())
        e_id = e.get_id()
        # Finding the arcs which contain the connector
        for a in s.get_arcs():
            if a.get_sad_id() == e_id:
                arc_s = a
                break
        for a in t.get_arcs():
            if a.get_sad_id() == e_id:
                arc_t = a
                break
        if no_repeat:
            return set(arc_s.get_ids() + arc_t.get_ids()[::-1])
        else:
            return arc_s.get_ids() + arc_t.get_ids()[::-1]

    def get_edge_arcs_coords(self, e, no_repeat=False):
        ids = self.get_edge_ids(e, no_repeat)
        coords = np.zeros(shape=(len(ids), 3), dtype=np.float32)
        for i, idx in enumerate(ids):
            coords[i, :] = self.__skel.GetPoint(idx)
        return coords

    # Returns an array with the filled value through and edge along its arcs
    # e: input edge
    # no_repeat: if True (defaul False) consecutive repeated points are ereased
    # f_len: if not None (default) it force the function to return arrays with this length, if original length was
    #        bigger it is equally sampled, otherwise they are set to f_mx.
    # f_mx: default value for filed (defualt 1.), only applicable if f_len is not None
    def get_edge_skel_field(self, e, no_repeat=False, f_len=None, f_mx=1.):
        coords = self.get_edge_arcs_coords(e, no_repeat)
        if f_mx is None:
            vals = np.zeros(shape=len(coords), dtype=np.float32)
            for i, coord in enumerate(coords):
                try:
                    vals[i] = trilin3d(self.__density, coord)
                except IndexError:
                    pass
        else:
            vals = f_mx * np.ones(shape=f_len, dtype=np.float32)
            for i, idx in enumerate(np.linspace(0, len(coords)-1, f_len).astype(np.int)):
                try:
                    vals[i] = trilin3d(self.__density, coords[idx, :])
                except IndexError:
                    pass
        return vals

    # Return an edge from its source and target vertices
    def get_edge_st(self, s, t):
        t_id = t.get_id()
        neighs, edges = self.get_vertex_neighbours(s.get_id())
        for i, n in enumerate(neighs):
            if n.get_id() == t_id:
                return edges[i]
        return None

    def get_manifolds(self):
        return self.__manifolds

    def get_density(self):
        return self.__density

    # Return the length of the arrays which hold ids
    def get_nid(self):
        return len(self.__vertices)

    # Get all VertexMCF objects in a list
    def get_vertices_list(self):
        vertices = list()
        for v in self.__vertices:
            if v is not None:
                vertices.append(v)
        return vertices

    # Get all EdgeMCF objects in a list
    def get_edges_list(self):
        edges = list()
        for e in self.__edges:
            if e is not None:
                edges.append(e)
        return edges

    # Get all ArcMCF objects in a list
    def get_arcs_list(self):
        arcs = list()
        for v in self.get_vertices_list():
            arcs += v.get_arcs()
        return arcs

    # Return a list with the arcs which form the edges
    def get_arc_edges_list(self):

        arcs = list()
        for e in self.get_edges_list():
            v_s = self.__vertices[e.get_source_id()]
            v_t = self.__vertices[e.get_target_id()]
            sad_id = e.get_id()
            for a_s in v_s.get_arcs():
                if a_s.get_sad_id() == sad_id:
                    arcs.append(a_s)
                    for a_t in v_t.get_arcs():
                        if a_t.get_sad_id() == sad_id:
                            arcs.append(a_t)

        return arcs

    # Return the length (in nm) of the arcs which compounds an edge
    def get_edge_length(self, edge):
        # Get arcs
        v_s = self.__vertices[edge.get_source_id()]
        v_t = self.__vertices[edge.get_target_id()]
        sad_id = edge.get_id()
        for a_s in v_s.get_arcs():
            if a_s.get_sad_id() == sad_id:
                for a_t in v_t.get_arcs():
                    if a_t.get_sad_id() == sad_id:
                        l = a_s.get_length(self.__skel) + a_t.get_length(self.__skel)
                        return self.__resolution * l

        error_msg = "Invalid edge architecture."
        raise pexceptions.PySegInputError(expr='get_edge_length (GraphMCF)', msg=error_msg)

    # Get a weighted length on each dimension (similar to Mahalanobis)
    # edge: input edge
    # w_x|y|z: weighting factors
    # total: if True (default False) only source and target points are taken into account
    def get_edge_length_2(self, edge, w_x, w_y, w_z, total=False):

        # Get arcs
        v_s = self.__vertices[edge.get_source_id()]
        v_t = self.__vertices[edge.get_target_id()]
        sad_id = edge.get_id()
        for a_s in v_s.get_arcs():
            if a_s.get_sad_id() == sad_id:
                for a_t in v_t.get_arcs():
                    if a_t.get_sad_id() == sad_id:
                        if total:
                            l = a_s.get_length_2(self.__skel, w_x, w_y, w_z) + \
                                a_t.get_length_2(self.__skel, w_x, w_y, w_z)
                        else:
                            m_s = self.__skel.GetPoint(a_s.get_min_id())
                            m_t = self.__skel.GetPoint(a_t.get_min_id())
                            l = w_x*math.fabs(m_s[0]-m_t[0]) + \
                                w_y*math.fabs(m_s[1]-m_t[1]) + \
                                w_z*math.fabs(m_s[2]-m_t[2])
                        return self.__resolution * l

        error_msg = "Invalid edge architecture."
        raise pexceptions.PySegInputError(expr='get_edge_length (GraphMCF)', msg=error_msg)

    # Return vertices of every subgraph in different lists
    def get_vertex_sg_lists(self):

        # Get subgraphs
        if self.__props_info.is_already(STR_GRAPH_ID) is None:
            self.find_subgraphs()

        # Get maximum graph id
        mx = -1
        vertices = self.get_vertices_list()
        key_id = self.__props_info.is_already(STR_GRAPH_ID)
        data_type = self.__props_info.get_type(index=key_id)
        data_type = disperse_io.TypesConverter().gt_to_numpy(data_type)
        for v in vertices:
            t = self.__props_info.get_prop_entry_fast(key_id, v.get_id(), 1, data_type)
            # t = self.__props_info.get_prop_entry(STR_GRAPH_ID, v.get_id())
            t = t[0] - 1
            if t > mx:
                mx = t

        # Create the lists
        if mx > -1:
            sg_lists = np.empty(shape=mx + 1, dtype=list)
            for i in range(len(sg_lists)):
                sg_lists[i] = list()
            for v in vertices:
                g_id = self.__props_info.get_prop_entry_fast(key_id, v.get_id(), 1, data_type)
                # g_id = self.__props_info.get_prop_entry(STR_GRAPH_ID, v.get_id())
                g_id = g_id[0] - 1
                if g_id != -1:
                    sg_lists[g_id].append(v)
            return sg_lists.tolist()

        else:
            return None

    # Return a graph_tool version of the current graph and if id_arr is True (default False)
    # the an array for indexing gt vertices from morse id is also returned
    # fupdate: if True (default False) the GraphGT is forced to be recomputed
    def get_gt(self, id_arr=False, fupdate=False):

        if (self.__graph_gt is not None) and (not fupdate):
            return self.__graph_gt

        graph = gt.Graph(directed=False)

        # Vertices
        vertices = self.get_vertices_list()
        vertices_gt = np.empty(shape=self.get_nid(), dtype=object)
        for v in vertices:
            vertices_gt[v.get_id()] = graph.add_vertex()

        # Edges
        edges = self.get_edges_list()
        edges_gt = np.empty(shape=self.get_nid(), dtype=object)
        for e in edges:
            edges_gt[e.get_id()] = graph.add_edge(vertices_gt[e.get_source_id()],
                                                  vertices_gt[e.get_target_id()])

        # Getting properties lists
        props_v, props_e = list(), list()
        keys = list()
        cont = 0
        nprops = self.__props_info.get_nprops()
        for i in range(MAX_NPROPS):
            prop_key = self.__props_info.get_key(i)
            if prop_key not in keys:
                p_type = self.__props_info.get_type(i)
                if prop_key == DPSTR_CELL:
                    p_type = 'int'
                n_comp = self.__props_info.get_ncomp(i)
                if n_comp > 1:
                    if p_type == 'uint8_t':
                        p_type = 'vector<uint8_t>'
                    elif p_type == 'short':
                        p_type = 'vector<short>'
                    elif p_type == 'int':
                        p_type = 'vector<int>'
                    elif p_type == 'long':
                        p_type = 'vector<long>'
                    elif p_type == 'float':
                        p_type = 'vector<float>'
                    else:
                        error_msg = "Data type " + p_type + " is not recognized!"
                        raise pexceptions.PySegInputError(expr='get_gt (GraphMCF)', msg=error_msg)
                try:
                    hold_v_prop = graph.new_vertex_property(p_type)
                    hold_e_prop = graph.new_edge_property(p_type)
                except ValueError:
                    print 'WARNING get_gt (GraphMCF): property ' + prop_key + ' could not be added.'
                props_v.append(hold_v_prop)
                props_e.append(hold_e_prop)
                keys.append(prop_key)
                cont += 1
                if cont >= nprops:
                    break

        # Getting properties values
        for key, prop_v, prop_e in zip(keys, props_v, props_e):
            prop_id = self.get_prop_id(key)
            n_comp = self.get_prop_ncomp(key_id=prop_id)
            p_type = disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=prop_id))
            if key == DPSTR_CELL:
                for v in vertices:
                    v_id = v.get_id()
                    v_gt = vertices_gt[v_id]
                    prop_v[v_gt] = v_id
                for e in edges:
                    e_id = e.get_id()
                    e_gt = edges_gt[e_id]
                    prop_e[e_gt] = e_id
            else:
                for v in vertices:
                    v_id = v.get_id()
                    v_gt = vertices_gt[v_id]
                    t = self.get_prop_entry_fast(prop_id, v_id, n_comp, p_type)
                    if len(t) > 1:
                        prop_v[v_gt] = np.asarray(t)
                    else:
                        prop_v[v_gt] = t[0]

                for e in edges:
                    e_id = e.get_id()
                    e_gt = edges_gt[e_id]
                    t = self.get_prop_entry_fast(prop_id, e_id, n_comp, p_type)
                    if len(t) > 1:
                        prop_e[e_gt] = np.asarray(t)
                    else:
                        prop_e[e_gt] = t[0]
            graph.vertex_properties[key] = prop_v
            graph.edge_properties[key] = prop_e

        if id_arr:
            return graph, vertices_gt
        return graph

    # Return property id in the TableProp for fast indexing
    # Return None if the property does not exist
    def get_prop_id(self, key):
        return self.__props_info.is_already(key)

    def get_prop_type(self, key_id=-1, key=None):
        return self.__props_info.get_type(index=key_id, key=key)

    def get_prop_entry_fast(self, key_id, id_entry, n_comp, data_type):
        return self.__props_info.get_prop_entry_fast(key_id, id_entry, n_comp, data_type)

    def get_prop_ncomp(self, key_id=-1, key=None):
        return self.__props_info.get_ncomp(index=key_id, key=key)

    # Return an array of values for a property an from a list of input vertices or edges
    def get_prop_values(self, prop_key, ids):
        prop_id = self.get_prop_id(prop_key)
        n_comp = self.get_prop_ncomp(key_id=prop_id)
        dtype = disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=prop_id))
        numel = len(ids)
        if n_comp > 1:
            array = np.zeros(shape=(numel, n_comp), dtype=dtype)
            for i in range(numel):
                t = self.get_prop_entry_fast(prop_id, ids[i], n_comp, dtype)
                array[i, :] = np.asarray(t)
        else:
            array = np.zeros(shape=numel, dtype=dtype)
            for i in range(numel):
                t = self.get_prop_entry_fast(prop_id, ids[i], n_comp, dtype)
                array[i] = t[0]
        return array

    # av_mode: if True (default False) the properties of arcs will be the properties values of
    # respective vertices
    # edges: if True (default False) only edge arcs are printed
    def get_vtp(self, av_mode=False, edges=False):

        # Initialization
        poly = vtk.vtkPolyData()
        poly.SetPoints(self.__skel.GetPoints())
        arrays = list()
        keys = list()
        cont = 0
        nprops = self.__props_info.get_nprops()
        for i in range(MAX_NPROPS):
            prop_key = self.__props_info.get_key(i)
            if prop_key not in keys:
                array = disperse_io.TypesConverter.gt_to_vtk(self.__props_info.get_type(i))
                array.SetName(self.__props_info.get_key(i))
                array.SetNumberOfComponents(self.__props_info.get_ncomp(i))
                arrays.append(array)
                keys.append(prop_key)
                cont += 1
                if cont >= nprops:
                    break

        # VTK Topology
        # Vertices
        verts = vtk.vtkCellArray()
        for v in self.get_vertices_list():
            verts.InsertNextCell(1)
            verts.InsertCellPoint(v.get_id())
            for array in arrays:
                key_id = self.__props_info.is_already(array.GetName())
                n_comp = array.GetNumberOfComponents()
                data_type = self.__props_info.get_type(index=key_id)
                data_type = disperse_io.TypesConverter().gt_to_numpy(data_type)
                array.InsertNextTuple(self.__props_info.get_prop_entry_fast(key_id, v.get_id(),
                                                                            n_comp, data_type))
                # array.InsertNextTuple(self.__props_info.get_prop_entry(array.GetName(),
                #                                                        v.get_id()))
        # Edges
        for e in self.get_edges_list():
            verts.InsertNextCell(1)
            verts.InsertCellPoint(e.get_id())
            for array in arrays:
                key_id = self.__props_info.is_already(array.GetName())
                n_comp = array.GetNumberOfComponents()
                data_type = self.__props_info.get_type(index=key_id)
                data_type = disperse_io.TypesConverter().gt_to_numpy(data_type)
                array.InsertNextTuple(self.__props_info.get_prop_entry_fast(key_id, e.get_id(),
                                                                            n_comp, data_type))
        # Arcs
        lines = vtk.vtkCellArray()
        if edges:
            for i, a in enumerate(self.get_arc_edges_list()):
                npoints = a.get_npoints()
                lines.InsertNextCell(npoints)
                for j in range(npoints):
                    lines.InsertCellPoint(a.get_point_id(j))
                if av_mode:
                    for array in arrays:
                        key_id = self.__props_info.is_already(array.GetName())
                        n_comp = array.GetNumberOfComponents()
                        data_type = self.__props_info.get_type(index=key_id)
                        data_type = disperse_io.TypesConverter().gt_to_numpy(data_type)
                        array.InsertNextTuple(self.__props_info.get_prop_entry_fast(key_id,
                                                                                    a.get_min_id(),
                                                                                    n_comp,
                                                                                    data_type))
                else:
                    for array in arrays:
                        key_id = self.__props_info.is_already(array.GetName())
                        n_comp = array.GetNumberOfComponents()
                        data_type = self.__props_info.get_type(index=key_id)
                        data_type = disperse_io.TypesConverter().gt_to_numpy(data_type)
                        array.InsertNextTuple(self.__props_info.get_prop_entry_fast(key_id,
                                                                                    a.get_sad_id(),
                                                                                    n_comp,
                                                                                    data_type))
        else:
            for i, a in enumerate(self.get_arcs_list()):
                npoints = a.get_npoints()
                lines.InsertNextCell(npoints)
                for j in range(npoints):
                    lines.InsertCellPoint(a.get_point_id(j))
                if av_mode:
                    for array in arrays:
                        key_id = self.__props_info.is_already(array.GetName())
                        n_comp = array.GetNumberOfComponents()
                        data_type = self.__props_info.get_type(index=key_id)
                        data_type = disperse_io.TypesConverter().gt_to_numpy(data_type)
                        array.InsertNextTuple(self.__props_info.get_prop_entry_fast(key_id,
                                                                                    a.get_min_id(),
                                                                                    n_comp,
                                                                                    data_type))
                else:
                    for array in arrays:
                        key_id = self.__props_info.is_already(array.GetName())
                        n_comp = array.GetNumberOfComponents()
                        data_type = self.__props_info.get_type(index=key_id)
                        data_type = disperse_io.TypesConverter().gt_to_numpy(data_type)
                        array.InsertNextTuple(self.__props_info.get_prop_entry_fast(key_id,
                                                                                    a.get_sad_id(),
                                                                                    n_comp,
                                                                                    data_type))


        # vtkPolyData construction
        poly.SetVerts(verts)
        poly.SetLines(lines)
        for array in arrays:
            poly.GetCellData().AddArray(array)

        return poly

    # Only vertices and edges in mask are stored
    # mask: binary mask where region to stored is tagged with True
    # av_mode: if True (default False) the properties of arcs will be the properties values of
    # respective vertices
    # edges: if True (default False) only edge arcs are printed
    def get_vtp_in_msk(self, mask, av_mode=False, edges=False):

        # Initialization
        poly = vtk.vtkPolyData()
        poly.SetPoints(self.__skel.GetPoints())
        arrays = list()
        keys = list()
        cont = 0
        nprops = self.__props_info.get_nprops()
        for i in range(MAX_NPROPS):
            prop_key = self.__props_info.get_key(i)
            if prop_key not in keys:
                array = disperse_io.TypesConverter.gt_to_vtk(self.__props_info.get_type(i))
                array.SetName(self.__props_info.get_key(i))
                array.SetNumberOfComponents(self.__props_info.get_ncomp(i))
                arrays.append(array)
                keys.append(prop_key)
                cont += 1
                if cont >= nprops:
                    break

        # Filter vertices and edges to stores
        seg_vertices, seg_edges, seg_arcs = list(), list(), list()
        if edges:
            for v in self.get_vertices_list():
                x, y, z = self.get_vertex_coords(v)
                try:
                    if mask[int(np.round(x)), int(np.round(y)), int(np.round(z))]:
                        seg_vertices.append(v)
                except IndexError:
                    pass
            for e in self.get_edges_list():
                x, y, z = self.get_edge_coords(e)
                try:
                    if mask[int(np.round(x)), int(np.round(y)), int(np.round(z))]:
                        seg_edges.append(e)
                        seg_arcs += self.get_edge_arcs(e)
                except IndexError:
                    pass
        else:
            for v in self.get_vertices_list():
                x, y, z = self.get_vertex_coords(v)
                try:
                    if mask[int(np.round(x)), int(np.round(y)), int(np.round(z))]:
                        seg_vertices.append(v)
                        seg_arcs += v.get_arcs()
                except IndexError:
                    pass
            for e in self.get_edges_list():
                x, y, z = self.get_edge_coords(e)
                try:
                    if mask[int(np.round(x)), int(np.round(y)), int(np.round(z))]:
                        seg_edges.append(e)
                except IndexError:
                    pass

        # VTK Topology
        # Vertices
        verts = vtk.vtkCellArray()
        for v in seg_vertices():
            verts.InsertNextCell(1)
            verts.InsertCellPoint(v.get_id())
            for array in arrays:
                key_id = self.__props_info.is_already(array.GetName())
                n_comp = array.GetNumberOfComponents()
                data_type = self.__props_info.get_type(index=key_id)
                data_type = disperse_io.TypesConverter().gt_to_numpy(data_type)
                array.InsertNextTuple(self.__props_info.get_prop_entry_fast(key_id, v.get_id(),
                                                                            n_comp, data_type))
                # array.InsertNextTuple(self.__props_info.get_prop_entry(array.GetName(),
                #                                                        v.get_id()))
        # Edges
        for e in seg_edges:
            verts.InsertNextCell(1)
            verts.InsertCellPoint(e.get_id())
            for array in arrays:
                key_id = self.__props_info.is_already(array.GetName())
                n_comp = array.GetNumberOfComponents()
                data_type = self.__props_info.get_type(index=key_id)
                data_type = disperse_io.TypesConverter().gt_to_numpy(data_type)
                array.InsertNextTuple(self.__props_info.get_prop_entry_fast(key_id, e.get_id(),
                                                                            n_comp, data_type))
        # Arcs
        lines = vtk.vtkCellArray()
        if edges:
            for i, a in enumerate(seg_arcs):
                npoints = a.get_npoints()
                lines.InsertNextCell(npoints)
                for j in range(npoints):
                    lines.InsertCellPoint(a.get_point_id(j))
                if av_mode:
                    for array in arrays:
                        key_id = self.__props_info.is_already(array.GetName())
                        n_comp = array.GetNumberOfComponents()
                        data_type = self.__props_info.get_type(index=key_id)
                        data_type = disperse_io.TypesConverter().gt_to_numpy(data_type)
                        array.InsertNextTuple(self.__props_info.get_prop_entry_fast(key_id,
                                                                                    a.get_min_id(),
                                                                                    n_comp,
                                                                                    data_type))
                else:
                    for array in arrays:
                        key_id = self.__props_info.is_already(array.GetName())
                        n_comp = array.GetNumberOfComponents()
                        data_type = self.__props_info.get_type(index=key_id)
                        data_type = disperse_io.TypesConverter().gt_to_numpy(data_type)
                        array.InsertNextTuple(self.__props_info.get_prop_entry_fast(key_id,
                                                                                    a.get_sad_id(),
                                                                                    n_comp,
                                                                                    data_type))
        else:
            for i, a in enumerate(seg_arcs):
                npoints = a.get_npoints()
                lines.InsertNextCell(npoints)
                for j in range(npoints):
                    lines.InsertCellPoint(a.get_point_id(j))
                if av_mode:
                    for array in arrays:
                        key_id = self.__props_info.is_already(array.GetName())
                        n_comp = array.GetNumberOfComponents()
                        data_type = self.__props_info.get_type(index=key_id)
                        data_type = disperse_io.TypesConverter().gt_to_numpy(data_type)
                        array.InsertNextTuple(self.__props_info.get_prop_entry_fast(key_id,
                                                                                    a.get_min_id(),
                                                                                    n_comp,
                                                                                    data_type))
                else:
                    for array in arrays:
                        key_id = self.__props_info.is_already(array.GetName())
                        n_comp = array.GetNumberOfComponents()
                        data_type = self.__props_info.get_type(index=key_id)
                        data_type = disperse_io.TypesConverter().gt_to_numpy(data_type)
                        array.InsertNextTuple(self.__props_info.get_prop_entry_fast(key_id,
                                                                                    a.get_sad_id(),
                                                                                    n_comp,
                                                                                    data_type))


        # vtkPolyData construction
        poly.SetVerts(verts)
        poly.SetLines(lines)
        for array in arrays:
            poly.GetCellData().AddArray(array)

        return poly

    # Only selected vertices and their edges are stored
    # v_ids: list of vertices index
    # av_mode: if True (default False) the properties of arcs will be the properties values of
    # respective vertices
    # edges: if True (default False) only edge arcs are printed
    def get_vtp_ids(self, v_ids, av_mode=False, edges=False):

        # Initialization
        poly = vtk.vtkPolyData()
        poly.SetPoints(self.__skel.GetPoints())
        arrays = list()
        keys = list()
        cont = 0
        nprops = self.__props_info.get_nprops()
        for i in range(MAX_NPROPS):
            prop_key = self.__props_info.get_key(i)
            if prop_key not in keys:
                array = disperse_io.TypesConverter.gt_to_vtk(self.__props_info.get_type(i))
                array.SetName(self.__props_info.get_key(i))
                array.SetNumberOfComponents(self.__props_info.get_ncomp(i))
                arrays.append(array)
                keys.append(prop_key)
                cont += 1
                if cont >= nprops:
                    break

        # Filter vertices and edges to stores
        seg_vertices, seg_edges, seg_arcs = list(), list(), list()
        lut_v = np.zeros(shape=self.get_nid(), dtype=np.bool)
        if edges:
            for v_id in v_ids:
                seg_vertices.append(self.get_vertex(v_id))
                lut_v[v_id] = True
            for e in self.get_edges_list():
                s_id, t_id = e.get_source_id(), e.get_target_id()
                if lut_v[s_id] and lut_v[t_id]:
                    seg_edges.append(e)
                    seg_arcs += self.get_edge_arcs(e)
        else:
            for v_id in v_ids:
                v = self.get_vertex(v_id)
                seg_vertices.append(v)
                seg_arcs += v.get_arcs()
                lut_v[v_id] = True
            for e in self.get_edges_list():
                s_id, t_id = e.get_source_id(), e.get_target_id()
                if lut_v[s_id] and lut_v[t_id]:
                    seg_edges.append(e)

        # VTK Topology
        # Vertices
        verts = vtk.vtkCellArray()
        for v in seg_vertices:
            verts.InsertNextCell(1)
            verts.InsertCellPoint(v.get_id())
            for array in arrays:
                key_id = self.__props_info.is_already(array.GetName())
                n_comp = array.GetNumberOfComponents()
                data_type = self.__props_info.get_type(index=key_id)
                data_type = disperse_io.TypesConverter().gt_to_numpy(data_type)
                array.InsertNextTuple(self.__props_info.get_prop_entry_fast(key_id, v.get_id(),
                                                                            n_comp, data_type))
                # array.InsertNextTuple(self.__props_info.get_prop_entry(array.GetName(),
                #                                                        v.get_id()))
        # Edges
        for e in seg_edges:
            verts.InsertNextCell(1)
            verts.InsertCellPoint(e.get_id())
            for array in arrays:
                key_id = self.__props_info.is_already(array.GetName())
                n_comp = array.GetNumberOfComponents()
                data_type = self.__props_info.get_type(index=key_id)
                data_type = disperse_io.TypesConverter().gt_to_numpy(data_type)
                array.InsertNextTuple(self.__props_info.get_prop_entry_fast(key_id, e.get_id(),
                                                                            n_comp, data_type))
        # Arcs
        lines = vtk.vtkCellArray()
        if edges:
            for i, a in enumerate(seg_arcs):
                npoints = a.get_npoints()
                lines.InsertNextCell(npoints)
                for j in range(npoints):
                    lines.InsertCellPoint(a.get_point_id(j))
                if av_mode:
                    for array in arrays:
                        key_id = self.__props_info.is_already(array.GetName())
                        n_comp = array.GetNumberOfComponents()
                        data_type = self.__props_info.get_type(index=key_id)
                        data_type = disperse_io.TypesConverter().gt_to_numpy(data_type)
                        array.InsertNextTuple(self.__props_info.get_prop_entry_fast(key_id,
                                                                                    a.get_min_id(),
                                                                                    n_comp,
                                                                                    data_type))
                else:
                    for array in arrays:
                        key_id = self.__props_info.is_already(array.GetName())
                        n_comp = array.GetNumberOfComponents()
                        data_type = self.__props_info.get_type(index=key_id)
                        data_type = disperse_io.TypesConverter().gt_to_numpy(data_type)
                        array.InsertNextTuple(self.__props_info.get_prop_entry_fast(key_id,
                                                                                    a.get_sad_id(),
                                                                                    n_comp,
                                                                                    data_type))
        else:
            for i, a in enumerate(seg_arcs):
                npoints = a.get_npoints()
                lines.InsertNextCell(npoints)
                for j in range(npoints):
                    lines.InsertCellPoint(a.get_point_id(j))
                if av_mode:
                    for array in arrays:
                        key_id = self.__props_info.is_already(array.GetName())
                        n_comp = array.GetNumberOfComponents()
                        data_type = self.__props_info.get_type(index=key_id)
                        data_type = disperse_io.TypesConverter().gt_to_numpy(data_type)
                        array.InsertNextTuple(self.__props_info.get_prop_entry_fast(key_id,
                                                                                    a.get_min_id(),
                                                                                    n_comp,
                                                                                    data_type))
                else:
                    for array in arrays:
                        key_id = self.__props_info.is_already(array.GetName())
                        n_comp = array.GetNumberOfComponents()
                        data_type = self.__props_info.get_type(index=key_id)
                        data_type = disperse_io.TypesConverter().gt_to_numpy(data_type)
                        array.InsertNextTuple(self.__props_info.get_prop_entry_fast(key_id,
                                                                                    a.get_sad_id(),
                                                                                    n_comp,
                                                                                    data_type))


        # vtkPolyData construction
        poly.SetVerts(verts)
        poly.SetLines(lines)
        for array in arrays:
            poly.GetCellData().AddArray(array)

        return poly

    def get_prop_max(self, key):
        return self.__props_info.get_prop_max(key)

    def get_prop_min(self, key):
        return self.__props_info.get_prop_min(key)

    # Generates an .vtp file with nodes as points and edges as rect lines
    # nodes: if True (default) nodes are stored as points
    # edges: if True (default) edges are stored as lines
    def get_scheme_vtp(self, nodes=True, edges=True):

        # Initialization
        poly = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        arrays = list()
        keys = list()
        cont = 0
        nprops = self.__props_info.get_nprops()
        for i in range(MAX_NPROPS
                       ):
            prop_key = self.__props_info.get_key(i)
            if prop_key not in keys:
                array = disperse_io.TypesConverter.gt_to_vtk(self.__props_info.get_type(i))
                array.SetName(self.__props_info.get_key(i))
                array.SetNumberOfComponents(self.__props_info.get_ncomp(i))
                arrays.append(array)
                keys.append(prop_key)
                cont += 1
                if cont >= nprops:
                    break

        # Geometry
        vertices = self.get_vertices_list()
        lut = np.zeros(shape=self.get_nid(), dtype=np.int)
        for i, v in enumerate(vertices):
            x, y, z = v.get_coordinates(self.__skel)
            points.InsertPoint(i, x, y, z)
            lut[v.get_id()] = i

        # Topology
        # Nodes
        verts = vtk.vtkCellArray()
        if nodes:
            for v in vertices:
                verts.InsertNextCell(1)
                verts.InsertCellPoint(lut[v.get_id()])
                for array in arrays:
                    key_id = self.__props_info.is_already(array.GetName())
                    n_comp = array.GetNumberOfComponents()
                    data_type = self.__props_info.get_type(index=key_id)
                    data_type = disperse_io.TypesConverter().gt_to_numpy(data_type)
                    array.InsertNextTuple(self.__props_info.get_prop_entry_fast(key_id, v.get_id(),
                                                                                n_comp, data_type))
        # Edges
        if edges:
            lines = vtk.vtkCellArray()
            for e in self.get_edges_list():
                lines.InsertNextCell(2)
                lines.InsertCellPoint(lut[e.get_source_id()])
                lines.InsertCellPoint(lut[e.get_target_id()])
                for array in arrays:
                    key_id = self.__props_info.is_already(array.GetName())
                    n_comp = array.GetNumberOfComponents()
                    data_type = self.__props_info.get_type(index=key_id)
                    data_type = disperse_io.TypesConverter().gt_to_numpy(data_type)
                    array.InsertNextTuple(self.__props_info.get_prop_entry_fast(key_id,
                                                                                e.get_id(),
                                                                                n_comp,
                                                                                data_type))

        # vtkPolyData construction
        poly.SetPoints(points)
        if nodes:
            poly.SetVerts(verts)
        if edges:
            poly.SetLines(lines)
        for array in arrays:
            poly.GetCellData().AddArray(array)

        return poly

    # Generates an .vtp file with the skeleton of the graph where graph skel points properties
    # are imported
    # mode: if 'node' (default) only nodes are stored, otherwise arcs are stored
    def get_skel_vtp(self, mode='node'):

        # Intialization
        poly = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        arrays_in = list()
        arrays_out = list()
        for i in range(self.__skel.GetPointData().GetNumberOfArrays()):
            array = self.__skel.GetPointData().GetArray(i)
            data_type = disperse_io.TypesConverter().vtk_to_numpy(array)
            hold = disperse_io.TypesConverter().numpy_to_vtk(data_type)
            hold.SetName(array.GetName())
            hold.SetNumberOfComponents(array.GetNumberOfComponents())
            arrays_in.append(array)
            arrays_out.append(hold)

        # Generating geometry and topology
        lut = np.ones(shape=self.__skel.GetNumberOfPoints(), dtype=np.bool)
        if mode == 'node':
            # Geometry and topology
            vertices = self.get_vertices_list()
            verts = vtk.vtkCellArray()
            for i, v in enumerate(vertices):
                v_id = v.get_id()
                if lut[v_id]:
                    x, y, z = v.get_coordinates(self.__skel)
                    points.InsertNextPoint(x, y, z)
                    verts.InsertNextCell(1)
                    verts.InsertCellPoint(i)
                    for j in range(len(arrays_in)):
                        arrays_out[j].InsertNextTuple(arrays_in[j].GetTuple(v.get_id()))
                    lut[v_id] = False
        # Storing the arcs as lines
        else:
            # Geometry and topology
            arcs = self.get_arcs_list()
            verts = vtk.vtkCellArray()
            point_id = 0
            for i, a in enumerate(arcs):
                for j in range(a.get_npoints()):
                    a_id = a.get_point_id(j)
                    if lut[a_id]:
                        x, y, z = self.__skel.GetPoint(a_id)
                        points.InsertNextPoint(x, y, z)
                        verts.InsertNextCell(1)
                        verts.InsertCellPoint(point_id)
                        for k in range(len(arrays_in)):
                            arrays_out[k].InsertNextTuple(arrays_in[k].GetTuple(a_id))
                        point_id += 1
                        lut[a_id] = False

        # Building the vtp object
        poly.SetPoints(points)
        poly.SetVerts(verts)
        for array in arrays_out:
            poly.GetPointData().AddArray(array)

        return poly

    # Returns the arc skeleton with the values of the scalar_field (numpy array) as point data
    def get_sfield_vtp(self, scalar_field, mode='node'):

        # Intialization
        poly = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        array = disperse_io.TypesConverter().numpy_to_vtk(np.float64)
        array.SetName('scalar_field')
        array.SetNumberOfComponents(1)

        # Generating geometry and topology
        lut = np.ones(shape=self.__skel.GetNumberOfPoints(), dtype=np.bool)
        # Geometry and topology
        arcs = self.get_arcs_list()
        verts = vtk.vtkCellArray()
        point_id = 0
        if mode == 'node':
            for v in self.get_vertices_list():
                x, y, z = self.__skel.GetPoint(v.get_id())
                points.InsertNextPoint(x, y, z)
                verts.InsertNextCell(1)
                verts.InsertCellPoint(point_id)
                t = (scalar_field[int(round(x)), int(round(y)), int(round(z))],)
                array.InsertNextTuple(t)
                point_id += 1
        else:
            for i, a in enumerate(arcs):
                for j in range(a.get_npoints()):
                    a_id = a.get_point_id(j)
                    if lut[a_id]:
                        x, y, z = self.__skel.GetPoint(a_id)
                        points.InsertNextPoint(x, y, z)
                        verts.InsertNextCell(1)
                        verts.InsertCellPoint(point_id)
                        t = (scalar_field[int(round(x)), int(round(y)), int(round(z))],)
                        array.InsertNextTuple(t)
                        point_id += 1
                        lut[a_id] = False

        # Building the vtp object
        poly.SetPoints(points)
        poly.SetVerts(verts)
        poly.GetPointData().AddArray(array)

        return poly

    # Return vertices DPSTR_CELL ids from a segmented region
    # prop: property key for segmentation
    # th: threshold
    # op: operator
    def get_th_vids(self, prop, th, op):

        ids = list()
        prop_ids_key = self.get_prop_id(DPSTR_CELL)
        key_id = self.__props_info.is_already(prop)
        if key_id is None:
            error_msg = 'The graph does not include %s property.' % prop
            raise pexceptions.PySegInputError(expr='theshold_vertices (GraphMCF)', msg=error_msg)
        n_comp = self.__props_info.get_ncomp(index=key_id)
        data_type = self.__props_info.get_type(index=key_id)
        data_type = disperse_io.TypesConverter().gt_to_numpy(data_type)

        for v in self.get_vertices_list():
            t = self.__props_info.get_prop_entry_fast(key_id, v.get_id(), n_comp, data_type)
            t = sum(t) / len(t)
            if op(t, th):
                idx = self.__props_info.get_prop_entry_fast(prop_ids_key, v.get_id(), 1, np.int)
                ids.append(idx)

        return ids

    # Cloud vertices points coordinates in a membrane slice
    # prop_dst: property key for measuring the distance to a membrane
    # slice_samp: range [low, high] in nm with the slice values (penetration tail)
    # Return: booth an array with points coordinates and, if asked, their cardinality
    def get_cloud_points_slice(self, prop_dst, slice_samp):

        # Initialisation
        key_id = self.get_prop_id(prop_dst)
        data_type = disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=key_id))

        # Find vertices within slice
        lut_ver = np.zeros(shape=self.get_nid(), dtype=np.int)
        v_ids = list()
        for v in self.get_vertices_list():
            v_id = v.get_id()
            dst = self.get_prop_entry_fast(key_id, v_id, 1, data_type)[0]
            if (dst >= slice_samp[0]) and (dst <= slice_samp[1]):
                if lut_ver[v_id] == 0:
                    v_ids.append(v_id)
                lut_ver[v_id] += 1

        # Finding coordinates
        coords = np.zeros(shape=(len(v_ids), 3), dtype=np.float)
        for i, v_id in enumerate(v_ids):
            coords[i, :] = self.__skel.GetPoint(v_id)
        return coords

    # Vertex clustering according connectivity within a slice
    # prop_dst: property key for measuring the distance to a membrane
    # slice_samp: range [low, high] in nm with the slice values (penetration tail)
    # Return: booth an array with points coordinates and cluster id
    def get_cloud_clst_slice(self, prop_dst, slice_samp):

        # Initialisation
        key_id = self.get_prop_id(prop_dst)
        data_type = disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=key_id))

        # Find tail vertices within slice
        lut_ver = (-1) * np.ones(shape=self.get_nid(), dtype=np.int)
        cont = 0
        v_ids = list()
        for v in self.get_vertices_list():
            v_id = v.get_id()
            dst = self.get_prop_entry_fast(key_id, v_id, 1, data_type)[0]
            if (dst >= slice_samp[0]) and (dst <= slice_samp[1]):
                if lut_ver[v_id] == -1:
                    lut_ver[v_id] = cont
                    v_ids.append(v_id)
                    cont += 1

        # Find edges within slice
        e_ids = list()
        for edge in self.get_edges_list():
            s_id = lut_ver[edge.get_source_id()]
            t_id = lut_ver[edge.get_target_id()]
            if (s_id > 0) and (t_id > 0):
                e_ids.append([s_id, t_id])

        # graph_tool building
        graph = gt.Graph(directed=False)
        vertices_gt = np.empty(shape=len(v_ids), dtype=object)
        for i in range(len(v_ids)):
            vertices_gt[i] = graph.add_vertex()
        for e_id in e_ids:
            graph.add_edge(vertices_gt[e_id[0]], vertices_gt[e_id[1]])

        # Subgraphs visitor initialization
        sgraph_id = graph.new_vertex_property("int")
        visitor = SubGraphVisitor(sgraph_id)

        # Find subgraphs
        coords = np.zeros(shape=(vertices_gt.shape[0], 3), dtype=np.float)
        for i, v in enumerate(vertices_gt):
            if sgraph_id[v] == 0:
                gt.dfs_search(graph, v, visitor)
                visitor.update_sgraphs_id()
            coords[i, :] = self.__skel.GetPoint(v_ids[i])

        return coords, np.asarray(sgraph_id.get_array(), dtype=np.int)

    # Find contact points in a segmentation border
    # seg: tomogram with the segmentation
    # lbl1: label for region 1
    # lbl2: label for region 2 (different)
    # Returns: coordinates contact points
    def get_cont_seg(self, seg, lbl1, lbl2):

        # Initialization
        edges_list = self.get_edges_list()
        coords = list()
        lut_ids = np.ones(shape=self.__skel.GetNumberOfPoints(), dtype=np.bool)

        # Loop for finding the edge which contains a connector point
        x_E, y_E, z_E = seg.shape
        for e in edges_list:
            s = self.get_vertex(e.get_source_id())
            t = self.get_vertex(e.get_target_id())
            x_s, y_s, z_s = self.get_vertex_coords(s)
            x_s, y_s, z_s = int(round(x_s)), int(round(y_s)), int(round(z_s))
            x_t, y_t, z_t = self.get_vertex_coords(t)
            x_t, y_t, z_t = int(round(x_t)), int(round(y_t)), int(round(z_t))
            if (x_s < x_E) and (y_s < y_E) and (z_s < z_E) and \
                    (x_t < x_E) and (y_t < y_E) and (z_t < z_E):
                s_lbl = seg[x_s, y_s, z_s]
                t_lbl = seg[x_t, y_t, z_t]
                # Check regions border edge
                if ((s_lbl == lbl1) and (t_lbl == lbl2)) or \
                        ((s_lbl == lbl2) and (t_lbl == lbl1)):
                    e_id = e.get_id()
                    # Finding the arcs which contain the connector
                    for a in s.get_arcs():
                        if a.get_sad_id() == e_id:
                            arc_s = a
                            break
                    for a in t.get_arcs():
                        if a.get_sad_id() == e_id:
                            arc_t = a
                            break
                    # Find the connector starting from minimum lbl1
                    lbl = s_lbl
                    if s_lbl == lbl1:
                        hold = arc_s
                        arc_s = arc_t
                        arc_t = hold
                        lbl = t_lbl
                    found = False
                    hold_p = arc_s.get_point_id(0)
                    for i in range(1, arc_s.get_npoints()):
                        curr_p = arc_s.get_point_id(i)
                        x_p, y_p, z_p = self.__skel.GetPoint(curr_p)
                        x_p_r, y_p_r, z_p_r = int(round(x_p)), int(round(y_p)), \
                                                int(round(z_p))
                        if seg[x_p_r, y_p_r, z_p_r] == lbl1:
                            p = np.asarray(self.__skel.GetPoint(hold_p))
                            if lut_ids[hold_p]:
                                coords.append(p)
                                lut_ids[hold_p] = False
                            found = True
                            break
                        hold_p = arc_s.get_point_id(i)
                    if not found:
                        for i in range(1, arc_t.get_npoints()):
                            hold_p = arc_t.get_point_id(i)
                            x_p, y_p, z_p = self.__skel.GetPoint(hold_p)
                            x_p_r, y_p_r, z_p_r = int(round(x_p)), int(round(y_p)), \
                                                    int(round(z_p))
                            if seg[x_p_r, y_p_r, z_p_r] == lbl:
                                p = np.asarray(self.__skel.GetPoint(hold_p))
                                if lut_ids[hold_p]:
                                    coords.append(p)
                                    lut_ids[hold_p] = False
                                found = True
                                break
                    if not found:
                        error_msg = 'Unexpected event.'
                        print 'WARNING (GraphMCF:get_cont_seg) :' + error_msg

        return np.asarray(coords, dtype=np.float)

    def set_prop(self, key, values):
        self.__props_info.set_prop(key, values)

    # Invert the values a property
    def invert_prop(self, key_old, key_new):
        self.__props_info.invert_prop(key_old, key_new)

    # Set the property value of an entry
    def set_prop_entry(self, key, value, id_entry):
        self.__props_info.set_prop_entry(key, value, id_entry)

    # Set the property value of an entry faster than set_prop_entry()
    def set_prop_entry_fast(self, key_id, value, id_entry, n_comp):
        self.__props_info.set_prop_entry_fast(key_id, value, id_entry, n_comp)

    #### External functionality

    def insert_vertex(self, vertex):
        self.__vertices[vertex.get_id()] = vertex

    def remove_vertex(self, vertex):
        v_id = vertex.get_id()
        self.__vertices[v_id] = None
        # Remove its properties from table
        self.__props_info.remove_entry(v_id)
        # Remove the edges of this vertex and their properties
        for a in vertex.get_arcs():
            e_id = a.get_sad_id()
            self.remove_edge(self.__edges[e_id])

    # Input list with vertices (or their ids)
    def remove_vertices_list(self, vertices):
        for v in vertices:
            try:
                self.remove_vertex(v)
            except AttributeError:
                self.remove_vertex(self.get_vertex(v))

    def insert_edge(self, edge):
        self.__edges[edge.get_id()] = edge

    def remove_edge(self, edge):
        if edge is None:
            return
        else:
            e_id = edge.get_id()
            self.__edges[e_id] = None
            self.__props_info.remove_entry(e_id)

    def build_vertex_geometry(self):

        vertices = self.get_vertices_list()
        # mx = int(np.max(self.__manifolds)) + 1
        mx = self.get_nid()
        lcoords = np.empty(shape=mx, dtype=list)
        ldensities = np.empty(shape=mx, dtype=list)
        for i in range(mx):
            lcoords[i] = list()
            ldensities[i] = list()

        # Create lists of coordinates and densities
        for x in range(self.__manifolds.shape[0]):
            for y in range(self.__manifolds.shape[1]):
                for z in range(self.__manifolds.shape[2]):
                    id = int(self.__manifolds[x, y, z])
                    if (id >= 0) and (id < mx):
                        try:
                            ldensities[id].append(self.__density[x, y, z])
                        except IndexError:
                            continue
                        lcoords[id].append((x, y, z))

        # Creates vertices geometry
        key_tot_id = self.__props_info.add_prop(STR_TOT_VDEN, 'float', 1, def_val=-1)
        key_avg_id = self.__props_info.add_prop(STR_AVG_VDEN, 'float', 1, def_val=-1)
        for i, v in enumerate(vertices):
            v_id = v.get_id()
            # print v_id, len(lcoords), len(ldensities)
            if v_id >= len(lcoords):
                print 'Jol'
            if (len(lcoords[v_id]) == 0) or (len(ldensities[v_id]) == 0):
                # Vertices without geometry are removed
                self.remove_vertex(self.get_vertex(v_id))
            else:
                geom = geometry.GeometryMCF(np.asarray(lcoords[v_id]),
                                            np.asarray(ldensities[v_id]))
                v.add_geometry(geom)
                v_id = v.get_id()
                # self.__props_info.set_prop_entry_fast(key_tot_id, (geom.get_total_density(),),
                #                                       v_id, 1)
                self.__props_info.set_prop_entry_fast(key_tot_id, (geom.get_total_density_inv(),),
                                                      v_id, 1)
                self.__props_info.set_prop_entry_fast(key_avg_id, (geom.get_avg_density(),),
                                                      v_id, 1)



    # basic_props: if True (default False) it only imports a few specific properties from
    #             the skeleton
    def build_from_skel(self, basic_props=False):

        # Initialization
        critical_index = None
        for i in range(self.__skel.GetPointData().GetNumberOfArrays()):
            if self.__skel.GetPointData().GetArrayName(i) == STR_CRITICAL_INDEX:
                critical_index = self.__skel.GetPointData().GetArray(i)
                break
        if critical_index is None:
            error_msg = "Input skeleton has no '%s' property." % STR_CRITICAL_INDEX
            raise pexceptions.PySegInputError(expr='build_from_skel (GraphMCF)', msg=error_msg)
        verts = self.__skel.GetVerts()
        nverts = verts.GetNumberOfCells()
        lines = self.__skel.GetLines()

        # Insert the vertices
        line_id = 0
        for i in range(nverts):
            pts = vtk.vtkIdList()
            verts.GetCell(line_id, pts)
            point_id = pts.GetId(0)
            if critical_index.GetTuple1(point_id) == DPID_CRITICAL_MIN:
                vertex = VertexMCF(i)
                self.insert_vertex(vertex)
            line_id = line_id + pts.GetNumberOfIds() + 1

        # Insert the arcs
        line_id = 0
        sa_lut = (-1) * np.ones(shape=(self.__skel.GetNumberOfPoints(), 2), dtype=np.int)
        for i in range(lines.GetNumberOfCells()):
            pts = vtk.vtkIdList()
            lines.GetCell(line_id, pts)
            line = np.zeros(shape=pts.GetNumberOfIds(), dtype=np.int)
            for j in range(len(line)):
                line[j] = pts.GetId(j)
            if critical_index.GetTuple1(line[0]) == DPID_CRITICAL_SAD:
                line[::-1] = line
            self.__vertices[line[0]].add_arc(ArcMCF(nverts + i, list(line)))
            sad_id = line[-1]
            if sa_lut[sad_id][0] == -1:
                sa_lut[sad_id][0] = line[0]
            else:
                sa_lut[sad_id][1] = line[0]
            line_id = line_id + pts.GetNumberOfIds() + 1

        # Insert the edges
        line_id = 0
        for i in range(nverts):
            pts = vtk.vtkIdList()
            verts.GetCell(line_id, pts)
            point_id = pts.GetId(0)
            if critical_index.GetTuple1(point_id) == DPID_CRITICAL_SAD:
                v1_id = sa_lut[point_id][0]
                v2_id = sa_lut[point_id][1]
                if (v1_id != -1) and (v2_id != -1):
                    edge = EdgeMCF(i, v1_id, v2_id)
                    self.insert_edge(edge)
            line_id = line_id + pts.GetNumberOfIds() + 1

        # Importing props
        # Point data
        vertices = self.get_vertices_list()
        edges = self.get_edges_list()
        for i in range(self.__skel.GetPointData().GetNumberOfArrays()):
            array = self.__skel.GetPointData().GetArray(i)
            array_name = array.GetName()
            if basic_props and (array_name != STR_FIELD_VALUE):
                continue
            n_comp = array.GetNumberOfComponents()
            self.__props_info.add_prop(array_name,
                                       disperse_io.TypesConverter().vtk_to_gt(array),
                                       n_comp, def_val=0)
            key_id = self.__props_info.is_already(array_name)
            for v in vertices:
                pts = vtk.vtkIdList()
                v_id = v.get_id()
                verts.GetCell(v_id * 2, pts)
                self.__props_info.set_prop_entry_fast(key_id, array.GetTuple(pts.GetId(0)),
                                                      v_id, n_comp)
            for e in edges:
                pts = vtk.vtkIdList()
                e_id = e.get_id()
                verts.GetCell(e_id * 2, pts)
                self.__props_info.set_prop_entry_fast(key_id, array.GetTuple(pts.GetId(0)),
                                                      e_id, n_comp)

        # Cell data
        if basic_props:
            return
        for i in range(self.__skel.GetCellData().GetNumberOfArrays()):
            array = self.__skel.GetCellData().GetArray(i)
            array_name = array.GetName()
            self.__props_info.add_prop(array_name,
                                       disperse_io.TypesConverter().vtk_to_gt(array),
                                       array.GetNumberOfComponents(), def_val=0)
            key_id = self.__props_info.is_already(array_name)
            n_comp = self.__props_info.get_ncomp(index=key_id)
            for v in vertices:
                self.__props_info.set_prop_entry_fast(key_id, array.GetTuple(v.get_id()),
                                                      v.get_id(), n_comp)
            for e in edges:
                self.__props_info.set_prop_entry_fast(key_id, array.GetTuple(e.get_id()),
                                                      e.get_id(), n_comp)

        # Inserting vertex persistence and pairs
        self.__props_info.add_prop(STR_V_PER, 'float', 1, def_val=0)
        key_per_id = self.__props_info.is_already(STR_V_PER)
        key_field_id = self.__props_info.is_already(STR_FIELD_VALUE)
        for v in vertices:
            v_per = self.compute_vertex_persistence(v, key_field_id)
            self.__props_info.set_prop_entry_fast(key_per_id, (v_per,), v.get_id(), 1)

    # Build a subgraph list from a property
    # key_prop: property key with the labels for the subgraphs, only boolean or int
    # properties are valid. IMPORTANT: negative labels are not valid
    def build_sgraphs_list(self, key_prop):

        # Initialization
        prop_id = self.get_prop_id(key_prop)
        str_type = self.get_prop_type(key_id=prop_id)
        n_comp = self.get_prop_ncomp(key_id=prop_id)
        if (str_type != 'int') and (str_type != 'bool'):
            error_msg = 'Only one component int and bool types are valid.'
            raise pexceptions.PySegInputError(expr='build_sgraphs_list (GraphMCF)',
                                              msg=error_msg)
        data_type = disperse_io.TypesConverter().gt_to_numpy(str_type)

        # Creating the helping lists and lut
        n_lbls = self.__props_info.get_prop_max(key_prop) + 1
        l_sgv = np.zeros(shape=n_lbls, dtype=object)
        l_sge = np.zeros(shape=n_lbls, dtype=object)
        for i in range(n_lbls):
            l_sgv[i] = list()
            l_sge[i] = list()
        lut = (-1) * np.ones(shape=self.get_nid(), dtype=data_type)

        # Filling the list of vertices
        for v in self.get_vertices_list():
            v_id = v.get_id()
            t = self.get_prop_entry_fast(prop_id, v_id, n_comp, data_type)[0]
            if t >= 0:
                lut[v_id] = t
                l_sgv[t].append(v_id)

        # Filling the list of edges
        for e in self.get_edges_list():
            s_lbl = lut[e.get_source_id()]
            if s_lbl != -1:
                t_lbl = lut[e.get_target_id()]
                if s_lbl == t_lbl:
                    l_sge[s_lbl].append(e.get_id())

        # Building the subgraphs
        sgraphs = list()
        for i, s in enumerate(l_sgv):
            if len(s) > 0:
                sgraphs.append(SubGraphMCF(self, l_sgv[i], l_sge[i]))

        return sgraphs


    # Compute vertex persistence as the highest absolute difference between the vertex field
    # value and the field value of its edges
    # vertex: the vertex
    # key_field_id: key identifier for field value property
    # Returns: vertex persistence
    def compute_vertex_persistence(self, vertex, key_field_id):

        per = 0
        v_id = vertex.get_id()
        hold = self.__props_info.get_prop_entry_fast(key_field_id, v_id, 1, np.float)
        v_field = hold[0]
        arcs = vertex.get_arcs()
        for a in arcs:
            a_id = a.get_sad_id()
            if a_id is not None:
                hold = self.__props_info.get_prop_entry_fast(key_field_id, a_id, 1, np.float)
                fval = hold[0]
                # Update persistence
                hold_per = math.fabs(v_field - fval)
                if hold_per > per:
                    per = hold_per

        return per

    # Find the pair vertex of an input vertex, that is, the vertex connected to the input
    # one though the lowest property value (typically field value)
    # vertex: the input vertex
    # key_field_id: key identifier for field value property
    # Returns: pair vertex and the arc used for edging (if they do not exist return None, None)
    def compute_pair_vertex(self, vertex, key_prop_id):

        pair = None
        e_pair = None
        a_pair = None
        h_fval = np.finfo(np.float).max
        arcs = vertex.get_arcs()
        for a in arcs:
            a_id = a.get_sad_id()
            if a_id is not None:
                edge = self.get_edge(a_id)
                if edge is not None:
                    hold = self.__props_info.get_prop_entry_fast(key_prop_id, a_id, 1, np.float)
                    fval = hold[0]
                    if fval < h_fval:
                        h_fval = fval
                        e_pair = edge
                        a_pair = a

        if e_pair is not None:
            p_id = e_pair.get_source_id()
            if p_id == vertex.get_id():
                p_id = e_pair.get_target_id()
            pair = self.get_vertex(p_id)

        return pair, a_pair

    # Get the pair vertex (if exist) of a given vertex
    # vertex: input vertex
    # key_pair_id: key identifier for vertex pair id property
    # Return: the pair vertex and the edge which joins both vertices
    def get_vertex_pair(self, vertex, key_pair_id):
        v_id = vertex.get_id()
        hold = self.__props_info.get_prop_entry_fast(key_pair_id, v_id, 1, np.float)
        p_vertex = self.get_vertex(int(hold[0]))
        if p_vertex is not None:
            p_id = p_vertex.get_id()
            if p_id is not None:
                neighs, edges = self.get_vertex_neighbours(p_vertex.get_id())
                for i, n in enumerate(neighs):
                    if n.get_id() == v_id:
                        return p_vertex, edges[i]
        return None, None


    # Performs a topological simplification based on vertices cancellation according
    # to its persistence.
    # VERY IMPORTANT: this method only works if the GraphMCF does not contain self-loops and
    # repeated edges
    # th_per: threshold for persistence (default None)
    # n: number of preserved vertices, if not None (default None) vertices are cancelled and
    # deleted until reaching n so th_per is not taken into account
    # prop_ref: property key for reference from a binary mask (only active if n is not None), default None
    def topological_simp(self, th_per=None, n=None, prop_ref=None):

        if (th_per is None) and (n is None):
            error_msg = 'Both \'th_per\' and \'n\' input parameters cannot be simultaneously None.'
            raise pexceptions.PySegInputError('topological_simp (GraphMCF)', error_msg)

        # Create a persistence ascend ordered lists with the initial vertices
        self.__v_lst = self.get_vertices_list()
        key_per_id = self.get_prop_id(STR_V_PER)
        key_pair_id = self.get_prop_id(self.__pair_prop_key)
        key_hid_id = self.__props_info.add_prop(STR_HID, 'int', 1, def_val=0)
        self.__per_lst = list()
        for i, v in enumerate(self.__v_lst):
            hold = self.__props_info.get_prop_entry_fast(key_per_id, v.get_id(), 1, np.float)
            self.__per_lst.append(hold[0])
            self.__props_info.set_prop_entry_fast(key_hid_id, (i,), v.get_id(), 1)
        self.__per_lst = np.asarray(self.__per_lst)
        self.__v_lst = np.asarray(self.__v_lst)
        mx = np.finfo(np.float).max

        if n is None:
            # Cancel vertices until emptying the list or reaching the threshold
            count = 0
            if len(self.__per_lst) > 0:
                ind = np.argmin(self.__per_lst)
                n_vertices = self.__v_lst.shape[0]
                while count < n_vertices:
                    vertex = self.__v_lst[ind]
                    self.__per_lst[ind] = mx
                    hold = self.__props_info.get_prop_entry_fast(key_per_id, vertex.get_id(),
                                                                 1, np.float)
                    # Threshold persistence condition
                    if hold[0] > th_per:
                        break
                    self.__cancel_vertex(vertex, key_pair_id, key_per_id, key_hid_id)
                    # Compute next vertex
                    ind = np.argmin(self.__per_lst)
                    count += 1
        else:
            if prop_ref is None:
                # Cancel vertices until reaching n value
                count = self.__v_lst.shape[0]
                if len(self.__per_lst) > 0:
                    ind = np.argmin(self.__per_lst)
                    while count > n:
                        vertex = self.__v_lst[ind]
                        self.__per_lst[ind] = mx
                        self.__cancel_vertex(vertex, key_pair_id, key_per_id, key_hid_id)
                        # Compute next vertex
                        ind = np.argmin(self.__per_lst)
                        count -= 1
            else:
                # Get lut of references
                lut_ref = np.zeros(shape=self.get_nid(), dtype=np.bool)
                prop_ref_id = self.get_prop_id(prop_ref)
                ref_n_comp = self.get_prop_ncomp(key=prop_ref)
                if (prop_ref_id is None) or (ref_n_comp != 1):
                    error_msg = 'The graph does not include %s as mask.' % prop_ref
                    raise pexceptions.PySegInputError(expr='topological_simp (GraphMCF)', msg=error_msg)
                prop_type = self.get_prop_type(key_id=prop_ref_id)
                prop_type = disperse_io.TypesConverter().gt_to_numpy(prop_type)
                count = 0
                for i, v in enumerate(self.__vertices):
                    if v is not None:
                        if self.__props_info.get_prop_entry_fast(prop_ref_id, v.get_id(), 1, prop_type)[0] > 0:
                            lut_ref[i] = True
                            count += 1
                # Cancel vertices until reaching n value
                if len(self.__per_lst) > 0:
                    ind = np.argmin(self.__per_lst)
                    while count > n:
                        vertex = self.__v_lst[ind]
                        self.__per_lst[ind] = mx
                        self.__cancel_vertex(vertex, key_pair_id, key_per_id, key_hid_id)
                        # Compute next vertex
                        ind = np.argmin(self.__per_lst)
                        if lut_ref[vertex.get_id()]:
                            count -= 1

        # Mark for deleting intermediate lists
        self.__v_lst = None
        self.__per_lst = None
        self.__props_info.remove_prop(STR_HID)

    # Update the persistence property STR_V_FPER of vertices (No vertex or edge is deleted)
    # VERY IMPORTANT: this method only works if the GraphMCF does not contain self-loops and
    # repeated edges
    # th_per: (None) maximum persistence, the procedure stops when this threshold is reached
    def compute_full_per(self, th_per=None):

        # Make a temporal copy of the GraphMCF so as to operate with it
        graph_copy = copy.deepcopy(self)
        key_fper_id = self.__props_info.add_prop(STR_V_FPER, 'float', 1, def_val=-1)

        # Create a persistence ascend ordered lists with the initial vertices
        graph_copy.__v_lst = graph_copy.get_vertices_list()
        key_per_id = graph_copy.get_prop_id(STR_V_PER)
        key_field_id = graph_copy.get_prop_id(STR_FIELD_VALUE)
        key_hid_id = graph_copy.__props_info.add_prop(STR_HID, 'int', 1, def_val=0)
        graph_copy.__per_lst = list()
        for i, v in enumerate(graph_copy.__v_lst):
            hold = graph_copy.__props_info.get_prop_entry_fast(key_per_id, v.get_id(), 1, np.float)
            graph_copy.__per_lst.append(hold[0])
            graph_copy.__props_info.set_prop_entry_fast(key_hid_id, (i,), v.get_id(), 1)
        graph_copy.__per_lst = np.asarray(graph_copy.__per_lst)
        graph_copy.__v_lst = np.asarray(graph_copy.__v_lst)
        mx = np.finfo(np.float).max

        # Cancel vertices until emptying the list or reaching the threshold
        count = 0
        if len(graph_copy.__per_lst) > 0:
            ind = np.argmin(graph_copy.__per_lst)
            n_vertices = graph_copy.__v_lst.shape[0]
            while count < n_vertices:
                vertex = graph_copy.__v_lst[ind]
                graph_copy.__per_lst[ind] = mx
                v_id = vertex.get_id()
                hold = graph_copy.__props_info.get_prop_entry_fast(key_per_id, v_id,
                                                                   1, np.float)
                # Threshold persistence condition
                if (th_per is not None) and (hold[0] > th_per):
                    break
                graph_copy.__cancel_vertex(vertex, key_field_id, key_per_id, key_hid_id)
                # Update persistence in the self GraphMCF
                self.__props_info.set_prop_entry_fast(key_fper_id, hold, v_id, 1)
                # Compute next vertex
                ind = np.argmin(graph_copy.__per_lst)
                count += 1

        # Delete the copy
        graph_copy.__v_lst = None
        graph_copy.__per_lst = None
        graph_copy.__props_info.remove_prop(STR_HID)
        del graph_copy

    # Delete all arcs with are not part of and edges (one of its extreme is not a vertex)
    def arc_simp(self):
        for a in self.get_arcs_list():
            vertex = self.get_vertex(a.get_min_id())
            edge = self.get_edge(a.get_sad_id())
            if (vertex is None) or (edge is None):
                vertex.del_arc(a)

    # Keep just the N closest neighbours
    # n_neighs: maximum number of neighs
    def edge_simp(self, n_neighs=1):

        # Edges to delete LUT
        lut = np.ones(shape=self.__skel.GetVerts().GetNumberOfCells(), dtype=np.bool)

        # Look edges which are not going to be deleted
        for v in self.get_vertices_list():
            v_id = v.get_id()
            neighs, edges = self.get_vertex_neighbours(v_id)
            l_edges = len(edges)
            if l_edges > n_neighs:
                lengths = np.zeros(shape=l_edges, dtype=np.float)
                ids = np.zeros(shape=l_edges, dtype=np.int)
                for i in range(l_edges):
                    lengths[i] = self.get_edge_length(edges[i])
                    ids[i] = edges[i].get_id()
                ids_sort = ids[np.argsort(lengths)]
                for j in range(n_neighs):
                    lut[ids_sort[j]] = False

        # Delete marked edges
        for i in range(len(lut)):
            if lut[i]:
                self.remove_edge(self.get_edge(i))

    # Delete vertices with a degree equal or lower than threshold
    # th_d: threshold for vertex degree
    def simp_vertices(self, th_d):
        for v in self.get_vertices_list():
            neighs, _ = self.get_vertex_neighbours(v.get_id())
            if len(neighs) <= th_d:
                self.remove_vertex(v)



    # fname: file name ended with .pkl
    def pickle(self, fname):

        # Dump pickable objects and store the file names of the unpickable objects
        stem, ext = os.path.splitext(fname)
        self.__skel_fname = stem + '_skel.vtp'
        pkl_f = open(fname, 'w')
        try:
            pickle.dump(self, pkl_f)
        finally:
            pkl_f.close()

        # Store unpickable objects
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(self.__skel_fname)
        vtk_ver = vtk.vtkVersion().GetVTKVersion()
        if int(vtk_ver[0]) < 6:
            writer.SetInput(self.__skel)
        else:
            writer.SetInputData(self.__skel)
        if writer.Write() != 1:
            error_msg = 'Error writing %s.' % self.__skel_fname
            raise pexceptions.PySegInputError(expr='pickle (GraphMCF)', msg=error_msg)

    # Threshold vertices according to a property
    # prop: vertex property key
    # thres: threshold
    # oper: operator
    # mask: if not None (default None), vertices on mask (non zero fg) will not be removed
    def threshold_vertices(self, prop, thres, oper, mask=None):

        key_id = self.__props_info.is_already(prop)
        if key_id is None:
            error_msg = 'The graph does not include %s property.' % prop
            raise pexceptions.PySegInputError(expr='theshold_vertices (GraphMCF)', msg=error_msg)
        n_comp = self.__props_info.get_ncomp(index=key_id)
        data_type = self.__props_info.get_type(index=key_id)
        data_type = disperse_io.TypesConverter().gt_to_numpy(data_type)

        if mask is None:
            for v in self.get_vertices_list():
                t = self.__props_info.get_prop_entry_fast(key_id, v.get_id(), n_comp, data_type)
                t = sum(t) / len(t)
                if oper(t, thres):
                    self.remove_vertex(v)
        else:
            for v in self.get_vertices_list():
                x, y, z = self.get_vertex_coords(v)
                if mask[int(np.round(x)), int(np.round(y)), int(np.round(z))] == 0:
                    t = self.__props_info.get_prop_entry_fast(key_id, v.get_id(), n_comp, data_type)
                    t = sum(t) / len(t)
                    if oper(t, thres):
                        self.remove_vertex(v)

    # Threshold vertices according to their degree (number of neighbours)
    # thres: threshold
    # oper: operator
    # mask: if not None (default None), vertices on mask (non zero fg) will not be removed
    def threshold_vertices_deg(self, thres, oper, mask=None):

        if mask is None:
            for v in self.get_vertices_list():
                neighs, _ = self.get_vertex_neighbours(v.get_id())
                if oper(len(neighs), thres):
                    self.remove_vertex(v)
        else:
            for v in self.get_vertices_list():
                x, y, z = self.get_vertex_coords(v)
                if mask[int(np.round(x)), int(np.round(y)), int(np.round(z))] == 0:
                    neighs, _ = self.get_vertex_neighbours(v.get_id())
                    if oper(len(neighs), thres):
                        self.remove_vertex(v)

    # Threshold vertices an already segmented area
    # prop: segmentation vertex property key
    # lbl: threshold
    # keep_b: if True (default False) region borders, i.e. vertices connected to external
    # regions and the edges which create these connections
    def threshold_seg_region(self, prop, lbl, keep_b=False):

        key_id = self.__props_info.is_already(prop)
        if key_id is None:
            error_msg = 'The graph does not include %s property.' % prop
            raise pexceptions.PySegInputError(expr='theshold_vertices (GraphMCF)', msg=error_msg)
        n_comp = self.__props_info.get_ncomp(index=key_id)
        data_type = self.__props_info.get_type(index=key_id)
        data_type = disperse_io.TypesConverter().gt_to_numpy(data_type)
        if isinstance(lbl, tuple):
            lbl_t = lbl
        else:
            lbl_t = (lbl,)

        if not keep_b:
            for v in self.get_vertices_list():
                t = self.__props_info.get_prop_entry_fast(key_id, v.get_id(), n_comp, data_type)
                if t == lbl_t:
                    self.remove_vertex(v)
        else:
            # Firstly, loop for deleting vertices
            for v in self.get_vertices_list():
                v_id = v.get_id()
                t = self.__props_info.get_prop_entry_fast(key_id, v_id, n_comp, data_type)
                if t == lbl_t:
                    to_del = True
                    neighs, _ = self.get_vertex_neighbours(v_id)
                    for n in neighs:
                        u = self.__props_info.get_prop_entry_fast(key_id, n.get_id(),
                                                                  n_comp, data_type)
                        if u != lbl_t:
                            to_del = False
                            break
                    if to_del:
                        self.remove_vertex(v)
            # Secondly, loop for deleting edges
            for e in self.get_edges_list():
                s_id = e.get_source_id()
                t_id = e.get_target_id()
                t = self.__props_info.get_prop_entry_fast(key_id, s_id, n_comp, data_type)
                u = self.__props_info.get_prop_entry_fast(key_id, t_id, n_comp, data_type)
                if (t == lbl_t) and (u == lbl_t):
                    self.remove_edge(e)

    # Threshold graph vertices in a specified list
    # v_list: list with the vertices id
    # in_mode: if True (default) vertices in list are removed, otherwise the rest are remove and the ones in the
    #          list are preserved
    def threshold_vertices_list(self, v_list, in_mode=True):
        if in_mode:
            for v_id in v_list:
                self.remove_vertex(self.get_vertex(v_id))
        else:
            prev_lut = np.zeros(shape=self.get_nid(), dtype=np.bool)
            for v_id in v_list:
                prev_lut[v_id] = True
            for v in self.get_vertices_list():
                v_id = v.get_id()
                if not prev_lut[v_id]:
                    self.remove_vertex(self.get_vertex(v_id))

    # Threshold vertices its geodesic distance to as segmented mask
    # mask: segmented (1-fg and 0-bg) tomogram
    # key_w: property key for edge metric
    # th_dst: maximum geodesic distance (threshold)
    # winv: if True (default False) values of weighting property are inverted
    def threshold_mask_dst(self, mask, key_w, th_dst, winv=False):

        # Adding segmentation property
        key_s = 'hold'
        self.add_scalar_field_nn(mask, key_s)
        graph_gt = GraphGT(self).get_gt()
        prop_w = graph_gt.edge_properties[key_w]
        if winv:
            prop_w.get_array()[:] = lin_map(prop_w.get_array(), lb=1, ub=0)
        prop_s = graph_gt.vertex_properties[key_s]
        prop_s_arr = prop_s.get_array().astype(np.bool)
        prop_i = graph_gt.vertex_properties[DPSTR_CELL]
        lut_th = np.ones(shape=self.get_nid(), dtype=np.bool)

        # Measuring geodesic distances
        dists_map = gt.shortest_distance(graph_gt, weights=prop_w)

        # Loop for checking shortest distance at every vertex
        for v in graph_gt.vertices():
            v_id = prop_i[v]
            if prop_s[v] > 0:
                lut_th[v_id] = False
            else:
                dist_map_arr = dists_map[v].get_array()
                if dist_map_arr[prop_s_arr].min() < th_dst:
                    lut_th[v_id] = False

        # Thresholding
        for v in self.get_vertices_list():
            if lut_th[v.get_id()]:
                self.remove_vertex(v)

        # Delete hold property
        self.__props_info.remove_prop(key=key_s)

    # Threshold vertices until preserving just the given number of vertices with the highest
    # or lowest property
    # n: number of vertices preserved
    # prop: key string for the property
    # prop_ref: property key for reference from a binary mask (only active if n is not None), default None
    def threshold_vertices_n(self, n, prop, mode='high', prop_ref=None):

        key_id = self.__props_info.is_already(prop)
        if key_id is None:
            error_msg = 'The graph does not include %s property.' % prop
            raise pexceptions.PySegInputError(expr='theshold_vertices_n (GraphMCF)', msg=error_msg)
        n_comp = self.__props_info.get_ncomp(index=key_id)
        data_type = self.__props_info.get_type(index=key_id)
        data_type = disperse_io.TypesConverter().gt_to_numpy(data_type)

        # List of ordered vertices
        vertices = np.asarray(self.get_vertices_list())
        if prop_ref is None:
            if n_comp == 1:
                arr_prop = np.zeros(shape=vertices.shape[0], dtype=data_type)
                for i, v in enumerate(vertices):
                    hold = self.__props_info.get_prop_entry_fast(key_id, v.get_id(),
                                                                 n_comp, data_type)
                    arr_prop[i] = hold[0]
            else:
                arr_prop = np.zeros(shape=(vertices.shape[0], n_comp), dtype=data_type)
                for i, v in enumerate(vertices):
                    hold = self.__props_info.get_prop_entry_fast(key_id, v.get_id(),
                                                                 n_comp, data_type)
                    hold = np.asarray(hold)
                    arr_prop[i] = math.sqrt(np.sum(hold * hold))
        else:
            ref_key_id = self.__props_info.is_already(prop_ref)
            ref_n_comp = self.__props_info.get_ncomp(key=prop_ref)
            if (ref_key_id is None) or (ref_n_comp != 1):
                error_msg = 'The graph does not include %s as mask.' % prop_ref
                raise pexceptions.PySegInputError(expr='theshold_vertices_n (GraphMCF)', msg=error_msg)
            ref_data_type = self.__props_info.get_type(index=ref_key_id)
            ref_data_type = disperse_io.TypesConverter().gt_to_numpy(ref_data_type)
            lut_ref = np.zeros(shape=vertices.shape[0], dtype=data_type)
            if n_comp == 1:
                arr_prop = np.zeros(shape=vertices.shape[0], dtype=data_type)
                for i, v in enumerate(vertices):
                    v_id = v.get_id()
                    hold = self.__props_info.get_prop_entry_fast(key_id, v_id, n_comp, data_type)
                    arr_prop[i] = hold[0]
                    if self.__props_info.get_prop_entry_fast(ref_key_id, v_id, 1, ref_data_type)[0] > 0:
                        lut_ref[i] = True
            else:
                arr_prop = np.zeros(shape=(vertices.shape[0], n_comp), dtype=data_type)
                for i, v in enumerate(vertices):
                    v_id = v.get_id()
                    hold = self.__props_info.get_prop_entry_fast(key_id, v_id, n_comp, data_type)
                    hold = np.asarray(hold)
                    arr_prop[i] = math.sqrt(np.sum(hold * hold))
                    if self.__props_info.get_prop_entry_fast(ref_key_id, v_id, 1, ref_data_type)[0] > 0:
                        lut_ref[i] = True
        ids = np.argsort(arr_prop)
        if prop_ref is not None:
            lut_ref = lut_ref[ids]
        vertices = vertices[ids]
        if mode == 'high':
            vertices = vertices[::-1]
            if prop_ref is not None:
                lut_ref = lut_ref[::-1]

        # Removing vertices
        if prop_ref is None:
            for i in range(n, vertices.shape[0]):
                self.remove_vertex(vertices[i])
        else:
            n_del = lut_ref.sum()
            for i in range(vertices.shape[0]-1, -1, -1):
                self.remove_vertex(vertices[i])
                if lut_ref[i]:
                    n_del -= 1
                if n_del <= n:
                    break

    # Threshold edges according to a property
    def threshold_edges(self, prop, thres, oper, mask=None):

        key_id = self.__props_info.is_already(prop)
        if self.__props_info.is_already(prop) is None:
            error_msg = 'The graph does not include %s property.' % prop
            raise pexceptions.PySegInputError(expr='theshold_edges (GraphMCF)', msg=error_msg)
        n_comp = self.__props_info.get_ncomp(index=key_id)
        data_type = self.__props_info.get_type(index=key_id)
        data_type = disperse_io.TypesConverter().gt_to_numpy(data_type)

        if mask is None:
            for e in self.get_edges_list():
                t = self.__props_info.get_prop_entry_fast(key_id, e.get_id(), n_comp, data_type)
                t = sum(t) / len(t)
                if oper(t, thres):
                    self.remove_edge(e)
        else:
            for e in self.get_edges_list():
                x, y, z = self.get_edge_coords(e)
                if mask[int(np.round(x)), int(np.round(y)), int(np.round(z))] == 0:
                    t = self.__props_info.get_prop_entry_fast(key_id, e.get_id(), n_comp, data_type)
                    t = sum(t) / len(t)
                    if oper(t, thres):
                        self.remove_edge(e)

    # Threshold edges util preserving just the given number of edges with the highest
    # or lowest property
    # n: number of edges preserved, if less or equal to zero the function does nothing
    # prop: key string for the property
    # mode: if 'high' (default) then vertices with highest property values are preserved,
    # otherwise those with the lowest
    # prop_ref: property key for reference from a binary mask (only active if n is not None), default None
    # fit: if True (default None) only edges in reference binary mask are processed
    #      (requires prop_ref not None)
    def threshold_edges_n(self, n, prop, mode='high', prop_ref=None, fit=True):

        if n <= 0:
            return

        key_id = self.__props_info.is_already(prop)
        if self.__props_info.is_already(prop) is None:
            error_msg = 'The graph does not include %s property.' % prop
            raise pexceptions.PySegInputError(expr='theshold_edges_n (GraphMCF)', msg=error_msg)
        n_comp = self.__props_info.get_ncomp(index=key_id)
        data_type = self.__props_info.get_type(index=key_id)
        data_type = disperse_io.TypesConverter().gt_to_numpy(data_type)

        # List of ordered vertices
        edges = np.asarray(self.get_edges_list())
        if prop_ref is None:
            if n_comp == 1:
                arr_prop = np.zeros(shape=edges.shape[0], dtype=data_type)
                for i, e in enumerate(edges):
                    hold = self.__props_info.get_prop_entry_fast(key_id, e.get_id(),
                                                                 n_comp, data_type)
                    arr_prop[i] = hold[0]
            else:
                arr_prop = np.zeros(shape=(edges.shape[0], n_comp), dtype=data_type)
                for i, e in enumerate(edges):
                    hold = self.__props_info.get_prop_entry_fast(key_id, e.get_id(),
                                                                 n_comp, data_type)
                    hold = np.asarray(hold)
                    arr_prop[i] = math.sqrt(np.sum(hold * hold))
        else:
            ref_key_id = self.__props_info.is_already(prop_ref)
            ref_n_comp = self.__props_info.get_ncomp(key=prop_ref)
            if (ref_key_id is None) or (ref_n_comp != 1):
                error_msg = 'The graph does not include %s as mask.' % prop_ref
                raise pexceptions.PySegInputError(expr='theshold_edges_n_ref (GraphMCF)', msg=error_msg)
            ref_data_type = self.__props_info.get_type(index=ref_key_id)
            ref_data_type = disperse_io.TypesConverter().gt_to_numpy(ref_data_type)
            lut_ref = np.zeros(shape=edges.shape[0], dtype=data_type)
            if n_comp == 1:
                arr_prop = np.zeros(shape=edges.shape[0], dtype=data_type)
                for i, e in enumerate(edges):
                    e_id = e.get_id()
                    hold = self.__props_info.get_prop_entry_fast(key_id, e_id, n_comp, data_type)
                    arr_prop[i] = hold[0]
                    if self.__props_info.get_prop_entry_fast(ref_key_id, e_id, 1, ref_data_type)[0] > 0:
                        lut_ref[i] = True
            else:
                arr_prop = np.zeros(shape=(edges.shape[0], n_comp), dtype=data_type)
                for i, e in enumerate(edges):
                    e_id = e.get_id()
                    hold = self.__props_info.get_prop_entry_fast(key_id, e_id, n_comp, data_type)
                    hold = np.asarray(hold)
                    arr_prop[i] = math.sqrt(np.sum(hold * hold))
                    if self.__props_info.get_prop_entry_fast(ref_key_id, e_id, 1, ref_data_type)[0] > 0:
                        lut_ref[i] = True
        ids = np.argsort(arr_prop)
        if prop_ref is not None:
            lut_ref = lut_ref[ids]
        edges = edges[ids]
        if mode == 'high':
            edges = edges[::-1]
            if prop_ref is not None:
                lut_ref = lut_ref[::-1]

        # Removing vertices
        if prop_ref is None:
            for i in range(n, edges.shape[0]):
                self.remove_edge(edges[i])
        else:
            if fit:
                n_del = lut_ref.sum()
                for i in range(edges.shape[0]-1, -1, -1):
                    if lut_ref[i]:
                        self.remove_edge(edges[i])
                        n_del -= 1
                    if n_del <= n:
                        break
            else:
                n_del = lut_ref.sum()
                for i in range(edges.shape[0]-1, -1, -1):
                    self.remove_edge(edges[i])
                    if lut_ref[i]:
                        n_del -= 1
                    if n_del <= n:
                        break

    # Threshold edges contained by a mask (it preserves those that go outside)
    # mask: 3D numpy array
    def threshold_edges_in_mask(self, mask):

        for e in self.get_edges_list():
            s_id = e.get_source_id()
            t_id = e.get_target_id()
            x_s, y_s, z_s = self.__skel.GetPoint(s_id)
            x_s, y_s, z_s = int(np.floor(x_s)), int(np.floor(y_s)), int(np.floor(z_s))
            x_t, y_t, z_t = self.__skel.GetPoint(t_id)
            x_t, y_t, z_t = int(np.floor(x_t)), int(np.floor(y_t)), int(np.floor(z_t))
            s_r = mask[x_s, y_s, z_s]
            s_t = mask[x_t, y_t, z_t]
            if (s_r > 0) and (s_t > 0):
                self.remove_edge(e)

    # Compute the vertices density relevance in the ArcGraph
    def compute_vertices_relevance(self):

        # Compute arcs total density
        vertices = self.get_vertices_list()
        densities = np.zeros(shape=len(vertices), dtype=np.float64)
        for i, v in enumerate(vertices):
            geom = v.get_geometry()
            densities[i] += geom.get_total_density()

        # Get cumulative distribution function
        densities = lin_map(densities, lb=densities.max(), ub=densities.min())
        densities /= densities.sum()
        arg = np.argsort(densities)
        densities_sort = densities[arg]
        cdf = np.zeros(shape=densities_sort.shape, dtype=np.float)
        for i in range(1, len(densities_sort)):
            cdf[i] = cdf[i - 1] + densities_sort[i]

        # Set the new property to all arcs
        self.__props_info.add_prop(STR_VERTEX_RELEVANCE,
                                   disperse_io.TypesConverter().numpy_to_gt(np.float), 1, 0)
        key_id = self.__props_info.is_already(STR_VERTEX_RELEVANCE)
        n_comp = self.__props_info.get_ncomp(index=key_id)
        for i, v in enumerate(vertices):
            # self.__props_info.set_prop_entry(STR_VERTEX_RELEVANCE, cdf[arg[i]], v.get_id())
            self.__props_info.set_prop_entry_fast(key_id, (cdf[arg[i]],), v.get_id(), n_comp)

    # Compute the graph density relevance in the GraphMCF
    def compute_sgraph_relevance(self):

        sgraphs = self.get_vertex_sg_lists()
        if (sgraphs is None) or (len(sgraphs) < 1):
            if self.__props_info.is_already(STR_GRAPH_RELEVANCE) is None:
                self.__props_info.add_prop(STR_GRAPH_RELEVANCE,
                                           disperse_io.TypesConverter().numpy_to_gt(np.float),
                                           1, -1)
            return
        densities = np.zeros(shape=len(sgraphs), dtype=np.float64)
        for i, g in enumerate(sgraphs):
            hold_density = 0
            for v in g:
                geom = v.get_geometry()
                if geom is not None:
                    hold_density += geom.get_total_density()
            densities[i] = hold_density

        # Get cumulative distribution function
        densities /= densities.sum()
        arg = np.argsort(densities)
        densities_sort = densities[arg]
        cdf = np.zeros(shape=len(densities_sort), dtype=np.float)
        for i in range(1, len(densities_sort)):
            cdf[i] = cdf[i - 1] + densities_sort[i]

        # Unsort cdf
        ucdf = np.zeros(shape=cdf.shape, dtype=cdf.dtype)
        for i in range(len(cdf)):
            ucdf[arg[i]] = cdf[i]

        # Set the new property to all vertices of the subgraphs
        key_id = self.__props_info.is_already(STR_GRAPH_RELEVANCE)
        if key_id is None:
            self.__props_info.add_prop(STR_GRAPH_RELEVANCE,
                                       disperse_io.TypesConverter().numpy_to_gt(np.float), 1, 0)
            key_id = self.__props_info.is_already(STR_GRAPH_RELEVANCE)
        n_comp = self.__props_info.get_ncomp(index=key_id)
        for i, g in enumerate(sgraphs):
            for v in g:
                # self.__props_info.set_prop_entry(STR_GRAPH_RELEVANCE, (ucdf[i],), v.get_id())
                self.__props_info.set_prop_entry_fast(key_id, (ucdf[i],), v.get_id(), n_comp)

    # Property stored as key
    # key: key string
    # w_x|y|z: dimension weighting
    # total: mode of computation
    def compute_edges_length(self, key=SGT_EDGE_LENGTH, w_x=1, w_y=1, w_z=1, total=False):
        key_id = self.__props_info.is_already(key)
        if key_id is None:
            key_id = self.__props_info.add_prop(key, 'float', 1, 0)
        if (w_x == 1) and (w_y == 1) and (w_z == 1) and (not total):
            for e in self.get_edges_list():
                self.__props_info.set_prop_entry_fast(key_id, (self.get_edge_length(e),),
                                                      e.get_id(), 1)
        else:
            for e in self.get_edges_list():
                self.__props_info.set_prop_entry_fast(key_id,
                                                      (self.get_edge_length_2(e, w_x, w_y, w_z, total),),
                                                      e.get_id(), 1)

    # Property stored as key
    # key: key string
    def compute_vertices_dst(self, key=STR_VERT_DST):
        key_id = self.__props_info.is_already(key)
        if key_id is None:
            key_id = self.__props_info.add_prop(key, 'float', 1, 0)
        for e in self.get_edges_list():
            s = self.get_vertex(e.get_source_id())
            t = self.get_vertex(e.get_target_id())
            x_s, y_s, z_s = self.get_vertex_coords(s)
            x_t, y_t, z_t = self.get_vertex_coords(t)
            hold = np.asarray((x_s-x_t, y_s-y_t, z_s-z_t), dtype=np.float)
            dst = math.sqrt(np.sum(hold * hold)) * self.get_resolution()
            self.__props_info.set_prop_entry_fast(key_id, (dst,), e.get_id(), 1)

    # Edge property stored as key, STR_FWVERT_DST is the product of STR_FIELD_VALUE and STR_VERT_DST
    # key: key string (default STR_FWVERT_DST), by the way STR_VERT_DST is also computed
    def compute_vertices_fwdst(self, key=STR_FWVERT_DST):
        key_id = self.__props_info.is_already(key)
        if key_id is None:
            key_id = self.__props_info.add_prop(key, 'float', 1, 0)
        key_vd = self.__props_info.is_already(STR_VERT_DST)
        if key_vd is None:
            self.compute_vertices_dst()
        key_f = self.__props_info.is_already(STR_FIELD_VALUE)
        for e in self.get_edges_list():
            e_id = e.get_id()
            dst = self.get_prop_entry_fast(key_vd, e_id, 1, np.float)[0]
            field = self.get_prop_entry_fast(key_f, e_id, 1, np.float)[0]
            self.__props_info.set_prop_entry_fast(key_id, (dst*field,), e.get_id(), 1)

    # Integrates density information along edges
    # The result is stored in STR_EDGE_INT property
    # field: if True (default False), integration similitude is multiplied by edge field value
    def compute_edges_integration(self, field=False):

        if self.__density is None:
            error_msg = 'This object must have a geometry for computing edge integration'
            raise pexceptions.PySegInputError(expr='compute_edges_integration (GraphMCF)',
                                              msg=error_msg)

        # Adding/Updating property
        key_id = self.__props_info.is_already(STR_EDGE_INT)
        if key_id is None:
            key_id = self.__props_info.add_prop(STR_EDGE_INT, 'float', 1, 0)

        for e in self.get_edges_list():

            # Get edge arcs
            v_s = self.__vertices[e.get_source_id()]
            v_t = self.__vertices[e.get_target_id()]
            sad_id = e.get_id()
            look = True
            v_s_arc = v_s.get_arcs()
            l_v_s_arcs = len(v_s_arc)
            i = 0
            v_t_arc = v_t.get_arcs()
            l_v_t_arcs = len(v_t_arc)
            while (i < l_v_s_arcs) and look:
                a_s = v_s_arc[i]
                if a_s.get_sad_id() == sad_id:
                    j = 0
                    while (j < l_v_t_arcs) and look:
                        a_t = v_t_arc[j]
                        if a_t.get_sad_id() == sad_id:
                            # Get points id
                            a_s_np = a_s.get_npoints()
                            a_t_np = a_t.get_npoints()
                            points = np.zeros(shape=a_s_np+a_t_np-1, dtype=np.int)
                            for i in range(a_t_np):
                                points[i] = a_t.get_point_id(i)
                            for i in range(a_s_np-1):
                                points[a_t_np+i] = a_s.get_point_id(i)
                            # Integration by trapezoidal rule
                            area_c = .0
                            length = .0
                            x1, y1, z1 = self.__skel.GetPoint(points[0])
                            for i in range(1, points.shape[0]):
                                x2, y2, z2 = self.__skel.GetPoint(points[i])
                                xh, yh, zh = x1 - x2, y1 - y2, z1 - z2
                                dist = math.sqrt(xh*xh + yh*yh + zh*zh)
                                length += dist
                                f_1 = trilin3d(self.__density, (x1, y1, z1))
                                f_2 = trilin3d(self.__density, (x2, y2, z2))
                                area_c += (0.5 * dist * (f_1 + f_2))
                                x1, y1, z1 = x2, y2, z2
                            # int similarity = (area_r - area_c) / area_r
                            x1, y1, z1 = self.__skel.GetPoint(sad_id)
                            d_s = trilin3d(self.__density, (x1, y1, z1))
                            s_int = (1/(d_s*length)) * area_c
                            self.set_prop_entry_fast(key_id, (s_int,), sad_id, 1)
                            look = False
                        j += 1
                i += 1

        if field:
            edges = self.get_edges_list()
            n_edges = len(edges)
            field = np.zeros(shape=n_edges, dtype=np.float)
            for i, e in enumerate(edges):
                x, y, z = self.get_edge_coords(e)
                field[i] = trilin3d(self.__density, (x, y, z))
            field = lin_map(field, lb=1, ub=0)
            for i, e in enumerate(edges):
                e_id = e.get_id()
                t = self.get_prop_entry_fast(key_id, e_id, 1, np.float)
                self.set_prop_entry_fast(key_id, (t[0]*field[i],), e_id, 1)

    # Computes vertices similitude for edges (min(d_v1,d_v2)/d_e) )
    def compute_edges_sim(self):

        if self.__density is None:
            error_msg = 'This object must have a geometry for computing edge integration'
            raise pexceptions.PySegInputError(expr='compute_edges_integration (GraphMCF)',
                                              msg=error_msg)

        # Adding/Updating property
        key_id = self.__props_info.is_already(STR_EDGE_SIM)
        if key_id is None:
            key_id = self.__props_info.add_prop(STR_EDGE_SIM, 'float', 1, 0)

        for e in self.get_edges_list():

            # Get vertices
            v_s = self.__vertices[e.get_source_id()]
            v_t = self.__vertices[e.get_target_id()]

            # Compute edge similitude
            x, y, z = self.__skel.GetPoint(v_s.get_id())
            d_v1 = trilin3d(self.__density, (x, y, z))
            x, y, z = self.__skel.GetPoint(v_t.get_id())
            d_v2 = trilin3d(self.__density, (x, y, z))
            e_id = e.get_id()
            x, y, z = self.__skel.GetPoint(e_id)
            d_e = trilin3d(self.__density, (x, y, z))
            if d_e == 0:
                hold1 = 0
            elif d_v1 < d_v2:
                hold1 = d_v1 / d_e
            else:
                hold1 = d_v2 / d_e
            self.set_prop_entry_fast(key_id, (hold1,), e_id, 1)

    # This metric is the product of int and 1/sim
    def compute_edge_filamentness(self):

        if self.__density is None:
            error_msg = 'This object must have a geometry for computing edge integration'
            raise pexceptions.PySegInputError(expr='compute_edge_filamentness (GraphMCF)',
                                              msg=error_msg)

        # Adding/Updating property
        key_id = self.__props_info.is_already(STR_EDGE_FNESS)
        if key_id is None:
            key_id = self.__props_info.add_prop(STR_EDGE_FNESS, 'float', 1, 0)

        for e in self.get_edges_list():

            # Get vertices
            v_s = self.__vertices[e.get_source_id()]
            v_t = self.__vertices[e.get_target_id()]

            # Get densities
            x, y, z = self.__skel.GetPoint(v_s.get_id())
            d_v1 = trilin3d(self.__density, (x, y, z))
            x, y, z = self.__skel.GetPoint(v_t.get_id())
            d_v2 = trilin3d(self.__density, (x, y, z))
            e_id = e.get_id()
            x, y, z = self.__skel.GetPoint(e_id)
            d_e = trilin3d(self.__density, (x, y, z))

            # Compute edge similitude
            if d_e == 0:
                hold1 = 0
            elif d_v1 < d_v2:
                hold1 = d_v1 / d_e
            else:
                hold1 = d_v2 / d_e

            # Compute integration
            hold2 = 0
            sad_id = e.get_id()
            look = True
            v_s_arc = v_s.get_arcs()
            l_v_s_arcs = len(v_s_arc)
            i = 0
            v_t_arc = v_t.get_arcs()
            l_v_t_arcs = len(v_t_arc)
            while (i < l_v_s_arcs) and look:
                a_s = v_s_arc[i]
                if a_s.get_sad_id() == sad_id:
                    j = 0
                    while (j < l_v_t_arcs) and look:
                        a_t = v_t_arc[j]
                        if a_t.get_sad_id() == sad_id:
                            # Get points id
                            a_s_np = a_s.get_npoints()
                            a_t_np = a_t.get_npoints()
                            points = np.zeros(shape=a_s_np+a_t_np-1, dtype=np.int)
                            for i in range(a_t_np):
                                points[i] = a_t.get_point_id(i)
                            for i in range(a_s_np-1):
                                points[a_t_np+i] = a_s.get_point_id(i)
                            # Integration by trapezoidal rule
                            area_c = .0
                            length = .0
                            x1, y1, z1 = self.__skel.GetPoint(points[0])
                            for i in range(1, points.shape[0]):
                                x2, y2, z2 = self.__skel.GetPoint(points[i])
                                xh, yh, zh = x1 - x2, y1 - y2, z1 - z2
                                dist = math.sqrt(xh*xh + yh*yh + zh*zh)
                                length += dist
                                f_1 = trilin3d(self.__density, (x1, y1, z1))
                                f_2 = trilin3d(self.__density, (x2, y2, z2))
                                area_c += (0.5 * dist * (f_1 + f_2))
                                # x1, y1, z1 = x2, y2, z2
                                # int similarity = (area_r - area_c) / area_r
                                x1, y1, z1 = self.__skel.GetPoint(sad_id)
                                d_s = trilin3d(self.__density, (x1, y1, z1))
                                d_s_l = d_s * length
                                if d_s_l > 0:
                                    hold2 = (1/(d_s*length)) * area_c
                                else:
                                    hold2 = 0
                            look = False
                        j += 1
                i += 1

            # The final result is the product
            self.set_prop_entry_fast(key_id, (hold1*hold2,), e_id, 1)

    # Compute edge affinity
    def compute_edge_affinity(self):

        # Adding/Updating property
        key_id = self.__props_info.is_already(STR_EDGE_AFFINITY)
        if key_id is None:
            key_id = self.__props_info.add_prop(STR_EDGE_AFFINITY, 'float', 1, 0)

        for e in self.get_edges_list():

            # Get vertices
            s_id = e.get_source_id()
            t_id = e.get_target_id()
            v_s = self.__vertices[s_id]
            v_t = self.__vertices[t_id]

            # Get field values
            x, y, z = self.__skel.GetPoint(v_s.get_id())
            d_v1 = trilin3d(self.__density, (x, y, z))
            x, y, z = self.__skel.GetPoint(v_t.get_id())
            d_v2 = trilin3d(self.__density, (x, y, z))
            e_id = e.get_id()
            x, y, z = self.__skel.GetPoint(e_id)
            d_e = trilin3d(self.__density, (x, y, z))

            # Compute edge affinity
            if d_e <= 0:
                aff = 0
            else:
                if d_v2 > d_v1:
                    aff = d_v2 / d_e
                else:
                    aff = d_v1 / d_e
                # aff = (d_v1+d_v2) / (2.*d_e)
            if aff > 1.:
                aff = 1.

            # Set edge property
            self.set_prop_entry_fast(key_id, (aff,), e_id, 1)

    def find_subgraphs(self):

        # graph_tool initialization
        graph = gt.Graph(directed=False)
        vertices = self.get_vertices_list()
        vertices_gt = np.empty(shape=len(self.__vertices), dtype=object)
        for v in vertices:
            vertices_gt[v.get_id()] = graph.add_vertex()
        edges = self.get_edges_list()
        for e in edges:
            graph.add_edge(vertices_gt[e.get_source_id()], vertices_gt[e.get_target_id()])

        # Subgraphs visitor initialization
        sgraph_id = graph.new_vertex_property("int")
        visitor = SubGraphVisitor(sgraph_id)

        # Find subgraphs
        for v in vertices:
            v_gt = vertices_gt[v.get_id()]
            if sgraph_id[v_gt] == 0:
                gt.dfs_search(graph, v_gt, visitor)
                visitor.update_sgraphs_id()

        # Set property
        key_id = self.__props_info.add_prop(STR_GRAPH_ID, 'int', 1, 0)

        n_comp = self.__props_info.get_ncomp(index=key_id)
        for v in vertices:
            v_gt = vertices_gt[v.get_id()]
            # self.__props_info.set_prop_entry(STR_GRAPH_ID, (sgraph_id[v_gt],), v.get_id())
            self.__props_info.set_prop_entry_fast(key_id, (sgraph_id[v_gt],), v.get_id(), n_comp)
        for e in edges:
            v_gt = vertices_gt[e.get_source_id()]
            # self.__props_info.set_prop_entry(STR_GRAPH_ID, (sgraph_id[v_gt],), v.get_id())
            self.__props_info.set_prop_entry_fast(key_id, (sgraph_id[v_gt],), e.get_id(), n_comp)

    # Compute the diameters of all subgraphs contained in graph
    # Uses pseudo_diameter of graph_tool package
    # update: if True (default False) graph finding is forced to be computed
    def compute_diameters(self, update=False):

        if (self.__props_info.is_already(STR_GRAPH_ID) is None) or update:
            self.find_subgraphs()
        n_graphs = self.__props_info.get_prop_max(STR_GRAPH_ID)
        if n_graphs < 1:
            return None
        graphs = np.empty(shape=n_graphs, dtype=object)
        weights = np.empty(shape=n_graphs, dtype=list)

        # Build graph_tool graphs
        key_id = self.__props_info.is_already(STR_GRAPH_ID)
        data_type = self.__props_info.get_type(index=key_id)
        data_type = disperse_io.TypesConverter().gt_to_numpy(data_type)
        for i in range(self.__props_info.get_nentries()):
            g_id = self.__props_info.get_prop_entry_fast(key_id, i, 1, data_type)
            if g_id[0] != -1:
                g_id = g_id[0] - 1
                if (g_id < n_graphs) and (graphs[g_id] is None):
                    graphs[g_id] = gt.Graph(directed=False)
                    weights[g_id] = list()
        vertices = self.get_vertices_list()
        vertices_gt = np.empty(shape=len(self.__vertices), dtype=object)
        for v in vertices:
            g_id = self.__props_info.get_prop_entry_fast(key_id, v.get_id(), 1, data_type)
            g_id = g_id[0] - 1
            if g_id != -1:
                vertices_gt[v.get_id()] = graphs[g_id].add_vertex()
        edges = self.get_edges_list()
        for e in edges:
            v_s = self.get_vertex(e.get_source_id()).get_id()
            g_id = self.__props_info.get_prop_entry_fast(key_id, v_s, 1, data_type)
            g_id = g_id[0] - 1
            if g_id != -1:
                v_t = self.get_vertex(e.get_target_id()).get_id()
                graphs[g_id].add_edge(vertices_gt[v_s], vertices_gt[v_t])
                weights[g_id].append(self.get_edge_length(e))

        # Compute weights
        w_props = np.empty(shape=len(graphs), dtype=object)
        for i, g in enumerate(graphs):
            w_prop = g.new_edge_property("float")
            w_prop.get_array()[:] = np.asarray(weights[i])
            w_props[i] = w_prop

        # Measure diameters
        diam = np.zeros(shape=len(graphs), dtype=np.float)
        for i, g in enumerate(graphs):
            t_diam, ends = gt.pseudo_diameter(g)
            diam[i] = gt.shortest_distance(g, ends[0], ends[1], w_props[i])

        # Add STR_GRAPH_DIAM property
        self.__props_info.add_prop(STR_GRAPH_DIAM,
                                   disperse_io.TypesConverter().numpy_to_gt(np.float), 1, 0)
        key_id2 = self.__props_info.is_already(STR_GRAPH_DIAM)
        n_comp = self.__props_info.get_ncomp(index=key_id2)
        for v in vertices:
            g_id = self.__props_info.get_prop_entry_fast(key_id, v.get_id(), 1, data_type)
            g_id = g_id[0] - 1
            self.__props_info.set_prop_entry_fast(key_id2, (diam[g_id],), v.get_id(), n_comp)

    # img: if img is None a new image is created with self.__density size
    # property: property name string for labeling the vertices
    # th_den: number of sigmas above (+) or below vertex geometry density mean for thresholding,
    #         if None no threshold is applied
    def print_vertices(self, img=None, property=DPSTR_CELL, th_den=None):

        if img is None:
            if self.__density is None:
                error_msg = 'The graph does not have geometry so image size cannot be estimated.'
                raise pexceptions.PySegInputError(expr='print_vertices (GraphMCF)', msg=error_msg)
            img = np.zeros(shape=self.__density.shape, dtype=np.float)

        if property == DPSTR_CELL:
            for v in self.get_vertices_list():
                v.get_geometry().print_in_numpy(img, v.get_id(), th_den)
        else:
            key_id = self.__props_info.is_already(key=property)
            if key_id is None:
                error_msg = 'This GraphMCF does not contain %s property.' % property
                raise pexceptions.PySegInputError(expr='print_vertices (GraphMCF)',
                                                  msg=error_msg)
            n_comp = self.__props_info.get_ncomp(index=key_id)
            if n_comp != 1:
                error_msg = 'Only one-dimensional can be printed.'
                raise pexceptions.PySegInputError(expr='print_vertices (GraphMCF)',
                                                  msg=error_msg)
            data_type = self.__props_info.get_type(index=key_id)
            data_type = disperse_io.TypesConverter().gt_to_numpy(data_type)
            for v in self.get_vertices_list():
                lbl = self.__props_info.get_prop_entry_fast(key_id, v.get_id(), n_comp, data_type)
                # lbl = self.__props_info.get_prop_entry(property, v.get_id())
                geom = v.get_geometry()
                if geom is not None:
                    geom.print_in_numpy(img, lbl[0], th_den)

        return img

    # Add a scalar field (3D numpy array) to the graph (vertices and edges)
    # field: input numpy array with the scalar field
    # name: string with key name for the property
    # manifolds: if True (default False) and a geometry has already been added to the object,
    # they are used as manifolds
    # neigh: (default None) neighbourhood size (in nm) for estimating the field value,
    # when manifolds is not used, if None this value is estimated through tri-linear
    # interpolation
    # mode: (default 'mean') valid 'sum', 'mean', 'max', 'min' and 'median'
    # offset: (default None) top-left-right corner coordinates of the subvolume from which the Graph was computed
    #         respect to field tomogram
    # bin: (default 1.)  tomogram bining respect to the Graph, it allows to compensate resolution differences
    #      between the graph and the input field
    # seg: (default None) segmented tomogram to ensure that information between two different region is not mixed,
    #      unused if neigh is zero or manifold is True
    def add_scalar_field(self, field, name, manifolds=False, neigh=None, mode='mean', offset=None, seg=None, bin=1.):

        # Creating the new property
        if bin <= 0:
            error_msg = 'Input bin must be greater than zero, current %s.' % str(bin)
            raise pexceptions.PySegInputError(expr='add_scalar_field (GraphMCF)', msg=error_msg)
        ibin = 1. / float(bin)
        self.__props_info.add_prop(key=name,
                                   type=disperse_io.TypesConverter().numpy_to_gt(field.dtype),
                                   ncomp=1,
                                   def_val=0)
        key_id = self.__props_info.is_already(name)
        if mode == 'sum':
            hold_fun = np.sum
        elif mode == 'mean':
            hold_fun = np.mean
        elif mode == 'median':
            hold_fun = np.median
        elif mode == 'max':
            hold_fun = np.max
        elif mode == 'min':
            hold_fun = np.min
        else:
            error_msg = 'Non valid mode %s.' % mode
            raise pexceptions.PySegInputError(expr='add_scalar_field (GraphMCF)', msg=error_msg)
        odtype = np.float32
        if neigh is not None:
            neigh = np.floor(0.5 * neigh * (1/self.__resolution))
            if neigh <= 0:
                neigh = None
            else:
                odtype = np.int
        if offset is None:
            offset = np.zeros(shape=3, dtype=odtype)
        else:
            offset = np.asarray(offset, dtype=odtype)

        if manifolds and (self.__manifolds is not None):

            # Vertices
            for v in self.get_vertices_list():
                v_id = v.get_id()
                v = hold_fun(v.get_geometry().get_densities())
                self.__props_info.set_prop_entry_fast(key_id, (v,), v_id, 1)
            # Edges
            for e in self.get_edges_list():
                s = self.get_vertex(e.get_source_id())
                t = self.get_vertex(e.get_target_id())
                v = hold_fun(np.concatenate((s.get_geometry().get_densities(),
                                            t.get_geometry().get_densities())))
                self.__props_info.set_prop_entry_fast(key_id, (v,), e.get_id(), 1)

        else:

            if neigh is None:

                # Vertices
                for v in self.get_vertices_list():
                    v_id = v.get_id()
                    point = np.asarray(self.__skel.GetPoint(v_id), dtype=odtype) * ibin
                    v = trilin3d(field, offset+point)
                    self.__props_info.set_prop_entry_fast(key_id, (v,), v_id, 1)
                # Edges
                for e in self.get_edges_list():
                    e_id = e.get_id()
                    point = np.asarray(self.__skel.GetPoint(e_id), dtype=odtype) * ibin
                    v = trilin3d(field, offset+point)
                    self.__props_info.set_prop_entry_fast(key_id, (v,), e_id, 1)

            else:

                if seg is None:
                    # Vertices
                    for v in self.get_vertices_list():
                        v_id = v.get_id()
                        point = np.asarray(self.__skel.GetPoint(v_id), dtype=np.float32) * ibin
                        point_f, point_c = np.floor(point).astype(odtype)+offset-neigh, \
                                           np.ceil(point).astype(odtype)+offset+neigh
                        try:
                            hold_field = field[point_f[0]:point_c[0], point_f[1]:point_c[1], point_f[2]:point_c[2]]
                            val = hold_fun(hold_field)
                        except ValueError:
                            val = -1
                        self.__props_info.set_prop_entry_fast(key_id, (val,), v_id, 1)
                    # Edges
                    for e in self.get_edges_list():
                        e_id = e.get_id()
                        point = np.asarray(self.__skel.GetPoint(e_id), dtype=np.float32) * ibin
                        point_f, point_c = np.floor(point).astype(odtype)+offset-neigh, \
                                           np.ceil(point).astype(odtype)+offset+neigh
                        try:
                            hold_field = field[point_f[0]:point_c[0], point_f[1]:point_c[1], point_f[2]:point_c[2]]
                            val = hold_fun(hold_field)
                        except ValueError:
                            val = -1
                        self.__props_info.set_prop_entry_fast(key_id, (val,), e_id, 1)

                else:
                    # Vertices
                    for v in self.get_vertices_list():
                        v_id = v.get_id()
                        point = np.asarray(self.__skel.GetPoint(v_id), dtype=np.float32) * ibin
                        point_f, point_c = np.floor(point).astype(odtype)+offset, np.ceil(point).astype(odtype)+offset
                        hold_field = field[point_f[0]:point_c[0], point_f[1]:point_c[1], point_f[2]:point_c[2]]
                        point = np.round(point).astype(np.int)
                        try:
                            sval = seg[point[0], point[1], point[2]]
                        except IndexError:
                            continue
                        hold_seg = seg[point_f[0]:point_c[0], point_f[1]:point_c[1], point_f[2]:point_c[2]]
                        hold_field = hold_field[hold_seg == sval]
                        try:
                            val = hold_fun(hold_field)
                        except ValueError:
                            val = -1
                        self.__props_info.set_prop_entry_fast(key_id, (val,), v_id, 1)
                    # Edges
                    for e in self.get_edges_list():
                        e_id = e.get_id()
                        point = np.asarray(self.__skel.GetPoint(e_id), dtype=np.float32) * ibin
                        point_f, point_c = np.floor(point).astype(odtype)+offset, np.ceil(point).astype(odtype)+offset
                        hold_field = field[point_f[0]:point_c[0], point_f[1]:point_c[1], point_f[2]:point_c[2]]
                        point = np.round(point).astype(np.int)
                        try:
                            sval = seg[point[0], point[1], point[2]]
                        except IndexError:
                            continue
                        hold_seg = seg[point_f[0]:point_c[0], point_f[1]:point_c[1], point_f[2]:point_c[2]]
                        hold_field = hold_field[hold_seg == sval]
                        try:
                            val = hold_fun(hold_field)
                        except ValueError:
                            val = -1
                        self.__props_info.set_prop_entry_fast(key_id, (val,), e_id, 1)

    # Add scalar field using nearest neighbour so no interpolation is applied
    # field: input tomogram
    # name: property name
    # clean: if True (default) already existing properties are cleaned before
    # bg: if not None (default) field voxels with this value are not added to the graph
    def add_scalar_field_nn(self, field, name, clean=False, bg=None):

        # Initialization
        key_id = self.__props_info.is_already(name)
        if clean or (key_id is None):
            key_id = self.__props_info.add_prop(key=name,
                                       type=disperse_io.TypesConverter().numpy_to_gt(field.dtype),
                                       ncomp=1,
                                       def_val=0)
        if bg is None:
            def_val = field.min()
        else:
            def_val = bg

        # Vertices
        for v in self.get_vertices_list():
            v_id = v.get_id()
            x, y, z = self.get_vertex_coords(v)
            x, y, z = int(round(x)), int(round(y)), int(round(z))
            try:
                hold_val = field[x, y, z]
                if (bg is None) or (hold_val != bg):
                    self.__props_info.set_prop_entry_fast(key_id, (hold_val,), v_id, 1)
            except IndexError:
                self.__props_info.set_prop_entry_fast(key_id, (def_val,), v_id, 1)
        # Edges
        for e in self.get_edges_list():
            e_id = e.get_id()
            x, y, z = self.get_edge_coords(e)
            x, y, z = int(round(x)), int(round(y)), int(round(z))
            try:
                hold_val = field[x, y, z]
                if (bg is None) or (hold_val != bg):
                    self.__props_info.set_prop_entry_fast(key_id, (hold_val,), e_id, 1)
            except IndexError:
                self.__props_info.set_prop_entry_fast(key_id, (def_val,), e_id, 1)

    # Add template matching information (as return by Pytom) to the Graph as Vertex and Edge properties
    # key_prop: string name for the generated properties, sufix '_cc', '_norm' and '_ang' will be added to the two new
    #           generated properties
    # scores: tomogram with the cross correlation map
    # angles: tomogram with the angles index
    # ang_lut: list of angles indexed by angles with the Euler angles rotation (in degrees)
    # t_normal: vector which represent the template normal used
    # d_nhood: diameter of the neighborhood of every vertex (o edge) in nm
    # offset: (default [0, 0, 0]) top-left-right corner coordinates of the sub-volume from which the Graph was computed
    #         respect to field tomogram
    # bin: (default 1.)  tomogram binning respect to the Graph, it allows to compensate resolution differences
    #      between the graph and the input field
    # Results: scores (key_prop+'_cc'), normal rotation (key_prop+'_norm') and rotation angles (key_prop+'_ang') in
    #          degrees are stored as graph properties
    def add_tm_field(self, key_prop, scores, angles, ang_lut, t_normal, d_nhood, offset=(0, 0, 0), bin=1.):

        # Input parsing
        offset = np.asarray(offset, dtype=np.int)
        if bin <= 0:
            error_msg = 'Input bin must be greater than zero, current %s.' % str(bin)
            raise pexceptions.PySegInputError(expr='add_scalar_field (GraphMCF)', msg=error_msg)
        ibin = 1. / float(bin)
        key_prop_s, key_prop_a, key_prop_n = key_prop+'_cc', key_prop+'_ang', key_prop+'_norm'
        data_type = 'float'
        key_id_s = self.__props_info.add_prop(key=key_prop_s, type=data_type, ncomp=1)
        key_id_a = self.__props_info.add_prop(key=key_prop_a, type=data_type, ncomp=3)
        key_id_n = self.__props_info.add_prop(key=key_prop_n, type=data_type, ncomp=3)
        rad = (.5*d_nhood) / self.__resolution # Neighborhood radius in voxels

        # Vertices
        for v in self.get_vertices_list():
            v_id = v.get_id()
            point = np.asarray(self.__skel.GetPoint(v_id), dtype=np.float32) * ibin
            point_f, point_c = np.floor(point-rad).astype(np.int)+offset, np.ceil(point+rad).astype(np.int)+offset
            hold_scores = scores[point_f[0]:point_c[0], point_f[1]:point_c[1], point_f[2]:point_c[2]].flatten()
            hold_angles = angles[point_f[0]:point_c[0], point_f[1]:point_c[1], point_f[2]:point_c[2]].flatten()
            try:
                idv = hold_scores.argmax()
            except ValueError:
                self.__props_info.set_prop_entry_fast(key_id_s, (-1,), v_id, 1)
                self.__props_info.set_prop_entry_fast(key_id_s, (0, 0, 0), v_id, 3)
                continue
            val = hold_scores[idv]
            eu_angs = np.asarray(ang_lut[:, hold_angles[idv], 0], dtype=np.float32)
            norm = rotate_3d_vector(t_normal, eu_angs, deg=True)
            self.__props_info.set_prop_entry_fast(key_id_s, (val,), v_id, 1)
            self.__props_info.set_prop_entry_fast(key_id_a, tuple(eu_angs), v_id, 3)
            self.__props_info.set_prop_entry_fast(key_id_n, tuple(norm), v_id, 3)
        # Edges
        for e in self.get_edges_list():
            e_id = e.get_id()
            point = np.asarray(self.__skel.GetPoint(e_id), dtype=np.float32) * ibin
            point_f, point_c = np.floor(point+rad).astype(np.int)+offset, np.ceil(point+rad).astype(np.int)+offset
            hold_scores = scores[point_f[0]:point_c[0], point_f[1]:point_c[1], point_f[2]:point_c[2]].flatten()
            hold_angles = angles[point_f[0]:point_c[0], point_f[1]:point_c[1], point_f[2]:point_c[2]].flatten()
            try:
                idv = hold_scores.argmax()
            except ValueError:
                self.__props_info.set_prop_entry_fast(key_id_s, (-1,), e_id, 1)
                self.__props_info.set_prop_entry_fast(key_id_s, (0, 0, 0), e_id, 3)
                continue
            val = hold_scores[idv]
            eu_angs = np.asarray(ang_lut[:, hold_angles[idv], 0], dtype=np.float32)
            norm = rotate_3d_vector(t_normal, eu_angs, deg=True)
            self.__props_info.set_prop_entry_fast(key_id_s, (val,), e_id, 1)
            self.__props_info.set_prop_entry_fast(key_id_a, tuple(eu_angs), e_id, 3)
            self.__props_info.set_prop_entry_fast(key_id_n, tuple(norm), e_id, 3)

    # Overwrites method add_tm_field() to receive directly Euler angles
    # key_prop: string name for the generated properties, sufix '_cc' and '_ang' will be added to the two new
    #           generated properties
    # scores: tomogram with the cross correlation map
    # phi|psi|the: tomograms with the euler angles according TOM convention in degrees
    # t_normal: vector which represent the template normal used
    # d_nhood: diameter of the neighborhood of every vertex (o edge) in nm
    def add_tm_field_eu(self, key_prop, scores, phi, psi, the, t_normal, d_nhood):

        # Input parsing
        key_prop_s, key_prop_a, key_prop_n = key_prop+'_cc', key_prop+'_ang', key_prop+'_norm'
        data_type = 'float'
        key_id_s = self.__props_info.add_prop(key=key_prop_s, type=data_type, ncomp=1)
        key_id_a = self.__props_info.add_prop(key=key_prop_a, type=data_type, ncomp=3)
        key_id_n = self.__props_info.add_prop(key=key_prop_n, type=data_type, ncomp=3)
        rad = (.5*d_nhood) / self.__resolution # Neighborhod radius in voxels

        # Vertices
        for v in self.get_vertices_list():
            v_id = v.get_id()
            point = np.asarray(self.__skel.GetPoint(v_id), dtype=np.float32)
            point_f, point_c = np.floor(point-rad).astype(np.int), np.ceil(point+rad).astype(np.int)
            hold_scores = scores[point_f[0]:point_c[0], point_f[1]:point_c[1], point_f[2]:point_c[2]].flatten()
            hold_phi = phi[point_f[0]:point_c[0], point_f[1]:point_c[1], point_f[2]:point_c[2]].flatten()
            hold_psi = psi[point_f[0]:point_c[0], point_f[1]:point_c[1], point_f[2]:point_c[2]].flatten()
            hold_the = the[point_f[0]:point_c[0], point_f[1]:point_c[1], point_f[2]:point_c[2]].flatten()
            try:
                idv = hold_scores.argmax()
            except ValueError:
                self.__props_info.set_prop_entry_fast(key_id_s, (-1,), v_id, 1)
                self.__props_info.set_prop_entry_fast(key_id_s, (0, 0, 0), v_id, 3)
                continue
            val = hold_scores[idv]
            eu_angs = np.asarray((hold_phi[idv], hold_psi[idv], hold_the[idv]), dtype=np.float32)
            norm = rotate_3d_vector(t_normal, eu_angs, deg=True)
            self.__props_info.set_prop_entry_fast(key_id_s, (val,), v_id, 1)
            self.__props_info.set_prop_entry_fast(key_id_a, tuple(eu_angs), v_id, 3)
            self.__props_info.set_prop_entry_fast(key_id_n, tuple(norm), v_id, 3)
        # Edges
        for e in self.get_edges_list():
            e_id = e.get_id()
            point = np.asarray(self.__skel.GetPoint(e_id), dtype=np.float32)
            point_f, point_c = np.floor(point+rad).astype(np.int), np.ceil(point+rad).astype(np.int)
            hold_scores = scores[point_f[0]:point_c[0], point_f[1]:point_c[1], point_f[2]:point_c[2]].flatten()
            hold_phi = phi[point_f[0]:point_c[0], point_f[1]:point_c[1], point_f[2]:point_c[2]].flatten()
            hold_psi = psi[point_f[0]:point_c[0], point_f[1]:point_c[1], point_f[2]:point_c[2]].flatten()
            hold_the = the[point_f[0]:point_c[0], point_f[1]:point_c[1], point_f[2]:point_c[2]].flatten()
            try:
                idv = hold_scores.argmax()
            except ValueError:
                self.__props_info.set_prop_entry_fast(key_id_s, (-1,), e_id, 1)
                self.__props_info.set_prop_entry_fast(key_id_s, (0, 0, 0), e_id, 3)
                continue
            val = hold_scores[idv]
            eu_angs = np.asarray((hold_phi[idv], hold_psi[idv], hold_the[idv]), dtype=np.float32)
            norm = rotate_3d_vector(t_normal, eu_angs, deg=True)
            self.__props_info.set_prop_entry_fast(key_id_s, (val,), e_id, 1)
            self.__props_info.set_prop_entry_fast(key_id_a, tuple(eu_angs), e_id, 3)
            self.__props_info.set_prop_entry_fast(key_id_n, tuple(norm), e_id, 3)

    # If this property already exists it is overwritten
    def add_prop(self, key, type, ncomp, def_val=-1):
        return self.__props_info.add_prop(key, type, ncomp, def_val)

    # Add a new property with by inverting the values of a previous one
    # the final key will be the original + '_inv'
    # edg: if True (default False) edges are also considered
    def add_prop_inv(self, key, edg=False):

        # Initialization
        key_inv = key + '_inv'
        key_id = self.get_prop_id(key)
        if key_id is None:
            error_msg = 'No property with key ' + key + ' found!'
            raise pexceptions.PySegInputError(expr='add_prop_inv (GraphMCF)', msg=error_msg)
        key_id_inv = self.get_prop_id(key_inv)
        if key_id_inv is not None:
            error_msg = 'Inverted property ' + key_inv + ' already exists!'
            raise pexceptions.PySegInputError(expr='add_prop_inv (GraphMCF)', msg=error_msg)
        n_comp = self.get_prop_ncomp(key_id=key_id)
        if n_comp != 1:
            error_msg = 'Input property ' + key + ' has ' + str(n_comp) + ' components, only 1 valid!'
            raise pexceptions.PySegInputError(expr='add_prop_inv (GraphMCF)', msg=error_msg)
        d_type_gt = self.get_prop_type(key_id=key_id)
        d_type = disperse_io.TypesConverter().gt_to_numpy(d_type_gt)
        key_id_inv = self.add_prop(key_inv, d_type_gt, 1)

        # Get property array
        if edg:
            vertices, edges = self.get_vertices_list(), self.get_edges_list()
            array = np.zeros(shape=len(vertices)+len(edges), dtype=d_type)
            for i in range(len(vertices)):
                array[i] = self.get_prop_entry_fast(key_id, vertices[i].get_id(), 1, d_type)[0]
            for i, j in zip(range(len(vertices), len(vertices)+len(edges)), range(len(edges))):
                array[i] = self.get_prop_entry_fast(key_id, edges[j].get_id(), 1, d_type)[0]
        else:
            vertices = self.get_vertices_list()
            array = np.zeros(shape=len(vertices), dtype=d_type)
            for i, v in enumerate(vertices):
                array[i] = self.get_prop_entry_fast(key_id, v.get_id(), 1, d_type)[0]

        # Invertion
        array_in = lin_map(array, lb=array.max(), ub=array.min())

        # Set inverted property array
        if edg:
            for i in range(len(vertices)):
                self.set_prop_entry_fast(key_id_inv, (array_in[i],), vertices[i].get_id(), 1)
            for i, j in zip(range(len(vertices), len(vertices)+len(edges)), range(len(edges))):
                self.set_prop_entry_fast(key_id_inv, (array_in[i],), edges[j].get_id(), 1)
        else:
            for i, v in enumerate(vertices):
                self.set_prop_entry_fast(key_id_inv, (array_in[i],), v.get_id(), 1)

    # Eliminates self loops edges to vertices
    def filter_self_edges(self):
        for e in self.get_edges_list():
            if e.get_source_id() == e.get_target_id():
                self.remove_edge(e)

    # Eliminates repeated edges (keep those with minimum field value)
    def filter_repeated_edges(self):

        key_id = self.__props_info.is_already(STR_FIELD_VALUE)
        data_type = self.__props_info.get_type(index=key_id)
        data_type = disperse_io.TypesConverter().gt_to_numpy(data_type)
        for v in self.get_vertices_list():
            neighs, edges = self.get_vertex_neighbours(v.get_id())
            neigh_ids = list()
            neigh_count = list()
            for i, n in enumerate(neighs):
                n_id = n.get_id()
                try:
                    hold_count = neigh_ids.index(n_id)
                except:
                    neigh_ids.append(n_id)
                    neigh_count.append(i)
                    continue
                curr_edge = edges[i]
                hold_edge = edges[neigh_count[hold_count]]
                curr_field = self.__props_info.get_prop_entry_fast(key_id,
                                                                   curr_edge.get_id(),
                                                                   1, data_type)
                hold_field = self.__props_info.get_prop_entry_fast(key_id,
                                                                   hold_edge.get_id(),
                                                                   1, data_type)
                if curr_field < hold_field:
                    neigh_count[hold_count] = i
                    self.remove_edge(hold_edge)
                else:
                    self.remove_edge(curr_edge)

    # Eliminates edges which are in missing wedge area
    # In this version only tilt axes perpendicular to XY are considered
    # wr_ang: wedge rotation angle in degrees [-90, 90]
    # tilt_ang: maximum tilt angle in degrees [0, 90]
    def filter_mw_edges(self, wr_ang, tilt_ang=0):

        # Precompute wedge vectors
        rho = np.radians(wr_ang)
        phi = np.radians(90 - tilt_ang)
        phi2 = np.pi - phi
        z = np.array((0, 0, 1))
        r = np.array((np.cos(rho), np.sin(rho), 0))

        # Loop for checking the edges
        warnings.filterwarnings('error')
        for e in self.get_edges_list():
            v0_id = e.get_source_id()
            v1_id = e.get_target_id()
            p0 = self.__skel.GetPoint(v0_id)
            p1 = self.__skel.GetPoint(v1_id)
            # Point vector w
            w = np.array(p1) - np.array(p0)
            # Projection of w on the plane with r as normal
            w_p = np.cross(r, np.cross(w, r))
            w_p_norm = np.sqrt(np.sum(w_p * w_p))
            try:
                # Getting angle between z and the projection of w
                w_p_norm_inv = 1 / w_p_norm
                phi_p = np.arccos(np.dot(w_p, z) * w_p_norm_inv)
            except RuntimeWarning:
                continue
            if (phi_p <= phi) or (phi_p >= phi2):
                self.remove_edge(e)

    # Import a prop from another
    def import_prop(self, graph, prop_key):

        # Add property to GraphMCF
        key_id_g = graph.get_prop_id(prop_key)
        d_type_gt = graph.get_prop_type(key_id=key_id_g)
        d_type = disperse_io.TypesConverter().gt_to_numpy(d_type_gt)
        n_comp = graph.get_prop_ncomp(key_id=key_id_g)
        key_id = self.get_prop_id(prop_key)
        if key_id is None:
            key_id = self.add_prop(prop_key, d_type_gt, n_comp)

        for v in graph.get_vertices_list():
            v_id = v.get_id()
            if self.get_vertex(v_id) is not None:
                t = graph.get_prop_entry_fast(key_id_g, v_id, n_comp, d_type)
                self.set_prop_entry_fast(key_id, t, v_id, n_comp)

        for e in graph.get_edges_list():
            e_id = e.get_id()
            if self.get_edge(e_id) is not None:
                t = graph.get_prop_entry_fast(key_id_g, e_id, n_comp, d_type)
                self.set_prop_entry_fast(key_id, t, e_id, n_comp)

    # Compute percentile threshold from a vertex or edge property
    # prop_key: property key, only one component properties are valid
    # vertex: True for vertex properties and False for edge properties
    # per_ct: percentile (%) criteria for finding the threshold
    def find_per_th(self, prop_key, vertex, per_ct):

        # Initialization
        key_id = self.get_prop_id(prop_key)
        if vertex:
            tokens = self.get_vertices_list()
        else:
            tokens = self.get_edges_list()
        n_comp = self.get_prop_ncomp(key_id=key_id)
        if n_comp != 1:
            error_msg = 'Only 1 component properties are valid!'
            raise pexceptions.PySegInputError(expr='__find_per_th (GraphMCF)', msg=error_msg)
        data_type = self.get_prop_type(key_id=key_id)

        # Loop for storing properties into an array
        type_np = disperse_io.TypesConverter().gt_to_numpy(data_type)
        arr = np.zeros(shape=len(tokens), dtype=type_np)
        for i, token in enumerate(tokens):
            arr[i] = self.__props_info.get_prop_entry_fast(key_id, token.get_id(), 1, type_np)[0]

        # Compute percentile
        return np.percentile(arr, per_ct)

    # Compute the bounding box which contains all vertices (minima) and edges (saddle points)
    # Returns: bounding box as [x_min, y_min, z_min, x_max, y_max, z_max] tuple
    def compute_bbox(self):

        # Initialization
        MAX_F = np.finfo(np.float).max
        MIN_F = np.finfo(np.float).min
        x_min, y_min, z_min = MAX_F, MAX_F, MAX_F
        x_max, y_max, z_max = MIN_F, MIN_F, MIN_F

        # Vertices
        for v in self.get_vertices_list():
            x, y, z = self.__skel.GetPoint(v.get_id())
            if x < x_min:
                x_min = x
            if y < y_min:
                y_min = y
            if z < z_min:
                z_min = z
            if x > x_max:
                x_max = x
            if y > y_max:
                y_max = y
            if z > z_max:
                z_max = z

        # Edges
        for e in self.get_edges_list():
            x, y, z = self.__skel.GetPoint(e.get_id())
            if x < x_min:
                x_min = x
            if y < y_min:
                y_min = y
            if z < z_min:
                z_min = z
            if x > x_max:
                x_max = x
            if y > y_max:
                y_max = y
            if z > z_max:
                z_max = z

        return x_min, y_min, z_min, x_max, y_max, z_max

    # Returns graph global statistics, vertices and edges per volume and ratio edges/vertices
    # mask: binary mask (default None) where True is fg for subvolume computation
    # Returns: a tuple of three scalars (see above)
    def compute_global_stat(self, mask=None):

        # Getting valid vertices, edges and volume dimension
        if mask is None:
            vertices = self.get_vertices_list()
            edges = self.get_edges_list()
            x_min, y_min, z_min, x_max, y_max, z_max = self.compute_bbox()
            # 3D embedded graph
            if z_max > 0:
                vol = (x_max-x_min) * (y_max-y_min) * (z_max-z_min)
                vol *= float(self.get_resolution() * self.get_resolution() * self.get_resolution())
            # 2D embedded graph
            else:
                vol = (x_max-x_min) * (y_max-y_min)
                vol *= float(self.get_resolution() * self.get_resolution())
        else:
            vertices = list()
            for v in self.get_vertices_list():
                x, y, z = self.get_vertex_coords(v)
                try:
                    if mask[int(round(x)), int(round(y)), int(round(z))]:
                        vertices.append(v)
                except IndexError:
                    pass
            edges = list()
            for e in self.get_edges_list():
                x, y, z = self.get_edge_coords(e)
                try:
                    if mask[int(round(x)), int(round(y)), int(round(z))]:
                        edges.append(e)
                except IndexError:
                    pass
            bin_mask = np.asarray(mask, dtype=np.bool)
            vol = float(bin_mask.sum())
            vol *= (self.get_resolution() * self.get_resolution() * self.get_resolution())

        # Computing statistics
        n_verts = float(len(vertices))
        n_edges = float(len(edges))
        if vol <= 0:
            return 0., 0., 0.
        elif n_verts == 0:
            return 1., 0., 0.
        else:
            return n_verts/vol, n_edges/vol, n_edges/n_verts

    # Graph simplification until reaching specified vertex and edge density
    # v_num: number of vertices after simplification
    # e_num: number of edges after simplification
    # v_den: vertex density (vertex/nm^3), if None (default) no vertex is deleted, only applicable if e_den is None
    # e_den: edge density (edge/nm^3), if None (default) no vertex is deleted, only applicable if e_den is None
    # v_prop: key string for vertex simplification property, if None (default) topological
    #         simplification is applied
    # e_prop: key string for vertex simplification property (default STR_VERT_DST)
    # v_mode: if 'high' (default) then vertices with highest property values are preserved,
    #       otherwise those with the lowest
    # e_mode: the same as v_mode but for vertices
    # mask: binary mask, True is fg, for setting region where the graph has representation,
    #       by default, None, bounding box is used but for precise computation is should be passed.
    #       If mask is not None, the graph is thresholded according to the mask
    # Returns: GraphMCF is filtered, and warning exception is raised if the a demanded density
    #          cannot be reached
    def graph_density_simp(self, v_num=None, e_num=None, v_den=None, e_den=None, v_prop=None,
                           e_prop=STR_VERT_DST, v_mode='high', e_mode='high', mask=None):

        # Compute valid region volume (nm^3)
        if (v_num is None) or (e_num is None):
            res3 = float(self.get_resolution() * self.get_resolution() * self.get_resolution())
            if mask is None:
                x_min, y_min, z_min, x_max, y_max, z_max = self.compute_bbox()
                if z_max > 0:
                    vol = (x_max-x_min) * (y_max-y_min) * (z_max-z_min) * res3
                else:
                    vol = (x_max-x_min) * (y_max-y_min) * self.get_resolution() * self.get_resolution()
            else:
                bin_mask = np.asarray(mask, dtype=np.bool)
                vol = float(bin_mask.sum()) * res3
                # Remove vertices and edges out of the mask
                for v in self.get_vertices_list():
                    x, y, z = self.get_vertex_coords(v)
                    try:
                        if not bin_mask[int(round(x)), int(round(y)), int(round(z))]:
                            self.remove_vertex(v)
                    except IndexError:
                        pass
                for e in self.get_edges_list():
                    x, y, z = self.get_edge_coords(e)
                    try:
                        if not bin_mask[int(round(x)), int(round(y)), int(round(z))]:
                            self.remove_edge(e)
                    except IndexError:
                        pass
            if vol == 0:
                error_msg = 'Valid region has null volume.'
                raise pexceptions.PySegInputWarning(expr='graph_density_simp (GraphMCF)', msg=error_msg)

        # Vertices simplification
        v_err = False
        n_tverts = None
        if v_num is not None:
            n_tverts = v_num
        elif v_den is not None:
            # Compute target number of vertices
            vertices = self.get_vertices_list()
            n_verts = float(len(vertices))
            n_tverts = int(round(v_den * vol))
            if n_verts <= n_tverts:
                n_tverts = int(n_verts)
                v_err = True

        # Delete vertices for reaching the target number
        if n_tverts is not None:
            if v_prop is None:
                self.topological_simp(n=n_tverts)
            else:
                self.threshold_vertices_n(n=n_tverts, prop=v_prop, mode=v_mode)

        # Edges simplification
        e_err = False
        n_tedgs = None
        if e_num is not None:
            n_tedgs = e_num
        elif e_den is not None:
            # Compute target number of edges
            edges = self.get_edges_list()
            n_edgs = float(len(edges))
            n_tedgs = int(round(e_den * vol))
            if n_edgs < n_tedgs:
                n_tedgs = int(n_edgs)
                e_err = True

        # Delete vertices for reaching the target number
        if n_tedgs is not None:
            self.threshold_edges_n(n=n_tedgs, prop=e_prop, mode=e_mode)

        # Raise warnings
        if v_err:
            curr_res = n_verts / vol
            error_msg = 'Demanded resolution for vertices could not be reached. \n'
            error_msg += 'Current vertex resolution is ' + str(curr_res) + ' vox/nm^3, '
            error_msg += 'asked ' + str(v_den) + ' vox/nm^3.'
            raise pexceptions.PySegInputWarning(expr='graph_density_simp (GraphMCF) \n', msg=error_msg)
        if e_err:
            curr_res = n_edgs / vol
            error_msg = 'Demanded resolution for edges could not be reached. '
            error_msg += 'Current edge resolution is ' + str(curr_res) + ' vox/nm^3, '
            error_msg += 'asked ' + str(e_den) + ' vox/nm^3.'
            raise pexceptions.PySegInputWarning(expr='graph_density_simp (GraphMCF) \n', msg=error_msg)

    # Graph simplification until reaching specified vertex and edge density from a reference
    # maks: binary mask for setting the reference
    # v_den: vertex density (vertex/nm^3), if None (default) no vertex is deleted, only applicable if e_den is None
    # e_den: edge density (edge/nm^3), if None (default) no vertex is deleted, only applicable if e_den is None
    # v_prop: key string for vertex simplification property, if None (default) topological
    #         simplification is applied
    # e_prop: key string for vertex simplification property (default STR_VERT_DST)
    # v_mode: if 'high' (default) then vertices with highest property values are preserved,
    #       otherwise those with the lowest
    # e_mode: the same as v_mode but for vertices
    # fit: if True (default None) only edges in reference binary mask are processed
    #      (requires prop_ref not None)
    # Returns: GraphMCF is filtered, and warning exception is raised if the a demanded density
    #          cannot be reached
    def graph_density_simp_ref(self, mask, v_den=None, e_den=None, v_prop=None, e_prop=None,
                               v_mode='high', e_mode='high', fit=False):

        # Compute valid region volume (nm^3)
        bin_mask = np.asarray(mask, dtype=np.bool)
        res3 = float(self.get_resolution() * self.get_resolution() * self.get_resolution())
        vol = float(bin_mask.sum()) * res3
        if vol == 0:
            error_msg = 'Valid region has null volume.'
            raise pexceptions.PySegInputWarning(expr='graph_density_simp_ref (GraphMCF)',
                                                msg=error_msg)

        # Add mask property
        self.add_scalar_field_nn(mask, STR_SIMP_MASK)
        ref_key_id = self.get_prop_id(STR_SIMP_MASK)

        # Vertices simplification
        v_err = False
        n_tverts = None
        if v_den is not None:
            # Compute target number of vertices
            n_tverts = int(round(v_den * vol))
            # Compute current number of vertices at the reference region
            n_verts = 0
            for v in self.get_vertices_list():
                if self.__props_info.get_prop_entry_fast(ref_key_id, v.get_id(), 1, mask.dtype.type)[0] > 0:
                    n_verts += 1
            if n_verts < n_tverts:
                n_tverts = int(n_verts)
                v_err = True

        # Delete vertices for reaching the target number
        if n_tverts is not None:
            if v_prop is None:
                self.topological_simp(n=n_tverts, prop_ref=STR_SIMP_MASK)
            else:
                self.threshold_vertices_n(n=n_tverts, prop=v_prop, mode=v_mode, prop_ref=STR_SIMP_MASK)

        # Edges simplification
        e_err = False
        n_tedgs = None
        if e_den is not None:
            # Compute target number of edges
            n_tedgs = int(round(e_den * vol))
            # Compute current number of edges at the reference region
            n_edgs = 0
            for e in self.get_edges_list():
                if self.__props_info.get_prop_entry_fast(ref_key_id, e.get_id(), 1, mask.dtype.type)[0] > 0:
                    n_edgs += 1
            if n_edgs < n_tedgs:
                n_tedgs = int(n_edgs)
                e_err = True

        # Delete vertices for reaching the target number
        if n_tedgs is not None:
            self.threshold_edges_n(n=n_tedgs, prop=e_prop, mode=e_mode, prop_ref=STR_SIMP_MASK,
                                   fit=fit)

        # Raise warnings
        if v_err:
            curr_res = n_verts / vol
            error_msg = 'Demanded resolution for vertices could not be reached. \n'
            error_msg += 'Current vertex resolution is ' + str(curr_res) + ' vox/nm^3, '
            error_msg += 'asked ' + str(v_den) + ' vox/nm^3.'
            raise pyseg.pexceptions.PySegInputWarning(expr='graph_density_simp_ref (GraphMCF)', msg=error_msg)
        if e_err:
            curr_res = n_edgs / vol
            error_msg = 'Demanded resolution for edges could not be reached. \n'
            error_msg += 'Current edge resolution is ' + str(curr_res) + ' vox/nm^3, '
            error_msg += 'asked ' + str(e_den) + ' vox/nm^3.'
            raise pyseg.pexceptions.PySegInputWarning(expr='graph_density_simp_ref (GraphMCF) \n', msg=error_msg)


    # For al edges computes its total curvatures (as curves in space)
    # Returns: properties STR_EDGE_UK, STR_EDGE_K, STR_EDGE_UT and STR_EDGE_T
    def compute_edge_curvatures(self):

        # Create the new properties
        key_id_uk = self.add_prop(STR_EDGE_UK, disperse_io.TypesConverter().numpy_to_gt(np.float), 1)
        key_id_ns = self.add_prop(STR_EDGE_NS, disperse_io.TypesConverter().numpy_to_gt(np.float), 1)
        key_id_ut = self.add_prop(STR_EDGE_UT, disperse_io.TypesConverter().numpy_to_gt(np.float), 1)
        key_id_bs = self.add_prop(STR_EDGE_BNS, disperse_io.TypesConverter().numpy_to_gt(np.float), 1)
        key_id_sin = self.add_prop(STR_EDGE_SIN, disperse_io.TypesConverter().numpy_to_gt(np.float), 1)
        key_id_al = self.add_prop(STR_EDGE_APL, disperse_io.TypesConverter().numpy_to_gt(np.float), 1)

        # Main loop for curvatures computation
        for e in self.get_edges_list():
            e_id = e.get_id()
            curve = diff_geom.SpaceCurve(self.get_edge_arcs_coords(e, no_repeat=True))
            self.set_prop_entry_fast(key_id_uk, (curve.get_total_uk(),), e_id, 1)
            self.set_prop_entry_fast(key_id_ns, (curve.get_normal_symmetry(),), e_id, 1)
            self.set_prop_entry_fast(key_id_ut, (curve.get_total_ut(),), e_id, 1)
            self.set_prop_entry_fast(key_id_bs, (curve.get_binormal_symmetry(),), e_id, 1)
            self.set_prop_entry_fast(key_id_sin, (curve.get_sinuosity(),), e_id, 1)
            self.set_prop_entry_fast(key_id_al, (curve.get_apex_length()*self.get_resolution(),), e_id, 1)

    # CLAHE property equalization, it does not work with negative values
    # (edge properties are interpolated from vertex properties)
    # prop_key: property key for equalizing
    #### CLAHE settings
    # N: number of grayscales (default 256)
    # clip_f: clipping factor in percentage (default 100)
    # s_max: maximum slop (default 4)
    # Returns: a new property called "prop_key+'_eq'" with the equalization in the range of ints [0,N]
    def clahe_prop(self, prop_key, N=256, clip_f=100, s_max=4):

        # Initialization
        graph = GraphGT(self)
        graph_gt = graph.get_gt()
        prop_v, prop_e = None, None
        try:
            prop_v = graph_gt.vertex_properties[prop_key]
            prop_v_a = prop_v.get_array()
            prop_v_eq = graph_gt.new_vertex_property('float')
            if prop_v_a.min() < 0:
                error_msg = 'Properties with negative values are not valid, nothing can be done.'
                raise pexceptions.PySegInputWarning(expr='clahe_prop (GraphMCF)',
                                                    msg=error_msg)
        except:
            pass
        try:
            prop_e = graph_gt.edge_properties[prop_key]
            prop_e_eq = graph_gt.new_edge_property('float')
            if prop_v_a.min() < 0:
                error_msg = 'Properties with negative values are not valid, nothing can be done.'
                raise pexceptions.PySegInputWarning(expr='clahe_prop (GraphMCF)',
                                                    msg=error_msg)
        except:
            pass
        if (prop_v is None) and (prop_e is None):
            error_msg = 'Property ' + prop_key + ' does not exist, nothing can be done.'
            raise pexceptions.PySegInputWarning(expr='clahe_prop (GraphMCF)',
                                                msg=error_msg)
        prop_w = graph_gt.edge_properties[SGT_EDGE_LENGTH]
        prop_key_eq = prop_key + '_eq'
        if N > graph_gt.num_vertices():
            N = graph_gt.num_vertices()

        # Measuring geodesic distances
        dists_map = gt.shortest_distance(graph_gt, weights=prop_w)

        if prop_v is not None:
            # Vertices loop
            for v in graph_gt.vertices():
                ids = np.argsort(dists_map[v].get_array())
                trans, x_arr = clahe_array(prop_v_a[ids[:N]], N, clip_f, s_max)
                hold = prop_v[v] - x_arr
                idx = np.argmin(hold * hold)
                prop_v_eq[v] = trans[idx]
            graph_gt.vertex_properties[prop_key_eq] = prop_v_eq
        if prop_e is not None:
            # Edges loop
            for e in graph_gt.edges():
                s, t = e.source(), e.target()
                # Vertex values interpolation
                prop_e_eq[e] = prop_e[e] * ((prop_v_eq[s]+prop_v_eq[t]) / float((prop_v[s]+prop_v[t])))
            graph_gt.edge_properties[prop_key_eq] = prop_e_eq

        # Insert properties to GraphMCF
        graph.add_prop_to_GraphMCF(self, prop_key_eq, up_index=True)

    # Special CLAHE adaptation for equalizing field_value property
    #### CLAHE settings
    # max_geo_dist: maximum geodesic radius for neighborhood
    # N: number of grayscales (default 256)
    # clip_f: clipping factor in percentage (default 100)
    # s_max: maximum slop (default 4)
    # Returns: a new property called "prop_key+'_eq'" with the equalization in the range of ints [0,N]
    def clahe_field_value(self, max_geo_dist, N=256, clip_f=100, s_max=4):

        # Initialization
        graph = GraphGT(self)
        graph_gt = graph.get_gt()
        prop_v = graph_gt.vertex_properties[STR_FIELD_VALUE]
        prop_v_a = prop_v.get_array()
        prop_v_eq = graph_gt.new_vertex_property('float')
        prop_e = graph_gt.edge_properties[STR_FIELD_VALUE]
        prop_e_eq = graph_gt.new_edge_property('float')
        prop_e_h = graph_gt.new_edge_property('float')
        prop_e_v = graph_gt.new_edge_property('float')
        prop_e_h.get_array()[:] = np.zeros(shape=graph_gt.num_edges(), dtype=np.float)
        prop_e_v.get_array()[:] = np.zeros(shape=graph_gt.num_edges(), dtype=np.float)
        prop_w = graph_gt.edge_properties[SGT_EDGE_LENGTH]
        prop_v_id = graph_gt.vertex_properties[DPSTR_CELL]
        prop_key_eq = 'field_value_eq'
        N_m = float(N - 1)
        if N_m <= 0:
            error_msg = 'Number of greyvalues must be greater than 1.'
            raise pexceptions.PySegInputWarning(expr='clahe_field_value (GraphMCF)',
                                                msg=error_msg)
        N_inv = 1. / N_m

        # Measuring geodesic distances
        dists_map = gt.shortest_distance(graph_gt, weights=prop_w)

        # Vertex equalization by CLAHE
        if prop_v is not None:
            # Vertices loop
            for v in graph_gt.vertices():
                map_arr = dists_map[v].get_array()
                ids = np.where((map_arr < max_geo_dist) & (map_arr > 0))[0]
                # Getting densities array
                arr = self.get_vertex(prop_v_id[v]).get_geometry().get_densities()
                for idx in ids:
                    hold_v = graph_gt.vertex(idx)
                    hold_arr = self.get_vertex(prop_v_id[hold_v]).get_geometry().get_densities()
                    arr = np.concatenate((arr, hold_arr))
                # Vertex equalization
                arr *= N_m
                trans = clahe_array2(arr, N, clip_f, s_max)
                hold_val = int(round(N_m*prop_v[v]))
                prop_v_eq[v] = trans[hold_val]
                # Edge equalization
                for e in v.out_edges():
                    if prop_e_v[e] < prop_v_eq[v]:
                        prop_e_v[e] = prop_v_eq[v]
                    hold_val = int(round(N_m*prop_e[e]))
                    prop_e_eq[e] += (.5 * trans[hold_val])

        # Edge regularization
        for e in graph_gt.edges():
            if prop_e_v[e] > prop_e_eq[e]:
                prop_e_eq[e] = prop_e_v[e]

        # Normalization
        prop_v_eq.get_array()[:] = N_inv * prop_v_eq.get_array()
        graph_gt.vertex_properties[prop_key_eq] = prop_v_eq
        prop_e_eq.get_array()[:] = N_inv * prop_e_eq.get_array()
        graph_gt.edge_properties[prop_key_eq] = prop_e_eq

        # Insert properties to GraphMCF
        graph.add_prop_to_GraphMCF(self, prop_key_eq, up_index=True)

    # Special CLAHE adaptation for equalizing field_value property along the embedded graph skeleton
    #### CLAHE settings
    # max_geo_dist: maximum geodesic radius for neighborhood
    # N: number of greyscales (default 256)
    # clip_f: clipping factor in percentage (default 100)
    # s_max: maximum slop (default 4)
    # Returns: a new property called "prop_key+'_eq'" with the equalization in the range of ints [0,1]
    def clahe_field_value_skel(self, max_geo_dist, N=256, clip_f=100, s_max=4):

        # Initialization
        graph = GraphGT(self)
        graph_gt = graph.get_gt()
        prop_v = graph_gt.vertex_properties[STR_FIELD_VALUE]
        prop_v_eq = graph_gt.new_vertex_property('float')
        prop_e = graph_gt.edge_properties[STR_FIELD_VALUE]
        prop_e_eq = graph_gt.new_edge_property('float')
        prop_e_h = graph_gt.new_edge_property('float')
        prop_e_v = graph_gt.new_edge_property('float')
        prop_e_h.get_array()[:] = np.zeros(shape=graph_gt.num_edges(), dtype=np.float)
        prop_e_v.get_array()[:] = np.zeros(shape=graph_gt.num_edges(), dtype=np.float)
        prop_w = graph_gt.edge_properties[SGT_EDGE_LENGTH]
        prop_v_id = graph_gt.vertex_properties[DPSTR_CELL]
        prop_key_eq = 'field_value_eq'
        N_m = float(N - 1)
        if N_m <= 0:
            error_msg = 'Number of greyvalues must be greater than 1.'
            raise pexceptions.PySegInputWarning(expr='clahe_field_value_skel (GraphMCF)',
                                                msg=error_msg)
        N_inv = 1. / N_m
        n_samp = 10 # Number of samples for every edge (function cte)

        # Measuring geodesic distances
        dists_map = gt.shortest_distance(graph_gt, weights=prop_w)

        # Vertex equalization by CLAHE
        if prop_v is not None:

            count, mx_count = 0., float(graph_gt.num_vertices())

            # Vertices loop
            for v in graph_gt.vertices():
                # Getting neighbourhood vertices
                map_arr = dists_map[v].get_array()
                ids = np.where((map_arr < max_geo_dist) & (map_arr > 0))[0]
                # Getting neighbourhood edges
                n_edges = list()
                for idx in ids:
                    hold_v = graph_gt.vertex(idx)
                    n_edges += self.get_vertex_neighbours(prop_v_id[hold_v])[1]
                # Getting field values on edges skeleton
                arr = np.ones(shape=(len(n_edges)*n_samp), dtype=np.float32)
                for i, e in enumerate(n_edges):
                    arr[i:i+n_samp] = self.get_edge_skel_field(e, no_repeat=True, f_len=n_samp)
                # Equalization map
                arr *= N_m
                trans = clahe_array2(arr, N, clip_f, s_max)
                # Vertex equalization
                hold_val = int(round(N_m*prop_v[v]))
                prop_v_eq[v] = trans[hold_val]
                # Edge equalization
                for e in v.out_edges():
                    hold_val = int(round(N_m*prop_e[e]))
                    prop_e_eq[e] = trans[hold_val]

                count += 1
                print 'CLAHE skel: progress ' + str(round((count/mx_count)*100., 2)) + ' %'

        # Normalization
        prop_v_eq.get_array()[:] = N_inv * prop_v_eq.get_array()
        graph_gt.vertex_properties[prop_key_eq] = prop_v_eq
        prop_e_eq.get_array()[:] = N_inv * prop_e_eq.get_array()
        graph_gt.edge_properties[prop_key_eq] = prop_e_eq

        # Insert properties to GraphMCF
        graph.add_prop_to_GraphMCF(self, prop_key_eq, up_index=True)

    def compute_graph_gt(self):
        self.__graph_gt = self.get_gt(fupdate=True)

    # Decimate vertices to keep those one with highest betweenss
    # dec: decimation factor, ex. 5
    # graph: input GraphGT, if None (default) it is computed
    # key_v: property key for weighting vertieces
    # key_e: property key for weighting edges
    # gt_update: if True (default False) GraphGT computation is forced
    def bet_decimation(self, dec, graph=None, key_v=None, key_e=None, gt_update=True):

        dec = float(dec)
        if dec <= 0:
            error_msg = 'Decimation factor must be higher that zeros.'
            raise pexceptions.PySegInputWarning(expr='bet_decimation (GraphMCF)', msg=error_msg)

        # Compute GraphGT
        if graph is None:
            graph = GraphGT(self)
        graph_gt = graph.get_gt()
        in_nv = graph_gt.num_vertices()
        d_nv = in_nv - math.ceil(in_nv/dec)

        # Compute betweeness
        graph.betweenness(mode='vertex', prop_name=SGT_BETWEENNESS, prop_v=key_v, prop_e=key_e)
        arr_id, arr_bet = graph_gt.vp[DPSTR_CELL].get_array(), graph_gt.vp[SGT_BETWEENNESS].get_array()

        # Insert properties to GraphMCF
        graph.add_prop_to_GraphMCF(self, SGT_BETWEENNESS, up_index=True)

        # Finding vertices ids with the highest betweeness
        sort_ids = np.argsort(arr_bet)

        # Vertices thresholding
        for idx in sort_ids[:d_nv]:
            self.remove_vertex(self.get_vertex(arr_id[idx]))

    # Make any operation in operator package with supports two operators (on vertices and edges)
    # Only valid for properties with one number of components
    # key_a|b: string key for the two input properties (a and b)
    # key_c: string key where the output will be stored
    # op: input operator
    def two_props_operator(self, key_a, key_b, key_c, op):

        # Initialization
        key_a_id = self.get_prop_id(key_a)
        a_ncomp = self.get_prop_ncomp(key_id=key_a_id)
        a_type = disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=key_a_id))
        key_b_id = self.get_prop_id(key_b)
        b_ncomp = self.get_prop_ncomp(key_id=key_b_id)
        b_type = disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=key_b_id))
        key_c_id = self.get_prop_id(key_c)
        c_ncomp = self.get_prop_ncomp(key_id=key_c_id)
        c_type = disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=key_c_id))
        if (a_ncomp != 1) or (b_ncomp != 1) or (c_ncomp != 1):
            error_msg = 'Only input properties with 1 component are valid.'
            raise pexceptions.PySegInputError(expr='two_props_operator (GraphMCF)',
                                              msg=error_msg)

        # Loops for operation
        for v in self.get_vertices_list():
            v_id = v.get_id()
            val_a = self.get_prop_entry_fast(key_a_id, v_id, 1, a_type)[0]
            val_b = self.get_prop_entry_fast(key_b_id, v_id, 1, b_type)[0]
            val_c = c_type(op(val_a, val_b))
            self.set_prop_entry_fast(key_c_id, (val_c,), v_id, 1)
        for e in self.get_edges_list():
            e_id = e.get_id()
            val_a = self.get_prop_entry_fast(key_a_id, e_id, 1, a_type)[0]
            val_b = self.get_prop_entry_fast(key_b_id, e_id, 1, b_type)[0]
            val_c = c_type(op(val_a, val_b))
            self.set_prop_entry_fast(key_c_id, (val_c,), e_id, 1)

    # Make any operation in operator package from input cte and all items addressed by prop_key
    # The output is stored (accumulated) in the same property
    # prop_key: string key for the input/output property
    # cte: tuple with input ctes (number of components must be equal to the number of componentes of the property)
    # op: input operator
    def prop_cte_operator(self, prop_key, cte, op):

        # Parsing
        key_id = self.get_prop_id(prop_key)
        if key_id is None:
            error_msg = 'Property ' + prop_key + ' not found!'
            raise pexceptions.PySegInputError(expr='prop_cte_operator (GraphMCF)', msg=error_msg)
        n_comp = self.get_prop_ncomp(key_id=key_id)
        if n_comp != len(cte):
            error_msg = 'Input cte has the same number of componentas than the selected property.'
            raise pexceptions.PySegInputError(expr='prop_cte_operator (GraphMCF)', msg=error_msg)
        c_type = disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=key_id))
        c_cte = list()
        for i in range(len(cte)):
            c_cte.append(cte[i])

        # Loops for operation
        for v in self.get_vertices_list():
            v_id = v.get_id()
            val_in = self.get_prop_entry_fast(key_id, v_id, 1, c_type)[0]
            val_out = list()
            for i in range(n_comp):
                val_out.append(op(c_cte[i], val_in[i]))
            self.set_prop_entry_fast(key_id, tuple(val_out), v_id, 1)
        for e in self.get_edges_list():
            e_id = e.get_id()
            val_in = self.get_prop_entry_fast(key_id, v_id, 1, c_type)[0]
            val_out = list()
            for i in range(n_comp):
                val_out.append(op(c_cte[i], val_in[i]))
            self.set_prop_entry_fast(key_id, tuple(val_out), e_id, 1)

    # Generates a binary mask where graph vertices or edges arc points are set to True
    # verts: if True (default) vertices points are printed, otherwise edge arc points
    def to_mask(self, verts=True):

        # Initialization
        mask = np.zeros(shape=self.__density.shape, dtype=np.bool)

        if verts:
            for v in self.get_vertices_list():
                x, y, z = self.get_vertex_coords(v)
                try:
                    mask[int(round(x)), int(round(y)), int(round(z))] = True
                except IndexError:
                    pass
        else:
            for e in self.get_edges_list():
                for coord in self.get_edge_arcs_coords(e):
                    try:
                        mask[int(round(coord[0])), int(round(coord[1])), int(round(coord[2]))] = True
                    except IndexError:
                        pass

        return mask

    # Compute vertex degree and store it in a property
    # key_v: property key where the degree will be stored (default SGT_NDEGREE)
    def compute_vertex_degree(self, key_v=SGT_NDEGREE):

        # Initialization
        prop_id = self.add_prop(key_v, 'int', 1)

        # Loop for computing vertex degrees
        for v in self.__vertices:
            if v is not None:
                v_id = v.get_id()
                neighs, _ = self.get_vertex_neighbours(v_id)
                self.set_prop_entry_fast(prop_id, (len(neighs),), v_id, 1)

    # Computes angles (in degrees) between edges and Z-axis
    # z_n: normal vector orthogonal to Z-axis (default [0, 0, 1])
    # key_v: if None (default) edge vectors are computed as the vector between their vertices, otherwise it is the
    #        key string of 3-components property which encodes edge vectors
    # Result: a new property is generated with angles values and key STR_EDGE_ZANG
    def compute_edge_zang(self, z_n=(0,0,1.), key_v=None):

        # New property initialization
        prop_id = self.get_prop_id(STR_EDGE_ZANG)
        if prop_id is None:
            prop_id = self.add_prop(STR_EDGE_ZANG, 'float', 1)
        Z_n = np.asarray(z_n, dtype=np.float32)
        prop_v_id = None
        if key_v is not None:
            prop_v_id = self.get_prop_id(key_v)
            if prop_v_id is None:
                error_msg = 'Input property ' + key_v + ' not found.'
                raise pexceptions.PySegInputError(expr='compute_edge_zang (GraphMCF)', msg=error_msg)
            n_comp = self.get_prop_ncomp(key_id=prop_v_id)
            if n_comp != 3:
                error_msg = 'Input property must have 3 components, current has ' + str(n_comp)
                raise pexceptions.PySegInputError(expr='compute_edge_zang (GraphMCF)', msg=error_msg)
            vtype = pyseg.disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=prop_v_id))

        # Edges loop
        if prop_v_id is None:
            for e in self.get_edges_list():
                X_s = self.get_vertex_coords(self.get_vertex(e.get_source_id()))
                X_t = self.get_vertex_coords(self.get_vertex(e.get_target_id()))
                X = np.asarray(X_t, dtype=np.float32) - np.asarray(X_s, dtype=np.float32)
                ang = math.degrees(angle_2vec_3D(X, Z_n))
                self.set_prop_entry_fast(prop_id, (ang,), e.get_id(), 1)
        else:
            for e in self.get_edges_list():
                X_t = self.get_prop_entry_fast(prop_v_id, e.get_id(), n_comp, vtype)
                ang = math.degrees(angle_2vec_3D(np.asarray(X_t, dtype=np.float32), Z_n))
                self.set_prop_entry_fast(prop_id, (ang,), e.get_id(), 1)

    # Computes edge vectors defined as Vs-Vt, where Vs is the edge source vertex and Vt is the edge target one
    # Result: a new property is generated with angles values and key STR_EDGE_VECT
    def compute_edge_vectors(self):

        # New property initialization
        prop_id = self.get_prop_id(STR_EDGE_VECT)
        if prop_id is None:
            prop_id = self.add_prop(STR_EDGE_VECT, 'float', 3, def_val=0)

        # Edges loop
        for e in self.get_edges_list():
            X_s = self.get_vertex_coords(self.get_vertex(e.get_source_id()))
            X_t = self.get_vertex_coords(self.get_vertex(e.get_target_id()))
            X = np.asarray(X_t, dtype=np.float32) - np.asarray(X_s, dtype=np.float32)
            self.set_prop_entry_fast(prop_id, (X[0], X[1], X[2],), e.get_id(), 3)

    # Computes vertices vectors defined as V-Vn, where Vn is farthest vertex neighbor
    # n_hood: neighbourhood radius, if None (default) only directly connected vertices are considered as neighbours
    # key_dst: key property for measuring distances (default STR_VERT_DST)
    # fupdate: if True (default False) force to update GraphGT
    # Result: a new property is generated with angles values and key STR_VERT_VECT
    def compute_vertex_vectors(self, n_hood=None, key_dst=STR_VERT_DST, fupdate=False):

        # Getting input graph
        if (self.__graph_gt is None) or fupdate:
            self.compute_graph_gt()
        graph_gt = self.__graph_gt

        # Input parsing
        if n_hood is not None:
            try:
                prop_d = graph_gt.ep[key_dst]
            except KeyError:
                error_msg = 'No valid distance property specified ' + key_dst
                raise pexceptions.PySegInputError(expr='compute_edge_vectors (GraphMCF)', msg=error_msg)
            prop_v = graph_gt.new_vertex_property('vector<float>')
            prop_id = graph_gt.vp[DPSTR_CELL]

        # Vertices loop
        if n_hood is None:
            for v in graph_gt.vertices():
                hold_dst, hold_e = 0, None
                for e in v.out_edges():
                    if prop_d[e] > hold_dst:
                        hold_dst, hold_e = prop_d[e], e
                if hold_e is not None:
                    n = e.source()
                    if int(v) == int(n):
                        n = e.target()
                    X_s = self.get_vertex_coords(self.get_vertex(prop_id[v]))
                    X_t = self.get_vertex_coords(self.get_vertex(prop_id[n]))
                    prop_v[v] = np.asarray(X_t, dtype=np.float32) - np.asarray(X_s, dtype=np.float32)
        else:
            dst_map = gt.shortest_distance(graph_gt, weights=prop_d)
            for v in graph_gt.vertices():
                dsts = dst_map[v].get_array()
                n_ids = np.where((dsts<=n_hood) & (dsts>0))[0]
                if len(n_ids) > 0:
                    n_dsts = dsts[n_ids]
                    n_id = n_ids[np.argmax(n_dsts)]
                    n = graph_gt.vertex(n_id)
                    X_s = self.get_vertex_coords(self.get_vertex(prop_id[v]))
                    X_t = self.get_vertex_coords(self.get_vertex(prop_id[n]))
                    prop_v[v] = np.asarray(X_t, dtype=np.float32) - np.asarray(X_s, dtype=np.float32)
                else:
                    prop_v[v] = np.asarray((0,0,0), dtype=np.float32)

        # Storing vertex vertices property
        graph_gt.vp[STR_VERT_VECT] = prop_v
        key_id = self.add_prop(STR_VERT_VECT, 'float', 3, def_val=0)
        if key_id is None:
            error_msg = 'Property ' + STR_VERT_VECT + ' could not be added.'
            raise pexceptions.PySegInputError(expr='compute_edge_vectors (GraphMCF)', msg=error_msg)
        for v in graph_gt.vertices():
            self.set_prop_entry_fast(key_id, tuple(prop_v[v]), prop_id[v], 3)

    # Suppress specific vertices
    # v_ids: list of vertex ids to suppress
    # rad_n: radius of the neighbourhood for suppression (default 0, just indexed vertices are suppressed)
    # key_dst: edge property key string to measure vertices distance (default SGT_EDGE_LENGTH)
    def suppress_vertices(self, v_ids, rad_n=0, key_dst=SGT_EDGE_LENGTH):

        # Input parsing
        if not hasattr(v_ids,'__len__'):
            error_msg = 'Input vertex indices must be a list.'
            raise pexceptions.PySegInputError(expr='suppress_vertices (GraphMCF)', msg=error_msg)
        if rad_n > 0:
            prop_id = self.get_prop_id(key_dst)
            if prop_id is None:
                error_msg = 'Property ' + key_dst + ' not found!'
                raise pexceptions.PySegInputError(expr='suppress_vertices (GraphMCF)', msg=error_msg)
            n_comp = self.get_prop_ncomp(key_id=prop_id)
            if n_comp != 1:
                error_msg = 'Only properties with 1 component are valid, current ' + str(n_comp)
                raise pexceptions.PySegInputError(expr='suppress_vertices (GraphMCF)', msg=error_msg)
            gtype = self.get_prop_type(key_id=prop_id)
            etype = disperse_io.TypesConverter().gt_to_numpy(gtype)

        # Removing vertices
        if rad_n > 0:

            # Build temporary graph
            lut_v = np.zeros(shape=self.get_nid(), dtype=object)
            graph = gt.Graph(directed=False)
            vertices = self.get_vertices_list()
            pv_arr = np.zeros(shape=len(vertices), dtype=np.int)
            for i, v in enumerate(vertices):
                v_id = v.get_id()
                lut_v[v_id] = graph.add_vertex()
                pv_arr[i] = v_id
            edges = self.get_edges_list()
            pe_arr = np.zeros(shape=len(edges), dtype=etype)
            for i, e in enumerate(edges):
                s_id, t_id, e_id = e.get_source_id(), e.get_target_id(), e.get_id()
                graph.add_edge(lut_v[s_id], lut_v[t_id])
                pe_arr[i] = self.get_prop_entry_fast(prop_id, e_id, 1, etype)[0]
            pe = graph.new_edge_property(gtype)
            pe.get_array()[:] = pe_arr
            dst_map = gt.shortest_distance(graph, weights=pe)

            # Finding neighbors
            for v_id in v_ids:
                v = lut_v[v_id]
                dists = dst_map[v].get_array()
                h_ids = np.where(dists <= rad_n)[0]
                for h_id in h_ids:
                    v = self.get_vertex(pv_arr[h_id])
                    if v is not None:
                        self.remove_vertex(v)

        else:
            for v_id in v_ids:
                v = self.get_vertex(v_id)
                if v is not None:
                    self.remove_vertex(v)

    # Generates a new copy of the current GraphMCF but this copy is purged so as to just keep specified vertices
    # v_ids: list of vertices ids for output subgraph
    # Returns: an new GraphMCF object with just the specified vertices
    def gen_subgraph(self, v_ids):

        # Initial graph copy
        hold_graph = copy.deepcopy(self)

        # Purging
        lut_v = np.zeros(shape=self.get_nid(), dtype=np.bool)
        for v_id in v_ids:
            lut_v[v_id] = True
        for v in hold_graph.get_vertices_list():
            if not lut_v[v.get_id()]:
                hold_graph.remove_vertex(v)

        return hold_graph

    # Find maximum filament persistence length
    # mn_len: minimum length to search
    # mx_len: maximum length to search
    # mx_ap: maximum third curvature fraction (rad) between (0, 2*pi)
    # samp_len: distance sampling for curves geometry
    # mx_sin: maximum sinuosity allowed, low values will increse speed but may be dangerous (default 3)
    # gen_fils: if True (default False) a set with the filaments found is also returned (SetFilaments)
    # npr: number of process, None (defaults) means they are adjusted automatically to the number of cores
    # Returns: to new graph_tool properties (for edges and vertices) called SGT_MAX_LP and SGT_MAX_LP_X
    def find_max_fil_persistence(self, mn_len, mx_len, mx_ktt, samp_len, gen_fils=False, npr=None):

        # Initialization
        per_id, len_id = self.add_prop(STR_MAX_LP, 'float', 1), self.add_prop('fil_length', 'float', 1)
        sin_id, apl_id = self.add_prop('fil_sin', 'float', 1), self.add_prop('fil_apl', 'float', 1)
        unk_id, unt_id = self.add_prop('fil_unk', 'float', 1), self.add_prop('fil_unt', 'float', 1)
        # VTK initialization
        set_fils = None
        if gen_fils:
            set_fils = SetSpaceCurve(list())

        # Multi-threading
        if npr is None:
            npr = mp.cpu_count()
        # npr = 1
        processes = list()
        # Static number of vertices division
        vertices = self.get_vertices_list()
        nv = len(self.get_vertices_list())
        spl_ids = np.array_split(np.arange(nv), npr)
        nid = self.get_nid()
        # Shared arrays idex: 0: persistence length, 1: filament length, 2: sinuosity, 3: apex length, 4: unsigned
        #                        normalized curvature, 5: unsigned normalized torsion
        per_mpa, len_mpa = mp.RawArray('f', nid), mp.RawArray('f', nid)
        apl_mpa, sin_mpa = mp.RawArray('f', nid), mp.RawArray('f', nid)
        unk_mpa, unt_mpa = mp.RawArray('f', nid), mp.RawArray('f', nid)
        sin_mpa[:] = -1.*np.ones(shape=nid, dtype=float)
        shared_arrs = (per_mpa, len_mpa, sin_mpa, apl_mpa, unk_mpa, unt_mpa)
        if gen_fils:
            manager = mp.Manager()
            fils_mpa = manager.list()
            for pr_id in range(npr):
                fils_mpa.append(list())

        # Vertices loop
        # import time
        # hold_time = time.time()
        for pr_id in range(npr):
            if gen_fils:
                pr = mp.Process(target=th_find_max_per, args=(pr_id, self, vertices, spl_ids[pr_id],
                                                              samp_len, mx_ktt,
                                                              mn_len, mx_len,
                                                              shared_arrs, fils_mpa))
            else:
                pr = mp.Process(target=th_find_max_per, args=(pr_id, self, vertices, spl_ids[pr_id],
                                                              samp_len, mx_ktt,
                                                              mn_len, mx_len,
                                                              shared_arrs, fils_mpa))
            pr.start()
            processes.append(pr)
        for pr in processes:
            pr.join()
        # print 'Time for ' + str(npr) + ' process: ' + str(time.time() - hold_time)
        gc.collect()

        # Join the filaments
        print 'Threads finished!'
        if gen_fils:
            # print fils_mpa[0]
            set_fils = SetSpaceCurve(fils_mpa[0])
            for pr_id in range(1, npr):
                # print fils_mpa[pr_id]
                set_fils.add(fils_mpa[pr_id])

        # Set vertices properties (maximum of its vertices criterium)
        for v in vertices:
            v_id = v.get_id()
            len_v = len_mpa[v_id]
            if len_v > 0:
                self.set_prop_entry_fast(per_id, (per_mpa[v_id],), v_id, 1)
                self.set_prop_entry_fast(len_id, (len_mpa[v_id],), v_id, 1)
                self.set_prop_entry_fast(sin_id, (sin_mpa[v_id],), v_id, 1)
                self.set_prop_entry_fast(apl_id, (apl_mpa[v_id],), v_id, 1)
                self.set_prop_entry_fast(unk_id, (unk_mpa[v_id]/len_v,), v_id, 1)
                self.set_prop_entry_fast(unt_id, (unt_mpa[v_id]/len_v,), v_id, 1)

        # Set edges property (maximum of its vertices criterium)
        for e in self.get_edges_list():
            e_id = e.get_id()
            len_v = len_mpa[e_id]
            if len_v > 0:
                self.set_prop_entry_fast(per_id, (per_mpa[e_id],), e_id, 1)
                self.set_prop_entry_fast(len_id, (len_mpa[e_id],), e_id, 1)
                self.set_prop_entry_fast(sin_id, (sin_mpa[e_id],), e_id, 1)
                self.set_prop_entry_fast(apl_id, (apl_mpa[e_id],), e_id, 1)
                self.set_prop_entry_fast(unk_id, (unk_mpa[e_id]/len_v,), e_id, 1)
                self.set_prop_entry_fast(unt_id, (unt_mpa[e_id]/len_v,), e_id, 1)

        if gen_fils:
            return set_fils

    # Generates a TomoPeaks object from the vertices
    # t_name: tomogram path
    # rs: coordinates rescaling to fit reference tomogram (default 1)
    # v_rot: vector property (number of components 3) for rotation information (default None, no ration information
    #        added)
    # a_rot: property key for storing rotation angles information, only valid if v_rot is not None
    # v_ref: reference vector (default [0, 0, 1])
    # conv: convention, valid: 'relion' (default)
    def gen_tomopeaks(self, t_name, rs=1., v_rot=None, a_rot=None, v_ref=(0, 0, 1), conv='relion'):

        # Initialization
        tomo = disperse_io.load_tomo(str(t_name), mmap=True)
        rs = float(rs)
        if v_rot is not None:
            prop_v_id = self.get_prop_id(v_rot)
            if prop_v_id is None:
                error_msg = 'Property ' + tomo + ' does not exist.'
                raise pexceptions.PySegInputError(expr='gen_tomopeaks (GraphMCF)', msg=error_msg)
            elif self.get_prop_ncomp(key_id=prop_v_id) != 3:
                error_msg = 'Property ' + tomo + ' must have 3 components.'
                raise pexceptions.PySegInputError(expr='gen_tomopeaks (GraphMCF)', msg=error_msg)
            prop_v_t = disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=prop_v_id))
        peaks = TomoPeaks(tomo.shape, name=t_name)

        # Loop for adding peaks from graph vertices
        vertices = self.get_vertices_list()
        for v in vertices:
            coords = self.get_vertex_coords(v) * rs
            peaks.add_peak(coords)

        # Rotation information
        if v_rot is not None:
            vals = list()
            for v in vertices:
                vals.append(self.get_prop_entry_fast(prop_v_id, v.get_id(), 3, prop_v_t))
            peaks.add_prop(v_rot, n_comp=3, vals=vals, dtype=prop_v_t)

        # Assign vector property as rotation
        peaks.vect_rotation_ref(v_rot, a_rot, v_ref=v_ref, conv=conv)


    # Radial distribution function on the graph
    # thick: shell thickness
    # max_d: maximum distance
    # n_samp: number of samples
    # n_samp2: fraction [0, 1] (default 0) of equally spaced sub-samples to store a vertex properties in the graph, typically greater than ss
    #        (default None)
    # edge_len: metric used to measure the edge length (default SGT_EDGE_LENGTH), rg, ss and st will be measured in
    #           the unit of this metric
    # mask: default (None), binary mask with the valid region, the mask will be applied to the tomogram
    # norm: if True (default) local density is normalized according to global density
    # npr: number of process, None (defaults) means they are adjusted automatically to the number of cores
    # Returns: a 2-tuple with the X and Y axis of the graph RDF and the corresponded properties rdf_* for vertices
    #           are added in the graph if ss_vp is not None
    def graph_rdf(self, thick, max_d, n_samp=50, n_samp2=0, edge_len=SGT_EDGE_LENGTH, mask=None, norm=True, npr=None):

        # Initialization
        rads = np.linspace(0, max_d, n_samp)
        bin_s = list()
        thick_h = .5 * float(thick)
        bin_s.append((rads[0], thick_h))
        for i in range(1, rads.shape[0]):
            h_l = rads[i] - thick_h
            if h_l < 0:
                h_l = 0
            bin_s.append((h_l, rads[i]+thick_h))
        nr = len(bin_s)
        if mask is None:
            mask = np.ones(shape=self.__density.shape, dtype=np.bool)
        if mask.shape != self.__density.shape:
            error_msg = 'Input mask must have the same shape as the input density.'
            raise pexceptions.PySegInputError(expr='graph_rdf (GraphMCF)', msg=error_msg)
        vol = mask.sum() * (self.__resolution**3)

        # Applying the mask
        self.add_scalar_field_nn(mask.astype(np.float32), 'mask')
        self.threshold_vertices('mask', 0, operator.eq)
        self.threshold_edges('mask', 0, operator.eq)
        dst_mask = SubVolDtrans(mask)

        # Compute GraphGT
        graph = GraphGT(self)
        graph_gt = graph.get_gt()
        nv = graph_gt.num_vertices()
        prop_id = graph_gt.vp[DPSTR_CELL]

        # Mult-processing
        if npr is None:
            npr = mp.cpu_count()
        npr = 1
        processes = list()
        # Static number of vertices division
        v_ids = np.arange(nv)
        spl_ids = np.array_split(np.arange(nv), npr)
        print str(nv), str(nr)
        print str(nv*nr)
        verts_rdf_mpa = mp.Array('f', nv*nr)

        # Computing distances matrix
        prop_e = graph_gt.ep[edge_len]
        dsts_mat = gt.shortest_distance(graph_gt, weights=prop_e, max_dist=max_d)

        # Vertices parallel loop
        import time
        hold_time = time.time()
        for pr_id in range(npr):
            hold_ids = spl_ids[pr_id]
            coords = np.zeros(shape=(len(hold_ids), 3), dtype=np.float32)
            hold_dsts_vects = list()
            for i, idx in enumerate(hold_ids):
                vg = graph_gt.vertex(idx)
                v = self.get_vertex(prop_id[graph_gt.vertex(idx)])
                coords[i, :] = self.get_vertex_coords(v)
                hold_dsts_vects.append(dsts_mat[vg].get_array())
            pr = mp.Process(target=pr_graph_rdf, args=(pr_id, max_d, bin_s, spl_ids[pr_id],
                                                       coords, hold_dsts_vects, dst_mask, self.__resolution,
                                                       verts_rdf_mpa))
            pr.start()
            processes.append(pr)
        for pr in processes:
            pr.join()
        print 'Time for ' + str(npr) + ' process: ' + str(time.time() - hold_time)
        gc.collect()

        # Computing final results
        # rdf = np.zeros(shape=nr, dtype=np.float)
        # for i in range(nv):
        #     rdf += verts_rdf_mpa[i:i+nr]
        # if norm:
        #     rdf /= vol
        rdf_mat = np.frombuffer(verts_rdf_mpa.get_obj(), dtype=np.float32).reshape(nv, nr)
        print str(nv), str(nr)
        print str(rdf_mat.shape)

        # Adding the properties to the graph
        ss_f = int(math.floor(n_samp2 * nr))
        if (ss_f > 0) and (ss_f <= nr):
            for i_rd in np.round(np.linspace(0, len(rads), ss_f)):
                if i_rd >= len(rads):
                    i_rd = len(rads) - 1
                if i_rd < 0:
                    i_rd = 0
                rd = rads[i_rd]
                key_id = self.add_prop('rdf_'+str(rd), 'float', 1)
                for v_gt_id in v_ids:
                    v_id = prop_id[graph_gt.vertex(v_gt_id)]
                    # print v_gt_id, v_id
                    val = rdf_mat[v_gt_id, i_rd]
                    self.set_prop_entry_fast(key_id, (val,), v_id, 1)

        print 'RDF successfully computed!'

        return rads, rdf_mat.sum(axis=0)

    # Generates the shortest path between two vertices in the graph
    # v_source: v_id for the starting vertex
    # v_target: v_id for the target vertex
    # prop_key: property key for measuring the geodesic distance (default SGT_EDGE_LENGTH)
    # Returns: the shortest path (list of vertex ids, and edges ids) between source and targets
    def find_shortest_path(self, v_source, v_target, prop_key=SGT_EDGE_LENGTH):

        # Input parsing
        if self.__graph_gt is None:
            error_msg = 'This function could not be run without comput GraphGT first.'
            raise pexceptions.PySegInputError(expr='find_shortest_path (GraphMCF)', msg=error_msg)

        # Shortest path algorithm
        try:
            s = gt.find_vertex(self.__graph_gt, self.__graph_gt.vertex_properties[DPSTR_CELL], v_source)[0]
            t = gt.find_vertex(self.__graph_gt, self.__graph_gt.vertex_properties[DPSTR_CELL], v_target)[0]
        except IndexError:
            return  None
        edge_prop = self.__graph_gt.edge_properties[prop_key]
        v_path, e_path = gt.shortest_path(self.__graph_gt, s, t, weights=edge_prop)

        v_path_ids, e_path_ids = list(), list()
        for v in v_path:
            v_path_ids.append(self.__graph_gt.vertex_properties[DPSTR_CELL][v])
        for e in e_path:
            e_path_ids.append(self.__graph_gt.edge_properties[DPSTR_CELL][e])

        return v_path_ids, e_path_ids

    #### Internal area for topology simplification helper functions

    # Cancel a vertex, if it has a pair the geometry an neighbours will be transferred to it,
    # otherwise the vertex is just removed. This method has been designed for working
    # exclusively with topological_simplification method
    # vertex: input vertex for being cancelled
    # key_prop_id: key identifier for edge property
    # key_per_id: key identifier for vertex persistence property
    # key_hid_id: key identifier for vertex hold id property
    # Returns: None, but this method cancel a vertex in the graph and update the persistence
    # list
    def __cancel_vertex(self, vertex, key_prop_id, key_per_id, key_hid_id):

        # Compute pair (and the arc which forms the edge with the input vertex)
        pair, a_pair = self.compute_pair_vertex(vertex, key_prop_id)
        ############# DEBUG
        # if pair is not None:
        #     print '\tPersistence of pair ' + str(pair.get_id()) + \
        #           ' is ' + str(self.compute_vertex_persistence(pair, key_field_id))
        ###################

        # Remove input vertex and its edges (before get its neighbours)
        v_id = vertex.get_id()
        neighs, edges_n = self.get_vertex_neighbours(v_id)
        self.remove_vertex(vertex)
        self.__props_info.set_prop_entry_fast(key_per_id, (0,), v_id, 1)

        # If pair =>
        if pair is not None:

            # Extend geometry of the pair
            geom_p = pair.get_geometry()
            if geom_p is not None:
                geom = vertex.get_geometry()
                if geom is not None:
                    geom_p.extend(geom)

            # Extend pair arcs with input vertex arcs
            a_pair_id = a_pair.get_id()
            hold_pairs = list()
            # Find old arcs
            for a in vertex.get_arcs():
                a_id = a.get_id()
                if a_id != a_pair_id:
                    hold_pairs.append(a)
                else:
                    # Get pair side arc
                    pair_arc = None
                    a_sad_id = a.get_sad_id()
                    for ap in pair.get_arcs():
                        if a_sad_id == ap.get_sad_id():
                            pair_arc = ap
                            break
                    if pair_arc is None:
                        error_msg = 'Pair arc not found'
                        raise pexceptions.PySegInputError(expr='__cancel_vertex (GraphMCF)',
                                                          msg=error_msg)
            # Arcs extension
            pair.del_arc(pair_arc)
            pair_arc.extend(a_pair, side='sad')
            for a in hold_pairs:
                hold = copy.deepcopy(pair_arc)
                a.extend(hold, side='min')
                pair.add_arc(a)

            # Create the new edges (avoid repeated edges, keep the one with the
            # lowest property value)
            pair_id = pair.get_id()
            p_neighs, p_edges_n = self.get_vertex_neighbours(pair_id)
            for i, n in enumerate(neighs):
                n_id = n.get_id()
                if n_id != pair_id:
                    new_edge = True
                    e_id = edges_n[i].get_id()
                    hold_fv = self.__props_info.get_prop_entry_fast(key_prop_id, e_id,
                                                                    1, np.float)
                    hold_fv = hold_fv[0]
                    for j, n2 in enumerate(p_neighs):
                        n2_id = n2.get_id()
                        if n2_id == n_id:
                            e2 = p_edges_n[j]
                            hold_fv2 = self.__props_info.get_prop_entry_fast(key_prop_id,
                                                                             e2.get_id(),
                                                                             1, np.float)
                            hold_fv2 = hold_fv2[0]
                            if hold_fv2 < hold_fv:
                                hold_fv = hold_fv2
                                new_edge = False
                            else:
                                self.remove_edge(e2)
                    if new_edge:
                        edge = EdgeMCF(e_id, pair_id, n_id)
                        self.insert_edge(edge)

            # Compute persistence in pair vertex (maybe it is increased)
            per = self.compute_vertex_persistence(pair, key_prop_id)

            ############# DEBUG
            # print '\tUpdated persistence of pair ' + str(pair.get_id()) + \
            #  ' is ' + str(per)
            ###################

            # Update persistence list
            self.__props_info.set_prop_entry_fast(key_per_id, (per,), pair_id, 1)
            hold = self.__props_info.get_prop_entry_fast(key_hid_id, pair_id, 1, np.int)
            self.__per_lst[hold[0]] = per

    # Merges two edges, they must share a local min, the other two will be set as source and
    # target vertices. The corresponding arc of the source vertex is also updated
    # s_edge: source edge
    # t_edge: target edge
    # key_field_id: key identifier for field value property
    def __merge_edges(self, s_edge, t_edge, key_field_id):

        # Check that both edges share an extrema
        both = False
        v2_id = s_edge.get_source_id()
        if v2_id == t_edge.get_source_id():
            both = True
            v1_id = s_edge.get_target_id()
            v3_id = t_edge.get_target_id()
        elif v2_id == t_edge.get_target_id():
            both = True
            v1_id = s_edge.get_target_id()
            v3_id = t_edge.get_source_id()
        if not both:
            v2_id = s_edge.get_target_id()
            if v2_id == t_edge.get_source_id():
                both = True
                v1_id = s_edge.get_source_id()
                v3_id = t_edge.get_target_id()
            elif v2_id == t_edge.get_target_id():
                both = True
                v1_id = s_edge.get_source_id()
                v3_id = t_edge.get_source_id()
        if not both:
            return None

        # Check if the both vertices are already neighbours and update the edge field value
        t_e_id = t_edge.get_id()
        s_e_id = s_edge.get_id()
        hold = self.__props_info.get_prop_entry_fast(key_field_id, t_e_id, 1, np.float)
        v2_field = hold[0]
        neighs, edges = self.get_vertex_neighbours(v1_id)
        for i, n in enumerate(neighs):
            e_id = edges[i].get_id()
            if (n.get_id() == v3_id) and (e_id != t_e_id):
                hold = self.__props_info.get_prop_entry_fast(key_field_id, e_id, 1, np.float)
                hold_field = hold[0]
                if hold_field < v2_field:
                    v2_field = hold_field
                self.__props_info.set_prop_entry_fast(key_field_id, (t_e_id,), e_id, 1)
                self.remove_edge(t_e_id)
                # TODO: activate this break for increasing the speed of topological simp
                # break


        # Remove old edges
        self.remove_edge(s_edge)
        self.remove_edge(t_edge)

        # Create the new one
        edge = EdgeMCF(t_e_id, v1_id, v3_id)
        self.insert_edge(edge)

        # Update the corresponding arc of the source vertex
        v_source = self.get_vertex(v1_id)
        s_arc = None
        for a in v_source.get_arcs():
            if a.get_sad_id() == s_e_id:
                s_arc = a
                break
        v_hold = self.get_vertex(v2_id)
        h_arc1 = None
        h_arc2 = None
        for a in v_hold.get_arcs():
            a_id = a.get_sad_id()
            if a_id == s_e_id:
                h_arc1 = a
            elif a_id == t_e_id:
                h_arc2 = a
            if (h_arc1 is not None) and (h_arc2 is not None):
                break
        h_arc2.extend(h_arc1, side='min')
        s_arc.extend(h_arc2, side='sad')

    #### Internal functionality area

    # Restore previous state
    def __setstate__(self, state):
        self.__dict__.update(state)
        # Restore unpickable objects
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(self.__skel_fname)
        reader.Update()
        self.__skel = reader.GetOutput()

    # Copy the object's state from self.__dict__ which contains all instance attributes.
    # Afterwards remove unpickable objects
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_GraphMCF__skel']
        return state
