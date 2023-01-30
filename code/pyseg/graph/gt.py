"""
Wrapping classes for using graph-tool package

# Author: Antonio Martinez-Sanchez (Max Planck Institute for Biochemistry)
# Date: 02.01.15
"""

__author__ = 'martinez'
try:
    from globals import *
except:
    from pyseg.globals import *
# import pyseg.globals as gl
import pyseg.disperse_io as disperse_io
import pyseg.filament as filament
# from morse import GraphMCF
import graph_tool.all as gt
import matplotlib.pyplot as plt
import scipy as sp
import sys
import math
from sklearn.cluster import AffinityPropagation, DBSCAN, AgglomerativeClustering, SpectralClustering
import multiprocessing as mp
import threading as mt
import pyseg
import copy
from pyseg.filament.globals import FilamentUDG

##### File global variables
C_AP_CMD = 'Concurrent_AP'
DEBUG = True
MAX_FLOAT = np.finfo(float).max

###### Package functions

# Computes the unit vector between two points and its length
# v0: origin point
# v1: ending point
# Return: 2-tuple; the unit vector and its length
def norm_vector_2pts(v0, v1):
    vect = v1 - v0
    norm = math.sqrt((vect * vect).sum())
    if norm <= 0:
        return np.zeros(shape=3, dtype=float), 0.
    else:
        return vect / norm, norm

# Applies Laplacian smoothing to a 3D curve
# curve: numpy array (n,3) with the 3D coordinates
# ld: correction factor (0, 1] (default .5)
def smooth_3D_curve(curve, ld=.5):

    # Initalization
    nc = curve.shape[0]
    if nc <= 2:
        return np.copy(curve)
    s_curve = np.zeros(shape=curve.shape, dtype=np.float32)

    # Laplacian kernel
    s_curve[0,:] = curve[0, :]
    for i in range(1, nc-1):
        p_m, p, p_p = curve[i-1, :], curve[i, :], curve[i+1, :]
        L = .5*(p_m+p_p)  - p
        s_curve[i, :] = p + ld*L
    s_curve[-1,:] = curve[-1, :]

    return s_curve

# coords: numpy array withe 3D coordinates shape=(n,3)
# mx_ang: maximum angle (between consecutive tangents) for curvature in degress, default 180.
# l_iter: number of iterations for Laplacian curve smoothing, if 0 (default) no smothing.
# Returns: a 2-tuple with maximum persistence found and index of the filament endpoint
def fil_max_persistence(coords, mx_ang=90., l_iter=0):

    # The trivial case
    mx_ang_rad = math.radians(mx_ang)
    nc = coords.shape[0]
    if nc <= 2:
        return -1., 0

    # Curve smoothing
    if l_iter <= 0:
        s_coords = coords
    else:
        s_coords = np.copy(coords)
        for i in range(l_iter):
            s_coords = smooth_3D_curve(coords, ld=.5)

    # Estimating filament length
    t_0, norm = norm_vector_2pts(s_coords[0, :], s_coords[1, :])
    inc_s = norm
    hold_ang = mx_ang_rad
    count = 0
    for i in range(1, nc-1):
        t_i_1, norm = norm_vector_2pts(s_coords[i, :], s_coords[i+1, :])
        ang = angle_2vec_3D(t_0, t_i_1)
        if ang > mx_ang_rad:
            break
        inc_s += norm
        hold_ang = ang
        count += 1

    # compute persistence
    try:
        print('L=' + str(inc_s) + ', ang=' + str(math.degrees(hold_ang)) + ', cos=' + str(math.cos(hold_ang)) + \
            ', log=' + str(math.log(math.cos(hold_ang))) + ', per=' + str(-inc_s/math.log(math.cos(hold_ang))))
        print('Nsteps ' + str(count) + ' of ' + str(nc-1))
        return -inc_s/math.log(math.cos(hold_ang)), inc_s
    except ValueError:
        return 0., 0.
    except ZeroDivisionError:
        return 0., 0.

# graph: graph_tool graph
# prop_key_v: property key for vertices
# prop_key_e: property key for edges to measure distances
# th: threshold for vertex property
# ns: neighborhood radius
# nn: minimum number of neighbors required
# prop_key_a: property key for angles (see mx_ang)
# mx_ang: maximum angle (default 180)
# Returns: a list with the peaks
def find_prop_peaks(graph, prop_key_v, prop_key_e, th, ns, nn, prop_key_a=None, mx_ang=180):

    # Input parsing
    try:
        prop_v = graph.vertex_properties[prop_key_v]
    except KeyError:
        error_msg = 'Property ' + prop_key_v + ' for vertices does not exist!'
        raise pexceptions.PySegInputError(expr='find_prop_peaks (pyseg.gt)', msg=error_msg)
    try:
        prop_id = graph.vertex_properties[DPSTR_CELL]
    except KeyError:
        error_msg = 'Property ' + DPSTR_CELL + ' for vertices does not exist!'
        raise pexceptions.PySegInputError(expr='find_prop_peaks (pyseg.gt)', msg=error_msg)
    try:
        prop_e = graph.edge_properties[prop_key_e]
    except KeyError:
        error_msg = 'Property ' + prop_key_e + ' for edges does not exist!'
        raise pexceptions.PySegInputError(expr='find_prop_peaks (pyseg.gt)', msg=error_msg)
    if prop_key_a is not None:
        try:
            prop_a = graph.vertex_properties[prop_key_a]
        except KeyError:
            error_msg = 'Property ' + prop_key_a + ' for angles does not exist!'
            raise pexceptions.PySegInputError(expr='find_prop_peaks (pyseg.gt)', msg=error_msg)

    # Measure all vertices distances and getting GT properties
    dists_map = gt.shortest_distance(graph, weights=prop_e)

    # Get vertices ordered by the input property
    v_vals = prop_v.get_array()
    n_vertices = len(v_vals)
    p_lut = np.zeros(shape=n_vertices, dtype=bool)
    p_lut_i = np.ones(shape=n_vertices, dtype=bool)
    try:
        mn = np.iinfo(v_vals.dtype).min
    except ValueError:
        mn = np.finfo(v_vals.dtype).min

    # Main loop
    peaks = list()
    # TODO: this loop can have a better (faster) implementation
    while p_lut.sum() < n_vertices:
        # Get candidate and neighbors
        id_s = np.argmax(v_vals)

        if graph.vp[DPSTR_CELL][graph.vertex(id_s)] == 94576:
            print('Jol')

        v = graph.vertex(id_s)
        dists = dists_map[v].get_array()
        n_ids = np.where((dists <= ns) & p_lut_i)[0]
        if prop_key_a is None:
            # Check peak criteria
            if (prop_v[v] >= th) and (len(n_ids) >= nn):
                peaks.append(prop_id[v])
            # Mark peak candidate and neighbors as processed
            v_vals[id_s] = mn
            p_lut[id_s] = True
            p_lut_i[id_s] = False
            for n_id in n_ids:
                v_vals[n_id] = mn
                p_lut[n_id] = True
                p_lut_i[n_id] = False
        else:
            # Check peak criteria
            if prop_a[v] <= mx_ang:
                if (prop_v[v] >= th) and (len(n_ids) >= nn):
                    peaks.append(prop_id[v])
                # Mark peak candidate and neighbors as processed
                v_vals[id_s] = mn
                p_lut[id_s] = True
                p_lut_i[id_s] = False
                for n_id in n_ids:

                    if graph.vp[DPSTR_CELL][graph.vertex(n_id)] == 94576:
                        print('Jol')

                    v_vals[n_id] = mn
                    p_lut[n_id] = True
                    p_lut_i[n_id] = False
            else:
                # Discard current vertex but not neighbors
                v_vals[id_s] = mn
                p_lut[id_s] = True
                p_lut_i[id_s] = False

    return peaks

##################################################################################################
# Class for creating a GT graph from GraphMCF
#
#
class GraphGT(object):

    #### Constructor Area

    # graph_mcf: GraphMCF instance, if it is a string the graph is loaded from disk, and if
    # None the graph is not initialized (it is needed to call either set_mcf_graph() or load)
    def __init__(self, graph_id):
        self.__coords = None
        self.__graph_mcf = None
        if isinstance(graph_id, str):
            self.load(graph_id)
        else:
            self.set_mcf_graph(graph_id)
        self.__cell_v_id = None
        self.__cell_e_id = None

    #### Get/Set

    def set_gt_vertex_property(self, key, prop_v):
        self.__graph.vertex_properties[key] = prop_v

    def set_gt_edge_property(self, key, prop_e):
        self.__graph.vertex_properties[key] = prop_e

    def set_mcf_graph(self, graph_mcf):
        self.__graph = graph_mcf.get_gt()
        prop_id = self.__graph.vertex_properties[DPSTR_CELL]
        prop_c = self.__graph.new_vertex_property('vector<float>')
        for v in self.__graph.vertices():
            prop_c[v] = graph_mcf.get_vertex_coords(graph_mcf.get_vertex(prop_id[v]))
        self.__graph.vertex_properties[SGT_COORDS] = prop_c
        self.__graph_mcf = graph_mcf

    def get_gt(self):
        return self.__graph

    def get_vertices(self):
        hold = list()
        for v in self.__graph.vertices():
            hold.append(v)
        return hold

    #### External functionality

    # Wrapping function for computing betweenness
    # mode: if 'vertex' (default) only vertex property is returned, else if 'edge' only edge
    # property is returned, otherwise both metrics are returned
    # prop_name: string used for indexing the properties
    def betweenness(self, mode='vertex', prop_name=SGT_BETWEENNESS, prop_v=None, prop_e=None):

        vprop = None
        if prop_v is not None:
            vprop = self.__graph.vp[prop_v]
        eprop = None
        if prop_e is not None:
            eprop = self.__graph.ep[prop_e]

        prop_v_out, prop_e_out = None, None
        if mode == 'vertex':
            prop_v_out = gt.betweenness(self.__graph, vprop, eprop)[0]
        elif mode == 'edge':
            prop_e_out = gt.betweenness(self.__graph, vprop, eprop)[1]
        else:
            prop_v_out, prop_e_out = gt.betweenness(self.__graph, vprop, eprop)

        if prop_v_out is not None:
            self.__graph.vertex_properties[prop_name] = prop_v_out
        if prop_e_out is not None:
            self.__graph.edge_properties[prop_name] = prop_e_out

    # Wrapping function for computing the minimum spanning tree
    # prop_name: string used for indexing the properties
    # prop_weight: string name for the property used for weighting (default None)
    def min_spanning_tree(self, prop_name=SGT_MIN_SP_TREE, prop_weight=None):

        if prop_weight is None:
            prop_e = gt.min_spanning_tree(self.__graph)
        else:
            prop_e = gt.min_spanning_tree(self.__graph,
                                          self.__graph.edge_properties[prop_weight])
        self.__graph.edge_properties[prop_name] = prop_e


    # Wrapping function for drawing the graph
    # v_color: string used for encoding the color of vertices (default None)
    # rm_vc=[min, max]: remapping value for the vertex color (default None)
    # cmap_vc=cm: string with the color map name used for vertex color (default hot)
    # e_color: string used for encoding the color of edges (default None)
    # rm_ec=[min, max]: remapping value for the edge color (default None)
    # cmap_ec=cm: string with the color map name used for edge color (default hot)
    # v_size: string used for encoding the size of vertices (default None)
    # rm_vs=[min, max]: remapping value for the vertex size (default None)
    # e_size: string used for encoding the size of edges (default None)
    # rm_es=[min, max]: remapping value for the edge size (default None)
    # output: output file name (ps, svg, pdf and png) or object (default gt.interactive_window())
    def draw(self, v_color=None, rm_vc=None, cmap_vc='hot',
             e_color=None, rm_ec=None, cmap_ec='hot',
             v_size=None, rm_vs=None,
             e_size=None, rm_es=None,
             output=None):

        # Setting vertex color
        n_verts = self.__graph.num_vertices()
        if (v_color is not None) and (n_verts > 0):
            v_c_prop = self.__graph.vertex_properties[v_color].get_array()[:]
            if rm_vc is not None:
                hold = lin_map(v_c_prop, lb=rm_vc[0], ub=rm_vc[1])
                v_c_prop = self.__graph.new_vertex_property(disperse_io.TypesConverter().numpy_to_gt(v_c_prop.dtype))
                v_c_prop.get_array()[:] = hold
            if cmap_vc is not None:
                cmap = plt.cm.get_cmap(cmap_vc, n_verts)
                cmap_vals = cmap(np.arange(n_verts))
                v_c_prop = self.__graph.new_vertex_property('vector<double>')
                for v in self.__graph.vertices():
                    v_c_prop[v] = cmap_vals[self.__graph.vertex_index[v], :]
        else:
            v_c_prop = [0.640625, 0, 0, 0.9]
        # Setting edge color
        n_edges = self.__graph.num_edges()
        if (e_color is not None) and (n_edges > 0):
            e_c_prop = self.__graph.edge_properties[e_color].get_array()[:]
            if rm_ec is not None:
                hold = lin_map(e_c_prop, lb=rm_ec[0], ub=rm_ec[1])
                e_c_prop = self.__graph.new_edge_property(disperse_io.TypesConverter().numpy_to_gt(e_c_prop.dtype))
                e_c_prop.get_array()[:] = hold
            if cmap_ec is not None:
                cmap = plt.cm.get_cmap(cmap_vc, n_edges)
                cmap_vals = cmap(np.arange(n_edges))
                e_c_prop = self.__graph.new_edge_property('vector<double>')
                for e in self.__graph.edges():
                    e_c_prop[e] = cmap_vals[self.__graph.edge_index[e], :]
        else:
            e_c_prop = [0., 0., 0., 1]
        # Setting vertex size
        if (v_size is not None) and (n_verts > 0):
            v_s_prop = self.__graph.vertex_properties[v_size].get_array()[:]
            if rm_vs is not None:
                hold = lin_map(v_s_prop, lb=rm_vs[0], ub=rm_vs[1])
                v_s_prop = self.__graph.new_vertex_property(disperse_io.TypesConverter().numpy_to_gt(v_s_prop.dtype))
                v_s_prop.get_array()[:] = hold
        else:
            v_s_prop = 5
        # Setting edge size
        if (e_size is not None) and (n_edges > 0):
            e_s_prop = self.__graph.edge_properties[e_size].get_array()[:]
            if rm_es is not None:
                hold = lin_map(e_s_prop, lb=rm_es[0], ub=rm_es[1])
                e_s_prop = self.__graph.new_edge_property(disperse_io.TypesConverter().numpy_to_gt(e_s_prop.dtype))
                e_s_prop.get_array()[:] = hold
        else:
            e_s_prop = 1.0

        # Drawing call
        if output is None:
            output = gt.interactive_window(self.__graph)
        gt.graph_draw(self.__graph, pos=gt.sfdp_layout(self.__graph), output=output,
                      vertex_size=v_s_prop, vertex_fill_color=v_c_prop,
                      edge_pen_width=e_s_prop, edge_color=e_c_prop)

    # Store in disk the graph (preferred format .gt)
    def save(self, filename):
        self.__graph.save(filename)

    # Load from disk the graph and overwrites the previous one (preferred format .gt)
    def load(self, filename):
        self.__graph = gt.load_graph(filename)

    # Add property (for vertices or edges) to a GraphMCF instance
    # graph_mcf: graph mcf (it should be the same using in the constructor)
    # key: key string for the property
    # up_index: if True (default) the id for nodes is updated (slower but ensures wer are
    # working with the current state of the graph)
    def add_prop_to_GraphMCF(self, graph_mcf, key, up_index=True):

        # Get keyed property
        try:
            prop_v = self.__graph.vertex_properties[key]
            vertex_mode = True
        except:
            vertex_mode = False
        try:
            prop_e = self.__graph.edge_properties[key]
            edge_mode = True
        except:
            edge_mode = False
        if (vertex_mode is False) and (edge_mode is False):
            error_msg = 'Property ' + key + ' not found.'
            raise pexceptions.PySegInputError(expr='add_prop_to_GraphMCF (GraphGT)',
                                              msg=error_msg)

        # Get index array
        if (self.__cell_v_id is None) or up_index:
            if vertex_mode:
                self.__cell_v_id = self.__graph.vertex_properties[DPSTR_CELL].get_array()
        if (self.__cell_e_id is None) or up_index:
            if edge_mode:
                self.__cell_e_id = self.__graph.edge_properties[DPSTR_CELL].get_array()

        if vertex_mode:
            p_data_v = prop_v.get_array()
            if len(p_data_v.shape) > 1:
                n_comp = p_data_v.shape[1]
            else:
                n_comp = 1
            d_type = disperse_io.TypesConverter().numpy_to_gt(p_data_v.dtype)
        if edge_mode:
            p_data_e = prop_e.get_array()
        if (vertex_mode is False) and edge_mode:
            if len(p_data_e.shape) > 1:
                n_comp = p_data_e.shape[1]
            else:
                n_comp = 1
            d_type = disperse_io.TypesConverter().numpy_to_gt(p_data_e.dtype)

        # Add property to GraphMCF
        graph_mcf.add_prop(key, d_type, n_comp, 0)
        key_id = graph_mcf.get_prop_id(key)
        if vertex_mode:
            for i in range(self.__cell_v_id.shape[0]):
                graph_mcf.set_prop_entry_fast(key_id, (p_data_v[i],), self.__cell_v_id[i],
                                              n_comp)
        if edge_mode:
            for i in range(self.__cell_e_id.shape[0]):
                graph_mcf.set_prop_entry_fast(key_id, (p_data_e[i],), self.__cell_e_id[i],
                                              n_comp)

    # Random Walk Repeated algorithm for several sources
    # source: list with the set of source nodes
    # prop_key: name for the output property map
    # c: restart probability [0, 1] (default 0.1) (the lower the more global analysis)
    # weight: property map used for edge weighting (None is valid and then all edges weight 1)
    # weight_v: property map used for vertex weighting (None is valid and then all vertices weight 1)
    # mode: steady-state computation mode, 1 (default) inverse, otherwise "on the fly"
    # inv: if True (default False) edge weighting property is inverted
    # returns: a property vertex map is added to the graph
    def multi_rwr(self, sources, prop_key, c=0.1, weight=False, weight_v=None, mode=1,
                  inv=False):

        # Initialization
        # Get transposed transition matrix
        if weight is not None:
            if inv:
                weight_p = self.__graph.edge_properties[weight]
                hold_array = weight_p.get_array()
                weight_p.get_array()[:] = lin_map(weight_p.get_array(),
                                                  lb=hold_array.max(), ub=hold_array.min())
            else:
                weight_p = self.__graph.edge_properties[weight]
            W = gt.transition(self.__graph, weight=weight_p)
        else:
            W = gt.transition(self.__graph)
        W = W.transpose()

        # Set up sources vector
        e = np.zeros(shape=self.__graph.num_vertices(), dtype=W.dtype)
        if weight_v is not None:
            weight_v_arr = self.__graph.vertex_properties[weight_v].get_array()[:]
        else:
            weight_v_arr = np.ones(shape=e.shape, dtype=W.dtype)
        for s in sources:
            s_id = self.__graph.vertex_index[s]
            e[s_id] = weight_v_arr[s_id]
        e = e * (1.0/np.sum(e))

        if mode == 1:
            Q = sp.sparse.identity(W.shape[0], dtype=W.dtype) - W.multiply(1-c)
            # Matrix inverted
            Q = sp.sparse.linalg.inv(Q.tocsc())
            Q = Q.multiply(c)
            Q = Q.tocsr()
            # Computation
            r = Q.dot(e)

        else:
            m = 8*(self.__graph.num_edges() + self.__graph.num_vertices())
            if m < 50:
                m = 50
            r = self.__onthefly_solver(e, W, c, 1e-10, m)

        # Set property in graph
        # r = lin_map(r, lb=0, ub=1)
        prop = self.__graph.new_vertex_property('float')
        # prop.get_array()[:] = lin_map(r, lb=0, ub=1)
        prop.get_array()[:] = r
        #prop.get_array()[:] = g_dists
        self.__graph.vertex_properties[prop_key] = prop

    # Applies page rank algorithm for mesure vertex centrality
    # damp: damping factor for the algorithm (default .85)
    # key_e: edges property key for weighting (default None)
    # einv: if True (default False) the values for edge weighting will be inverted
    def page_rank(self, damp=.85, key_e=None, einv=False):

        # Pre-processing graph properties
        if key_e is not None:
            if einv:
                field = self.__graph.edge_properties[key_e].get_array()
                field_inv = lin_map(field, lb=1, ub=0)
                weights = self.__graph.new_edge_property(disperse_io.TypesConverter().numpy_to_gt(field_inv.dtype))
                weights.get_array()[:] = field_inv
            else:
                field = self.__graph.edge_properties[key_e].get_array()
                weights = self.__graph.new_edge_property(disperse_io.TypesConverter().numpy_to_gt(field.dtype))
                weights.get_array()[:] = field
        else:
            weights = None

        # Page rank
        prop_v = gt.pagerank(self.__graph, damp, weight=weights)

        # Store the result
        self.__graph.vertex_properties[SGT_PAGERANK] = prop_v

    # Vertices clustering through affinity propagation
    # key_e: edges property string for distances computing
    # preference: float value which controls the number of clusters (default None, automatically computed)
    # ainv: if True (default False) the values of the affinity matrix will be inverted
    # damp: damping factor, float [0.5 1] (default 0.5)
    # conv_iter: convergence iterations, int (default 15)
    # max_iter: maximum iterations, int (default 200)
    # rand: if True (default False) the output labels for clusters a randomized
    # mx_sp: if not None (default None), it sets the distance threshold in nm for building
    #        a sparse similarity matrix
    # ditype: data type for temporary shortest distance matrix, if None (default) it is set automatically to edge
    #         property type. In some cases allow to avoid MemoryErrors.
    # verbose: if True (default False) the number of clusters found and the preference if it computed automatically
    #          is printed (see preference)
    # Return: stored in a vertex property map called 'aff_clusters'
    def aff_propagation(self, key_e, preference=None, ainv=False, damp=0.5, conv_iter=15, max_iter=200,
                        rand=False, ditype=None, verbose=False):

        # Preprocessing graph properties
        if ainv:
            field = self.__graph.edge_properties[key_e].get_array()
            field_inv = lin_map(field, lb=1, ub=0)
            weights = self.__graph.new_edge_property(disperse_io.TypesConverter().numpy_to_gt(field_inv.dtype))
            weights.get_array()[:] = field_inv
        else:
            field = self.__graph.edge_properties[key_e].get_array()
            weights = self.__graph.new_edge_property(disperse_io.TypesConverter().numpy_to_gt(field.dtype))
            weights.get_array()[:] = field
        self.__graph.edge_properties['hold'] = weights
        if ditype is None:
            ditype = weights.get_array().dtype
        try:
            d_max = np.iinfo(ditype).max
        except ValueError:
            d_max = np.finfo(ditype).max

        # Affinity propagation
        dists = gt.shortest_distance(self.__graph, weights=self.__graph.edge_properties['hold'])

        # Convert distances from PropertyMap to ndarray matrix workable by skit-learn functionality
        mx = 0
        dst_mat = list()
        for i, v in enumerate(self.__graph.vertices()):
            arr = dists[v].get_array()
            arr_idx = arr[arr < d_max]
            h_mx, h_mn = arr_idx.max(), arr_idx.min()
            if h_mx > mx:
                mx = h_mx
            dst_mat.append(arr.astype(ditype))
            del arr
        del dists
        # Invert values os as to have a positive affinity matrix
        dst_mat = np.asarray(dst_mat)
        id_dst_mat = dst_mat < d_max
        hold_dst_mat = dst_mat[id_dst_mat]
        hold_dst_mat *= -1.
        hold_dst_mat += mx
        dst_mat[id_dst_mat] = hold_dst_mat
        id_dst_mat = np.invert(id_dst_mat)
        dst_mat[id_dst_mat] = ditype(0)
        np.fill_diagonal(dst_mat, ditype(0))
        del hold_dst_mat

        # Affinity propagation
        if preference is None:
            id_dst_mat = np.invert(id_dst_mat)
            preference = np.median(dst_mat[id_dst_mat])
            if verbose:
                print('GraphGT.aff_propagation(): Preference computed: ' + str(preference))
        del id_dst_mat
        aff = AffinityPropagation(damping=damp,
                                  convergence_iter=conv_iter,
                                  max_iter=max_iter,
                                  affinity='precomputed',
                                  preference=preference)
        aff.fit(dst_mat)

        # Store the result
        self.__graph.vertex_properties[STR_AFF_CLUST] = self.__graph.new_vertex_property('int')
        self.__graph.vertex_properties[STR_AFF_CENTER] = self.__graph.new_vertex_property('int')
        # Labels
        n_labels = aff.labels_.max()
        if verbose:
            print('GraphGT.aff_propagation(): Number of clusters found: ' + str(n_labels))
        if rand and (n_labels > 0):
            lut_rand = np.random.randint(0, n_labels, len(aff.labels_))
            array = self.__graph.vertex_properties[STR_AFF_CLUST].get_array()
            for i in range(len(array)):
                array[i] = lut_rand[aff.labels_[i]]
            self.__graph.vertex_properties[STR_AFF_CLUST].get_array()[:] = array
        else:
            self.__graph.vertex_properties[STR_AFF_CLUST].get_array()[:] = aff.labels_
        # Centers
        centers = (-1) * np.ones(shape=self.__graph.num_vertices(), dtype=int)
        for i in range(len(aff.cluster_centers_indices_)):
            v = aff.cluster_centers_indices_[i]
            centers[v] = aff.labels_[v]
        self.__graph.vertex_properties[STR_AFF_CENTER].get_array()[:] = centers

    # Vertices clustering through DBSCAN
    # key_dst: string to the edge property used to measure vertices distances
    # eps: (default 0.5) The maximum distance between two samples for them to be considered as in the same neighborhood.
    # min_samples: (default 5) The number of samples in a neighborhood for a point to be considered
    #              as a core point. This includes the point itself.
    # rand: if True (default False) the output labels for clusters a randomized
    # ditype: data type for temporary shortest distance matrix, if None (default) it is set automatically to edge
    #         property type. In some cases allow to avoid MemoryErrors.
    # Return: stored in a vertex property map called 'dbscan_clusters'
    def dbscan(self, key_dst, eps=0.5, min_samples=5, rand=False, ditype=None):

        # Preprocessing graph properties
        prop_d = self.__graph.edge_properties[key_dst]
        if ditype is None:
            ditype = prop_d.get_array().dtype
        try:
            d_max = np.iinfo(ditype).max
        except ValueError:
            d_max = np.finfo(ditype).max

        # Measuring distance through the graph
        dists = gt.shortest_distance(self.__graph, weights=prop_d)

        # Convert distances from PropertyMap to ndarray matrix workable by skit-learn functionality
        dst_mat = list()
        for i, v in enumerate(self.__graph.vertices()):
            arr = dists[v].get_array()
            dst_mat.append(arr.astype(ditype))
            del arr
        del dists
        dst_mat = np.asarray(dst_mat)
        # Ceiling max values for avoiding further overflows
        id_max = dst_mat >= d_max
        dst_mat[id_max] = d_max

        # DBSCAN for graph distances
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        dbscan.fit(dst_mat)

        # Store the result
        self.__graph.vertex_properties[STR_DBSCAN_CLUST] = self.__graph.new_vertex_property('int')
        # Labels
        if rand:
            n_labels = len(dbscan.labels_)
            lut_rand = np.random.randint(0, n_labels, n_labels)
            array = self.__graph.vertex_properties[STR_DBSCAN_CLUST].get_array()
            for i in range(len(array)):
                array[i] = lut_rand[dbscan.labels_[i]]
            self.__graph.vertex_properties[STR_DBSCAN_CLUST].get_array()[:] = array
        else:
            self.__graph.vertex_properties[STR_DBSCAN_CLUST].get_array()[:] = dbscan.labels_

    # Spectral clustering (normalized graph cuts)
    # n_clst: number of clusters
    # affinity: edges property string for building the affinity matrix
    # ainv: if True (default False) the values of the affinity matrix will be inverted
    # rand: if True (default False) the output labels for clusters a randomized
    # Return: stored in a vertex property map called 'sp_clusters'
    def spectral_clst(self, n_clst, affinity, ainv=False, rand=False):

        # Pre-processing graph properties
        if ainv:
            field = self.__graph.edge_properties[affinity].get_array()
            field_inv = lin_map(field, lb=1, ub=0)
            weights = self.__graph.new_edge_property(disperse_io.TypesConverter().numpy_to_gt(field_inv.dtype))
            weights.get_array()[:] = field_inv
        else:
            field = self.__graph.edge_properties[affinity].get_array()
            weights = self.__graph.new_edge_property(disperse_io.TypesConverter().numpy_to_gt(field.dtype))
            weights.get_array()[:] = field
        self.__graph.edge_properties['hold'] = weights
        data_type = field.dtype
        try:
            d_max = np.iinfo(data_type).max
        except:
            d_max = np.finfo(data_type).max

        # Spectral clustering
        dists = gt.shortest_distance(self.__graph, weights=self.__graph.edge_properties['hold'])
        spc = SpectralClustering(n_clusters=n_clst,
                                 eigen_solver='arpack',
                                 assign_labels='discretize',
                                 affinity='precomputed')

        # Affinity matrix
        n_samples = self.__graph.num_vertices()
        aff_mat = np.zeros(shape=(n_samples, n_samples), dtype=data_type)
        for i, v in enumerate(self.__graph.vertices()):
            aff_mat[i, :] = dists[v]
        # Ceiling max values for avoiding further overflows
        id_max = aff_mat == d_max
        aff_mat[id_max] = np.sum(aff_mat[np.invert(id_max)])
        # Invert affinity
        aff_mat = lin_map(aff_mat, lb=1., ub=np.finfo(float).eps)
        spc.fit(aff_mat)

        # Store the result
        self.__graph.vertex_properties[STR_SP_CLUST] = self.__graph.new_vertex_property('int')
        # Labels
        if rand:
            n_labels = len(spc.labels_)
            lut_rand = np.random.randint(0, n_labels, n_labels)
            array = self.__graph.vertex_properties[STR_SP_CLUST].get_array()
            for i in range(len(array)):
                array[i] = lut_rand[spc.labels_[i]]
            self.__graph.vertex_properties[STR_SP_CLUST].get_array()[:] = array
        else:
            self.__graph.vertex_properties[STR_SP_CLUST].get_array()[:] = spc.labels_


    # Hirarchical agglomerative clustering
    # n_clst: number of clusters
    # skel: Graph MCF skeleton for getting vertex coordinates
    # conn: if True (default) graph connectivity information is used
    # affinity: edges property string for weighting the connectivity matrix, if None (default)
    # all weights are set to 1. ONLY POSITIVE VALUES ARE CONSIDERED
    # rand: if True (default False) the output labels for clusters a randomized
    # Return: stored in a vertex property map called 'agg_clusters'
    def agg_clst(self, n_clst, skel, conn=True, affinity=None, rand=False):

        # Initialization
        i_prop = self.__graph.vertex_properties[DPSTR_CELL]

        # Get vertices coordinates
        n_samples = self.__graph.num_vertices()
        X = np.zeros(shape=(n_samples, 3), dtype=float)
        for i, v in enumerate(self.__graph.vertices()):
            X[i, :] = skel.GetPoint(i_prop[v])

        # Connectivity matrix
        n_clst_f = n_clst
        conn_mat = None
        if conn:
            # Connectivity matrix based on neighbours
            conn_mat = sp.sparse.lil_matrix((n_samples, n_samples), dtype=float)
            for v in self.__graph.vertices():
                i = int(v)
                conn_mat[i, i] = 1
                for w in v.out_neighbours():
                    # conn_mat[i, int(w)] = weights[self.__graph.edge(v, w)]
                    conn_mat[i, int(w)] = 1
            # Add weak edges to connectivity matrix so as to have a fully connected graph
            self.__full_connectivity(X, conn_mat)

        # Affinity matrix
        if affinity is not None:
            weights = self.__graph.edge_properties[affinity]
            data_type = weights.get_array().dtype
            try:
                d_max = np.iinfo(data_type).max
            except:
                d_max = np.finfo(data_type).max
            n_samples = self.__graph.num_vertices()
            aff_mat = np.zeros(shape=(n_samples, n_samples), dtype=data_type)
            dists = gt.shortest_distance(self.__graph, weights=weights)
            for i, v in enumerate(self.__graph.vertices()):
                aff_mat[i, :] = dists[v]
            # Ceiling max values for avoiding further overflows
            id_max = aff_mat == d_max
            aff_mat[id_max] = np.sum(aff_mat[np.invert(id_max)])
            # Affinity matrix requires negative distances
            aff_mat *= -1.

        # Clustering
        if affinity is None:
            agg = AgglomerativeClustering(n_clusters=n_clst,
                                          connectivity=conn_mat,
                                          affinity='euclidean',
                                          linkage='ward')
        else:
            agg = AgglomerativeClustering(n_clusters=10,
                                          connectivity=conn_mat,
                                          affinity=aff_mat,
                                          linkage='average')
        clusters = agg.fit(X)

        # Store the result
        self.__graph.vertex_properties[STR_AH_CLUST] = self.__graph.new_vertex_property('int')
        # Labels
        if rand:
            n_labels = len(clusters.labels_)
            lut_rand = np.random.randint(0, n_labels, n_labels)
            array = self.__graph.vertex_properties[STR_AH_CLUST].get_array()
            for i in range(len(array)):
                array[i] = lut_rand[clusters.labels_[i]]
            self.__graph.vertex_properties[STR_AH_CLUST].get_array()[:] = array
        else:
            self.__graph.vertex_properties[STR_AH_CLUST].get_array()[:] = clusters.labels_

    # Vertices clustering through community blockmodel
    # affinity: edges property string for building the affinity matrix
    # ainv: if True (default False) the values of the affinity matrix will be inverted
    # b: growing factor for the logistic mapping of the input values (default None, no remapping is
    # applied)
    # rand: if True (default False) the output labels for clusters a randomized
    # Return: stored in a vertex property map called 'dbscan_clusters'
    def community_bm(self, affinity, ainv=False, b=None, rand=False):

        # Preprocessing graph properties
        if ainv:
            field = self.__graph.edge_properties[affinity].get_array()
            if b is None:
                    field_inv = lin_map(field, lb=1, ub=0)
            else:
                field_inv = gen_log_map(field, b=b, lb=np.max(field), ub=np.min(field))
            weights = self.__graph.new_edge_property(disperse_io.TypesConverter().numpy_to_gt(field_inv.dtype))
            weights.get_array()[:] = field_inv
        else:
            field = self.__graph.edge_properties[affinity].get_array()
            weights = self.__graph.new_edge_property(disperse_io.TypesConverter().numpy_to_gt(field.dtype))
            if b is not None:
                field = gen_log_map(field, b=b)
            weights.get_array()[:] = field
        self.__graph.edge_properties['hold'] = weights

        # Community blockmodel
        state = gt.community_structure(self.__graph, 10, 1000, weight=weights, t_range=(5, 0.1), verbose=True)
        labels = state.get_array()

        # Store the result
        self.__graph.vertex_properties[STR_BM_CLUST] = self.__graph.new_vertex_property('int')
        if rand:
            n_labels = len(labels)
            lut_rand = np.random.randint(0, n_labels, n_labels)
            array = self.__graph.vertex_properties[STR_DBSCAN_CLUST].get_array()
            for i in range(len(array)):
                array[i] = lut_rand[labels[i]]
            self.__graph.vertex_properties[STR_BM_CLUST].get_array()[:] = array
        else:
            self.__graph.vertex_properties[STR_BM_CLUST].get_array()[:] = labels

    # Solves multisource and multisink maximum flow problem
    # graph_mcf: the parent GraphMCF
    # mask: numpy file with the segmentation for sources (mask[x,y,z]==1) and
    #       sinks (mask[x,y,z]==2)
    # prop_w: (None) string key with the edge capacities
    # einv: (False) if True edge properties are inverted
    # alg: algorithm used, valid: 'ek', 'bk' or 'pr' (default)
    # Result: edge property map with the residual capacities (capacity - flow)
    def compute_max_flow(self, graph_mcf, mask, prop_w, einv=False, alg='pr'):

        # Initialization
        graph_b = gt.Graph(directed=True)

        # Vertices
        n_vertices = self.__graph.num_vertices()
        graph_b.add_vertex(n=n_vertices)
        s_source = graph_b.add_vertex()
        s_sink = graph_b.add_vertex()

        # Edges
        hold = list()
        prop = self.__graph.edge_properties[prop_w]
        for e in self.__graph.edges():
            s = e.source()
            t = e.target()
            cap = prop[e]
            graph_b.add_edge(s, t)
            hold.append(cap)
            graph_b.add_edge(t, s)
            hold.append(cap)

        # Virtual edges
        n_edges = 0
        self.__graph.vertex_properties[STR_FLOW_SS] = self.__graph.new_vertex_property('int')
        self.__graph.vertex_properties[STR_FLOW_SS].get_array()[:] = (-1) * np.ones(shape=n_vertices,
                                                                                    dtype=int)
        for v in self.__graph.vertices():
            v_id = self.__graph.vertex_properties[DPSTR_CELL][v]
            if v_id != -1:
                vertex = graph_mcf.get_vertex(v_id)
                x, y, z = graph_mcf.get_vertex_coords(vertex)
                lbl = mask[int(round(x)), int(round(y)), int(round(z))]
                if lbl == 1:
                    graph_b.add_edge(s_source, int(v))
                    n_edges += 1
                    self.__graph.vertex_properties[STR_FLOW_SS][v] = 1
                elif lbl == 2:
                    graph_b.add_edge(int(v), s_sink)
                    n_edges += 1
                    self.__graph.vertex_properties[STR_FLOW_SS][v] = 2

        # Edge capacity
        field = np.asarray(hold)
        if einv:
            field_norm = lin_map(field, lb=1, ub=0)
        else:
            field_norm = lin_map(field, lb=0, ub=1)
        data_type = disperse_io.TypesConverter().numpy_to_gt(field_norm.dtype)
        capacity = graph_b.new_edge_property(data_type)
        capacity_arr = np.concatenate((field, np.ones(shape=n_edges, dtype=field.dtype)))
        capacity.get_array()[:] = capacity_arr

        # gt.graph_draw(graph_b, ecolor=capacity)

        # Solve max flow
        if alg == 'ek':
            capacities = gt.edmonds_karp_max_flow(graph_b, s_source, s_sink, capacity)
            hold = self.__graph.new_edge_property(data_type)
        elif alg == 'bk':
            capacities = gt.boykov_kolmogorov_max_flow(graph_b, s_source, s_sink,
                                                       capacity)
            hold = self.__graph.new_edge_property(data_type)
        else:
            capacities = gt.push_relabel_max_flow(graph_b, s_source, s_sink, capacity)
            hold = self.__graph.new_edge_property(data_type)

        # Insert properties
        cap_arr = capacities.get_array()
        hold_arr = hold.get_array()
        for i in range(0, field.shape[0], 2):
            v1 = capacity_arr[i] - cap_arr[i]
            v2 = capacity_arr[i+1] - cap_arr[i+1]
            if v2 > v1:
                v1 = v2
            hold_arr[i/2] = v1

        if alg == 'ek':
            self.__graph.edge_properties[STR_MFLOW_EK] = self.__graph.new_edge_property(data_type)
            self.__graph.edge_properties[STR_MFLOW_EK].get_array()[:] = hold_arr
        elif alg == 'bk':
            self.__graph.edge_properties[STR_MFLOW_BK] = self.__graph.new_edge_property(data_type)
            self.__graph.edge_properties[STR_MFLOW_BK].get_array()[:] = hold_arr
        else:
            self.__graph.edge_properties[STR_MFLOW_PR] = self.__graph.new_edge_property(data_type)
            self.__graph.edge_properties[STR_MFLOW_PR].get_array()[:] = hold_arr

    # Computes Scale based Clustering Coefficient
    # scale: maximum distance in nm
    # prop_v: property key for weighting the vertices
    # vinv: if True the vertex property is inverted
    def scc(self, scale, prop_v=None, vinv=False):

        # Initialization
        if prop_v is not None:
            prop_v_p = self.__graph.vertex_properties[prop_v]
            field_v = prop_v_p.get_array()
            if vinv:
                field_v = lin_map(field_v, lb=field_v.max(), ub=field_v.min())
        else:
            field_v = np.ones(shape=self.__graph.num_vertices(), dtype=float)
        prop_e_d = self.__graph.edge_properties[SGT_EDGE_LENGTH]
        prop_scc = self.__graph.new_vertex_property('float')
        scale = float(scale)


        # Loop
        for s in self.__graph.vertices():
            # Neighbourhood
            dist_map = gt.shortest_distance(self.__graph,
                                            source=s,
                                            weights=prop_e_d,
                                            max_dist=scale)
            ids = np.where(dist_map.get_array() < scale)[0]
            # Computing SCC for each vertex
            hold_sum = 0.
            for i in range(ids.shape[0]):
                v = self.__graph.vertex(ids[i])
                hold_sum += field_v[int(v)]
            prop_scc[s] = hold_sum

        # Storing the property
        self.__graph.vertex_properties[STR_SCC] = prop_scc

    # Filter for weighting the amount of neighbours places at a specific geodesic distance
    # key: name of the output property
    # dst: target geodesic distance (in nm)
    # sig: sigma for the Gaussian function (the smaller the sharper filtering), must be greater than zero
    # prop_key_v: property key for weighting the vertices (default None)
    # prop_key_s: property key for sources, >0 valued vertices, (default None, all vertices are considered)
    # n_sig: limit for the local analysis to dst +/-n_sig*sig (default 3)
    # Returns: a new vertex float property [0, 1] property is added to the graph with name in key
    def geo_neigh_filter(self, key, dst, sig, prop_key_v=None, prop_key_s=None, n_sig=3.):

        # Initialization
        if sig <= 0:
            error_msg = 'Input sigma must be greater than zero, current is ' + str(sig)
            raise pexceptions.PySegInputError(expr='geo_neigh_filter (GraphGT)',
                                              msg=error_msg)
        if prop_key_v is None:
            prop_v = self.__graph.new_vertex_property('float', vals=1)
        else:
            prop_v = self.__graph.vertex_properties[prop_key_v]
        if prop_key_s is not None:
            prop_s = self.__graph.vertex_properties[prop_key_s]
        prop_e = self.__graph.edge_properties[SGT_EDGE_LENGTH]
        prop_k = self.__graph.new_vertex_property('float')
        n_sig_h = sig * n_sig
        n_sigs = (dst-n_sig_h, dst+n_sig_h)
        if n_sigs[0] < 0:
            n_sigs[0] = 0

        # Compute normalization constant, Gaussian integral distances range
        c = -1./(2.*sig*sig)
        c_a = math.fabs(c)
        cte_1 = math.sqrt(np.pi/(4.*c_a))
        cte_2 = math.sqrt(c_a)
        int_l =  cte_1 * sp.special.erf(cte_2*(n_sig-dst))
        int_h =  cte_1 * sp.special.erf(cte_2*n_sig_h)
        r_l, r_h = n_sigs[0], n_sigs[1]
        vol = ((4.*np.pi)/3.) * (r_h*r_h*r_h - r_l*r_l*r_l)
        n_f = vol * (int_h - int_l)
        if n_f > 0:
            n_f = 1. / n_f
        else:
            n_f = 0

        # Getting sources list
        if prop_s is None:
            sources = self.__graph.vertices()
            n_s = self.__graph.num_vertices()
        else:
            sources = list()
            for v in self.__graph.vertices():
                if prop_s[v] > 0:
                    sources.append(v)
            n_s = len(sources)

        # Multi-threading
        tlock = mt.Lock()
        n_th = mp.cpu_count()
        if n_th > n_s:
            n_th = n_s
        threads = list()

        # Loop for sources
        # Static division in threads by sources
        spl_i = np.array_split(np.arange(0, n_s), n_th)
        for id_i in spl_i:
            th = mt.Thread(target=self.__th_geo_neigh_filter,
                           args=(id_i, sources, prop_v, prop_e, sig, n_sigs, c, n_f, prop_k, tlock))
            th.start()
            threads.append(th)
        for th in threads:
            th.join()

        # Normalization, for comparing results obtained with different scales
        prop_k.get_array()[:] = prop_k.get_array() * n_f

        # Setting the new vertex property
        self.__graph.vertex_properties[key] = prop_k

    # Detect clusters of vertices with an specific size
    # key: name of the output property
    # rad: cluster radius (nm)
    # len_d: length of the depletion region, ring surrounding clustering region where vertices in this region will
    #        contribute negatively
    # prop_key_v: property key for weighting the vertices (default None)
    # prop_key_s: property key for sources, >0 valued vertices, (default None, all vertices are considered)
    # mode_2d: if True (default False), the normalization are computed for 2-manifolds instead 3-manifolds
    # Returns: a new vertex float property [0, 1] is added to the graph with name in key
    def geo_size_filter(self, key, rad, len_d, prop_key_v=None, prop_key_s=None, mode_2d=False):

        # Initialization
        if rad <= 0:
            error_msg = 'Input rad must be greater than zero, current is ' + str(rad)
            raise pexceptions.PySegInputError(expr='geo_size_filter (GraphGT)',
                                              msg=error_msg)
        if prop_key_v is None:
            prop_v = self.__graph.new_vertex_property('float', vals=1)
        else:
            prop_v = self.__graph.vertex_properties[prop_key_v]
        prop_s = None
        if prop_key_s is not None:
            prop_s = self.__graph.vertex_properties[prop_key_s]
        prop_e = self.__graph.edge_properties[SGT_EDGE_LENGTH]
        prop_k = self.__graph.new_vertex_property('float')

        # Compute normalization constant
        if mode_2d:
            cte = 1. / (np.pi * rad * rad)
        else:
            cte = 1. / ((4./3.) * np.pi * rad * rad * rad)
        rad_t = rad + len_d

        # Getting sources list
        sources = list()
        if prop_s is None:
            for v in self.__graph.vertices():
                sources.append(v)
            n_s = self.__graph.num_vertices()
        else:
            for v in self.__graph.vertices():
                if prop_s[v] > 0:
                    sources.append(v)
            n_s = len(sources)
        sources = np.asarray(sources, dtype=object)

        # Multi-threading
        tlock = mt.Lock()
        n_th = mp.cpu_count()
        if n_th > n_s:
            n_th = n_s
        threads = list()

        # Loop for sources
        # Static division in threads by sources
        spl_i = np.array_split(np.arange(0, n_s), n_th)
        for id_i in spl_i:
            th = mt.Thread(target=self.__th_geo_size_filter,
                           args=(id_i, sources, prop_v, prop_e, rad, rad_t, cte, prop_k, tlock))
            th.start()
            threads.append(th)
        for th in threads:
            th.join()

        # Normalization and data regularization
        prop_k_arr = prop_k.get_array()
        prop_k_arr *= cte
        prop_k_arr[prop_k_arr < 0] = .0
        prop_k.get_array()[:] = prop_k_arr

        # Setting the new vertex property
        self.__graph.vertex_properties[key] = prop_k

    # Computes Geodesic Gaussian Filter on the graph
    # sig: sigma for the Gaussian function
    # prop_v: property key for weighting the vertices
    # prop_e: property key for weighting the edge
    # vinv: if True the vertex property is inverted
    # einv: if True the edge property is inverted
    # vnorm: if True the vertex property is normalized to [0, 1]
    # enorm: if True the vertex property is normalized to [0, 1]
    # energy: if True energy normalization is active
    # strg: if True (default False) straight paths are overweighted, straightness is defined as 1./sinuosity
    def ggf(self, sig, prop_v=None, prop_e=None, vinv=False, einv=False, vnorm=False,
            enorm=False, energy=True):

        # Initialization
        if prop_v is not None:
            prop_v_p = self.__graph.vertex_properties[prop_v]
            field_v = prop_v_p.get_array()
            if vinv:
                if vnorm:
                    field_v = pyseg.globals.lin_map(field_v, lb=1, ub=0)
                else:
                    field_v = pyseg.globals.lin_map(field_v, lb=field_v.max(), ub=field_v.min())
            elif vnorm:
                field_v = pyseg.globals.lin_map(field_v, lb=0, ub=1)
        else:
            field_v = np.ones(shape=self.__graph.num_vertices(), dtype=float)
        prop_e_p = None
        if prop_e is not None:
            prop_e_p = self.__graph.edge_properties[prop_e]
            field_e = prop_e_p.get_array()
            if einv:
                if enorm:
                    field_e = pyseg.globals.lin_map(field_e, lb=1, ub=0)
                else:
                    field_e = pyseg.globals.lin_map(field_e, lb=field_e.max(), ub=field_e.min())
            elif enorm:
                field_e = pyseg.globals.lin_map(field_e, lb=0, ub=1)
            prop_e_p.get_array()[:] = field_e
        prop_ggf = self.__graph.new_vertex_property('float')
        s3 = 3. * sig
        c = (-1.) / (2.*sig*sig)

        # Filtering
        for s in self.__graph.vertices():
            dist_map = gt.shortest_distance(self.__graph,
                                            source=s,
                                            weights=prop_e_p,
                                            max_dist=s3)
            ids = np.where(dist_map.get_array() < s3)[0]
            # Computing energy preserving coefficients
            fields = np.zeros(shape=ids.shape[0], dtype=float)
            coeffs = np.zeros(shape=ids.shape[0], dtype=float)
            hold_sum = 0
            for i in range(ids.shape[0]):
                v = self.__graph.vertex(ids[i])
                fields[i] = field_v[int(v)]
                dst = dist_map[v]
                hold = math.exp(c * dst * dst)
                coeffs[i] = hold
                hold_sum += hold
            if hold_sum > 0:
                # Convolution
                if energy:
                    coeffs *= (1/hold_sum)
                    prop_ggf[s] = np.sum(fields * coeffs)
                else:
                    prop_ggf[s] = np.sum(coeffs)

        # Storing the property
        self.__graph.vertex_properties[STR_GGF] = prop_ggf

    # Computes Geodesic Gaussian Filter on the graph, where the only possible metrics for lengths are
    # SGT_EDGE_LENGTH or STR_VERT_DST, this increases dramatically the computation efficiency compared with ggf()
    # sig: sigma for the Gaussian function
    # prop_v: property key for weighting the vertices
    # v_dst: if True (default False) metric for the geodesic distance is STR_VERT_DST instead of SGT_EDGE_LENGTH
    # vinv: if True the vertex property is inverted
    # strg: if True (default False) straight paths are overweighted, straightness is defined as 1./sinuosity,
    #       it option is only available is a GraphMCF is set
    # mems: if this is True (default False), a memory saving version is run, that mean a significant
    #       computation time increase, only use this if memory errors appear
    # find_mx: if True (default False), the maximum instead the sum is computed
    # Returns: a new vertex and edge properties with key STR_GGF
    def ggf_l(self, ref_dst, sig, prop_v=None, v_dst=None, vinv=False, strg=False, mems=False, find_mx=True):

        # Initialization
        if prop_v is not None:
            prop_v_p = self.__graph.vertex_properties[prop_v]
            field_v = prop_v_p.get_array()
            if vinv:
                field_v = pyseg.globals.lin_map(field_v, lb=field_v.max(), ub=field_v.min())
        else:
            field_v = np.ones(shape=self.__graph.num_vertices(), dtype=float)
        if v_dst:
            prop_e_p = self.__graph.edge_properties[STR_VERT_DST]
        else:
            prop_e_p = self.__graph.edge_properties[SGT_EDGE_LENGTH]
        if strg:
            prop_id = self.__graph.vertex_properties[DPSTR_CELL]
            try:
                skel = self.__graph_mcf.get_skel()
            except AttributeError:
                error_msg = '\'strg\' option requires a GraphMCF initialization'
                raise pexceptions.PySegInputError(expr='ggf_l (GraphGT)', msg=error_msg)
        prop_ggf = self.__graph.new_vertex_property('float')
        s3 = 3. * sig
        c = (-1.) / (2.*sig*sig)

        # Measuring distances
        if not mems:
            dists_map = gt.shortest_distance(self.__graph, weights=prop_e_p)

        # Filtering
        for s in self.__graph.vertices():
            if strg:
                pt_s = np.asarray(skel.GetPoint(prop_id[s]), dtype=np.float32)
            if mems:
                dist_map = gt.shortest_distance(self.__graph, source=s, weights=prop_e_p, max_dist=s3)
            else:
                dist_map = dists_map[s]
            ids = np.where(dist_map.get_array() < s3)[0]
            hold = 0
            for idx in ids:
                v = self.__graph.vertex(idx)
                dst = dist_map[v]
                hold_dst = ref_dst - dst
                if strg:
                    pt_v = np.asarray(skel.GetPoint(prop_id[v]), dtype=np.float32)
                    eu_dst = pt_v - pt_s
                    eu_dst = math.sqrt((eu_dst*eu_dst).sum())
                    strg_v = 0
                    if dst > 0:
                        strg_v = eu_dst / dst
                    if find_mx:
                        aux = (field_v[v] * strg_v * math.exp(c * hold_dst * hold_dst))
                        if aux > hold:
                            hold = aux
                    else:
                        hold += (field_v[v] * strg_v * math.exp(c * hold_dst * hold_dst))
                else:
                    if find_mx:
                        aux = (field_v[v] * math.exp(c * hold_dst * hold_dst))
                        if aux > hold:
                            hold = aux
                    else:
                        hold += (field_v[v] * math.exp(c * hold_dst * hold_dst))
            prop_ggf[s] = hold

        # Estimate edge centrality for every edge as the minimum of its two vertex centralities
        prop_e_c = self.__graph.new_edge_property('float')
        for e in self.__graph.edges():
            c_s, c_t = prop_ggf[e.source()], prop_ggf[e.target()]
            if c_s < c_t:
                prop_e_c[e] = c_s
            else:
                prop_e_c[e] = c_t

        # Storing the property
        self.__graph.vertex_properties[STR_GGF] = prop_ggf
        self.__graph.edge_properties[STR_GGF] = prop_e_c

    # Find maximum filament persistence length
    # mx_len: maximum length to search
    # mx_ang: maximum angle (between consecutive tangents) for curvature in degrees, default 90.
    # l_iter: number of iterations for Laplacian curve smoothing, if 0 (default) no smoothing.
    # samp: if not None it sets curve sampling distance in nm
    # mx_sin: maximum sinuosity allowed, low values will increse speed but may be dangerous (default 3)
    # vtp_p: if not None (default), path to a vtp file to store filaments found, it has only debugging purposes
    # Returns: to new graph_tool properties (for edges and vertices) called SGT_MAX_LP and SGT_MAX_LP_X
    def find_max_fil_persistence(self, mx_len, max_ang=90., l_iter=0, samp=None, mx_sin=10, vtp_p=None):

        # Initialization
        prop_e, prop_id_v, prop_id_e = self.__graph.ep[SGT_EDGE_LENGTH], self.__graph.vp[DPSTR_CELL], \
                                       self.__graph.ep[DPSTR_CELL]
        try:
            skel = self.__graph_mcf.get_skel()
        except AttributeError:
            error_msg = 'This function requires a GraphMCF initialization'
            raise pexceptions.PySegInputError(expr='find_max_fil_persistence (GraphGT)', msg=error_msg)
        prop_v_p, prop_v_x = self.__graph.new_vertex_property('float'), self.__graph.new_vertex_property('float')
        prop_e_p, prop_e_x = self.__graph.new_edge_property('float'), self.__graph.new_edge_property('float')
        nv = self.__graph.num_vertices()
        if samp is not None:
            samp_v = samp / self.__graph_mcf.get_resolution()

        # VTK initialization
        if vtp_p is not None:
            poly = vtk.vtkPolyData()
            points = vtk.vtkPoints()
            lines = vtk.vtkCellArray()
            per_arr = vtk.vtkFloatArray()
            per_arr.SetName('persistence length')
            per_arr.SetNumberOfComponents(1)
            xper_arr = vtk.vtkFloatArray()
            xper_arr.SetName('length')
            xper_arr.SetNumberOfComponents(1)
            pt_id = 0

        # Filtering
        count = 1
        for s in self.__graph.vertices():

            if vtp_p is not None:
                print(str(count) + ' of ' + str(nv))
                count += 1

            # Getting neighbours
            pt_s = np.asarray(skel.GetPoint(prop_id_v[s]), dtype=np.float32)
            per, x_per = 0., 0.
            n_lut = np.ones(shape=nv, dtype=bool)
            n_lut[int(s)] = False
            dist_map, pred_map = gt.shortest_distance(self.__graph, source=s, weights=prop_e, max_dist=mx_len,
                                                      pred_map=True)
            dist_map_arr = dist_map.get_array()
            ids = np.where(dist_map_arr < mx_len)[0]
            s_ids = np.argsort(dist_map_arr[ids])[::-1]

            # Neighbours loop
            for idx in ids[s_ids]:
                t = self.__graph.vertex(idx)
                pt_t = np.asarray(skel.GetPoint(prop_id_v[t]), dtype=np.float32)
                eu_dst = pt_t - pt_s
                eu_dst = math.sqrt((eu_dst * eu_dst).sum())
                if (eu_dst > 0) and n_lut[int(t)] and ((dist_map_arr[idx]/eu_dst) < mx_sin):

                    # Getting the path
                    v_path, e_path = gt.shortest_path(self.__graph, source=s, target=t, weights=prop_e,
                                                      pred_map=pred_map)
                    vid_l, eid_l = list(), list()
                    vid_l.append(prop_id_v[v_path[0]])
                    n_lut[int(v_path[0])] = False
                    for i in range(len(e_path)):
                        vh = v_path[i+1]
                        eid_l.append(prop_id_e[e_path[i]])
                        vid_l.append(prop_id_v[vh])
                        n_lut[int(vh)] = False
                    fil = FilamentUDG(vid_l, eid_l, self.__graph_mcf)
                    if samp is None:
                        coords = fil.get_curve_coords()
                    else:
                        coords = fil.gen_downsample(samp=samp_v)
                    if coords is None:
                        continue

                    # Computing maximum persistence for this path
                    hold_per, p_id = fil_max_persistence(coords, max_ang, l_iter)

                    # Updating vertex persistence
                    if hold_per > per:
                        x_per = p_id
                        per = hold_per

                        if vtp_p is not None:
                            coords = fil.get_curve_coords()
                            lines.InsertNextCell(coords.shape[0])
                            per_arr.InsertNextTuple((per,))
                            xper_arr.InsertNextTuple((x_per,))
                            for coord in coords:
                                points.InsertPoint(pt_id, coord)
                                lines.InsertCellPoint(pt_id)
                                pt_id += 1
                else:
                    n_lut[int(t)] = False

            # Set final vertex persistence
            prop_v_p[s] = per
            prop_v_x[s] = x_per

        # Estimate edge properties the minimum of its two vertex
        for e in self.__graph.edges():
            s, t = e.source(), e.target()
            p_s, p_t = prop_v_p[s], prop_v_p[t]
            if p_s < p_t:
                prop_e_p[e] = p_s
                prop_e_x[e] = prop_v_x[s]
            else:
                prop_e_p[e] = p_t
                prop_e_x[e] = prop_v_x[t]

        # Storing the property
        self.__graph.vp[SGT_MAX_LP] = prop_v_p
        self.__graph.vp[SGT_MAX_LP_X] = prop_v_x
        self.__graph.ep[SGT_MAX_LP] = prop_e_p
        self.__graph.ep[SGT_MAX_LP_X] = prop_e_x

        if vtp_p is not None:
            poly.SetLines(lines)
            poly.SetPoints(points)
            poly.GetCellData().AddArray(per_arr)
            poly.GetCellData().AddArray(xper_arr)
            disperse_io.save_vtp(poly, vtp_p)

    # Computes Bilateral Filter on the graph
    # ss: sigma for the Gaussian function for SGT_EDGE_LENGTH property
    # sr: sigma for the Gaussian function for STR_EDGE_FNESS property
    # prop_v: property key for weighting the vertices
    # vinv: if True the vertex property is inverted
    def bilateral(self, ss, sr, prop_v=None, vinv=False):

        # Initialization
        if prop_v is not None:
            prop_v_p = self.__graph.vertex_properties[prop_v]
            field_v = prop_v_p.get_array()
            if vinv:
                field_v = lin_map(field_v, lb=field_v.max(), ub=field_v.min())
        else:
            field_v = np.ones(shape=self.__graph.num_vertices(), dtype=float)
        prop_e_d = self.__graph.edge_properties[SGT_EDGE_LENGTH]
        prop_e_r = self.__graph.edge_properties[STR_EDGE_FNESS]
        # field_r = prop_e_r.get_array()
        # field_r = lin_map(field_r, lb=field_r.max(), ub=field_r.min())
        # prop_e_r.get_array()[:] = field_r
        prop_bil = self.__graph.new_vertex_property('float')
        s3 = 3. * ss
        cs = (-1) / (2*ss*ss)
        cr = (-1) / (2*sr*sr)

        # Filtering
        for s in self.__graph.vertices():
            dist_map = gt.shortest_distance(self.__graph,
                                            source=s,
                                            weights=prop_e_d,
                                            max_dist=s3)
            ids = np.where(dist_map.get_array() < s3)[0]
            # Computing energy preserving coefficients
            fields = np.zeros(shape=ids.shape[0], dtype=float)
            coeffs = np.zeros(shape=ids.shape[0], dtype=float)
            hold_sum = 0
            for i in range(ids.shape[0]):
                v = self.__graph.vertex(ids[i])
                fields[i] = field_v[int(v)]
                dsts = dist_map[v]
                hold_s = math.exp(cs * dsts * dsts)
                _, e_path = gt.shortest_path(self.__graph, source=s, target=v,
                                             weights=prop_e_r)
                dstr = 1
                for e in e_path:
                    # dstr *= prop_e_r[e]
                    if prop_e_r[e] < dstr:
                        dstr = prop_e_r[e]
                dstr = 1 - dstr
                hold_r = math.exp(cr * dstr * dstr)
                hold = hold_s * hold_r
                # print len(e_path), hold_s, hold_r
                coeffs[i] = hold_r
                hold_sum += hold_s
            if hold_sum > 0:
                # coeffs *= (1/hold_sum)
                # Convolution
                prop_bil[s] = np.sum(fields * coeffs)

        # Storing the property
        self.__graph.vertex_properties[STR_BIL] = prop_bil

    # Computes vertex centrality for paths with length in [dist_min, dist_max]
    # prop_key: string key for the computed vertex property
    # prop_e_key: string key for the edge weighting
    # dist_min: minimum distance for paths
    # dist_max: maximum distance for paths
    # prop_s: int property with value 1 for sources, if None (default) all vertices are
    # considered sources
    # sd_cte: sinuosity degradation constant, if 0 (default) it is not applied
    # Return: centrality for paths with fixed lengths stored in a vertex and edge properties called
    # prop_key
    def fix_len_path_centrality(self, prop_key, prop_e_key, dist_min, dist_max, prop_s=None, sd_cte=0):

        # Input parsing
        if (sd_cte > 0) and (self.__graph_mcf is None):
            error_msg = 'A reference GraphMCF is required for applying sinuosity penalization.'
            raise pexceptions.PySegInputError(expr='fix_len_path_centrality (GraphGT)', msg=error_msg)

        # Initialization
        if prop_s is None:
            prop_s = self.__graph.new_vertex_property('int')
            prop_s.get_array()[:] = np.ones(shape=self.__graph.num_vertices(), dtype=int)
        source_ids = list()
        for s in self.__graph.vertices():
            source_ids.append(int(s))
        n_s = len(source_ids)
        prop_v_c_a = np.zeros(shape=self.__graph.num_vertices(), dtype=np.float32)

        # Multi-threading
        n_th = mp.cpu_count()
        if n_th > n_s:
            n_th = n_s
        threads = list()

        # Loop for sources
        # Static division in threads by sources
        spl_i = np.array_split(np.arange(0, n_s), n_th)
        for id_i in spl_i:
            # Create a copy of the graph so as to minimize threads interaction
            h_graph = copy.deepcopy(self.__graph)
            th = mt.Thread(target=self.__fix_len_path_centrality,
                           args=(id_i, source_ids, h_graph, prop_e_key, dist_max, prop_v_c_a, sd_cte))
            th.start()
            threads.append(th)
        for th in threads:
            th.join()

        # Estimate edge centrality for every edge as the minimum of its two vertex centralities
        prop_e_c = self.__graph.new_edge_property('float')
        for e in self.__graph.edges():
            s, t = e.source(), e.target()
            c_s, c_t = prop_v_c_a[int(s)], prop_v_c_a[int(t)]
            if c_s < c_t:
                prop_e_c[e] = c_s
            else:
                prop_e_c[e] = c_t

        # Inserting the new properties
        prop_v_c = self.__graph.new_vertex_property('float')
        prop_v_c.get_array()[:] = prop_v_c_a
        self.__graph.vertex_properties[prop_key] = prop_v_c
        self.__graph.edge_properties[prop_key] = prop_e_c

    # Find vertices in filaments within a length and sinuosity range, a strength parameter is associated to every
    # voxel
    # key_o: key string for the output property (default [1. MAX_FLOAT))
    # rg_len: filament length range (2-tuple) (default [.0 MAX_FLOAT))
    # rg_sin: sinuosity range (2-tuple)
    # mode_eq: if False (default) STR_FIELD_VALUE is used for measuring filament strength, otherwise
    #          STR_FIELD_VALUE_EQ
    # mode_dst: if False (default) SGT_EDGE_LENGTH is used for measuring filament strength, otherwise STR_VERT_DST
    # Returns: a new vertex and edge property with 'key_o' as key is generated with the strength
    #          of the strongest filament which goes through every vertex
    def geo_centrality(self, key_o, rg_len=(1., MAX_FLOAT), rg_sin=(.0, MAX_FLOAT), mode_eq=False, mode_dst=False):

        # Initialization
        if self.__graph_mcf is None:
            error_msg = 'A reference GraphMCF is required for applying sinuosity penalization.'
            raise pexceptions.PySegInputError(expr='geo_centrality (GraphGT)', msg=error_msg)
        if mode_dst:
            prop_l = self.__graph.ep[STR_VERT_DST]
        else:
            prop_l = self.__graph.ep[SGT_EDGE_LENGTH]
        if mode_eq:
            prop_f = self.__graph.ep[STR_FIELD_VALUE_EQ]
        else:
            prop_f = self.__graph.ep[STR_FIELD_VALUE]
        prop_o = self.__graph.new_vertex_property('float')
        prop_o_e = self.__graph.new_edge_property('float')
        prop_w, prop_w_i = self.__graph.new_edge_property('float'), self.__graph.new_edge_property('float')
        prop_w_a = prop_l.get_array() * prop_f.get_array()
        # prop_w_a_i = lin_map(prop_w_a, lb=prop_w_a.max(), ub=prop_w_a.min())
        prop_w_i.get_array()[:] = prop_w_a
        nv = self.__graph.num_vertices()

        # Pre-computing metrics
        lens_map = gt.shortest_distance(self.__graph, weights=prop_l)
        affs_map = gt.shortest_distance(self.__graph, weights=prop_w_i)
        eus_map = self.__vertices_eu_map()

        # Vertices loop
        for v in self.__graph.vertices():

            # Extract potential filaments
            v_id = int(v)
            dist_map = lens_map[v].get_array()
            eu_map = eus_map[v_id, :]
            dist_map[dist_map >= MAX_FLOAT] = 0.
            eu_map[eu_map <= 0] = -1.
            sin_map = dist_map / eu_map
            id_mask = (dist_map > 0.) & \
                      (dist_map >= rg_len[0]) & (dist_map < rg_len[1]) & \
                      (sin_map >= rg_sin[0]) & (sin_map < rg_sin[1])
            ids = np.where(id_mask)[0]
            if len(ids) == 0:
                continue

            # Loop for finding the strongest filament
            affs = affs_map[v].get_array()[ids]
            affs_s = np.argsort(affs)[::-1]
            # _, prop_p = gt.shortest_distance(self.__graph, source=v, weights=prop_w_i,
            #                                  max_dist=rg_len[1], pred_map=True)
            for idx_s in affs_s:
                t_id = ids[idx_s]
                t = self.__graph.vertex(t_id)
                # Compute path for target
                _, edges = gt.shortest_path(self.__graph, source=v, target=t, weights=prop_w_i, pred_map=None)
                length = .0
                for e in edges:
                    length += prop_l[e]
                if eus_map[v_id, t_id] <= 0:
                    continue
                sin = length / eus_map[v_id, t_id]
                # print sin - sin_map[t_id]
                if (sin >= rg_sin[0]) and (sin < rg_sin[1]):
                    prop_o[v] = affs[idx_s]
                    break

        # Edges loop (vertex minimum)
        for e in self.__graph.edges():
            s_id, t_id = e.source(), e.target()
            s, t = self.__graph.vertex(s_id), self.__graph.vertex(t_id)
            s_v, t_v = prop_o[s], prop_o[t]
            if s_v < t_v:
                prop_o_e[e] = s_v
            else:
                prop_o_e[e] = t_v

        self.__graph.vp[key_o] = prop_o
        self.__graph.ep[key_o] = prop_o_e

    # Metric for computing all shortest paths (local) centrality but weighting smooth paths (low sinuosity) and highly
    # connected ones. This function requires to have a GraphMCF reference tomogram (Algorithm 2)
    # dsts: distances for analysis (if list multi-scale analysis)
    # sg: sigma for analysis
    # key_o: key string for the output property
    # key_s: key string for path length measure (SGT_EDGE_LENGTH)
    # key_v: key string for vertex strength measure (default None, all vertices weight the same)
    # w_s: sinuosity fallen constant, [0, +infty] the smallest the slowest decay
    # Returns: a new vertex property with 'key_o' as key is generated
    def geo_centrality_2(self, dsts, sg, key_o, max_dst=MAX_FLOAT, key_s=SGT_EDGE_LENGTH, key_v=None, w_s=1):

        # Initialization
        if not hasattr(dsts, '__len__'):
            dsts = list(dsts)
        if self.__graph_mcf is None:
            error_msg = 'A reference GraphMCF is required for applying sinuosity penalization.'
            raise pexceptions.PySegInputError(expr='geo_centrality (GraphGT)', msg=error_msg)
        if key_s is not None:
            prop_s = self.__graph.ep[key_s]
        else:
            prop_s = self.__graph.new_vertex_property('float')
            prop_s.get_array()[:] = np.ones(shape=self.__graph.num_edges(), dtype=np.float32)
        if key_v is not None:
            prop_v = self.__graph.vp[key_v]
        else:
            prop_v = self.__graph.new_vertex_property('float')
            prop_v.get_array()[:] = np.ones(shape=self.__graph.num_vertices(), dtype=np.float32)
        prop_o = self.__graph.new_vertex_property('float')
        prop_o.get_array()[:] = np.zeros(shape=self.__graph.num_vertices(), dtype=np.float32)

        # Pre-computing metrics for sinuosity
        dists_map = gt.shortest_distance(self.__graph, weights=prop_s)
        eus_map = self.__vertices_eu_map()

        # Distances ranges


        # Vertices loop
        for v in self.__graph.vertices():

            # Extract neighbours indices
            v_id = int(v)
            dist_map = dists_map[v].get_array()
            ids = np.where((dist_map > 0) & (dist_map < max_dst))[0]

            # Skip trivial events
            if ids.shape[0] > 0:

                # This vertex radiates to neighbours
                dist_a, eu_a = dist_map[ids], eus_map[v_id, ids]
                sin = dist_a / eu_a
                sin_e = np.exp(-1.* w_s * (sin-1.))
                cte = prop_v[v] / float(ids.shape[0])
                for i, n_id in enumerate(ids):
                    n = self.__graph.vertex(n_id)
                    hold = prop_o[n]
                    hold += (cte*sin_e[i])
                    prop_o[n] = hold

        self.__graph.vp[key_o] = prop_o


    # Find paths which connects two segmented regions
    # out_key: property name for the output
    # seg_key: vertex property name for segmentation
    # s_lbl: label for source region in segmentation property
    # t_lbl: label for target region in segmentation property
    # w_key: name property for edge weighting
    # max_dist: maximum distance (in terms of SGT_EDGE_LENGTH) (default None)
    # Returns: paths centrality in a edge an vertex properties called 'out_key' and
    # 'edge_+out_key'
    def find_paths_seg(self, out_key, seg_key, s_lbl, t_lbl, w_key, max_dist=None):

        # Initialization
        prop_e_w = self.__graph.edge_properties[w_key]
        prop_e_d = self.__graph.edge_properties[SGT_EDGE_LENGTH]
        prop_v_s = self.__graph.vertex_properties[seg_key]
        prop_v_c = self.__graph.new_vertex_property('int')
        prop_e_c = self.__graph.new_edge_property('int')
        prop_v_t = self.__graph.new_vertex_property('bool')
        prop_v_t.get_array()[:] = np.zeros(shape=self.__graph.num_vertices(), dtype=bool)

        # Getting sources and targets
        sources = list()
        for v in self.__graph.vertices():
            if prop_v_s[v] == s_lbl:
                sources.append(v)
            if prop_v_s[v] == t_lbl:
                prop_v_t[v] = True

        # Loop for sources
        for s in sources:
            dist_map = gt.shortest_distance(self.__graph, source=s, weights=prop_e_d,
                                            max_dist=max_dist)
            ids = np.where(dist_map.get_array() < max_dist)[0]
            # Loop for reachable targets
            for i in range(ids.shape[0]):
                    t = self.__graph.vertex(ids[i])
                    if prop_v_t[t]:
                        v_path, e_path = gt.shortest_path(self.__graph, source=s, target=t,
                                                          weights=prop_e_w)
                        # Checking again the path length
                        length = 0
                        for e in e_path:
                            length += prop_e_d[e]
                        if length <= max_dist:
                            # Update vertices and edges centrality
                            for v in v_path:
                                prop_v_c[v] += 1
                            for e in e_path:
                                prop_e_c[e] += 1

        # Storing the new properties
        self.__graph.vertex_properties[out_key] = prop_v_c
        self.__graph.edge_properties['edge_'+out_key] = prop_e_c


    # Find subgraphs
    # key_v: string for the vertex property where graph_id is stored
    # Returns: the total number of subgraphs.
    def find_subgraphs(self, key_v):

        sgraph_id = self.__graph.new_vertex_property("int", vals=0)
        visitor = pyseg.factory.SubGraphVisitor(sgraph_id)
        for v in self.__graph.vertices():
            if sgraph_id[v] == 0:
                gt.dfs_search(self.__graph, v, visitor)
                visitor.update_sgraphs_id()
        self.__graph.vertex_properties[key_v] = sgraph_id

        return visitor.get_num_sgraphs()

    # Generates a subgraph (graph_tool graph, not a GraphGT) from a list of seed vertices id
    # v_list: list of vertices id (DPSTR_CELL)
    def gen_sgraph_vlist(self, v_list):

        # Creating LUTs for converting id between graphs
        # v_list_max = np.asarray(v_list).max() + 1
        prop_old_id = self.__graph.vertex_properties[DPSTR_CELL]
        id_max = prop_old_id.get_array().max() + 1
        lut_v = -1 * np.ones(shape=id_max, dtype=int)
        lut_v_id = -1 * np.ones(shape=id_max, dtype=int)
        lut_v_inv = -1 * np.ones(shape=id_max, dtype=int)
        for v in self.__graph.vertices():
            lut_v_id[prop_old_id[v]] = int(v)

        # New graph initialization
        graph = gt.Graph(directed=True)

        # Creating vertices
        for i, v_id in enumerate(v_list):
            v_old = int(lut_v_id[v_id])
            v_new = int(graph.add_vertex())
            lut_v[v_old] = v_new
            lut_v_inv[v_new] = v_old

        # Creating edges (only those which connect two survivor vertices are created)
        l_e_inv = list()
        for e in self.__graph.edges():
            s_new, t_new = lut_v[int(e.source())], lut_v[int(e.target())]
            if (s_new != -1) and (t_new != -1):
                graph.add_edge(graph.vertex(s_new), graph.vertex(t_new))
                l_e_inv.append(e)

        # Adding properties
        for prop in self.__graph.properties:
            p_type, p_key = prop[0], prop[1]
            if p_type == 'v':
                old_p = self.__graph.vp[p_key]
                new_p = graph.new_vertex_property(old_p.value_type())
                for v in graph.vertices():
                    new_p[v] = old_p[self.__graph.vertex(lut_v_inv[int(v)])]
                graph.vp[p_key] = new_p
            elif p_type == 'e':
                old_p = self.__graph.ep[p_key]
                new_p = graph.new_edge_property(old_p.value_type())
                cont = 0
                for e in graph.edges():
                    new_p[e] = old_p[l_e_inv[cont]]
                    cont += 1
                graph.ep[p_key] = new_p
            elif p_type == 'g':
                old_p = self.__graph.gp[p_key]
                new_p = graph.new_graph_property(old_p.value_type())
                graph.ep[p_key] = new_p
                graph.ep[p_key] = old_p

        return graph

    # Find peaks by (cluster centers) by DBSCAN graph clustering algorithm
    # prop_key: property name for stored the clustering property
    # key_dst: string to the edge property used to measure vertices distances
    # key_weight: string to the vertex property used to vertices weighting, peaks will be the maximum valued vertex of
    #             a cluster, if it is a GraphMCF object the peak will be the closest cluster center mass
    # eps: (default 0.5) The maximum distance between two samples for them to be considered as in the same neighborhood.
    # min_samples: (default 5) The number of samples in a neighborhood for a point to be considered
    #              as a core point. This includes the point itself.
    # ditype: data type for temporary shortest distance matrix, if None (default) it is set automatically to edge
    #         property type. In some cases allow to avoid MemoryErrors.
    # Returns: peaks ids (DPSTR_CELL) for clusters centers, clusters labels are stored internally in vertex property
    #          call prop_key where outliers have -1 label
    def find_peaks_dbscan(self, prop_key, key_dst, key_weigth, eps=.5, min_samples=5, ditype=None):

        # Input parsing
        try:
            prop_e = self.__graph.ep[key_dst]
        except KeyError:
            error_msg = 'Property ' + key_dst + ' does not exist for edges.'
            raise pexceptions.PySegInputError(expr='find_peaks_dbscan (GraphGT)', msg=error_msg)
        prop_w = None
        if not isinstance(key_weigth, pyseg.graph.GraphMCF):
            try:
                prop_w = self.__graph.vp[key_weigth].get_array()
            except KeyError:
                error_msg = 'Property ' + key_weigth + ' does not exist for vertices.'
                raise pexceptions.PySegInputError(expr='find_peaks_dbscan (GraphGT)', msg=error_msg)
        prop_id = self.__graph.vp[DPSTR_CELL].get_array()

        # DBSCAN
        self.dbscan(key_dst, eps, min_samples, rand=False, ditype=ditype)

        # Storing clustering in the input property
        prop_v = self.__graph.new_vertex_property('int')
        arr = self.__graph.vp[STR_DBSCAN_CLUST].get_array()
        prop_v.get_array()[:] = arr
        self.__graph.vp[prop_key] = prop_v

        # Finding peaks
        peaks = list()
        mx_lbl = arr.max()
        for lbl in range(mx_lbl):
            ids = np.where(arr == lbl)[0]
            if prop_w is None:
                coords = key_weigth.get_vertices_coords(ids)
                cm = coords.mean(axis=0)
                hold = coords - cm
                idx = ids[((hold*hold).sum(axis=1)).argmin()]
            else:
                hold = np.zeros(shape=len(ids), dtype=prop_w.dtype)
                for i, idx in enumerate(ids):
                    hold[i] = prop_w[idx]
                idx = ids[hold.argmax()]
            peaks.append(prop_id[idx])

        return peaks

    # Vertices suppression so as to ensure a minimum distance among them, it return a list of vertices ids
    # scale: scale suppression in nm
    # key_v: property weight key for sorting the vertices
    # conn: if True (default) suppression is only apply between connected vertices, otherwise the graph is treated
    #       just as a cloud of points
    # Returns: a list with the vertices ids (for GraphMCF) to delete
    def vertex_scale_supression(self, scale, key_v, conn=True):

        # Sorting (descending) the vertices
        try:
            prop_v = self.__graph.vp[key_v]
        except KeyError:
            error_msg = 'Property ' + key_v + ' does not exist for vertices.'
            raise pexceptions.PySegInputError(expr='vertex_scale_supression (GraphGT)', msg=error_msg)
        sort_ids = np.argsort(prop_v.get_array())[::-1]
        free_lut = np.ones(shape=self.__graph.num_vertices(), dtype=bool)
        del_l = list()
        prop_id = self.__graph.vp[DPSTR_CELL]

        if conn:

            # Graph version
            prop_e = self.__graph.ep[STR_VERT_DST]
            for s_idx in sort_ids:
                if free_lut[s_idx]:
                    s = self.__graph.vertex(s_idx)
                    dst_map = gt.shortest_distance(self.__graph,
                                                   source=s,
                                                   weights=prop_e,
                                                   max_dist=scale).get_array()
                    n_ids = np.where((dst_map>0) & (dst_map<=scale))[0]
                    for n_idx in n_ids:
                        if free_lut[n_idx]:
                            del_l.append(prop_id[self.__graph.vertex(n_idx)])
                            free_lut[n_idx] = False

        else:

            # Euclidean space version
            dsts_map = self.__vertices_eu_map()
            for s_idx in sort_ids:
                if free_lut[s_idx]:
                    dst_map = dsts_map[s_idx, :]
                    n_ids = np.where((dst_map>0) & (dst_map<=scale))[0]
                    for n_idx in n_ids:
                        if free_lut[n_idx]:
                            del_l.append(prop_id[self.__graph.vertex(n_idx)])
                            free_lut[n_idx] = False

        return del_l

    ########## INTERNAL FUNCTION AREA

    # Single thread for geodesic neighbors filter
    def __th_geo_neigh_filter(self, ids, sources, prop_v, prop_g, sig, n_sigs, c, prop_k, tlock):

        # Loop on sources
        for idx in ids:
            s = sources[idx]

            # Compute geodesic distance
            dist_map = gt.shortest_distance(self.__graph,
                                            source=s,
                                            weights=prop_g,
                                            max_dist=n_sigs[1])
            dists = dist_map.get_array()

            # Get neighbours in range
            n_ids = np.where((dists>n_sigs[0]) & (dists<n_sigs[1]))[0]

            # Compute weights in neighbours
            for n_idx in n_ids:
                v = self.__graph.vertex(n_idx)
                g = prop_v[v] * math.exp(dist_map[v]*c)
                tlock.acquire()
                prop_k[v] += g
                tlock.release()

    # Single thread for geodesic neighbors filter
    def __th_geo_size_filter(self, ids, sources, prop_v, prop_g, rad, rad_t, c, prop_k, tlock):

        # Loop on sources
        for idx in ids:
            s = sources[idx]

            # Compute geodesic distance
            dist_map = gt.shortest_distance(self.__graph,
                                            source=s,
                                            weights=prop_g,
                                            max_dist=rad_t)
            dists = dist_map.get_array()

            # Get neighbours in ranges
            n_ids = np.where(dists <= rad)[0]
            d_ids = np.where((dists>rad) & (dists<rad_t))[0]

            # Compute weights in neighbours
            for n_idx in n_ids:
                v = self.__graph.vertex(n_idx)
                hold = prop_v[v]
                tlock.acquire()
                prop_k[v] += hold
                tlock.release()

            # Compute weights in depletion zone
            for d_idx in d_ids:
                v = self.__graph.vertex(d_idx)
                hold = prop_v[v]
                tlock.acquire()
                prop_k[v] -= hold
                tlock.release()

    # Single thread for computing 'fix_len_path_centrality()' function
    def __fix_len_path_centrality(self, s_ids, source_ids, graph, prop_e_key, dist_max,
                                  prop_v_c_a, sd_cte):

        # Graph properties
        prop_e_d = graph.edge_properties[SGT_EDGE_LENGTH]
        prop_e_r = graph.edge_properties[prop_e_key]
        prop_v_id = graph.vertex_properties[DPSTR_CELL]
        prop_e_id = graph.edge_properties[DPSTR_CELL]

        # Main loop
        count = 0
        for s_id in s_ids:
            count += 1
            print(str(count) + ' of ' + str(len(s_ids)))
            s = graph.vertex(s_id)
            dist_map, p_map = gt.shortest_distance(graph, source=s, weights=prop_e_d, max_dist=dist_max,
                                                   pred_map=True)
            ids = np.where(dist_map.get_array() < dist_max)[0]
            for idx in ids:
                t = self.__graph.vertex(idx)
                v_path, e_path = gt.shortest_path(graph, source=s, target=t, weights=prop_e_r,
                                                  pred_map=p_map)
                if len(e_path) > 0:
                    # Measuring path length
                    length = 0
                    for e in e_path:
                        length += prop_e_d[e]
                    if sd_cte <= 0:
                        hold = 1. * len(v_path)
                    else:
                        v_ids = np.zeros(shape=len(v_path), dtype=int)
                        e_ids = np.zeros(shape=len(e_path), dtype=int)
                        for j, v in enumerate(v_path):
                            v_ids[j] = prop_v_id[v]
                        for j, e in enumerate(e_path):
                            e_ids[j] = prop_e_id[e]
                        # print e_ids
                        fil = filament.FilamentUDG(v_ids, e_ids, self.__graph_mcf)
                        hold = fil.get_sinuosity()
                        if hold > 0:
                            hold = fil.get_dst() / hold
                        else:
                            hold = 0.
                    # Update source centrality
                    prop_v_c_a[s_id] += hold

    # RWR linear equation solver without using inverse matrix operation
    # e: sources array (dense)
    # W: transposed transition matrix matrix (sparse)
    # c: restart probability
    # ep: threshold increment (in L2 norm) for stoping
    # m: maximum number of iterations (default 50, minimum 1)
    # Return: the relevance vector for the input source
    def __onthefly_solver(self, e, W, c, ep, m=50):

        W_p = W.multiply(1-c)
        p_s = e
        l2 = np.finfo('float').max
        count = 0
        while (l2 > ep) and (count < m):
            # Computation
            p_s_1 = W_p.dot(p_s) + c*e
            hold = p_s_1 - p_s
            l2 = np.sqrt(np.sum(hold * hold))
            p_s = p_s_1
            count += 1

        if count == m:
            print('WARNING (GraphGT, __onthefly_solver): Maximum number of ' \
                                 'iterations reached.', file=sys.stderr)

        return p_s_1

    # It adds weak edges to connectivity matrix so as to have a fully connected graph.
    # (Used for agglomerative hierarchical clustering)
    # X: arrays with point coordinates
    # conn_mat: connectivity matrix where the edges will be added
    def __full_connectivity(self, X, conn_mat):

        # Subgraphs visitor initialization
        sgraph_id = self.__graph.new_vertex_property("int", vals=0)
        visitor = pyseg.factory.SubGraphVisitor(sgraph_id)

        # Find subgraphs
        for v in self.__graph.vertices():
            if sgraph_id[v] == 0:
                gt.dfs_search(self.__graph, v, visitor)
                visitor.update_sgraphs_id()
        n_sgraphs = visitor.get_num_sgraphs()
        n_samples = X.shape[0]
        mx = np.finfo(float).max
        sources = np.zeros(shape=(n_sgraphs, n_sgraphs), dtype=float)
        targets = np.zeros(shape=(n_sgraphs, n_sgraphs), dtype=float)
        dists = mx * np.ones(shape=(n_sgraphs, n_sgraphs), dtype=float)
        sgraph_id_arr = sgraph_id.get_array()

        # Measuring distances among all points (FORCE BRUTE, O(Nxlog(N)))
        for i in range(n_samples):
            s_id = sgraph_id_arr[i] - 1
            for j in range(i+1, n_samples):
                t_id = sgraph_id_arr[j] - 1
                if s_id != t_id:
                    hold = X[i, :] - X[j, :]
                    dist = math.sqrt(np.sum(hold*hold))
                    if dist < dists[s_id, t_id]:
                        dists[s_id, t_id] = dist
                        sources[s_id, t_id] = i
                        targets[s_id, t_id] = j
                        dists[t_id, s_id] = dist
                        sources[t_id, s_id] = i
                        targets[t_id, s_id] = j

        # Build a temporal graph for connecting all subgraphs
        hold_graph = gt.Graph(directed=False)
        hold_graph.add_vertex(n_sgraphs)
        n_edges = int((n_sgraphs * (n_sgraphs - 1)) * 0.5)
        s_prop_arr = np.zeros(shape=n_edges, dtype=int)
        t_prop_arr = np.zeros(shape=n_edges, dtype=int)
        d_prop_arr = np.zeros(shape=n_edges, dtype=float)
        count = 0
        for i in range(n_sgraphs):
            for j in range(i+1, n_sgraphs):
                hold_graph.add_edge(hold_graph.vertex(i), hold_graph.vertex(j))
                s_prop_arr[count] = sources[i, j]
                t_prop_arr[count] = targets[i, j]
                d_prop_arr[count] = dists[i, j]
                count += 1
        d_prop = hold_graph.new_edge_property('float')
        d_prop.get_array()[:] = d_prop_arr

        # Computing minimum spanning tree for maximum connectivity simplification
        tree_map = gt.min_spanning_tree(hold_graph, weights=d_prop)

        # gt.graph_draw(hold_graph,
        #               output='/home/martinez/workspace/temp/hold.pdf')
        # hold_graph.set_edge_filter(tree_map)
        # gt.graph_draw(hold_graph,
        #               output='/home/martinez/workspace/temp/hold_tree.pdf')

        # Set the new edges to conectivity matrix
        mn = np.finfo(float).eps
        for i, e in enumerate(hold_graph.edges()):
            if tree_map[e] == 1:
                s = s_prop_arr[i]
                t = t_prop_arr[i]
                conn_mat[s, t] = mn

    # Computes a dense matrix with euclidean distances among vertices (it requires self.__graph_mcf not None)
    def __vertices_eu_map(self):

        # Getting vertices coordinates
        nv = self.__graph.num_vertices()
        prop_id = self.__graph.vp[DPSTR_CELL]
        coords = np.zeros(shape=(nv, 3), dtype=np.float32)
        skel = self.__graph_mcf.get_skel()
        for i in range(nv):
            v = self.__graph.vertex(i)
            x, y, z = skel.GetPoint(prop_id[v])
            coords[i, 0], coords[i, 1], coords[i, 2] = x, y, z

        # Compute euclidean distances matrix
        dists = np.zeros(shape=(nv, nv), dtype=np.float32)
        for i in range(nv):
            hold = coords[i, :] - coords
            hold = np.sqrt((hold*hold).sum(axis=1))
            dists[i, :] = hold

        return dists
