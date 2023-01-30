"""
Classes for modelling the structures connected to membranes

# Author: Antonio Martinez-Sanchez (Max Planck Institute for Biochemistry)
# Date: 12.06.15
"""

__author__ = 'martinez'

import numpy as np
import vtk
import math
import pyseg as ps
import graph_tool.all as gt
import copy
import operator
from sklearn.cluster import AffinityPropagation

try:
    import pexceptions
except:
    from pyseg import pexceptions

##### Global variables

MB_SEG_LBL = 1
MB_SIDE_LBL = 2
MB_BG_LBL = 0
STR_CONN_LBL = 'conn_lbl'
STR_CONN_DEG = 'conn_deg'
STR_CONN_LEN = 'conn_len'
STR_CONN_EFI = 'conn_efi'
STR_CONN_CID = 'cell_id'
STR_CONN_CLST = 'conn_clst'
STR_CONN_CENT = 'conn_cent'
DEG_EPS = 1 # pix
STR_SEG = 'conn_seg'

###########################################################################################
# Class for extracting the cloud of connectors (protein-membrane contact points) of
# a membrane
###########################################################################################

class CloudMbConn(object):

    # graph: parent GraphMCF
    # seg: tomogram with the membrane segmentation (membrane must be labeled as 1)
    def __init__(self, graph, seg):
        self.__graph = graph
        self.__skel = graph.get_skel()
        self.__seg = seg
        self.__coords = None
        self.__lbls = None
        self.__deg = None
        self.__i_points = 0 # Connectors index (non-repeated)
        self.__build()

    #### Set/Get methods area

    # Returns connectors coordinates as an array
    # sub: if True (default) coordinates are returned with subpixel precision
    # lbl: if not None (default) only connectors with label == 'lbl' are returned
    def get_coords(self, sub=True, lbl=None):

        if lbl is None:
            coords = self.__coords
        else:
            coords = list()
            for i, l in enumerate(self.__lbls):
                if l == lbl:
                    coords.append(self.__coords[i])
            coords = np.asarray(coords)

        if sub:
            return coords
        return np.round(coords)

    # Returns connector locations in a VTK file (vtp)
    def get_vtp(self):

        # Initialization
        points = vtk.vtkPoints()
        cells = vtk.vtkCellArray()
        lbl_data = vtk.vtkIntArray()
        lbl_data.SetNumberOfComponents(1)
        lbl_data.SetName(STR_CONN_LBL)
        deg_data = vtk.vtkIntArray()
        deg_data.SetNumberOfComponents(1)
        deg_data.SetName(STR_CONN_DEG)

        # Loop
        # for i, l in enumerate(self.__lbls):
        for i in range(self.__i_points + 1):
            x, y, z = self.__coords[i]
            points.InsertPoint(i, x, y, z)
            cells.InsertNextCell(1)
            cells.InsertCellPoint(i)
            lbl_data.InsertTuple1(i, self.__lbls[i])
            deg_data.InsertTuple1(i, self.__deg[i])

        # Poly building
        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        poly.SetVerts(cells)
        poly.GetCellData().AddArray(lbl_data)
        poly.GetCellData().AddArray(deg_data)

        return poly

    #### Internal functionality area

    # Find the connectors from the input GraphMCF and segmentation
    def __build(self):

        # Initialization
        edges_list = self.__graph.get_edges_list()
        n_edges = len(edges_list)
        self.__coords = (-1) * np.ones(shape=(n_edges, 3), dtype=float)
        self.__deg = np.zeros(shape=n_edges, dtype=int)
        self.__lbls = (-1) * np.ones(shape=n_edges, dtype=int)
        self.__i_points = 0

        # Loop for finding the edge which contains a connector point
        x_E, y_E, z_E = self.__seg.shape
        for e in edges_list:
            s = self.__graph.get_vertex(e.get_source_id())
            t = self.__graph.get_vertex(e.get_target_id())
            x_s, y_s, z_s = self.__graph.get_vertex_coords(s)
            x_s, y_s, z_s = int(round(x_s)), int(round(y_s)), int(round(z_s))
            x_t, y_t, z_t = self.__graph.get_vertex_coords(t)
            x_t, y_t, z_t = int(round(x_t)), int(round(y_t)), int(round(z_t))
            if (x_s < x_E) and (y_s < y_E) and (z_s < z_E) and \
                    (x_t < x_E) and (y_t < y_E) and (z_t < z_E):
                s_lbl = self.__seg[x_s, y_s, z_s]
                t_lbl = self.__seg[x_t, y_t, z_t]
                # Check transmembrane edge
                if (s_lbl != t_lbl) and ((s_lbl == MB_SEG_LBL) or (t_lbl == MB_SEG_LBL)):
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
                    # Find the connector starting from minimum out of membrane
                    lbl = s_lbl
                    if s_lbl == MB_SEG_LBL:
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
                        if self.__seg[x_p_r, y_p_r, z_p_r] == MB_SEG_LBL:
                            p = np.asarray(self.__skel.GetPoint(hold_p))
                            p_id = self.__is_already(p)
                            if p_id is None:
                                self.__coords[self.__i_points] = p
                                self.__lbls[self.__i_points] = lbl
                                self.__deg[self.__i_points] += 1
                                self.__i_points += 1
                            else:
                                self.__deg[p_id] += 1
                            found = True
                            break
                        hold_p = arc_s.get_point_id(i)
                    if not found:
                        for i in range(1, arc_t.get_npoints()):
                            hold_p = arc_t.get_point_id(i)
                            x_p, y_p, z_p = self.__skel.GetPoint(hold_p)
                            x_p_r, y_p_r, z_p_r = int(round(x_p)), int(round(y_p)), \
                                                    int(round(z_p))
                            if self.__seg[x_p_r, y_p_r, z_p_r] == lbl:
                                p = np.asarray(self.__skel.GetPoint(hold_p))
                                p_id = self.__is_already(p)
                                if p_id is None:
                                    self.__coords[self.__i_points] = p
                                    self.__lbls[self.__i_points] = lbl
                                    self.__deg[self.__i_points] += 1
                                    self.__i_points += 1
                                else:
                                    self.__deg[p_id] += 1
                                found = True
                                break
                    if not found:
                        error_msg = 'Unexpected event.'
                        print('WARNING: ' + error_msg)
                        # raise pexceptions.PySegTransitionError(expr='__build (CloudMbConn)',
                        #                                        msg=error_msg)

    # Detect if there is already a connector in DEG_EPS radius and if true returns its
    # id, otherwise return None
    # p: connector coordinates as
    def __is_already(self, p):

        if self.__i_points > 0:
            hold = self.__coords[0:self.__i_points, :]
            hold = p - hold
            hold = np.sum(hold*hold, axis=1)
            p_id = hold.argmin()
            if math.sqrt(hold[p_id]) < DEG_EPS:
                return p_id
        return None

###########################################################################################
# Class for building the graph connectors attached to a membrane at one side
###########################################################################################

class ConnSubGraph(object):

    # NOTE: the subgraph is actually not built until build() is not successfully called
    # graph: parent GraphMCF
    # seg: tomogram with the membrane segmentation (membrane must be labeled as 1 and
    # its side to 2)
    def __init__(self, graph, seg, density):
        self.__graph = copy.deepcopy(graph)
        self.__skel = graph.get_skel()
        self.__seg = seg
        self.__conn_p = None
        self.__conn_l = None
        self.__sgraph = None
        self.__conns = None
        self.__lengths = None
        self.__clsts = None
        self.__cents = None
        self.__density = density

    #### Set/Get methods area

    def get_sub_graph(self):
        return self.__graph

    # Print connectors (1) and directly attached vertices (2) and paths between them (3)
    def get_conn_vtp(self):

        if self.__conn_p is None:
            return None

        # Initialization
        points = vtk.vtkPoints()
        verts = vtk.vtkCellArray()
        lines = vtk.vtkCellArray()
        lbl_data = vtk.vtkIntArray()
        lbl_data.SetNumberOfComponents(1)
        lbl_data.SetName(STR_CONN_LBL)
        cid_data = vtk.vtkIntArray()
        cid_data.SetNumberOfComponents(1)
        cid_data.SetName(STR_CONN_CID)
        len_data = vtk.vtkFloatArray()
        len_data.SetNumberOfComponents(1)
        len_data.SetName(STR_CONN_LEN)
        if self.__clsts is not None:
            clst_data = vtk.vtkIntArray()
            clst_data.SetNumberOfComponents(1)
            clst_data.SetName(STR_CONN_CLST)
            cent_data = vtk.vtkIntArray()
            cent_data.SetNumberOfComponents(1)
            cent_data.SetName(STR_CONN_CENT)
            c_count = 0

        # Loop
        point_id = 0
        cell_id = 0
        for i, paths in enumerate(self.__conn_p):
            if len(paths) > 0:
                for j, path in enumerate(paths):
                    # Connector
                    x, y, z = self.__skel.GetPoint(path[-1])
                    points.InsertPoint(point_id, x, y, z)
                    verts.InsertNextCell(1)
                    verts.InsertCellPoint(point_id)
                    lbl_data.InsertNextTuple((1,))
                    len_data.InsertNextTuple((self.__conn_l[i][j],))
                    if self.__clsts is not None:
                        clst_data.InsertNextTuple((self.__clsts[c_count],))
                        cent_data.InsertNextTuple((self.__cents[c_count],))
                        c_count += 1
                    cid_data.InsertNextTuple((cell_id,))
                    point_id += 1
                    cell_id += 1
                    # Directly attached vertex
                    x, y, z = self.__skel.GetPoint(path[0])
                    points.InsertPoint(point_id, x, y, z)
                    verts.InsertNextCell(1)
                    verts.InsertCellPoint(point_id)
                    lbl_data.InsertNextTuple((2,))
                    len_data.InsertNextTuple((self.__conn_l[i][j],))
                    if self.__clsts is not None:
                        clst_data.InsertNextTuple((-1,))
                        cent_data.InsertNextTuple((-1,))
                    cid_data.InsertNextTuple((cell_id,))
                    point_id += 1
                    cell_id += 1

        # Paths
        for i, paths in enumerate(self.__conn_p):
            if len(paths) > 0:
                for j, path in enumerate(paths):
                    lines.InsertNextCell(len(path))
                    for p in path:
                        x, y, z = self.__skel.GetPoint(p)
                        points.InsertPoint(point_id, x, y, z)
                        lines.InsertCellPoint(point_id)
                        point_id += 1
                    lbl_data.InsertNextTuple((3,))
                    len_data.InsertNextTuple((self.__conn_l[i][j],))
                    if self.__clsts is not None:
                        clst_data.InsertNextTuple((-1,))
                        cent_data.InsertNextTuple((-1,))
                    cid_data.InsertNextTuple((cell_id,))
                    cell_id += 1

        # Poly building
        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        poly.SetVerts(verts)
        poly.SetLines(lines)
        poly.GetCellData().AddArray(lbl_data)
        poly.GetCellData().AddArray(len_data)
        poly.GetCellData().AddArray(cid_data)
        if self.__clsts is not None:
            poly.GetCellData().AddArray(clst_data)
            poly.GetCellData().AddArray(cent_data)

        return poly

    # Print paths and lengths
    def get_graph_mcf(self):
        return self.__graph

    #### Internal functionality area

    # Build the connectors graph
    # conn_len: maximum length for connectors
    # edge_len: maximum length for edges
    def build(self, conn_len=None, edge_len=None):

        # Initialization
        nid = self.__graph.get_nid()
        self.__conn_p = np.zeros(shape=nid, dtype=object)
        self.__conn_l = np.zeros(shape=nid, dtype=object)
        for i in range(nid):
            self.__conn_p[i] = list()
            self.__conn_l[i] = list()

        # Finding connectors
        # Loop for finding the edge which contains a connector point
        for e in self.__graph.get_edges_list():
            s = self.__graph.get_vertex(e.get_source_id())
            t = self.__graph.get_vertex(e.get_target_id())
            x_s, y_s, z_s = self.__graph.get_vertex_coords(s)
            x_t, y_t, z_t = self.__graph.get_vertex_coords(t)
            s_lbl = self.__seg[int(round(x_s)), int(round(y_s)), int(round(z_s))]
            t_lbl = self.__seg[int(round(x_t)), int(round(y_t)), int(round(z_t))]
            # Check transmembrane edge
            if (s_lbl != t_lbl) and \
                    ((s_lbl == MB_SIDE_LBL) or (t_lbl == MB_SIDE_LBL)):
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
                # Find the connector starting from minimum out of membrane
                vc = s.get_id()
                if s_lbl != MB_SIDE_LBL:
                    hold = arc_s
                    arc_s = arc_t
                    arc_t = hold
                    vc = t.get_id()
                found = False
                p_path = list()
                hold_l = 0
                hold_p = arc_s.get_point_id(0)
                x_h, y_h, z_h = self.__skel.GetPoint(hold_p)
                p_path.append(hold_p)
                for i in range(1, arc_s.get_npoints()):
                    curr_p = arc_s.get_point_id(i)
                    x_p, y_p, z_p = self.__skel.GetPoint(curr_p)
                    x_p_r, y_p_r, z_p_r = int(round(x_p)), int(round(y_p)), \
                                            int(round(z_p))
                    if self.__seg[x_p_r, y_p_r, z_p_r] == MB_SEG_LBL:
                        found = True
                        break
                    x_h, y_h, z_h = x_h-x_p, y_h-y_p, z_h-z_p
                    hold_l += math.sqrt(x_h*x_h + y_h*y_h + z_h*z_h)
                    x_h, y_h, z_h = x_p, y_p, z_p
                    if (conn_len is not None) and (hold_l > conn_len):
                        break
                    hold_p = curr_p
                    p_path.append(hold_p)
                if not found:
                    pt_lst = arc_t.get_ids()[::-1]
                    for i in range(1, len(pt_lst)):
                        curr_p = pt_lst[i]
                        x_p, y_p, z_p = self.__skel.GetPoint(curr_p)
                        x_p_r, y_p_r, z_p_r = int(round(x_p)), int(round(y_p)), \
                                                int(round(z_p))
                        if self.__seg[x_p_r, y_p_r, z_p_r] == MB_SEG_LBL:
                            found = True
                            break
                        x_h, y_h, z_h = x_h-x_p, y_h-y_p, z_h-z_p
                        hold_l += math.sqrt(x_h*x_h + y_h*y_h + z_h*z_h)
                        x_h, y_h, z_h = x_p, y_p, z_p
                        if (conn_len is not None) and (hold_l > conn_len):
                            break
                        hold_p = curr_p
                        p_path.append(hold_p)
                if (conn_len is not None) and (hold_l > conn_len):
                    pass
                elif not found:
                    error_msg = 'Unexpected event.'
                    print('WARNING: ' + error_msg)
                    # raise pexceptions.PySegTransitionError(expr='__build (CloudMbConn)',
                    #                                        msg=error_msg)
                else:
                    self.__conn_p[vc].append(p_path)

        # Graph filtering
        self.__graph.add_scalar_field_nn(self.__seg, STR_SEG)
        self.__graph.threshold_seg_region(STR_SEG, MB_BG_LBL, keep_b=False)
        self.__graph.threshold_seg_region(STR_SEG, MB_SEG_LBL, keep_b=False)
        if edge_len is not None:
            self.__graph.threshold_edges(ps.globals.SGT_EDGE_LENGTH, edge_len, operator.gt)
        graph_gt_h = ps.graph.GraphGT(self.__graph)
        graph_gt = graph_gt_h.get_gt()
        w_prop = graph_gt.edge_properties[ps.globals.STR_EDGE_FNESS]
        # Inverting weighting property
        w_prop.get_array()[:] = 1. / w_prop.get_array()
        l_prop = graph_gt.edge_properties[ps.globals.SGT_EDGE_LENGTH]
        i_prop = graph_gt.vertex_properties[ps.globals.DPSTR_CELL]

        # Computing connectors vertex properties
        cv_prop_typ = graph_gt.new_vertex_property('int', vals=1)
        cv_prop_len = graph_gt.new_vertex_property('float', vals=-1.)
        nconn_lut = np.ones(shape=graph_gt.num_vertices(), dtype=bool)
        for v in graph_gt.vertices():
            v_id = i_prop[v]
            p_paths = self.__conn_p[v_id]
            if len(p_paths) > 0:
                for path in p_paths:
                    hold_l = 0
                    xh, yh, zh = self.__skel.GetPoint(path[0])
                    for p in path[1::]:
                        x, y, z = self.__skel.GetPoint(p)
                        xp, yp, zp = xh-x, yh-y, zh-z
                        length = math.sqrt(xp*xp + yp*yp + zp*zp)
                        hold_l += length
                        xh, yh, zh = x, y, z
                    self.__conn_l[v_id].append(hold_l)
                cv_prop_len[v] = np.asarray(self.__conn_l[v_id]).min()
                cv_prop_typ[v] = 2
                nconn_lut[int(v)] = False

        # Computing vertex properties
        mx = np.finfo(w_prop.get_array().dtype).max
        gdists_map = gt.shortest_distance(graph_gt, weights=w_prop)
        for v in graph_gt.vertices():
            gdists = gdists_map[v].get_array()
            gdists[nconn_lut] = np.inf
            id_min = gdists.argmin()
            t = graph_gt.vertex(id_min)
            if gdists[id_min] < mx:
                dist = gt.shortest_distance(graph_gt, source=v, target=t, weights=l_prop)
                cv_prop_len[v] = cv_prop_len[t] + dist

        graph_gt.vertex_properties[STR_CONN_LEN] = cv_prop_len
        self.__sgraph = graph_gt
        graph_gt_h.add_prop_to_GraphMCF(self.__graph, STR_CONN_LEN, up_index=True)

        # Store connectors as class variable
        self.__conns, self.__lengths = self.__get_conn_array()
        self.__clsts = None
        self.__cents = None

    # Clusters the connectors
    def conn_clst(self):

        # Initialization
        n_conns = self.__conns.shape[0]
        mx = np.finfo(float).max
        dist_mat = mx * np.ones(shape=(n_conns, n_conns), dtype=float)
        graph_gt = ps.graph.GraphGT(self.__graph).get_gt()
        i_prop = graph_gt.vertex_properties[ps.globals.DPSTR_CELL]
        w_prop = graph_gt.edge_properties[ps.globals.SGT_EDGE_LENGTH]

        # Computing distances matrix
        for i in range(n_conns):
            s_id = self.__conns[i]
            dist_mat[i, i] = 0.
            for j in range(i+1, n_conns):
                t_id = self.__conns[j]
                if s_id == t_id:
                    dist_mat[i, j] = self.__lengths[i] + self.__lengths[j]
                else:
                    s = gt.find_vertex(graph_gt, i_prop, s_id)[0]
                    t = gt.find_vertex(graph_gt, i_prop, t_id)[0]
                    dist = gt.shortest_distance(graph_gt, source=s, target=t,
                                                weights=w_prop)
                    if dist < mx:
                        dist_mat[i, j] = dist + self.__lengths[i] + self.__lengths[j]

        # Ceiling max values for avoiding further overflows
        id_max = dist_mat == mx
        dist_mat[id_max] = np.sum(dist_mat[np.invert(id_max)])
        # Clustering
        cluster = AffinityPropagation(affinity='precomputed', preference=-100)
        cluster.fit((-1) * dist_mat)

        # Store the result
        lut_rand = np.random.randint(0, n_conns, n_conns)
        self.__clsts = np.zeros(shape=n_conns, dtype=int)
        for i in range(n_conns):
            self.__clsts[i] = lut_rand[cluster.labels_[i]]
        # Centers
        self.__cents = (-1) * np.ones(shape=n_conns, dtype=int)
        for i in range(len(cluster.cluster_centers_indices_)):
            v = cluster.cluster_centers_indices_[i]
            self.__cents[v] = lut_rand[cluster.labels_[v]]

    # Wrapping function for vertices thresholding
    def threshold_vertices(self, key_prop, th, oper):
        self.__graph.threshold_vertices(key_prop, th, oper)

#### Internal functionality area

    def __get_conn_array(self):
        conns = list()
        lengths = list()
        for paths in self.__conn_p:
            if len(paths) > 0:
                for path in paths:
                    conns.append(path[0])
                    hold_l = 0
                    xh, yh, zh = self.__skel.GetPoint(path[0])
                    for p in path[1::]:
                        x, y, z = self.__skel.GetPoint(p)
                        xp, yp, zp = xh-x, yh-y, zh-z
                        hold_l += math.sqrt(xp*xp + yp*yp + zp*zp)
                        xh, yh, zh = x, y, z
                    lengths.append(hold_l)

        return np.asarray(conns, dtype=int), np.asarray(lengths, dtype=float)




