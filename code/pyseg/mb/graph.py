"""
Classes for extending GraphMCF functionality for membranes

# Author: Antonio Martinez-Sanchez (Max Planck Institute for Biochemistry)
# Date: 03.11.15
"""

__author__ = 'martinez'

from pyseg.globals import *
from pyseg.graph import GraphGT
from pyseg import disperse_io
from pyseg import pexceptions
import math
import graph_tool.all as gt
from variables import *
from pyseg.graph import GraphMCF
from pyseg.factory import SubGraphVisitor
from pyseg.filament import FilamentLDG, FilamentUDG, NetFilamentsSyn


XML_MBT_MIN = 'in'
XML_MBT_MOUT = 'out'
XML_MBT_MPERI = 'per_in'
XML_MBT_MPERO = 'per_out'

##################################################################################################
# Generic extension for dealing with GraphMCF of membranes
#
#
class MbGraphMCF(GraphMCF):

    #### Constructor Area

    # During building
    # skel: DisPerSe skeleton
    # manifolds: DisPerSe manifolds
    # density: image density map
    # mb_seg: tomogram with membrane segmentation (1-mb, 2-inside and 3-outside)
    def __init__(self, skel, manifolds, density, mb_seg):
        super(MbGraphMCF, self).__init__(skel, manifolds, density)
        self.__mb_seg = mb_seg
        self.__mb_fils = np.zeros(shape=self.get_nid(), dtype=object)
        self.add_scalar_field_nn(self.__mb_seg, MB_SEG)
        self.__mb_dst = disperse_io.seg_dist_trans(self.__mb_seg == MB_LBL) * self.get_resolution()

    #### Set/Get functionality

    #### External functionality

    def get_mb_vertices_list(self):
        verts_mb = list()
        prop_seg_id = self.get_prop_id(MB_SEG)
        for v in self.get_vertices_list():
            if self.get_prop_entry_fast(prop_seg_id, v.get_id(), 1, np.int)[0] == MB_LBL:
                verts_mb.append(v)
        return verts_mb

    # Euclidean (through euclidean space) shortest distance to membrane
    # Returns: result stored as property MB_EU_DST
    def compute_mb_eu_dst(self):
        self.add_scalar_field(self.__mb_dst, MB_EU_DST)

    # Geodesic (through graph and euclidean) distance to shortest contact point,
    # it also computes differential geometry property for filaments
    # update: if True (default) then membrane segmentation is added as scalar field and if edges length
    # and edge filamentness are computed
    # Returns: result stored as property MB_GEO_DST, MB_GEO_LEN and MB_GEO_SIM
    def compute_mb_geo(self, update=True):

        # Initialization
        if update or (self.get_prop_id(MB_SEG) is None):
            self.add_scalar_field_nn(self.__mb_seg, MB_SEG)
        if update or (self.get_prop_id(SGT_EDGE_LENGTH) is None):
            self.compute_edges_length(SGT_EDGE_LENGTH, 1, 1, 1, False)
        if update or (self.get_prop_id(STR_EDGE_FNESS) is None):
            self.compute_edge_filamentness()
        key_dst_id = self.get_prop_id(MB_GEO_DST)
        if key_dst_id is None:
            key_dst_id = self.add_prop(MB_GEO_DST, 'float', 1)
        key_len_id = self.get_prop_id(MB_GEO_LEN)
        if key_len_id is None:
            key_len_id = self.add_prop(MB_GEO_LEN, 'float', 1)
        key_sin_id = self.get_prop_id(MB_GEO_SIN)
        if key_sin_id is None:
            key_sin_id = self.add_prop(MB_GEO_SIN, 'float', 1)
        key_kt_id = self.get_prop_id(MB_GEO_KT)
        if key_kt_id is None:
            key_kt_id = self.add_prop(MB_GEO_KT, 'float', 1)
        key_tt_id = self.get_prop_id(MB_GEO_TT)
        if key_tt_id is None:
            key_tt_id = self.add_prop(MB_GEO_TT, 'float', 1)
        key_ns_id = self.get_prop_id(MB_GEO_NS)
        if key_ns_id is None:
            key_ns_id = self.add_prop(MB_GEO_NS, 'float', 1)
        key_bs_id = self.get_prop_id(MB_GEO_BS)
        if key_bs_id is None:
            key_bs_id = self.add_prop(MB_GEO_BS, 'float', 1)
        key_al_id = self.get_prop_id(MB_GEO_APL)
        if key_al_id is None:
            key_al_id = self.add_prop(MB_GEO_APL, 'float', 1)
        key_cont_id = self.get_prop_id(MB_CONT_COORD)
        if key_cont_id is None:
            key_cont_id = self.add_prop(MB_CONT_COORD, 'float', 3)

        # Getting the graph GraphGT
        graph_gt = GraphGT(self).get_gt()
        # prop_efn = graph_gt.edge_properties[STR_FIELD_VALUE]
        prop_elen = graph_gt.edge_properties[SGT_EDGE_LENGTH]
        prop_seg = graph_gt.vertex_properties[MB_SEG]
        prop_eid = graph_gt.edge_properties[DPSTR_CELL]
        prop_vid = graph_gt.vertex_properties[DPSTR_CELL]

        # Creating LUTs
        lut_in = list()
        lut_out = list()
        for v in graph_gt.vertices():
            seg_lbl = prop_seg[v]
            if seg_lbl == MB_IN_LBL:
                lut_in.append(int(v))
            elif seg_lbl == MB_OUT_LBL:
                lut_out.append(int(v))
            elif seg_lbl == MB_LBL:
                lut_in.append(int(v))
                lut_out.append(int(v))
        lut_in = np.asarray(lut_in, dtype=np.int)
        lut_out = np.asarray(lut_out, dtype=np.int)

        # Loop for all vertices
        for v in graph_gt.vertices():
            # Look for vertices out the membrane
            seg_lbl = prop_seg[v]
            if seg_lbl != MB_LBL:
                # Getting lut for vertices on the other side of the membrane
                if seg_lbl == MB_IN_LBL:
                    lut_oth = lut_out
                elif seg_lbl == MB_OUT_LBL:
                    lut_oth = lut_in
                else:
                    continue
                # Finding the shortest geodesic path to other vertices which crosses the membrane
                # dists = gt.shortest_distance(graph_gt, source=v, weights=prop_efn).get_array()
                dists = gt.shortest_distance(graph_gt, source=v, weights=prop_elen).get_array()
                dists_pot = dists[lut_oth]
                if len(dists_pot) > 0:
                    t = graph_gt.vertex(lut_oth[np.argmin(dists_pot)])
                    # Finding geodesic shortest path to shortest vertex
                    # v_path, e_path = gt.shortest_path(graph_gt, v, t, weights=prop_efn)
                    v_path, e_path = gt.shortest_path(graph_gt, v, t, weights=prop_elen)
                    if len(e_path) > 0:
                        contact = None
                        # Measuring the path (on euclidean space) until membrane contact point
                        fil_v_ids = list()
                        for h_v in v_path[0:-1]:
                            fil_v_ids.append(prop_vid[h_v])
                        fil_p_ids = list()
                        # Initialize the filament starting point
                        fil_p_ids.append(prop_vid[v])
                        path_len = .0
                        for e in e_path[0:-1]:
                            path_len += prop_elen[e]
                            # Add in a ordered ways path points to the filament
                            p_ids = self.get_edge_ids(self.get_edge(prop_eid[e]))
                            if p_ids[-1] == fil_p_ids[-1]:
                                p_ids = p_ids[::-1]
                            fil_p_ids += p_ids[1::]
                        edge = self.get_edge(prop_eid[e_path[-1]])
                        e_ids = self.get_edge_ids(edge)
                        e_coords = np.asarray(self.get_edge_arcs_coords(edge), dtype=np.float)
                        e_coords_int = np.asarray(e_coords.round(), dtype=np.int)
                        try:
                            xi, yi, zi = e_coords_int[0, :]
                            mb_i = self.__mb_seg[xi, yi, zi]
                            # Reverse for ending with vertex through membrane
                            if  mb_i != seg_lbl:
                                e_coords = e_coords[::-1]
                                e_coords_int = e_coords_int[::-1]
                                e_ids = e_ids[::-1]
                            contact = e_coords[0, :]
                            fil_p_ids.append(e_ids[0])
                            # Loop for length increasing
                            xh, yh, zh = e_coords[0, :]
                            for i in range(1, e_coords.shape[0]):
                                x, y, z = e_coords_int[i, :]
                                if self.__mb_seg[x, y, z] == seg_lbl:
                                    x, y, z = e_coords[i, :]
                                    hold_len = np.asarray((x-xh, y-yh, z-zh), dtype=np.float)
                                    path_len += (math.sqrt((hold_len*hold_len).sum()) * self.get_resolution())
                                    xh, yh, zh = x, y, z
                                    contact = e_coords[i, :]
                                    fil_p_ids.append(e_ids[i])
                                else:
                                    break
                        except IndexError:
                            pass

                        # Compute metrics only if a contact is found
                        if contact is not None:

                            # Add filament
                            v_id = prop_vid[v]
                            if len(fil_p_ids) > 2:
                                self.add_mb_fil(v_id, fil_v_ids, fil_p_ids)
                                fil = self.__mb_fils[v_id]
                                self.set_prop_entry_fast(key_dst_id, (fil.get_dst(),), v_id, 1)
                                self.set_prop_entry_fast(key_len_id, (fil.get_length(),), v_id, 1)
                                self.set_prop_entry_fast(key_sin_id, (fil.get_sinuosity(),), v_id, 1)
                                self.set_prop_entry_fast(key_kt_id, (fil.get_total_k(),), v_id, 1)
                                self.set_prop_entry_fast(key_tt_id, (fil.get_total_t(),), v_id, 1)
                                self.set_prop_entry_fast(key_ns_id, (fil.get_total_ns(),), v_id, 1)
                                self.set_prop_entry_fast(key_bs_id, (fil.get_total_bs(),), v_id, 1)
                                self.set_prop_entry_fast(key_al_id, (fil.get_apex_length(),), v_id, 1)
                                self.set_prop_entry_fast(key_cont_id, (contact[0], contact[1], contact[2]),
                                                         v_id, 3)
                            else:
                                print 'WARNING: filament with less than 3 points!'

    def add_mb_fil(self, v_id, v_ids, p_ids):
        self.__mb_fils[v_id] = FilamentLDG(v_ids, p_ids, self)

    def get_mb_fil(self, v_id):
        return self.__mb_fils[v_id]

    # Cloud of coordinates for all vertices in filaments of a membrane slice
    # slices: Slice object with the membrane slice information
    # Return: an array with points coordinates
    def get_cloud_mb_slice_fils(self, slice):

        # Initialization
        seg_id = self.get_prop_id(MB_SEG)
        seg_dt = disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=seg_id))
        eu_id = self.get_prop_id(MB_EU_DST)
        eu_dt = disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=eu_id))
        geo_id = self.get_prop_id(MB_GEO_DST)
        geo_dt = disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=geo_id))
        gl_id = self.get_prop_id(MB_GEO_LEN)
        gl_dt = disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=gl_id))
        sin_id = self.get_prop_id(MB_GEO_SIN)
        sin_dt = disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=sin_id))

        # Compute GraphGT
        if self._GraphMCF__graph_gt is None:
            self.compute_graph_gt()
        graph_gt = self._GraphMCF__graph_gt
        lut_gt = (-1) * np.ones(shape=self.get_nid(), dtype=object)
        prop_vid = graph_gt.vertex_properties[DPSTR_CELL]
        prop_vs = graph_gt.vertex_properties[MB_SEG]
        prop_elen = graph_gt.edge_properties[SGT_EDGE_LENGTH]

        # Find tail vertices within slice
        lut_ver = (-1) * np.ones(shape=self.get_nid(), dtype=np.int)
        cont = 0
        v_ids = list()
        for v in graph_gt.vertices():
            v_id = prop_vid[v]
            seg = self.get_prop_entry_fast(seg_id, v_id, 1, seg_dt)[0]
            eu = self.get_prop_entry_fast(eu_id, v_id, 1, eu_dt)[0]
            geo = self.get_prop_entry_fast(geo_id, v_id, 1, geo_dt)[0]
            gl = self.get_prop_entry_fast(gl_id, v_id, 1, gl_dt)[0]
            sin = self.get_prop_entry_fast(sin_id, v_id, 1, sin_dt)[0]
            if slice.test(side=seg, eu_dst=eu, geo_dst=geo, geo_len=gl, sin=sin):
                if lut_ver[v_id] == -1:
                    lut_ver[v_id] = cont
                    v_ids.append(v_id)
                    lut_gt[v_id] = int(v)
                    cont += 1

        # Find shortest paths (filaments)
        mat_geo = gt.shortest_distance(graph_gt, weights=prop_elen)
        h_v_ids = list()
        for v_id in v_ids:
            # print 'v_id=' + str(v_id)
            s = graph_gt.vertex(lut_gt[v_id])
            s_lbl = prop_vs[s]
            ts_gt = np.argsort(mat_geo[s])
            for t_gt in ts_gt:
                t = graph_gt.vertex(t_gt)
                if prop_vs[t] != s_lbl:
                    v_path, _ = gt.shortest_path(graph_gt, s, t, weights=prop_elen)
                    for v_p in v_path[:-1]:
                        if prop_vs[v_p] != s_lbl:
                            break
                        v_p_id = prop_vid[v_p]
                        if lut_ver[v_p_id] == -1:
                            lut_ver[v_p_id] = cont
                            h_v_ids.append(v_p_id)
                            cont += 1
                    break
        v_ids += h_v_ids

        # Find edges within slice
        e_ids = list()
        for edge in self.get_edges_list():
            s_id = lut_ver[edge.get_source_id()]
            t_id = lut_ver[edge.get_target_id()]
            if (s_id > -1) and (t_id > -1):
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
        rids = np.zeros(shape=vertices_gt.shape[0], dtype=np.int)
        for i, v in enumerate(vertices_gt):
            if sgraph_id[v] == 0:
                gt.dfs_search(graph, v, visitor)
                visitor.update_sgraphs_id()
            v_id = v_ids[i]
            rids[i] = v_id
            coords[i, :] = self._GraphMCF__skel.GetPoint(v_id)

        # Clusters filtering
        ids = sgraph_id.get_array()
        max_id = ids.max() + 1
        lut_counts = np.zeros(shape=max_id, dtype=np.int)
        for idy in ids:
            lut_counts[idy] += 1
        hold_coords = list()
        hold_ids = list()
        for idy in range(1, max_id):
            counts = lut_counts[idy]
            if counts > 0:
                if slice.test(cnv=counts):
                    c_ids = np.where(ids == idy)[0]
                    if len(c_ids) > 0:
                        for c_id in c_ids:
                            hold_coords.append(coords[c_id, :])
                            hold_ids.append(rids[c_id])

        # Computing mask
        mask_out = (self.__mb_seg == slice.get_side())
        mask_out *= ((self.__mb_dst >= slice.get_eu_dst_low()) * (self.__mb_dst <= slice.get_eu_dst_high()))

        return np.asarray(hold_coords, dtype=np.float), np.asarray(hold_ids, dtype=np.int), mask_out

    # Cloud of points in a slice
    # slices: Slice object with the membrane slice information
    # cont_mode: if True (default false) contact points coordinates, instead of vertex ones, are returned
    # graph: if not None (default) it contains the already compute GraphGT, it can save a lot of time
    #         if this function is going to be called several times
    # cont_prop: if cont_mode active, then it may get an array from GraphMCF properties associated to the contact
    #            points. (default STR_FIELD_VALUE_INV)
    # conf_fus: if cont_mode active, mode for contact point fusion criterium, valid: 'max'
    # Return: an array with points coordinates, vertices ids and mask. Raises ValueError if points found
    def get_cloud_mb_slice(self, slice, cont_mode=False, graph_gt=None, cont_prop=STR_FIELD_VALUE_INV,
                           cont_fus='max'):
        # Initialisation
        seg_id = self.get_prop_id(MB_SEG)
        seg_dt = disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=seg_id))
        eu_id = self.get_prop_id(MB_EU_DST)
        eu_dt = disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=eu_id))
        geo_id = self.get_prop_id(MB_GEO_DST)
        geo_dt = disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=geo_id))
        gl_id = self.get_prop_id(MB_GEO_LEN)
        gl_dt = disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=gl_id))
        sin_id = self.get_prop_id(MB_GEO_SIN)
        sin_dt = disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=sin_id))
        if cont_mode:
            cont_id = self.get_prop_id(MB_CONT_COORD)
            cont_dt = disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=cont_id))
            cont_pid = self.get_prop_id(cont_prop)
            cont_pdt = disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=cont_pid))
            if cont_fus != 'max':
                error_msg = 'Input contact point property fusion criterium ' + cont_fus + ' is not valid.'
                raise pexceptions.PySegInputError(expr='get_cloud_mb_slice (MbGraphMCF)', msg=error_msg)

        # Find tail vertices within slice
        h_v_ids = list()
        for v in self.get_vertices_list():
            v_id = v.get_id()
            seg = self.get_prop_entry_fast(seg_id, v_id, 1, seg_dt)[0]
            eu = self.get_prop_entry_fast(eu_id, v_id, 1, eu_dt)[0]
            geo = self.get_prop_entry_fast(geo_id, v_id, 1, geo_dt)[0]
            gl = self.get_prop_entry_fast(gl_id, v_id, 1, gl_dt)[0]
            sin = self.get_prop_entry_fast(sin_id, v_id, 1, sin_dt)[0]
            if slice.test(side=seg, eu_dst=eu, geo_dst=geo, geo_len=gl, sin=sin):
                h_v_ids.append(v_id)

        # Vertices thresholding
        th_l = slice.get_list_th()
        if len(th_l) > 0:
            if graph_gt is None:
                graph_gt = GraphGT(self)
            for th in slice.get_list_th():
                h_v_ids = self.threshold_slice(h_v_ids, th, graph_gt)

        # Creating vertices LUT
        v_ids = list()
        cont = 0
        lut_ver = (-1) * np.ones(shape=self.get_nid(), dtype=np.int)
        for v_id in h_v_ids:
            if lut_ver[v_id] == -1:
                lut_ver[v_id] = cont
                v_ids.append(v_id)
                cont += 1

        # Find edges within slice
        e_ids = list()
        for edge in self.get_edges_list():
            s_id = lut_ver[edge.get_source_id()]
            t_id = lut_ver[edge.get_target_id()]
            if (s_id > -1) and (t_id > -1):
                e_ids.append([s_id, t_id])

        # graph_tool building
        vertices_gt = np.empty(shape=len(v_ids), dtype=object)
        graph = gt.Graph(directed=False)
        for i in range(len(v_ids)):
            vertices_gt[i] = graph.add_vertex()
        for e_id in e_ids:
            graph.add_edge(vertices_gt[e_id[0]], vertices_gt[e_id[1]])

        # Subgraphs visitor initialization
        sgraph_id = graph.new_vertex_property("int")
        visitor = SubGraphVisitor(sgraph_id)

        # Find subgraphs
        coords = np.zeros(shape=(vertices_gt.shape[0], 3), dtype=np.float)
        rids = np.zeros(shape=vertices_gt.shape[0], dtype=np.int)
        if cont_fus == 'max':
            cont_p = (-np.inf) * np.ones(shape=vertices_gt.shape[0], dtype=np.int)
        for i, v in enumerate(vertices_gt):
            if sgraph_id[v] == 0:
                gt.dfs_search(graph, v, visitor)
                visitor.update_sgraphs_id()
            v_id = v_ids[i]
            rids[i] = v_id
            if cont_mode:
                coords[i, :] = self.get_prop_entry_fast(cont_id, v_id, 3, cont_dt)
                val = self.get_prop_entry_fast(cont_pid, v_id, 3, cont_pdt)[0]
                if cont_fus == 'max':
                    if val > cont_p[i]:
                        cont_p[i] = val
            else:
                coords[i, :] = self._GraphMCF__skel.GetPoint(v_id)

        # Clusters filtering
        ids = sgraph_id.get_array()
        max_id = ids.max()
        lut_counts = np.zeros(shape=max_id + 1, dtype=np.int)
        for idy in ids:
            lut_counts[idy] += 1
        hold_coords = list()
        hold_ids = list()
        for idy in range(1, max_id + 1):
            counts = lut_counts[idy]
            if slice.test(cnv=counts):
                c_ids = np.where(ids == idy)[0]
                for c_id in c_ids:
                    hold_coords.append(coords[c_id, :])
                    hold_ids.append(rids[c_id])

        # Computing mask
        mask_out = (self.__mb_seg == slice.get_side())
        if cont_mode:
            hold_mb = self.__mb_dst / self.get_resolution()
            mask_out *= ((hold_mb >= 0) * (hold_mb <= 2))
        else:
            mask_out *= ((self.__mb_dst >= slice.get_eu_dst_low()) * (self.__mb_dst <= slice.get_eu_dst_high()))
        if cont_mode:
            return np.asarray(hold_coords, dtype=np.float), np.asarray(hold_ids, dtype=np.int), mask_out, cont_p
        else:
            return np.asarray(hold_coords, dtype=np.float), np.asarray(hold_ids, dtype=np.int), mask_out

    def get_cloud_mb_slice_pick(self, slice, cont_mode=False, graph_gt=None, cont_prop=STR_FIELD_VALUE_INV, cont_fus='max'):
        """
        Cloud of points in a slice for picking
        :param slice: slice object with the membrane slice information
        :param cont_mode: if 0 then vertices localization are directly provided, if 1 then the contact points associated
         to the each edge between membrane and exterior, and if 2 then the last vertex points before the membrane is provided
        :param graph_gt: if not None (default) it contains the already compute GraphGT, it can save a lot of time
                         if this function is going to be called several times
        :param cont_prop: if cont_mode active, then it may get an array from GraphMCF properties associated to the contact
    #                     points. (default STR_FIELD_VALUE_INV)
        :param cont_fus: if cont_mode active, mode for contact point fusion criterium, valid: 'max'
        :return: an array with points coordinates, vertices ids and mask. Raises ValueError if points found
        """

        # Initialisation
        seg_id = self.get_prop_id(MB_SEG)
        seg_dt = disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=seg_id))
        eu_id = self.get_prop_id(MB_EU_DST)
        eu_dt = disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=eu_id))
        geo_id = self.get_prop_id(MB_GEO_DST)
        geo_dt = disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=geo_id))
        gl_id = self.get_prop_id(MB_GEO_LEN)
        gl_dt = disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=gl_id))
        sin_id = self.get_prop_id(MB_GEO_SIN)
        sin_dt = disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=sin_id))
        if cont_mode > 0:
            cont_id = self.get_prop_id(MB_CONT_COORD)
            cont_dt = disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=cont_id))
            cont_pid = self.get_prop_id(cont_prop)
            cont_pdt = disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=cont_pid))
            if cont_fus != 'max':
                error_msg = 'Input contact point property fusion criterium ' + cont_fus + ' is not valid.'
                raise pexceptions.PySegInputError(expr='get_cloud_mb_slice (MbGraphMCF)', msg=error_msg)

        # Find tail vertices within slice
        h_v_ids = list()
        for v in self.get_vertices_list():
            v_id = v.get_id()
            seg = self.get_prop_entry_fast(seg_id, v_id, 1, seg_dt)[0]
            eu = self.get_prop_entry_fast(eu_id, v_id, 1, eu_dt)[0]
            geo = self.get_prop_entry_fast(geo_id, v_id, 1, geo_dt)[0]
            gl = self.get_prop_entry_fast(gl_id, v_id, 1, gl_dt)[0]
            sin = self.get_prop_entry_fast(sin_id, v_id, 1, sin_dt)[0]
            if slice.test(side=seg, eu_dst=eu, geo_dst=geo, geo_len=gl, sin=sin):
                h_v_ids.append(v_id)

        # Vertices thresholding
        th_l = slice.get_list_th()
        if len(th_l) > 0:
            if graph_gt is None:
                graph_gt = GraphGT(self)
            for th in slice.get_list_th():
                h_v_ids = self.threshold_slice(h_v_ids, th, graph_gt)

        # Creating vertices LUT
        v_ids = list()
        cont = 0
        lut_ver = (-1) * np.ones(shape=self.get_nid(), dtype=np.int)
        for v_id in h_v_ids:
            if lut_ver[v_id] == -1:
                lut_ver[v_id] = cont
                v_ids.append(v_id)
                cont += 1

        if cont_mode > 0:
            # Find edges within slice
            coords, ids, cont_p = list(), list(), list()
            for edge in self.get_edges_list():
                s_id = edge.get_source_id()
                t_id = edge.get_target_id()
                if self.get_prop_entry_fast(seg_id, s_id, 1, seg_dt)[0] != self.get_prop_entry_fast(seg_id, t_id, 1, seg_dt)[0]:
                    if cont_mode == 1:
                        coords.append(self.get_prop_entry_fast(cont_id, s_id, 3, cont_dt))
                    else:
                        coords.append(self._GraphMCF__skel.GetPoint(t_id))
                    val = self.get_prop_entry_fast(cont_pid, t_id, 3, cont_pdt)[0]
                    cont_p.append(val)
                    ids.append(t_id)
        else:
            # Find vertices within slice
            coords, ids, cont_p = list(), list(), list()
            for vertex in self.get_vertices_list():
                v_id = vertex.get_id()
                coords.append(self._GraphMCF__skel.GetPoint(v_id))
                ids.append(v_id)

        # Computing mask
        mask_out = (self.__mb_seg == slice.get_side())
        if cont_mode:
            hold_mb = self.__mb_dst / self.get_resolution()
            mask_out *= ((hold_mb >= 0) * (hold_mb <= 2))
        else:
            mask_out *= ((self.__mb_dst >= slice.get_eu_dst_low()) * (self.__mb_dst <= slice.get_eu_dst_high()))
        if cont_mode:
            return np.asarray(coords, dtype=np.float), np.asarray(ids, dtype=np.int), mask_out, \
                   np.asarray(cont_p, dtype=np.float)
        else:
            return np.asarray(coords, dtype=np.float), np.asarray(ids, dtype=np.int), mask_out

    # Filters a cloud of points from a slice by applying linker restrictions
    def filter_linker_cloud(self, c_s, c_id_s, c_t, c_id_t, linker):

        # Initialization
        graph_gt = GraphGT(self).get_gt()
        prop_vid = graph_gt.vertex_properties[DPSTR_CELL]
        # prop_ew = graph_gt.edge_properties[STR_FIELD_VALUE]
        prop_el = graph_gt.edge_properties[SGT_EDGE_LENGTH]

        # Getting sources and targets lists
        sources = list()
        for v_id in c_id_s:
            v_list = gt.find_vertex(graph_gt, prop_vid, v_id)
            if len(v_list) > 0:
                sources.append(v_list[0])
        targets = list()
        for v_id in c_id_t:
            v_list = gt.find_vertex(graph_gt, prop_vid, v_id)
            if len(v_list) > 0:
                targets.append(v_list[0])

        # Building neighbours array for sources
        mat_geo = gt.shortest_distance(graph_gt, weights=prop_el)
        neighs = np.zeros(shape=len(sources), dtype=np.int)
        for i, s in enumerate(sources):
            for j, t in enumerate(targets):
                geo_dst = mat_geo[s][t]
                x_s, y_s, z_s = self._GraphMCF__skel.GetPoint(prop_vid[s])
                x_t, y_t, z_t = self._GraphMCF__skel.GetPoint(prop_vid[t])
                hold = np.asarray((x_s-x_t, y_s-y_t, z_s-z_t), dtype=np.float)
                eu_dst = math.sqrt((hold*hold).sum()) * self.get_resolution()
                if eu_dst <= 0:
                    sin = 1.
                else:
                    sin = geo_dst / eu_dst
                if linker.test(geo_dst=geo_dst, eu_dst=eu_dst, sin=sin):
                    neighs[i] += 1

        # Test number of neighbours in linked slice and cloud filtering
        hold_cloud = list()
        hold_ids = list()
        for i, s in enumerate(sources):
            if linker.test(neighs=neighs[i]):
                hold_cloud.append(c_s[i])
                hold_ids.append(c_id_s[i])

        return np.asarray(hold_cloud, dtype=np.float), np.asarray(hold_ids, dtype=np.int)

    # Generates a VTK poly data from a cloud of vertices a membrane slice
    # v_ids: vertices id which represent a membrane slice
    # Return: a VTK poly data object (.vtp)
    def slice_to_vtp(self, v_ids):

        # Initialization
        poly = vtk.vtkPolyData()
        poly.SetPoints(self._GraphMCF__skel.GetPoints())
        verts = vtk.vtkCellArray()
        lines = vtk.vtkCellArray()
        struct = vtk.vtkIntArray()
        struct.SetNumberOfComponents(1)
        struct.SetName(MB_VTP_STR)
        eu_dst = vtk.vtkFloatArray()
        eu_dst.SetNumberOfComponents(1)
        eu_dst.SetName(MB_EU_DST)
        geo_dst = vtk.vtkFloatArray()
        geo_dst.SetNumberOfComponents(1)
        geo_dst.SetName(MB_GEO_DST)
        geo_len = vtk.vtkFloatArray()
        geo_len.SetNumberOfComponents(1)
        geo_len.SetName(MB_GEO_LEN)
        sin = vtk.vtkFloatArray()
        sin.SetNumberOfComponents(1)
        sin.SetName(MB_GEO_SIN)
        kt = vtk.vtkFloatArray()
        kt.SetNumberOfComponents(1)
        kt.SetName(MB_GEO_KT)
        tt = vtk.vtkFloatArray()
        tt.SetNumberOfComponents(1)
        tt.SetName(MB_GEO_TT)
        ns = vtk.vtkFloatArray()
        ns.SetNumberOfComponents(1)
        ns.SetName(MB_GEO_NS)
        bs = vtk.vtkFloatArray()
        bs.SetNumberOfComponents(1)
        bs.SetName(MB_GEO_BS)
        apl = vtk.vtkFloatArray()
        apl.SetNumberOfComponents(1)
        apl.SetName(MB_GEO_APL)
        prop_eu = self.get_prop_id(MB_EU_DST)
        prop_geo = self.get_prop_id(MB_GEO_DST)
        prop_gl = self.get_prop_id(MB_GEO_LEN)
        prop_sin = self.get_prop_id(MB_GEO_SIN)
        prop_kt = self.get_prop_id(MB_GEO_KT)
        prop_tt = self.get_prop_id(MB_GEO_TT)
        prop_ns = self.get_prop_id(MB_GEO_NS)
        prop_bs = self.get_prop_id(MB_GEO_BS)
        prop_apl = self.get_prop_id(MB_GEO_APL)

        # VTK Topology
        # Vertices
        lut_ver = np.zeros(shape=self.get_nid(), dtype=np.bool)
        for v_id in v_ids:
            verts.InsertNextCell(1)
            struct.InsertNextTuple((MB_VTP_STR_E,))
            eu_dst.InsertNextTuple(self.get_prop_entry_fast(prop_eu, v_id, 1, np.float))
            geo_dst.InsertNextTuple(self.get_prop_entry_fast(prop_geo, v_id, 1, np.float))
            geo_len.InsertNextTuple(self.get_prop_entry_fast(prop_gl, v_id, 1, np.float))
            sin.InsertNextTuple(self.get_prop_entry_fast(prop_sin, v_id, 1, np.float))
            kt.InsertNextTuple(self.get_prop_entry_fast(prop_kt, v_id, 1, np.float))
            tt.InsertNextTuple(self.get_prop_entry_fast(prop_tt, v_id, 1, np.float))
            ns.InsertNextTuple(self.get_prop_entry_fast(prop_ns, v_id, 1, np.float))
            bs.InsertNextTuple(self.get_prop_entry_fast(prop_bs, v_id, 1, np.float))
            apl.InsertNextTuple(self.get_prop_entry_fast(prop_apl, v_id, 1, np.float))
            verts.InsertCellPoint(v_id)
            lut_ver[v_id] = True
        # Contact points
        for v_id in v_ids:
            fil = self.get_mb_fil(v_id)
            if isinstance(fil, FilamentLDG):
                verts.InsertNextCell(1)
                struct.InsertNextTuple((MB_VTP_STR_C,))
                eu_dst.InsertNextTuple(self.get_prop_entry_fast(prop_eu, v_id, 1, np.float))
                geo_dst.InsertNextTuple(self.get_prop_entry_fast(prop_geo, v_id, 1, np.float))
                geo_len.InsertNextTuple(self.get_prop_entry_fast(prop_gl, v_id, 1, np.float))
                sin.InsertNextTuple(self.get_prop_entry_fast(prop_sin, v_id, 1, np.float))
                kt.InsertNextTuple(self.get_prop_entry_fast(prop_kt, v_id, 1, np.float))
                tt.InsertNextTuple(self.get_prop_entry_fast(prop_tt, v_id, 1, np.float))
                ns.InsertNextTuple(self.get_prop_entry_fast(prop_ns, v_id, 1, np.float))
                bs.InsertNextTuple(self.get_prop_entry_fast(prop_bs, v_id, 1, np.float))
                apl.InsertNextTuple(self.get_prop_entry_fast(prop_apl, v_id, 1, np.float))
                verts.InsertCellPoint(self.get_mb_fil(v_id).get_point_ids()[-1])
        # Filament paths
        for v_id in v_ids:
            fil = self.get_mb_fil(v_id)
            if isinstance(fil, FilamentLDG):
                p_ids = fil.get_point_ids()
                n_points = len(p_ids)
                lines.InsertNextCell(n_points)
                struct.InsertNextTuple((MB_VTP_STR_F,))
                eu_dst.InsertNextTuple(self.get_prop_entry_fast(prop_eu, v_id, 1, np.float))
                geo_dst.InsertNextTuple(self.get_prop_entry_fast(prop_geo, v_id, 1, np.float))
                geo_len.InsertNextTuple(self.get_prop_entry_fast(prop_gl, v_id, 1, np.float))
                sin.InsertNextTuple(self.get_prop_entry_fast(prop_sin, v_id, 1, np.float))
                kt.InsertNextTuple(self.get_prop_entry_fast(prop_kt, v_id, 1, np.float))
                tt.InsertNextTuple(self.get_prop_entry_fast(prop_tt, v_id, 1, np.float))
                ns.InsertNextTuple(self.get_prop_entry_fast(prop_ns, v_id, 1, np.float))
                bs.InsertNextTuple(self.get_prop_entry_fast(prop_bs, v_id, 1, np.float))
                apl.InsertNextTuple(self.get_prop_entry_fast(prop_apl, v_id, 1, np.float))
                for i in range(n_points):
                    lines.InsertCellPoint(p_ids[i])
        # Edges
        for e in self.get_edges_list():
            s_id = e.get_source_id()
            t_id = e.get_target_id()
            if lut_ver[s_id] and lut_ver[t_id]:
                lines.InsertNextCell(2)
                struct.InsertNextTuple((MB_VTP_STR_E,))
                eu_dst.InsertNextTuple((-1,))
                geo_dst.InsertNextTuple((-1,))
                geo_len.InsertNextTuple((-1,))
                sin.InsertNextTuple((-1,))
                kt.InsertNextTuple((-1,))
                tt.InsertNextTuple((-1,))
                ns.InsertNextTuple((-1,))
                bs.InsertNextTuple((-1,))
                apl.InsertNextTuple((-1,))
                lines.InsertCellPoint(s_id)
                lines.InsertCellPoint(t_id)

        # Building the vtp object
        poly.SetVerts(verts)
        poly.SetLines(lines)
        poly.GetCellData().AddArray(struct)
        poly.GetCellData().AddArray(eu_dst)
        poly.GetCellData().AddArray(geo_dst)
        poly.GetCellData().AddArray(geo_len)
        poly.GetCellData().AddArray(sin)
        poly.GetCellData().AddArray(kt)
        poly.GetCellData().AddArray(tt)
        poly.GetCellData().AddArray(ns)
        poly.GetCellData().AddArray(bs)
        poly.GetCellData().AddArray(apl)

        return poly

    # Generates a VTK poly data from the internal structure of the membrane
    # v_ids: vertices id which represent a membrane slice, if none the membrane mask is used
    # av_mode: if True (default False) the properties of arcs will be the properties values of
    # respective vertices
    # edges: if True (default False) only edge arcs are printed
    # Return: a VTK poly data object (.vtp)
    def mb_to_vtp(self, v_ids, av_mode=True, edges=False):

        if v_ids is None:
            return self.get_vtp_in_msk(self.__mb_seg == MB_SEG, av_mode, edges)
        else:
            return self.get_vtp_ids(v_ids, av_mode, edges)

    # Generates a binary segmented tomogram (True-fg) from a list of vertices (or a membrane slice)
    # v_ids: membrane slice cloud of vertices id
    # th_den: number of sigmas above (+) or below vertex geometry density mean for thresholding,
    #         if None no threshold is applied
    # slc: if True (default False) the list of vertices is considered a membrane slice
    def print_slice(self, v_ids, th_den=None, slc=False):

        # Initialization
        img = np.zeros(shape=self._GraphMCF__density.shape, dtype=np.bool)

        if slc:
            for v_id in v_ids:
                fil = self.get_mb_fil(v_id)
                if isinstance(fil, FilamentLDG):
                    for f_id in fil.get_vertex_ids():
                        v = self.get_vertex(f_id)
                        v.get_geometry().print_in_numpy(img, True, th_den)
        else:
            for v_id in v_ids:
                v = self.get_vertex(v_id)
                v.get_geometry().print_in_numpy(img, True, th_den)

        return img

    # Set an edge property values into a fixed value so as to protect them from edge simplification
    # (see graph_density_simp)
    # prop_key: property to set
    # mb_dst: distance to membrane for protecting (in nm)
    # val: setting value
    # Return: a new property key for using in graph_density_simp for edge simplification
    def protect_mb_edges_prop(self, prop_key, mb_dst, val):

        # Getting region to protect
        tomod = disperse_io.seg_dist_trans(self.__mb_seg == MB_LBL) * self.get_resolution()
        bin_mask = tomod < mb_dst

        # Get old property and creation of the new
        key_id = self.get_prop_id(prop_key)
        data_type = self.get_prop_type(key_id=key_id)
        dt = disperse_io.TypesConverter().gt_to_numpy(data_type)
        ncomp = self.get_prop_ncomp(key_id=key_id)
        new_prop_key = prop_key + '_ptc'
        new_key_id = self.add_prop(new_prop_key, data_type, ncomp)

        # Set values of the new property
        for v in self.get_vertices_list():
            v_id = v.get_id()
            t = list(self.get_prop_entry_fast(key_id, v_id, ncomp, dt))
            try:
                x, y, z = self.get_vertex_coords(v)
                if bin_mask[int(round(x)), int(round(y)), int(round(z))]:
                    for i in range(len(t)):
                        t[i] = val
            except IndexError:
                pass
            self.set_prop_entry_fast(new_key_id, tuple(t), v_id, ncomp)
        for e in self.get_edges_list():
            e_id = e.get_id()
            t = list(self.get_prop_entry_fast(key_id, e_id, ncomp, dt))
            try:
                x, y, z = self.get_edge_coords(e)
                if bin_mask[int(round(x)), int(round(y)), int(round(z))]:
                    for i in range(len(t)):
                        t[i] = val
            except IndexError:
                pass
            self.set_prop_entry_fast(new_key_id, tuple(t), e_id, ncomp)

        return new_prop_key

    # Threshold trans-membrane vertices simplification
    # prop: vertex property key
    # den: desired vertex density (vertex/nm^3)
    # mode: if 'high' (default) then vertices with highest property values are preserved,
    #       otherwise those with the lowest
    def trans_mb_simp(self, prop_key, den, mode='high'):

        # Building membrane mask
        mb_mask = self.__mb_seg == MB_LBL

        # Compute valid region volume (nm^3)
        res3 = float(self.get_resolution() * self.get_resolution() * self.get_resolution())
        vol = float(mb_mask.sum()) * res3
        if vol == 0:
            error_msg = 'Membrane region has null volume.'
            raise pexceptions.PySegInputWarning(expr='thres_trans_mb (GraphMCF)', msg=error_msg)
        prop_id = self.get_prop_id(prop_key)
        p_type = disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=prop_id))
        n_comp = self.get_prop_ncomp(key_id=prop_id)
        if n_comp > 1:
            error_msg = 'Property ' + prop_key + ' has more than one components.'
            raise pexceptions.PySegInputError(expr='thres_trans_mb (GraphMCF)', msg=error_msg)

        # Compute target number of vertices with membrane
        mb_verts = np.asarray(self.get_mb_vertices_list())
        n_verts = float(len(mb_verts))
        n_tverts = int(round(den * vol))
        if n_verts < n_tverts:
            error_msg = 'Desired density cannot be achieved because current is ' + str(n_verts/vol)
            raise pexceptions.PySegInputWarning(expr='thres_trans_mb (GraphMCF)', msg=error_msg)

        # List of ordered vertices
        arr_prop = np.zeros(shape=mb_verts.shape[0], dtype=p_type)
        for i, v in enumerate(mb_verts):
            arr_prop[i] = self.__props_info.get_prop_entry_fast(prop_id, v.get_id(), n_comp, p_type)[0]
        ids = np.argsort(arr_prop)
        mb_verts = mb_verts[ids]
        if mode == 'high':
            mb_verts = mb_verts[::-1]

        # Removing vertices
        for i in range(n_tverts, mb_verts.shape[0]):
            self.remove_vertex(mb_verts[i])

    # Threshold the vertices of a slice
    # v_ids: list with the coordinates
    # v_ids: list with vertices ids for the slice
    # thres: ThresSlice object with the configuration for thresholding
    # Result: a sub-set of the input coords and v_ids
    def slice_vertex_threshold(self, coords, v_ids, thres):

        # Initialization
        prop_key = thres.get_prop_key()
        prop_id = self.get_prop_id(prop_key)
        p_type = disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=prop_id))
        n_comp = self.get_prop_ncomp(key_id=prop_id)
        if n_comp > 1:
            error_msg = 'Property ' + prop_key + ' has more than one components.'
            raise pexceptions.PySegInputError(expr='slice_vertex_threshold (GraphMCF)', msg=error_msg)

        hold_ids = list()
        hold_coords = list()
        for (coord, v_id) in zip(coords, v_ids):
            val = self.get_prop_entry_fast(prop_id, v_id, n_comp, p_type)[0]
            if thres.test(val):
                hold_coords.append(coord)
                hold_ids.append(v_id)

        return np.asarray(hold_coords, dtype=np.float), np.asarray(hold_ids, dtype=np.int)


    # Compute angle between a vector (3 components property) and the normal to a membrane (only vertices)
    # prop_3d: property key for the vector
    # v_ids: list with the vertices ids, if None (default) all vertices are considered
    # Result: a property with angles in degrees stored as prop_3d+'_ang', the normals as prop_3d+'_norm'
    # Returns: the property keys for normals, and angles
    def angle_vector_norms(self, prop_3d, v_ids=None):

        # Input parsing and initialization
        prop_id = self.get_prop_id(prop_3d)
        if prop_id is None:
            error_msg = 'Input property ' + prop_3d + ' does not exist.'
            raise pexceptions.PySegInputError(expr='angle_vector_norms (MbGraphMCF)', msg=error_msg)
        n_comp = self.get_prop_ncomp(key_id=prop_id)
        if n_comp != 3:
            error_msg = 'Only input properties with 3 component are valid, current ' + str(n_comp) + '.'
            raise pexceptions.PySegInputError(expr='angle_vector_norms (MbGraphMCF)', msg=error_msg)
        dtype = disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=prop_id))
        prop_id_n = self.get_prop_id(MB_CONT_COORD)
        dtype_n = disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=prop_id_n))

        # Get vertices list
        if v_ids is None:
            v_ids = list()
            for v in self._GraphMCF__vertices:
                if v is not None:
                    v_ids.append(v.get_id())
        norms = np.zeros(shape=(len(v_ids), 3), dtype=np.float32)
        angles = np.zeros(shape=(len(v_ids), 3), dtype=np.float32)

        # Loop for computing the normals and angles
        for i, v_id in enumerate(v_ids):
            t = self.get_prop_entry_fast(prop_id_n, v_id, 3, dtype_n)
            norm = np.asarray(self._GraphMCF__skel.GetPoint(v_id), dtype=np.float32) - np.asarray(t, dtype=np.float32)
            vect = np.asarray(self.get_prop_entry_fast(prop_id, v_id, 3, dtype), dtype=dtype)
            angles[i] = math.degrees(angle_2vec_3D(norm, vect))
            norms[i, :] = norm

        # Inserting properties and values
        norm_key, ang_key = prop_3d+'_norm', prop_3d+'_ang'
        gtype_f = disperse_io.TypesConverter().numpy_to_gt(np.float32)
        norm_id, ang_id = self.add_prop(norm_key, gtype_f, 3), self.add_prop(ang_key, gtype_f, 1)
        for (v_id, ang, norm) in zip(v_ids, angles, norms):
            self.set_prop_entry_fast(ang_id, ang, v_id, 1)
            self.set_prop_entry_fast(norm_id, norm, v_id, 3)

        return norm_key, ang_key

    # Applies a property threshold to a slice
    # v_ids: slice vertices ids for being thresholded
    # th: Threshold object for a GraphMCF property
    # graph_gt: if not None (default) it contains the already compute GraphGT, it can save a lot of time
    #         if this function is going to be called several times
    # Returns: a list with the thresholded vertices ids
    def threshold_slice(self, v_ids, th, graph_gt=None):

        # Getting the GraphGT
        if graph_gt is None:
            graph_gt = GraphGT(self).get_gt()
        prop_id = graph_gt.vertex_properties[DPSTR_CELL]
        prop_th = graph_gt.vertex_properties[th.get_prop_key()].get_array()

        # Get LUT for slice vertices
        vertices_gt = np.empty(shape=self.get_nid(), dtype=object)
        prop_arr = np.zeros(shape=len(v_ids), dtype=prop_th.dtype)
        for v in graph_gt.vertices():
            vertices_gt[prop_id[v]] = v
        cont = 0
        for v_id in v_ids:
            if vertices_gt[v_id] is not None:
                prop_arr[cont] = prop_th[vertices_gt[v_id]]
                cont += 1

        # Apply the threshold
        if th.get_mode() == XML_MBT_MPERI:
            per_th_l = np.percentile(prop_arr, th.get_value_low())
            per_th_h = np.percentile(prop_arr, th.get_value_high())
            mask = (prop_arr >= per_th_l) & (prop_arr <= per_th_h)
        elif th.get_mode() == XML_MBT_MPERO:
            per_th_l = np.percentile(prop_arr, th.get_value_low())
            per_th_h = np.percentile(prop_arr, th.get_value_high())
            mask = (prop_arr < per_th_l) | (prop_arr > per_th_h)
        elif th.get_mode() == XML_MBT_MIN:
            mask = (prop_arr >= th.get_value_low()) & (prop_arr <= th.get_value_high())
        else:
            mask = (prop_arr < th.get_value_low()) | (prop_arr > th.get_value_high())

        return list(np.asarray(v_ids)[mask])


    # Generates a filament network where filaments are specified by a set of sources and targets
    # s_ids: list with sources ids
    # t_ids: list with targets ids
    # rg_dst: range for valid geodesic distances in nm
    # rg_sin: range for sinuosities
    # graph_gt: by default None, otherwise pre-computed GraphGT
    # key_l: property for measuring the length (default STR_VERT_DST)
    def gen_fil_network(self, s_ids, t_ids, rg_dst, rg_sin, graph_gt=None, key_l=STR_VERT_DST):

        # Initialization
        if (not hasattr(rg_dst, '__len__')) or (len(rg_dst) != 2):
            error_msg = 'The input range of distances must be a 2-tuple.'
            raise pexceptions.PySegInputError(expr='gen_fil_network (MbGraphMCF)', msg=error_msg)
        if rg_dst[0] < 0:
            error_msg = 'Lowest distance must greater or equal to zero, current ' + str(rg_dst[0]) + '.'
            raise pexceptions.PySegInputError(expr='gen_fil_network (MbGraphMCF)', msg=error_msg)
        if (not hasattr(rg_sin, '__len__')) or (len(rg_sin) != 2):
            error_msg = 'The input range of sinuosity must be a 2-tuple.'
            raise pexceptions.PySegInputError(expr='gen_fil_network (MbGraphMCF)', msg=error_msg)
        if rg_dst[0] < 1:
            error_msg = 'Lowest sinuosity must greater or equal to zero, current ' + str(rg_sin[0]) + '.'
            raise pexceptions.PySegInputError(expr='gen_fil_network (MbGraphMCF)', msg=error_msg)
        if (not hasattr(s_ids, '__len__')) or (not hasattr(t_ids, '__len__')):
            error_msg = 'Input source and target Ids must be iterables.'
            raise pexceptions.PySegInputError(expr='gen_fil_network (MbGraphMCF)', msg=error_msg)
        fils = list()

        # Compute GraphGT
        if graph_gt is None:
            graph = GraphGT(self)
            graph_gt = graph.get_gt()
        else:
            if not isinstance(graph_gt, gt.Graph):
                error_msg = 'Input graph_gt must be an instance of gt.Graph.'
                raise pexceptions.PySegInputError(expr='gen_fil_network (MbGraphMCF)', msg=error_msg)
        prop_id, prop_id_e = graph_gt.vp[DPSTR_CELL], graph_gt.ep[DPSTR_CELL]
        try:
            prop_l = graph_gt.ep[key_l]
        except KeyError:
            error_msg = 'Requires the previous computations of ' + STR_VERT_DST + ' property'
            raise pexceptions.PySegInputError(expr='gen_fil_network (MbGraphMCF)', msg=error_msg)

        # Vertices lut
        nv = graph_gt.num_vertices()
        nids = self.get_nid()
        s_lut, t_lut = np.zeros(shape=nids, dtype=np.bool), np.zeros(shape=nids, dtype=np.bool)
        for i, s_id in enumerate(s_ids):
            s_lut[s_id] = True
        for t_id in t_ids:
            t_lut[t_id] = True
        len_s, len_t = len(s_ids), len(t_ids)
        sg_ids = np.zeros(shape=len_s, dtype=object)
        sg_lut, tg_lut = np.zeros(shape=nv, dtype=np.bool), np.zeros(shape=nv, dtype=np.bool)
        count_s = 0
        for v in graph_gt.vertices():
            v_id = prop_id[v]
            if s_lut[v_id]:
                sg_ids[count_s] = v
                sg_lut[int(v)] = True
                count_s += 1
            if t_lut[v_id]:
                tg_lut[int(v)] = True

        # Loop for sources
        # count = 0
        for s in sg_ids:

            # print str(count) + ' of ' + str(len_s)
            # count += 1

            # Compute geodesic distances to source
            s_mcf = self.get_vertex(prop_id[s])
            pt_s = np.asarray(self.get_vertex_coords(s_mcf))
            dst_map, pred_map = gt.shortest_distance(graph_gt, source=s, weights=prop_l, max_dist=rg_dst[1],
                                                     pred_map=True)

            # Find potential targets
            h_tg_lut = np.copy(tg_lut)
            dst_map_arr = dst_map.get_array()
            n_ids = np.where(tg_lut & (dst_map_arr>=rg_dst[0]) & (dst_map_arr<=rg_dst[1]))[0]
            sort_ids = np.argsort(dst_map_arr[n_ids])

            # Loop for found targets
            for n_id in n_ids[sort_ids]:

                # Check already visited
                t = graph_gt.vertex(n_id)
                if not h_tg_lut[int(t)]:
                    continue

                # Find the paths
                v_path, e_path = gt.shortest_path(graph_gt, source=s, target=t, weights=prop_l, pred_map=pred_map)

                # Build the filament
                lv = len(v_path)
                if lv > 1:
                    length = 0
                    vl_ids, el_ids = list(), list()
                    vh_id = prop_id[v_path[0]]
                    vl_ids.append(vh_id)
                    # el_ids.append(vh_id)
                    for i in range(1,lv):
                        vh, eh = v_path[i], e_path[i-1]
                        vh_id = prop_id[vh]
                        # Rewind if source
                        if sg_lut[int(vh)]:
                            length = 0
                            vl_ids, el_ids = list(), list()
                            vl_ids.append(vh_id)
                            # el_ids.append(vh_id)
                        # Check potential target
                        elif h_tg_lut[int(vh)]:
                            vl_ids.append(vh_id)
                            el_ids.append(prop_id_e[eh])
                            # el_ids.append(vh_id)
                            length += prop_l[eh]
                            # Check length
                            if (length>=rg_dst[0]) and (length<=rg_dst[1]):
                                # Check sinuosity
                                t_mcf = self.get_vertex(vh_id)
                                pt_t = np.asarray(self.get_vertex_coords(t_mcf))
                                eu_dst = pt_s - pt_t
                                eu_dst = math.sqrt((eu_dst*eu_dst).sum()) * self.get_resolution()
                                sin = dst_map_arr[n_id] / eu_dst
                                if (sin>=rg_sin[0]) and (sin<=rg_sin[1]):
                                    # Filament found
                                    fils.append(FilamentUDG(vl_ids[:], el_ids[:], self))
                            # No more filaments in this path
                            for j in range(i, lv):
                                h_tg_lut[int(v_path[j])] = False
                            break
                        else:
                            vl_ids.append(vh_id)
                            el_ids.append(prop_id_e[eh])
                            # el_ids.append(vh_id)
                            length += prop_l[eh]

        # Build the network
        return NetFilamentsSyn(fils)

##################################################################################################
# Generic extension for dealing with GraphMCF of synapses, analogously it can be used for any junction of two membranes
#
#
class SynGraphMCF(GraphMCF):

    #### Constructor Area

    # During building
    # skel: DisPerSe skeleton
    # manifolds: DisPerSe manifolds
    # density: image density map
    # mb_seg: tomogram with synapse segmentation (1-pst_mb, 2-pre_mb, 3-psd, 4-az, 5-cleft and otherwise-bg)
    def __init__(self, skel, manifolds, density, syn_seg):
        super(SynGraphMCF, self).__init__(skel, manifolds, density)
        self.build_from_skel(basic_props=False)
        self.__syn_seg = syn_seg
        self.__mb_fils_pst = np.zeros(shape=self.get_nid(), dtype=object)
        self.__mb_fils_pre = np.zeros(shape=self.get_nid(), dtype=object)
        self.add_scalar_field_nn(self.__syn_seg, SYN_SEG)
        self.__mb_dst_pst = disperse_io.seg_dist_trans(self.__syn_seg == SYN_PST_LBL) * self.get_resolution()
        self.__mb_dst_pre = disperse_io.seg_dist_trans(self.__syn_seg == SYN_PRE_LBL) * self.get_resolution()

    #### Set/Get functionality

    #### External functionality

    # Returns vertices within the membrane segmentation
    # mb_lbl: if 1 then pst else if 2 pre
    def get_mb_vertices_list(self, mb_lbl):
        verts_mb = list()
        prop_seg_id = self.get_prop_id(SYN_SEG)
        for v in self.get_vertices_list():
            if self.get_prop_entry_fast(prop_seg_id, v.get_id(), 1, np.int)[0] == mb_lbl:
                verts_mb.append(v)
        return verts_mb

    # Euclidean (through euclidean space) shortest distance to two membranes of the synapse
    # Returns: result stored as property MB_PST_EU_DST and MB_PRE_EU_DST
    def compute_mb_eu_dst(self):
        self.add_scalar_field(self.__mb_dst_pst, MB_PST_EU_DST)
        self.add_scalar_field(self.__mb_dst_pre, MB_PRE_EU_DST)

    # Geodesic (through graph and euclidean) distance to shortest contact point,
    # it also computes differential geometry property for filaments
    # Returns: result stored as property MB_PST|PRE_GEO_DST, MB_PST|PRE_GEO_LEN and MB_PST|PRE_GEO_SIM
    def compute_mb_geo(self, update=True):

        # Initialization
        if update or (self.get_prop_id(SYN_SEG) is None):
            self.add_scalar_field_nn(self.__syn_seg, SYN_SEG)
        if update or (self.get_prop_id(SGT_EDGE_LENGTH) is None):
            self.compute_edges_length(SGT_EDGE_LENGTH, 1, 1, 1, False)
        if update or (self.get_prop_id(STR_EDGE_FNESS) is None):
            self.compute_edge_filamentness()
        key_pst_dst_id = self.get_prop_id(MB_PST_GEO_DST)
        if key_pst_dst_id is None:
            key_pst_dst_id = self.add_prop(MB_PST_GEO_DST, 'float', 1)
        key_pre_dst_id = self.get_prop_id(MB_PRE_GEO_DST)
        if key_pre_dst_id is None:
            key_pre_dst_id = self.add_prop(MB_PRE_GEO_DST, 'float', 1)
        key_pst_len_id = self.get_prop_id(MB_PST_GEO_LEN)
        if key_pst_len_id is None:
            key_pst_len_id = self.add_prop(MB_PST_GEO_LEN, 'float', 1)
        key_pre_len_id = self.get_prop_id(MB_PRE_GEO_LEN)
        if key_pre_len_id is None:
            key_pre_len_id = self.add_prop(MB_PRE_GEO_LEN, 'float', 1)
        key_pst_sin_id = self.get_prop_id(MB_PST_GEO_SIN)
        if key_pst_sin_id is None:
            key_pst_sin_id = self.add_prop(MB_PST_GEO_SIN, 'float', 1)
        key_pre_sin_id = self.get_prop_id(MB_PRE_GEO_SIN)
        if key_pre_sin_id is None:
            key_pre_sin_id = self.add_prop(MB_PRE_GEO_SIN, 'float', 1)
        key_pst_cont_id = self.get_prop_id(MB_PST_CONT_COORD)
        if key_pst_cont_id is None:
            key_pst_cont_id = self.add_prop(MB_PST_CONT_COORD, 'float', 3)
        key_pre_cont_id = self.get_prop_id(MB_PRE_CONT_COORD)
        if key_pre_cont_id is None:
            key_pre_cont_id = self.add_prop(MB_PRE_CONT_COORD, 'float', 3)

        # Getting the graph GraphGT
        graph_gt = GraphGT(self).get_gt()
        prop_elen = graph_gt.edge_properties[SGT_EDGE_LENGTH]
        prop_seg = graph_gt.vertex_properties[SYN_SEG]
        prop_eid = graph_gt.edge_properties[DPSTR_CELL]
        prop_vid = graph_gt.vertex_properties[DPSTR_CELL]

        # Creating LUTs
        lut_psd, lut_clf_pst = list(), list()
        lut_az, lut_clf_pre = list(), list()
        for v in graph_gt.vertices():
            i_v = int(v)
            seg_lbl = prop_seg[v]
            if seg_lbl == SYN_PSD_LBL:
                lut_psd.append(i_v)
            if seg_lbl == SYN_AZ_LBL:
                lut_az.append(i_v)
            elif seg_lbl == SYN_CLF_LBL:
                lut_clf_pst.append(i_v)
                lut_clf_pre.append(i_v)
            elif seg_lbl == SYN_PST_LBL:
                lut_psd.append(i_v)
                lut_clf_pst.append(i_v)
            elif seg_lbl == SYN_PRE_LBL:
                lut_az.append(i_v)
                lut_clf_pre.append(i_v)
        lut_psd, lut_clf_pst = np.asarray(lut_psd, dtype=np.int), np.asarray(lut_clf_pst, dtype=np.int)
        lut_az, lut_clf_pre = np.asarray(lut_az, dtype=np.int), np.asarray(lut_clf_pre, dtype=np.int)

        # Loop for all vertices
        for v in graph_gt.vertices():

            # Look for vertices out the membranes
            seg_lbl = prop_seg[v]
            if (seg_lbl != SYN_PST_LBL) and (seg_lbl != SYN_PRE_LBL):
                # Getting lut for vertices on the other side of the membrane
                if seg_lbl == SYN_PSD_LBL:
                    lut_oth_pst, lut_oth_pre = lut_clf_pst, None
                elif seg_lbl == SYN_AZ_LBL:
                    lut_oth_pst, lut_oth_pre = None, lut_clf_pre
                elif seg_lbl == SYN_CLF_LBL:
                    lut_oth_pst, lut_oth_pre = lut_psd, lut_az
                else:
                    continue

                # Checking the sides
                t_l, side_l = list(), list()
                dists = gt.shortest_distance(graph_gt, source=v, weights=prop_elen).get_array()
                if lut_oth_pst is not None:
                    dists_pot_pst = dists[lut_oth_pst]
                    if len(dists_pot_pst) > 0:
                        t_l.append(graph_gt.vertex(lut_oth_pst[np.argmin(dists_pot_pst)]))
                        side_l.append(SYN_PST_LBL)
                if lut_oth_pre is not None:
                    dists_pot_pre = dists[lut_oth_pre]
                    if len(dists_pot_pre) > 0:
                        t_l.append(graph_gt.vertex(lut_oth_pre[np.argmin(dists_pot_pre)]))
                        side_l.append(SYN_PRE_LBL)

                # Finding the shortest geodesic path to other vertices which crosses the membrane
                for (t, sd) in zip(t_l, side_l):
                    v_path, e_path = gt.shortest_path(graph_gt, v, t, weights=prop_elen)
                    if len(e_path) > 0:
                        contact = None
                        # Measuring the path (on euclidean space) until membrane contact point
                        fil_v_ids = list()
                        for h_v in v_path[0:-1]:
                            fil_v_ids.append(prop_vid[h_v])
                        fil_p_ids = list()
                        # Initialize the filament starting point
                        fil_p_ids.append(prop_vid[v])
                        path_len = .0
                        for e in e_path[0:-1]:
                            path_len += prop_elen[e]
                            # Add in a ordered ways path points to the filament
                            p_ids = self.get_edge_ids(self.get_edge(prop_eid[e]))
                            if p_ids[-1] == fil_p_ids[-1]:
                                p_ids = p_ids[::-1]
                            fil_p_ids += p_ids[1::]
                        edge = self.get_edge(prop_eid[e_path[-1]])
                        e_ids = self.get_edge_ids(edge)
                        e_coords = np.asarray(self.get_edge_arcs_coords(edge), dtype=np.float)
                        e_coords_int = np.asarray(e_coords.round(), dtype=np.int)
                        try:
                            xi, yi, zi = e_coords_int[0, :]
                            # Reverse for ending with vertex through membrane
                            if  self.__syn_seg[xi, yi, zi] != seg_lbl:
                                e_coords = e_coords[::-1]
                                e_coords_int = e_coords_int[::-1]
                                e_ids = e_ids[::-1]
                            contact = e_coords[0, :]
                            fil_p_ids.append(e_ids[0])
                            # Loop for length increasing
                            xh, yh, zh = e_coords[0, :]
                            for i in range(1, e_coords.shape[0]):
                                x, y, z = e_coords_int[i, :]
                                if self.__syn_seg[x, y, z] == seg_lbl:
                                    x, y, z = e_coords[i, :]
                                    hold_len = np.asarray((x-xh, y-yh, z-zh), dtype=np.float)
                                    path_len += (math.sqrt((hold_len*hold_len).sum()) * self.get_resolution())
                                    xh, yh, zh = x, y, z
                                    contact = e_coords[i, :]
                                    fil_p_ids.append(e_ids[i])
                                else:
                                    break
                        except IndexError:
                            pass

                        # Compute metrics only if a contact is found
                        if contact is not None:

                            # Add filament
                            v_id = prop_vid[v]
                            if len(fil_p_ids) > 2:
                                try:
                                    fil = FilamentLDG(fil_v_ids, fil_p_ids, self)
                                except IndexError:
                                    print 'WARNING: filament with less than 3 points!'
                                if sd == SYN_PST_LBL:
                                    self.__mb_fils_pst[v_id] = fil
                                    self.set_prop_entry_fast(key_pst_dst_id, (fil.get_dst(),), v_id, 1)
                                    self.set_prop_entry_fast(key_pst_len_id, (fil.get_length(),), v_id, 1)
                                    self.set_prop_entry_fast(key_pst_sin_id, (fil.get_sinuosity(),), v_id, 1)
                                    self.set_prop_entry_fast(key_pst_cont_id, (contact[0], contact[1], contact[2]),
                                                             v_id, 3)
                                elif sd == SYN_PRE_LBL:
                                    self.__mb_fils_pre[v_id] = fil
                                    self.set_prop_entry_fast(key_pre_dst_id, (fil.get_dst(),), v_id, 1)
                                    self.set_prop_entry_fast(key_pre_len_id, (fil.get_length(),), v_id, 1)
                                    self.set_prop_entry_fast(key_pre_sin_id, (fil.get_sinuosity(),), v_id, 1)
                                    self.set_prop_entry_fast(key_pre_cont_id, (contact[0], contact[1], contact[2]),
                                                             v_id, 3)
                            else:
                                print 'WARNING: filament with less than 3 points!'

    # Returns: returns a 2-tuple (pst, pre) with the the two possible filaments (if a filament does not exist is None)
    # if mb_id == SYN_PST|PRE_LBL only one filament is returned
    def get_mb_fil(self, v_id, mb_id=None):
        if mb_id is None:
            return self.__mb_fils_pst[v_id], self.__mb_dst_pre[v_id]
        elif mb_id == SYN_PST_LBL:
            return self.__mb_fils_pst[v_id]
        elif mb_id == SYN_PRE_LBL:
            return self.__mb_fils_pre[v_id]
        else:
            error_msg = 'Non valid membrane label ' + str(mb_id)
            raise pexceptions.PySegInputError(expr='get_mb_fil (SynGraphMCF)', msg=error_msg)

    # Cloud of points in a slice
    # slices: Slice object with the membrane slice information
    # cont_mode: if True (default False) contact points coordinates, instead of vertex ones, are returned
    # graph: if not None (default) it contains the already compute GraphGT, it can save a lot of time
    #         if this function is going to be called several times
    # cont_prop: if cont_mode active, then it may get an array from GraphMCF properties associated to the contact
    #            points. (default STR_FIELD_VALUE_INV)
    # conf_fus: if cont_mode active, mode for contact point fusion criterium, valid: 'max'
    # Return: an array with points coordinates, vertices ids and mask, and contact point properties array in contact
    #         mode
    def get_cloud_mb_slice(self, slice, cont_mode=False, graph_gt=None, cont_prop=STR_FIELD_VALUE_INV, cont_fus='max'):

        # Initialisation
        seg_id = self.get_prop_id(SYN_SEG)
        seg_dt = disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=seg_id))
        mb_id = slice.get_mb()
        if mb_id == SYN_PST_LBL:
            eu_id = self.get_prop_id(MB_PST_EU_DST)
            eu_dt = disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=eu_id))
            geo_id = self.get_prop_id(MB_PST_GEO_DST)
            geo_dt = disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=geo_id))
            gl_id = self.get_prop_id(MB_PST_GEO_LEN)
            gl_dt = disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=gl_id))
            sin_id = self.get_prop_id(MB_PST_GEO_SIN)
            sin_dt = disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=sin_id))
            gol_id = self.get_prop_id(MB_PRE_GEO_LEN)
            gol_dt = disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=gol_id))
            if cont_mode:
                cont_id = self.get_prop_id(MB_PST_CONT_COORD)
                cont_dt = disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=cont_id))
                cont_pid = self.get_prop_id(cont_prop)
                cont_pdt = disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=cont_pid))
                if cont_fus != 'max':
                    error_msg = 'Input contact point property fusion criterium ' + cont_fus + ' is not valid.'
                    raise pexceptions.PySegInputError(expr='get_cloud_mb_slice (SynGraphMCF)', msg=error_msg)
        elif mb_id == SYN_PRE_LBL:
            eu_id = self.get_prop_id(MB_PRE_EU_DST)
            eu_dt = disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=eu_id))
            geo_id = self.get_prop_id(MB_PRE_GEO_DST)
            geo_dt = disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=geo_id))
            gl_id = self.get_prop_id(MB_PRE_GEO_LEN)
            gl_dt = disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=gl_id))
            sin_id = self.get_prop_id(MB_PRE_GEO_SIN)
            sin_dt = disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=sin_id))
            gol_id = self.get_prop_id(MB_PST_GEO_LEN)
            gol_dt = disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=gol_id))
            if cont_mode:
                cont_id = self.get_prop_id(MB_PRE_CONT_COORD)
                cont_dt = disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=cont_id))
                cont_pid = self.get_prop_id(cont_prop)
                cont_pdt = disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=cont_pid))
                if cont_fus != 'max':
                    error_msg = 'Input contact point property fusion criterium ' + cont_fus + ' is not valid.'
                    raise pexceptions.PySegInputError(expr='get_cloud_mb_slice (SynGraphMCF)', msg=error_msg)
        else:
            error_msg = 'Non valid membrane label ' + str(mb_id)
            raise pexceptions.PySegInputError(expr='get_cloud_mb_slice (SynGraphMCF)', msg=error_msg)

        # Find tail vertices within slice
        h_v_ids = list()
        for v in self.get_vertices_list():
            v_id = v.get_id()
            seg = self.get_prop_entry_fast(seg_id, v_id, 1, seg_dt)[0]
            eu = self.get_prop_entry_fast(eu_id, v_id, 1, eu_dt)[0]
            geo = self.get_prop_entry_fast(geo_id, v_id, 1, geo_dt)[0]
            gl = self.get_prop_entry_fast(gl_id, v_id, 1, gl_dt)[0]
            sin = self.get_prop_entry_fast(sin_id, v_id, 1, sin_dt)[0]
            gol = self.get_prop_entry_fast(gol_id, v_id, 1, gol_dt)[0]
            # if (gol > 0) and (gl > 0) and (gol < gl):
            #     continue
            if slice.test(seg=seg, eu_dst=eu, geo_dst=geo, geo_len=gl, sin=sin):
                    h_v_ids.append(v_id)

        # Vertices thresholding
        th_l = slice.get_list_th()
        if len(th_l) > 0:
            if graph_gt is None:
                graph_gt = GraphGT(self)
            for th in slice.get_list_th():
                h_v_ids = self.threshold_slice(h_v_ids, th, graph_gt)

        # Creating vertices LUT
        v_ids = list()
        cont = 0
        lut_ver = (-1) * np.ones(shape=self.get_nid(), dtype=np.int)
        for v_id in h_v_ids:
            if lut_ver[v_id] == -1:
                lut_ver[v_id] = cont
                v_ids.append(v_id)
                cont += 1

        # Find edges within slice
        e_ids = list()
        for edge in self.get_edges_list():
            s_id = lut_ver[edge.get_source_id()]
            t_id = lut_ver[edge.get_target_id()]
            if (s_id > -1) and (t_id > -1):
                e_ids.append([s_id, t_id])

        # graph_tool building
        vertices_gt = np.empty(shape=len(v_ids), dtype=object)
        graph = gt.Graph(directed=False)
        for i in range(len(v_ids)):
            vertices_gt[i] = graph.add_vertex()
        for e_id in e_ids:
            graph.add_edge(vertices_gt[e_id[0]], vertices_gt[e_id[1]])

        # Subgraphs visitor initialization
        sgraph_id = graph.new_vertex_property("int")
        visitor = SubGraphVisitor(sgraph_id)

        # Find subgraphs
        coords = np.zeros(shape=(vertices_gt.shape[0], 3), dtype=np.float)
        rids = np.zeros(shape=vertices_gt.shape[0], dtype=np.int)
        if cont_fus == 'max':
            cont_p = (-np.inf) * np.ones(shape=vertices_gt.shape[0], dtype=np.int)
        for i, v in enumerate(vertices_gt):
            if sgraph_id[v] == 0:
                gt.dfs_search(graph, v, visitor)
                visitor.update_sgraphs_id()
            v_id = v_ids[i]
            rids[i] = v_id
            if cont_mode:
                coords[i, :] = self.get_prop_entry_fast(cont_id, v_id, 3, cont_dt)
                val = self.get_prop_entry_fast(cont_pid, v_id, 3, cont_pdt)[0]
                if cont_fus == 'max':
                    if val > cont_p[i]:
                        cont_p[i] = val
            else:
                coords[i, :] = self._GraphMCF__skel.GetPoint(v_id)

        # Clusters filtering
        ids = sgraph_id.get_array()
        max_id = ids.max()
        lut_counts = np.zeros(shape=max_id+1, dtype=np.int)
        for idy in ids:
            lut_counts[idy] += 1
        hold_coords = list()
        hold_ids = list()
        for idy in range(1, max_id+1):
            counts = lut_counts[idy]
            if slice.test(cnv=counts):
                c_ids = np.where(ids == idy)[0]
                for c_id in c_ids:
                    hold_coords.append(coords[c_id, :])
                    hold_ids.append(rids[c_id])

        # Computing mask
        mask_out = (self.__syn_seg == slice.get_seg())
        if mb_id == SYN_PST_LBL:
            if cont_mode:
                hold_mb = self.__mb_dst_pst / self.get_resolution()
                mask_out *= ((hold_mb >= 0) * (hold_mb <= 2))
            else:
                mask_out *= ((self.__mb_dst_pst >= slice.get_eu_dst_low()) *
                             (self.__mb_dst_pst <= slice.get_eu_dst_high()))
        else:
            if cont_mode:
                hold_mb = self.__mb_dst_pre / self.get_resolution()
                mask_out *= ((hold_mb >= 0) * (hold_mb <= 2))
            else:
                mask_out *= ((self.__mb_dst_pre >= slice.get_eu_dst_low()) *
                             (self.__mb_dst_pre <= slice.get_eu_dst_high()))
                # disperse_io.save_numpy(mask_out, '/fs/pool/pool-ruben/antonio/nuc_mito/hold_1.mrc')
        if cont_mode:
            return np.asarray(hold_coords, dtype=np.float), np.asarray(hold_ids, dtype=np.int), mask_out, cont_p
        else:
            return np.asarray(hold_coords, dtype=np.float), np.asarray(hold_ids, dtype=np.int), mask_out

    # Generates a VTK poly data from a cloud of vertices of a synapse slice
    # v_ids: vertices id which represent a membrane slice
    # mb_id: identifies the membrane which corresponds with the slice
    # Return: a VTK poly data object (.vtp)
    def slice_to_vtp(self, v_ids, mb_id):

        # Initialization
        poly = vtk.vtkPolyData()
        poly.SetPoints(self._GraphMCF__skel.GetPoints())
        verts = vtk.vtkCellArray()
        lines = vtk.vtkCellArray()
        struct = vtk.vtkIntArray()
        struct.SetNumberOfComponents(1)
        struct.SetName(MB_VTP_STR)
        eu_dst = vtk.vtkFloatArray()
        eu_dst.SetNumberOfComponents(1)
        eu_dst.SetName(MB_EU_DST)
        geo_dst = vtk.vtkFloatArray()
        geo_dst.SetNumberOfComponents(1)
        geo_dst.SetName(MB_GEO_DST)
        geo_len = vtk.vtkFloatArray()
        geo_len.SetNumberOfComponents(1)
        geo_len.SetName(MB_GEO_LEN)
        sin = vtk.vtkFloatArray()
        sin.SetNumberOfComponents(1)
        sin.SetName(MB_GEO_SIN)
        kt = vtk.vtkFloatArray()
        kt.SetNumberOfComponents(1)
        kt.SetName(MB_GEO_KT)
        tt = vtk.vtkFloatArray()
        tt.SetNumberOfComponents(1)
        tt.SetName(MB_GEO_TT)
        ns = vtk.vtkFloatArray()
        ns.SetNumberOfComponents(1)
        ns.SetName(MB_GEO_NS)
        bs = vtk.vtkFloatArray()
        bs.SetNumberOfComponents(1)
        bs.SetName(MB_GEO_BS)
        apl = vtk.vtkFloatArray()
        apl.SetNumberOfComponents(1)
        apl.SetName(MB_GEO_APL)
        if mb_id == SYN_PST_LBL:
            prop_eu = self.get_prop_id(MB_PST_EU_DST)
            prop_geo = self.get_prop_id(MB_PST_GEO_DST)
            prop_gl = self.get_prop_id(MB_PST_GEO_LEN)
            prop_sin = self.get_prop_id(MB_PST_GEO_SIN)
        elif mb_id == SYN_PRE_LBL:
            prop_eu = self.get_prop_id(MB_PRE_EU_DST)
            prop_geo = self.get_prop_id(MB_PRE_GEO_DST)
            prop_gl = self.get_prop_id(MB_PRE_GEO_LEN)
            prop_sin = self.get_prop_id(MB_PRE_GEO_SIN)

        # VTK Topology
        # Vertices
        lut_ver = np.zeros(shape=self.get_nid(), dtype=np.bool)
        for v_id in v_ids:
            verts.InsertNextCell(1)
            struct.InsertNextTuple((MB_VTP_STR_E,))
            eu_dst.InsertNextTuple(self.get_prop_entry_fast(prop_eu, v_id, 1, np.float))
            geo_dst.InsertNextTuple(self.get_prop_entry_fast(prop_geo, v_id, 1, np.float))
            geo_len.InsertNextTuple(self.get_prop_entry_fast(prop_gl, v_id, 1, np.float))
            sin.InsertNextTuple(self.get_prop_entry_fast(prop_sin, v_id, 1, np.float))
            kt.InsertNextTuple((-1,))
            tt.InsertNextTuple((-1,))
            ns.InsertNextTuple((-1,))
            bs.InsertNextTuple((-1,))
            apl.InsertNextTuple((-1,))
            verts.InsertCellPoint(v_id)
            lut_ver[v_id] = True
        # Contact points
        for v_id in v_ids:
            fil = self.get_mb_fil(v_id, mb_id)
            if isinstance(fil, FilamentLDG):
                verts.InsertNextCell(1)
                struct.InsertNextTuple((MB_VTP_STR_C,))
                eu_dst.InsertNextTuple(self.get_prop_entry_fast(prop_eu, v_id, 1, np.float))
                geo_dst.InsertNextTuple(self.get_prop_entry_fast(prop_geo, v_id, 1, np.float))
                geo_len.InsertNextTuple(self.get_prop_entry_fast(prop_gl, v_id, 1, np.float))
                sin.InsertNextTuple(self.get_prop_entry_fast(prop_sin, v_id, 1, np.float))
                kt.InsertNextTuple((fil.get_total_k(),))
                tt.InsertNextTuple((fil.get_total_t(),))
                ns.InsertNextTuple((fil.get_total_ns(),))
                bs.InsertNextTuple((fil.get_total_bs(),))
                apl.InsertNextTuple((fil.get_apex_length(),))
                verts.InsertCellPoint(fil.get_point_ids()[-1])
        # Filament paths
        for v_id in v_ids:
            fil = self.get_mb_fil(v_id, mb_id)
            if isinstance(fil, FilamentLDG):
                p_ids = fil.get_point_ids()
                n_points = len(p_ids)
                lines.InsertNextCell(n_points)
                struct.InsertNextTuple((MB_VTP_STR_F,))
                eu_dst.InsertNextTuple(self.get_prop_entry_fast(prop_eu, v_id, 1, np.float))
                geo_dst.InsertNextTuple(self.get_prop_entry_fast(prop_geo, v_id, 1, np.float))
                geo_len.InsertNextTuple(self.get_prop_entry_fast(prop_gl, v_id, 1, np.float))
                sin.InsertNextTuple(self.get_prop_entry_fast(prop_sin, v_id, 1, np.float))
                kt.InsertNextTuple((fil.get_total_k(),))
                tt.InsertNextTuple((fil.get_total_t(),))
                ns.InsertNextTuple((fil.get_total_ns(),))
                bs.InsertNextTuple((fil.get_total_bs(),))
                apl.InsertNextTuple((fil.get_apex_length(),))
                for i in range(n_points):
                    lines.InsertCellPoint(p_ids[i])
        # Edges
        for e in self.get_edges_list():
            s_id = e.get_source_id()
            t_id = e.get_target_id()
            if lut_ver[s_id] and lut_ver[t_id]:
                lines.InsertNextCell(2)
                struct.InsertNextTuple((MB_VTP_STR_E,))
                eu_dst.InsertNextTuple((-1,))
                geo_dst.InsertNextTuple((-1,))
                geo_len.InsertNextTuple((-1,))
                sin.InsertNextTuple((-1,))
                kt.InsertNextTuple((-1,))
                tt.InsertNextTuple((-1,))
                ns.InsertNextTuple((-1,))
                bs.InsertNextTuple((-1,))
                apl.InsertNextTuple((-1,))
                lines.InsertCellPoint(s_id)
                lines.InsertCellPoint(t_id)

        # Building the vtp object
        poly.SetVerts(verts)
        poly.SetLines(lines)
        poly.GetCellData().AddArray(struct)
        poly.GetCellData().AddArray(eu_dst)
        poly.GetCellData().AddArray(geo_dst)
        poly.GetCellData().AddArray(geo_len)
        poly.GetCellData().AddArray(sin)
        poly.GetCellData().AddArray(kt)
        poly.GetCellData().AddArray(tt)
        poly.GetCellData().AddArray(ns)
        poly.GetCellData().AddArray(bs)
        poly.GetCellData().AddArray(apl)

        return poly

    # Applies RWR on a list of sources
    # s_ids: list with the sources id
    # key: prop key were the result will be stored, if it does not exist it is created
    # c: restart probability [0, 1] (default 0.1) (the lower the more global analysis)
    # key_w_v: prop key for vertices weighting (default None)
    # key_w_e: prop key for edges weighting (default None)
    # inv: if True (default False) edge weighting property is inverted
    # graph: if not None (default) it contains the already compute GraphGT, it can save a lot of time
    #         if this function is going to be called several times
    def rwr_sources(self, s_ids, key, c=.0, key_w_v=None, key_w_e=None, inv=False, graph=None):

        # Getting the GraphGT
        if graph is None:
            graph = GraphGT(self)
        graph_gt = graph.get_gt()

        # Get sources
        vertices_gt = np.empty(shape=self.get_nid(), dtype=object)
        prop_id = graph_gt.vertex_properties[DPSTR_CELL]
        for v in graph_gt.vertices():
            vertices_gt[prop_id[v]] = v
        sources = list()
        for v_i in s_ids:
            v = vertices_gt[v_i]
            if v is not None:
                sources.append(v)

        # RWR
        graph.multi_rwr(sources, key, c, key_w_e, key_w_v, mode=2, inv=inv)
        graph.add_prop_to_GraphMCF(self, key, up_index=True)

    # Applies geodesic neighbours filter from list of ids with the sources
    # s_ids: list with the sources id
    # key: prop key were the result will be stored, if it does not exist it is created
    # scale: neighborhood radius in nm
    # sig: sigma (controls neighborhood sharpness) in nm
    # n_sig: limits neighbourhood shape (default 3)
    # key_w_v: prop key for vertices weighting (default None)
    # graph: if not None (default) it contains the already compute GraphGT, it can save a lot of time
    #         if this function is going to be called several times
    def gnf_sources(self, s_ids, key, scale, sig, n_sig, key_w_v=None, graph=None):

        # Getting the GraphGT
        if graph is None:
            graph = GraphGT(self)
        graph_gt = graph.get_gt()

        # Get sources
        prop_s = graph_gt.new_vertex_property('int', vals=0)
        vertices_gt = np.empty(shape=self.get_nid(), dtype=object)
        prop_id = graph_gt.vertex_properties[DPSTR_CELL]
        for v in graph_gt.vertices():
            vertices_gt[prop_id[v]] = v
        for v_i in s_ids:
            v = vertices_gt[v_i]
            if v is not None:
                prop_s[v] = 1

        # Filtering
        graph.geodesic_neighbors_filter(key, scale, sig, prop_key_v=key_w_v, prop_key_s=prop_s, n_sig=n_sig)
        graph.add_prop_to_GraphMCF(self, key, up_index=True)

    # Applies a property threshold to a slice
    # v_ids: slice vertices ids for being thresholded
    # th: Threshold object for a GraphMCF property
    # graph_gt: if not None (default) it contains the already compute GraphGT, it can save a lot of time
    #         if this function is going to be called several times
    # Returns: a list with the thresholded vertices ids
    def threshold_slice(self, v_ids, th, graph_gt=None):

        # Getting the GraphGT
        if graph_gt is None:
            graph_gt = GraphGT(self).get_gt()
        prop_id = graph_gt.vertex_properties[DPSTR_CELL]
        prop_th = graph_gt.vertex_properties[th.get_prop_key()].get_array()

        # Get LUT for slice vertices
        vertices_gt = np.empty(shape=self.get_nid(), dtype=object)
        prop_arr = np.zeros(shape=len(v_ids), dtype=prop_th.dtype)
        for v in graph_gt.vertices():
            vertices_gt[prop_id[v]] = v
        cont = 0
        for v_id in v_ids:
            if vertices_gt[v_id] is not None:
                prop_arr[cont] = prop_th[vertices_gt[v_id]]
                cont += 1

        # Apply the threshold
        if th.get_mode() == XML_MBT_MPERI:
            per_th_l = np.percentile(prop_arr, th.get_value_low())
            per_th_h = np.percentile(prop_arr, th.get_value_high())
            mask = (prop_arr >= per_th_l) & (prop_arr <= per_th_h)
        elif th.get_mode() == XML_MBT_MPERO:
            per_th_l = np.percentile(prop_arr, th.get_value_low())
            per_th_h = np.percentile(prop_arr, th.get_value_high())
            mask = (prop_arr < per_th_l) | (prop_arr > per_th_h)
        elif th.get_mode() == XML_MBT_MIN:
            mask = (prop_arr >= th.get_value_low()) & (prop_arr <= th.get_value_high())
        else:
            mask = (prop_arr < th.get_value_low()) | (prop_arr > th.get_value_high())

        return list(np.asarray(v_ids)[mask])

    # Compute angle between a vector (3 components property) and the normal to a membrane (only vertices)
    # prop_3d: property key for the vector
    # mb: membrane for computing the normals, valid: SYN_PST_LBL or SYN_PRE_LBL
    # v_ids: list with the vertices ids, if None (default) all vertices are considered
    # Result: a property with angles in degrees stored as prop_3d+'_'+SYN_XXX_LBL+'_ang', the normals as SYN_XXX_LBL+'_norm'
    # Returns: the property keys for normals, and angles
    def angle_vector_norms(self, prop_3d, mb, v_ids=None):

        # Input parsing and initialization
        prop_id = self.get_prop_id(prop_3d)
        if prop_id is None:
            error_msg = 'Input property ' + prop_3d + ' does not exist.'
            raise pexceptions.PySegInputError(expr='angle_vector_norms (SynGraphMCF)', msg=error_msg)
        n_comp = self.get_prop_ncomp(key_id=prop_id)
        if n_comp != 3:
            error_msg = 'Only input properties with 3 component are valid, current ' + str(n_comp) + '.'
            raise pexceptions.PySegInputError(expr='angle_vector_norms (SynGraphMCF)', msg=error_msg)
        dtype = disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=prop_id))
        if mb == SYN_PST_LBL:
            mb_key = MB_PST_CONT_COORD
        elif mb == SYN_PRE_LBL:
            mb_key = MB_PRE_CONT_COORD
        else:
            error_msg = 'Non valid membrane label for SynGraphMCF ' + mb
            raise pexceptions.PySegInputError(expr='angle_vector_norms (SynGraphMCF)', msg=error_msg)
        prop_id_n = self.get_prop_id(mb_key)
        dtype_n = disperse_io.TypesConverter().gt_to_numpy(self.get_prop_type(key_id=prop_id_n))

        # Get vertices list
        if v_ids is None:
            v_ids = list()
            for v in self._GraphMCF__vertices:
                if v is not None:
                    v_ids.append(v.get_id())
        norms = np.zeros(shape=(len(v_ids), 3), dtype=np.float32)
        angles = np.zeros(shape=(len(v_ids), 3), dtype=np.float32)

        # Loop for computing the normals and angles
        for i, v_id in enumerate(v_ids):
            t = self.get_prop_entry_fast(prop_id_n, v_id, 3, dtype_n)
            norm = np.asarray(self._GraphMCF__skel.GetPoint(v_id), dtype=np.float32) - np.asarray(t, dtype=np.float32)
            vect = np.asarray(self.get_prop_entry_fast(prop_id, v_id, 3, dtype), dtype=dtype)
            angles[i] = math.degrees(angle_2vec_3D(norm, vect))
            norms[i, :] = norm

        # Inserting properties and values
        norm_key, ang_key = prop_3d+mb_key+'_norm', prop_3d+'_'+mb_key+'_ang'
        gtype_f = disperse_io.TypesConverter().numpy_to_gt(np.float32)
        norm_id, ang_id = self.add_prop(norm_key, gtype_f, 3), self.add_prop(ang_key, gtype_f, 1)
        for (v_id, ang, norm) in zip(v_ids, angles, norms):
            self.set_prop_entry_fast(ang_id, ang, v_id, 1)
            self.set_prop_entry_fast(norm_id, norm, v_id, 3)

        return norm_key, ang_key

    # Generates a filament network where filaments are specified by a set of sources and targets
    # s_ids: list with sources ids
    # t_ids: list with targets ids
    # rg_dst: range for valid geodesic distances in nm
    # rg_sin: range for sinuosities
    # graph_gt: by default None, otherwise pre-computed GraphGT
    # key_l: property for measuring the length (default STR_VERT_DST)
    def gen_fil_network(self, s_ids, t_ids, rg_dst, rg_sin, graph_gt=None, key_l=STR_VERT_DST):

        # Initialization
        if (not hasattr(rg_dst, '__len__')) or (len(rg_dst) != 2):
            error_msg = 'The input range of distances must be a 2-tuple.'
            raise pexceptions.PySegInputError(expr='gen_fil_network (SynGraphMCF)', msg=error_msg)
        if rg_dst[0] < 0:
            error_msg = 'Lowest distance must greater or equal to zero, current ' + str(rg_dst[0]) + '.'
            raise pexceptions.PySegInputError(expr='gen_fil_network (SynGraphMCF)', msg=error_msg)
        if (not hasattr(rg_sin, '__len__')) or (len(rg_sin) != 2):
            error_msg = 'The input range of sinuosity must be a 2-tuple.'
            raise pexceptions.PySegInputError(expr='gen_fil_network (SynGraphMCF)', msg=error_msg)
        if rg_dst[0] < 1:
            error_msg = 'Lowest sinuosity must greater or equal to zero, current ' + str(rg_sin[0]) + '.'
            raise pexceptions.PySegInputError(expr='gen_fil_network (SynGraphMCF)', msg=error_msg)
        if (not hasattr(s_ids, '__len__')) or (not hasattr(t_ids, '__len__')):
            error_msg = 'Input source and target Ids must be iterables.'
            raise pexceptions.PySegInputError(expr='gen_fil_network (SynGraphMCF)', msg=error_msg)
        fils = list()

        # Compute GraphGT
        if graph_gt is None:
            graph = GraphGT(self)
            graph_gt = graph.get_gt()
        else:
            if not isinstance(graph_gt, gt.Graph):
                error_msg = 'Input graph_gt must be an instance of gt.Graph.'
                raise pexceptions.PySegInputError(expr='gen_fil_network (SynGraphMCF)', msg=error_msg)
        prop_id, prop_id_e = graph_gt.vp[DPSTR_CELL], graph_gt.ep[DPSTR_CELL]
        try:
            prop_l = graph_gt.ep[key_l]
        except KeyError:
            error_msg = 'Requires the previous computations of ' + STR_VERT_DST + ' property'
            raise pexceptions.PySegInputError(expr='gen_fil_network (SynGraphMCF)', msg=error_msg)

        # Vertices lut
        nv = graph_gt.num_vertices()
        nids = self.get_nid()
        s_lut, t_lut = np.zeros(shape=nids, dtype=np.bool), np.zeros(shape=nids, dtype=np.bool)
        for i, s_id in enumerate(s_ids):
            s_lut[s_id] = True
        for t_id in t_ids:
            t_lut[t_id] = True
        len_s, len_t = len(s_ids), len(t_ids)
        sg_ids = np.zeros(shape=len_s, dtype=object)
        sg_lut, tg_lut = np.zeros(shape=nv, dtype=np.bool), np.zeros(shape=nv, dtype=np.bool)
        count_s = 0
        for v in graph_gt.vertices():
            v_id = prop_id[v]
            if s_lut[v_id]:
                sg_ids[count_s] = v
                sg_lut[int(v)] = True
                count_s += 1
            if t_lut[v_id]:
                tg_lut[int(v)] = True

        # Loop for sources
        # count = 0
        for s in sg_ids:

            # print str(count) + ' of ' + str(len_s)
            # count += 1

            # Compute geodesic distances to source
            s_mcf = self.get_vertex(prop_id[s])
            pt_s = np.asarray(self.get_vertex_coords(s_mcf))
            dst_map, pred_map = gt.shortest_distance(graph_gt, source=s, weights=prop_l, max_dist=rg_dst[1],
                                                     pred_map=True)

            # Find potential targets
            h_tg_lut = np.copy(tg_lut)
            dst_map_arr = dst_map.get_array()
            n_ids = np.where(tg_lut & (dst_map_arr>=rg_dst[0]) & (dst_map_arr<=rg_dst[1]))[0]
            sort_ids = np.argsort(dst_map_arr[n_ids])

            # Loop for found targets
            for n_id in n_ids[sort_ids]:

                # Check already visited
                t = graph_gt.vertex(n_id)
                if not h_tg_lut[int(t)]:
                    continue

                # Find the paths
                v_path, e_path = gt.shortest_path(graph_gt, source=s, target=t, weights=prop_l, pred_map=pred_map)

                # Build the filament
                lv = len(v_path)
                if lv > 1:
                    length = 0
                    vl_ids, el_ids = list(), list()
                    vh_id = prop_id[v_path[0]]
                    vl_ids.append(vh_id)
                    # el_ids.append(vh_id)
                    for i in range(1,lv):
                        vh, eh = v_path[i], e_path[i-1]
                        vh_id = prop_id[vh]
                        # Rewind if source
                        if sg_lut[int(vh)]:
                            length = 0
                            vl_ids, el_ids = list(), list()
                            vl_ids.append(vh_id)
                            # el_ids.append(vh_id)
                        # Check potential target
                        elif h_tg_lut[int(vh)]:
                            vl_ids.append(vh_id)
                            el_ids.append(prop_id_e[eh])
                            # el_ids.append(vh_id)
                            length += prop_l[eh]
                            # Check length
                            if (length>=rg_dst[0]) and (length<=rg_dst[1]):
                                # Check sinuosity
                                t_mcf = self.get_vertex(vh_id)
                                pt_t = np.asarray(self.get_vertex_coords(t_mcf))
                                eu_dst = pt_s - pt_t
                                eu_dst = math.sqrt((eu_dst*eu_dst).sum()) * self.get_resolution()
                                sin = dst_map_arr[n_id] / eu_dst
                                if (sin>=rg_sin[0]) and (sin<=rg_sin[1]):
                                    # Filament found
                                    fils.append(FilamentUDG(vl_ids[:], el_ids[:], self))
                            # No more filaments in this path
                            for j in range(i, lv):
                                h_tg_lut[int(v_path[j])] = False
                            break
                        else:
                            vl_ids.append(vh_id)
                            el_ids.append(prop_id_e[eh])
                            # el_ids.append(vh_id)
                            length += prop_l[eh]

        # Build the network
        return NetFilamentsSyn(fils)
