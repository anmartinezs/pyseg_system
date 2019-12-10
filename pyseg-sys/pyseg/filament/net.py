"""
Set of specific utilities for retrieving a filaments from a tomogram

# Author: Antonio Martinez-Sanchez (Max Planck Institute for Biochemistry)
# Date: 15.09.15
"""

__author__ = 'Antonio Martinez-Sanchez'

import pyseg as ps
import csv
import vtk
import graph_tool.all as gt
from pyseg.globals.variables import *
from pyseg.filament.variables import *
from pyseg.filament import FilamentU
from pyseg.filament.globals import FilVisitor

##########################################################################################
# Class for modelling the SpaceCurves contained by a list
##########################################################################################

class SetSpaceCurve(object):

    # curves: list of input Filaments
    def __init__(self, curves):
        self.__curves = curves

    #### Set/Get methods area

    # Add one, or more curves in a list or in another SetSpaceCurve
    def add(self, item):
        if isinstance(item, list):
            self.__curves += item
        elif isinstance(item, SetSpaceCurve):
            self.__curves += item.get_curves_list()
        elif hasattr(item, '__len__'):
            for curve in item:
                self.__curves.append(curve)
        else:
            self.__curves.append(item)

    def get_curves_list(self):
        return self.__curves

    # Convert the curves set into a VTK PolyData object
    def get_vtp(self):

        # Initialization
        appender = vtk.vtkAppendPolyData()

        # Loop for curves
        for curve in self.__curves:
            # Append to holder poly data
            hold_vtp = curve.get_vtp(add_geom=True)
            # print 'hold = ' + str(hold_vtp.GetNumberOfPoints())
            appender.AddInputData(hold_vtp)
            appender.Update()
            # print 'total = ' + str(appender.GetOutput().GetNumberOfPoints())

        return appender.GetOutput()

##########################################################################################
# Class for modelling the filaments contained by a tomogram
##########################################################################################

class NetFilaments(object):

    # graph: parent GraphMCF
    # min_len: minimum length for the Filaments (default 0)
    # max_len: maximum length for the Filaments (default MAX_FLOAT)
    # key_edge: edge metric for nodes similarity in the graph (default STR_EDGE_FNESS)
    def __init__(self, graph, min_len=0, max_len=MAX_FLOAT, key_edge=STR_EDGE_FNESS):
        self.__graph = graph
        self.__graph_gt = None
        self.__min_len = min_len
        self.__max_len = max_len
        self.__key_edge = key_edge
        self.__fils = list()
        self.__build()

    #### Set/Get methods area

    def get_graph(self):
        return self.__graph

    # Convert the filament network into a VTK PolyData object
    # mode: if 1 (default) filament properties are computed in vertex mode, otherwise path mode
    # paths: if 1 (default) vertices are represented though edge paths, otherwise through
    #        vertices locations
    def get_vtp(self, mode=1, paths=1):

        # Initialization
        point_id = 0
        cell_id = 0
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        cell_data = vtk.vtkIntArray()
        cell_data.SetNumberOfComponents(1)
        cell_data.SetName(STR_CELL)
        len_data = vtk.vtkFloatArray()
        len_data.SetNumberOfComponents(1)
        len_data.SetName(STR_FIL_LEN)
        ct_data = vtk.vtkFloatArray()
        ct_data.SetNumberOfComponents(1)
        ct_data.SetName(STR_FIL_CT)
        sin_data = vtk.vtkFloatArray()
        sin_data.SetNumberOfComponents(1)
        sin_data.SetName(STR_FIL_SIN)
        fness_data = vtk.vtkFloatArray()
        fness_data.SetNumberOfComponents(1)
        fness_data.SetName(STR_FIL_FNESS)
        smo_data = vtk.vtkFloatArray()
        smo_data.SetNumberOfComponents(1)
        smo_data.SetName(STR_FIL_SMO)
        mc_data = vtk.vtkFloatArray()
        mc_data.SetNumberOfComponents(1)
        mc_data.SetName(STR_FIL_MC)

        # Write lines
        for i, f in enumerate(self.__fils):

            # Getting children if demanded
            if paths == 1:
                coords = f.get_path_coords()
            else:
                coords = f.get_vertex_coords()
            lines.InsertNextCell(coords.shape[0])
            for c in coords:
                points.InsertPoint(point_id, c[0], c[1], c[2])
                lines.InsertCellPoint(point_id)
                point_id += 1
            cell_id += 1
            cell_data.InsertNextTuple((cell_id,))
            len_data.InsertNextTuple((f.get_length(mode=mode),))
            ct_data.InsertNextTuple((f.get_total_curvature(mode=mode),))
            sin_data.InsertNextTuple((f.get_sinuosity(mode=mode),))
            fness_data.InsertNextTuple((f.get_filamentness(),))
            smo_data.InsertNextTuple((f.get_smoothness(mode=mode),))
            mc_data.InsertNextTuple((f.get_max_curvature(mode=mode),))

        # Poly building
        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        poly.SetLines(lines)
        poly.GetCellData().AddArray(cell_data)
        poly.GetCellData().AddArray(len_data)
        poly.GetCellData().AddArray(ct_data)
        poly.GetCellData().AddArray(sin_data)
        poly.GetCellData().AddArray(fness_data)
        poly.GetCellData().AddArray(smo_data)
        poly.GetCellData().AddArray(mc_data)

        return poly

    #### External functionality area

    # Filaments properties are accumulated (a kind of centrality) in the internal graph
    def store_fils_in_graph(self):

        # Compute filament properties
        len_a = np.zeros(shape=len(self.__fils), dtype=np.float)
        ct_a = np.zeros(shape=len(self.__fils), dtype=np.float)
        sin_a = np.zeros(shape=len(self.__fils), dtype=np.float)
        fness_a = np.zeros(shape=len(self.__fils), dtype=np.float)
        smo_a = np.zeros(shape=len(self.__fils), dtype=np.float)
        mc_a = np.zeros(shape=len(self.__fils), dtype=np.float)
        for i, f in enumerate(self.__fils):
            len_a[i] = f.get_length()
            ct_a[i] = f.get_total_curvature()
            sin_a[i] = f.get_sinuosity()
            fness_a[i] = f.get_filamentness()
            smo_a[i] = f.get_smoothness()
            mc_a[i] = f.get_sinuosity()

        # Normalize parameters
        s_len = len_a.sum()
        if s_len > 0:
            len_a /= s_len
        else:
            len_a = (-1) * np.ones(shape=len(self.__fils), dtype=np.float)
        s_ct = ct_a.sum()
        if s_len > 0:
            ct_a = 1 - (ct_a/s_ct)
        else:
            ct_a = (-1) * np.ones(shape=len(self.__fils), dtype=np.float)
        s_sin = sin_a.sum()
        if s_sin > 0:
            sin_a = 1 - (sin_a/s_sin)
        else:
            sin_a = (-1) * np.ones(shape=len(self.__fils), dtype=np.float)
        s_fness = fness_a.sum()
        if s_fness > 0:
            fness_a = 1 - (fness_a/s_fness)
        else:
            fness_a = (-1) * np.ones(shape=len(self.__fils), dtype=np.float)
        s_smo = smo_a.sum()
        if s_smo > 0:
            smo_a /= s_smo
        else:
            smo_a = (-1) * np.ones(shape=len(self.__fils), dtype=np.float)
        s_mc = mc_a.sum()
        if s_mc > 0:
            mc_a = 1 - (mc_a / s_mc)
        else:
            mc_a = (-1) * np.ones(shape=len(self.__fils), dtype=np.float)

        # Insert the new properties
        len_id = self.__graph.add_prop(STR_FIL_LEN, 'float', 1, def_val=0.)
        ct_id = self.__graph.add_prop(STR_FIL_CT, 'float', 1, def_val=0.)
        sin_id = self.__graph.add_prop(STR_FIL_SIN, 'float', 1, def_val=0.)
        fness_id = self.__graph.add_prop(STR_FIL_FNESS, 'float', 1, def_val=0.)
        smo_id = self.__graph.add_prop(STR_FIL_SMO, 'float', 1, def_val=0.)
        mc_id = self.__graph.add_prop(STR_FIL_MC, 'float', 1, def_val=0.)

        # Accumulate property into vertices and edges
        n_lut = np.zeros(shape=self.__graph.get_nid(), dtype=np.float)
        for i, fil in enumerate(self.__fils):
            for v in fil.get_vertices():
                v_id = v.get_id()
                n_lut[v_id] += 1.
                hold_t = self.__graph.get_prop_entry_fast(len_id, v_id, 1, np.float)
                hold = hold_t[0] + len_a[i]
                self.__graph.set_prop_entry_fast(len_id, (hold,), v_id, 1)
                hold_t = self.__graph.get_prop_entry_fast(ct_id, v_id, 1, np.float)
                hold = hold_t[0] + ct_a[i]
                self.__graph.set_prop_entry_fast(ct_id, (hold,), v_id, 1)
                hold_t = self.__graph.get_prop_entry_fast(sin_id, v_id, 1, np.float)
                hold = hold_t[0] + sin_a[i]
                self.__graph.set_prop_entry_fast(sin_id, (hold,), v_id, 1)
                hold_t = self.__graph.get_prop_entry_fast(fness_id, v_id, 1, np.float)
                hold = hold_t[0] + fness_a[i]
                self.__graph.set_prop_entry_fast(fness_id, (hold,), v_id, 1)
                hold_t = self.__graph.get_prop_entry_fast(smo_id, v_id, 1, np.float)
                hold = hold_t[0] + smo_a[i]
                self.__graph.set_prop_entry_fast(smo_id, (hold,), v_id, 1)
                hold_t = self.__graph.get_prop_entry_fast(mc_id, v_id, 1, np.float)
                hold = hold_t[0] + mc_a[i]
                self.__graph.set_prop_entry_fast(mc_id, (hold,), v_id, 1)
            for e in fil.get_edges():
                e_id = e.get_id()
                n_lut[e_id] += 1.
                hold_t = self.__graph.get_prop_entry_fast(len_id, e_id, 1, np.float)
                hold = hold_t[0] + len_a[i]
                self.__graph.set_prop_entry_fast(len_id, (hold,), e_id, 1)
                hold_t = self.__graph.get_prop_entry_fast(ct_id, e_id, 1, np.float)
                hold = hold_t[0] + ct_a[i]
                self.__graph.set_prop_entry_fast(ct_id, (hold,), e_id, 1)
                hold_t = self.__graph.get_prop_entry_fast(sin_id, e_id, 1, np.float)
                hold = hold_t[0] + sin_a[i]
                self.__graph.set_prop_entry_fast(sin_id, (hold,), e_id, 1)
                hold_t = self.__graph.get_prop_entry_fast(fness_id, e_id, 1, np.float)
                hold = hold_t[0] + fness_a[i]
                self.__graph.set_prop_entry_fast(fness_id, (hold,), e_id, 1)
                hold_t = self.__graph.get_prop_entry_fast(smo_id, e_id, 1, np.float)
                hold = hold_t[0] + smo_a[i]
                self.__graph.set_prop_entry_fast(smo_id, (hold,), e_id, 1)
                hold_t = self.__graph.get_prop_entry_fast(mc_id, e_id, 1, np.float)
                hold = hold_t[0] + mc_a[i]
                self.__graph.set_prop_entry_fast(mc_id, (hold,), e_id, 1)

        # Vertices and edges normalization
        for v in self.__graph.get_vertices_list():
            v_id = v.get_id()
            n_val = n_lut[v_id]
            if n_val > 0.:
                hold_t = self.__graph.get_prop_entry_fast(len_id, v_id, 1, np.float)
                hold = hold_t[0] / n_val
                self.__graph.set_prop_entry_fast(len_id, (hold,), v_id, 1)
                hold_t = self.__graph.get_prop_entry_fast(ct_id, v_id, 1, np.float)
                hold = hold_t[0] / n_val
                self.__graph.set_prop_entry_fast(ct_id, (hold,), v_id, 1)
                hold_t = self.__graph.get_prop_entry_fast(sin_id, v_id, 1, np.float)
                hold = hold_t[0] / n_val
                self.__graph.set_prop_entry_fast(sin_id, (hold,), v_id, 1)
                hold_t = self.__graph.get_prop_entry_fast(fness_id, v_id, 1, np.float)
                hold = hold_t[0] / n_val
                self.__graph.set_prop_entry_fast(fness_id, (hold,), v_id, 1)
                hold_t = self.__graph.get_prop_entry_fast(smo_id, v_id, 1, np.float)
                hold = hold_t[0] / n_val
                self.__graph.set_prop_entry_fast(smo_id, (hold,), v_id, 1)
                hold_t = self.__graph.get_prop_entry_fast(mc_id, v_id, 1, np.float)
                hold = hold_t[0] / n_val
                self.__graph.set_prop_entry_fast(mc_id, (hold,), v_id, 1)
        for e in self.__graph.get_edges_list():
            e_id = e.get_id()
            n_val = n_lut[e_id]
            if n_val > 0.:
                hold_t = self.__graph.get_prop_entry_fast(len_id, e_id, 1, np.float)
                hold = hold_t[0] / n_val
                self.__graph.set_prop_entry_fast(len_id, (hold,), e_id, 1)
                hold_t = self.__graph.get_prop_entry_fast(ct_id, e_id, 1, np.float)
                hold = hold_t[0] / n_val
                self.__graph.set_prop_entry_fast(ct_id, (hold,), e_id, 1)
                hold_t = self.__graph.get_prop_entry_fast(sin_id, e_id, 1, np.float)
                hold = hold_t[0] / n_val
                self.__graph.set_prop_entry_fast(sin_id, (hold,), e_id, 1)
                hold_t = self.__graph.get_prop_entry_fast(fness_id, e_id, 1, np.float)
                hold = hold_t[0] / n_val
                self.__graph.set_prop_entry_fast(fness_id, (hold,), e_id, 1)
                hold_t = self.__graph.get_prop_entry_fast(smo_id, e_id, 1, np.float)
                hold = hold_t[0] / n_val
                self.__graph.set_prop_entry_fast(smo_id, (hold,), e_id, 1)
                hold_t = self.__graph.get_prop_entry_fast(mc_id, e_id, 1, np.float)
                hold = hold_t[0] / n_val
                self.__graph.set_prop_entry_fast(mc_id, (hold,), e_id, 1)

    #### Internal functionality area

    def __build(self):

        # Getting GraphGT
        self.__graph_gt = ps.graph.GraphGT(self.__graph)
        graph_gt = self.__graph_gt.get_gt()
        prop_con = graph_gt.edge_properties[STR_EDGE_FNESS]
        prop_v_i = graph_gt.vertex_properties[DPSTR_CELL]
        prop_e_i = graph_gt.edge_properties[DPSTR_CELL]

        # Visiting procedure initialization
        n_vertices = graph_gt.num_vertices()
        connt = np.zeros(shape=(n_vertices, n_vertices), dtype=np.bool)

        # Main loop for finding filaments at every vertex
        for source in graph_gt.vertices():

            prog = (100. * int(source)) / graph_gt.num_vertices()
            print '\t\tProgress ' + str(round(prog, 2)) + '% ...'

            # An isolated vertex cannot be a Filament
            if sum(1 for _ in source.all_edges()) <= 0:
                continue

            # Search filaments in source neighbourhood
            visitor = FilVisitor(graph_gt, source, self.__min_len, self.__max_len)
            gt.dijkstra_search(graph_gt, source, prop_con, visitor)
            hold_v_paths, hold_e_paths = visitor.get_paths()

            # Build the filaments
            for i, v_path in enumerate(hold_v_paths):
                head, tail = v_path[0], v_path[-1]
                head_i, tail_i = int(head), int(v_path[-1])
                if not(connt[head_i, tail_i]) and not(connt[tail_i, head_i]):
                    v_list = list()
                    e_list = list()
                    e_path = hold_e_paths[i]
                    for j in range(len(v_path) - 1):
                        v_id = prop_v_i[v_path[j]]
                        e_id = prop_e_i[e_path[j]]
                        v_list.append(self.__graph.get_vertex(v_id))
                        e_list.append(self.__graph.get_edge(e_id))
                    v_id = prop_v_i[v_path[-1]]
                    v_list.append(self.__graph.get_vertex(v_id))
                    # Building a filament
                    self.__fils.append(FilamentU(self.__graph, v_list, e_list))
                    # Set as unconnected already processed pair of vertices
                    connt[head_i, tail_i] = True
                    connt[tail_i, head_i] = True

##########################################################################################
# Class for modelling the filaments obtained from a Synapse
##########################################################################################

class NetFilamentsSyn(object):

    # fils: list of Filaments extracted from a SynGraphMCF object
    def __init__(self, fils):
        self.__fils = fils

    #### Set/Get methods area

    # Convert the filament network into a VTK PolyData object
    # paths: if 1 (default) vertices are represented though edge paths, otherwise through
    #        vertices locations
    def get_vtp(self, paths=1):

        # Initialization
        point_id = 0
        cell_id = 0
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        cell_data = vtk.vtkIntArray()
        cell_data.SetNumberOfComponents(1)
        cell_data.SetName(STR_CELL)
        len_data = vtk.vtkFloatArray()
        len_data.SetNumberOfComponents(1)
        len_data.SetName(STR_FIL_LEN)
        ct_data = vtk.vtkFloatArray()
        ct_data.SetNumberOfComponents(1)
        ct_data.SetName(STR_FIL_CT)
        tt_data = vtk.vtkFloatArray()
        tt_data.SetNumberOfComponents(1)
        tt_data.SetName(STR_FIL_TT)
        ns_data = vtk.vtkFloatArray()
        ns_data.SetNumberOfComponents(1)
        ns_data.SetName(STR_FIL_NS)
        bs_data = vtk.vtkFloatArray()
        bs_data.SetNumberOfComponents(1)
        bs_data.SetName(STR_FIL_BNS)
        sin_data = vtk.vtkFloatArray()
        sin_data.SetNumberOfComponents(1)
        sin_data.SetName(STR_FIL_SIN)
        apl_data = vtk.vtkFloatArray()
        apl_data.SetNumberOfComponents(1)
        apl_data.SetName(STR_FIL_APL)
        eud_data = vtk.vtkFloatArray()
        eud_data.SetNumberOfComponents(1)
        eud_data.SetName(STR_FIL_EUD)
        fne_data = vtk.vtkFloatArray()
        fne_data.SetNumberOfComponents(1)
        fne_data.SetName(STR_FIL_FNESS)

        # Write lines
        for i, f in enumerate(self.__fils):

            # Getting children if demanded
            if paths == 1:
                coords = f.get_curve_coords()
            else:
                coords = f.get_vertex_coords()
            lines.InsertNextCell(coords.shape[0])
            for c in coords:
                points.InsertPoint(point_id, c[0], c[1], c[2])
                lines.InsertCellPoint(point_id)
                point_id += 1
            cell_id += 1
            cell_data.InsertNextTuple((cell_id,))
            len_data.InsertNextTuple((f.get_length(),))
            ct_data.InsertNextTuple((f.get_total_k(),))
            tt_data.InsertNextTuple((f.get_total_t(),))
            ns_data.InsertNextTuple((f.get_total_ns(),))
            bs_data.InsertNextTuple((f.get_total_bs(),))
            sin_data.InsertNextTuple((f.get_sinuosity(),))
            apl_data.InsertNextTuple((f.get_apex_length(),))
            eud_data.InsertNextTuple((f.get_dst(),))
            fne_data.InsertNextTuple((f.get_filamentness(),))

        # Poly building
        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        poly.SetLines(lines)
        poly.GetCellData().AddArray(cell_data)
        poly.GetCellData().AddArray(len_data)
        poly.GetCellData().AddArray(ct_data)
        poly.GetCellData().AddArray(tt_data)
        poly.GetCellData().AddArray(ns_data)
        poly.GetCellData().AddArray(bs_data)
        poly.GetCellData().AddArray(sin_data)
        poly.GetCellData().AddArray(apl_data)
        poly.GetCellData().AddArray(eud_data)
        poly.GetCellData().AddArray(fne_data)

        return poly

    #### External functionality

    # Filter filaments in the network by their sinuosity
    # low_v: low threshold
    # high_v: high threshold
    # inc: if True (default) the survival filaments are [l, h], otherwise (-infty, l)U(h, infty)
    def filter_by_sinuosity(self, low_th, high_th, inc=True):
        hold_l = self.__fils
        self.__fils = list()
        for fil in hold_l:
            sin = fil.get_sinuosity()
            if inc:
                if (sin >= low_th) and (sin <= high_th):
                    self.__fils.append(fil)
            else:
                if (sin < low_th) or (sin > high_th):
                    self.__fils.append(fil)

    # Filter filaments in the network by Euclidean distance between their extrema
    # low_v: low threshold
    # high_v: high threshold
    # inc: if True (default) the survival filaments are [l, h], otherwise (-infty, l)U(h, infty)
    def filter_by_eud(self, low_th, high_th, inc=True):
        hold_l = self.__fils
        self.__fils = list()
        for fil in hold_l:
            sin = fil.get_dst()
            if inc:
                if (sin >= low_th) and (sin <= high_th):
                    self.__fils.append(fil)
            else:
                if (sin < low_th) or (sin > high_th):
                    self.__fils.append(fil)

    # Filter the vertices of a GraphMCF depending on their presence in the filament network
    # graph: the input GraphMCF to be filtered
    # mode: filtration mode, if 'in' (default) vertices not present in the network are filter out, otherwise those
    #        in the network are the filtered ones
    def filter_graph_mcf(self, graph, mode='in'):

        # Build network vertices LUT
        lut = np.zeros(shape=graph.get_nid(), dtype=np.bool)
        for f in self.__fils:
            for v_id in f.get_vertex_ids():
                try:
                    lut[v_id] = True
                except IndexError:
                    pass

        # Graph filtering
        if mode == 'in':
            for v in graph.get_vertices_list():
                if not lut[v.get_id()]:
                    graph.remove_vertex(v)
        else:
            for v in graph.get_vertices_list():
                if lut[v.get_id()]:
                    graph.remove_vertex(v)

    # Adds a filament property to a GraphMCF object
    # graph: the input GraphMCF to be filtered
    # key_p: filament property, valid: STR_FIL_FNESS (default)
    # mode: fusion mode criteria, valid: 'max' (default)
    def add_vprop_graph_mcf(self, graph, key_p=STR_FIL_FNESS, mode='max'):

        # Initialization
        mode = str(mode)
        if mode != 'max':
            error_msg = 'Input fusion criterium ' + mode + ' is not valid.'
            raise ps.pexceptions.PySegInputError(expr='add_vprop_graph_mcf (NetFilamentsSyn)', msg=error_msg)
        key_p = str(key_p)
        if key_p != STR_FIL_FNESS:
            error_msg = 'Input filamente property ' + key_p + ' not available.'
            raise ps.pexceptions.PySegInputError(expr='add_vprop_graph_mcf (NetFilamentsSyn)', msg=error_msg)
        try:
            key_id = graph.add_prop(key_p, 'float', 1, def_val=-1)
        except Exception:
            error_msg = 'Property could not be added to input graph with class ' + str(graph.__class__) + '.'
            raise ps.pexceptions.PySegInputError(expr='add_vprop_graph_mcf (NetFilamentsSyn)', msg=error_msg)

        # Build network vertices LUT
        if mode == 'max':
            if key_p == STR_FIL_FNESS:
                for f in self.__fils:
                    hold_fv = f.get_filamentness()
                    for v_id in f.get_vertex_ids():
                        try:
                            # Update property value
                            val = graph.get_prop_entry_fast(key_id, v_id, 1, np.float32)[0]
                            if hold_fv > val:
                                graph.set_prop_entry_fast(key_id, (hold_fv,), v_id, 1)
                        except Exception:
                            pass



