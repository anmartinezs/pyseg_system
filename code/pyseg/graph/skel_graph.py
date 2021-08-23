"""
Classes for expressing a DisPerSe skeleton as a graph

# Author: Antonio Martinez-Sanchez (Max Planck Institute for Biochemistry)
"""

__author__ = 'martinez'

try:
    import pickle as pickle
except:
    import pickle
from .core import *

#####################################################################################################
# Class for holding a Graph which describes a DISPERSE skeleton
# Just VTK_LINE cells are processed
#
class SkelGraph(Graph):

    #### Constructor Area
    # By default vertex property STR_VERTEX_ID (id point in skeleton) and edges properties STR_EDGE_LENGTH and 'weight'
    # skel: vtk poly data where lines will be used for building a graph
    def __init__(self, skel):
        super(SkelGraph, self).__init__(skel)

    #### Set functions area

    #### Get functions area

    # Get the vtkPolyData which represent the graph
    def get_vtp(self):

        super(SkelGraph, self).get_vtp()

        # Set geometry properties of vertices
        for i, v in enumerate(self.get_vertices_list()):
            point_id = v.get_id()
            for j in range(self._Graph__v_prop_info.get_nprops()):
                t = v.get_property(self._Graph__v_prop_info.get_key(j))
                if isinstance(t, tuple):
                    for k in range(self._Graph__v_prop_info.get_ncomp(j)):
                        self._Graph__parrays[j].SetComponent(point_id, k, t[k])
                else:
                    self._Graph__parrays[j].SetComponent(point_id, 0, t)
            self._Graph__vid_array.SetComponent(point_id, 0, point_id)

        # Topology (Each edge is expressed as a line)
        lines = vtk.vtkCellArray()
        for i, e in enumerate(self.get_edges_list()):
            lines.InsertNextCell(2)
            lines.InsertCellPoint(e.get_start().get_id())
            lines.InsertCellPoint(e.get_end().get_id())
            for j in range(self._Graph__e_prop_info.get_nprops()):
                t = e.get_property(self._Graph__e_prop_info.get_key(j))
                if isinstance(t, tuple):
                    for k in range(self._Graph__e_prop_info.get_ncomp(j)):
                        self._Graph__earrays[j].InsertComponent(i, k, t[k])
                else:
                    self._Graph__earrays[j].InsertComponent(i, 0, t)
            self._Graph__length_array.InsertComponent(i, 0, e.get_length())

        # Build final poly
        self._Graph__poly.SetLines(lines)
        self._Graph__poly.GetPointData().AddArray(self._Graph__vid_array)
        for i in range(self._Graph__v_prop_info.get_nprops()):
            self._Graph__poly.GetPointData().AddArray(self._Graph__parrays[i])
        self._Graph__poly.GetCellData().AddArray(self._Graph__length_array)
        for i in range(self._Graph__e_prop_info.get_nprops()):
            self._Graph__poly.GetCellData().AddArray(self._Graph__earrays[i])

        # Delete intermediate variables
        del self._Graph__parrays
        del self._Graph__vid_array
        del self._Graph_Graph__earrays
        del self._Graph__length_array

        return self._Graph__poly

    #### Functionality area

    # This function must be call for having access to the most functionalities of this class
    # build = if it is False (defult True) the object is update but no vertex nor edges is inserted
    def update(self, build=True):

        super(SkelGraph, self).update()

        if build:

            # Convert skel point in lines into vertices and set edges between connected points
            nvprops = len(self._Graph__v_props_array)
            neprops = len(self._Graph__e_props_array)
            v_props_name = self._Graph__v_prop_info.get_keys()
            e_props_name = self._Graph__e_prop_info.get_keys()
            for i in range(self._Graph__skel.GetNumberOfCells()):
                cell = self._Graph__skel.GetCell(i)
                if (cell.GetNumberOfPoints() > 1) and cell.IsLinear():
                    pts = cell.GetPointIds()
                    for j in range(1, pts.GetNumberOfIds()):
                        point_id_v1 = pts.GetId(j-1)
                        point_id_v2 = pts.GetId(j)
                        # Catching VTK properties
                        v1_props_value = list()
                        v2_props_value = list()
                        for k in range(nvprops):
                            v1_props_value.append(self._Graph__v_props_array[k].GetTuple(point_id_v1))
                            v2_props_value.append(self._Graph__v_props_array[k].GetTuple(point_id_v2))
                        # Adding vertices to the graph
                        v1 = self.insert_vertex(point_id_v1, v1_props_value, v_props_name)
                        v2 = self.insert_vertex(point_id_v2, v2_props_value, v_props_name)
                        # Adding edges to the graph (no self loops)
                        if v1.get_id() != v2.get_id():
                            e_props_value = list()
                            for k in range(neprops):
                                e_props_value.append(self._Graph__e_props_array[k].GetTuple(i))
                            self.insert_edge(v1, v2, e_props_value, e_props_name)

    # Insert a new edge in the graph only
    # v_start: initial vertex
    # v_end: end vertex
    # props_value: list with the values for the vertex properties
    # props_name: list with the name of the vertex properties
    # return: the vertex id in the graph
    def insert_edge(self, v_start, v_end, props_value=None, props_name=None):

        # Create and insert the edge
        e = Edge(v_start, v_end)
        super(SkelGraph, self).insert_edge(e, v_start, v_end, props_value, props_name)

        return e

    # Use this method for pickling instead of call pickle.dump() directly
    # fname: file name ended with .pkl
    def pickle(self, fname):

        # Dump pickable objects and store the file names of the unpickable objects
        stem, ext = os.path.splitext(fname)
        self._Graph__skel_fname = stem + '_skel.vtp'
        pkl_f = open(fname, 'w')
        try:
            pickle.dump(self, pkl_f)
        finally:
            pkl_f.close()

        # Store unpickable objects
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(self._Graph__skel_fname)
        vtk_ver = vtk.vtkVersion().GetVTKVersion()
        if int(vtk_ver[0]) < 6:
            writer.SetInput(self._Graph__skel)
        else:
            writer.SetInputData(self._Graph__skel)
        if writer.Write() != 1:
            error_msg = 'Error writing %s.' % (self._Graph__skel_fname)
            raise pexceptions.PySegInputError(expr='pickle (SkelGraph)', msg=error_msg)

    # Threshold vertex maxima according to a property
    def threshold_maxima(self, prop, thres, oper):

        if self._Graph__v_prop_info.is_already(prop) is None:
            error_msg = 'The graph does not include %s property.' % prop
            raise pexceptions.PySegInputError(expr='theshold_maxima (SkelGraph)', msg=error_msg)

        if self._Graph__v_prop_info.is_already(STR_CRITICAL_VERTEX) is None:
            self.find_critical_points()

        for v in self.get_vertices_list():
            if v.get_property(key=STR_CRITICAL_VERTEX) == CRITICAL_MAX:
                if not oper(v.get_property(key=prop), thres):
                    self.remove_vertex(v.get_id())

    # Threshold vertex maxima according to a property
    def threshold_minima(self, prop, thres, oper):

        if self._Graph__v_prop_info.is_already(prop) is None:
            error_msg = 'The graph does not include %s property.' % prop
            raise pexceptions.PySegInputError(expr='theshold_minima (SkelGraph)', msg=error_msg)

        if self._Graph__v_prop_info.is_already(STR_CRITICAL_VERTEX) is None:
            self.find_critical_points()

        for v in self.get_vertices_list():
            if v.get_property(key=STR_CRITICAL_VERTEX) == CRITICAL_MIN:
                if not oper(v.get_property(key=prop), thres):
                    self.remove_vertex(v.get_id())





