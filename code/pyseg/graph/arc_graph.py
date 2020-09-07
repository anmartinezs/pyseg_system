"""
Clases for reducing the size of a SkelGraph but keeping its topology

# Author: Antonio Martinez-Sanchez (Max Planck Institute for Biochemistry)
# Date: 16.09.14
"""

__author__ = 'martinez'

from .core import *
try:
    import disperse_io
except:
    import pyseg.disperse_io
try:
    import pickle as pickle
except:
    import pickle

#####################################################################################################
# Class for holding an Arc. An arc is a set of connected vertices with linear topology.
#
#
class Arc(object):

    #### Constructor Area

    def __init__(self, vertices):
        # The vertices are connected according to its order (v1, v2, ..., vn) -> (v1-v2-...-vn)
        self.__vertices = vertices
        self.__properties = list()
        self.__prop_names = list()
        self.__geometry = None
        self.__edge = None

    #### Set functions area

    def set_edge(self, edge):
        self.__edge = edge

    # If idx is not None the property is addressed according to its index in the list of property,
    # This access will be faster but is dangerous because properties can be added and removed
    # key: string key of the property, no property with name key no action is performed
    # value: can be any object
    def set_property(self, key, value, idx=None):
        if idx is None:
            idx = self.is_property(key)
        self.__properties[idx] = value

    #### Get functions area

    def get_vertices(self):
        return self.__vertices

    def get_vertex(self, idx):
        return self.__vertices[idx]

    def get_start_vertex(self):
        return self.__vertices[0]

    def get_end_vertex(self):
        return self.__vertices[-1]

    def get_length(self):

        vh = self.__vertices[0]
        length = 0
        for i in range(1, len(self.__vertices)):
            x1, x2, x3 = vh.get_coordinates()
            y1, y2, y3 = self.__vertices[i].get_coordinates()
            vh = self.__vertices[i]
            x1 = x1 - y1
            x2 = x2 - y2
            x3 = x3 - y3
            length += math.sqrt(x1*x1 + x2*x2 + x3*x3)

        return length

    # Returns the number of points or (sub-)vertices which compound the arc line
    def get_nvertices(self):
        return len(self.__vertices)

    def get_vertex(self, i):
        return self.__vertices[i]

    # If idx is not None the property is addressed according to its index in the list of property,
    # This access will be faster but is dangerous because properties can be added and removed
    # key: string key of the property, no property with name key no action is performed
    def get_property(self, key, idx=None):
        if idx is None:
            idx = self.is_property(key)
        if idx is not None:
            return self.__properties[idx]
        else:
            return None

    def get_num_properties(self):
        return len(self.__prop_names)

    def get_property_name(self, idx):
        return self.__prop_names[idx]

    #### Functionality area

    # If this already exists return its index, otherwise None
    # key: string name
    def is_property(self, key):
        try:
            idx = self.__prop_names.index(key)
        except ValueError:
            return None
        return idx

    # Add a new property
    # key: string name
    # value: can be any object
    def add_property(self, key, value=None):
        self.__prop_names.append(key)
        self.__properties.append(value)

    # Remove property
    # If idx is not None the property is addressed according to its index in the list of property,
    # This access will be faster but is dangerous because properties can be added and removed
    # key: string key of the property, no property with name key no action is performed
    def remove_property(self, key, idx=None):
        if idx is None:
            idx = self.is_property(key)
        self.__prop_names.pop(idx)
        self.__properties.pop(idx)

    def add_geometry(self, manifold, density):

        # Checks that booth tomograms have the same size
        if manifold.shape != density.shape:
            error_msg = 'Manifold and Density tomograms have different size.'
            raise pexceptions.PySegInputError(expr='add_geometry (Arc)', msg=error_msg)

        # Creates geometry
        verts_coords = np.ndarray(shape=(len(self.__vertices), 3), dtype=tuple)
        for i, v in enumerate(self.get_vertices()):
            t = v.get_coordinates()
            verts_coords[i][0] = t[0]
            verts_coords[i][1] = t[1]
            verts_coords[i][2] = t[2]
        self.__geometry = geometry.ArcGeometry(verts_coords, manifold, density)

    # Reverse the order of the vertices
    def reverse(self):
        self.__vertices.reverse()

    # Creates a new Arc with the same state
    def copy(self):
        hold_vertices = list()
        for v in self.__vertices:
            hold_vertices.append(v.copy())
        hold = Arc(self.__vertices)
        for i in range(self.get_num_properties()):
            hold.add_property(self.get_property_name(i), self.get_property(key=None, idx=i))
        if self.__geometry is not None:
            self.add_geometry(self.__geometry.get_manifold(), self.__geometry.get_density())
        return hold


#####################################################################################################
# Class for represting the edges in an ArcGraph
#
#
class ArcEdge(Edge):

    #### Constructor Area

    # arc: list of Arc objects which holds the topology and geometry of the edge, all of them must
    # have the same start and end vertices
    def __init__(self, arcs):
        a_s = arcs[0].get_start_vertex()
        a_e = arcs[0].get_end_vertex()
        for a in arcs:
            if (a.get_nvertices() < 2) or (a_s != a.get_start_vertex()) or (a_e != a.get_end_vertex()):
                error_msg = 'All arcs must have the same star and end vertices.'
                raise pexceptions.PySegInputWarning(expr='__init__ (ArcEdge)', msg=error_msg)
        super(ArcEdge, self).__init__(a_s, a_e)
        self.__arcs = arcs

    #### Set functions area

    #### Get functions area

    def get_narcs(self):
        return len(self.__arcs)

    def get_arc(self, i):
        return self.__arcs[i]

    def get_arcs(self):
        return self.__arcs

    def get_start_vertex(self):
        return self.__arcs[0].get_start_vertex()

    def get_end_vertex(self):
        return self.__arcs[0].get_end_vertex()

    def remove_arc(self, arc):
        self.__arcs.remove(arc)


    ##### Functionality area

    @classmethod
    def has_geometry(cls):
        return True

    # manifold: numpy array with the labels for the manifolds
    # density: numpy array with the density map
    def add_geometry(self, manifold, density):
        for a in self.__arcs:
            a.add_geometry(manifold, density)

#####################################################################################################
# Class for compressing the complexity of SkelGraph where vertices connected through line of vertices
# are collapsed to arcs. Now arcs also have geometry.
# The skeleton has to be first processed by a factory vtk vertices will become the graph's vertices,
# line cell will become the arcs
#
class ArcGraph(Graph):

    def __init__(self, skel):
        super(ArcGraph, self).__init__(skel)
        self.__a_prop_info = PropInfo()

    #### Set functions area

    #### Get functions area

    def get_arcs_list(self):
        arcs = list()
        for e in self.get_edges_list():
            for a in e.get_arcs():
                arcs.append(a)
        return arcs

    def get_arc_prop_info(self):
        return self.__a_prop_info

    def get_num_arcs(self):
        return len(self.get_arcs_list())

    # Get the vtkPolyData which represent the graph
    def get_vtp(self):

        super(ArcGraph, self).get_vtp()

        # Initialization of the arc properties
        aarrays = list()
        for i in range(self.__a_prop_info.get_nprops()):
            array = disperse_io.TypesConverter.gt_to_vtk(self.__a_prop_info.get_type(i))
            array.SetName(self.__a_prop_info.get_key(i))
            array.SetNumberOfComponents(self.__a_prop_info.get_ncomp(i))
            aarrays.append(array)
        # Id for Arcs
        aid_array = vtk.vtkIntArray()
        aid_array.SetName(STR_ARCS_ID)

        # Set geometry properties of vertices and its topology
        verts = vtk.vtkCellArray()
        for i, v in enumerate(self.get_vertices_list()):
            for j in range(self._Graph__v_prop_info.get_nprops()):
                point_id = v.get_id()
                t = v.get_property(self._Graph__v_prop_info.get_key(j))
                if isinstance(t, tuple):
                    for k in range(self._Graph__v_prop_info.get_ncomp(j)):
                        self._Graph__parrays[j].InsertComponent(point_id, k, t[k])
                else:
                    self._Graph__parrays[j].InsertComponent(point_id, 0, t)
            self._Graph__vid_array.InsertComponent(point_id, 0, point_id)
            self._Graph__length_array.InsertComponent(point_id, 0, -1)
            aid_array.InsertComponent(point_id, 0, -1)
            verts.InsertNextCell(1)
            verts.InsertCellPoint(point_id)

        # Edges topology
        nlines = verts.GetNumberOfCells()
        lines = vtk.vtkCellArray()
        edges = self.get_edges_list()
        aid = 1
        for i, e in enumerate(edges):
            arcs = edges[i].get_arcs()
            for a in arcs:
                nverts = a.get_nvertices()
                lines.InsertNextCell(nverts)
                for v in a.get_vertices():
                    lines.InsertCellPoint(v.get_id())
                for j in range(self.__a_prop_info.get_nprops()):
                    t = e.get_property(self.__a_prop_info.get_key(j))
                    if isinstance(t, tuple):
                        for k in range(self.__a_prop_info.get_ncomp(j)):
                            aarrays[j].InsertComponent(nlines, k, t[k])
                    else:
                        aarrays[j].InsertComponent(nlines, 0, t[0])
                aid_array.InsertComponent(nlines, 0, aid)
                self._Graph__length_array.InsertComponent(nlines, 0, a.get_length())
                nlines += 1
                aid += 1

        # Build final poly
        self._Graph__poly.SetVerts(verts)
        self._Graph__poly.SetLines(lines)
        self._Graph__poly.GetPointData().AddArray(self._Graph__vid_array)
        for i in range(self._Graph__v_prop_info.get_nprops()):
            self._Graph__poly.GetPointData().AddArray(self._Graph__parrays[i])
        self._Graph__poly.GetCellData().AddArray(aid_array)
        self._Graph__poly.GetCellData().AddArray(self._Graph__length_array)
        for i in range(self.__a_prop_info.get_nprops()):
            self._Graph__poly.GetCellData().AddArray(aarrays[i])

        # Delete intermediate variables
        del self._Graph__parrays
        del self._Graph__vid_array
        del self._Graph__earrays
        del self._Graph__length_array

        return self._Graph__poly


    #### Functionality area

    # name: Value STR_EDGE_LENGTH for property name is not allowed and if the name already exists
    # character '_' is added
    # type: graph-tool data type
    # ncomp: number of components of the property (default 1)
    # def_value: initial value for the property (default 0), it could also be an array of values
    def add_arc_property(self, name, type, ncomp=1, def_value=0):

        if name != STR_ARCS_ID:
            if self.__a_prop_info.is_already(name) is not None:
                name += '_'
            for a in self.get_arcs_list():
                a.add_property(key=name, value=def_value)
            self.__a_prop_info.add_prop(name, type, ncomp)
            return
        return NO_PROPERTY_INSERTED

    def remove_arc_property(self, name):
        if self.__a_prop_info.is_already(name) is not None:
            for a in self.get_arcs_list():
                a.remove_property(key=name)
        self.__a_prop_info.remove_prop(key=name)


    # This function must be call for having access to the most functionalities of this class
    # build = if it is False (default True) the object is update but no vertex nor edges is inserted
    def update(self, build=True):

        super(ArcGraph, self).update()

        if build:

            # Parse the skeleton
            if not isinstance(self._Graph__skel, vtk.vtkPolyData):
                error_msg = 'The skeleton must be a vtkPolyData.'
                raise pexceptions.PySegInputWarning(expr='update (ArcGraph)', msg=error_msg)
            lines = self._Graph__skel.GetLines()
            verts = self._Graph__skel.GetVerts()

            # Firstly, vertices are inserted
            nvprops = len(self._Graph__v_props_array)
            v_props_name = self._Graph__v_prop_info.get_keys()
            line_id = 0
            for i in range(verts.GetNumberOfCells()):
                pts = vtk.vtkIdList()
                verts.GetCell(line_id, pts)
                point_id = pts.GetId(0)
                # Catching VTK properties
                v_props_value = list()
                for k in range(nvprops):
                    v_props_value.append(self._Graph__v_props_array[k].GetTuple(point_id))
                # Adding vertices to the graph
                self.insert_vertex(point_id, v_props_value, v_props_name)
                line_id = line_id + pts.GetNumberOfIds() + 1

            # Secondly, arcs are inserted
            # neprops = len(self._Graph__e_props_array)
            # e_props_name = self._Graph__e_prop_info.get_keys()
            hold_v_start = None
            hold_v_end = None
            line_id = 0
            for i in range(lines.GetNumberOfCells()):
                pts = vtk.vtkIdList()
                lines.GetCell(line_id, pts)
                npoints = pts.GetNumberOfIds()
                if npoints >= 2:
                    # Consecutive arcs with the same start and end will be part of the same edge
                    v_start = self.get_vertex(pts.GetId(0))
                    v_end = self.get_vertex(pts.GetId(npoints-1))
                    arc_vertices = list()
                    # Add intermediate vertices to the arc
                    for j in range(1, npoints-1):
                        arc_vertices.append(Vertex(pts.GetId(j), self._Graph__skel))
                    if (hold_v_start is None) or (hold_v_end is None) \
                            or not (((hold_v_start.get_id() == v_start.get_id()) \
                            and (hold_v_end.get_id() == v_end.get_id())) \
                            or ((hold_v_start.get_id() == v_end.get_id()) \
                            and (hold_v_end.get_id() == v_end.get_id()))):
                        arcs_list = list()
                        hold_v_start = v_start
                        hold_v_end = v_end
                    # Add the head and the tail of the arc
                    arc_vertices.insert(0, v_start)
                    arc_vertices.append(v_end)
                    arcs_list.append(Arc(arc_vertices))
                    self.insert_edge(arcs_list)
                line_id = line_id + npoints + 1

    # Insert a new edge in the graph only
    # v_start: initial vertex
    # v_end: end vertex
    # props_value: list with the values for the vertex properties
    # props_name: list with the name of the vertex properties
    # arc: comprises the geometry of the arc
    # return: the vertex id in the graph
    def insert_edge(self, arc, props_value=None, props_name=None):

        e = ArcEdge(arc)
        for a in arc:
            a.set_edge(e)
        super(ArcGraph, self).insert_edge(e, e.get_start(), e.get_end(), props_value, props_name)

        return e

    # Remove an arc if it is the only one in the edge also the edge
    def remove_arc(self, arc):
        e = arc.get_edge()
        e.remove_arc(arc)
        if e.get_narcs() <= 0:
            self.remove_edge(e)

    # Compute arc max density
    # norm_range: if a range (two elements tuple) is passed then the densities are normalized to
    # this range. [min max]
    def compute_arcs_max_density(self, norm_range=None):

        if self._Graph__v_prop_info.is_already(STR_DENSITY_VERTEX) is None:
            error_msg = 'Vertices must have %s property.' % STR_DENSITY_VERTEX
            raise pexceptions.PySegInputWarning(expr='compute_arcs_max_density (ArcGraph)', msg=error_msg)

        # Computing
        arcs = self.get_arcs_list()
        l_max = np.zeros(shape=len(arcs), dtype=np.float)
        for i, a in enumerate(arcs):
            hold = np.finfo(np.float).min
            for v in a.get_vertices():
                v_den = v.get_property(STR_DENSITY_VERTEX)
                if v_den > hold:
                    hold = v_den
            l_max[i] = hold

        if norm_range is not None:
            l_max = lin_map(l_max, norm_range[0], norm_range[1])

        # Adding/setting the property
        if self.__a_prop_info.is_already(STR_ARC_MAX_DENSITY) is None:
            self.add_arc_property(STR_ARC_MAX_DENSITY, disperse_io.TypesConverter().numpy_to_gt(np.float))
        if len(arcs) > 0:
            idx = arcs[0].is_property(STR_ARC_MAX_DENSITY)
            for i, a in enumerate(arcs):
                a.set_property(STR_ARC_MAX_DENSITY, l_max[i], idx)

    # Compare two arcs of this ArgGraph.
    # Return True if the vertices of both arcs in the graph are the same
    def arcs_eq(self, arc_1, arc_2):

        vertices_1 = arc_1.get_vertices()
        vertices_2 = arc_2.get_vertices()
        n_vertices_1 = len(vertices_1)
        n_vertices_2 = len(vertices_2)

        if n_vertices_1 != n_vertices_2:
            return False
        for i in range(n_vertices_1):
            if vertices_1[i].get_id() != vertices_2[i].get_id():
                return False
        return True

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
            raise pexceptions.PySegInputError(expr='pickle (ArcGraph)', msg=error_msg)


    # From an input SkelGraph copies the properties of its vertices to non-extremal vertices
    # of the arcs
    def copy_varcs_prop(self, skel_graph):

        # Copy properties
        for a in self.get_arcs_list():
            for i in range(1, a.get_nvertices()):
                v = a.get_vertex(i)
                v_skel = skel_graph.get_vertex(v.get_id())
                names = v_skel.get_prop_names_list()
                values = v_skel.get_properties_list()
                for j in range(len(names)):
                    v.add_property(names[j], values[j])

    # Copies the prop info of other instance of ArcGraph
    def copy_prop_info(self, arc_graph):

        v_prop_info, e_prop_info = arc_graph.get_prop_info()
        a_prop_info = arc_graph.get_arc_prop_info()
        for i in range(v_prop_info.get_nprops()):
            key = v_prop_info.get_key(i)
            if self.__Graph_v_prop_info.is_already() is None:
                self.__Graph_v_prop_info.add_property(key=key, type=v_prop_info.get_type(i),
                                                      ncomp=v_prop_info.get_ncomp(i))
        for i in range(e_prop_info.get_nprops()):
            key = e_prop_info.get_key(i)
            if self.__Graph_e_prop_info.is_already() is None:
                self.__Graph_e_prop_info.add_property(key=key, type=e_prop_info.get_type(i),
                                                      ncomp=e_prop_info.get_ncomp(i))
        for i in range(a_prop_info.get_nprops()):
            key = a_prop_info.get_key(i)
            if self.__a_prop_info.is_already() is None:
                self.__a_prop_info.add_property(key=key, type=a_prop_info.get_type(i),
                                                ncomp=a_prop_info.get_ncomp(i))

    # Compute the arc density relevance in the ArcGraph
    def compute_arc_relevance(self):

        if (self._Graph__manifolds is None) and (self._Graph__densigy is None):
            return

        # Compute arcs total density
        arcs = self.get_arcs_list()
        densities = np.zeros(shape=len(arcs), dtype=np.float64)
        for i, a in enumerate(arcs):
            for v in a.get_vertices():
                geom = v.get_geometry()
                densities[i] += geom.get_total_density()

        # Get cumulative distribution function
        densities = lin_map(densities, lb=densities.max(), ub=densities.min())
        densities /= densities.sum()
        arg = np.argsort(densities)
        densities_sort = densities[arg]
        cdf = np.zeros(shape=densities.shape, dtype=np.float)
        for i in range(1, densities_sort.shape):
            cdf[i] = cdf[i-1] + densities_sort[i]

        # Set the new property to all arcs
        self.remove_arc_property(STR_ARC_RELEVANCE)
        self.add_arc_property(self, STR_ARC_RELEVANCE,
                              disperse_io.TypesConverter().numpy_to_gt(cdf.dtype), 1, 0)
        for i, a in enumerate(arcs):
            a.set_property(STR_ARC_RELEVANCE, cdf[i])

    def theshold_arcs(self, prop, thres, oper):

        if self.__Graph_a_prop_info.is_already(prop) is None:
            error_msg = 'The graph does not include %s property.' % prop
            raise pexceptions.PySegInputError(expr='theshold_arcs (ArcGraph)', msg=error_msg)

        for a in self.get_arcs_list():
            if not oper(a.get_property(key=prop), thres):
                self.remove_arc(a)


    #### Internal functionality area

    def __setstate__(self, state):
        super(ArcGraph, self).__setstate__(state)
        # self.__dict__.update(state)
        self.__update_arc_skel()

    # Update the skel of all the vertices in the arcs. Used for unplicking
    def __update_arc_skel(self):
        for a in self.get_arcs_list():
            for v in a.get_vertices():
                v.set_skel(self._Graph__skel)



