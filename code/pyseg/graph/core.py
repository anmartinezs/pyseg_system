"""
General definitions for the graph package

# Author: Antonio Martinez-Sanchez (Max Planck Institute for Biochemistry)
# Date: 19.09.14
"""


__author__ = 'martinez'

from . import geometry
try:
    import disperse_io
except:
    import pyseg.disperse_io
import scipy.sparse as sp
try:
    from globals import *
except:
    from pyseg.globals import *
from abc import *
import graph_tool.all as gt

#####################################################################################################
# Class for holding just the information with the id and the neighbours of a vertex
#
class VertexCore(object):

    #### Constructor Area

    def __init__(self, id, skel):
        self.__id = id # Id for the SkelGraph class
        self.__skel = skel
        self.__neighbours = list()
        # For avoiding cycling objects during pickling
        self.__neighbours_id = list()

    #### Pickling methods

    # Copy the object's state from self.__dict__ which contains all instance attributes.
    # Afterwards remove unpickable objects
    def __getstate__(self):
        self.get_neighbours_id()
        state = self.__dict__.copy()
        # Delete VTK objects
        del state['_VertexCore__skel']
        # For avoiding cycles
        state['_VertexCore__neighbours'] = None
        return state

    # Restore previous state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__neighbours = list()

    #### Set functions area

    def set_skel(self, skel):
        self.__skel = skel

    #### Get functions area

    def get_id(self):
        return self.__id

    def get_skel(self):
        return self.__skel

    # Return the neighbours list
    def get_neighbours(self):
        return self.__neighbours

    # Return the neighbours id list
    def get_neighbours_id(self, update=True):
        if update:
            self.__neighbours_id = list()
            for v in self.__neighbours:
                self.__neighbours_id.append(v.get_id())
        return self.__neighbours_id

    # Return the number of neighbour vertices
    def get_num_neighbours(self):
        return len(self.__neighbours)

    #### Functionality area

    # Add 'n' as neighbour vertex only if it is not in the neighbours list already
    # Return: index in list or -1 if the vertex was already added
    def add_neighbour(self, n):
        try:
            self.__neighbours.index(n)
            return -1
        except ValueError:
            self.__neighbours.append(n)
            return len(self.__neighbours) - 1

    # Remove neighbour
    # Remove the 'n' from neighbours list
    # Return: 1 if 'n' was in the list and was successfully remove, -1 otherwise
    def remove_neighbour(self, n):
        try:
            self.__neighbours.remove(n)
            return 1
        except ValueError:
            return -1

#####################################################################################################
# Class for holding a Vertex
#
class Vertex(VertexCore):

    #### Constructor Area

    def __init__(self, id, skel):
        super(Vertex, self).__init__(id, skel)
        self.__geometry = None
        self.__prop_names = list()
        self.__properties = list()
        self.__vertices = None

    #### Set functions area

    # If idx is not None the property is addressed according to its index in the list of property,
    # This access will be faster but is dangerous because properties can be added and removed
    # key: string key of the property, no property with name key no action is performed
    # value: can be any object
    def set_property(self, key, value, idx=None):
        if idx is None:
            idx = self.is_property(key)
        self.__properties[idx] = value

    #### Get functions area

    def get_geometry(self):

        if self.__geometry is None:
            error_msg = 'No geometry added to this vertex.'
            raise pexceptions.PySegInputWarning(expr='get_geometry (Vertex)', msg=error_msg)

        return self.__geometry

    def get_coordinates(self):
        return self._VertexCore__skel.GetPoint(self._VertexCore__id)

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

    def get_properties_list(self):
        return self.__properties

    def get_prop_names_list(self):
        return self.__prop_names

    # Get a VetexCore version of the self
    def get_core(self):
        v = VertexCore(self._VertexCore__id, self._VertexCore__skel)
        for n in self.get_neighbours():
            v.add_neighbour(n)
        return v

    def get_num_properties(self):
        return len(self.__prop_names)

    def get_property_name(self, idx):
        return self.__prop_names[idx]

    #### Functionality area

    # manifold: numpy array with the labels for the manifolds
    # density: numpy array with the density map
    def add_geometry(self, manifold, density):

        # Checks that booth tomograms have the same size
        if manifold.shape != density.shape:
            error_msg = 'Manifold and Density tomograms have different size.'
            raise pexceptions.PySegInputError(expr='add_geometry (Vertex)', msg=error_msg)

        # Creates geometry
        self.__geometry = geometry.PointGeometry(self.get_coordinates(), manifold, density)

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

    # Creates a new Vertex with the same state (not valid for child classes)
    # TODO: AND WITHOUT REFERENCE TO NEIGHBOURS
    def copy(self):
        # Creates new Vertex
        hold = Vertex(self.get_id(), self.get_skel())
        # Set property
        for i in range(self.get_num_properties()):
            hold.add_property(self.get_property_name(i), self.get_property(key=None, idx=i))
        # Set geometry
        if self.__geometry is not None:
            hold.add_geometry(self.__geometry.get_manifold(), self.__geometry.get_density())
        return hold

#####################################################################################################
# Class for holding an Edge
#
class Edge(object):

    #### Constructor Area

    # [start, end] -> Start and End vertices
    def __init__(self, start, end):
        self.__start = start
        self.__end = end
        self.__length = self.get_length()
        self.__prop_names = list()
        self.__properties = list()

    #### Set functions area

    # If idx is not None the property is addressed according to its index in the list of property,
    # This access will be faster but is dangerous because properties can be added and removed
    # key: string key of the property, no property with name key no action is performed
    # value: can be any object
    def set_property(self, key, value, idx=None):
        if idx is None:
            idx = self.is_property(key)
        self.__properties[idx] = value

    #### Get functions area

    def get_length(self):

        x1, x2, x3 = self.__start.get_coordinates()
        y1, y2, y3 = self.__end.get_coordinates()
        x1 = x1 - y1
        x2 = x2 - y2
        x3 = x3 - y3
        return math.sqrt(x1*x1 + x2*x2 + x3*x3)

    def get_start(self):
        return self.__start

    def get_end(self):
        return self.__end


    # If idx is not None the property is addressed according to its index in the list of property,
    # This access will be faster but is dangerous because properties can be added and removed
    # key: string key of the property, no property with name key no action is performed
    def get_property(self, key, idx=None):
        if idx is None:
            idx = self.is_property(key)
        return self.__properties[idx]

    def get_properties_list(self):
        return self.__properties

    def get_prop_names_list(self):
        return self.__prop_names

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
    def add_property(self, key, value):
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

    @classmethod
    def has_geometry(cls):
        return False

#########################################################################################################
# Class for holding the information of the properties
#
class PropInfo(object):

    ###### Constructor Area

    # key-> string which identifies the property
    # type-> only 'int' and 'double' are currently accepted
    # ncomp-> number of components
    def __init__(self):
        self.__key = list()
        self.__type = list()
        self.__ncomp = list()

    ##### Get Area

    def get_nprops(self):
        return len(self.__key)

    # If this already exists return its index, otherwise None
    def is_already(self, key):

        try:
            idx = self.__key.index(key)
        except:
            return None

        return idx

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
                raise pexceptions.PySegInputWarning(expr='get_type (PropInfo)', msg=error_msg)
            return self.__type[idx]

    def get_ncomp(self, index=-1, key=None):
        if key is None:
            return self.__ncomp[index]
        else:
            try:
                idx = self.__key.index(key)
            except:
                error_msg = "No property found with the key '%s'." % key
                raise pexceptions.PySegInputWarning(expr='get_ncomp (PropInfo)', msg=error_msg)
            return self.__ncomp[idx]

    ##### Functionality area

    def add_prop(self, key, type, ncomp):
        self.__key.append(key)
        self.__type.append(type)
        self.__ncomp.append(ncomp)

    # Remove by index or key (if key parameter is different from 0)
    def remove_prop(self, idx=-1, key=None):

        if key is None:
            if idx < 0:
                self.__key.pop()
                self.__type.pop()
                self.__ncomp.pop()
            else:
                self.__key.remove(idx)
                self.__type.remove(idx)
                self.__ncomp.remove(idx)
        else:
            idx = self.is_already(key)
            if idx is not None:
                self.__key.remove(idx)
                self.__type.remove(idx)
                self.__ncomp.remove(idx)

#####################################################################################################
# Abstract class for the graphs which holds topological and geometrical information
#
#
class Graph(metaclass=ABCMeta):

    # For Abstract Base Classes in python
    def __init__(self, skel):
        self.__skel = skel
        self.__v_prop_info = PropInfo()
        self.__e_prop_info = PropInfo()
        self.__manifolds = None
        self.__density = None
        # Edges are expressed in a Sparse Matrix
        npoints = self.__skel.GetNumberOfPoints()
        self.__edges_c = sp.dok_matrix((npoints, npoints), dtype=np.int)
        self.__edges_m = sp.dok_matrix((npoints, npoints), dtype=np.bool)
        self.__edges = list()
        self.__vertices = np.empty(shape=npoints, dtype=object)
        # For graph properties
        self.__prop_names = list()
        self.__properties = list()
        # For get_vtp method
        self.__poly = None
        self.__parrays = None
        self.__vid_array = None
        self.__earrays = None
        self.__length_array = None
        # For update method
        self.__v_props_array = list()
        self.__e_props_array = list()
        # For pickling VTK objects
        self.__skel_fname = None

    #### Set functions area


    #### Get functions area

    # If idx is not None the property is addressed according to its index in the list of property,
    # This access will be faster but is dangerous because properties can be added and removed
    # key: string key of the property, no property with name key no action is performed
    # value: can be any object
    def set_property(self, key, value, idx=None):
        if idx is None:
            idx = self.is_property(key)
        self.__properties[idx] = value

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

    def get_properties_list(self):
        return self.__properties

    def get_prop_names_list(self):
        return self.__prop_names

    def get_num_properties(self):
        return len(self.__prop_names)

    def get_property_name(self, idx):
        return self.__prop_names[idx]

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

    def get_skel(self):
        return self.__skel

    # Get Vertex object from point_id, it returns None if no Vertex found with this point_id
    def get_vertex(self, point_id):
        return self.__vertices[point_id]

    # Returns a list wiht the vertices which are neigbours to the input vertex
    def get_vertex_neighbours(self, point_id):
        vertex = self.__vertices[point_id]
        if vertex is not None:
            return vertex.get_neighbours()
        else:
            return None

    # Get all Vertex objects in a list
    # core: if True (default = False) the vertices are copied into into a list of VetexCore objects
    def get_vertices_list(self, core=False):
        vertices = list()
        if core:
            for v in self.__vertices:
                if v is not None:
                    vertices.append(v.get_core())
        else:
            for v in self.__vertices:
                if v is not None:
                    vertices.append(v)
        return vertices

    # Get all Edge objects in a list
    def get_edges_list(self):
        edges = list()
        for e in self.__edges:
            if e is not None:
                edges.append(e)
        return edges

    # Return the PropInfo for vertices and edges
    def get_prop_info(self):
        return self.__v_prop_info, self.__e_prop_info

    # Return the edge from the point id of its vertices
    # Note that 'get_edge(end_id, start_id)' will return the same edge
    def get_edge(self, start_id, end_id):
        s_id = int(start_id)
        e_id = int(end_id)
        if self.__edges_m[s_id, e_id]:
            return self.__edges[int(self.__edges_c[s_id, e_id])]
        return None

    # Return the current number of vertices
    def get_num_vertices(self):
        return len(self.get_vertices_list())

    # Return the current number of vertices
    def get_num_edges(self):
        return len(self.get_edges_list())

    # Get the vtkPolyData which represent the graph
    # The abstract class implements the vtkDataArray creation and the geometry setting for vertices,
    # the edge's geometry and topology MUST be done by the child method
    def get_vtp(self):

        # Initialization
        self.__poly = vtk.vtkPolyData()
        self.__poly.SetPoints(self.__skel.GetPoints())
        self.__parrays = list()
        for i in range(self.__v_prop_info.get_nprops()):
            array = disperse_io.TypesConverter.gt_to_vtk(self.__v_prop_info.get_type(i))
            array.SetName(self.__v_prop_info.get_key(i))
            array.SetNumberOfComponents(self.__v_prop_info.get_ncomp(i))
            for j in range(self.__poly.GetNumberOfPoints()):
                for k in range(self.__v_prop_info.get_ncomp(i)):
                    array.InsertComponent(j, k, -1)
            self.__parrays.append(array)
        # Vertex STR_VERTEX_ID default property
        self.__vid_array = vtk.vtkIntArray()
        self.__vid_array.SetName(STR_VERTEX_ID)
        for i in range(self.__poly.GetNumberOfPoints()):
            self.__vid_array.InsertComponent(i, 0, -1)
        self.__earrays = list()
        for i in range(self.__e_prop_info.get_nprops()):
            array = disperse_io.TypesConverter.gt_to_vtk(self.__e_prop_info.get_type(i))
            array.SetName(self.__e_prop_info.get_key(i))
            array.SetNumberOfComponents(self.__e_prop_info.get_ncomp(i))
            self.__earrays.append(array)
        # Edge STR_EDGE_LENGTH default
        self.__length_array = vtk.vtkDoubleArray()
        self.__length_array.SetName(STR_EDGE_LENGTH)


    #### Functionality area

    # This function must be call for having access to the most functionalities of this class
    def update(self):

        # Creating the properties arrays, array names cannot be duplicated
        self.__v_props_array = list()
        for i in range(self.__skel.GetPointData().GetNumberOfArrays()):
            array = self.__skel.GetPointData().GetArray(i)
            ncomp = array.GetNumberOfComponents()
            if ncomp == 1:
                v_props_type = disperse_io.TypesConverter().vtk_to_gt(array)
            else:
                v_props_type = disperse_io.TypesConverter().vtk_to_gt(array, True)
            if self.add_vertex_property(array.GetName(), v_props_type, ncomp) != NO_PROPERTY_INSERTED:
                self.__v_props_array.append(array)
        self.__e_props_array = list()
        for i in range(self.__skel.GetCellData().GetNumberOfArrays()):
            array = self.__skel.GetCellData().GetArray(i)
            ncomp = array.GetNumberOfComponents()
            if ncomp == 1:
                e_props_type = disperse_io.TypesConverter().vtk_to_gt(array)
            else:
                e_props_type = disperse_io.TypesConverter().vtk_to_gt(array, True)
            if self.add_edge_property(array.GetName(), e_props_type, ncomp) != NO_PROPERTY_INSERTED:
                self.__e_props_array.append(array)

    # Insert a new vertex in the graph only if the point id has not been inserted before
    # point_id: id of the point in the skel
    # props_value: list with the values for the vertex properties
    # props_name: list with the name of the vertex properties
    # return: the new or already inserted vertex in the graph, properties are allways set.
    # TODO: If an error occurs the vertex is not inserted and None is returned
    def insert_vertex(self, point_id, props_value=None, props_name=None):

        # Check if the vertex exist
        if self.__vertices[point_id] is None:
            # Create the vertex
            v = Vertex(point_id, self.__skel)
            if (self.__manifolds is not None) and (self.__density is not None):
                v.add_geometry(self.__manifolds, self.__density)
            self.__vertices[point_id] = v
            # Create and set default property
            v.add_property(key=STR_VERTEX_ID, value=point_id)
            # Set optional properties
            if props_value is not None:
                for i in range(len(props_value)):
                    v.add_property(key=props_name[i], value=props_value[i])
        else:
            v = self.__vertices[point_id]

        return v

    # Only one edge is allowed between to vertices
    # Returns: ITEM_INSERTED or NO_ITEM_INSERTED
    def insert_edge(self, e, v_start, v_end, props_value=None, props_name=None):

        # Connection matrix must be symmetric (undirected graph)
        # An edges only is inserted once, i.e. no loops allowed between two neighbours
        s_id = v_start.get_id()
        e_id = v_end.get_id()
        if (self.__edges_m[s_id, e_id] == 0) and (self.__edges_m[e_id, s_id] == 0):

            self.__edges.append(e)
            idx = len(self.__edges) - 1
            self.__edges_c[s_id, e_id] = idx
            self.__edges_c[e_id, s_id] = idx
            self.__edges_m[s_id, e_id] = True
            self.__edges_m[e_id, s_id] = True

            # Add neighbours
            v_start = e.get_start()
            v_end = e.get_end()
            v_start.add_neighbour(v_end)
            v_end.add_neighbour(v_start)

            # Set default property
            e.add_property(key=STR_EDGE_LENGTH, value=e.get_length())

            # Set optional properties
            if props_value is not None:
                for i in range(len(props_value)):
                    e.add_property(key=props_name[i], value=props_value[i])

            return ITEM_INSERTED

        else:
            return NO_ITEM_INSERTED

    # Delete a vertex identified by its id in the skel or the vertex label
    def remove_vertex(self, point_id):
        vertex = self.__vertices[point_id]
        # Remove the vertex from the neighbour list of its neighbours
        for n in vertex.get_neighbours():
            n.remove_neighbour(vertex)
            # Delete edge
            self.remove_edge(self.get_edge(vertex.get_id(), n.get_id()))
        del vertex
        self.__vertices[point_id] = None

    # Delete and edge from the graph
    # Returns: ITEM_DELETED (or NO_ITEM_DELETED), if the item has (or it had already) been deleted
    def remove_edge(self, edge):

        start_id = edge.get_start().get_id()
        end_id = edge.get_end().get_id()
        if (self.__edges_m[start_id, end_id] != 0) or (self.__edges_m[end_id, start_id] != 0):
            idx = int(self.__edges_c[start_id, end_id])
            self.__edges_c[start_id, end_id] = 0
            self.__edges_c[end_id, start_id] = 0
            self.__edges_m[start_id, end_id] = False
            self.__edges_m[end_id, start_id] = False
            self.__edges[idx] = None
            return ITEM_DELETED
        else:
            return NO_ITEM_DELETED

    # If none values passed as 'manifold' and 'density' each vertex created after the call of
    # this function is provided with a PointGeometry. Both inputs should be numpy arrays with
    # the same size
    def add_geometry(self, manifolds=None, density=None):

        if (manifolds is not None) and (density is not None):
            self.__manifolds = manifolds
            self.__density = density

        # name: Value STR_VERTEX_ID for property name is not allowed and if the name already exists
    # character '_' is added
    # type: graph-tool data type
    # ncomp: number of components of the property (default 1)
    # def_value: initial value for the property (default 0)
    # Return NO_PROPERTY_INSERTED if the property has not been inserted
    def add_vertex_property(self, name, type, ncomp=1, def_value=0):

        if name != STR_VERTEX_ID:
            if self.__v_prop_info.is_already(name) is not None:
                name += '_'
            for v in self.get_vertices_list():
                v.add_property(key=name, value=def_value)
            self.__v_prop_info.add_prop(name, type, ncomp)
            return
        return NO_PROPERTY_INSERTED

    # name: Value STR_EDGE_LENGTH for property name is not allowed and if the name already exists
    # character '_' is added
    # type: graph-tool data type
    # ncomp: number of components of the property (default 1)
    # def_value: initial value for the property (default 0), it could also be an array of values
    def add_edge_property(self, name, type, ncomp=1, def_value=0):

        if name != STR_EDGE_LENGTH:
            if self.__e_prop_info.is_already(name) is not None:
                name += '_'
            for e in self.get_edges_list():
                e.add_property(key=name, value=def_value)
            self.__e_prop_info.add_prop(name, type, ncomp)
            return
        return NO_PROPERTY_INSERTED

    # Print the geometry of the set of vertices which compounds the ArcGraph into a numpy ndarray
    # property: key of the vertex property used for labeling, if None (default) density is printed
    # bg_level: level for the background (default=0)
    # th_den: number of sigmas above (+) or below vertex geometry density mean for thresholding,
    #         if None no threshold is applied
    # minima: if True (default) only local minimum vertices are printed
    # lbl: label used if property is None
    def print_vertices(self, property=None, lbl=1, bg_level=0, th_den=None, minima=True):

        if (self.__manifolds is None) or (self.__density is None):
            error_msg = 'Geometry has to be provided to vertex.'
            raise pexceptions.PySegInputWarning(expr='print_vertices (ArcGraph)', msg=error_msg)

        # Build hold volume
        if property is None:
            vol = bg_level * np.ones(self.__density.shape, np.float32)
        else:
            if property == STR_VERTEX_ID:
                np_type = np.int32
            else:
                if self.__v_prop_info.get_ncomp(key=property) != 1:
                    error_msg = 'Tensorial properties cannot be printed.'
                    raise pexceptions.PySegInputWarning(expr='print_vertices (ArcGraph)', msg=error_msg)
                np_type = disperse_io.TypesConverter().gt_to_numpy(self.__v_prop_info.get_type(key=property))
            vol = bg_level * np.ones(self.__density.shape, np_type)

        # Loop for printing the geometry of each vertex
        if property is not None:
            if property == STR_VERTEX_ID:
                for v in self.get_vertices_list():
                    if minima:
                        if v.is_property(STR_CRITICAL_VERTEX) \
                                and v.get_property(STR_CRITICAL_VERTEX) == CRITICAL_MIN:
                            geom = v.get_geometry()
                            geom.print_in_numpy(vol, v.get_id(), th_den)
                    else:
                        geom = v.get_geometry()
                        geom.print_in_numpy(vol, v.get_id(), th_den)
            else:
                for v in self.get_vertices_list():
                    if minima:
                        if v.is_property(STR_CRITICAL_VERTEX) \
                                and v.get_property(STR_CRITICAL_VERTEX) == CRITICAL_MIN:
                            geom = v.get_geometry()
                            geom.print_in_numpy(vol, v.get_property(property), th_den)
                    else:
                        geom = v.get_geometry()
                        geom.print_in_numpy(vol, v.get_property(property), th_den)
        else:
            for v in self.get_vertices_list():
                if minima:
                    if v.is_property(STR_CRITICAL_VERTEX) \
                            and v.get_property(STR_CRITICAL_VERTEX) == CRITICAL_MIN:
                        geom = v.get_geometry()
                        geom.print_in_numpy(vol=vol, lbl=lbl, th_den=th_den)
                else:
                    geom = v.get_geometry()
                    geom.print_in_numpy(vol=vol, lbl=lbl, th_den=th_den)

        return vol

    # Update just the geometry of the vertices and edges
    def update_geometry(self, manifolds=None, density=None):

        if manifolds is not None:
            self._Graph__manifolds = manifolds
        if density is not None:
            self._Graph__density = density

        # Loop for adding geometry to all vertices
        for v in self.get_vertices_list():
            v.add_geometry(self._Graph__manifolds, self._Graph__density)

        # Loop for adding geometry to all vertices
        for e in self.get_edges_list():
            if e.has_geometry():
                e.add_geometry(self._Graph__manifolds, self._Graph__density)

    # Identifies the type of critical point represented for each vertex and stores this information
    # as STR_CRITICAL_VERTEX vertex property.
    # This functions also associates a density value property (STR_DENSITY_VERTEX) to all vertices
    def find_critical_points(self):

        if self._Graph__density is None:
            error_msg = 'No density filed found.'
            raise pexceptions.PySegInputWarning(expr='find_critical_points (SkelGraph)', msg=error_msg)
        vertices = self.get_vertices_list()
        if len(vertices) <= 0:
            error_msg = 'Graph without vertices.'
            raise pexceptions.PySegInputWarning(expr='find_critical_points (SkelGraph)', msg=error_msg)

        # Get densities of the vertex and its neighbours
        for v in vertices:
            coord = v.get_coordinates()
            # coord = (int(round(coord[0])), int(round(coord[1])), int(round(coord[2])))
            v_den = trilin_tomo(coord[0], coord[1], coord[2], self._Graph__density)
            if v.is_property(STR_DENSITY_VERTEX):
                v.set_property(STR_DENSITY_VERTEX, v_den)
            else:
                v.add_property(STR_DENSITY_VERTEX, v_den)
            n_dens = np.zeros(v.get_num_neighbours(), dtype=np.float32)
            for i, n in enumerate(v.get_neighbours()):
                ncoord = n.get_coordinates()
                # ncoord = (int(round(ncoord[0])), int(round(ncoord[1])), int(round(ncoord[2])))
                n_dens[i] = trilin_tomo(ncoord[0], ncoord[1], ncoord[2], self._Graph__density)
                if n.is_property(STR_DENSITY_VERTEX):
                    n.set_property(STR_DENSITY_VERTEX, n_dens[i])
                else:
                    n.add_property(STR_DENSITY_VERTEX, n_dens[i])

            # Get the type of critical point
            if np.sum(v_den < n_dens) == 0:
                if v.is_property(STR_CRITICAL_VERTEX):
                    v.set_property(STR_CRITICAL_VERTEX, CRITICAL_MAX)
                else:
                    v.add_property(STR_CRITICAL_VERTEX, CRITICAL_MAX)
            elif np.sum(v_den > n_dens) == 0:
                if v.is_property(STR_CRITICAL_VERTEX):
                    v.set_property(STR_CRITICAL_VERTEX, CRITICAL_MIN)
                else:
                    v.add_property(STR_CRITICAL_VERTEX, CRITICAL_MIN)
            else:
                if v.is_property(STR_CRITICAL_VERTEX):
                    v.set_property(STR_CRITICAL_VERTEX, CRITICAL_SAD)
                else:
                    v.add_property(STR_CRITICAL_VERTEX, CRITICAL_SAD)

        # Update vertex properties list
        if self.__v_prop_info.is_already(STR_CRITICAL_VERTEX) is None:
            self.__v_prop_info.add_prop(STR_CRITICAL_VERTEX, 'int', 1)
        if self.__v_prop_info.is_already(STR_DENSITY_VERTEX) is None:
            self.__v_prop_info.add_prop(STR_DENSITY_VERTEX, 'float', 1)

    # Uses pseudo_diameter of graph_tool package
    # mode: 'geom' the output is in the geometric metric, 'topo' the output is in number of nodes
    # resolution: nm per voxel for the geometric metric
    def compute_diameter(self, mode='geom', resolution=1):

        # Build graph_tool graph
        graph = gt.Graph(directed=False)
        for v in self.__vertices:
            graph.add_vertex(v)
        weights = np.zeros(shape=len(self.__edges), dtype=np.float)
        for i, e in enumerate(self.__edges):
            graph.add_edge(e.get_start(), e.get_end())
            weights[i] = e.get_length() * resolution
        w_prop = graph.new_edge_property("float")
        w_prop.get_array()[:] = weights

        # Measure pseudo diamter
        t_diam, ends = gt.pseudo_diameter(graph)

        if mode == 'topo':
            return t_diam
        else:
            dist = gt.shortest_distance(graph, ends[0], ends[1], weights)
            return dist.a[0]

    # Method for pickling
    @abstractmethod
    def pickle(self, fname):
        raise NotImplementedError('pickle (Graph)')


    ############## Internal functionality area

    def __update_vertices_skel(self):
        # Set the skeleton to all vertex
        for v in self.get_vertices_list():
            v.set_skel(self.__skel)

    # Restore previous state
    def __setstate__(self, state):
        self.__dict__.update(state)
        # Restore unpickable objects
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(self._Graph__skel_fname)
        reader.Update()
        self.__skel = reader.GetOutput()
        self.__update_vertices_skel()
        # Restore neighbours in vertices
        for v in self.get_vertices_list():
            for n_id in v.get_neighbours_id(update=False):
                v.add_neighbour(self.get_vertex(n_id))
        # Create hold unpickable objects (set to default value)
        self.__poly = None
        self.__parrays = None
        self.__vid_array = None
        self.__earrays = None
        self.__length_array = None
        # For update method
        self.__v_props_array = list()
        self.__e_props_array = list()

    # Copy the object's state from self.__dict__ which contains all instance attributes.
    # Afterwards remove unpickable objects
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_Graph__skel']
        del state['_Graph__poly']
        del state['_Graph__parrays']
        del state['_Graph__vid_array']
        del state['_Graph__earrays']
        del state['_Graph__length_array']
        del state['_Graph__v_props_array']
        del state['_Graph__e_props_array']
        return state