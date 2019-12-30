"""
Classes for building proper SkelGraphs objects for structures attached to biological
membranes

# Author: Antonio Martinez-Sanchez (Max Planck Institute for Biochemistry)
# Date: 08.09.14
"""

__author__ = 'martinez'
__version__ = "$Revision: 002 $"

try:
    import disperse_io
except:
    import pyseg.disperse_io
try:
    import graph
except:
    import pyseg.graph
try:
    from globals import *
except:
    from pyseg.globals import *
import utils
import operator
from abc import *
import math


#####################################################################################################
#   Abstract class for membrane factories
#
#
class MembFactory(object):

    # For Abstract Base Classes in python
    __metaclass__ = ABCMeta

    #### Constructor Area
    # By default vertex property STR_VERTEX_ID (id point in skeleton) and edges properties STR_EDGE_LENGTH and 'weight'
    # skel: vtk poly data where lines will be used for building a graph (from DisPerSe)
    # manifolds: numpy 3darray for extracting the geometry
    # density: numpy 3darray with the original tomogram
    # memb_seg: numpy 3darray with the segmentation of the membranes (>0: fg, 0: bg)
    def __init__(self, skel, manifolds, density, memb_seg):
        self.__skel = skel
        self.__filt_skel = None
        self.__density = density
        self.__manifolds = manifolds
        self.__memb_seg = memb_seg
        self.__mb_thickness = 0
        self.__mb_thickness2 = 0
        self.__resolution = 1
        self.__sgraph = None
        self.__agraph = None

    #### Set functions area

    # In nm/voxel, by default 1
    def set_resolution(self, resolution=1):
        self.__resolution = resolution

    # Setting membrane thickness in nm
    # Lipid bilayer thick should be around [4.5, 6.1] nm
    def set_memb_thickness(self, nm=0):
        self.__mb_thickness = nm
        self.__mb_thickness2 = self.__mb_thickness * 0.5

    #### Get functions area

    def get_SkelGraph(self):
        return self.__sgraph

    def get_ArcGraph(self):
        return self.__agraph

    #### Functionality area

    @abstractmethod
    def build_skelgraph(self):
        raise NotImplementedError('build_skelgraph() (MembFactory). '
                                  'Abstract method, it requires an implementation.')

    # Build the ArcGraph from the current state of the objects, all input parameters
    # should be set before by the constructor and calling set methods
    # supdate: forces to rebuild the SkelGraph first, even if it was already built
    def build_arcgraph(self, supdate=False):

        if (self.__sgraph is None) or supdate:
            self.build_skelgraph()
        a_skel = vtk.vtkPolyData()
        self._MembFactory__agraph = graph.ArcGraph(a_skel)

        # Copying the vertices for having a list of processed ones
        vertices = self.__sgraph.get_vertices_list(core=True)

        # Loop for processing the vertices
        true_vertices = list()
        for v in vertices:
            n_neigh = v.get_num_neighbours()
            if n_neigh != 2:
                true_vertices.append(self.__sgraph.get_vertex(v.get_id()))

        # Loop for adding the arcs
        arcs_graph = list()
        for v in true_vertices:
            arcs_v = list()
            # Loop for arcs in a vertex
            for neigh in v.get_neighbours():
                # Track the arc
                prev = v
                current = neigh
                arc_list = list()
                while True:
                    n_neighs = current.get_num_neighbours()
                    if n_neighs == 2:
                        arc_list.append(current)
                        hold_neighs = current.get_neighbours()
                        if hold_neighs[0].get_id() == prev.get_id():
                            prev = current
                            current = hold_neighs[1]
                        else:
                            prev = current
                            current = hold_neighs[0]
                    else:
                        break
                # Add head and tail vertices to the arc
                arc_list.insert(0, v)
                arc_list.append(current)
                arcs_v.append(graph.Arc(arc_list))
            arcs_graph.append(arcs_v)

        # Building the new skeleton for the arc graph (only point attributes are added)
        a_skel.SetPoints(self.__skel.GetPoints())
        for i in range(self.__skel.GetPointData().GetNumberOfArrays()):
            a_skel.GetPointData().AddArray(self.__skel.GetPointData().GetArray(i))
        # Adding the vertices
        verts = vtk.vtkCellArray()
        for v in true_vertices:
            verts.InsertNextCell(1)
            verts.InsertCellPoint(v.get_id())
        # Adding the arcs
        lines = vtk.vtkCellArray()
        for e in arcs_graph:
            for a in e:
                nverts = a.get_nvertices()
                lines.InsertNextCell(nverts)
                for i in range(nverts):
                    lines.InsertCellPoint(a.get_vertex(i).get_id())
        a_skel.SetVerts(verts)
        a_skel.SetLines(lines)

        # Building the ArcGraph
        self._MembFactory__agraph = graph.ArcGraph(a_skel)
        self._MembFactory__agraph.update()
        self._MembFactory__agraph.update_geometry(self.__manifolds, self.__density)

#####################################################################################################
#   Class for building a Skelgraph and/or Arcgraph for studying the structures attached to a membrane
#
#
class MembSideSkelGraph(MembFactory):

    #### Constructor Area
    # By default vertex property STR_VERTEX_ID (id point in skeleton) and edges properties STR_EDGE_LENGTH and 'weight'
    # skel: vtk poly data where lines will be used for building a graph (from DisPerSe)
    # manifolds: numpy 3darray for extrating the geometry
    # density: numpy 3darray with the original tomogram
    # memb_seg: numpy 3darray with the segmentation of the membrane (1: bg, 0: fg)
    def __init__(self, skel, manifolds, density, memb_seg):
        super(MembSideSkelGraph, self).__init__(skel, manifolds, density, memb_seg)
        self.__memb_surf = None
        self.__max_dist = None
        self.set_side()
        self.__dist_filt = vtkClosestPointAlgorithm()

    #### Set functions area

    # Max distance for considering that a structure is attached to membrane
    # None is valid
    def set_max_distance(self, nm=None):
        self.__max_dist = nm

    # Set the side of the membrane which will be processed (STR_SIGN_P: positive (default), STR_SIGN_N: negative)
    def set_side(self, side=STR_SIGN_P):
        if side == STR_SIGN_N:
            self.__side = side
        else:
            self.__side = STR_SIGN_P

    #### Get functions area

    #### Functionality area

    # Build the SkelGraph from the current state of the objects, all input parameters
    # should be set before by the constructor and calling set methods
    def build_skelgraph(self):

        # Convert membrane segmentation into surface
        self.__memb_surf = disperse_io.gen_surface(self._MembFactory__memb_seg)
        # By default no name is assigned to normals scalar field
        nf_found = False
        for i in range(self.__memb_surf.GetPointData().GetNumberOfArrays()):
            array = self.__memb_surf.GetPointData().GetArray(i)
            if (array.GetNumberOfComponents() == 3) and (array.GetName() is None):
                array.SetName(STR_NFIELD_ARRAY_NAME)
                nf_found = True
                break
        if not nf_found:
            error_msg = 'No normal field found in membrane surface.'
            raise pexceptions.PySegInputWarning(expr='build_skelgraph (MembSideSkelGraph)', msg=error_msg)
        self.__dist_filt.SetInputData(self.__memb_surf)
        self.__dist_filt.set_normal_field(STR_NFIELD_ARRAY_NAME)
        self.__dist_filt.initialize()

        # Filter the skeleton with the border filter
        algorithm = vtkFilterSurfBorderAlgorithm()
        algorithm.SetInputData(self._MembFactory__skel)
        algorithm.set_surf(self.__memb_surf)
        algorithm.Execute()
        self._MembFactory__filt_skel = algorithm.GetOutput()

        # Build the initial graph
        self._MembFactory__sgraph = graph.SkelGraph(self._MembFactory__filt_skel)
        self._MembFactory__sgraph.update()

        # Refine the graph for keeping just the vertices within attachment area,
        # between membrane edge to max_distance
        # The first loop just label vertex
        self._MembFactory__sgraph.add_vertex_property(STR_ATYPE_ARRAY_NAME,
                                                      'int', 1, VERTEX_TYPE_NO_PROC)
        for v in self._MembFactory__sgraph.get_vertices_list():
            if v.get_property(STR_ATYPE_ARRAY_NAME) == VERTEX_TYPE_NO_PROC:
                v_hold_type = self.__is_in_area(v)
                if v_hold_type != VERTEX_TYPE_OUTER:
                    v_id = v.get_id()
                    neighbours = self._MembFactory__sgraph.get_vertex_neighbours(v_id)
                    is_anchor = False
                    for n in neighbours:
                        n_hold_type = self.__is_in_area(n)
                        if n_hold_type == VERTEX_TYPE_OUTER:
                            n.set_property(STR_ATYPE_ARRAY_NAME, VERTEX_TYPE_OUTER)
                            is_anchor = True
                    if is_anchor:
                        v.set_property(STR_ATYPE_ARRAY_NAME, v_hold_type)
                    else:
                        v.set_property(STR_ATYPE_ARRAY_NAME, VERTEX_TYPE_INNER)
                else:
                    v.set_property(STR_ATYPE_ARRAY_NAME, VERTEX_TYPE_OUTER)

        # Loop for deleting vertices labeled as outer
        for v in self._MembFactory__sgraph.get_vertices_list():
            if v.get_property(STR_ATYPE_ARRAY_NAME) == VERTEX_TYPE_OUTER:
                self._MembFactory__sgraph.remove_vertex(v.get_id())

        # Adding geometry for the refined SkelGraph
        self._MembFactory__sgraph.update_geometry(self._MembFactory__manifolds, self._MembFactory__density)

        # Filter vertices geometries for not crossing the membrane
        if self.__side == STR_SIGN_P:
            utils.graph_geom_distance_filter(self._MembFactory__sgraph, self.__dist_filt, 0, operator.gt)
        else:
            utils.graph_geom_distance_filter(self._MembFactory__sgraph, self.__dist_filt, 0, operator.lt)

        self._MembFactory__sgraph.find_critical_points()


    #### Internal functionality area

    # Check if a vertex is within attachment area and if it is closer to membrane than the external
    # region
    # Outputs: VERTEX_TYPE_OUTER, VERTEX_TYPE_ANCHOR_FRONT, VERTEX_TYPE_ANCHOR_BACK
    def __is_in_area(self, v):

        coord = v.get_coordinates()
        dist, dotp = self.__dist_filt.evaluate(coord[0], coord[1], coord[2])
        dist *= self._MembFactory__resolution
        if (dist >= self._MembFactory__mb_thickness2) and (dist <= self.__max_dist):
            if ((self.__side == STR_SIGN_P) and (dotp >= 0)) \
                    or ((self.__side == STR_SIGN_N) and (dotp <= 0)):
                if abs(dist-self._MembFactory__mb_thickness2) <= abs(dist-self.__max_dist):
                    return VERTEX_TYPE_ANCHOR_FRONT
                else:
                    return VERTEX_TYPE_ANCHOR_BACK

        return VERTEX_TYPE_OUTER


##########################################################################################
#   Class for building a Skelgraph and/or Arcgraph for studying the structures attached
#   between two membranes
#
#
class MembInterSkelGraph(MembFactory):

    #### Constructor Area
    # By default vertex property STR_VERTEX_ID (id point in skeleton) and edges properties STR_EDGE_LENGTH and 'weight'
    # skel: vtk poly data where lines will be used for building a graph (from DisPerSe)
    # manifolds: numpy 3darray for extrating the geometry
    # density: numpy 3darray with the original tomogram
    # memb_seg: numpy 3darray with the segmentation of the membrane ((1, 2): bg, 0: fg)
    def __init__(self, skel, manifolds, density, memb_seg):
        super(MembInterSkelGraph, self).__init__(skel, manifolds, density, memb_seg)
        self.__memb_surf_1 = None
        self.__memb_surf_2 = None
        self.__dist_filt_1 = vtkClosestPointAlgorithm()
        self.__dist_filt_2 = vtkClosestPointAlgorithm()

    #### Set functions area

    #### Get functions area

    #### Functionality area

    # Build the SkelGraph from the current state of the objects, all input parameters
    # should be set before by the constructor and calling set methods
    def build_skelgraph(self):

        # Convert membranes segmentation into surfaces
        self.__memb_surf_1 = disperse_io.gen_surface(self._MembFactory__memb_seg == 1)
        self.__memb_surf_2 = disperse_io.gen_surface(self._MembFactory__memb_seg == 2)
        # By default no name is assigned to normals scalar field
        nf_found = False
        for i in range(self.__memb_surf_1.GetPointData().GetNumberOfArrays()):
            array = self.__memb_surf_1.GetPointData().GetArray(i)
            if (array.GetNumberOfComponents() == 3) and (array.GetName() is None):
                array.SetName(STR_NFIELD_ARRAY_NAME)
                nf_found = True
                break
        if not nf_found:
            error_msg = 'No normal field found in membrane surface.'
            raise pexceptions.PySegInputWarning(expr='build_skelgraph (MembInterSkelGraph)', msg=error_msg)
        nf_found = False
        for i in range(self.__memb_surf_2.GetPointData().GetNumberOfArrays()):
            array = self.__memb_surf_2.GetPointData().GetArray(i)
            if (array.GetNumberOfComponents() == 3) and (array.GetName() is None):
                array.SetName(STR_NFIELD_ARRAY_NAME)
                nf_found = True
                break
        if not nf_found:
            error_msg = 'No normal field found in membrane surface.'
            raise pexceptions.PySegInputWarning(expr='build_skelgraph (MembInterSkelGraph)', msg=error_msg)

        # Membrane orientation
        self.__orient_membranes()

        # Distance filters initialization
        self.__dist_filt_1.SetInputData(self.__memb_surf_1)
        self.__dist_filt_1.set_normal_field(STR_NFIELD_ARRAY_NAME)
        self.__dist_filt_1.initialize()
        self.__dist_filt_2.SetInputData(self.__memb_surf_2)
        self.__dist_filt_2.set_normal_field(STR_NFIELD_ARRAY_NAME)
        self.__dist_filt_2.initialize()

        # Filter the skeleton with the border filter
        algorithm = vtkFilterSurfBorderAlgorithm()
        algorithm.SetInputData(self._MembFactory__skel)
        algorithm.set_surf(self.__memb_surf_1)
        algorithm.Execute()
        self._MembFactory__filt_skel = algorithm.GetOutput()
        del algorithm
        algorithm = vtkFilterSurfBorderAlgorithm()
        algorithm.SetInputData(self._MembFactory__filt_skel)
        algorithm.set_surf(self.__memb_surf_2)
        algorithm.Execute()
        self._MembFactory__filt_skel = algorithm.GetOutput()

        # Build the initial graph
        self._MembFactory__sgraph = graph.SkelGraph(self._MembFactory__filt_skel)
        self._MembFactory__sgraph.update()

        # Refine the graph for keeping just the vertices within attachment area,
        self._MembFactory__sgraph.add_vertex_property(STR_ATYPE_ARRAY_NAME,
                                                      'int', 1, VERTEX_TYPE_NO_PROC)
        for v in self._MembFactory__sgraph.get_vertices_list():
            if v.get_property(STR_ATYPE_ARRAY_NAME) == VERTEX_TYPE_NO_PROC:
                v_hold_type = self.__is_in_area(v)
                if v_hold_type != VERTEX_TYPE_OUTER:
                    v_id = v.get_id()
                    neighbours = self._MembFactory__sgraph.get_vertex_neighbours(v_id)
                    is_anchor = False
                    for n in neighbours:
                        n_hold_type = self.__is_in_area(n)
                        if n_hold_type == VERTEX_TYPE_OUTER:
                            n.set_property(STR_ATYPE_ARRAY_NAME, VERTEX_TYPE_OUTER)
                            is_anchor = True
                    if is_anchor:
                        v.set_property(STR_ATYPE_ARRAY_NAME, v_hold_type)
                    else:
                        v.set_property(STR_ATYPE_ARRAY_NAME, VERTEX_TYPE_INNER)
                else:
                    v.set_property(STR_ATYPE_ARRAY_NAME, VERTEX_TYPE_OUTER)

        # Loop for deleting vertices labeled as outer
        for v in self._MembFactory__sgraph.get_vertices_list():
            if v.get_property(STR_ATYPE_ARRAY_NAME) == VERTEX_TYPE_OUTER:
                self._MembFactory__sgraph.remove_vertex(v.get_id())

        # Adding geometry for the refined SkelGraph
        self._MembFactory__sgraph.update_geometry(self._MembFactory__manifolds, self._MembFactory__density)

        # Filter vertices geometries for not crossing the membrane
        # Normals of booth membranes have been oriented for being positive
        utils.graph_geom_distance_filter(self._MembFactory__sgraph, self.__dist_filt_1, 0, operator.gt)
        utils.graph_geom_distance_filter(self._MembFactory__sgraph, self.__dist_filt_2, 0, operator.gt)

        self._MembFactory__sgraph.find_critical_points()


    #### Internal functionality area

    # Check if a vertex is within attachment area and if it is closer to membrane than the external
    # region
    # Outputs: VERTEX_TYPE_OUTER, VERTEX_TYPE_ANCHOR_FRONT, VERTEX_TYPE_ANCHOR_BACK
    def __is_in_area(self, v):

        coord = v.get_coordinates()
        dist_1, dotp_1 = self.__dist_filt_1.evaluate(coord[0], coord[1], coord[2])
        dist_1 = math.fabs(dist_1 * self._MembFactory__resolution)
        dist_2, dotp_2 = self.__dist_filt_2.evaluate(coord[0], coord[1], coord[2])
        dist_2 = math.fabs(dist_2 * self._MembFactory__resolution)
        if (dotp_1 >= 0) and (dotp_2 >= 0) \
                and (dist_1 > self._MembFactory__mb_thickness2) \
                and (dist_2 > self._MembFactory__mb_thickness2):
            if dist_1 < dist_2:
                return VERTEX_TYPE_ANCHOR_FRONT
            else:
                return VERTEX_TYPE_ANCHOR_BACK
        else:
            return VERTEX_TYPE_OUTER


    # Orient the membranes normals for being opposed at the their closest point
    def __orient_membranes(self):

        orient_surf_against_surf(self.__memb_surf_1, self.__memb_surf_2)
        orient_surf_against_surf(self.__memb_surf_2, self.__memb_surf_1)


##########################################################################################
#   Class for building a Skelgraph and/or Arcgraph for studying the transmembrane
#   structures
#
#
class MembTransSkelGraph(MembFactory):

    #### Constructor Area
    # By default vertex property STR_VERTEX_ID (id point in skeleton) and edges properties STR_EDGE_LENGTH and 'weight'
    # skel: vtk poly data where lines will be used for building a graph (from DisPerSe)
    # manifolds: numpy 3darray for extrating the geometry
    # density: numpy 3darray with the original tomogram
    # memb_seg: numpy 3darray with the segmentation of the membrane (1: bg, 0: fg)
    def __init__(self, skel, manifolds, density, memb_seg):
        super(MembTransSkelGraph, self).__init__(skel, manifolds, density, memb_seg)
        self.__memb_surf = None
        self.__dist_filt = vtkClosestPointAlgorithm()

    #### Set functions area

    #### Get functions area

    #### Functionality area

    # Build the SkelGraph from the current state of the objects, all input parameters
    # should be set before by the constructor and calling set methods
    def build_skelgraph(self):

        # Convert membrane segmentation into surface
        self.__memb_surf = disperse_io.gen_surface(self._MembFactory__memb_seg)
        # By default no name is assigned to normals scalar field
        nf_found = False
        for i in range(self.__memb_surf.GetPointData().GetNumberOfArrays()):
            array = self.__memb_surf.GetPointData().GetArray(i)
            if (array.GetNumberOfComponents() == 3) and (array.GetName() is None):
                array.SetName(STR_NFIELD_ARRAY_NAME)
                nf_found = True
                break
        if not nf_found:
            error_msg = 'No normal field found in membrane surface.'
            raise pexceptions.PySegInputWarning(expr='build_skelgraph (MembSideSkelGraph)', msg=error_msg)
        self.__dist_filt.SetInputData(self.__memb_surf)
        self.__dist_filt.set_normal_field(STR_NFIELD_ARRAY_NAME)
        self.__dist_filt.initialize()

        # Filter the skeleton with the border filter
        algorithm = vtkFilterSurfBorderAlgorithm()
        algorithm.SetInputData(self._MembFactory__skel)
        algorithm.set_surf(self.__memb_surf)
        algorithm.Execute()
        self._MembFactory__filt_skel = algorithm.GetOutput()

        # Build the initial graph
        self._MembFactory__sgraph = graph.SkelGraph(self._MembFactory__filt_skel)
        self._MembFactory__sgraph.update()

        # Refine the graph for keeping just the vertices within attachment area,
        # between membrane edge to max_distance
        # The first loop just label vertex
        self._MembFactory__sgraph.add_vertex_property(STR_ATYPE_ARRAY_NAME,
                                                      'int', 1, VERTEX_TYPE_NO_PROC)
        for v in self._MembFactory__sgraph.get_vertices_list():
            if v.get_property(STR_ATYPE_ARRAY_NAME) == VERTEX_TYPE_NO_PROC:
                v_hold_type = self.__is_in_area(v)
                if v_hold_type != VERTEX_TYPE_OUTER:
                    v_id = v.get_id()
                    neighbours = self._MembFactory__sgraph.get_vertex_neighbours(v_id)
                    is_anchor = False
                    for n in neighbours:
                        n_hold_type = self.__is_in_area(n)
                        if n_hold_type == VERTEX_TYPE_OUTER:
                            n.set_property(STR_ATYPE_ARRAY_NAME, VERTEX_TYPE_OUTER)
                            is_anchor = True
                    if is_anchor:
                        v.set_property(STR_ATYPE_ARRAY_NAME, v_hold_type)
                    else:
                        v.set_property(STR_ATYPE_ARRAY_NAME, VERTEX_TYPE_INNER)
                else:
                    v.set_property(STR_ATYPE_ARRAY_NAME, VERTEX_TYPE_OUTER)

        # Loop for deleting vertices labeled as outer
        for v in self._MembFactory__sgraph.get_vertices_list():
            if v.get_property(STR_ATYPE_ARRAY_NAME) == VERTEX_TYPE_OUTER:
                self._MembFactory__sgraph.remove_vertex(v.get_id())

        # Adding geometry for the refined SkelGraph
        self._MembFactory__sgraph.update_geometry(self._MembFactory__manifolds, self._MembFactory__density)

        # Filter vertices geometries for not crossing the limits
        utils.graph_geom_distance_filter(self._MembFactory__sgraph, self.__dist_filt,
                                         self._MembFactory__mb_thickness2, operator.gt)
        # utils.graph_geom_distance_filter(self._MembFactory__sgraph, self.__dist_filt,
        #                                  -self._MembFactory__mb_thickness2, operator.lt)

        self._MembFactory__sgraph.find_critical_points()


    #### Internal functionality area

    # Check if a vertex is within attachment area and if it is closer to membrane than the external
    # region
    # Outputs: VERTEX_TYPE_OUTER, VERTEX_TYPE_ANCHOR_FRONT, VERTEX_TYPE_ANCHOR_BACK
    def __is_in_area(self, v):

        coord = v.get_coordinates()
        dist, dotp = self.__dist_filt.evaluate(coord[0], coord[1], coord[2])
        if dist < self._MembFactory__mb_thickness2:
            if dotp >= 0:
                return VERTEX_TYPE_ANCHOR_FRONT
            else:
                return VERTEX_TYPE_ANCHOR_BACK
        else:
            return VERTEX_TYPE_OUTER