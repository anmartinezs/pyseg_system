"""
Clases for resenting networks, collections of geometric objects with geometry and without topology

# Author: Antonio Martinez-Sanchez (Max Planck Institute for Biochemistry)
# Date: 16.10.14
"""

__author__ = 'martinez'

try:
    from graph import *
except:
    from pyseg.graph import *
try:
    from globals import *
except:
    from pyseg.globals import *
try:
    import disperse_io
except:
    import pyseg.disperse_io

#####################################################################################################
#   Class for defining a Filament
#
#
class Filament(object):

    #### Constructor Area
    # vertices: list of vertices
    # arc: list the arcs which connect the vertices
    def __init__(self, vertices, arcs, parse=False):
        self.__vertices = vertices
        self.__arcs = arcs
        self.__length = None
        self.__geometry = None
        self.__prop_names = list()
        self.__properties = list()
        self.__prop_types = list()
        if parse:
            if not self.__parse_arcs():
                error_msg = 'Filament parsing error.'
                raise pexceptions.PySegInputError(expr='__init__ (Filament)', msg=error_msg)

    #### Get functions area

    def get_num_vertices(self):
        return len(self.__vertices)

    def get_num_arcs(self):
        return len(self.__arcs)

    def get_arcs(self):
        return self.__arcs

    # Returns an ordered list with all the vertices which compounds the arcs of the filament
    def get_arc_vertices(self, parse=False):

        hold_verts = list()

        # Parsing
        if parse:
            try:
                for i in range(len(self.__vertices)-1):
                    if (self.__vertices[i].get_id() != self.__arcs[i].get_start_vertex().get_id()) \
                            or \
                        (self.__vertices[i+1].get_id() != self.__arcs[i].get_end_vertex().get_id()):
                        self.__arcs[i].reverse()
                        # raise Exception
                    for v in self.__arcs[i].get_vertices():
                        hold_verts.append(v)
            except Exception:
                error_msg = 'Parsing vertices error.'
                raise pexceptions.PySegTransitionError(expr='get_arc_vertices (Filament)',
                                                       msg=error_msg)

        else:
            for a in self.__arcs:
                for v in a.get_vertices():
                    hold_verts.append(v)

        return hold_verts

    def get_length(self):
        if self.__length is None:
            length = 0
            for a in self.__arcs:
                length += a.get_length()
            return length
        else:
            return self.__length

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

    def get_property_type(self, key, idx=None):
        if idx is None:
            idx = self.is_property(key)
        if idx is not None:
            return self.__prop_types[idx]
        else:
            return None

    def get_length(self):
        length = 0
        for a in self.__arcs:
            length += a.get_length()
        return length

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
    # type: in graph tool format
    # value: can be any object
    def add_property(self, key, type, value=None):
        self.__prop_names.append(key)
        self.__prop_types.append(type)
        self.__properties.append(value)

    # Remove property
    # If idx is not None the property is addressed according to its index in the list of property,
    # This access will be faster but is dangerous because properties can be added and removed
    # key: string key of the property, no property with name key no action is performed
    def remove_property(self, key, idx=None):
        if idx is None:
            idx = self.is_property(key)
        self.__prop_names.pop(idx)
        self.__prop_types.pop(idx)
        self.__properties.pop(idx)

    # An ArcGeometry object is generated because it is enough for representing the geometry of
    # a filament
    def add_geometry(self, manifold, density):

        # Checks that booth tomograms have the same size
        if manifold.shape != density.shape:
            error_msg = 'Manifold and Density tomograms have different size.'
            raise pexceptions.PySegInputError(expr='add_geometry (Filament)', msg=error_msg)

        # Get all vertices

        # Creates geometry
        arc_verts = self.get_arc_vertices()
        verts_coords = np.ndarray(shape=(len(arc_verts), 3), dtype=tuple)
        for i, v in enumerate(arc_verts):
            t = v.get_coordinates()
            verts_coords[i][0] = t[0]
            verts_coords[i][1] = t[1]
            verts_coords[i][2] = t[2]
        self.__geometry = ArcGeometry(verts_coords, manifold, density)

    # img: if img is a numpy array (None default) the filament is printed in the array, otherwise
    # an minimum numpy 3D array is created for holding the filament
    # th_den: number of sigmas above (+) or below vertex geometry density mean for thresholding,
    #         if None no threshold is applied
    def print_mask(self, img=None, th_den=None):

        if self.__geometry is None:
            return None

        if img is None:
            img = np.zeros(self.__geometry.get_size(), np.bool)
        else:
            offset = self.__geometry.get_offset()
            size = self.__geometry.get_size()
            if (img.shape[0] < (offset[0] + size[0])) or (img.shape[1] < (offset[1] + size[1])) or \
               (img.shape[2] < (offset[2] + size[2])):
                error_msg = 'The input image cannot contain this geometry.'
                raise pexceptions.PySegInputWarning(expr='print_mask (Filament)', msg=error_msg)

        self.__geometry.print_in_numpy(vol=img, lbl=1, th_den=th_den)

    # Returns either topology or geometry (default) bounds. If no geometry if found topology bounds
    # are returned.
    def get_bound(self, mode='geometry'):

        if (mode == 'geometry') and (self.__geometry is not None):
            return self.__geometry.get_bound()
        else:
            xl = np.finfo(np.float).max
            yl = xl
            zl = xl
            xh = np.finfo(np.float).min
            yh = xh
            zh = xh
            for v in self.get_arc_vertices():
                x, y, z = v.get_coordinates()
                if x < xl:
                    xl = x
                if y < yl:
                    yl = y
                if z < zl:
                    zl = z
                if x > xh:
                    xh = x
                if y > yh:
                    yh = y
                if z > zh:
                    zh = z
            return xl, yl, zl, xh, yh, zh

    #### Internal functionality area

    # Check that the list of arcs is connected and consecutive as well as vertices represents
    # the arcs extremes
    def __parse_arcs(self):

        if len(self.__vertices) != len(self.__arcs)+1:
            return False
        v_count = 0
        for i in range(len(self.__arcs)):
            start_v_id = self.__vertices[v_count].get_id()
            end_v_id = self.__vertices[v_count+1].get_id()
            start_a_id = self.__arcs[i].get_start_vertex().get_id()
            end_a_id = self.__arcs[i].get_end_vertex().get_id()
            if (start_v_id != start_a_id) or (end_v_id != end_a_id):
                self.__arcs[i].reverse()
                start_a_id = self.__arcs[i].get_start_vertex().get_id()
                end_a_id = self.__arcs[i].get_end_vertex().get_id()
                if (start_v_id != start_a_id) or (end_v_id != end_a_id):
                    return False
            v_count += 1

        return True

####################################################################################################
#   Base class for networks
#   TODO: IMPLEMENT WITH SHARED FUNCTIONALITY
#
#
class Network(object):
    pass

#####################################################################################################
#   Class for holding a network of filaments
#
#
class NetFilaments(Network):

    #### Constructor Area
    # filaments: list of filaments
    # arc_graph: ArcGraph object used for getting this network
    def __init__(self, filaments, arc_graph):
        self.__filaments = filaments
        self.__arc_graph = arc_graph
        self.__f_prop_info = PropInfo()
        self.__v_list = None
        self.__manifold = None
        self.__density = None

    #### Get methods area

    # Returns all vertices in the network in a list
    # update: if is False and the list has been already calculated the it is returned directly
    def get_vertices_list(self, update=True):

        if (not update) and (self.__v_list is not None):
            return self.__v_list

        # Get the whole list of vertices
        self.__v_list = list()
        for f in self.__filaments:
            for a in f.get_arcs():
                for v in a.get_vertices():
                    self.__v_list.append(v)

        return self.__v_list

    #### Functionality area

    # name: Value STR_FIL_ID for property name is not allowed and if the name already exists
    # character '_' is added
    # type: graph-tool data type
    # ncomp: number of components of the property (default 1)
    # def_value: initial value for the property (default 0), it could also be an array of values
    def add_property(self, name, type, ncomp=1, def_value=0):

        if name != STR_FIL_ID:
            if self.__f_prop_info.is_already(name) is not None:
                name += '_'
            for a in self.__filaments:
                a.add_property(key=name, value=def_value)
            self.__f_prop_info.add_prop(name, type, ncomp)
            return
        return NO_PROPERTY_INSERTED

    # Add the geometry to all filaments of the network
    def add_geometry(self, manifold, density):
        for f in self.__filaments:
            f.add_geometry(manifold, density)
        self.__manifold = manifold
        self.__density = density

    # Get the number the number of times which an arc is repeated in a list of filaments.
    # It is stored as STR_ARC_REDUNDANCY property in the arcs
    def arc_redundancy(self):

        # Add the property
        self.__arc_graph.add_arc_property(STR_ARC_REDUNDANCY, 'int', ncomp=1, def_value=0)

        # Get the whole list of arcs
        arcs = list()
        for f in self.__filaments:
            for a in f.get_arcs():
                arcs.append(a)

        # Set the value
        n_arcs = len(arcs)
        for i in range(n_arcs):
            for j in range(i+1, n_arcs):
                a_i = arcs[i]
                a_j = arcs[j]
                if arcs[i] is arcs[j]:
                    a_i.set_property(STR_ARC_REDUNDANCY,
                                         a_i.get_property(STR_ARC_REDUNDANCY) + 1)
                    a_j.set_property(STR_ARC_REDUNDANCY,
                                         a_j.get_property(STR_ARC_REDUNDANCY) + 1)

    # Get the number the number of times which an vertex is repeated in the network.
    # It is stored as STR_VER_REDUNDANCY property in the arcs
    def vertex_redundancy(self):

        # Get the whole list of vertices
        self.get_vertices_list(False)

        # Counting vertex redundancy
        lut = np.zeros(shape=self.__arc_graph.get_skel().GetNumberOfPoints(), dtype=np.int)
        for v in self.__v_list:
            lut[v.get_id()] += 1

        # Add property to ArcGraph vertex properties list
        self.__arc_graph.add_vertex_property(name=STR_VER_REDUNDANCY, type='int')

        # Setting vertex redundancy
        for v in self.__v_list:
            if v.is_property(STR_VER_REDUNDANCY) is None:
                v.add_property(key=STR_VER_REDUNDANCY, value=lut[v.get_id()])
            else:
                v.set_property(key=STR_VER_REDUNDANCY, value=lut[v.get_id()])


    # Analyze the network an return the list of arcs that compounds it without redundancy
    def get_arcs_nored(self):

        # Points lut
        skel = self.__arc_graph.get_skel()
        lut = np.zeros(shape=skel.GetNumberOfPoints(), dtype=np.bool)
        for f in self.__filaments:
            for a in f.get_arcs():
                for v in a.get_vertices():
                    lut[v.get_id()] = True

        # Add arcs whose points have not been added before
        arcs = list()
        for f in self.__filaments:
            for a in f.get_arcs():
                count = 0
                for v in a.get_vertices():
                    if lut[v.get_id()]:
                        count += 1
                if a.get_nvertices() > count:
                    arcs.append(a)

        return arcs

    # Returns array with the length of the filaments
    def get_fil_lengths(self):
        lengths = np.zeros(shape=len(self.__filaments), dtype=np.float)
        for i, f in enumerate(self.__filaments):
            lengths[i] = f.get_length()
        return lengths

    # Create a vtkPolyData for a list of filaments (all filaments/arcs must have the same properties)
    # mode: Either 'filament' (default) or 'arc'. It selects what kind of structure will represent
    # the cells
    def get_vtp(self, mode='filament'):

        # Initialization
        skel = self.__arc_graph.get_skel()
        poly = vtk.vtkPolyData()
        poly.SetPoints(skel.GetPoints())
        varrays = list()
        v_prop_info, e_prop_info = self.__arc_graph.get_prop_info()
        # Vertex STR_VERTEX_ID default property
        vid_array = vtk.vtkIntArray()
        vid_array.SetName(STR_VERTEX_ID)
        for i in range(v_prop_info.get_nprops()):
            array = disperse_io.TypesConverter.gt_to_vtk(v_prop_info.get_type(i))
            array.SetName(v_prop_info.get_key(i))
            array.SetNumberOfComponents(v_prop_info.get_ncomp(i))
            for j in range(poly.GetNumberOfPoints()):
                for k in range(v_prop_info.get_ncomp(i)):
                    array.InsertComponent(j, k, -1)
                vid_array.InsertComponent(j, 0, j)
            varrays.append(array)

        if mode == 'filament':
            # Initialization of the arc properties
            a_prop_info = self.__f_prop_info
            aarrays = list()
            for i in range(a_prop_info.get_nprops()):
                array = disperse_io.TypesConverter.gt_to_vtk(a_prop_info.get_type(i))
                array.SetName(a_prop_info.get_key(i))
                array.SetNumberOfComponents(a_prop_info.get_ncomp(i))
                aarrays.append(array)
            fid_array = vtk.vtkIntArray()
            fid_array.SetName(STR_FIL_ID)
        elif mode == 'arc':
            # Initialization of the arc properties
            a_prop_info = self.__arc_graph.get_arc_prop_info()
            aarrays = list()
            for i in range(a_prop_info.get_nprops()):
                array = disperse_io.TypesConverter.gt_to_vtk(a_prop_info.get_type(i))
                array.SetName(a_prop_info.get_key(i))
                array.SetNumberOfComponents(a_prop_info.get_ncomp(i))
                aarrays.append(array)
        else:
            error_msg = '%s invalid mode option' % mode
            raise pexceptions.PySegInputWarning(expr='get_vtp (Filament)', msg=error_msg)
        # Arc STR_EDGE_LENGTH default
        length_array = vtk.vtkDoubleArray()
        length_array.SetName(STR_EDGE_LENGTH)

        lines = vtk.vtkCellArray()
        if mode == 'filament':
            # Filaments topology
            for i, f in enumerate(self.__filaments):
                vertices = f.get_arc_vertices(parse=True)
                lines.InsertNextCell(len(vertices))
                for v in vertices:
                    point_id = v.get_id()
                    lines.InsertCellPoint(point_id)
                    for j in range(v_prop_info.get_nprops()):
                        t = v.get_property(v_prop_info.get_key(j))
                        if isinstance(t, tuple):
                            for k in range(v_prop_info.get_ncomp(j)):
                                varrays[j].SetComponent(point_id, k, t[k])
                        elif t is None:
                            varrays[j].SetComponent(point_id, 0, -1)
                        else:
                            varrays[j].SetComponent(point_id, 0, t)
                for j in range(self.__f_prop_info.get_nprops()):
                    t = f.get_property(self.__f_prop_info.get_key(j))
                    if isinstance(t, tuple):
                        for k in range(self.__f_prop_info.get_ncomp(j)):
                            aarrays[j].InsertComponent(i, k, t[k])
                    else:
                        aarrays[j].InsertComponent(i, 0, t)
                fid_array.InsertComponent(i, 0, i)
                length_array.InsertComponent(i, 0, f.get_length())
        else:
            # Filaments topology
            nlines = 0
            for f in self.__filaments:
                arcs = f.get_arcs()
                for i, a in enumerate(arcs):
                    vertices = a.get_vertices()
                    lines.InsertNextCell(len(vertices))
                    for v in vertices:
                        lines.InsertCellPoint(v.get_id())
                    for j in range(a_prop_info.get_nprops()):
                        t = a.get_property(a_prop_info.get_key(j))
                        if isinstance(t, tuple):
                            for k in range(a_prop_info.get_ncomp(j)):
                                aarrays[j].InsertComponent(nlines, k, t[k])
                        else:
                            aarrays[j].InsertComponent(nlines, 0, t)
                    length_array.InsertComponent(nlines, 0, a.get_length())
                    nlines += 1

        # Build final poly
        poly.SetLines(lines)
        poly.GetCellData().AddArray(length_array)
        if mode == 'filament':
            for i in range(self.__f_prop_info.get_nprops()):
                poly.GetCellData().AddArray(aarrays[i])
            poly.GetCellData().AddArray(fid_array)
        else:
            for i in range(a_prop_info.get_nprops()):
                poly.GetCellData().AddArray(aarrays[i])
        poly.GetPointData().AddArray(vid_array)
        for i in range(v_prop_info.get_nprops()):
            poly.GetPointData().AddArray(varrays[i])

        return poly

    # Returns either topology or geometry (default) bounds. If no geometry if found topology bounds
    # are returned.
    def get_bound(self, mode='geometry'):

        if (mode == 'geometry') and (self.__manifold is not None) and (self.__manifold is not None):
            hold_mode = 'geometry'
        else:
            hold_mode = 'topology'

        xhl = np.finfo(np.float).max
        yhl = xhl
        zhl = xhl
        xhh = np.finfo(np.float).min
        yhh = xhh
        zhh = xhh
        for f in self.__filaments:
            xl, yl, zl, xh, yh, zh = f.get_bound(hold_mode)
            if xl < xhl:
                xhl = xl
            if yl < yhl:
                yhl = yl
            if zl < zhl:
                zhl = zl
            if xh > xhh:
                xhh = xh
            if yh > yhh:
                yhh = yh
            if zh > zhh:
                zhh = zh

        return xhl, yhl, zhl, xhh, yhh, zhh


    # img: if img is a numpy array (None default) the filament is printed in the array, otherwise
    # an minimum numpy 3D array is created for holding the filament
    # th_den: number of sigmas above (+) or below vertex geometry density mean for thresholding,
    #         if None no threshold is applied
    # manifolds, density: it these images are passed the method directly print the mask without
    # requiring adding a geometry to the network previously
    # Return: a new image if img is None, the input printed if not and None if there is no geometry
    def print_mask(self, img=None, th_den=None, manifold=None, density=None):

        if (manifold is None) and (density is None):

            xl, yl, zl, xh, yh, zh = self.get_bound(mode='geometry')
            sz = (int(np.ceil(xh-xl)), int(np.ceil(yh-yl)), int(np.ceil(zh-zl)))
            if img is None:
                img = np.zeros(shape=sz, dtype=np.bool)
            else:
                if (img.shape[0] < xh) or (img.shape[1] < yh) or (img.shape[2] < zh):
                    error_msg = 'The input image cannot contain this geometry.'
                    raise pexceptions.PySegInputWarning(expr='print_mask (Filament)', msg=error_msg)

            if (self.__manifold is None) or (self.__density is None):
                return None
            # Print form geometries
            for f in self.__filaments:
                f.print_mask(img, th_den)

        else:

            if img is None:
                img = np.zeros(shape=manifold.shape, dtype=np.bool)
            else:
                if (img.shape != manifold.shape) or (img.shape != density.shape):
                    error_msg = 'The input image must be equal than manifold and density files.'
                    raise pexceptions.PySegInputWarning(expr='print_mask (NetFilament)',
                                                        msg=error_msg)

            # Lut for keeping already processed points
            lut = np.ones(shape=self.__arc_graph.get_skel().GetNumberOfPoints(),
                          dtype=np.bool)
            for f in self.__filaments:
                for v in f.get_arc_vertices():
                    point_id = v.get_id()
                    if lut[point_id]:
                        coord = v.get_coordinates()
                        lbl = manifold[int(round(coord[0])),
                                       int(round(coord[1])),
                                       int(round(coord[2]))]
                        idx = np.where(manifold == lbl)
                        xmin = np.min(idx[0])
                        ymin = np.min(idx[1])
                        zmin = np.min(idx[2])
                        xmax = np.max(idx[0]) + 1
                        ymax = np.max(idx[1]) + 1
                        zmax = np.max(idx[2]) + 1
                        densities = density[idx]
                        threshold = densities.mean() - th_den*densities.std()
                        subvol = density[xmin:xmax, ymin:ymax, zmin:zmax]
                        idx2 = np.where(subvol < threshold)
                        idx2 = list(idx2)
                        idx2[0].setflags(write=True)
                        idx2[0] = idx2[0] + xmin
                        idx2[1].setflags(write=True)
                        idx2[1] = idx2[1] + ymin
                        idx2[2].setflags(write=True)
                        idx2[2] = idx2[2] + zmin
                        img[idx2] = True
                        lut[point_id] = False

        return img

#####################################################################################################
#   Class for holding a network of ArcGraphs
#
#
class NetArcGraphs(Network):

    #### Constructor Area
    # arc_graphs: list of ArcGraph objects, all of them must share their skel
    def __init__(self, arc_graphs):
        self.__arc_graphs = arc_graphs
        self.__skel = None
        if len(arc_graphs) > 0:
            self.__skel = arc_graphs[0].get_skel()
            for i in range(1, len(arc_graphs)):
                if self.__skel is not arc_graphs[i].get_skel():
                    error_msg = 'The skel must be the same object in all ArcGraphs.'
                    raise pexceptions.PySegInputWarning(expr='print_mask (NetArcGraphs)',
                                                        msg=error_msg)
        self.__v_prop_info = PropInfo()
        self.__e_prop_info = PropInfo()
        self.__a_prop_info = PropInfo()
        self.__g_prop_info = PropInfo()
        self.__join_agraphs_props()
        self.__manifold = None
        self.__density = None

    #### Functionality area

    # Add a new property or reset an already existing property to 'def_value'
    # TODO: modify the rest of add_XXX_property for having an equivalent behaviour
    def add_graph_property(self, name, type, ncomp=1, def_value=0):

        if self.__g_prop_info.is_already(name) is None:
            for g in self.__arc_graphs:
                g.add_property(key=name, value=def_value)
            self.__g_prop_info.add_prop(name, type, ncomp)
        else:
            for a in self.__arc_graphs:
                a.add_property(key=name, value=def_value)

    # Add the geometry to all filaments of the network
    def add_geometry(self, manifold, density):
        for f in self.__filaments:
            f.add_geometry(manifold, density)
        self.__manifold = manifold
        self.__density = density

    # Returns either topology or geometry (default) bounds. If no geometry if found topology bounds
    # are returned.
    def get_bound(self, mode='geometry'):

        if (mode == 'geometry') and (self.__manifold is not None) and (self.__manifold is not None):
            hold_mode = 'geometry'
        else:
            hold_mode = 'topology'

        xhl = np.finfo(np.float).max
        yhl = xhl
        zhl = xhl
        xhh = np.finfo(np.float).min
        yhh = xhh
        zhh = xhh
        for f in self.__filaments:
            xl, yl, zl, xh, yh, zh = f.get_bound(hold_mode)
            if xl < xhl:
                xhl = xl
            if yl < yhl:
                yhl = yl
            if zl < zhl:
                zhl = zl
            if xh > xhh:
                xhh = xh
            if yh > yhh:
                yhh = yh
            if zh > zhh:
                zhh = zh

        return xhl, yhl, zhl, xhh, yhh, zhh

    # Create a vtkPolyData for a list of filaments (all filaments/arcs must have the same properties)
    # mode: Either 'graph' (default) or 'arc'. It selects what kind of structure will represent
    # the cells
    def get_vtp(self, mode='graph'):

        # Initialization
        skel = self.__skel
        poly = vtk.vtkPolyData()
        poly.SetPoints(skel.GetPoints())
        varrays = list()
        v_prop_info = self.__v_prop_info
        # Vertex STR_VERTEX_ID default property
        vid_array = vtk.vtkIntArray()
        vid_array.SetName(STR_VERTEX_ID)
        for i in range(v_prop_info.get_nprops()):
            array = disperse_io.TypesConverter.gt_to_vtk(v_prop_info.get_type(i))
            array.SetName(v_prop_info.get_key(i))
            array.SetNumberOfComponents(v_prop_info.get_ncomp(i))
            for j in range(poly.GetNumberOfPoints()):
                for k in range(v_prop_info.get_ncomp(i)):
                    array.InsertComponent(j, k, -1)
                vid_array.InsertComponent(j, 0, j)
            varrays.append(array)

        if mode == 'graph':
            # Initialization of the arc properties
            a_prop_info = self.__g_prop_info
            aarrays = list()
            for i in range(a_prop_info.get_nprops()):
                array = disperse_io.TypesConverter.gt_to_vtk(a_prop_info.get_type(i))
                array.SetName(a_prop_info.get_key(i))
                array.SetNumberOfComponents(a_prop_info.get_ncomp(i))
                aarrays.append(array)
            fid_array = vtk.vtkIntArray()
            fid_array.SetName(STR_ARCGRAPH_ID)
        elif mode == 'arc':
            # Initialization of the arc properties
            a_prop_info = self.__a_prop_info
            aarrays = list()
            for i in range(a_prop_info.get_nprops()):
                array = disperse_io.TypesConverter.gt_to_vtk(a_prop_info.get_type(i))
                array.SetName(a_prop_info.get_key(i))
                array.SetNumberOfComponents(a_prop_info.get_ncomp(i))
                aarrays.append(array)
        else:
            error_msg = '%s invalid mode option' % mode
            raise pexceptions.PySegInputWarning(expr='get_vtp (NetArcGraphs)', msg=error_msg)
        # Arc STR_EDGE_LENGTH default
        length_array = vtk.vtkDoubleArray()
        length_array.SetName(STR_EDGE_LENGTH)

        lines = vtk.vtkCellArray()
        if mode == 'graph':
            # Filaments topology
            for i, f in enumerate(self.__arc_graphs):
                vertices = f.get_arc_vertices(parse=True)
                lines.InsertNextCell(len(vertices))
                for v in vertices:
                    point_id = v.get_id()
                    lines.InsertCellPoint(point_id)
                    for j in range(v_prop_info.get_nprops()):
                        t = v.get_property(v_prop_info.get_key(j))
                        if isinstance(t, tuple):
                            for k in range(v_prop_info.get_ncomp(j)):
                                varrays[j].SetComponent(point_id, k, t[k])
                        elif t is None:
                            varrays[j].SetComponent(point_id, 0, -1)
                        else:
                            varrays[j].SetComponent(point_id, 0, t)
                for j in range(a_prop_info.get_nprops()):
                    t = f.get_property(a_prop_info.get_key(j))
                    if isinstance(t, tuple):
                        for k in range(a_prop_info.get_ncomp(j)):
                            aarrays[j].InsertComponent(i, k, t[k])
                    else:
                        aarrays[j].InsertComponent(i, 0, t)
                fid_array.InsertComponent(i, 0, i)
                length_array.InsertComponent(i, 0, f.get_length())
        else:
            # Filaments topology
            nlines = 0
            for f in self.__arc_graphs:
                arcs = f.get_arcs()
                for i, a in enumerate(arcs):
                    vertices = a.get_vertices()
                    lines.InsertNextCell(len(vertices))
                    for v in vertices:
                        lines.InsertCellPoint(v.get_id())
                    for j in range(a_prop_info.get_nprops()):
                        t = a.get_property(a_prop_info.get_key(j))
                        if isinstance(t, tuple):
                            for k in range(a_prop_info.get_ncomp(j)):
                                aarrays[j].InsertComponent(nlines, k, t[k])
                        else:
                            aarrays[j].InsertComponent(nlines, 0, t)
                    length_array.InsertComponent(nlines, 0, a.get_length())
                    nlines += 1

        # Build final poly
        poly.SetLines(lines)
        poly.GetCellData().AddArray(length_array)
        if mode == 'filament':
            for i in range(a_prop_info.get_nprops()):
                poly.GetCellData().AddArray(aarrays[i])
            poly.GetCellData().AddArray(fid_array)
        else:
            for i in range(a_prop_info.get_nprops()):
                poly.GetCellData().AddArray(aarrays[i])
        poly.GetPointData().AddArray(vid_array)
        for i in range(v_prop_info.get_nprops()):
            poly.GetPointData().AddArray(varrays[i])

        return poly

    # Segment graphs geometry in a numpy array
    # img: if img is a numpy array (None default) the filament is printed in the array, otherwise
    # an minimum numpy 3D array is created for holding the filament
    # th_den: number of sigmas above (+) or below vertex geometry density mean for thresholding,
    #         if None no threshold is applied
    # manifolds, density: it these images are passed the method directly print the mask without
    # requiring adding a geometry to the network previously
    # mode: 'graph' (default), 'arcs' or 'vertices'.
    # Return: a new image if img is None, the input printed if not and None if there is no geometry
    def print_mask(self, img=None, th_den=None, manifold=None, density=None, mode='graph'):

        if (manifold is None) and (density is None):

            xl, yl, zl, xh, yh, zh = self.get_bound(mode='geometry')
            sz = (int(np.ceil(xh-xl)), int(np.ceil(yh-yl)), int(np.ceil(zh-zl)))
            if img is None:
                img = np.zeros(shape=sz, dtype=np.int)
            else:
                if (img.shape[0] < xh) or (img.shape[1] < yh) or (img.shape[2] < zh):
                    error_msg = 'The input image cannot contain this geometry.'
                    raise pexceptions.PySegInputWarning(expr='print_mask (Filament)', msg=error_msg)

            if (self.__manifold is None) or (self.__density is None):
                return None

            # Print from geometries
            if mode == 'graph':
                for i, g in enumerate(self.__arc_graphs):
                    for a in g.get_arc_list():
                        for v in a.get_vertices():
                            v.get_geometry().print_in_numpy(vol=img, lbl=i, th_den=th_den)
            elif mode == 'arcs':
                for g in self.__arc_graphs:
                    for i, a in enumerate(g.get_arc_list()):
                        for v in a.get_vertices():
                            v.get_geometry().print_in_numpy(vol=img, lbl=i, th_den=th_den)
            if mode == 'vertices':
                for g in self.__arc_graphs:
                    for i, v in enumerate(g.get_vertices_list()):
                        v.get_geometry().print_in_numpy(vol=img, lbl=i, th_den=th_den)

        else:

            if img is None:
                img = np.zeros(shape=manifold.shape, dtype=np.int)
            else:
                if (img.shape != manifold.shape) or (img.shape != density.shape):
                    error_msg = 'The input image must be equal than manifold and density files.'
                    raise pexceptions.PySegInputWarning(expr='print_mask (NetFilament)',
                                                        msg=error_msg)

            # Get all points which must be printed and their label
            points = Hash()
            if mode == 'graph':
                for i, g in enumerate(self.__arc_graphs):
                    for a in g.get_arc_list():
                        for v in a.get_vertices():
                            points.append(key=v, value=i)
            elif mode == 'arcs':
                for g in self.__arc_graphs:
                    for i, a in enumerate(g.get_arc_list()):
                        for v in a.get_vertices():
                            points.append(key=v, value=i)
            if mode == 'vertices':
                for g in self.__arc_graphs:
                    for i, v in enumerate(g.get_vertices_list()):
                        points.append(key=v, value=i)

            # Lut for keeping already processed points
            lut = np.ones(shape=self.__skel.GetNumberOfPoints(), dtype=np.bool)
            for i in range(len(points)):
                v = points.get_key(i)
                point_id = v.get_id()
                if lut[point_id]:
                    coord = v.get_coordinates()
                    lbl = manifold[int(round(coord[0])),
                                   int(round(coord[1])),
                                   int(round(coord[2]))]
                    idx = np.where(manifold == lbl)
                    xmin = np.min(idx[0])
                    ymin = np.min(idx[1])
                    zmin = np.min(idx[2])
                    xmax = np.max(idx[0]) + 1
                    ymax = np.max(idx[1]) + 1
                    zmax = np.max(idx[2]) + 1
                    densities = density[idx]
                    threshold = densities.mean() - th_den*densities.std()
                    subvol = density[xmin:xmax, ymin:ymax, zmin:zmax]
                    idx2 = np.where(subvol < threshold)
                    idx2 = list(idx2)
                    idx2[0].setflags(write=True)
                    idx2[0] = idx2[0] + xmin
                    idx2[1].setflags(write=True)
                    idx2[1] = idx2[1] + ymin
                    idx2[2].setflags(write=True)
                    idx2[2] = idx2[2] + zmin
                    img[idx2] = points.get_value(i)
                    lut[point_id] = False

        return img

    def compute_diameters(self, resolution=1):
        self.add_graph_property(STR_GRAPH_DIAM, disperse_io.TypesConverter().numpy_to_gt(np.float),
                                ncomp=1, def_value=0)
        for g in self.__arc_graphs:
            g.compute_diameter(type='topo', resolution=resolution)

    #### Internal functionality area

    # Keep in __ag_prop_info the properties shared for all vertices
    def __join_agraphs_props(self):

        if len(self.__arc_graphs) <= 0:
            return

        # Initial properties
        v_prop_info, e_prop_info = self.__arc_graphs[0].get_prop_info()
        a_prop_info = self.__arc_graphs[0].get_arc_prop_info()
        for i in range(v_prop_info.get_nprops):
            key = v_prop_info.get_key(i)
            if self.__v_prop_info.is_already(key) is not None:
                self.__v_prop_info.add_prop(key=key, type=v_prop_info.get_type(i),
                                                ncomp=v_prop_info.get_ncomp(i))
        for i in range(e_prop_info.get_nprops):
            key = e_prop_info.get_key(i)
            if self.__e_prop_info.is_already(key) is not None:
                self.__e_prop_info.add_prop(key=key, type=e_prop_info.get_type(i),
                                                ncomp=e_prop_info.get_ncomp(i))
        for i in range(a_prop_info.get_nprops):
            key = a_prop_info.get_key(i)
            if self.__a_prop_info.is_already(key) is not None:
                self.__a_prop_info.add_prop(key=key, type=a_prop_info.get_type(i),
                                            ncomp=a_prop_info.get_ncomp(i))

        # Purge props not included in all graphs
        del_names = list()
        for i in range(self.__v_prop_info.get_nprops()):
            key = self.__v_prop_info.get_key(i)
            for g in self.__arc_graphs:
                v_prop_info, _ = g.get_prop_info()
                if v_prop_info.is_already(key) is None:
                    del_names.append(key)
                    break
        for name in del_names:
            self.__v_prop_info.remove_prop(key=name)
        del_names = list()
        for i in range(self.__e_prop_info.get_nprops()):
            key = self.__e_prop_info.get_key(i)
            for g in self.__arc_graphs:
                _, e_prop_info = g.get_prop_info()
                if e_prop_info.is_already(key) is None:
                    del_names.append(key)
                    break
        for name in del_names:
            self.__e_prop_info.remove_prop(key=name)
        del_names = list()
        for i in range(self.__a_prop_info.get_nprops()):
            key = self.__a_prop_info.get_key(i)
            for g in self.__arc_graphs:
                a_prop_info = g.get_arc_prop_info()
                if a_prop_info.is_already(key) is None:
                    del_names.append(key)
                    break
        for name in del_names:
            self.__a_prop_info.remove_prop(key=name)

