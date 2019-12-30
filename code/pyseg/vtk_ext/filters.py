__author__ = 'martinez'

##########################################################################################
#
#   Set of classes which follow the VTK filter interface and extend the functionality of VTK
#
##########################################################################################

import vtk
import numpy as np
import math

# Class for containing a set of edges
class SetEdges(object):

    def __init__(self):
        self.__origins = list()
        self.__targets = list()
        self.__faces = list()

    # Add and new edges [orign, target] into the lists and initialises the faces number to 1.
    # If this edge already exist no new one is added, just the faces number is incremented
    def add_edge(self, origin, target):

        nedges = len(self.__origins)

        if nedges > 0:
            for i in range(len(self.__origins)):
                if ((self.__origins[i] == origin) and (self.__targets[i] == target)) or ((self.__origins[i] == target) and (self.__targets[i] == origin)):
                    self.__faces[i] += 1
                    return

        self.__origins.append(origin)
        self.__targets.append(target)
        self.__faces.append(1)

    # Returns a Numpy matrix, rows are the edges and columns origin, target and faces
    def print_numpy(self):

        return np.array([np.asarray(self.__origins), np.asarray(self.__targets), np.asarray(self.__faces)])

# Count faces for every line in a polygonal mesh
def count_faces(mesh):

    polys_cells = mesh.GetPolys()
    polys = polys_cells.GetData()

    # Get faces number
    edges = SetEdges()
    poly_id = 0
    for i in range(polys_cells.GetNumberOfCells()):
        npoints = polys.GetValue(poly_id)
        if npoints >= 2:
            start = poly_id + 2
            end = poly_id + npoints + 1
            for j in range(start, end):
                edges.add_edge(polys.GetValue(j-1), polys.GetValue(j))
            edges.add_edge(polys.GetValue(start-1), polys.GetValue(end-1))
        poly_id = poly_id + npoints + 1

    # Build output poly
    faces = vtk.vtkIntArray()
    faces.SetName('faces')
    lines = vtk.vtkCellArray()
    table = edges.print_numpy()
    for i in range(table.shape[1]):
        lines.InsertNextCell(2)
        lines.InsertCellPoint(table[0][i])
        lines.InsertCellPoint(table[1][i])
        faces.InsertNextValue(table[2][i])
    out_polys = vtk.vtkPolyData()
    out_polys.SetPoints(mesh.GetPoints())
    out_polys.SetLines(lines)
    out_polys.GetCellData().AddArray(faces)

    return out_polys

##########################################################################################
#
# Filter which gets the border of a open surface (polygonal mesh)
# Input -> Polydata Mesh
# Output -> Polydata expresed as a set of lines with the number of faces for every one in
#           attributes, it will be 1 for border lines and 2 inner lines.
#           TODO: Add the posibility of return the border as a closed curves
#
class vtkSurfBorderAlgorithm(vtk.vtkPolyDataAlgorithm):

    # Abstract method for filtering task
    def Execute(self):

        self.SetOutput(count_faces(self.GetInput()))


##########################################################################################
#
# From an input poly data set the closest cell point in another poly data
# Input -> The poly data where measure are taken and the reference poly data (see set_reference())
# Output -> The same poly but and attributes with the cell id, distance and cell mass centers are added
#
class vtkClosestCellAlgorithm(vtk.vtkPolyDataAlgorithm):

    def __init__(self):
        # super(vtkClosestCellAlgorithm, self).__init__()
        self.__reference = None
        self.__copy = True

    def set_reference(self, poly):
        self.__reference = poly

    # If switch is True it adds all input attributes from the input poly to the output
    def set_copy_input_attributes(self, switch=True):
        self.__copy = switch

    # Abstract method for filtering task
    def Execute(self):

        reference = self.__reference

        if reference is not None:

            # VTK arrays
            input_polys = self.GetInput()
            massca = vtk.vtkFloatArray()
            massca.SetName('ccell_center')
            dista = vtk.vtkFloatArray()
            dista.SetName('ccell_dist')
            cellida = vtk.vtkIntArray()
            cellida.SetName('ccell_id')
            massca.SetNumberOfComponents(3)

            # Array for increasing processing speed
            rncells = reference.GetNumberOfCells()
            cpolyp = np.zeros([rncells, 3], dtype=np.float)

            # Get center of mass for all cells
            for i in range(rncells):
                cell = vtk.vtkGenericCell()
                reference.GetCell(i, cell)
                npoints = cell.GetNumberOfPoints()
                pts = cell.GetPoints()
                coords = np.zeros([npoints, 3], dtype=np.float)
                for k in range(npoints):
                    coords[k][:] = pts.GetPoint(k)
                mcoord = np.mean(coords, 0)
                cpolyp[i][:] = mcoord
                del cell
                del coords

            # Get the sortest cell in reference for every cell in input
            for i in range(input_polys.GetNumberOfCells()):
                # Get center of mass of the current input cell
                cell = vtk.vtkGenericCell()
                input_polys.GetCell(i, cell)
                npoints = cell.GetNumberOfPoints()
                pts = cell.GetPoints()
                coords = np.zeros([npoints, 3], dtype=np.float)
                for k in range(npoints):
                    coords[k][:] = pts.GetPoint(k)
                mcoord = np.mean(coords, 0)
                # Measuring the distance (Euclidean) with all the reference cells
                darr = np.sqrt(np.sum(np.square(cpolyp-mcoord), axis=1))
                amin = darr.argmin()
                # Store the results as attribute
                massca.InsertTuple3(i, cpolyp[amin][0], cpolyp[amin][1], cpolyp[amin][2])
                dista.InsertTuple1(i, darr[amin])
                cellida.InsertTuple1(i, amin)
                del cell
                del coords

            # Building the output poly data
            output = vtk.vtkPolyData()
            output.SetVerts(input_polys.GetVerts())
            output.SetLines(input_polys.GetLines())
            output.SetStrips(input_polys.GetStrips())
            output.SetPolys(input_polys.GetPolys())
            output.SetPoints(input_polys.GetPoints())

            # Adding preivous attributes
            if self.__copy:
                copy_alg = vtkAddAttributesAlgorithm()
                copy_alg.SetInputData(output)
                copy_alg.set_reference(input_polys)
                copy_alg.Execute()

            # Adding new attributes
            output.GetCellData().AddArray(massca)
            output.GetCellData().AddArray(dista)
            output.GetCellData().AddArray(cellida)
            self.SetOutput(output)

        else:
            raise NameError('set_reference() must be succesfully called before Execute() in a vtkClosestCellAlgorithm object.')

##########################################################################################
#
# Add all attributes from the reference poly data to the input. Both must have the save geometry
# and topology
# Input -> The poly data where attributes will be added and the reference poly data (see set_reference())
#          from which the attributes will be taken
#
class vtkAddAttributesAlgorithm(vtk.vtkPolyDataAlgorithm):

    def __init__(self):
        self.__reference = None

    def set_reference(self, poly):
        self.__reference = poly

    def Execute(self):

        reference = self.__reference
        if reference is not None:

            # Check that geometry and topology are equals in input and reference
            input_poly = self.GetInput()
            npoints = input_poly.GetNumberOfPoints()
            ncells = input_poly.GetNumberOfCells()
            if (npoints != reference.GetNumberOfPoints()) or (ncells != reference.GetNumberOfCells()):
                raise NameError('Input and reference poly data must have the same geometry and topology in a vtkAddAttributesAlgorithm object.')

            # Copy point attributes
            point_data = input_poly.GetPointData()
            rpoint_data = reference.GetPointData()
            for i in range(rpoint_data.GetNumberOfArrays()):
                point_data.AddArray(rpoint_data.GetArray(i))

            # Copy cell attributes
            cell_data = input_poly.GetCellData()
            rcell_data = reference.GetCellData()
            for i in range(rcell_data.GetNumberOfArrays()):
                cell_data.AddArray(rcell_data.GetArray(i))

        else:
            raise NameError('set_reference() must be succesfully called before Execute() in a vtkAddAttributesAlgorithm object.')


##########################################################################################
#
# Class for filtering all cells closer to a surface border
# Input -> An input data and a reference surface (set_surf())
# Output -> The input poly data without the cells whose closet surface's cell is a in surface border
#
class vtkFilterSurfBorderAlgorithm(vtk.vtkPolyDataAlgorithm):

    def __init__(self):
        self.__surf = None
        self.__border = None
        self.__keep_geom = True

    def set_surf(self, poly):
        self.__surf = poly
        self.__border = None

    def set_keep_geom(self, keep=True):
        self.__keep_geom = keep

    def Execute(self):

        if self.__surf is not None:

            # Extracting the surface border
            if self.__border is None:
                border_filter = vtkSurfBorderAlgorithm()
                border_filter.SetInputData(self.__surf)
                border_filter.Execute()
                self.__border = border_filter.GetOutput()

            # Getting the closest surf cell
            closest_filter = vtkClosestCellAlgorithm()
            closest_filter.set_reference(self.__border)
            closest_filter.SetInputData(self.GetInput())
            closest_filter.Execute()
            closest_poly = closest_filter.GetOutput()

            # Getting the attributes
            attp_arrays = list()
            attp_out_arrays = list()
            if self.__keep_geom:
                for i in range(closest_poly.GetPointData().GetNumberOfArrays()):
                    attp_out_arrays.append(closest_poly.GetPointData().GetArray(i))
            else:
                for i in range(closest_poly.GetPointData().GetNumberOfArrays()):
                    array = closest_poly.GetPointData().GetArray(i)
                    klass = array.__class__
                    arrayb = klass()
                    arrayb.SetName(array.GetName())
                    arrayb.SetNumberOfComponents(array.GetNumberOfComponents())
                    attp_arrays.append(array)
                    attp_out_arrays.append(arrayb)
            attc_arrays = list()
            attc_out_arrays = list()
            for i in range(closest_poly.GetCellData().GetNumberOfArrays()):
                array = closest_poly.GetCellData().GetArray(i)
                klass = array.__class__
                arrayb = klass()
                arrayb.SetName(array.GetName())
                arrayb.SetNumberOfComponents(array.GetNumberOfComponents())
                attc_arrays.append(array)
                attc_out_arrays.append(arrayb)

            # Keeping cells whose closest surface cell is not in border
            for i in range(closest_poly.GetCellData().GetNumberOfArrays()):
                if closest_poly.GetCellData().GetArrayName(i) == 'ccell_id':
                    ccell_array = closest_poly.GetCellData().GetArray(i)
                    break
            for i in range(self.__border.GetCellData().GetNumberOfArrays()):
                if self.__border.GetCellData().GetArrayName(i) == 'faces':
                    faces_array = self.__border.GetCellData().GetArray(i)
                    break

            ncatt = len(attc_arrays)
            if self.__keep_geom:
                npatt = len(attp_out_arrays)
                # Filter the cell but keeping the original points (geometry)
                out_poly = vtk.vtkPolyData()
                out_poly.SetPoints(closest_poly.GetPoints())
                out_poly.Allocate()
                # Creating the topoogy
                for i in range(closest_poly.GetNumberOfCells()):
                    if faces_array.GetTuple1(int(ccell_array.GetTuple1(i))) >= 2:
                        cell = vtk.vtkGenericCell()
                        closest_poly.GetCell(i, cell)
                        npoints = cell.GetNumberOfPoints()
                        ids = vtk.vtkIdList()
                        for j in range(npoints):
                            ids.InsertNextId(cell.GetPointId(j))
                        out_poly.InsertNextCell(cell.GetCellType(), ids)
                        for k in range(ncatt):
                            attc_out_arrays[k].InsertNextTuple(attc_arrays[k].GetTuple(i))

            # In keep geometry is disabled
            else:
                npatt = len(attp_arrays)
                points_f = np.zeros(closest_poly.GetNumberOfPoints(), dtype=np.bool)
                cells_f = list()
                cells_f_att = list()
                for i in range(ccell_array.GetNumberOfTuples()):
                    if faces_array.GetTuple1(int(ccell_array.GetTuple1(i))) >= 2:
                        cell = vtk.vtkGenericCell()
                        closest_poly.GetCell(i, cell)
                        npoints = cell.GetNumberOfPoints()
                        for k in range(npoints):
                            points_f[cell.GetPointId(k)] = True
                        cells_f.append(cell)
                        cells_f_tuples = list()
                        for k in range(ncatt):
                            cells_f_tuples.append(attc_arrays[k].GetTuple(i))
                        cells_f_att.append(cells_f_tuples)

                # Creating Geometry
                count = 0
                lut = (-1) * np.ones(points_f.size, dtype=np.int)
                out_points = vtk.vtkPoints()
                for i in range(lut.size):
                    if points_f[i]:
                        x, y, z = closest_poly.GetPoint(i)
                        out_points.InsertNextPoint(x, y, z)
                        for k in range(npatt):
                            attp_out_arrays[k].InsertNextTuple(attp_arrays[k].GetTuple(i))
                        lut[i] = count
                        count += 1
                out_poly = vtk.vtkPolyData()
                out_poly.SetPoints(out_points)
                out_poly.Allocate()
                # Creating Topology
                for i in range(len(cells_f)):
                    cell = cells_f[i]
                    npoints = cell.GetNumberOfPoints()
                    ids = vtk.vtkIdList()
                    # Convert cell points
                    for k in range(npoints):
                        ids.InsertNextId(lut[cell.GetPointId(k)])
                    out_poly.InsertNextCell(cell.GetCellType(), ids)
                    for k in range(ncatt):
                        tuples = cells_f_att[i]
                        attc_out_arrays[k].InsertNextTuple(tuples[k])

            # Adding the attributes
            for k in range(npatt):
                out_poly.GetPointData().AddArray(attp_out_arrays[k])
            for k in range(ncatt):
                out_poly.GetCellData().AddArray(attc_out_arrays[k])

            self.SetOutput(out_poly)

        else:
            raise NameError('set_surf() must be succesfully called before Execute() in a vtkFilterSurfBorderAlgorithm object.')

##########################################################################################
#
# Filter for measuring the closest distance between individual points and the closest point of the
# cells of a poly data
# Input -> The reference polydata
#
class vtkClosestPointAlgorithm(vtk.vtkPolyDataAlgorithm):

    def __init__(self):
        self.__points_x = None
        self.__points_y = None
        self.__points_z = None
        self.__points_id = None
        self.__normal_field = None

    # Called after SetInputData() is called the filter pre-computes the reference poly points so further
    # calls to evaluate() will be faster.
    def initialize(self):

        # Check input
        input_poly = self.GetInput()
        if input_poly is None:
            raise NameError('No reference poly data, SetInputData() must be succesfully called before.')

        hold_points_x = list()
        hold_points_y = list()
        hold_points_z = list()
        hold_points_id = list()
        # Adding surface points
        for i in range(input_poly.GetNumberOfCells()):
            cell = vtk.vtkGenericCell()
            input_poly.GetCell(i, cell)
            npoints = cell.GetNumberOfPoints()
            pts = cell.GetPoints()
            ids = cell.GetPointIds()
            for k in range(npoints):
                x, y, z = pts.GetPoint(k)
                hold_points_x.append(x)
                hold_points_y.append(y)
                hold_points_z.append(z)
                hold_points_id.append(ids.GetId(k))

        self.__points_x = np.asarray(hold_points_x, dtype=np.float32)
        self.__points_y = np.asarray(hold_points_y, dtype=np.float32)
        self.__points_z = np.asarray(hold_points_z, dtype=np.float32)
        self.__points_id = np.asarray(hold_points_id, dtype=np.int)

    # normal_field: must contain a vector field or a the string name to the attribute with the
    # point normals. If no valid normal field is addressed this property will stay to None so
    # evaluate() will not estimate the dot product. If normal_field is None the first scalar field
    # associated to input poly data is used
    def set_normal_field(self, normal_field=None):

        input_poly = self.GetInput()
        if isinstance(normal_field, str):
            for i in range(input_poly.GetPointData().GetNumberOfArrays()):
                if input_poly.GetPointData().GetArrayName(i) == normal_field:
                    array = input_poly.GetPointData().GetArray(i)
                    if array.GetNumberOfComponents() == 3:
                        self.__normal_field = array
                        return
        if normal_field is None:
            for i in range(input_poly.GetPointData().GetNumberOfArrays()):
                array = input_poly.GetPointData().GetArray(i)
                if array.GetNumberOfComponents() == 3:
                    self.__normal_field = array
                    return

        # If normal_field is vtk.vtkDataArray() child must have GetNumberOfComponents() attribute
        try:
            if normal_field.GetNumberOfComponents() == 3:
                self.__normal_field = normal_field
                return
        except AttributeError:
            self.__normal_field = None

        # Default
        self.__normal_field = None

    # Makes the dot product between the input point and the closest point normal
    # both vectors are previously normalized
    def __dot_norm(self, pin, pnorm_id):

        # Point and vector coordinates
        input_poly = self.GetInput()
        pnorm = input_poly.GetPoint(pnorm_id)
        norm = self.__normal_field.GetTuple(pnorm_id)
        norm = np.asarray(norm, dtype=np.float32)
        v = np.asarray(pnorm, dtype=np.float32) - np.asarray(pin, dtype=np.float32)

        # Normalization
        mv = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
        if mv > 0:
            v = v / mv
        else:
            return 0
        mnorm = math.sqrt(norm[0]*norm[0] + norm[1]*norm[1] + norm[2]*norm[2])
        if mnorm > 0:
            norm = norm / mnorm
        else:
            return 0

        return v[0]*norm[0] + v[1]*norm[1] + v[2]*norm[2]


    # From an input point [x, y, z] measures its distance to the closest point in a cell of the reference
    # poly. If initialize is called after SetInputData() is called the filter pre-computes the reference
    # poly points so further calls to evaluate() will be faster.
    # Return the euclidean distance.
    # if normal_field is set (see set_normal_field()) the dot product with the normal to the closest point
    # is also returned
    def evaluate(self, x, y, z):

        if self.__points_x is not None:

            # Fast mode (requires a previous call to initialize())
            hold_x = self.__points_x - x
            hold_y = self.__points_y - y
            hold_z = self.__points_z - z
            holdxx = hold_x * hold_x
            holdyy = hold_y * hold_y
            holdzz = hold_z * hold_z
            hold = holdxx + holdyy
            hold = hold + holdzz
            # Estimating dot product if required
            if self.__normal_field is not None:
                return math.sqrt(np.min(hold)), \
                       self.__dot_norm([x, y, z], self.__points_id[np.argmin(hold)])
            else:
                return math.sqrt(np.min(hold))

        # Check input
        input_poly = self.GetInput()
        if input_poly is None:
            raise NameError('No reference poly data, SetInputData() must be successfully called before.')

        # Cells loop for looking the minimum
        min_dist = np.finfo('d').max
        point_id = -1
        for i in range(input_poly.GetNumberOfCells()):
            cell = vtk.vtkGenericCell()
            input_poly.GetCell(i, cell)
            npoints = cell.GetNumberOfPoints()
            pts = cell.GetPoints()
            for k in range(npoints):
                mp = pts.GetPoint(k)
                s1 = x-mp[0]
                s2 = y-mp[1]
                s3 = z-mp[2]
                dist = s1*s1 + s2*s2 + s3*s3
                if dist < min_dist:
                    min_dist = dist
                    ids = cell.GetPointIds()
                    point_id = ids.GetId(k)

        # Estimating dot product if required
        if self.__normal_field is not None:
            return math.sqrt(min_dist), self.__dot_norm([x, y, z], point_id)
        else:
            return math.sqrt(min_dist)

    # From an input point [x, y, z] return the point id of the closest point in the surface
    def evaluate_id(self, x, y, z):

        if self.__points_x is not None:

            # Fast mode (requires a previous call to initialize())
            hold_x = self.__points_x - x
            hold_y = self.__points_y - y
            hold_z = self.__points_z - z
            holdxx = hold_x * hold_x
            holdyy = hold_y * hold_y
            holdzz = hold_z * hold_z
            hold = holdxx + holdyy
            hold = hold + holdzz
            # Estimating dot product if required
            return self.__points_id[np.argmin(hold)]

        # Check input
        input_poly = self.GetInput()
        if input_poly is None:
            raise NameError('No reference poly data, SetInputData() must be succesfully called before.')

        # Cells loop for looking the minimum
        min_dist = np.finfo('d').max
        point_id = -1
        for i in range(input_poly.GetNumberOfCells()):
            cell = vtk.vtkGenericCell()
            input_poly.GetCell(i, cell)
            npoints = cell.GetNumberOfPoints()
            pts = cell.GetPoints()
            for k in range(npoints):
                mp = pts.GetPoint(k)
                s1 = x-mp[0]
                s2 = y-mp[1]
                s3 = z-mp[2]
                dist = s1*s1 + s2*s2 + s3*s3
                if dist < min_dist:
                    min_dist = dist
                    ids = cell.GetPointIds()
                    point_id = ids.GetId(k)

        return point_id

##########################################################################################
#
# Filter redundant points in a DISPERSE skeleton. I don't know why but these points are
# returned by DISPERSE.
#
# Input -> The polydata for being filtered. The resolution (1 by default) is the maximum distance
#          for redundant points
#
class vtkFilterRedundacyAlgorithm(vtk.vtkPolyDataAlgorithm):

    def __init__(self):
        self.__resolution = 1

    def set_resolution(self, resolution):
        self.__resolution = resolution

    # Based on vtkClosestPointAlgorith
    # Resolution parameter is not used, resolutions=1 is always assumed
    def Execute(self):

        # Check input
        input_poly = self.GetInput()
        if input_poly is None:
            raise NameError('No reference poly data, SetInputData() must be successfully called before.')

        # Getting the attributes
        attp_arrays = list()
        attp_out_arrays = list()
        n_p_arr = input_poly.GetPointData().GetNumberOfArrays()
        for i in range(n_p_arr):
            array = input_poly.GetPointData().GetArray(i)
            klass = array.__class__
            arrayb = klass()
            arrayb.SetName(array.GetName())
            arrayb.SetNumberOfComponents(array.GetNumberOfComponents())
            attp_arrays.append(array)
            attp_out_arrays.append(arrayb)
        attc_arrays = list()
        attc_out_arrays = list()
        n_c_arr = input_poly.GetCellData().GetNumberOfArrays()
        for i in range(n_c_arr):
            array = input_poly.GetCellData().GetArray(i)
            klass = array.__class__
            arrayb = klass()
            arrayb.SetName(array.GetName())
            arrayb.SetNumberOfComponents(array.GetNumberOfComponents())
            attc_arrays.append(array)
            attc_out_arrays.append(arrayb)

        # Mask precomputations
        m_x = 0
        m_y = 0
        m_z = 0
        npoints = input_poly.GetNumberOfPoints()
        for i in range(npoints):
            x, y, z = input_poly.GetPoint(i)
            x = int(round(x))
            y = int(round(y))
            z = int(round(z))
            if x > m_x:
                m_x = x
            if y > m_y:
                m_y = y
            if z > m_z:
                m_z = z
        m_x += 1
        m_y += 1
        m_z += 1
        mask = (-1) * np.ones(shape=(m_x, m_y, m_z), dtype=np.int)

        # Loop for detecting the redundant points and creating the new geometry
        new_points = vtk.vtkPoints()
        nred_points = (-1) * np.ones(npoints, dtype=np.int)
        p_count = 0
        for i in range(npoints):
            x, y, z = input_poly.GetPoint(i)
            xc = int(round(x))
            yc = int(round(y))
            zc = int(round(z))
            if mask[xc, yc, zc] >= 0:
                # Mark for redundant
                nred_points[i] = mask[xc, yc, zc]
            else:
                # Mark for processed
                new_points.InsertNextPoint(x, y, z)
                for j in range(n_p_arr):
                    attp_out_arrays[j].InsertNextTuple(attp_arrays[j].GetTuple(i))
                mask[xc, yc, zc] = p_count
                nred_points[i] = p_count
                p_count += 1

        # Creating new topology without points redundancy
        output_poly = vtk.vtkPolyData()
        output_poly.SetPoints(new_points)
        output_poly.Allocate()
        for i in range(input_poly.GetNumberOfCells()):
            cell = vtk.vtkGenericCell()
            input_poly.GetCell(i, cell)
            ids = vtk.vtkIdList()
            for j in range(cell.GetNumberOfPoints()):
                ids.InsertNextId(nred_points[cell.GetPointId(j)])
            output_poly.InsertNextCell(cell.GetCellType(), ids)
            for j in range(n_c_arr):
                attc_out_arrays[j].InsertNextTuple(attc_arrays[j].GetTuple(i))

        # Adding attributes
        for i in range(n_p_arr):
            output_poly.GetPointData().AddArray(attp_out_arrays[i])
        for i in range(n_c_arr):
            output_poly.GetCellData().AddArray(attc_out_arrays[i])

        # Set the output
        self.SetOutput(output_poly)


    # This method is too slow for big datasets
    def __old_Execute(self):

        # Check input
        input_poly = self.GetInput()
        if input_poly is None:
            raise NameError('No reference poly data, SetInputData() must be successfully called before.')
        output_points = vtk.vtkPoints()

        # Getting the attributes
        attp_arrays = list()
        attp_out_arrays = list()
        for i in range(input_poly.GetPointData().GetNumberOfArrays()):
            array = input_poly.GetPointData().GetArray(i)
            klass = array.__class__
            arrayb = klass()
            arrayb.SetName(array.GetName())
            arrayb.SetNumberOfComponents(array.GetNumberOfComponents())
            attp_arrays.append(array)
            attp_out_arrays.append(arrayb)
        attc_arrays = list()
        attc_out_arrays = list()
        for i in range(input_poly.GetCellData().GetNumberOfArrays()):
            array = input_poly.GetCellData().GetArray(i)
            klass = array.__class__
            arrayb = klass()
            arrayb.SetName(array.GetName())
            arrayb.SetNumberOfComponents(array.GetNumberOfComponents())
            attc_arrays.append(array)
            attc_out_arrays.append(arrayb)

        # Loop for points
        list_coords = list()
        list_ids = list()
        npoints = input_poly.GetNumberOfPoints()
        lut = (-1) * np.ones(npoints, dtype=np.int)
        x1, y1, z1 = input_poly.GetPoint(0)
        output_points.InsertNextPoint(x1, y1, z1)
        nattp = len(attp_arrays)
        for i in range(nattp):
            attp_out_arrays[i].InsertNextTuple(attp_arrays[i].GetTuple(0))
        list_coords.append([x1, y1, z1])
        list_ids.append(0)
        lut[0] = 0
        count = 1
        hid = 0
        for i in range(1, npoints):
            delete = False
            x1, y1, z1 = input_poly.GetPoint(i)
            for j in range(len(list_coords)):
                x2, y2, z2 = list_coords[j]
                hx = x1 - x2
                hy = y1 - y2
                hz = z1 - z2
                dist = math.sqrt(hx*hx + hy*hy + hz*hz)
                if dist <= self.__resolution:
                    delete = True
                    hid = j
                    break
            if delete:
                lut[i] = lut[list_ids[hid]]
            else:
                lut[i] = count
                count += 1
                output_points.InsertNextPoint(x1, y1, z1)
                list_coords.append([x1, y1, z1])
                list_ids.append(i)
                for j in range(nattp):
                    attp_out_arrays[j].InsertNextTuple(attp_arrays[j].GetTuple(i))

        output_poly = vtk.vtkPolyData()
        output_poly.SetPoints(output_points)
        output_poly.Allocate()
        # Creating Topology
        nattc = len(attc_arrays)
        for i in range(input_poly.GetNumberOfCells()):
            cell = vtk.vtkGenericCell()
            input_poly.GetCell(i, cell)
            ids = vtk.vtkIdList()
            for j in range(cell.GetNumberOfPoints()):
                ids.InsertNextId(lut[cell.GetPointId(j)])
            output_poly.InsertNextCell(cell.GetCellType(), ids)
            for j in range(nattc):
                attc_out_arrays[j].InsertNextTuple(attc_arrays[j].GetTuple(i))

        # Adding attributes
        for i in range(nattp):
            output_poly.GetPointData().AddArray(attp_out_arrays[i])
        for i in range(nattc):
            output_poly.GetCellData().AddArray(attc_out_arrays[i])

        # Set the output
        self.SetOutput(output_poly)