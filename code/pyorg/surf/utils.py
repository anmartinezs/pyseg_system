"""
Pool of useful functions for processing surfaces

# Author: Antonio Martinez-Sanchez (Max Planck Institute for Biochemistry)
# Date: 1.05.17
"""

import vtk
import math
import numpy as np
from pyorg import disperse_io

__author__ = 'Antonio Martinez-Sanchez'


# Iso-surface on an input 3D volume
# tomo: input 3D numpy array
# th: iso-surface threshold
# flip_axis: if not None (default) it specifies the axis to flip (valid: 0, 1 or 3)
# closed: if True (default False) if forces to generate a closed surface, VERY IMPORTANT: closed output
#         is only guaranteed for input boolean tomograms
# normals: normals orientation, valid 'inwards' (default), otherwise orientation is outwards
# Returns: a vtkPolyData object only made up of triangles
def iso_surface(tomo, th, flp=None, closed=False, normals='inwards'):

    # Marching cubes configuration
    march = vtk.vtkMarchingCubes()
    tomo_vtk = disperse_io.numpy_to_vti(tomo)
    if closed:
        # print str(tomo_vtk.GetExtent()), str(tomo.shape)
        padder = vtk.vtkImageConstantPad()
        padder.SetInputData(tomo_vtk)
        padder.SetConstant(0)
        padder.SetOutputWholeExtent(-1, tomo.shape[0], -1, tomo.shape[1], -1, tomo.shape[2])
        padder.Update()
        tomo_vtk = padder.GetOutput()

    # Flipping
    if flp is not None:
        flp_i = int(flp)
        if (flp_i >= 0) and (flp_i <= 3):
            fliper = vtk.vtkImageFlip()
            fliper.SetFilteredAxis(2)
            fliper.SetInputData(tomo_vtk)
            fliper.Update()
            tomo_vtk = fliper.GetOutput()

    # Running Marching Cubes
    march.SetInputData(tomo_vtk)
    march.SetValue(0, th)
    march.Update()
    hold_poly = march.GetOutput()

    # Filtering
    hold_poly = poly_filter_triangles(hold_poly)

    # Normals orientation
    orienter = vtk.vtkPolyDataNormals()
    orienter.SetInputData(hold_poly)
    orienter.AutoOrientNormalsOn()
    if normals == 'inwards':
        orienter.FlipNormalsOn()
    orienter.Update()
    hold_poly = orienter.GetOutput()

    if closed and (not is_closed_surface(hold_poly)):
        raise RuntimeError

    return hold_poly


# Swap XY coordinates in a vtkPolyData
def poly_swapxy(ref_poly):

    # Initialization
    new_pts = vtk.vtkPoints()
    old_pts = ref_poly.GetPoints()

    # Loop for swapping
    for i in range(old_pts.GetNumberOfPoints()):
        pt = old_pts.GetPoint(i)
        new_pts.InsertPoint(i, [pt[1], pt[0], pt[2]])

    # Updating the poly data
    ref_poly.SetPoints(new_pts)

    return ref_poly


# Filter a vtkPolyData to keep just the polys which are triangles
# poly: input vtkPolyData
# Returns: the input poly filtered
def poly_filter_triangles(poly):
    cut_tr = vtk.vtkTriangleFilter()
    cut_tr.SetInputData(poly)
    cut_tr.PassVertsOff()
    cut_tr.PassLinesOff()
    cut_tr.Update()
    return cut_tr.GetOutput()


# Decimate a vtkPolyData
# poly: input vtkPolyData
# dec: Specify the desired reduction in the total number of polygons, default None (not applied)
#      (e.g., if TargetReduction is set to 0.9,
#      this filter will try to reduce the data set to 10% of its original size).
# Returns: the input poly filtered
def poly_decimate(poly, dec):
    tr_dec = vtk.vtkDecimatePro()
    tr_dec.SetInputData(poly)
    tr_dec.SetTargetReduction(dec)
    tr_dec.Update()
    return tr_dec.GetOutput()


# Returns the intersection of 2 polys (closed surfaces, A intersected by B)
# poly_i: input polys
def poly_2_intersection(poly_a, poly_b):
    poly_bool = vtk.vtkBooleanOperationPolyDataFilter()
    poly_bool.SetOperationToIntersection()
    poly_bool.SetInputData(0, poly_a)
    poly_bool.SetInputData(1, poly_b)
    poly_bool.Update()
    return poly_bool.GetOutput()


# Check if two polys (closed surfaces, A is intersected by B)
# poly_i: input polys
# TIP: it is recommended to set A with the poly data with less cells
def is_2_polys_intersect(poly_a, poly_b):
    selector = vtk.vtkSelectEnclosedPoints()
    selector.Initialize(poly_a)
    cells = poly_b.GetPolys()
    vid = vtk.vtkIdList()
    for i in range(cells.GetNumberOfPolys()):
        cells.GetCell(i, vid)
        for j in range(vid.GetNumberOfIds()):
            if selector.IsInsideSurface(poly_b.GetPoint(vid.GetId(j))) == 1:
                return True
    return False

# Returns the difference between to polys (A-B)
# poly_i: input polys
def poly_2_difference(poly_a, poly_b):
    poly_bool = vtk.vtkBooleanOperationPolyDataFilter()
    poly_bool.SetOperationToDiference()
    poly_bool.SetInputData(0, poly_a)
    poly_bool.SetInputData(1, poly_b)
    poly_bool.Update()
    return poly_bool.GetOutput()


# Converts a point into a poly
# point: 3-tuple with point coordinates
# normal: 3-tuple with the normal to be associated as property (default None)
# n_name: name for the normal (default 'normal')
def point_to_poly(point, normal=None, n_name='n_normal'):
    poly = vtk.vtkPolyData()
    p_points = vtk.vtkPoints()
    p_cells = vtk.vtkCellArray()
    p_points.InsertNextPoint(point)
    p_cells.InsertNextCell(1)
    p_cells.InsertCellPoint(0)
    poly.SetPoints(p_points)
    poly.SetVerts(p_cells)
    if normal is not None:
        p_norm = vtk.vtkFloatArray()
        p_norm.SetName(n_name)
        p_norm.SetNumberOfComponents(3)
        p_norm.InsertTuple(0, normal)
        poly.GetPointData().AddArray(p_norm)
    return poly


# Converts an iterable of points into a poly
# points: iterable with 3D points coordinates
# normals: iterable with coordinates for the normals
# n_name: name for the normal (default 'normal')
def points_to_poly(points, normals=None, n_name='n_normal'):
    poly = vtk.vtkPolyData()
    p_points = vtk.vtkPoints()
    p_cells = vtk.vtkCellArray()
    if normals is not None:
        p_norm = vtk.vtkFloatArray()
        p_norm.SetName(n_name)
        p_norm.SetNumberOfComponents(3)
        for i, point, normal in zip(range(len(points)), points, normals):
            p_points.InsertNextPoint(point)
            p_cells.InsertNextCell(1)
            p_cells.InsertCellPoint(i)
            p_norm.InsertTuple(i, normal)
    else:
        for i, point in enumerate(points):
            p_points.InsertNextPoint(point)
            p_cells.InsertNextCell(1)
            p_cells.InsertCellPoint(i)
    poly.SetPoints(p_points)
    poly.SetVerts(p_cells)
    if normals is not None:
        poly.GetPointData().AddArray(p_norm)
    return poly


# Clip poly data with another
def clip_poly(poly, poly_clipper):
    measurer = vtk.vtkImplicitPolyDataDistance()
    measurer.SetInput(poly_clipper)
    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(poly)
    clipper.SetClipFunction(measurer)
    clipper.SetValue(0)
    clipper.Update()
    return clipper.GetOutput()


# Checks if an input vtkPolyData is a closed surface
# poly: input vtkPolyData to check
def is_closed_surface(poly):
    selector = vtk.vtkSelectEnclosedPoints()
    selector.CheckSurfaceOn()
    selector.SetSurfaceData(poly)
    if selector.GetCheckSurface() > 0:
        return True
    else:
        return False


# Find the closest point in a vtkPolyData
# Returns: closest point coordinates
def vtp_closest_point(poly, point):
    point_tree = vtk.vtkKdTreePointLocator()
    point_tree.SetDataSet(poly)
    point_tree.BuildLocator()
    cpoint_id = point_tree.FindClosestPoint(point)
    return poly.GetPoint(cpoint_id)

# Find the closet point in a vtkPolyData to another vtkPolyData
# vtp_1|2: input vtkPolyData objects
# Returns: closest point in vtp_1 to vtp_2 ID (-1 mean unexpected event) and the distance
def vtp_to_vtp_closest_point(vtp_1, vtp_2):
    point_tree = vtk.vtkKdTreePointLocator()
    point_tree.SetDataSet(vtp_2)
    point_tree.BuildLocator()
    point_id, point_dst = -1, np.finfo(np.float).max
    for i in xrange(vtp_1.GetNumberOfPoints()):
        point_1 = vtp_1.GetPoint(i)
        point_2_id = point_tree.FindClosestPoint(point_1)
        point_2 = vtp_2.GetPoint(point_2_id)
        hold = np.asarray(point_1, dtype=np.float) - np.asarray(point_2, dtype=np.float)
        hold_dst = np.sqrt((hold * hold).sum())
        if hold_dst < point_dst:
            point_id, point_dst = i, hold_dst
    return point_id, point_dst

# Expands a dictionary with statistics from a ListTomoParticles into a matrix by taking into account the number of
# points of every tomogram
# dict_in: input dictionary
# list_in: input ListTomoParticles
# Return: a matrix where MxN, where M are the distances, an N the number of entries
def stat_dict_to_mat(dict_in, list_in):

    # Initialization
    hold_mat = list()

    for key, arr in zip(dict_in.iterkeys(), dict_in.itervalues()):
        tomo = list_in.get_tomo_by_key(key)
        # Parse simulation keys
        for i in range(tomo.get_num_particles()):
            hold_mat.append(arr)

    return np.asarray(hold_mat, dtype=np.float)

# Computes p-values from experimental data and simulations dictionaries
# rads: range of radius
# dic_exp: dictionary with the experimental arrays
# dic_sim: dictionary of matrices with the distributions
# Returns: a dictionary with the signed p-values in percentage and the correspoindent radius [r_l, r_h, p_l, p_h]
def list_tomoparticles_pvalues(rads, dic_exp, dic_sim):

    # Initialization
    p_values = dict()

    # Loop dictrionary entries
    for key, exp_arr in zip(dic_exp.iterkeys(), dic_exp.itervalues()):
        try:
            sim_mat = dic_sim[key]
        except KeyError:
            print 'WARNING (list_tomoparticles_pvalues): key ' + key + ' not in simulation dictionary!'
            continue
        # Get the percentile maximum loop
        p_values[key] = [0., 0., 0., 0.]
        pers = np.zeros(shape=exp_arr.shape, dtype=np.float32)
        for i in xrange(len(exp_arr)):
            rad, exp_value, sim_arr = rads[i], exp_arr[i], sim_mat[:, i]
            sim_len = float(len(sim_arr))
            if sim_len > 0:
                per_l, per_h = float((sim_arr > exp_value).sum()), float((sim_arr < exp_value).sum())
                per_l, per_h = -100.*(per_l/sim_len), 100.*(per_h/sim_len)
                if per_l < p_values[key][2]:
                    p_values[key][0], p_values[key][2] = rad, per_l
                if per_h > p_values[key][3]:
                    p_values[key][1], p_values[key][3] = rad, per_h
            else:
                p_values[key] = [0., 0., 0., 0.]

    return p_values


def line_2_pts(pt_1, pt_2, step):
    """
    Generates the coordiantes in the line between pt_1 and pt_2
    :param pt_1/2: line extrema
    :param step: point distance step
    :return: an array with size [n, 3] where n is the number of point coordiantes
    """

    # Initialization
    v = pt_2 - pt_1
    v_len = math.sqrt((v * v).sum())
    if v_len <= 0:
        coords = np.zeros(shape=(2,len(pt_1)), dtype=np.float32)
        coords[0, :] = pt_1
        coords[1, :] = pt_2
        return coords
    v_norm = v / v_len

    # Loop for tracing the line
    hold_dst, coords = 0, list()
    coords.append(pt_1)
    while hold_dst < v_len:
        hold_dst += step
        coord = pt_1 + v_norm * hold_dst
        coords.append(coord)
    coords.append(pt_2)

    return np.asarray(coords, dtype=np.float32)


def points_to_polyline(coords):
    """
    Generates a vtkPolyData as a poly line
    :param coords: iterable with line coordinates
    :return: a vtkPolyData object
    """
    points = vtk.vtkPoints()
    for coord in coords:
        points.InsertNextPoint(coord)
    line = vtk.vtkPolyLine()
    line.GetPointIds().SetNumberOfIds(coords.shape[0])
    for i in range(coords.shape[0]):
        line.GetPointIds().SetId(i, i)
    cells = vtk.vtkCellArray()
    cells.InsertNextCell(line)
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetLines(cells)
    return poly


def is_point_inside_surf(point, selector, conv_iter, max_iter):
    """
    Call repeatedly to VTK methods until a consensuous is reached
    :param point: point coordinates
    :param selector: vtkSelectEnclosedPoints object for the surface
    :param conv_iter: number of iterations for convergence
    :param max_iter: maximum number of iterations
    :return: True if the opoint is inside after the repeated calls, otherwise False
    """
    count_p, count_n, total = 0, 0, 0
    for i in range(max_iter):
        if selector.IsInsideSurface(point) > 0:
            count_p += 1
            count_n = 0
            total += 1
        else:
            count_p = 0
            count_n += 1
            total -= 1
        if count_p >= conv_iter:
            return True
        if count_n >= conv_iter:
            return False
    if total > 0:
        return True
    else:
        return False