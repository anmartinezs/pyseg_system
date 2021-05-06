"""
Collection of functions for helping to segment microtubes in tomograms

# Author: Antonio Martinez-Sanchez (Max Planck Institute for Biochemistry)
# Date: 1.07.17
"""

import csv
import vtk
from .utils import *
from sklearn.cluster import MeanShift

__author__ = 'Antonio Martinez-Sanchez'


# Clean an directory contents (directory is preserved)
# dir: directory path
def clean_dir(dir):
    for root, dirs, files in os.walk(dir):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))

def get_sub_copy(tomo, sub_pt, sub_shape):
    '''
    Returns the a subvolume of a tomogram from a center and a shape
    :param tomo: input tomogram
    :param sub_pt: subtomogram center point
    :param sub_shape: output subtomogram shape (all dimensions must be even)
    :return: a copy with the subvolume or a VOI
    '''

    # Initialization
    nx, ny, nz = int(sub_shape[0]), int(sub_shape[1]), int(sub_shape[2])
    mx, my, mz = tomo.shape[0], tomo.shape[1], tomo.shape[2]
    mx1, my1, mz1 = mx - 1, my - 1, mz - 1
    hl_x, hl_y, hl_z = int(nx * .5), int(ny * .5), int(nz * .5)
    x, y, z = int(round(sub_pt[0])), int(round(sub_pt[1])), int(round(sub_pt[2]))

    # Compute bounding restriction
    # off_l_x, off_l_y, off_l_z = x - hl_x + 1, y - hl_y + 1, z - hl_z + 1
    off_l_x, off_l_y, off_l_z = x - hl_x, y - hl_y, z - hl_z
    # off_h_x, off_h_y, off_h_z = x + hl_x + 1, y + hl_y + 1, z + hl_z + 1
    off_h_x, off_h_y, off_h_z = x + hl_x, y + hl_y, z + hl_z
    dif_l_x, dif_l_y, dif_l_z = 0, 0, 0
    dif_h_x, dif_h_y, dif_h_z = nx, ny, nz
    if off_l_x < 0:
        # dif_l_x = abs(off_l_x) - 1
        dif_l_x = abs(off_l_x)
        off_l_x = 0
    if off_l_y < 0:
        # dif_l_y = abs(off_l_y) - 1
        dif_l_y = abs(off_l_y)
        off_l_y = 0
    if off_l_z < 0:
        # dif_l_z = abs(off_l_z) - 1
        dif_l_z = abs(off_l_z)
        off_l_z = 0
    if off_h_x >= mx:
        dif_h_x = nx - off_h_x + mx1
        off_h_x = mx1
    if off_h_y >= my:
        dif_h_y = ny - off_h_y + my1
        off_h_y = my1
    if off_h_z >= mz:
        dif_h_z = nz - off_h_z + mz1
        off_h_z = mz1

    # Make the subvolume copy
    hold_sv = np.zeros(shape=np.asarray(sub_shape, dtype=np.int), dtype=tomo.dtype)
    hold_sv[dif_l_x:dif_h_x, dif_l_y:dif_h_y, dif_l_z:dif_h_z] = tomo[off_l_x:off_h_x, off_l_y:off_h_y, off_l_z:off_h_z]

    return hold_sv

# Read microtubes centerline samples and group them by microtube ID
# fname: CSV file name
# coords_cols: X, Y and Z column numbers in the CSV file
# id_col: microtube ID colum number in the CSV file
# Returns: a dictionary indexed with the microtube id with centerline coodinates in a list
def read_csv_mts(fname, coords_cols, id_col):

    # Initialization
    mt_dict = None

    # Open the file to read
    with open(fname, 'r') as in_file:
        reader = csv.reader(in_file)

        # Reading loop
        coords, ids = list(), list()
        for row in reader:
            x, y, z, idx = float(row[coords_cols[0]]), float(row[coords_cols[1]]), float(row[coords_cols[2]]),\
                           int(row[id_col])
            coords.append(np.asarray((x, y, z), dtype=np.float))
            ids.append(idx)

        # Dictionary creation
        mt_dict = dict.fromkeys(set(ids))
        for key in mt_dict.keys():
            mt_dict[key] = list()
        for key, coord in zip(ids, coords):
            mt_dict[key].append(coord)

    return mt_dict


# Converts as set of points into a binary mask
# points: iterable with the points coordinates
# mask_shape: shape of the output mask
# inv: if False (default) then True-fg and False-bg, otherwise these values are inverted
# Returns: a 3D numpy binray array
def points_to_mask(points, mask_shape, inv=False):
    mask = np.zeros(shape=mask_shape, dtype=np.bool)
    for point in points:
        i, j, k = int(round(point[0])), int(round(point[1])), int(round(point[2]))
        if (i < 0) or (j < 0) or (k < 0) or \
            (i >= mask_shape[0]) or (j >= mask_shape[1]) or (k >= mask_shape[2]):
            continue
        else:
            mask[i, j, k] = True
    if inv:
        return np.invert(mask)
    else:
        return mask


# Mean shift clustering for points in 3D space
# coords: points 3D coordinates in numpy array with size [n_points, 3]
# bandwith: bandwidth used in RBF kernel
# cluster_all: if True standard behaviour, if not all points and clustered so orphand point are
#              associated to trivial clusters
# Returns: cluster labels array [n_points]
def cluster_3d_mean_shift(coords, bandwidth, cluster_all=False):

    # Input parsing
    if (not isinstance(coords, np.ndarray)) or (len(coords.shape) != 2) or (coords.shape[1] != 3):
        error_msg = 'Input coords must be numpy array of 3D coordinates (size=[n_points, 3]).'
        raise pexceptions.PySegInputError(expr='cluster_3D_mean_shift', msg=error_msg)
    if bandwidth <= 0:
        error_msg = 'Input bandwith must be greater than zero.'
        raise pexceptions.PySegInputError(expr='cluster_3D_mean_shift', msg=error_msg)
    bw_f = float(bandwidth)

    # Call to MeanShift
    mshift = MeanShift(bandwidth=bw_f, cluster_all=True, bin_seeding=True)
    mshift.fit(coords)
    labels = np.asarray(mshift.labels_)

    # Orphans processing
    if cluster_all:
        labels_max = labels.max()
        for i, lbl in enumerate(labels):
            if lbl == -1:
                labels_max += 1
                labels[i] = labels_max

    return labels


# Computes center of gravity for every cluster
# coords: coordinates array [n_points, 3]
# labels: cluster labels array [n_points]
def clusters_cg(coords, labels):

    # Input parsing
    if (not isinstance(coords, np.ndarray)) or (len(coords.shape) != 2) or (coords.shape[1] != 3):
        error_msg = 'Input coords must be numpy array of 3D coordinates (size=[n_points, 3]).'
        raise pexceptions.PySegInputError(expr='clusters_cg', msg=error_msg)
    if (not isinstance(labels, np.ndarray)) or (len(labels.shape) != 1) or \
            (labels.shape[0] != coords.shape[0]):
        error_msg = 'Input labels must be array with size=[n_points].'
        raise pexceptions.PySegInputError(expr='clusters_cg', msg=error_msg)

    # Center of gravity loop computations
    u_labels = np.unique(labels)
    n_lbls = len(u_labels)
    n_points_lut = dict.fromkeys(u_labels)
    cgs = dict.fromkeys(u_labels)
    for lbl in u_labels:
        cgs[lbl] = np.zeros(shape=3, dtype=np.float)
        n_points_lut[lbl] = 0
    for point, lbl in zip(coords, labels):
        cgs[lbl] += point
        n_points_lut[lbl] += 1
    # Averaging loop
    for lbl in u_labels:
        cgs[lbl] *= (1./float(n_points_lut[lbl]))

    return np.asarray(list(cgs.values()), dtype=np.float)


# Converts cluster of points int a vtkPolyData
# points: array with 3D points coordinates [n_points, 3]
# labels: array with point labels [n_points] with cluster labels,
#         if None default every point correspond with a cluster
# centers: cluster centers array [n_unique_labels] (default None)
def clusters_to_poly(points, labels=None, centers=None):

    # Input parsing
    if (not isinstance(points, np.ndarray)) or (len(points.shape) != 2) or (points.shape[1] != 3):
        error_msg = 'Input coords must be numpy array of 3D coordinates (size=[n_points, 3]).'
        raise pexceptions.PySegInputError(expr='points_to_poly', msg=error_msg)
    if labels is not None:
        if (not isinstance(labels, np.ndarray)) or (len(labels.shape) != 1) or \
                (labels.shape[0] != points.shape[0]):
            error_msg = 'Input labels must be array with size=[n_points].'
            raise pexceptions.PySegInputError(expr='points_to_poly', msg=error_msg)
    if centers is not None:
        if not isinstance(centers, np.ndarray):
            error_msg = 'Input centers must be array with size=[n_unique_labels].'
            raise pexceptions.PySegInputError(expr='points_to_poly', msg=error_msg)

    # Initialization
    poly = vtk.vtkPolyData()
    p_points = vtk.vtkPoints()
    p_cells = vtk.vtkCellArray()
    plabels = vtk.vtkIntArray()
    plabels.SetNumberOfComponents(1)
    plabels.SetName('label')
    pcenters = vtk.vtkIntArray()
    pcenters.SetNumberOfComponents(1)
    pcenters.SetName('center')

    # Points loop
    for i, point in enumerate(points):
        p_points.InsertNextPoint(point)
        p_cells.InsertNextCell(1)
        p_cells.InsertCellPoint(i)
        if labels is None:
            plabels.InsertTuple1(i, i)
        else:
            plabels.InsertTuple1(i, labels[i])
        if centers is None:
            pcenters.InsertTuple1(i, 1)
        else:
            pcenters.InsertTuple1(i, -1)

    # Inserting centers
    if centers is not None:
        for i in range(centers.shape[0]):
            p_i = points.shape[0] + i
            p_points.InsertNextPoint(centers[i])
            p_cells.InsertNextCell(1)
            p_cells.InsertCellPoint(p_i)
            plabels.InsertTuple1(p_i, -1)
            pcenters.InsertTuple1(p_i, 1)

    # Building the polydata
    poly.SetPoints(p_points)
    poly.SetVerts(p_cells)
    poly.GetCellData().AddArray(plabels)
    poly.GetCellData().AddArray(pcenters)

    return poly


# Computes rotation angles of from an input vector to fit reference [0,0,1] vector having a free Euler angle
# in Relion format
# First Euler angle (Rotation) is assumed 0
# v_in: input vector
# mode: either 'active' (default) or 'pasive'
# Returns: a 2-tuple with the Euler angles in Relion format
def vect_to_zrelion(v_in, mode='active'):

    # Normalization
    v_m = np.asarray((v_in[1], v_in[0], v_in[2]), dtype=np.float32)
    try:
        n = v_m / math.sqrt((v_m*v_m).sum())
    except ZeroDivisionError:
        print('WARNING (vect_rotation_ref): vector with module 0 cannot be rotated!')
        return 0., 0., 0.

    # Computing angles in Extrinsic ZYZ system
    alpha = np.arccos(n[2])
    beta = np.arctan2(n[1], n[0])

    # Transform to Relion system (intrinsic ZY'Z'' where rho is free)
    rot, tilt, psi = 0., unroll_angle(math.degrees(alpha), deg=True), \
                     unroll_angle(180.-math.degrees(beta), deg=True)

    # By default is active, invert if passive
    if mode == 'passive':
        M = rot_mat_relion(rot, tilt, psi, deg=True)
        rot, tilt, psi = rot_mat_eu_relion(M.T, deg=True)

    return rot, tilt, psi


def randomize_voxel_mask(vol, mask, ref='fg'):
    """
    Function to randomize voxel density value in masked volumes
    :param vol: volume with the density map
    :param mask: volume with the binary mask (fg: True, bg: False)
    :param ref: 'fg' (default) indicates that (ref: fg, ref: bg)
    :return: a copy of vol but with the pixel in region marked as 'fg' in 'ref'
    """

    # Initialization
    o_vol = np.copy(vol)

    # Finding 'bg' and reference
    bg_ids = np.where(mask == False)
    if ref == 'fg':
        ref_ids = np.where(mask)
    else:
        ref_ids = np.where(mask == False)

    # Randomization
    rnd_ids = np.random.randint(0, len(ref_ids[0]), size=len(bg_ids[0]))
    for i in range(len(bg_ids[0])):
        rnd_id = rnd_ids[i]
        x, y, z = bg_ids[0][i], bg_ids[1][i], bg_ids[2][i]
        rnd_x, rnd_y, rnd_z = ref_ids[0][rnd_id], ref_ids[1][rnd_id], ref_ids[2][rnd_id]
        o_vol[x, y, z] = vol[rnd_x, rnd_y, rnd_z]

    return o_vol
