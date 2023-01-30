__author__ = 'martinez'

import math
import shutil
from .variables import *
import scipy as sp
import random as rnd
from pyorg import pexceptions
try:
    import pickle as pickle
except:
    import pickle


###### Global analysis

# Unpickle a Generic PySeg object
def unpickle_obj(fname):

    # Load pickable state
    f_pkl = open(fname, 'rb')
    try:
        gen_obj = pickle.load(f_pkl)
    except UnicodeDecodeError:
        gen_obj = pickle.load(f_pkl, encoding='latin1')
    finally:
        f_pkl.close()

    return gen_obj

# epsilon for testing whether a number is close to zero
ZERO_EPS = np.finfo(float).eps * 4.0

# Trilinear interpolation of a coordinate point within a tomogram (voxel size must be 1)
# x, y, z: input coordinates
# tomogra: input tomogram
# The ouput is the density value
def trilin_tomo(x, y, z, tomogram):

    # Check input dimensions
    if len(tomogram.shape) != 3:
        error_msg = 'The input tomogram is not a 3D array.'
        raise pexceptions.PySegInputError(expr='trilin_tomo', msg=error_msg)
    xc = int(math.ceil(x))
    yc = int(math.ceil(y))
    zc = int(math.ceil(z))
    xf = int(math.floor(x))
    yf = int(math.floor(y))
    zf = int(math.floor(z))
    if (xc >= tomogram.shape[0]) or (yc >= tomogram.shape[1]) or (zc >= tomogram.shape[2]) or \
            (xf < 0) or (yf < 0) or (zf < 0):
        error_msg = 'Input coordinates out of tomogram bounds'
        raise pexceptions.PySegInputError(expr='trilin_tomo', msg=error_msg)

    # Get neigbourhood values
    v000 = float(tomogram[xf, yf, zf])
    v100 = float(tomogram[xc, yf, zf])
    v010 = float(tomogram[xf, yc, zf])
    v001 = float(tomogram[xf, yf, zc])
    v101 = float(tomogram[xc, yf, zc])
    v011 = float(tomogram[xf, yc, zc])
    v110 = float(tomogram[xc, yc, zf])
    v111 = float(tomogram[xc, yc, zc])

    # Coordinates correction
    xn = x - xf
    yn = y - yf
    zn = z - zf
    x1 = 1 - xn
    y1 = 1 - yn
    z1 = 1 - zn

    # Interpolation
    return (v000 * x1 * y1 * z1) + (v100 * xn * y1 * z1) + (v010 * x1 * yn * z1) + \
           (v001 * x1 * y1 * zn) + (v101 * xn * y1 * zn) + (v011 * x1 * yn * zn) + \
           (v110 * xn * yn * z1) + (v111 * xn * yn * zn)

# Applies a linear mapping to the input array for getting an array in the specified range
def lin_map(array, lb=0, ub=1):
    a = np.max(array)
    i = np.min(array)
    den = a - i
    if den == 0:
        return array
    m = (ub - lb) / den
    c = ub - m*a
    return m*array + c

# Contrast enhancement by standard deviation cropping
# std: if not None (default 3), number of standard deviations to density cropping
# lb: output lower bound for remapping
# ub: output upper bound for remapping
# mask: if not None (default) set ROI
def cont_en_std(array, nstd=3, lb=0, ub=1, mask=None):

    # Masking
    if mask is None:
        hold_array = array
    else:
        m_ids = np.asarray(mask > 0, dtype=bool)
        hold_array = array[m_ids]

    # Remapping parameters estimation
    if nstd is None:
        ui, li = hold_array.max(), hold_array.min()
    else:
        mn = hold_array.mean()
        std = hold_array.std()
        ui = mn + nstd*std
        li = mn - nstd*std
    m = (ub - lb) / float((ui - li))
    b = ub - m*ui

    # Remapping
    rarray = hold_array*m + b

    # Density cropping
    rarray[rarray > ub] = ub
    rarray[rarray < lb] = lb

    # Unmasking
    if mask is None:
        farray = rarray
    else:
        farray = np.zeros(shape=array.shape, dtype=array.dtype)
        farray[m_ids] = rarray

    return farray

# Applies a generalized logistic mapping to the input array for getting an array in
# the specified range
# array: input numpy array
# b: growing rate (default 0.5)
# lb: lower output bound (default None, it is computed from input data)
# ub: upper output bound (default None, it is computed from input data)
def gen_log_map(array, b=0.5, lb=None, ub=None):
    if lb is None:
        lb = np.min(array)
    if ub is None:
        ub = np.max(array)
    v = 0.5
    M = np.mean(array)
    hold = 1 + v * np.exp(-b*(array-M))
    return lb + ((ub-lb) / np.power(hold, 1/v))


# Makes the dotproduct between the input point and the closest point normal
# both vectors are previously normalized
# Input must be float numpy array
def dot_norm(p, pnorm, norm):

    # Point and vector coordinates
    v = pnorm - p

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

# _dims=[min, max]: crops voxels out of the dimensions
def crop_cube(vol, x_dims, y_dims, z_dims, value_border=0):

    # Check dimensions
    if x_dims[0] < 0:
        x_dims[0] = 0
    if x_dims[1] > vol.shape[0]:
        x_dims[1] = vol.shape[0]
    if y_dims[0] < 0:
        y_dims[0] = 0
    if z_dims[1] > vol.shape[1]:
        z_dims[1] = vol.shape[1]
    if z_dims[0] < 0:
        z_dims[0] = 0
    if z_dims[1] > vol.shape[2]:
        z_dims[1] = vol.shape[2]

    # Cropping
    hold_vol = value_border * np.ones(shape=vol.shape, dtype=float)
    hold_vol[x_dims[0]:x_dims[1], y_dims[0]:y_dims[1], z_dims[0]:z_dims[1]] = \
        vol[x_dims[0]:x_dims[1], y_dims[0]:y_dims[1], z_dims[0]:z_dims[1]]

    return hold_vol

##########################################################################################
#   Field Estimators: they generates a density map from a sparse set of points input
#
#

# Linear interpolation from eight neighbours in 3D
# tomo: input 3D numpy array
# p: coordinates of the point for being interpolated
# Returns: interpolation value, if the input point is in the border (or out) a zero is returned
def trilin3d(tomo, p):

    x0, y0, z0 = int(np.floor(p[0])), int(np.floor(p[1])), int(np.floor(p[2]))
    x1, y1, z1 = int(np.ceil(p[0])), int(np.ceil(p[1])), int(np.ceil(p[2]))
    xd, yd, zd = p[0]-x0, p[1]-y0, p[2]-z0
    try:
        v000 = tomo[x0, y0, z0]
        v001 = tomo[x0, y0, z1]
        v010 = tomo[x0, y1, z0]
        v011 = tomo[x0, y1, z1]
        v100 = tomo[x1, y0, z0]
        v101 = tomo[x1, y0, z1]
        v110 = tomo[x1, y1, z0]
        v111 = tomo[x1, y1, z1]
    except IndexError:
        return 0

    # Trilinear approximation
    c00 = v000*(1-xd) + v100*xd
    c10 = v010*(1-xd) + v110*xd
    c01 = v001*(1-xd) + v101*xd
    c11 = v011*(1-xd) + v111*xd
    c0 = c00*(1-yd) + c10*yd
    c1 = c01*(1-yd) + c11*yd

    return c0*(1-zd) + c1*zd

# Linear interpolation from four neighbours in 2D
# tomo: input 2D numpy array
# p: coordinates of the point for being interpolated
# Returns: interpolation value, if the input point is in the border (or out) a zero is returned
def trilin2d(tomo, p):

    x0, y0 = int(np.floor(p[0])), int(np.floor(p[1]))
    x1, y1 = int(np.ceil(p[0])), int(np.ceil(p[1]))
    xd, yd = p[0]-x0, p[1]-y0
    try:
        a00 = tomo[x0, y0]
        a10 = tomo[x1, y0] - tomo[x0, y0]
        a01 = tomo[x0, y1] - tomo[x0, y0]
        a11 = tomo[x1, y1] + tomo[x0, y0] - tomo[x1, y0] - tomo[x0, y1]
    except IndexError:
        return 0

    return a00 + a10*xd + a01*yd + a11*xd*yd

# Add missing wedge to distortion
# In this version only tilt axes perpendicular to XY are considered
# in_shape: output tomogram shape (nx, ny, nz)
# wr_ang: wedge rotation angle in degrees [-90, 90]
# tilt_ang: maximum tilt angles in degrees [0, 90]
# Returns: a tomogram with a wedge shape
def wedge(in_shape, wr_ang, tilt_ang=90.):

    # Precompute wedge vectors
    rho = np.radians(wr_ang)
    phi = np.radians(90 - tilt_ang[0])
    phi2 = np.pi - phi
    # print phi, phi2
    # z = np.array((0, 0, 1))
    r = np.array((np.cos(rho), np.sin(rho), 0))

    # Creating the wedge
    nx, ny, nz = in_shape
    nx2, ny2, nz2 = nx * 0.5, ny * 0.5, nz * 0.5
    X, Y, Z = np.meshgrid(np.linspace(-nx2, nx2, nx),
                          np.linspace(-ny2, ny2, ny),
                          np.linspace(-nz2, nz2, nz))
    # Proyecting angles on the plane with r as normal
    X_h, Y_h, Z_h = Y*r[2] + Z*r[1], Z*r[0] + X*r[2], X*r[1] + Y*r[0]
    X, Y, Z = Y_h*r[2] + Z_h*r[1], Z_h*r[0] + X_h*r[2], X_h*r[1] + Y_h*r[0]
    W = np.sqrt(X*X + Y*Y + Z*Z)
    # To avoid divide by zero
    Id = W == 0
    W[Id] = np.finfo(W.dtype).resolution
    W_inv = 1 / W
    P = np.arccos(Z * W_inv)

    # Wedge criteria
    Id = np.logical_and(P > phi, P < phi2)
    W = np.zeros(shape=W.shape, dtype=float)
    W[Id] = 1

    return W


def add_mw(tomo, wr_ang, tilt_ang=0, norm='01'):
    """
    Add missing wedge to distortion (so far, just symmetric wedges are considered)
    :param tomo: input tomogram (ndarray)
    :param wr_ang: wedge rotation angle in degrees [-90, 90]
    :param tilt_ang: maximum tilt angle in degrees [0, 90]
    :param norm: normalization mode, valid: '01' (default), 'mn_std' and None
    :return: the tomogram with the missing wedge applied
    """

    # Creating the wedge
    nx, ny, nz = tomo.shape
    nx2, ny2, nz2 = tomo.shape[0] * 0.5, tomo.shape[1] * 0.5, tomo.shape[2] * 0.5
    X, Y, Z = np.meshgrid(np.linspace(-nx2, nx2, nx),
                          np.linspace(-ny2, ny2, ny),
                          np.linspace(-nz2, nz2, nz))
    phi = np.radians(90.-tilt_ang)
    W = (math.tan(phi) <= (np.abs(X) / np.abs(Z))).astype(np.float32)
    if wr_ang > 0:
        W = tomo_rot(W, (wr_ang, 0, 0), conv='mat')

    # from pyorg import disperse_io
    # disperse_io.save_numpy(W, '/fs/home/martinez/workspace/pyseg_system/data/tutorials/synth_sumb/mics/wedge.mrc')

    # Adding MW distortion through Fourier method
    hold = np.real(np.fft.ifftn(np.fft.ifftshift(np.fft.fftshift(np.fft.fftn(tomo)) * W)))
    # disperse_io.save_numpy(hold, '/fs/home/martinez/workspace/pyseg_system/data/tutorials/synth_sumb/mics/hold.mrc')
    if norm == '01':
        return lin_map(hold, lb=0, ub=1)
    elif norm == 'mn_std':
        return (hold-np.mean(hold)) / np.std(hold)
    else:
        return hold

# Computes Average Nearest Neighbour Ratio from a volume with cloud of points embedded
# vol: input numpy tomogram
# mask: numpy tomogram for making
# Return: the estimated ANN, mean shortest distance and its variance
def ann_vol(vol, mask=None):

    # Initialization
    if mask is None:
        est_vol = float(np.prod(vol.shape))
        msk_vol = vol
    else:
        est_vol = float(mask.sum())
        msk_vol = (vol * mask).astype(bool)
    n_points = msk_vol.sum()
    if (est_vol == 0) or (n_points == 0):
        return 0, 0, 0
    d_e = 1. / math.pow(n_points/est_vol, 1./3.)
    cloud = get_cloud_coords(msk_vol)
    dists = np.zeros(shape=cloud.shape[0], dtype=float)

    # Shortest distance loop
    for i in range(len(dists)):
        hold = cloud[i] - cloud
        hold = np.sum(hold*hold, axis=1)
        hold[i] = np.inf
        dists[i] = math.sqrt(np.min(hold))

    mn = dists.mean()
    if d_e > 0:
        return mn/d_e, mn, dists.var()
    else:
        return 0, mn, dists.var()


# Returns an array with the coordinates of non-zero points
# cloud: input tomogram
def get_cloud_coords(cloud):

    coords = list()

    ids = np.where(cloud > 0)
    for i in range(len(ids[0])):
        coords.append(np.asarray((ids[0][i], ids[1][i], ids[2][i])))

    return np.asarray(coords, dtype=float)

# Computes local curvature of a curve in space
# curve: array of n points in a 3D space
# Return: an n-2 array with the local curvatures, if n <= 2 [0] is returned
def compute_space_k(curve):

    # Initialization
    n_p = curve.shape[0]
    if n_p <= 2:
        return np.zeros(shape=1, dtype=float)
    curvatures = np.zeros(shape=n_p-2, dtype=float)

    # Loop for computing the curvatures
    cont = 0
    for i in range(1, curve.shape[0]-1):
        # Samples
        v_l1, v, v_p1 = curve[i-1], curve[i], curve[i+1]
        # Tangent vectors
        t_l1, t = v - v_l1, v_p1 - v
        m_t_l1 = math.sqrt(t_l1[0]*t_l1[0] + t_l1[1]*t_l1[1] + t_l1[2]*t_l1[2])
        if m_t_l1 <= 0:
            t_l1[0], t_l1[1], t_l1[2] = 0., 0., 0.
        else:
            t_l1 /= m_t_l1
        m_t = math.sqrt(t[0]*t[0] + t[1]*t[1] + t[2]*t[2])
        if m_t <= 0:
            t[0], t[1], t[2] = 0., 0., 0.
        else:
            t /= m_t
        # Angle between tangents
        alpha = angle_2vec_3D(t_l1, t)
        # Arc length
        h_1 = .5 * (v_p1-v)
        h_2 = .5 * (v-v_l1)
        l_1 = math.sqrt(h_1[0]*h_1[0] + h_1[1]*h_1[1] + h_1[2]*h_1[2])
        l_2 = math.sqrt(h_2[0]*h_2[0] + h_2[1]*h_2[1] + h_2[2]*h_2[2])
        # Curvature as the ratio between tangents angle change and arc length
        hold = l_1 + l_2
        if hold == 0.:
            curvatures[cont] = 0.
        else:
            curvatures[cont] = alpha / hold
        cont += 1

    return curvatures

# Computes local curvature of a curve in space
# curve: array of n points in a 2D space
# Return: an n-2 array with the local curvatures, if n <= 2 [0] is returned
def compute_plane_k(curve):

    # Initialization
    n_p = curve.shape[0]
    if n_p <= 2:
        return np.zeros(shape=1, dtype=float)
    curvatures = np.zeros(shape=n_p-2, dtype=float)

    # Loop for computing the curvatures
    cont = 0
    for i in range(1, curve.shape[0]-1):
        # Samples
        v_l1, v, v_p1 = curve[i-1], curve[i], curve[i+1]
        # Tangent vectors
        t_l1, t = v - v_l1, v_p1 - v
        m_t_l1 = math.sqrt(t_l1[0]*t_l1[0] + t_l1[1]*t_l1[1])
        if m_t_l1 <= 0:
            t_l1[0], t_l1[1] = 0., 0.
        else:
            t_l1 /= m_t_l1
        m_t = math.sqrt(t[0]*t[0] + t[1]*t[1])
        if m_t <= 0:
            t[0], t[1] = 0., 0.
        else:
            t /= m_t
        # Angle between tangents
        alpha = angle_2vec_2D(t_l1, t)
        # Arc length
        h_1 = .5 * (v_p1-v)
        h_2 = .5 * (v-v_l1)
        l_1 = math.sqrt(h_1[0]*h_1[0] + h_1[1]*h_1[1])
        l_2 = math.sqrt(h_2[0]*h_2[0] + h_2[1]*h_2[1])
        # Curvature as the ration between tangents angle change and arc length
        hold = l_1 + l_2
        if hold == 0.:
            curvatures[cont] = 0.
        else:
            curvatures[cont] = alpha / hold
        cont += 1

    return curvatures

# Smallest angle in radians between two vector in 3D space
def angle_2vec_3D(v_p, v_q):
    dt = v_p[0]*v_q[0] + v_p[1]*v_q[1] + v_p[2]*v_q[2]
    m_p = math.sqrt(v_p[0]*v_p[0] + v_p[1]*v_p[1] + v_p[2]*v_p[2])
    m_q = math.sqrt(v_q[0]*v_q[0] + v_q[1]*v_q[1] + v_q[2]*v_q[2])
    hold = m_p * m_q
    if hold == 0.:
        if dt >= 0.:
            hold = 1.
        else:
            hold = -1.
    else:
        hold = dt / hold
    if hold < -1.:
        hold = -1.
    elif hold > 1.:
        hold = 1.
    return math.acos(hold)

# Smallest angle in radians between two vector in 2D space
def angle_2vec_2D(v_p, v_q):
    dt = v_p[0]*v_q[0] + v_p[1]*v_q[1]
    m_p = math.sqrt(v_p[0]*v_p[0] + v_p[1]*v_p[1])
    m_q = math.sqrt(v_q[0]*v_q[0] + v_q[1]*v_q[1])
    hold = m_p * m_q
    if hold == 0.:
        if dt >= 0.:
            hold = 1.
        else:
            hold = -1.
    else:
        hold = dt / hold
    if hold < -1.:
        hold = -1.
    elif hold > 1.:
        hold = 1.
    return math.acos(hold)

def closest_points(point, points, nn=1):
    """
    Find the closest points to an input reference by Euclidean distance
    :param point: Input reference point
    :param points: Array of points to look for the closest neighbour
    :param nn: (default 1) number of neighbours to look for
    :return: An array with the closest points found
    """

    eu_dsts = point - points
    eu_dsts = np.sqrt((eu_dsts * eu_dsts).sum(axis=1))
    n_ids = np.argsort(eu_dsts)
    out_points = np.zeros(shape=(nn, 3))
    for i in range(nn):
        out_points[i] = points[n_ids[i], :]
    return out_points

# Flip cloud of points coordinates along specified dimensions
# dim_id: dimensions index, it must in [0, 1, ..., nd]
def flip_cloud(cloud, dim_id=0):
    mx, mn = cloud[:, dim_id].max(), cloud[:, dim_id].min()
    hold = np.zeros(shape=cloud.shape, dtype=cloud.dtype)
    for dim in range(cloud.shape[1]):
        if dim == dim_id:
            hold[:, dim] = (mx - cloud[:, dim]) + mn
        else:
            hold[:, dim] = cloud[:, dim]
    return hold

# Computes normal vector to a 3d point cloud (no orientation information)
# cloud: numpy array with shape [n, 3], n-points and 3-dimensions
# weights: point weights, if None (default) all points are weighted equally
def normal3d_point_cloud(cloud, weights=None):

    # Initialization
    k = cloud.shape[0]
    k_inv = 1. / float(k)
    if weights is not None:
        h_weights = lin_map(weights, lb=1., ub=0)
        w_s = h_weights.sum()
        if w_s > 0:
            n_weights = h_weights / w_s
    pc = k_inv * np.sum(cloud, axis=0)
    pm_00, pm_11, pm_22 = pc[0]*pc[0], pc[1]*pc[1], pc[2]*pc[2]
    pm_01, pm_02, pm_12 = pc[0]*pc[1], pc[0]*pc[2], pc[1]*pc[2]
    P = np.zeros(shape=(3, 3), dtype=float)
    P[0][0], P[0][1], P[0][2] = pm_00, pm_01, pm_02
    P[1][0], P[1][1], P[1][2] = pm_01, pm_11, pm_12
    P[2][0], P[2][1], P[2][2] = pm_02, pm_12, pm_22

    # Covariance matrix
    M = np.zeros(shape=(3, 3), dtype=float)
    if weights is None:
        for i in range(k):
            p = cloud[i, :]
            p_00, p_11, p_22 = p[0]*p[0], p[1]*p[1], p[2]*p[2]
            p_01, p_02, p_12 = p[0]*p[1], p[0]*p[2], p[1]*p[2]
            M[0][0] += p_00
            M[0][1] += p_01
            M[0][2] += p_02
            M[1][0] += p_01
            M[1][1] += p_11
            M[1][2] += p_12
            M[2][0] += p_02
            M[2][1] += p_12
            M[2][2] += p_22
        M = k_inv*M - P
    else:
        H = np.zeros(shape=(3, 3), dtype=float)
        for i in range(k):
            p = cloud[i, :]
            p_00, p_11, p_22 = p[0]*p[0], p[1]*p[1], p[2]*p[2]
            p_01, p_02, p_12 = p[0]*p[1], p[0]*p[2], p[1]*p[2]
            H[0][0] = p_00
            H[0][1] = p_01
            H[0][2] = p_02
            H[1][0] = p_01
            H[1][1] = p_11
            H[1][2] = p_12
            H[2][0] = p_02
            H[2][1] = p_12
            H[2][2] = p_22
            M += (n_weights[i] * H)
        M -= P

    # Eigen-problem
    W, V = np.linalg.eig(M)

    # Finding 'shortest' eigenvector
    return V[np.argsort(W*W)[0], :]

# Evaluates a 3x3 vector field taking into the account the border effects
# field: the vector filed [X, Y, Z, 3]
# point: point location
# Returns: 3D vector as numpy array
def evaluate_field_3x3(field, point):
    # Check border
    x, y, z = int(round(point[0])), int(round(point[1])), int(round(point[2]))
    n_x, n_y, n_z = field.shape[0], field.shape[1], field.shape[2]
    if x <= 0:
        x = 0
    elif x >= n_x:
        x = n_x - 1
    if y <= 0:
        y = 0
    elif y >= n_y:
        y = n_y - 1
    if z <= 0:
        z = 0
    elif z >= n_z:
        z = n_z - 1
    # Evaluation
    return field[x, y, z, :]

# Contrast Limited Adaptive Histogram Equalization (CLAHE) on a onedimensional array
# A. M. Rez, Journal of VLSI Signal Processing 30, 35-44, 2004
# array: input array
# N: number of grayscales (default 256)
# clip_f: clipping factor in percentage (default 100)
# s_max: maximum slope (default 4)
# Returns: return the bins of input and the transformation function (t[n]=trans[x[n]])
def clahe_array(array, N=256, clip_f=100, s_max=4):

    # Initialization
    M, N_f, clip_f, s_max = array.shape[0], float(N), float(clip_f), float(s_max)
    beta = (float(M)/N_f) * (1+(clip_f/100)*(s_max-1))

    # Compute histogram
    hist, x = np.histogram(array, bins=N)

    # Correct histogram
    ex = 0.
    for i in range(N):
        if hist[i] > beta:
            ex += (hist[i] - beta)
            hist[i] = beta
    m = ex / N_f
    for i in range(N):
        if hist[i] < (beta-m):
            hist[i] += m
            ex -= m
        elif hist[i] < beta:
            ex += (hist[i] - beta)
            hist[i] = beta
    while ex > 0:
        for i in range(N):
            if ex > 0:
                if hist[i] < beta:
                    hist[i] += 1
                    ex -= 1

    # Compute CDF
    x_trans = .5 * (x[1:]+x[:-1])
    try:
        return ((N_f-1.) / M) * np.cumsum(hist), x_trans
    except ZeroDivisionError:
        print('WARNING (clahe_array): CDF is zero!')
        return np.zeros(shape=len(hist), dtype=hist.dtype), x_trans

# Equal to clahe_array but now only transformation array is returned whose range are [0,N-1] and has
# N elements
def clahe_array2(array, N=256, clip_f=100, s_max=4):

    # Initialization
    M, N_f, clip_f, s_max = float(array.shape[0]), float(N), float(clip_f), float(s_max)
    beta = (M/N_f) * (1+(clip_f/100)*(s_max-1))

    # Compute histogram
    hist, _ = np.histogram(array+1, bins=np.arange(N+1))

    # Correct histogram
    ex = 0.
    for i in range(N):
        if hist[i] > beta:
            ex += (hist[i] - beta)
            hist[i] = beta
    m = ex / N_f
    for i in range(N):
        if hist[i] < (beta-m):
            hist[i] += m
            ex -= m
        elif hist[i] < beta:
            ex += (hist[i] - beta)
            hist[i] = beta
    while ex > 0:
        for i in range(N):
            if ex > 0:
                if hist[i] < beta:
                    hist[i] += 1
                    ex -= 1

    # Compute CDF
    try:
        return  ((N_f-1.) / M) * np.cumsum(hist)
    except ZeroDivisionError:
        print('WARNING (clahe_array): CDF is zero!')
        return np.zeros(shape=len(hist), dtype=hist.dtype)

# Compute center of gravity from point coordinates in 2D
def cgravity_points_2d(points):
    m00, m01, m10 = .0, .0, .0
    for point in points:
        m00 += 1.
        m01 += point[1]
        m10 += point[0]
    return np.asarray((m10/m00, m01/m00))

# Applies digital low pass Butterworth filter to a tomogram
# tomo: input tomogram 3D
# cl: cut of frequency (voxels)
# n: filter order (default 5)
# fs: if fs is True, default False then the filter is return in the fourier space (zero shifted) instead the filtered version of
#     the input tomogram
# Returns: the filtered tomogram
def tomo_butter_low(tomo, cl, n=5, fs=False):

    # Initialization
    if (not isinstance(tomo, np.ndarray)) and (len(tomo.shape) != 3):
        error_msg = 'Input mask must be a 3D numpy array with odd dimensions.'
        raise pexceptions.PySegInputError(expr='tomo_butter_low', msg=error_msg)
    if cl <= 0:
        error_msg = 'Cut off frequency must be greater than zero, current: ' + str(cl)
        raise pexceptions.PySegInputError(expr='tomo_butter_low', msg=error_msg)
    nx, ny, nz = (tomo.shape[0]-1)*.5, (tomo.shape[1]-1)*.5, (tomo.shape[2]-1)*.5

    # Building the filter
    if (nx % 1) == 0:
        arr_x = np.concatenate((np.arange(-nx, 0, 1), np.arange(0, nx+1, 1)))
    else:
        if nx < 1:
            arr_x = np.arange(0, 1)
        else:
            nx = math.ceil(nx)
            arr_x = np.concatenate((np.arange(-nx, 0, 1), np.arange(0, nx, 1)))
    if (ny % 1) == 0:
        arr_y = np.concatenate((np.arange(-ny, 0, 1), np.arange(0, ny+1, 1)))
    else:
        if ny < 1:
            arr_y = np.arange(0, 1)
        else:
            ny = math.ceil(ny)
            arr_y = np.concatenate((np.arange(-ny, 0, 1), np.arange(0, ny, 1)))
    if (nz % 1) == 0:
        arr_z = np.concatenate((np.arange(-nz, 0, 1), np.arange(0, nz+1, 1)))
    else:
        if nz < 1:
            arr_z = np.arange(0, 1)
        else:
            nz = math.ceil(nz)
            arr_z = np.concatenate((np.arange(-nz, 0, 1), np.arange(0, nz, 1)))
    [X, Y, Z] = np.meshgrid(arr_x, arr_y, arr_z, indexing='ij')
    X = X.astype(np.float32, copy=False)
    Y = Y.astype(np.float32, copy=False)
    Z = Z.astype(np.float32, copy=False)
    R = np.sqrt(X*X + Y*Y + Z*Z)
    del X
    del Y
    del Z
    Hl = 1. / (1.+np.power(R/cl, 2*n))
    if fs:
        return Hl
    del R

    # disperse_io.save_numpy(Hl, '/home/martinez/workspace/disperse/data/marion/cluster/out/Hl.mrc')
    # hold = np.zeros(tomo.shape, dtype=np.float32)
    # hold[:, :, 0] = np.fft.ifft2(np.fft.fft2(tomo[:, :, 0]) * np.fft.fftshift(Hl[:, :, 0]))
    # disperse_io.save_numpy(np.abs(hold), '/home/martinez/workspace/disperse/data/marion/cluster/out/tomo.mrc')

    # Filtering
    return np.real(np.fft.ifftn(np.fft.fftn(tomo) * np.fft.fftshift(Hl)))

# Implements an algorithm for estimating signed distance form input segmentation slice by slice along Z axis
# IMPORTANT: only tested for smooth membranes
# tomo: input 3D array with the segmentation
# lbl: label for segmented region, if None (default) segmented region is every voxel >0
# res: voxel resolution, pixel/nm (default 1)
# del_b: if True (default) no reliable border segmentation are discarded (set to zero)
# mode_2d: if True (default False) the distances are computed in 2D slice by slice instead of 3D
# get_point: if True (default False) reference point for orientation is also returned
# set_point: if not None (default) sets the reference point
# Returns: a tomogram with the signed distance in nm
def signed_distance_2d(tomo, lbl=None, res=1, del_b=True, mode_2d=False, get_point=False, set_point=None):

    # Parsing
    if (not isinstance(tomo, np.ndarray)) or (len(tomo.shape) != 3):
        error_msg = 'Input tomogram must be a 3D numpy array.'
        raise pexceptions.PySegInputError(expr='signed_distance_2d', msg=error_msg)
    if lbl is None:
        tomo_h = np.invert(tomo > 0)
    else:
        tomo_h = np.invert(tomo == lbl)
    if set_point is not None:
        if (not hasattr(set_point, '__len__')) or (len(set_point) != 2):
            error_msg = 'Input reference point must be a 3-tuple.'
            raise pexceptions.PySegInputError(expr='signed_distance_2d', msg=error_msg)
    tomo_s = np.zeros(shape=tomo_h.shape, dtype=np.float32)

    # Coordinates system
    X, Y = np.meshgrid(np.arange(0, tomo_h.shape[0]), np.arange(0, tomo_h.shape[1]), indexing='ij')

    # Find fartest point in the middle slice
    if set_point is not None:
        pt_f = set_point
    else:
        nx2 = int(math.floor(tomo.shape[2] * .5))
        img_d = sp.ndimage.morphology.distance_transform_edt(tomo_h[:, :, nx2], return_indices=False)
        pt_f = np.unravel_index(img_d.argmax(), img_d.shape)

    # Loop on slices
    for z in range(tomo_h.shape[2]):

        # Slices witout segmentation region are not valid
        if tomo_h[:, :, z].min() == 0:

            # Unsigned 2D distance transform
            img_d, ids =  sp.ndimage.morphology.distance_transform_edt(tomo_h[:, :, z], return_indices=True)

            # Coordinates to sortest segmented points
            seg_X, seg_Y = ids[0].astype(np.float32), ids[1].astype(np.float32)

            # Compute normalized vector fields (just in valid regions)
            vfx_1, vfy_1 = X-seg_X, Y-seg_Y
            vfx_f, vfy_f = float(pt_f[0])-seg_X, float(pt_f[1])-seg_Y
            vf_1_m = np.sqrt(vfx_1*vfx_1 + vfy_1*vfy_1)
            vf_f_m = np.sqrt(vfx_f*vfx_f + vfy_f*vfy_f)
            id_m = (vf_1_m != 0) & (vf_f_m != 0)
            vf_1_m_c = vf_1_m[id_m]
            vfx_1_c = vfx_1[id_m]/vf_1_m_c
            vfy_1_c = vfy_1[id_m]/vf_1_m_c
            vf_f_m_c = vf_f_m[id_m]
            vfx_f_c = vfx_f[id_m]/vf_f_m_c
            vfy_f_c = vfy_f[id_m]/vf_f_m_c

            # Compute angle field (just dot product, arccos is redundant)
            dota = vfx_1_c*vfx_f_c + vfy_1_c*vfy_f_c

            # Add sign to angles
            hold = np.zeros(shape=(tomo_s.shape[0], tomo_s.shape[1]), dtype=tomo_s.dtype)
            hold[id_m] = np.sign(dota)
            tomo_s[:, :, z] = hold

    # Assign sign information to the 3D unsigned distance transform
    if mode_2d:
        img_d = np.zeros(shape=tomo_h.shape, dtype=np.float32)
        for z in range(tomo_h.shape[2]):
            h_tomo = tomo_h[:, :, z]
            h_img_d = sp.ndimage.morphology.distance_transform_edt(h_tomo, sampling=res, return_indices=False)
            h_img_d *= tomo_s[:, :, z]
            if del_b:
                h_dtp = sp.ndimage.morphology.distance_transform_edt(h_img_d>0, sampling=res, return_indices=False)
                h_dtn = sp.ndimage.morphology.distance_transform_edt(h_img_d<0, sampling=res, return_indices=False)
                h_dtz = np.abs(h_img_d)
                mask = (h_dtp<h_dtz) & (h_dtn<h_dtz)
                h_img_d[mask] = 0
            img_d[:, :, z] = h_img_d

    else:
        if del_b:
            h_img_d = sp.ndimage.morphology.distance_transform_edt(tomo_h, sampling=res, return_indices=False)
            h_img_d *= tomo_s
            h_dtp = sp.ndimage.morphology.distance_transform_edt(h_img_d>0, sampling=res, return_indices=False)
            h_dtn = sp.ndimage.morphology.distance_transform_edt(h_img_d<0, sampling=res, return_indices=False)
            h_dtz = np.abs(h_img_d)
            mask = (h_dtp<h_dtz) & (h_dtn<h_dtz)
            h_img_d[mask] = 0
            img_d = h_img_d
        else:
            img_d = sp.ndimage.morphology.distance_transform_edt(tomo_h, sampling=res, return_indices=False)

    if get_point:
        return img_d, pt_f
    else:
        return img_d


def rotate_3d_vector(vector, euler, deg=True, conv='mat'):
    """
    Rotates a vector from Euler angles using the TOM toolbox convention
    :param vector: 3d vector to rotate
    :param euler: euler angles as the form [phi, psi, theta]
    :param deg: if True (default) input angles in degrees, otherwise in radians
    :param conv: rotation convention, valid: 'mat' (Matlab or TOM, default) and 'relion'
    :return: the rotated vector as a numpy array
    """

    # Initialization
    if deg:
        phi, psi, theta = math.radians(euler[0]), math.radians(euler[1]), math.radians(euler[2])
    else:
        phi, psi, theta = euler[0], euler[1], euler[2]

    if conv == 'relion':
        M = rot_mat_relion(phi, psi, theta, deg=False)
        # rm00, rm01, rm02 = M[0, 0], M[0, 1], M[0, 2]
        # rm10, rm11, rm12 = M[1, 0], M[1, 1], M[1, 2]
        # rm20, rm21, rm22 = M[2, 0], M[2, 1], M[2, 2]
        rm00, rm01, rm02 = M[0, 0], M[1, 0], M[2, 0]
        rm10, rm11, rm12 = M[0, 1], M[1, 1], M[2, 1]
        rm20, rm21, rm22 = M[0, 2], M[1, 2], M[2, 2]
    else:
        # Building rotation matrix
        rm00 = math.cos(psi)*math.cos(phi) - math.cos(theta)*math.sin(psi)*math.sin(phi)
        rm10 = math.sin(psi)*math.cos(phi) + math.cos(theta)*math.cos(psi)*math.sin(phi)
        rm20 = math.sin(theta)*math.sin(phi)
        rm01 = math.cos(psi)*math.sin(phi) - math.cos(theta)*math.sin(psi)*math.cos(phi)
        rm11 = -math.sin(psi)*math.sin(phi) + math.cos(theta)*math.cos(psi)*math.cos(phi)
        rm21 = math.sin(theta)*math.cos(phi)
        rm02 = math.sin(theta)*math.sin(psi)
        rm12 = -math.sin(theta)*math.cos(psi)
        rm22 = math.cos(theta)

    # Rotation
    rot = np.zeros(shape=3, dtype=np.float32)
    rot[0] = rm00*vector[0] + rm01*vector[1] + rm02*vector[2]
    rot[1] = rm10*vector[0] + rm11*vector[1] + rm12*vector[2]
    rot[2] = rm20*vector[0] + rm21*vector[1] + rm22*vector[2]

    return rot

# Returns input variable x if x\in[lb,ub], otherwise lb if x<lb or ub if x>ub is returned
def set_in_range(x, lb, ub):
    if x < lb:
        return lb
    elif x > ub:
        return ub
    else:
        return x

# Creates 3D rotation matrix as TOM convention from euler angles (phi, psi, the)
# phi, psi, the: the Euler angles
# deg: if True (default) angles are computed in degrees, otherwise radians
def rot_mat(phi, psi, the, deg=True):
    if deg:
        phi, psi, the = math.radians(phi), math.radians(psi), math.radians(the)
    rot = np.zeros(shape=(3, 3), dtype=np.float32)
    rot[0][0] = math.cos(psi)*math.cos(phi) - math.cos(the)*math.sin(psi)*math.sin(phi)
    rot[0][1] = math.sin(psi)*math.cos(phi) + math.cos(the)*math.cos(psi)*math.sin(phi)
    rot[0][2] = math.sin(the)*math.sin(phi)
    rot[1][0] = -math.cos(psi)*math.sin(phi) - math.cos(the)*math.sin(psi)*math.cos(phi)
    rot[1][1] = -math.sin(psi)*math.sin(phi) + math.cos(the)*math.cos(psi)*math.cos(phi)
    rot[1][2] = math.sin(the)*math.cos(phi)
    rot[2][0] = math.sin(the)*math.sin(psi)
    rot[2][1] = -math.sin(the)*math.cos(psi)
    rot[2][2] = math.cos(the)
    return np.mat(rot)

def rot_mat_pytom(phi=0, psi=0, the=0):
    """Creates a rotation matrix using ZXZ convention (phi: z1, the: x, psi: z2).

    @param phi: 1st rotate around Z axis, in degree.
    @param psi: 3rd rotate around Z axis, in degree.
    @param the: 2nd rotate around X axis, in degree.
    @return: a rotation matrix.
    """

    # Transfer the angle to Euclidean
    phi = -float(phi) * np.pi / 180.0
    the = -float(the) * np.pi / 180.0
    psi = -float(psi) * np.pi / 180.0
    sin_alpha = np.sin(phi)
    cos_alpha = np.cos(phi)
    sin_beta = np.sin(the)
    cos_beta = np.cos(the)
    sin_gamma = np.sin(psi)
    cos_gamma = np.cos(psi)

    # Calculate inverse rotation matrix
    R = np.zeros((3, 3), dtype='float32')

    R[0, 0] = cos_alpha * cos_gamma - cos_beta * sin_alpha \
        * sin_gamma
    R[1, 0] = -cos_alpha * sin_gamma - cos_beta * sin_alpha \
        * cos_gamma
    R[2, 0] = sin_beta * sin_alpha

    R[0, 1] = sin_alpha * cos_gamma + cos_beta * cos_alpha \
        * sin_gamma
    R[1, 1] = -sin_alpha * sin_gamma + cos_beta * cos_alpha \
        * cos_gamma
    R[2, 1] = -sin_beta * cos_alpha

    R[0, 2] = sin_beta * sin_gamma
    R[1, 2] = sin_beta * cos_gamma
    R[2, 2] = cos_beta

    return R


def rotate3d_pytom(data, phi=0, psi=0, the=0, center=None, order=2):
    """Rotate a 3D data using ZXZ convention (phi: z1, the: x, psi: z2).

    @param data: data to be rotated.
    @param phi: 1st rotate around Z axis, in degree.
    @param psi: 3rd rotate around Z axis, in degree.
    @param the: 2nd rotate around X axis, in degree.
    @param center: rotation center.

    @return: the data after rotation.
    """
    # Figure out the rotation center
    if center is None:
        cx = data.shape[0] / 2
        cy = data.shape[1] / 2
        cz = data.shape[2] / 2
    else:
        assert len(center) == 3
        (cx, cy, cz) = center

    # Transfer the angle to Euclidean
    phi = -float(phi) * np.pi / 180.0
    the = -float(the) * np.pi / 180.0
    psi = -float(psi) * np.pi / 180.0
    sin_alpha = np.sin(phi)
    cos_alpha = np.cos(phi)
    sin_beta = np.sin(the)
    cos_beta = np.cos(the)
    sin_gamma = np.sin(psi)
    cos_gamma = np.cos(psi)

    # Calculate inverse rotation matrix
    Inv_R = np.zeros((3, 3), dtype='float32')

    Inv_R[0, 0] = cos_alpha * cos_gamma - cos_beta * sin_alpha \
        * sin_gamma
    Inv_R[0, 1] = -cos_alpha * sin_gamma - cos_beta * sin_alpha \
        * cos_gamma
    Inv_R[0, 2] = sin_beta * sin_alpha

    Inv_R[1, 0] = sin_alpha * cos_gamma + cos_beta * cos_alpha \
        * sin_gamma
    Inv_R[1, 1] = -sin_alpha * sin_gamma + cos_beta * cos_alpha \
        * cos_gamma
    Inv_R[1, 2] = -sin_beta * cos_alpha

    Inv_R[2, 0] = sin_beta * sin_gamma
    Inv_R[2, 1] = sin_beta * cos_gamma
    Inv_R[2, 2] = cos_beta

    from scipy import mgrid
    grid = mgrid[-cx:data.shape[0]-cx, -cy:data.shape[1]-cy, -cz:data.shape[2]-cz]
    temp = grid.reshape((3, grid.size / 3))
    temp = np.dot(Inv_R, temp)
    grid = np.reshape(temp, grid.shape)
    grid[0] += cx
    grid[1] += cy
    grid[2] += cz

    # Interpolation
    from scipy.ndimage import map_coordinates
    d = map_coordinates(data, grid, order=order)

    return d

# Creates 3D rotation matrix in intrisic ZY'Z'' convention
# rot, tilt, psi: the Euler angles
# deg: if True (default) angles are computed in degrees, otherwise radians
def rot_mat_zyz(rot, tilt, psi, deg=True):
    if deg:
        rot, tilt, psi = math.radians(rot), math.radians(tilt), math.radians(psi)
    mt = np.zeros(shape=(3, 3), dtype=np.float32)
    c1, s1 = math.cos(rot), math.sin(rot)
    c2, s2 = math.cos(tilt), math.sin(tilt)
    c3, s3 = math.cos(psi), math.sin(psi)
    mt[0][0] = c1*c2*c3 - s1*s3
    mt[0][1] = -c3*s1 - c1*c2*s3
    mt[0][2] = c1*s2
    mt[1][0] = c1*s3 + c2*c3*s1
    mt[1][1] = c1*c3 - c2*s1*s3
    mt[1][2] = s1*s2
    mt[2][0] = -c3*s2
    mt[2][1] = s2*s3
    mt[2][2] = c2
    return np.mat(mt)

# Creates 3D rotation matrix according Relion convention
# TODO: According to Relion's documentation they have ZY'Z'' convention but this is not what I've found when digged
# TODO: into their code. This is a direct translation from their code https://github.com/jjcorreao/relion/blob/master/relion-1.3/src/euler.cpp
# rot, tilt, psi: the Euler angles
# deg: if True (default) angles are computed in degrees, otherwise radians
def rot_mat_relion(rot, tilt, psi, deg=True):

    # XMIPP doc
    if deg:
        rot, tilt, psi = math.radians(rot), math.radians(tilt), math.radians(psi)
    mt = np.zeros(shape=(3, 3), dtype=np.float32)
    ca, sa = math.cos(rot), math.sin(rot)
    cb, sb = math.cos(tilt), math.sin(tilt)
    cg, sg = math.cos(psi), math.sin(psi)
    cc, cs = cb*ca, cb*sa
    sc, ss = sb*ca, sb*sa

    # mt[0][0] = cg*cc - sg*sa
    # mt[0][1] = cg*cs + sg*ca
    # mt[0][2] = -cg*sb
    # mt[1][0] = -sg*cc - cg*sa
    # mt[1][1] = -sg*cs + cg*ca
    # mt[1][2] = sg*sb
    # mt[2][0] = sc
    # mt[2][1] = ss
    # mt[2][2] = cb

    # XMIPP doc inverted
    mt[0][0] = cg*cc - sg*sa
    mt[1][0] = cg*cs + sg*ca
    mt[2][0] = -cg*sb
    mt[0][1] = -sg*cc - cg*sa
    mt[1][1] = -sg*cs + cg*ca
    mt[2][1] = sg*sb
    mt[0][2] = sc
    mt[1][2] = ss
    mt[2][2] = cb

    return np.mat(mt)

    # # tom_eulerconver_xmipp.m code
    # if deg:
    #     hrot, htilt, hpsi = math.radians(-psi), math.radians(-tilt), math.radians(-rot)
    # else:
    #     hrot, htilt, hpsi = -psi, -tilt, -rot

    # mt1 = np.matrix([[math.cos(hrot), -math.sin(hrot), 0.],
    #                  [math.sin(hrot), math.cos(hrot), 0.],
    #                  [0., 0., 1.]])
    # mt2 = np.matrix([[math.cos(htilt), 0., math.sin(htilt)],
    #                  [0., 1., 0.],
    #                  [-math.sin(htilt), 0., math.cos(htilt)]])
    # mt3 = np.matrix([[math.cos(hpsi), -math.sin(hpsi), 0.],
    #                  [math.sin(hpsi), math.cos(hpsi), 0.],
    #                  [0., 0., 1.]])

    # return mt1 * mt2 * mt3

# Returns Euler angles from rotation matrix according to Relion's convention
# TODO: According to Relion's documentation they have ZY'Z'' convention but this is not what I've found when digged
# TODO: into their code. This is a direct translation from their code https://github.com/jjcorreao/relion/blob/master/relion-1.3/src/euler.cpp
# A: rotation matrix
# deg: if True (default) angles are returned in degrees, otherwise in radians
# Ruturns: an array with rot, tilt and psi angles
def rot_mat_eu_relion(A, deg=True):

    # XMIPP doc inverted
    A = np.copy(A.T)

    # Input parsing
    if (A.shape[0] != 3) or (A.shape[1] != 3):
        error_msg = 'Rotation matrix must have a shape 3x3.'
        raise pexceptions.PySegInputError(expr='rot_mat_relion', msg=error_msg)

    abs_sb = math.sqrt(A[0, 2]*A[0, 2] + A[1, 2] * A[1, 2])

    if abs_sb > 16*FLT_EPSILON:
        gamma = math.atan2(A[1, 2], -A[0, 2])
        alpha = math.atan2(A[2, 1], A[2, 0])
        if math.fabs(math.sin(gamma)) < FLT_EPSILON:
            sign_sb = np.sign(-A[0, 2] / math.cos(gamma))
        else:
            if math.sin(gamma) > 0:
                sign_sb = np.sign(A[1, 2])
            else:
                sign_sb = -1. * np.sign(A[1, 2])
        beta = math.atan2(sign_sb*abs_sb, A[2, 2])
    else:
        if np.sign(A[2, 2]) > 0:
            # Let's consider the matrix as a rotation around Z
            alpha = 0.
            beta  = 0.
            gamma = math.atan2(-A[1, 0], A[0, 0])
        else:
            alpha = 0.
            beta = np.pi
            gamma = math.atan2(A[1, 0], -A[0, 0])

    if deg:
        return math.degrees(alpha), math.degrees(beta), math.degrees(gamma)
    else:
        return alpha, beta, gamma

def rot_mat_2_vectors(v_a, v_b):
    """
    Compute the rotation matrix to transform v_a into v_b
    :param v_a: origin vector
    :param v_b: destination vector
    :return: the rotation matrix
    """

    # Input parsing
    n_a, n_b = math.sqrt((v_a * v_a).sum()), math.sqrt((v_b * v_b).sum())
    if (n_a <= 0) or (n_b <= 0):
        raise ValueError
    v_a_n, v_b_n = v_a / n_a, v_b / n_b

    # Computing the matrix
    v = np.cross(v_a_n, v_b_n)
    c = np.dot(v_a_n, v_b_n)
    c2 = 1. / (1. - c)
    I = np.identity(3)
    V = np.matrix([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])
    R = I + V + V*V*c2

    return R

# Computes quaternion from an input rotation matrix
# Code extracted: from http://www.lfd.uci.edu/~gohlke/code/transformations.py.html
# rot: rotation numpy matrix
# is_prec: if True (default False) the input matrix is assumed to be a precise rotation
#          matrix and a faster algorithm is used.
def rot_to_quat(rot, is_prec=False):

    M = np.array(rot, dtype=np.float64, copy=False)[:4, :4]

    if is_prec:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])

    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                         [m01+m10,     m11-m00-m22, 0.0,         0.0],
                         [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                         [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]

    if q[0] < 0.0:
        np.negative(q, q)

    return q

# From an input quaternion computes its rotation matrix
# Code extracted: from http://www.lfd.uci.edu/~gohlke/code/transformations.py.html
# quat: quaternion (4-tuple numpy array)
# is_prec: if True (default False) the input matrix is assumed to be a precise rotation
#          matrix and a faster algorithm is used.
def quat_to_rot(quat):
    q = np.array(quat, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < ZERO_EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([[1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0]],
                     [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0]],
                     [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2]]])

# Generate a pseudo-random uniformly distributed unit quaternion (4-tuple as numpy array)
# Code from: J.J. Kuffern, ICRA 2004
def rand_quat():
    s = rnd.random()
    s1, s2 = math.sqrt(1.-s), math.sqrt(s)
    rho1, rho2 = 2.*np.pi*rnd.random(), 2.*np.pi*rnd.random()
    w, x, y, z = math.cos(rho2)*s2, math.sin(rho1)*s1, math.cos(rho1)*s1, math.sin(rho2)*s2
    return np.asarray((w, x, y, z), dtype=np.float32)

# From an input Euler angles and translation vector applies transformation: 1st rotation, 2nd translation
# So as to speed up its performance no parsing is applying to inputs
# Formula: Xp = R*X + T
# Xs: input coordinates ((3,1) shaped array is assumed)
# phi|psi|the: euler angles
# T: translation vector ((3,1) shaped array is assumed)
# deg: if True (default) angles are computed in degrees, otherwise radians
def rot_and_trans(Xs, phi, psi, the, T, deg=True):
    rot = rot_mat(phi, psi, the, deg)
    return rot*np.mat(Xs) + T

# Inverse of rot_and_trans(), so inv_rot_and_trans(rot_and_trans(Xs,...),...) == Xs
# Formula: X = R^(T)*Xp + R^(T)*T   : by rotation matrix properties R^(T)==R^(-1)
# Xs: input coordinates ((3,1) shaped array is assumed)
# phi|psi|the: euler angles
# T: translation vector ((3,1) shaped array is assumed)
# deg: if True (default) angles are computed in degrees, otherwise radians
def inv_rot_and_trans(Xs, phi, psi, the, T, deg=True):
    rot_inv = rot_mat(phi, psi, the, deg).T
    return rot_inv*np.mat(Xs) - rot_inv*T

# Rotates a tomogram according to the specified Euler angles
# tomo: the tomo numpy.ndarray (it must be 3D) to rotate
# deg: if True (default) angles are in degrees
# eu_angs: Euler angles in convention phi, psi, the
# center: center for rotation, if None (default) the tomogram center is used
# conv: rotation convention, valid: 'mat' (Matlab, default), 'zyz' intrinsic and 'relion'
# active: it True (default) then rotation is active (direct), otherwise it is passive (inverse)
# The rest of parameters are equivalent to scipy.ndimage.interpolation.map_coordinates
def tomo_rot(tomo, eu_angs, conv='mat', active=True, deg=True, center=None, order=3, mode='constant', cval=0.0, prefilter=True):

    # Input parsing
    if (not isinstance(tomo, np.ndarray)) or (len(tomo.shape) != 3):
        error_msg = 'Input tomogram must be a 3D numpy array.'
        raise pexceptions.PySegInputError(expr='tomo_rot', msg=error_msg)
    if (not hasattr(eu_angs, '__len__')) or (len(eu_angs) != 3):
        error_msg = 'Euler angles must be 3-tubple.'
        raise pexceptions.PySegInputError(expr='tomo_rot', msg=error_msg)
    if center is None:
        if conv == 'relion':
            center = np.round(.5 * np.asarray(tomo.shape, dtype=np.float32))
        else:
            center = .5 * (np.asarray(tomo.shape, dtype=np.float32) - 1)
    else:
        if (not hasattr(center, '__len__')) or (len(center) != 3):
            error_msg = 'Center must be 3-tubple.'
            raise pexceptions.PySegInputError(expr='tomo_rot', msg=error_msg)
    if (conv != 'mat') and (conv != 'zyz') and (conv != 'relion'):
        error_msg = 'Input convention ' + conv + ' is not valid.'
        raise pexceptions.PySegInputError(expr='tomo_rot', msg=error_msg)
    ts = tomo.size

    # Compute rotation matrix coeficients (first index: rows, second index: comlumn)
    if conv == 'mat':
        R = rot_mat(eu_angs[0], eu_angs[1], eu_angs[2], deg=deg)
    elif conv == 'zyz':
        R = rot_mat_zyz(eu_angs[0], eu_angs[1], eu_angs[2], deg=deg)
    elif conv == 'relion':
        R = rot_mat_relion(eu_angs[0], eu_angs[1], eu_angs[2], deg=deg)
    if not active:
        R = R.T

    # Compute grid
    X, Y, Z = np.meshgrid(np.arange(tomo.shape[0]), np.arange(tomo.shape[1]), np.arange(tomo.shape[2]),
                          indexing='xy')

    # From indices to coordinates
    X, Y, Z = (X-center[1]).astype(np.float32), (Y-center[0]).astype(np.float32), (Z-center[2]).astype(np.float32)

    # Grid rotation
    Xr = X*R[0, 0] + Y*R[0, 1] + Z*R[0, 2]
    Yr = X*R[1, 0] + Y*R[1, 1] + Z*R[1, 2]
    Zr = X*R[2, 0] + Y*R[2, 1] + Z*R[2, 2]

    # From coordinates to indices
    X, Y, Z = Xr+center[1], Yr+center[0], Zr+center[2]

    # Re-mapping (interpolation)
    inds = np.zeros(shape=(3, ts), dtype=np.float32)
    inds[0, :], inds[1, :], inds[2, :] = X.reshape(ts), Y.reshape(ts), Z.reshape(ts)
    tomo_r = sp.ndimage.interpolation.map_coordinates(tomo, inds, order=order, mode=mode,
                                                      cval=cval, prefilter=prefilter)

    return tomo_r.reshape(tomo.shape)


# Tomogram shift in Fourier space
# tomo: the tomo numpy.ndarray (it must be 3D) to shift
# shift: 3-tuple with the shifting for every dimension
def tomo_shift(tomo, shift):

    # Input parsing
    if (not isinstance(tomo, np.ndarray)) or (len(tomo.shape) != 3):
        error_msg = 'Input tomogram must be a 3D numpy array.'
        raise pexceptions.PySegInputError(expr='tomo_shift', msg=error_msg)
    if (not hasattr(shift, '__len__')) or (len(shift) != 3):
        error_msg = 'shift must be 3-tubple.'
        raise pexceptions.PySegInputError(expr='tomo_shift', msg=error_msg)
    dx, dy, dz = float(tomo.shape[0]), float(tomo.shape[1]), float(tomo.shape[2])
    dx2, dy2, dz2 = math.floor(.5*dx), math.floor(.5*dy), math.floor(.5*dz)
    if isinstance(shift, np.ndarray):
        delta = np.copy(shift)
    else:
        delta = np.asarray(shift, dtype=np.float32)
    dim = np.asarray((dx, dy, dz), dtype=np.float32)

    # Generating the grid
    x_l, y_l, z_l = -dx2, -dy2, -dz2
    x_h, y_h, z_h = -dx2+dx, -dy2+dy, -dz2+dz
    X, Y, Z = np.meshgrid(np.arange(x_l, x_h), np.arange(y_l, y_h), np.arange(z_l, z_h), indexing='xy')

    # Check for trivial dimensions
    ids = np.where(dim <= 1)[0]
    delta[ids] = 0

    # Shift grid in Fourier space
    delta[0], delta[1], delta[2] = delta[0]/dx, delta[1]/dy, delta[2]/dz
    X = np.fft.ifftshift(delta[0]*X + delta[1]*Y + delta[2]*Z)
    del Y, Z

    # Tomogram shifting in Fourier space
    j = np.complex(0, 1)
    img = np.fft.fftn(tomo)
    return np.real(np.fft.ifftn(img * np.exp(-2.*np.pi*j*X)))

# Coordinates suppression so as to ensure a minimum euclidean distance among them, it returns a list
# with the indices of the coordinates to delete
# coords: iterable of coordinates
# scale: scale suppression (euclidean distance)
# weights: array with the points weights of every coordinate to give priorities (default None)
# Returns: a list with the indices of the coordinates to delete
def coords_scale_supression(coords, scale, weights=None):

    # Initialization
    del_l = list()
    del_lut = np.zeros(shape=len(coords), dtype=bool)

    coords = np.asarray(coords, dtype=np.float32)
    if weights is None:
        s_ids = np.arange(len(coords))
    else:
        s_ids = np.argsort(weights)[::-1]

    # Coordinates loop
    for s_id in s_ids:
        # Not process already deleted coords
        if not del_lut[s_id]:
            eu_dsts = coords[s_id, :] - coords
            eu_dsts = np.sqrt((eu_dsts * eu_dsts).sum(axis=1))
            # Finding closest points
            n_ids = np.where(eu_dsts < scale)[0]
            # Add to deletion list
            for idx in n_ids:
                if (idx != s_id) and (not del_lut[idx]):
                    del_l.append(idx)
                    del_lut[idx] = True

    return del_l

# Unroll an angle [-infty, infty] to fit range [-180, 180] (or (-pi, pi) in radians)
# angle: input angle
# deg: if True (defult) the angle is in degrees, otherwise in radians
def unroll_angle(angle, deg=True):
    fang = float(angle)
    if deg:
        mx_ang, mx_ang2 = 360., 180.
    else:
        mx_ang, mx_ang2 = 2*np.pi, np.pi
    ang_mod, ang_sgn = np.abs(fang), np.sign(fang)
    ur_ang = ang_mod % mx_ang
    if ur_ang > mx_ang2:
        return -1. * ang_sgn * (mx_ang - ur_ang)
    else:
        return ang_sgn * ur_ang

# Clean an directory contents (directory is preserved)
# dir: directory path
def clean_dir(dir):
    for root, dirs, files in os.walk(dir):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))


# Sort dict1 input dictionary using dict2.values() (must be scalars)
# Returns the sorted list of keys and values from dict1
def sort_dict(dict1, dict2, reverse=False):
    s_keys, s_values = list(), list()
    for h_key in sorted(list(dict2.keys()), key=dict2.get, reverse=reverse):
        try:
            s_values.append(dict1[h_key])
            s_keys.append(h_key)
        except KeyError:
            continue
    return s_keys, s_values


def gen_wedge_vol(shape, semi_ang, rot_ang, dtype=bool):
    """
    Generates an volume with a wedge
    :param shape: 3-tuple with the input sizes
    :param semi_ang: wedge semi-angle in degrees
    :param rot_ang: rotation angle (around Z axis) for the wedge in degrees
    :param dtype: data type (default bool)
    :return: a volume with the specified shape and wedge with semi-angle 'ang'
    """

    # Input parsing
    ang = (float(semi_ang)*np.pi) / 180.

    # Creating the grid
    dx, dy, dz = float(shape[0]), float(shape[1]), float(shape[2])
    dx2, dy2, dz2 = math.floor(.5 * dx), math.floor(.5 * dy), math.floor(.5 * dz)
    x_l, y_l, z_l = -dx2, -dy2, -dz2
    x_h, y_h, z_h = -dx2 + dx, -dy2 + dy, -dz2 + dz
    X, Y, Z = np.meshgrid(np.arange(x_l, x_h), np.arange(y_l, y_h), np.arange(z_l, z_h), indexing='xy')
    del Y

    # Creating the wedge
    wedge = np.one(shape=shape, dtype=dtype)
    Ids = np.tan(ang) > (np.abs(X) / np.abs(Z))
    wedge[Ids] = 0

    # Rotation angle
    wedge = tomo_rot(wedge, (rot_ang, 0, 0))

    return wedge


def tomo_binning(tomo, nbin):
    """
    Bin N times an input tomogram
    :param tomo: tomogram to bin
    :param nbin: number of binnings
    :return: a binned version of the input tomogram
    """
    hold_tomo = np.copy(tomo)
    for i in range(nbin):
        bin_tomo = 0.125 * (hold_tomo[0:hold_tomo.shape[0]-1:2, 0:hold_tomo.shape[1]-1:2, 0:hold_tomo.shape[2]-1:2] +
                   hold_tomo[0:hold_tomo.shape[0]-1:2, 1:hold_tomo.shape[1]:2, 0:hold_tomo.shape[2]-1:2] +
                   hold_tomo[1:hold_tomo.shape[0]:2, 0:hold_tomo.shape[1]-1:2, 0:hold_tomo.shape[2]-1:2] +
                   hold_tomo[1:hold_tomo.shape[0]:2, 1:hold_tomo.shape[1]:2, 0:hold_tomo.shape[2]-1:2] +
                   hold_tomo[0:hold_tomo.shape[0]-1:2, 0:hold_tomo.shape[1]-1:2, 1:hold_tomo.shape[2]:2] +
                   hold_tomo[0:hold_tomo.shape[0]-1:2, 1:hold_tomo.shape[1]:2, 1:hold_tomo.shape[2]:2] +
                   hold_tomo[1:hold_tomo.shape[0]:2, 0:hold_tomo.shape[1]-1:2, 1:hold_tomo.shape[2]:2] +
                   hold_tomo[1:hold_tomo.shape[0]:2, 1:hold_tomo.shape[1]:2, 1:hold_tomo.shape[2]:2])
        hold_tomo = bin_tomo
    return bin_tomo


def geo_point_dst_sphere(point, samples, s_cent, s_rad):
    """
    Measures the geodesic distances to a point on sphere (normal vectors method)
    :param point: point coordinates as n1array
    :param samples: coordinates of the samples as an ndarray (n, 3)
    :param s_cent: sphere center
    :param s_rad: sphere radius
    :return: a n1array with the distances of each samples to a point
    """

    # Computing the normals
    ns_1 = point - s_cent
    ns_1 /= np.linalg.norm(ns_1)
    ns_2 = samples - s_cent
    ns_2 /= np.linalg.norm(ns_2)

    # Computing the distances
    dsts = np.zeros(shape=len(ns_2), dtype=float)
    for i in range(len(ns_2)):
        dsts[i] = s_rad * np.arctan2(np.linalg.norm(np.cross(ns_1, ns_2[i, :])),
                                     np.dot(ns_1, ns_2[i, :]))
    return dsts


def gen_urnd_sphere_cap(s_rad, c_rad, a_cap=None):
    """
    Generate a random point from distribution uniformly distributed on sphere cap
    :param s_rad: sphere radius
    :param c_rad: cap radius
    :param a_cap: axis vector for the cap (optional, default None then north pole used (0, 0, 1))
    :return: the coordinates of point generated as a 1darray with respect the sphere center
    """

    # Compute cap angle
    phi_cap = np.arccos(1. - (c_rad**2/(2.*s_rad**2)))

    # Uniform point on the cap on north pole
    phi = phi_cap * np.random.rand() # np.arccos(2. * phi_cap * np.random.rand() - 1.)
    rho = 2. * np.pi * np.random.rand()
    x = s_rad * np.sin(phi) * np.cos(rho)
    y = s_rad * np.sin(phi) * np.sin(rho)
    z = s_rad * np.cos(phi)

    # Move from pole to cap center
    if a_cap is not None:
        s_a_cap = np.asarray((a_cap[1], a_cap[0], a_cap[2]))
        rot, tilt, psi = vect_to_zrelion(s_a_cap, mode='active')
        vect = rotate_3d_vector((x, y, z), (rot, tilt, psi), deg=True, conv='relion')
        return vect

    return np.asarray((x, y, z), dtype=float)


def vect_to_zrelion(v_in, mode='active'):
    """
    Computes rotation angles of from an input vector to fit reference [0,0,1] vector having a free Euler angle in Relion
    format. First Euler angle (Rotation) is assumed 0
    :param v_in: input vector
    :param mode: either 'active' (default) or 'pasive'
    :return: a 2-tuple with the Euler angles in Relion format
    """

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

def relion_norm(tomo, mask=None, inv=True):
    """
    Relion tomogram normalization
    :param tomo: input tomogram
    :param mask: if None (default) the whole tomogram is used for computing the statistics otherwise just the masked region
    :param inv: if True the values are inverted (default)
    :return:
    """

    # Input parsing
    if mask is None:
        mask = np.ones(shape=tomo.shape, dtype=bool)

    # Inversion
    if inv:
        hold_tomo = -1. * tomo
    else:
        hold_tomo = tomo

    # Statistics
    stat_tomo = hold_tomo[mask>0]
    mn, st = stat_tomo.mean(), stat_tomo.std()

    # Histogram equalization
    tomo_out = np.zeros(shape=tomo.shape, dtype=np.float32)
    if st > 0:
        tomo_out = (hold_tomo-mn) / st
    else:
        print('WARNING (relion_norm): standard deviation=' + str(st))

    return tomo_out

################################################################################################
#  Class for modelling a thresholding procedure
#  If th_high >= th_low then everything in range [th_low, th_high] is pass the test
#  Otherwise everything in range (-infty,th_high)U(th_low,infty) is pass the test
#
class Threshold(object):

    def __init__(self, th_low, th_high):
        self.__th_low = None
        self.__th_high = None
        self.set_low(th_low)
        self.set_high(th_high)

    def set_low(self, th_low):
        if th_low is None:
            self.__th_low = - np.inf
        else:
            self.__th_low = th_low

    def set_high(self, th_high):
        if th_high is None:
            self.__th_high = np.inf
        else:
            self.__th_high = th_high

    def get_low(self):
        return self.__th_low

    def get_high(self):
        return self.__th_high

    def test(self, in_val):
        if self.__th_high >= self.__th_low:
            if (in_val >= self.__th_low) and (in_val <= self.__th_high):
                return True
            else:
                return False
        else:
            if (in_val >= self.__th_low) and (in_val <= self.__th_high):
                return False
            else:
                return True

    def tostring(self):
        if self.__th_low <= self.__th_high:
            return '[' + str(self.__th_low) + ', ' + str(self.__th_high) + ']'
        else:
            return '(-inf, ' + str(self.__th_high) + ') U (' + str(self.__th_low) + ', inf)'


##########################################################################################
#   Associative list for index values with a key
#   Reimplement all list() functionality (with the exception of sort())
#
class Hash(object):

    def __init__(self):
        self.__keys = list()
        self.__values = list()

    def append(self, key, value):
        self.__keys.append(key)
        self.__values.append(value)

    def extend(self, keys, values):
        self.__keys.extend(keys)
        self.__values.extend(values)

    def insert(self, i, key, value):
        self.__keys.insert(i, key)
        self.__values.insert(i, value)

    def remove(self, key, value):
        self.__keys.remove(key)
        self.__values.remove(value)

    def pop(self, i):
        if i is None:
            key = self.__keys.pop()
            value = self.__values.pop()
        else:
            key = self.__keys.pop(i)
            value = self.__values.pop(i)

    def index_key(self, key):
        self.__keys.index(key)

    def index_value(self, value):
        self.__values.index(value)

    def count_key(self, key):
        self.__keys.count(key)

    def count_value(self, value):
        self.__values.count(value)

    def reverse(self):
        self.__keys.reverse()
        self.__values.reverse()

    def get_key(self, i):
        return self.__keys[i]

    def get_value(self, i):
        return self.__values[i]
