"""
Classes for curves

# Author: Antonio Martinez-Sanchez (Max Planck Institute for Biochemistry)
# Date: 29.02.2016
"""

__author__ = 'martinez'

import vtk
import math
import numpy as np
from pyseg.globals.utils import angle_2vec_3D, closest_points
import warnings

###### Global variables
PI_2 = .5 * np.pi
MAX_PER_ANG = .25 * np.pi
MAX_FLOAT = np.finfo('float').max

# ####################################################################################################
# This class represents a spaced curve from a sequence of discrete samples (coordinates in 3D)
# Numerical approximation of discrete differential geometry from
# Boutin M. "Numerically Invariant Signature Curves" Int. J. Comput. Vision, 40(3): 235-248, 2000
#
#
class SpaceCurve(object):

    # #### Constructor Area

    # samples: array with the sequences samples of the curve
    # mode: computation mode, 1: precise, 2 fast (default)
    # do_geom: if True (default) curve geometric properties are computed during construction, otherwise not (this is
    #          useful for temporary curves)
    def __init__(self, samples, mode=2, do_geom=True):
        self.__samples = np.asarray(samples, dtype=float)
        self.__mode = mode
        self.__apex_id = -1
        self.__ds = None
        self.__lengths = None
        self.__usg_k = None
        self.__sg_k = None
        self.__usg_t = None
        self.__sg_t = None
        self.__length = .0
        self.__tot_uk = .0
        self.__tot_k = .0
        self.__tot_ut = .0
        self.__tot_t = .0
        self.__tot_ukt = .0
        self.__per = .0
        self.__ns = .0
        self.__bs = .0
        self.__al = -1.
        self.__sin = .0
        if do_geom:
            self.compute_geom()


    # Compute all geometric descriptors
    def compute_geom(self):
        self.__compute_ds()
        self.__length = self.__ds.sum()
        self.__compute_lengths()
        self.__compute_usg_k()
        self.__compute_sg_k()
        self.__compute_usg_t()
        self.__compute_sg_t()
        self.__tot_uk = (self.__usg_k * self.__ds).sum()
        self.__tot_k = (self.__sg_k * self.__ds).sum()
        self.__tot_ut = (self.__usg_t * self.__ds).sum()
        self.__tot_t = (self.__sg_t * self.__ds).sum()
        self.__tot_ukt = (np.sqrt((self.__usg_k*self.__usg_k) + (self.__usg_t*self.__usg_t)) * self.__ds).sum()
        self.__compute_per_length()
        self.__compute_ns()
        self.__compute_bs()
        self.__compute_al()
        self.__compute_sin()

    # External functionality area

    def get_nsamples(self):
        return self.__samples.shape[0]

    def get_samples(self):
        return self.__samples

    def get_sample(self, idx):
        return self.__samples[idx, :]

    def get_start_sample(self):
        return self.__samples[0, :]

    def get_end_sample(self):
        return self.__samples[-1, :]

    def get_lengths(self):
        return self.__lengths

    def get_length(self):
        return self.__length

    def get_total_uk(self):
        return self.__tot_uk

    def get_total_k(self):
        return self.__tot_k

    def get_total_ut(self):
        return self.__tot_ut

    def get_total_t(self):
        return self.__tot_t

    def get_total_ukt(self):
        return self.__tot_ukt

    def get_normal_symmetry(self):
        return self.__ns

    def get_binormal_symmetry(self):
        return self.__bs

    def get_apex_length(self, update=False):
        if update:
            self.__compute_al()
        return self.__al

    def get_sinuosity(self):
        return self.__sin

    def get_ds(self):
        return self.__ds

    def get_uk(self):
        return self.__usg_k

    def get_k(self):
        return self.__sg_k

    def get_ut(self):
        return self.__usg_t

    def get_t(self):
        return self.__sg_t

    def get_per_length(self, update=False):
        if update:
            self.__compute_per_length()
        return self.__per

    # Return a vtkPolyData which contains the curve
    # add_geom: if True geometry properties are added otherwise not
    def get_vtp(self, add_geom=True):

        # Initialization
        poly, points, lines = vtk.vtkPolyData(), vtk.vtkPoints(), vtk.vtkCellArray()
        if add_geom:
            # Point properties
            pds_data = vtk.vtkFloatArray()
            pds_data.SetNumberOfComponents(1)
            pds_data.SetName('ds')
            plens_data = vtk.vtkFloatArray()
            plens_data.SetNumberOfComponents(1)
            plens_data.SetName('lengths')
            puk_data = vtk.vtkFloatArray()
            puk_data.SetNumberOfComponents(1)
            puk_data.SetName('u_k')
            psk_data = vtk.vtkFloatArray()
            psk_data.SetNumberOfComponents(1)
            psk_data.SetName('s_k')
            put_data = vtk.vtkFloatArray()
            put_data.SetNumberOfComponents(1)
            put_data.SetName('u_t')
            pst_data = vtk.vtkFloatArray()
            pst_data.SetNumberOfComponents(1)
            pst_data.SetName('s_t')
            # Cell properties
            clen_data = vtk.vtkFloatArray()
            clen_data.SetNumberOfComponents(1)
            clen_data.SetName('length')
            ctuk_data = vtk.vtkFloatArray()
            ctuk_data.SetNumberOfComponents(1)
            ctuk_data.SetName('u_total_k')
            ctk_data = vtk.vtkFloatArray()
            ctk_data.SetNumberOfComponents(1)
            ctk_data.SetName('total_k')
            ctut_data = vtk.vtkFloatArray()
            ctut_data.SetNumberOfComponents(1)
            ctut_data.SetName('u_total_t')
            ctt_data = vtk.vtkFloatArray()
            ctt_data.SetNumberOfComponents(1)
            ctt_data.SetName('total_t')
            cper_data = vtk.vtkFloatArray()
            cper_data.SetNumberOfComponents(1)
            cper_data.SetName('per_length')
            cukt_data = vtk.vtkFloatArray()
            cukt_data.SetNumberOfComponents(1)
            cukt_data.SetName('total_ukt')
            cns_data = vtk.vtkFloatArray()
            cns_data.SetNumberOfComponents(1)
            cns_data.SetName('normal_sim')
            cbs_data = vtk.vtkFloatArray()
            cbs_data.SetNumberOfComponents(1)
            cbs_data.SetName('binormal_sim')
            cal_data = vtk.vtkFloatArray()
            cal_data.SetNumberOfComponents(1)
            cal_data.SetName('apex_length')
            csin_data = vtk.vtkFloatArray()
            csin_data.SetNumberOfComponents(1)
            csin_data.SetName('sinuosity')

        # Line creation
        lines.InsertNextCell(self.get_nsamples())
        if add_geom:
            # Adding cell properties
            clen_data.InsertNextTuple((self.__length,))
            ctuk_data.InsertNextTuple((self.__tot_uk,))
            ctk_data.InsertNextTuple((self.__tot_k,))
            ctut_data.InsertNextTuple((self.__tot_ut,))
            ctt_data.InsertNextTuple((self.__tot_t,))
            cukt_data.InsertNextTuple((self.__tot_ukt,))
            cper_data.InsertNextTuple((self.__per,))
            cns_data.InsertNextTuple((self.__ns,))
            cbs_data.InsertNextTuple((self.__bs,))
            cal_data.InsertNextTuple((self.__al,))
            csin_data.InsertNextTuple((self.__sin,))
            for i, point in enumerate(self.get_samples()):
                points.InsertNextPoint(point)
                lines.InsertCellPoint(i)
                # Adding point properties
                pds_data.InsertNextTuple((self.__ds[i],))
                plens_data.InsertNextTuple((self.__lengths[i],))
                puk_data.InsertNextTuple((self.__usg_k[i],))
                psk_data.InsertNextTuple((self.__sg_k[i],))
                put_data.InsertNextTuple((self.__usg_t[i],))
                pst_data.InsertNextTuple((self.__sg_t[i],))
        else:
            for i, point in enumerate(self.get_samples()):
                points.InsertNextPoint(point)
                lines.InsertCellPoint(i)
        poly.SetPoints(points)
        poly.SetLines(lines)
        if add_geom:
            # Point properties
            poly.GetPointData().AddArray(pds_data)
            poly.GetPointData().AddArray(plens_data)
            poly.GetPointData().AddArray(puk_data)
            poly.GetPointData().AddArray(psk_data)
            poly.GetPointData().AddArray(put_data)
            poly.GetPointData().AddArray(pst_data)
            # Cell properties
            poly.GetCellData().AddArray(clen_data)
            poly.GetCellData().AddArray(ctuk_data)
            poly.GetCellData().AddArray(ctk_data)
            poly.GetCellData().AddArray(ctut_data)
            poly.GetCellData().AddArray(ctt_data)
            poly.GetCellData().AddArray(cukt_data)
            poly.GetCellData().AddArray(cper_data)
            poly.GetCellData().AddArray(cns_data)
            poly.GetCellData().AddArray(cbs_data)
            poly.GetCellData().AddArray(cal_data)
            poly.GetCellData().AddArray(csin_data)
        return poly

    ###### External functionality area

    # Returns a new SpaceCurve whose samples are the decimation of the current
    # n_samp: number of samples for the decimated curve
    def gen_decimated(self, n_samp):
        # decimator = vtk.vtkDecimatePolylineFilter()
        decimator = vtk.vtkSplineFilter()
        decimator.SetSubdivideToSpecified()
        decimator.SetNumberOfSubdivisions(n_samp-1)
        poly = self.get_vtp(add_geom=False)
        decimator.SetInputData(poly)
        decimator.Update()
        poly_dec = decimator.GetOutput()
        coords = list()
        for i in range(poly_dec.GetNumberOfPoints()):
            coords.append(np.asarray(poly_dec.GetPoint(i), dtype=float))
        return SpaceCurve(coords)

    def compute_point_intersection(self, point):
        """
        Compute curve intersection point between the curve and an input point, the intersection point is defined as
        the line intersection for Point-Line distance between the input point an the two closest curve samples.
        Point-Line distance estimation taken from: http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
        :param point: Input point
        :return:
        """

        # Finding the two closest samples on the curve
        samps = self.get_samples()
        cpoints = closest_points(point, samps, nn=2)
        p0, p1, p2 = point, cpoints[0, :], cpoints[1, :]

        # Intersection point
        hold_a, hold_b = p1 - p0, p2 - p1
        t = -(np.dot(hold_a, hold_b)) / (hold_b * hold_b).sum()
        return p1 + (p2 - p1)*t

    def compute_point_normal(self, point):
        """
        Compute the normal between a point and the curve, it is defined as the normalized vector between the curve
        intersection point and the input point
        :param point: Input point
        :return: The normalized normal vector
        """

        normal = self.compute_point_intersection(point) - point
        norm = math.sqrt((normal * normal).sum())
        if norm <= 0:
            return np.asarray((0., 0., 0.))
        else:
            return normal / norm

    # #### Internal functionality area

    # Linear extrapolation for x
    def __lin_extra(self, x, x_k, x_k1, y_k, y_k1):
        ds = x_k - x_k1
        if ds == 0:
            return 0.
        else:
            hold = y_k1 + ((x - x_k1)/ds)*(y_k - y_k1)
            if hold < 0:
                return 0.
            else:
                return hold

    # Lagrange extrapolation from tre points
    def __3_pts_lagrange_extra(self, x, x_1, x_2, x_3, y_1, y_2, y_3):
        n_1 = (x-x_2) * (x-x_3)
        n_2 = (x-x_1) * (x-x_3)
        n_3 = (x-x_1) * (x-x_2)
        d_1 = (x_1-x_2) * (x_1-x_3)
        d_2 = (x_2-x_1) * (x_2-x_3)
        d_3 = (x_3-x_1) * (x_3-x_2)
        if (d_1 == 0) or (d_2 == 0) or (d_3 == 3):
            return 0.
        else:
            return (n_1/d_1)*y_1 + (n_2/d_2)*y_2 + (n_3/d_3)*y_3

    # Compute signed area of a parallelogram
    def __pl_sg_area(self, p_i, p_j, p_k, p_l):
        vij = p_i - p_j
        vkl = p_k - p_l
        return vij[0]*vkl[1] - vkl[0]*vij[1]

    # Euclidean distance between two points
    def __dist_2_pts(self, p_0, p_1):
        hold = p_0 - p_1
        return math.sqrt((hold * hold).sum())

    # Height of a triangle respect to p_1, that is, distance of p_1 to line (p_0, p_2)
    def __tri_h(self, p_0, p_1, p_2):
        vr = p_2 - p_0
        vp = p_0 - p_1
        vpm = math.sqrt((vp*vp).sum())
        if vpm <= 0:
            return 0.
        else:
            vh = np.cross(vr, vp)
            return math.sqrt((vh*vh).sum()) / vpm

    # Height of a tetrahedron respect to p_3, that is, distance of p_3 to plane (p_0, p_1, p_2)
    def __tetra_h(self, p_0, p_1, p_2, p_3):
        n = np.cross(p_1-p_0, p_2-p_0)
        nm = math.sqrt((n*n).sum())
        if nm <= 0:
            return 0.
        else:
            return math.fabs(np.dot(n/nm, p_3-p_0))

    # Heron formula for triangle area from its three sides
    def __tri_area_3_sides(self, a, b, c):
        p = .5 * (a + b + c)
        hold = p * (p-a) * (p-b) * (p-c)
        if hold <= 0:
            return 0.
        else:
            return math.sqrt(hold)

    # Numerical estimator for unsigned curvature from 3 input points
    # Returns: unsigned curvature estimation for p_1
    def __usg_k_3_pts(self, p_0, p_1, p_2):
        a = self.__dist_2_pts(p_0, p_1)
        b = self.__dist_2_pts(p_1, p_2)
        c = self.__dist_2_pts(p_0, p_2)
        hold = a * b * c
        if hold == 0:
            return 0.
        else:
            return (4.*self.__tri_area_3_sides(a, b, c)) / hold

    # Numerical estimator for unsigned curvature from 5 input points
    # Returns: unsigned curvature estimation for p_2
    def __usg_k_5_pts(self, p_0, p_1, p_2, p_3, p_4):

        # Computed signed areas of the parallelograms
        a_012 = self.__pl_sg_area(p_0, p_1, p_0, p_2)
        a_013 = self.__pl_sg_area(p_0, p_1, p_0, p_3)
        a_014 = self.__pl_sg_area(p_0, p_1, p_0, p_4)
        a_023 = self.__pl_sg_area(p_0, p_2, p_0, p_3)
        a_024 = self.__pl_sg_area(p_0, p_2, p_0, p_4)
        a_034 = self.__pl_sg_area(p_0, p_3, p_0, p_4)
        a_123 = self.__pl_sg_area(p_1, p_2, p_1, p_3)
        a_124 = self.__pl_sg_area(p_1, p_2, p_1, p_4)
        a_134 = self.__pl_sg_area(p_1, p_3, p_1, p_4)
        a_234 = self.__pl_sg_area(p_2, p_3, p_2, p_4)
        a_1234 = self.__pl_sg_area(p_1, p_2, p_3, p_4)
        a_1234_2 = a_1234 * a_1234

        # Intermediate computations
        t = .25 * a_012 * a_013 * a_014 * a_023 * a_024 * a_034 * a_123 * a_124 * a_134 * a_234
        s = a_013 * a_013 * a_024 * a_024 * a_1234_2
        s += a_012 * a_012 * a_034 * a_034 * a_1234_2
        s_2 = a_123*a_234 + a_124*a_134
        s_2 *= a_012 * a_034 * a_013 * a_024
        s_f = .25 * (s - 2.*s_2)

        if t <= 0:
            return 0
        else:
            return s_f / (t**(2./3.))
            # return s_f

    # Numerical estimator for signed curvature from 2 input points
    # Returns: signed curvature estimation for p_0
    def __sg_k_2_pts(self, p_0, p_1, uk_0, uk_1):
        b = self.__dist_2_pts(p_0, p_1)
        if b <= 0:
            return 0
        else:
            return (uk_1-uk_0) / b

    # Numerical estimator for signed curvature from 5 input points
    # Returns: signed curvature estimation for p_2
    def __sg_k_5_pts(self, p_0, p_1, p_2, p_3, p_4, uk_1, uk_2, uk_3):
        a = self.__dist_2_pts(p_1, p_2)
        b = self.__dist_2_pts(p_2, p_3)
        d = self.__dist_2_pts(p_3, p_4)
        g = self.__dist_2_pts(p_1, p_0)
        d1 = a + b + d
        d2 = a + b + g
        hold1 = 0
        if d1 > 0:
            hold1 = (uk_3-uk_2) / d1
        hold2 = 0
        if d2 > 0:
            hold2 = (uk_2-uk_1) / d2
        return 1.5*(hold1 + hold2)

    # Numerical estimator for unsigned torsion from 4 input points (version 1)
    # Returns: unsigned torsion estimation for p_1
    def __usg_t_4_pts_1(self, p_0, p_1, p_2, p_3, uk_1):
        d = self.__dist_2_pts(p_2, p_3)
        e = self.__dist_2_pts(p_1, p_3)
        f = self.__dist_2_pts(p_0, p_3)
        h = self.__tetra_h(p_0, p_1, p_2, p_3)

        hold = d * e * f * uk_1
        if hold <= 0:
            return .0
        else:
            return (6.*h) / hold

    # Numerical estimator for unsigned torsion from 4 input points (version 2)
    # Returns: unsigned torsion estimation for p_1
    def __usg_t_4_pts_2(self, p_0, p_1, p_2, p_3):
        b = self.__dist_2_pts(p_1, p_2)
        d = self.__dist_2_pts(p_2, p_3)
        e = self.__dist_2_pts(p_1, p_3)
        f = self.__dist_2_pts(p_0, p_3)
        hold = f * self.__tri_area_3_sides(e, b, d)
        if hold <= 0:
            return .0
        else:
            h = self.__tetra_h(p_0, p_1, p_2, p_3)
            return (1.5*h*b) / hold

    # Numerical estimator for signed torsion from 5 input points
    # Returns: unsigned torsion estimation for p_2
    def __sg_t_5_pts(self, p_0, p_1, p_2, p_3, p_4, uk_2, k_2, ut_1, ut_2, ut_3):
        if uk_2 <= 0:
            return 0.
        a = self.__dist_2_pts(p_1, p_2)
        b = self.__dist_2_pts(p_2, p_3)
        d = self.__dist_2_pts(p_3, p_4)
        g = self.__dist_2_pts(p_0, p_1)
        h = self.__tri_h(p_1, p_2, p_3)
        hold_1 = 2*a + 2*b + 2*d + h + g
        if hold_1 <= 0:
            return 0.
        else:
            hold_2 = 2*a + 2*b - 2*d - 3*h + g
            hold_3 = (ut_2*k_2) / (6*uk_2)
            return 4. * ((ut_3 - ut_1 + (hold_2*hold_3)) / hold_1)

    # Computes length differential
    def __compute_ds(self):

        # Initialization
        n_points = self.__samples.shape[0]
        ds = np.zeros(shape=n_points, dtype=float)

        # Regular cases
        for i in range(1, n_points):
            ds[i] = self.__dist_2_pts(self.__samples[i-1], self.__samples[i])

        self.__ds = ds

    def __compute_lengths(self):
        self.__lengths = np.zeros(shape=self.__ds.shape, dtype=float)
        for i in range(1, len(self.__ds)):
            self.__lengths[i] = self.__lengths[i-1] + self.__ds[i]

    # Estimates local curvature along the whole curve
    def __compute_usg_k(self):

        # Initialization
        n_samples = self.__samples.shape[0]
        usg_k = np.zeros(shape=n_samples, dtype=float)
        if n_samples <= 2:
            self.__usg_k = usg_k
            return

        # Regular cases
        if self.__mode == 1:
            for i in range(2, n_samples-2):
                p_0, p_1 = self.__samples[i-2, :], self.__samples[i-1, :]
                p_2 = self.__samples[i, :]
                p_3, p_4 = self.__samples[i+1, :], self.__samples[i+2, :]
                usg_k[i] = self.__usg_k_5_pts(p_0, p_1, p_2, p_3, p_4)
        else:
            for i in range(1, n_samples-1):
                p_0, p_1, p_2  = self.__samples[i-1, :], self.__samples[i, :], self.__samples[i+1, :]
                usg_k[i] = self.__usg_k_3_pts(p_0, p_1, p_2)

        # Extremal cases
        p_0, p_1, p_2 = self.__samples[0, :], self.__samples[1, :], self.__samples[2, :]
        usg_k[1] = self.__usg_k_3_pts(p_0, p_1, p_2)
        usg_k[0] = self.__lin_extra(0, self.__ds[1], self.__ds[1]+self.__ds[2], usg_k[1], usg_k[2])
        p_0, p_1, p_2 = self.__samples[-1, :], self.__samples[-2, :], self.__samples[-3, :]
        usg_k[-2] = self.__usg_k_3_pts(p_0, p_1, p_2)
        usg_k[-1] = self.__lin_extra(self.__length, self.__length-self.__ds[-1],
                                     self.__length-self.__ds[-1]-self.__ds[-2], usg_k[-2], usg_k[-3])

        self.__usg_k = usg_k

    # Estimates local curvature derivative along the whole curve
    # Requires the previous computation of the unsigned curvature (self.__usg_k)
    def __compute_sg_k(self):

        # Initialization
        n_samples = self.__samples.shape[0]
        sg_k = np.zeros(shape=n_samples, dtype=float)
        if n_samples <= 2:
            self.__sg_k = sg_k
            return

        # Regular cases
        for i in range(2, n_samples-2):
            p_0, p_1 = self.__samples[i-2, :], self.__samples[i-1, :]
            p_2 = self.__samples[i, :]
            p_3, p_4 = self.__samples[i+1, :], self.__samples[i+2, :]
            uk_1, uk_2, uk_3 = self.__usg_k[i-1], self.__usg_k[i], self.__usg_k[i+1]
            sg_k[i] = self.__sg_k_5_pts(p_0, p_1, p_2, p_3, p_4, uk_1, uk_2, uk_3)

        # Extremal cases
        p_1, p_2 = self.__samples[1, :], self.__samples[2, :]
        uk_1, uk_2 = self.__usg_k[1], self.__usg_k[2]
        sg_k[1] = self.__sg_k_2_pts(p_1, p_2, uk_1, uk_2)
        sg_k[0] = self.__lin_extra(0, self.__ds[:2].sum(), self.__ds[:3].sum(), sg_k[1], sg_k[2])
        p_1, p_2 = self.__samples[-3, :], self.__samples[-2, :]
        uk_1, uk_2 = self.__usg_k[-3], self.__usg_k[-2]
        sg_k[-2] = self.__sg_k_2_pts(p_1, p_2, uk_1, uk_2)
        sg_k[-1] = self.__lin_extra(self.__length, self.__length-self.__ds[-1:].sum(),
                                    self.__length-self.__ds[-2:].sum(), sg_k[-2], sg_k[-3])

        self.__sg_k = sg_k

    # Estimates local unsigned torsion along the whole curve
    def __compute_usg_t(self):

        # Initialization
        n_samples = self.__samples.shape[0]
        usg_t = np.zeros(shape=n_samples, dtype=float)
        if n_samples <= 3:
            self.__usg_t = usg_t
            return

        # Regular cases
        for i in range(2, n_samples-2):
            p_0, p_1 = self.__samples[i-1, :], self.__samples[i, :]
            p_2, p_3 = self.__samples[i+1, :], self.__samples[i+2, :]
            uk_1 = self.__usg_k[i]
            usg_t_1 = self.__usg_t_4_pts_1(p_0, p_1, p_2, p_3, uk_1)
            usg_t_2 = self.__usg_t_4_pts_2(p_0, p_1, p_2, p_3)
            usg_t[i] = .5 * (usg_t_1 + usg_t_2)

        # Extremal cases
        p_0, p_1, p_2, p_3 = self.__samples[0, :], self.__samples[1, :], self.__samples[2, :], \
                             self.__samples[3, :]
        uk_1 = self.__usg_k[1]
        usg_t_1 = self.__usg_t_4_pts_1(p_0, p_1, p_2, p_3, uk_1)
        usg_t_2 = self.__usg_t_4_pts_2(p_0, p_1, p_2, p_3)
        usg_t[1] = .5 * (usg_t_1 + usg_t_2)
        usg_t[0] = self.__3_pts_lagrange_extra(0, self.__ds[:2].sum(), self.__ds[:3].sum(), self.__ds[:4].sum(),
                                               usg_t[1], usg_t[2], usg_t[3])
        usg_t[-2] = self.__lin_extra(self.__length-self.__ds[-1:].sum(), self.__length-self.__ds[-2:].sum(),
                                     self.__length-self.__ds[-3:].sum(), usg_t[-3], usg_t[-4])
        usg_t[-1] = self.__3_pts_lagrange_extra(self.__length, self.__length-self.__ds[-1:].sum(),
                                               self.__length-self.__ds[-2:].sum(),
                                               self.__length-self.__ds[-3:].sum(),
                                               usg_t[-2], usg_t[-3], usg_t[-4])

        self.__usg_t = usg_t

    # Estimates local torsion derivative along the whole curve
    def __compute_sg_t(self):

        # Initialization
        n_samples = self.__samples.shape[0]
        sg_t = np.zeros(shape=n_samples, dtype=float)
        if n_samples <= 3:
            self.__sg_t = sg_t
            return

        # Regular cases
        for i in range(2, n_samples-2):
            p_0, p_1 = self.__samples[i-2, :], self.__samples[i-1, :]
            p_2 = self.__samples[i, :]
            p_3, p_4 = self.__samples[i+1, :], self.__samples[i+2, :]
            uk_2, k_2 = self.__usg_k[i], self.__sg_k[i]
            ut_1, ut_2, ut_3 = self.__usg_t[i-1], self.__usg_t[i], self.__usg_t[i+1]
            sg_t[i] = self.__sg_t_5_pts(p_0, p_1, p_2, p_3, p_4, uk_2, k_2, ut_1, ut_2, ut_3)

        # Extremal cases
        sg_t[1] = self.__lin_extra(self.__ds[:2].sum(), self.__ds[:3].sum(), self.__ds[:4].sum(),
                                   sg_t[2], sg_t[3])
        sg_t[0] = self.__3_pts_lagrange_extra(0, self.__ds[:2].sum(), self.__ds[:3].sum(), self.__ds[:4].sum(),
                                              sg_t[1], sg_t[2], sg_t[3])
        sg_t[-2] = self.__lin_extra(self.__length-self.__ds[-1:].sum(), self.__length-self.__ds[-2:].sum(),
                                    self.__length-self.__ds[-3:].sum(), sg_t[-3], sg_t[-4])
        sg_t[-1] = self.__3_pts_lagrange_extra(self.__length, self.__length-self.__ds[-1:].sum(),
                                               self.__length-self.__ds[-2:].sum(),
                                               self.__length-self.__ds[-3:].sum(),
                                               sg_t[-2], sg_t[-3], sg_t[-4])

        self.__sg_t = sg_t

    # Compute accumulated normal symmetry
    # Requires the previous computation of local and total unsigned curvature
    def __compute_ns(self):

        # Initialization
        n_samples = self.__samples.shape[0]
        if n_samples <= 2:
            self.__ns = 1.
            return

        # Normal accumulation vector
        n = np.zeros(shape=3, dtype=float)
        for i in range(1, n_samples-1):
            p_0, p_1, p_2 = self.__samples[i-1, :], self.__samples[i, :], self.__samples[i+1, :]
            # Update normal accumulation
            n_h = 2*p_1 - p_0 - p_2
            n_h_m = math.sqrt((n_h * n_h).sum())
            if n_h_m > 0:
                n_h /= n_h_m
                n += ((self.__usg_k[i]*self.__ds[i]) * n_h)

        # Extrema cases (end)
        p_0, p_1, p_2 = self.__samples[-3, :], self.__samples[-2, :], self.__samples[-1, :]
        n_h = 2*p_1 - p_0 - p_2
        n_h_m = math.sqrt((n_h * n_h).sum())
        if n_h_m > 0:
            n_h /= n_h_m
            n += ((self.__usg_k[-1]*self.__ds[-1]) * n_h)

        # Compute total value of symmetry
        n_m = math.sqrt((n * n).sum())

        total = self.__tot_uk
        if total <= 0:
            self.__ns = 1.
        else:
            self.__ns = 1. - (1./total) * n_m

    # Compute accumulated binormal symmetry
    # Requires the previous computation of local and total unsigned torsion
    def __compute_bs(self):

        # Initialization
        n_samples = self.__samples.shape[0]
        if n_samples <= 2:
            self.__bs = 1.
            return

        # Normal accumulation vector
        b = np.zeros(shape=3, dtype=float)
        for i in range(1, n_samples-1):
            p_0, p_1, p_2 = self.__samples[i-1, :], self.__samples[i, :], self.__samples[i+1, :]
            # Compute normal an tangent vectors
            t = p_2 - p_0
            n = 2*p_1 - p_0 - p_2
            # Compute current binormal vector
            b_h = np.cross(t, n)
            b_h_m = math.sqrt((b_h * b_h).sum())
            if b_h_m > 0:
                b_h /= b_h_m
                # Update accumulated vector
                b += ((self.__usg_t[i]*self.__ds[i]) * b_h)

        # Extrema cases (end)
        p_0, p_1, p_2 = self.__samples[-3, :], self.__samples[-2, :], self.__samples[-1, :]
        t = p_2 - p_0
        n = 2*p_1 - p_0 - p_2
        # Compute current binormal vector
        b_h = np.cross(t, n)
        b_h_m = math.sqrt((b_h * b_h).sum())
        if b_h_m > 0:
            b_h /= b_h_m
            # Update accumulated vector
            b += ((self.__usg_t[-1]*self.__ds[-1]) * b_h)

        # Compute total value of symmetry
        b_m = math.sqrt((b * b).sum())

        total = self.__tot_ut
        if total <= 0:
            self.__bs = 1.
        else:
            self.__bs = 1. - (1./total) * b_m

    # Curve apex length, maximum distance of curve point from curve axis (line which contains p_start and p_end)
    def __compute_al(self):

        # Initialization
        n_samples = self.__samples.shape[0]
        if n_samples <= 2:
            self.__al = -1
            return

        # Compute curve axis line
        p_start, p_end = self.__samples[0, :],  self.__samples[-1, :]
        v_a = p_end - p_start
        v_a_m = math.sqrt((v_a * v_a).sum())
        if v_a_m <= 0:
            self.__al = 0.

        # Finding maximum distance
        hold = np.cross(v_a, p_start-self.__samples)

        # Find apex coordinates
        dsts = np.sqrt(np.sum(hold * hold, axis=1))
        a_id = np.argmax(dsts)
        self.__apex_id = a_id

        try:
            self.__al = dsts[a_id] / v_a_m
        except RuntimeWarning:
            print(f"curve.py:765 dsts[a_id] '{dsts[a_id]}', v_a_m '{v_a_m}'")

    # Compute curve sinuosity (ratio between geodesic and d(p_start, p_end))
    # Requires previous computation of curve length
    def __compute_sin(self):
        eu_dst = self.__samples[-1, :] - self.__samples[0, :]
        eu_dst = math.sqrt((eu_dst * eu_dst).sum())
        if eu_dst <= 0:
            self.__sin = -1.
        else:
            self.__sin = self.__length / eu_dst

    # Compute persistence length (Apex and star point are the reference points)
    def __compute_per_length(self):
        if self.__apex_id < 0:
            self.__compute_al()
        # Check that persistence can be computed
        if self.__apex_id < 2:
            self.__per = -1.
            return
        # Starting vector
        start_v = self.__samples[1] - self.__samples[0]
        env_v = self.__samples[self.__apex_id] - self.__samples[self.__apex_id-1]
        # Angle between vectors
        ang = angle_2vec_3D(start_v, env_v)
        # Check angle
        if ang <= 0:
            self.__per = -1.
        elif ang < MAX_PER_ANG:
            if self.__ds is None:
                self.__compute_ds()
            length = self.__ds[:self.__apex_id].sum()
            # print 'L=' + str(length) + ', A=' + str(ang)
            self.__per =  -length / math.log(math.cos(ang))

