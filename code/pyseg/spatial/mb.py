"""
Classes for doing the spatial analysis of structures in membranes

# Author: Antonio Martinez-Sanchez (Max Planck Institute for Biochemistry)
# Date: 12.06.15
"""

__author__ = 'martinez'

from pyto.segmentation.cluster import Cluster
from .globals import *

try:
    import pexceptions
except:
    from pyseg import pexceptions

##### PACKAGE VARIABLES

STR_KMEANS_IDS = 'kmeans_ids'
STR_HR_IDS = 'hr_ids'
STR_NND = 'nnd'
MB_LBL = 1

###########################################################################################
# Class for doing a spatial analysis of a cloud of connectors
# VERY IMPORTANT: This analyzer suppose a completely plane membrane (rectangular without
# curvature) with no thickness. Since input membrane is not treated as a surface
# and computed distances are not geodesic the precision are degraded as membrane
# curvature and thickness increase
###########################################################################################

class CMCAnalyzer(object):

    # cloud: array with cloud of points coordinates (n, 3)
    # seg: tomogram with the membrane segmentation (membrane must be labeled as 1)
    def __init__(self, cloud, seg):
        self.__cloud = cloud
        self.__seg = seg
        self.__nnde = self.nnde(self.__cloud)
        self.__km = None
        self.__hr = None
        self.__km_ids = None
        self.__hr_ids = None

    #### Set/Get methods area

    def get_nconn(self):
        return self.__cloud.shape[0]

    def get_cloud_coords(self, cloud_2d=False, coord=None):
        if cloud_2d:
            return self.compress_plane_2d(coord=coord)
        else:
            return self.__cloud

    # Return Pyto cluster object for k-means
    def get_clusters_km(self):
        return self.__km

    # Return Pyto cluster object for hierarchical
    def get_clusters_hr(self):
        return self.__hr

    # Returns membrane bounding box (x_m, x_M, y_m, y_M, *z_m, *z_M )
    def get_bounds(self, comp_2d=False, coord=None):

        if comp_2d:
            return self.__plane_bounds(coord=coord)
        else:
            mb_ids = np.where(self.__seg == MB_LBL)
            return np.min(mb_ids[0][:]), np.min(mb_ids[1][:]), np.min(mb_ids[2][:]), \
                   np.max(mb_ids[0][:]), np.max(mb_ids[1][:]), np.max(mb_ids[2][:]),

    # Returns segmentation points coordinates
    # comp_2d: if True (default False) coordinates are compressed to two dimensions
    def get_seg_cloud(self, comp_2d=False):

        if comp_2d:
            x_m, x_M, y_m, y_M = self.__plane_bounds()
            X, Y = np.meshgrid(list(range(x_m, x_M)), list(range(y_m, y_M)))
            n = X.shape.sum()
            cloud = np.zeros(shape=(n, 2), dtype=float)
            cloud[:, 0] = np.reshape(X, n)
            cloud[:, 1] = np.reshape(Y, n)
            return cloud
        else:
            cloud = list()
            mb_ids = np.where(self.__seg == MB_LBL)
            for i in range(len(mb_ids[0])):
                cloud.append(mb_ids[:][i])
            return np.asarray(cloud)

    # Returns connector locations in a VTK file (vtp)
    def get_vtp(self):

        # Initialization
        points = vtk.vtkPoints()
        cells = vtk.vtkCellArray()
        if self.__km_ids is not None:
            km_data = vtk.vtkIntArray()
            km_data.SetNumberOfComponents(1)
            km_data.SetName(STR_KMEANS_IDS)
        if self.__hr_ids is not None:
            hr_data = vtk.vtkIntArray()
            hr_data.SetNumberOfComponents(1)
            hr_data.SetName(STR_HR_IDS)
        if self.__nnde is not None:
            nnd_data = vtk.vtkFloatArray()
            nnd_data.SetNumberOfComponents(1)
            nnd_data.SetName(STR_NND)

        # Loop
        for i in range(self.__cloud.shape[0]):
            x, y, z = self.__cloud[i]
            points.InsertPoint(i, x, y, z)
            cells.InsertNextCell(1)
            cells.InsertCellPoint(i)
            if self.__km_ids is not None:
                km_data.InsertTuple1(i, self.__km_ids[i])
            if self.__hr_ids is not None:
                hr_data.InsertTuple1(i, self.__hr_ids[i])
            if self.__nnde is not None:
                nnd_data.InsertTuple1(i, self.__nnde[i])

        # Poly building
        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        poly.SetVerts(cells)
        if self.__km_ids is not None:
            poly.GetCellData().AddArray(km_data)
        if self.__hr_ids is not None:
            poly.GetCellData().AddArray(hr_data)
        if self.__nnde is not None:
            poly.GetCellData().AddArray(nnd_data)

        return poly

    #### External functionality area

    # t-test for the means of cloud of points shortest distance
    # Returns: t-test value (array), and the p-value
    def ttest_ind(self, ref_cloud):
        return sp.stats.ttest_ind(self.__nnde, self.nnde(ref_cloud))

    # Computes the fraction of clusters referred to cluster size
    # n: number of samples
    # Returns: fraction of clustered and radius
    def fraction_clustered(self, n):

        # Initialization
        d_max = np.max(self.fnde(self.__cloud))
        samples = np.linspace(0, d_max*0.1, n)
        fraction = np.zeros(shape=n, dtype=float)
        n_points_inv = 1.0 / float(len(self.__nnde))

        # Computation
        for i, s in enumerate(samples):
            fraction[i] = (self.__nnde <= s).sum() * n_points_inv

        return fraction, samples

    # Compares clustering with a reference
    # ref_cloud: reference cloud, if None (default) a random with uniform distribution
    # is generated
    # ref_clust: reference clusters (Pyto objects)
    # mode: clustering type, valid: k-means (default) and hierarchical
    # Returns: random similarty, Fowlkes-Mallows and variation of information
    def cluster_comp(self, ref_clust, mode='k-means'):

        if mode == 'k-means':
            clust = self.__km
        elif mode == 'hierarchical':
            clust = self.__hr
        else:
            error_msg = 'Not valid clustering criterion: ' + mode
            raise pexceptions.PySegTransitionError(expr='cluster_comp (CMCAnalyzer)',
                                                   msg=error_msg)

        if clust is None:
            error_msg = 'No internal clustering, call cluster() method first.'
            raise pexceptions.PySegInputError(expr='cluster_comp (CMCAnalyzer)',
                                              msg=error_msg)

        clust.findSimilarity(reference=ref_clust)
        return clust.rand, clust.bflat, clust.findSimilarityVI(reference=ref_clust)

    # Compute clustering (information are stored internally)
    # n: number of clusters
    # mode: clustering type, valid: k-means (default) and hierarchical
    def cluster(self, n, mode='k-means'):

        if mode == 'k-means':
            clust = Cluster.kmeans(data=self.__cloud, k=n)
            c_ids = clust.getClusters()
            self.__km = clust
        elif mode == 'hierarchical':
            clust = Cluster.hierarchical(data=self.__cloud)
            clust.extractFlat(threshold=n, criterion='maxclust')
            c_ids = clust.getClusters()
            self.__hr = clust
        else:
            error_msg = 'Not valid clustering criterion: ' + mode
            raise pexceptions.PySegTransitionError(expr='cluster (CMCAnalyzer)',
                                                   msg=error_msg)
        self.__set_ids(c_ids, mode)

    # Performs Averaged Nearest Neighbour test
    # mode: mode for area computation, valid: plane (default), volume.
    # thick: membrane thickness in pixel units, only valid for mode=volume
    # Returns: ANN z-score
    def annt(self, mode='plane', thick=1, coord=None):

        # Computing distances and area
        area = self.mb_area(mode, thick)
        if mode == 'plane':
            dists = self.nnde(comp_2d=True, coord=coord)
        else:
            dists = self.nnde(coord=coord)
        n = len(dists)

        # Computing ANN z-score
        d_o = np.mean(dists)
        if area <= 0:
            return -1
        area_inv = 1.0 / float(area)
        d_e = 0.5 / math.sqrt(n * area_inv)
        s_e = 0.26136 / math.sqrt(n * n * area_inv)
        return (d_o - d_e) / s_e

    # Expected nearest neighbour mean
    # mode: mode for area computation, valid: plane (default), volume.
    # thick: membrane thickness in pixel units, only valid for mode=volume
    # Returns: expected nearest neighbour distance in pixels
    def anne(self, mode='plane', thick=1, coord=None):
        return 0.5 / math.sqrt(self.get_nconn() / float(self.mb_area(mode, thick, coord=coord)))

    # Estimated nearest neighbour mean
    # mode: mode for area computation, valid: plane (default), volume.
    # Returns: estimated nearest neighbour distance mean in pixels
    def annm(self, mode='plane', coord=None):
        if mode == 'plane':
            dists = self.nnde(comp_2d=True, coord=coord)
        else:
            dists = self.nnde(coord=coord)
        return np.mean(dists)

    # Estimated nearest neighbour variance
    # mode: mode for area computation, valid: plane (default), volume.
    # Returns: estimated nearest neighbour distance variance in pixels
    def annv(self, mode='plane', coord=None):
        if mode == 'plane':
            dists = self.nnde(comp_2d=True, coord=coord)
        else:
            dists = self.nnde(coord=coord)
        return np.var(dists)

    # Crossed Averaged Nearest Neighbour test
    # cloud: input cloud analyzer for crossing
    # Returns: z-value
    def cannt(self, analyzer, mode='plane', thick=1, coord=None):

        n = self.get_nconn()
        a = self.mb_area(mode, thick, coord=coord)
        s_e = 0.26136 / float(math.sqrt((n*n)/float(a)))

        return (self.cannm(analyzer, mode, coord=coord) - self.anne(mode, thick, coord=coord)) / s_e

    # Crossed estimated nearest neighbour mean
    # mode: mode for area computation, valid: plane (default), volume.
    # analyzer: input cloud analyzer for crossing
    # Returns: estimated nearest neighbour distance mean in pixels
    def cannm(self, analyzer, mode='plane', coord=None):

        # Computing crossed distances
        if mode == 'plane':
            cloud = self.compress_plane_2d(coord=coord)
            cloud_r = analyzer.get_cloud_coords(cloud_2d=True, coord=coord)
        else:
            cloud_r = analyzer.get_cloud_coords(cloud_2d=False, coord=coord)
            cloud = self.__cloud
        dists = np.zeros(shape=cloud.shape[0], dtype=float)

        # Shortest distance loop
        for i in range(len(dists)):
            hold = cloud[i] - cloud_r
            hold = np.sum(hold*hold, axis=1)
            dists[i] = math.sqrt(np.min(hold))

        return np.mean(dists)

    # Crossed estimated nearest neighbour variance
    # mode: mode for area computation, valid: plane (default), volume.
    # analyzer: input cloud analyzer for crossing
    # Returns: estimated nearest neighbour distance variance in pixels
    def cannv(self, analyzer, mode='plane', coord=None):

        # Computing crossed distances
        if mode == 'plane':
            cloud = self.compress_plane_2d(coord=coord)
            cloud_r = analyzer.get_cloud_coords(cloud_2d=True, coord=coord)
        else:
            cloud_r = analyzer.get_cloud_coords(cloud_2d=False, coord=coord)
            cloud = self.__cloud
        dists = np.zeros(shape=cloud.shape[0], dtype=float)

        # Shortest distance loop
        for i in range(len(dists)):
            hold = cloud[i] - cloud_r
            hold = np.sum(hold*hold, axis=1)
            dists[i] = math.sqrt(np.min(hold))

        return np.var(dists)

    # G-Function
    # n: number of samples
    # Return: G-Function sampled n times paired with their distances
    def g_function(self, n, mode='plane', coord=None):

        # Geting ANND distribution
        if mode == 'plane':
            dists = self.nnde(comp_2d=True, coord=coord)
        else:
            dists = self.nnde(coord=coord)

        # Computing CDF
        hist, x = np.histogram(dists, bins=n, normed=True)
        dx = x[1] - x[0]
        cum = np.cumsum(hist)*dx
        n1 = n + 1
        hold = np.zeros(shape=n1, dtype=cum.dtype)
        hold[1:n1] = cum
        return hold, x

    # F-Function
    # analyzer: for crossing
    # n: number of samples
    # Return: G-Function sampled n times
    def f_function(self, analyzer, n, mode='plane', coord=None):

        # Compressing if needed
        if mode == 'plane':
            cloud = self.compress_plane_2d(coord=coord)
            cloud_r = analyzer.get_cloud_coords(cloud_2d=True, coord=coord)
        else:
            cloud_r = analyzer.get_cloud_coords(cloud_2d=False, coord=coord)
            cloud = self.__cloud
        dists = np.zeros(shape=cloud.shape[0], dtype=float)

        # Computing crossed distances
        for i in range(len(dists)):
            hold = cloud[i] - cloud_r
            hold = np.sum(hold*hold, axis=1)
            dists[i] = math.sqrt(np.min(hold))

        # Computing CDF
        hist, x = np.histogram(dists, bins=n, normed=True)
        dx = x[1] - x[0]
        cum = np.cumsum(hist)*dx
        n1 = n + 1
        hold = np.zeros(shape=n1, dtype=cum.dtype)
        hold[1:n1] = cum
        return hold, x

    # Estimation of Ripley's K along several samples.
    # VERY IMPORTANT: This test suppose that all points are contained by an euclidean
    # plane
    # n: number of samples
    # cloud: if not None (default None) the crossed Ripley's K factor is computed
    # bord: if True (default), border compensation activated
    # Return: Ripley's K coefficient and distance samples
    def ripley_t(self, n, cloud=None, bord=True, coord=None):

        # Initialization
        area = self.mb_area(mode='plane')
        d_max = math.sqrt(0.5 * area)
        ts = np.linspace(0, d_max, n)
        A = 1

        # Project
        if cloud is not None:
            cloud_2d_r = self.compress_plane_2d(cloud, coord=coord)
        else:
            cloud_2d_r = self.compress_plane_2d(coord=coord)
        cloud_2d_p = self.compress_plane_2d(coord=coord)
        x_min, x_max, y_min, y_max = self.__plane_bounds()
        N_p = cloud_2d_p.shape[0]
        N = cloud_2d_r.shape[0]
        L_acc = np.zeros(shape=n, dtype=float)
        # D = 1.0 / float(math.pi * N * (N-1))
        D = 1.0 / float(N*N)

        # Cluster radius loop
        for k, t in enumerate(ts):

            # Points loop
            for i in range(N_p):

                hold = cloud_2d_p[i] - cloud_2d_r
                dists = np.sqrt(np.sum(hold*hold, axis=1))
                ids = np.where(dists < t)
                c_0 = cloud_2d_p[i, 0]
                c_1 = cloud_2d_p[i, 1]

                # Neighbour loop
                for j in range(len(ids[0])):

                    r = dists[ids[0][j]]

                    if r > 0:

                        if bord:
                            ####### Compute residual areas on edges
                            r2 = r * r
                            r_2_inv = 1.0 / float(2.0 * r)
                            # Horizontal-Bottom
                            B = -2 * c_0
                            h = y_min - c_1
                            C = c_0*c_0 + h*h - r2
                            s_0, s_1 = self.__q_solver(A, B, C)
                            area_hb = 0
                            if (s_0 is not None) and (s_1 is not None):
                                a = math.fabs(s_0 - s_1)
                                psin = a * r_2_inv
                                if math.fabs(psin) < 1:
                                    rho = 2 * math.asin(a * r_2_inv)
                                    area_hb = 0.5*r2 * (rho - math.sin(rho))
                                else:
                                    area_hb = 0.5 * math.pi * r2
                            # Horizontal-Top
                            h = y_max - c_1
                            C = c_0*c_0 + h*h - r2
                            s_0, s_1 = self.__q_solver(A, B, C)
                            area_ht = 0
                            if (s_0 is not None) and (s_1 is not None):
                                a = math.fabs(s_0 - s_1)
                                psin = a * r_2_inv
                                if math.fabs(psin) < 1:
                                    rho = 2 * math.asin(a * r_2_inv)
                                    area_ht = 0.5*r2 * (rho - math.sin(rho))
                                else:
                                    area_ht = 0.5 * math.pi * r2
                            # Vertical-Left
                            B = -2 * c_1
                            h = x_min - c_0
                            C = h*h + c_1*c_1 - r2
                            s_0, s_1 = self.__q_solver(A, B, C)
                            area_vl = 0
                            if (s_0 is not None) and (s_1 is not None):
                                a = math.fabs(s_0 - s_1)
                                psin = a * r_2_inv
                                if math.fabs(psin) < 1:
                                    rho = 2 * math.asin(a * r_2_inv)
                                    area_vl = 0.5*r2 * (rho - math.sin(rho))
                                else:
                                    area_vl = 0.5 * math.pi * r2
                            # Vertical-Right
                            h = x_max - c_0
                            C = h*h + c_1*c_1 - r2
                            s_0, s_1 = self.__q_solver(A, B, C)
                            area_vr = 0
                            if (s_0 is not None) and (s_1 is not None):
                                a = math.fabs(s_0 - s_1)
                                psin = a * r_2_inv
                                if math.fabs(psin) < 1:
                                    rho = 2 * math.asin(a * r_2_inv)
                                    area_vr = 0.5*r2 * (rho - math.sin(rho))
                                else:
                                    area_vr = 0.5 * math.pi * r2

                            # Compute weighting factor
                            a_c = math.pi * r2
                            area_h = a_c - (area_hb + area_ht + area_vl + area_vr)
                            w = 0
                            if a_c > 0:
                                w = area_h / a_c

                            # point contribution
                            L_acc[k] += w

                        else:
                            L_acc[k] += 1

        # Final computation
        L_acc = np.sqrt((area*L_acc) * D)

        return L_acc, ts

    # Estimates membrane surface area in pixel units. VERY IMPORT: currently all valid
    # modes are just approximations, valid: plane (default), volume.
    # thick: membrane thickness in pixel units, only valid for mode=volume
    def mb_area(self, mode='plane', thick=1, coord=None):

        mb_ids = np.where(self.__seg == MB_LBL)

        if mode == 'plane':
            if coord is not None:
                if coord == 0:
                    a = np.max(mb_ids[1][:]) - np.min(mb_ids[1][:])
                    b = np.max(mb_ids[2][:]) - np.min(mb_ids[2][:])
                elif coord == 1:
                    a = np.max(mb_ids[0][:]) - np.min(mb_ids[0][:])
                    b = np.max(mb_ids[2][:]) - np.min(mb_ids[2][:])
                else:
                    a = np.max(mb_ids[0][:]) - np.min(mb_ids[0][:])
                    b = np.max(mb_ids[1][:]) - np.min(mb_ids[1][:])
                return a * b
            else:
                a = np.max(mb_ids[0][:]) - np.min(mb_ids[0][:])
                b = np.max(mb_ids[1][:]) - np.min(mb_ids[1][:])
                c = np.max(mb_ids[2][:]) - np.min(mb_ids[2][:])
                l = np.sort(np.asarray((a, b, c)))
            return l[1] * l[2]
        elif mode == 'volume':
            vol = (self.__seg == 1).sum()
            return vol / float(thick)
        else:
            error_msg = 'Not valid mode: ' + mode
            raise pexceptions.PySegInputError(expr='mb_area (CMCAnalyzer)', msg=error_msg)


    # Generates a random cloud of points within the membrane
    # n: number of points
    # mode: distribution model, valid: uniform (default) and uniform-2d
    def gen_rand_cloud(self, n, mode='uniform'):

        # Getting membrane coords
        mb_ids = np.where(self.__seg == MB_LBL)
        n_vox = len(mb_ids[0])

        if mode == 'uniform':

            # Generate random
            rand = np.random.uniform(size=n_vox)
            as_rand = np.argsort(rand)

            # Build coordinates
            ids = as_rand[0:n]
            hold = np.zeros(shape=(n, 3), dtype=float)
            hold[:, 0] = mb_ids[0][ids]
            hold[:, 1] = mb_ids[1][ids]
            hold[:, 2] = mb_ids[2][ids]

            return hold

        elif mode == 'uniform-2d':

            x_m, x_M, y_m, y_M = self.__plane_bounds()
            hold = np.zeros(shape=(n, 2), dtype=float)
            hold[:, 0] = np.random.randint(x_m, x_M, n)
            hold[:, 1] = np.random.randint(y_m, y_M, n)

            return hold

        else:
            error_msg = 'Invalid distribution: ' + mode
            raise pexceptions.PySegTransitionError(expr='__gen_rand_cloud (CMCAnalyzer)',
                                                   msg=error_msg)

    # Computes Nearest Neighbour Distance of a cloud of points in a Euclidean
    # space
    # cloud: array with point coordinates (n, 3), if None (default) internal cloud of
    # points are used
    # comp_2d: if True (default False) coordinates are compressed along the least
    # significant dimension
    def nnde(self, cloud=None, comp_2d=False, coord=None):

        # Initialization
        if cloud is None:
            cloud = self.__cloud
        if comp_2d:
            w_cloud = self.compress_plane_2d(cloud, coord=coord)
        else:
            w_cloud = cloud
        dists = np.zeros(shape=w_cloud.shape[0], dtype=float)

        # Shortest distance loop
        for i in range(len(dists)):
            hold = w_cloud[i] - w_cloud
            hold = np.sum(hold*hold, axis=1)
            hold[i] = np.inf
            dists[i] = math.sqrt(np.min(hold))
            # if dists[i] == 0:
            #     print 'Warning: repeated coordinate found!'

        return dists

    # Computes Nearest Neighbour Distance of all points in the membrane segmented volume to
    # the cloud of connectors
    # cloud: array with point coordinates (n, 3), if None (default) internal cloud of
    # points are used
    # comp_2d: if True (default False) coordinates are compressed along the least
    # significant dimension
    def nnde_seg(self, cloud=None, comp_2d=False, coord=None):

        # Initialization
        if cloud is None:
            cloud = self.__cloud
        if comp_2d:
            w_cloud = self.compress_plane_2d(cloud, coord=coord)
        else:
            w_cloud = cloud
        s_cloud = self.get_seg_cloud(comp_2d)
        dists = np.zeros(shape=s_cloud.shape[0], dtype=float)

        # Shortest distance loop
        for i in range(len(dists)):
            hold = s_cloud[i] - w_cloud
            hold = np.sum(hold*hold, axis=1)
            dists[i] = math.sqrt(np.min(hold))

        return dists

    # Computes Farthest Neighbour Distance of a cloud of points in a Euclidean
    # space
    # cloud: array with point coordinates (n, 3), if None (default) internal cloud of
    # points are used
    # comp_2d: if True (default False) coordinates are compressed along the least
    # significant dimension
    def fnde(self, cloud=None, comp_2d=False, coord=None):

        # Initialization
        if cloud is None:
            cloud = self.__cloud
        if comp_2d:
            w_cloud = self.compress_plane_2d(cloud, coord=coord)
        else:
            w_cloud = cloud
        dists = np.zeros(shape=cloud.shape[0], dtype=float)

        # Farthest distance loop
        for i in range(len(dists)):
            hold = w_cloud[i] - w_cloud
            hold = np.sum(hold*hold, axis=1)
            dists[i] = math.sqrt(np.max(hold))

        return dists

    # Compress the cloud of points so as to have 2D coordinates
    # cloud: array with point coordinates (n, 3), if None (default) internal cloud of
    # coord: if not None (default None) it masks the coordinate to delete
    # points are used
    def compress_plane_2d(self, cloud=None, coord=None):

        if cloud is None:
            cloud = self.__cloud

        # If input is already 2D it is returned directly
        if cloud.shape[1] == 2:
            return cloud

        if coord is not None:
            cloud_2d = np.zeros(shape=(cloud.shape[0], 2), dtype=float)
            if coord == 0:
                cloud_2d[:, 0] = cloud[:, 1]
                cloud_2d[:, 1] = cloud[:, 2]
            elif coord == 1:
                cloud_2d[:, 0] = cloud[:, 0]
                cloud_2d[:, 1] = cloud[:, 2]
            else:
                cloud_2d[:, 0] = cloud[:, 0]
                cloud_2d[:, 1] = cloud[:, 1]
        else:
            mb_ids = np.where(self.__seg == MB_LBL)
            a = np.max(mb_ids[0][:]) - np.min(mb_ids[0][:])
            b = np.max(mb_ids[1][:]) - np.min(mb_ids[1][:])
            c = np.max(mb_ids[2][:]) - np.min(mb_ids[2][:])
            ids = np.argsort(np.asarray((a, b, c)))
            cloud_2d = np.zeros(shape=(cloud.shape[0], 2), dtype=float)
            cloud_2d[:, 0] = cloud[:, ids[2]]
            cloud_2d[:, 1] = cloud[:, ids[1]]

        return cloud_2d

    #### Internal functionality area

    # Solves a quadratic equation
    # Returns: two arrays with solutions if they exist, otherwise for the corresponding
    # non-existing solution
    def __q_solver(self, a, b, c):

        h = b*b - 4*a*c
        k = 1. / float(2 * a)
        if h < 0:
            return None, None
        elif h == 0:
            return -b * k, None
        else:
            return (-b + math.sqrt(h))*k, (-b - math.sqrt(h))*k

    # Interface for updating point cluster id lists from the cluster ser returned by Pyto
    def __set_ids(self, clust_set, list_key='kmeans'):

        hl = np.zeros(shape=self.__cloud.shape[0], dtype=int)

        for i, set in enumerate(clust_set):
            if set is not None:
                l = list(set)
                for j in range(len(l)):
                    hl[l[j]-1] = i

        if list_key == 'k-means':
            self.__km_ids = hl
        elif list_key == 'hierarchical':
            self.__hr_ids = hl

    # Get bounds of the compressed plane
    # cloud: array with point coordinates (n, 3), if None (default) internal cloud of
    # points are used
    # coord: if not None (default None) it masks the coordinate to delete
    # Returns: bounds coordinates for the plane (x_m, x_M, y_m, y_M)
    def __plane_bounds(self, cloud=None, coord=None):

        if cloud is None:
            cloud = self.__cloud

        # mb_ids = np.where(self.__seg == MB_LBL)
        # a = np.max(mb_ids[0][:]) - np.min(mb_ids[0][:])
        # b = np.max(mb_ids[1][:]) - np.min(mb_ids[1][:])
        # c = np.max(mb_ids[2][:]) - np.min(mb_ids[2][:])
        # ids = np.argsort(np.asarray((a, b, c)))
        #
        # cloud_2d = np.zeros(shape=(cloud.shape[0], 2), dtype=float)
        # cloud_2d[:, 0] = cloud[:, ids[2]]
        # cloud_2d[:, 1] = cloud[:, ids[1]]

        cloud_2d = self.compress_plane_2d(cloud, coord=coord)

        return cloud_2d[:, 0].min(), cloud_2d[:, 0].max(), \
               cloud_2d[:, 1].min(), cloud_2d[:, 1].max()



