"""
General Classes for modelling filaments

# Author: Antonio Martinez-Sanchez (Max Planck Institute for Biochemistry)
# Date: 15.09.15
"""

__author__ = 'martinez'

import pyseg as ps
import heapq as pq
from pyseg.globals import *
from pyseg.filament.variables import *
from pyseg import pexceptions
# import pexceptions

##### Global variables

WHITE = 0
GRAY = 1
BLACK = 2

#####################################################################################################
#   Priority Queue for efficient GraphMCF vertices insertion-extraction in Dijkstra search
#   Designe speciffically for being used by FilPerVisitor().find_max_per_filament()
#
class DijkstraVQueue(object):

    def __init__(self):
        self.__pq = list()
        self.__finder = dict()

    def get_nelem(self):
        return len(self.__pq)

    def is_empty(self):
        if len(self.__pq) == 0:
            return True
        else:
            return False

    # Insert (or update) a vertex in ordered position in the priority queue
    # v_id: vertex id in GraphMCF
    def insert(self, v_id, dst):
        if v_id in self.__finder:
            # Mark as removed
            entry = self.__finder.pop(v_id)
            entry[-1] = None
        entry = [dst, v_id]
        self.__finder[v_id] = entry
        pq.heappush(self.__pq, entry)

    # Extract vertex id with the minimum distance
    def extract(self):
        while self.__pq:
            dst, v_id = pq.heappop(self.__pq)
            if v_id is not None:
                del self.__finder[v_id]
                return v_id
        raise KeyError('pop from an empty priority queue')

#####################################################################################################
#   Class for implementing the functionality of Base class DijkstraVisitor of graph_tool package, with
#   the intention of finding filaments in a graph, distance es measured by edge length property
#
#
class FilVisitor(gt.DijkstraVisitor):

    # graph: input graph gt
    # source: source vertex
    # min_dist: minimum filament length
    # max_dist: maximum filament length and distance for stopping the search
    def __init__(self, graph, source, min_len, max_len):
        self.__graph = graph
        self.__source = source
        self.__prop_e_l = graph.edge_properties[ps.globals.SGT_EDGE_LENGTH]
        self.__prop_v_d = graph.new_vertex_property('float')
        self.__prop_v_d.get_array()[:] = MAX_FLOAT * np.ones(shape=(self.__graph.num_vertices()),
                                                             dtype=float)
        self.__prop_v_p = graph.new_vertex_property('int')
        self.__prop_v_e = np.zeros(shape=self.__graph.num_vertices(), dtype=object)
        self.__min_len = min_len
        self.__max_len = max_len
        self.__v_paths = list()
        self.__e_paths = list()
        # print 'A new visitor has born!'

    # Return Vertex and edges id for filaments paths
    def get_paths(self):
        return self.__v_paths, self.__e_paths

    def discover_vertex(self, u):
        # print 'vertex discovered -> p(' + str(int(u)) + ')'
        if int(u) == int(self.__source):
            self.__prop_v_d[u] = 0.
            self.__prop_v_p[u] = int(u)

    def edge_relaxed(self, e):
        # print 'edge relaxed -> e(' + str(int(e.source())) + ', ' + str(int(e.target())) + ')'
        (u, v) = tuple(e)
        dist = self.__prop_e_l[e] + self.__prop_v_d[u]
        # Update distances
        self.__prop_v_d[v] = dist
        self.__prop_v_p[v] = int(u)
        self.__prop_v_e[int(v)] = e

    # def examine_vertex(self, u):
        # n_edges = sum(1 for _ in u.all_edges())
        # if n_edges > 0:
        #     print 'Jol'
        # print 'vertex examined -> u(' + str(int(u)) + ') with ' + \
        #       str(n_edges) + ' edges.'

    # def examine_edge(self, e):
    #     print 'edge examined -> e(' + str(int(e.source())) + ', ' + str(int(e.target())) + ')'

    def finish_vertex(self, u):

        dist = self.__prop_v_d[self.__graph.vertex(self.__prop_v_p[u])]
        # print 'source=' + str(self.__source) + '; dist=' + str(dist) + '; p(' + str(int(self.__prop_v_p[u])) + ') <- ' + str(int(u))
        if (dist < MAX_FLOAT) and (dist > self.__min_len) and (self.__prop_v_p[u] != int(u)):

            # Back track trough shortest path
            path = list()
            dists = list()
            edges = list()
            curr = self.__prop_v_p[u]
            curr_v = self.__graph.vertex(curr)
            path.append(curr_v)
            dists.append(self.__prop_v_d[curr_v])
            prev = self.__prop_v_p[curr_v]
            # Stop criterion, source vertex is reached
            while curr != prev:
                next_v = curr
                curr = prev
                curr_v = self.__graph.vertex(curr)
                prev = self.__prop_v_p[curr_v]
                path.append(curr_v)
                dists.append(self.__prop_v_d[curr_v])
                edges.append(self.__prop_v_e[next_v])
            # An isolated vertex cannot be a Filament
            if len(path) < 2:
                return
            path = path[::-1]
            dists = dists[::-1]
            edges = edges[::-1]

            # Storing all valid filaments, those greater than min length
            n_v = len(dists)
            # arg_d = n_v + 1
            # for i in range(n_v):
            #     if dists[i] > self.__min_len:
            #         arg_d = i
            #         break
            # An isolated vertex cannot be a Filament
            # if arg_d < 2:
            #     return
            # for i in range(arg_d, n_v):
            #     self.__v_paths.append(path[0:i])
            #     self.__e_paths.append(edges[0:i-1])
            self.__v_paths.append(path[0:n_v])
            self.__e_paths.append(edges[0:n_v-1])

        # Stop the search
        dist2 = self.__prop_v_d[u]
        if dist2 > self.__max_len:
            raise gt.StopSearch

#####################################################################################################
#   Visitor class "inspired" (and not based) on Dijkstra search algorithm for filament persitence length estimation
#
#
class FilPerVisitor(object):

    # graph: input GraphMCF (network topology and geometry)
    # s_id: source vertex id (for GraphGT)
    # samp_l: distance for sampling curves geometry
    # ktt_limit: total (third) curvature limit (radians [0, 2*pi]
    # min_dist: minimum filament length (it must be greater or equal to 5*samp_l)
    # max_dist: maximum filament length and distance for stopping the search
    def __init__(self, graph, s_id, samp_l, ktt_limit, min_len, max_len):
        self.__graph = graph
        self.__s_id = s_id
        self.__samp_l = float(samp_l)
        hold_min_len = 5 * self.__samp_l
        if min_len < hold_min_len:
            self.__min_len = hold_min_len
        else:
            self.__min_len = float(min_len)
        if max_len < self.__min_len:
            self.__max_len = self.__min_len
        else:
            self.__max_len = float(max_len)
        self.__ktt_limit = float(ktt_limit)

    # External functionality

    # For every vertex computes find the path which generates the maximum persitence length
    # gen_fil: if True (default False) the filament (FilamentUDG) which correspond with the maximum persistence
    #          found is returned
    # Returns: the filament length, and its persistence length, and the correspondent filament, and the
    #          the correspondent filament for geometrical parameters if required
    def find_max_per_filament(self, gen_fil=False):

        # Initialization
        el_id = self.__graph.get_prop_id(SGT_EDGE_LENGTH)
        if el_id is None:
            error_msg = 'Edge length of the reference Graph must be computed before call this function'
            raise pexceptions.PySegInputError(expr='find_max_per_filament (FilPerVisitor)',
                                              msg=error_msg)
        f_len = 0
        hold_fil = None
        hold_film = None
        # vox_apex_l = self.__apex_limit / self.__graph.get_resolution()
        queue = DijkstraVQueue()

        # Dijkstra

        # Initialize all vertices
        dsts = np.inf * np.ones(shape=self.__graph.get_nid(), dtype=np.float32)
        pred_e = (-1) * np.ones(shape=self.__graph.get_nid(), dtype=int)
        pred = self.__s_id * np.ones(shape=self.__graph.get_nid(), dtype=int)
        disc = np.zeros(shape=self.__graph.get_nid(), dtype=bool)
        dsts[self.__s_id] = 0.

        # Discover vertex source
        queue.insert(self.__s_id, 0.)

        while True:

            # Discover vertex u
            try:
                u_id = queue.extract()
            except KeyError:
                # (Priority queue empty) Termination condition
                break
            d_u = dsts[u_id]
            if d_u > self.__max_len:
                # (Maximum length reached) Termination condition
                break
            elif disc[u_id]:
                # Skip this path
                continue

            # print str(queue.get_nelem()), str(d_u)

            neighs, edges = self.__graph.get_vertex_neighbours(u_id)
            for (v, e) in zip(neighs, edges):

                # Examine vertex v (edge e=(u,v))
                v_id, e_id = v.get_id(), e.get_id()
                d_v = dsts[v_id]
                w_e = self.__graph.get_prop_entry_fast(el_id, e_id, 1, np.float32)[0]

                if (w_e+d_u) < d_v:
                    # Edge relaxation
                    new_d_u= w_e + d_u
                    dsts[v_id] = new_d_u
                    pred[v_id] = u_id
                    pred_e[v_id] = e_id
                    queue.insert(v_id, new_d_u)

                elif d_v == np.inf:
                    # Discover vertex v (if not during relaxation)
                    queue.insert(v_id, dsts[v_id])

            # Finish vertex u

            # Check length
            if d_u >= self.__min_len:
                if d_u <= self.__max_len:

                    # Recover current path and its length
                    curve, film = self.__recover_curve(u_id, pred, pred_e)

                    # Minimum value for reliable computation
                    n_samp = int(d_u/self.__samp_l) + 1
                    if n_samp >= 5:

                        # Curve decimation
                        curve_dec = curve.gen_decimated(n_samp)
                        # curve_dec = curve

                        # Maximum apex condition
                        # if curve_dec.get_total_ukt() > self.__ktt_limit:
                        if curve_dec.get_sinuosity() > self.__ktt_limit:
                            # Finish paths through this vertex
                            for v in neighs:
                                disc[v.get_id()] = True
                        else:
                            # Compute persistence length
                            hold_len = curve_dec.get_length()
                            if hold_len > f_len:
                                f_len = hold_len
                                hold_fil = curve_dec
                                hold_film = film

        if gen_fil:
            if hold_fil is not None:
                hold_fil.compute_geom()
            return f_len, hold_film, hold_fil
        else:
            if hold_fil is not None:
                hold_fil.compute_geom()
            return f_len, hold_film

    ##### Internal functionality area

    def __recover_fil(self, v_id, pred_v, pred_e):
        v_ids, e_ids = list(), list()
        c_id = v_id
        while c_id != self.__s_id:
            v_ids.insert(0, c_id)
            e_ids.insert(0, pred_e[c_id])
            c_id = pred_v[c_id]
        v_ids.insert(0, self.__s_id)
        return FilamentUDG(v_ids, e_ids, self.__graph)

    def __recover_curve(self, v_id, pred_v, pred_e):
        p_ids, v_ids, e_ids = list(), list(), list()
        skel = self.__graph.get_skel()
        c_id = v_id
        while c_id != self.__s_id:
            p_ids.insert(0, np.asarray(skel.GetPoint(c_id), dtype=float)*self.__graph.get_resolution())
            v_ids.insert(0, self.__graph.get_vertex(c_id))
            p_ids.insert(0, np.asarray(skel.GetPoint(pred_e[c_id]), dtype=float)*self.__graph.get_resolution())
            e_ids.insert(0, self.__graph.get_edge(pred_e[c_id]))
            c_id = pred_v[c_id]
        p_ids.insert(0, np.asarray(skel.GetPoint(self.__s_id), dtype=float)*self.__graph.get_resolution())
        v_ids.insert(0, self.__graph.get_vertex(self.__s_id))
        return ps.diff_geom.SpaceCurve(p_ids, do_geom=False), FilamentU(self.__graph, v_ids, e_ids, geom=False)

###########################################################################################
# Class for modelling a filament (oriented curve)
###########################################################################################

class Filament(object):

    # graph: parent GraphMCF
    # vertices: list of ordered vertices (VertexMCF)
    # edge: list of ordered edges (EdgeMCF), v{i} -> e{i} -> v{i+1}
    # dst_field: Distance to membrane field
    def __init__(self, graph, vertices, edges, dst_field):
        self.__graph = graph
        self.__vertices = vertices
        self.__edges = edges
        self.__dst_field = dst_field
        self.__ids = self.__get_path_ids()
        self.__coords = self.__get_path_coords()
        self.__pen = self.__get_penetration()
        self.__pent = self.__get_pen_tail()
        self.__fness = self.__get_filamentness()
        self.__sness = self.__get_sness()
        self.__mness = self.__get_mness()
        self.__dens = self.__get_path_den()

    #### Set/Get methods area

    def get_edges(self):
        return self.__edges

    def get_vertices(self):
        return self.__vertices

    def get_num_vertices(self):
        return len(self.__vertices)

    def get_head(self):
        return self.__vertices[0]

    def get_tail(self):
        return self.__vertices[-1]

    def get_tail_coords(self):
        return self.__coords[-1]

    # Return filament path coordinates order from head to tail
    def get_path_coords(self, start_id=None):
        if start_id is None:
            return self.__coords
        else:
            ids = self.get_path_ids()
            s_id = 0
            try:
                s_id = ids.index(start_id)
            except ValueError:
                print('WARNING (Filament.get_path_coords): starting id ' + start_id \
                      + ' is not found.')
            coords = np.zeros(shape=(len(ids)-s_id, 3), dtype=float)
            skel = self.__graph.get_skel()
            for i, idx in enumerate(ids[s_id::]):
                coords[i, :] = skel.GetPoint(ids[i])
            return coords

    # Return filament path ids order from head to tail
    def get_path_ids(self):
        return self.__ids

    # Return vertices coordinates in a numpy array
    def get_vertex_coords(self):
        skel = self.__graph.get_skel()
        coords = np.zeros(shape=(self.get_num_vertices(), 3), dtype=float)
        for i, v in enumerate(self.__vertices):
            coords[i, :] = skel.GetPoint(v.get_id())
        return coords

    # start_id: if not None (default) only ids after it are considered
    def get_length(self, start_id=None):
        if start_id is None:
            length = 0
            for e in self.__edges:
                length += self.__graph.get_edge_length(e)
            return length
        else:
            ids = self.get_path_ids()
            s_id = 0
            try:
                s_id = ids.index(start_id)
            except ValueError:
                print('WARNING (Filament.get_length): starting id ' + start_id \
                      + ' is not found.')
            length = 0
            skel = self.__graph.get_skel()
            for i in range(s_id, len(ids)-1):
                x1, y1, z1 = skel.GetPoint(ids[i])
                x2, y2, z2 = skel.GetPoint(ids[i+1])
                hold = np.asarray((x1-x2, y1-y2, z1-z2), dtype=float)
                length += math.sqrt(np.sum(hold*hold))
            return length * self.__graph.get_resolution()

    # For getting the distance from a contact point to the first vertex in the filament
    # cont_id: id of the contact point
    def get_cont_length(self, cont_id):
        v_id = self.__vertices[1].get_id()
        ids = self.get_path_ids()
        s_id = 0
        t_id = 0
        try:
            s_id = ids.index(cont_id)
        except ValueError:
            print('WARNING (Filament.get_cont_length): id ' + cont_id + ' is not found.')
        try:
            t_id = ids.index(v_id)
        except ValueError:
            print('WARNING (Filament.get_cont_length): id ' + cont_id + ' is not found.')
        length = 0
        skel = self.__graph.get_skel()
        for i in range(s_id, t_id):
            x1, y1, z1 = skel.GetPoint(ids[i])
            x2, y2, z2 = skel.GetPoint(ids[i+1])
            hold = np.asarray((x1-x2, y1-y2, z1-z2), dtype=float)
            length += math.sqrt(np.sum(hold*hold))
        return length * self.__graph.get_resolution()

    # cont: if not None (default), this point is used a head
    def get_head_tail_dist(self, cont=None):
        skel = self.__graph.get_skel()
        if cont is None:
            x_h, y_h, z_h = skel.GetPoint(self.__vertices[0].get_id())
        else:
            x_h, y_h, z_h = skel.GetPoint(cont)
        x_t, y_t, z_t = skel.GetPoint(self.__vertices[-1].get_id())
        hold = np.asarray((x_h-x_t, y_h-y_t, z_h-z_t), dtype=float)
        return math.sqrt(np.sum(hold*hold))

    def get_penetration(self):
        return self.__pen

    # This metric is the product of int and 1/sim
    def get_filamentness(self):
        return self.__fness

    def get_pen_len_ratio(self):
        length = self.get_length()
        if length == 0:
            return 0.
        else:
            return self.get_penetration() / length

    # Get the tail penetration
    def get_pen_tail(self):
        return self.__pent

    def get_sness(self):
        return self.__sness

    def get_mness(self):
        return self.__mness

    # start_id: if not None (default) only ids after it are considered
    def get_dness(self, start_id=None):
            ids = self.get_path_ids()
            s_id = 0
            if start_id is not None:
                try:
                    s_id = ids.index(start_id)
                except ValueError:
                    print('WARNING (Filament.get_dness): starting id ' + start_id \
                          + ' is not found.')
            return np.percentile(1. - self.__dens[s_id::], 5)

    # Computes total curvature
    def get_total_curvature(self, start_id=None):

        # Getting curve coordinates in space
        curve = self.get_path_coords(start_id)
        curve *= self.__graph.get_resolution()

        # Computing curvatures
        curvatures = compute_space_k(curve)

        # Curvature integral
        total_k = 0.
        for i in range(1, curve.shape[0]-1):
            v_i_l1, v_i, v_i_p1 = curve[i-1, :], curve[i, :], curve[i+1, :]
            h_1 = v_i_p1 - v_i
            h_2 = v_i-v_i_l1
            h_1 = math.sqrt(h_1[0]*h_1[0] + h_1[1]*h_1[1] + h_1[2]*h_1[2])
            h_2 = math.sqrt(h_2[0]*h_2[0] + h_2[1]*h_2[1] + h_2[2]*h_2[2])
            total_k += (0.5 * (h_1 + h_2) * curvatures[i-1])

        return total_k

    # Computes a smoothness metric based on total curvature
    def get_smoothness(self, start_id=None):

        # Getting curve coordinates in space
        curve = self.get_path_coords(start_id)
        curve *= self.__graph.get_resolution()

        # Computing curvatures
        curvatures = compute_space_k(curve)

        # Square for avoiding orientation information
        curvatures *= curvatures

        # Curvature integral
        total_k = 0.
        for i in range(1, curve.shape[0]-1):
            v_i_l1, v_i, v_i_p1 = curve[i-1, :], curve[i, :], curve[i+1, :]
            h_1 = v_i_p1 - v_i
            h_2 = v_i-v_i_l1
            h_1 = math.sqrt(h_1[0]*h_1[0] + h_1[1]*h_1[1] + h_1[2]*h_1[2])
            h_2 = math.sqrt(h_2[0]*h_2[0] + h_2[1]*h_2[1] + h_2[2]*h_2[2])
            hold = 0.5 * (h_1 + h_2) * curvatures[i-1]
            total_k += (hold * hold)

        length = self.get_length(start_id)
        if length <= 0:
            return 0.
        else:
            return total_k / length

    # Computes maximum local curvature along the whole network
    def get_max_curvature(self, start_id=None):

        # Getting curve coordinates in space
        curve = self.get_path_coords(start_id)
        curve *= self.__graph.get_resolution()

        # Computing curvatures
        curvatures = np.absolute(compute_space_k(curve))

        # Maximum
        return curvatures.max()

    # Computes curve sinuosity (ration between length and distance between extremes)
    # mode: if 1 (default) vertex mode, otherwise path mode
    def get_sinuosity(self, start_id=None):
        length = self.get_length(start_id)
        if length == 0:
            return 0
        dst = self.get_head_tail_dist(cont=start_id)
        if dst == 0:
            return 0
        else:
            return length / dst

    ##### Internal functionality area

    def __get_path_den(self):
        skel = self.__graph.get_skel()
        ids = self.get_path_ids()
        density = self.__graph.get_density()
        dens = np.zeros(shape=len(ids), dtype=float)
        for i, idx in enumerate(ids):
            x, y, z = skel.GetPoint(idx)
            dens[i] = ps.globals.trilin3d(density, (x, y, z))
        return dens

    def __get_path_ids(self):
        ids = self.__graph.get_edge_ids(self.__edges[0])
        if ids[0] != self.get_head().get_id():
            ids = ids[::-1]
        for i in range(1, len(self.__edges)):
            hold_ids = self.__graph.get_edge_ids(self.__edges[i])
            if ids[-1] == hold_ids[-1]:
                hold_ids = hold_ids[::-1]
            # if ids[-1] != hold_ids[0]:
            #     print 'Jol'
            ids += hold_ids[1::]
        return ids

    def __get_path_coords(self):
        skel = self.__graph.get_skel()
        ids = self.get_path_ids()
        coords = np.zeros(shape=(len(ids), 3), dtype=float)
        for i in range(coords.shape[0]):
            coords[i, :] = skel.GetPoint(ids[i])
        return coords

    def __get_penetration(self):
        mx = 0.
        coords = self.get_path_coords()
        for c in coords:
            hold = ps.globals.trilin3d(self.__dst_field, c)
            if hold > mx:
                mx = hold
        return mx

    def __get_filamentness(self):

        # Initialization
        density = self.__graph.get_density()
        if density is None:
            error_msg = 'The parent GraphMCF must have a geometry for computing filamentness'
            raise pexceptions.PySegInputError(expr='filamentness (Filament)',
                                              msg=error_msg)

        # Edges filament loop
        hold_mx = 0
        area_c = .0
        length = .0
        coords = self.get_path_coords()
        for i in range(coords.shape[0]-1):
            # Compute separatrices integral (trapezoidal rule), length and maximum
            x1, y1, z1 = coords[i, 0], coords[i, 1], coords[i, 2]
            f_1 = ps.globals.trilin3d(density, (x1, y1, z1))
            x2, y2, z2 = coords[i+1, 0], coords[i+1, 1], coords[i+1, 2]
            xh, yh, zh = x1 - x2, y1 - y2, z1 - z2
            dist = math.sqrt(xh*xh + yh*yh + zh*zh)
            length += dist
            f_2 = ps.globals.trilin3d(density, (x2, y2, z2))
            area_c += (0.5 * dist * (f_1 + f_2))
            if f_2 > hold_mx:
                hold_mx = f_2

        # Filamentness computation
        if (hold_mx <= 0) or (length <= 0):
            return 0.
        else:
            return area_c / (hold_mx * length)

    def __get_pen_tail(self):
        skel = self.__graph.get_skel()
        return ps.globals.trilin3d(self.__dst_field,
                                   skel.GetPoint(self.get_tail().get_id()))

    def __get_sness(self):
        hold = 0.
        key_id = self.__graph.get_prop_id(ps.globals.STR_FIELD_VALUE)
        for e in self.__edges:
            t = self.__graph.get_prop_entry_fast(key_id, e.get_id(), 1, float)[0]
            hold += (1-t)
        n_edges = float(len(self.__edges))
        return 1. - (hold / n_edges)

    def __get_mness(self):
        hold = 0.
        key_id = self.__graph.get_prop_id(ps.globals.STR_FIELD_VALUE)
        for e in self.__edges:
            t = self.__graph.get_prop_entry_fast(key_id, e.get_id(), 1, float)[0]
            if t > hold:
                hold = t
        return 1. - hold

###########################################################################################
# Class for modelling a filament (unoriented curve)
###########################################################################################

class FilamentU(object):

    # graph: parent GraphMCF
    # vertices: list of ordered vertices (VertexMCF)
    # edge: list of ordered edges (EdgeMCF), v{i} -> e{i} -> v{i+1}
    # geom: if True (default) geometric parameters are computed during building
    def __init__(self, graph, vertices, edges, geom=True):
        self.__graph = graph
        self.__vertices = vertices
        self.__edges = edges
        if geom:
            self.__ids = self.__get_path_ids()
            self.__coords = self.__get_path_coords()
            self.__fness = self.__get_filamentness()

    #### Set/Get methods area

    def get_edges(self):
        return self.__edges

    def get_vertices(self):
        return self.__vertices

    def get_num_vertices(self):
        return len(self.__vertices)

    def get_head(self):
        return self.__vertices[0]

    def get_tail(self):
        return self.__vertices[-1]

    # Return filament path coordinates order from head to tail
    def get_path_coords(self):
        return self.__coords

    # Return filament path ids order from head to tail
    def get_path_ids(self):
        return self.__ids

    # Return vertices coordinates in a numpy array
    def get_vertex_coords(self):
        skel = self.__graph.get_skel()
        coords = np.zeros(shape=(self.get_num_vertices(), 3), dtype=float)
        for i, v in enumerate(self.__vertices):
            coords[i, :] = skel.GetPoint(v.get_id())
        return coords

    # mode: if 1 (default) vertex mode, otherwise path mode
    # start_id: if not None (default) only ids after it are considered, only valid for path mode
    def get_length(self, mode=1, start_id=None):
        if mode == 1:
            length = 0
            coords = self.get_vertex_coords()
            for i in range(coords.shape[0] - 1):
                x1, y1, z1 = coords[i, 0], coords[i, 1], coords[i, 2]
                x2, y2, z2 = coords[i+1, 0], coords[i+1, 1], coords[i+1, 2]
                hold = np.asarray((x1-x2, y1-y2, z1-z2), dtype=float)
                length += math.sqrt(np.sum(hold*hold))
            return length * self.__graph.get_resolution()
        else:
            if start_id is None:
                length = 0
                for e in self.__edges:
                    length += self.__graph.get_edge_length(e)
                return length
            else:
                ids = self.get_path_ids()
                s_id = 0
                try:
                    s_id = ids.index(start_id)
                except ValueError:
                    print('WARNING (FilamentU.get_length): starting id ' + start_id \
                          + ' is not found.')
                length = 0
                skel = self.__graph.get_skel()
                for i in range(s_id, len(ids)-1):
                    x1, y1, z1 = skel.GetPoint(ids[i])
                    x2, y2, z2 = skel.GetPoint(ids[i+1])
                    hold = np.asarray((x1-x2, y1-y2, z1-z2), dtype=float)
                    length += math.sqrt(np.sum(hold*hold))
                return length * self.__graph.get_resolution()

    # cont: if not None (default), this point is used a head
    def get_head_tail_dist(self, cont=None):
        skel = self.__graph.get_skel()
        if cont is None:
            x_h, y_h, z_h = skel.GetPoint(self.__vertices[0].get_id())
        else:
            x_h, y_h, z_h = skel.GetPoint(cont)
        x_t, y_t, z_t = skel.GetPoint(self.__vertices[-1].get_id())
        hold = np.asarray((x_h-x_t, y_h-y_t, z_h-z_t), dtype=float)
        return math.sqrt(np.sum(hold*hold)) * self.__graph.get_resolution()

    # This metric is the product of int and 1/sim
    def get_filamentness(self):
        return self.__fness

    # Computes total curvature
    # mode: if 1 (default) vertex mode, otherwise path mode
    def get_total_curvature(self, mode=1):

        # Getting curve coordinates in space
        if mode == 1:
            curve = self.get_vertex_coords()
        else:
            curve = self.get_path_coords()
        curve *= self.__graph.get_resolution()

        # Computing curvatures
        curvatures = compute_space_k(curve)

        # Curvature integral
        total_k = 0.
        for i in range(1, curve.shape[0]-1):
            v_i_l1, v_i, v_i_p1 = curve[i-1, :], curve[i, :], curve[i+1, :]
            h_1 = v_i_p1 - v_i
            h_2 = v_i-v_i_l1
            h_1 = math.sqrt(h_1[0]*h_1[0] + h_1[1]*h_1[1] + h_1[2]*h_1[2])
            h_2 = math.sqrt(h_2[0]*h_2[0] + h_2[1]*h_2[1] + h_2[2]*h_2[2])
            total_k += (0.5 * (h_1 + h_2) * curvatures[i-1])

        return total_k

    # Computes a smoothness metric based on total curvature
    # mode: if 1 (default) vertex mode, otherwise path mode
    def get_smoothness(self, mode=1):

        # Getting curve coordinates in space
        if mode == 1:
            curve = self.get_vertex_coords()
        else:
            curve = self.get_path_coords()
        curve *= self.__graph.get_resolution()

        # Computing curvatures
        curvatures = compute_space_k(curve)

        # Square for avoiding orientation information
        curvatures *= curvatures

        # Curvature integral
        total_k = 0.
        for i in range(1, curve.shape[0]-1):
            v_i_l1, v_i, v_i_p1 = curve[i-1, :], curve[i, :], curve[i+1, :]
            h_1 = v_i_p1 - v_i
            h_2 = v_i-v_i_l1
            h_1 = math.sqrt(h_1[0]*h_1[0] + h_1[1]*h_1[1] + h_1[2]*h_1[2])
            h_2 = math.sqrt(h_2[0]*h_2[0] + h_2[1]*h_2[1] + h_2[2]*h_2[2])
            hold = 0.5 * (h_1 + h_2) * curvatures[i-1]
            total_k += (hold * hold)

        length = self.get_length(mode=mode)
        if length <= 0:
            return 0.
        else:
            return total_k / length

    # Computes maximum local curvature along the whole network
    # mode: if 1 (default) vertex mode, otherwise path mode
    def get_max_curvature(self, mode=1):

        # Getting curve coordinates in space
        if mode == 1:
            curve = self.get_vertex_coords()
        else:
            curve = self.get_path_coords()
        curve *= self.__graph.get_resolution()

        # Computing curvatures
        curvatures = np.absolute(compute_space_k(curve))

        # Maximum
        return curvatures.max()

    # Computes curve sinuosity (ration between length and distance between extremes)
    # mode: if 1 (default) vertex mode, otherwise path mode
    def get_sinuosity(self, mode=1):
        length = self.get_length(mode=mode)
        if length == 0:
            return 0
        dst = self.get_head_tail_dist()
        if dst == 0:
            return 0
        else:
            return length / dst

##### Internal functionality area

    def __get_path_den(self):
        skel = self.__graph.get_skel()
        ids = self.get_path_ids()
        density = self.__graph.get_density()
        dens = np.zeros(shape=len(ids), dtype=float)
        for i, idx in enumerate(ids):
            x, y, z = skel.GetPoint(idx)
            dens[i] = ps.globals.trilin3d(density, (x, y, z))
        return dens

    def __get_path_ids(self):
        ids = self.__graph.get_edge_ids(self.__edges[0])
        if ids[0] != self.get_head().get_id():
            ids = ids[::-1]
        for i in range(1, len(self.__edges)):
            hold_ids = self.__graph.get_edge_ids(self.__edges[i])
            if ids[-1] == hold_ids[-1]:
                hold_ids = hold_ids[::-1]
            # if ids[-1] != hold_ids[0]:
            #     print 'Jol'
            ids += hold_ids[1::]
        return ids

    def __get_path_coords(self):
        skel = self.__graph.get_skel()
        ids = self.get_path_ids()
        coords = np.zeros(shape=(len(ids), 3), dtype=float)
        for i in range(coords.shape[0]):
            coords[i, :] = skel.GetPoint(ids[i])
        return coords

    def __get_filamentness(self):

        # Initialization
        density = self.__graph.get_density()
        if density is None:
            error_msg = 'The parent GraphMCF must have a geometry for computing filamentness'
            raise pexceptions.PySegInputError(expr='filamentness (Filament)',
                                              msg=error_msg)

        # Edges filament loop
        hold_mx = 0
        area_c = .0
        length = .0
        coords = self.get_path_coords()
        for i in range(coords.shape[0]-1):
            # Compute separatrices integral (trapezoidal rule), length and maximum
            x1, y1, z1 = coords[i, 0], coords[i, 1], coords[i, 2]
            f_1 = ps.globals.trilin3d(density, (x1, y1, z1))
            x2, y2, z2 = coords[i+1, 0], coords[i+1, 1], coords[i+1, 2]
            xh, yh, zh = x1 - x2, y1 - y2, z1 - z2
            dist = math.sqrt(xh*xh + yh*yh + zh*zh)
            length += dist
            f_2 = ps.globals.trilin3d(density, (x2, y2, z2))
            area_c += (0.5 * dist * (f_1 + f_2))
            if f_2 > hold_mx:
                hold_mx = f_2

        # Filamentness computation
        if (hold_mx <= 0) or (length <= 0):
            return 0.
        else:
            return area_c / (hold_mx * length)

###########################################################################################
# Class for modelling a filament which is lighter than Filament
###########################################################################################

class FilamentL(object):

    # v_ids: list of vertices ids
    # p_ids: list of ordered points in graph skeleton from contact point to vertex
    def __init__(self, v_ids, p_ids):
        self.__v_ids = v_ids
        self.__p_ids = p_ids


    #### Get/Set methods

    def get_point_ids(self):
        return self.__p_ids

    def get_vertex_ids(self):
        return self.__v_ids

###########################################################################################
# Class derived from FilamentL with additional functionality for getting differential geometry
# descriptors
###########################################################################################

class FilamentLDG(FilamentL):

    # v_ids: list of vertices ids
    # p_ids: list of ordered points in graph skeleton from contact point to vertex
    # graph: parent GraphMCF
    def __init__(self, v_ids, p_ids, graph):
        super(FilamentLDG, self).__init__(v_ids, p_ids)
        self.__graph = graph
        # Purge curve coordinates
        coords = list()
        skel = self.__graph.get_skel()
        hold_pt = skel.GetPoint(p_ids[0])
        for i in p_ids[1:]:
            curr_pt = skel.GetPoint(i)
            if (hold_pt[0] != curr_pt[0]) or (hold_pt[1] != curr_pt[1]) or (hold_pt[2] != curr_pt[2]):
                coords.append(np.asarray(curr_pt, dtype=float))
                hold_pt = curr_pt
        self.__curve = ps.diff_geom.SpaceCurve(coords)

    #### Get/Set methods

    def get_vertices(self):
        vertices = list()
        for v_id in self.get_vertex_ids():
            vertices.append(self.__graph.get_vertex(v_id))
        return vertices

    def get_vertex_coords(self):
        skel = self.__graph.get_skel()
        coords = list()
        for v_id in self.get_vertex_ids():
            coords.append(skel.GetPoint(v_id))
        return np.asarray(coords, dtype=float)

    def get_curve_coords(self):
        return self.__curve.get_samples()

    def get_curve(self):
        return self.__curve

    # Returns unsigned total curvature
    def get_total_k(self):
        return self.__curve.get_total_uk()

    # Returns unsigned total torsion
    def get_total_t(self):
        return self.__curve.get_total_ut()

    # Returns normal symmetry
    def get_total_ns(self):
        return self.__curve.get_normal_symmetry()

    # Returns binormal symmetry
    def get_total_bs(self):
        return self.__curve.get_binormal_symmetry()

    # Returns sinuosity
    def get_sinuosity(self):
        return self.__curve.get_sinuosity()

    # Returns apex length (in nm)
    def get_apex_length(self):
        return self.__curve.get_apex_length() * self.__graph.get_resolution()

    # Returns geodesic length (in nm)
    def get_length(self):
        return self.__curve.get_length() * self.__graph.get_resolution()

    # Distance between the two extrema
    def get_dst(self):
        pt1, pt2 = self.__curve.get_start_sample(), self.__curve.get_end_sample()
        hold = pt1 - pt2
        return math.sqrt((hold * hold).sum())

    def get_filamentness(self):

        # Initialization
        density = self.__graph.get_density()
        if density is None:
            error_msg = 'The parent GraphMCF must have a geometry for computing filamentness'
            raise pexceptions.PySegInputError(expr='filamentness (Filament)',
                                              msg=error_msg)

        # Edges filament loop
        hold_mx = 0
        area_c = .0
        length = .0
        coords = self.get_curve_coords()
        for i in range(coords.shape[0]-1):
            # Compute separatrices integral (trapezoidal rule), length and maximum
            x1, y1, z1 = coords[i, 0], coords[i, 1], coords[i, 2]
            f_1 = ps.globals.trilin3d(density, (x1, y1, z1))
            x2, y2, z2 = coords[i+1, 0], coords[i+1, 1], coords[i+1, 2]
            xh, yh, zh = x1 - x2, y1 - y2, z1 - z2
            dist = math.sqrt(xh*xh + yh*yh + zh*zh)
            length += dist
            f_2 = ps.globals.trilin3d(density, (x2, y2, z2))
            area_c += (0.5 * dist * (f_1 + f_2))
            if f_2 > hold_mx:
                hold_mx = f_2

        # Filamentness computation
        if (hold_mx <= 0) or (length <= 0):
            return 0.
        else:
            return area_c / (hold_mx * length)

    ##### External functionality

    # Equally spaced of the filament downsampling, the starting and ending point will be the same as in original
    # filamente but the length of the last piece is not guaranteed
    # samp: sampling distance
    # Returns: the new downsampled curve in space, if the filament has a length < than 3*samp then None is returned
    def gen_downsample(self, samp):

        # Getting current samples and lengths
        out_coords = list()
        coords, lengths = self.__curve.get_samples(), self.__curve.get_lengths()
        nc = coords.shape[0]

        # Checking trivial events
        if nc <= 2:
            return None

        # Finding samples
        hold_coord = coords[0, :]
        out_coords.append(hold_coord)
        curr_l = samp
        for (length, coord) in zip(lengths[1:nc-1], coords[1:nc-1]):
            # Take a sample
            if length > curr_l:
                vect = coord - hold_coord
                vect_n = math.sqrt((vect * vect).sum())
                if vect_n <= 0:
                    out_coords.append(coord)
                else:
                    vect /= vect_n
                    dif_l = length - curr_l
                    out_coords.append(hold_coord+dif_l*vect)
                curr_l += samp
            hold_coord = coord

        if len(out_coords) < 2:
            return None
        else:
            out_coords.append(coords[-1, :])
            return np.asarray(out_coords, dtype=float)

###########################################################################################
# Class derived from FilamentLDG but here the curve is automatically inferred from an input sequence
# of EdgeMCF
###########################################################################################

class FilamentUDG(FilamentLDG):

    # v_ids: list of vertices ids
    # p_ids: list of edges ids
    # graph: parent GraphMCF
    def __init__(self, v_ids, e_ids, graph):
        # Find ordered point ids
        p_ids = list()
        for v_id, e_id in zip(v_ids[:-1], e_ids):
            edge = graph.get_edge(e_id)
            s_id = edge.get_source_id()
            h_p_ids = graph.get_edge_ids(edge)
            if s_id != v_id:
                h_p_ids = h_p_ids[::-1]
            p_ids += h_p_ids
        # Call base constructor
        super(FilamentUDG, self).__init__(v_ids, p_ids, graph)

    #### Get/Set methods

