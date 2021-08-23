"""
General Classes for the package

# Author: Antonio Martinez-Sanchez (Max Planck Institute for Biochemistry)
# Date: 15.09.15
"""

__author__ = 'martinez'

from pyseg.globals import *
import graph_tool.all as gt
from .variables import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

try:
    import pickle as pickle
except:
    import pickle

######## File global variables

eps_comp = 0.001


#####################################################################################################
#   Class for implementing the functionality of Base class DijkstraVisitor of graph_tool package,
#   with the intention of finding filaments in a graph, distance es measured by
#   edge length property (STR_2GT_EL)
#
#
class FilVisitor2(gt.DijkstraVisitor):

    # graph: input graph gt
    # source: source vertex
    # min_dist: minimum filament length
    # max_dist: maximum filament length and distance for stopping the search
    def __init__(self, graph, source, min_len, max_len):
        self.__graph = graph
        self.__source = source
        self.__prop_e_l = graph.edge_properties[STR_2GT_EL]
        self.__prop_v_d = graph.new_vertex_property('float')
        self.__prop_v_d.get_array()[:] = MAX_FLOAT * np.ones(shape=(self.__graph.num_vertices()),
                                                             dtype=np.float)
        self.__prop_v_p = graph.new_vertex_property('int')
        self.__prop_v_e = np.zeros(shape=self.__graph.num_vertices(), dtype=object)
        self.__min_len = min_len
        self.__max_len = max_len
        self.__v_paths = list()
        self.__e_paths = list()

    # Return Vertex and edges id for filaments paths
    def get_paths(self):
        return self.__v_paths, self.__e_paths

    def discover_vertex(self, u):
        if int(u) == int(self.__source):
            self.__prop_v_d[u] = 0.
            self.__prop_v_p[u] = int(u)

    def edge_relaxed(self, e):
        (u, v) = tuple(e)
        dist = self.__prop_e_l[e] + self.__prop_v_d[u]
        # Update distances
        self.__prop_v_d[v] = dist
        self.__prop_v_p[v] = int(u)
        self.__prop_v_e[int(v)] = e

    def finish_vertex(self, u):

        dist = self.__prop_v_d[self.__graph.vertex(self.__prop_v_p[u])]
        if (dist < MAX_FLOAT) and (dist > self.__min_len) and (self.__prop_v_p[u] != int(u)):

            # Back track trough shortest path
            path = list()
            edges = list()
            curr = self.__prop_v_p[u]
            curr_v = self.__graph.vertex(curr)
            path.append(curr_v)
            prev = self.__prop_v_p[curr_v]
            # Stop criterion, source vertex is reached
            while curr != prev:
                next_v = curr
                curr = prev
                curr_v = self.__graph.vertex(curr)
                prev = self.__prop_v_p[curr_v]
                path.append(curr_v)
                edges.append(self.__prop_v_e[next_v])
            # An isolated vertex cannot be a Filament
            if len(path) < 2:
                return
            path = path[::-1]
            edges = edges[::-1]

            # Storing all valid filaments, those greater than min length
            self.__v_paths.append(path)
            self.__e_paths.append(edges)

        # Stop the search
        dist2 = self.__prop_v_d[u]
        if dist2 > self.__max_len:
            raise gt.StopSearch

###################################################################################################
# Class for comparing and plotting different function graphs
#
#
class FuncComparator(object):

    #### Set/Get methods
    # g_name: name for the comparator
    def __init__(self, g_name):
        self.__g_name = g_name
        self.__xs = list()
        self.__ys = list()
        self.__names = list()

    #### External functionality area

    # name: string with function name
    # x_arr: numpy array with de independent variable
    # y_arr: numpy array with de dependant variable
    def insert_graph(self, name, x_arr, y_arr):

        # Check dimensions
        if len(self.__xs) == 0:
            if x_arr.shape != y_arr.shape:
                error_msg = 'Input arrays have not the same dimension.'
                raise pexceptions.PySegInputError(expr='insert_graph (FuncComparer)', msg=error_msg)
        else:
            if (self.__xs[0].shape != x_arr.shape) or (self.__ys[0].shape != y_arr.shape):
                error_msg = 'Input arrays have different dimension compared with previous ones.'
                raise pexceptions.PySegInputError(expr='insert_graph (FuncComparer)', msg=error_msg)
            # elif np.absolute(self.__x - x_arr).sum() > eps_comp:
            #     error_msg = 'Independent variable must be element-wise equal to the previous ones.'
            #     raise pexceptions.PySegInputError(expr='insert_graph (FuncComparer)', msg=error_msg)

        # Insertion
        self.__names.append(name)
        self.__xs.append(np.asarray(x_arr, dtype=np.float))
        self.__ys.append(np.asarray(y_arr, dtype=np.float))

    # Plots comparison among already inserted functions
    # block: if True (default False) waits for closing windows for finishing the execution
    # plot_inserted: if True (default) all inserted graphs are plotted
    # leg_num: if False (default) legend will contain inserted name, otherwise it will contain
    #          numbers
    # leg_loc: legend location (default 4)
    def plot_comparison(self, block=False, plot_inserted=True, leg_num=False, leg_loc=4):

        # Computing statistics
        ys = np.array(self.__ys, dtype=np.float)
        xs = np.array(self.__xs, dtype=np.float)
        means = ys.mean(axis=0)
        medians = np.median(ys, axis=0)
        variances = ys.var(axis=0)
        stds = ys.std(axis=0)
        crs = np.corrcoef(ys)
        # crs = np.corrcoef(ys, y=xs)

        # Plotting comparison statistics
        fig_count = 0
        # Plot inserted graphs
        if plot_inserted:
            fig_count += 1
            plt.figure(fig_count)
            plt.title(self.__g_name + ' functions')
            plt.xlabel('Sample')
            plt.ylabel('Value')
            lines = list()
            color = cm.rainbow(np.linspace(0, 1, len(self.__ys)))
            if leg_num:
                for (num, x_arr, y_arr, c) in zip(np.arange(1, len(self.__names)+1), self.__xs, self.__ys, color):
                    line, = plt.plot(x_arr, y_arr, c=c, label=str(num))
                    lines.append(line)
            else:
                for (name, x_arr, y_arr, c) in zip(self.__names, self.__xs, self.__ys, color):
                    line, = plt.plot(x_arr, y_arr, c=c, label=name)
                    lines.append(line)
            plt.legend(handles=lines, loc=leg_loc, ncol=1, borderaxespad=0.)

        # Check if equally sampled variable
        eqs = True
        for x_arr in self.__xs:
            if np.absolute(self.__xs[0] - x_arr).sum() > eps_comp:
                eqs = False

        if eqs:
            # Plot means
            fig_count += 1
            plt.figure(fig_count)
            plt.title('Means and std for ' + self.__g_name)
            plt.xlabel('Sample')
            plt.ylabel('Mean - std')
            plt.plot(self.__xs[0], means)
            plt.plot(self.__xs[0], means + stds, 'k--')
            plt.plot(self.__xs[0], means - stds, 'k--')
            # Plot medians
            fig_count += 1
            plt.figure(fig_count)
            plt.title('Medians for ' + self.__g_name)
            plt.xlabel('Sample')
            plt.ylabel('Median')
            plt.plot(self.__xs[0], medians)
            # Plot variances
            fig_count += 1
            plt.figure(fig_count)
            plt.title('Variances for ' + self.__g_name)
            plt.xlabel('Sample')
            plt.ylabel('Var')
            plt.plot(self.__xs[0], variances)
            # Plot correlation
            fig_count += 1
            plt.figure(fig_count)
            plt.title('Cross-correlation maxtrix for ' + self.__g_name)
            plt.xlabel('Insertion')
            plt.ylabel('Insertion')
            plt.pcolor(crs, cmap='jet', vmin=-1, vmax=1)
            plt.colorbar()
            plt.yticks(np.arange(.5, crs.shape[1]+.5),list(range(crs.shape[1])))
            plt.xticks(np.arange(.5, crs.shape[0]+.5),list(range(crs.shape[0])))

        # Show
        plt.show(block=block)

    # Stores plot figures in a png file
    # path: path to the folder where figures will be stored
    # plot_inserted: if True (default) all inserted graphs are stored
    # leg_num: if False (default) legend will contain inserted name, otherwise it will contain
    #          numbers
    # leg_loc: legend location (default 4)
    def store_figs(self, path, plot_inserted=True, leg_num=False, leg_loc=4):

        # Computing statistics
        ys = np.array(self.__ys, dtype=np.float)
        xs = np.array(self.__xs, dtype=np.float)
        means = ys.mean(axis=0)
        medians = np.median(ys, axis=0)
        variances = ys.var(axis=0)
        stds = ys.std(axis=0)
        crs = np.corrcoef(ys)
        # crs = np.corrcoef(ys, y=xs)

        # Plotting comparison statistics
        fig_count = 0
        # Plot inserted graphs
        if plot_inserted:
            fig_count += 1
            plt.figure(fig_count)
            plt.title(self.__g_name + ' functions')
            plt.xlabel('Sample')
            plt.ylabel('Value')
            lines = list()
            color = cm.rainbow(np.linspace(0, 1, len(self.__ys)))
            if leg_num:
                for (num, x_arr, y_arr, c) in zip(np.arange(1, len(self.__names)+1), self.__xs, self.__ys, color):
                    line, = plt.plot(x_arr, y_arr, c=c, label=str(num))
                    lines.append(line)
            else:
                for (name, x_arr, y_arr, c) in zip(self.__names, self.__xs, self.__ys, color):
                    line, = plt.plot(x_arr, y_arr, c=c, label=name)
                    lines.append(line)
            plt.legend(handles=lines, loc=leg_loc, ncol=1, borderaxespad=0.)
            plt.savefig(path+'/all.png')
            plt.close()

        # Check if equally sampled variable
        eqs = True
        for x_arr in self.__xs:
            if np.absolute(self.__xs[0] - x_arr).sum() > eps_comp:
                eqs = False

        if eqs:
            # Plot means
            fig_count += 1
            plt.figure(fig_count)
            plt.title('Means and std for ' + self.__g_name)
            plt.xlabel('Sample')
            plt.ylabel('Mean - Std')
            plt.plot(self.__xs[0], means)
            plt.plot(self.__xs[0], means + stds, 'k--')
            plt.plot(self.__xs[0], means - stds, 'k--')
            plt.savefig(path+'/mn.png')
            plt.close()
            # Plot medians
            fig_count += 1
            plt.figure(fig_count)
            plt.title('Medians for ' + self.__g_name)
            plt.xlabel('Sample')
            plt.ylabel('Median')
            plt.plot(self.__xs[0], medians)
            plt.savefig(path+'/med.png')
            plt.close()
            # Plot variances
            fig_count += 1
            plt.figure(fig_count)
            plt.title('Variances for ' + self.__g_name)
            plt.xlabel('Sample')
            plt.ylabel('Var')
            plt.plot(self.__xs[0], variances)
            plt.savefig(path+'/var.png')
            plt.close()
            # Plot correlation
            fig_count += 1
            plt.figure(fig_count)
            plt.title('Cross-correlation maxtrix for ' + self.__g_name)
            plt.xlabel('Insertion')
            plt.ylabel('Insertion')
            plt.pcolor(crs, cmap='jet', vmin=-1, vmax=1)
            plt.colorbar()
            plt.yticks(np.arange(.5, crs.shape[1]+.5),list(range(crs.shape[1])))
            plt.xticks(np.arange(.5, crs.shape[0]+.5),list(range(crs.shape[0])))
            plt.savefig(path+'/cor.png')
            plt.close()

    # Pickling the object state
    # fname: full path for the pickle file
    def pickle(self, fname):
        pkl_f = open(fname, 'w')
        try:
            pickle.dump(self, pkl_f)
        finally:
            pkl_f.close()