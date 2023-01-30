"""
Classes for modelling the structures connected to membranes as filaments

# Author: Antonio Martinez-Sanchez (Max Planck Institute for Biochemistry)
# Date: 12.07.15
"""

__author__ = 'martinez'

import pyseg as ps
import copy
import sklearn
from pyseg.globals import *
from pyseg import pexceptions # as pexceptions
from scipy.ndimage.morphology import distance_transform_edt
from skimage.morphology import convex_hull_image
from pyseg.factory import SubGraphVisitor
from .variables import *
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from pyseg.filament import Filament

try:
    import pickle as pickle
except:
    import pickle

#### File global variables


###########################################################################################
# Class for modelling a contact point (Filament-Membrane)
###########################################################################################

class Contact(object):

    # graph: parent GraphMCF
    # fil: parent Filament
    # seg: tomogram with the membrane segmentation (membrane must be labeled as 1 (MB_LBL))
    # norm: normal vector (3d array) to membrane surface, default None because it used to be
    #       computed after Contact instantiation
    def __init__(self, graph, fil, seg, norm=None):
        self.__graph = graph
        self.__fil = fil
        self.__norm = norm
        self.__seg = seg
        self.__id = None
        self.__coords = None
        self.__clst_id = None
        self.__clst_id_eu = None
        self.__card = -1
        self.__find()

    #### Set/Get methods area

    def set_clst_id(self, clst_id):
        self.__clst_id = clst_id

    def set_clst_id_eu(self, clst_id):
        self.__clst_id_eu = clst_id

    def get_clst_id(self):
        return self.__clst_id

    def get_clst_id_eu(self):
        return self.__clst_id_eu

    def get_id(self):
        return self.__id

    def get_coords(self):
        return self.__coords

    def get_norm(self):
        return self.__norm

    def set_norm(self, norm):
        self.__norm = norm

    def set_card(self, card):
        self.__card = card

    def get_card(self):
        return self.__card

    #### Internal functionality area

    def __find(self):

        skel = self.__graph.get_skel()

        # Getting the edge
        edges = self.__fil.get_edges()
        vertices = self.__fil.get_vertices()
        e_id = edges[0].get_id()
        s = vertices[0]
        t = vertices[1]

        # Finding the arcs which contain the connector
        for a in s.get_arcs():
            if a.get_sad_id() == e_id:
                arc_s = a
                break
        for a in t.get_arcs():
            if a.get_sad_id() == e_id:
                arc_t = a
                break

        # Loops for finding the contact point
        found = False
        for i in range(1, arc_s.get_npoints()):
            curr_p = arc_s.get_point_id(i)
            x_p, y_p, z_p = skel.GetPoint(curr_p)
            x_p_r, y_p_r, z_p_r = int(round(x_p)), int(round(y_p)), int(round(z_p))
            if self.__seg[x_p_r, y_p_r, z_p_r] != MB_LBL:
                self.__coords = np.asarray((x_p, y_p, z_p))
                self.__id = curr_p
                found = True
                break
        if not found:
            ids = arc_t.get_ids()[::-1]
            for i in range(0, len(ids)):
                curr_p = ids[i]
                x_p, y_p, z_p = skel.GetPoint(curr_p)
                x_p_r, y_p_r, z_p_r = int(round(x_p)), int(round(y_p)), int(round(z_p))
                if self.__seg[x_p_r, y_p_r, z_p_r] != MB_LBL:
                    self.__coords = np.asarray((x_p, y_p, z_p))
                    self.__id = curr_p
                    found = True
                    break
        if not found:
            print('\tWarning: Unexpected event creating a Contact')
            curr_p = ids[-1]
            self.__coords = np.asarray(skel.GetPoint(curr_p))
            self.__id = curr_p
            # error_msg = 'Unexpected event.'
            # raise pexceptions.PySegInputWarning(expr='__find (Contact)', msg=error_msg)

##########################################################################################
# Class for modelling the filaments contained by a membrane
##########################################################################################

class MbFilaments(object):

    # graph: parent GraphMCF
    # seg: tomogram with the membrane segmentation (membrane must be labeled as 1 (MB_LBL),
    # and the allowed regions by 2 (MB_IN_LBL) and 3 (MB_OUT_LBL), respectively inside and outside)
    # nrad: neighbourhood radius for computing contact normals (in nm)
    # min_length: minimum length for the filaments (in nm)
    # max_length: maximum length for the filaments (in nm)
    def __init__(self, graph, seg, nrad=5, min_length=0., max_length=MAX_FLOAT):
        self.__graph_in = graph
        self.__graph_out = None
        self.__graph_gt_in = None
        self.__graph_gt_out = None
        self.__seg = seg
        self.__mb_dst = None
        self.__fils_in = list()
        self.__fils_out = list()
        self.__cont_in = list()
        self.__cont_out = list()
        self.__graph_fn_in = None
        self.__graph_fn_out = None
        self.__build(min_length, max_length, nrad)

    #### Set/Get methods area

    def get_graph_in(self):
        return self.__graph_in

    def get_graph_out(self):
        return self.__graph_out

    def get_density(self):
        return self.__graph_in.get_density()

    def get_resolution(self):
        return self.__graph_in.get_resolution()

    # Store contact points
    # force_update: if True (default) force to update all properties
    def get_cont_vtp(self, force_update=True):

        # Initialization
        point_id = 0
        cell_id = 0
        skel_in = self.__graph_in.get_skel()
        skel_out = self.__graph_out.get_skel()
        points = vtk.vtkPoints()
        verts = vtk.vtkCellArray()
        cell_data = vtk.vtkIntArray()
        cell_data.SetNumberOfComponents(1)
        cell_data.SetName(STR_CELL)
        str_data = vtk.vtkIntArray()
        str_data.SetNumberOfComponents(1)
        str_data.SetName(STR_STR)
        side_data = vtk.vtkIntArray()
        side_data.SetNumberOfComponents(1)
        side_data.SetName(STR_SIDE)
        len_data = vtk.vtkFloatArray()
        len_data.SetNumberOfComponents(1)
        len_data.SetName(STR_LEN)
        pen_data = vtk.vtkFloatArray()
        pen_data.SetNumberOfComponents(1)
        pen_data.SetName(STR_PEN)
        pl_data = vtk.vtkFloatArray()
        pl_data.SetNumberOfComponents(1)
        pl_data.SetName(STR_PEN_LEN)
        dst_data = vtk.vtkFloatArray()
        pet_data = vtk.vtkFloatArray()
        pet_data.SetNumberOfComponents(1)
        pet_data.SetName(STR_PENT)
        dst_data.SetNumberOfComponents(1)
        dst_data.SetName(STR_DST)
        fness_data = vtk.vtkFloatArray()
        fness_data.SetNumberOfComponents(1)
        fness_data.SetName(STR_FNESS)
        sness_data = vtk.vtkFloatArray()
        sness_data.SetNumberOfComponents(1)
        sness_data.SetName(STR_SNESS)
        mness_data = vtk.vtkFloatArray()
        mness_data.SetNumberOfComponents(1)
        mness_data.SetName(STR_MNESS)
        dness_data = vtk.vtkFloatArray()
        dness_data.SetNumberOfComponents(1)
        dness_data.SetName(STR_DNESS)
        clst_data = vtk.vtkIntArray()
        clst_data.SetNumberOfComponents(1)
        clst_data.SetName(STR_CLST)
        clste_data = vtk.vtkIntArray()
        clste_data.SetNumberOfComponents(1)
        clste_data.SetName(STR_CLST_EU)
        ct_data = vtk.vtkFloatArray()
        ct_data.SetNumberOfComponents(1)
        ct_data.SetName(STR_CT)
        mc_data = vtk.vtkFloatArray()
        mc_data.SetNumberOfComponents(1)
        mc_data.SetName(STR_MC)
        sim_data = vtk.vtkFloatArray()
        sim_data.SetNumberOfComponents(1)
        sim_data.SetName(STR_SIN)
        smo_data = vtk.vtkFloatArray()
        smo_data.SetNumberOfComponents(1)
        smo_data.SetName(STR_SMO)
        alpha_data = vtk.vtkFloatArray()
        alpha_data.SetNumberOfComponents(1)
        alpha_data.SetName(STR_ALPHA)
        beta_data = vtk.vtkFloatArray()
        beta_data.SetNumberOfComponents(1)
        beta_data.SetName(STR_BETA)
        nor_data = vtk.vtkFloatArray()
        nor_data.SetNumberOfComponents(3)
        nor_data.SetName(STR_NORM)
        car_data = vtk.vtkIntArray()
        car_data.SetNumberOfComponents(1)
        car_data.SetName(STR_CARD)

        # Update properties
        if force_update:
            self.compute_cont_card()

        # Writing the contact points
        for i, cont in enumerate(self.__cont_in):
            cont = self.__cont_in[i]
            cont_id = cont.get_id()
            if cont.get_clst_id() is not None:
                clst_id = cont.get_clst_id()
            else:
                clst_id = -1
            if cont.get_clst_id_eu() is not None:
                clste_id = cont.get_clst_id_eu()
            else:
                clste_id = -1
            f = self.__fils_in[i]
            x, y, z = skel_in.GetPoint(cont_id)
            points.InsertPoint(point_id, x, y, z)
            verts.InsertNextCell(1)
            verts.InsertCellPoint(point_id)
            cell_data.InsertNextTuple((cell_id,))
            str_data.InsertNextTuple((STR_C,))
            side_data.InsertNextTuple((MB_IN_LBL,))
            len_data.InsertNextTuple((f.get_length(cont_id),))
            dst_data.InsertNextTuple((f.get_head_tail_dist(cont_id),))
            pen_data.InsertNextTuple((f.get_penetration(),))
            pl_data.InsertNextTuple((f.get_pen_len_ratio(),))
            pet_data.InsertNextTuple((f.get_pen_tail(),))
            fness_data.InsertNextTuple((f.get_filamentness(),))
            sness_data.InsertNextTuple((f.get_sness(),))
            mness_data.InsertNextTuple((f.get_mness(),))
            dness_data.InsertNextTuple((f.get_dness(cont_id),))
            clst_data.InsertNextTuple((clst_id,))
            clste_data.InsertNextTuple((clste_id,))
            ct_data.InsertNextTuple((f.get_total_curvature(cont_id),))
            mc_data.InsertNextTuple((f.get_max_curvature(cont_id),))
            sim_data.InsertNextTuple((f.get_sinuosity(cont_id),))
            smo_data.InsertNextTuple((f.get_smoothness(cont_id),))
            alpha_data.InsertNextTuple((self.__alpha(f, cont),))
            beta_data.InsertNextTuple((self.__beta(f, cont),))
            nor_data.InsertNextTuple(tuple(cont.get_norm()))
            car_data.InsertNextTuple((cont.get_card(),))
            point_id += 1
            cell_id += 1
        for i, cont in enumerate(self.__cont_out):
            cont = self.__cont_out[i]
            cont_id = cont.get_id()
            if cont.get_clst_id() is not None:
                clst_id = cont.get_clst_id()
            else:
                clst_id = -1
            if cont.get_clst_id_eu() is not None:
                clste_id = cont.get_clst_id_eu()
            else:
                clste_id = -1
            f = self.__fils_out[i]
            x, y, z = skel_out.GetPoint(cont_id)
            points.InsertPoint(point_id, x, y, z)
            verts.InsertNextCell(1)
            verts.InsertCellPoint(point_id)
            cell_data.InsertNextTuple((cell_id,))
            str_data.InsertNextTuple((STR_C,))
            side_data.InsertNextTuple((MB_OUT_LBL,))
            len_data.InsertNextTuple((f.get_length(cont_id),))
            dst_data.InsertNextTuple((f.get_head_tail_dist(cont_id),))
            pen_data.InsertNextTuple((f.get_penetration(),))
            pl_data.InsertNextTuple((f.get_pen_len_ratio(),))
            pet_data.InsertNextTuple((f.get_pen_tail(),))
            fness_data.InsertNextTuple((f.get_filamentness(),))
            sness_data.InsertNextTuple((f.get_sness(),))
            mness_data.InsertNextTuple((f.get_mness(),))
            dness_data.InsertNextTuple((f.get_dness(cont_id),))
            clst_data.InsertNextTuple((clst_id,))
            clste_data.InsertNextTuple((clste_id,))
            ct_data.InsertNextTuple((f.get_total_curvature(cont_id),))
            mc_data.InsertNextTuple((f.get_max_curvature(cont_id),))
            sim_data.InsertNextTuple((f.get_sinuosity(cont_id),))
            smo_data.InsertNextTuple((f.get_smoothness(cont_id),))
            alpha_data.InsertNextTuple((self.__alpha(f, cont),))
            beta_data.InsertNextTuple((self.__beta(f, cont),))
            nor_data.InsertNextTuple(tuple(cont.get_norm()))
            car_data.InsertNextTuple((cont.get_card(),))
            point_id += 1
            cell_id += 1


        # Poly building
        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        poly.SetVerts(verts)
        poly.GetCellData().AddArray(cell_data)
        poly.GetCellData().AddArray(str_data)
        poly.GetCellData().AddArray(side_data)
        poly.GetCellData().AddArray(len_data)
        poly.GetCellData().AddArray(pen_data)
        poly.GetCellData().AddArray(pl_data)
        poly.GetCellData().AddArray(pet_data)
        poly.GetCellData().AddArray(dst_data)
        poly.GetCellData().AddArray(fness_data)
        poly.GetCellData().AddArray(sness_data)
        poly.GetCellData().AddArray(mness_data)
        poly.GetCellData().AddArray(dness_data)
        poly.GetCellData().AddArray(clst_data)
        poly.GetCellData().AddArray(clste_data)
        poly.GetCellData().AddArray(ct_data)
        poly.GetCellData().AddArray(mc_data)
        poly.GetCellData().AddArray(sim_data)
        poly.GetCellData().AddArray(smo_data)
        poly.GetCellData().AddArray(alpha_data)
        poly.GetCellData().AddArray(beta_data)
        poly.GetCellData().AddArray(nor_data)
        poly.GetCellData().AddArray(car_data)

        return poly

    # Storing filaments scheme
    # force_update: if True (default) force to update all properties
    def get_sch_vtp(self, force_update=True):

        # Initialization
        point_id = 0
        cell_id = 0
        skel_in = self.__graph_in.get_skel()
        skel_out = self.__graph_out.get_skel()
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        cell_data = vtk.vtkIntArray()
        cell_data.SetNumberOfComponents(1)
        cell_data.SetName(STR_CELL)
        str_data = vtk.vtkIntArray()
        str_data.SetNumberOfComponents(1)
        str_data.SetName(STR_STR)
        side_data = vtk.vtkIntArray()
        side_data.SetNumberOfComponents(1)
        side_data.SetName(STR_SIDE)
        len_data = vtk.vtkFloatArray()
        len_data.SetNumberOfComponents(1)
        len_data.SetName(STR_LEN)
        pen_data = vtk.vtkFloatArray()
        pen_data.SetNumberOfComponents(1)
        pen_data.SetName(STR_PEN)
        pl_data = vtk.vtkFloatArray()
        pl_data.SetNumberOfComponents(1)
        pl_data.SetName(STR_PEN_LEN)
        dst_data = vtk.vtkFloatArray()
        pet_data = vtk.vtkFloatArray()
        pet_data.SetNumberOfComponents(1)
        pet_data.SetName(STR_PENT)
        dst_data.SetNumberOfComponents(1)
        dst_data.SetName(STR_DST)
        fness_data = vtk.vtkFloatArray()
        fness_data.SetNumberOfComponents(1)
        fness_data.SetName(STR_FNESS)
        sness_data = vtk.vtkFloatArray()
        sness_data.SetNumberOfComponents(1)
        sness_data.SetName(STR_SNESS)
        mness_data = vtk.vtkFloatArray()
        mness_data.SetNumberOfComponents(1)
        mness_data.SetName(STR_MNESS)
        dness_data = vtk.vtkFloatArray()
        dness_data.SetNumberOfComponents(1)
        dness_data.SetName(STR_DNESS)
        clst_data = vtk.vtkIntArray()
        clst_data.SetNumberOfComponents(1)
        clst_data.SetName(STR_CLST)
        clste_data = vtk.vtkIntArray()
        clste_data.SetNumberOfComponents(1)
        clste_data.SetName(STR_CLST_EU)
        ct_data = vtk.vtkFloatArray()
        ct_data.SetNumberOfComponents(1)
        ct_data.SetName(STR_CT)
        mc_data = vtk.vtkFloatArray()
        mc_data.SetNumberOfComponents(1)
        mc_data.SetName(STR_MC)
        sim_data = vtk.vtkFloatArray()
        sim_data.SetNumberOfComponents(1)
        sim_data.SetName(STR_SIN)
        smo_data = vtk.vtkFloatArray()
        smo_data.SetNumberOfComponents(1)
        smo_data.SetName(STR_SMO)
        alpha_data = vtk.vtkFloatArray()
        alpha_data.SetNumberOfComponents(1)
        alpha_data.SetName(STR_ALPHA)
        beta_data = vtk.vtkFloatArray()
        beta_data.SetNumberOfComponents(1)
        beta_data.SetName(STR_BETA)
        nor_data = vtk.vtkFloatArray()
        nor_data.SetNumberOfComponents(3)
        nor_data.SetName(STR_NORM)
        car_data = vtk.vtkIntArray()
        car_data.SetNumberOfComponents(1)
        car_data.SetName(STR_CARD)

        # Update properties
        if force_update:
            self.compute_cont_card()

        # Write verts
        for i, f in enumerate(self.__fils_in):
            cont = self.__cont_in[i]
            cont_id = cont.get_id()
            if cont.get_clst_id() is not None:
                clst_id = cont.get_clst_id()
            else:
                clst_id = -1
            if cont.get_clst_id_eu() is not None:
                clste_id = cont.get_clst_id_eu()
            else:
                clste_id = -1
            tail = f.get_tail()
            x_c, y_c, z_c = skel_in.GetPoint(cont_id)
            x_t, y_t, z_t = skel_in.GetPoint(tail.get_id())
            lines.InsertNextCell(2)
            points.InsertPoint(point_id, x_c, y_c, z_c)
            lines.InsertCellPoint(point_id)
            point_id += 1
            points.InsertPoint(point_id, x_t, y_t, z_t)
            lines.InsertCellPoint(point_id)
            point_id += 1
            cell_data.InsertNextTuple((cell_id,))
            str_data.InsertNextTuple((STR_T,))
            side_data.InsertNextTuple((MB_IN_LBL,))
            len_data.InsertNextTuple((f.get_length(cont_id),))
            dst_data.InsertNextTuple((f.get_head_tail_dist(cont_id),))
            pen_data.InsertNextTuple((f.get_penetration(),))
            pl_data.InsertNextTuple((f.get_pen_len_ratio(),))
            pet_data.InsertNextTuple((f.get_pen_tail(),))
            fness_data.InsertNextTuple((f.get_filamentness(),))
            sness_data.InsertNextTuple((f.get_sness(),))
            mness_data.InsertNextTuple((f.get_mness(),))
            dness_data.InsertNextTuple((f.get_dness(cont_id),))
            clst_data.InsertNextTuple((clst_id,))
            clste_data.InsertNextTuple((clste_id,))
            ct_data.InsertNextTuple((f.get_total_curvature(cont_id),))
            mc_data.InsertNextTuple((f.get_max_curvature(cont_id),))
            sim_data.InsertNextTuple((f.get_sinuosity(cont_id),))
            smo_data.InsertNextTuple((f.get_smoothness(cont_id),))
            alpha_data.InsertNextTuple((self.__alpha(f, cont),))
            beta_data.InsertNextTuple((self.__beta(f, cont),))
            nor_data.InsertNextTuple(tuple(cont.get_norm()))
            car_data.InsertNextTuple((cont.get_card(),))
            cell_id += 1

        for i, f in enumerate(self.__fils_out):
            cont = self.__cont_out[i]
            cont_id = cont.get_id()
            if cont.get_clst_id() is not None:
                clst_id = cont.get_clst_id()
            else:
                clst_id = -1
            if cont.get_clst_id_eu() is not None:
                clste_id = cont.get_clst_id_eu()
            else:
                clste_id = -1
            tail = f.get_tail()
            x_c, y_c, z_c = skel_out.GetPoint(cont_id)
            x_t, y_t, z_t = skel_out.GetPoint(tail.get_id())
            lines.InsertNextCell(2)
            points.InsertPoint(point_id, x_c, y_c, z_c)
            lines.InsertCellPoint(point_id)
            point_id += 1
            points.InsertPoint(point_id, x_t, y_t, z_t)
            lines.InsertCellPoint(point_id)
            point_id += 1
            cell_data.InsertNextTuple((cell_id,))
            str_data.InsertNextTuple((STR_T,))
            side_data.InsertNextTuple((MB_OUT_LBL,))
            len_data.InsertNextTuple((f.get_length(cont_id),))
            dst_data.InsertNextTuple((f.get_head_tail_dist(cont_id),))
            pen_data.InsertNextTuple((f.get_penetration(),))
            pl_data.InsertNextTuple((f.get_pen_len_ratio(),))
            pet_data.InsertNextTuple((f.get_pen_tail(),))
            fness_data.InsertNextTuple((f.get_filamentness(),))
            sness_data.InsertNextTuple((f.get_sness(),))
            mness_data.InsertNextTuple((f.get_mness(),))
            dness_data.InsertNextTuple((f.get_dness(cont_id),))
            clst_data.InsertNextTuple((clst_id,))
            clste_data.InsertNextTuple((clste_id,))
            ct_data.InsertNextTuple((f.get_total_curvature(cont_id),))
            mc_data.InsertNextTuple((f.get_max_curvature(cont_id),))
            sim_data.InsertNextTuple((f.get_sinuosity(cont_id),))
            smo_data.InsertNextTuple((f.get_smoothness(cont_id),))
            alpha_data.InsertNextTuple((self.__alpha(f, cont),))
            beta_data.InsertNextTuple((self.__beta(f, cont),))
            nor_data.InsertNextTuple(tuple(cont.get_norm()))
            car_data.InsertNextTuple((cont.get_card(),))
            cell_id += 1

        # Poly building
        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        poly.SetLines(lines)
        poly.GetCellData().AddArray(cell_data)
        poly.GetCellData().AddArray(str_data)
        poly.GetCellData().AddArray(side_data)
        poly.GetCellData().AddArray(len_data)
        poly.GetCellData().AddArray(pen_data)
        poly.GetCellData().AddArray(pl_data)
        poly.GetCellData().AddArray(pet_data)
        poly.GetCellData().AddArray(dst_data)
        poly.GetCellData().AddArray(fness_data)
        poly.GetCellData().AddArray(sness_data)
        poly.GetCellData().AddArray(mness_data)
        poly.GetCellData().AddArray(dness_data)
        poly.GetCellData().AddArray(clst_data)
        poly.GetCellData().AddArray(clste_data)
        poly.GetCellData().AddArray(ct_data)
        poly.GetCellData().AddArray(mc_data)
        poly.GetCellData().AddArray(sim_data)
        poly.GetCellData().AddArray(smo_data)
        poly.GetCellData().AddArray(alpha_data)
        poly.GetCellData().AddArray(beta_data)
        poly.GetCellData().AddArray(nor_data)
        poly.GetCellData().AddArray(car_data)

        return poly

    # Storing filaments path
    # force_update: if True (default) force to update all properties
    def get_path_vtp(self, force_update=True):

        # Initialization
        point_id = 0
        cell_id = 0
        skel_in = self.__graph_in.get_skel()
        skel_out = self.__graph_out.get_skel()
        points = vtk.vtkPoints()
        verts = vtk.vtkCellArray()
        lines = vtk.vtkCellArray()
        cell_data = vtk.vtkIntArray()
        cell_data.SetNumberOfComponents(1)
        cell_data.SetName(STR_CELL)
        str_data = vtk.vtkIntArray()
        str_data.SetNumberOfComponents(1)
        str_data.SetName(STR_STR)
        side_data = vtk.vtkIntArray()
        side_data.SetNumberOfComponents(1)
        side_data.SetName(STR_SIDE)
        len_data = vtk.vtkFloatArray()
        len_data.SetNumberOfComponents(1)
        len_data.SetName(STR_LEN)
        pen_data = vtk.vtkFloatArray()
        pen_data.SetNumberOfComponents(1)
        pen_data.SetName(STR_PEN)
        pl_data = vtk.vtkFloatArray()
        pl_data.SetNumberOfComponents(1)
        pl_data.SetName(STR_PEN_LEN)
        dst_data = vtk.vtkFloatArray()
        pet_data = vtk.vtkFloatArray()
        pet_data.SetNumberOfComponents(1)
        pet_data.SetName(STR_PENT)
        dst_data.SetNumberOfComponents(1)
        dst_data.SetName(STR_DST)
        fness_data = vtk.vtkFloatArray()
        fness_data.SetNumberOfComponents(1)
        fness_data.SetName(STR_FNESS)
        sness_data = vtk.vtkFloatArray()
        sness_data.SetNumberOfComponents(1)
        sness_data.SetName(STR_SNESS)
        mness_data = vtk.vtkFloatArray()
        mness_data.SetNumberOfComponents(1)
        mness_data.SetName(STR_MNESS)
        dness_data = vtk.vtkFloatArray()
        dness_data.SetNumberOfComponents(1)
        dness_data.SetName(STR_DNESS)
        clst_data = vtk.vtkIntArray()
        clst_data.SetNumberOfComponents(1)
        clst_data.SetName(STR_CLST)
        clste_data = vtk.vtkIntArray()
        clste_data.SetNumberOfComponents(1)
        clste_data.SetName(STR_CLST_EU)
        ct_data = vtk.vtkFloatArray()
        ct_data.SetNumberOfComponents(1)
        ct_data.SetName(STR_CT)
        mc_data = vtk.vtkFloatArray()
        mc_data.SetNumberOfComponents(1)
        mc_data.SetName(STR_MC)
        sim_data = vtk.vtkFloatArray()
        sim_data.SetNumberOfComponents(1)
        sim_data.SetName(STR_SIN)
        smo_data = vtk.vtkFloatArray()
        smo_data.SetNumberOfComponents(1)
        smo_data.SetName(STR_SMO)
        alpha_data = vtk.vtkFloatArray()
        alpha_data.SetNumberOfComponents(1)
        alpha_data.SetName(STR_ALPHA)
        beta_data = vtk.vtkFloatArray()
        beta_data.SetNumberOfComponents(1)
        beta_data.SetName(STR_BETA)
        nor_data = vtk.vtkFloatArray()
        nor_data.SetNumberOfComponents(3)
        nor_data.SetName(STR_NORM)
        car_data = vtk.vtkIntArray()
        car_data.SetNumberOfComponents(1)
        car_data.SetName(STR_CARD)

        # Update properties
        if force_update:
            self.compute_cont_card()


        # Write verts
        for i, f in enumerate(self.__fils_in):
            cont = self.__cont_in[i]
            cont_id = cont.get_id()
            if cont.get_clst_id() is not None:
                clst_id = cont.get_clst_id()
            else:
                clst_id = -1
            if cont.get_clst_id_eu() is not None:
                clste_id = cont.get_clst_id_eu()
            else:
                clste_id = -1
            # Head
            head = f.get_head()
            x, y, z = skel_in.GetPoint(head.get_id())
            points.InsertPoint(point_id, x, y, z)
            verts.InsertNextCell(1)
            verts.InsertCellPoint(point_id)
            cell_data.InsertNextTuple((cell_id,))
            str_data.InsertNextTuple((STR_H,))
            side_data.InsertNextTuple((MB_IN_LBL,))
            len_data.InsertNextTuple((f.get_length(cont_id),))
            dst_data.InsertNextTuple((f.get_head_tail_dist(cont_id),))
            pen_data.InsertNextTuple((f.get_penetration(),))
            pl_data.InsertNextTuple((f.get_pen_len_ratio(),))
            pet_data.InsertNextTuple((f.get_pen_tail(),))
            fness_data.InsertNextTuple((f.get_filamentness(),))
            sness_data.InsertNextTuple((f.get_sness(),))
            mness_data.InsertNextTuple((f.get_mness(),))
            dness_data.InsertNextTuple((f.get_dness(cont_id),))
            clst_data.InsertNextTuple((clst_id,))
            clste_data.InsertNextTuple((clste_id,))
            ct_data.InsertNextTuple((f.get_total_curvature(cont_id),))
            mc_data.InsertNextTuple((f.get_max_curvature(cont_id),))
            sim_data.InsertNextTuple((f.get_sinuosity(cont_id),))
            smo_data.InsertNextTuple((f.get_smoothness(cont_id),))
            alpha_data.InsertNextTuple((self.__alpha(f, cont),))
            beta_data.InsertNextTuple((self.__beta(f, cont),))
            nor_data.InsertNextTuple(tuple(cont.get_norm()))
            car_data.InsertNextTuple((cont.get_card(),))
            point_id += 1
            cell_id += 1
            # Tail
            tail = f.get_tail()
            x, y, z = skel_in.GetPoint(tail.get_id())
            points.InsertPoint(point_id, x, y, z)
            verts.InsertNextCell(1)
            verts.InsertCellPoint(point_id)
            cell_data.InsertNextTuple((cell_id,))
            str_data.InsertNextTuple((STR_T,))
            side_data.InsertNextTuple((MB_IN_LBL,))
            len_data.InsertNextTuple((f.get_length(cont_id),))
            dst_data.InsertNextTuple((f.get_head_tail_dist(cont_id),))
            pen_data.InsertNextTuple((f.get_penetration(),))
            pl_data.InsertNextTuple((f.get_pen_len_ratio(),))
            pet_data.InsertNextTuple((f.get_pen_tail(),))
            fness_data.InsertNextTuple((f.get_filamentness(),))
            sness_data.InsertNextTuple((f.get_sness(),))
            mness_data.InsertNextTuple((f.get_mness(),))
            dness_data.InsertNextTuple((f.get_dness(cont_id),))
            clst_data.InsertNextTuple((clst_id,))
            clste_data.InsertNextTuple((clste_id,))
            ct_data.InsertNextTuple((f.get_total_curvature(cont_id),))
            mc_data.InsertNextTuple((f.get_max_curvature(cont_id),))
            sim_data.InsertNextTuple((f.get_sinuosity(cont_id),))
            smo_data.InsertNextTuple((f.get_smoothness(cont_id),))
            alpha_data.InsertNextTuple((self.__alpha(f, cont),))
            beta_data.InsertNextTuple((self.__beta(f, cont),))
            nor_data.InsertNextTuple(tuple(cont.get_norm()))
            car_data.InsertNextTuple((cont.get_card(),))
            point_id += 1
            cell_id += 1

        for i, f in enumerate(self.__fils_out):
            cont = self.__cont_out[i]
            cont_id = cont.get_id()
            if cont.get_clst_id() is not None:
                clst_id = cont.get_clst_id()
            else:
                clst_id = -1
            if cont.get_clst_id_eu() is not None:
                clste_id = cont.get_clst_id_eu()
            else:
                clste_id = -1
            # Head
            head = f.get_head()
            x, y, z = skel_out.GetPoint(head.get_id())
            points.InsertPoint(point_id, x, y, z)
            verts.InsertNextCell(1)
            verts.InsertCellPoint(point_id)
            cell_data.InsertNextTuple((cell_id,))
            str_data.InsertNextTuple((STR_H,))
            side_data.InsertNextTuple((MB_OUT_LBL,))
            len_data.InsertNextTuple((f.get_length(cont_id),))
            dst_data.InsertNextTuple((f.get_head_tail_dist(cont_id),))
            pen_data.InsertNextTuple((f.get_penetration(),))
            pl_data.InsertNextTuple((f.get_pen_len_ratio(),))
            pet_data.InsertNextTuple((f.get_pen_tail(),))
            fness_data.InsertNextTuple((f.get_filamentness(),))
            sness_data.InsertNextTuple((f.get_sness(),))
            mness_data.InsertNextTuple((f.get_mness(),))
            dness_data.InsertNextTuple((f.get_dness(cont_id),))
            clst_data.InsertNextTuple((clst_id,))
            clste_data.InsertNextTuple((clste_id,))
            ct_data.InsertNextTuple((f.get_total_curvature(cont_id),))
            mc_data.InsertNextTuple((f.get_max_curvature(cont_id),))
            sim_data.InsertNextTuple((f.get_sinuosity(cont_id),))
            smo_data.InsertNextTuple((f.get_smoothness(cont_id),))
            alpha_data.InsertNextTuple((self.__alpha(f, cont),))
            beta_data.InsertNextTuple((self.__beta(f, cont),))
            nor_data.InsertNextTuple(tuple(cont.get_norm()))
            car_data.InsertNextTuple((cont.get_card(),))
            point_id += 1
            cell_id += 1
            # Tail
            tail = f.get_tail()
            x, y, z = skel_out.GetPoint(tail.get_id())
            points.InsertPoint(point_id, x, y, z)
            verts.InsertNextCell(1)
            verts.InsertCellPoint(point_id)
            cell_data.InsertNextTuple((cell_id,))
            str_data.InsertNextTuple((STR_T,))
            side_data.InsertNextTuple((MB_OUT_LBL,))
            len_data.InsertNextTuple((f.get_length(cont_id),))
            dst_data.InsertNextTuple((f.get_head_tail_dist(cont_id),))
            pen_data.InsertNextTuple((f.get_penetration(),))
            pl_data.InsertNextTuple((f.get_pen_len_ratio(),))
            pet_data.InsertNextTuple((f.get_pen_tail(),))
            fness_data.InsertNextTuple((f.get_filamentness(),))
            sness_data.InsertNextTuple((f.get_sness(),))
            mness_data.InsertNextTuple((f.get_mness(),))
            dness_data.InsertNextTuple((f.get_dness(cont_id),))
            clst_data.InsertNextTuple((clst_id,))
            clste_data.InsertNextTuple((clste_id,))
            ct_data.InsertNextTuple((f.get_total_curvature(cont_id),))
            mc_data.InsertNextTuple((f.get_max_curvature(cont_id),))
            sim_data.InsertNextTuple((f.get_sinuosity(cont_id),))
            smo_data.InsertNextTuple((f.get_smoothness(cont_id),))
            alpha_data.InsertNextTuple((self.__alpha(f, cont),))
            beta_data.InsertNextTuple((self.__beta(f, cont),))
            nor_data.InsertNextTuple(tuple(cont.get_norm()))
            car_data.InsertNextTuple((cont.get_card(),))
            point_id += 1
            cell_id += 1

        # Write lines
        for i, f in enumerate(self.__fils_in):
            cont = self.__cont_in[i]
            cont_id = cont.get_id()
            if cont.get_clst_id() is not None:
                clst_id = cont.get_clst_id()
            else:
                clst_id = -1
            if cont.get_clst_id_eu() is not None:
                clste_id = cont.get_clst_id_eu()
            else:
                clste_id = -1
            coords = f.get_path_coords()
            lines.InsertNextCell(coords.shape[0])
            for c in coords:
                points.InsertPoint(point_id, c[0], c[1], c[2])
                lines.InsertCellPoint(point_id)
                point_id += 1
            cell_data.InsertNextTuple((cell_id,))
            str_data.InsertNextTuple((STR_F,))
            side_data.InsertNextTuple((MB_IN_LBL,))
            len_data.InsertNextTuple((f.get_length(cont_id),))
            dst_data.InsertNextTuple((f.get_head_tail_dist(cont_id),))
            pen_data.InsertNextTuple((f.get_penetration(),))
            pl_data.InsertNextTuple((f.get_pen_len_ratio(),))
            pet_data.InsertNextTuple((f.get_pen_tail(),))
            fness_data.InsertNextTuple((f.get_filamentness(),))
            sness_data.InsertNextTuple((f.get_sness(),))
            mness_data.InsertNextTuple((f.get_mness(),))
            dness_data.InsertNextTuple((f.get_dness(cont_id),))
            clst_data.InsertNextTuple((clst_id,))
            clste_data.InsertNextTuple((clste_id,))
            ct_data.InsertNextTuple((f.get_total_curvature(cont_id),))
            mc_data.InsertNextTuple((f.get_max_curvature(cont_id),))
            sim_data.InsertNextTuple((f.get_sinuosity(cont_id),))
            smo_data.InsertNextTuple((f.get_smoothness(cont_id),))
            alpha_data.InsertNextTuple((self.__alpha(f, cont),))
            beta_data.InsertNextTuple((self.__beta(f, cont),))
            nor_data.InsertNextTuple(tuple(cont.get_norm()))
            car_data.InsertNextTuple((cont.get_card(),))
            cell_id += 1
        for i, f in enumerate(self.__fils_out):
            cont = self.__cont_out[i]
            cont_id = cont.get_id()
            if cont.get_clst_id() is not None:
                clst_id = cont.get_clst_id()
            else:
                clst_id = -1
            if cont.get_clst_id_eu() is not None:
                clste_id = cont.get_clst_id_eu()
            else:
                clste_id = -1
            coords = f.get_path_coords()
            lines.InsertNextCell(coords.shape[0])
            for c in coords:
                points.InsertPoint(point_id, c[0], c[1], c[2])
                lines.InsertCellPoint(point_id)
                point_id += 1
            cell_data.InsertNextTuple((cell_id,))
            str_data.InsertNextTuple((STR_F,))
            side_data.InsertNextTuple((MB_OUT_LBL,))
            len_data.InsertNextTuple((f.get_length(cont_id),))
            dst_data.InsertNextTuple((f.get_head_tail_dist(cont_id),))
            pen_data.InsertNextTuple((f.get_penetration(),))
            pl_data.InsertNextTuple((f.get_pen_len_ratio(),))
            pet_data.InsertNextTuple((f.get_pen_tail(),))
            fness_data.InsertNextTuple((f.get_filamentness(),))
            sness_data.InsertNextTuple((f.get_sness(),))
            mness_data.InsertNextTuple((f.get_mness(),))
            dness_data.InsertNextTuple((f.get_dness(cont_id),))
            clst_data.InsertNextTuple((clst_id,))
            clste_data.InsertNextTuple((clste_id,))
            ct_data.InsertNextTuple((f.get_total_curvature(cont_id),))
            mc_data.InsertNextTuple((f.get_max_curvature(cont_id),))
            sim_data.InsertNextTuple((f.get_sinuosity(cont_id),))
            smo_data.InsertNextTuple((f.get_smoothness(cont_id),))
            alpha_data.InsertNextTuple((self.__alpha(f, cont),))
            beta_data.InsertNextTuple((self.__beta(f, cont),))
            nor_data.InsertNextTuple(tuple(cont.get_norm()))
            car_data.InsertNextTuple((cont.get_card(),))
            cell_id += 1

        # Poly building
        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        poly.SetVerts(verts)
        poly.SetLines(lines)
        poly.GetCellData().AddArray(cell_data)
        poly.GetCellData().AddArray(str_data)
        poly.GetCellData().AddArray(side_data)
        poly.GetCellData().AddArray(len_data)
        poly.GetCellData().AddArray(pen_data)
        poly.GetCellData().AddArray(pl_data)
        poly.GetCellData().AddArray(pet_data)
        poly.GetCellData().AddArray(dst_data)
        poly.GetCellData().AddArray(fness_data)
        poly.GetCellData().AddArray(sness_data)
        poly.GetCellData().AddArray(mness_data)
        poly.GetCellData().AddArray(dness_data)
        poly.GetCellData().AddArray(clst_data)
        poly.GetCellData().AddArray(clste_data)
        poly.GetCellData().AddArray(ct_data)
        poly.GetCellData().AddArray(mc_data)
        poly.GetCellData().AddArray(sim_data)
        poly.GetCellData().AddArray(smo_data)
        poly.GetCellData().AddArray(alpha_data)
        poly.GetCellData().AddArray(beta_data)
        poly.GetCellData().AddArray(nor_data)
        poly.GetCellData().AddArray(car_data)

        return poly

    # side: if MB_IN_LBL inside structures are considered, if MB_OUT_LBL then outside,
    # otherwise (default) both
    def get_heads_id(self, side=None):

        heads = list()
        if (side is None) or (side is MB_IN_LBL):
            for f in self.__fils_in:
                heads.append(f.get_head().get_id())
        if (side is None) or (side is MB_OUT_LBL):
            for f in self.__fils_out:
                heads.append(f.get_head().get_id())

        return heads

    # side: if MB_IN_LBL inside structures are considered, if MB_OUT_LBL then outside,
    # otherwise (default) both
    def get_tails_id(self, side=None):

        tails = list()
        if (side is None) or (side is MB_IN_LBL):
            for f in self.__fils_in:
                tails.append(f.get_tail().get_id())
        if (side is None) or (side is MB_OUT_LBL):
            for f in self.__fils_out:
                tails.append(f.get_tail().get_id())

        return tails

    # side: if MB_IN_LBL inside structures are considered, if MB_OUT_LBL then outside,
    # otherwise (default) both
    def get_conts_id(self, side=None):

        conts = list()
        if (side is None) or (side is MB_IN_LBL):
            for c in self.__cont_in:
                conts.append(c.get_id())
        if (side is None) or (side is MB_OUT_LBL):
            for c in self.__cont_out:
                conts.append(c.get_id())

        return conts

    # side: if MB_IN_LBL inside structures are considered, if MB_OUT_LBL then outside,
    # otherwise (default) both
    def get_conts_clst_id(self, side=None):

        conts = list()
        if (side is None) or (side is MB_IN_LBL):
            for c in self.__cont_in:
                conts.append(c.get_clst_id())
        if (side is None) or (side is MB_OUT_LBL):
            for c in self.__cont_out:
                conts.append(c.get_clst_id())

        return conts

    # side: if MB_IN_LBL inside structures are considered, if MB_OUT_LBL then outside,
    # otherwise (default) both
    # struct: if STR_H heads, if STR_T tail, if STR_C contact points, otherwise (default) both
    # eps: if not None (default None) points that are closer than eps are considered as
    # repeated and they are deleted
    # card: if True (default False) the cardinality is also returned
    # Return: an array with points coordinates and their cardinality (optional)
    def get_cloud_points(self, side=None, struct=None, eps=None, card=False):

        # Getting points ids
        ids = list()
        if card:
            conts = list()
            if side == MB_IN_LBL:
                hold_conts = self.__cont_in
            else:
                hold_conts = self.__cont_out
        if struct == STR_H:
            ids += self.get_heads_id(side)
            if card:
                conts += hold_conts
        elif struct == STR_T:
            ids += self.get_tails_id(side)
            if card:
                conts += hold_conts
        elif struct == STR_C:
            ids += self.get_conts_id(side)
            if card:
                conts += hold_conts
        else:
            ids += self.get_heads_id(side)
            ids += self.get_tails_id(side)
            ids += self.get_conts_id(side)
            if card:
                conts = (hold_conts + hold_conts + hold_conts)

        # Getting coordinates
        skel = self.__graph_in.get_skel()
        hold_coords = np.zeros(shape=(len(ids), 3), dtype=float)
        if card:
            hold_cards = np.zeros(shape=len(ids), dtype=int)
            for i, idx in enumerate(ids):
                hold_coords[i, :] = skel.GetPoint(idx)
                hold_cards[i] = conts[i].get_card()
        else:
            for i, idx in enumerate(ids):
                hold_coords[i, :] = skel.GetPoint(idx)

        # Filter repeated coordinates
        if eps is not None:
            coords = list()
            if card:
                cards = list()
            del_lut = np.ones(shape=hold_coords.shape[0], dtype=bool)
            for i, coord in enumerate(hold_coords):
                if del_lut[i]:
                    coords.append(coord)
                    if card:
                        cards.append(hold_cards[i])
                    hold = coord - hold_coords
                    hold = np.sum(hold*hold, axis=1)
                    hold[i] = np.inf
                    dists = np.sqrt(hold)
                    eps_ids = np.where(dists < eps)[0]
                    for idx in eps_ids:
                        del_lut[idx] = False
            if card:
                return np.asarray(coords, dtype=float), np.asarray(cards, dtype=int)
            else:
                return np.asarray(coords, dtype=float)
        else:
            if card:
                return hold_conts, hold_cards
            else:
                return hold_coords

    # side: if MB_IN_LBL inside structures are considered, if MB_OUT_LBL then outside,
    # otherwise (default) both
    # struct: if STR_H heads, if STR_T tail, if STR_C contact points, otherwise (default) both
    # eps: if not None (default None) points that are closer than eps are considered as
    # repeated and they are deleted
    # Return: booth an array with points coordinates and cluster id
    def get_cloud_clst_id(self, side=None, struct=None, eps=None):

        # Getting points ids
        ids = list()
        clsts = list()
        if struct == STR_H:
            ids += self.get_heads_id(side)
            clsts += self.get_conts_clst_id(side)
        elif struct == STR_T:
            ids += self.get_tails_id(side)
            clsts += self.get_conts_clst_id(side)
        elif struct == STR_C:
            ids += self.get_conts_id(side)
            clsts += self.get_conts_clst_id(side)
        else:
            ids += self.get_heads_id(side)
            clsts += self.get_conts_clst_id(side)
            ids += self.get_tails_id(side)
            clsts += self.get_conts_clst_id(side)
            ids += self.get_conts_id(side)
            clsts += self.get_conts_clst_id(side)

        # Getting coordinates
        skel = self.__graph_in.get_skel()
        hold_clsts = np.zeros(shape=len(clsts), dtype=int)
        hold_coords = np.zeros(shape=(len(ids), 3), dtype=float)
        for i, idx in enumerate(ids):
            hold_coords[i, :] = skel.GetPoint(idx)
            hold_clsts[i] = clsts[i]

        # Filter repeated coordinates
        if eps is not None:
            coords = list()
            clsts_l = list()
            del_lut = np.ones(shape=hold_coords.shape[0], dtype=bool)
            for i, coord in enumerate(hold_coords):
                if del_lut[i]:
                    coords.append(coord)
                    clsts_l.append(clsts[i])
                    hold = coord - hold_coords
                    hold = np.sum(hold*hold, axis=1)
                    hold[i] = np.inf
                    dists = np.sqrt(hold)
                    eps_ids = np.where(dists < eps)[0]
                    for idx in eps_ids:
                        del_lut[idx] = False
            return np.asarray(coords, dtype=float), np.asarray(clsts_l, int)
        else:
            return hold_coords, hold_clsts

    # Clustering accroding connectivity within a slice
    # side: if MB_IN_LBL inside structures are considered, if MB_OUT_LBL then outside,
    # otherwise (default) both
    # slice_samp: range [low, high] in nm with the slice values (penetration tail)
    # Return: booth an array with points coordinates and cluster id
    def get_cloud_clst_slice(self, side, slice_samp):

        # Getting a copy of the graph graph
        if side == MB_IN_LBL:
            graph = self.__graph_in
            fils = self.__fils_in
        elif side == MB_OUT_LBL:
            graph = self.__graph_out
            fils = self.__fils_out
        else:
            error_msg = 'Non-valid side.'
            raise pexceptions.PySegInputError(expr='get_cloud_clst_slice (MbFilaments)',
                                              msg=error_msg)
        skel = graph.get_skel()

        # Find tail vertices within slice
        lut_ver = (-1) * np.ones(shape=graph.get_nid(), dtype=int)
        cont = 0
        v_ids = list()
        for f in fils:
            pent = f.get_pen_tail()
            if (pent >= slice_samp[0]) and (pent <= slice_samp[1]):
                v_id = f.get_tail().get_id()
                if lut_ver[v_id] == -1:
                    lut_ver[v_id] = cont
                    v_ids.append(v_id)
                    cont += 1

        # Find edges within slice
        e_ids = list()
        for edge in graph.get_edges_list():
            s_id = lut_ver[edge.get_source_id()]
            t_id = lut_ver[edge.get_target_id()]
            if (s_id > 0) and (t_id > 0):
                e_ids.append([s_id, t_id])

        # graph_tool building
        graph = gt.Graph(directed=False)
        vertices_gt = np.empty(shape=len(v_ids), dtype=object)
        for i in range(len(v_ids)):
            vertices_gt[i] = graph.add_vertex()
        for e_id in e_ids:
            graph.add_edge(vertices_gt[e_id[0]], vertices_gt[e_id[1]])

        # Subgraphs visitor initialization
        sgraph_id = graph.new_vertex_property("int")
        visitor = SubGraphVisitor(sgraph_id)

        # Find subgraphs
        coords = np.zeros(shape=(vertices_gt.shape[0], 3), dtype=float)
        for i, v in enumerate(vertices_gt):
            if sgraph_id[v] == 0:
                gt.dfs_search(graph, v, visitor)
                visitor.update_sgraphs_id()
            coords[i, :] = skel.GetPoint(v_ids[i])

        return coords, sgraph_id.get_array()

    # Cloud filament tail points coordinates in a slice
    # side: if MB_IN_LBL inside structures are considered, if MB_OUT_LBL then outside,
    # otherwise (default) both
    # slice_samp: range [low, high] in nm with the slice values (penetration tail)
    # card: if True (default False) tail cardianlity is returned
    # Return: booth an array with points coordinates and cluster id
    def get_cloud_points_slice(self, side, slice_samp, card=False):

        # Getting a copy of the graph graph
        if side == MB_IN_LBL:
            graph = self.__graph_in
            fils = self.__fils_in
        elif side == MB_OUT_LBL:
            graph = self.__graph_out
            fils = self.__fils_out
        else:
            error_msg = 'Non-valid side.'
            raise pexceptions.PySegInputError(expr='get_cloud_clst_slice (MbFilaments)',
                                              msg=error_msg)
        skel = graph.get_skel()

        # Find tail vertices within slice
        lut_ver = np.zeros(shape=graph.get_nid(), dtype=int)
        v_ids = list()
        for f in fils:
            pent = f.get_pen_tail()
            if (pent >= slice_samp[0]) and (pent <= slice_samp[1]):
                v_id = f.get_tail().get_id()
                if lut_ver[v_id] == 0:
                    v_ids.append(v_id)
                lut_ver[v_id] += 1

        # Finding coordinates
        coords = np.zeros(shape=(len(v_ids), 3), dtype=float)
        if card:
            cards = np.zeros(shape=len(v_ids), dtype=int)
            for i, v_id in enumerate(v_ids):
                coords[i, :] = skel.GetPoint(v_id)
                cards[i] = lut_ver[v_id]
            return coords, cards
        else:
            for i, v_id in enumerate(v_ids):
                coords[i, :] = skel.GetPoint(v_id)
            return coords

    #### External functionality area

    # Filament thresholding based on their head to tail dist
    # th: thresholding object
    # mode: if 'simple' default just filament a thresholded, in 'complete' firstly a simple
    #       thresholding is done and all filaments with share a vertex with an already deleted
    #       filament is also deleted
    def threshold_dst(self, th, mode='simple'):
        self.__threhold_fils(th, mode, func_name='get_head_tail_dist')

    # Filament thresholding based on tail pentration
    # th: thresholding object
    # mode: if 'simple' default just filament a thresholded, in 'complete' firstly a simple
    #       thresholding is done and all filaments with share a vertex with an already deleted
    #       filament is also deleted
    def threshold_pent(self, th, mode='simple'):
        self.__threhold_fils(th, mode, func_name='get_pen_tail')

    # Filament thresholding based on penetration
    # th: thresholding object
    # mode: if 'simple' default just filament a thresholded, in 'complete' firstly a simple
    #       thresholding is done and all filaments with share a vertex with an already deleted
    #       filament is also deleted
    def threshold_pen(self, th, mode='simple'):
        self.__threhold_fils(th, mode, func_name='get_penetration')

    # Filament thresholding based on length
    # th: thresholding object
    # mode: if 'simple' default just filament a thresholded, in 'complete' firstly a simple
    #       thresholding is done and all filaments with share a vertex with an already deleted
    #       filament is also deleted
    def threshold_len(self, th, mode='simple'):
        self.__threhold_fils(th, mode, func_name='get_length')

    # Filament thresholding based on dness metric
    # th: thresholding object
    # mode: if 'simple' default just filament a thresholded, in 'complete' firstly a simple
    #       thresholding is done and all filaments with share a vertex with an already deleted
    #       filament is also deleted
    def threshold_dness(self, th, mode='simple'):
        self.__threhold_fils(th, mode, func_name='get_dness')

    # Filament thresholding based on total curvature
    # th: thresholding object
    # mode: if 'simple' default just filament a thresholded, in 'complete' firstly a simple
    #       threholding is done and all filaments with share a vertex with an already deleted
    #       filament is also deleted
    def threshold_ct(self, th, mode='simple'):
        self.__threhold_fils(th, mode, func_name='get_total_curvature')

    # Filament thresholding based on max curvature
    # th: thresholding object
    # mode: if 'simple' default just filament a thresholded, in 'complete' firstly a simple
    #       threholding is done and all filaments with share a vertex with an already deleted
    #       filament is also deleted
    def threshold_mc(self, th, mode='simple'):
        self.__threhold_fils(th, mode, func_name='get_max_curvature')

    # Filament thresholding based on smoothness
    # th: thresholding object
    # mode: if 'simple' default just filament a thresholded, in 'complete' firstly a simple
    #       threholding is done and all filaments with share a vertex with an already deleted
    #       filament is also deleted
    def threshold_smo(self, th, mode='simple'):
        self.__threhold_fils(th, mode, func_name='get_smoothness')

    # Filament thresholding based on sinuosity
    # th: thresholding object
    # mode: if 'simple' default just filament a thresholded, in 'complete' firstly a simple
    #       threholding is done and all filaments with share a vertex with an already deleted
    #       filament is also deleted
    def threshold_sin(self, th, mode='simple'):
        self.__threhold_fils(th, mode, func_name='get_sinuosity')

    # Contact points (also filaments) thresholding based on sinuosity
    # th: thresholding object
    # mode: if 'simple' default just filament a thresholded, in 'complete' firstly a simple
    #       threholding is done and all filaments with share a vertex with an already deleted
    #       filament is also deleted
    # update: if True (default) cardinality is recomputed
    def threshold_card(self, th, mode='simple', update=True):
        if update:
            self.compute_cont_card()
        self.__threhold_conts(th, mode, func_name='get_card')

    # Threshold filaments according to its closest distance to another filament in the opposite
    # side
    # dst_th: minimum distance for thresholding (nm)
    def threshold_cdst_conts(self, dst_th):

        # Initialization
        lut_keep_in = np.zeros(shape=len(self.__cont_in), dtype=bool)
        lut_keep_out = np.zeros(shape=len(self.__cont_out), dtype=bool)
        res = self.__graph_in.get_resolution()

        # Get clouds
        cloud_in = self.get_cloud_points(MB_IN_LBL, STR_C, eps=None)
        cloud_out = self.get_cloud_points(MB_OUT_LBL, STR_C, eps=None)

        # Computing crossed distances
        for i, p in enumerate(cloud_in):
            hold = p - cloud_out
            hold = np.sum(hold*hold, axis=1).min()
            if (math.sqrt(hold) * res) < dst_th:
                lut_keep_in[i] = True
        for i, p in enumerate(cloud_out):
            hold = p - cloud_in
            hold = np.sum(hold*hold, axis=1).min()
            if (math.sqrt(hold) * res) < dst_th:
                lut_keep_out[i] = True

        # Updating Contact and Filament lists
        hold_cont_in = list()
        hold_fil_in = list()
        for i, c in enumerate(self.__cont_in):
            if lut_keep_in[i]:
                hold_cont_in.append(c)
                hold_fil_in.append(self.__fils_in[i])
        self.__cont_in = hold_cont_in
        self.__fils_in = hold_fil_in
        hold_cont_out = list()
        hold_fil_out = list()
        for i, c in enumerate(self.__cont_out):
            if lut_keep_out[i]:
                hold_cont_out.append(c)
                hold_fil_out.append(self.__fils_out[i])
        self.__cont_out = hold_cont_out
        self.__fils_out = hold_fil_out

    # Clusters contact points by using euclidean distances through filaments
    # side: set membrane side
    def cont_clusters_eu(self, side):

        # Initialization
        if side == MB_IN_LBL:
            conts = self.__cont_in
            graph = self.__graph_in
        elif side == MB_OUT_LBL:
            conts = self.__cont_out
            graph = self.__graph_out
        else:
            error_msg = 'Non-valid side.'
            raise pexceptions.PySegInputError(expr='cont_clusters_eu (MbFilaments)',
                                              msg=error_msg)
        lut_conts = (-1) * np.ones(shape=graph.get_skel().GetNumberOfPoints(), dtype=int)
        inn_coords = list()

        # Finding inner vertices
        n_conts = 0
        for c in conts:
            i_id = c.get_id()
            if lut_conts[i_id] == -1:
                lut_conts[i_id] = n_conts
                inn_coords.append(c.get_coords())
                n_conts += 1
        inn_coords = np.asarray(inn_coords, dtype=float)

        # Affinity propagation
        aff = sklearn.cluster.AffinityPropagation(affinity='euclidean')
        aff.fit(inn_coords)

        # Set contact class id
        for c in conts:
            c.set_clst_id_eu(aff.labels_[lut_conts[c.get_id()]])

    # Clusters contact points by using geodesic distances through filaments
    # side: set membrane side
    # approx: if activated, True (default False), shortest_path is based on pure geodesic distance
    # and not in connectivity
    def cont_clusters(self, side, approx=False):

        # Initialization
        if side == MB_IN_LBL:
            fils = self.__fils_in
            conts = self.__cont_in
            graph = copy.deepcopy(self.__graph_in)
        elif side == MB_OUT_LBL:
            fils = self.__fils_out
            conts = self.__cont_out
            graph = copy.deepcopy(self.__graph_out)
        else:
            error_msg = 'Non-valid side.'
            raise pexceptions.PySegInputError(expr='cont_clusters (MbFilaments)', msg=error_msg)

        # Delete vertex within a membrane
        graph.threshold_seg_region(MB_SEG, MB_LBL, keep_b=False)

        # Delete all edges which pass within the membrane
        skel = graph.get_skel()
        for e in graph.get_edges_list():
            e_ids = graph.get_edge_ids(e)
            for idx in e_ids:
                x, y, z = skel.GetPoint(idx)
                if self.__seg[int(round(x)), int(round(y)), int(round(z))] == MB_LBL:
                    graph.remove_edge(e)
                    break

        # Hold variables
        graph_GT = ps.graph.GraphGT(graph)
        graph_gt = graph_GT.get_gt()
        prop_id_v = graph_gt.vertex_properties[ps.globals.DPSTR_CELL]
        prop_fv_e = graph_gt.edge_properties[ps.globals.STR_FIELD_VALUE]
        prop_el_e = graph_gt.edge_properties[ps.globals.SGT_EDGE_LENGTH]
        lut_conts = (-1) * np.ones(shape=graph.get_skel().GetNumberOfPoints(), dtype=int)
        inn_vertices = list()
        inn_dsts = list()


        # Finding inner vertices
        n_conts = 0
        for i, c in enumerate(conts):
            c_id = c.get_id()
            if lut_conts[c_id] == -1:
                lut_conts[c_id] = n_conts
                v_id = fils[i].get_vertices()[1].get_id()
                inn_vertices.append(gt.find_vertex(graph_gt, prop_id_v, v_id)[0])
                inn_dsts.append(fils[i].get_cont_length(conts[i].get_id()))
                n_conts += 1
        inn_dsts = np.asarray(inn_dsts, dtype=float)

        # Getting affinity matrix
        aff_mat = np.zeros(shape=(n_conts, n_conts), dtype=float)
        if not approx:
            for i in range(n_conts):
                s = inn_vertices[i]
                for j in range(i+1, n_conts):
                    t = inn_vertices[j]
                    if int(s) == int(t):
                        dst = inn_dsts[i] + inn_dsts[j]
                    else:
                        # Finding most connected path
                        _, e_path = gt.shortest_path(graph_gt, s, t, prop_fv_e)
                        # Measuring the distance
                        dst = MAX_FLOAT
                        if len(e_path) > 0:
                            dst = 0
                            for e in e_path:
                                dst += prop_el_e[e]
                            dst += (inn_dsts[i] + inn_dsts[j])
                    aff_mat[i, j] = dst
                    aff_mat[j, i] = dst
        else:
            # Measure all distances at once
            dists_map = gt.shortest_distance(graph_gt, weights=prop_el_e)
            for i in range(n_conts):
                s = inn_vertices[i]
                for j in range(i+1, n_conts):
                    t = inn_vertices[j]
                    dst = dists_map[s][t]
                    if dst < MAX_FLOAT:
                        dst += (inn_dsts[i] + inn_dsts[j])
                    aff_mat[i, j] = dst
                    aff_mat[j, i] = dst

        # Affinity propagation
        aff = sklearn.cluster.AffinityPropagation(affinity='precomputed')
        # Ceiling max values for avoiding further overflows
        hold_aff_mat = copy.deepcopy(aff_mat)
        id_max = aff_mat == MAX_FLOAT
        aff_mat[id_max] = np.sum(aff_mat[np.invert(id_max)])
        # Affinity propagation requires negative distances
        aff.fit((-1) * aff_mat)

        # # Set contact class id
        # for c in conts:
        #     c.set_clst_id(aff.labels_[lut_conts[c.get_id()]])

        # 2nd label based purely on connectivity
        lbls2 = (-1) * np.ones(shape=n_conts, dtype=int)
        cont_lst = np.arange(n_conts).tolist()
        lbl2 = 0
        while len(cont_lst) > 0:
            cont = cont_lst.pop(0)
            lbls2[cont] = lbl2
            to_del = list()
            for n_cont in cont_lst:
                if hold_aff_mat[cont, n_cont] < MAX_FLOAT:
                    to_del.append(n_cont)
            for n_cont in to_del:
                cont_lst.remove(n_cont)
                lbls2[n_cont] = lbl2
            lbl2 += 1

        # Update affinity propagation labeling for not clustering together unconnected points
        labels = (-1) * np.ones(shape=aff.labels_.shape, dtype=aff.labels_.dtype)
        lbl3 = 0
        cont_lst = np.arange(n_conts).tolist()
        while len(cont_lst) > 0:
            cont = cont_lst.pop(0)
            h_lbl1 = aff.labels_[cont]
            h_lbl2 = lbls2[cont]
            labels[cont] = lbl3
            to_del = list()
            for n_cont in cont_lst:
                n_lbl1 = aff.labels_[n_cont]
                n_lbl2 = lbls2[n_cont]
                if (n_lbl1 == h_lbl1) and (n_lbl2 == h_lbl2):
                    to_del.append(n_cont)
            for n_cont in to_del:
                cont_lst.remove(n_cont)
                labels[n_cont] = lbl3
            lbl3 += 1

        # Set contact class id
        for c in conts:
            c.set_clst_id(labels[lut_conts[c.get_id()]])

    # Clusters contact points by simple connectivity
    # side: set membrane side
    def cont_clusters_conn(self, side):

        # Initialization
        if side == MB_IN_LBL:
            fils = self.__fils_in
            conts = self.__cont_in
            graph = copy.deepcopy(self.__graph_in)
        elif side == MB_OUT_LBL:
            fils = self.__fils_out
            conts = self.__cont_out
            graph = copy.deepcopy(self.__graph_out)
        else:
            error_msg = 'Non-valid side.'
            raise pexceptions.PySegInputError(expr='cont_clusters_conn (MbFilaments)', msg=error_msg)

        # Delete vertex within a membrane
        graph.threshold_seg_region(MB_SEG, MB_LBL, keep_b=False)

        # Delete all edges which pass within the membrane
        skel = graph.get_skel()
        for e in graph.get_edges_list():
            e_ids = graph.get_edge_ids(e)
            for idx in e_ids:
                x, y, z = skel.GetPoint(idx)
                if self.__seg[int(round(x)), int(round(y)), int(round(z))] == MB_LBL:
                    graph.remove_edge(e)
                    break

        # pyseg.disperse_io.save_vtp(graph.get_vtp(),
        #                            '/home/martinez/workspace/disperse/data/psd_an1/zd/pst/mb_fil/hold.vtp')

        # Hold variables
        graph_GT = ps.graph.GraphGT(graph)
        graph_gt = graph_GT.get_gt()
        prop_id_v = graph_gt.vertex_properties[ps.globals.DPSTR_CELL]
        prop_el_e = graph_gt.edge_properties[ps.globals.SGT_EDGE_LENGTH]
        lut_conts = (-2) * np.ones(shape=graph.get_skel().GetNumberOfPoints(), dtype=int)
        inn_vertices = list()


        # Finding inner vertices
        l_conts = list()
        for i, c in enumerate(conts):
            c_id = c.get_id()
            if lut_conts[c_id] == -2:
                lut_conts[c_id] = -1
                v_id = fils[i].get_vertices()[1].get_id()
                inn_vertices.append(gt.find_vertex(graph_gt, prop_id_v, v_id)[0])
                l_conts.append(c)
        n_conts = len(l_conts)

        # Measure all distances at once
        lut_det = np.zeros(shape=n_conts, dtype=bool)
        id_c = 0
        # ids = np.random.randint(0, n_conts, n_conts)
        ids = np.arange(n_conts)
        dists_map = gt.shortest_distance(graph_gt, weights=prop_el_e)
        for i, c in enumerate(l_conts):
            if not lut_det[i]:
                lbl = ids[id_c]
                id_c += 1
                s = inn_vertices[i]
                lut_conts[c.get_id()] = lbl
                lut_det[i] = True
                for j, n in enumerate(l_conts):
                    if not lut_det[j]:
                        t = inn_vertices[j]
                        if dists_map[s][t] < MAX_FLOAT:
                            lut_conts[n.get_id()] = lbl
                            lut_det[j] = True

        # Set contact class id
        for c in conts:
            c.set_clst_id(lut_conts[c.get_id()])

    # fname: file name ended with .pkl
    def pickle(self, fname):

        # Dump pickable objects and store the file names of the unpickable objects
        stem, ext = os.path.splitext(fname)
        self.__graph_fn_in = stem + '_graph_in.vtp'
        self.__graph_fn_out = stem + '_graph_out.vtp'

        # Pickle the GraphMCF
        self.__graph_in.pickle(self.__graph_fn_in)
        self.__graph_out.pickle(self.__graph_fn_out)

        # Pickle the rest of the object
        pkl_f = open(fname, 'w')
        try:
            pickle.dump(self, pkl_f)
        finally:
            pkl_f.close()

    # Computes contact points cardinality, how many times is this contant point visited by a
    # filamente in the network
    def compute_cont_card(self):

        # In graph
        # Counting cardinality
        lut_card = np.zeros(shape=self.__graph_in.get_skel().GetNumberOfPoints(), dtype=int)
        for c in self.__cont_in:
            lut_card[c.get_id()] += 1
        # Setting cardinality
        for c in self.__cont_in:
            c.set_card(lut_card[c.get_id()])

        # Out graph
        # Counting cardinality
        lut_card = np.zeros(shape=self.__graph_out.get_skel().GetNumberOfPoints(), dtype=int)
        for c in self.__cont_out:
            lut_card[c.get_id()] += 1
        # Setting cardinality
        for c in self.__cont_out:
            c.set_card(lut_card[c.get_id()])

    #### Internal functionality area

    # Stub for thesholding function
    def __threhold_fils(self, th, mode, func_name):
        if mode == 'simple':
            # Inside
            hold_fils = list()
            hold_conts = list()
            for i, f in enumerate(self.__fils_in):
                c = self.__cont_in[i]
                func = getattr(f, func_name)
                try:
                    val = th.test(func(c.get_id()))
                except TypeError:
                    val = th.test(func())
                if val:
                    hold_fils.append(f)
                    hold_conts.append(c)
            self.__fils_in = hold_fils
            self.__cont_in = hold_conts
            # Outside
            hold_fils = list()
            hold_conts = list()
            for i, f in enumerate(self.__fils_out):
                c = self.__cont_out[i]
                func = getattr(f, func_name)
                try:
                    val = th.test(func(c.get_id()))
                except TypeError:
                    val = th.test(func())
                if val:
                    hold_fils.append(f)
                    hold_conts.append(c)
            self.__fils_out = hold_fils
            self.__cont_out = hold_conts
        else:
            # Inside
            # Loop for making vertices
            lut = np.ones(shape=self.__graph_in.get_nid(), dtype=bool)
            for i, f in enumerate(self.__fils_in):
                c = self.__cont_in[i]
                func = getattr(f, func_name)
                try:
                    val = th.test(func(c.get_id()))
                except TypeError:
                    val = th.test(func())
                if val:
                    for v in f.get_vertices():
                        lut[v.get_id()] = False
            hold_fils = list()
            hold_conts = list()
            for i, f in enumerate(self.__fils_in):
                lock = True
                for v in f.get_vertices():
                    if lut[v.get_id()]:
                        lock = False
                        break
                if lock:
                    hold_fils.append(f)
                    hold_conts.append(self.__cont_in[i])
            self.__fils_in = hold_fils
            self.__cont_in = hold_conts
            # Outside
            # Loop for making vertices
            lut = np.ones(shape=self.__graph_out.get_nid(), dtype=bool)
            for i, f in enumerate(self.__fils_out):
                c = self.__cont_out[i]
                func = getattr(f, func_name)
                try:
                    val = th.test(func(c.get_id()))
                except TypeError:
                    val = th.test(func())
                if val:
                    for v in f.get_vertices()[1::]:
                        lut[v.get_id()] = False
            hold_fils = list()
            hold_conts = list()
            for i, f in enumerate(self.__fils_out):
                lock = True
                for v in f.get_vertices()[1::]:
                    if lut[v.get_id()]:
                        lock = False
                        break
                if lock:
                    hold_fils.append(f)
                    hold_conts.append(self.__cont_out[i])
            self.__fils_out = hold_fils
            self.__cont_out = hold_conts
        self.__filter_graphs()

    # Stub for thesholding based on Contat function
    def __threhold_conts(self, th, mode, func_name):
        if mode == 'simple':
            # Inside
            hold_fils = list()
            hold_conts = list()
            for i, f in enumerate(self.__fils_in):
                c = self.__cont_in[i]
                func = getattr(c, func_name)
                val = th.test(func())
                if val:
                    hold_fils.append(f)
                    hold_conts.append(c)
            self.__fils_in = hold_fils
            self.__cont_in = hold_conts
            # Outside
            hold_fils = list()
            hold_conts = list()
            for i, f in enumerate(self.__fils_out):
                c = self.__cont_out[i]
                func = getattr(c, func_name)
                val = th.test(func())
                if val:
                    hold_fils.append(f)
                    hold_conts.append(c)
            self.__fils_out = hold_fils
            self.__cont_out = hold_conts
        else:
            # Inside
            # Loop for making vertices
            lut = np.ones(shape=self.__graph_in.get_nid(), dtype=bool)
            for i, f in enumerate(self.__fils_in):
                c = self.__cont_in[i]
                func = getattr(c, func_name)
                val = th.test(func())
                if val:
                    for v in f.get_vertices():
                        lut[v.get_id()] = False
            hold_fils = list()
            hold_conts = list()
            for i, f in enumerate(self.__fils_in):
                lock = True
                for v in f.get_vertices():
                    if lut[v.get_id()]:
                        lock = False
                        break
                if lock:
                    hold_fils.append(f)
                    hold_conts.append(self.__cont_in[i])
            self.__fils_in = hold_fils
            self.__cont_in = hold_conts
            # Outside
            # Loop for making vertices
            lut = np.ones(shape=self.__graph_out.get_nid(), dtype=bool)
            for i, f in enumerate(self.__fils_out):
                c = self.__cont_out[i]
                func = getattr(c, func_name)
                val = th.test(func())
                if val:
                    for v in f.get_vertices()[1::]:
                        lut[v.get_id()] = False
            hold_fils = list()
            hold_conts = list()
            for i, f in enumerate(self.__fils_out):
                lock = True
                for v in f.get_vertices()[1::]:
                    if lut[v.get_id()]:
                        lock = False
                        break
                if lock:
                    hold_fils.append(f)
                    hold_conts.append(self.__cont_out[i])
            self.__fils_out = hold_fils
            self.__cont_out = hold_conts
        self.__filter_graphs()

    # Restore previous state
    def __setstate__(self, state):
        self.__dict__.update(state)
        # Restore unpickable objects
        self.__graph_in = ps.factory.unpickle_obj(self.__graph_fn_in)
        self.__graph_out = ps.factory.unpickle_obj(self.__graph_fn_out)

    # Copy the object's state from self.__dict__ which contains all instance attributes.
    # Afterwards remove unpickable objects
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_MbFilaments__graph_in']
        del state['_MbFilaments__graph_out']
        return state

    # Find all filaments for the segmented membrane, every filament has his head
    def __build(self, min_length, max_length, nrad):

        # Getting shortest distance transform to membrane
        self.__mb_dst = distance_transform_edt(np.invert(self.__seg == 1))
        self.__mb_dst *= self.__graph_in.get_resolution()

        # Graph membrane region segmentation
        self.__graph_in.add_scalar_field_nn(self.__seg, MB_SEG)
        self.__graph_out = copy.deepcopy(self.__graph_in)
        self.__graph_in.threshold_seg_region(MB_SEG, MB_OUT_LBL, keep_b=False)
        self.__graph_in.threshold_seg_region(MB_SEG, MB_LBL, keep_b=True)
        self.__graph_out.threshold_seg_region(MB_SEG, MB_IN_LBL, keep_b=False)
        self.__graph_out.threshold_seg_region(MB_SEG, MB_LBL, keep_b=True)

        # Getting GraphGT
        self.__graph_gt_in = ps.graph.GraphGT(self.__graph_in)
        graph_gt_in = self.__graph_gt_in.get_gt()
        prop_seg_in = graph_gt_in.vertex_properties[MB_SEG]
        prop_id_v_in = graph_gt_in.vertex_properties[ps.globals.DPSTR_CELL]
        prop_id_e_in = graph_gt_in.edge_properties[ps.globals.DPSTR_CELL]
        # prop_len_in = graph_gt_in.edge_properties[ps.globals.SGT_EDGE_LENGTH]
        prop_len_in = graph_gt_in.edge_properties[ps.globals.STR_VERT_DST]
        prop_con_in = graph_gt_in.edge_properties[ps.globals.STR_FIELD_VALUE]

        self.__graph_gt_out = ps.graph.GraphGT(self.__graph_out)
        graph_gt_out = self.__graph_gt_out.get_gt()
        prop_seg_out = graph_gt_out.vertex_properties[MB_SEG]
        prop_id_v_out = graph_gt_out.vertex_properties[ps.globals.DPSTR_CELL]
        prop_id_e_out = graph_gt_out.edge_properties[ps.globals.DPSTR_CELL]
        # prop_len_out = graph_gt_out.edge_properties[ps.globals.SGT_EDGE_LENGTH]
        prop_len_out = graph_gt_out.edge_properties[ps.globals.STR_VERT_DST]
        prop_con_out = graph_gt_out.edge_properties[ps.globals.STR_FIELD_VALUE]

        # Get sources lists (heads)
        sources_in = list()
        sources_out = list()
        for e in graph_gt_in.edges():
            s, t = e.source(), e.target()
            s_lbl, t_lbl = prop_seg_in[s], prop_seg_in[t]
            if s_lbl == MB_LBL:
                if t_lbl == MB_IN_LBL:
                    sources_in.append(s)
            elif t_lbl == MB_LBL:
                if s_lbl == MB_IN_LBL:
                    sources_in.append(t)
        for e in graph_gt_out.edges():
            s, t = e.source(), e.target()
            s_lbl, t_lbl = prop_seg_out[s], prop_seg_out[t]
            if s_lbl == MB_LBL:
                if t_lbl == MB_OUT_LBL:
                    sources_out.append(s)
            elif t_lbl == MB_LBL:
                if s_lbl == MB_OUT_LBL:
                    sources_out.append(t)

        # Get side masks
        mask_in = prop_seg_in.get_array() == MB_IN_LBL
        mask_out = prop_seg_out.get_array() == MB_OUT_LBL

        # Measuring all shortest distances
        dists_map_in = gt.shortest_distance(graph_gt_in, weights=prop_len_in)
        dists_map_out = gt.shortest_distance(graph_gt_out, weights=prop_len_out)

        # Loop for finding the heads (inside)
        for h in sources_in:
            dists = dists_map_in[h].get_array()
            ids = np.where((dists > min_length) & (dists < max_length) & mask_in)[0]
            # Loop for finding the filaments with share this head
            for idt in ids:
                t = graph_gt_in.vertex(idt)
                v_path, e_path = gt.shortest_path(graph_gt_in, h, t, weights=prop_con_in)
                hold_length = 0
                vertex_list = list()
                edge_list = list()
                for i, e in enumerate(e_path):
                    vertex_list.append(self.__graph_in.get_vertex(prop_id_v_in[v_path[i]]))
                    edge = self.__graph_in.get_edge(prop_id_e_in[e])
                    hold_length += self.__graph_in.get_edge_length(edge)
                    edge_list.append(edge)
                vertex_list.append(self.__graph_in.get_vertex(prop_id_v_in[v_path[-1]]))
                if (hold_length > min_length) and (hold_length < max_length):
                    # Building a filament
                    self.__fils_in.append(Filament(self.__graph_in,
                                                   vertex_list,
                                                   edge_list,
                                                   self.__mb_dst))

        # Loop for finding the heads (outside)
        for h in sources_out:
            dists = dists_map_out[h].get_array()
            ids = np.where((dists > min_length) & (dists < max_length) & mask_out)[0]
            # Loop for finding the filaments with share this head
            for idt in ids:
                t = graph_gt_out.vertex(idt)
                v_path, e_path = gt.shortest_path(graph_gt_out, h, t, weights=prop_con_out)
                hold_length = 0
                vertex_list = list()
                edge_list = list()
                for i, e in enumerate(e_path):
                    vertex_list.append(self.__graph_out.get_vertex(prop_id_v_out[v_path[i]]))
                    edge = self.__graph_out.get_edge(prop_id_e_out[e])
                    hold_length += self.__graph_out.get_edge_length(edge)
                    edge_list.append(edge)
                vertex_list.append(self.__graph_out.get_vertex(prop_id_v_out[v_path[-1]]))
                if (hold_length > min_length) and (hold_length < max_length):
                    # Building a filament
                    self.__fils_out.append(Filament(self.__graph_out,
                                                    vertex_list,
                                                    edge_list,
                                                    self.__mb_dst))

        # Finding/building mb-filament contact points
        c_in_ids = list()
        for f in self.__fils_in:
            cont = Contact(self.__graph_in, f, self.__seg)
            self.__cont_in.append(cont)
            c_in_ids.append(cont.get_id())
        c_out_ids = list()
        for f in self.__fils_out:
            cont = Contact(self.__graph_out, f, self.__seg)
            self.__cont_out.append(cont)
            c_out_ids.append(cont.get_id())

        # Filter the graphs with the filaments
        self.__filter_graphs()

        # Computing normal vector for contact points
        # Computing surface clouds of points
        surf_cloud_in = list()
        surf_cloud_out = list()
        mb_ids = np.where(self.__seg == MB_LBL)
        n_x, n_y, n_z = self.__seg.shape
        for i in range(len(mb_ids[0])):
            ids = np.asarray((mb_ids[0][i], mb_ids[1][i], mb_ids[2][i]), dtype=int)
            x_min, y_min, z_min = ids - 2
            x_max, y_max, z_max = ids + 2
            if x_min <= 0:
                x_min = 0
            if x_max >= n_x:
                x_max = n_x - 1
            if y_min <= 0:
                y_min = 0
            if y_max >= n_y:
                y_max = n_y - 1
            if z_min <= 0:
                z_min = 0
            if z_max >= n_z:
                z_max = n_z - 1
            neigh = self.__seg[x_min:x_max, y_min:y_max, z_min:z_max]
            if (neigh == MB_IN_LBL).sum() > 0:
                surf_cloud_in.append(ids)
            if (neigh == MB_OUT_LBL).sum() > 0:
                surf_cloud_out.append(ids)
        # Compute normal for every contact id
        if len(self.__cont_in) > 0:
            surf_cloud_in = np.asarray(surf_cloud_in, dtype=float)
            field_in = self.__compute_normal_field(surf_cloud_in, self.__seg, MB_IN_LBL, nrad)
            c_in_ids = np.asarray(c_in_ids, dtype=int)
            c_max = c_in_ids.max()
            lut_ids = np.zeros(shape=c_max+1, dtype=bool)
            lut_v = np.zeros(shape=(c_max+1, 3), dtype=float)
            skel = self.__graph_in.get_skel()
            for ct in self.__cont_in:
                ct_id = ct.get_id()
                # Find normal only if it has not been already computed
                if not lut_ids[ct_id]:
                    point = np.asarray(skel.GetPoint(ct_id), dtype=float)
                    norm = evaluate_field_3x3(field_in, point)
                    # Normalization
                    n_norm = np.sqrt(np.sum(norm * norm))
                    if n_norm > 0:
                        norm /= n_norm
                    # Update LUTs
                    lut_ids[ct_id] = True
                    lut_v[ct_id][:] = norm
                # Set normal in contact point
                ct.set_norm(lut_v[ct_id])
        if len(self.__cont_out) > 0:
            surf_cloud_out = np.asarray(surf_cloud_out, dtype=float)
            field_out = self.__compute_normal_field(surf_cloud_out, self.__seg, MB_OUT_LBL, nrad)
            c_out_ids = np.asarray(c_out_ids, dtype=int)
            c_max = c_out_ids.max()
            lut_ids = np.zeros(shape=c_max+1, dtype=bool)
            lut_v = np.zeros(shape=(c_max+1, 3), dtype=float)
            skel = self.__graph_out.get_skel()
            for ct in self.__cont_out:
                ct_id = ct.get_id()
                # Find normal only if it has not been already computed
                if not lut_ids[ct_id]:
                    point = np.asarray(skel.GetPoint(ct_id), dtype=float)
                    norm = evaluate_field_3x3(field_out, point)
                    # Normalization
                    n_norm = np.sqrt(np.sum(norm * norm))
                    if n_norm > 0:
                        norm /= n_norm
                    # Update LUTs
                    lut_ids[ct_id] = True
                    lut_v[ct_id][:] = norm
                # Set normal in contact point
                ct.set_norm(lut_v[ct_id])

    def __filter_graphs(self):

        lut_del = np.ones(shape=self.__graph_in.get_nid(), dtype=bool)
        for f in self.__fils_in:
            for v in f.get_vertices():
                lut_del[v.get_id()] = False
            for e in f.get_edges():
                lut_del[e.get_id()] = False
        for f in self.__fils_out:
            for v in f.get_vertices():
                lut_del[v.get_id()] = False
            for e in f.get_edges():
                lut_del[e.get_id()] = False
        for v in self.__graph_in.get_vertices_list():
            if lut_del[v.get_id()]:
                self.__graph_in.remove_vertex(v)
        for e in self.__graph_in.get_edges_list():
            if lut_del[e.get_id()]:
                self.__graph_in.remove_edge(e)
        for v in self.__graph_out.get_vertices_list():
            if lut_del[v.get_id()]:
                self.__graph_out.remove_vertex(v)
        for e in self.__graph_out.get_edges_list():
            if lut_del[e.get_id()]:
                self.__graph_out.remove_edge(e)

    # Computes the inclination angle in degrees for Filament-Contact pair
    def __alpha(self, fil, cont):

        # Compute contact to filament tail vector
        v = fil.get_tail_coords() - cont.get_coords()
        vnorm = math.sqrt((v * v).sum())
        if vnorm <= 0:
            return .0
        v /= vnorm

        # Compute smallest angle between this vector and contact normal
        n = cont.get_norm()
        nnorm = math.sqrt((n * n).sum())
        if nnorm <= 0:
            return .0
        n /= nnorm
        return (180./np.pi) * math.acos((n*v).sum())

    # Computes the inclination angle in degrees of a filament against missing wedge axis
    # [0, 0, 1]
    def __beta(self, fil, cont):

        # Compute contact to filament tail vector
        v = fil.get_tail_coords() - cont.get_coords()
        vnorm = math.sqrt((v * v).sum())
        if vnorm <= 0:
            return .0
        v /= vnorm

        # Compute smallest angle between this vector and missing wedge axis
        n = np.asarray((0, 0, 1), dtype=float)
        nnorm = math.sqrt((n * n).sum())
        if nnorm <= 0:
            return .0
        n /= nnorm
        b1 = math.acos((n*v).sum())
        v *= -1
        b2 = math.acos((n*v).sum())
        if b1 <= b2:
            return (180./np.pi) * b1
        else:
            return (180./np.pi) * b2

    # Computes the vector field of normals from a cloud of points
    def __compute_normal_field(self, cloud, seg, lbl, nrad):

        # Computing seed normals
        n_x, n_y, n_z = seg.shape
        n_field = np.zeros(shape=(n_x, n_y, n_z, 3), dtype=float)
        for p in cloud:
            hold = cloud - p
            hold = np.sqrt(np.sum(hold * hold, axis=1))
            n_ids = np.where(hold < nrad)[0]
            if len(n_ids) > 0:
                n_cloud = cloud[n_ids][:]
                # Computing normal and its orientation
                norm = normal3d_point_cloud(n_cloud)
                p_n = p + norm
                p_nn = p - norm
                th1 = trilin3d(self.__seg, p_n) - lbl
                th2 = trilin3d(self.__seg, p_nn) - lbl
                th1 *= th1
                th2 *= th2
                if th1 >= th2:
                    norm *= -1
                n_field[int(p[0]), int(p[1]), int(p[2]), :] = norm

        hold_fname = '/home/martinez/workspace/disperse/data/jbp/mb_fil/hold_1.mrc'
        hold = np.sqrt(n_field * n_field)
        ps.disperse_io.save_numpy(hold.sum(axis=3), hold_fname)

        # Normal field Gaussian smoothing to each channel
        n_field[:, :, :, 0] = sp.ndimage.filters.gaussian_filter(n_field[:, :, :, 0], nrad)
        n_field[:, :, :, 1] = sp.ndimage.filters.gaussian_filter(n_field[:, :, :, 1], nrad)
        n_field[:, :, :, 2] = sp.ndimage.filters.gaussian_filter(n_field[:, :, :, 2], nrad)

        hold_fname = '/home/martinez/workspace/disperse/data/jbp/mb_fil/hold_2.mrc'
        hold = np.sqrt(n_field * n_field)
        ps.disperse_io.save_numpy(hold.sum(axis=3), hold_fname)

        return n_field

##########################################################################################
# Class for modelling Connectors Domain (a set of connectors)
##########################################################################################

class ConnDom(object):

    # cloud: cloud of points with coordinates of the contact points
    # lbls: label (integers) for every point for defining the sets
    # box: box with encloses the cloud
    def __init__(self, cloud, lbls, box):
        self.__cloud = np.asarray(cloud, dtype=float)
        self.__lbls = np.asarray(lbls, dtype=int)
        self.__box = box
        self.__chulls = list()
        self.__c_clouds = list()
        self.__c_ids = list()
        self.__moments = list()
        self.__del_mask = None
        self.__build()

    #### Set/Get methods area

    def get_lbls(self):
        return self.__lbls

    def get_box(self):
        return self.__box

    def get_cloud(self):
        return self.__cloud

    def get_clsts_list(self):
        return copy.deepcopy(self.__c_clouds)

    def get_clst_npoints(self):
        npoints = np.zeros(shape=len(self.__c_clouds), dtype=int)
        for i, clst in enumerate(self.__c_clouds):
            npoints[i] = len(clst)
        return npoints

    def get_clst_areas(self):
        areas = np.zeros(shape=len(self.__moments), dtype=float)
        for i, m in enumerate(self.__moments):
            areas[i] = m['m00']
        return areas

    def get_clst_densities(self):
        npoints = self.get_clst_npoints()
        areas = self.get_clst_areas()
        return np.asarray(npoints, dtype=float) / areas

    # Computing clusters' Center of Gravity
    def get_clst_cg(self):
        cent = np.zeros(shape=(len(self.__moments), 2), dtype=float)
        for i, m in enumerate(self.__moments):
            cent[i, 0] = m['m10'] / m['m00']
            cent[i, 1] = m['m01'] / m['m00']
        return cent

    # Computing clusters' Semi-major axes
    def get_clst_axes(self):
        axes = np.zeros(shape=(len(self.__moments), 2), dtype=float)
        for i, m in enumerate(self.__moments):
            hold = m['mu20'] - m['mu02']
            hold *= hold
            hold1 = 0.5 * (m['mu20'] + m['mu02'])
            hold2 = 4.*m['mu11']*m['mu11']
            if hold2 < hold:
                axes[i, 0] = -1
                axes[i, 1] = -1
            else:
                hold2 = math.sqrt(hold2 - hold)
                axes[i, 0] = math.sqrt(hold1 + hold2)
                if hold1 < hold2:
                    axes[i, 1] = -1
                else:
                    axes[i, 1] = math.sqrt(hold1 - hold2)

        return axes

    # Computing clusters' orientation
    def get_clst_orient(self):
        orients = np.zeros(shape=len(self.__moments), dtype=float)
        for i, m in enumerate(self.__moments):
            orients[i] = 0.5 * math.atan2(2.*m['mu11'], m['mu20']-m['mu02'])
        return orients

    # Computing clusters' roundness
    def get_clst_round(self):
        rounds = np.zeros(shape=len(self.__chulls), dtype=float)
        for i, chull in enumerate(self.__chulls):
            # Computing perimeter
            chull_hold = np.asarray(chull, np.uint8)
            countour = cv.findContours(chull_hold,
                                       mode=cv.RETR_EXTERNAL,
                                       method=cv.CHAIN_APPROX_SIMPLE)
            # if len(countour) != 1:
            #     error_msg = 'Unexpected shape.'
            #     raise pexceptions.PySegInputError(expr='get_clst_round (ConnDom)',
            #                                       msg=error_msg)
            if len(countour[0]) == 0:
                rounds[i] = -1
            else:
                per = cv.arcLength(countour[0][0], closed=True)
                rounds[i] = (per * per) / (2. * np.pi * self.__moments[i]['m00'])
        return rounds

     # Computing clusters' eccentricity
    def get_clst_ecc(self):
        eccs = np.zeros(shape=len(self.__moments), dtype=float)
        for i, m in enumerate(self.__moments):
            hold1 = m['mu20'] - m['mu02']
            hold2 = m['mu20'] + m['mu02']
            hold3 = hold1*hold1 - 4.*m['mu11']*m['mu11']
            if hold3 == 0:
                eccs[i] = 0
            elif hold2 == 0:
                eccs[i] = -1
            else:
                eccs[i] = hold3 / (hold2 * hold2)
        return eccs

    # Generates a random distribution of the internal clusters
    # tries: number of tries for getting the less overlapped location for every cluster
    # Returns: an array with new centroids
    def get_rand_clsts(self, tries=50):

        # Initialization
        # Get the number of different labels
        set_lbls = np.array(list(set(self.__lbls)))
        n_cgs = np.zeros(shape=(len(set_lbls), 2), dtype=float)
        # Mask for checking the overlapping
        off_x = math.floor(self.__box[1])
        off_y = math.floor(self.__box[0])
        m, n = math.ceil(self.__box[3]) - off_x + 1, math.ceil(self.__box[2]) - off_y + 1
        mask = np.zeros(shape=(m, n), dtype=bool)

        # Loop for clusters
        for i, lbl in enumerate(set_lbls):
            # Get all contact point for the current domain
            ids = np.where(self.__lbls == lbl)[0]
            c_cloud = self.__cloud[ids]
            # Translate to base coordinates and computes minimum distance to center of gravity
            cg = c_cloud.mean(axis=0)
            f_cloud = c_cloud - cg
            min_dst = np.sqrt((f_cloud*f_cloud).sum(axis=1)).min()
            # Compute valid search areas
            dst_t = sp.ndimage.morphology.distance_transform_edt(np.invert(mask))
            mask_dst = np.zeros(shape=mask.shape, dtype=mask.dtype)
            mask_dst[dst_t > min_dst] = True
            if mask_dst.sum() <= 0:
                error_msg = 'Mask fully overlapped.'
                raise pexceptions.PySegTransitionError(expr='get_rand_clsts (ConnDom)',
                                                       msg=error_msg)
            # Keep the best try (lower overlapping)
            h_cgs = np.zeros(shape=(tries, 2), dtype=float)
            min_ov = MAX_FLOAT
            h_cg = None
            h_chull = np.zeros(shape=mask.shape, dtype=mask.dtype)
            for c_try in range(tries):
                # Random selection for the new centroid from valid areas
                m_ids = np.where(mask_dst)
                r_x, r_y = np.random.randint(0, len(m_ids[0])), np.random.randint(0, len(m_ids[1]))
                cg_x, cg_y = m_ids[0][r_x], m_ids[1][r_y]
                # Rotate randomly against base center [0, 0]
                rho = np.random.rand() * (2*np.pi)
                sinr, cosr = math.sin(rho), math.cos(rho)
                r_cloud = np.zeros(shape=f_cloud.shape, dtype=f_cloud.dtype)
                r_cloud[:, 0] = f_cloud[:, 0]*cosr - f_cloud[:, 1]*sinr
                r_cloud[:, 1] = f_cloud[:, 0]*sinr + f_cloud[:, 1]*cosr
                # Translation to randomly already selected center
                n_cg = np.asarray((cg_x, cg_y) , dtype=float)
                v = n_cg - cg
                t_cloud = r_cloud + v
                chull, _ = self.__compute_chull_no_bound(t_cloud)
                # Update minimum overlap
                ov = (chull * mask).sum()
                if ov < min_ov:
                    min_ov = ov
                    h_cg = n_cg
                    h_chull = chull
                else:
                    if h_cg is None:
                        h_cg = n_cg
                h_cgs[c_try, :] = n_cg

            # Update mask
            mask[h_chull] = True
            # Get new center transposed
            n_cgs[i, 0] = h_cg[1]
            n_cgs[i, 1] = h_cg[0]

        return n_cgs

    # Generates a random distribution of the contact points
    # Returns: an array with new contacts
    def get_rand_contacts(self):

        # Initialization
        # Generate random contacts
        contacts = np.zeros(shape=(self.__cloud.shape[0], 2), dtype=float)
        hold = np.random.rand(self.__cloud.shape[0], 2)
        # Fit to box
        off_x = math.floor(self.__box[1])
        off_y = math.floor(self.__box[0])
        m, n = math.ceil(self.__box[3]) - off_x + 1, math.ceil(self.__box[2]) - off_y + 1
        contacts[:, 1] = off_x + m * hold[:, 0]
        contacts[:, 0] = off_y + n * hold[:, 1]

        return contacts


    #### External functionality area

    # th_*: threshold object for every cluster property
    def threshold(self, th_npoints=None, th_areas=None, th_den=None, th_round=None):
        if th_npoints is not None:
            self.__threshold(th_npoints, 'get_clst_npoints')
        if th_areas is not None:
            self.__threshold(th_areas, 'get_clst_areas')
        if th_den is not None:
            self.__threshold(th_den, 'get_clst_densities')
        if th_round is not None:
            self.__threshold(th_round, 'get_clst_round')

    # Print convex hull image
    # prop: string which identifies cluster property, or the property values array
    # Return: float single channel numpy array image
    def print_image(self, prop=None):

        # Creating the holding image
        off_x = math.floor(self.__box[1])
        off_y = math.floor(self.__box[0])
        m, n = int(math.ceil(self.__box[3]) - off_x + 1), int(math.ceil(self.__box[2]) - off_y + 1)
        img = np.zeros(shape=(m, n), dtype=float)
        if prop is not None:
            if isinstance(prop, str):
                values = getattr(self, 'get_clst_'+prop)()
            else:
                values = prop
        else:
            values = np.ones(shape=len(self.__chulls), dtype=float)

        # Printing the convex hull
        for i, chull in enumerate(self.__chulls):
            img[chull] = values[i]

        return img

    # Plot the clusters into a figure in blocking mode
    # prop: string which identifies cluster property
    # centers: if True (default False) the centers of gravity are printed, otherwise the convex hulls
    def plot_clusters(self, prop=None, centers=False):

        # plt.ion()
        plt.figure()
        title = str('Clusters cloud ')
        if centers:
            title += 'centers '
        if prop is not None:
            title += ('for property ' + prop)
        plt.title(title)
        plt.xlabel('X')
        plt.ylabel('Y')
        if centers:
            plt.axis('scaled')
            plt.xlim(self.__box[0], self.__box[2])
            plt.ylim(self.__box[1], self.__box[3])
            cgs = self.get_clst_cg()
            if prop is None:
                plt.scatter(cgs[:, 0], cgs[:, 1])
            else:
                values = getattr(self, 'get_clst_'+prop)()
                cax = plt.scatter(cgs[:, 0], cgs[:, 1], c=values, cmap=cm.jet)
                plt.colorbar(cax, orientation='horizontal')
        else:
            img = self.print_image(prop)
            if prop is None:
                plt.imshow(img, interpolation='nearest')
            else:
                cax = plt.imshow(img, interpolation='nearest', cmap=cm.jet)
                plt.colorbar(cax, orientation='horizontal')

        plt.show(block=True)

    # Borderness metric for measuring in a cloud of points within a convex polygon are organized
    # either close to the polygon border 1 or the center 0.
    # cloud: cloud of point coordinates distributed on the polygon
    # dilate: if not None (default None) it sets the number of dilation at every convex hull
    # Returns: borderness value and the amount of points on the polygon (the only ones used
    # for the bordeness computation)
    def get_clst_border(self, cloud, dilate=None):

        # Creating dilation mask
        mask = None
        do_iilate = (dilate is not None) and (dilate > 0)
        if do_iilate:
            mask = sp.ndimage.generate_binary_structure(2, 1)

        # Get centers of gravity
        cgs = self.get_clst_cg()

        # Translating cloud coordinates
        off_x = math.floor(self.__box[1])
        off_y = math.floor(self.__box[0])
        c_cloud = np.zeros(shape=cloud.shape, dtype=float)
        c_cloud[:, 0] = cloud[:, 0] - off_x
        c_cloud[:, 1] = cloud[:, 1] - off_y

        # Loop for computing border for every convex hull
        border = np.zeros(shape=cgs.shape[0], dtype=float)
        n_pts = np.zeros(shape=cgs.shape[0], dtype=int)
        if do_iilate:
            for i, chull in enumerate(self.__chulls):
                d_chull = sp.ndimage.morphology.binary_dilation(chull,
                                                                structure=mask,
                                                                iterations=dilate)
                border[i], n_pts[i] = self.__compute_cont_borderness(d_chull, cgs[i], c_cloud)
        else:
            for i, chull in enumerate(self.__chulls):
                border[i], n_pts[i] = self.__compute_cont_borderness(chull, cgs[i], c_cloud)

        return border, n_pts

    # Returns the binary mask where False values represented the region occupied by clustered
    # segmented out
    def get_del_mask(self):
        return self.__del_mask

    # fname: file name ended with .pkl
    def pickle(self, fname):

        pkl_f = open(fname, 'w')
        try:
            pickle.dump(self, pkl_f)
        finally:
            pkl_f.close()

    #### Internal functionality area

    def __build(self):

        # Initialization
        lut_lbls = np.ones(shape=self.__lbls.max()+1, dtype=bool)
        off_x = int(math.floor(self.__box[1]))
        off_y = int(math.floor(self.__box[0]))
        m, n = int(math.ceil(self.__box[3]) - off_x + 1), int(math.ceil(self.__box[2]) - off_y + 1)
        self.__del_mask = np.ones(shape=(m, n), dtype=bool)

        # Loop for every label
        for lbl in self.__lbls:
            # Check this label is not processed yet
            if lut_lbls[lbl]:
                # Get all contact point for the current domain
                ids = np.where(self.__lbls == lbl)[0]
                c_cloud = self.__cloud[ids]
                self.__c_clouds.append(c_cloud)
                self.__c_ids.append(ids)
                chull = self.__compute_chull(c_cloud)
                self.__chulls.append(chull)
                chull_hold = np.asarray(chull, np.uint8)
                self.__moments.append(cv.moments(chull_hold, binaryImage=True))
                # Mark label as processed
                lut_lbls[lbl] = False

    # Stub for thresholding
    def __threshold(self, th, func_name):
        # Initialization
        hold_clouds = list()
        hold_lbls = list()
        hold_chulls = list()
        hold_c_clouds = list()
        hold_moments = list()
        hold_c_ids = list()

        func = getattr(self, func_name)
        values = func()
        for (c_cloud, lbl, chull, mmnts, value) in \
            zip(self.__c_clouds, self.__c_ids, self.__chulls, self.__moments, values):
            if th.test(value):
                # If the test is passed the cluster is kept
                hold_c_clouds.append(c_cloud)
                hold_c_ids.append(lbl)
                hold_chulls.append(chull)
                hold_moments.append(mmnts)
                # ids = np.where(self.__lbls == lbl)[0]
                hold_clouds += list(self.__cloud[lbl])
                hold_lbls += list(self.__lbls[lbl])
            else:
                self.__del_mask[chull] = False

        # Update class attributes
        self.__cloud = np.asarray(hold_clouds, dtype=float)
        self.__lbls = np.asarray(hold_lbls, dtype=int)
        self.__chulls = list(hold_chulls)
        self.__c_clouds = list(hold_c_clouds)
        self.__moments = list(hold_moments)
        self.__c_ids = list(hold_c_ids)

    # Returns convex hull and discard points out of bounds are discarded and no exception is
    #           raised, instead in a second variable a true is returned
    def __compute_chull_no_bound(self, c_cloud):

        # Create holding image
        off_x = int(math.floor(self.__box[1]))
        off_y = int(math.floor(self.__box[0]))
        m, n = int(math.ceil(self.__box[3]) - off_x + 2), int(math.ceil(self.__box[2]) - off_y + 2)
        img = np.zeros(shape=(m, n), dtype=bool)

        # Filling holding image
        hold = np.asarray(np.round(c_cloud), dtype=int)
        hold[:, 0] -= off_y
        hold[:, 1] -= off_x
        excep = False
        p_count = 0
        for p in hold:
            try:
                img[p[1], p[0]] = True
            except IndexError:
                excep = True
                continue
            p_count += 1

        # Computing the convex hull
        if p_count > 0:
            chull = np.asarray(convex_hull_image(img), dtype=bool)
        else:
            chull = img

        return chull, excep

    def __compute_chull(self, c_cloud):

        # Create holding image
        off_x = int(math.floor(self.__box[1]))
        off_y = int(math.floor(self.__box[0]))
        m, n = int(math.ceil(self.__box[3]) - off_x + 1), int(math.ceil(self.__box[2]) - off_y + 1)
        img = np.zeros(shape=(m, n), dtype=bool)

        # Filling holding image
        hold = np.asarray(np.round(c_cloud), dtype=int)
        hold[:, 0] -= off_y
        hold[:, 1] -= off_x
        for p in hold:
            # try:
            img[p[1], p[0]] = True
            # except IndexError:
            #     print '\tWarning: __compute_chull IndexError'

        # Computing the convex hull
        chull = np.asarray(convex_hull_image(img), dtype=bool)

        return chull

    # Borderness metric for measuring in a cloud of points within a convex polygon are organized
    # either close to the polygon border 1 or the center 0.
    # chull: binary mask where True is fg polygon and False is bg
    # cg: polygon center of gravity
    # c_cloud: cloud of point coordinates distributed on the polygon
    # Returns: borderness value and the amount of points on the polygon (the only ones used
    # for the bordeness computation)
    def __compute_cont_borderness(self, chull, cg, c_cloud):

        # Distance transform
        ids = np.zeros(((np.ndim(chull),) + chull.shape), dtype=np.int32)
        sp.ndimage.morphology.distance_transform_edt(chull, return_indices=True, indices=ids)

        # Loop for points
        n_points = 0
        accum = 0
        for p in c_cloud:
            pt = p[::-1]
            # Is on the polygon?
            if trilin2d(chull, pt) > 0.:
                # Getting border closest point
                pi = np.round(p).astype(int)
                idy, idx = ids[0, pi[1], pi[0]], ids[1, pi[1], pi[0]]
                b = [idx, idy]
                # Distance point to center
                hold = p - cg
                dpc = math.sqrt(np.sum(hold * hold))
                # Polygon radius
                hold = cg - b
                rad = math.sqrt(np.sum(hold * hold))

                if rad > 0:
                    accum += dpc / rad
                n_points += 1

        if n_points > 0:
            return (accum / n_points), n_points
        else:
            return 0., 0
