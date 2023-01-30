"""

    Script for cleft segmentation

    Input:  - Pickle files with MbGraphMCF for post- and pre-synaptic membranes
            - Membranes segmentation files
            - xml file with mb_slice definitions

    Output: - Cloud files
            - Segmentations

"""

__author__ = 'Antonio Martinez-Sanchez'

########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '/home/martinez/workspace/disperse/data/psd_an1'

# Input pickles
pst_in_pkl = ROOT_PATH + '/zd/pst/mb_graph/syn_14_9_bin2_sirt_rot_crop2.pkl'
pre_in_pkl = ROOT_PATH + '/zd/pre/mb_graph/syn_14_9_bin2_sirt_rot_crop2.pkl'
# Input membrane segmentation
pst_in_seg = ROOT_PATH + '/in/zd/bin2/syn_14_9_bin2_crop2_pst_seg.fits'
pre_in_seg = ROOT_PATH + '/in/zd/bin2/syn_14_9_bin2_crop2_pre_seg.fits'

del_coord =1

####### Output data

output_dir = ROOT_PATH + '/zd/cleft/syn_14_9_2'

###### Slices settings file

slices_file = ROOT_PATH + '/zd/cleft/slices_cleft_2.xml'
cont_file = ROOT_PATH + '/zd/cleft/cont_cleft_2.xml'

######### Segmentation
th_den = 0 # None

########################################################################################
# AUXILIAR FUNCTIONALITY
########################################################################################

################# Package import

import os
import vtk
import time
import pyseg as ps
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from pyseg.xml_io import SliceSet
from pyseg.spatial.plane import GroupPlotter, make_plane, make_plane_box, purge_repeat_coords2

########## Global variables

C_PST_LB_1 = 1
C_PST_LB_2 = 2
C_PST_LB_1_2 = 3
C_PRE_LB_1 = 4
C_PRE_LB_2 = 5
C_PRE_LB_1_2 = 6
C_PST_PRE_LB = 7
V_CONT = 1
V_FIL = 2
L_CONT = 3
L_FIL = 4

eps_cont = 0.1 # nm
valid_names = ('cleft_1', 'cleft_2')

# Slice set parameters
group_names = ('pst_1', 'pst_2', 'pst_1_2', 'pre_1', 'pre_2', 'pre_1_2', 'inter')
n_samp, n_sim_f, r_max, r_bord, p_f = 80, 200, 60, 2, 5

######### Functions

# From membrane slices generates cleft volumetric segmentations
def gen_clft_seg(pst_graph, pre_graph, pst_cl_ids, pre_cl_ids, th_den):

    # Initialization
    pre_skel = pre_graph.get_skel()
    pst_skel = pst_graph.get_skel()
    den_shape = pst_graph.get_density().shape
    seg1 = np.zeros(shape=den_shape, dtype=int)
    seg2 = np.zeros(shape=den_shape, dtype=int)
    seg4 = np.zeros(shape=den_shape, dtype=int)
    seg5 = np.zeros(shape=den_shape, dtype=int)
    pst_ids = list()
    pre_ids = list()
    pst_ids2 = list()
    pre_ids2 = list()
    pst_cl = list()
    pre_cl = list()

    # Tomograms labelling
    # Post-synaptic membrane layer 1
    for idx in pst_cl_ids[0]:
        array = pst_graph.get_vertex(idx).get_geometry().get_array_mask(mode=1, th_den=th_den)
        for point in array:
            seg1[point[0], point[1], point[2]] = C_PST_LB_1
    # Post-synaptic membrane layer 2
    for idx in pst_cl_ids[1]:
        array = pst_graph.get_vertex(idx).get_geometry().get_array_mask(mode=1, th_den=th_den)
        for point in array:
            seg2[point[0], point[1], point[2]] = C_PST_LB_2
    # Pre-synaptic membrane layer 1
    for idx in pre_cl_ids[0]:
        array = pre_graph.get_vertex(idx).get_geometry().get_array_mask(mode=1, th_den=th_den)
        for point in array:
            seg4[point[0], point[1], point[2]] = C_PRE_LB_1
    # Pre-synaptic membrane layer 2
    for idx in pre_cl_ids[1]:
        array = pre_graph.get_vertex(idx).get_geometry().get_array_mask(mode=1, th_den=th_den)
        for point in array:
            seg5[point[0], point[1], point[2]] = C_PRE_LB_2
    # Post-synaptic layers 1 and 2 intersection
    seg3 = seg1 * seg2
    s_id = seg3 > 0
    seg1[s_id] = 0
    seg2[s_id] = 0
    seg3[s_id] = C_PST_LB_1_2
    # Pre-synaptic layers 1 and 2 intersection
    seg6 = seg4 * seg5
    s_id = seg6 > 0
    seg4[s_id] = 0
    seg5[s_id] = 0
    seg6[s_id] = C_PRE_LB_1_2
    # Pst- and pre-synaptic intersection
    pst_seg = (seg1 > 0).astype(int) + (seg2 > 0).astype(int) + (seg3 > 0).astype(int)
    pre_seg = (seg4 > 0).astype(int) + (seg5 > 0).astype(int) + (seg6 > 0).astype(int)
    seg7 = pst_seg * pre_seg
    s_id = seg7 > 0
    seg1[s_id] = 0
    seg2[s_id] = 0
    seg3[s_id] = 0
    seg4[s_id] = 0
    seg5[s_id] = 0
    seg6[s_id] = 0
    seg7[s_id] = C_PST_PRE_LB

    # Vertex assignment

    # Inter-layer
    pst_hl = list(pst_clc_ids[0]) + list(pst_clc_ids[1])
    pre_hl = list(pre_clc_ids[0]) + list(pre_clc_ids[1])
    lut_int_pst = np.ones(shape=pst_graph.get_nid(), dtype=bool)
    lut_int_pre = np.ones(shape=pre_graph.get_nid(), dtype=bool)
    for idx in pst_hl:
        point = pst_graph.get_vertex_coords(pst_graph.get_vertex(idx))
        if seg7[int(round(point[0])), int(round(point[1])), int(round(point[2]))] == C_PST_PRE_LB:
            pst_cl.append(pst_skel.GetPoint(idx))
            pst_ids.append(idx)
            pst_ids2.append(C_PST_PRE_LB)
            lut_int_pst[idx] = False
    for idx in pre_hl:
        point = pre_graph.get_vertex_coords(pre_graph.get_vertex(idx))
        if seg7[int(round(point[0])), int(round(point[1])), int(round(point[2]))] == C_PST_PRE_LB:
            pre_cl.append(pre_skel.GetPoint(idx))
            pre_ids.append(idx)
            pre_ids2.append(C_PST_PRE_LB)
            lut_int_pre[idx] = False

    # Post-synaptic layer1
    for idx in pst_cl_ids[0]:
        if lut_int_pst[idx]:
            point = pst_graph.get_vertex_coords(pst_graph.get_vertex(idx))
            if seg3[int(round(point[0])), int(round(point[1])), int(round(point[2]))] == C_PST_LB_1_2:
                pst_cl.append(pst_skel.GetPoint(idx))
                pst_ids.append(idx)
                pst_ids2.append(C_PST_LB_1_2)
            elif seg1[int(round(point[0])), int(round(point[1])), int(round(point[2]))] == C_PST_LB_1:
                pst_cl.append(pst_skel.GetPoint(idx))
                pst_ids.append(idx)
                pst_ids2.append(C_PST_LB_1)
    # Post-synaptic layer2
    for idx in pst_cl_ids[1]:
        if lut_int_pst[idx]:
            point = pst_graph.get_vertex_coords(pst_graph.get_vertex(idx))
            if seg2[int(round(point[0])), int(round(point[1])), int(round(point[2]))] == C_PST_LB_2:
                pst_cl.append(pst_skel.GetPoint(idx))
                pst_ids.append(idx)
                pst_ids2.append(C_PST_LB_2)

    # Pre-synaptic layer1
    for idx in pre_cl_ids[0]:
        if lut_int_pre[idx]:
            point = pre_graph.get_vertex_coords(pre_graph.get_vertex(idx))
            if seg6[int(round(point[0])), int(round(point[1])), int(round(point[2]))] == C_PRE_LB_1_2:
                pre_cl.append(pre_skel.GetPoint(idx))
                pre_ids.append(idx)
                pre_ids2.append(C_PRE_LB_1_2)
            elif seg4[int(round(point[0])), int(round(point[1])), int(round(point[2]))] == C_PRE_LB_1:
                pre_cl.append(pre_skel.GetPoint(idx))
                pre_ids.append(idx)
                pre_ids2.append(C_PRE_LB_1)
    # Pre-synaptic layer2
    for idx in pre_cl_ids[1]:
        if lut_int_pre[idx]:
            point = pre_graph.get_vertex_coords(pre_graph.get_vertex(idx))
            if seg5[int(round(point[0])), int(round(point[1])), int(round(point[2]))] == C_PRE_LB_2:
                pre_cl.append(pre_skel.GetPoint(idx))
                pre_ids.append(idx)
                pre_ids2.append(C_PRE_LB_2)

    return (seg1, seg2, seg3, seg4, seg5, seg6, seg7), [pre_cl, pst_cl], [pre_ids, pst_ids], [pre_ids2, pst_ids2]

# Plotting and storing clouds
def store_plt_clouds(clft_cl, clft_ids2, o_dir):

    fig_count = 1
    plt.figure(fig_count)
    plt.title('Pre-synaptic filaments')
    plt.xlabel('X (nm)')
    plt.ylabel('Y (nm)')
    plt.xlim(box[0], box[2])
    plt.ylim(box[1], box[3])
    color1 = cm.rainbow(np.linspace(0, 1, 3))
    marks = ('o', 's', '^')
    h_lbls = ('1', '2', '1_2')
    cloud_1 = list()
    cloud_2 = list()
    cloud_3 = list()
    for i in range(len(clft_cl[0])):
        idx = clft_ids2[0][i]
        if idx == C_PRE_LB_1:
            cloud_1.append(clft_cl[0][i])
        elif idx == C_PRE_LB_2:
            cloud_2.append(clft_cl[0][i])
        elif idx == C_PRE_LB_1_2:
            cloud_3.append(clft_cl[0][i])
    lines = list()
    lbls = list()
    for i, cloud in enumerate((cloud_1, cloud_2, cloud_3)):
        if len(cloud) > 0:
            cloud = make_plane(np.asarray(cloud, dtype=float), del_coord)
            line = plt.scatter(cloud[:, 0], cloud[:, 1], c=color1[i], marker=marks[i])
            lines.append(line)
            lbls.append(h_lbls[i])
    plt.legend(lines, lbls)
    plt.savefig(o_dir + '/' + 'pre_cloud.png')
    plt.close()

    fig_count += 1
    plt.figure(fig_count)
    plt.title('Post-synaptic filaments')
    plt.xlabel('X (nm)')
    plt.ylabel('Y (nm)')
    plt.xlim(box[0], box[2])
    plt.ylim(box[1], box[3])
    cloud_1 = list()
    cloud_2 = list()
    cloud_3 = list()
    for i in range(len(clft_cl[1])):
        idx = clft_ids2[1][i]
        if idx == C_PST_LB_1:
            cloud_1.append(clft_cl[1][i])
        elif idx == C_PST_LB_2:
            cloud_2.append(clft_cl[1][i])
        elif idx == C_PST_LB_1_2:
            cloud_3.append(clft_cl[1][i])
    lines = list()
    lbls = list()
    for i, cloud in enumerate((cloud_1, cloud_2, cloud_3)):
        if len(cloud) > 0:
            cloud = make_plane(np.asarray(cloud, dtype=float), del_coord)
            line = plt.scatter(cloud[:, 0], cloud[:, 1], c=color1[i], marker=marks[i])
            lines.append(line)
            lbls.append(h_lbls[i])
    plt.legend(lines, lbls)
    plt.savefig(o_dir + '/' + 'pst_cloud.png')
    plt.close()

    fig_count += 1
    plt.figure(fig_count)
    plt.title('Inter-synaptic vertices')
    plt.xlabel('X (nm)')
    plt.ylabel('Y (nm)')
    plt.xlim(box[0], box[2])
    plt.ylim(box[1], box[3])
    cloud_1 = list()
    for i in range(len(clft_cl[0])):
        idx = clft_ids2[0][i]
        if idx == C_PST_PRE_LB:
            cloud_1.append(clft_cl[0][i])
    for i in range(len(clft_cl[1])):
        idx = clft_ids2[1][i]
        if idx == C_PST_PRE_LB:
            cloud_1.append(clft_cl[1][i])
    if len(cloud_1) > 0:
        cloud_1 = make_plane(np.asarray(cloud_1, dtype=float), del_coord)
        plt.scatter(cloud_1[:, 0], cloud_1[:, 1])
    plt.savefig(o_dir + '/' + 'inter_cloud.png')
    plt.close()

# Plotting and storing membrane contact points clouds
def store_plt_cont_clouds(pre_clc, pre_clc_ids, pst_clc, pst_clc_ids, clft_ids, cleft_ids2, o_dir):

    # Initialization
    pst_ids = list()
    pre_ids = list()
    pst_ids2 = list()
    pre_ids2 = list()
    pst_cl = list()
    pre_cl = list()
    pre2_clc = list(pre_clc[0]) + list(pre_clc[1])
    pst2_clc = list(pst_clc[0]) + list(pst_clc[1])
    pre2_clc_ids = list(pre_clc_ids[0]) + list(pre_clc_ids[1])
    pst2_clc_ids = list(pst_clc_ids[0]) + list(pst_clc_ids[1])

    fig_count = 1
    plt.figure(fig_count)
    plt.title('Pre-synaptic contact points')
    plt.xlabel('X (nm)')
    plt.ylabel('Y (nm)')
    plt.xlim(box[0], box[2])
    plt.ylim(box[1], box[3])
    color1 = cm.rainbow(np.linspace(0, 1, 4))
    h_lbls = ('1', '2', '1_2', 'pst_pre')
    marks = ('o', 's', '^', 'p')
    cloud_1, cloud_2, cloud_3, cloud_4 = list(), list(), list(), list()
    for i in range(len(pre2_clc)):
        idx_c = pre2_clc_ids[i]
        try:
            idx = clft_ids[0].index(idx_c)
        except ValueError:
            continue
        lbl = cleft_ids2[0][idx]
        if lbl == C_PRE_LB_1:
            cloud_1.append(pre2_clc[i])
        elif lbl == C_PRE_LB_2:
            cloud_2.append(pre2_clc[i])
        elif lbl == C_PRE_LB_1_2:
            cloud_3.append(pre2_clc[i])
        else:
            cloud_4.append(pre2_clc[i])
        pre_cl.append(pre2_clc[i])
        pre_ids.append(idx_c)
        pre_ids2.append(lbl)
    lines = list()
    lbls = list()
    for i, cloud in enumerate((cloud_1, cloud_2, cloud_3, cloud_4)):
        if len(cloud) > 0:
            cloud = make_plane(np.asarray(cloud, dtype=float), del_coord)
            line = plt.scatter(cloud[:, 0], cloud[:, 1], c=color1[i], marker=marks[i])
            lines.append(line)
            lbls.append(h_lbls[i])
    plt.legend(lines, lbls)
    plt.savefig(o_dir + '/' + 'pre_cont_cloud.png')
    plt.close()

    fig_count += 1
    plt.figure(fig_count)
    plt.title('Pst-synaptic contact points')
    plt.xlabel('X (nm)')
    plt.ylabel('Y (nm)')
    plt.xlim(box[0], box[2])
    plt.ylim(box[1], box[3])
    cloud_1, cloud_2, cloud_3, cloud_4 = list(), list(), list(), list()
    for i in range(len(pst2_clc)):
        idx_c = pst2_clc_ids[i]
        try:
            idx = clft_ids[1].index(idx_c)
        except ValueError:
            continue
        lbl = cleft_ids2[1][idx]
        if lbl == C_PST_LB_1:
            cloud_1.append(pst2_clc[i])
        elif lbl == C_PST_LB_2:
            cloud_2.append(pst2_clc[i])
        elif lbl == C_PST_LB_1_2:
            cloud_3.append(pst2_clc[i])
        else:
            cloud_4.append(pst2_clc[i])
        pst_cl.append(pst2_clc[i])
        pst_ids.append(idx_c)
        pst_ids2.append(lbl)
    lines = list()
    lbls = list()
    for i, cloud in enumerate((cloud_1, cloud_2, cloud_3, cloud_4)):
        if len(cloud) > 0:
            cloud = make_plane(np.asarray(cloud, dtype=float), del_coord)
            line = plt.scatter(cloud[:, 0], cloud[:, 1], c=color1[i], marker=marks[i])
            lines.append(line)
            lbls.append(h_lbls[i])
    plt.legend(lines, lbls)
    plt.savefig(o_dir + '/' + 'pst_cont_cloud.png')
    plt.close()

    return [pre_cl, pst_cl], [pre_ids, pst_ids], [pre_ids2, pst_ids2]

# Generating and storing cleft poly data
def store_vtp(pst_graph, pre_graph, clft_cl, clft_ids, clft_ids2, cont_cl, cont_ids, cont_ids2, o_dir):

    # Initialization
    poly = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    verts = vtk.vtkCellArray()
    lines = vtk.vtkCellArray()
    struct = ps.disperse_io.TypesConverter().numpy_to_vtk(int)
    struct.SetName('struct')
    struct.SetNumberOfComponents(1)
    ids = ps.disperse_io.TypesConverter().numpy_to_vtk(int)
    ids.SetName('ids')
    ids.SetNumberOfComponents(1)
    lut_pre = (-1) * np.ones(shape=pre_graph.get_nid(), dtype=int)
    lut_pst = (-1) * np.ones(shape=pst_graph.get_nid(), dtype=int)
    lut_pre2 = (-1) * np.ones(shape=pre_graph.get_nid(), dtype=int)
    lut_pst2 = (-1) * np.ones(shape=pst_graph.get_nid(), dtype=int)
    lut_cont_pre = (-1) * np.ones(shape=len(cont_ids[0]), dtype=int)
    lut_cont_pst = (-1) * np.ones(shape=len(cont_ids[1]), dtype=int)
    lut_cont_pre2 = (-1) * np.ones(shape=len(cont_ids[0]), dtype=int)
    lut_cont_pst2 = (-1) * np.ones(shape=len(cont_ids[1]), dtype=int)
    point_id = 0

    # Vertices
    # Filaments
    # Pre-synaptic
    for (cl, idx, ids2) in zip(clft_cl[0], clft_ids[0], clft_ids2[0]):
        points.InsertPoint(point_id, cl[0], cl[1], cl[2])
        lut_pre[idx] = point_id
        verts.InsertNextCell(1)
        verts.InsertCellPoint(point_id)
        struct.InsertNextTuple((V_FIL,))
        ids.InsertNextTuple((ids2,))
        lut_pre2[idx] = ids2
        point_id += 1
    # Pst-synaptic
    for (cl, idx, ids2) in zip(clft_cl[1], clft_ids[1], clft_ids2[1]):
        points.InsertPoint(point_id, cl[0], cl[1], cl[2])
        lut_pst[idx] = point_id
        verts.InsertNextCell(1)
        verts.InsertCellPoint(point_id)
        struct.InsertNextTuple((V_FIL,))
        ids.InsertNextTuple((ids2,))
        lut_pst2[idx] = ids2
        point_id += 1
    # Contact points
    # Pre-synaptic
    cont = 0
    for (cl, idx, ids2) in zip(cont_cl[0], cont_ids[0], cont_ids2[0]):
        points.InsertPoint(point_id, cl[0], cl[1], cl[2])
        # lut_pre[idx] = point_id
        verts.InsertNextCell(1)
        verts.InsertCellPoint(point_id)
        struct.InsertNextTuple((V_CONT,))
        ids.InsertNextTuple((ids2,))
        lut_pre2[idx] = ids2
        lut_cont_pre[cont] = idx
        lut_cont_pre2[cont] = point_id
        cont += 1
        point_id += 1
    # Pst-synaptic
    cont = 0
    for (cl, idx, ids2) in zip(cont_cl[1], cont_ids[1], cont_ids2[1]):
        points.InsertPoint(point_id, cl[0], cl[1], cl[2])
        # lut_pst[idx] = point_id
        verts.InsertNextCell(1)
        verts.InsertCellPoint(point_id)
        struct.InsertNextTuple((V_CONT,))
        ids.InsertNextTuple((ids2,))
        lut_pst2[idx] = ids2
        lut_cont_pst[cont] = idx
        lut_cont_pst2[cont] = point_id
        cont += 1
        point_id += 1

    # Lines
    # Filaments
    # Pre-synaptic
    for edge in pre_graph.get_edges_list():
        s_id = lut_pre[edge.get_source_id()]
        t_id = lut_pre[edge.get_target_id()]
        if (s_id > -1) and (t_id > -1):
            lines.InsertNextCell(2)
            lines.InsertCellPoint(s_id)
            lines.InsertCellPoint(t_id)
            struct.InsertNextTuple((L_FIL,))
            s_idx2, t_idx2 = lut_pre2[s_id], lut_pre2[t_id]
            if s_idx2 == t_idx2:
                ids.InsertNextTuple((s_idx2,))
            else:
                ids.InsertNextTuple((-1,))
    # Pst-synaptic
    for edge in pst_graph.get_edges_list():
        s_id = lut_pst[edge.get_source_id()]
        t_id = lut_pst[edge.get_target_id()]
        if (s_id > -1) and (t_id > -1):
            lines.InsertNextCell(2)
            lines.InsertCellPoint(s_id)
            lines.InsertCellPoint(t_id)
            struct.InsertNextTuple((L_FIL,))
            s_idx2, t_idx2 = lut_pst2[s_id], lut_pst2[t_id]
            if s_idx2 == t_idx2:
                ids.InsertNextTuple((s_idx2,))
            else:
                ids.InsertNextTuple((-1,))
    # Contact points
    # Pre-synaptic
    for (idx, p_id) in zip(lut_cont_pre, lut_cont_pre2):
        lines.InsertNextCell(2)
        lines.InsertCellPoint(p_id)
        lines.InsertCellPoint(lut_pre[idx])
        struct.InsertNextTuple((L_CONT,))
        ids.InsertNextTuple((lut_pre2[idx],))
    # Pst-synaptic
    for (idx, p_id) in zip(lut_cont_pst, lut_cont_pst2):
        lines.InsertNextCell(2)
        lines.InsertCellPoint(p_id)
        lines.InsertCellPoint(lut_pst[idx])
        struct.InsertNextTuple((L_CONT,))
        ids.InsertNextTuple((lut_pst2[idx],))

    # Building the polydata
    poly.SetPoints(points)
    poly.SetVerts(verts)
    poly.SetLines(lines)
    poly.GetCellData().AddArray(struct)
    poly.GetCellData().AddArray(ids)

    # Storing
    ps.disperse_io.save_vtp(poly, o_dir + '/cleft.vtp')

########################################################################################
# MAIN ROUTINE
########################################################################################

########## Print initial message

print('Spatial analysis for cleft segmentation.')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('Options:')
print('\tPre-synaptic pickle file(s): ' + pre_in_pkl)
print('\tPost-synaptic pickle file(s): ' + pst_in_pkl)
print('\tPre-synaptic segmentation file(s): ' + pre_in_seg)
print('\tPost-synaptic segmentation file(s): ' + pst_in_seg)
print('\tDimensions to delete: ' + str(del_coord))
print('\tOutput directory: ' + str(output_dir))
print('\tSlices files: ' + slices_file + ', ' + cont_file)
print('')

######### Process

print('Loading the input graphs:')
path, fname = os.path.split(pre_in_pkl)
pre_stem_name, _ = os.path.splitext(fname)
path, fname = os.path.split(pst_in_pkl)
pst_stem_name, _ = os.path.splitext(fname)

print('\tUnpicking graph ' + pre_in_pkl)
pre_graph = ps.factory.unpickle_obj(pre_in_pkl)
pre_seg = ps.disperse_io.load_tomo(pre_in_seg)

print('\tUnpicking graph ' + pst_in_pkl)
pst_graph = ps.factory.unpickle_obj(pst_in_pkl)
pst_seg = ps.disperse_io.load_tomo(pst_in_seg)

print('\tLoading XML file with the slices...')
v_slices = SliceSet(slices_file)
c_slices = SliceSet(cont_file)
box = make_plane_box(pst_graph.compute_bbox(), coord=del_coord) * pst_graph.get_resolution()

print('\tSlices loop:')
pre_cl, pst_cl, pre_cl_ids, pst_cl_ids = list(), list(), list(), list()
pre_clc, pst_clc, pre_clc_ids, pst_clc_ids = list(), list(), list(), list()
for valid_name in valid_names:

    print('\t\tLooking for slice named ' + valid_name)
    sl, cl = None, None
    for (v_slice, c_slice) in zip(v_slices.get_slices_list(), c_slices.get_slices_list()):
        if v_slice.get_name() == valid_name:
            sl = v_slice
        if c_slice.get_cont():
            if c_slice.get_name() == valid_name:
                cl = c_slice
                print('\t\t\t-Slice found.')
                break

    if (sl is not None) and (cl is not None):

        print('\t\tProcessing slice ' + sl.get_name() + ':')
        print('\t\t\t-Euclidean distance: (' + sl.get_eu_dst_sign() + ')[' \
              + str(sl.get_eu_dst_low()) + ', ' + str(sl.get_eu_dst_high()) + '] nm')
        print('\t\t\t-Geodesic distance: (' + sl.get_geo_dst_sign() + ')[' \
              + str(sl.get_geo_dst_low()) + ', ' + str(sl.get_geo_dst_high()) + '] nm')
        print('\t\t\t-Geodesic length: (' + sl.get_geo_len_sign() + ')[' \
              + str(sl.get_geo_len_low()) + ', ' + str(sl.get_geo_len_high()) + '] nm')
        print('\t\t\t-Sinuosity: (' + sl.get_sin_sign() + ')[' \
              + str(sl.get_sin_low()) + ', ' + str(sl.get_sin_high()) + '] nm')
        print('\t\t\t-Cluster number of points: (' + sl.get_cnv_sign() + ')[' \
              + str(sl.get_cnv_low()) + ', ' + str(sl.get_cnv_high()) + ']')
        print('\t\t\tGetting membrane slice filaments...')
        pre_cloud, h_pre_cloud_ids = pre_graph.get_cloud_mb_slice_fils(sl)
        pst_cloud, h_pst_cloud_ids = pst_graph.get_cloud_mb_slice_fils(sl)
        pre_cloud, pre_cloud_ids = purge_repeat_coords2(np.asarray(pre_cloud, dtype=float) * pre_graph.get_resolution(),
                                                        h_pre_cloud_ids, eps_cont)
        pst_cloud, pst_cloud_ids = purge_repeat_coords2(np.asarray(pst_cloud, dtype=float) * pst_graph.get_resolution(),
                                                        h_pst_cloud_ids, eps_cont)
        pre_cl.append(pre_cloud)
        pst_cl.append(pst_cloud)
        pre_cl_ids.append(pre_cloud_ids)
        pst_cl_ids.append(pst_cloud_ids)
        print('\t\t\t-Vertices found (pre and post): ' + str(pre_cloud.shape[0]) + ', ' + str(pst_cloud.shape[0]))

        print('\t\tProcessing slice for contact points ' + cl.get_name() + ':')
        print('\t\t\t-Euclidean distance: (' + cl.get_eu_dst_sign() + ')[' \
              + str(cl.get_eu_dst_low()) + ', ' + str(cl.get_eu_dst_high()) + '] nm')
        print('\t\t\t-Geodesic distance: (' + cl.get_geo_dst_sign() + ')[' \
              + str(cl.get_geo_dst_low()) + ', ' + str(cl.get_geo_dst_high()) + '] nm')
        print('\t\t\t-Geodesic length: (' + cl.get_geo_len_sign() + ')[' \
              + str(cl.get_geo_len_low()) + ', ' + str(cl.get_geo_len_high()) + '] nm')
        print('\t\t\t-Sinuosity: (' + cl.get_sin_sign() + ')[' \
              + str(cl.get_sin_low()) + ', ' + str(cl.get_sin_high()) + '] nm')
        print('\t\t\t-Cluster number of points: (' + cl.get_cnv_sign() + ')[' \
              + str(cl.get_cnv_low()) + ', ' + str(cl.get_cnv_high()) + ']')
        print('\t\t\tGetting membrane slice filaments...')
        pre_cloud_cont, h_pre_cloud_ids_cont = pre_graph.get_cloud_mb_slice(cl, cont_mode=True)
        pst_cloud_cont, h_pst_cloud_ids_cont = pst_graph.get_cloud_mb_slice(cl, cont_mode=True)
        pre_cloud_cont, pre_cloud_ids_cont = purge_repeat_coords2(np.asarray(pre_cloud_cont, dtype=float) * pre_graph.get_resolution(),
                                                                  h_pre_cloud_ids_cont, eps_cont)
        pst_cloud_cont, pst_cloud_ids_cont = purge_repeat_coords2(np.asarray(pst_cloud_cont, dtype=float) * pst_graph.get_resolution(),
                                                                  h_pst_cloud_ids_cont, eps_cont)
        pre_clc.append(pre_cloud_cont)
        pst_clc.append(pst_cloud_cont)
        pre_clc_ids.append(pre_cloud_ids_cont)
        pst_clc_ids.append(pst_cloud_ids_cont)
        print('\t\t\t-Contact points found (pre and post): ' + str(pre_cloud_cont.shape[0]) + \
              ', ' + str(pst_cloud_cont.shape[0]))


print('\tGenerating segmentation...')
clft_segs, clft_cl, clft_ids, clft_ids2 = gen_clft_seg(pst_graph, pre_graph, pst_cl_ids, pre_cl_ids, th_den)

print('\tStoring clouds...')
store_plt_clouds(clft_cl, clft_ids2, output_dir)
cont_cl, cont_ids, cont_ids2 = store_plt_cont_clouds(pre_clc, pre_clc_ids, pst_clc, pst_clc_ids,
                                                     clft_ids, clft_ids2, output_dir)

print('\tStoring poly data...')
cont_cl_ur = list()
cont_cl_ur.append(np.asarray(cont_cl[0], dtype=float) / pst_graph.get_resolution())
cont_cl_ur.append(np.asarray(cont_cl[1], dtype=float) / pst_graph.get_resolution())
store_vtp(pst_graph, pre_graph, clft_cl, clft_ids, clft_ids2, cont_cl_ur, cont_ids, cont_ids2, output_dir)

print('\tStoring segmentation...')
for i, seg in enumerate(clft_segs):
    output_seg = output_dir + '/' + 'cleft_seg_' + str(i+1) + '.vti'
    ps.disperse_io.save_numpy(seg, output_seg)

print('\tStoring the set of clouds...')
set_clouds = ps.spatial.SetCloudsP(box, n_samp, n_sim_f, r_max, r_bord, p_f)
cloud_1, cloud_2, cloud_3, cloud_4, cloud_5, cloud_6, cloud_7 = list(), list(), list(), list(), list(), list(), list()
h_clft_cl = list(make_plane(np.asarray(clft_cl[0], dtype=float), del_coord)) + \
            list(make_plane(np.asarray(clft_cl[1], dtype=float), del_coord))
h_clft_ids2 = list(clft_ids2[0]) + list(clft_ids2[1])
for (cl, ids2) in zip(h_clft_cl, h_clft_ids2):
    if ids2 == C_PST_LB_1:
        cloud_1.append(cl)
    elif ids2 == C_PST_LB_2:
        cloud_2.append(cl)
    elif ids2 == C_PST_LB_1_2:
        cloud_3.append(cl)
    elif ids2 == C_PRE_LB_1:
        cloud_4.append(cl)
    elif ids2 == C_PRE_LB_2:
        cloud_5.append(cl)
    elif ids2 == C_PRE_LB_1_2:
        cloud_6.append(cl)
    elif ids2 == C_PST_PRE_LB:
        cloud_7.append(cl)
set_clouds.insert_cloud(np.asarray(cloud_1, dtype=float), group_names[0])
set_clouds.insert_cloud(np.asarray(cloud_2, dtype=float), group_names[1])
set_clouds.insert_cloud(np.asarray(cloud_3, dtype=float), group_names[2])
set_clouds.insert_cloud(np.asarray(cloud_4, dtype=float), group_names[3])
set_clouds.insert_cloud(np.asarray(cloud_5, dtype=float), group_names[4])
set_clouds.insert_cloud(np.asarray(cloud_6, dtype=float), group_names[5])
set_clouds.insert_cloud(np.asarray(cloud_7, dtype=float), group_names[6])
output_pkl = output_dir + '/' + 'clouds_set.pkl'
set_clouds.pickle(output_pkl)
set_clouds_cont = ps.spatial.SetCloudsP(box, n_samp, n_sim_f, r_max, r_bord, p_f)
cloud_1, cloud_2, cloud_3, cloud_4, cloud_5, cloud_6, cloud_7 = list(), list(), list(), list(), list(), list(), list()
h_clft_cl = list(make_plane(np.asarray(cont_cl[0], dtype=float), del_coord)) + \
            list(make_plane(np.asarray(cont_cl[1], dtype=float), del_coord))
h_clft_ids2 = list(cont_ids2[0]) + list(cont_ids2[1])
for (cl, ids2) in zip(h_clft_cl, h_clft_ids2):
    if ids2 == C_PST_LB_1:
        cloud_1.append(cl)
    elif ids2 == C_PST_LB_2:
        cloud_2.append(cl)
    elif ids2 == C_PST_LB_1_2:
        cloud_3.append(cl)
    elif ids2 == C_PRE_LB_1:
        cloud_4.append(cl)
    elif ids2 == C_PRE_LB_2:
        cloud_5.append(cl)
    elif ids2 == C_PRE_LB_1_2:
        cloud_6.append(cl)
    elif ids2 == C_PST_PRE_LB:
        cloud_7.append(cl)
set_clouds_cont.insert_cloud(np.asarray(cloud_1, dtype=float), group_names[0])
set_clouds_cont.insert_cloud(np.asarray(cloud_2, dtype=float), group_names[1])
set_clouds_cont.insert_cloud(np.asarray(cloud_3, dtype=float), group_names[2])
set_clouds_cont.insert_cloud(np.asarray(cloud_4, dtype=float), group_names[3])
set_clouds_cont.insert_cloud(np.asarray(cloud_5, dtype=float), group_names[4])
set_clouds_cont.insert_cloud(np.asarray(cloud_6, dtype=float), group_names[5])
set_clouds_cont.insert_cloud(np.asarray(cloud_7, dtype=float), group_names[6])
output_pkl = output_dir + '/' + 'cont_clouds_set.pkl'
set_clouds_cont.pickle(output_pkl)



