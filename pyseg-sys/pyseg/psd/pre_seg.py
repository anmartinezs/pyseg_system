"""

    Script for pre-synaptic active zone segmentation

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
pre_in_pkl = ROOT_PATH + '/uli/pre/mb_graph/syn_11_1_bin2_sirt_rot_crop2.pkl'
# Input membrane segmentation
pre_in_seg = ROOT_PATH + '/in/uli/bin2/syn_11_1_bin2_crop2_pre_seg.fits'

del_coord =1

####### Output data

output_dir = ROOT_PATH + '/zd/pre_seg/syn_11_1_2'

###### Slices settings file

slices_file = ROOT_PATH + '/zd/pre_seg/slices_pre_2.xml'
cont_file = ROOT_PATH + '/zd/pre_seg/cont_pre_2.xml'

######### Segmentation
th_den = 0 # None

########################################################################################
# AUXILIARY FUNCTIONALITY
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
from pyseg.spatial.plane import make_plane, make_plane_box, purge_repeat_coords2

########## Global variables

C_PRE_LB_1 = 1
V_CONT = 1
V_FIL = 2
L_CONT = 3
L_FIL = 4

eps_cont = 0.1 # nm

# Slice set parameters
group_names = ('pre_1',)
n_samp, n_sim_f, r_max, r_bord, p_f = 80, 200, 60, 2, 5

######### Functions

# From membrane slices generates cleft volumetric segmentations
def gen_pre_seg(pre_graph, pre_cl_ids, th_den):

    # Initialization
    pre_skel = pre_graph.get_skel()
    den_shape = pre_graph.get_density().shape
    seg1 = np.zeros(shape=den_shape, dtype=np.int)
    pre_ids = list()
    pre_ids2 = list()
    pre_cl = list()

    # Tomograms labelling
    # Post-synaptic membrane layer 1
    for idx in pre_cl_ids:
        array = pre_graph.get_vertex(idx).get_geometry().get_array_mask(mode=1, th_den=th_den)
        for point in array:
            seg1[point[0], point[1], point[2]] = C_PRE_LB_1

    # Vertex assignment

    # Post-synaptic layer1
    for idx in pre_cl_ids:
        point = pre_graph.get_vertex_coords(pre_graph.get_vertex(idx))
        if seg1[int(round(point[0])), int(round(point[1])), int(round(point[2]))] == C_PRE_LB_1:
            pre_cl.append(pre_skel.GetPoint(idx))
            pre_ids.append(idx)
            pre_ids2.append(C_PRE_LB_1)

    return seg1, pre_cl, pre_ids, pre_ids2

# Plotting and storing clouds
def store_plt_clouds(pre_cl, pre_ids2, o_dir):

    fig_count = 1
    plt.figure(fig_count)
    plt.title('Pre-synaptic filaments')
    plt.xlabel('X (nm)')
    plt.ylabel('Y (nm)')
    plt.xlim(box[0], box[2])
    plt.ylim(box[1], box[3])
    cloud_1 = list()
    color1 = cm.rainbow(np.linspace(0, 1, 1))
    marks = ('o')
    h_lbls = ('1', '2', '1_2')
    for i in range(len(pre_cl)):
        idx = pre_ids2[i]
        if idx == C_PRE_LB_1:
            cloud_1.append(pre_cl[i])
    lines = list()
    lbls = list()
    for i, cloud in enumerate((cloud_1,)):
        if len(cloud) > 0:
            cloud = make_plane(np.asarray(cloud, dtype=np.float), del_coord)
            line = plt.scatter(cloud[:, 0], cloud[:, 1], c=color1[i], marker=marks[i])
            lines.append(line)
            lbls.append(h_lbls[i])
    plt.legend(lines, lbls)
    plt.savefig(o_dir + '/' + 'pst_cloud.png')
    plt.close()

# Plotting and storing membrane contact points clouds
def store_plt_cont_clouds(pre_clc, pre_clc_ids, pre_ids, pre_ids2, o_dir):

    # Initialization
    pre2_ids = list()
    pre2_ids2 = list()
    pre_cl = list()

    fig_count = 1
    plt.figure(fig_count)
    plt.title('Pst-synaptic contact points')
    plt.xlabel('X (nm)')
    plt.ylabel('Y (nm)')
    plt.xlim(box[0], box[2])
    plt.ylim(box[1], box[3])
    color1 = cm.rainbow(np.linspace(0, 1, 1))
    marks = ('o')
    h_lbls = ('1')
    cloud_1 = list()
    for i in range(len(pre_clc)):
        idx_c = pre_clc_ids[i]
        try:
            idx = pre_ids.index(idx_c)
        except ValueError:
            continue
        lbl = pre_ids2[idx]
        if lbl == C_PRE_LB_1:
            cloud_1.append(pre_clc[i])
        pre_cl.append(pre_clc[i])
        pre2_ids.append(idx_c)
        pre2_ids2.append(lbl)
    lines = list()
    lbls = list()
    for i, cloud in enumerate((cloud_1,)):
        if len(cloud) > 0:
            cloud = make_plane(np.asarray(cloud, dtype=np.float), del_coord)
            line = plt.scatter(cloud[:, 0], cloud[:, 1], c=color1[i], marker=marks[i])
            lines.append(line)
            lbls.append(h_lbls[i])
    plt.legend(lines, lbls)
    plt.savefig(o_dir + '/' + 'pst_cont_cloud.png')
    plt.close()

    return pre_cl, pre2_ids, pre2_ids2

# Generating and storing cleft poly data
def store_vtp(pre_graph, pre_cl, pre_ids, pre_ids2, cont_cl, cont_ids, cont_ids2, o_dir):

    # Initialization
    poly = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    verts = vtk.vtkCellArray()
    lines = vtk.vtkCellArray()
    struct = ps.disperse_io.TypesConverter().numpy_to_vtk(np.int)
    struct.SetName('struct')
    struct.SetNumberOfComponents(1)
    ids = ps.disperse_io.TypesConverter().numpy_to_vtk(np.int)
    ids.SetName('ids')
    ids.SetNumberOfComponents(1)
    lut_pre = (-1) * np.ones(shape=pre_graph.get_nid(), dtype=np.int)
    lut_pre2 = (-1) * np.ones(shape=pre_graph.get_nid(), dtype=np.int)
    lut_cont_pre = (-1) * np.ones(shape=len(cont_ids), dtype=np.int)
    lut_cont_pre2 = (-1) * np.ones(shape=len(cont_ids), dtype=np.int)
    point_id = 0

    # Vertices
    # Filaments
    # Pre-synaptic
    for (cl, idx, ids2) in zip(pre_cl, pre_ids, pre_ids2):
        points.InsertPoint(point_id, cl[0], cl[1], cl[2])
        lut_pre[idx] = point_id
        verts.InsertNextCell(1)
        verts.InsertCellPoint(point_id)
        struct.InsertNextTuple((V_FIL,))
        ids.InsertNextTuple((ids2,))
        lut_pre2[idx] = ids2
        point_id += 1
    # Contact points
    # Pre-synaptic
    cont = 0
    for (cl, idx, ids2) in zip(cont_cl, cont_ids, cont_ids2):
        points.InsertPoint(point_id, cl[0], cl[1], cl[2])
        verts.InsertNextCell(1)
        verts.InsertCellPoint(point_id)
        struct.InsertNextTuple((V_CONT,))
        ids.InsertNextTuple((ids2,))
        lut_pre2[idx] = ids2
        lut_cont_pre[cont] = idx
        lut_cont_pre2[cont] = point_id
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
    # Contact points
    # Pre-synaptic
    for (idx, p_id) in zip(lut_cont_pre, lut_cont_pre2):
        lines.InsertNextCell(2)
        lines.InsertCellPoint(p_id)
        lines.InsertCellPoint(lut_pre[idx])
        struct.InsertNextTuple((L_CONT,))
        ids.InsertNextTuple((lut_pre2[idx],))

    # Building the polydata
    poly.SetPoints(points)
    poly.SetVerts(verts)
    poly.SetLines(lines)
    poly.GetCellData().AddArray(struct)
    poly.GetCellData().AddArray(ids)

    # Storing
    ps.disperse_io.save_vtp(poly, o_dir + '/pre.vtp')

########################################################################################
# MAIN ROUTINE
########################################################################################

########## Print initial message

print 'Spatial analysis for PSD segmentation.'
print '\tAuthor: ' + __author__
print '\tDate: ' + time.strftime("%c") + '\n'
print 'Options:'
print '\tPre-synaptic pickle file(s): ' + pre_in_pkl
print '\tPre-synaptic segmentation file(s): ' + pre_in_seg
print '\tDimensions to delete: ' + str(del_coord)
print '\tOutput directory: ' + str(output_dir)
print '\tSlices files: ' + slices_file + ', ' + cont_file
print ''

######### Process

print 'Loading the input graphs:'
path, fname = os.path.split(pre_in_pkl)
pst_stem_name, _ = os.path.splitext(fname)

print '\tUnpicking graph ' + pre_in_pkl
pre_graph = ps.factory.unpickle_obj(pre_in_pkl)
pre_seg = ps.disperse_io.load_tomo(pre_in_seg)

print '\tLoading XML file with the slices...'
v_slices = SliceSet(slices_file)
c_slices = SliceSet(cont_file)
box = make_plane_box(pre_graph.compute_bbox(), coord=del_coord) * pre_graph.get_resolution()

print '\tSlices loop:'
sl = v_slices.get_slices_list()[0]
cl = c_slices.get_slices_list()[0]
print '\t\tLooking for slice named ' + sl.get_name()

print '\t\tProcessing slice ' + sl.get_name() + ':'
print '\t\t\t-Euclidean distance: (' + sl.get_eu_dst_sign() + ')[' \
              + str(sl.get_eu_dst_low()) + ', ' + str(sl.get_eu_dst_high()) + '] nm'
print '\t\t\t-Geodesic distance: (' + sl.get_geo_dst_sign() + ')[' \
              + str(sl.get_geo_dst_low()) + ', ' + str(sl.get_geo_dst_high()) + '] nm'
print '\t\t\t-Geodesic length: (' + sl.get_geo_len_sign() + ')[' \
              + str(sl.get_geo_len_low()) + ', ' + str(sl.get_geo_len_high()) + '] nm'
print '\t\t\t-Sinuosity: (' + sl.get_sin_sign() + ')[' \
              + str(sl.get_sin_low()) + ', ' + str(sl.get_sin_high()) + '] nm'
print '\t\t\t-Cluster number of points: (' + sl.get_cnv_sign() + ')[' \
              + str(sl.get_cnv_low()) + ', ' + str(sl.get_cnv_high()) + ']'
print '\t\t\tGetting membrane slice filaments...'
pre_cloud, h_pre_cloud_ids = pre_graph.get_cloud_mb_slice_fils(sl)
pre_cloud, pre_cloud_ids = purge_repeat_coords2(np.asarray(pre_cloud, dtype=np.float) * pre_graph.get_resolution(),
                                                        h_pre_cloud_ids, eps_cont)
print '\t\t\t-Vertices found: ' + str(pre_cloud.shape[0])

print '\t\tProcessing slice for contact points ' + cl.get_name() + ':'
print '\t\t\t-Euclidean distance: (' + cl.get_eu_dst_sign() + ')[' \
              + str(cl.get_eu_dst_low()) + ', ' + str(cl.get_eu_dst_high()) + '] nm'
print '\t\t\t-Geodesic distance: (' + cl.get_geo_dst_sign() + ')[' \
              + str(cl.get_geo_dst_low()) + ', ' + str(cl.get_geo_dst_high()) + '] nm'
print '\t\t\t-Geodesic length: (' + cl.get_geo_len_sign() + ')[' \
              + str(cl.get_geo_len_low()) + ', ' + str(cl.get_geo_len_high()) + '] nm'
print '\t\t\t-Sinuosity: (' + cl.get_sin_sign() + ')[' \
              + str(cl.get_sin_low()) + ', ' + str(cl.get_sin_high()) + '] nm'
print '\t\t\t-Cluster number of points: (' + cl.get_cnv_sign() + ')[' \
              + str(cl.get_cnv_low()) + ', ' + str(cl.get_cnv_high()) + ']'
print '\t\t\tGetting membrane slice filaments...'
pre_cloud_cont, h_pre_cloud_ids_cont = pre_graph.get_cloud_mb_slice(cl, cont_mode=True)
pre_cloud_cont, pre_cloud_ids_cont = purge_repeat_coords2(np.asarray(pre_cloud_cont, dtype=np.float) * pre_graph.get_resolution(),
                                                                  h_pre_cloud_ids_cont, eps_cont)
print '\t\t\t-Contact points found: '  + str(pre_cloud_cont.shape[0])


print '\tGenerating segmentation...'
pa_seg, pa_cl, pa_ids, pa_ids2 = gen_pre_seg(pre_graph, pre_cloud_ids, th_den)

print '\tStoring clouds...'
store_plt_clouds(pa_cl, pa_ids2, output_dir)
cont_cl, cont_ids, cont_ids2 = store_plt_cont_clouds(pre_cloud_cont, pre_cloud_ids_cont, pa_ids, pa_ids2, output_dir)

print '\tStoring poly data...'
cont_cl = np.asarray(cont_cl, dtype=np.float) / pre_graph.get_resolution()
store_vtp(pre_graph, pa_cl, pa_ids, pa_ids2, cont_cl, cont_ids, cont_ids2, output_dir)

print '\tStoring segmentation...'
output_seg = output_dir + '/' + 'pre_seg_1.vti'
ps.disperse_io.save_numpy(pa_seg, output_seg)

print '\tStoring the set of clouds...'
set_clouds = ps.spatial.SetCloudsP(box, n_samp, n_sim_f, r_max, r_bord, p_f)
cloud_1 = list()
h_pa_cl = list(make_plane(np.asarray(pa_cl, dtype=np.float), del_coord))
h_pa_ids2 = list(pa_ids2)
for (cl, ids2) in zip(h_pa_cl, h_pa_ids2):
    if ids2 == C_PRE_LB_1:
        cloud_1.append(cl)
set_clouds.insert_cloud(np.asarray(cloud_1, dtype=np.float), group_names[0])
output_pkl = output_dir + '/' + 'clouds_set.pkl'
set_clouds.pickle(output_pkl)



