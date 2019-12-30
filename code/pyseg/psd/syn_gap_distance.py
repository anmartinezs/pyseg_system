"""

    Script for analyzing synapse gap distance

    Input:  - A STAR file with the synapse segmentations


    Output: - Synapse gap length distribution

"""

import vtk
import numpy as np
import scipy as sp

__author__ = 'Antonio Martinez-Sanchez'


###### Global variables

MB_PST, MB_PRE = 1, 2
START_IDX, END_STR = 4, '_bin2'
PER_LOW, PER_HIGH = 5, 95 # %

########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '/fs/pool/pool-lucic2/antonio/workspace/psd_an/'

# Input STAR file
in_star = ROOT_PATH + '/in/syn_seg_all_2.star'

####### Output data

out_dir = ROOT_PATH + '/ex/syn/gap_dst/run1'

###### Parameters

sg_res = 0.684
sg_sg = 0.5
sg_th = 0.5

########################################################################################
# MAIN ROUTINE
########################################################################################

# Global Functions

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
    if isinstance(tomo, np.ndarray):
        tomo_vtk = ps.disperse_io.numpy_to_vti(tomo)
    else:
        tomo_vtk = tomo
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

# Get synapse label from segmentation full path
# in_seg: full path to synapse segmentation file
def syn_lbl(in_seg):
    stem = os.path.splitext(os.path.split(in_seg)[1])[0]
    end_idx = stem.index(END_STR)
    return stem[START_IDX:end_idx]

# Measures the closest distance for every point pair between two surfaces
# Only point whose normals have a negative dot-product (normals opposed) are processed
def syn_gap_dsts(pst_surf, pre_surf):

    # Initialization
    normals_pst, normals_pre = pst_surf.GetPointData().GetNormals(), pre_surf.GetPointData().GetNormals()
    locator_pst, locator_pre = vtk.vtkPointLocator(), vtk.vtkPointLocator()
    locator_pst.SetDataSet(pst_surf)
    locator_pre.SetDataSet(pre_surf)
    dsts = list()

    # From to pst to pre
    for i in xrange(pst_surf.GetNumberOfPoints()):
        pst_pt, pst_n = pst_surf.GetPoint(i), normals_pst.GetTuple(i)
        pre_pt_id = locator_pre.FindClosestPoint(pst_pt)
        pre_pt, pre_n = pre_surf.GetPoint(pre_pt_id), normals_pre.GetTuple(pre_pt_id)
        if np.dot(np.asarray(pst_n), np.asarray(pre_n)) < 0:
            hold = np.asarray(pst_pt) - np.asarray(pre_pt)
            dsts.append(np.sqrt((hold*hold).sum()))

    # From to pre to pst
    for i in xrange(pre_surf.GetNumberOfPoints()):
        pre_pt, pre_n = pre_surf.GetPoint(i), normals_pre.GetTuple(i)
        pst_pt_id = locator_pst.FindClosestPoint(pre_pt)
        pst_pt, pst_n = pst_surf.GetPoint(pst_pt_id), normals_pst.GetTuple(pst_pt_id)
        if np.dot(np.asarray(pre_n), np.asarray(pst_n)) < 0:
            hold = np.asarray(pre_pt) - np.asarray(pst_pt)
            dsts.append(np.sqrt((hold*hold).sum()))

    return np.asarray(dsts, dtype=np.float32)

################# Package import

import os
import sys
import time
import pyseg as ps
import matplotlib.pyplot as plt

try:
    import cPickle as pickle
except ImportError:
    import pickle


########## Global variables

########## Print initial message

print 'Synapse gap distance analysis.'
print '\tAuthor: ' + __author__
print '\tDate: ' + time.strftime("%c") + '\n'
print 'Options:'
print '\tInput STAR file with segmentations: ' + in_star
print '\tOutput directory: ' + out_dir
print '\tPre-processing segmentation options:'
print '\t\t-Pixel size: ' + str(sg_res)
print '\t\t-Gaussian smoothing sigma: ' + str(sg_sg)
print '\t\t-Iso-surface threshold: ' + str(sg_th)

######### Process

print 'Main Routine: '

print '\tLoading input STAR file...'
star = ps.sub.Star()
try:
    star.load(in_star)
except ps.pexceptions.PySegInputError as e:
    print 'ERROR: input STAR file could not be loaded because of "' + e.get_message() + '"'
    print 'Terminated. (' + time.strftime("%c") + ')'
    sys.exit(-1)
in_segs = star.get_column_data('_psSegImage')
if in_segs is None:
    print 'ERROR: input STAR does not contain any synapse segmentation to process.'
    print 'Terminated. (' + time.strftime("%c") + ')'
    sys.exit(-1)

print '\tMain loop:'
dst_mns, dst_stds, dst_lbls = list(), list(), list()
dst_per_lows, dst_per_highs = list(), list()
for in_seg in in_segs:

    print '\t\tLoading file: ' + in_seg
    seg = ps.disperse_io.load_tomo(in_seg)
    pst_seg = sp.ndimage.filters.gaussian_filter(np.asarray(seg==MB_PST, dtype=np.float32), 3.0)
    pre_seg = sp.ndimage.filters.gaussian_filter(np.asarray(seg==MB_PRE, dtype=np.float32), 3.0)

    print '\t\tGenerating membrane surfaces...'
    pst_surf = iso_surface(pst_seg, sg_th, closed=True, normals='outwards')
    pre_surf = iso_surface(pre_seg, sg_th, closed=True, normals='outwards')

    print '\t\tComputing distances...'
    dsts = syn_gap_dsts(pst_surf, pre_surf) * sg_res
    try:
        lbl = syn_lbl(in_seg)
    except ValueError:
        print 'WARNING: segmentation file uncorrectly formatted, no synapse label can be extracted.'
        continue
    dst_lbls.append(lbl)
    dst_mns.append(dsts.mean())
    dst_stds.append(dsts.std())
    dst_per_lows.append(np.percentile(dsts, PER_LOW))
    dst_per_highs.append(np.percentile(dsts, PER_HIGH))
    print '\t\t\t+Mean: ' + str(dst_mns[-1]) + ' nm'
    print '\t\t\t+Std: ' + str(dst_stds[-1]) + ' nm'
    print '\t\t\t+Percentiles: [' + str(dst_per_lows[-1]) + ', ' + str(dst_per_highs[-1]) + '] +  nm'

    print '\t\tSaving generated surfaces...'
    ps.disperse_io.save_vtp(pst_surf, out_dir + '/syn_' + lbl + END_STR + '_pst_surf.vtp')
    ps.disperse_io.save_vtp(pre_surf, out_dir + '/syn_' + lbl + END_STR + '_pre_surf.vtp')

print '\tPickling results...'
with open(out_dir + '/dst_lbls.pkl', 'w') as file_lbls:
    pickle.dump(dst_lbls, file_lbls)
with open(out_dir + '/dst_mns.pkl', 'w') as file_mns:
    pickle.dump(dst_mns, file_mns)
with open(out_dir + '/dst_stds.pkl', 'w') as file_stds:
    pickle.dump(dst_stds, file_stds)
with open(out_dir + '/dst_per_lows.pkl', 'w') as file_per_lows:
    pickle.dump(dst_per_lows, file_per_lows)
with open(out_dir + '/dst_per_highs.pkl', 'w') as file_per_highs:
    pickle.dump(dst_per_lows, file_per_highs)

print '\tPlotting the results (mean, std) in nm: '
for lbl, mn, st in zip(dst_lbls, dst_mns, dst_stds):
    print '\t\t-' + lbl + ': ' + str(mn) + ', ' + str(st)
plt.figure()
plt.title('Synapse gap distances')
plt.xlabel('Mean (nm)')
plt.ylabel('Std (nm)')
plt.plot(np.asarray(dst_mns), np.asarray(dst_stds))
plt.show(block=True)
plt.close()
plt.figure()
plt.title('Synapse gap distances II')
plt.xlabel('Percentile ' + str(PER_LOW) + ' %')
plt.ylabel('Percentile ' + str(PER_HIGH) + ' %')
plt.plot(np.asarray(dst_per_lows), np.asarray(dst_per_highs))
plt.show(block=True)
plt.close()

print 'Terminated. (' + time.strftime("%c") + ')'
