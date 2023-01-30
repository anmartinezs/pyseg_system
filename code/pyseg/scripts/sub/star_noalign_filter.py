"""

    Generates a filtered copy of a STAR file where miss aligned particles are filtered.
    Miss alignment is computed by comparison against another reference STAR file.

    Input:  - STAR file to filter
            - Reference STAR file
            - Miss alignment parameters

    Output: - A copy of input STAR file where only aligned entries are preserved

"""

__author__ = 'Antonio Martinez-Sanchez'


###### Global variables

import pyseg as ps
import vtk
import scipy as sp

MIC_NAME, IMG_NAME = '_rlnMicrographName', '_rlnImageName'
N_BINS = 10.

########################################################################################
# PARAMETERS
########################################################################################

# ROOT_PATH = '/home/martinez/pool/pool-lucic2/antonio/workspace/psd_an/ex/syn/sub/relion/fils/pst_t/ch/pst_cont_10nm'
ROOT_PATH = '/fs/pool/pool-lucic2/antonio/workspace/psd_an/ex/syn/sub/relion/fils/pst'
ROOT_PATH_MIC = '/fs/pool/pool-lucic2/antonio/workspace/psd_an/in/rln/tomos'
ROOT_PATH_SEG = '/fs/pool/pool-lucic2/antonio/workspace/psd_an'

# Input star file
# in_star = ROOT_PATH + '/Refine3D/run1_data.star'
in_star = ROOT_PATH + '/ref_2/mask_20_50_12/klass_2_gather.star'

# Reference star file
# ref_star = ROOT_PATH + '/particles_pst_cont_10nm.star'
ref_star = ROOT_PATH + '/particles_pst_cont.star'
out_stem = 'clean2_ref'

# Output directory for results
# out_dir = ROOT_PATH + '/recon/ref/clean'
out_dir = ROOT_PATH + '/ref_3/clean'

###### Parameters

do_del = False
ref_vect = (0., 0., 1.) # Reference vector
max_angle = 35 # deg
dst_rg = [5, 14] # nm
part_dim = 64 # voxels

####### Reference segmentation

seg_save = True
seg_sg = 10 # Voxels
seg_th = 0.45
seg_lbl = 3 # 3-PSD
seg_res = 0.684 #nm/voxel
in_mics = (ROOT_PATH_MIC+'/syn_14_9_bin2/syn_14_9_bin2.mrc',
           ROOT_PATH_MIC+'/syn_14_13_bin2/syn_14_13_bin2.mrc',
           ROOT_PATH_MIC+'/syn_14_14_bin2/syn_14_14_bin2.mrc',
           ROOT_PATH_MIC+'/syn_14_15_bin2/syn_14_15_bin2.mrc',
           ROOT_PATH_MIC+'/syn_11_2_bin2/syn_11_2_bin2.mrc',
           ROOT_PATH_MIC+'/syn_11_5_bin2/syn_11_5_bin2.mrc',
           ROOT_PATH_MIC+'/syn_11_6_bin2/syn_11_6_bin2.mrc',
           ROOT_PATH_MIC+'/syn_11_9_bin2/syn_11_9_bin2.mrc',
           ROOT_PATH_MIC+'/syn_13_1_bin2/syn_13_1_bin2.mrc',
           ROOT_PATH_MIC+'/syn_13_3_bin2/syn_13_3_bin2.mrc',
           ROOT_PATH_MIC+'/syn_14_17_bin2/syn_14_17_bin2.mrc',
           ROOT_PATH_MIC+'/syn_14_18_bin2/syn_14_18_bin2.mrc',
           ROOT_PATH_MIC+'/syn_14_19_bin2/syn_14_19_bin2.mrc',
           ROOT_PATH_MIC+'/syn_14_20_bin2/syn_14_20_bin2.mrc',
           ROOT_PATH_MIC+'/syn_14_22_bin2/syn_14_22_bin2.mrc',
           ROOT_PATH_MIC+'/syn_14_24_bin2/syn_14_24_bin2.mrc',
           ROOT_PATH_MIC+'/syn_14_25_bin2/syn_14_25_bin2.mrc',
           ROOT_PATH_MIC+'/syn_14_26_bin2/syn_14_26_bin2.mrc',
           ROOT_PATH_MIC+'/syn_14_27_bin2/syn_14_27_bin2.mrc',
           ROOT_PATH_MIC+'/syn_14_28_bin2/syn_14_28_bin2.mrc',
           # ROOT_PATH_MIC+'/syn_14_32_bin2/syn_14_32_bin2.mrc',
           # ROOT_PATH_MIC+'/syn_14_33_bin2/syn_14_33_bin2.mrc',
           )
in_segs =  (ROOT_PATH_SEG + '/in/zd/bin2/syn_14_9_bin2_crop2_syn.fits',
	    ROOT_PATH_SEG + '/in/zd/bin2/syn_14_13_bin2_crop2_syn.fits',
	    ROOT_PATH_SEG + '/in/zd/bin2/syn_14_14_bin2_crop2_syn.fits',
            # ROOT_PATH_SEG + '/in/zd/bin2/syn_14_15_bin2_crop2_syn.fits',
	    # ROOT_PATH_SEG + '/in/uli/bin2/syn_11_2_bin2_rot_crop2_seg.fits',
	    # ROOT_PATH_SEG + '/in/uli/bin2/syn_11_5_bin2_rot_crop2_seg.fits',
	    # ROOT_PATH_SEG + '/in/uli/bin2/syn_11_6_bin2_rot_crop2_seg.fits',
	    # ROOT_PATH_SEG + '/in/uli/bin2/syn_11_9_bin2_rot_crop2_seg.fits',
	    # ROOT_PATH_SEG + '/in/uli/bin2/syn_13_1_bin2_rot_crop2_seg.fits',
	    # ROOT_PATH_SEG + '/in/uli/bin2/syn_13_3_bin2_rot_crop2_seg.fits',
            # ROOT_PATH_SEG + '/in/ch/bin2/syn_14_17_bin2_rot_crop2_seg.fits',
            # ROOT_PATH_SEG + '/in/ch/bin2/syn_14_18_bin2_rot_crop2_seg.fits',
            # ROOT_PATH_SEG + '/in/ch/bin2/syn_14_19_bin2_rot_crop2_seg.fits',
            # ROOT_PATH_SEG + '/in/ch/bin2/syn_14_20_bin2_rot_crop2_seg.fits',
            # ROOT_PATH_SEG + '/in/ch/bin2/syn_14_22_bin2_rot_crop2_seg.fits',
            # ROOT_PATH_SEG + '/in/ch/bin2/syn_14_24_bin2_rot_crop2_seg.fits',
            # ROOT_PATH_SEG + '/in/ch/bin2/syn_14_25_bin2_rot_crop2_seg.fits',
            # ROOT_PATH_SEG + '/in/ch/bin2/syn_14_26_bin2_rot_crop2_seg.fits',
            # ROOT_PATH_SEG + '/in/ch/bin2/syn_14_27_bin2_rot_crop2_seg.fits',
            # ROOT_PATH_SEG + '/in/ch/bin2/syn_14_28_bin2_rot_crop2_seg.fits',
            # ROOT_PATH_SEG + '/in/ch/bin2/syn_14_32_bin2_rot_crop2_seg.fits',
            # ROOT_PATH_SEG + '/in/ch/bin2/syn_14_33_bin2_rot_crop2_seg.fits',
            )
in_offs = ((911,447,159),
           (847,767,87),
	   (991,747,107),
           # (883,731,99),
	   # (471,771,299),
           # (1203,963,331),
           # (827,531,427),
           # (99,747,283),
           # (703,483,219),
           # (899,1259,275),
           # (999,979,619),
           # (439,759,459),
           # (419,819,479),
           # (479,699,619),
           # (599,719,399),
           # (1059,659,439),
           # (579,959,359),
           # (639,919,379),
           # (999,879,379),
           # (879,799,399),
           # (1099,839,519),
           # (679,719,359),
           )
in_rots = ((0,0,2),
           (0,0,67),
           (0,0,-19),
           # (0,0,44),
	   # (0,0,-27),
           # (0,0,-30),
           # (0,0,12),
           # (0,0,10),
           # (0,0,23),
           # (0,0,20),
           # (0,0,20),
           # (0,0,0),
           # (0,0,55),
           # (0,0,-45),
           # (0,0,60),
           # (0,0,-50),
           # (0,0,-55),
           # (0,0,-45),
           # (0,0,-16),
           # (0,0,-55),
           # (0,0,-85),
           # (0,0,-55),
           )


########################################################################################
# MAIN ROUTINE
########################################################################################

# Rotation (extrinsic XYZ) around reference tomogram center
def tomo_ref_poly_rotation(ref_poly, in_tomo, ang_rot):

    # Initialization
    tomo = ps.disperse_io.load_tomo(in_tomo, mmap=True)
    center = .5 * np.asarray(tomo.shape, dtype=np.float32)

    # Setting up the transformations
    # Centering
    cent_tr = vtk.vtkTransform()
    cent_tr.Translate(-1.*center)
    tr_cent = vtk.vtkTransformPolyDataFilter()
    tr_cent.SetTransform(cent_tr)
    # Rotation
    rot_tr = vtk.vtkTransform()
    rot_tr.RotateX(ang_rot[0])
    rot_tr.RotateY(ang_rot[1])
    rot_tr.RotateZ(ang_rot[2])
    tr_rot = vtk.vtkTransformPolyDataFilter()
    tr_rot.SetTransform(rot_tr)
    # Back-center
    bak_tr = vtk.vtkTransform()
    bak_tr.Translate(center)
    tr_bak = vtk.vtkTransformPolyDataFilter()
    tr_bak.SetTransform(bak_tr)

    # Apply the filters
    tr_cent.SetInputData(ref_poly)
    tr_cent.Update()
    tr_rot.SetInputData(tr_cent.GetOutput())
    tr_rot.Update()
    tr_bak.SetInputData(tr_rot.GetOutput())
    tr_bak.Update()

    return tr_bak.GetOutput()

# Translating a vtkPolyData
def tomo_ref_poly_trans(ref_poly, rt_tr):

    # Setting up the transformation
    cent_tr = vtk.vtkTransform()
    cent_tr.Translate(rt_tr)
    tr_cent = vtk.vtkTransformPolyDataFilter()
    tr_cent.SetTransform(cent_tr)

    # Apply the filters
    tr_cent.SetInputData(ref_poly)
    tr_cent.Update()

    return tr_cent.GetOutput()

# Swap XY coordinates in a vtkPolyData
def tomo_poly_swapxy(ref_poly):

    # Initialization
    new_pts = vtk.vtkPoints()
    old_pts = ref_poly.GetPoints()

    # Loop for swapping
    for i in range(old_pts.GetNumberOfPoints()):
        pt = old_pts.GetPoint(i)
        new_pts.InsertPoint(i, [pt[1], pt[0], pt[2]])

    # Updating the poly data
    ref_poly.SetPoints(new_pts)

    return ref_poly

# Gets a smooth surface from a segmented region
# sg: sigma for gaussian smoothing
# th: iso-surface threshold
# Returns: a polydata with the surface found
def tomo_smooth_surf(seg, sg, th):

    # Smoothing
    seg_s = sp.ndimage.filters.gaussian_filter(seg.astype(float), sg)
    seg_vti = ps.disperse_io.numpy_to_vti(seg_s)

    # Iso-surface
    surfaces = vtk.vtkMarchingCubes()
    surfaces.SetInputData(seg_vti)
    surfaces.ComputeNormalsOn()
    surfaces.ComputeGradientsOn()
    surfaces.SetValue(0, th)
    surfaces.Update()

    # # Keep just the largest surface
    # con_filter = vtk.vtkPolyDataConnectivityFilter()
    # con_filter.SetInputData(surfaces.GetOutput())
    # con_filter.SetExtractionModeToLargestRegion()

    # return con_filter.GetOutput()

    return surfaces.GetOutput()

# Finds the closest points of a poly to another poly
# Returns an array of coordiantes with the closes distance point on poly_ref to poly
def tomo_2poly_dst(poly_ref, poly):

    # Initialization
    points = np.zeros(shape=(poly.GetNumberOfPoints(),3), dtype=float)
    dst_filter = ps.vtk_ext.vtkClosestPointAlgorithm()
    dst_filter.SetInputData(poly_ref)
    dst_filter.initialize()

    # ps.disperse_io.save_vtp(poly_ref, out_dir+'/hold_ref.vtp')

    # Finding the closest points
    for i in range(poly.GetNumberOfPoints()):
        x, y, z = poly.GetPoint(i)
        dst_pid = dst_filter.evaluate_id(x, y, z)
        points[i, :] = np.asarray(poly_ref.GetPoint(int(dst_pid)), dtype=float)

    # ps.disperse_io.save_vtp(poly, out_dir+'/hold.vtp')

    return points


################# Package import

import os
import sys
import time
import math
import numpy as np
import matplotlib.pyplot as plt

########## Print initial message

print('Filtering miss-aligned entries in a STAR file.')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('Options:')
print('\tInput file: ' + in_star)
print('\tReference file: ' + ref_star)
print('\tOutput directory: ' + out_dir)
print('\tOutput stem: ' + out_stem)
print('\tInput segmentation:')
print('\t\t-Micrograph names:' + str(in_mics))
print('\t\t-Segmentation tomograms:' + str(in_segs))
if seg_save:
    print('\t\t-Save segmentation active.')
print('\t\t-Surface estimation options:')
print('\t\t\t-Sigma for smoothing: ' + str(seg_sg) + ' voxels')
print('\t\t\t-Threshold: ' + str(seg_th))
print('\t\t-Rigid body transformation:')
print('\t\t\t-Segmentation offsets (voxels):' + str(in_offs))
print('\t\t\t-Segmentation rotations (degrees):' + str(in_rots))
print('\tMiss-alignment parameters:')
print('\t\t-Reference vector: ' + str(ref_vect))
print('\t\t-Maximum angle: ' + str(max_angle) + ' deg')
print('\t\t-Reference surface distance range: ' + str(dst_rg) + ' voxels')
print('')

######### Process

print('Main Routine: ')

print('\tLoading input and reference STAR files...')
star = ps.sub.Star()
try:
    star.load(in_star)
except ps.pexceptions.PySegInputError as e:
    print('ERROR: input STAR file could not be read because of "' + str(e.msg, e.expr) + '"')
    sys.exit(-1)
r_star = ps.sub.Star()
try:
    r_star.load(ref_star)
except ps.pexceptions.PySegInputError as e:
    print('ERROR: input STAR file could not be read because of "' + str(e.msg, e.expr) + '"')
    sys.exit(-1)

print('\tCreating the poly data...')
nrows = star.get_nrows()
ref_vect = np.asarray(ref_vect, dtype=float)
points = np.zeros(shape=len(in_segs), dtype=object)
cells = np.zeros(shape=len(in_segs), dtype=object)
vects = np.zeros(shape=len(in_segs), dtype=object)
vects_r = np.zeros(shape=len(in_segs), dtype=object)
pangs = np.zeros(shape=len(in_segs), dtype=object)
pdsts = np.zeros(shape=len(in_segs), dtype=object)
row_ids = np.zeros(shape=len(in_segs), dtype=object)
curr_pts = np.zeros(shape=len(in_segs), dtype=int)
for row in range(nrows):
    mic = star.get_element(MIC_NAME, row)
    try:
        mic_idx = in_mics.index(mic)
    except ValueError:
        continue
    if not isinstance(points[mic_idx], vtk.vtkPoints):
        points[mic_idx], cells[mic_idx] = vtk.vtkPoints(), vtk.vtkCellArray()
        row_id = vtk.vtkIntArray()
        row_id.SetName('Row ID')
        row_id.SetNumberOfComponents(1)
        row_ids[mic_idx] = row_id
        vect = vtk.vtkFloatArray()
        vect.SetName('Particle vector')
        vect.SetNumberOfComponents(3)
        vects[mic_idx] = vect
        vect_r = vtk.vtkFloatArray()
        vect_r.SetName('Reference vector')
        vect_r.SetNumberOfComponents(3)
        vects_r[mic_idx] = vect_r
        pang = vtk.vtkFloatArray()
        pang.SetName('Angle')
        pang.SetNumberOfComponents(1)
        pangs[mic_idx] = pang
        pdst = vtk.vtkFloatArray()
        pdst.SetName('Distance')
        pdst.SetNumberOfComponents(1)
        pdsts[mic_idx] = pdst
    # Getting particle information
    c_x, c_y, c_z = star.get_element('_rlnCoordinateX', row), star.get_element('_rlnCoordinateY', row), \
                    star.get_element('_rlnCoordinateZ', row)
    try:
        rot, psi, tilt = star.get_element('_rlnAngleRot', row), star.get_element('_rlnAnglePsi', row), \
                         star.get_element('_rlnAngleTilt', row)
    except KeyError:
        print('ERROR: invalid input STAR file!')
        print('Unsuccessfully terminated. (' + time.strftime("%c") + ')')
    try:
        o_x, o_y, o_z = star.get_element('_rlnOriginX', row), star.get_element('_rlnOriginY', row), \
                        star.get_element('_rlnOriginZ', row)
    except KeyError:
        o_x, o_y, o_z = 0., 0., 0.
    coord = np.asarray((c_x-o_x, c_y-o_y, c_z-o_z), dtype=np.float32)
    # coord = np.asarray((c_y-o_y, c_x-o_x, c_z-o_z), dtype=np.float32)
    points[mic_idx].InsertPoint(curr_pts[mic_idx], coord)
    cells[mic_idx].InsertNextCell(1)
    cells[mic_idx].InsertCellPoint(curr_pts[mic_idx])
    row_ids[mic_idx].InsertNextTuple((row,))
    # Rotate vector from STAR file angles
    mat_r = ps.globals.rot_mat_relion(rot, tilt, psi, deg=True)
    rot_v = mat_r.T * ref_vect.reshape(3, 1)
    rot_v = np.asarray((rot_v[1, 0], rot_v[0, 0], rot_v[2, 0]), dtype=float)
    # Rotate to fit reference tomogram
    mat_r = ps.globals.rot_mat(in_rots[mic_idx][0], -1.*in_rots[mic_idx][2], in_rots[mic_idx][1])
    rot_v = mat_r * rot_v.reshape(3, 1)
    rot_v = np.asarray((rot_v[0, 0], rot_v[1, 0], rot_v[2, 0]), dtype=float)
    vects[mic_idx].InsertNextTuple(tuple(rot_v))
    pangs[mic_idx].InsertNextTuple((-1.,))
    pdsts[mic_idx].InsertNextTuple((-1.,))
    vects_r[mic_idx].InsertNextTuple((0., 0., 0))
    curr_pts[mic_idx] += 1

print('\tComputing miss-alignment:')
el_to_del = np.ones(shape=nrows, dtype=bool)
dsts, angs = list(), list()
for mic_idx, mic in enumerate(in_mics):
    print('\t\t-Processing micrograph: ' + str(mic))
    if isinstance(points[mic_idx], vtk.vtkPoints):
        # Polydata creation
        row_id = row_ids[mic_idx]
        pang, pdst, vect, vect_r = pangs[mic_idx], pdsts[mic_idx], vects[mic_idx], vects_r[mic_idx]
        poly = vtk.vtkPolyData()
        poly.SetPoints(points[mic_idx])
        poly.SetVerts(cells[mic_idx])
        poly.GetPointData().AddArray(row_id)
        poly.GetPointData().AddArray(vect)
        # Rigid body transformations
        poly = tomo_poly_swapxy(poly)
        poly = tomo_ref_poly_rotation(poly, mic, in_rots[mic_idx])
        poly = tomo_ref_poly_trans(poly, -1.*np.asarray(in_offs[mic_idx], dtype=float))
        # Estimate reference surface
        surf = tomo_smooth_surf(ps.disperse_io.load_tomo(in_segs[mic_idx])==seg_lbl, seg_sg, seg_th)
        # Find closest point on surface
        hold_pts = tomo_2poly_dst(surf, poly)
        parts, del_parts = 0, 0
        for i, hold_pt in enumerate(hold_pts):
            parts += 1
            del_parts += 1
            row = int(row_id.GetTuple(i)[0])
            coord = poly.GetPoint(i)
            coord = np.round(coord).astype(int)
            hold_vect = coord.astype(float) - hold_pt
            vect_r.SetTuple(i, (hold_vect[0], hold_vect[1], hold_vect[2]))
            dst = math.sqrt((hold_vect*hold_vect).sum())
            # print 'DEBUG: dst= ' + str(dst) + ' nm'
            ang = math.degrees(ps.globals.angle_2vec_3D(vects[mic_idx].GetTuple(i), hold_vect))
            # print 'DEBUG: v= ' + str(vects[mic_idx].GetTuple(i)) + ', v_ref=' + str(hold_vect) + ', ang=' + str(ang) + ' degs'
            if (dst >= dst_rg[0]) and (dst <= dst_rg[1]):
                if (ang>=0) and (ang <= max_angle):
                    el_to_del[row] = False
                    del_parts -= 1
            pang.SetTuple(i, (ang,))
            pdst.SetTuple(i, (dst,))
            dsts.append(dst)
            angs.append(ang)
        poly.GetPointData().AddArray(pang)
        poly.GetPointData().AddArray(pdst)
        poly.GetPointData().AddArray(vect_r)
        print('\t\t\t' + str(del_parts) + ' of ' + str(parts) + ' will be deleted in this micrograph')
        mic_stem = os.path.splitext(os.path.split(mic)[1])[0]
        out_poly = out_dir + '/' + out_stem + '_' + mic_stem + '.vtp'
        print('\t\t\tPoly data stored in: ' + out_poly)
        ps.disperse_io.save_vtp(poly, out_poly)
        if seg_save:
            out_seg = out_dir + '/' + out_stem + '_' + mic_stem + '_seg_surf.vtp'
            print('\t\t\tReference region segmentation saved in: ' + out_poly)
            ps.disperse_io.save_vtp(surf, out_seg)

print('\tPlotting statistics...')
if (len(dsts) == 0) or (len(angs) == 0):
    print('\tWARNING: Nothing to plot')
    print('Successfully terminated. (' + time.strftime("%c") + ')')
dsts, angs = np.asarray(dsts, dtype=float), np.asarray(angs, dtype=float)
plt.hist(angs, normed=1)
plt.xlabel('Degrees')
plt.ylabel('Probability')
plt.title('Angles Histogram')
plt.xlim(0, 180)
plt.grid(True)
plt.show(block=True)
plt.hist(dsts, normed=1)
plt.xlabel('Voxels')
plt.ylabel('Probability')
plt.title('Shifting Histogram')
plt.xlim(0, dst_rg[0]+dst_rg[1])
plt.grid(True)
plt.show(block=True)

print('\tFinding entries to delete...')
del_rows = list()
for i, to_del in enumerate(el_to_del):
    if to_del:
        del_rows.append(i)

print('\tParticles marked for deletion: ' + str(len(del_rows)) + ' of ' + str(star.get_nrows()))

if do_del:
    print('\tDeleting rows...')
    star.del_rows(del_rows)

    out_file = out_dir + '/' + out_stem + '_' + os.path.splitext(os.path.split(in_star)[1])[0] + '_malign.star'
    print('\tStoring the results in: ' + out_file)
    star.store(out_file)


print('Successfully terminated. (' + time.strftime("%c") + ')')