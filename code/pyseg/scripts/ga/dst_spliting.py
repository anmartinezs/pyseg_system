"""

    Measures distance from STAR files particles to segmentations and splits them according their distance

    Input:  - STAR files
            - Segmentation
            - Distance parameters
            - Particle template preprocessing settings

    Output: - The splited STAR files

"""

__author__ = 'Antonio Martinez-Sanchez'


###### Global variables

import vtk
import numpy as np
import scipy as sp
import pyseg as ps

ROT_COL, TILT_COL, PSI_COL = '_rlnAngleRot', '_rlnAngleTilt', '_rlnAnglePsi'
XO_COL, YO_COL, ZO_COL = '_rlnOriginX', '_rlnOriginY', '_rlnOriginZ'
key_col = '_rlnClassNumber'

########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '/fs/pool/pool-lucic2/antonio/tomograms/qguo/net/render'

# Input
in_stars = (ROOT_PATH + '/B1G4L4T1/B1G4L4T1_GS1.star',
            ROOT_PATH + '/B1G4L4T1/B1G4L4T1_GS2.star',
            ROOT_PATH + '/B1G4L4T1/B1G4L4T1_SPS1.star',
            ROOT_PATH + '/B1G4L4T1/B1G4L4T1_SPS2.star',
            ROOT_PATH + '/B2G1L1T1/B2G1L1T1_GS1.star',
            ROOT_PATH + '/B2G1L1T1/B2G1L1T1_GS2.star',
            ROOT_PATH + '/B2G1L1T1/B2G1L1T1_SPS1.star',
            ROOT_PATH + '/B2G1L1T1/B2G1L1T1_SPS2.star',
            ROOT_PATH + '/B2G1L7T1/B2G1L7T1_GS1.star',
            ROOT_PATH + '/B2G1L7T1/B2G1L7T1_GS2.star',
            ROOT_PATH + '/B2G1L7T1/B2G1L7T1_SPS1.star',
            ROOT_PATH + '/B2G1L7T1/B2G1L7T1_SPS2.star',)
in_temps = (ROOT_PATH + '/B2G1L1T1/02_GS_good_class001.mrc',
            ROOT_PATH + '/B2G1L1T1/02_GS_good_class001.mrc',
            ROOT_PATH + '/B2G1L1T1/02_GS_good_class001.mrc',
            ROOT_PATH + '/B2G1L1T1/02_GS_good_class001.mrc',
            ROOT_PATH + '/B2G1L1T1/02_GS_good_class001.mrc',
            ROOT_PATH + '/B2G1L1T1/02_GS_good_class001.mrc',
            ROOT_PATH + '/B2G1L1T1/02_GS_good_class001.mrc',
            ROOT_PATH + '/B2G1L1T1/02_GS_good_class001.mrc',
            ROOT_PATH + '/B2G1L1T1/02_GS_good_class001.mrc',
            ROOT_PATH + '/B2G1L1T1/02_GS_good_class001.mrc',
            ROOT_PATH + '/B2G1L1T1/02_GS_good_class001.mrc')
in_masks = (ROOT_PATH + '/B1G4L4T1/S_bin_mask_swap_B1G4L4T1.vti',
            ROOT_PATH + '/B1G4L4T1/S_bin_mask_swap_B1G4L4T1.vti',
            ROOT_PATH + '/B1G4L4T1/S_bin_mask_swap_B1G4L4T1.vti',
            ROOT_PATH + '/B1G4L4T1/S_bin_mask_swap_B1G4L4T1.vti',
            ROOT_PATH + '/B2G1L1T1/S_bin_mask_swap_B2G1L1T1.vti',
            ROOT_PATH + '/B2G1L1T1/S_bin_mask_swap_B2G1L1T1.vti',
            ROOT_PATH + '/B2G1L1T1/S_bin_mask_swap_B2G1L1T1.vti',
            ROOT_PATH + '/B2G1L1T1/S_bin_mask_swap_B2G1L1T1.vti',
            ROOT_PATH + '/B2G1L7T1/S_bin_mask_swap_B2G1L7T1.vti',
            ROOT_PATH + '/B2G1L7T1/S_bin_mask_swap_B2G1L7T1.vti',
            ROOT_PATH + '/B2G1L7T1/S_bin_mask_swap_B2G1L7T1.vti',
            ROOT_PATH + '/B2G1L7T1/S_bin_mask_swap_B2G1L7T1.vti')
in_tomos = (ROOT_PATH + '/B1G4L4T1/filter_B1G4L4T1.mrc',
            ROOT_PATH + '/B1G4L4T1/filter_B1G4L4T1.mrc',
            ROOT_PATH + '/B1G4L4T1/filter_B1G4L4T1.mrc',
            ROOT_PATH + '/B1G4L4T1/filter_B1G4L4T1.mrc',
            ROOT_PATH + '/B2G1L1T1/filter_B1G4L4T1.mrc',
            ROOT_PATH + '/B2G1L1T1/filter_B1G4L4T1.mrc',
            ROOT_PATH + '/B2G1L1T1/filter_B1G4L4T1.mrc',
            ROOT_PATH + '/B2G1L1T1/filter_B1G4L4T1.mrc',
            ROOT_PATH + '/B2G1L7T1/filter_B1G4L4T1.mrc',
            ROOT_PATH + '/B2G1L7T1/filter_B1G4L4T1.mrc',
            ROOT_PATH + '/B2G1L7T1/filter_B1G4L4T1.mrc',
            ROOT_PATH + '/B2G1L7T1/filter_B1G4L4T1.mrc')

# Output
out_dir = ROOT_PATH + '/dsts'

# Class filtering
classes = None

###### Reference file

rf_res = 1.368 # nm/voxel

###### Template pre-processing parameters

tp_cube = False # If true a cube with input template dimensions is used
tp_th = 0.035 # 0.122 # Threshold for iso-surface
tp_dec = 0.5 # Decimation factor before rescaling
tp_rsc = 1./4. # Particle rescaling to fit the reference file resolution
tp_cmass = True # If True then the center of mass is computed instead the picked postion

###### Rigid transformation parameters

rt_flip = False
rt_orig = True
rt_swapxy = True

###### Masking

mk_rsc = 2. # Mask rescaling factor to fit the reference file resolution
mk_dst = 20 # nm, distance to grow the mask

##### Helping functions

# Generates a cubic surface with the dimensions of the input tomogram
def tomo_poly_iso_cube(tomo, dec=None):

    # Creating the cube
    cube = np.zeros(shape=tomo.shape, dtype=np.float32)
    cube[1:tomo.shape[0]-1, 1:tomo.shape[1]-1, 1:tomo.shape[2]-1] = 1.

    # Marching cubes configuration
    march = vtk.vtkMarchingCubes()
    cube_vtk = ps.disperse_io.numpy_to_vti(cube)
    march.SetInputData(cube_vtk)
    march.SetValue(0, .5)

    # Running Marching Cubes
    march.Update()
    hold_poly = march.GetOutput()

    # Decimation filter
    if dec is not None:
        tr_dec = vtk.vtkDecimatePro()
        tr_dec.SetInputData(hold_poly)
        tr_dec.SetTargetReduction(dec)
        tr_dec.Update()
        hold_poly = tr_dec.GetOutput()

    return hold_poly

# Iso-surface on a tomogram
def tomo_poly_iso(tomo, th, flp, dec=None):

    # Marching cubes configuration
    march = vtk.vtkMarchingCubes()
    # tomo_vtk = numpy_support.numpy_to_vtk(num_array=tomo.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
    tomo_vtk = ps.disperse_io.numpy_to_vti(tomo)

    # Flipping
    if flp:
        fliper = vtk.vtkImageFlip()
        fliper.SetFilteredAxis(2)
        fliper.SetInputData(tomo_vtk)
        fliper.Update()
        tomo_vtk = fliper.GetOutput()
    march.SetInputData(tomo_vtk)
    march.SetValue(0, th)

    # Running Marching Cubes
    march.Update()
    hold_poly = march.GetOutput()

    # Decimation filter
    if dec is not None:
        tr_dec = vtk.vtkDecimatePro()
        tr_dec.SetInputData(hold_poly)
        tr_dec.SetTargetReduction(dec)
        tr_dec.Update()
        hold_poly = tr_dec.GetOutput()

    return hold_poly

# Appends the template surface in the position specified by the peaks
# tpeaks: TomoPeaks object
# temp_poly: template vtkPolyData
# rot_cols: 3-tuple of strings with the keys for angles rotation, so far only Relion's convention is used
# temp_shape: template bounding box shape
# do_cmass: if True (default False) then the center of the mass of the density is computed instead the particle position
# Returns: poly with the template copies in their positions, the center of mass of every copy in a list
def tomo_poly_peaks(tpeaks, temp_poly, rot_cols, temp_shape, do_origin=True, do_cmass=False):

    # Initialization
    cmass = list()
    appender = vtk.vtkAppendPolyData()
    cmas_computer = vtk.vtkCenterOfMass()

    # Box compensation
    # box_orig = np.asarray((-.5*temp_shape[0], -.5*temp_shape[1], -.5*temp_shape[2]), dtype=float)
    box_tr = vtk.vtkTransform()
    box_tr.Translate(-.5*temp_shape[0], -.5*temp_shape[1], -.5*temp_shape[2])

    # Loop for peaks
    # count = 0
    for peak in tpeaks.get_peaks_list():

        # if count > 5:
        #     break
        # count += 1

        # Getting geometrical information
        coords = peak.get_prop_val(ps.sub.PK_COORDS)
        rot_ang, tilt_ang, psi_ang = peak.get_prop_val(rot_cols[0]), peak.get_prop_val(rot_cols[1]), \
                                     peak.get_prop_val(rot_cols[2])

        # Temporal copy
        hold_poly = vtk.vtkPolyData()
        hold_poly.DeepCopy(temp_poly)

        # Translation form bounding box top-left-up corner to center
        tr_box = vtk.vtkTransformPolyDataFilter()
        tr_box.SetInputData(hold_poly)
        tr_box.SetTransform(box_tr)
        tr_box.Update()
        hold_poly = tr_box.GetOutput()

        # Rotation around particle center
        rot_mat = ps.globals.rot_mat_relion(rot_ang, tilt_ang, psi_ang)
        rot_mat_inv = rot_mat.T
        mat_rot = vtk.vtkMatrix4x4()
        for i in range(3):
            for j in range(3):
                mat_rot.SetElement(i, j, rot_mat_inv[i, j])
                # mat_rot.SetElement(i, j, rot_mat[i, j])
        rot_tr = vtk.vtkTransform()
        rot_tr.SetMatrix(mat_rot)
        tr_rot = vtk.vtkTransformPolyDataFilter()
        tr_rot.SetInputData(hold_poly)
        tr_rot.SetTransform(rot_tr)
        tr_rot.Update()
        hold_poly = tr_rot.GetOutput()

        # Translation to origin (Shifting)
        if do_origin:
            orig_x, orig_y, orig_z = peak.get_prop_val(XO_COL), peak.get_prop_val(YO_COL), peak.get_prop_val(ZO_COL)
            orig_tr = vtk.vtkTransform()
            orig_tr.Translate(-orig_x, -orig_y, -orig_z)
            # orig_tr.Translate(orig_x, orig_y, orig_z)
            tr_orig = vtk.vtkTransformPolyDataFilter()
            tr_orig.SetInputData(hold_poly)
            tr_orig.SetTransform(orig_tr)
            tr_orig.Update()
            hold_poly = tr_orig.GetOutput()

        # Translation to final location
        tra_tr = vtk.vtkTransform()
        if do_origin:
            orig_x, orig_y, orig_z = peak.get_prop_val(XO_COL), peak.get_prop_val(YO_COL), peak.get_prop_val(ZO_COL)
            tra_tr.Translate(coords+np.asarray((orig_x, orig_y, orig_z), dtype=float))
        else:
            tra_tr.Translate(coords)
        tr_tra = vtk.vtkTransformPolyDataFilter()
        tr_tra.SetInputData(hold_poly)
        tr_tra.SetTransform(tra_tr)
        tr_tra.Update()
        hold_poly = tr_tra.GetOutput()

        # Append to holder poly data
        appender.AddInputData(hold_poly)

        # Center of mass computation
        if do_cmass:
            cmas_computer.SetInputData(hold_poly)
            cmas_computer.SetUseScalarsAsWeights(False)
            cmas_computer.Update()
            cmass.append(np.asarray(cmas_computer.GetCenter(), dtype=float))
        else:
            if do_origin:
                cmass.append(coords+np.asarray((orig_x, orig_y, orig_z), dtype=float))
            else:
                cmass.append(coords)

    # Apply changes
    appender.Update()

    return appender.GetOutput(), cmass

# Scaling a vtkPolyData
def poly_scale(ref_poly, scale, dec=True):

    # Setting up the transformation
    cent_tr = vtk.vtkTransform()
    cent_tr.Scale(scale, scale, scale)
    tr_cent = vtk.vtkTransformPolyDataFilter()
    tr_cent.SetTransform(cent_tr)

    # Apply the rescaling filter
    tr_cent.SetInputData(ref_poly)
    tr_cent.Update()
    hold_poly = tr_cent.GetOutput()

    # Decimation filter
    red = 1. - scale
    if dec and (red < 1.):
        tr_dec = vtk.vtkDecimatePro()
        tr_dec.SetInputData(hold_poly)
        tr_dec.SetTargetReduction(red)
        tr_dec.Update()
        hold_poly = tr_dec.GetOutput()

    return hold_poly

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

# Apply a mask to a list of coordinates and returns vtkPolyData with the survivors
def cmass_poly_mask(cmass, mask, tomo_dst, swapxy):

    # Initialization
    poly = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    verts = vtk.vtkCellArray()
    if tomo_dst is not None:
        dsts = vtk.vtkFloatArray()
        dsts.SetName('Distance to Seg.')
        dsts.SetNumberOfComponents(1)


    # Applying mask
    count = 0
    for cmas in cmass:
        x, y, z = cmas[0], cmas[1], cmas[2]
        try:
            if mask[int(round(x)), int(round(y)), int(round(z))]:
                # Insert the center of mass in the poly data
                if tomo_dst is not None:
                    dsts.InsertNextTuple((ps.globals.trilin3d(tomo_dst, (x, y, z)),))
                if swapxy:
                    points.InsertNextPoint((y, x, z))
                else:
                    points.InsertNextPoint((x, y, z))
                verts.InsertNextCell(1)
                verts.InsertCellPoint(count)
                count += 1
        except IndexError:
            pass

    # Creates the poly
    poly.SetPoints(points)
    poly.SetVerts(verts)
    if tomo_dst is not None:
        poly.GetPointData().AddArray(dsts)

    return poly

# Measures distance from points to vtk Poly data
def cmass_poly_dst(cmass, seg_poly):

    # Initialization
    poly = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    verts = vtk.vtkCellArray()
    dsts = vtk.vtkFloatArray()
    dsts.SetName('Distance to Seg.')
    dsts.SetNumberOfComponents(1)

    # Cell point locator
    loc = vtk.vtkPointLocator()
    loc.SetDataSet(seg_poly);
    loc.BuildLocator();

    # Applying mask
    dsts_l = list()
    count = 0
    for cmas in cmass:
        x, y, z = cmas[0], cmas[1], cmas[2]
        # p = (x, y, z)
        p = (y, x, z)
        hold_id = loc.FindClosestPoint(p)
        hold_p = seg_poly.GetPoint(hold_id)
        points.InsertNextPoint(p)
        verts.InsertNextCell(1)
        verts.InsertCellPoint(count)
        dst = np.asarray(hold_p, dtype=float) - np.asarray(p, dtype=float)
        dst = np.sqrt((dst*dst).sum())
        dsts.InsertNextTuple((dst,))
        dsts_l.append(dst)
        count += 1

    # Creates the poly
    poly.SetPoints(points)
    poly.SetVerts(verts)
    poly.GetPointData().AddArray(dsts)

    return poly, np.asarray(dsts_l, dtype=float)

########################################################################################
# MAIN ROUTINE
########################################################################################

################# Package import

import os
import gc
import sys
import time
import scipy

########## Print initial message

print('Rendering a template in a tomogram.')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('Options:')
print('\tInput STAR files: ' + str(in_stars))
print('\tInput masks: ' + str(in_masks))
print('\tOutput directory: ' + out_dir)
print('\tReference Resolution: ' + str(rf_res) + ' nm/voxel')
print('\tMasks: ' + str(in_masks))
if mk_rsc != 1:
    print('\t\tRe-scaling factor: ' + str(mk_rsc))
print('\t\tDistance to grow: ' + str(mk_dst) + ' nm')
print('\tTemplate Pre-processing: ')
print('\t\t-Initial decimation factor: ' + str(tp_dec))
print('\t\t-Re-scaling factor: ' + str(tp_rsc))
if tp_cmass:
    print('\t\t-Points in the center of mass.')
else:
    print('\t\t-Points in the picked postions.')
print('')

######### Process

print('Main procedure loop:')

for (in_star, in_temp, in_mask, in_tomo) in zip(in_stars, in_temps, in_masks, in_tomos):

    print('\tLoading the input STAR file: ' + str(in_star))
    star = ps.sub.Star()
    star.load(in_star)
    print('\t\t-Number of particles: ' + str(star.get_nrows()))

    print('\tGeneration tomogram peaks object...')
    tpeaks = star.gen_tomo_peaks(in_tomo, klass=None, orig=rt_orig, full_path=False, micro=False)

    print('\tGenerate an instance template poly data: ' + str(in_temp))
    temp = ps.disperse_io.load_tomo(in_temp)
    temp_poly = tomo_poly_iso(temp, tp_th, rt_flip, tp_dec)


    print('\tRender templates in the reference tomogram...')
    temp_rss = np.asarray(temp.shape, dtype=float)
    ref_poly, cmass = tomo_poly_peaks(tpeaks, temp_poly, (ROT_COL, TILT_COL, PSI_COL), temp_rss, rt_orig,
                                      do_cmass=tp_cmass)
    gc.collect()

    print('\tProcessing mask:' + str(in_mask))
    mask = ps.disperse_io.load_tomo(in_mask)
    mask[mask>0] = 1
    density = scipy.ndimage.filters.gaussian_filter(mask, 1.)
    mask = np.swapaxes(mask, 0, 2)
    if mk_rsc != 1:
        mask = sp.ndimage.interpolation.zoom(mask, mk_rsc)

    print('\t\tMeasuring particle segmentation distance...')
    dst_poly = tomo_poly_iso(mask, 0.9, False, None)
    dst_poly = poly_scale(dst_poly, 1./tp_rsc, dec=True)
    print('\t\t-Number of center of mass: ' + str(len(cmass)))
    assert len(cmass) == star.get_nrows()
    cmass_poly, dsts = cmass_poly_dst(cmass, dst_poly)
    gc.collect()

    print('\t\tGenrating the splited STAR files...')
    star.add_column('_DistanceToFib', val=dsts, no_parse=True)

    print('\tStoring the results...')
    out_stem = os.path.splitext(os.path.split(in_star)[1])[0]
    star.store(out_dir + '/' + out_stem + '_dst.star')
    ps.disperse_io.save_vtp(temp_poly, out_dir+'/'+out_stem+'_temp.vtp')
    ps.disperse_io.save_vtp(tomo_poly_swapxy(cmass_poly), out_dir+'/'+out_stem+'_cmass.vtp')
    ps.disperse_io.save_vtp(ref_poly, out_dir+'/'+out_stem+'.vtp')
    ps.disperse_io.save_vtp(tomo_poly_swapxy(dst_poly), out_dir+'/'+out_stem+'_seg.vtp')