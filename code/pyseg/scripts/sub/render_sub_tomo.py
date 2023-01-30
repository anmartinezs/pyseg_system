"""

    Renders template copies in a tomogram from a STAR file

    Input:  - STAR file
            - Reference tomogram
            - Template pre-processing parameters
            - Rigid transformations parameters

    Output: - A set of VTK poly data for rendering the templates on the reference tomogram

"""

__author__ = 'Antonio Martinez-Sanchez'


###### Global variables

import vtk
import copy
import numpy as np
import pyseg as ps

ROT_COL, TILT_COL, PSI_COL = '_rlnAngleRot', '_rlnAngleTilt', '_rlnAnglePsi'
XO_COL, YO_COL, ZO_COL = '_rlnOriginX', '_rlnOriginY', '_rlnOriginZ'
key_col = '_rlnClassNumber'

########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '/fs/pool/pool-lucic2/antonio/workspace/psd_an'

# Input
# in_star = ROOT_PATH + '/ex/syn/sub/relion/fils/pst_t/ch/pst_cont_10nm/particles_pst_cont_10nm_prior_rot_rnd.star'
in_star = ROOT_PATH + '/ex/syn/sub/relion/fils/pst_t/ch/pst_cont5/ref_2/c1/run5_data.star'
in_tomo = ROOT_PATH + '/in/zd/bin2/syn_14_14_bin2.mrc'
in_temp = ROOT_PATH + '/ex/syn/sub/relion/fils/pst_t/ch/pst_cont5/ref_2/c1/run5_class001.mrc'

# Output
out_stem = 'syn_14_14_c1' # If None the stem of input template is used
out_dir = ROOT_PATH+'/ex/syn/sub/relion/fils/pst_t/ch/pst_cont5/ref_2/recon/c1'

# Class filtering
classes = None

###### Template pre-processing parameters

tp_th = 0.171 # Threshold for iso-surface
tp_rsc = 1. # Particle rescaling to fit the reference file resolution

###### Rigid transformation parameters

rt_flip = False
rt_orig = True
rt_swapxy = True

# rt_tr = (471,771,299) # Translation in pixels
# rt_tr = (991,747,107) # (747,991,107) # Translation in pixels
# rt_rot = (0, 0, -27)
# rt_rot = (0, 0, -19) # Rotation angles for axes XYZ in degrees

rt_tr =  (991,747,107)# (911,447,159),
         #  (847,767,87)
	     # (991,747,107),
         # (883,731,99),
	     # (471,771,299),
         # (1203,963,331),
         # (827,531,427),
         # (99,747,283),
         # (703,483,219),
         # (899,1259,275),
         # (999,979,619),
         #  (439,759,459),
         #  (419,819,479),
         #  (479,699,619),
         #  (599,719,399),
         #  (1059,659,439),
         #  (579,959,359),
         #  (639,919,379),
         #  (999,879,379),
         #  (879,799,399),
         #  (1099,839,519),
         #  (679,719,359),

rt_rot =  (0,0,-19)# (0,0,2),
          # (0,0,67)
          # (0,0,-19),
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



##### Helping functions

# Computes 4x4 rotation and translation matrix from the 3 Eulers angles (degrees) in Relion's convention
# a translation vector
def rot_trans_mat_relion(eu_angs, tr_vec):
    mat = np.zeros(shape=(4, 4), dtype=float)
    rot_mat = ps.globals.rot_mat_relion(eu_angs[0], eu_angs[1], eu_angs[2], deg=True)
    mat[:3, :3] = rot_mat
    mat[:3, 3] = tr_vec
    mat[3, 3] = 1.
    return mat

# Iso-surface on a tomogram
def tomo_poly_iso(tomo, th, flp):

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

    return march.GetOutput()

# Appends the template surface in the position specified by the peaks
# tpeaks: TomoPeaks object
# temp_poly: template vtkPolyData
# rot_cols: 3-tuple of strings with the keys for angles rotation, so far only Relion's convention is used
# temp_shape: template bounding box shape
# Returns: poly with the template copies in their positions, the center of mass of every copy in a list
def tomo_poly_peaks(tpeaks, temp_poly, rot_cols, temp_shape, do_origin=True):

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
            try:
                orig_x, orig_y, orig_z = peak.get_prop_val(XO_COL), peak.get_prop_val(YO_COL), peak.get_prop_val(ZO_COL)
            except KeyError:
                orig_x, orig_y, orig_z = 0, 0, 0
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
            try:
                orig_x, orig_y, orig_z = peak.get_prop_val(XO_COL), peak.get_prop_val(YO_COL), peak.get_prop_val(ZO_COL)
            except KeyError:
                orig_x, orig_y, orig_z = 0, 0, 0
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
        cmas_computer.SetInputData(hold_poly)
        cmas_computer.SetUseScalarsAsWeights(False)
        cmas_computer.Update()
        cmass.append(np.asarray(cmas_computer.GetCenter(), dtype=float))

    # Apply changes
    appender.Update()

    return appender.GetOutput(), cmass

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

# Converts an array of 3D coordinates into a vtkPolyData
def cloud_point_to_poly(cmass):

    # Initialization
    poly = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    verts = vtk.vtkCellArray()

    # Applying mask
    count = 0
    for cmas in cmass:
        x, y, z = cmas[0], cmas[1], cmas[2]
        # Insert the center of mass in the poly data
        points.InsertNextPoint((x, y, z))
        verts.InsertNextCell(1)
        verts.InsertCellPoint(count)
        count += 1

    # Creates the poly
    poly.SetPoints(points)
    poly.SetVerts(verts)

    return poly

########################################################################################
# MAIN ROUTINE
########################################################################################

################# Package import

import os
import sys
import time

########## Print initial message

print('Rendering a template in a tomogram.')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('Options:')
print('\tInput STAR file: ' + in_star)
print('\tInput reference tomogram file: ' + in_tomo)
print('\tInput template tomogram: ' + in_temp)
if out_stem is None:
    _, temp_fname = os.path.split(in_temp)
    out_stem, _ = os.path.splitext(temp_fname)
print('\tOutput directory: ' + out_dir)
print('\tOutput suffix: ' + out_stem)
print('\tTemplate Pre-processing: ')
print('\t\t-Template threshold: ' + str(tp_th))
print('\t\t-Re-scaling factor: ' + str(tp_rsc))
print('\tRigid transformations: ')
if rt_flip:
    print('\t\t-Template flipping active.')
if rt_orig:
    print('\t\t-Origin compensation.')
if rt_swapxy:
    print('\t\t-Swapping XY coordinates.')
print('\t\t-Rotation extrinsic XYZ: ' + str(rt_rot))
print('\t\t-Un-cropping with: ' + str(rt_tr))
print('')

######### Process

print('Main procedure:')

print('\tLoading the input STAR file...')
star = ps.sub.Star()
star.load(in_star)

if classes is not None:
    print('\tFiltering to keep classes ' + str(classes) + '...')
    del_rows = list()
    for i in range(star.get_nrows()):
        if not(star.get_element(key_col, i) in classes):
            del_rows.append(i)
    star.del_rows(del_rows)

print('\tScaling particle coordinates...')
star.scale_coords(tp_rsc)

print('\tGeneration tomogram peaks object...')
tpeaks = star.gen_tomo_peaks(in_tomo, klass=None, orig=rt_orig, full_path=False)

print('\tGenerate template poly data...')
temp = ps.disperse_io.load_tomo(in_temp)
temp_poly = tomo_poly_iso(temp, tp_th, rt_flip)
ps.disperse_io.save_vtp(temp_poly, out_dir+'/'+out_stem+'_temp.vtp')

print('\tRender templates in the reference tomogram...')
temp_rss = np.asarray(temp.shape, dtype=float) * tp_rsc
ref_poly, cmass = tomo_poly_peaks(tpeaks, temp_poly, (ROT_COL, TILT_COL, PSI_COL), temp_rss, rt_orig)
cmass_poly = cloud_point_to_poly(cmass)

print('\tRigid body translations...')
if rt_swapxy:
    ref_poly = tomo_poly_swapxy(ref_poly)
    cmass_poly = tomo_poly_swapxy(cmass_poly)
ref_poly = tomo_ref_poly_rotation(ref_poly, in_tomo, rt_rot)
ref_poly = tomo_ref_poly_trans(ref_poly, -1.*np.asarray(rt_tr, dtype=np.float32))
cmass_poly = tomo_ref_poly_rotation(cmass_poly, in_tomo, rt_rot)
cmass_poly = tomo_ref_poly_trans(cmass_poly, -1.*np.asarray(rt_tr, dtype=np.float32))

print('\tStoring the results...')
ps.disperse_io.save_vtp(ref_poly, out_dir+'/'+out_stem+'.vtp')
ps.disperse_io.save_vtp(cmass_poly, out_dir+'/'+out_stem+'_cmass.vtp')
# ps.disperse_io.save_numpy(tpeaks.to_tomo_cloud(), out_dir+'/'+out_stem+'.mrc')
