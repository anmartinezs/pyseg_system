############################################################################################################
# Set of functions for reading the results returned by DisPerSe tools
#
#

import vtk
import numpy
import scipy
import pyfits
# import math
# from pyorg import pexceptions
import warnings
from pyto.io.image_io import ImageIO as ImageIO

from pyorg.globals.utils import *
from vtk.util import numpy_support

__author__ = 'martinez'

##########################################################################
# CONSTANTS
#
MAX_DIST_SURF = 3

############################################################################################################
# manifold_from_vtk_to_img: Function for loading a manifold from a VTK file
# Input:
#       filename: String with the path to the VTK file, this file should be generated with DISPERSE and cointains
#                 a manifold
#       outputdir: Directory where the output image will be stored
# Return:
#        OK: >0
#        Error: <0
def manifold_from_vtk_to_img(filename, outputdir):

    # Read the source file.
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    output = reader.GetOutput()

    # Get the POINTS of the DATASET
    points = output.GetPoints()

    # Get the FIELD (source_index) of the DATASET.
    field = output.GetAttributesAsFieldData(0)
    field_si = field.GetArray(0)
    if points.GetNumberOfPoints() != field_si.GetNumberOfTuples():
        print "Error (1): input file \"%s\" is corrupted.\n" % filename
        return -1

    # Create the image holder.
    points.ComputeBounds()
    (ox, nx, oy, ny, oz, nz) = points.GetBounds()
    if (ox != 0) or (oy != 0) or (oz != 0) or (nx <= 0) or (ny <= 0):
        print "Error (2): input file \"%s\" is corrupted.\n" % filename
        return -2
    if nz != 0:
        print "Error (3): input file \"%s\" should be 2D.\n" % filename
        return -3
    nx = math.ceil(nx)
    ny = math.ceil(ny)
    hold = numpy.zeros([nx+1, ny+1])

    # Paint image holder
    for k in range(0, points.GetNumberOfPoints()-1):
        x, y, z = points.GetPoint(k)
        f = field_si.GetComponent(k, 0)
        hold[round(x), round(y)] = round(f)

    #### DEBUG: show image result
    #import pylab as plt
    #plt.imshow(hold)
    #plt.show()

    # Save image in disk
    inputpath, file = os.path.split(filename)
    stem, ext = os.path.splitext(file)
    outputfile = "%s/%s.fits" % (outputdir, stem)
    #### DEBUG: using PIL package
    #holdi = numpy.int16(hold)
    #img = Image.fromarray(holdi)
    #img.convert('L').save(outputfile)
    pyfits.writeto(outputfile, hold)

    # Return
    return 0

############################################################################################################
# manifold_from_vtk_to_img: Function for loading a manifold from DisPerSe file and stores it in VTI/FITS formats
# Note: DisPerSe has a very strage way of storing the data
# Input:
#       filename: String with the path to the VTK file, this file should be generated with DISPERSE and contain
#                 a manifold
#       outputdir   : Directory where the output image will be stored
#       format: currently only 'vti' (default), 'mrc' or 'em' formats are allowed
#       transpose: DisPerSe data used to require being transposed
#       pad: for adding padding pixel in the borde. Format [p1x, p1y, p1z, p2x, p2y, p2z]
# Return:
#        If an error occurs an Exception is raised
#        If outputdir is None the manifold is returned instead being stored
def manifold3d_from_vtu_to_img(filename, outputdir=None, format='vti', transpose=False,
                               pad=[0, 0, 0, 0, 0, 0]):

    if not filename.endswith('vtu'):
        raise pexceptions.PySegInputWarning(expr='numpy_to_vti', msg='Only input format \'vtu\' is allowed.')

    # Read the source file.
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    output = reader.GetOutput()

    # Get the POINTS of the DATASET
    points = output.GetPoints()

    # Transpose
    if transpose is True:
        transpose_poly(output)

    # Get the FIELD (source_index) of the DATASET.
    cell_data = None
    for i in range(output.GetCellData().GetNumberOfArrays()):
        if output.GetCellData().GetArrayName(i) == 'source_index':
            cell_data = output.GetCellData().GetArray(i)
    if cell_data is None:
        error_msg = "Error (2): input file \"%s\" is corrupted.\n" % filename
        raise pexceptions.PySegInputError(expr='manifold3d_from_vtu_to_img', msg=error_msg)

    # Create the image holder.
    points.ComputeBounds()
    ox, nx, oy, ny, oz, nz = points.GetBounds()
    if (nx <= 0) or (ny <= 0) or (nz < 0):
        error_msg = "Error (2): input file \"%s\" is corrupted.\n" % filename
        raise pexceptions.PySegInputError(expr='manifold3d_from_vtu_to_img', msg=error_msg)

    # Paint image holder
    # I think this is redundant because DisPerSe uses Tetra cells for representing the manifolds (I don't know why).
    # Therefore this loop does much redundant work
    nx = int(math.ceil(nx + pad[3]))
    ny = int(math.ceil(ny + pad[4]))
    nz = int(math.ceil(nz + pad[5]))
    px = pad[0] - ox
    if px < 0:
        px = 0
    py = pad[1] - oy
    if py < 0:
        py = 0
    pz = pad[2] - oz
    if pz < 0:
        pz = 0
    if format is 'vti':
        vtk_ver = vtk.vtkVersion().GetVTKVersion()
        image_manifold = vtk.vtkImageData()
        if nz == 0:
            # image_manifold.SetExtent(0, nx-1, 0, ny-1, 0, nz)
            image_manifold.SetExtent(0, nx, 0, ny, 0, nz)
        else:
            image_manifold.SetExtent(0, nx-1, 0, ny-1, 0, nz-1)
        if int(vtk_ver[0]) < 6:
            image_manifold.SetScalarTypeToFloat()
            image_manifold.AllocateScalars()
        else:
            image_manifold.AllocateScalars(vtk.VTK_DOUBLE, 1)
        array_manifold = vtk.vtkDoubleArray()
        # For 2D images
        if nz == 0:
            array_manifold.SetNumberOfTuples((nx+1) * (ny+1))
        else:
            array_manifold.SetNumberOfTuples(nx * ny * nz)
        for i in range(output.GetNumberOfCells()):
            cell = output.GetCell(i)
            cell_points = cell.GetPoints()
            for j in range(cell_points.GetNumberOfPoints()):
                x, y, z = cell_points.GetPoint(j)
                coord = [int(round(x+px)), int(round(y+py)), int(round(z+pz))]
                point_id = image_manifold.ComputePointId(coord)
                if point_id <= array_manifold.GetNumberOfTuples():
                    array_manifold.SetTuple1(image_manifold.ComputePointId(coord), cell_data.GetValue(i))
    else:
        Nx = nx
        Nxy = nx * ny
        hold = numpy.zeros(nx * ny * nz)
        for i in range(output.GetNumberOfCells()):
            cell = output.GetCell(i)
            cell_points = cell.GetPoints()
            for j in range(cell_points.GetNumberOfPoints()):
                x, y, z = cell_points.GetPoint(j)
                x += px
                y += py
                z += pz
                hold[x+y*Nx+z*Nxy] = cell_data.GetValue(i)

    # Save image in disk
    inputpath, file = os.path.split(filename)
    stem, ext = os.path.splitext(file)
    if format is 'vti':
        image_manifold.GetPointData().SetScalars(array_manifold)
        if outputdir is None:
            return image_manifold
        else:
            outputfile = "%s/%s.vti" % (outputdir, stem)
            writer = vtk.vtkXMLImageDataWriter()
            writer.SetFileName(outputfile)
            vtk_ver = vtk.vtkVersion().GetVTKVersion()
            if int(vtk_ver[0]) < 6:
                writer.SetInput(image_manifold)
            else:
                writer.SetInputData(image_manifold)
            if writer.Write() != 1:
                raise pexceptions.PySegInputError(expr='manifold3d_from_vtu_to_img', msg='Error (3) writing the .vti file.')
    elif format is 'mrc':
        image_manifold = ImageIO()
        hold = hold.reshape((nx+1, ny+1, nz+1), order='F')
        image_manifold.setData(data=hold.astype(numpy.float32))
        if outputdir is None:
            return image_manifold
        else:
            outputfile = "%s/%s.mrc" % (outputdir, stem)
            image_manifold.writeMRC(file=outputfile)
    elif format is 'em':
        image_manifold = ImageIO()
        hold = hold.reshape((nx+1, ny+1, nz+1), order='F')
        image_manifold.setData(data=hold.astype(numpy.float32))
        if outputdir is None:
            return image_manifold
        else:
            outputfile = "%s/%s.em" % (outputdir, stem)
            image_manifold.writeEM(file=outputfile)
    else:
        raise pexceptions.PySegInputWarning(expr='manifold3d_from_vtu_to_img', msg='Only \'vti\', \'mrc\' and \'em\' formats are allowed')


############################################################################################################
# mrc_to_vtk: Read an MRC file an save it in VTK format (*.vti)
#
# Input:
#       filename: String with the path to the MRC file
#       outputdir: Directory where the output VTK file will be stored (vti extension is added to the original name)
#
def mrc_to_vtk(filename, outputdir):

    ### Get VTK version
    vtk_ver = vtk.vtkVersion().GetVTKVersion()

    ### Read the source file.
    mrc = ImageIO()
    mrc.read(filename)
    mrc.readMRCHeader()

    ### Convert to VTK format

    # First create VTK holder object
    image = vtk.vtkImageData()
    image.SetDimensions(mrc.mrcHeader[0], mrc.mrcHeader[1], mrc.mrcHeader[2])
    image.SetSpacing(mrc.mrcHeader[0]/mrc.mrcHeader[10], mrc.mrcHeader[1]/mrc.mrcHeader[11], mrc.mrcHeader[2]/mrc.mrcHeader[12])
    if mrc.mrcHeader[3] == 0:
        if int(vtk_ver[0]) < 6:
            image.SetScalarTypeToUnsignedChar()
            image.AllocateScalars()
        else:
            image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    elif mrc.mrcHeader[3] == 1:
        if int(vtk_ver[0]) < 6:
            image.SetScalarTypeToShort()
            image.AllocateScalars()
        else:
            image.AllocateScalars(vtk.VTK_SHORT, 1)
    else:
        if int(vtk_ver[0]) < 6:
            image.SetScalarTypeToFloat()
            image.AllocateScalars()
        else:
            image.AllocateScalars(vtk.VTK_FLOAT, 1)

    # Copy scalars from the MRC data
    array = numpy.ascontiguousarray(mrc.data)
    array = array.reshape(array.size, order='F')
    scalars_mrc = numpy_support.numpy_to_vtk(array)
    image.GetPointData().SetScalars(scalars_mrc)

    ### Save VTK file
    writer = vtk.vtkXMLImageDataWriter()
    stem, ext = os.path.splitext(filename)
    ofname = "%s.vti" % stem
    writer.SetFileName(ofname)
    if int(vtk_ver[0]) < 6:
        writer.SetInput(image)
    else:
        writer.SetInputData(image)
    if writer.Write() != 1:
        raise NameError('Error writting the .vti file!!!')

############################################################################################################
# vti_to_img: Read a .vti image (point data sacalar field) and store it in MRC, EM or FITS
#
# Input:
#       filename: String with the path to the input file
#       outputdir: Directory where the output file will be stored (vti extension is added to the original name)
#       ext: extension of the format that will be used ('mrc', 'em' and 'fits')
#       mode: data storen in; 1- short, otherwise- float
#
def vti_to_img(filename, outputdir, ext, mode=0):

    # Load .vti file
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(filename)
    try:
        reader.Update()
    except:
        error_msg = 'File %s could not be read.' % filename
        raise pexceptions.PySegInputError(expr='vti_to_img', msg=error_msg)
    output = reader.GetOutput()

    # Read tomogram data
    output.ComputeBounds()
    nx, ny, nz = output.GetDimensions()
    scalars = output.GetPointData().GetScalars()
    hold = numpy.zeros([nx, ny, nz])
    for i in range(scalars.GetNumberOfTuples()):
        [x, y, z] = output.GetPoint(i)
        hold[x, y, z] = scalars.GetTuple1(i)

    # Data conversion
    if mode == 1:
        data_type = 'uint16'
        array = hold.astype(dtype=numpy.uint16)
    else:
        data_type = 'float32'
        array = hold.astype(dtype=numpy.float32)

    # Create the output path
    inputpath, file = os.path.split(filename)
    stem, ext_in = os.path.splitext(file)
    outputfile = '%s/%s.%s' % (outputdir, stem, ext)

    # Store the result
    if ext == 'mrc':
        image = ImageIO()
        image.setData(array)
        print outputfile
        try:
            image.writeMRC(outputfile)
        except:
            error_msg = 'File %s could not be writen.' % outputfile
            raise pexceptions.PySegInputError(expr='vti_to_img', msg=error_msg)
    elif ext == 'em':
        image = ImageIO()
        image.setData(array)
        try:
            image.writeEM(outputfile, dataType=data_type)
        except:
           error_msg = 'File %s could not be writen.' % outputfile
           raise pexceptions.PySegInputError(expr='vti_to_img', msg=error_msg)
    elif ext == 'fits':
        try:
            pyfits.writeto(outputfile, array, clobber=True, output_verify='silentfix')
        except:
            error_msg = 'File %s could not be writen.' % outputfile
            raise pexceptions.PySegInputError(expr='vti_to_img', msg=error_msg)
    else:
        error_msg = 'Format %s not writable.' % ext
        raise pexceptions.PySegInputError(expr='vti_to_img', msg=error_msg)

# Gen a surface from a segmented tomogram
# tomo: the input segmentation, a numpy ndarray or a string with the file name to a segmented
#          tomogram, only 'fits', 'mrc', 'em' and 'vti' formats allowed
# lbl: label for the foreground
# mask: If True (default) the input segmentation is used as mask for the surface
# purge_ratio: if greater than 1 (default) then 1 every purge_ratio points of tomo are randomly
# deleted
# field: if True (default False) return the polarity distance scalar field
# verbose: print out messages for checking the progress
# Return: output surface (vtkPolyData)
def gen_surface(tomo, lbl=1, mask=True, purge_ratio=1, field=False, mode_2d=False, verbose=False):

    # Check input format
    if isinstance(tomo, str):
        fname, fext = os.path.splitext(tomo)
        if fext == '.fits':
            tomo = pyfits.getdata(tomo)
        elif fext == '.mrc':
            hold = ImageIO()
            hold.readMRC(file=tomo)
            tomo = hold.data
        elif fext == '.em':
            hold = ImageIO()
            hold.readEM(file=tomo)
            tomo = hold.data
        elif fext == '.vti':
            reader = vtk.vtkXMLImageDataReader()
            reader.SetFileName(tomo)
            reader.Update()
            tomo = vti_to_numpy(reader.GetOutput())
        else:
            error_msg = 'Format %s not readable.' % fext
            raise pexceptions.PySegInputError(expr='gen_surface', msg=error_msg)
    elif not isinstance(tomo, numpy.ndarray):
        error_msg = 'Input must be either a file name or a ndarray.'
        raise pexceptions.PySegInputError(expr='gen_surface', msg=error_msg)

    # Load file with the cloud of points
    nx, ny, nz = tomo.shape
    cloud = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    cloud.SetPoints(points)

    if purge_ratio <= 1:
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    if tomo[x, y, z] == lbl:
                        points.InsertNextPoint(x, y, z)
    else:
        count = 0
        mx_value = purge_ratio - 1
        purge = numpy.random.randint(0, purge_ratio+1, nx*ny*nz)
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    if purge[count] == mx_value:
                        if tomo[x, y, z] == lbl:
                            points.InsertNextPoint(x, y, z)
                    count += 1

    if verbose:
        print 'Cloud of points loaded...'

    # Creating the isosurface
    surf = vtk.vtkSurfaceReconstructionFilter()
    # surf.SetSampleSpacing(2)
    surf.SetSampleSpacing(purge_ratio)
    #surf.SetNeighborhoodSize(10)
    surf.SetInputData(cloud)
    contf = vtk.vtkContourFilter()
    contf.SetInputConnection(surf.GetOutputPort())
    # if thick is None:
    contf.SetValue(0, 0)
    # else:
        # contf.SetValue(0, thick)

    # Sometimes the contouring algorithm can create a volume whose gradient
    # vector and ordering of polygon (using the right hand rule) are
    # inconsistent. vtkReverseSense cures    this problem.
    reverse = vtk.vtkReverseSense()
    reverse.SetInputConnection(contf.GetOutputPort())
    reverse.ReverseCellsOn()
    reverse.ReverseNormalsOn()
    reverse.Update()
    rsurf = reverse.GetOutput()

    if verbose:
        print 'Isosurfaces generated...'

    # Translate and scale to the proper positions
    cloud.ComputeBounds()
    rsurf.ComputeBounds()
    xmin, xmax, ymin, ymax, zmin, zmax = cloud.GetBounds()
    rxmin, rxmax, rymin, rymax, rzmin, rzmax = rsurf.GetBounds()
    scale_x = (xmax-xmin) / (rxmax-rxmin)
    scale_y = (ymax-ymin) / (rymax-rymin)
    denom = rzmax - rzmin
    num = zmax - xmin
    if (denom == 0) or (num == 0):
        scale_z = 1
    else:
        scale_z = (zmax-zmin) / (rzmax-rzmin)
    transp = vtk.vtkTransform()
    transp.Translate(xmin, ymin, zmin)
    transp.Scale(scale_x, scale_y, scale_z)
    transp.Translate(-rxmin, -rymin, -rzmin)
    tpd = vtk.vtkTransformPolyDataFilter()
    tpd.SetInputData(rsurf)
    tpd.SetTransform(transp)
    tpd.Update()
    tsurf = tpd.GetOutput()

    if verbose:
        print 'Rescaled and translated...'

    # Masking according to distance to the original segmentation
    if mask:
        tomod = scipy.ndimage.morphology.distance_transform_edt(numpy.invert(tomo.astype(numpy.bool)))
        for i in range(tsurf.GetNumberOfCells()):

            # Check if all points which made up the polygon are in the mask
            points_cell = tsurf.GetCell(i).GetPoints()
            count = 0
            for j in range(0, points_cell.GetNumberOfPoints()):
                x, y, z = points_cell.GetPoint(j)
                if tomod[int(round(x)), int(round(y)), int(round(z))] > MAX_DIST_SURF:
                    count += 1

            if count > 0:
                tsurf.DeleteCell(i)

        # Release free memory
        tsurf.RemoveDeletedCells()

        if verbose:
            print 'Mask applied...'

        # Field distance
    if field:

        # Get normal attributes
        norm_flt = vtk.vtkPolyDataNormals()
        norm_flt.SetInputData(tsurf)
        norm_flt.ComputeCellNormalsOn()
        norm_flt.AutoOrientNormalsOn()
        norm_flt.ConsistencyOn()
        norm_flt.Update()
        tsurf = norm_flt.GetOutput()
        # for i in range(tsurf.GetPointData().GetNumberOfArrays()):
        #    array = tsurf.GetPointData().GetArray(i)
        #    if array.GetNumberOfComponents() == 3:
        #        break
        array = tsurf.GetCellData().GetNormals()

        # Build membrane mask
        tomoh = numpy.ones(shape=tomo.shape, dtype=numpy.bool)
        tomon = numpy.ones(shape=(tomo.shape[0], tomo.shape[1], tomo.shape[2], 3),
                           dtype=TypesConverter().vtk_to_numpy(array))
        # for i in range(tsurf.GetNumberOfCells()):
        #     points_cell = tsurf.GetCell(i).GetPoints()
        #     for j in range(0, points_cell.GetNumberOfPoints()):
        #         x, y, z = points_cell.GetPoint(j)
        #         # print x, y, z, array.GetTuple(j)
        #         x, y, z = int(round(x)), int(round(y)), int(round(z))
        #         tomo[x, y, z] = False
        #         tomon[x, y, z, :] = array.GetTuple(j)
        for i in range(tsurf.GetNumberOfCells()):
            points_cell = tsurf.GetCell(i).GetPoints()
            for j in range(0, points_cell.GetNumberOfPoints()):
                x, y, z = points_cell.GetPoint(j)
                # print x, y, z, array.GetTuple(j)
                x, y, z = int(round(x)), int(round(y)), int(round(z))
                if tomo[x, y, z] == lbl:
                    tomoh[x, y, z] = False
                    tomon[x, y, z, :] = array.GetTuple(i)

        # Distance transform
        tomod, ids = scipy.ndimage.morphology.distance_transform_edt(tomoh, return_indices=True)

        # Compute polarity
        if mode_2d:
            for x in range(nx):
                for y in range(ny):
                    for z in range(nz):
                        i_x, i_y, i_z = ids[0, x, y, z], ids[1, x, y, z], ids[2, x, y, z]
                        norm = tomon[i_x, i_y, i_z]
                        norm[2] = 0
                        pnorm = (i_x, i_y, 0)
                        p = (x, y, 0)
                        dprod = dot_norm(numpy.asarray(p, dtype=numpy.float),
                                         numpy.asarray(pnorm, dtype=numpy.float),
                                         numpy.asarray(norm, dtype=numpy.float))
                        tomod[x, y, z] = tomod[x, y, z] * numpy.sign(dprod)
        else:
            for x in range(nx):
                for y in range(ny):
                    for z in range(nz):
                        i_x, i_y, i_z = ids[0, x, y, z], ids[1, x, y, z], ids[2, x, y, z]
                        hold_norm = tomon[i_x, i_y, i_z]
                        norm = hold_norm
                        # norm[0] = (-1) * hold_norm[1]
                        # norm[1] = hold_norm[0]
                        # norm[2] = hold_norm[2]
                        pnorm = (i_x, i_y, i_z)
                        p = (x, y, z)
                        dprod = dot_norm(numpy.asarray(pnorm, dtype=numpy.float),
                                         numpy.asarray(p, dtype=numpy.float),
                                         numpy.asarray(norm, dtype=numpy.float))
                        tomod[x, y, z] = tomod[x, y, z] * numpy.sign(dprod)

        if verbose:
            print 'Distance field generated...'

        return tsurf, tomod

    if verbose:
        print 'Finished!'

    return tsurf

# TODO: Version for testing
# Generates a surface from a segmented tomogram
# tomo: the input segmentation, a numpy ndarray or a string with the file name to a segmented
#          tomogram, only 'fits', 'mrc', 'em' and 'vti' formats allowed
# lbl: label for the foreground
# purge_ratio: if greater than 1 (default) then 1 every purge_ratio points of tomo are randomly
# deleted
# cloud: if True (default False) a vtkPolyData with the cloud of points and normals is also returned
# verbose: print out messages for checking the progress
# Return: output surface (vtkPolyData), output cloud if cloud=True (vtkPolyData)
def gen_surface_cloud(tomo, lbl=1, purge_ratio=1, cloud=False, verbose=False):

    if verbose:
        print 'Running disperse_io.get_surface_test:'
        if isinstance(tomo, str):
            print '\tinput file: ' + tomo
        else:
            print '\tndarray'
        print '\tlbl: ' + str(lbl)
        print '\tpurge_ratio: ' + str(purge_ratio)
        print '\tcloud: ' + str(cloud)
        print ''

    # Check input format
    if isinstance(tomo, str):
        tomo = load_tomo(tomo)
    elif not isinstance(tomo, numpy.ndarray):
        error_msg = 'Input must be either a file name or a ndarray.'
        raise pexceptions.PySegInputError(expr='gen_surface_cloud', msg=error_msg)

    # Load file with the cloud of points
    nx, ny, nz = tomo.shape
    cloud = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    cloud.SetPoints(points)
    if purge_ratio <= 1:
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    if tomo[x, y, z] == lbl:
                        points.InsertNextPoint(x, y, z)
    else:
        count = 0
        mx_value = purge_ratio - 1
        purge = numpy.random.randint(0, purge_ratio+1, nx*ny*nz)
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    if purge[count] == mx_value:
                        if tomo[x, y, z] == lbl:
                            points.InsertNextPoint(x, y, z)
                    count += 1

    if verbose:
        print 'Cloud of points loaded...'

    # Creating the isosurface
    surf = vtk.vtkSurfaceReconstructionFilter()
    surf.SetSampleSpacing(purge_ratio)
    surf.SetNeighborhoodSize(10)
    surf.SetInputData(cloud)
    contf = vtk.vtkContourFilter()
    contf.SetInputConnection(surf.GetOutputPort())
    # if thick is None:
    contf.SetValue(0, 0)
    # else:
        # contf.SetValue(0, thick)

    # Sometimes the contouring algorithm can create a volume whose gradient
    # vector and ordering of polygon (using the right hand rule) are
    # inconsistent. vtkReverseSense cures    this problem.
    reverse = vtk.vtkReverseSense()
    reverse.SetInputConnection(contf.GetOutputPort())
    reverse.ReverseCellsOn()
    reverse.ReverseNormalsOn()
    reverse.Update()
    rsurf = reverse.GetOutput()

    if verbose:
        print 'Isosurfaces generated...'

    # Translate and scale to the proper positions
    cloud.ComputeBounds()
    rsurf.ComputeBounds()
    xmin, xmax, ymin, ymax, zmin, zmax = cloud.GetBounds()
    rxmin, rxmax, rymin, rymax, rzmin, rzmax = rsurf.GetBounds()
    scale_x = (xmax-xmin) / (rxmax-rxmin)
    scale_y = (ymax-ymin) / (rymax-rymin)
    denom = rzmax - rzmin
    num = zmax - xmin
    if (denom == 0) or (num == 0):
        scale_z = 1
    else:
        scale_z = (zmax-zmin) / (rzmax-rzmin)
    transp = vtk.vtkTransform()
    transp.Translate(xmin, ymin, zmin)
    transp.Scale(scale_x, scale_y, scale_z)
    transp.Translate(-rxmin, -rymin, -rzmin)
    tpd = vtk.vtkTransformPolyDataFilter()
    tpd.SetInputData(rsurf)
    tpd.SetTransform(transp)
    tpd.Update()
    tsurf = tpd.GetOutput()

    if verbose:
        print 'Rescaled and translated...'

    # Masking according to distance to the original segmentation
    tomod = scipy.ndimage.morphology.distance_transform_edt(numpy.invert(tomo.astype(numpy.bool)))
    for i in range(tsurf.GetNumberOfCells()):

        # Check if all points which made up the polygon are in the mask
        points_cell = tsurf.GetCell(i).GetPoints()
        count = 0
        npoints = points_cell.GetNumberOfPoints()
        for j in range(0, npoints):
            x, y, z = points_cell.GetPoint(j)
            if tomod[int(round(x)), int(round(y)), int(round(z))] < MAX_DIST_SURF:
                count += 1

        if count < npoints:
            tsurf.DeleteCell(i)

    # Release free memory
    tsurf.RemoveDeletedCells()

    if verbose:
        print 'Mask applied...'

    if cloud:

        # Compute the normals
        norm_flt = vtk.vtkPolyDataNormals()
        norm_flt.SetInputData(tsurf)
        norm_flt.ComputePointNormalsOff()
        norm_flt.ComputeCellNormalsOn()
        # Optional parameters
        norm_flt.SetFeatureAngle(30)
        norm_flt.SplittingOff()
        norm_flt.ConsistencyOn()
        norm_flt.AutoOrientNormalsOn()
        norm_flt.NonManifoldTraversalOn()
        norm_flt.Update()
        tsurf = norm_flt.GetOutput()

        points = vtk.vtkPoints()
        verts = vtk.vtkCellArray()
        array = tsurf.GetCellData().GetNormals()
        array_cloud = vtk.vtkFloatArray()
        array_cloud.SetNumberOfComponents(array.GetNumberOfComponents())
        for i in range(tsurf.GetNumberOfCells()):
            cell = tsurf.GetCell(i)
            cell_points = cell.GetPoints()
            npoints = cell.GetNumberOfPoints()
            (x_a, y_a, z_a) = (0, 0, 0)
            # Get the averaged point
            for j in range(npoints):
                x, y, z = cell_points.GetPoint(j)
                x_a += x
                y_a += y
                z_a += z
            point_id = points.InsertNextPoint(x_a/npoints, y_a/npoints, z_a/npoints)
            verts.InsertNextCell(1)
            verts.InsertCellPoint(point_id)
            array_cloud.InsertNextTuple(array.GetTuple(i))
        cloud_poly = vtk.vtkPolyData()
        cloud_poly.SetPoints(points)
        cloud_poly.SetVerts(verts)
        array_cloud.SetName(STR_CLOUD_NORMALS)
        # cloud_poly.GetCellData().AddArray(array_cloud)
        cloud_poly.GetPointData().AddArray(array_cloud)

        if verbose:
            print 'Surface and points cloud successfully generated!'

        return tsurf, cloud_poly

    if verbose:
        print 'Surface successfully generated!'

    return tsurf

# Gaussian smoothing vector field
# could: vtp file with a cloud of points and a vector field called STR_CLOUD_NORMALS as
# attribute
# sigma: gaussian smoothing factor
# size: 3-tuple with the dimensions of the output array if None (default) the bounds of the
#       input poly
def gauss_smooth_vfield(cloud, sigma, size=None):

    # Build holding 3-channels tomogram
    tomo_in = vtp_to_numpy(cloud, size, STR_CLOUD_NORMALS)
    if (len(tomo_in.shape) != 4) or (tomo_in.shape[3] != 3):
        error_msg = 'Input vtkPolyData has no vector field called ' + STR_CLOUD_NORMALS + '.'
        raise pexceptions.PySegInputError(expr='gauss_smooth_vfield', msg=error_msg)
    if tomo_in.dtype is not np.float64:
        tomo_in = tomo_in.astype(np.float64)

    # Gaussian filtration to each channel separatedly
    tomo_x = sp.ndimage.filters.gaussian_filter(tomo_in[:, :, :, 0], sigma)
    tomo_y = sp.ndimage.filters.gaussian_filter(tomo_in[:, :, :, 1], sigma)
    tomo_z = sp.ndimage.filters.gaussian_filter(tomo_in[:, :, :, 2], sigma)
    del tomo_in

    # Updating vector field attribute
    prop_array = None
    for i in range(cloud.GetPointData().GetNumberOfArrays()):
        if cloud.GetPointData().GetArrayName(i) == STR_CLOUD_NORMALS:
            prop_array = cloud.GetPointData().GetArray(i)
            break
    for i in range(cloud.GetNumberOfPoints()):
        p = cloud.GetPoint(i)
        v_x = trilin3d(tomo_x, p)
        v_y = trilin3d(tomo_y, p)
        v_z = trilin3d(tomo_z, p)
        prop_array.SetTuple3(i, v_x, v_y, v_z)

# Generates a volume (ndarray) with the signed distance to the closest point in the input
# cloud set (it must have signed normals)
# cloud: vtkPolyData with cloud of points
# dims=(nx, ny, nz): dimensions for the output volume
def signed_dist_cloud(cloud, dims):

    # Create a KDTree of the input points
    point_tree = vtk.vtkKdTreePointLocator()
    point_tree.SetDataSet(cloud)
    point_tree.BuildLocator()

    # Get normals
    normals = None
    for i in range(cloud.GetPointData().GetNumberOfArrays()):
        if cloud.GetPointData().GetArrayName(i) == STR_CLOUD_NORMALS:
            normals = cloud.GetPointData().GetArray(i)
            break
    if normals is None:
        error_msg = 'The cloud point set does not contain normals.'
        raise pexceptions.PySegInputError(expr='signed_dist_cloud', msg=error_msg)

    # Distance measure for the whole tomogram
    dist_tomo = numpy.zeros(shape=dims, dtype=numpy.float)
    for x in range(dims[0]):
        for y in range(dims[1]):
            for z in range(dims[2]):
                # Find closest point
                point = (x, y, z)
                cpoint_id = point_tree.FindClosestPoint(point)
                cpoint = cloud.GetPoint(cpoint_id)
                # Get normal
                normal = normals.GetTuple(cpoint_id)
                # Measure distance
                dist = vtk.vtkPlane.DistanceToPlane(point, normal, cpoint)
                vect = numpy.zeros(shape=3, dtype=np.float)
                vtk.vtkMath.Subtract(point, cpoint, vect)
                if vtk.vtkMath.Dot(vect, normal) > .0:
                    dist_tomo[x, y, z] = dist
                else:
                    dist_tomo[x, y, z] = (-1) * dist

    return dist_tomo

# Distance transform for a segmentation
# seg: binary segmentation (ndarray, 1 (or True) is fg)
def seg_dist_trans(seg):

    if seg.dtype != numpy.bool:
        tomod = scipy.ndimage.morphology.distance_transform_edt(numpy.invert(seg.astype(numpy.bool)))
    else:
        tomod = scipy.ndimage.morphology.distance_transform_edt(numpy.invert(seg))

    return tomod

# Load tomogram in numpy format
# fname: full path to the tomogram
# mmap: if True (default False) a numpy.memmap object is loaded instead of numpy.ndarray, which means that data
#       are not loaded completely to memory, this is useful only for very large tomograms. Only valid with formats
#       MRC and EM.
#       VERY IMPORTANT: This subclass of ndarray has some unpleasant interaction with some operations,
#       because it does not quite fit properly as a subclass of numpy.ndarray
def load_tomo(fname, mmap=False):
    '''
    Load tomogram in disk in numpy format (valid formats: .rec, .mrc, .em, .vti and .fits)
    :param fname: full path to the tomogram
    :param mmap: if True (default False) a numpy.memmap object is loaded instead of numpy.ndarray, which means that data
        are not loaded completely to memory, this is useful only for very large tomograms. Only valid with formats
        MRC and EM.
        VERY IMPORTANT: This subclass of ndarray has some unpleasant interaction with some operations,
        because it does not quite fit properly as a subclass of numpy.ndarray
    :return: numpy array
    '''

    # Input parsing
    stem, ext = os.path.splitext(fname)
    if mmap and (not ((ext == '.mrc') or (ext == '.rec') or (ext == '.em') or (ext == '.vti') or (ext == '.fits'))):
        error_msg = 'mmap option is only valid for .mrc, .rec, .em and .fits formats, current ' + ext
        raise pexceptions.PySegInputError(expr='load_tomo', msg=error_msg)

    if ext == '.fits':
        im_data = pyfits.getdata(fname).transpose()
    elif (ext == '.mrc') or (ext == '.rec'):
        image = ImageIO()
        if mmap:
            image.readMRC(fname, memmap=mmap)
        else:
            image.readMRC(fname)
        im_data = image.data
    elif ext == '.em':
        image = ImageIO()
        if mmap:
            image.readEM(fname, memmap=mmap)
        else:
            image.readEM(fname)
        im_data = image.data
    elif ext == '.vti':
        stem, sfname = os.path.split(fname)
        im_data = load_vti(sfname, stem)
    else:
        error_msg = '%s is non valid format.' % ext
        raise pexceptions.PySegInputError(expr='load_tomo', msg=error_msg)

    # For avoiding 2D arrays
    if len(im_data.shape) == 2:
        im_data = numpy.reshape(im_data, (im_data.shape[0], im_data.shape[1], 1))

    return im_data

# Load data from VTK PolyData file called fname
def load_poly(fname):

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(fname)
    reader.Update()

    return reader.GetOutput()

def load_vti(fname, in_dir):
    '''
    Load a vtkImageData objecte froma a .vti file
    :param fname: input filename
    :param in_dir: input directory
    :return: the loaded vtkImageData
    '''

    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(in_dir + '/' + fname)
    reader.Update()

    return reader.GetOutput()

# Store data from VTK PolyData file called fname
def save_vtp(poly, fname):

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(fname)
    writer.SetInputData(poly)
    if writer.Write() != 1:
        error_msg = 'Error writing the file %s.' % fname
        raise pexceptions.PySegInputError(expr='save_vtp', msg=error_msg)

# No return, the procedure is directly applied over input poly's points
def transpose_poly(poly):

    # Points permutation loop
    points = poly.GetPoints()
    for k in range(0, points.GetNumberOfPoints()):
        x, y, z = points.GetPoint(k)
        points.SetPoint(k, z, y, x)

def numpy_to_vti(array, spacing=[1, 1, 1]):
    '''
    Converts a 3D numpy array into vtkImageData object
    :param array: 3D numpy array
    :param spacing: distance between pixels
    :return: a vtkImageData object
    '''

    # Flattern the input array
    array_1d = numpy_support.numpy_to_vtk(num_array=array.ravel(order='F'), deep=True,
                                          array_type=vtk.VTK_FLOAT)

    # Create the new vtkImageData
    nx, ny, nz = array.shape
    image = vtk.vtkImageData()
    image.SetSpacing(spacing)
    image.SetDimensions(nx, ny, nz)
    image.AllocateScalars(vtk.VTK_FLOAT, 1)
    image.GetPointData().SetScalars(array_1d)

    return image

def add_numpy_to_vti(vti, array, name=None):
    '''
    Add the values of 3D numpy array as scalar filed to a vtkImageData object
    :param vti: vtkImageData object
    :param array: 3D numpy array
    :param name: name assigned to the scalar field
    :return: None (a new scalar field is added to vti)
    '''

    # Creating the new scalar field
    sfield = vtk.vtkFloatArray()
    if name is not None:
        sfield.SetName(name)
    sfield.SetNumberOfComponents(1)

    # Flattern the input array
    array_1d = numpy_support.numpy_to_vtk(num_array=array.ravel(order='F'), deep=True,
                                          array_type=vtk.VTK_FLOAT)

    # Addting the new scalar field
    vti.GetPointData().AddArray(vti)

def save_vti(image, fname, outputdir):

    writer = vtk.vtkXMLImageDataWriter()
    outputfile = outputdir + '/' + fname
    writer.SetFileName(outputfile)
    writer.SetInputData(image)
    if writer.Write() != 1:
        error_msg = 'Error writing the %s file on %s.' % fname, outputdir
        raise pexceptions.PySegInputError(expr='save_vti', msg=error_msg)

# Input vti must be a scalar field
def vti_to_numpy(image, transpose=True):

    # Read tomogram data
    image.ComputeBounds()
    nx, ny, nz = image.GetDimensions()
    scalars = image.GetPointData().GetScalars()
    if transpose:
        dout = numpy.zeros(shape=[nz, ny, nx], dtype=TypesConverter.vtk_to_numpy(scalars))
        for i in range(scalars.GetNumberOfTuples()):
            [x, y, z] = image.GetPoint(i)
            dout[int(z), int(y), int(x)] = scalars.GetTuple1(i)
    else:
        dout = numpy.zeros(shape=[nx, ny, nz], dtype=TypesConverter.vtk_to_numpy(scalars))
        for i in range(scalars.GetNumberOfTuples()):
            [x, y, z] = image.GetPoint(i)
            dout[int(x), int(y), int(z)] = scalars.GetTuple1(i)

    return dout

# Generates a numpy 3D dense array (tomogram) from the geometry of an input vtkPolyData
# poly: input vtkPolyData
# size: 3-tuple with the dimensions of the output array if None (default) the bounds of the
#       input poly
# key_prop: if None (default) poly points are set to 1, otherwise it is the key string of the
#           point property used
def vtp_to_numpy(poly, size=None, key_prop=None):

    # Initialization
    prop_array = None
    n_comp = 1
    for i in range(poly.GetPointData().GetNumberOfArrays()):
        if poly.GetPointData().GetArrayName(i) == key_prop:
            prop_array = poly.GetPointData().GetArray(i)
            break
    if size is None:
        poly.ComputeBounds()
        ox, nx, oy, ny, oz, nz = poly.GetBounds()
        size = (int(nx-ox+1), int(ny-oy+1), int(nz-oz+1))
    if prop_array is None:
        tomo = numpy.zeros(shape=size, dtype=np.int8)
    else:
        n_comp = prop_array.GetNumberOfComponents()
        if n_comp > 1:
            tomo_shape = (size[0], size[1], size[2], n_comp)
            tomo = numpy.zeros(shape=tomo_shape,
                               dtype=TypesConverter().vtk_to_numpy(prop_array))
        else:
            tomo = numpy.zeros(shape=size, dtype=TypesConverter().vtk_to_numpy(prop_array))

    # Loop for setting the points
    if prop_array is None:
        for i in range(poly.GetNumberOfPoints()):
            x, y, z = poly.GetPoint(i)
            x, y, z = int(round(x)), int(round(y)), int(round(z))
            tomo[x, y, z] = 1
    else:
        if n_comp > 1:
            for i in range(poly.GetNumberOfPoints()):
                x, y, z = poly.GetPoint(i)
                x, y, z = int(round(x)), int(round(y)), int(round(z))
                t = prop_array.GetTuple(i)
                for j in range(n_comp):
                    tomo[x, y, z, j] = t[j]
        else:
            for i in range(poly.GetNumberOfPoints()):
                x, y, z = poly.GetPoint(i)
                x, y, z = int(round(x)), int(round(y)), int(round(z))
                tomo[x, y, z] = prop_array.GetTuple1(i)

    return tomo

def save_numpy(array, fname):

    _, ext = os.path.splitext(fname)

    # Parse input array for fulfilling format requirements
    if (ext == '.mrc') or (ext == '.em'):
        if (array.dtype != 'ubyte') and (array.dtype != 'int16') and (array.dtype != 'float32'):
            array = array.astype('float32')
        # if (len(array.shape) == 3) and (array.shape[2] == 1):
        #   array = numpy.reshape(array, (array.shape[0], array.shape[1]))

    if ext == '.vti':
        pname, fnameh = os.path.split(fname)
        save_vti(numpy_to_vti(array), fnameh, pname)
    elif ext == '.fits':
        warnings.resetwarnings()
        warnings.filterwarnings('ignore', category=UserWarning, append=True)
        pyfits.writeto(fname, array, clobber=True, output_verify='silentfix')
        warnings.resetwarnings()
        warnings.filterwarnings('always', category=UserWarning, append=True)
    elif ext == '.mrc':
        img = ImageIO()
        # img.setData(numpy.transpose(array, (1, 0, 2)))
        img.setData(array)
        img.writeMRC(fname)
    elif ext == '.em':
        img = ImageIO()
        # img.setData(numpy.transpose(array, (1, 0, 2)))
        img.setData(array)
        img.writeEM(fname)
    else:
        error_msg = 'Format not valid %s.' % ext
        raise pexceptions.PySegInputError(expr='save_numpy', msg=error_msg)

# Crop a volume to fit a segmentation
# vol: input tomogram
# seg: segmentation tomogram
# lbl: segmentation label (default 1)
# off: offset voxels for the subvolume (default 0)
# Return: cropped suvolume, cropped segmentation, top-left subvolume coordinates
def crop_lbl_tomo(vol, seg, lbl=1, off=0):

    # Initialization
    if vol.shape != seg.shape:
        error_msg = 'Volume and its segmentation must have the same shape'
        raise pexceptions.PySegInputError(expr='crop_lbl_tomo', msg=error_msg)

    # Compute segmentation extension
    ids_x, ids_y, ids_z = numpy.where(seg == lbl)
    l_x, u_x = int(ids_x.min()-off), int(ids_x.max()+off)
    l_y, u_y = int(ids_y.min()-off), int(ids_y.max()+off)
    l_z, u_z = int(ids_z.min()-off), int(ids_z.max()+off)
    if l_x < 0:
        l_x = 0
    if u_x > (seg.shape[0]-1):
        u_x = seg.shape[0] - 1
    if l_y < 0:
        l_y = 0
    if u_y > (seg.shape[1]-1):
        u_y = seg.shape[1] - 1
    if l_z < 0:
        l_z = 0
    if u_z > (seg.shape[2]-1):
        u_z = seg.shape[2] - 1

    # Cropping subvolumes
    svol = vol[l_x:u_x, l_y:u_y, l_z:u_z]
    sseg = seg[l_x:u_x, l_y:u_y, l_z:u_z]

    return svol, sseg, (l_x, l_y, l_z)

#####################################################################################
# Subprocess which can be parallelized
#
#

# For slicing the measure of the signed distance
# args =
def find_sign_dist_slice(args):

    # Unpack args
    (slice, tomo, tree, cloud, normals) = args

    dims = tomo.shape
    for x in range(dims[0]):
        for y in range(dims[1]):
            # Find closest point
            point = (x, y, slice)
            cpoint_id = tree.FindClosestPoint(point)
            cpoint = cloud.GetPoint(cpoint_id)
            # Get normal
            normal = normals.GetTuple(cpoint_id)
            # Measure distance
            dist = vtk.vtkPlane.DistanceToPlane(point, normal, cpoint)
            vect = numpy.zeros(shape=3, dtype=np.float)
            vtk.vtkMath.Subtract(point, cpoint, vect)
            if vtk.vtkMath.Dot(vect, normal) > .0:
                tomo[x, y, slice] = dist
            else:
                tomo[x, y, slice] = (-1) * dist


#####################################################################################
# Static class for converting types between the diffrent libraries: Numpy, VTK and Graph-tool
# In general if types does not match exactly data are upcasted
# TODO: Revise the whole package before 04.09.2014 for using this class instead manual conv.
#
#
class TypesConverter(object):

    # Creates a vtk array object from the equivalent numpy data type
    @staticmethod
    def numpy_to_vtk(din):

        # Check than a type object is passed
        if (not isinstance(din, type)) and (not isinstance(din, np.dtype)):
            error_msg = 'type object required as input.' % din
            raise pexceptions.PySegInputError(expr='numpy_to_vtk_array (TypesConverter)', msg=error_msg)

        if din == numpy.bool:
            return vtk.vtkBitArray()
        elif din == numpy.int:
            return vtk.vtkIntArray()
        elif din == numpy.int8:
            return vtk.vtkTypeInt8Array()
        elif din == numpy.int16:
            return vtk.vtkTypeInt16Array()
        elif din == numpy.int32:
            return vtk.vtkTypeInt32Array()
        elif din == numpy.int64:
            return vtk.vtkTypeInt64Array()
        elif din == numpy.uint8:
            return vtk.vtkTypeUInt8Array()
        elif din == numpy.uint16:
            return vtk.vtkTypeUInt16Array()
        elif din == numpy.uint32:
            return vtk.vtkTypeUInt32Array()
        elif din == numpy.uint64:
            return vtk.vtkTypeUInt64Array()
        elif din == numpy.float32:
            return vtk.vtkFloatArray()
        elif din == numpy.float64:
            return vtk.vtkDoubleArray()
        else:
            error_msg = 'Numpy type not identified'
            raise pexceptions.PySegInputError(expr='numpy_to_vtk_array (TypesConverter)', msg=error_msg)

    # Creates a numpy data type object from the equivalent vtk object
    @staticmethod
    def vtk_to_numpy(din):

        # Check than a type object is passed
        if not isinstance(din, vtk.vtkDataArray):
            error_msg = 'vtkDataArray object required as input.' % din
            raise pexceptions.PySegInputError(expr='vtk_to_numpy (TypesConverter)', msg=error_msg)

        if isinstance(din, vtk.vtkBitArray):
            return numpy.bool
        elif isinstance(din, vtk.vtkIntArray) or isinstance(din, vtk.vtkTypeInt32Array):
            return numpy.int
        elif isinstance(din, vtk.vtkTypeInt8Array):
            return numpy.int8
        elif isinstance(din, vtk.vtkTypeInt16Array):
            return numpy.int16
        elif isinstance(din, vtk.vtkTypeInt64Array):
            return numpy.int64
        elif isinstance(din, vtk.vtkTypeUInt8Array):
            return numpy.uint8
        elif isinstance(din, vtk.vtkTypeUInt16Array):
            return numpy.uint16
        elif isinstance(din, vtk.vtkTypeUInt32Array):
            return numpy.uint32
        elif isinstance(din, vtk.vtkTypeUInt64Array):
            return numpy.uint64
        elif isinstance(din, vtk.vtkFloatArray) or isinstance(din, vtk.vtkTypeFloat32Array):
            return numpy.float32
        elif isinstance(din, vtk.vtkDoubleArray) or isinstance(din, vtk.vtkTypeFloat64Array):
            return numpy.float64
        else:
            error_msg = 'VTK type not identified.' % din
            raise pexceptions.PySegInputError(expr='numpy_to_vtk_array (TypesConverter)', msg=error_msg)

    # Convert a numpy type into the equivalent graph tool alias
    # array: if true, equivalent array type is used
    @staticmethod
    def numpy_to_gt(din, array=False):

        # Check than a type object is passed
        if (not isinstance(din, type)) and (not isinstance(din, numpy.dtype)):
            error_msg = 'type object required as input.'
            raise pexceptions.PySegInputError(expr='numpy_to_gt (TypesConverter)',
                                                    msg=error_msg)

        if array is False:
            if din == numpy.bool:
                return 'uint8_t'
            elif (din == numpy.int8) or (din == numpy.int16) or (din == numpy.uint8):
                return 'short'
            elif (din == numpy.int32) or (din == numpy.int) or (din == numpy.uint16):
                return 'int'
            elif (din == numpy.int64) or (din == numpy.uint32):
                return 'long'
            elif (din == numpy.float) or (din == numpy.float32) or (din == numpy.float64):
                return 'float'
            else:
                if din.name == 'bool':
                    return 'uint8_t'
                elif (din.name == 'int8') or (din.name == 'int16') or (din.name == 'uint8'):
                    return 'short'
                elif (din.name == 'int32') or (din.name == 'int') or (din.name == 'uint16'):
                    return 'int'
                elif (din.name == 'int64') or (din.name == 'uint32'):
                    return 'long'
                elif (din.name == 'float') or (din.name == 'float32') or (din.name == 'float64'):
                    return 'float'
                else:
                    error_msg = 'Numpy type not identified. Objects are not accepted.' % din
                    raise pexceptions.PySegInputError(expr='numpy_to_gt (TypesConverter)', msg=error_msg)
        else:
            if din == numpy.bool:
                return 'vector<uint8_t>'
            elif (din == numpy.int8) or (din == numpy.int16) or (din == numpy.uint8):
                return 'vector<short>'
            elif (din == numpy.int32) or (din == numpy.int) or (din == numpy.uint16):
                return 'vector<int>'
            elif (din == numpy.int64) or (din == numpy.uint32):
                return 'vector<long>'
            elif (din == numpy.float) or (din == numpy.float32) or (din == numpy.float64):
                return 'vector<float>'
            else:
                error_msg = 'Numpy type not identified. Objects are not accepted.' % din
                raise pexceptions.PySegInputError(expr='numpy_to_gt (TypesConverter)', msg=error_msg)

    # From the graph tool string alias return the numpy data type
    @staticmethod
    def gt_to_numpy(din):

        # Check than a type object is passed
        if not isinstance(din, str):
            error_msg = 'str object required as input.' % din
            raise pexceptions.PySegInputError(expr='gt_to_numpy (TypesConverter)', msg=error_msg)

        if (din == 'uint8_t') or (din == 'vector<uint8_t>'):
            return numpy.bool
        elif (din == 'short') or (din == 'vector<short>'):
            return numpy.int16
        elif (din == 'int') or (din == 'vector<int>'):
            return numpy.int32
        elif (din == 'long') or (din == 'vector<long>'):
            return numpy.int64
        elif (din == 'float') or (din == 'vector<float>'):
            return numpy.float
        else:
            error_msg = 'Graph tool alias not identified.' % din
            raise pexceptions.PySegInputError(expr='gt_to_numpy (TypesConverter)', msg=error_msg)

    # From vtk data array object to graph tool string alias
    @staticmethod
    def vtk_to_gt(din, array=False):

        # Check than a type object is passed
        if not isinstance(din, vtk.vtkDataArray):
            error_msg = 'vtkDataArray object required as input.' % din
            raise pexceptions.PySegInputError(expr='vtk_to_gt (TypesConverter)', msg=error_msg)

        if din.GetNumberOfComponents() == 1:
            if isinstance(din, vtk.vtkBitArray):
                return 'uint8_t'
            elif isinstance(din, vtk.vtkIntArray) or isinstance(din, vtk.vtkTypeInt32Array):
                return 'int'
            elif isinstance(din, vtk.vtkTypeInt8Array):
                return 'short'
            elif isinstance(din, vtk.vtkTypeInt16Array):
                return 'short'
            elif isinstance(din, vtk.vtkTypeInt64Array):
                return 'long'
            elif isinstance(din, vtk.vtkTypeUInt8Array):
                return 'short'
            elif isinstance(din, vtk.vtkTypeUInt16Array):
                return 'int'
            elif isinstance(din, vtk.vtkTypeUInt32Array):
                return 'long'
            elif isinstance(din, vtk.vtkTypeUInt64Array):
                return 'float'
            elif isinstance(din, vtk.vtkFloatArray) or isinstance(din, vtk.vtkTypeFloat32Array):
                return 'float'
            elif isinstance(din, vtk.vtkDoubleArray) or isinstance(din, vtk.vtkTypeFloat64Array):
                return 'float'
            else:
                error_msg = 'VTK type not identified.' % din
                raise pexceptions.PySegInputError(expr='vtk_to_gt (TypesConverter)', msg=error_msg)
        else:
            if isinstance(din, vtk.vtkBitArray):
                return 'vector<uint8_t>'
            elif isinstance(din, vtk.vtkIntArray) or isinstance(din, vtk.vtkTypeInt32Array):
                return 'vector<int>'
            elif isinstance(din, vtk.vtkTypeInt8Array):
                return 'vector<short>'
            elif isinstance(din, vtk.vtkTypeInt16Array):
                return 'vector<short>'
            elif isinstance(din, vtk.vtkTypeInt64Array):
                return 'vector<long>'
            elif isinstance(din, vtk.vtkTypeUInt8Array):
                return 'vector<short>'
            elif isinstance(din, vtk.vtkTypeUInt16Array):
                return 'vector<int>'
            elif isinstance(din, vtk.vtkTypeUInt32Array):
                return 'vector<long>'
            elif isinstance(din, vtk.vtkTypeUInt64Array):
                return 'vector<float>'
            elif isinstance(din, vtk.vtkFloatArray) or isinstance(din, vtk.vtkTypeFloat32Array):
                return 'vector<float>'
            elif isinstance(din, vtk.vktDoubleArray) or isinstance(din, vtk.vtkTypeFloat64Array):
                return 'vector<float>'
            else:
                error_msg = 'VTK type not identified.' % din
                raise pexceptions.PySegInputError(expr='vtk_to_gt (TypesConverter)', msg=error_msg)

    # From the graph tool string alias creates an equivalent vtkDataArray object
    @staticmethod
    def gt_to_vtk(din):

        # Check than a type object is passed
        if not isinstance(din, str):
            error_msg = 'str object required as input.' % din
            raise pexceptions.PySegInputError(expr='gt_to_vtk (TypesConverter)', msg=error_msg)

        if (din == 'uint8_t') or (din == 'vector<uint8_t>'):
            return vtk.vtkBitArray()
        elif (din == 'short') or (din == 'vector<short>'):
            return vtk.vtkTypeInt16Array()
        elif (din == 'int') or (din == 'vector<int>'):
            return vtk.vtkIntArray()
        elif (din == 'long') or (din == 'vector<long>'):
            return vtk.vtkTypeInt64Array()
        elif (din == 'float') or (din == 'vector<float>'):
            return vtk.vtkFloatArray()
        else:
            error_msg = 'Graph tool alias not identified.' % din
            raise pexceptions.PySegInputError(expr='gt_to_vtk (TypesConverter)', msg=error_msg)