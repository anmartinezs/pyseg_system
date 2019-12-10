from unittest import TestCase
from pyseg.globals import *
from pyseg.graph import SkelGraph
from pyseg.disperse_io import *
import vtk
import pyfits

__author__ = 'martinez'

class TestPointGeometry(TestCase):

    def test_Class(self):

        # TODO: skel_small.vtp is not a proper input
        # Reading the DISPERSE skeleton
        print 'Loading skeleton...'
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(DATA_DIR + '/skel_small.vtp')
        reader.Update()
        skel = reader.GetOutput()
        del reader

        # Reading the image with the manifolds
        print 'Loading manifolds...'
        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(DATA_DIR + '/manifolds.vti')
        reader.Update()
        manifolds = vti_to_numpy(reader.GetOutput())
        del reader

        # Reading the density map
        print 'Loading density...'
        density = pyfits.getdata(DATA_DIR + '/density.fits')

        # Building the SkelGraph object
        print 'Building the graph from the skeleton...'
        sgraph = SkelGraph(skel)
        sgraph.add_geometry(manifolds, density)
        sgraph.update()

        # Print the segmentation: densities and vertex labels
        print 'Printing vertices in a image...'
        segs = sgraph.print_vertices()
        lbls = sgraph.print_vertices(STR_VERTEX_ID)

        # Store result
        segs_img = ImageIO()
        segs_img.setData(segs.transpose())
        segs_img.write(file=DATA_DIR+'/skel_seg_density.mrc', fileFormat='mrc')
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(DATA_DIR + '/skel_seg_lbl.vti')
        vtk_ver = vtk.vtkVersion().GetVTKVersion()
        if int(vtk_ver[0]) < 6:
            writer.SetInput(numpy_to_vti(lbls))
        else:
            writer.SetInputData(numpy_to_vti(lbls))
        out_writer = writer.Write()
        numpy_to_vti(lbls)
        self.assertEqual(out_writer, 1, 'Labels stored properly.')

        # Check that all vertex coordinates are within the final segmentation
        good_lbls = 0
        for i in range(skel.GetNumberOfPoints()):
            v = sgraph.get_vertex(i)
            if v is not None:
                coord = v.get_coordinates()
                x = int(coord[0])
                y = int(coord[1])
                z = int(coord[2])
                if lbls[x, y, z] == manifolds[x, y, z]:
                    good_lbls += 1

        error_msg = 'Some vertices have their coordinates out of their geometry.'
        self.assertEqual(good_lbls, sgraph.get_num_vertices(), error_msg)