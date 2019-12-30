from unittest import TestCase
from pyseg.globals import *
from pyseg.disperse_io import *
from pyseg.vtk_ext import vtkFilterRedundacyAlgorithm
from psd import PsdSynap
import vtk
import numpy as np

__author__ = 'martinez'

# Test variables
TEST_RESOLUTION = 1.888
TEST_MB_THICK = 5
TEST_MB_THICK_2 = TEST_MB_THICK * 0.5
TEST_MAX_DIST_POST = 50
TEST_MAX_DIST_PRE = 10

class TestPsdSynap(TestCase):

    def test_Class(self):

        # Read the DisPerSe skeleton
        print 'Loading skeleton...'
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(DATA_DIR + '/skel.vtp')
        reader.Update()
        skel = reader.GetOutput()
        del reader

        # Filtering the redundancy of the skeleton
        red_filt = vtkFilterRedundacyAlgorithm()
        red_filt.SetInputData(skel)
        red_filt.Execute()
        skel = red_filt.GetOutput()

        # Read the density
        print 'Loading density...'
        density = pyfits.getdata(DATA_DIR + '/density.fits')

        # Read the manifolds
        print 'Loading manifolds...'
        reader = vtk.vtkXMLImageDataReader()
        # reader.SetFileName(DATA_DIR + '/manifolds.vti')
        reader.SetFileName(DATA_DIR + '/manifolds2.vti')
        reader.Update()
        manifolds = vti_to_numpy(reader.GetOutput())
        del reader

        # Read the segmented membrane
        print 'Loading membrane segmentation...'
        seg = pyfits.getdata(DATA_DIR + '/mb_seg.fits')

        # Factoring a MembSideSkelGraph object
        print 'Creating the PsdSynap object...'
        psd = PsdSynap(skel, manifolds, density, seg)
        psd.set_resolution(TEST_RESOLUTION)
        psd.set_memb_thickness(TEST_MB_THICK)
        psd.set_max_dist_post(TEST_MAX_DIST_POST)
        psd.set_max_dist_pre(TEST_MAX_DIST_PRE)

        print 'Building the SkelGraphs'
        psd.build_sgraphs()

        print 'Building the ArcGraphs'
        psd.build_agraphs()

        # Write the output skel graph
        print 'Storing the graphs as VTK poly data...'
        poly = psd.get_vtp('skel')
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(DATA_DIR + '/psd_graph.vtp')
        vtk_ver = vtk.vtkVersion().GetVTKVersion()
        if int(vtk_ver[0]) < 6:
            writer.SetInput(poly)
        else:
            writer.SetInputData(poly)
        out_writer = writer.Write()
        self.assertEqual(out_writer, 1, 'Graph not stored properly.')
        del writer
        poly_a = psd.get_vtp('arc')
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(DATA_DIR + '/psd_sgraph.vtp')
        vtk_ver = vtk.vtkVersion().GetVTKVersion()
        if int(vtk_ver[0]) < 6:
            writer.SetInput(poly_a)
        else:
            writer.SetInputData(poly_a)
        out_writer = writer.Write()
        self.assertEqual(out_writer, 1, 'Graph not stored properly.')

        # Check that the geometries have been built properly
        # Print the segmentation: densities and vertex labels
        print 'Printing vertices in a image...'
        psd.print_vertices(DATA_DIR + '/seg_psd.vti', 'skel')
        psd.print_vertices(DATA_DIR + '/sega_psd.mrc', 'arc')