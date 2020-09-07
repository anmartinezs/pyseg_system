from unittest import TestCase
from pyseg.globals import *
from pyseg.disperse_io import *
from pyseg.factory import MembTransSkelGraph
from pyseg.vtk_ext import vtkClosestPointAlgorithm
from pyseg.vtk_ext import vtkFilterRedundacyAlgorithm
import vtk
import numpy as np

__author__ = 'martinez'

# Test variables
TEST_RESOLUTION = 1.888
TEST_MB_THICK = 5
TEST_MB_THICK_2 = TEST_MB_THICK * 0.5


class TestMembTransSkelGraph(TestCase):

    def test_Class(self):

        # Read the DisPerSe skeleton
        print('Loading skeleton...')
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
        print('Loading density...')
        density = fits.getdata(DATA_DIR + '/density.fits')

        # Read the manifolds
        print('Loading manifolds...')
        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(DATA_DIR + '/manifolds2.vti')
        reader.Update()
        manifolds = reader.GetOutput()
        # Tomogram permutation
        # permuter = vtk.vtkImagePermute()
        # permuter.SetInputData(manifolds)
        # permuter.SetFilteredAxes(2, 1, 0)
        # permuter.Update()
        # manifolds = permuter.GetOutput()
        manifolds = vti_to_numpy(manifolds)
        del reader
        # TODO: DEBUG
        # hold = ImageIO()
        # hold.setData(manifolds)
        # hold.writeMRC(DATA_DIR + '/hold.mrc')

        # Read the segmented membrane
        print('Loading membrane segmentation...')
        seg = fits.getdata(DATA_DIR + '/mb_seg.fits')

        # Factoring a MembSideSkelGraph object
        print('Factoring the MembSideSkelGraph object...')
        mb_skel = MembTransSkelGraph(skel, manifolds, density, seg)
        mb_skel.set_resolution(TEST_RESOLUTION)
        mb_skel.set_memb_thickness(TEST_MB_THICK)
        mb_skel.build_skelgraph()

        print('Getting the ArcGraph...')
        mb_skel.build_arcgraph()

        # Write the output skel graph
        print('Storing the graphs as VTK poly data...')
        sgraph = mb_skel.get_SkelGraph()
        poly = sgraph.get_vtp()
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(DATA_DIR + '/mb_trans_graph.vtp')
        vtk_ver = vtk.vtkVersion().GetVTKVersion()
        if int(vtk_ver[0]) < 6:
            writer.SetInput(poly)
        else:
            writer.SetInputData(poly)
        out_writer = writer.Write()
        self.assertEqual(out_writer, 1, 'Graph not stored properly.')
        del writer
        agraph = mb_skel.get_ArcGraph()
        poly_a = agraph.get_vtp()
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(DATA_DIR + '/mb_trans_sgraph.vtp')
        vtk_ver = vtk.vtkVersion().GetVTKVersion()
        if int(vtk_ver[0]) < 6:
            writer.SetInput(poly_a)
        else:
            writer.SetInputData(poly_a)
        out_writer = writer.Write()
        self.assertEqual(out_writer, 1, 'Graph not stored properly.')

        # Check that the geometries have been built properly
        # Print the segmentation: densities and vertex labels
        print('Printing vertices in a image...')
        segs = sgraph.print_vertices(STR_VERTEX_ID, th_den=None)
        lbls = sgraph.print_vertices(STR_VERTEX_ID, th_den=None)
        # Store result
        segs_img = ImageIO()
        hold = segs.transpose()
        hold = hold.astype(np.int16)
        segs_img.setData(hold)
        segs_img.write(file=DATA_DIR+'/skel_seg_trans_density.mrc', fileFormat='mrc')
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(DATA_DIR + '/skel_seg_trans_lbl.vti')
        vtk_ver = vtk.vtkVersion().GetVTKVersion()
        if int(vtk_ver[0]) < 6:
            writer.SetInput(numpy_to_vti(lbls))
        else:
            writer.SetInputData(numpy_to_vti(lbls))
        out_writer = writer.Write()
        # vol = numpy_to_vti(lbls)
        self.assertEqual(out_writer, 1, 'Labels stored properly.')

        # Check that all vertex coordinates are within the final segmentation
        print('Checking the object has been printed properly...')
        good_lbls = 0
        for i in range(skel.GetNumberOfPoints()):
            v = sgraph.get_vertex(i)
            if (v is not None) and (v.get_property(STR_CRITICAL_VERTEX, CRITICAL_MIN)):
                coord = v.get_coordinates()
                x = round(coord[0])
                y = round(coord[1])
                z = round(coord[2])
                if lbls[x, y, z] != 0:
                    good_lbls += 1

        error_msg = 'Some vertices have their coordinates out of their geometry.'
        self.assertEqual(good_lbls, sgraph.get_num_vertices(), error_msg)