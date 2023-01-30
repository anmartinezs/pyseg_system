from unittest import TestCase
from pyseg.globals import *
from pyseg.disperse_io import *
from pyseg.factory import MembSideSkelGraph
from pyseg.vtk_ext import vtkClosestPointAlgorithm
from pyseg.vtk_ext import vtkFilterRedundacyAlgorithm
import vtk
import numpy as np

__author__ = 'martinez'

# Test variables
TEST_RESOLUTION = 1.888
TEST_MB_THICK = 5
TEST_MB_THICK_2 = TEST_MB_THICK * 0.5
TEST_MAX_DIST = 50

class TestMembSideSkelGraph(TestCase):

    def test_Class(self):

        # Read the DisPerSe skeleton
        print('Loading skeleton...')
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(DATA_DIR + '/skel2.vtp')
        # reader.SetFileName(DATA_DIR + '/skel_small.vtp')
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
        # reader.SetFileName(DATA_DIR + '/manifolds.vti')
        reader.SetFileName(DATA_DIR + '/manifolds2.vti')
        reader.Update()
        manifolds = vti_to_numpy(reader.GetOutput())
        del reader

        # Read the segmented membrane
        print('Loading membrane segmentation...')
        hold = fits.getdata(DATA_DIR + '/mb_seg.fits')
        seg = np.zeros(hold.shape, dtype=int)
        # Keep as foreground just the postsynaptic membrane
        seg[hold == 1] = 1

        # Factoring a MembSideSkelGraph object
        print('Factoring the MembSideSkelGraph object...')
        mb_skel = MembSideSkelGraph(skel, manifolds, density, seg)
        mb_skel.set_resolution(TEST_RESOLUTION)
        mb_skel.set_memb_thickness(TEST_MB_THICK)
        mb_skel.set_max_distance(TEST_MAX_DIST)
        mb_skel.set_side(STR_SIGN_N)
        mb_skel.build_skelgraph()

        print('Getting the ArcGraph...')
        mb_skel.build_arcgraph()

        # Write the output skel graph
        print('Storing the graphs as VTK poly data...')
        sgraph = mb_skel.get_SkelGraph()
        poly = sgraph.get_vtp()
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(DATA_DIR + '/mb_side_graph.vtp')
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
        writer.SetFileName(DATA_DIR + '/mb_side_sgraph.vtp')
        vtk_ver = vtk.vtkVersion().GetVTKVersion()
        if int(vtk_ver[0]) < 6:
            writer.SetInput(poly_a)
        else:
            writer.SetInputData(poly_a)
        out_writer = writer.Write()
        self.assertEqual(out_writer, 1, 'Graph not stored properly.')

        # Loop for checking the graph
        print('Checking the object has been built properly...')
        dev = 0
        dist_filt = vtkClosestPointAlgorithm()
        hold_surf = gen_surface(seg)
        dist_filt.SetInputData(hold_surf)
        dist_filt.set_normal_field(hold_surf.GetPointData().GetArray(0))
        dist_filt.initialize()
        vertices = sgraph.get_vertices_list()
        for v in vertices:
            # For checking that distance is in range
            coord = v.get_coordinates()
            dist, dprod = dist_filt.evaluate(coord[0], coord[1], coord[2])
            # Positive configuration
            dist = abs(dist * TEST_RESOLUTION)
            if (dist < TEST_MB_THICK_2) or (dist > TEST_MAX_DIST):
                print(dist)

        error_msg = 'Error, number of vertices out of the specified area inserted %d.' % dev
        self.assertEqual(dev, 0, error_msg)

        # Check that the geometries have been built properly
        # Print the segmentation: densities and vertex labels
        print('Printing vertices in a image...')
        segs = sgraph.print_vertices(STR_VERTEX_ID, th_den=0)
        lbls = sgraph.print_vertices(STR_VERTEX_ID, th_den=None)
        # Store result
        segs_img = ImageIO()
        segs_img.setData(segs.transpose().astype(np.float32))
        segs_img.write(file=DATA_DIR+'/skel_seg_density.mrc', fileFormat='mrc')
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(DATA_DIR + '/skel_seg_lbl.vti')
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
        n_min = 0
        for i in range(skel.GetNumberOfPoints()):
            v = sgraph.get_vertex(i)
            if v is not None:
                coord = v.get_coordinates()
                x = int(round(coord[0]))
                y = int(round(coord[1]))
                z = int(round(coord[2]))
                # geom = v.get_geometry()
                # svol = geom.get_numpy_mask(v.get_id())
                if v.get_property(STR_CRITICAL_VERTEX) == CRITICAL_MIN:
                    if lbls[x, y, z] != 0:
                        good_lbls += 1
                    n_min += 1

        error_msg = 'Some vertices have their coordinates out of their geometry.'
        self.assertEqual(good_lbls, n_min, error_msg)
