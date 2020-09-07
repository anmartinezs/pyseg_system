from unittest import TestCase
from pyseg.vtk_ext import vtkClosestPointAlgorithm
import vtk
import time
from pyseg.globals import *

__author__ = 'martinez'

class TestVtkClosestPointAlgorithm(TestCase):

    def test_Evaluate(self):

        # Read reference
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(DATA_DIR + '/surf.vtp')
        reader.Update()
        reference = reader.GetOutput()
        del reader

        # Read poly
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(DATA_DIR + '/anchors.vtp')
        reader.Update()
        poly = reader.GetOutput()
        del reader

        # Set-up the filter
        filter_p = vtkClosestPointAlgorithm()
        filter_p.SetInputData(reference)

        # Measure distance to poly points (slow way)
        distance = vtk.vtkFloatArray()
        distance.SetName('surf_distance')
        points = poly.GetPoints()
        dist_t1 = 0
        tic = time.time()
        for i in range(poly.GetNumberOfPoints()):
            x, y, z = points.GetPoint(i)
            dist = filter_p.evaluate(x, y, z)
            distance.InsertNextValue(dist)
            dist_t1 += dist
        elapsed1 = time.time() - tic
        print('Slow mode time = %.2f s' % (elapsed1))
        print('Slow mode accumulated result = %.3f' % (dist_t1))
        poly.GetPointData().AddArray(distance)

        # Measure distance to poly points (fast way)
        distance2 = vtk.vtkFloatArray()
        distance2.SetName('surf_distance2')
        filter_p.initialize()
        dist_t2 = 0
        tic = time.time()
        for i in range(poly.GetNumberOfPoints()):
            x, y, z = points.GetPoint(i)
            dist = filter_p.evaluate(x, y, z)
            distance2.InsertNextValue(dist)
            dist_t2 += dist
        elapsed2 = time.time() - tic
        print('Fast mode time = %.2f s' % (elapsed2))
        print('Fast mode accumulated result = %.3f' % (dist_t2))
        poly.GetPointData().AddArray(distance2)

        # Measuring distance and dotproduct
        dprod = vtk.vtkFloatArray()
        dprod.SetName('dot_prod')
        filter_p.set_normal_field(STR_NFIELD_ARRAY_NAME)
        dist_t3 = 0
        tic = time.time()
        for i in range(poly.GetNumberOfPoints()):
            x, y, z = points.GetPoint(i)
            dist, dprodv = filter_p.evaluate(x, y, z)
            dprod.InsertNextValue(dprodv)
            dist_t3 += dist
        elapsed2 = time.time() - tic
        print('Dot product mode time = %.2f s' % (elapsed2))
        print('Dot product mode accumulated result = %.3f' % (dist_t3))
        poly.GetPointData().AddArray(dprod)

        # Assertion and results storing
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(DATA_DIR + '/surf_distance_anchors.vtp')
        vtk_ver = vtk.vtkVersion().GetVTKVersion()
        if int(vtk_ver[0]) < 6:
            writer.SetInput(poly)
        else:
            writer.SetInputData(poly)
        out_writer = writer.Write()

        self.assertEqual(out_writer, 1, 'Poly not saved properly.')
        self.assertGreater(elapsed1, elapsed2, 'Fast version is slower than the Slow one!!')
        self.assertAlmostEqual(dist_t1, dist_t2, 3, 'Distances estimated between the different methods are not approximately equal.')
        self.assertAlmostEqual(dist_t1, dist_t3, 3, 'Distances estimated between the different methods are not approximately equal.')