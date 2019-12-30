from unittest import TestCase
from pyseg.vtk_ext import vtkClosestCellAlgorithm
import vtk

__author__ = 'martinez'


class TestVtkClosestCellAlgorithm(TestCase):

    def test_Execute(self):

        # Read reference
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName('../data/border.vtp')
        reader.Update()
        reference = reader.GetOutput()
        del reader

        # Read input poly
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName('../data/skel.vtp')
        reader.Update()
        input_poly = reader.GetOutput()
        del reader

        # Filtering
        algorithm = vtkClosestCellAlgorithm()
        algorithm.SetInputData(input_poly)
        algorithm.set_reference(reference)
        algorithm.Execute()
        poly_filt = algorithm.GetOutput()

        # Assertion and results storing
        self.assertIsInstance(poly_filt, vtk.vtkPolyData, 'A non poly data returned.')

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName('../data/skel_closest_border.vtp')
        vtk_ver = vtk.vtkVersion().GetVTKVersion()
        if int(vtk_ver[0]) < 6:
            writer.SetInput(poly_filt)
        else:
            writer.SetInputData(poly_filt)
        out_writer = writer.Write()

        self.assertEqual(out_writer, 1, 'Border not saved properly.')
        self.assertGreater(poly_filt.GetNumberOfCells(), 0, 'Border without cells.')