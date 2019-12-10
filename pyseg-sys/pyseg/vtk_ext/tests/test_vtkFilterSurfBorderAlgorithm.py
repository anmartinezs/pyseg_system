from unittest import TestCase
from pyseg.globals import *
from pyseg.vtk_ext import vtkFilterSurfBorderAlgorithm
import vtk

__author__ = 'martinez'


class TestVtkFilterSurfBorderAlgorithm(TestCase):

    def test_Execute(self):

        # Read reference
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(DATA_DIR + '/surf.vtp')
        reader.Update()
        surf = reader.GetOutput()
        del reader

        # Read input poly
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(DATA_DIR + '/skel.vtp')
        reader.Update()
        input_poly = reader.GetOutput()
        del reader

        # Filtering
        algorithm = vtkFilterSurfBorderAlgorithm()
        algorithm.SetInputData(input_poly)
        algorithm.set_surf(surf)
        algorithm.Execute()
        poly_filt = algorithm.GetOutput()

        # Assertion and results storing
        self.assertIsInstance(poly_filt, vtk.vtkPolyData, 'A non poly data returned.')

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(DATA_DIR + '/skel_border_filt.vtp')
        vtk_ver = vtk.vtkVersion().GetVTKVersion()
        if int(vtk_ver[0]) < 6:
            writer.SetInput(poly_filt)
        else:
            writer.SetInputData(poly_filt)
        out_writer = writer.Write()

        self.assertEqual(out_writer, 1, 'Border not saved properly.')
        self.assertGreater(poly_filt.GetNumberOfCells(), 0, 'Border without cells.')