from unittest import TestCase
from pyseg.vtk_ext import vtkSurfBorderAlgorithm
import vtk

__author__ = 'martinez'


class TestVtkSurfBorderAlgorithm(TestCase):

    def test_Execute(self):

        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName('../data/surf.vtp')
        reader.Update()

        filter = vtkSurfBorderAlgorithm()
        filter.SetInputData(reader.GetOutput())
        filter.Execute()
        border = filter.GetOutput()

        self.assertIsInstance(border, vtk.vtkPolyData, 'A non poly data returned.')

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName('../data/border.vtp')
        vtk_ver = vtk.vtkVersion().GetVTKVersion()
        if int(vtk_ver[0]) < 6:
            writer.SetInput(border)
        else:
            writer.SetInputData(border)
        out_writer = writer.Write()

        self.assertEqual(out_writer, 1, 'Border not saved properly.')
        self.assertGreater(border.GetNumberOfCells(), 0, 'Border without cells.')