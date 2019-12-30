from unittest import TestCase
from pyseg.vtk_ext import vtkFilterRedundacyAlgorithm
from pyseg.globals import *
import vtk

__author__ = 'martinez'


class TestVtkFilterRedundacyAlgorithm(TestCase):

    def test_Execute(self):

        reader = vtk.vtkXMLPolyDataReader()
        # reader.SetFileName('../data/anchors.vtp')
        reader.SetFileName(DATA_DIR + '/skel.vtp')
        reader.Update()
        input_poly = reader.GetOutput()

        filter = vtkFilterRedundacyAlgorithm()
        filter.SetInputData(input_poly)
        # filter.set_resolution(0.1)
        filter.Execute()
        output_poly = filter.GetOutput()

        self.assertIsInstance(output_poly, vtk.vtkPolyData, 'A non poly data returned.')

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(DATA_DIR + '/skel_no_red.vtp')
        vtk_ver = vtk.vtkVersion().GetVTKVersion()
        if int(vtk_ver[0]) < 6:
            writer.SetInput(output_poly)
        else:
            writer.SetInputData(output_poly)
        out_writer = writer.Write()

        self.assertEqual(out_writer, 1, 'Output not saved properly.')
        self.assertGreater(input_poly.GetNumberOfPoints(), output_poly.GetNumberOfPoints(), 'No redundancy found.')

        # Check attributes
        atts_input = list()
        for i in range(input_poly.GetPointData().GetNumberOfArrays()):
            atts_input.append(input_poly.GetPointData().GetArrayName(i))
        for i in range(input_poly.GetCellData().GetNumberOfArrays()):
            atts_input.append(input_poly.GetCellData().GetArrayName(i))
        atts_output = list()
        for i in range(output_poly.GetPointData().GetNumberOfArrays()):
            atts_output.append(output_poly.GetPointData().GetArrayName(i))
        for i in range(output_poly.GetCellData().GetNumberOfArrays()):
            atts_output.append(output_poly.GetCellData().GetArrayName(i))

        self.assertEqual(len(atts_input), len(atts_output), 'Output size attribute is not correct')
        for i in range(len(atts_input)):
            self.assertTrue(atts_output[i] in atts_input, 'Attribute %s was not found in output.' % (atts_input[i]))