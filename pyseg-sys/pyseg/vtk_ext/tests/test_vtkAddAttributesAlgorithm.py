from unittest import TestCase
from pyseg.vtk_ext import vtkAddAttributesAlgorithm
import vtk

__author__ = 'martinez'


class TestVtkAddAttributesAlgorithm(TestCase):

    def test_Execute(self):

        # Read reference
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName('../data/border.vtp')
        reader.Update()
        reference = reader.GetOutput()
        del reader

        # Read input poly
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName('../data/border.vtp')
        reader.Update()
        input_poly = reader.GetOutput()
        del reader

        # Filtering
        algorithm = vtkAddAttributesAlgorithm()
        algorithm.SetInputData(input_poly)
        algorithm.set_reference(reference)
        algorithm.Execute()

        # Output information analysis
        atts_reference = list()
        for i in range(reference.GetPointData().GetNumberOfArrays()):
            atts_reference.append(reference.GetPointData().GetArrayName(i))
        for i in range(reference.GetCellData().GetNumberOfArrays()):
            atts_reference.append(reference.GetCellData().GetArrayName(i))
        atts_input = list()
        for i in range(input_poly.GetPointData().GetNumberOfArrays()):
            atts_input.append(input_poly.GetPointData().GetArrayName(i))
        for i in range(input_poly.GetCellData().GetNumberOfArrays()):
            atts_input.append(input_poly.GetCellData().GetArrayName(i))

        self.assertEqual(2*len(atts_reference), len(atts_input), 'Output size attribute is not correct')
        for i in range(len(atts_reference)):
            self.assertTrue(atts_reference[i] in atts_input, 'Attribute %s was not found in output.' % (atts_input[i]))