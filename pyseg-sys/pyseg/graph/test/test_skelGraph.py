from unittest import TestCase
from pyseg.globals import *
from pyseg.graph import SkelGraph
import vtk
import numpy as np

__author__ = 'martinez'


class TestSkelGraph(TestCase):

    def test_Class(self):

        # Reading the DISPERSE skeleton
        print 'Loading skeleton...'
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(DATA_DIR + '/skel.vtp')
        # reader.SetFileName(DATA_DIR + '/skel_small.vtp')
        reader.Update()
        skel = reader.GetOutput()
        del reader

        # Building the SkelGraph object
        print 'Building the graph from the skeleton...'
        sgraph = SkelGraph(skel)
        sgraph.update()

        # Getting VTK poly data
        print 'Storing the graph as VTK poly data...'
        poly = sgraph.get_vtp()
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(DATA_DIR + '/skel_graph.vtp')
        vtk_ver = vtk.vtkVersion().GetVTKVersion()
        if int(vtk_ver[0]) < 6:
            writer.SetInput(poly)
        else:
            writer.SetInputData(poly)
        out_writer = writer.Write()

        self.assertEqual(out_writer, 1, 'Graph not stored properly.')

        # Rebuilding the graph
        print 'Rebuilding a new graph from the stored poly data...'
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(DATA_DIR + '/skel_graph.vtp')
        reader.Update()
        skel2 = reader.GetOutput()
        sgraph2 = SkelGraph(skel2)
        sgraph2.update()

        # Comparing booth graphs
        print 'Comparing both graphs...'
        error = False
        vprops1, eprops1 = sgraph.get_prop_info()
        vprops2, eprops2 = sgraph.get_prop_info()
        # Getting props lists
        lvprops1 = list()
        lvprops2 = list()
        for i in range(vprops1.get_nprops()):
            key1 = vprops1.get_key(i)
            if vprops2.is_already(key1) is None:
                key2 = key1 + '_'
                if vprops2.is_already(key1) is not None:
                    lvprops1.append(key1)
                    lvprops2.append(key2)
            else:
                lvprops1.append(key1)
                lvprops2.append(key1)
        leprops1 = list()
        leprops2 = list()
        for i in range(eprops1.get_nprops()):
            key1 = eprops1.get_key(i)
            if eprops2.is_already(key1) is None:
                key2 = key1 + '_'
                if eprops2.is_already(key1) is not None:
                    leprops1.append(key1)
                    leprops2.append(key2)
            else:
                leprops1.append(key1)
                leprops2.append(key1)

        # Vertices
        error_msg = 'The graph has been modified during the building and/or the storing process.\nTheir vertices have different property values.'
        for i in range(skel.GetNumberOfPoints()):
            # point_id = points.GetValue(i)
            n1 = sgraph.get_vertex_neighbours(i)
            n2 = sgraph.get_vertex_neighbours(i)
            # print n1, n2
            if (n1 is not None) and (len(n1) > 0):
                if (n2 is not None) and (len(n1) != len(n2)):
                    error = True
                    break
                for j in range(len(n1)):
                    for k in range(len(lvprops1)):
                        vp1 = n1[j].get_property(lvprops1[k])
                        vp2 = n2[j].get_property(lvprops2[k])
                        self.assertAlmostEqual(np.sum(vp1), np.sum(vp2), 7, error_msg)
        error_msg = 'The graph has been modified during the building and/or the storing process.\nThey have different vertices.'
        self.assertEqual(error, False, error_msg)

        # Edges
        error = False
        nedges = 0
        error_msg = 'The graph has been modified during the building and/or the storing process.\nTheir edges have different property values.'
        for i in range(skel.GetNumberOfCells()):
            cell = skel.GetCell(i)
            if cell.GetNumberOfPoints() > 1:
                pts = cell.GetPointIds()
                for j in range(1, pts.GetNumberOfIds()):
                    point_id_v1 = pts.GetId(j-1)
                    point_id_v2 = pts.GetId(j)
                    e1 = sgraph.get_edge(point_id_v1, point_id_v2)
                    e2 = sgraph2.get_edge(point_id_v1, point_id_v2)
                    nedges += 1
                    if (e1 is None) or (e2 is None):
                        error = True
                        break
                    for k in range(len(leprops1)):
                        ep1 = e1.get_property(leprops1[k])
                        ep2 = e2.get_property(leprops2[k])
                        self.assertAlmostEqual(np.sum(ep2), np.sum(ep1), 7, error_msg)
        error_msg = 'The graph has been modified during the building and/or the storing process.\nThey have different edges.'
        self.assertEqual(error, False, error_msg)
        error_msg = 'The graph has not been built completely (1).'
        self.assertEqual(sgraph.get_num_edges(), nedges, error_msg)
        error_msg = 'The graph has not been built completely (2).'
        self.assertEqual(sgraph2.get_num_edges(), nedges, error_msg)