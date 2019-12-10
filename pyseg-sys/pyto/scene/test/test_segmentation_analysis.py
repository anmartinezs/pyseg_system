"""

Tests module segmentation_analysis

# Author: Vladan Lucic
# $Id: test_segmentation_analysis.py 1311 2016-06-13 12:41:50Z vladan $
"""

__version__ = "$Revision: 1311 $"

from copy import copy, deepcopy
import unittest

import numpy
import numpy.testing as np_test 
import scipy

#from pyto.segmentation.grey import Grey
#from pyto.segmentation.segment import Segment
#from pyto.segmentation.hierarchy import Hierarchy
#from pyto.segmentation.thresh_conn import ThreshConn
#import common as common
from pyto.scene.segmentation_analysis import SegmentationAnalysis
import pyto.segmentation.test.common as seg_cmn
from pyto.segmentation.cleft import Cleft
from pyto.scene.cleft_regions import CleftRegions


class TestSegmentationAnalysis(np_test.TestCase):
    """
    """

    def setUp(self):
        """
        """
        self.reorder_warning = True

    def makeCleftLayers(self):
        """
        Returns a CleftRegions object made using image 1, boundary 1
        """

        cleft = Cleft(data=seg_cmn.bound_1.data, bound1Id=3, bound2Id=4, 
                      cleftId=5)
        cleft_layers = CleftRegions(image=seg_cmn.image_1, cleft=cleft)
        cleft_layers.makeLayers(nBoundLayers=1)
        cleft_layers.findDensity(regions=cleft_layers.regions)

        return cleft_layers

    def testTcSegmentAnalyze(self):
        """
        Tests tcSegmentAnalyze() and also getLabels().
        """

        ######################################################
        #
        # Image 1, boundary 1, order <, boundary >= 1
        #

        # make CleftLayers
        cleft_layers = self.makeCleftLayers()

        # do tc segmentation and analysis 
        se = SegmentationAnalysis(image=seg_cmn.image_1, 
                                  boundary=seg_cmn.bound_1)
        se.setSegmentationParameters(boundaryIds=[3, 4], nBoundary=1, 
                                     boundCount='at_least', mask=5)
        se.setAnalysisParameters(distanceRegion=seg_cmn.bound_1, 
                                 distanceId=[3], distanceMode='min', 
                                 cleftLayers=cleft_layers)
        tc_iter = se.tcSegmentAnalyze(
            thresh=seg_cmn.threshold_1, order='<', count=True, doDensity=True, 
            doMorphology=True, doLength=True, doTopology=True, 
            doDistanceTo=True, doBoundDistance=True, doCleftContacts=True)

        # make and test individual segments
        id_dict = {}
        for count in range(len(seg_cmn.threshold_1)):

            # make level
            new_se, level, curr_thresh = tc_iter.next()
            seg = new_se.segments

            # test parameters
            np_test.assert_equal(new_se.nBoundary, se.nBoundary)
            np_test.assert_equal(new_se.boundCount, se.boundCount)
            np_test.assert_equal(new_se.structElConn, se.structElConn)
            np_test.assert_equal(new_se.contactStructElConn, 
                                 se.contactStructElConn)
            np_test.assert_equal(new_se.countStructElConn, se.countStructElConn)
            np_test.assert_equal(new_se.lengthContact, se.lengthContact)
            np_test.assert_equal(new_se.lengthLine, se.lengthLine)

            # test levels and threshold
            np_test.assert_equal(level, count)
            np_test.assert_equal(curr_thresh, seg_cmn.threshold_1[level])

            # test getLabels()
            np_test.assert_equal(new_se.labels, new_se.segments)
            np_test.assert_equal(new_se.ids, new_se.segments.ids)

            # test ids
            id_dict = seg_cmn.id_correspondence(actual=seg.data[2:6, 1:9], 
                            desired=seg_cmn.data_1[level], current=id_dict)
            converted_ids = [id_dict[id_] for id_ in seg_cmn.levelIds_1[level]]
            #np_test.assert_equal(seg.ids, seg_cmn.levelIds_1[level])
            np_test.assert_equal(seg.ids, converted_ids)

            # test segments 
            try:
                np_test.assert_equal(seg.data[2:6, 1:9], seg_cmn.data_1[level])
            except AssertionError:
                np_test.assert_equal(seg.data[2:6, 1:9]>0, 
                                     seg_cmn.data_1[level]>0)
                if self.reorder_warning:
                    print(
                        "The exact id assignment is different from what it was "
                        + "when this test was written, but this really depends "
                        + "on internals of scipy.ndimage. Considering that the "
                        + "segments are correct, most likely everything is ok.")
                self.reorder_warning = False
            np_test.assert_equal(
                seg.contacts.findSegments(boundaryIds=[3,4], nBoundary=1),
                seg_cmn.levelIds_1[level])

            # test boundaries
            bound_found = seg.contacts.findBoundaries(segmentIds=seg.ids, 
                                                      nSegment=1)
            if level == 0:
                np_test.assert_equal(bound_found, [])
            elif level == 1:
                np_test.assert_equal(bound_found, [3])
            else:
                np_test.assert_equal(bound_found, [3,4])
            
            # test segment analysis
            np_test.assert_almost_equal(se._regionDensity.mean, 
                                        seg_cmn.image_ar_inset_1.mean())
            np_test.assert_almost_equal(new_se.regionDensity.mean, 
                                        seg_cmn.image_ar_inset_1.mean())
            np_test.assert_almost_equal(new_se.backgroundDensity.mean, 
                                        seg_cmn.bkg_density_1[count])
            if se._distance is not None:
                np_test.assert_almost_equal(se._distance.distance[seg.ids], 
                                         seg_cmn.distance_to_3_min_1[seg.ids])
                np_test.assert_almost_equal(new_se.distance.distance[seg.ids], 
                                         seg_cmn.distance_to_3_min_1[seg.ids])
            # test bound distance
            if (level >= 0) and (level < 2):
                np_test.assert_equal((se.boundDistance.distance > 0).sum(), 0)
            elif level == 2:
                dist = se.boundDistance.distance
                np_test.assert_equal((dist > 0).sum(), 1)
                np_test.assert_equal((dist[3:6] == 5).sum(), 1)
            elif level == 3:
                dist = se.boundDistance.distance
                np_test.assert_equal((dist[6:10] == 5).sum(), 2)
            elif level == 4:
                np_test.assert_equal((se._boundDistance.distance > 0).sum(), 2)
                dist = se.boundDistance.distance
                np_test.assert_equal(dist[10] == 5, True)
                np_test.assert_equal(dist[11] == 5, True)
            elif level > 4:
                dist = se._boundDistance.distance
                np_test.assert_equal((dist[12:] == 5).sum(), 1)
 
            # test contacts
            if level == 0:
                np_test.assert_equal(se._nContacts, [[0], [0], [0]])
                np_test.assert_equal(se._surfaceDensityContacts, [0, 0, 0])
                np_test.assert_equal(se._surfaceDensitySegments, [0, 0, 0])
            if level == 1:
                np_test.assert_equal(se._nContacts, 
                                     [[2, 1, 1], 
                                      [2, 1, 1],
                                      [0, 0, 0]])
                np_test.assert_equal(se._surfaceDensityContacts, 
                                     [2./16, 2./8, 0])
                np_test.assert_equal(se._surfaceDensitySegments, 
                                     [2./8, 2./8, 2./8])
            if level == 2:
                np_test.assert_equal(se._nContacts, 
                                     [[4, 0, 0, 1, 2, 1], 
                                      [2, 0, 0, 1, 1, 0],
                                      [2, 0, 0, 0, 1, 1]])
                np_test.assert_equal(se._surfaceDensityContacts, 
                                     [4./16, 2./8, 2./8])
                np_test.assert_equal(se._surfaceDensitySegments, 
                                     [3./8, 3./8, 3./8])
            if level == 5:
                np_test.assert_equal(se._nContacts, 
                                     [[6, 0,0,0,0,0,0,0,0,0,0,0, 6],  
                                      [2, 0,0,0,0,0,0,0,0,0,0,0, 2],
                                      [4, 0,0,0,0,0,0,0,0,0,0,0, 4]])
                np_test.assert_equal(se._surfaceDensityContacts, 
                                     [6./16, 2./8, 4./8])
                np_test.assert_equal(se._surfaceDensitySegments, 
                                     [1./8, 1./8, 1./8])

        # hierarchy
        tc = tc_iter.send(True) 

        # test hierarchical segmentation
        converted_ids = [id_dict[id_] for id_ in seg_cmn.ids_1]
        np_test.assert_equal(tc.levelIds, seg_cmn.levelIds_1)

        # test analysis of the hierarchy
        np_test.assert_almost_equal(se.density.mean[tc.ids], 
                                    seg_cmn.density_1[tc.ids])
        np_test.assert_almost_equal(se.regionDensity.mean, 
                                    seg_cmn.image_ar_inset_1.mean())
        np_test.assert_equal(se.morphology.volume[1:], seg_cmn.volume_1[1:])
        #print se.morphology.length

        # test distance to
        np_test.assert_almost_equal(se.distance.distance[tc.ids], 
                                    seg_cmn.distance_to_3_min_1[converted_ids])

        # test bound distance
        dist = se.boundDistance.distance
        np_test.assert_equal((dist == 5).sum(), 8)
        np_test.assert_equal((dist > 5).sum(), 0)
        np_test.assert_equal(((dist > 0) & (dist < 5)).sum(), 0)

        ##########################################################
        #
        # Same as above but multi distanceTo
        #

        # do tc segmentation and analysis 
        se = SegmentationAnalysis(image=seg_cmn.image_1, 
                                  boundary=seg_cmn.bound_1)
        se.setSegmentationParameters(boundaryIds=[3, 4], nBoundary=1, 
                                     boundCount='at_least', mask=5)
        se.setAnalysisParameters(distanceRegion=seg_cmn.bound_1, 
                        distanceId=([3], [3,4]), distanceMode=('min', 'mean'),
                        distanceName=('min_3', 'mean_3_4'))
        tc_iter = se.tcSegmentAnalyze(
            thresh=seg_cmn.threshold_1, order='<', doDensity=True, 
            doMorphology=True, doLength=True, doTopology=True, 
            doDistanceTo=True, doBoundDistance=True)

        # make and test individual segments
        id_dict = {}
        for count in range(len(seg_cmn.threshold_1)):

            # make level
            new_se, level, curr_thresh = tc_iter.next()
            seg = new_se.segments

            id_dict = seg_cmn.id_correspondence(actual=seg.data[2:6, 1:9], 
                            desired=seg_cmn.data_1[level], current=id_dict)
            converted_ids = [id_dict[id_] for id_ in seg_cmn.levelIds_1[level]]

        # hierarchy
        tc = tc_iter.send(True) 
        tc.useInset(inset=[slice(2,6), slice(1,9)], mode='abs', useFull=True,
                    expand=True)
        np_test.assert_equal(tc.data>0, seg_cmn.data_1[-1]>0)
        converted_ids = [id_dict[id_] for id_ in seg_cmn.ids_1]

        # test distance to
        np_test.assert_almost_equal(se.min_3.distance[tc.ids], 
                                    seg_cmn.distance_to_3_min_1[converted_ids])
        np_test.assert_almost_equal(se.mean_3_4.distance[tc.ids], 
                                  seg_cmn.distance_to_3_4_mean_1[converted_ids])
        np_test.assert_almost_equal(se.mean_3_4.closestRegion[tc.ids], 
                                  seg_cmn.closest_region_1[converted_ids])

        
        ######################################################
        #
        # Image 1, boundary 1, order >, multi distance
        #

        # do tc segmentation and analysis 
        se = SegmentationAnalysis(image=seg_cmn.image_1, 
                                  boundary=seg_cmn.bound_1)
        se.setSegmentationParameters(boundaryIds=[3, 4], nBoundary=1, 
                                     boundCount='at_least', mask=5)
        se.setAnalysisParameters(distanceRegion=seg_cmn.bound_1, 
                        distanceId=([3], [3,4]), distanceMode=('min', 'mean'),
                        distanceName=('min_3', 'mean_3_4'))
        tc_iter = se.tcSegmentAnalyze(
            thresh=seg_cmn.threshold_1, order='>', doDensity=True, 
            doMorphology=True, doLength=True, doTopology=True, 
            doDistanceTo=True, doBoundDistance=True)

        # make and test individual segments
        id_dict = {}
        for count in range(len(seg_cmn.threshold_1)):

            # make level
            #print 'Count: ', count
            new_se, level, curr_thresh = tc_iter.next()
            seg = new_se.segments

            # test levels and threshold
            np_test.assert_equal(level, 0)
            np_test.assert_equal(curr_thresh, seg_cmn.threshold_1[-count-1])

            # test ids
            id_dict = seg_cmn.id_correspondence(actual=seg.data[2:6, 1:9], 
                            desired=seg_cmn.data_1[-count-1], current=id_dict)
            inv_id_dict = dict([(val, key) for key, val in id_dict.items()])
            converted_ids = [inv_id_dict[id_] for id_ 
                             in seg.ids]

            # test segments 
            try:
                desired = seg.reorder(data=seg_cmn.data_1[-count-1], 
                                      order=id_dict)
                np_test.assert_equal(seg.data[2:6, 1:9], desired)
            except AssertionError:
                np_test.assert_equal(seg.data[2:6, 1:9]>0, 
                                     seg_cmn.data_1[-count-1]>0)
                if self.reorder_warning:
                    print(
                        "The exact id assignment is different from what it was "
                        + "when this test was written, but this really depends "
                        + "on internals of scipy.ndimage. Considering that the "
                        + "segments are correct, most likely everything is ok.")
                self.reorder_warning = False
            np_test.assert_equal(
                seg.contacts.findSegments(boundaryIds=[3,4], nBoundary=1),
                seg.ids)

            # test boundaries
            bound_found = seg.contacts.findBoundaries(segmentIds=seg.ids, 
                                                      nSegment=1)
            if len(seg_cmn.threshold_1)-count-1 == 0:
                np_test.assert_equal(bound_found, [])
            elif len(seg_cmn.threshold_1)-count-1 == 1:
                np_test.assert_equal(bound_found, [3])
            else:
                np_test.assert_equal(bound_found, [3,4])
            
            # test segment analysis
            np_test.assert_almost_equal(se._regionDensity.mean, 
                                        seg_cmn.image_ar_inset_1.mean())
            if se._min_3 is not None:
                np_test.assert_almost_equal(se._min_3.distance[seg.ids], 
                                   seg_cmn.distance_to_3_min_1[converted_ids])
                np_test.assert_almost_equal(new_se.min_3.distance[seg.ids], 
                                   seg_cmn.distance_to_3_min_1[converted_ids])

        # hierarchy
        tc = tc_iter.send(True) 
        converted_ids = [id_dict[id_] for id_ in seg_cmn.ids_1]

        # test analysis of the hierarchy
        np_test.assert_almost_equal(se.density.mean[converted_ids], 
                                    seg_cmn.density_1[tc.ids])
        np_test.assert_equal(se.morphology.volume[converted_ids], 
                             seg_cmn.volume_1[tc.ids])

        # test distance to
        np_test.assert_almost_equal(se.min_3.distance[converted_ids], 
                                    seg_cmn.distance_to_3_min_1[tc.ids])
        np_test.assert_almost_equal(se.mean_3_4.distance[converted_ids], 
                                  seg_cmn.distance_to_3_4_mean_1[tc.ids])
        np_test.assert_almost_equal(se.mean_3_4.closestRegion[converted_ids], 
                                  seg_cmn.closest_region_1[tc.ids])

    def testClassify(self):
        """
        Tests classify and all individual classification methods
        """

        # prepare for tc segmentation 
        se = SegmentationAnalysis(image=seg_cmn.image_1, 
                                  boundary=seg_cmn.bound_1)
        se.setSegmentationParameters(boundaryIds=[3, 4], nBoundary=1, 
                                     boundCount='at_least', mask=5)
        se.setAnalysisParameters(distanceRegion=seg_cmn.bound_1, 
                                 distanceId=[3], distanceMode='min')
        tc_iter = se.tcSegmentAnalyze(
            thresh=seg_cmn.threshold_1, order='<', doDensity=True, 
            doMorphology=True, doLength=True, doTopology=True, 
            doDistanceTo=True, doBoundDistance=True)

        # make hierarchy
        id_dict = {}
        for count in range(len(seg_cmn.threshold_1)):
            seg, level, curr_thresh = tc_iter.next()
        tc = tc_iter.send(True) 

        #######################################################
        #
        # Volume + bound ids
        #

        # set classification parameters and desired results
        se.addClassificationParameters(type='volume', 
                                       args={'volumes':[0, 10, 50]})
        se.addClassificationParameters(type='contacted_ids', 
                                       args={'ids':[3], 'rest':True})
        desired_names = ['_vol-0-10_bound-3', '_vol-0-10_bound-rest', 
                         '_vol-10-50_bound-3', '_vol-10-50_bound-rest']
        desired_ids = [[1,2,3,4,6,7,10], [5,8,9], [11,12,13,14], []]

        # classify and test
        ind = 0
        for hi, name in se.classify(hierarchy=tc):
            np_test.assert_equal(name, desired_names[ind])
            np_test.assert_equal(hi.ids, desired_ids[ind])
            np_test.assert_equal(hi.contacts.segments, hi.ids)
            ind += 1

        #######################################################
        #
        # Volume + n bound
        #

        # make a hierarchy
        tc_iter = se.tcSegmentAnalyze(
            thresh=seg_cmn.threshold_1, order='<', doDensity=True, 
            doMorphology=True, doLength=True, doTopology=True, 
            doDistanceTo=True, doBoundDistance=True)
        for count in range(len(seg_cmn.threshold_1)):
            seg, level, curr_thresh = tc_iter.next()
        tc = tc_iter.send(True) 

        # set classification parameters and desired results
        se.removeClassificationParameters()
        se.addClassificationParameters(type='volume',
                                       args={'volumes':[2, 10, 20], 
                                             'names':['small', 'big']})
        se.addClassificationParameters(type='n_contacted', 
                                       args={'nContacted':[1,2], 
                                             'names':['teth', 'conn']})
        desired_names = ['_small_teth', '_small_conn', 
                         '_big_teth', '_big_conn']
        desired_ids = [[2,3], [4,6,7,10], [], [11,12]]

        # classify and test
        ind = 0
        for hi, name in se.classify(hierarchy=tc): 
            np_test.assert_equal(name, desired_names[ind])
            np_test.assert_equal(hi.ids, desired_ids[ind])
            np_test.assert_equal(hi.contacts.segments, hi.ids)
            ind += 1

        #######################################################
        #
        # New + volume + bound ids
        #

        # make a hierarchy
        tc_iter = se.tcSegmentAnalyze(
            thresh=seg_cmn.threshold_1, order='<', doDensity=True, 
            doMorphology=True, doLength=True, doTopology=True, 
            doDistanceTo=True, doBoundDistance=True)
        for count in range(len(seg_cmn.threshold_1)):
            seg, level, curr_thresh = tc_iter.next()
        tc = tc_iter.send(True) 

        # set classification parameters and desired results
        se.removeClassificationParameters()
        se.addClassificationParameters(type='keep', args={'mode':'new'})
        se.addClassificationParameters(type='volume',
                                       args={'volumes':[1, 2, 10], 
                                             'names':['tiny', 'small']})
        se.addClassificationParameters(type='contacted_ids', 
                                       args={'ids':[3], 'rest':True, 
                                             'names':['con-3', 'rest']})
        desired_names = ['_new_tiny_con-3', '_new_tiny_rest', 
                         '_new_small_con-3', '_new_small_rest']
        desired_ids = [[1], [5,8,9], [2], []]

        # classify and test
        ind = 0
        for hi, name in se.classify(hierarchy=tc): 
            np_test.assert_equal(name, desired_names[ind])
            np_test.assert_equal(hi.ids, desired_ids[ind])
            np_test.assert_equal(hi.contacts.segments, hi.ids)
            ind += 1

    def testClassifyAnalyze(self):
        """
        Tests classifyAnalyze() method, and also getLabels()
        """

        # make CleftLayers and set desired cleft contacts
        cleft_layers = self.makeCleftLayers()
        desired_n_contacts = [
            [[12, 1, 1, 1, 2, 0, 2, 2, 0, 0, 3], 
             [7, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1],
             [5, 0, 0, 0, 1, 0, 1, 1, 0, 0, 2]], 
            [[3, 0, 0, 0, 0, 1, 0, 0, 1, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [3, 0, 0, 0, 0, 1, 0, 0, 1, 1]], 
            [[21, 0,0,0,0,0,0,0,0,0,0, 3, 6, 6, 6],
             [7, 0,0,0,0,0,0,0,0,0,0, 1, 2, 2, 2],
             [14, 0,0,0,0,0,0,0,0,0,0, 2, 4, 4, 4]], 
            [[0],
             [0],
             [0]]]
        desired_surface_density_contacts = [
            [12/16., 7/8., 5/8.], [3/16., 0, 3/8.], 
            [21/16., 7/8., 14/8.], [0, 0, 0]]
        desired_surface_density_segments = [
            [7/8., 7/8., 7/8.], [3/8., 3/8., 3/8.],
            [4/8., 4/8., 4/8.], [0, 0, 0]] 

        ########################################################
        #
        # Scenario 1:
        #   - segment and analyze
        #   - classify

        # prepare for tc segmentation 
        se = SegmentationAnalysis(image=seg_cmn.image_1, 
                                  boundary=seg_cmn.bound_1)
        se.setSegmentationParameters(boundaryIds=[3, 4], nBoundary=1, 
                                     boundCount='at_least', mask=5)
        se.setAnalysisParameters(distanceRegion=seg_cmn.bound_1, 
                        distanceId=([3], [3,4]), distanceMode=('min', 'mean'),
                        distanceName=('min_3', 'mean_3_4'),
                                 cleftLayers=cleft_layers)
        tc_iter = se.tcSegmentAnalyze(
            thresh=seg_cmn.threshold_1, order='<', count=True, doDensity=True, 
            doMorphology=True, doLength=True, doTopology=True, 
            doDistanceTo=True, doBoundDistance=True, doCleftContacts=True)

        # make hierarchy
        id_dict = {}
        for count in range(len(seg_cmn.threshold_1)):
            seg, level, curr_thresh = tc_iter.next()
        tc = tc_iter.send(True) 
 
        # set classification parameters and desired results
        se.addClassificationParameters(type='volume', 
                                       args={'volumes':[0, 10, 50]})
        se.addClassificationParameters(type='contacted_ids', 
                                       args={'ids':[3], 'rest':True})
        desired_names = ['_vol-0-10_bound-3', '_vol-0-10_bound-rest', 
                         '_vol-10-50_bound-3', '_vol-10-50_bound-rest']
        desired_ids = [[1,2,3,4,6,7,10], [5,8,9], [11,12,13,14], []]

        # classify and test 
        ind = 0
        for new, name in se.classifyAnalyze(hierarchy=tc, doDensity=False, 
            doMorphology=False, doLength=False, doTopology=False, 
            doDistanceTo=False, doBoundDistance=True, doCleftContacts=True):
            
            # test classification
            np_test.assert_equal(name, desired_names[ind])
            np_test.assert_equal(new.hierarchy.ids, desired_ids[ind])
        
            # test getLabels()
            np_test.assert_equal(new.labels, new.hierarchy)
            np_test.assert_equal(new.ids, new.hierarchy.ids)
        
            # test analysis
            np_test.assert_almost_equal(
                new.morphology.volume[new.morphology.ids], 
                seg_cmn.volume_1[new.hierarchy.ids])
            np_test.assert_almost_equal(
                new.topology.euler[new.topology.ids], 
                seg_cmn.euler_1[new.hierarchy.ids])
            np_test.assert_almost_equal(
                new.topology.nFaces[new.topology.ids], 
                seg_cmn.n_faces_1[new.hierarchy.ids])
            np_test.assert_almost_equal(new.density.mean[new.density.ids], 
                                        seg_cmn.density_1[new.hierarchy.ids])
            np_test.assert_almost_equal(
                new.min_3.distance[new.min_3.ids], 
                seg_cmn.distance_to_3_min_1[new.hierarchy.ids])
            np_test.assert_almost_equal(
                new.mean_3_4.distance[new.mean_3_4.ids], 
                seg_cmn.distance_to_3_4_mean_1[new.hierarchy.ids])
            np_test.assert_almost_equal(
                new.mean_3_4.closestRegion[new.mean_3_4.ids], 
                seg_cmn.closest_region_1[new.hierarchy.ids])
        
            # test cleft contacts
            np_test.assert_almost_equal(new.nContacts, desired_n_contacts[ind])
            np_test.assert_almost_equal(new.surfaceDensityContacts, 
                                        desired_surface_density_contacts[ind])
            np_test.assert_almost_equal(new.surfaceDensitySegments, 
                                        desired_surface_density_segments[ind])

            ind += 1 

        ######################################################
        #
        # Scenario 2:
        #   - segment wo analysis
        #   - classify and analyze

        # prepare for tc segmentation 
        se = SegmentationAnalysis(image=seg_cmn.image_1, 
                                  boundary=seg_cmn.bound_1)
        se.setSegmentationParameters(boundaryIds=[3, 4], nBoundary=1, 
                                     boundCount='at_least', mask=5)
        se.setAnalysisParameters(distanceRegion=seg_cmn.bound_1, 
                        distanceId=([3], [3,4]), distanceMode=('min', 'mean'),
                        distanceName=('min_3', 'mean_3_4'),
                                 cleftLayers=cleft_layers)
        tc_iter = se.tcSegmentAnalyze(
            thresh=seg_cmn.threshold_1, order='<', count=True, doDensity=False, 
            doMorphology=False, doLength=False, doTopology=False, 
            doDistanceTo=False, doBoundDistance=True, doCleftContacts=False)

        # make hierarchy
        for count in range(len(seg_cmn.threshold_1)):
            seg, level, curr_thresh = tc_iter.next()
        tc = tc_iter.send(True) 
 
        # set classification parameters and desired results
        se.addClassificationParameters(type='volume', 
                                       args={'volumes':[0, 10, 50]})
        se.addClassificationParameters(type='contacted_ids', 
                                       args={'ids':[3], 'rest':True})
        desired_names = ['_vol-0-10_bound-3', '_vol-0-10_bound-rest', 
                         '_vol-10-50_bound-3', '_vol-10-50_bound-rest']
        desired_ids = [[1,2,3,4,6,7,10], [5,8,9], [11,12,13,14], []]

        # classify and test 
        ind = 0
        for new_2, name in se.classifyAnalyze(hierarchy=tc, doDensity=True, 
            doMorphology=True, doLength=True, doTopology=True, 
            doDistanceTo=True, doBoundDistance=True, doCleftContacts=True):

            # test classification
            np_test.assert_equal(name, desired_names[ind])
            np_test.assert_equal(new_2.hierarchy.ids, desired_ids[ind])

            # test analysis
            np_test.assert_almost_equal(
                new_2.morphology.volume[new_2.morphology.ids], 
                seg_cmn.volume_1[new_2.hierarchy.ids])
            np_test.assert_almost_equal(
                new_2.topology.euler[new_2.topology.ids], 
                seg_cmn.euler_1[new_2.hierarchy.ids])
            if len(new_2.hierarchy.ids) > 0:                
                np_test.assert_almost_equal(
                    new_2.topology.nFaces[new_2.topology.ids,:], 
                    seg_cmn.n_faces_1[new_2.hierarchy.ids])
            else:
                np_test.assert_equal(len(new_2.topology.nFaces), 0)
            np_test.assert_almost_equal(
                new_2.density.mean[new_2.density.ids], 
                seg_cmn.density_1[new_2.hierarchy.ids])
            np_test.assert_almost_equal(
                new_2.min_3.distance[new_2.min_3.ids], 
                seg_cmn.distance_to_3_min_1[new_2.hierarchy.ids])
            np_test.assert_almost_equal(
                new_2.mean_3_4.distance[new_2.mean_3_4.ids], 
                seg_cmn.distance_to_3_4_mean_1[new_2.hierarchy.ids])
            np_test.assert_almost_equal(
                new_2.mean_3_4.closestRegion[new_2.mean_3_4.ids], 
                seg_cmn.closest_region_1[new_2.hierarchy.ids])

            # test cleft contacts
            np_test.assert_almost_equal(new_2.nContacts, 
                                        desired_n_contacts[ind])
            np_test.assert_almost_equal(new_2.surfaceDensityContacts, 
                                        desired_surface_density_contacts[ind])
            np_test.assert_almost_equal(new_2.surfaceDensitySegments, 
                                        desired_surface_density_segments[ind])

            ind += 1 


if __name__ == '__main__':
    suite = \
        unittest.TestLoader().loadTestsFromTestCase(TestSegmentationAnalysis)
    unittest.TextTestRunner(verbosity=2).run(suite)
