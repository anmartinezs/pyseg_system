"""

Tests module experiments
 
# Author: Vladan Lucic
# $Id$
"""
from __future__ import unicode_literals

__version__ = "$Revision$"

import sys
from copy import copy, deepcopy
import pickle
import os.path
import unittest

import numpy
import numpy.testing as np_test 
import scipy

from pyto.analysis.experiment import Experiment


class TestExperiment(np_test.TestCase):
    """
    """

    def setUp(self):
        """
        """
        pass

    def testTransformByIds(self):
        """
        Tests transformByIds()
        """
        
        # same ids
        old_ids = [2, 1, 4, 3]
        old_values = [20, 10, 40, 30]
        new_ids = [3, 2, 1, 4]
        exp = Experiment()
        new_values = exp.transformByIds(
            ids=old_ids, new_ids=new_ids, values=old_values, default=100)
        np_test.assert_equal(new_values, [30, 20, 10, 40])

        # different ids
        old_ids = [4, 1, 7, 6]
        old_values = [40, 10, 70, 60]
        new_ids = [4, 2, 1, 5, 7]
        exp = Experiment()
        new_values = exp.transformByIds(
            ids=old_ids, new_ids=new_ids, values=old_values, default=100)
        np_test.assert_equal(new_values, [40, 100, 10, 100, 70])

        # same ids square form
        old_ids = [2, 1, 5, 3]
        new_ids = [3, 2, 1, 5]
        old_values = numpy.array([[22, 21, 25, 23],
                                  [12, 11, 15, 13],
                                  [52, 51, 55, 53],
                                  [32, 31, 35, 33]])
        exp = Experiment()
        new_values = exp.transformByIds(
            ids=old_ids, new_ids=new_ids, values=old_values, default=100, 
            mode='square')
        np_test.assert_equal(new_values, 
                             numpy.array([[33, 32, 31, 35],
                                          [23, 22, 21, 25],
                                          [13, 12, 11, 15],
                                          [53, 52, 51, 55]]))

        # different ids square form
        old_ids = [2, 1, 5, 3]
        new_ids = [3, 4, 1]
        old_values = numpy.array([[22, 21, 25, 23],
                                  [12, 11, 15, 13],
                                  [52, 51, 55, 53],
                                  [32, 31, 35, 33]])
        exp = Experiment()
        new_values = exp.transformByIds(
            ids=old_ids, new_ids=new_ids, values=old_values, default=100, 
            mode='square')
        np_test.assert_equal(new_values, 
                             numpy.array([[33, 100, 31],
                                          [100, 100, 100],
                                          [13, 100, 11]]))

        # different ids square form
        old_ids = [2, 1, 5, 3]
        new_ids = [3, 6, 1]
        old_values = numpy.array([[22, 21, 25, 23],
                                  [12, 11, 15, 13],
                                  [52, 51, 55, 53],
                                  [32, 31, 35, 33]])
        exp = Experiment()
        new_values = exp.transformByIds(
            ids=old_ids, new_ids=new_ids, values=old_values, default=100, 
            mode='square')
        np_test.assert_equal(new_values, 
                             numpy.array([[33, 100, 31],
                                          [100, 100, 100],
                                          [13, 100, 11]]))

        # same ids vector_pair form
        old_ids = [2, 1, 5, 3]
        new_ids = [3, 2, 1, 5]
        old_values = numpy.array([21, 25, 23, 15, 13, 53])
        exp = Experiment()
        new_values = exp.transformByIds(
            ids=old_ids, new_ids=new_ids, values=old_values, 
            mode='vector_pair')
        np_test.assert_equal(new_values, [23, 13, 53, 21, 25, 15]) 

        # different ids vector_pair form
        old_ids = [2, 1, 7, 4]
        new_ids = [4, 3, 2, 1]
        old_values = numpy.array([21, 72, 24, 17, 14, 74])
        exp = Experiment()
        new_values = exp.transformByIds(
            ids=old_ids, new_ids=new_ids, values=old_values, 
            mode='vector_pair')
        np_test.assert_equal(new_values, [-1, 24, 14, -1, -1, 21]) 

        # different ids vector_pair form
        old_ids = [2, 1, 7, 4]
        new_ids = [4, 8, 2, 1]
        old_values = numpy.array([21, 72, 24, 17, 14, 74])
        exp = Experiment()
        new_values = exp.transformByIds(
            ids=old_ids, new_ids=new_ids, values=old_values, 
            mode='vector_pair')
        np_test.assert_equal(new_values, [-1, 24, 14, -1, -1, 21]) 

        # different ids vector_pair form
        old_ids = [2, 1, 7, 4]
        new_ids = [5, 8, 21]
        old_values = numpy.array([21, 72, 24, 17, 14, 74])
        exp = Experiment()
        new_values = exp.transformByIds(
            ids=old_ids, new_ids=new_ids, values=old_values, default=100,
            mode='vector_pair')
        np_test.assert_equal(new_values, [100, 100, 100]) 



if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestExperiment)
    unittest.TextTestRunner(verbosity=2).run(suite)
