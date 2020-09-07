"""
Tests class Features.


# Author: Vladan Lucic
# $Id$
"""
from __future__ import unicode_literals

__version__ = "$Revision$"

from copy import copy, deepcopy
import unittest

import numpy
import numpy.testing as np_test 
import scipy

from pyto.segmentation.features import Features

class TestFeatures(np_test.TestCase):
    """
    """

    def setUp(self):
        """
        """
        this = Features()
        this.setIds(ids=[1,5])
        this.dataNames = ['data']
        this.data = numpy.array([-1, 10, -1, -1, -1, 50]) 
        self.this = this

    def testExtractOne(self):
        """
        Tests extractOne()
        """
        
        this = deepcopy(self.this)
        actual = this.extractOne(id_=4, array_=True)
        np_test.assert_equal(actual, None)
        actual = this.extractOne(id_=4, array_=False)
        np_test.assert_equal(actual, None)
        actual = this.extractOne(id_=5, array_=False)
        np_test.assert_equal(isinstance(actual, Features), True)
        np_test.assert_equal(actual.data, 50)
        actual = this.extractOne(id_=5, array_=True)
        np_test.assert_equal(actual.data[0], 50)

    def testMerge(self):
        """
        Tests merge
        """

        # replace mode
        this = deepcopy(self.this)
        initial_id = id(this)
        other = deepcopy(this)
        other.setIds(ids=[2, 7])
        other.data = numpy.array([-1, -1, 20, -1, -1, -1, -1, 70])
        this.merge(new=other, mode='replace')
        np_test.assert_equal(this.ids, [1, 2, 5, 7])
        np_test.assert_equal(this.data[this.ids], [10, 20, 50, 70])
        np_test.assert_equal(id(this), initial_id)
        
        # mode replace, mode0 replace
        yet = Features()
        yet.setIds(ids=[4,9])
        yet.dataNames=['data']
        yet.data = numpy.array([49, 0, 0, 0, 40, 0, 0, 0, 0, 90])
        yet2 = Features()
        yet2.setIds(ids=[3,8])
        yet2.dataNames=['data']
        yet2.data = numpy.array([38, 0, 0, 30, 0, 0, 0, 0, 80])
        yet.merge(new=yet2, mode='replace', mode0='replace')
        np_test.assert_equal(yet.ids, [3, 4, 8, 9])
        np_test.assert_equal(yet.data, [38, 0, 0, 30, 40, 0, 0, 0, 80, 90])

        # mode replace, mode0 add
        yet = Features()
        yet.setIds(ids=[4,9])
        yet.dataNames=['data']
        yet.data = numpy.array([49, 0, 0, 0, 40, 0, 0, 0, 0, 90])
        yet2 = Features()
        yet2.setIds(ids=[3,8])
        yet2.dataNames=['data']
        yet2.data = numpy.array([38, 0, 0, 30, 0, 0, 0, 0, 80])
        yet.merge(new=yet2, mode='replace', mode0='add')
        np_test.assert_equal(yet.ids, [3, 4, 8, 9])
        np_test.assert_equal(yet.data, [87, 0, 0, 30, 40, 0, 0, 0, 80, 90])

        # mode replace, mode0 arbitrary number
        yet = Features()
        yet.setIds(ids=[4,9])
        yet.dataNames=['data']
        yet.data = numpy.array([49, 0, 0, 0, 40, 0, 0, 0, 0, 90])
        yet2 = Features()
        yet2.setIds(ids=[3,8])
        yet2.dataNames=['data']
        yet2.data = numpy.array([38, 0, 0, 30, 0, 0, 0, 0, 80])
        yet.merge(new=yet2, mode='replace', mode0=-15)
        np_test.assert_equal(yet.ids, [3, 4, 8, 9])
        np_test.assert_equal(yet.data, [-15, 0, 0, 30, 40, 0, 0, 0, 80, 90])

        # add mode
        yet = Features()
        yet.setIds(ids=[4,9])
        yet.dataNames=['data']
        yet.data = numpy.array([0, 0, 0, 0, 40, 0, 0, 0, 0, 90])
        this.merge(new=yet, mode='add')
        np_test.assert_equal(this.ids, [1, 2, 4, 5, 7, 9])
        np_test.assert_equal(this.data[this.ids], [10, 20, 39, 50, 70, 90])
        np_test.assert_equal(id(this), initial_id)

        # current object no segments
        this = Features()
        this.dataNames = ['data']
        initial_id = id(this)
        other = Features()
        other.dataNames = ['data']
        other.setIds(ids=[2, 7])
        other.data = numpy.array([-1, -1, 20, -1, -1, -1, -1, 70])
        this.merge(new=other, mode='replace')
        np_test.assert_equal(this.ids, [2, 7])
        np_test.assert_equal(this.data[this.ids], [20, 70])
        np_test.assert_equal(id(this), initial_id)
        
        # new object None
        this = Features()
        this.setIds(ids=[1,5])
        this.dataNames = ['data']
        this.data = numpy.array([-1, 10, -1, -1, -1, 50]) 
        initial_id = id(this)
        other = None
        this.merge(new=other, mode='replace')
        np_test.assert_equal(this.ids, [1, 5])
        np_test.assert_equal(this.data[this.ids], [10, 50])
        np_test.assert_equal(id(this), initial_id)

        # no ids in this, other None
        this = Features()
        this.merge(new=None, mode='replace')
        np_test.assert_equal(this.ids is None, True)
        this = Features()
        this.merge(new=None, mode='add')
        np_test.assert_equal(this.ids is None, True)

        # no ids attribute (shouldn't happen, but just in case)
        this = Features()
        del this._ids
        other = Features()
        this.merge(new=other, mode='replace')
        np_test.assert_equal(this.ids, [])


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFeatures)
    unittest.TextTestRunner(verbosity=2).run(suite)
