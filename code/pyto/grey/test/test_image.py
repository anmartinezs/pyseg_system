"""

Tests module image

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

from pyto.grey.image import Image
#import common


class TestImage(np_test.TestCase):
    """
    """

    def setUp(self):
        """
        """
        pass

    def testLimit(self):
        """
        Test limit()
        """

        image = self.makeImage()
        image.limit(limit=2, mode='std', size=3)
        np_test.assert_almost_equal(image.data[5, 5], 5)
        np_test.assert_almost_equal(image.data[7, 7], 0)
        desired = numpy.array([
                [10, 10, 0],
                [10, 10, 0],
                [0, 0, 0]])
        np_test.assert_almost_equal(image.data[0:3, 0:3], desired)


    def makeImage(self):
        """
        Returns an image
        """

        data = numpy.zeros(100).reshape(10,10)
        data[0:3, 0:3] = 10
        data[5, 5] = 5
        data[7, 7] = 7

        image = Image(data=data)
        return image


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestImage)
    unittest.TextTestRunner(verbosity=2).run(suite)
