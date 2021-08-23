"""

Tests module numpy_plus.

Currently tests trim_slice() only.

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id$
"""
from __future__ import unicode_literals

__version__ = "$Revision$"

from copy import copy, deepcopy
import unittest

import numpy
import numpy.testing as np_test 
import scipy

from  pyto.util.numpy_plus import trim_slice

class TestNumpyPlus(np_test.TestCase):
    """
    """

    def test_trim_slice(self):
        """
        Tests trim_slice()
        """

        # inset inside
        result = trim_slice(slice_nd=[slice(1, 3), slice(2, 5)], 
                                    shape=[4, 5])
        desired = ([slice(1, 3), slice(2, 5)], [slice(0, 2), slice(0, 3)])
        np_test.assert_equal(result, desired)

        # partially outside
        result = trim_slice(slice_nd=[slice(3, 6), slice(3, 6)], 
                                    shape=[4, 5])
        desired = ((slice(3, 4), slice(3, 5)), (slice(0, 1), slice(0, 2)))
        np_test.assert_equal(result, desired)

        # partially outside, negative side
        result = trim_slice(slice_nd=[slice(-2, 1), slice(3, 6)], 
                                    shape=[4, 5])
        desired = ((slice(0, 1), slice(3, 5)), (slice(2, 3), slice(0, 2)))
        np_test.assert_equal(result, desired)

        # completely outside
        result = trim_slice(slice_nd=[slice(2, 4), slice(6, 8)], 
                                    shape=[4, 5])
        np_test.assert_equal(result[0] is None, True)
        np_test.assert_equal(result[1] is None, True)

        # slices supersed of shape
        result = trim_slice(slice_nd=[slice(-2, 6), slice(2, 5)], 
                                    shape=[4, 5])
        desired = ((slice(0, 4), slice(2, 5)), (slice(2, 6), slice(0, 3)))
        np_test.assert_equal(result, desired)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestNumpyPlus)
    unittest.TextTestRunner(verbosity=2).run(suite)
