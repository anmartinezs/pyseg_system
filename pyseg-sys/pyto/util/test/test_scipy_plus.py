"""

Tests module pyto.util.scipy_plus.

# Author: Vladan Lucic
# $Id: test_scipy_plus.py 1062 2014-10-10 15:31:02Z vladan $
"""

__version__ = "$Revision: 1062 $"

import unittest

import numpy
import numpy.testing as np_test 
import scipy

from pyto.util.scipy_plus import *


class TestScipyPlus(np_test.TestCase):
    """
    """

    def setUp(self):
        """
        """
        pass

    def test_chisquare_2(self):
        """
        Tests chisquare2()
        """

        # Statistics for the biological sciences, pg 110
        chisq, p = chisquare_2(f_obs_1=numpy.array([20, 30]), 
                               f_obs_2=numpy.array([24, 26]),
                               yates=True)
        np_test.assert_almost_equal(chisq, 0.364, decimal=2)
        np_test.assert_equal(p>0.25, True)
        np_test.assert_almost_equal(p, 0.546, decimal=3)
    
        # Statistics for the biological sciences, pg 111
        chisq, p = chisquare_2(f_obs_1=numpy.array([60, 32, 28]), 
                               f_obs_2=numpy.array([28, 17, 45]))
        np_test.assert_almost_equal(chisq, 16.23, decimal=2)
        np_test.assert_equal(p<0.005, True)
        np_test.assert_almost_equal(p, 0.0003, decimal=4)
        desired = scipy.stats.chi2.sf(chisq, 2)
        np_test.assert_almost_equal(p, desired, decimal=4)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestScipyPlus)
    unittest.TextTestRunner(verbosity=2).run(suite)
