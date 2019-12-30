"""

Tests module util.bulk . 

Currently tests only bulk module.. 

# Author: Vladan Lucic
# $Id: test_bulk.py 765 2010-10-05 17:15:02Z vladan $
"""

__version__ = "$Revision: 765 $"

import os
import sys
import unittest

import numpy
import numpy.testing as np_test 

import pyto.util.bulk 


class TestBulk(np_test.TestCase):

    def setUp(self):
        """
        """

        # path of this module
        self.this_dir, self.this_base = os.path.split(os.path.abspath(__file__))
       
    def testRunPath(self):
        """
        Tests run_path
        """
 
        # tests calling run_path one by one
        for dir_ in ['dir_1', 'dir_2', 'dir_1']: 

            # check that module is loaded and executed
            module_executed = False
            try:
                path = os.path.join(self.this_dir, dir_, 'script.py')
                #before_module_index = pyto.util.bulk._module_index
                pyto.util.bulk.run_path([path])
            except pyto.util.test.test_bulk._TestBulkError:
                module_executed = True
            np_test.assert_equal(module_executed, True)

            # check module directly
            #module_index = pyto.util.bulk._module_index
            #if module_index - before_module_index == 1:
            #    mod = sys.modules['script_' + str(module_index)]
            #else:
            #    print "something's funny"

    def tearDown(self):
        """
        """

        # nothing for now
        pass


class _TestBulkError(Exception): 
    """
    Exception class used to pass info from executed module
    """
    pass
    

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBulk)
    unittest.TextTestRunner(verbosity=2).run(suite)
