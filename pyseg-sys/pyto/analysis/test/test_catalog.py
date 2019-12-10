"""

Tests module catalog

# Author: Vladan Lucic
# $Id: test_catalog.py 1485 2018-10-04 14:35:01Z vladan $
"""

__version__ = "$Revision: 1485 $"

from copy import copy, deepcopy
import unittest
import os.path

import numpy
import numpy.testing as np_test 
import scipy

from pyto.analysis.catalog import Catalog
import common


class TestCatalog(np_test.TestCase):
    """
    """

    def setUp(self):
        """
        """

        #self.relative_dir = 'all_catalogs'

    def makeInstance(self):
        """
        Returns an istance of Catalog()
        """

        curr_dir, base = os.path.split(__file__)
        rel_dir =  os.path.join(curr_dir, common.rel_catalogs_dir)
        catal = Catalog() 
        catal.read(dir=rel_dir, catalog=r'catalog_[0-9]*\.', 
                   type='distributed')

        return catal

    def testRead(self):
        """
        Tests read
        """

        ###################################################
        #
        # Absolute path to catalogs directory
        #

        curr_dir, base = os.path.split(os.path.abspath(__file__))
        abs_dir =  os.path.join(curr_dir, common.rel_catalogs_dir)
        catal = Catalog() 
        catal.read(dir=abs_dir, catalog=r'catalog_[0-9]*\.', 
                   type='distributed')
        #print 'catal._db: ', catal._db

        # test feature
        np_test.assert_equal(catal._db['feature']['exp_1'], 'good')
        np_test.assert_equal(catal._db['feature']['exp_2'], 'bad')
        np_test.assert_equal(catal._db['feature']['exp_3'], 'good')
        np_test.assert_equal(catal._db['feature']['exp_4'], 'bad')

        # test results
        desired = os.path.realpath(os.path.join(curr_dir, 'catalogs_b', 
                                               'results_1.dat'))
        np_test.assert_equal(
            os.path.realpath(catal._db['results_file']['exp_1']), desired)
        desired = os.path.realpath(os.path.join(curr_dir, 'catalogs_b', 
                                               'results_2.dat'))
        np_test.assert_equal(
            os.path.realpath(catal._db['results_file']['exp_2']), desired)
        desired = os.path.realpath(os.path.join(curr_dir, 'catalogs_a', 
                                               'results_3.dat'))
        np_test.assert_equal(
            os.path.realpath(catal._db['results_file']['exp_3']), desired)
        desired = os.path.realpath(os.path.join(curr_dir, 'catalogs_a', 
                                      'catalogs_c', 'res', 'results_4.dat'))
        np_test.assert_equal(
            os.path.realpath(catal._db['results_file']['exp_4']), desired)

        ###################################################
        #
        # Absolute path to catalogs directory, test identifiers
        #

        catal = Catalog() 
        catal.read(dir=abs_dir, catalog=r'catalog_[0-9]*\.', 
                   type='distributed', identifiers=['exp_2', 'exp_3'])

        # test feature
        np_test.assert_equal(catal._db['feature'].get('exp_1', 'foo'), 'foo')
        np_test.assert_equal(catal._db['feature']['exp_2'], 'bad')
        np_test.assert_equal(catal._db['feature']['exp_3'], 'good')
        np_test.assert_equal(catal._db['feature'].get('exp_4', 'foo'), 'foo')

        ###################################################
        #
        # Absolute path to catalogs directory
        #

        curr_dir, base = os.path.split(__file__)
        rel_dir =  os.path.join(curr_dir, common.rel_catalogs_dir)
        catal = Catalog() 
        catal.read(dir=rel_dir, catalog=r'catalog_[0-9]*\.', 
                   type='distributed')
        #print 'catal._db: ', catal._db

        # test results
        desired = os.path.realpath(os.path.join(curr_dir, 'catalogs_b', 
                                               'results_1.dat'))
        np_test.assert_equal(catal._db['results_file']['exp_1'], desired)
        desired = os.path.realpath(os.path.join(curr_dir, 'catalogs_b', 
                                               'results_2.dat'))
        np_test.assert_equal(catal._db['results_file']['exp_2'], desired)
        desired = os.path.realpath(os.path.join(curr_dir, 'catalogs_a', 
                                               'results_3.dat'))
        np_test.assert_equal(catal._db['results_file']['exp_3'], desired)
        desired = os.path.realpath(os.path.join(curr_dir, 'catalogs_a', 
                                      'catalogs_c', 'res', 'results_4.dat'))
        np_test.assert_equal(catal._db['results_file']['exp_4'], desired)

    def testMakeGroups(self):
        """
        Tests makeGroups
        """

        # read db
        #curr_dir, base = os.path.split(__file__)
        #rel_dir =  os.path.join(curr_dir, common.rel_catalogs_dir)
        #catal = Catalog() 
        #catal.read(dir=rel_dir, catalog=r'catalog_[0-9]*\.', 
        #           type='distributed')
        #print 'catal._db: ', catal._db

        # use all identifiers
        catal = self.makeInstance()
        catal.makeGroups()
        #for name in catal.getProperties():
        #    print "property: ", name, "  value: ", getattr(catal, name)
        desired = {'first' : {'exp_2':'bad', 'exp_3':'good'},
                   'second' : {'exp_1':'good', 'exp_4':'bad'}}
        np_test.assert_equal(catal.feature, desired)

        # specify all identifiers
        catal = self.makeInstance()
        catal.makeGroups(include=['exp_2', 'exp_3', 'exp_1', 'exp_4'])
        desired = {'first' : {'exp_2':'bad', 'exp_3':'good'},
                   'second' : {'exp_1':'good', 'exp_4':'bad'}}
        np_test.assert_equal(catal.feature, desired)

        # specify some identifiers
        catal = self.makeInstance()
        catal.makeGroups(include=['exp_2', 'exp_3', 'exp_4'])
        desired = {'first' : {'exp_2':'bad', 'exp_3':'good'},
                   'second' : {'exp_4':'bad'}}
        np_test.assert_equal(catal.feature, desired)

        # specify some identifiers
        catal = self.makeInstance()
        catal.makeGroups(include=['exp_2', 'exp_3'])
        desired = {'first' : {'exp_2':'bad', 'exp_3':'good'}}
        np_test.assert_equal(catal.feature, desired)

        # some properties not in all experiments
        catal = self.makeInstance()
        catal._db['partial'] = {'exp_2' : 2, 'exp_3' : 3}
        catal._db['categ'] = {'exp_2' : 2, 'exp_3' : 3}
        # print catal._db.keys()
        # failed to find name that comes first in _db.keys()
        catal.makeGroups()
        desired = {'first' : {'exp_2':2, 'exp_3':3}}
        np_test.assert_equal(catal.partial, desired)
        desired = {'first' : {'exp_2':'bad', 'exp_3':'good'},
                   'second' : {'exp_1':'good', 'exp_4':'bad'}}
        np_test.assert_equal(catal.feature, desired)

        # use all identifiers
        catal = self.makeInstance()
        catal.makeGroups()
        #for name in catal.getProperties():
        #    print "property: ", name, "  value: ", getattr(catal, name)
        desired = {'first' : {'exp_2':'bad', 'exp_3':'good'},
                   'second' : {'exp_1':'good', 'exp_4':'bad'}}
        np_test.assert_equal(catal.feature, desired)

        # put everything in one (dummy) group
        catal = self.makeInstance()
        catal.makeGroups(
            feature=None, singleGroupName='ggg', singleFeature='fff')
        desired = set(
            ['results_file', 'results_obj', 'cleft_columns_obj', 'feature',
             'pixel_size', 'category', 'fff'])
        np_test.assert_equal(set(catal.getProperties()), desired)
        desired = {'ggg' : {'exp_2':'ggg', 'exp_3':'ggg', 
                    'exp_1':'ggg', 'exp_4':'ggg'}}
        np_test.assert_equal(catal.fff, desired)
        desired = {'ggg' : {
            'exp_2':'bad', 'exp_3':'good', 'exp_1':'good', 'exp_4':'bad'}}
        np_test.assert_equal(catal.feature, desired)

    def testGetProperties(self):
        """
        Tests getProperties()
        """

        # make instance
        catal = self.makeInstance()
        catal.makeGroups()

        # test
        props = catal.getProperties()
        desired = set(
            ['results_file', 'results_obj', 'cleft_columns_obj', 'feature',
             'pixel_size', 'category'])
        np_test.assert_equal(set(props), desired)

    def testAdd(self):
        """
        Tests add()
        """

        # read catalog
        curr_dir, base = os.path.split(os.path.abspath(__file__))
        abs_dir =  os.path.join(curr_dir, common.rel_catalogs_dir)
        catal = Catalog() 
        catal.read(dir=abs_dir, catalog=r'catalog_[0-9]*\.', 
                   type='distributed')

        # add new property
        new_values = {'exp_1' : 'new_1', 'exp_3' : 'new_3'}
        catal.add(name='new', values=new_values)
        np_test.assert_equal(catal._db['new']['exp_1'], 'new_1')
        np_test.assert_equal(catal._db['new'].get('exp_2', "none"), 'none')
        np_test.assert_equal(catal._db['new']['exp_3'], 'new_3')
        np_test.assert_equal(catal._db['new'].get('exp_4', "none"), 'none')
        #print catal._db

        # add an existing property, overwrite
        new_values = {'exp_2' : 'new_2', 'exp_3' : 'new_33'}
        catal.add(name='new', values=new_values, overwrite=True)
        np_test.assert_equal(catal._db['new']['exp_1'], 'new_1')
        np_test.assert_equal(catal._db['new']['exp_2'], 'new_2')
        np_test.assert_equal(catal._db['new']['exp_3'], 'new_33')
        np_test.assert_equal(catal._db['new'].get('exp_4', "none"), 'none')

        # add an existing property, not overwrite
        new_values = {'exp_2' : 'new_2', 'exp_5' : 'new_55'}
        np_test.assert_raises(
            ValueError, catal.add, 
            **{'name':'new', 'values':new_values, 'overwrite':False})

        # test other features
        np_test.assert_equal(catal._db['feature']['exp_1'], 'good')
        np_test.assert_equal(catal._db['feature']['exp_2'], 'bad')
        np_test.assert_equal(catal._db['feature']['exp_3'], 'good')
        np_test.assert_equal(catal._db['feature']['exp_4'], 'bad')

    def testPool(self):
        """
        Test pool()
        """

        # make instance
        catal = self.makeInstance()
        catal.makeGroups(feature='category')

        # test
        catal.pool(categories=['first', 'second'], name='together')
        desired = {'exp_2' : 3.2, 'exp_3' : 3.3} 
        np_test.assert_equal(catal.pixel_size['first'], desired)
        desired = {'exp_1' : 3.1, 'exp_4' : 3.4} 
        np_test.assert_equal(catal.pixel_size['second'], desired)
        desired = {'exp_1' : 3.1, 'exp_2' : 3.2, 'exp_3' : 3.3, 'exp_4' : 3.4} 
        np_test.assert_equal(catal.pixel_size['together'], desired)
        


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCatalog)
    unittest.TextTestRunner(verbosity=2).run(suite)
