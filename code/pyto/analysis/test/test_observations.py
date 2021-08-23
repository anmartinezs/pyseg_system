"""

Tests module observations
 
# Author: Vladan Lucic
# $Id$
"""
from __future__ import unicode_literals
from __future__ import division
from builtins import zip
from builtins import range
#from past.utils import old_div

__version__ = "$Revision$"

import sys
from copy import copy, deepcopy
import pickle
import os.path
import unittest

import numpy
import numpy.testing as np_test 
import scipy

from pyto.analysis.catalog import Catalog
from pyto.analysis.observations import Observations

# set output
out = sys.stdout  # tests output, but prints lots of things 
out = None        # clean output but doesn't test print


class TestObservations(np_test.TestCase):
    """
    """

    def setUp(self):
        """
        """
        self.obs = self.makeInstance()
        
    @classmethod
    def makeInstance(cls):
        """
        """
        obs = Observations()
        obs.identifiers = ['exp_1', 'exp_2', 'exp_5', 'exp_6']
        obs.categories = ['odd', 'even', 'odd', 'even']
        obs.ids = [numpy.array([1,3,5]), numpy.array([1,2,3]),
                   numpy.array([]), numpy.array([6])]
        obs.mean = [numpy.array([2., 6, 10]), numpy.array([2., 4, 6]),
                    numpy.array([]), numpy.array([12])]
        obs.pair = [numpy.array([[1,2], [3,4], [5,6]]), 
                    numpy.array([[2,4], [6,8], [10,12]]), 
                    numpy.array([]), numpy.array([[20,40]])]
        obs.scalar = [1, 2, 5, 6]
        obs.indexed = set(['ids', 'mean', 'pair'])
        obs.properties = set(['ids', 'identifiers', 'categories', 'mean', 
                              'scalar', 'pair'])

        return obs

    def testInitializeProperties(self):
        """
        Test initializeProperties()
        """

        # one name
        obs = Observations()
        obs.initializeProperties(names='aha')
        np_test.assert_equal('aha' in obs.properties, True)
        np_test.assert_equal(obs.aha, [])

        # more than one name
        obs = Observations()
        names = ['a', 'b']
        obs.initializeProperties(names=names)
        np_test.assert_equal(obs.properties.issuperset(set(names)), True)
        for nam in names:
            np_test.assert_equal(getattr(obs, nam), [])

    def test_get_indexed_data(self):
        """
        Tests get_indexed_data()
        """

        # standard
        ind_data = self.obs.indexed_data
        np_test.assert_equal(
            ind_data.columns.tolist(), ['identifiers', 'ids', 'mean', 'pair'])
        np_test.assert_equal(
            ind_data.identifiers.unique().tolist(),
            ['exp_1', 'exp_2', 'exp_6'])
        np_test.assert_equal(
            ind_data.ids.tolist(), numpy.hstack(self.obs.ids))
        np_test.assert_equal(
            ind_data['mean'].tolist(), numpy.hstack(self.obs.mean))
        np_test.assert_equal(
            ind_data.pair.tolist(),
            [[1,2], [3,4], [5,6], [2,4], [6,8], [10,12], [20,40]])

        # some identifiers
        ind_data = self.obs.get_indexed_data(
            identifiers=['exp_6', 'dummy', 'exp_2'], additional=['scalar'])
        np_test.assert_equal(
            ind_data.columns.tolist(),
            ['identifiers', 'ids', 'mean', 'pair', 'scalar'])
        np_test.assert_equal(ind_data.ids.tolist(), [6, 1, 2, 3])
        np_test.assert_equal(ind_data['mean'].tolist(), [12, 2, 4, 6])
        np_test.assert_equal(ind_data.scalar.tolist(), [6, 2, 2, 2])

        # names and additional
        ind_data = self.obs.get_indexed_data(
            identifiers=['exp_6', 'dummy', 'exp_2'],
            names=['pair', 'mean'], additional=['scalar'])
        np_test.assert_equal(
            ind_data.columns.tolist(),
            ['identifiers', 'ids', 'pair', 'mean', 'scalar'])

        # no experiemnts
        empty = Observations()
        data = empty.get_indexed_data()
        np_test.assert_equal(data is None, True)
        
    def test_get_scalar_data(self):
        """
        Tests get_scalar_data()
        """

        # no args
        data = self.obs.scalar_data
        np_test.assert_equal(
            data.columns.tolist(), ['identifiers', 'categories', 'scalar'])
        np_test.assert_equal(
            data.identifiers.unique().tolist(),
            ['exp_1', 'exp_2', 'exp_5', 'exp_6'])
        np_test.assert_equal(
            data.scalar.tolist(), self.obs.scalar)
         
        # some args
        data = self.obs.get_scalar_data(
            identifiers=['exp_6', 'dummy', 'exp_2'])
        np_test.assert_equal(
            data.columns.tolist(), ['identifiers', 'categories', 'scalar'])
        np_test.assert_equal(
            data.identifiers.unique().tolist(), ['exp_6', 'exp_2'])
        np_test.assert_equal(data.scalar.tolist(), [6, 2])
        
        # some args
        data = self.obs.get_scalar_data(
            identifiers=['exp_6', 'dummy', 'exp_2'], names=['scalar'])
        np_test.assert_equal(
            data.columns.tolist(), ['identifiers', 'scalar'])
        np_test.assert_equal(
            data.identifiers.unique().tolist(), ['exp_6', 'exp_2'])
        np_test.assert_equal(data.scalar.tolist(), [6, 2])
        
        # some args
        data = self.obs.get_scalar_data(
            identifiers=['exp_6', 'dummy', 'exp_2'],
            names=['scalar', 'categories'])
        np_test.assert_equal(
            data.columns.tolist(), ['identifiers', 'scalar', 'categories'])
        np_test.assert_equal(
            data.identifiers.unique().tolist(), ['exp_6', 'exp_2'])
        np_test.assert_equal(data.scalar.tolist(), [6, 2])

        # no experiemnts
        empty = Observations()
        data = empty.get_scalar_data()
        np_test.assert_equal(data is None, True)
        
    def testGetExperiment(self):
        """
        Tests getExperiment()
        """

        ex_2 = self.obs.getExperiment(identifier='exp_2')
        np_test.assert_equal(ex_2.identifier, 'exp_2')
        np_test.assert_equal(ex_2.indexed, self.obs.indexed)
        np_test.assert_equal(self.obs.properties.difference(ex_2.properties),
                             set(['identifiers']))
        np_test.assert_equal('identifier' in ex_2.properties, True)
        np_test.assert_equal(ex_2.ids, numpy.array([1,2,3]))
        np_test.assert_almost_equal(ex_2.mean, numpy.array([2, 4, 6]))

    def testGetExperimentIndex(self):
        """
        Tests getExperimentIndex()
        """
        np_test.assert_equal(self.obs.getExperimentIndex(identifier='exp_1'), 0)
        np_test.assert_equal(self.obs.getExperimentIndex(identifier='exp_6'), 3)

    def testAddExperiment(self):
        """
        Tests addExperiment()
        """

        # get experiments
        ex_6 = self.obs.getExperiment(identifier='exp_6')
        ex_2 = self.obs.getExperiment(identifier='exp_2')
        ex_1 = self.obs.getExperiment(identifier='exp_1')
        ex_5 = self.obs.getExperiment(identifier='exp_5')

        # mane new instance
        new = Observations()

        # add one and check
        new.addExperiment(experiment=ex_1)
        np_test.assert_equal(new.identifiers, ['exp_1'])
        np_test.assert_equal(new.properties, 
                             set(['ids', 'identifiers', 'categories', 'mean', 
                                  'scalar', 'pair']))
        np_test.assert_equal(new.indexed, set(['ids', 'mean', 'pair']))
        np_test.assert_equal(new.ids, [[1,3,5]])
        np_test.assert_equal(new.categories, ['odd'])
        np_test.assert_equal(new.mean, [[2., 6, 10]])

        # add all and check
        new.addExperiment(experiment=ex_2)
        new.addExperiment(experiment=ex_5)
        new.addExperiment(experiment=ex_6)
        np_test.assert_equal(new.identifiers, 
                             ['exp_1', 'exp_2', 'exp_5', 'exp_6'])
        np_test.assert_equal(new.properties, 
                             set(['ids', 'identifiers', 'categories', 'mean',
                                  'scalar', 'pair']))
        np_test.assert_equal(new.indexed, set(['ids', 'mean', 'pair']))
        np_test.assert_equal(new.ids, [[1,3,5], [1,2,3], [], [6]])
        np_test.assert_equal(new.categories, ['odd', 'even', 'odd', 'even'])
        np_test.assert_equal(new.mean, [[2., 6, 10], [2., 4, 6], [], [12]])

    def testAddData(self):
        """
        Tests addData()
        """

        # make two instances
        obs = self.makeInstance()
        obs_2 = self.makeInstance()
        def plus(x, y): return x + y 
        obs_2.apply(funct=plus, args=['mean'], kwargs={'y':10}, name='mean_2')
        obs_2.scalar_2 = [11, 22, 55, 66]
        obs_2.properties.add('scalar_2')
        
        # general test
        obs.addData(source=obs_2, names=['mean_2', 'scalar_2'])
        np_test.assert_equal('mean_2' in obs.properties, True)
        np_test.assert_equal('mean_2' in obs.indexed, True)
        np_test.assert_equal(obs.mean_2, 
                             [[12., 16, 20], [12., 14, 16], [], [22]])
        np_test.assert_equal('scalar_2' in obs.properties, True)
        np_test.assert_equal('scalar_2' in obs.indexed, False)
        np_test.assert_equal(obs.scalar_2, [11, 22, 55, 66]) 

        # test copy = True
        exp_1_mean_2 = obs_2.getValue(property='mean_2', identifier='exp_1')
        exp_1_mean_2[1] = 26
        np_test.assert_equal(
            obs.getValue(property='mean_2', identifier='exp_1'), 
            [12, 16, 20])
        np_test.assert_equal(
            obs_2.getValue(property='mean_2', identifier='exp_1'), 
            [12, 26, 20])

        # test copy = False
        obs.addData(source=obs_2, names=['mean_2'], copy=False)
        exp_1_mean_2[1] = 36
        np_test.assert_equal(
            obs.getValue(property='mean_2', identifier='exp_1'), 
            [12, 36, 20])

        # test changing names
        obs = self.makeInstance()
        obs_2 = self.makeInstance()
        def plus(x, y): return x + y 
        obs_2.apply(funct=plus, args=['mean'], kwargs={'y':10}, name='mean_2')
        obs_2.scalar_2 = [11, 22, 55, 66]
        obs_2.properties.add('scalar_2')
        obs.addData(source=obs_2, 
                    names={'mean_2':'mean_3', 'scalar_2':'scalar_3'})
        np_test.assert_equal('mean_2' not in obs.properties, True)
        np_test.assert_equal('mean_3' in obs.properties, True)
        np_test.assert_equal('mean_3' in obs.indexed, True)
        np_test.assert_equal(obs.mean_3, 
                             [[12., 16, 20], [12., 14, 16], [], [22]])
        np_test.assert_equal('scalar_3' in obs.properties, True)
        np_test.assert_equal('scalar_3' in obs.indexed, False)
        np_test.assert_equal(obs.scalar_3, [11, 22, 55, 66]) 

    def testJoinExperiments(self):
        """
        Tests joinExperiments()
        """
        
        # scalar property, mode join
        exp = self.obs.joinExperiments(name='scalar', mode='join')
        np_test.assert_equal(len(exp.scalar), 4) 
        np_test.assert_equal(set(exp.scalar), set([1, 2, 5, 6]))
        
        # scalar property first, mode join, identifiers
        exp = self.obs.joinExperiments(name=['scalar', 'mean'], mode='join',
                                       identifiers=['exp_6', 'exp_5', 'exp_1'])
        np_test.assert_equal(exp.scalar, [6, 5, 1])
        np_test.assert_equal(exp.ids, list(range(1,4)))
        np_test.assert_equal(exp.mean, [12, 2, 6, 10])
        np_test.assert_equal(set(exp.properties), 
                             set(['identifier', 'scalar', 'mean', 
                                  'ids', 'idNames']))
        np_test.assert_equal(set(exp.indexed), 
                             set(['ids', 'idNames', 'scalar']))

        # indexed property first, mode join, identifiers
        exp = self.obs.joinExperiments(name=['mean', 'scalar'], mode='join',
                                       identifiers=['exp_5', 'exp_2', 'exp_6'])
        np_test.assert_equal(exp.scalar, [5, 2, 6])
        np_test.assert_equal(exp.ids, list(range(1,5)))
        np_test.assert_equal(exp.mean, [2, 4, 6, 12])
        np_test.assert_equal(set(exp.properties), 
                             set(['identifier', 'scalar', 'mean', 
                                  'ids', 'idNames']))
        np_test.assert_equal(set(exp.indexed), 
                             set(['ids', 'idNames', 'mean']))

    def testRemove(self):
        """
        Tests remove()
        """

        obs = self.makeInstance()

        # test nonexistant removal
        np_test.assert_equal(obs.identifiers, 
                             ['exp_1', 'exp_2', 'exp_5', 'exp_6'])
        obs.remove(identifier='exp_7')
        np_test.assert_equal(obs.identifiers, 
                             ['exp_1', 'exp_2', 'exp_5', 'exp_6'])
        
        # test one removal
        np_test.assert_equal(obs.identifiers, 
                             ['exp_1', 'exp_2', 'exp_5', 'exp_6'])
        obs.remove(identifier='exp_5')
        np_test.assert_equal(obs.identifiers, 
                             ['exp_1', 'exp_2', 'exp_6'])
        np_test.assert_equal(len(obs.mean), 3)
        np_test.assert_equal(
            obs.ids,
            [numpy.array([1,3,5]), numpy.array([1,2,3]), numpy.array([6])])
        np_test.assert_equal(
            obs.mean,
            [numpy.array([2., 6, 10]), numpy.array([2., 4, 6]),
             numpy.array([12])])
        np_test.assert_equal(
            obs.pair,
            [numpy.array([[1,2], [3,4], [5,6]]), 
             numpy.array([[2,4], [6,8], [10,12]]), 
             numpy.array([[20,40]])])
        np_test.assert_equal(obs.scalar, [1, 2, 6])

        # test another removal
        obs.remove(identifier='exp_2')
        np_test.assert_equal(obs.identifiers, 
                             ['exp_1', 'exp_6'])
        np_test.assert_equal(len(obs.mean), 2)
        np_test.assert_equal(
            obs.ids,
            [numpy.array([1,3,5]), numpy.array([6])])
        np_test.assert_equal(
            obs.mean,
            [numpy.array([2., 6, 10]), numpy.array([12])])
        np_test.assert_equal(
            obs.pair,
            [numpy.array([[1,2], [3,4], [5,6]]), numpy.array([[20,40]])])
        np_test.assert_equal(obs.scalar, [1, 6])

        # test non-existing identifier another time
        np_test.assert_equal(
            obs.ids,
            [numpy.array([1,3,5]), numpy.array([6])])
        obs.remove(identifier='exp_2')
        np_test.assert_equal(len(obs.identifiers), 2)
        np_test.assert_equal(
            obs.ids,
            [numpy.array([1,3,5]), numpy.array([6])])

    def testKeep(self):
        """
        Tests keep()
        """

        # test 
        obs = self.makeInstance()
        np_test.assert_equal(obs.identifiers, 
                             ['exp_1', 'exp_2', 'exp_5', 'exp_6'])
        obs.keep(identifiers=['exp_2', 'exp_6'])
        np_test.assert_equal(len(obs.mean), 2)
        np_test.assert_equal(
            obs.ids, [numpy.array([1,2,3]), numpy.array([6])])
        np_test.assert_equal(
            obs.mean,
            [numpy.array([2., 4, 6]), numpy.array([12])])
        np_test.assert_equal(
            obs.pair,
            [numpy.array([[2,4], [6,8], [10,12]]), 
             numpy.array([[20,40]])])
        np_test.assert_equal(obs.scalar, [2, 6])

        # test 
        obs = self.makeInstance()
        np_test.assert_equal(obs.identifiers, 
                             ['exp_1', 'exp_2', 'exp_5', 'exp_6'])
        obs.keep(identifiers=['exp_13', 'exp_1', 'exp_15', 'exp_6'])
        np_test.assert_equal(obs.identifiers, ['exp_1', 'exp_6'])
        np_test.assert_equal(len(obs.mean), 2)
        np_test.assert_equal(
            obs.ids, [numpy.array([1,3,5]), numpy.array([6])])
        np_test.assert_equal(
            obs.mean, [numpy.array([2., 6, 10]), numpy.array([12])])

    def testGetValue(self):
        """
        Tests getValue()
        """
        
        value = self.obs.getValue(identifier='exp_2', property='categories')
        np_test.assert_equal(value, 'even')
        value = self.obs.getValue(identifier='exp_6', property='mean')
        np_test.assert_equal(value, numpy.array([12]))
        value = self.obs.getValue(identifier='exp_5', property='ids') 
        np_test.assert_equal(value, numpy.array([]))

        # with ids
        value = self.obs.getValue(identifier='exp_1', property='mean')
        np_test.assert_equal(value, numpy.array([2,6,10]))
        value = self.obs.getValue(identifier='exp_1', property='mean', ids=3)
        np_test.assert_equal(value, 6)
        value = self.obs.getValue(identifier='exp_1', property='mean', ids=5)
        np_test.assert_equal(value, 10)
        value = self.obs.getValue(identifier='exp_1', property='mean', 
                                  ids=[3,5,1])
        np_test.assert_equal(value, [6, 10, 2])
        value = self.obs.getValue(identifier='exp_2', property='mean', 
                                  ids=[3,2])
        np_test.assert_equal(value, [6, 4])

        # using name argument
        value = self.obs.getValue(identifier='exp_2', name='categories')
        np_test.assert_equal(value, 'even')
        value = self.obs.getValue(identifier='exp_6', name='mean')
        np_test.assert_equal(value, numpy.array([12]))
        value = self.obs.getValue(identifier='exp_5', name='ids') 
        np_test.assert_equal(value, numpy.array([]))

        value = self.obs.getValue(identifier='exp_1', name='mean')
        np_test.assert_equal(value, numpy.array([2,6,10]))
        value = self.obs.getValue(identifier='exp_1', name='mean', ids=3)
        np_test.assert_equal(value, 6)
        value = self.obs.getValue(identifier='exp_1', name='mean', ids=5)
        np_test.assert_equal(value, 10)
        value = self.obs.getValue(identifier='exp_1', name='mean', 
                                  ids=[3,5,1])
        np_test.assert_equal(value, [6, 10, 2])
        value = self.obs.getValue(identifier='exp_2', name='mean', 
                                  ids=[3,2])
        np_test.assert_equal(value, [6, 4])
 
        # non-existing identifier
        np_test.assert_raises(
            ValueError, self.obs.getValue, 
            {'identifier':'exp_22', 'name':'mean'})

    def testSetValue(self):
        """
        Tests setValue()
        """

        obs = self.makeInstance()

        # changing a non-indexed value
        value = obs.getValue(identifier='exp_6', property='mean')
        np_test.assert_almost_equal(value, numpy.array([12]))
        obs.setValue(identifier='exp_6', property='mean', 
                     value=numpy.array([13.])) 
        value = obs.getValue(identifier='exp_6', property='mean')
        desired = [numpy.array([2., 6, 10]), numpy.array([2., 4, 6]),
                    numpy.array([]), numpy.array([13.])]
        np_test.assert_equal(obs.mean, desired)

        # argument name
        obs.setValue(identifier='exp_6', name='mean', 
                     value=numpy.array([14.])) 
        value = obs.getValue(identifier='exp_6', name='mean')
        desired = [numpy.array([2., 6, 10]), numpy.array([2., 4, 6]),
                   numpy.array([]), numpy.array([14.])]
        np_test.assert_equal(obs.mean, desired)

        # new property, non-indexed
        obs.setValue(identifier='exp_5', property='size', value='large', 
                     default='small')
        for current, desired in zip(obs.size, 
                                    ['small', 'small', 'large', 'small']):
            np_test.assert_equal(current, desired) 
        np_test.assert_equal('size' in obs.properties, True)

        # new indexed property
        new_std = numpy.array([5, 4, 3])
        obs.setValue(identifier='exp_1', property='std', 
                     value=new_std, indexed=True, default=-1) 
        value = obs.getValue(identifier='exp_1', property='std')
        np_test.assert_almost_equal(value, new_std)
        for current, desired in zip(obs.std, [new_std, [-1,-1,-1], [], [-1]]):
            try:
                np_test.assert_almost_equal(current, desired)
            except TypeError:
                if not ((current is None) and (desired is None)):
                    raise
        desired = set(['ids', 'identifiers', 'categories', 'std', 
                       'mean', 'size', 'scalar', 'pair'])
        np_test.assert_equal(obs.properties, desired)

        # change one element of the previous property
        obs.setValue(identifier='exp_1', property='std', 
                     value=10, id_=3, indexed=True) 
        value = obs.getValue(identifier='exp_1', property='std')
        np_test.assert_almost_equal(value, [5, 10, 3])
        obs.setValue(identifier='exp_1', property='std', 
                     value=15, id_=5, indexed=True) 
        value = obs.getValue(identifier='exp_1', property='std')
        np_test.assert_almost_equal(value, [5, 10, 15])
        single_value = obs.getValue(identifier='exp_1', property='std', ids=5)
        np_test.assert_almost_equal(15, single_value)
 
        # change more than one element of the previous property
        obs.setValue(identifier='exp_1', property='std', 
                     value=[11, 14], id_=[3, 5], indexed=True) 
        value = obs.getValue(identifier='exp_1', property='std')
        np_test.assert_almost_equal(value, [5, 11, 14])

        # existing experiment and new property, single indexed
        np_test.assert_equal('volume2' in obs.properties, False)
        obs.setValue(identifier='exp_1', property='volume2', 
                     value=33, indexed=True, id_=1, default=-1)
        np_test.assert_equal('volume2' in obs.properties, True)
        np_test.assert_equal('volume2' in obs.indexed, True)
        value = obs.getValue(identifier='exp_1', property='volume2', ids=1)
        np_test.assert_almost_equal(value, 33) 
        value = obs.getValue(identifier='exp_1', property='volume2', ids=3)
        np_test.assert_almost_equal(value, -1) 
        value = obs.getValue(identifier='exp_5', property='volume2')
        np_test.assert_almost_equal(value, []) 
        value = obs.getValue(identifier='exp_6', property='volume2')
        np_test.assert_almost_equal(value, [-1]) 

         # existing experiment and new property, >1 indexed
        np_test.assert_equal('volume22' in obs.properties, False)
        obs.setValue(identifier='exp_1', property='volume22', 
                     value=[33,55], indexed=True, id_=[1,5], default=-1)
        np_test.assert_equal('volume22' in obs.properties, True)
        np_test.assert_equal('volume22' in obs.indexed, True)
        value = obs.getValue(identifier='exp_1', property='volume22', ids=1)
        np_test.assert_almost_equal(value, 33) 
        value = obs.getValue(identifier='exp_1', property='volume22', ids=3)
        np_test.assert_almost_equal(value, -1) 
        value = obs.getValue(identifier='exp_1', property='volume22', ids=5)
        np_test.assert_almost_equal(value, 55) 
        value = obs.getValue(identifier='exp_5', property='volume22')
        np_test.assert_almost_equal(value, []) 
        value = obs.getValue(identifier='exp_6', property='volume22')
        np_test.assert_almost_equal(value, [-1]) 

        # existing experiment and new property, indexed all ids
        np_test.assert_equal('volume3' in obs.properties, False)
        obs.setValue(identifier='exp_2', property='volume3', 
                     value=[33,34,35], indexed=True, default=-1)
        np_test.assert_equal('volume3' in obs.properties, True)
        np_test.assert_equal('volume3' in obs.indexed, True)
        value = obs.getValue(identifier='exp_1', property='volume3')
        np_test.assert_almost_equal(value, [-1, -1, -1]) 
        value = obs.getValue(identifier='exp_2', property='volume3')
        np_test.assert_almost_equal(value, [33, 34, 35]) 
 
        # all existing experiments, new property, indexed default values
        np_test.assert_equal('volume4' in obs.properties, False)
        obs.setValue(identifier=None, property='volume4', 
                     indexed=True, default=-4)
        np_test.assert_equal(
            obs.volume4,
            [[-4, -4, -4], [-4,-4,-4], [], [-4]])
        np_test.assert_equal('volume4' in obs.properties, True)
        np_test.assert_equal('volume4' in obs.indexed, True)
      
        # existing experiment and new property, non-indexed
        np_test.assert_equal('scalar3' in obs.properties, False)
        obs.setValue(identifier='exp_5', property='scalar3', 
                     value=33, indexed=False, default=-3)
        value = obs.getValue(identifier='exp_1', property='scalar3')
        np_test.assert_almost_equal(value, -3) 
        value = obs.getValue(identifier='exp_2', property='scalar3')
        np_test.assert_almost_equal(value, -3) 
        value = obs.getValue(identifier='exp_5', property='scalar3')
        np_test.assert_almost_equal(value, 33) 
        value = obs.getValue(identifier='exp_6', property='scalar3')
        np_test.assert_almost_equal(value, -3) 

        # new non-indexed property, same value for all experiments
        obs.setValue(property='common', default='ok')
        np_test.assert_equal(obs.common, ['ok'] * len(obs.identifiers))
        np_test.assert_equal('common' in obs.properties, True)
        np_test.assert_equal(obs.common, ['ok','ok','ok','ok']) 

        # new experiment but existing property, non-indexed
        np_test.assert_equal('exp_55' in obs.identifiers, False) 
        obs.setValue(identifier='exp_55', property='size', value='v_large')
        np_test.assert_equal('exp_55' in obs.identifiers, True) 
        np_test.assert_equal(
            obs.getValue(property='size', identifier='exp_55'), 'v_large')
        obs.setValue(identifier='exp_56', property='volume', value=[2,3,4])
        np_test.assert_equal(
            obs.getValue(property='volume', identifier='exp_56'), [2,3,4])

        # new experiment but existing property, indexed
        np_test.assert_equal('exp_57' in obs.identifiers, False) 
        np_test.assert_raises(
            ValueError, obs.setValue, [],
            {'identifier':'exp_57', 'property':'volume', 'value':57, 
             'indexed':True})
        np_test.assert_equal('exp_57' in obs.identifiers, False) 

        # new experiment and new property, non-indexed
        np_test.assert_equal('exp_12' in obs.identifiers, False) 
        obs.setValue(identifier='exp_12', property='scalar_12', 
                     value=12, indexed=False, default=-2) 
        np_test.assert_equal('exp_12' in obs.identifiers, True) 
        np_test.assert_equal('exp_12' in obs.indexed, False) 
        np_test.assert_equal(
            obs.getValue(identifier='exp_12', name='scalar_12'), 12) 
        np_test.assert_equal(
            obs.getValue(identifier='exp_2', name='scalar_12'), -2) 
 
        # new experiment and new property, indexed, all values
        np_test.assert_equal('exp_11' in obs.identifiers, False) 
        np_test.assert_equal('volume_11' not in obs.properties, True)
        np_test.assert_equal('volume_11' not in obs.indexed, True)
        obs.setValue(identifier='exp_11', property='volume_11', 
                     value=[11, 11.1], indexed=True, default=-11) 
        value = obs.getValue(identifier='exp_11', name='volume_11')
        np_test.assert_almost_equal([11, 11.1], value) 
        value = obs.getValue(identifier='exp_1', name='volume_11')
        np_test.assert_almost_equal([-11, -11, -11], value) 
        np_test.assert_equal('exp_11' in obs.identifiers, True) 
        np_test.assert_equal('volume_11' in obs.properties, True)
        np_test.assert_equal('volume_11' in obs.indexed, True)
 
        # new experiment and new property, indexed, single values
        np_test.assert_equal('exp_13' in obs.identifiers, False) 
        np_test.assert_equal('volume_13' not in obs.properties, True)
        np_test.assert_equal('volume_13' not in obs.indexed, True)
        np_test.assert_raises(
            ValueError, obs.setValue, [],
            {'identifier':'exp_13', 'property':'volume_13', 
             'value':13, 'indexed':True, 'id_':3,'default':-13})
        np_test.assert_equal('exp_13' in obs.identifiers, False) 
        np_test.assert_equal('volume_13' not in obs.properties, True)
        np_test.assert_equal('volume_13' not in obs.indexed, True)
 
       # start from empty object
        empty = Observations()
        empty.setValue(property='data', identifier='i1', value=1)
        np_test.assert_equal(empty.properties, set(['identifiers', 'data']))
        np_test.assert_equal(empty.identifiers, ['i1'])
        np_test.assert_equal(
            empty.getValue(property='data', identifier='i1'), 1)
        empty.setValue(property='data', identifier='i3', value=3)
        np_test.assert_equal(empty.properties, set(['identifiers', 'data']))
        np_test.assert_equal(empty.identifiers, ['i1', 'i3'])
        np_test.assert_equal(
            empty.getValue(property='data', identifier='i3'), 3)

        #

    def testExtract(self):
        """
        Tests extract()
        """

        obs = self.makeInstance()

        # extract by condition
        cond = [[True, True, False], [False, True, False], [], [True]]
        extracted = obs.extract(condition=cond)
        np_test.assert_equal(extracted.indexed, obs.indexed)
        np_test.assert_equal(extracted.properties, obs.properties)
        np_test.assert_equal(extracted.ids, [[1, 3], [2], [], [6]])
        np_test.assert_equal(extracted.mean, [[2., 6], [4.], [], [12]])
        np_test.assert_equal(extracted.scalar, [1, 2, 5, 6])
        np_test.assert_equal(
            extracted.pair,
            [[[1,2], [3,4]], [[6,8]], [], [[20,40]]])

        # extract by values

    def testApply(self):
        """
        Tests apply()
        """

        obs = self.makeInstance()
        def plus(x, y): return x + y 

        # name is None
        res = obs.apply(funct=plus, args=['mean'], kwargs={'y':10})
        np_test.assert_equal(res, [[12,16,20], [12,14,16], [], [22]])
        
        # name given
        obs.apply(funct=plus, args=['mean'], kwargs={'y':10}, name='new')
        np_test.assert_equal(obs.new, [[12,16,20], [12,14,16], [], [22]])
        np_test.assert_equal('new' in obs.properties, True)
        np_test.assert_equal('new' in obs.indexed, True)

        # more than one arg
        res2 = obs.apply(funct=numpy.subtract, args=['new', 'mean'])
        np_test.assert_equal(res2, [[10,10,10], [10,10,10], [], [10]])

        # more than one arg and name
        obs.apply(funct=numpy.subtract, args=('new', 'mean'), name='ten')
        np_test.assert_equal(obs.ten, [[10,10,10], [10,10,10], [], [10]])
        np_test.assert_equal('ten' in obs.properties, True)
        np_test.assert_equal('new' in obs.indexed, True)

        # overwrite propery
        obs.apply(funct=plus, args=['mean'], kwargs={'y':10}, name='mean')
        np_test.assert_equal(obs.mean, [[12,16,20], [12,14,16], [], [22]])

        # scalar
        obs.apply(funct=plus, args=['scalar'], kwargs={'y':10}, 
                  name='schifted_scalar', indexed=False)
        np_test.assert_equal('schifted_scalar' in obs.properties, True)
        np_test.assert_equal('schifted_scalar' in obs.indexed, False)
        np_test.assert_equal(obs.schifted_scalar, [11, 12, 15, 16])

    def testPixels2nm(self):
        """
        Tests pixels2nm()
        """

        obs = self.makeInstance()
        conversion = {'exp_1':0.5, 'exp_2':1, 'exp_5':2, 'exp_6':3}

        val = obs.pixels2nm(name='mean', conversion=conversion)
        des = [obs.mean[0]*0.5, obs.mean[1], obs.mean[2]*2., obs.mean[3]*3.]
        for actual, desired in zip(val, des):
            np_test.assert_almost_equal(actual, desired)

        val = obs.pixels2nm(name='mean', conversion=conversion, power=2)
        des = [obs.mean[0]*0.25, obs.mean[1], obs.mean[2]*4., obs.mean[3]*9.]
        for actual, desired in zip(val, des):
            np_test.assert_almost_equal(actual, desired)

        val = obs.pixels2nm(name='mean', conversion=conversion, power=-2)
        des = [obs.mean[0]/0.25, obs.mean[1], obs.mean[2]/4., obs.mean[3]/9.]
        for actual, desired in zip(val, des):
            np_test.assert_almost_equal(actual, desired)        

        val = obs.pixels2nm(name='scalar', conversion=conversion, power=-2)
        des = [obs.scalar[0]/0.25, obs.scalar[1], 
               obs.scalar[2]/4., obs.scalar[3]/9.]
        for actual, desired in zip(val, des):
            np_test.assert_almost_equal(actual, desired)        

    def testAddCatalog(self):
        """
        Tests addCatalog()
        """

        # make catalog
        cat = Catalog()
        cat._db = {'file' : {'exp_1' : 'file_1', 'exp_6' : 'file_6', 
                             'exp_2' : 'file_2', 'exp_5' : 'file_5'}, 
                   'order' : {'exp_2' : 'first', 'exp_5' : 'second',
                              'exp_1' : 'first', 'exp_6' : 'second'}
                   }

        # test
        obs = self.makeInstance()
        obs.addCatalog(catalog=cat)
        np_test.assert_equal(obs.file, ['file_1', 'file_2', 'file_5', 'file_6'])
        np_test.assert_equal(obs.order, ['first', 'first', 'second', 'second'])
        desired = set(['ids', 'identifiers', 'categories', 'order', 
                       'file', 'mean', 'scalar', 'pair'])
        np_test.assert_equal(obs.properties, desired)

    def testDoStats(self):
        """
        Test doStats() without arg bins
        """

        # all identifiers
        obs = self.makeInstance()
        stats = obs.doStats(name='mean', new=True)
        np_test.assert_equal(stats.identifiers, obs.identifiers)
        np_test.assert_equal(stats.properties, 
                             set(['identifiers', 'data', 'mean', 'std', 
                                  'n', 'sem']))
        np_test.assert_equal(stats.data, obs.mean)
        np_test.assert_equal(
                stats.getValue(property='data', identifier='exp_1'), [2,6,10])
        np_test.assert_equal(
                stats.getValue(property='data', identifier='exp_2'), [2,4,6])
        np_test.assert_equal(
                stats.getValue(property='data', identifier='exp_5'), [])
        np_test.assert_equal(
                stats.getValue(property='data', identifier='exp_6'), [12])
        np_test.assert_equal(numpy.array(stats.mean)[[0,1,3]], [6, 4, 12])
        np_test.assert_equal(numpy.isnan(stats.mean[2]), True)
        np_test.assert_equal(numpy.isnan(
            stats.getValue(property='mean', identifier='exp_5')), True)
        np_test.assert_almost_equal(
            numpy.array(stats.std)[[0,1,2,3]], 
            [4., 2., numpy.nan, numpy.nan])
        np_test.assert_almost_equal(
            numpy.array(stats.sem)[[0,1,3]], 
            [4/numpy.sqrt(3), 2/numpy.sqrt(3), numpy.nan])
        np_test.assert_equal(numpy.isnan(stats.sem[2]), True)
        np_test.assert_equal(stats.n, [3, 3, 0, 1])

        # some identifiers
        obs = self.makeInstance()
        some_ident = ['exp_11', 'exp_6', 'exp_5', 'exp_2']
        stats = obs.doStats(name='mean', new=True, identifiers=some_ident)
        np_test.assert_equal(stats.identifiers, ['exp_6', 'exp_5', 'exp_2'])
        np_test.assert_equal(stats.properties, 
                             set(['identifiers', 'data', 'mean', 'std', 
                                  'n', 'sem']))
        np_test.assert_almost_equal(
                stats.getValue(property='data', identifier='exp_2'), [2,4,6])
        np_test.assert_almost_equal(
                stats.getValue(property='data', identifier='exp_5'), [])
        np_test.assert_almost_equal(
                stats.getValue(property='data', identifier='exp_6'), [12])
        np_test.assert_almost_equal(
                stats.getValue(property='mean', identifier='exp_2'), 4)
        np_test.assert_almost_equal(
                stats.getValue(property='mean', identifier='exp_6'), 12)
        np_test.assert_equal(
            numpy.isnan(stats.getValue(property='mean', identifier='exp_5')), 
                        True)
        np_test.assert_almost_equal(
                stats.getValue(property='std', identifier='exp_2'), 2)
        np_test.assert_almost_equal(
                stats.getValue(property='std', identifier='exp_5'), numpy.nan)
        np_test.assert_almost_equal(
                stats.getValue(property='std', identifier='exp_6'), numpy.nan)
        np_test.assert_almost_equal(
                stats.getValue(property='sem', identifier='exp_2'), 
                2/numpy.sqrt(3))
        np_test.assert_almost_equal(
                stats.getValue(property='sem', identifier='exp_6'), numpy.nan)
        np_test.assert_equal(
            numpy.isnan(stats.getValue(property='sem', identifier='exp_5')), 
                        True)
        np_test.assert_equal(
                stats.getValue(property='n', identifier='exp_2'), 3)
        np_test.assert_equal(
                stats.getValue(property='n', identifier='exp_5'), 0)
        np_test.assert_equal(
                stats.getValue(property='n', identifier='exp_6'), 1)

        # effectivly no identifiers
        obs = self.makeInstance()
        some_ident = ['exp_11', 'exp_16']
        stats = obs.doStats(name='mean', new=True, identifiers=some_ident)
        names = ['data', 'mean', 'std', 'n', 'sem']
        np_test.assert_equal(stats.properties.issuperset(set(names)), True)
        for nam in names:
            np_test.assert_equal(getattr(stats, nam), [])

        # nan
        obs = self.makeInstance()
        obs.setValue(identifier='exp_2', name='mean', id_=3, value=numpy.nan)
        stats = obs.doStats(name='mean', new=True, remove_nan=True)
        np_test.assert_almost_equal(
            stats.getValue(property='data', identifier='exp_1'), [2,6,10])
        np_test.assert_almost_equal(
            stats.getValue(property='data', identifier='exp_2'), [2,4])
        np_test.assert_almost_equal(
            stats.getValue(property='mean', identifier='exp_2'), 3)
        np_test.assert_almost_equal(
            stats.getValue(property='std', identifier='exp_2'), numpy.sqrt(2))

    def testDoStatsBins(self):
        """
        Test doStats() with bins argument  
        """

        # all identifiers
        obs = self.makeInstance()
        stats = obs.doStats(name='mean', bins=[0,5,20], fraction=0, new=True)
        np_test.assert_equal(stats.identifiers, obs.identifiers)
        np_test.assert_equal(
            stats.properties, 
            set(['ids', 'identifiers', 'data', 'histogram', 'probability', 
                 'fraction', 'n']))
        np_test.assert_equal(
            stats.indexed, set(['ids', 'histogram', 'probability']))
        np_test.assert_equal(
                stats.getValue(property='ids', identifier='exp_2'), [0,5])
        np_test.assert_equal(
                stats.getValue(property='histogram', identifier='exp_1'), 
                [1, 2])
        np_test.assert_equal(
                stats.getValue(property='histogram', identifier='exp_2'), 
                [2, 1])
        np_test.assert_equal(
                stats.getValue(property='histogram', identifier='exp_5'), 
                [0, 0])
        np_test.assert_equal(
                stats.getValue(property='histogram', identifier='exp_6'), 
                [0, 1])
        np_test.assert_equal(
                stats.getValue(property='probability', identifier='exp_2'), 
                [2/3., 1/3.])
        np_test.assert_equal(
                stats.getValue(property='probability', identifier='exp_6'), 
                [0, 1])
        np_test.assert_equal(
                stats.getValue(property='fraction', identifier='exp_1'), 1/3.)
        np_test.assert_equal(
                stats.getValue(property='fraction', identifier='exp_5'), 
                numpy.nan)
        np_test.assert_equal(
                stats.getValue(property='n', identifier='exp_2'), 3)
        np_test.assert_equal(
                stats.getValue(property='n', identifier='exp_6'), 1)
        np_test.assert_equal(
                stats.getValue(property='data', identifier='exp_6'), [12])
        np_test.assert_equal(
                stats.getValue(property='data', identifier='exp_1'), 
                [2, 6, 10.])

        # nan
        obs = self.makeInstance()
        obs.setValue(identifier='exp_2', name='mean', id_=2, value=numpy.nan)
        stats = obs.doStats(name='mean', new=True, bins=[0,5,20], 
                            fraction=0, remove_nan=True)
        np_test.assert_almost_equal(
            stats.getValue(property='histogram', identifier='exp_1'), [1,2])
        np_test.assert_almost_equal(
            stats.getValue(property='histogram', identifier='exp_2'), [1,1])

    def testDoStatsByIndex(self): 
        """
        Tests doStatsByIndex()
        """

        obs = self.makeInstance()
        stats = obs.doStatsByIndex(name='mean', identifiers=['exp_1', 'exp_2'],
                                   identifier='ident')

        np_test.assert_equal(
            stats.properties, 
            set(['ids', 'identifier', 'mean', 'std', 'sem', 'n', 'data'])) 
        np_test.assert_equal(stats.indexed, set(['ids', 'mean', 'std', 'sem'])) 
        np_test.assert_equal(stats.ids, [1, 2, 3]) 
        np_test.assert_almost_equal(stats.data, 
                                    numpy.array([[2, 6, 10], [2, 4, 6]]))
        np_test.assert_almost_equal(stats.mean, [2, 5, 8])
        np_test.assert_almost_equal(stats.std, 
                                    [0, numpy.sqrt(2), numpy.sqrt(8)])
        np_test.assert_almost_equal(stats.n, 2)
        np_test.assert_almost_equal(stats.sem, [0, 1, 2])
        np_test.assert_equal(stats.identifier, 'ident')

    def testDoInference(self):
        """
        Tests doInfrence
        """

        obs = self.makeInstance()
        stats = obs.doStats(name='mean', new=True)

        # single reference
        reference='exp_1'
        stats.doInference(test='t', reference=reference)
        np_test.assert_equal(
            stats.properties, 
            set(['identifiers', 'data', 'mean', 'std', 'n',  
                 'sem', 'testValue', 'confidence', 'testSymbol', 'reference']))
        np_test.assert_equal(stats.testSymbol, ['t'] * 4), 
        np_test.assert_almost_equal(numpy.array(stats.testValue)[[0,1,3]], 
                                    [0, -0.7746, numpy.nan], decimal=3)
        np_test.assert_equal(numpy.isnan(stats.testValue[2]), True)
        np_test.assert_almost_equal(numpy.array(stats.confidence)[[0,1,3]], 
                                    [1, 0.4818, numpy.nan], decimal=3)
        np_test.assert_equal(numpy.isnan(stats.confidence[2]), True)
        np_test.assert_equal(stats.reference, 
                             ['exp_1', 'exp_1', 'exp_1', 'exp_1'])

        # single reference, some identifiers
        some_ident = ['exp_11', 'exp_6', 'exp_5', 'exp_1']
        stats2 = obs.doStats(name='mean', new=True, identifiers=some_ident)
        reference='exp_1'
        stats2.doInference(test='t', reference=reference, 
                          identifiers=['exp_1', 'exp_5', 'exp_6'])
        np_test.assert_equal(stats2.identifiers, ['exp_6', 'exp_5', 'exp_1'])
        np_test.assert_equal(
            stats2.properties, 
            set(['identifiers', 'data', 'mean', 'std', 'n',  
                 'sem', 'testValue', 'confidence', 'testSymbol', 'reference']))
        np_test.assert_almost_equal(
                stats2.getValue(property='testValue', identifier='exp_1'), 0)
        np_test.assert_almost_equal(
                stats2.getValue(property='testValue', identifier='exp_6'), 
                numpy.nan, decimal=3)
        np_test.assert_equal(
            numpy.isnan(stats2.getValue(property='testValue', 
                                       identifier='exp_5')), 
            True)
        np_test.assert_equal(stats2.reference, 
                             ['exp_1', 'exp_1', 'exp_1'])

        # another single reference
        stats.doInference(test='t', reference='exp_2')
        np_test.assert_almost_equal(numpy.array(stats.testValue)[[0,1,3]], 
                                    [0.7746, 0, numpy.nan], decimal=3)
        np_test.assert_equal(numpy.isnan(stats.testValue[2]), True)

        # multiple reference, dictionary
        reference={'exp_1':'exp_2', 'exp_2':'exp_1', 
                   'exp_5':'exp_2', 'exp_6':'exp_2'}
        stats.doInference(test='t', reference=reference)
        np_test.assert_almost_equal(numpy.array(stats.testValue)[[0,1,3]], 
                                    [0.7746, -0.7746, numpy.nan], decimal=3)
        np_test.assert_equal(numpy.isnan(stats.testValue[2]), True)
        for ident in stats.identifiers:
            np_test.assert_equal(
                stats.getValue(identifier=ident, property='reference'),  
                reference[ident])

        # multiple reference, list
        reference=['exp_2', 'exp_1', 'exp_2', 'exp_2']
        stats.doInference(test='t', reference=reference)
        np_test.assert_almost_equal(numpy.array(stats.testValue)[[0,1,3]], 
                                    [0.7746, -0.7746, numpy.nan], decimal=3)
        np_test.assert_equal(numpy.isnan(stats.testValue[2]), True)
        np_test.assert_equal(stats.reference, reference)

        # effectivly no identifiers

    def testDoCorrelation(self):
        """
        Tests doCorrelation()
        """

        # add another indexed property
        obs = self.makeInstance()
        def plus(x, y): return x + y 
        obs.apply(funct=plus, args=['mean'], kwargs={'y':10}, name='mean_2')

        # mode None, new
        corr = obs.doCorrelation(xName='mean', yName='mean_2', mode=None, 
                                 test='r', new=True, out=out)
        np_test.assert_equal(
            corr.properties, 
            set(['xData','yData','testValue','testSymbol','confidence','n',
                 'identifiers'])) 
        np_test.assert_equal(corr.xData, obs.mean)
        for actual, desired in zip(corr.yData, obs.mean):
            np_test.assert_equal(actual, numpy.array(desired) + 10)
        np_test.assert_equal(corr.n, [3, 3, 0, 1])
        np_test.assert_equal(corr.testSymbol, ['r'] * 4)
        for ind in [0, 1]:
            np_test.assert_equal(corr.testValue[ind], 1)
            np_test.assert_equal(
                corr.confidence[ind], scipy.stats.pearsonr([1,2,3], [1,2,3])[1])
        for ind in [2, 3]:
            np_test.assert_equal(numpy.isnan(corr.testValue[ind]), True)
            np_test.assert_equal(numpy.isnan(corr.confidence[ind]), True)
        
        # mode None, new, identifiers
        corr = obs.doCorrelation(xName='mean', yName='mean_2', mode=None, 
                                 test='r', new=True, 
                                 identifiers=['exp_6', 'exp_1'], out=out)
        np_test.assert_equal(
            corr.properties, 
            set(['xData','yData','testValue','testSymbol','confidence','n',
                 'identifiers'])) 
        np_test.assert_equal(corr.xData, [[12], [2, 6, 10]])
        np_test.assert_equal(corr.yData, [[22], [12, 16, 20]])
        np_test.assert_equal(corr.n, [1, 3])
        np_test.assert_equal(corr.testSymbol, ['r'] * 2)
        for ind in [1]:
            np_test.assert_equal(corr.testValue[ind], 1)
            np_test.assert_equal(
                corr.confidence[ind], scipy.stats.pearsonr([1,2,3], [1,2,3])[1])
        for ind in [0]:
            np_test.assert_equal(numpy.isnan(corr.testValue[ind]), True)
            np_test.assert_equal(numpy.isnan(corr.confidence[ind]), True)
        
        # mode None, no new
        obs.doCorrelation(xName='mean', yName='mean_2', mode=None, test='r', 
                          new=False, out=out)
        np_test.assert_equal(
             set(['testValue','testSymbol','confidence','n',
                 'identifiers']) <= obs.properties, True)
        np_test.assert_equal(obs.n, [3, 3, 0, 1])
        np_test.assert_equal(obs.testSymbol, ['r'] * 4)
        for ind in [0, 1]:
            np_test.assert_equal(obs.testValue[ind], 1)
            np_test.assert_equal(
                obs.confidence[ind], scipy.stats.pearsonr([1,2,3], [1,2,3])[1])
        for ind in [2, 3]:
            np_test.assert_equal(numpy.isnan(obs.testValue[ind]), True)
            np_test.assert_equal(numpy.isnan(obs.confidence[ind]), True)

        # mode 'join', new, indexed property
        exp = obs.doCorrelation(xName='mean', yName='mean_2', mode='join', 
                                test='r', new=True, out=out)
        np_test.assert_equal(
            exp.properties, 
            set(['xData','yData','testValue','testSymbol','confidence','n',
                 'identifier'])) 
        np_test.assert_equal(exp.xData, [2, 6, 10, 2, 4, 6, 12])
        np_test.assert_equal(exp.yData, [12, 16, 20, 12, 14, 16, 22])
        desired = scipy.stats.pearsonr([2, 6, 10, 2, 4, 6, 12], 
                                       [12, 16, 20, 12, 14, 16, 22])
        np_test.assert_equal(exp.testValue, desired[0])
        np_test.assert_equal(exp.confidence, desired[1])

        # mode 'join', new, identifiers, indexed property
        exp = obs.doCorrelation(xName='mean', yName='mean_2', mode='join', 
                                test='r', new=True, 
                                identifiers=['exp_6', 'exp_1'], out=out)
        np_test.assert_equal(
            exp.properties, 
            set(['xData','yData','testValue','testSymbol','confidence','n',
                 'identifier'])) 
        x_data = [12, 2, 6, 10]
        y_data = [22, 12, 16, 20]
        np_test.assert_equal(exp.xData, x_data)
        np_test.assert_equal(exp.yData, y_data)
        desired = scipy.stats.pearsonr(x_data, y_data)
        np_test.assert_equal(exp.testValue, desired[0])
        np_test.assert_equal(exp.confidence, desired[1])

        # mode 'join', new, scalar property
        obs.scalar_2 = [1,3,4,6]
        obs.properties.add('scalar_2')
        exp = obs.doCorrelation(xName='scalar', yName='scalar_2', mode='join', 
                                test='r', new=True, out=out)
        np_test.assert_equal(
            exp.properties, 
            set(['xData','yData','testValue','testSymbol','confidence','n',
                 'identifier'])) 
        np_test.assert_equal(exp.xData, [1, 2, 5, 6])
        np_test.assert_equal(exp.yData, [1, 3, 4, 6])
        desired = scipy.stats.pearsonr([1, 2, 5, 6], [1, 3, 4, 6])
        np_test.assert_equal(exp.testValue, desired[0])
        np_test.assert_equal(exp.confidence, desired[1])


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestObservations)
    unittest.TextTestRunner(verbosity=2).run(suite)
