"""
Tests class Labels.

Tests for remove(), reorder(), restrict() and inset related methods are 
implemented in the moment.

# Author: Vladan Lucic
# $Id: test_labels.py 1435 2017-03-27 14:26:36Z vladan $
"""

__version__ = "$Revision: 1435 $"

from copy import copy, deepcopy
import unittest

import numpy
import numpy.testing as np_test 
import scipy

from pyto.segmentation.labels import Labels

class TestLabels(np_test.TestCase):
    """
    """

    def setUp(self):
        
        # make image
        self.small_array = numpy.array(
            [[0, 1, 1, 2, 5],
             [0, 1, 0, 2, 5],
             [0, 1, 0, 0, 7]])
        self.small = Labels(data=self.small_array)
        self.small_2 = Labels(data=numpy.zeros((4,6), dtype='int'))
        self.small_2.data[0:3, 1:6] = self.small_array
        self.labels_1 = Labels(numpy.ones((5,6), dtype='int'))

    def testRemove(self):
        """
        Tests _remove()
        """

        desired = numpy.array(\
            [[0, 6, 6, 2, 5],
             [0, 6, 0, 2, 5],
             [0, 6, 0, 0, 6]])

        # use remove()
        self.small.setData(data=self.small.data.copy())
        self.small.remove(ids=[1, 7], value=6)
        np_test.assert_equal(self.small.data, desired)

        # _remove, mode 'remove'
        removed = self.small._remove(data=self.small_array.copy(), 
                                     remove=[1, 7], value=6, mode='remove')
        np_test.assert_equal(removed, desired)
        removed = self.small._remove(data=self.small_array.copy(), 
                                     keep=[2, 5], value=6, mode='remove')
        np_test.assert_equal(removed, desired)

        # _remove, mode 'remove', working on array inset
        removed = self.small_2._remove(data=self.small_array.copy(), 
                                       remove=[1, 7], value=6, mode='remove')
        np_test.assert_equal(removed, desired)
        
        # _remove, mode 'keep'
        kept = self.small._remove(data=self.small_array.copy(), 
                                  keep=[2, 5], value=6, mode='keep')
        np_test.assert_equal(kept, desired)
        kept = self.small._remove(data=self.small_array.copy(), 
                                  remove=[1, 7], value=6, mode='keep')
        np_test.assert_equal(kept, desired)

        # _remove, mode 'keep'
        kept = self.small_2._remove(data=self.small_array.copy(), 
                                    keep=[2, 5], value=6, mode='keep')
        np_test.assert_equal(kept, desired)

        # remove, mode 'auto', does remove
        removed = self.small._remove(data=self.small_array.copy(), 
                                     remove=[1], value=6, mode='remove')
        desired = numpy.array(\
            [[0, 6, 6, 2, 5],
             [0, 6, 0, 2, 5],
             [0, 6, 0, 0, 7]])
        np_test.assert_equal(removed, desired)

        # remove, mode 'auto', does keep
        removed = self.small._remove(data=self.small_array.copy(), 
                                     keep=[5], value=6, mode='remove')
        desired = numpy.array(\
            [[0, 6, 6, 6, 5],
             [0, 6, 0, 6, 5],
             [0, 6, 0, 0, 6]])
        np_test.assert_equal(removed, desired)

    def testRestrict(self):
        """
        Tests restrict()
        """

        # all ids
        small_2 = Labels(self.small_2.data.copy())
        data = numpy.array(
            [[1, 1, 1, 1, 1, 1, 1],
             [2, 2, 2, 2, 2, 2, 2],
             [3, 3, 3, 3, 3, 3, 3],
             [4, 4, 4, 4, 4, 4, 4]])
        labels = Labels(data)
        labels.restrict(mask=small_2)
        labels.useInset(inset=(slice(0,4), slice(0,8)), 
                        mode='abs', useFull=True, expand=True)
        np_test.assert_equal(
            labels.data,
            numpy.array([[0, 0, 1, 1, 1, 1, 0, 0],
                         [0, 0, 2, 0, 2, 2, 0, 0],
                         [0, 0, 3, 0, 0, 3, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0]]))
    
        # all ids, update=False
        small_2 = Labels(self.small_2.data.copy())
        data = numpy.array(
            [[1, 1, 1, 1, 1, 1, 1],
             [2, 2, 2, 2, 2, 2, 2],
             [3, 3, 3, 3, 3, 3, 3],
             [4, 4, 4, 4, 4, 4, 4]])
        labels = Labels(data)
        new = labels.restrict(mask=small_2, update=False)
        np_test.assert_equal(
            new,
            numpy.array([[0, 0, 1, 1, 1, 1, 0],
                         [0, 0, 2, 0, 2, 2, 0],
                         [0, 0, 3, 0, 0, 3, 0],
                         [0, 0, 0, 0, 0, 0, 0]]))
        np_test.assert_equal(
            labels.data,
            numpy.array([[1, 1, 1, 1, 1, 1, 1],
                         [2, 2, 2, 2, 2, 2, 2],
                         [3, 3, 3, 3, 3, 3, 3],
                         [4, 4, 4, 4, 4, 4, 4]]))

        # some ids
        small_2 = Labels(self.small_2.data.copy())
        small_2_inset = small_2.inset
        data = numpy.array(
            [[1, 1, 1, 1, 1, 1, 1, 1],
             [2, 2, 2, 2, 2, 2, 2, 2],
             [3, 3, 3, 3, 3, 3, 3, 3],
             [4, 4, 4, 4, 4, 4, 4, 4]])
        labels = Labels(data)
        labels.useInset(inset=(slice(0,4), slice(0,7)), mode='abs')
        labels.restrict(mask=small_2, ids=[1, 2, 4])
        labels.useInset(inset=(slice(0,4), slice(0,8)), 
                        mode='abs', useFull=True, expand=True)
        np_test.assert_equal(
            labels.data,
            numpy.array([[0, 0, 1, 1, 1, 1, 0, 1],
                         [0, 0, 2, 0, 2, 2, 0, 2],
                         [3, 3, 3, 3, 3, 3, 3, 3],
                         [0, 0, 0, 0, 0, 0, 0, 4]]))
        np_test.assert_equal(small_2.inset, small_2_inset)
    
        # some ids, update=False
        small_2 = Labels(self.small_2.data.copy())
        small_2_inset = small_2.inset
        data = numpy.array(
            [[1, 1, 1, 1, 1, 1, 1, 1],
             [2, 2, 2, 2, 2, 2, 2, 2],
             [3, 3, 3, 3, 3, 3, 3, 3],
             [4, 4, 4, 4, 4, 4, 4, 4]])
        labels = Labels(data)
        labels.useInset(inset=(slice(0,4), slice(0,7)), mode='abs')
        new = labels.restrict(mask=small_2, ids=[1, 2, 4], update=False)
        np_test.assert_equal(
            new,
            numpy.array([[0, 0, 1, 1, 1, 1, 0],
                         [0, 0, 2, 0, 2, 2, 0],
                         [3, 3, 3, 3, 3, 3, 3],
                         [0, 0, 0, 0, 0, 0, 0]]))
    
        # some ids, mask larger than data
        small_2 = Labels(self.small_2.data.copy())
        small_2_inset = small_2.inset
        data = numpy.array(
            [[1, 1, 1, 1, 1, 1, 1, 1],
             [2, 2, 2, 2, 2, 2, 2, 2],
             [3, 3, 3, 3, 3, 3, 3, 3],
             [4, 4, 4, 4, 4, 4, 4, 4]])
        labels = Labels(data)
        labels.useInset(inset=(slice(1,4), slice(0,5)), mode='abs')
        labels.restrict(mask=small_2, ids=[1, 2, 4])
        labels.useInset(inset=(slice(0,4), slice(0,8)), 
                        mode='abs', useFull=True, expand=True)
        np_test.assert_equal(
            labels.data,
            numpy.array([[1, 1, 1, 1, 1, 1, 1, 1],
                         [0, 0, 2, 0, 2, 2, 2, 2],
                         [3, 3, 3, 3, 3, 3, 3, 3],
                         [0, 0, 0, 0, 0, 4, 4, 4]]))
        np_test.assert_equal(small_2.inset, small_2_inset)

    def testReorder(self):
        """
        Tests reorder
        """
        
        # order without gaps
        small = Labels(data=self.small_array)
        small.setIds(ids=[1,2,5,7])
        order = small.reorder()
        desired = numpy.array(
            [[0, 1, 1, 2, 3],
             [0, 1, 0, 2, 3],
             [0, 1, 0, 0, 4]])
        np_test.assert_equal(small.data, desired)
        np_test.assert_equal(small.ids, [1,2,3,4])
        np_test.assert_equal(order, {1:1, 2:2, 5:3, 7:4})

        # reorder without repetition
        small = Labels(data=self.small_array)
        small.setIds(ids=[1,2,5,7])
        small.reorder(order={1:2, 2:5, 5:1, 7:6})
        desired = numpy.array(
            [[0, 2, 2, 5, 1],
             [0, 2, 0, 5, 1],
             [0, 2, 0, 0, 6]])
        np_test.assert_equal(small.data, desired)
        np_test.assert_equal(small.ids, [1,2,5,6])

        # reorder with repetition
        small = Labels(data=self.small_array)
        small.setIds(ids=[1,2,5,7])
        small.reorder(order={1:2, 2:2, 5:1, 7:2})
        desired = numpy.array(
            [[0, 2, 2, 2, 1],
             [0, 2, 0, 2, 1],
             [0, 2, 0, 0, 2]])
        np_test.assert_equal(small.data, desired)
        np_test.assert_equal(small.ids, [1,2])

        # reorder external
        small = Labels(data=self.small_array)
        small.setIds(ids=[1,2,5,7])
        reordered = small.reorder(order={1:2, 2:5, 5:1, 7:6}, data=small.data)
        desired = numpy.array(
            [[0, 2, 2, 5, 1],
             [0, 2, 0, 5, 1],
             [0, 2, 0, 0, 6]])
        np_test.assert_equal(small.data, self.small_array)
        np_test.assert_equal(reordered, desired)
        np_test.assert_equal(small.ids, [1,2,5,7])

        # reorder with clean
        small = Labels(data=self.small_array)
        small.setIds(ids=[1,2,5,7])
        small.reorder(order={2:5, 5:1, 7:6}, clean=True)
        desired = numpy.array(
            [[0, 0, 0, 5, 1],
             [0, 0, 0, 5, 1],
             [0, 0, 0, 0, 6]])
        np_test.assert_equal(small.data, desired)
        np_test.assert_equal(small.ids, [1,5,6])

    def testFindInset(self):
        """
        Tests findInset()
        """

        # no additional
        small = Labels(data=self.small_array)
        small.useInset(inset=[slice(1,3), slice(1,5)])
        ids=[2,7]
        abs_inset = small.findInset(ids=ids, mode='abs')
        np_test.assert_equal(abs_inset, [slice(1,3), slice(3,5)]) 
        rel_inset = small.findInset(ids=ids, mode='rel')
        np_test.assert_equal(rel_inset, [slice(0,2), slice(2,4)])

        # with additional, both first and additional the smallest possible
        small = Labels(data=self.small_array)
        ids = [7]
        small.makeInset(ids=ids)
        small_2 = Labels(data=self.small_array)
        additional_ids = [2]
        small_2.makeInset(ids=additional_ids)
        rel_inset = small.findInset(ids=ids, additional=small_2, 
                                    additionalIds=additional_ids)
        np_test.assert_equal(rel_inset, [slice(-2,1), slice(-1,1)])
        abs_inset = small.findInset(ids=ids, mode='abs', additional=small_2, 
                                    additionalIds=additional_ids)
        np_test.assert_equal(abs_inset, [slice(0,3), slice(3,5)])

        # with additional, not the smallest
        small = Labels(data=self.small_array)
        small.useInset(inset=[slice(1,3), slice(2,5)])
        ids = [7]
        small_2 = Labels(data=self.small_array)
        small_2.useInset(inset=[slice(0,2), slice(1,5)])
        additional_ids = [2]
        abs_inset = small.findInset(
            ids=ids, mode='abs', additional=small_2, 
            additionalIds=additional_ids)
        np_test.assert_equal(abs_inset, [slice(0,3), slice(3,5)])
        rel_inset = small.findInset(
            ids=ids, additional=small_2, mode='rel', 
            additionalIds=additional_ids)
        np_test.assert_equal(rel_inset, [slice(-1,2), slice(1,3)])

        # inset without data 
        small = Labels(data=self.small_array)
        small.useInset(inset=[slice(1,3), slice(1,5)])
        abs_inset = small.findInset(ids=8, mode='abs')
        np_test.assert_equal(abs_inset is None, True)
        
        # inset without data with additional 
        small = Labels(data=self.small_array)
        small.useInset(inset=[slice(1,3), slice(1,5)])
        self.labels_1.useInset(inset=[slice(0,4), slice(2,6)])
        abs_inset = small.findInset(
            ids=[8,9], mode='abs', additional=self.labels_1, additionalIds=[1])
        np_test.assert_equal(abs_inset, [slice(0,4), slice(2,6)])
        rel_inset = small.findInset(
            ids=[9], mode='rel', additional=self.labels_1, additionalIds=[1])
        np_test.assert_equal(rel_inset, [slice(-1,3), slice(1,5)])

        # extend, no initial inset, expand
        small = Labels(data=self.small_array)
        abs_inset = small.findInset(ids=[2,5], mode='abs')
        np_test.assert_equal(abs_inset, [slice(0,2), slice(3,5)]) 
        abs_inset = small.findInset(ids=[2,5], mode='abs', extend=1)
        np_test.assert_equal(abs_inset, [slice(0,3), slice(2,6)]) 
        abs_inset = small.findInset(ids=[2,5], mode='abs', extend=2)
        np_test.assert_equal(abs_inset, [slice(0,4), slice(1,7)]) 
        abs_inset = small.findInset(ids=[2,5], mode='abs', extend=4)
        np_test.assert_equal(abs_inset, [slice(0,6), slice(0,9)]) 
        rel_inset = small.findInset(ids=[2,5], mode='rel')
        np_test.assert_equal(rel_inset, [slice(0,2), slice(3,5)])
        abs_inset = small.findInset(ids=[2,5], mode='rel', extend=2)
        np_test.assert_equal(abs_inset, [slice(0,4), slice(1,7)]) 

        # extend, w initial inset, expand
        small = Labels(data=self.small_array)
        small.useInset(inset=[slice(1,3), slice(1,5)])
        abs_inset = small.findInset(ids=[2,5], mode='abs')
        np_test.assert_equal(abs_inset, [slice(1,2), slice(3,5)]) 
        abs_inset = small.findInset(ids=[2,5], mode='abs', extend=1)
        np_test.assert_equal(abs_inset, [slice(0,3), slice(2,6)]) 
        abs_inset = small.findInset(ids=[2,5], mode='abs', extend=2)
        np_test.assert_equal(abs_inset, [slice(0,4), slice(1,7)]) 
        abs_inset = small.findInset(ids=[2,5], mode='abs', extend=4)
        np_test.assert_equal(abs_inset, [slice(0,6), slice(0,9)]) 
        rel_inset = small.findInset(ids=[2,5], mode='rel')
        np_test.assert_equal(rel_inset, [slice(0,1), slice(2,4)])
        abs_inset = small.findInset(ids=[2,5], mode='rel', extend=2)
        np_test.assert_equal(abs_inset, [slice(-1,3), slice(0,6)]) 
        abs_inset = small.findInset(ids=[2,5], mode='rel', extend=3)
        np_test.assert_equal(abs_inset, [slice(-1,4), slice(-1,7)]) 

        # extend, expand
        small = Labels(data=self.small_array)
        small.useInset(inset=[slice(1,3), slice(1,5)], expand=False)
        abs_inset = small.findInset(ids=[2,5], mode='abs')
        np_test.assert_equal(abs_inset, [slice(1,2), slice(3,5)]) 
        abs_inset = small.findInset(
            ids=[2,5], mode='abs', extend=1, expand=False)
        np_test.assert_equal(abs_inset, [slice(1,3), slice(2,5)]) 
        abs_inset = small.findInset(
            ids=[2,5], mode='abs', extend=2, expand=False)
        np_test.assert_equal(abs_inset, [slice(1,3), slice(1,5)]) 

        # additional, extend, any expand
        small = Labels(data=self.small_array)
        small.useInset(inset=[slice(1,3), slice(2,5)])
        ids = [7]
        small_2 = Labels(data=self.small_array)
        small_2.useInset(inset=[slice(0,2), slice(1,5)])
        additional_ids = [2]
        abs_inset = small.findInset(
            ids=ids, mode='abs', additional=small_2, 
            additionalIds=additional_ids)
        np_test.assert_equal(abs_inset, [slice(0,3), slice(3,5)])
        abs_inset = small.findInset(
            ids=ids, mode='abs', additional=small_2, 
            additionalIds=additional_ids, expand=False)
        np_test.assert_equal(abs_inset, [slice(1,3), slice(3,5)])
        abs_inset = small.findInset(
            ids=ids, mode='abs', additional=small_2, 
            additionalIds=additional_ids, extend=1, expand=True)
        np_test.assert_equal(abs_inset, [slice(0,4), slice(2,6)])
        abs_inset = small.findInset(
            ids=ids, mode='abs', additional=small_2, 
            additionalIds=additional_ids, extend=1, expand=False)
        np_test.assert_equal(abs_inset, [slice(1,3), slice(2,5)])

        rel_inset = small.findInset(
            ids=ids, additional=small_2, mode='rel', 
            additionalIds=additional_ids)
        np_test.assert_equal(rel_inset, [slice(-1,2), slice(1,3)])
        rel_inset = small.findInset(
            ids=ids, mode='rel', additional=small_2, 
            additionalIds=additional_ids, expand=False)
        np_test.assert_equal(rel_inset, [slice(0,2), slice(1,3)])
        rel_inset = small.findInset(
            ids=ids, mode='rel', additional=small_2, 
            additionalIds=additional_ids, extend=1, expand=True)
        np_test.assert_equal(rel_inset, [slice(-1,3), slice(0,4)])
        rel_inset = small.findInset(
            ids=ids, mode='rel', additional=small_2, 
            additionalIds=additional_ids, extend=1, expand=False)
        np_test.assert_equal(rel_inset, [slice(0,2), slice(0,3)])

    def testMakeInset(self):
        """
        Tests makeInset and useInset
        """

        # all ids
        small = Labels(data=self.small_array)
        ids=[1,2,5,7]
        desired_inset = [slice(0,3), slice(1,5)]
        np_test.assert_equal(small.findInset(ids=ids), desired_inset)
        small.makeInset(ids=ids)
        np_test.assert_equal(small.inset, desired_inset)
        np_test.assert_equal(small.data, self.small_array[desired_inset])
        
        # some ids
        small = Labels(data=self.small_array)
        ids=[2,5]
        desired_inset = [slice(0,2), slice(3,5)]
        np_test.assert_equal(small.findInset(ids=ids), desired_inset)
        small.makeInset(ids=ids)
        np_test.assert_equal(small.inset, desired_inset)
        np_test.assert_equal(small.data, self.small_array[desired_inset])
        
        # some ids with extend
        small = Labels(data=self.small_array)
        ids=[2,5]
        desired_inset = [slice(0,3), slice(2,6)]
        np_test.assert_equal(small.findInset(ids=ids, extend=1), desired_inset)
        small.makeInset(ids=ids, extend=1)
        np_test.assert_equal(small.inset, desired_inset)
        np_test.assert_equal(
            small.data, 
            numpy.array([[1,2,5,0],
                         [0,2,5,0],
                         [0,0,7,0]]))
        
        # wrong ids
        small = Labels(data=self.small_array)
        ids = [3,6]
        desired_inset = [slice(0,0), slice(0,0)]
        np_test.assert_equal(small.findInset(ids=ids), None)
        small.makeInset(ids=ids)
        np_test.assert_equal(small.inset, desired_inset)
        np_test.assert_equal(small.data, self.small_array[desired_inset])
        
        # no ids
        small = Labels(data=self.small_array)
        ids = []
        desired_inset = [slice(0,0), slice(0,0)]
        np_test.assert_equal(small.findInset(ids=ids), None)
        small.makeInset(ids=ids)
        np_test.assert_equal(small.inset, desired_inset)
        np_test.assert_equal(small.data, self.small_array[desired_inset])
        
        # no data 
        small = Labels(data=numpy.zeros((0,0), dtype=int))
        ids = [1]
        desired_inset = [slice(0,0), slice(0,0)]
        np_test.assert_equal(small.findInset(ids=ids), None)
        small.makeInset(ids=ids)
        np_test.assert_equal(small.inset, desired_inset)
        np_test.assert_equal(small.data, self.small_array[desired_inset])
        
        # with additional
        small = Labels(data=self.small_array)
        ids = [2]
        small_2 = Labels(data=self.small_array)
        additional_ids = [1]
        desired_inset = [slice(0,3), slice(1,4)]
        actual_inset = small.findInset(ids=ids, additional=small_2, 
                                       additionalIds=additional_ids)
        np_test.assert_equal(actual_inset, desired_inset)
        small.makeInset(ids=ids, additional=small_2, 
                        additionalIds=additional_ids)
        np_test.assert_equal(small.inset, desired_inset)
        np_test.assert_equal(small.data, self.small_array[desired_inset])
        
        # with additional, update=False
        small = Labels(data=self.small_array)
        ids = [2]
        small_2 = Labels(data=self.small_array)
        additional_ids = [1]
        desired_inset = [slice(0,3), slice(1,4)]
        actual_inset = small.findInset(ids=ids, additional=small_2, 
                                       additionalIds=additional_ids)
        np_test.assert_equal(actual_inset, desired_inset)
        prev_inset = small.inset
        prev_data = small.data
        new_data = small.makeInset(ids=ids, additional=small_2, 
                                   additionalIds=additional_ids, update=False)
        np_test.assert_equal(new_data, self.small_array[desired_inset])
        np_test.assert_equal(small.inset, prev_inset)
        np_test.assert_equal(small.data, prev_data)
        
        # no ids with additional
        small = Labels(data=self.small_array)
        ids = []
        small_2 = Labels(data=self.small_array)
        additional_ids = [1]
        desired_inset = [slice(0,3), slice(1,3)]
        actual_inset = small.findInset(ids=ids, additional=small_2, 
                                       additionalIds=additional_ids)
        np_test.assert_equal(actual_inset, desired_inset)
        small.makeInset(ids=ids, additional=small_2, 
                        additionalIds=additional_ids)
        np_test.assert_equal(small.inset, desired_inset)
        np_test.assert_equal(small.data, self.small_array[desired_inset])

        # ids with wrong additional
        small = Labels(data=self.small_array)
        ids = [2]
        small_2 = Labels(data=self.small_array)
        additional_ids = [3]
        desired_inset = [slice(0,2), slice(3,4)]
        actual_inset = small.findInset(ids=ids, additional=small_2, 
                                       additionalIds=additional_ids)
        np_test.assert_equal(actual_inset, desired_inset)
        small.makeInset(ids=ids, additional=small_2, 
                        additionalIds=additional_ids)
        np_test.assert_equal(small.inset, desired_inset)
        np_test.assert_equal(small.data, self.small_array[desired_inset])
                
    def testGetPointsAll(self):
        """
        Tests getPoints() mode 'all'
        """

        # make data
        data = numpy.array(
            [[1, 1, 1, 1, 1, 1, 1],
             [2, 2, 2, 2, 2, 2, 2],
             [3, 3, 3, 3, 3, 3, 3],
             [4, 4, 4, 4, 4, 4, 4]])

        # mode all, no inset
        labels = Labels(data=data)
        desired = (numpy.array([1, 1, 1, 1, 1, 1, 1]),
                   numpy.array([0, 1, 2, 3, 4, 5, 6]))
        np_test.assert_equal(labels.getPoints(ids=2, mode='all', 
                                              format_='numpy'), desired)
        desired = [[1, 0],
                   [1, 1],
                   [1, 2],
                   [1, 3],
                   [1, 4],
                   [1, 5],
                   [1, 6]]
        np_test.assert_equal(
            labels.getPoints(ids=[2], mode='all', format_='coordinates'), 
            desired)

        # mode all, inset
        labels = Labels(data=data[1:4, 2:6])
        labels.setInset([slice(1, 4), slice(2, 6)])
        desired = (numpy.array([1, 1, 1, 1]),
                   numpy.array([2, 3, 4, 5]))
        np_test.assert_equal(labels.getPoints(ids=[2], mode='all', 
                                              format_='numpy'), desired)

        # mode all, multi ids, inset
        data_diag = numpy.array(
            [[2, 3, 4, 0, 0],
             [1, 2, 3, 4, 0],
             [0, 1, 2, 3, 4],
             [0, 0, 1, 2, 3]])
        labels = Labels(data=data_diag)
        desired = (numpy.array([0, 0, 1, 1, 2, 2, 3, 3]), 
                   numpy.array([0, 1, 1, 2, 2, 3, 3, 4]))
        np_test.assert_equal(
            labels.getPoints(ids=[2, 3], mode='all', format_='numpy'), desired)
        labels = Labels(data=data_diag[1:4, 2:5])
        labels.setInset([slice(1, 4), slice(2, 5)])
        desired = (numpy.array([1, 2, 2, 3, 3]), numpy.array([2, 2, 3, 3, 4]))
        np_test.assert_equal(
            labels.getPoints(ids=[2, 3], mode='all', format_='numpy'), desired)

        # mode all, inset + additional inset
        labels = Labels(data=data_diag[1:4, 1:5])
        labels.setInset([slice(1, 4), slice(1, 5)])
        desired = (numpy.array([1, 2]), numpy.array([3, 4]))
        np_test.assert_equal(
            labels.getPoints(ids=4, mode='all', format_='numpy'), desired)

    def testGetPointsGeodesic(self):
        """
        Tests getPoints(mode='geodesic') and getPointsGeodesic().

        Run several times because the methods tested depend on a random 
        variable.

        """

        for i in range(10):
            self.basicGetPointsGeodesic()

    def basicGetPointsGeodesic(self):
        """
        Single test for getPoints(mode='geodesic') and getPointsGeodesic().
        """

        # more geodesic, distance=2 (complicated because random)
        data = numpy.array([[0, 1, 1, 1, 2, 2, 2, 0],
                            [0, 1, 1, 1, 2, 2, 2, 0],
                            [0, 1, 1, 1, 2, 2, 2, 0]])
        labels = Labels(data=data)
        result = labels.getPoints(ids=[1], mode='geodesic', distance=2, 
                                  connectivity=1)
        result = result.tolist()
        if len(result) == 5:
            desired = [[0, 1], [0, 3], [1, 2], [2, 1], [2, 3]]
        elif len(result) == 4:
            desired = [[0, 2], [1, 1], [1, 3], [2, 2]]
        elif len(result) == 3:
            if [1, 2] in result:
                if [0, 1] in result:
                    desired = [[0, 1], [1, 2], [2, 3]] 
                elif [0, 3] in result:
                    desired = [[0, 3], [1, 2], [2, 1]]
            elif [0, 1] in result:
                if [0, 3] in result:
                    desired = [[0, 1], [0, 3], [2, 2]]
                elif [2, 1] in result:
                    desired = [[0, 1], [2, 1], [1, 3]]
                else:
                    desired = [[0, 1], [1, 3], [2, 2]]
            elif [2, 3] in result:
                if [0, 3] in result:
                    desired = [[0, 3], [1, 1], [2, 3]]
                elif [2, 1] in result:
                    desired = [[0, 2], [2, 1], [2, 3]]
                else:
                    desired = [[2, 3], [1, 1], [0, 2]] 
            elif [0, 3] in result:
                desired = [[0, 3], [1, 1], [2, 2]]
            elif [2, 1] in result:
                desired = [[2, 1], [1, 3], [0, 2]]
        for des in desired:
            np_test.assert_equal(des in result, True)
        for res in result:
            np_test.assert_equal(res in desired, True)

        # mode geodesic, distance=3, inset
        labels = Labels(data=data[1:3, 2:8])
        labels.setInset([slice(1, 3), slice(2, 8)])
        result = labels.getPoints(ids=[2], mode='geodesic', distance=3, 
                                  connectivity=1)
        result = result.tolist()
        if len(result) == 1:
            np_test.assert_equal(result[0][1], 5)
        elif len(result) == 2:
            desired = []
            if [1, 4] in result:
                desired = [[1, 4], [2, 6]]
            elif [2, 4] in result:
                desired = [[2, 4], [1, 6]]
            for des in desired: 
                np_test.assert_equal(des in result, True)
            for res in result:
                np_test.assert_equal(res in desired, True)

    def testMagnify(self):
        """
        Tests magnify()
        """

        small = deepcopy(self.small)
        small.magnify(factor=3) 
        np_test.assert_equal(small.data.shape, (9, 15))
        np_test.assert_equal(small.data.dtype, self.small.data.dtype)
        np_test.assert_equal(
            small.data[2:7, 8:14], 
            numpy.array(
                [[1,2,2,2,5,5], 
                 [0,2,2,2,5,5],
                 [0,2,2,2,5,5],
                 [0,2,2,2,5,5],
                 [0,0,0,0,7,7]])
            )


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLabels)
    unittest.TextTestRunner(verbosity=2).run(suite)
