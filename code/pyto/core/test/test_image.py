"""

Tests module image

# Author: Vladan Lucic
# $Id$
"""
from __future__ import unicode_literals

__version__ = "$Revision$"

from copy import copy, deepcopy
import os.path
import unittest
#from numpy.testing import *

import numpy
import numpy.testing as np_test 
import scipy

from pyto.core.image import Image
from pyto.io.image_io import ImageIO

class TestImage(np_test.TestCase):
    """
    """

    def setUp(self):
        
        # make image
        array = numpy.arange(100).reshape(10,10)        
        self.image = Image(data=array)

        # set attributes
        self.image.xxx = [1,2,3]
        self.image.yyy = [4,5,6]

        large_array = array[1:9, 1:8].copy()
        self.large = Image(data=large_array)

        # set absolute path to current dir
        working_dir = os.getcwd()
        file_dir, name = os.path.split(__file__)
        self.dir = os.path.join(working_dir, file_dir)

        # image file names
        self.big_file_name = os.path.join(
            self.dir, '../../io/test/big_head.mrc')
        self.small_file_name = os.path.join(
            self.dir, '../../io/test/small.mrc')
        self.modified_file_name_mrc = os.path.join(
            self.dir, '../../io/test/modified.mrc')
        self.modified_file_name_raw = os.path.join(
            self.dir, '../../io/test/modified.raw')

    def testRelativeToAbsoluteInset(self):
        """
        Tests relativeToAbsoluteInset()
        """
        
        image = deepcopy(self.image)
        image_inset = [slice(2,5), slice(4,6)]
        image.useInset(inset=image_inset, mode='abs')

        # intersect
        inset = [slice(1,5), slice(-3,-2)]
        res = image.relativeToAbsoluteInset(inset=inset)
        np_test.assert_equal(res, [slice(3,7), slice(1,2)])

        # intersect below 0
        inset = [slice(-4,-2), slice(3,5)]
        res = image.relativeToAbsoluteInset(inset=inset)
        np_test.assert_equal(res, [slice(-2,0), slice(7,9)])

    def testAbsoluteToRelativeInset(self):
        """
        Tests absoluteToRelativeInset()
        """
        
        image = deepcopy(self.image)
        image_inset = [slice(2,5), slice(4,6)]
        image.useInset(inset=image_inset, mode='abs')

        # intersect
        inset = [slice(3,7), slice(1,2)]
        res = image.absoluteToRelativeInset(inset=inset)
        np_test.assert_equal(res, [slice(1,5), slice(-3,-2)])

        # intersect below 0
        inset = [slice(-2,0), slice(7,9)]
        res = image.absoluteToRelativeInset(inset=inset)
        np_test.assert_equal(res, [slice(-4,-2), slice(3,5)])

        # intersect below 0
        inset = [slice(-2,3), slice(7,9)]
        res = image.absoluteToRelativeInset(inset=inset)
        np_test.assert_equal(res, [slice(-4,1), slice(3,5)])

    def testUseInset(self):
        """
        """

        # absolute inset
        image = deepcopy(self.image)
        inset = [slice(2,5), slice(4,6)]
        image.useInset(inset=inset, mode='abs')
        desired_array = numpy.array([[24, 25],
                               [34, 35],
                               [44, 45]])
        np_test.assert_equal(image.data, desired_array)

        # absolute inset no update
        image = deepcopy(self.image)
        inset = [slice(2,5), slice(4,6)]
        new_data = image.useInset(inset=inset, mode='abs', update=False)
        desired_array = numpy.array([[24, 25],
                               [34, 35],
                               [44, 45]])
        np_test.assert_equal(new_data, desired_array)
        np_test.assert_equal(image.data, self.image.data)

        # absolute inset from an inset
        large = deepcopy(self.image)
        inset = [slice(2,5), slice(4,6)]
        large.useInset(inset=inset, mode='abs')
        desired_array = numpy.array([[24, 25],
                               [34, 35],
                               [44, 45]])
        np_test.assert_equal(large.data, desired_array)

        # relative inset
        large = deepcopy(self.large)
        inset = [slice(2,5), slice(4,6)]
        large.useInset(inset=inset, mode='rel')
        desired_array = numpy.array([[35, 36],
                               [45, 46],
                               [55, 56]])
        np_test.assert_equal(large.data, desired_array)

        # use full
        full_inset = inset=[slice(0,10), slice(0,10)]
        image = deepcopy(self.image)
        inset = [slice(2,5), slice(4,6)]
        image.saveFull()
        image.useInset(inset=inset, mode='abs')
        image.data[0,0] = 100
        image.useInset(inset=full_inset, mode='abs', useFull=True)
        np_test.assert_equal(image.data[2,4], 100)
        np_test.assert_equal(image.data[9,9], 99)

        # do not allow to use full
        full_inset = inset=[slice(0,10), slice(0,10)]
        image = deepcopy(self.image)
        inset = [slice(2,5), slice(4,6)]
        image.saveFull()
        image.useInset(inset=inset, mode='abs')
        kwargs = {'inset':full_inset, 'mode':'abs', 'useFull':False}
        self.assertRaises(ValueError, image.useInset, **kwargs)

        # expand
        full_inset = inset=[slice(0,10), slice(0,10)]
        image = deepcopy(self.image)
        inset = [slice(2,5), slice(4,6)]
        image.useInset(inset=inset, mode='abs')
        image.data[0,0] = 100
        image.useInset(inset=full_inset, mode='abs', expand=True)
        np_test.assert_equal(image.data[2,4], 100)
        np_test.assert_equal(image.data[9,9], 0)

        # expand, no update
        full_inset = [slice(0,10), slice(0,10)]
        med_inset = [slice(1,5), slice(4,7)]
        inset = [slice(2,5), slice(4,6)]
        image = deepcopy(self.image)
        image.useInset(inset=inset, mode='abs')
        image.data[0,0] = 100
        new_data = image.useInset(inset=med_inset, mode='abs', 
                                  expand=True, update=False)
        np_test.assert_equal(new_data[2,1], 35)
        np_test.assert_equal(new_data[0,2], 0)
        np_test.assert_equal(image.inset, inset)
        new_data = image.useInset(inset=full_inset, mode='abs', 
                                  expand=True, update=False)
        np_test.assert_equal(new_data[2,4], 100)
        np_test.assert_equal(new_data[9,9], 0)
        np_test.assert_equal(image.data[0,0], 100)
        np_test.assert_equal(image.data[1:,:], self.image.data[tuple(inset)][1:,:])
        np_test.assert_equal(image.inset, inset)

        # use full, expand, update  
        image = deepcopy(self.image)
        image.useInset(inset=inset, mode='abs')
        image.useInset(inset=med_inset, mode='abs', 
                       useFull=True, expand=True, update=True)
        np_test.assert_equal(image.data[2,1], 35)
        np_test.assert_equal(image.data[0,2], 16)
        np_test.assert_equal(image.inset, med_inset)

        # use full, expand, no update  
        image = deepcopy(self.image)
        image.useInset(inset=inset, mode='abs')
        new_data = image.useInset(inset=med_inset, mode='abs', 
                                  useFull=True, expand=True, update=False)
        np_test.assert_equal(new_data[2,1], 35)
        np_test.assert_equal(new_data[0,2], 16)
        np_test.assert_equal(image.inset, inset)

        # use full, expand, update, no overlap  
        image = deepcopy(self.image)
        inset = [slice(2,5), slice(4,6)]
        inset2 = [slice(4,6), slice(6,9)]
        image.useInset(inset=inset, mode='abs')
        image.useInset(inset=inset2, mode='abs', 
                       useFull=True, expand=True, update=True)
        np_test.assert_equal(image.inset, inset2)
        desired = numpy.array(
            [[46, 47, 48],
             [56, 57, 58]])
        np_test.assert_equal(image.data, desired)

        # no use full, expand, update, no overlap  
        image = deepcopy(self.image)
        inset = [slice(2,5), slice(4,6)]
        inset2 = [slice(4,6), slice(6,9)]
        image.useInset(inset=inset, mode='abs')
        image.useInset(inset=inset2, mode='abs', 
                       useFull=False, expand=True, update=True)
        np_test.assert_equal(image.inset, inset2)
        np_test.assert_equal(image.data, numpy.zeros((2,3)))

        # no use full, expand, update, inset 0
        image = deepcopy(self.image)
        inset = [slice(2,5), slice(4,6)]
        inset2 = [slice(0,0), slice(0,0)]
        image.useInset(inset=inset, mode='abs')
        image.useInset(inset=inset2, mode='abs', 
                       useFull=False, expand=True, update=True)
        np_test.assert_equal(image.inset, inset2)
        np_test.assert_equal(image.data, numpy.zeros((0,0)))

    def testExpandInset(self):
        """
        Tests expandInset()
        """

        # expand, update
        image = deepcopy(self.image)
        inset = [slice(2,5), slice(4,6)]
        image.useInset(inset=inset, mode='abs')
        new_inset = [slice(1,6), slice(2,6)]
        image.expandInset(inset=new_inset, update=True)
        np_test.assert_equal(image.inset, new_inset)
        new_data = numpy.array(
            [[0, 0, 0, 0],
             [0, 0, 24, 25],
             [0, 0, 34, 35],
             [0, 0, 44, 45],
             [0, 0, 0, 0]])
        np_test.assert_equal(image.data, new_data)
 
        # expand, no update
        image = deepcopy(self.image)
        inset = [slice(2,5), slice(4,6)]
        image.useInset(inset=inset, mode='abs')
        new_inset = [slice(1,6), slice(2,6)]
        new_data = image.expandInset(inset=new_inset, update=False)
        np_test.assert_equal(image.inset, inset)
        desired_data = numpy.array(
            [[24, 25],
             [34, 35],
             [44, 45]])
        np_test.assert_equal(image.data, desired_data)
        desired_data = numpy.array(
            [[0, 0, 0, 0],
             [0, 0, 24, 25],
             [0, 0, 34, 35],
             [0, 0, 44, 45],
             [0, 0, 0, 0]])
        np_test.assert_equal(new_data, desired_data)

        # partial overlap
        image = deepcopy(self.image)
        inset = [slice(2,5), slice(4,6)]
        image.useInset(inset=inset, mode='abs')
        new_inset = [slice(3,6), slice(2,5)]
        new_data = image.expandInset(inset=new_inset, update=True, value=9)
        np_test.assert_equal(image.inset, new_inset)
        desired_data = numpy.array(
            [[9, 9, 34],
             [9, 9, 44],
             [9, 9, 9]])
        np_test.assert_equal(image.data, desired_data)

        # completely inside
        image = deepcopy(self.image)
        inset = [slice(2,5), slice(4,6)]
        image.useInset(inset=inset, mode='abs')
        new_inset = [slice(3,5), slice(4,5)]
        new_data = image.expandInset(inset=new_inset, update=True)
        np_test.assert_equal(image.inset, new_inset)
        desired_data = numpy.array(
            [[34],
             [44]])
        np_test.assert_equal(image.data, desired_data)

        # completely inside
        image = deepcopy(self.image)
        inset = [slice(2,5), slice(3,6)]
        image.useInset(inset=inset, mode='abs')
        new_inset = [slice(2,3), slice(4,6)]
        new_data = image.expandInset(inset=new_inset, update=True)
        np_test.assert_equal(image.inset, new_inset)
        desired_data = numpy.array(
            [[24, 25]])
        np_test.assert_equal(image.data, desired_data)

        # completely outside
        image = deepcopy(self.image)
        inset = [slice(2,5), slice(3,6)]
        image.useInset(inset=inset, mode='abs')
        new_inset = [slice(5,7), slice(7,10)]
        new_data = image.expandInset(inset=new_inset, update=True)
        np_test.assert_equal(image.inset, new_inset)
        desired_data = numpy.zeros((2,3))
        np_test.assert_equal(image.data, desired_data)

        # 0
        image = deepcopy(self.image)
        inset = [slice(2,5), slice(3,6)]
        image.useInset(inset=inset, mode='abs')
        new_inset = [slice(0,0), slice(0,0)]
        new_data = image.expandInset(inset=new_inset, update=True)
        np_test.assert_equal(image.inset, new_inset)
        desired_data = numpy.zeros((0,0))
        np_test.assert_equal(image.data, desired_data)

    def testIsInside(self):
        """
        Tests isInside()
        """
        
        # inset inside self.inset
        image = deepcopy(self.image)
        inset = [slice(2,5), slice(3,7)]
        inset2 = [slice(3,5), slice(3,4)]
        image.useInset(inset=inset, mode='abs')
        res = image.isInside(inset=inset2)
        np_test.assert_equal(res, True)
    
        # self.inset inside inset
        image = deepcopy(self.image)
        inset = [slice(2,5), slice(3,7)]
        inset2 = [slice(3,5), slice(3,4)]
        image.useInset(inset=inset2, mode='abs')
        res = image.isInside(inset=inset)
        np_test.assert_equal(res, False)
        
        # overlap
        inset = [slice(2,5), slice(3,7)]
        inset2 = [slice(3,8), slice(2,4)]
        image = deepcopy(self.image)
        image.useInset(inset=inset2, mode='abs')
        res = image.isInside(inset=inset)
        np_test.assert_equal(res, False)

        # overlap
        inset = [slice(2,5), slice(3,7)]
        inset2 = [slice(1,6), slice(4,7)]
        image = deepcopy(self.image)
        image.useInset(inset=inset2, mode='abs')
        res = image.isInside(inset=inset)
        np_test.assert_equal(res, False)

        # no overlap
        inset = [slice(2,5), slice(3,7)]
        inset2 = [slice(5,6), slice(6,10)]
        image = deepcopy(self.image)
        image.useInset(inset=inset2, mode='abs')
        res = image.isInside(inset=inset)
        np_test.assert_equal(res, False)
        
    def testHasOvelap(self):
        """
        Tests hasOverlap()
        """
        
        # inset inside self.inset
        image = deepcopy(self.image)
        inset = [slice(2,5), slice(3,7)]
        inset2 = [slice(3,5), slice(3,4)]
        image.useInset(inset=inset, mode='abs')
        res = image.hasOverlap(inset=inset2)
        np_test.assert_equal(res, True)
    
        # self.inset inside inset
        image = deepcopy(self.image)
        inset = [slice(2,5), slice(3,7)]
        inset2 = [slice(3,5), slice(3,4)]
        image.useInset(inset=inset2, mode='abs')
        res = image.hasOverlap(inset=inset)
        np_test.assert_equal(res, True)
        
        # overlap
        inset = [slice(2,5), slice(3,7)]
        inset2 = [slice(4,8), slice(2,4)]
        image = deepcopy(self.image)
        image.useInset(inset=inset2, mode='abs')
        res = image.hasOverlap(inset=inset)
        np_test.assert_equal(res, True)

        # overlap
        inset = [slice(2,5), slice(3,7)]
        inset2 = [slice(1,6), slice(4,7)]
        image = deepcopy(self.image)
        image.useInset(inset=inset, mode='abs')
        res = image.hasOverlap(inset=inset2)
        np_test.assert_equal(res, True)

        # no overlap
        inset = [slice(2,5), slice(3,7)]
        inset2 = [slice(5,6), slice(5,10)]
        image = deepcopy(self.image)
        image.useInset(inset=inset, mode='abs')
        res = image.hasOverlap(inset=inset2)
        np_test.assert_equal(res, False)
        
        # no overlap
        inset = [slice(2,5), slice(3,7)]
        inset2 = [slice(5,6), slice(8,10)]
        image = deepcopy(self.image)
        image.useInset(inset=inset, mode='abs')
        res = image.hasOverlap(inset=inset2)
        np_test.assert_equal(res, False)
        
        # no overlap
        inset = [slice(2,5), slice(3,7)]
        inset2 = [slice(1,5), slice(7,10)]
        image = deepcopy(self.image)
        image.useInset(inset=inset, mode='abs')
        res = image.hasOverlap(inset=inset2)
        np_test.assert_equal(res, False)

    def testFindEnclosingInset(self):
        """
        Tests findEnclosingInset()
        """
        
        # inset2
        inset = self.image.findEnclosingInset(
            inset=[slice(2, 5), slice(1, 3)], inset2=[slice(1, 4), slice(3, 6)])
        np_test.assert_equal(inset, [slice(1, 5), slice(1, 6)])
        
        # self.inset
        image = deepcopy(self.image)
        image.useInset(inset=[slice(4, 6), slice(0, 4)], mode='abs')
        inset = image.findEnclosingInset(inset=[slice(2, 5), slice(1, 3)])
        np_test.assert_equal(inset, [slice(2, 6), slice(0, 4)])

        # inset2 inside inset
        inset = [slice(2,5), slice(3,7)]
        inset2 = [slice(3,5), slice(3,4)]
        res = self.image.findEnclosingInset(inset=inset, inset2=inset2)
        np_test.assert_equal(res, inset)

        # inset inside inset2
        inset = [slice(2,5), slice(3,7)]
        inset2 = [slice(3,5), slice(3,4)]
        res = self.image.findEnclosingInset(inset=inset2, inset2=inset)
        np_test.assert_equal(res, inset)

        # overlap
        inset = [slice(2,5), slice(3,7)]
        inset2 = [slice(3,8), slice(2,4)]
        res = self.image.findEnclosingInset(inset=inset, inset2=inset2)
        np_test.assert_equal(res, [slice(2,8), slice(2,7)])

        # overlap
        inset = [slice(2,5), slice(3,7)]
        inset2 = [slice(1,6), slice(4,7)]
        res = self.image.findEnclosingInset(inset=inset, inset2=inset2)
        np_test.assert_equal(res, [slice(1,6), slice(3,7)])
       
        # no overlap
        inset = [slice(2,5), slice(3,7)]
        inset2 = [slice(5,6), slice(8,10)]
        res = self.image.findEnclosingInset(inset=inset, inset2=inset2)
        np_test.assert_equal(res, [slice(2,6), slice(3,10)])
       
    def testFindIntersectingInset(self):
        """
        Tests findIntersectingInset()
        """
        
        # inset2
        inset = self.image.findIntersectingInset(
            inset=[slice(2, 5), slice(1, 3)], inset2=[slice(1, 4), slice(3, 6)])
        np_test.assert_equal(inset, [slice(2, 4), slice(3, 3)])
        
        # self.inset
        image = deepcopy(self.image)
        image.useInset(inset=[slice(4, 6), slice(0, 4)], mode='abs')
        inset = image.findIntersectingInset(
            inset=[slice(2, 5), slice(1, 3)])
        np_test.assert_equal(inset, [slice(4, 5), slice(1, 3)])

        # no overlap
        inset = [slice(2,5), slice(3,7)]
        inset2 = [slice(5,6), slice(8,10)]
        res = self.image.findIntersectingInset(inset=inset, inset2=inset2)
        np_test.assert_equal(res, [slice(5,5), slice(8,7)])
       
    def testNewFromInset(self):
        """
        Tests if copy/deepcopy of attrbutes works properly
        """

        # tests default args
        inset = [slice(2,5), slice(4,6)]
        new = self.image.newFromInset(inset=inset, copyData=True, deepcp=True)
        new.xxx[1] = 12
        new.yyy[1] = 15
        desired_array = numpy.array([[24, 25],
                               [34, 35],
                               [44, 45]])
        np_test.assert_equal(new.data, desired_array)
        np_test.assert_equal(new.inset, inset)
        np_test.assert_equal(self.image.xxx, [1,2,3])
        np_test.assert_equal(self.image.yyy, [4, 5, 6])
        inset = [slice(2,5), slice(4,6)]

        # tests if copy/deepcopy of attrbutes works properly
        new = self.image.newFromInset(
            inset=inset, copyData=True, deepcp=True, noDeepcp=['yyy'])
        new.xxx[1] = 12
        new.yyy[1] = 15
        desired_array = numpy.array([[24, 25],
                               [34, 35],
                               [44, 45]])
        np_test.assert_equal(new.data, desired_array)
        np_test.assert_equal(new.inset, inset)
        np_test.assert_equal(self.image.xxx, [1,2,3])
        np_test.assert_equal(self.image.yyy, [4, 15, 6])

    def testRead(self):
        """
        Tests read()
        """

        # 
        mrc = Image.read(
            file=os.path.normpath(
                os.path.join(self.dir, '../../io/test/new-head_int16.mrc')))
        np_test.assert_equal(mrc.pixelsize, 0.4)
        np_test.assert_equal(mrc.fileFormat, 'mrc')
        np_test.assert_equal(mrc.data[14,8,10], -14)
        np_test.assert_equal(mrc.memmap, False)

        # mrc with header
        mrc = Image.read(
            file=os.path.normpath(
                os.path.join(self.dir, '../../io/test/new-head_int16.mrc')),
            header=True)
        np_test.assert_equal(mrc.header is None, False)
        np_test.assert_equal(mrc.pixelsize, 0.4)
        np_test.assert_equal(mrc.fileFormat, 'mrc')
        np_test.assert_equal(mrc.data[14,8,10], -14)
        np_test.assert_equal(mrc.memmap, False)

        # em with header
        em = Image.read(
            file=os.path.normpath(
                os.path.join(self.dir, '../../io/test/mac-file.em')),
            header=True)
        np_test.assert_equal(em.header is None, False)

        # with memmap
        mrc = Image.read(
            file=os.path.normpath(
                os.path.join(self.dir, '../../io/test/new-head_int16.mrc')),
            memmap=True)
        np_test.assert_equal(mrc.pixelsize, 0.4)
        np_test.assert_equal(mrc.fileFormat, 'mrc')
        np_test.assert_equal(mrc.data[14,8,10], -14)
        np_test.assert_equal(mrc.memmap, True)

    def test_modify(self):
        """
        Tests modify(), implicitely tests reading and writting
        mrc header by pyto.io.ImageIO
        """

        # modify mrc image
        def fun_1(image):
            dat =  image.data + 1
            return dat
        # requires Image.modify(memmap=False) 
        #def fun_1(image): return image.data + 1
        Image.modify(
            old=self.big_file_name, new=self.modified_file_name_mrc,
            fun=fun_1, memmap=True)
        new = Image.read(
            file=self.modified_file_name_mrc, header=True, memmap=True)
        old = Image.read(file=self.big_file_name, header=True, memmap=True)

        # check data
        np_test.assert_equal(
            new.data[1, 10, :], numpy.arange(11001, 11101))
        np_test.assert_equal(
            new.data[2, :, 15], numpy.arange(20016, 30016, 100))
        
        # check header
        np_test.assert_almost_equal(new.pixelsize, old.pixelsize)
        np_test.assert_almost_equal(new.header[0:19], old.header[0:19])
        np_test.assert_almost_equal(new.header[22:25], old.header[22:25])
        np_test.assert_equal(True, new.header[25] == old.header[25])
        header_len = len(new.header)
        np_test.assert_almost_equal(
            new.header[26:header_len-1], old.header[26:header_len-1])
        np_test.assert_equal(
            True, new.header[header_len-1] == old.header[header_len-1])
        
        # modify mrc image and write as raw
        def fun_v(image, value):
            data = image.data + value
            return data
        modified = Image.modify(
            old=self.big_file_name, new=self.modified_file_name_raw,
            fun=fun_v, fun_kwargs={'value' : 4})
        new = Image.read(
            file=self.modified_file_name_raw, shape=modified.data.shape,
            dataType=modified.data.dtype, memmap=True)
        old = Image.read(file=self.big_file_name, header=True, memmap=True)

        # check data
        np_test.assert_equal(
            new.data[1, 10, :], numpy.arange(11004, 11104))
        np_test.assert_equal(
            new.data[2, :, 15], numpy.arange(20019, 30019, 100))
        
    def test_cut(self):
        """
        Tests cut(), implicitely tests reading and writting 
        mrc header by pyto.io.ImageIO
        """

        # cut image
        inset = [slice(1, 4), slice(10, 30), slice(50, 60)]
        Image.cut(
            old=self.big_file_name, new=self.small_file_name, inset=inset)

        # check data
        new = Image.read(file=self.small_file_name, header=True, memmap=True)
        np_test.assert_equal(
            new.data[1, 10, :], numpy.arange(22050, 22060))
        np_test.assert_equal(
            new.data[2, 6:16, 8], numpy.arange(31658, 32658, 100))

        # check header
        old = Image.read(file=self.big_file_name, header=True, memmap=True)
        np_test.assert_almost_equal(new.pixelsize, old.pixelsize)
        np_test.assert_equal(len(new.header), len(old.header))
        np_test.assert_equal(new.header[0:3], [3, 20, 10])
        np_test.assert_equal(new.header[7:10], [3, 20, 10])
        np_test.assert_almost_equal(
            new.header[10:13], numpy.array([3, 20, 10]) * old.pixelsize * 10,
            decimal=5)
        np_test.assert_equal(new.header[3:7], old.header[3:7])
        np_test.assert_almost_equal(new.header[13:19], old.header[13:19])
        np_test.assert_almost_equal(new.header[22:25], old.header[22:25])
        np_test.assert_equal(True, new.header[25] == old.header[25])
        #np_test.assert_string_equal(new.header[25], old.header[25])
        header_len = len(new.header)
        np_test.assert_almost_equal(
            new.header[26:header_len-1], old.header[26:header_len-1])
        np_test.assert_equal(
            True, new.header[header_len-1] == old.header[header_len-1])

    def tearDown(self):
        """
        Remove temporary files
        """
        try:
            os.remove(self.small_file_name)
        except OSError:
            pass
        try:
            os.remove(self.modified_file_name_mrc)
        except OSError:
            pass
        try:
            os.remove(self.modified_file_name_raw)
        except OSError:
            pass


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestImage)
    unittest.TextTestRunner(verbosity=2).run(suite)
