"""

Tests module image_io
 
# Author: Vladan Lucic
# $Id: test_pickled.py 983 2013-09-13 16:44:04Z vladan $
"""

__version__ = "$Revision: 983 $"

from copy import copy, deepcopy
import pickle
import os.path
import unittest

import numpy
import numpy.testing as np_test 
import scipy

from pyto.io.image_io import ImageIO


class TestImageIO(np_test.TestCase):
    """
    Tests class ImageIO
    """

    def setUp(self):
        """
        Sets absolute path to this file directory and saves it as self.dir
        """

        # set absolute path to current dir
        working_dir = os.getcwd()
        file_dir, name = os.path.split(__file__)
        self.dir = os.path.join(working_dir, file_dir)

        # make raw file
        self.raw_shape = (4,3,2)
        self.raw_dtype = 'int16'
        self.raw_data = numpy.arange(
            24, dtype=self.raw_dtype).reshape(self.raw_shape)
        raw = ImageIO()
        self.raw_file_name = 'data.raw'
        raw.write(file=self.raw_file_name, data=self.raw_data)

    def testRead(self):
        """
        Tests reading EM and MRC files
        """
        
        # EM tomo 
        em = ImageIO()
        em.read(file=os.path.join(self.dir, "bin-2.em"))
        expected = numpy.array([[-0.0242,   -0.0250,    0.0883],
                                [0.0640,    0.0071,   -0.1300],
                                [-0.0421,   -0.0392,   -0.0312]])
        np_test.assert_almost_equal(em.data[50:53, 120:123, 40], expected,
                                    decimal=4)
        expected = numpy.array([[-0.0573,    0.0569,    0.0386],
                                [0.1309,    0.1211,   -0.0881],
                                [-0.0110,   -0.0240,    0.0347]])
        np_test.assert_almost_equal(em.data[150:153, 20:23, 10], expected,
                                    decimal=4)
        np_test.assert_equal(em.byteOrder, '<')
        np_test.assert_equal(em.arrayOrder, 'FORTRAN')
        np_test.assert_equal(em.dataType, 'float32')
        np_test.assert_equal(em.data.dtype, numpy.dtype('float32'))
        np_test.assert_equal(em.memmap, False)

        # EM tomo with memory map
        em.read(file=os.path.join(self.dir, "bin-2.em"), memmap=True)
        expected = numpy.array([[-0.0242,   -0.0250,    0.0883],
                                [0.0640,    0.0071,   -0.1300],
                                [-0.0421,   -0.0392,   -0.0312]])
        np_test.assert_almost_equal(em.data[50:53, 120:123, 40], expected,
                                    decimal=4)
        expected = numpy.array([[-0.0573,    0.0569,    0.0386],
                                [0.1309,    0.1211,   -0.0881],
                                [-0.0110,   -0.0240,    0.0347]])
        np_test.assert_almost_equal(em.data[150:153, 20:23, 10], expected,
                                    decimal=4)
        np_test.assert_equal(em.byteOrder, '<')
        np_test.assert_equal(em.arrayOrder, 'FORTRAN')
        np_test.assert_equal(em.dataType, 'float32')
        np_test.assert_equal(em.data.dtype, numpy.dtype('float32'))
        np_test.assert_equal(em.memmap, True)

        # EM, big-endian
        em = ImageIO()
        em.read(file=os.path.join(self.dir, "mac-file.em"))
        np_test.assert_equal(em.byteOrder, '>')

        # EM, little-endian
        em = ImageIO()
        em.read(file=os.path.join(self.dir, "pc-file.em"))
        np_test.assert_equal(em.byteOrder, '<')
        em.read(file=os.path.join(self.dir, "pc-file.em"), memmap=True)
        np_test.assert_equal(em.byteOrder, '<')

        # MRC tomo
        mrc = ImageIO()
        mrc.read(file=os.path.join(self.dir, "bin-2.mrc"))
        expected = numpy.array([[-0.0242,   -0.0250,    0.0883],
                                [0.0640,    0.0071,   -0.1300],
                                [-0.0421,   -0.0392,   -0.0312]])
        np_test.assert_almost_equal(mrc.data[50:53, 120:123, 40], expected,
                                    decimal=4)
        expected = numpy.array([[-0.0573,    0.0569,    0.0386],
                                [0.1309,    0.1211,   -0.0881],
                                [-0.0110,   -0.0240,    0.0347]])
        np_test.assert_almost_equal(mrc.data[150:153, 20:23, 10], expected,
                                    decimal=4)
        np_test.assert_equal(mrc.byteOrder, '<')
        np_test.assert_equal(mrc.arrayOrder, 'FORTRAN')
        np_test.assert_equal(mrc.dataType, 'float32')
        np_test.assert_equal(mrc.data.dtype, numpy.dtype('float32'))
        np_test.assert_equal(mrc.memmap, False)

        # MRC tomo with memmap
        mrc = ImageIO()
        mrc.read(file=os.path.join(self.dir, "bin-2.mrc"), memmap=True)
        expected = numpy.array([[-0.0242,   -0.0250,    0.0883],
                                [0.0640,    0.0071,   -0.1300],
                                [-0.0421,   -0.0392,   -0.0312]])
        np_test.assert_almost_equal(mrc.data[50:53, 120:123, 40], expected,
                                    decimal=4)
        expected = numpy.array([[-0.0573,    0.0569,    0.0386],
                                [0.1309,    0.1211,   -0.0881],
                                [-0.0110,   -0.0240,    0.0347]])
        np_test.assert_almost_equal(mrc.data[150:153, 20:23, 10], expected,
                                    decimal=4)
        np_test.assert_equal(mrc.byteOrder, '<')
        np_test.assert_equal(mrc.arrayOrder, 'FORTRAN')
        np_test.assert_equal(mrc.dataType, 'float32')
        np_test.assert_equal(mrc.data.dtype, numpy.dtype('float32'))
        np_test.assert_equal(mrc.memmap, True)

        # MRC tomo with extended header
        mrc = ImageIO()
        mrc.read(file=os.path.join(self.dir, "bin-2_ext.mrc"), memmap=False)
        expected = numpy.array([[-0.0242,   -0.0250,    0.0883],
                                [0.0640,    0.0071,   -0.1300],
                                [-0.0421,   -0.0392,   -0.0312]])
        np_test.assert_almost_equal(mrc.data[50:53, 120:123, 40], expected,
                                    decimal=4)
        expected = numpy.array([[-0.0573,    0.0569,    0.0386],
                                [0.1309,    0.1211,   -0.0881],
                                [-0.0110,   -0.0240,    0.0347]])
        np_test.assert_almost_equal(mrc.data[150:153, 20:23, 10], expected,
                                    decimal=4)
        np_test.assert_equal(mrc.byteOrder, '<')
        np_test.assert_equal(mrc.arrayOrder, 'FORTRAN')
        np_test.assert_equal(mrc.dataType, 'float32')
        np_test.assert_equal(mrc.data.dtype, numpy.dtype('float32'))
        np_test.assert_equal(mrc.memmap, False)
        np_test.assert_equal(mrc.extendedHeaderLength, 5120)

        # MRC tomo with extended header and with memmap
        mrc = ImageIO()
        mrc.read(file=os.path.join(self.dir, "bin-2_ext.mrc"), memmap=True)
        expected = numpy.array([[-0.0242,   -0.0250,    0.0883],
                                [0.0640,    0.0071,   -0.1300],
                                [-0.0421,   -0.0392,   -0.0312]])
        np_test.assert_almost_equal(mrc.data[50:53, 120:123, 40], expected,
                                    decimal=4)
        expected = numpy.array([[-0.0573,    0.0569,    0.0386],
                                [0.1309,    0.1211,   -0.0881],
                                [-0.0110,   -0.0240,    0.0347]])
        np_test.assert_almost_equal(mrc.data[150:153, 20:23, 10], expected,
                                    decimal=4)
        np_test.assert_equal(mrc.byteOrder, '<')
        np_test.assert_equal(mrc.arrayOrder, 'FORTRAN')
        np_test.assert_equal(mrc.dataType, 'float32')
        np_test.assert_equal(mrc.data.dtype, numpy.dtype('float32'))
        np_test.assert_equal(mrc.memmap, True)
        np_test.assert_equal(mrc.extendedHeaderLength, 5120)

        # another MRC tomo (generated by and)
        mrc = ImageIO()
        mrc.read(file=os.path.join(self.dir, "and-tomo.mrc"))
        expected = numpy.array([[-0.0329,   -0.0006,   -0.0698],
                                [-0.0101,   -0.1196,   -0.1295],
                                [0.0844,   -0.0400,   -0.0716]])
        np_test.assert_almost_equal(mrc.data[50:53, 120:123, 40], expected,
                                    decimal=4)
        expected = numpy.array([[-0.0019,   -0.0085,    0.0036],
                                [0.0781,    0.0279,   -0.0365],
                                [0.0210,   -0.0193,   -0.0355]])
        np_test.assert_almost_equal(mrc.data[150:153, 20:23, 60], expected,
                                    decimal=4)
        np_test.assert_equal(mrc.dataType, 'float32')
        np_test.assert_equal(mrc.data.dtype, numpy.dtype('float32'))
        np_test.assert_equal(mrc.memmap, False)

        # another MRC tomo (generated by and) with memmap
        mrc = ImageIO()
        mrc.read(file=os.path.join(self.dir, "and-tomo.mrc"), memmap=True)
        expected = numpy.array([[-0.0329,   -0.0006,   -0.0698],
                                [-0.0101,   -0.1196,   -0.1295],
                                [0.0844,   -0.0400,   -0.0716]])
        np_test.assert_almost_equal(mrc.data[50:53, 120:123, 40], expected,
                                    decimal=4)
        expected = numpy.array([[-0.0019,   -0.0085,    0.0036],
                                [0.0781,    0.0279,   -0.0365],
                                [0.0210,   -0.0193,   -0.0355]])
        np_test.assert_almost_equal(mrc.data[150:153, 20:23, 60], expected,
                                    decimal=4)
        np_test.assert_equal(mrc.dataType, 'float32')
        np_test.assert_equal(mrc.data.dtype, numpy.dtype('float32'))
        np_test.assert_equal(mrc.memmap, True)

        # mrc with the opposite byte order
        mrc2 = ImageIO()
        mrc2.read(file=os.path.join(self.dir, "swapped_byte_order.mrc"))
        expected = numpy.array(
            [[ 0.000,    0.000],
             [-0.341,   -6.702],
             [0.782,  -11.780],
             [0.327,  -14.298],
             [-0.691,  -17.411],
             [-0.337,  -18.076],
             [-0.669,  -19.157],
             [-0.799,  -20.400],
             [-0.793,  -21.286],
             [-1.008,  -21.386]])
        np_test.assert_almost_equal(mrc2.data[:,:,0], expected, decimal=3)
        np_test.assert_equal(mrc2.memmap, False)
        raised = False
        try:
            mrc2.read(
                file=os.path.join(self.dir, "swapped_byte_order.mrc"), 
                memmap=True)
        except ValueError:
            raised = True
        np_test.assert_equal(raised, True)
        np_test.assert_equal(mrc2.memmap, True)
 
        # new style header mrc
        mrc_new = ImageIO()
        mrc_new.read(file=os.path.join(self.dir, 'new-head_int16.mrc'))
        np_test.assert_equal(mrc_new.dataType, 'int16')
        np_test.assert_equal(mrc_new.data.dtype, numpy.dtype('int16'))
        np_test.assert_equal(mrc_new.byteOrder, '<')
        np_test.assert_equal(mrc_new.arrayOrder, 'FORTRAN')
        np_test.assert_equal(mrc_new.shape, (40,30,20))
        np_test.assert_equal(mrc_new.pixel, [0.4, 0.4, 0.4])
        np_test.assert_equal(mrc_new.pixelsize, 0.4)
        np_test.assert_equal(mrc_new.data[14,8,10], -14)
        np_test.assert_equal(mrc_new.data[15,23,12], 10)
        np_test.assert_equal(mrc_new.data[23,29,16], 2)
        np_test.assert_equal(mrc_new.memmap, False)

        # new style header mrc
        mrc_new = ImageIO()
        mrc_new.read(
            file=os.path.join(self.dir, 'new-head_int16.mrc'), memmap=True)
        np_test.assert_equal(mrc_new.dataType, 'int16')
        np_test.assert_equal(mrc_new.data.dtype, numpy.dtype('int16'))
        np_test.assert_equal(mrc_new.byteOrder, '<')
        np_test.assert_equal(mrc_new.arrayOrder, 'FORTRAN')
        np_test.assert_equal(mrc_new.shape, (40,30,20))
        np_test.assert_equal(mrc_new.pixel, [0.4, 0.4, 0.4])
        np_test.assert_equal(mrc_new.pixelsize, 0.4)
        np_test.assert_equal(mrc_new.data[14,8,10], -14)
        np_test.assert_equal(mrc_new.data[15,23,12], 10)
        np_test.assert_equal(mrc_new.data[23,29,16], 2)
        np_test.assert_equal(mrc_new.memmap, True)
        np_test.assert_equal(mrc_new.n_labels, 9)
        np_test.assert_equal(len(mrc_new.labels), 9)
        desired = (
            "COMBINEFFT: Combined FFT from two tomograms             " 
            + "07-Oct-13  17:15:24" )
        np_test.assert_equal(len(mrc_new.labels[3]), 80)
        np_test.assert_equal(mrc_new.labels[3][:len(desired)], desired)
        desired = (
            "NEWSTACK: Images copied                                 10-Oct-13  18:00:03")
        np_test.assert_equal(len(mrc_new.labels[6]), 80)
        np_test.assert_equal(mrc_new.labels[6][:len(desired)], desired)

        # test raw file
        raw = ImageIO()
        raw.read(
            file=self.raw_file_name, dataType=self.raw_dtype, 
            shape=self.raw_shape)
        np_test.assert_equal(raw.data, self.raw_data)
        np_test.assert_equal(raw.memmap, False)

        # test raw file with memmap
        raw = ImageIO()
        raw.read(
            file=self.raw_file_name, dataType=self.raw_dtype, 
            shape=self.raw_shape, memmap=True)
        np_test.assert_equal(raw.data, self.raw_data)
        np_test.assert_equal(raw.memmap, True)


    def testWrite(self):
        """
        Tests write (and implicitly read), for em, mrc and raw format.
        """

        # arrays
        ar_uint8 = numpy.array([54, 200, 5, 7, 45, 123], 
                               dtype='uint8').reshape((3,1,2))
        ar_int8 = numpy.array([54, 2, -5, 7, 45, 123], 
                               dtype='uint8').reshape((3,1,2))
        ar_uint16 = numpy.array([1034, 546, 248, 40000, 2345, 365, 4876, 563],
                               dtype='uint16').reshape((2,2,2))
        ar_int16 = numpy.array([1034, 546, -248, 156, 2345, 365, -4876, 563],
                               dtype='int16').reshape((2,2,2))
        ar_int32 = numpy.array([1034, 56546, -223448, 156, 
                                2345, 2**31-10, -884876, 563],
                               dtype='int32').reshape((2,2,2))
        ar_uint32 = numpy.array([1034, 56546, 223448, 156, 
                                 2345, 365, 884876, 2**32-10],
                               dtype='uint32').reshape((2,2,2))
        ar_int8_2 = numpy.arange(24, dtype='int8').reshape((4,3,2))
        ar_int16_2 = numpy.arange(24, dtype='int16').reshape((4,3,2))
        ar2_int16 = numpy.array([1034, 546, -248, 156, 2345, 365, -4876, 563],
                                dtype='int16').reshape((2,4))
        ar_int16_f = numpy.array(
            [1034, 546, -248, 156, 2345, 365, -4876, 563],
            dtype='int16', order='F').reshape((2,2,2))
        ar_int16_c = numpy.array(
            [1034, 546, -248, 156, 2345, 365, -4876, 563],
            dtype='int16', order='C').reshape((2,2,2))
  
        # em uint8
        file_out = ImageIO()
        file_out.write(file='_test.em', data=ar_uint8)
        file_in = ImageIO()
        file_in.read(file='_test.em')
        np_test.assert_equal(file_in.dataType, 'uint8')
        np_test.assert_equal(file_in.data, ar_uint8)

        # em uint16
        file_out = ImageIO()
        file_out.write(file='_test.em', data=ar_uint16)
        file_in = ImageIO()
        file_in.read(file='_test.em')
        np_test.assert_equal(file_in.dataType, 'uint16')
        np_test.assert_equal(file_in.data, ar_uint16)

        # em int16 converted to int32, safe casting
        file_out = ImageIO()
        file_out.write(file='_test.em', data=ar_int16, dataType='int32', 
                       casting='safe')
        file_in = ImageIO()
        file_in.read(file='_test.em')
        np_test.assert_equal(file_in.dataType, 'int32')
        np_test.assert_equal(file_in.data, ar_int16)

        # em int16, safe casting 
        file_out = ImageIO()
        np_test.assert_raises(
            TypeError,
            file_out.write,
            **{'file':'_test.em', 'data':ar_int16, 'casting':'safe'})

        # em int16 converted to uint16, unsafe casting
        file_out = ImageIO()
        file_out.write(file='_test.em', data=ar_int16, dataType='uint16', 
                       casting='unsafe')
        file_in = ImageIO()
        file_in.read(file='_test.em')
        np_test.assert_equal(file_in.dataType, 'uint16')
        np_test.assert_equal(file_in.data.dtype, numpy.dtype('uint16'))
        #np_test.assert_equal(file_in.data, ar_int16) should fail
        np_test.assert_equal(file_in.data[0,1,0] == ar_int16[0,1,0], False)

        # em int16 to uint16, safe casting 
        file_out = ImageIO()
        np_test.assert_raises(
            TypeError,
            file_out.write,
            **{'file':'_test.em', 'data':ar_int16, 'dataType':'uint16', 
               'casting':'safe'})

        # em uint16 to int16, unsafe casting 
        file_out = ImageIO()
        np_test.assert_raises(
            TypeError,
            file_out.write,
            **{'file':'_test.em', 'data':ar_uint16, 'dataType':'int16', 
               'casting':'unsafe'})

        # em uint32 to int32, safe casting 
        file_out = ImageIO()
        np_test.assert_raises(
            TypeError,
            file_out.write,
            **{'file':'_test.em', 'data':ar_uint32, 'dataType':'int32', 
               'casting':'safe'})

        # em uint32 converted to int32, unsafe casting
        file_out = ImageIO()
        file_out.write(file='_test.em', data=ar_uint32, dataType='int32', 
                       casting='unsafe')
        file_in = ImageIO()
        file_in.read(file='_test.em')
        np_test.assert_equal(file_in.dataType, 'int32')
        #np_test.assert_equal(file_in.data, ar_uint32) should fail
        np_test.assert_equal(file_in.data[0,0,0] == ar_uint32[0,0,0], True)
        np_test.assert_equal(file_in.data[1,1,1] == ar_uint32[1,1,1], False)

        # em uint32 to float32, safe casting 
        file_out = ImageIO()
        np_test.assert_raises(
            TypeError,
            file_out.write,
            **{'file':'_test.em', 'data':ar_uint32, 'dataType':'float32', 
               'casting':'safe'})

        # em uint32 to float32, unsafe casting
        file_out = ImageIO()
        file_out.write(file='_test.em', data=ar_uint32, dataType='float32', 
                       casting='unsafe')
        file_in = ImageIO()
        file_in.read(file='_test.em')
        np_test.assert_equal(file_in.dataType, 'float32')
        #np_test.assert_almost_equal(file_in.data, ar_uint32)  should fail
        np_test.assert_almost_equal(
            file_in.data[0,0,0] == ar_uint32[0,0,0], True)
        np_test.assert_almost_equal(
            file_in.data[1,1,1] == ar_uint32[1,1,1], False)

        # em int32 to float32, unsafe casting
        file_out = ImageIO()
        file_out.write(file='_test.em', data=ar_int32, dataType='float32', 
                       casting='unsafe')
        file_in = ImageIO()
        file_in.read(file='_test.em')
        np_test.assert_equal(file_in.dataType, 'float32')
        #np_test.assert_almost_equal(file_in.data, ar_int32)  should fail
        np_test.assert_almost_equal(
            file_in.data[0,0,0] == ar_int32[0,0,0], True)
        np_test.assert_almost_equal(
            file_in.data[1,0,1] == ar_int32[1,0,1], False)

        # em int32 to float64, safe casting
        file_out = ImageIO()
        file_out.write(file='_test.em', data=ar_int32, dataType='float64', 
                       casting='safe')
        file_in = ImageIO()
        file_in.read(file='_test.em')
        np_test.assert_equal(file_in.dataType, 'float64')
        np_test.assert_almost_equal(file_in.data, ar_int32)

        # mrc data type and shape from args
        file_out = ImageIO()
        file_out.write(
            file='_test.mrc', data=ar_int8_2, shape=(2,3,4), dataType='int16')
        file_in = ImageIO()
        file_in.read(file='_test.mrc')
        np_test.assert_equal(file_in.dataType, 'int16')
        np_test.assert_equal(file_in.shape, (2,3,4))

        # mrc data type and shape from previously given data
        file_out = ImageIO()
        file_out.setData(ar_int16_2)
        file_out.write(file='_test.mrc')
        file_in = ImageIO()
        file_in.read(file='_test.mrc')
        np_test.assert_equal(file_in.dataType, 'int16')
        np_test.assert_equal(file_in.shape, (4,3,2))

        # mrc data type and shape from attributes
        file_out = ImageIO()
        file_out.data = ar_int8_2
        file_out.shape = (2,3,4)
        file_out.dataType = 'int16'
        file_out.write(file='_test.mrc')
        file_in = ImageIO()
        file_in.read(file='_test.mrc')
        np_test.assert_equal(file_in.dataType, 'int16')
        np_test.assert_equal(file_in.shape, (2,3,4))

        # mrc data type and shape from data
        file_out = ImageIO()
        file_out.write(file='_test.mrc', data=ar_int16_2)
        file_in = ImageIO()
        file_in.read(file='_test.mrc')
        np_test.assert_equal(file_in.dataType, 'int16')
        np_test.assert_equal(file_in.shape, (4,3,2))

        # mrc uint8, same as ubyte
        file_out = ImageIO()
        file_out.write(file='_test.mrc', data=ar_uint8)
        file_in = ImageIO()
        file_in.read(file='_test.mrc')
        np_test.assert_equal(file_in.dataType, 'ubyte')
        np_test.assert_almost_equal(file_in.data, ar_uint8)

        # mrc uint16 
        file_out = ImageIO()
        np_test.assert_raises(
            (KeyError, TypeError),
            file_out.write,
            **{'file':'_test.mrc', 'data':ar_uint16})

        # mrc uint16 to int16, safe casting
        file_out = ImageIO()
        np_test.assert_raises(
            TypeError,
            file_out.write,
            **{'file':'_test.mrc', 'data':ar_uint16, 'dataType':'ubyte', 
               'casting':'safe'})

        # mrc uint16 to int16, unsafe casting
        file_out = ImageIO()
        file_out.write(file='_test.mrc', data=ar_uint16, dataType='int16', 
                       casting='unsafe')
        file_in = ImageIO()
        file_in.read(file='_test.mrc')
        np_test.assert_equal(file_in.dataType, 'int16')
        #np_test.assert_almost_equal(file_in.data, ar_uint16)  should fail
        np_test.assert_equal(file_in.data[0,0,0] == ar_uint16[0,0,0], True)
        np_test.assert_equal(file_in.data[0,1,1] == ar_uint16[0,1,1], False)

        # mrc int16
        file_out = ImageIO()
        file_out.write(file='_test.mrc', data=ar_int16, pixel=2.3)
        file_in = ImageIO()
        file_in.read(file='_test.mrc')
        np_test.assert_equal(file_in.dataType, 'int16')
        np_test.assert_equal(file_in.data, ar_int16)
        np_test.assert_equal(file_in.pixel, [2.3, 2.3, 2.3])
        np_test.assert_equal(file_in.pixelsize, 2.3)

        # mrc int16 2D
        file_out = ImageIO()
        file_out.write(file='_test.mrc', data=ar2_int16, pixel=3.4)
        file_in = ImageIO()
        file_in.read(file='_test.mrc')
        np_test.assert_equal(file_in.dataType, 'int16')
        np_test.assert_equal(file_in.data[:,:,0], ar2_int16)        
        np_test.assert_equal(file_in.pixelsize, 3.4)

        # mrc int8 to int16
        file_out = ImageIO()
        file_out.write(file='_test.mrc', data=ar_int8, dataType='int16', 
                       casting='safe')
        file_in = ImageIO()
        file_in.read(file='_test.mrc')
        np_test.assert_equal(file_in.dataType, 'int16')
        np_test.assert_equal(file_in.data, ar_int8)

        # mrc int32 
        file_out = ImageIO()
        np_test.assert_raises(
            (KeyError, TypeError),
            file_out.write,
            **{'file':'_test.mrc', 'data':ar_int32})

        # mrc int32 to int16
        file_out = ImageIO()
        np_test.assert_raises(
            TypeError,
            file_out.write,
            **{'file':'_test.mrc', 'data':ar_int32, 'dataType':'int16',
               'casting':'safe'})

        # mrc int32 to float32
        file_out = ImageIO()
        np_test.assert_raises(
            TypeError,
            file_out.write,
            **{'file':'_test.mrc', 'data':ar_int32, 'dataType':'float32',
               'casting':'safe'})

        # mrc int32 to complex64
        file_out = ImageIO()
        np_test.assert_raises(
            TypeError,
            file_out.write,
            **{'file':'_test.mrc', 'data':ar_int32, 'dataType':'complex64',
               'casting':'safe'})

        # raw int16
        file_out = ImageIO()
        file_out.write(file='_test.raw', data=ar_int16)
        file_in = ImageIO()
        file_in.read(file='_test.raw', dataType='int16', shape=(2,2,2))
        np_test.assert_equal(file_in.dataType, 'int16')
        np_test.assert_equal(file_in.data, ar_int16)

        # raw int8 to int16
        file_out = ImageIO()
        file_out.write(file='_test.raw', data=ar_int8, dataType='int16')
        file_in = ImageIO()
        file_in.read(file='_test.raw', dataType='int16', shape=(3,1,2))
        np_test.assert_equal(file_in.dataType, 'int16')
        np_test.assert_equal(file_in.data, ar_int8)

        # raw int16 to int8
        file_out = ImageIO()
        np_test.assert_raises(
            TypeError,
            file_out.write,
            **{'file':'_test.raw', 'data':ar_int16, 'dataType':'int8',
               'casting':'safe'})

        # explain error messages printed before 
        print("It's fine if few error messages were printed just before " +
              "this line, because they have been caught.")

        # shape param
        file_out = ImageIO()
        file_out.write(file='_test.mrc', data=ar_int16, dataType='int16')
        file_in = ImageIO()
        file_in.read(file='_test.mrc', dataType='int16')
        np_test.assert_equal(file_in.data.shape, (2,2,2))
        file_out = ImageIO()
        file_out.write(
            file='_test.mrc', data=ar_int16, dataType='int16', shape=(1,4,2))
        file_in = ImageIO()
        file_in.read(file='_test.mrc', dataType='int16')
        np_test.assert_equal(file_in.data.shape, (1,4,2))
        file_out.write(
            file='_test.mrc', data=ar_int16, dataType='int16', shape=(4,2))
        file_in.readHeader(file='_test.mrc')
        file_in.read(file='_test.mrc', dataType='int16')
        np_test.assert_equal(file_in.data.shape, (4,2,1))
        file_in.read(file='_test.mrc', dataType='int16', shape=(2,2,2))
        np_test.assert_equal(file_in.data.shape, (2,2,2))

        # array order C, read write default (F)
        file_out = ImageIO()
        file_out.write(file='_test.mrc', data=ar_int16_c)
        file_in = ImageIO()
        file_in.read(file='_test.mrc')
        np_test.assert_equal(file_in.data, ar_int16_c)

        # array order C, read write C
        file_out = ImageIO()
        file_out.write(file='_test.mrc', data=ar_int16_c, arrayOrder='C')
        file_in = ImageIO()
        file_in.read(file='_test.mrc', arrayOrder='C')
        np_test.assert_equal(file_in.data, ar_int16_c)

        # array order F, read write default (F)
        file_out = ImageIO()
        file_out.write(file='_test.mrc', data=ar_int16_f)
        file_in = ImageIO()
        file_in.read(file='_test.mrc')
        np_test.assert_equal(file_in.data, ar_int16_f)

        # array order F, read write F
        file_out = ImageIO()
        file_out.write(file='_test.mrc', data=ar_int16_f, arrayOrder='F')
        file_in = ImageIO()
        file_in.read(file='_test.mrc', arrayOrder='F')
        np_test.assert_equal(file_in.data, ar_int16_f)

    def testPixelSize(self):
        """
        Tests pixel size in read and write
        """

        # arrays
        #ar_int8_2 = numpy.arange(24, dtype='int8').reshape((4,3,2))
        ar_int16_2 = numpy.arange(24, dtype='int16').reshape((4,3,2))

        #
        file_out = ImageIO()
        file_out.write(file='_test.mrc', data=ar_int16_2, pixel=2.1)
        file_in = ImageIO()
        file_in.read(file='_test.mrc')
        np_test.assert_almost_equal(file_in.pixel, 2.1)

    def tearDown(self):
        """
        Remove temporary files
        """
        try:
            os.remove(os.path.join(self.dir, '_test.em'))
        except OSError:
            pass
        try:
            os.remove(os.path.join(self.dir, '_test.mrc'))
        except OSError:
            pass
        try:
            os.remove(os.path.join(self.dir, '_test.raw'))
        except OSError:
            pass
        try:
            os.remove(os.path.join(self.dir, self.raw_file_name))
        except OSError:
            pass


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestImageIO)
    unittest.TextTestRunner(verbosity=2).run(suite)
