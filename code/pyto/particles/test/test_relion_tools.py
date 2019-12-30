"""

Tests relion_tools.py

# Author: Vladan Lucic (Max Planck Institute of Biochemistry)
# $Id: test_relion_tools.py 1451 2017-04-30 10:08:58Z vladan $
"""

__version__ = "$Revision: 1451 $"


import os
import unittest

import numpy as np
import numpy.testing as np_test 


import pyto.particles.relion_tools as relion_tools

class TestRelionTools(np_test.TestCase):
    """
    """

    def setUp(self):
        curr_dir, base = os.path.split(__file__)
        self.test_dir =  os.path.join(curr_dir, 'test_data')
        self.test_base =  os.path.join(curr_dir, 'test_data/')


    def test_get_n_particle_class_change(self):
        """
        Tests get_n_particle_class_change()
        """

        # class_ None
        res = relion_tools.get_n_particle_class_change(
            basename=self.test_base, iters=[2,5], class_=None, mode='change',
            fraction=False, tablename='data_', out='list')
        np_test.assert_equal(res, [20])
        res = relion_tools.get_n_particle_class_change(
            basename=self.test_base, iters=[2,5], class_=None, mode='change',
            fraction=True, tablename='data_', out='dict')
        np_test.assert_equal(res, {5 : 0.4})

        # individual classes
        res = relion_tools.get_n_particle_class_change(
            basename=self.test_base, iters=[2,5], class_=[1,2,3,4,5], 
            mode='change', fraction=False, tablename='data_', out='dict')
        np_test.assert_equal(res, {5 : [3, 5, 4, 6, 2]})
        res = relion_tools.get_n_particle_class_change(
            basename=self.test_base, iters=[2,5], class_=[1,2,3,4,5], 
            mode='change', fraction=True, tablename='data_', out='dict')
        np_test.assert_equal(res, {5 : [3/10., 5/10., 4/10., 6/13., 2/7.]})
        res = relion_tools.get_n_particle_class_change(
            basename=self.test_base, iters=[2,5], class_=[1,2,3,4,5], 
            mode='change', fraction=True, tablename='data_', out='list')
        np_test.assert_equal(res, [[3/10., 5/10., 4/10., 6/13., 2/7.]])
       
       # grouped classes
        res = relion_tools.get_n_particle_class_change(
            basename=self.test_base, iters=[2,5], class_=[1,(2,3,4),5], 
            mode='change', fraction=False, tablename='data_', out='dict')
        np_test.assert_equal(res, {5 : [3, 7, 2]})
        res = relion_tools.get_n_particle_class_change(
            basename=self.test_base, iters=[2,5], class_=[1,(2,4),(3,5)], 
            mode='change', fraction=True, tablename='data_', out='dict')
        np_test.assert_equal(res, {5 : [3/10., 8/23., 5/17.]})
       
        # individual classes to_from
        res = relion_tools.get_n_particle_class_change(
            basename=self.test_base, iters=[2,5], class_=[1,2,3,4,5], 
            mode='to_from', fraction=False, tablename='data_', out='dict')
        np_test.assert_equal(
            res, {5 : np.array([[7, 1, 2, 0, 0],
                                [0, 5, 2, 3, 0],
                                [0, 0, 6, 3, 1],
                                [2, 0, 0, 7, 1],
                                [1, 4, 0, 0, 5]])})

        # grouped classes to_from
        res = relion_tools.get_n_particle_class_change(
            basename=self.test_base, iters=[2,5], class_=[1,(2,4),(3,5)], 
            mode='to_from', fraction=False, tablename='data_', out='dict')
        np_test.assert_equal(
            res, {5 : np.array([[7, 1, 2],
                                [2, 15, 3],
                                [1, 7, 12]])})
        res = relion_tools.get_n_particle_class_change(
            basename=self.test_base, iters=[2,5], class_=[1,(2,3,4),5],
            mode='to_from', fraction=True, tablename='data_', out='dict')
        np_test.assert_equal(
            res, {5 : np.array([[7/10., 3/33., 0],
                                [2/10., 26/33., 2/7.],
                                [1/10., 4/33., 5/7.]])})
        res = relion_tools.get_n_particle_class_change(
            basename=self.test_base, iters=[2,5], class_=[1,(2,3,4),5],
            mode='to_from', fraction=True, tablename='data_', out='list')
        np_test.assert_equal(
            res, [np.array([[7/10., 3/33., 0],
                            [2/10., 26/33., 2/7.],
                            [1/10., 4/33., 5/7.]])])

    def test_find_file(self):
        """
        Tests find_file()
        """

        # no continuation
        res = relion_tools.find_file(
            basename=self.test_base, suffix='_data.star', iter_=2)
        np_test.assert_equal(res, [self.test_base + '_it002_data.star'])
        res = relion_tools.find_file(
            basename=self.test_base, suffix='_data.star', iter_=3)
        np_test.assert_equal(res is None, True) 
        res = relion_tools.find_file(
            basename=self.test_base, suffix='_data.star', iter_=2, half=2)
        np_test.assert_equal(res is None, True) 
        res = relion_tools.find_file(
            basename=self.test_base, suffix='_data.star', iter_=None)
        np_test.assert_equal(res is None, True) 

        # with continuation
        print("Watch now")
        res = relion_tools.find_file(
            basename=self.test_dir+'/a', suffix='_model.star', iter_=4, half=2)
        np_test.assert_equal(
            res, [self.test_dir + '/a_it004_half2_model.star'])
        res = relion_tools.find_file(
            basename=self.test_dir+'/a', suffix='_model.star', iter_=5, half=2)
        np_test.assert_equal(
            set(res), 
            set([self.test_dir + '/a_ct4_it005_half2_model.star',
                 self.test_dir + '/a_ct5_it005_half2_model.star']))

    def test_two_way_class(self):
        """
        Tests two_way_class
        """

        res = relion_tools.two_way_class(
            basename=self.test_base, iters=[5], mode=('class', 'find'),
            pattern=(range(1,6), ['tomo-reconstruct-01','tomo-reconstruct-02']),
            label=('rlnClassNumber', 'rlnMicrographName'), tablename='data_',
            suffix='_data.star', type_=(int, str), iter_format='_it%03d',
            method='contingency')
        np_test.assert_equal(
            res[5].contingency, [[10, 0], [5, 5], [2, 8], [7, 6], [6, 1]]) 
        np_test.assert_almost_equal(
            res[5].fraction, 
            [[1, 0], [0.5, 0.5], [0.2, 0.8], [7/13., 6/13.], [6/7., 1/7.]]) 
        np_test.assert_almost_equal(
            res[5].total_fract[0], [30/50., 20/50.])
        np_test.assert_almost_equal(
            res[5].total_fract[1], [0.2, 0.2, 0.2, 13/50., 7/50.])

        res = relion_tools.two_way_class(
            basename=self.test_base, iters=5, mode=('class', 'find'),
            pattern=([(1,2), 3, (4,5)], [
                'tomo-reconstruct-01','tomo-reconstruct-02']),
            label=('rlnClassNumber', 'rlnMicrographName'), tablename='data_',
            suffix='_data.star', type_=(int, str), iter_format='_it%03d',
            method='contingency')
        np_test.assert_equal(
            res.contingency, [[15, 5], [2, 8], [13, 7]]) 
        np_test.assert_almost_equal(
            res.fraction, [[0.75, 0.25], [0.2, 0.8], [13/20., 7/20.]]) 
        np_test.assert_almost_equal(
            res.total_fract[0], [30/50., 20/50.])
        np_test.assert_almost_equal(
            res.total_fract[1], [0.4, 0.2, 20/50.])

        res = relion_tools.class_group_interact(
            method='contingency', basename=self.test_base, iters=5, 
            classes=range(1,6), group_mode='find',
            group_pattern=['tomo-reconstruct-01','tomo-reconstruct-02'],
            group_label='rlnMicrographName', group_type=str,
            tablename='data_')
        np_test.assert_equal(
            res.contingency, [[10, 0], [5, 5], [2, 8], [7, 6], [6, 1]]) 
        np_test.assert_almost_equal(
            res.fraction, 
            [[1, 0], [0.5, 0.5], [0.2, 0.8], [7/13., 6/13.], [6/7., 1/7.]]) 

    def test_symmetrize_structure(self):
        """
        Tests symmetrize_structure()
        """

        # array: up in +z, end in +y 
        gg = np.zeros((10,10,10))
        gg[4,2:7,2:7] = np.array(
            [[0,0,0,0,0],
             [0,0,0,0,0],
             [1,2,3,4,5],
             [0,0,0,0,6],
             [0,0,0,0,0]])

        # mask
        ma = np.zeros((10,10,10))
        ma[4,2:7,2:7] = np.array(
            [[0,0,0,0,0],
             [0,0,0,0,0],
             [0,0,0,1,1],
             [0,0,0,0,1],
             [0,0,0,0,0]])

        # c2 mask
        res = relion_tools.symmetrize_structure(
            structure=gg, symmetry='C2', origin=[4,4,4], mask=ma)
        desired_4xx = np.array(
            [[0,0,0,0,0],
             [0,0,0,0,3],
             [0,0,0,4,5],
             [0,0,0,0,3],
             [0,0,0,0,0]])
        desired_x4x = np.array(
            [[0,0,0,0,0],
             [0,0,0,0,0],
             [0,0,0,4,5],
             [0,0,0,0,0],
             [0,0,0,0,0]])
        np_test.assert_almost_equal(res[4,2:7,2:7], desired_4xx)
        np_test.assert_almost_equal(res[2:7,4,2:7], desired_x4x)

        # c4
        res = relion_tools.symmetrize_structure(
            structure=gg, symmetry='C4', origin=[4,4,4])
        desired_4xx = np.array(
            [[0,0,0,0,0],
             [0,0,0,0,1.5],
             [1,2,3,4,5],
             [0,0,0,0,1.5],
             [0,0,0,0,0]])
        desired_x4x = np.array(
            [[0,0,0,0,0],
             [0,0,0,0,1.5],
             [1,2,3,4,5],
             [0,0,0,0,1.5],
             [0,0,0,0,0]])
        np_test.assert_almost_equal(res[4,2:7,2:7], desired_4xx)
        np_test.assert_almost_equal(res[2:7,4,2:7], desired_x4x)

        # d2
        res = relion_tools.symmetrize_structure(
            structure=gg, symmetry='D2', origin=[4,4,4])
        desired_4xx = np.array(
            [[0,0,0,0,0],
             [1.5,0,0,0,1.5],
             [3,3,3,3,3],
             [1.5,0,0,0,1.5],
             [0,0,0,0,0]])
        desired_x4x = np.array(
            [[0,0,0,0,0],
             [0,0,0,0,0],
             [3,3,3,3,3],
             [0,0,0,0,0],
             [0,0,0,0,0]])
        np_test.assert_almost_equal(res[4,2:7,2:7], desired_4xx)
        np_test.assert_almost_equal(res[2:7,4,2:7], desired_x4x)

        # d4
        res = relion_tools.symmetrize_structure(
            structure=gg, symmetry='D4', origin=[4,4,4])
        desired_4 = np.array(
            [[0,0,0,0,0],
             [0.75,0,0,0,0.75],
             [3,3,3,3,3],
             [0.75,0,0,0,0.75],
             [0,0,0,0,0]])
        np_test.assert_almost_equal(res[4,2:7,2:7], desired_4)
        np_test.assert_almost_equal(res[2:7,4,2:7], desired_4)


        

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRelionTools)
    unittest.TextTestRunner(verbosity=2).run(suite)


