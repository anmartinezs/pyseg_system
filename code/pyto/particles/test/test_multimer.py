"""

Tests multimer.py

# Author: Vladan Lucic (Max Planck Institute of Biochemistry)
# $Id: test_phantom.py 1451 2017-04-30 10:08:58Z vladan $
"""

__version__ = "$Revision: 1451 $"


import os
import unittest

import numpy as np
import numpy.testing as np_test 

from pyto.geometry.rigid_3d import Rigid3D as Rigid3D
from pyto.particles.multimer import Multimer as Multimer
from pyto.particles.phantom import Phantom as Phantom

class TestMultimer(np_test.TestCase):
    """
    """

    def setUp(self):
        """
        """
        
        # binary phantom closed wo overlap
        base_size = [2,3,8]
        prot_size = [2,2,6]
        base_pos = [0,0,0]
        prot_pos = [0,3,2]
        self.phantom_1_closed = Phantom.make_2box(
            base_size=base_size, protrusion_size=prot_size, base_pos=base_pos,
            protrusion_pos=prot_pos, shape=[20,20,20], center=[0,0,0], 
            origin=[10,10,10], binary=True)

        # binary phantom extended wo overlap
        base_size = [2,3,8]
        prot_size = [2,6,2]
        base_pos = [0,0,0]
        prot_pos = [0,3,6]
        self.phantom_1_extend = Phantom.make_2box(
            base_size=base_size, protrusion_size=prot_size, base_pos=base_pos,
            protrusion_pos=prot_pos, shape=[20,20,20], center=[0,0,0], 
            origin=[10,10,10], binary=True)

        # not binary phantom w overlap
        base_size = [2,3,8]
        prot_size = [2,6,5]
        base_pos = [0,0,0]
        prot_pos = [0,3,3]
        self.phantom_1_over = Phantom.make_2box(
            base_size=base_size, protrusion_size=prot_size, base_pos=base_pos,
            protrusion_pos=prot_pos, shape=[20,20,20], center=[0,0,0], 
            origin=[10,10,10], binary=True)
       
    def test_make_symmetry_transforms(self):
        """
        Test make_symmetry_transforms()
        """

        multi = Multimer(symmetry='C1')
        trans = multi.make_symmetry_transforms(affine=False)
        np_test.assert_equal(trans, [[0,0,0]])

        # angles
        multi = Multimer()
        trans = multi.make_symmetry_transforms(
            symmetry='d6', affine=False, degree=True)
        np_test.assert_equal(
            trans, 
            [[0,0,0], [60,0,0], [120,0,0], [180,0,0], [240,0,0], [300,0,0], 
             [0,180,0], [60,180,0], [120,180,0], [180,180,0], 
             [240,180,0], [300,180,0]]) 

        # Rigid3D
        multi = Multimer(symmetry='c4')
        trans = multi.make_symmetry_transforms(affine=True)
        np_test.assert_equal(len(trans), 4)
        np_test.assert_almost_equal(
            Rigid3D.extract_euler(r=trans[0].gl, mode='zxz_ex_active'), [0,0,0])
        np_test.assert_almost_equal(
            Rigid3D.extract_euler(r=trans[1].q, mode='zxz_ex_active'), 
            [np.pi/2.,0,0])
        np_test.assert_almost_equal(
            Rigid3D.extract_euler(r=trans[2].gl, mode='zxz_ex_active'), 
            [np.pi,0,0])
        np_test.assert_almost_equal(
            Rigid3D.extract_euler(r=trans[3].q, mode='zxz_ex_active'), 
            [-np.pi/2,0,0])
        np_test.assert_almost_equal(trans[0].s_scalar, 1)
        np_test.assert_almost_equal(trans[1].translation, [0,0,0])

        # Rigid3D
        multi = Multimer(symmetry='D3')
        trans = multi.make_symmetry_transforms(affine=True)
        np_test.assert_equal(len(trans), 6)
        np_test.assert_almost_equal(
            Rigid3D.extract_euler(r=trans[1].gl, mode='zxz_ex_active'), 
            [2 * np.pi / 3, 0, 0])
        np_test.assert_almost_equal(
            Rigid3D.extract_euler(r=trans[4].gl, mode='zxz_ex_active'), 
            [2 * np.pi / 3, np.pi, 0])
        np_test.assert_almost_equal(
            Rigid3D.extract_euler(r=trans[2].gl, mode='zxz_ex_active'), 
            [-2 * np.pi / 3, 0, 0])

    def test_sym_rotate(self):
        """
        Tests symmetry_rotate(), and implicitly initialize_sym_rotated(),
        set_sym_rotated() and get_sym_rotated().
        """

        # heteromer d2
        multi = Multimer(
            symmetry='d2', 
            monomers=[self.phantom_1_extend, self.phantom_1_closed])
        multi.sym_rotate()
        closed_x_0 = np.array(
            [[1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1],
             [0, 0, 1, 1, 1, 1, 1, 1],
             [0, 0, 1, 1, 1, 1, 1, 1],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0]])
        extend_x_0 = np.array(
            [[1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1],
             [0, 0, 0, 0, 0, 0, 1, 1],
             [0, 0, 0, 0, 0, 0, 1, 1],
             [0, 0, 0, 0, 0, 0, 1, 1],
             [0, 0, 0, 0, 0, 0, 1, 1],
             [0, 0, 0, 0, 0, 0, 1, 1],
             [0, 0, 0, 0, 0, 0, 1, 1]])
        np_test.assert_almost_equal(
            multi.get_sym_rotated(mono_index=0,sym_index=0)[10, 10:19, 10:18], 
            extend_x_0)
        np_test.assert_almost_equal(
            multi.get_sym_rotated(mono_index=0,sym_index=0)[11, 10:19, 10:18], 
            extend_x_0)
        np_test.assert_almost_equal(
            multi.get_sym_rotated(mono_index=0,sym_index=1)[10, 10:1:-1, 10:18],
            extend_x_0)
        np_test.assert_almost_equal(
            multi.get_sym_rotated(
                mono_index=0,sym_index=2)[11, 10:1:-1, 10:2:-1],
            extend_x_0)
        np_test.assert_almost_equal(
            multi.get_sym_rotated(mono_index=0,sym_index=3)[10, 10:19, 10:2:-1],
            extend_x_0)
        np_test.assert_almost_equal(
            multi.get_sym_rotated(mono_index=1,sym_index=0)[11, 10:19, 10:18], 
            closed_x_0)
        np_test.assert_almost_equal(
            multi.get_sym_rotated(mono_index=1,sym_index=0)[10, 10:19, 10:18], 
            closed_x_0)
        np_test.assert_almost_equal(
            multi.get_sym_rotated(mono_index=1,sym_index=1)[9, 10:1:-1, 10:18],
            closed_x_0)
        np_test.assert_almost_equal(
            multi.get_sym_rotated(
                mono_index=1,sym_index=2)[10, 10:1:-1, 10:2:-1],
            closed_x_0)
        np_test.assert_almost_equal(
            multi.get_sym_rotated(mono_index=1,sym_index=3)[9, 10:19, 10:2:-1],
            closed_x_0)

        # heteromer c4
        multi = Multimer(
            symmetry='C4', 
            monomers=[self.phantom_1_extend, self.phantom_1_closed])
        multi.sym_rotate()
        closed_x_0 = np.array(
            [[1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1],
             [0, 0, 1, 1, 1, 1, 1, 1],
             [0, 0, 1, 1, 1, 1, 1, 1],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0]])
        extend_x_0 = np.array(
            [[1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1],
             [0, 0, 0, 0, 0, 0, 1, 1],
             [0, 0, 0, 0, 0, 0, 1, 1],
             [0, 0, 0, 0, 0, 0, 1, 1],
             [0, 0, 0, 0, 0, 0, 1, 1],
             [0, 0, 0, 0, 0, 0, 1, 1],
             [0, 0, 0, 0, 0, 0, 1, 1]])
        np_test.assert_almost_equal(
            multi.get_sym_rotated(mono_index=0,sym_index=0)[10, 10:19, 10:18], 
            extend_x_0)
        np_test.assert_almost_equal(
            multi.get_sym_rotated(mono_index=0,sym_index=0)[11, 10:19, 10:18], 
            extend_x_0)
        np_test.assert_almost_equal(
            multi.get_sym_rotated(mono_index=0,sym_index=1)[10:1:-1, 10, 10:18],
            extend_x_0)
        np_test.assert_almost_equal(
            multi.get_sym_rotated(mono_index=0,sym_index=2)[9, 10:1:-1, 10:18],
            extend_x_0)
        np_test.assert_almost_equal(
            multi.get_sym_rotated(mono_index=0,sym_index=3)[10:19, 10, 10:18],
            extend_x_0)
        np_test.assert_almost_equal(
            multi.get_sym_rotated(mono_index=1,sym_index=0)[10, 10:19, 10:18], 
            closed_x_0)
        np_test.assert_almost_equal(
            multi.get_sym_rotated(mono_index=1,sym_index=0)[11, 10:19, 10:18], 
            closed_x_0)
        np_test.assert_almost_equal(
            multi.get_sym_rotated(mono_index=1,sym_index=1)[10:1:-1, 11, 10:18],
            closed_x_0)
        np_test.assert_almost_equal(
            multi.get_sym_rotated(mono_index=1,sym_index=2)[10, 10:1:-1, 10:18],
            closed_x_0)
        np_test.assert_almost_equal(
            multi.get_sym_rotated(mono_index=1,sym_index=3)[10:19, 9, 10:18],
            closed_x_0)

    def test_initialize_arrangement(self):
        """
        Tests initialize_arrangement()
        """

        # two multimer types
        multi = Multimer()
        composition=([3, 2, 1], [2, 2, 2])
        probability = (0.7, 0.3)
        multi.initialize_arrangement(
            composition=composition, probability=probability)
        np_test.assert_equal(multi.composition, composition)
        np_test.assert_equal(multi.multimer_type_probability, probability)
        np_test.assert_equal(
            multi.initial_arrangement, ([0, 0, 0, 1, 1, 2], [0, 0, 1, 1, 2, 2]))
        
        # one multimer type
        multi = Multimer()
        composition=[3, 2, 1, 3]
        probability = None
        multi.initialize_arrangement(
            composition=composition, probability=probability)
        np_test.assert_equal(multi.composition, (composition,))
        np_test.assert_equal(multi.multimer_type_probability, (1,))
        np_test.assert_equal(
            multi.initial_arrangement, ([0, 0, 0, 1, 1, 2, 3, 3, 3],))

    def test_arrange_monomres(self):
        """
        Tests arrange_monomers()
        """

        # one multimer type
        multi = Multimer()
        composition=[3, 2, 1, 3]
        probability = None
        multi.initialize_arrangement(
            composition=composition, probability=probability)
        for ind in range(10):
            res = multi.arrange_monomers()
            #print res
            np_test.assert_equal(np.bincount(res), composition)

        # two multimer types
        multi = Multimer()
        composition=([3, 2, 1], [2, 2, 2])
        probability = (0.7, 0.3)
        multi.initialize_arrangement(
            composition=composition, probability=probability)
        for ind in range(10):
            #res = multi.arrange_monomers()
            print res
            np_test.assert_equal(
                ((np.bincount(res)== composition[0]).all() 
                 or (np.bincount(res)== composition[1]).all()), True) 
            
        # should also test probability of multimer types, somehow
        

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMultimer)
    unittest.TextTestRunner(verbosity=2).run(suite)
