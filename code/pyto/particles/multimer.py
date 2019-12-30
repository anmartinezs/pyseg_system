"""
Contains class Multimer for manupulation of multimers

Work in progress

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id: phantom.py 1449 2017-04-30 10:03:37Z vladan $
"""

__version__ = "$Revision: 1449 $"

import itertools
import logging
import numpy as np
import scipy as sp

from pyto.core.image import Image as Image
from pyto.geometry.rigid_3d import Rigid3D as Rigid3D

class Multimer(object):
    """

    """

    def __init__(self, symmetry=None, monomers=None):
        """
        """

        self.symmetry = symmetry
        self.monomers = monomers
        #if (symmetry is not None) and (monomers is not None):
        #    self.symmetrize_monomers()

    def sym_rotate(self, monomers=None, symmetry=None, order=1):
        """
        Rotates all monomers according to all rotations contained in a
        given symmetry group. 

        Both the momomers and the symmetry group can be specified as
        arguments or as already defined attributes with the same names.

        The rotated monomers are saved in a table and can be accessed 
        using get_sym_rotated() method.

        Arguments:
          - monomers: list of monomers (Image objects)
          - symmetry: symmetry group, currently implemented Cn and Dn (such
          as ('C2', 'D3', ..., both capital and small letters can be used)
          - order: interpolation spline order, default 1 (n-linear),
          passed directly to Affine.transformArray() and from there to
          sp.ndimage.map_coordinates()

        Sets attributes:
          - self.monomers and self.symmetry to the given argument
          values if not None
          - self.sym_rotations: rotations as defined by 
          self.make_symmetry_transforms
        """

        # parse arguments
        if symmetry is not None:
            self.symmetry = symmetry
        if monomers is not None:
            self.monomers = monomers

        # make transformations
        sym_rotations = self.make_symmetry_transforms(affine=True)
        self.sym_rotations = sym_rotations
       
        # make table of all symmetry rotations of all monomers
        self.initialize_sym_rotated()
        for mono_ind, mono in enumerate(self.monomers):
            for transf_ind, trans in enumerate(sym_rotations):
                curr_array = trans.transformArray(
                    array=mono.data, origin=mono.origin, order=order, 
                    mode='constant')
                self.set_sym_rotated(
                    mono_index=mono_ind, sym_index=transf_ind, 
                    array=curr_array)
       
    def initialize_sym_rotated(self):
        """
        Initializes the object to hold all roated monomers.
        """
        self.sym_rotated = np.empty(
            (len(self.monomers), len(self.sym_rotations)), dtype=object)
         
    def set_sym_rotated(self, mono_index, sym_index, array):
        """
        Sets a rotated monomer image for a specified monomer and rotation.

        Arguments:
          - mono_index: index of the monomer in the self.monomers
          - sym_index: index of the rotation in the self.sym_rotations
          - array (ndarray): rotated monimer image

        Returns (ndarray): rotated monomer
        """
        self.sym_rotated[mono_index, sym_index] = array

    def get_sym_rotated(self, mono_index, sym_index):
        """
        Returns a rotated monomer image for a specified monomer and rotation.

        Arguments:
          - mono_index: index of the monomer in the self.monomers
          - sym_index: index of the rotation in the self.sym_rotations
          
        Return: (ndarray) rotated monomer image
        """
        return self.sym_rotated[mono_index, sym_index]

    def make_symmetry_transforms(
            self, symmetry=None, affine=True, degree=False):
        """
        Returns a list of 3D (rigid) rotations corresponding to the specified
        symmetry.
        
        Arg affine determines whther the 3D rigid rotations returned are
        affile.Rigid3D objects or lists of Euler angles (phi, theta and
        psi). The angles can be in radians or degerees, depending on 
        arg degree.

        In case of 'Cn' symmeties, the returned rotations are ordered by
        increasing phi, from 0 to 2 (n-1) pi / n. For 'Dn' symmetries,
        first the symmetries for theta = 0 are listed (in the order of 
        increasing phi) and then the theta = 180 symmetries (in the same 
        order). The order does not depend on the type of rotations, that
        is value of arg affine. 

        Arguments:
          - symmetry: symmetry group, currently implemented Cn and Dn (such as
          ('C2', 'D3', ..., both capital and small letters can be used)
          - affine: flag the determies the type of the returned rotations 
          - degree: flag indicating if the returned angles are in degrees
          or in radians (only if affine is False)

        Returns: list of rotations
        """
 
        if symmetry is not None:
            self.symmetry = symmetry

        # set how angles will be computed
        if affine or not degree:
            pi = np.pi
        else:
            pi = 180.

        # make angles list
        sym_n = int(self.symmetry[1])
        sym_phi = 2. * pi * np.arange(sym_n) / float(sym_n)
        if (self.symmetry[0] == 'c') or (self.symmetry[0] == 'C'):
            angle_iter = [[phi, 0, 0] for phi in sym_phi]

        elif (self.symmetry[0] == 'd') or (self.symmetry[0] == 'D'):   
            sym_theta = [0, pi]
            phi_theta_iter = itertools.product(sym_theta, sym_phi, [0])
            angle_iter = [
                [phi_theta[1], phi_theta[0], 0] for phi_theta in phi_theta_iter]

        else:
            raise ValueError("Sorry, only C and D symmetries are implemented")

        # make Rigid3D transformation objects if needed
        if affine:
            angle_iter = [Rigid3D(
                q=Rigid3D.make_r_euler(angles=angles, mode='zxz_ex_active')) 
             for angles in angle_iter]

        return angle_iter

    def initialize_arrangement(self, composition, probability):
        """
        Defines the subunit compositions for one or more multimer types. 
        Attributes set here are necessary to generate rundom multimers
        by self.arrange_monomers().

        Arg composition defines the numbers of monomers for one or more 
        multimer types and arg probability defines the probability of the 
        monomer types.

        For example:
          
          composition = ([3, 2, 1], [2, 2, 2])
          probability = (0.7, 0.3)

        means that there are two multimer types which occure with 
        probabilities 0.7 and 0.3. Both multimers contain three different
        monomer types and are composed of 6 monomers. The first multimer 
        contains 3 monomers of the first monomer type, 2 of the second and 
        1 of the third. The second multimer type contains 2 monomers of each 
        three monomer types. The resulting attribute:
    
          self.initial_arrangement = (
            [0, 0, 0, 1, 1, 2], 
            [0, 0, 1, 1, 2, 2])

        where monomer types are labeled by 0 - 2.

        Arguments:
          - composition: list or tuple of lists, where each list corresponds
          to one multimer type and the elements of the list show the 
          number of monomers in this multimer type
          - probability: None or tuple of ints, where each int corresponds
          to one multimer and shows the probabilities of the sorresponding 
        multimer types

        Sets attributes:
          - self.composition: like arg composition, but wrapped in a tuple 
          in case arg composition is a list
          - self.multimer_type_probability: like arg probability but
          normalized to 1 and wrapped in a tuple in case arg probability 
          is a list
          - self.initial_arrangement: list of lists, where each sublist 
          shows a generic arrangement of subunits for a multimer type
        """

        # adjust for multiple multimer types
        if not isinstance(composition, tuple):
            composition = (composition, )
            probability = (1,)

        # make sure probability sums to 1
        if np.sum(probability) != 1:
            probability = np.asarray(probability) / np.sum(probability)

        # save attributes
        self.composition = composition
        self.multimer_type_probability = probability

        # initial arrangements for all multimer types
        self.initial_arrangement = []
        for comp in composition:

            # make initial arrangement for one multimer type
            initial = []
            for mono_ind, frequency in enumerate(comp):
                initial = initial + [mono_ind] * frequency

            # add current arrangement
            self.initial_arrangement.append(initial)

    def arrange_monomers(self):
        """
        Randomly pick one of multimer types from self.composition according
        to probabilitis self.multimer_type_probability and then makes
        a random permutation of monomers (of the picked multimer).

        Method self.initialize_arrangement() has to be executed before
        calling this method.

        Returns: (list) monomer arrangement where elements are 
        integers starting from 0 that denote monomers.
        """

        # chose multimer type
        multi_ind = np.random.choice(
            len(self.multimer_type_probability), size=1, 
            p=self.multimer_type_probability) 

        # arrange monomers 
        multimer = np.random.permutation(self.initial_arrangement[multi_ind])

        return multimer
