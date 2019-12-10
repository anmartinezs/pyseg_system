"""
Contains class Phantom for making phantom images

Work in progress

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id: phantom.py 1527 2019-04-10 13:47:34Z vladan $
"""

__version__ = "$Revision: 1527 $"

import itertools
import logging
import numpy as np
import scipy as sp

from pyto.core.image import Image as Image
from pyto.particles.multimer import Multimer as Multimer
from pyto.geometry.rigid_3d import Rigid3D as Rigid3D

class Phantom(Image):
    """

    """

    def __init__(self, data=None):
        """
        """
        super(Phantom, self).__init__(data)
        
        #self.image = image
        #if (self.image is not None) and isinstance(self.image, np.ndarray):
        #    self.image = Image(self.image)

    @classmethod
    def transform_phantoms(
            cls, phantoms, tranforms=None, rotations=None, translations=None):
        """
        Abandoned
        """

        # check phantoms

        # generate transforms if needed

        # transform each phantom; smooth; save, write star



    @classmethod
    def make_multimer_fat(cls, monomers, symmetry, origin=None, order=1, binary=True):
        """
        Probably abandoned
        """

        # ToDo make random distribution of monomers

        # make symmetry transformation
        transfs = cls.make_symmetry_transforms(symmetry=symmetry, affine=True)

        # check monomers
        if isinstance(monomers, (list, tuple)):
            monomers_loc = monomers
            if len(monomers) != len(transfs):
                raise ValueError("Number of monomers has to match symmetry.")
        else:
            monomers_loc = [monomers] * len(transfs)
        if len(monomers_loc) == 0:
            raise ValueError("No monomers")
    
        # tranform
        final_array = np.zeros_like(monomers_loc[0].image.data) 
        for mono, trans in zip(monomers_loc, transfs):
            transf_array = trans.transformArray(
                array=mono.image.data, origin=mono.origin, order=order, 
                mode='constant')
            final_array = final_array + transf_array
                      
        # make binary if needed
        if binary:
            final_array[final_array>1] = 1

        # make and return new instance
        multi_obj = cls(final_array)
        return multi_obj
  
    def make_homomer(
            self, symmetry, origin=None, order=1, binary=True, threshold=1e-5):
        """
        Makes a homomer of a phantom given by this instance, by symmetrizing 
        (rotating) the phantom according to the given symmetry. The 
        homomer is returned as a new instance of this class.
        
        Arguments:
          - symmetry: symmetry, currently implemented Cn and Dn (such as
          ('C2', 'D3', ..., both capital and small letters can be used)
          - origin: (list or array of length 3) coordinates of the 
          rotation origin
          - order: interpolation spline order, default 1 (n-linear),
          passed directly to Affine.transformArray() and from there to
          sp.ndimage.map_coordinates()
          - binary: if True all elements of the muiltimer that are greater
          or equal to arg threshold are set to 1, otherwise overlapping 
          elements are added 
          - threshold: threshold used if arg binary is True
     
        Returns: instance of this class having attributes:
          - image: (pyto.core.Image) phantom image
          - origin: arg origin
        """

        # check origin
        if origin is None:
            origin =self.origin
                      
        # initialize
        multi = Multimer(symmetry=symmetry, monomers=[self])
        transfs = multi.make_symmetry_transforms(affine=True)
        final_array = np.zeros_like(self.data) 
                      
        # tranform
        for trans in transfs:
            transf_array = trans.transformArray(
                array=self.data, origin=origin, order=order, 
                mode='constant')
            final_array = final_array + transf_array
                      
        # make binary if needed
        if binary:
            final_array = np.where(final_array>=threshold, 1., 0)

        # make and return new instance
        multi_obj = self.__class__(final_array)
        multi_obj.origin = origin
        return multi_obj

    @classmethod
    def make_2box(
            cls, base_size, base_pos, shape, protrusion_size, protrusion_pos, 
            center=[0,0,0], origin=None, binary=True):
        """
        Makes phantom consisting of two 3D rectangles, here called base
        and protrusion.

        A phantom is first created by placing the two rectangles according 
        to the specified size and (lower-left corner) positions. Then the 
        phantom is translated in x y and z so that the voxel at the position  
        specified by arg center is moved to the positions specified by
        arg origin.

        Initial rectangles have value 1. Values of the overlapping array 
        elements (pixels) in the resulting phantom are determined by arg 
        binary. It it is False, the rectangles are added (the overlap is 2) 
        and if it is True the overlap (like the rest of the rectangles) 
        gets value 1. The background is 0 in both cases.

        Arguments:
          - base_size, protrusion_size: (list or array of length 3) x, y 
          and z sizes of the base and protrusion rectangles 
          - base_pos, protrusion_pos: (list or array of length 3) x, y 
          and z positions of the minimal (lower-left) corner of the base 
          and protrusion rectangles 
          - shape: (list or array of length 3) shape of the phantom image
          - center: (list or array of length 3) special position on the 
          phantom
          - origin: (list or array of length 3) position to which the center 
          voxel is translated
          - binary: if True all elements of the phantom are set to 1, 
          otherwise overlapping elements are added 
     
        Returns: instance of this class having attributes:
          - image: (pyto.core.Image) phantom image
          - origin: arg origin
        """

        # parse arguments
        #base_x, base_y, base_z = base_size
        #prot_x, prot_y, prot_z = protrusion_size

        # find shift 
        center = np.asarray(center)
        if origin is None:
            origin = (np.asarray(shape) / 2).astype(int)
        else:
            origin = np.asarray(origin)
        shift = origin - center

        # define g-phantom using slices
        base_slices = [
            slice(pos, pos+size) for size, pos in zip(base_size, base_pos)] 
        prot_slices = [
            slice(pos, pos+size) for size, pos 
            in zip(protrusion_size, protrusion_pos)]

  
        # move phantom definition so that center is at origin
        base_slices = [
            slice(sl.start+shi, sl.stop+shi) 
            for sl, shi in zip(base_slices, shift)]
        prot_slices = [
            slice(sl.start+shi, sl.stop+shi) 
            for sl, shi in zip(prot_slices, shift)]

        # make image
        im_array = np.zeros(shape=shape)
        im_array[base_slices] = 1
        if binary:
            im_array[prot_slices] = 1
        else:
            im_array[prot_slices] += 1
        
        # make instance
        phantom = cls(im_array)
        phantom.origin = origin

        return phantom
