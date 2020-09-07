"""
Class SerialEM contains methods for manipulations of meta-data (parameters) of 
a tomographic series acquired by SerialEM.

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id$
"""
from __future__ import unicode_literals
from builtins import zip
from builtins import object

__version__ = "$Revision$" 

import re
import os
import logging
import numpy as np

from pyto.grey.image import Image

class SerialEM(object):
    """ 
    """

    def __init__(
            self, dir_=None, mdoc=None, counte=None, 
            projection_dir=None, projection_suffix='_angle_%.1f.mrc'):
        """
        If arg dir_ is given, finds (multiple) mdoc files in dir_ and checks 
        that the corresponding projection stacks are there. All mdoc files in 
        (arg) dir_ are expected to belong to the same tilt series.

        Alternatively, if arg mdoc is given, just saves it as an attribute.

        Either dir_ or mdoc should be specified.

        Normally, pixel values in SerialEM series show the "real" counts
        multiplied by a factor (19.2 on MPI Titan2-K2). This factor
        should be given as arg counte. Attribute self.counte is set 
        directly from arg counte, but it can be also set externaly from
        FrameSeries.

        Arguments:
          - dir_: directory where mdoc and stacks corresponding to one tilt 
          series are located
          - mdoc: mdoc file path
          - counte: counts per electron for SerialEM projection stack
          - projection suffix: format of the suffix containing tilt
          angle that is added to the stack name to form file names for
          individual projections 

        Sets:
          - self.dir: arg dir_
          - self.mdoc_path: arg mdoc
          - self.mdocs: list of mdoc file names (without directory)
          - self.stacks: list of stack file names (without directory)
          - self.stack_paths: list of stack paths
          - self.counte: counts per pixel in serialem generated projection stack
          (different from the one in frames if gain correction was not done
          during tomo acquisition)
        """

        # check args
        if (dir_ is not None) and (mdoc is not None):
            raise ValueError("Either dir_ or mdoc argument should be given")

        # set dir
        self.dir = dir_
        self.mdoc_path = mdoc
        self.projection_dir = projection_dir
        self.projection_suffix = projection_suffix

        # set counts per electron (factor introduced by serialem)
        self.counte = counte

        # set mode
        multi_mode = False
        if mdoc is None:
            multi_mode = True 

        # additional settings if multiple files
        if multi_mode:

            # find mdoc files
            self.mdocs = [name for name in os.listdir(dir_) 
                          if name.endswith('mdoc')]

            # find projection stacks
            self.stacks = [name.rsplit('.', 1)[0] for name in self.mdocs]

            # set stack paths
            self.stack_paths = [
                os.path.join(self.dir, st_name) for st_name in self.stacks]

            # check they correspond to each other
            for name in self.stacks:
                path = os.path.join(self.dir, name)
                if not os.path.exists(path):
                    logging.warning(
                        "SerialEM stack " + path + " does not exist.")

    @property
    def projection_names(self):
        """
        Makes projection_name as by concatenating stack_name (without 
        extension) and tilt angle formatted using self.projection_suffix.
        """
        #format = '_angle_%.1f.mrc'
        vals = [name.rsplit('.', 1)[0] + (self.projection_suffix % angle) 
                for name, angle in zip(self.stack_names, self.tilt_angles)] 
        return np.asarray(vals)

    def parse_mdocs(self, sort=True):
        """
        Parses one or more mdoc file(s) that correspond to a single series and
        saves some of the parameters.

        All mdoc files in (arg) dir_ are expected to belong to the
        same tilt series.

        Attributes self.dir, self.mdocs and self.stacks need to be set before
        running this method. This is normaly done using arg dir_ in 
        intantiation of this instance.

        Elements of ndarray type variables that are set here correspond to 
        individual projections and are ordered in the order these projections 
        were acquired.

        Arguments:
          - sort: flag indicating if parameters should be sorted according to
          tilt angles

        Sets:
          - self.apixel: pixel size in A
          - self.z_values: (ndarray) projection positions in the stack 
          (read from the ZValue in mdoc file)
          - self.tilt_angles: (ndarray) tilt angle in deg
          - self.exposure_times: (ndarray) exposure times in s
          - self.orig_frame_names: (ndarray) names of projection image files
          (without directory path), or None if not found
          - self.stack_names: (ndarray) names of serial em generated projection 
          stack(s), for each projection
          - self.projection_names: (ndarray) projection file names of the 
          form: stack_name:z_value
          - self.projection_paths: (ndarray) has the form: projection
          file paths, obtained from self.projection_dir and 
          self.projection_names
          - the above ndarray attributes (z_values, tilt_angles, 
          exposure_times, orig_frame_names, stack_names and projection_names) 
          are ordered in the same way
        """

        # initialize
        pixel_list = []
        self.z_values = []
        #self.projection_name = []
        self.tilt_angles = []
        self.exposure_times = []
        self.stack_names = []
        self.orig_frame_names = []
 
        # parse mdocs
        for mdoc, stack in zip(self.mdocs, self.stacks):
            curr_path = os.path.join(self.dir, mdoc)
            curr_mdoc = self.__class__(mdoc=curr_path)
            curr_mdoc.parse_single_mdoc()
            #pix, zvals, angles, frame_names = \
            #    self.parse_single_mdoc(file_=curr_path)
            pixel_list.append(curr_mdoc.apixel)
            self.z_values.extend(curr_mdoc.z_values)
            self.tilt_angles.extend(curr_mdoc.tilt_angles)
            self.exposure_times.extend(curr_mdoc.exposure_times)
            self.stack_names.extend([stack] * len(curr_mdoc.z_values))
            self.orig_frame_names.extend(curr_mdoc.orig_frame_names)

        # check pixel
        if not (np.asarray(pixel_list) == pixel_list[0]).all():
            raise ValueError("Pixel sizes differ in different mdoc files")
        self.apixel = pixel_list[0]

        # check orig_frame_names
        if None in self.orig_frame_names:
            self.orig_frame_names is None

        # sort
        if sort:
            argsort = np.asarray(self.tilt_angles).argsort()
            self.z_values = np.asarray(self.z_values)[argsort]
            self.tilt_angles = np.asarray(self.tilt_angles)[argsort]
            self.exposure_times = np.asarray(self.exposure_times)[argsort]
            self.orig_frame_names = np.asarray(self.orig_frame_names)[argsort]
            self.stack_names = np.asarray(self.stack_names)[argsort]

        # make projection paths
        if self.projection_dir is not None:
            self.projection_paths = np.array([
                os.path.join(self.projection_dir, name)
                for name in self.projection_names])

    def parse_single_mdoc(self):
        """
        Parses a single mdoc file (generated by serialEM)

        Attribute self.mdoc_path need to be set. This is normaly done using 
        arg dir_ in intantiation of this instance.

        Sets attributes: 
          - self.apixel
          - self.z_value
          - self.tilt_angles
          - seld.exposure_times
          - self.orig_frame_names
        """

        # initialize
        mdoc_fd = open(self.mdoc_path)
        pixel_found = False
        z_values = []
        tilt_angles = []
        exposure_times = []
        orig_frame_names = []

        # read line by line
        for line in mdoc_fd:

            # pixel size
            if line.startswith('PixelSpacing'):
                line_split = line.split('=')
                current_pixel = float(line_split[-1])
                if not pixel_found:
                    pixel = current_pixel
                    pixel_found = True
                else:
                    if pixel != current_pixel:
                        raise ValueError("Pixel value changed in mdoc file")
                continue

            # z value
            if line.startswith('[ZValue'):
                zval = line.rsplit(']', 1)[0]
                zval = int(zval.split('=')[-1])
                z_values.append(zval)
                continue

            # tilt angle
            if line.startswith('TiltAngle'):
                angle = line.strip()
                angle = float((angle.split('='))[-1])
                tilt_angles.append(angle)
                continue

            # exposure time
            if line.startswith('ExposureTime'):
                exposure = line.strip()
                exposure = float((exposure.split('='))[-1])
                exposure_times.append(exposure)
                continue

            # original frames
            if line.startswith('SubFramePath'):
                line = line.strip()
                orig_frames_path = (line.split('='))[-1]
                orig_frame_n = (orig_frames_path.split('\\'))[-1]
                orig_frame_names.append(orig_frame_n)
                continue

        # checks
        if (len(z_values) != len(tilt_angles)):
            raise ValueError("Mdoc file was not parsed properly")
        if len(orig_frame_names) == 0:
            orig_frame_names = None
        elif len(z_values) != len(orig_frame_names):
            raise ValueError("Mdoc file was not parsed properly")

        # set attributes
        self.apixel = pixel
        self.z_values = z_values
        self.tilt_angles = tilt_angles
        self.exposure_times = exposure_times
        self.orig_frame_names = orig_frame_names

            
