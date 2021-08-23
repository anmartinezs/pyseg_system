"""
Class Series contains methods for manipulations of a series of images.

Meant for but limited to a tomographic series.

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id$
"""
from __future__ import unicode_literals
from __future__ import division
from builtins import zip
from builtins import str
from builtins import range
from builtins import object
#from past.utils import old_div

__version__ = "$Revision$"

import re
import os
import os.path
import logging
import numpy

from pyto.io.image_io import ImageIO
from pyto.grey.image import Image

class Series(object):
    """
    Methods for manipulations of a series of images. 

    Meant for but not limited to a tomographic series. Images can be saved in 
    separate files or as a single stack.

    Important methods:
      - self.sort: sorts series, fixes file headers and limits the data
      (only for separate images)
      - self.fix: fix headers (only for separate images)
      - self.getDose: calculated the total (series) dose 
      - self.images: generator that yields names of all files of a series
      (only for separate images)
    
    Common usage:

      myseries = Series(files='dir/base.em')
      myseries.sort(out='new_directory')
      myseries.fix(mode='polara_fei-tomo', microscope='polara-1_01-09')
    """

    def __init__(self, path=None, mode=None, stack=False, tilt_file=None):
        """
        Sets path and mode that determine the series.

        A series contains all files that match path. Path cosists of a 
        directory and a file name, where the file name is taken to be a 
        regular expression.

        Mode determines how the pattern match is performed. The options are:
          - search: (like re.search) pattern somewhere in the target string
          - match: (like re.match) target string begins with the pattern
          - exact: whole target string is mathed (both the beginning and the
          end of the file name are anchored)

        Arguments:
          - paths: full file path in the form dir/file or just file, where file 
          is a regular expression 
          - mode: pattern matching mode
          - stack: flag indicating if the series is in a single stack
          - tilt_file: name of the SerialEM type tilt angles file

        Sets:
          - self.path
          - self.mode
          - self.stack
          - self.tiltAngles: tilt angles read from tilt_file
        """

        # set attributes
        self.path = path
        self.mode = mode
        self.stack = stack

        if stack and (tilt_file is not None): 
            self.tiltAngles = self.readTiltAngles(file_=tilt_file)
        else:
            self.tiltAngles = None

        # initialize out (write-related) attributes
        self.setOutMode()

    ######################################################
    # 
    # Methods that manipulte whole series.
    #
    ######################################################

    def sort(self, seq=None, out=None, pad=0, start=1, fix_mode=None, 
             microscope=None, limit=None, limit_mode='std', size=5, 
             byte_order=None, test=False):
        """
        Sorts series, fixes image headers, corrects the data and writes the
        sorted and corrected images. 

        A series is sorted by tilt angle that is read from each image, or by the
        elements of seq (not necessarily angles) if given.

        Sorted images names have the form:
          directory/name + number + extension.
        where out is: directory/name_.extension and numbers start with start
        argument and are padded with pad (argument) number of zeros (see
        self.images).

        Argument fix_mode detemines how the headers are fixed:
          - None: no fixing
          - 'polara_fei-tomo': for series obtained on polara with the FEI tomo
          software
          - 'krios_fei-tomo': for series obtained on krios with the FEI tomo
          software
          - 'cm300': for series from CM300
        (see pyto.io.image_io). 

        If fix_mode is 'polara_fei-tomo', then arg microscope has to be 
        specified. The allowed values are specified in microscope_db.py. 
        Currently (r564) these are: 'polara-1_01-07', 'polara-1_01-09' and 
        'polara-2_01-09'.

        If fix_mode is None, microscope does nor need to be specified.
        Series recorded by SerialEM typically do not need fixing.

        Works for individual images only (not for stacks).

        Arguments:
          - seq: if specified this sequence used for sorting projection,
          otherwise tilt angles are used
          - out: sorted series path
          - pad: number of digits of a projection number in the sorted 
          series file name
          - start: start number for sorted series file names 
          - fix_mode: determined if and how headers are fixed
          - microscope: microscope type, see pyto.io.microscope_db
          - limit_mode: 'abs' or 'std'
          - limit: absolute greyscale limit if limit_mode is 'abs', or the 
          number of stds above and below the mean 
          - size: size of the subarray used to find the replacement value 
          for a voxel that's out of the limits 
          - byte_order: '<' or '>', default is the byte order of the machine
          - test: if True all steps are done except writing corrected images
        
        """

        # shortcuts
        path = self.path
        match_mode = self.mode

        # make tilt angles - file names dictionary
        # ToDo: use self.sortPath() instead
        in_paths = []
        images_iter = self.images(mode=match_mode)
        if seq is None:
            seq = []

            # get tilt angles from file headers
            for in_path in images_iter:
                image = ImageIO(file=in_path)
                image.readHeader()
                seq.append(image.tiltAngle)
                in_paths.append(in_path)
                image.file_.close()

        else:

            # use seq to sort
            for in_path in images_iter:
                in_paths.append(in_path)

        # sort (note: dictionary angle:in_path fails for multiple files with
        # same angles)
        seq_arr = numpy.array(seq)
        in_paths_arr = numpy.array(in_paths)
        sort_ind = seq_arr.argsort()
        sorted_seq = seq_arr[sort_ind]
        sorted_in_paths = in_paths_arr[sort_ind]

        # parse out and make out directory if it doesn't exists
        if out is not None:
            out_dir, out_base_pat = os.path.split(out)
            if out_dir == '':
                out_dir = '.'
            if (not test) and (not os.path.isdir(out_dir)):
                os.makedirs(out_dir)
        else:
            out_path = None

        # initialize file counter
        if start is None:
            ind = None
        else:
            ind = start - 1

        # loop over sorted in files
        for (item, in_path) in zip(sorted_seq, sorted_in_paths):
            if ind is not None: ind += 1

            # parse in file path
            in_dir, in_base = os.path.split(in_path)

            if out is not None:

                # make out file path
                out_base = self.convertBase(in_base=in_base,
                                     out_base=out_base_pat, pad=pad, index=ind)
                out_path = os.path.join(out_dir, out_base)

                # read files
                im_io = ImageIO(file=in_path)
                im_io.read()

                # log
                logging.info("%5.1f: %s -> %s  %6.1f %6.1f" \
                                 % (item, in_path, out_path, \
                                        im_io.data.mean(), im_io.data.std()))

                # fix
                if fix_mode is not None:
                    im_io.fix(mode=fix_mode, microscope=microscope)

                # limit
                if limit is not None:
                    image = Image(im_io.data)
                    image.limit(limit=limit, mode=limit_mode, size=size)

                # write fixed file
                if not test:
                    if byte_order is None:
                        byte_order = im_io.machineByteOrder
                    im_io.write(file=out_path, byteOrder=byte_order,
                                data=image.data)

            else:

                logging.info(" " + str(item) + ": " + in_path + " -> "\
                             + str(out_path))

    def fix(self, mode=None, microscope=None, out=None, pad=0, start=1):
        """
        Fixes wrong data in headers.

        Mode determines which values are fixed. Currently defined modes are:
          - 'polara_fei-tomo': images obtained on Polara (at MPI of 
          Biochemistry) using FEI tomography package and saved in EM 
          format.
          - 'krios_fei-tomo': images obtained on Krios (at MPI of 
          Biochemistry) using FEI tomography package and saved in EM 
          format.
          - 'cm300': images from cm300 in EM format
          - None: doesn't do anyhing
          
        If mode is 'polara_fei-tomo', then arg microscope has to be specified. 
        The allowed values are specified in microscope_db.py. Currently (r564)
        these are: 'polara-1_01-07', 'polara-1_01-09' and 'polara-2_01-09'.

        Works for individual images only (not for stacks), because stacks
        are typically recorded by SerialEM and do not need fixing.

        Arguments:
          - out: directory where the fixed images are written. The fixed and
          the original images have the same base names.
          - mode: fix mode
        """

        # parse arguments
        #if path is None: path = self.path
        #if mode is None: mode = self.mode

        # check for out, pad and start also?
        
        # make out directory if it doesn't exists
        if (out is not None) and (not os.path.isdir(out)):
            os.makedirs(out)

        # loop over images
        images_iter = self.images(out=out, pad=pad, start=start)
        for (in_path, out_path) in images_iter:

            # read image file
            image = ImageIO(file=in_path)
            image.read()

            # fix
            image.fix(mode=mode, microscope=microscope)

            # write fixed files
            image.write(file=out_path)

    def getDose(self, conversion=1, mode='pixel_size', projection_dose=False):
        """
        Calculates dose for the whole series (in e/A^2) and the average counts
        per pixel..

        Logs average, std, min and max counts for each image to INFO.

        Mode determines how the pixels are converted to A^2:
        - 'pixel_size': use pixelsize from the image header (default)
        - 'cm300_fix': use only for cm300 series with headers that were not
        corrected

        Arguments:
          - conversion: number of electrons per pixel (if known microscope use
          pyto.io.microscope_db.conversion[microscope]).
          - mode: 'pixel_size' should be used, other moders were used in
          special cases
          - projection_dose: flag indicating if dose [e/A^2] for each 
          projection is returned

        Returns total_dose, mean_count:
          - total dose in e/A^2
          - mean count in counts/pixel
          - projection_dose (if projection_dose=True) 2D ndarray containing
          tilt angle and projection dose for all projections 
          (shape n_projections x 2)
        """

        # set images to z positions if projections are in a stack, or
        # to projection file names if individual projection files
        if self.stack:
            image_stack = Image.read(file=self.path, memmap=True)
            images = numpy.array(list(range(image_stack.data.shape[2])))
            images = images[numpy.argsort(self.tiltAngles)]
        else:
            images = list(self.images())
            images = self.sortPaths(paths=images, mode='tilt_angle')

        tot_dose = 0
        tot_count = 0
        num = 0.
        proj_dose = []
        for fi in images:

            # read single projection file
            if self.stack:
                image = Image(data=image_stack.data[:,:,fi])
                if isinstance(image_stack.pixelsize, (numpy.ndarray, list)):
                    image.pixelsize = image_stack.pixelsize[0]
                else:
                    image.pixelsize = image_stack.pixelsize
                image.tiltAngle = self.tiltAngles[fi]
            else:
                image = ImageIO(file=fi)
                image.read()

            # calculate
            mean = image.data.mean()
            tot_count += mean
            num += 1
            if mode == 'pixel_size':
                loc_dose = mean / (conversion * image.pixelsize**2)
            elif (mode == 'cm300_fix') or (image.pixelsize == 2.048):
                ccd_pixelsize = 30000
                pixelsize = ccd_pixelsize / image.magnification
                loc_dose = mean / (conversion * pixelsize**2)
            else:
                raise ValueError("Sorry, dose calculation mode: " + mode + 
                                 " is not recognized.")  
            tot_dose += loc_dose

            # save and print stats for an individual image
            proj_dose.append([image.tiltAngle, loc_dose/100.])
            logging.info('%s, (%5.1f): mean=%6.1f, std=%6.1f, min=%d, max=%d' 
                         % (fi, image.tiltAngle, mean, image.data.std(), 
                            image.data.min(), image.data.max()))

        # mean counts
        mean_count = tot_count / num

        # total dose per A^2
        tot_dose = tot_dose / 100

        # print total values
        logging.info("Mean count per pixel: %6.1f" % mean_count)
        logging.info("Total electron dose per A^2: %6.1f" % tot_dose) 
        
        if projection_dose:
            return tot_dose, mean_count, numpy.asarray(proj_dose)
        else:
            return tot_dose, mean_count


    ######################################################
    # 
    # Other methods
    #
    ######################################################

    def images(self, path=None, mode=None, out=None, pad=0, start=1):
        """
        Generator that (when instantiated) yields paths for files that match
        path. If out is not None, a path for the corresponding output file is
        also returned for each matched (input) file.

        Mode determines how the pattern match is performed. The options are:
          - search: (like re.search)
          - match: (like re.match)
          - exact: both the beginning and the end of the file name are anchored

        Each path is separated into: directory, clean root, number and
        extension. For example, mydir/myfile_002.em gives: 'mydir', 'myfile_',
        '002', '.em'. If not given in argument out, directory file and
        extension are taken from a corresponding (mached, or input) file.
        Output files are numbered starting from start and the pading
        determined if leading zeros are present in these numbers. Values None
        for pad and start mean that the form and the number of the input file
        is retained.

        Works for individual images only (not for stacks).

        For example:

          images(path='indir/infile', out=outdir/outfile, pad=2, start=1)

        yields the following input and output files:

          indir/infile_000.em, outdir/outfile_01.em
          indir/infile_001.em, outdir/outfile_02.em
          ...
        
        Arguments:
          - path: pattern that matches input files
          - mode: determines how the file names are matched
          - out: output path
          - pad: number of digits (paded by zeros if needed) of the number
          pard of out file names
          - start: output files are number starting with this argument

        Yields:
          - in_path (if out is None)
          - in_path, out_path (if out is not None)
        """

        # parse arguments
        if path is None: path = self.path
        if mode is None: mode = self.mode

        # check for out, pad and start also?
        
        # extract directory and make re for base
        dir_, base_re = self.makeReg(path=path, mode=mode)

        #  split out
        if out is not None:
            out_dir, out_base = os.path.split(out)

        # find all files
        if start is None:
            ind = None
        else:
            ind = start - 1
        for fi in os.listdir(dir_):

            # find files that match fi
            if base_re.search(fi) is not None:
                if ind is not None: ind += 1

                if out is None:

                    # yield only in file path
                    yield os.path.join(dir_, fi)

                else:

                    # make output file base from the actual input base 
                    new_base = self.convertBase(in_base=fi, out_base=out_base,
                                                pad=pad, index=ind)

                    # yield both in and out file paths
                    yield (os.path.join(dir_, fi), 
                           os.path.join(out_dir, new_base)) 

    def sortPaths(self, paths, mode='num', sequence=None):
        """
        Sorts list of file path and returns it.

        If arg mode is 'num', paths are sorted by series number, which are 
        expected to be between the rightmost '_' and '.' in the file names.

        If arg mode is 'tilt_angle', paths are sorted by tilt angle. 

        If arg mode is 'sequence', paths are sorted by arg sequence that has
        to have the same length as paths.

        Arguments:
          - paths: list of file paths (with or without directory)
          - mode: sorting mode, 'num', 'tilt_angle', or 'sequence'
          - sequence: sequence of values corresponding to paths

        Returns sorted list of paths. 
        """

        if mode == 'tilt_angle':

            # get tilt angles from file headers
            unsorted = []
            for file_ in paths:
                image = ImageIO(file=file_)
                image.readHeader()
                unsorted.append(image.tiltAngle)

        elif mode == 'num':

            # get file numbers
            unsorted = [int(self.splitBase(base=file_)[1]) for file_ in paths]

        elif mode == 'sequence':

            unsorted = sequence

        else:
            raise ValueError('Mode ', mode, " not understood. Allowed values ",
                             "are 'num', 'tilt_angle' and 'sequence'.")

        # sort file numbers and paths
        unsorted = numpy.array(unsorted)
        sort_ind = unsorted.argsort()
        sorted_paths = [paths[ind] for ind in sort_ind]

        return sorted_paths

    def makeReg(self, path=None, mode=None):
        """
        """
          
        # parse path (remove?)
        if path is None:
            path = self.path
        if mode is None:
            mode = self.mode        

        # find directory and base 
        (dir_, base) = os.path.split(path)
        if dir_ == '': dir_ = '.'
        if base == '': base = '.*'

        # add anchors at the beginning and at the end of base if needed 
        if mode == 'exact':
            base = '^' + base + '$'
        elif mode == 'match':
            base = '^' + base

        # return 
        return dir_, re.compile(base)

    def convertBase(self, in_base, out_base, index, pad=0):
        """

        """
        
        #  parse out_base
        out_clean_root, out_num, out_ext = self.splitBase(out_base)

        # if output base or extension is not given use input           
        in_clean_root, in_num, in_ext = self.splitBase(in_base)
        if out_clean_root == '':
            out_clean_root = in_clean_root
        if out_ext == '':
            out_ext = in_ext

        # prepare num part
        if index is None:
            index = int(in_num)
        if pad is None:
            out_num = in_num
        else:
            out_num = ('%0' + str(pad) + 'd') % index

        # assemble out base
        new_out_base = out_clean_root + out_num + out_ext
        return new_out_base

    def splitBase(self, base):
        """
        Splits base of a file name into three parts and returns them. 

        The last '_' before the extension separates the first two parts, while
        the last '.' separates the last two. The idea is that the second part
        is a number. 

        For example:

          > series.splitBase('neu-2_gfp_23.mrc') 
          > 'neu-2_gfp_', '23', '.mrc'

        Argument:
          - base: file base, that is a file name with of without directory/ies

        Returns tuple of three strings
        
        """

        # sanity check
        if (base is None) or (base == ''):
            return '', None, ''

        # split base to root and extension
        root, ext = os.path.splitext(base)

        # split root
        num_pos = root.rfind('_') + 1
        if num_pos > 0:
            clean_root = root[:num_pos]
            num = root[num_pos:]
        else:
            clean_root = base
            num = None

        return clean_root, num, ext

    def setOutMode(self, mode=None, start=1):
        """
        Sets attributes
        """

        self.outMode = mode
        self.outStart = start

    @classmethod
    def readTiltAngles(cls, file_):
        """
        Reads angles from a SerialEm type tilt angles file (one angle per line,
        ordered like the corresponding projections in a stack).

        Argument:
          - file_: tilt file name

        Returns ndarray of tilt angles
        """
        angles = numpy.loadtxt(file_)
        return angles

        
