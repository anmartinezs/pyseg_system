"""
Basic greyscale analysis of aligned frames and original (non-aligned)
frame stacks.

Contains:
  - analyze_frame(): function that displays the analysis results
  - class FrameSeries: getting and analysing aligned and original frames

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id: frame_series.py 1534 2019-04-10 15:06:28Z vladan $
"""

__version__ = "$Revision: 1534 $"

import re
import os
import logging
import warnings
import numpy as np
import matplotlib.pyplot as plt

from serial_em import SerialEM
from pyto.grey.image import Image


def analyze_frames(
        serialem_dir, serialem_projection_dir=None, align_program=None, 
        orig_frames_dir=None,  aligned_frames_dir=None, aligned_prefix=None, 
        ucsf_translations_path=None, apixel=None, counte=None, 
        gain_corr_acquisition=True, 
        print_stats=True,  print_summary_stats=True, plot_frames=True, 
        plot_projections=True, bins=range(200), names_only=False):
    """
    Calculates basic pixel value stats for all aligned frames (projection)
    belonging to a tomo series, as well as for non-aligned (original)
    frames. Also prints total tomo series stats.

    The behavior depends on the arguments specified:

    At least arg serialem_dir has to be specified. In this case counts per 
    pixel are calculated for the projections of the SerialEM generated stack. 
    If arg counte is specified, counts are converted to electrons.

    If arg aligned_frames_dir is specified, counts per pixel are calculated for
    aligned frames (requires arg align_prefix). If arg counte is specified, 
    counts are converted to electrons. Aligned frames files have to have the 
    same name as the corresponding frame stacks prefixed with arg 
    aligned_prefix, except that the extension can be the same or mrc (useful 
    in case frame stacks are saved as tifs and aligned frames as mrcs.

    If fs.counte != fs.serialem.counte, as it happens when 
    gain_corr_acquisition is False, counts for serialem projections are 
    multiplied by  fs.counte / fs.serialem.counte before they are plotted.
    This brings aligned frames and serialem projection counts to the same 
    scale.
    
    If arg frames_dir is specified, counts per pixel are calculated for
    (non-aligned) frames and converted to electrons.

    If arg align_program is specified, frame alignment shifts (translations)
    are shown.

    If arg serialem_projection_dir is not None, projections from serialem
    stack are written as separate files in mrc format.

    Intended for use in an environment such as Jupyter notebook.

    Uses FrameSeries.projections() to get images and calculate stats. Check
    that method for the required file and directory name conventions. 

    Arguments:
      - serialem_dir: directory where serial em mdoc and stacks are
      - serialem_projection_dir: directory where individual projections
      from serialem are writen, on None if these should not be writen 
      - orig_frames_dir: directory where original (unaligned) frames are
      - aligned_frames_dir: directory where aligned frames are located
      - align_program: alignment program used ('dm', 'dimi' or 'ucsf')
      - aligned_prefix: prefix added to (unaligned) frames to make aligned
      frames file names
      - ucsf_translations_path: file path for the ucsf alignment log file
      (sent to stdout by ucsf alignment program)
      - apixel: pixel size in A
      - counte: number of counts per electron, None to use the value present
      in frame stack mrc files 
      - gain_corr_acquisition: flag indicating whether gain correction was 
      done during tomo acquisition
      - print_stats: flag indicating whether stats for individual projections 
      should be printed
      - print_summary_stats: flag indicating whether total series stats
      should be printed
      - bins: bind for counts histogram plots
      - plot_frames: flag indicationg whether counts histograms for individual
      frames should be plotted
      - plot_projections: flag indicating whether counts histograms for 
      projections (aligned frames and serialem projections)
      - names_only: if True only file names are shown, stats are not calculated
      not plotted, for testing
    """

    # initialize and make aligned frames (projections) iterator
    fs = FrameSeries(
        serialem_dir=serialem_dir, orig_frames_dir=orig_frames_dir, 
        serialem_projection_dir=serialem_projection_dir,
        aligned_frames_dir=aligned_frames_dir, align=align_program, 
        aligned_prefix=aligned_prefix, 
        ucsf_translations_path=ucsf_translations_path, counte=counte,
        gain_corr_acquisition=gain_corr_acquisition)
    projection_iter = fs.projections(
        serialem_stack=True, translations=True, names_only=names_only, 
        projections=None)

    # print table header
    if (print_stats or print_summary_stats) and not names_only:
        print
        print('                                  mean   std   min    max   mean    mean   ')
        print('                                  c/pix  c/pix c/pix  c/pix e/A^2 e/(pix s)')

    # loop over aligned frames (projections) 
    for proj in projection_iter:

        # print tilt angle
        if print_stats:
            print("Tilt angle: %6.2f" % proj['tilt_angle'])

        # get images
        original = proj.get('original', None)
        original_flat = proj.get('original_flat', None)
        aligned = proj.get('aligned', None)
        translations = proj.get('translations', None)
        serialem = proj.get('serialem', None)

        # write serialem projections 
        if not names_only:
            if serialem_projection_dir is not None:
                try:
                    os.makedirs(serialem_projection_dir)
                except OSError: pass
                serialem.write(
                    file=proj['serialem_projection'], pixel=fs.apixel)

        # plot, print data
        if not names_only:

            # plot count histogram count for frames of this projection
            if plot_frames and (original is not None): 
                plt.figure()
                plt.hist(original.data.flatten(), bins=bins, label='frames')
                plt.yscale('log')
                plt.xlabel('Counts')
                plt.ylabel('N pixels')
                plt.legend()
                plt.figure()

            # plot count histogram for this projection
            if plot_projections and (original_flat is not None): 
                plt.hist(
                    original_flat.data.flatten(), bins=bins, 
                    label='flat frames')
            if plot_projections and (aligned is not None): 
                plt.hist(
                    aligned.data.flatten(), bins=bins, label='aligned frames')
            if plot_projections and (serialem is not None): 
                if fs.counte == fs.serialem.counte:
                    serialem_adjust = 1
                else:
                    serialem_adjust = fs.counte / fs.serialem.counte
                plt.hist(
                    serialem_adjust * serialem.data.flatten(), bins=bins, 
                    label='serialem')
            if plot_frames or plot_projections:
                #plt.axis([plt.axis()[0], plt.axis()[1], 0, 10**7])
                plt.xlabel('Counts')
                plt.ylabel('N pixels')
                plt.legend()
                plt.show()

            # check if counts per electron known
            if (original_flat is not None) or (counte is not None):
                counte_known = True
            else:
                counte_known = False

            # print basic greyscale stats
            if print_stats and (original is not None): 
                print(
                    'Frames        %s: %5.1f  %5.1f  %5.1f  %6.1f  %5.3f   %5.2f  ' % 
                    (proj['original_name'], 
                     original.mean, original.std, original_min, original_max, 
                     original.mean_ea, original.mean/proj['exposure_time']))
            if print_stats and (original_flat is not None): 
                print(
                    'Flat frames   %s: %5.1f  %5.1f %5.1f  %6.1f   %5.3f' % 
                    (proj['original_name'], original_flat.mean, 
                     original_flat.std, original_flat.min, original_flat_max, 
                     original_flat.mean_ea))
            if print_stats and (aligned is not None): 
                if counte_known: 
                    print('Aligned       %s: %5.1f  %5.1f  %5.1f %6.1f %5.3f' % 
                          (proj['aligned_name'], aligned.mean, 
                           aligned.std, aligned.min, aligned.max,
                           aligned.mean_ea))
                else:
                    print(
                        'Aligned       %s: %5.1f  %5.1f %5.1f %6.1f' % 
                        (proj['aligned_name'], aligned.mean, aligned.std, 
                         aligned.min, aligned.max))
            if print_stats and (serialem is not None): 
                if counte_known: 
                    print(
                        'SerialEM      %s: %5.1f  %5.1f  %5.1f %6.1f %5.3f' % 
                        (proj['serialem_name'], serialem.mean, 
                         serialem.std, serialem.min, serialem.max, 
                         serialem.mean_ea))
                else:
                    print(
                        'SerialEM      %s: %5.1f  %5.1f %5.1f %6.1f' % 
                        (proj['serialem_name'], serialem.mean, serialem.std, 
                         serialem.min, serialem.max))

            # print translations
            if translations is not None:
                print("Frame translations: ")
                for trans in translations:
                    print("\t %7.3f  %7.3f" % (trans[0], trans[1]))
            if print_stats: print

        else:

            # only print file names
            if  print_stats and (original_flat is not None): 
                print('\t Frames        %s:' % proj['original_name'])
            if  print_stats and (aligned_frames_dir is not None): 
                print('\t Aligned       %s:' % proj['aligned_name'])
            if print_stats: print(
                    '\t SerialEM      %s:' % proj['serialem_name'])

    # print summary stats
    if print_summary_stats and not names_only:
        if original_flat is not None: 
            print(
                'Total original: %6.1f c/pix = %6.2f e/pix = %6.2f e/A^2' % 
                (fs.original_total, fs.original_total / fs.counte, 
                 fs.original_total_ea))
        if aligned_frames_dir is not None: 
             if counte_known: 
                 print(
                     'Total aligned:  %6.1f c/pix = %6.2f e/pix = %6.2f e/A^2' %
                     (fs.aligned_total, fs.aligned_total / fs.counte, 
                      fs.aligned_total_ea))
             else:
                 print('Total aligned:  %6.1f c/pix ' % (fs.aligned_total, ))
        if serialem is not None: 
             if counte_known: 
                 print(
                     'Total serialem: %6.1f c/pix = %6.2f e/pix = %6.2f e/A^2' %
                     (fs.serialem_total, fs.serialem_total / fs.serialem.counte,
                      fs.serialem_total_ea))
             else:
                 print('Total serialem:  %6.1f c/pix ' % (fs.serialem_total, ))
        print
        if counte_known:
            print(
                'Pixel = %5.2f A  Conversion = %5.2f c/e' % (
                    fs.apixel, fs.counte))
        else:
            print('Pixel = %5.2f A' % (fs.apixel,))

    return fs


class FrameSeries(object):
    """
    Basic manipulation of frame series (frame stacks) and the resulting 
    projection stack. 

    Note: currently implemented for series recorded by SerialEM.
    """

    def __init__(
            self, serialem_dir, serialem_projection_dir=None, 
            orig_frames_dir=None, aligned_frames_dir=None, 
            align=None, aligned_prefix=None, ucsf_translations_path=None, 
            apixel=None, counte=None, gain_corr_acquisition=True):
        """
        Sets attributes from arguments and reads serial em (mdoc) file.

        If arg aligned prefix is None, it is set to the default values for
        dm ('') and dimi alignments ('sum.'). If ucsf alignment (MotionCorr) 
        is specified (arg alignment) aligned prefix has to be given.

        MotionCorr is expected to run in serial mode (-serial 1). See 
        split__ucsf_translations() for additional info.

        Arguments:
          - serialem_dir: directory where serial em mdoc and stacks are
          - serialem_projection_dir:  directory where individual projections
          from serialem are writen, if needed
          - orig_frames_dir: directory where original (unaligned) frames are
          - aligned_frames_dir: directory where aligned frames are located
          - align: alignment program used ('dm', 'dimi' or 'ucsf' for 
          MotionCorr)
          - aligned_prefix: prefix added to (unaligned) frames to make aligned
          frames file names
          - ucsf_translations_path: file path for the ucsf alignment 
          (MotionCore) log file (sent to stdout by ucsf alignment program)
          - apixel: pixel size in A
          - counte: number of counts per electron, or None to set it at a later
          point from the value present in frame stack mrc files (using 
          projections(), for example
          - gain_corr_acquisition: flag indicating whether gain correction was 
          done during tomo acquisition

        Attributes set:
          - all arguments are saved as attributes with same names 
          - self.serialem: SerialEM object 
          - self.dm_aligned_prefix = "sum."
          - self.dm_translation_prefix = "Measured drift for "
          - self.aligned_path_prefix: self.aligned_frames_dir / 
          self.aligned_prefix
          - self.ucsf_translations_path: motioncorr (ucsf) translations path
          - self.apixel: pixel size in A
          - self.counte: counts per electrons
          - self.gain_corr_acquisition: from gain_corr_acquisition 
          - self.frames_guess_counte: (set to 1.) guess for the multiplication
          factor SerialEM uses to scale projection stack frames when gain 
          correction is not done during acquisition (1 should be correct)
          - self.serialem_guess_counte: (set to 19.2) guess for the 
          multiplication factor SerialEM uses to scale projection stack 
          when gain correction is not done during acquisition (microscope 
          dependent)
        """
        
        # constants
        self.dm_aligned_prefix = "sum."
        self.dm_translation_prefix = "Measured drift for "
        self.frames_guess_counte = 1.
        self.serialem_guess_counte = 19.2
        
        # figure out frame prefix
        if aligned_prefix is None:
            if align is None:
                self.aligned_prefix = ''
            if align == 'dm':
                self.aligned_prefix = self.dm_aligned_prefix
            elif align == 'dimi':
                self.aligned_prefix = ''
            elif align == 'ucsf':
                if aligned_prefix is None:
                    raise ValueError(
                        "Argument 'aligned_prefix' has to be specified for "
                        + "ucsf align.")
            else:
                raise ValueError(
                    "Argument align '" + align + "' was not understood. "
                    + "Valid options are None, 'dm', 'dimi', and 'ucsf'")
        else:
            self.aligned_prefix = aligned_prefix

        # sets attributes
        self.align = align
        self.serialem_dir = serialem_dir
        self.serialem_projection_dir = serialem_projection_dir
        self.orig_frames_dir = orig_frames_dir
        self.aligned_frames_dir = aligned_frames_dir
        if self.aligned_frames_dir is not None:
            self.aligned_path_prefix = os.path.join(
                self.aligned_frames_dir, self.aligned_prefix)

        # split ucsf translations
        if (((align == 'ucsf') or  (align == 'motioncorr')) 
            and (ucsf_translations_path is not None)):
            self.split_ucsf_translations(ucsf_translations_path)
        self.ucsf_translations_path = ucsf_translations_path

        # set pixel size and counts per electron
        self.apixel = apixel
        self.gain_corr_acquisition = gain_corr_acquisition
        if  (counte is None) or gain_corr_acquisition:
            self.counte = counte
        else:
            self.counte = self.frames_guess_counte
            
        # read serieal em (mdoc) files
        self.serialem = SerialEM(
            dir_=self.serialem_dir, counte=counte,
            projection_dir=serialem_projection_dir)

    def projections(self, projections=None, serialem_stack=False, stats=True,
                translations=False, names_only=False):
        """
        Generator that yields projection images. At each iteratition one or 
        more images corresponding to the current tilt angels are yielded. 

        The yielded images are selected in the following way:
          - All serialEM log (mdoc) files residing in self.serialem directory 
          are combined (they should all correspond to the same tilt series)
          - Names (without directory path) of the projection frame stacks (each 
          image is a stack of frames) and tilt angles are read from serialEM 
          mdoc file(s) and sorted by tilt angle. 
          - SerialEM projections are read from serialEM projection stacks that 
          are located in self.serialem directory and correspond to the 
          mdoc files
          - Complete paths to the projection frame stacks are obtained
          from self.orig_frames_dir and their names
          - Projection frame stacks are flattened (along z-axis) to 
          non-frame-aligned projection images
          - Complete paths to frame aligned projection images are obtained from
          self.aligned_frames_dir and the projection frame stack names. If such
          file doesn't exist (because frame stacks are tiffs and aligned 
          projections are mrcs, for example) the extensions or aligned 
          projections are changed to mrc. If these don't exist either, a 
          warning is generated.
          - Paths to files containing frame alignment translations are  
          determined from frame aligned projection file names (for Dimitri
          and DM alignment implemetnations)

        The following file / directory name conventions have to be followed:
          - In MotionCorr (UCSF) alignment the aligned projection file names 
          have a user specified prefix
          - In Dimi alignment the frame stacks and the aligned projection
          images have the same file names (but they reside in different 
          directories)
          - In DM alignment the aligned projection file names are obtained
          by adding prefix 'sum' (more precisely self.dm_aligned_prefix) to
          the corresponding frame stacks
          - In DM alignment the thanslation file names are obtained
          by adding prefix 'Measured drift for ' (more precisely 
          self.dm_translation_prefix) to the corresponding frame stacks

        In case pixel size has not been set already (self.apixel is None), 
        it is set to the value reported in mdoc file.

        If self.counte and self.serialem.counte are not None, they are not
        modified here (these are used to convert from counts to electrons).

        Alternatively, if self.counte is None, it is set from frame stacks mrc
        header. If frames stacks are not specified (self.orig_frames_dir), or 
        are mot in the mrc format, self.counte is set to 
        self.frames_guess_counte (normally set to 1).

        If self.counte is set from the frames stack mrc heared and 
        self.gain_corr_acquisition is True, self.serialem.counte is set to
        the same value. Otherwise it is set to self.serialem_guess_counte.
        Note that the default value (19.2) is microscope dependent.

        Arguments:
          - serialem_stack: flag indicating if projections from serial em stack
          are used
          - projections: indices of projections that are used, can be
            - None: all projections
            - list: list of projection indices
            - int: randomly pick this number of projections
          - stats: flag indicating if basic image statistics are calculated
          - names_only: flag indicating if instead of projection images the
          corresponding file names are yielded (useful for testing)
          - translations: flag indicating if frame alignment translations
          are also read (ignored if self.align is None)

        Sets:
          - self.apixel: pixel size in A
          - self.counte: counts per pixel (only if it wasn't set previously)
          - self.original_total_ea: total electrons per A for original frames
          - self.original_mean_ea: mean electrons per A per original frame
          - self.aligned_total_ea: total electrons per A for aligned frames
          - self.serial_total_ea: total electrons per A for serial stack

        Yields a dictionary that contains the following (key, value) pairs:
          - 'tilt_angle': tilt angle
          - 'exposure_time': exposure time
          - 'original': (pyto.grey.Image) projection frames stack (non-aligned)
          - 'original_flat': (pyto.grey.Image) projection obtained by summing 
           up original (non-aligned) frames without aligning them 
          - 'aligned': (pyto.grey.Image) projection obtained from aligned frames
          - 'translations': (ndarray) frame alignment translations
          - 'serialem': (pyto.grey.Image) serial em projection image
          - 'serialem_name': name of the projection (stack name + projection
          index in the order recorded)
          - 'serialem_projection': path to projections obtained
          from serialem
        All pyto.grey.Image objects have the following attributes set:
          - mean, std, var, min, max: mean, std, avriance, min, max counts 
          per pixel
        """

        # check whether to read translations
        translations = translations and (self.align is not None)

        # read mdoc file info and sort by tilt angle
        self.serialem.parse_mdocs(sort=True)

        # set pixel size
        if self.apixel is None:
            self.apixel = self.serialem.apixel

        # set projection indices
        all_projs = range(len(self.serialem.tilt_angles))
        if projections is not None:
            if isinstance(projections, (list, tuple)):
                proj_indices = projections
            elif isinstance(projections, int):
                np.random.shuffle(all_projs) 
                proj_indices = all_projs[0:projections]
            else:
                raise ValueError(
                    "Argument 'projections' has to be an int, list or None")
        else:
            proj_indices = all_projs

        # loop over projections
        prev_serialem_zslice_path = None
        self.original_total = 0
        self.aligned_total = 0
        self.serialem_total = 0
        n_projections = 0
        self.aligned_paths_list = []
        for proj_ind in proj_indices:

            # tilt angle and exposure time
            tilt_angle = self.serialem.tilt_angles[proj_ind]
            exposure_time = self.serialem.exposure_times[proj_ind]

            # read and flatten original frames
            if self.orig_frames_dir is not None:
                orig_frame_path = os.path.join(
                    self.orig_frames_dir, 
                    self.serialem.orig_frame_names[proj_ind])
                if not names_only:
                    orig = Image.read(file=orig_frame_path, header=True)
                    orig_flat_data = orig.data.sum(axis=2)
                    orig_flat = Image(data=orig_flat_data)
                    if stats:
                        try:
                            if ((self.counte is None) 
                                and (orig.fileFormat is not None)
                                and (orig.fileFormat == 'mrc')):
                                self.counte = self.find_counte(
                                    header=orig.header)
                        except AttributeError:
                            # just in case orig.fileFormat not defined
                            pass

                        orig.getStats(apixel=self.apixel, counte=self.counte)
                        self.original_total = self.original_total + orig.mean
                        orig_flat.getStats(apixel=self.apixel, 
                                           counte=self.counte)

            # read aligned frames
            if self.aligned_frames_dir is not None:

                # figure out aligned frames file path
                aligned_name = (self.aligned_prefix + 
                                self.serialem.orig_frame_names[proj_ind])
                aligned_path = os.path.join(
                    self.aligned_frames_dir, aligned_name)
                if not os.path.exists(aligned_path):
                    aligned_name = aligned_name.rsplit('.', 1)[0] + '.mrc'
                    aligned_path = os.path.join(
                        self.aligned_frames_dir, aligned_name)
                if os.path.exists(aligned_path):
                    self.aligned_paths_list.append(aligned_path)
                else:
                    warnings.warn(
                        "Aligned file " + aligned_path + " doesn't exist.")

                # read image and frame translations, make stats
                if not names_only:
                    aligned = Image.read(file=aligned_path)
                    if stats:
                        aligned.getStats(apixel=self.apixel, counte=self.counte)
                        self.aligned_total = self.aligned_total + aligned.mean
                    if translations:
                        #if (self.align == 'ucsf') or (self.align == 'motioncorr'):
                            #frame_trans = self.read_frame_translations(
                                #path=self.ucsf_translations_path)
                        frame_trans = self.read_frame_translations(
                            path=aligned_path)

            # read serialem projections
            z_value = self.serialem.z_values[proj_ind]
            if serialem_stack:
                serialem_zslice_path = os.path.join(
                    self.serialem_dir, self.serialem.stack_names[proj_ind])
                if not names_only:
                    if ((prev_serialem_zslice_path is None) 
                        or (prev_serialem_zslice_path != serialem_zslice_path)):
                        serialem_stack = Image.read(file=serialem_zslice_path, 
                                                    fileFormat='mrc')
                        prev_serialem_zslice_path = serialem_zslice_path
                    serialem_data = serialem_stack.data[:,:,z_value]
                    serialem_proj = Image(data=serialem_data)

                    # set counts per electron if needed
                    if self.serialem.counte is None:
                        if self.gain_corr_acquisition:
                            self.serialem.counte = self.counte
                        else:
                            self.serialem.counte = self.serialem_guess_counte
 
                    # calculate stats
                    if stats:
                        serialem_proj.getStats(
                            apixel=self.apixel, counte=self.serialem.counte)
                        self.serialem_total = (
                            self.serialem_total + serialem_proj.mean)

            # yield the current projection image in all available forms
            n_projections = n_projections + 1
            result = {'tilt_angle' : tilt_angle}
            result['exposure_time'] = exposure_time
            if self.orig_frames_dir is not None: 
                result['original_name'] = \
                    self.serialem.orig_frame_names[proj_ind]
                if not names_only:
                    result['original'] = orig
                    result['original_flat'] = orig_flat
            if self.aligned_frames_dir is not None:
                result['aligned_name'] = \
                    self.serialem.orig_frame_names[proj_ind]
                if not names_only:
                    result['aligned'] = aligned
            if translations:
                result['translations'] = frame_trans
            if serialem_stack:
                if not names_only:
                    result['serialem'] = serialem_proj
                result['serialem_name'] = (
                    self.serialem.stack_names[proj_ind] + ':' + str(z_value))
                if self.serialem.projection_dir is not None:
                    result['serialem_projection'] = (
                        self.serialem.projection_paths[proj_ind])
            yield result
                
        # set total values
        self.original_mean = self.original_total / float(n_projections)
        if (self.apixel is not None) and (self.counte is not None):
            conversion = self.apixel * self.apixel * self.counte
            sem_conversion =  self.apixel * self.apixel * self.serialem.counte
            self.original_total_ea = self.original_total / float(conversion)
            self.original_mean_ea = self.original_mean / float(conversion)
            self.aligned_total_ea = self.aligned_total / float(conversion)
            self.serialem_total_ea = self.serialem_total / float(sem_conversion)

    def find_counte(self, header):
        """
        Reads number of counts per electron from the specified header. 

        This header has to be from a frames stack im mrc format generated by
        DM and SerialEM. Written for Gatan K2 detector.

        Argument:
          - header: Frames stack header separated in a list (as attribute
          pyto.grey.Image.header or pyto.io.ImageIO.header)

        Returns number of counts per electron, or None if not found.
        """

        # checks?
        
        # string that preceeds counts per electron
        pre_strings = [
            'SerialEMCCD: Dose frac. image, scaled by',
            'SerialEMCCD: Dose fractionation image, scaled by']

        # get to the beginning of counts per electron
        record = header[-1]
        found = False
        for pre_str in pre_strings:
            if record.find(pre_str) > -1:
                rest = record.split(pre_str, 1)[1].strip()
                found = True
                break
        if not found:
            raise ValueError("Counts per electron could not be read.")
            #return None

        # read the number
        number = ''
        for char_ in rest:
            if (char_.isdigit()) or (char_ == '.'): 
                number = number + char_
            else:
                break
        counte = float(number)

        return counte

    def read_frame_translations(self, path):
        """
        Reads translations (shifts) between frames of aligned frames image 
        specified by arg path, which was obtained by Dimi, DM or 
        ucsf (MotionCore) alignement.

        MotionCorr is can run in in both serial (-serial 1) and individual
        mode (-serial 0). See split_ucsf_translations() for additional info.

        For dm and dimi alignment, log files areexpected to reside in the
        same directory with aligned frames. For motioncorr, these
        directories can be different, but then self.ucsf_translations_path
        has to be set.

        Argument:
          - path: aligned frames image path
          - align: alignment program: 'dm', 'dimi', 'motioncorr' (same as 
          'ucsf') or None if shifts should not be read
        
        Returns array of translations (n_frames x 2) or None if the frame 
        alignemend file corresponding to path was not found 
        """
        
        if self.align is None:
            
            # no shifts found
            return None

        elif self.align == 'dm':

            # dm aligned
            dir_, base = os.path.split(path)
            base = base.lstrip(self.dm_aligned_prefix)
            base = self.dm_translation_prefix + base
            trans_path = os.path.join(dir_, base)
            trans = Image.read(file=trans_path)
            return trans.data

        elif self.align == 'dimi':

            # dimi aligned
            log_path = path + '.log'
            if os.path.isfile(log_path):

                # read translations from Dimi log
                numbers = False
                data_list = []
                for line in open(log_path):
                    if not numbers and ('Relative' in line):
                        numbers = True
                        continue
                    if numbers:
                        if line.isspace(): break
                        data_list.append(
                            np.fromstring(line, sep=' ', dtype=float))
                data_array = np.array(data_list)
                return data_array

        elif (self.align == 'ucsf') or (self.align == 'motioncorr'):

            # ucsf aligned
            data_list = []
            log_name = os.path.split(path)[1]
            log_name = log_name.rsplit('.', 1)[0] + '.log'
            log_dir = os.path.split(self.ucsf_translations_path)[0]
            log_path = os.path.join(log_dir, log_name)
            for line in open(log_path):
                if line.strip().startswith('...... Frame'):
                    number_str = line.rsplit('shift:')[1].strip()
                    data_list.append(
                        np.fromstring(number_str, sep=' ', dtype=float))
            data_array = np.array(data_list)
            return data_array
                    
        # frame alignment file not found
        raise ValueError(
            "Alignment program " + self.align + "(self.align) was not "
            + "understood. Valid options are None, 'dm', 'dimi', 'ucsf' "
            + " and 'motioncorr'")
        return None

    def split_ucsf_translations(self, path):
        """
        Extracts translations for multiple frames stacks aligned using UCSF 
        (MotionCorr) saved in a single log file, and writes them separately.

        Works for motioncorr run in both serial (serial=1) and individual 
        mode (serial=0). See split_ucsf_translations_serial() and 
        split_ucsf_translations_individual().

        Arguments:
          - path: path to the UCSF (MotionCorr) log file
        """
      
        if path is None: return

        # determine whether serial
        serial = True
        for line in open(path):

            if line.strip().startswith('-Serial'):
                split_line = line.split()
                if len(split_line) == 2:
                    serial_number = split_line[1]
                    if serial_number == '0':
                        serial = False
                        break
            # in case -Serial not found, assume it is serial
        #path.close

        # call appropriate method do split file
        if serial:
            self.split_ucsf_translations_serial(path)
        else:
            self.split_ucsf_translations_individual(path)

    def split_ucsf_translations_serial(self, path):
        """
        Extracts translations for multiple frames stacks aligned using UCSF 
        (MotionCorr) saved in a single log file, and writes them separately.

        Only the file names and translations are saved.

        MotionCorr is expected to run in serial mode (-serial 1). Takes care 
        of the fact that a frame stacks file can be opened (and reported in 
        the log file) before shifts of the previous frames stack are 
        calculated (and reported). The only assumption is that the order the
        frame stacks are open is the same as the order of reported shifts.
        This was confirmed by inspection of log files (10.2017). 

        Arguments:
          - path: path to the UCSF (MotionCorr) log file
        """

        if path is None:
            return

        log_fds = []
        log_fds_ind = 0
        files_saved_ind = 0
        in_shifts = False
        in_saved = False

        # Looks for lines that show loaded files (frame stacks), shifts and
        # saved files (aligned frames). Beacuse a file can be opened before
        # the shifts of a previous one were determined, keeps track of the
        # order of loaded files, frame shifts and saved files.
        for line in open(path):
            line = line

            if in_shifts:

                # shift lines
                log_fds[log_fds_ind].write(line)                    
                if len(line.strip()) == 0:
                    in_shifts = False                    
                    log_fds_ind = log_fds_ind + 1

            elif in_saved:

                # file saved lines
                log_fds[files_saved_ind].write(line)
                if len(line.strip()) == 0:
                    in_saved = False
                    log_fds[files_saved_ind].close()
                    files_saved_ind = files_saved_ind + 1
                    
            elif line.strip().endswith(': loaded'):
                frames_path = line.rsplit(': loaded')[0]
                # removed because in case ucsf alignment was not run from the 
                # notebook directory, paths from shift file aren't good here
                #if os.path.isfile(frames_path):

                # starting with new frames stack 
                frames_name = os.path.split(frames_path)[1]
                log_name = frames_name.rsplit('.', 1)[0] + '.log'
                log_path = self.aligned_path_prefix + log_name
                fd = open(log_path, 'w')
                log_fds.append(fd)
                fd.write(line)
                fd.write(os.linesep)

            elif line.strip() == 'Full-frame alignment shift':

                # starting shift lines
                log_fds[files_saved_ind].write(line)
                in_shifts = True

            elif line.strip() == 'Corrected sum has been saved in:':
                
                # before saved
                log_fds[files_saved_ind].write(line)
                in_saved = True

    def split_ucsf_translations_individual(self, path):
        """
        Extracts translations for multiple frames stacks aligned using UCSF 
        (MotionCorr) saved in a single log file, and writes them separately.

        MotionCorr is expected to run in individual mode (-serial 0) where
        the complete log is saved is one log file. This method simply splits the
        log file into separate log file (one for each frames stack).

        Arguments:
          - path: path to the UCSF (MotionCorr) log file

        Returns: Number of individual log files
        """

        # get directory part of path
        dir_ = os.path.split(path)[0]

        file_lines = []
        one_log_path = None
        count = 0
        for line in open(path):

            # write file and start a next one
            if  line.strip().startswith("Usage"):
                if one_log_path is not None:
                    #print("writing in " + one_log_path) 
                    log_fd = open(one_log_path, 'w')
                    log_fd.writelines(file_lines)
                    log_fd.close()
                    one_log_path = None
                    count += 1
                    file_lines = []

            # save line
            file_lines.append(line)

            # form new log file path 
            if line.strip().startswith("-OutMrc"):
                split_line = line.strip().split()
                if len(split_line) > 1:
                    out_mrc_path = split_line[1]
                    out_mrc_file = os.path.split(out_mrc_path)[1]
                    one_log_file = out_mrc_file.rsplit('.', 1)[0] + '.log'
                    one_log_path = os.path.join(dir_, one_log_file)

        # write the last log file
        if one_log_path is not None:
            #print("writing in " + one_log_path) 
            log_fd = open(one_log_path, 'w')
            log_fd.writelines(file_lines)
            log_fd.close()
            count += 1

        return count
