"""
Functions related to ctf.

Currently only few that allow running ctffind from console or notebook.

Work in progress.

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id: ctf.py 1485 2018-10-04 14:35:01Z vladan $
"""

__version__ = "$Revision: 1485 $"

import os
import subprocess
import logging
import numpy as np
import matplotlib.pyplot as plt

import pyto.util.nested
from pyto.io.image_io import ImageIO
from pyto.grey.image import Image


class Ctf(object):
    """
    Determination of CTF by external tools
    """

    # prefix for validation attributed obtained from gctf
    validation_prefix = "validation_"

    # default params ctffind 4.0.17, also 4.1
    default_params_ctffind = {
        "pixel_a":1, "cs":2.7, "amp":0.1, "phase":"no", 'box':512, 
        'min_res':30, 'max_res':5, 'min_def':5000, 'max_def':50000, 
        'def_step':500, 'astig':100, 'known_astig':'no', 'slow_search':'yes',
        'restraint_astig':'yes', 'tolerated_astig':200,
        'phase':'yes', 'min_phase':0, 'max_phase':2, 'phase_step':0.1,
        'expert':'no'}

    # parameter list for ctffind 4.0.17 (currently not used, left for reference)
    param_names_ctffind_4_0 = [
        'pixel_a', 'voltage', 'cs', 'amp', 'box', 'min_res', 'max_res', 
        'min_def', 'max_def', 'def_step', 'astig', 'phase', 
        'min_phase', 'max_phase', 'phase_step']

    # default parameter list for 4.1; consistent with default_params_ctffind
    param_names_ctffind_4_1 = [
        'pixel_a', 'voltage', 'cs', 'amp', 'box', 'min_res', 'max_res', 
        'min_def', 'max_def', 'def_step', 'known_astig', 'slow_search',
        'restraint_astig','tolerated_astig',
        'phase', 'min_phase', 'max_phase', 'phase_step', 'expert']

    def __init__(self):
        """
        Initializes common attributes
        """

        # attributes
        self.image_path_orig = []
        self.image_inds = []
        self.image_path = []
        self.ctf_path = []
        self.phases = []
        self.defoci_1 = []
        self.defoci_2 = []
        self.defoci = []
        self.resolution = []
        self.pixel_a = []
        self.angle = []

    @classmethod
    def find(
        cls, image_dir, image_prefix, ctf_dir, params, pixel_a=None, 
        flatten='auto', tool='ctffind', executable=None,
        param_file='ctf_params.txt', fast=False, max_images=None, 
        plot_ctf=True, plot_ps=True, b_plot=True, exp_f_plot=False, 
        show_legend=True, plot_phases=True, plot_defoci=True, 
        plot_resolution=True, print_results=True, print_validation=False):
        """
        Determines and shows CTF fits for multiple images. 

        All files located in (arg) image_dir whose namess start with (arg)
        image_prefix and that have extension mrc, em or st are selected
        for the ctf determination.

        If a selected file is 3D (image stack), and arg flatten is True or 
        'auto', all z-slices are summed up (saved in ctf_dir) and the ctf 
        is detemined on the resulting (flattened. Alternatively, if arg 
        flatten is False, z-slices are extracted, saved in ctf_dir and 
        analyzed separately.

        All resulting files, as well as the extraced or flattened images 
        (in case of 3D files) are saved or moved to directory ctf_dir.

        CTF is determined using external tools. Current options are:
          - CTFFIND
          - gCTF 
        These tools have to be installed externally.

        Parameters for the ctf tools are specified as a dictionary (arg params).
        Parameters used for both ctffind and gctf are:
          - 'pixel_a', 'voltage', 'cs', 'amp', 'box', 'min_res', 'max_res', 
            'min_def', 'max_def', 'def_step', 'astig', 'phase', 
            'min_phase', 'max_phase', 'phase_step'
        Voltage ('voltage') should always be specified. The pixel size 
        (pixel_a) has to be specified in case it can not be read from 
        the image header. All other parameters are optional, if they are
        not specified the ctffind / gctg default values are used.

        The default values should be fine for single particle images.
        Parameter recommendations for phase plate images are given in
        the ctffind / gctf documentation.

        In case of ctffind, arg params can also be a list containing the 
        parameter values in the same order as specified above, starting
        with voltage.

        Important for ctffind: Because the required arguments differ between 
        versions 4.0 and 4.1, as well as depend on values specified, it is 
        not guaranteed that the dictionary form of arg params will work.
        In case of problems, specify params as a list.

        In addition, all other gctf arguments can also be specified 
        (without '--'). It is suggested to use:
          'do_EPA':'', 'do_validation':''

        Parameter units are the same as in the ctf deterimantion tools.

        Intended for use in an environment such as Jupyter notebook.

        Arguments:
          - image_dir: directory where images reside
          - image prefix: beginning of image file(s)
          - ctf_dir: directory where the ctf determination results and 
          extracted images are saved
          - pixel_a: pixel size in A
          - params: ctf determination parameters
          - flatten: indicated whether 3D images should be flatten (True or 
          'auto') or not (False).
          - tool:  name of the ctf detmination tool
          - executable: ctf tool executable
          - param_file: name of the temporary parameter file 
          - fast: flag indicating whether ctffind --fast option is used
          - print_results: flag indicating if phase and defoci found 
          are printed for each analyzed image
          - plot_ctf: flag indicating whether ctf is plotted for each 
          analyzed image
          - show_legend: flag indicating whether a legend is shown on ctf graphs
          - plot_phases, plot_defoci: flags indicating whether a graph 
          containing phases and defoci of all images respectivelly are plotted
          - max_images: max number if image analyzed, for testing

        Returns an instance of this class. The following attributes are all 
        lists where elements correspond to individual images:
          - image_path_orig: image path of the input file
          - image_path: image path of the image that is actually used
          to deterime ctf. It differs from image_path_orig if the original
          (input) image is a stack that is flattened or used to extract slices
          - image_inds: index of a slice extracted for a stack
          - ctf_path: path of the ctf fit image
          - defocus_1, defocus_2, defocus: defoci along the two axes and the
          mean defocus in um
          - angle: defocus (astigmatism) angle
          - phase: phase shift in multiples of pi
          - resolution: resolution in nm
          - ccc: correlation coefficient
          - pixel_a: pixel size in A
          - b_factor: b-factor (gctf only)
        """

        # initialize
        index = 0
        new = cls()
        print_head = True
        if plot_ctf and fast: 
            print(
                "Warning: CTF will not be plotted because fast execution"
                + " was chosen")

        # check which ctf tool to use
        if tool == 'ctffind':
            if executable is None:
                executable = 'ctffind'
        elif tool == 'gctf':
            if executable is None:
                executable = 'gctf'
        else:
            raise ValueError(
                "CTF determination tool " + str(tool) + " was not understood.")
        new.tool = tool

        # cftfind on all images
        file_list = np.sort(os.listdir(image_dir))
        for image_name in file_list:

            # skip files that are not images
            if not image_name.startswith(image_prefix): continue
            if not (image_name.endswith('.mrc') or image_name.endswith('.st') 
                    or image_name.endswith('.em')): 
                continue
            if image_name.endswith('ctf.mrc'): continue

            # set input image path    
            image_path = os.path.join(image_dir, image_name)

            # figure out if to flatten or not (just once, assume all files 
            # are the same)
            im_io = ImageIO(file=image_path)
            if image_name.endswith('.st'):
                im_io.readHeader(fileFormat='mrc')
            else:
                im_io.readHeader()
            z_dim = im_io.shape[2]
            n_digits = int(np.ceil(np.log10(z_dim)))
            if isinstance(flatten, bool):
                pass
            elif isinstance(flatten, str) and (flatten == 'auto'):
                if z_dim > 1: 
                    flatten = True
                else:
                    flatten = False
            else:
                raise ValueError(
                    "Argument flatten: "+ str(flatten) +" was not understood.") 

            # load stack and prepare image name, if need to extract images
            if (z_dim > 1) and not flatten:
                image_dir, image_name = os.path.split(image_path)
                image_base, image_extension = image_name.rsplit('.', 1)
                image_name_new_tmplt = (
                    image_base + '_%0' + str(n_digits) + 'd.mrc')
                if image_name.endswith('.st'):
                    stack = Image.read(
                        image_path, memmap=True, fileFormat='mrc')
                else:
                    stack = Image.read(image_path, memmap=True)
            else:
                image_path_to_read = image_path

            # find ctf of the current image or stack
            for image_in_stack_ind in range(z_dim):

                # extract and save images if needed
                if (z_dim > 1) and not flatten:
                    if not os.path.exists(ctf_dir): os.makedirs(ctf_dir)
                    image_path_to_read = os.path.join(
                        ctf_dir, (image_name_new_tmplt % image_in_stack_ind))
                    one_image = Image()
                    one_image.data = stack.data[:,:,image_in_stack_ind]
                    one_image.write(
                        file=image_path_to_read, pixel=stack.pixelsize)

                # save image path retlated
                new.image_path_orig.append(image_path)
                new.image_inds.append(image_in_stack_ind)
                new.image_path.append(image_path_to_read)

                # find ctf
                if tool == 'ctffind':

                    # ctffind
                    res_one = cls.ctffind(
                        image_path=image_path_to_read, flatten=flatten, 
                        ctf_dir=ctf_dir, executable=executable, 
                        pixel_a=pixel_a, params=params, 
                        param_file=param_file, fast=fast, print_head=print_head,
                        print_results= print_results, 
                        plot_ctf=plot_ctf, show_legend=show_legend)

                elif tool == 'gctf':

                    # gctf
                    res_one = cls.gctf(
                        image_path=image_path_to_read, params=params, 
                        pixel_a=pixel_a, flatten=flatten, ctf_dir=ctf_dir, 
                        executable=executable,  
                        plot_ctf=plot_ctf, plot_ps=plot_ps ,b_plot=b_plot, 
                        exp_f_plot=exp_f_plot, show_legend=show_legend,
                        print_results=print_results, 
                        print_head=print_head, 
                        print_validation=print_validation)
 
                    # save gctf specific data
                    try:
                        new.b_factor.append(res_one['b_factor'])
                    except AttributeError:
                        new.b_factor = [res_one['b_factor']]
                    for name, value in res_one.items():
                        if name.startswith(cls.validation_prefix):
                            try:
                                previous_val = getattr(new, name)
                                previous_val.append(value)
                                setattr(new, name, previous_val)
                            except AttributeError:
                                setattr(new, name, [value])

                else:
                    raise ValueError("Sorry tool: " + tool + " was not found.")

                # save data common for ctffind and gctf
                new.phases.append(res_one["phase"])
                new.defoci.append(res_one["defocus"])
                new.defoci_1.append(res_one['defocus_1'])
                new.defoci_2.append(res_one['defocus_2'])
                new.resolution.append(res_one['resolution'])
                new.pixel_a.append(res_one['pixel_a'])
                new.angle.append(res_one['angle'])
                new.ctf_path.append(res_one['ctf_path'])

                # keep track of n images processed so far
                print_head = False
                index = index + 1
                if (max_images is not None) and (index > max_images): break
                if flatten: break

        # plot phases
        if plot_phases:
            plt.figure()
            plt.bar(range(index), new.phases)
            plt.plot([0, index], [0.5, 0.5], 'r--')
            plt.ylabel('Phase shift [$\pi$]')
            plt.xlabel('Images')
            plt.title("Phase shift summary")

        # plot defocus
        if plot_defoci:
            plt.figure()
            plt.bar(range(index), new.defoci)
            plt.ylabel('Defocus [$\mu m$]')
            plt.xlabel('Images')
            plt.title("Defocus summary")

        # plot resolution
        if plot_resolution:
            plt.figure()
            plt.bar(range(index), new.resolution)
            plt.ylabel('Resolution [nm]')
            plt.xlabel('Images')
            plt.title("Resolution summary")

        return new

    @classmethod
    def ctffind(
        cls, image_path, ctf_dir, params, pixel_a=None, flatten=False, 
        executable='ctffind', param_file='ctf_params.txt', fast=False, 
        print_results=True, print_head=True, 
        plot_ctf=True, show_legend=True):
        """
        Determines and shows CTF fits of one image using ctffind.

        See find() for more information.
        """

        # make ctf dir if doesn't exist
        if not os.path.exists(ctf_dir): os.makedirs(ctf_dir)

        # find pixel size
        if pixel_a is None:
            pixel_a = cls.read_pixel_size(image_path=image_path) 

        # flatten frame stack
        if flatten:
            image_path = cls.flatten_stack(
                stack_path=image_path, flat_dir=ctf_dir)

        # default params ctffind 4.0.17 (moved to top of this file anyway)
        #default_params = {
        #    "pixel_a":1, "cs":2.7, "amp":0.1, "phase":"no", 'box':512, 
        #    'min_res':30, 'max_res':5, 'min_def':5000, 'max_def':50000, 
        #    'def_step':500, 'astig':100, 'phase':'no', 'min_phase':0, 
        #    'max_phase':2, 'phase_step':0.1}
        #param_names = [
        #    'pixel_a', 'voltage', 'cs', 'amp', 'box', 'min_res', 'max_res', 
        #    'min_def', 'max_def', 'def_step', 'astig', 'phase', 
        #    'min_phase', 'max_phase', 'phase_step']
            
        # keep params if list, add default if dict
        if isinstance(params, list):
            comb_params = [pixel_a] + params
        elif isinstance(params, dict):
            params_dict = cls.default_params_ctffind.copy()
            params_dict.update(params)
            params_dict['pixel_a'] = pixel_a
            param_names = cls.make_param_names_ctffind(params=params_dict)
            comb_params = [params_dict[name] for name in param_names]

        # set ctffind out paths
        image_dir, image_name = os.path.split(image_path)
        image_base, image_extension = image_name.rsplit('.', 1)
        ctf_path = os.path.join(ctf_dir, image_base + '_ctf.mrc')   
        ctf_txt_path = os.path.join(ctf_dir, image_base + '_ctf.txt')
        ctf_avrot_path = os.path.join(ctf_dir, image_base + '_ctf_avrot.txt')

        # wite ctf parameters to a file
        param_path = os.path.join(ctf_dir, param_file)
        pf = open(param_path, 'w')
        pf.write(image_path + '\n')
        pf.write(ctf_path + '\n')
        str_params = [str(par) + '\n' for par in comb_params]
        pf.writelines(str_params)
        pf.flush()

        # execute ctffind
        # shell commands that work:
        #     - ctffind < param_path
        #     - cat params.txt | ctffind
        #print(image)
        if fast:
            ctf_cmd = [executable, '--fast']
        else:
            ctf_cmd = [executable]
        try:
            subprocess.check_call(ctf_cmd, stdin=open(param_path))
        except Exception as exc:
            # workaround for ctffind command returning code 255 (4.1.8, 09.2018)
            logging.debug('CalledProcessError: ' + str(exc))
            
        # read results:
        ctf_txt = np.loadtxt(ctf_txt_path)
        results = {
            "defocus_1":ctf_txt[1]/10000., "defocus_2":ctf_txt[2]/10000., 
            "angle" : ctf_txt[3], "phase":ctf_txt[4]/np.pi, 
            "ccc" : ctf_txt[5], "resolution" : ctf_txt[6] / 10., 
            'pixel_a':pixel_a}
        results['defocus'] = (results['defocus_1'] + results['defocus_2']) / 2.
        results['ctf_path'] = ctf_path

        # prepare header for defoci and phases
        if print_head:
            left_space = ' ' * ((len(image_name) - 5) / 2)
            right_space = ' ' * ((len(image_name) - 4) / 2)
            head_1 = (
                left_space + "Image" + right_space + 
                " Defocus 1 Defocus 2 Phase Resolution")
            head_2 = (
                left_space + "     " + right_space + 
                "    um        um      [pi]      nm   ")

        # prepare results
        if print_results:
            data_format = '%s %6.2f    %6.2f   %6.2f   %6.2f  '
            data_vars = (
                image_name, results["defocus_1"], results["defocus_2"], 
                results["phase"], results["resolution"])

        # print
        if print_head:
            print(head_1)
            print(head_2)
        if print_results:
            print(data_format % data_vars)

        # plot ctf
        if plot_ctf:
            plt.figure()
            avrot_data = np.loadtxt(ctf_avrot_path)
            x_data = avrot_data[0] / pixel_a
            plt.plot(x_data, avrot_data[2], 'g-', label='PS')
            plt.plot(
                x_data, avrot_data[3], color='orange', linewidth=2, 
                label='CTF fit')
            plt.plot(
                x_data, avrot_data[4], color='blue', linewidth=2, 
                label='Quality')
            plt.ylim(-0.1, 1.1)
            plt.xlabel("Spatial frequency [1/A])")
            plt.ylabel("Amplitude")
            if show_legend: plt.legend()
            plt.show()

        return results

    @classmethod
    def make_param_names_ctffind(cls, params):
        """
        Makes a list of parameter names that's suitable for ctffind 4.1 and
        it is in accordance with the specified params.

        Argument:
          - params: dict of parameters

        Returns parameter list
        """

        # optional parts
        if params['restraint_astig'] in ['yes', 'y']:
            restraint_astig_part = ['restraint_astig','tolerated_astig']
        else:
            restraint_astig_part = ['restraint_astig']
        if (params['phase'] == 'yes') or (params['phase'] == 'y'):
            phase_part = ['phase', 'min_phase', 'max_phase', 'phase_step']
        else:
            phase_part = ['phase']

        # combine
        param_names = (
            cls.param_names_ctffind_4_1[:12] + restraint_astig_part
            + phase_part + ['expert'])

        return param_names
    
    @classmethod
    def gctf(
        cls, image_path, ctf_dir, params, pixel_a=None, flatten=False, 
        executable='gctf', plot_ps=True, plot_ctf=True, 
        b_plot=True, exp_f_plot=False, show_legend=True, 
        print_results=True, print_head=True, print_validation=False):
        """
        Determines and shows CTF fits of one image using gctf.

        See find() for more information.
        """ 

        # make ctf dir if doesn't exist
        if not os.path.exists(ctf_dir): os.makedirs(ctf_dir)

        # find pixel size
        if pixel_a is None:
            pixel_a = cls.read_pixel_size(image_path=image_path) 

        # flatten frame stack if needed
        if flatten:
            image_path = cls.flatten_stack(
                stack_path=image_path, flat_dir=ctf_dir)

        # prepare parameters
        gctf_names = {
            'pixel_a':'apix', 'voltage':'kV', 'cs':'Cs', 'amp':'ac', 
            'box':'boxsize', 'min_res':'resL', 'max_res':'resH', 
            'min_def':'defL', 'max_def':'defH', 'def_step':'defS', 
            'astig':'astm', 'phase':'phase', 'min_phase':'phase_shift_L', 
            'max_phase':'phase_shift_H', 'phase_step':'phase_shift_S'}
        params["pixel_a"] = pixel_a 
        params_list = [
            ["--" + gctf_names.get(key, key), str(val)] 
            for key, val in params.items()]
        params_list = pyto.util.nested.flatten(params_list)
        params_list = [par for par in params_list if len(par) > 0]
        #print(params_list)

        # execute ctffind
        ctf_cmd = [executable] + params_list + [image_path]
        call_status = subprocess.check_call(ctf_cmd)

        # set gctf out paths
        image_dir, image_name = os.path.split(image_path)
        image_base, image_extension = image_name.rsplit('.', 1)
        epa_path = os.path.join(ctf_dir, image_base + '_EPA.log')
        gctf_path = os.path.join(ctf_dir, image_base + '_gctf.log')    
        ctf_path = os.path.join(ctf_dir, image_base + '.ctf')   
        tmp_epa_path = os.path.join(image_dir, image_base + '_EPA.log')
        tmp_gctf_path = os.path.join(image_dir, image_base + '_gctf.log')    
        tmp_ctf_path = os.path.join(image_dir, image_base + '.ctf')   

        # move generated files to ctf_dir
        if image_dir != ctf_dir:
            call_status = subprocess.check_call(['mv', tmp_epa_path, epa_path])
            call_status = subprocess.check_call(
                ['mv', tmp_gctf_path, gctf_path])
            call_status = subprocess.check_call(['mv', tmp_ctf_path, ctf_path])
            call_status = subprocess.check_call(
                ['mv', 'micrographs_all_gctf.star', ctf_dir])

        # read results
        in_last_cycle = False
        in_last_cycle_data = False
        validation_lines = []
        for line in open(gctf_path):

            # read defocus
            if line.find('LAST CYCLE') >= 0: 
                in_last_cycle = True
                #print line.strip('\n')
            elif in_last_cycle and (line.find('Defocus_U') >= 0): 
                #print line.strip('\n')
                head_split = line.strip().split()
                in_last_cycle_data = True
            elif in_last_cycle_data:
                #print line.strip('\n')
                data_split = line.strip().split()[:-2]
                in_last_cycle_data = False

            # read res limit and b factor
            elif in_last_cycle and line.startswith('Resolution limit'): 
                resolution = float(line.split()[-1])
            elif in_last_cycle and line.startswith('Estimated Bfactor'): 
                b_factor = float(line.split()[-1])
                in_last_cycle = False

            # read validation
            elif line.find('VALIDATION_SCORE') >= 0:
                validation_lines.append(line.strip('\n'))

        # extract results
        results_native = dict(
            [(head, float(value)) 
             for head, value in zip(head_split, data_split)])
        results_native["Defocus_U"] = results_native["Defocus_U"] / 10000.
        results_native["Defocus_V"] = results_native["Defocus_V"] / 10000.
        #print(results_native)
        key_dict = {
            "Defocus_U":"defocus_1", "Defocus_V":"defocus_2",
            "Angle":"angle", "CCC":"ccc", "Phase_shift":"phase"}
        results = dict([
            (key_dict[old_key], value)
            for old_key, value in results_native.items()])
        results['defocus'] = (results['defocus_1'] + results['defocus_2']) / 2.
        results['phase'] = results.get('phase', 0) / 180.
        results["resolution"] = resolution / 10.
        results["b_factor"] = b_factor
        #if results.get("phase") is None: results["phase"] = 0
        results['ctf_path'] = ctf_path
        results['pixel_a'] = pixel_a
        for val_line in validation_lines:
            val_list = val_line.strip().split()
            name_suf =  val_list[0].replace('-', '_')
            results[cls.validation_prefix + name_suf] = int(val_list[-1])

        # prepare header for defoci and phases
        if print_head:
            left_space = ' ' * ((len(image_name) - 5) / 2)
            right_space = ' ' * ((len(image_name) - 4) / 2)
            head_1 = (
                left_space + "Image" + right_space + 
                " Defocus 1 Defocus 2 Phase Resolution")
            head_2 = (
                left_space + "     " + right_space + 
                "    um        um      [pi]      nm   ")

        # prepare results
        if print_results:
            data_format = '%s %6.2f    %6.2f   %6.2f   %6.2f  '
            data_vars = (
                image_name, results["defocus_1"], results["defocus_2"], 
                results["phase"], results["resolution"])

        # add validation to header and results
        val_names = np.sort(
            [val_nam for val_nam in results
             if val_nam.startswith(cls.validation_prefix)])[::-1]
        for val_nam in val_names:
            if print_head:
                head_1 += (" " + val_nam.split(cls.validation_prefix, 1)[1])
                head_2 += "       "
            if print_results:
                data_format += '   %2d  '
                data_vars += (results[val_nam],)

        # print
        if print_head:
            print(head_1)
            print(head_2)
        if print_results:
            print(data_format % data_vars)

        # print validation
        if print_validation:
            for val_line in validation_lines:
                print val_line

        # plot ctf
        epa = np.loadtxt(epa_path, skiprows=1)
        if plot_ps:
            plt.figure()
            plt.plot(1./epa[:,0], epa[:,2])
            plt.ylabel('ln(|F|)')
            #if show_legend: plt.legend()
            plt.show()
        if plot_ctf:
            plt.figure()
            if b_plot:
                exp_b = np.exp(-b_factor * 1./epa[:,0]**2 / 4.)
            else:
                exp_b = 1
            plt.plot(1./epa[:,0], epa[:,1] * exp_b, label="CTF fit")
            if exp_f_plot:
                plt.plot(
                    1./epa[:,0], np.exp(epa[:,3]), label="$e^{ln(|F|-Bg)}$")
            else:
                plt.plot(1./epa[:,0], epa[:,3], label="$ln(|F|-Bg)$")
            plt.xlabel('Resolution [1/A]')
            if show_legend: plt.legend()
            plt.show()

        # return
        return results

    @classmethod
    def read_pixel_size(cls, image_path):
        """
        Reads pixel size from an image file.

        Raises ValueError if pixel size can not be read from the image

        Argument:
          - image_path: image path

        Returns: pixel size in A
        """

        image_io = ImageIO()
        if image_path.endswith('.st'):
            image_io.readHeader(file=image_path, fileFormat='mrc')
        else:
            image_io.readHeader(file=image_path)
        if image_io.pixel is not None:
            if isinstance(image_io.pixel, (list, tuple)):
                pixel_a = 10 * image_io.pixel[0] 
            else:
                pixel_a = 10 * image_io.pixel
        else:
            raise ValueError(
                "Pixel size could not be found from image " + image_path +
                ". Please specify pixel_a as an argument.")

        # in case of 0 pix size
        if pixel_a == 0:
            raise ValueError(
                "Pixel size could not be found from image " + image_path +
                ". Please specify pixel_a as an argument.")

        return pixel_a

    @classmethod
    def flatten_stack(cls, stack_path, flat_dir):
        """
        Flattens image stack, that is sums up all z-slices and writes
        the resulting (flat) image).

        Arguments:
          - stack_path: path to the image stack
          - flat_path: path where the resulting image is saved

        Returns resulting image path
        """

        # parse stack path
        stack_dir, stack_name = os.path.split(stack_path)
        stack_base, stack_extension = stack_name.rsplit('.', 1)
        if stack_extension == 'st': 
            stack_extension = 'mrc'
            file_format = 'mrc'
        else:
            file_format = None

        # read, flatten and write
        flat_path = os.path.join(
            flat_dir, stack_base + '_flat.' + stack_extension)
        frame = Image.read(file=stack_path, fileFormat=file_format)
        frame.data = np.sum(frame.data, axis=2, dtype=frame.data.dtype)
        frame.write(file=flat_path, pixel=frame.pixelsize)

        return flat_path
