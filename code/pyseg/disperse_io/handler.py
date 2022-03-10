"""
Classes for handling (running) the DisPerSe commands

# Author: Antonio Martinez-Sanchez (Max Planck Institute for Biochemistry)
# Date: 16.10.14
"""

__author__ = 'martinez'

import time
import warnings
import subprocess
import logging
from astropy.io import fits
from . import disperse_io
import shutil
from pyto.io import ImageIO as ImageIO
from pyseg import pexceptions
try:
    from globals import *
except:
    from pyseg.globals import *

####################################################################################################
#
# Class for handling (running) the DisPerSe commands
#
#
class DisPerSe(object):

    ##### Constructor area
    # input_file: file name of the image for being processed
    # work_dir: working directory where intermediate result are stored.
    def __init__(self, input_file, work_dir):
        self.__work_dir = work_dir
        self.__create_work_dir()
        self.__input = self.__parse_input_image(input_file)
        self.__orig_input = input_file
        self.__mask = None
        self.__pad = [0, 0, 0, 0, 0, 0]
        self.__dump_manifolds = None
        self.__cut = None
        self.__nsig = None
        self.__robust = False
        self.__v_as_min = False
        self.__dump_arcs = -1
        self.__smooth = 0
        self.__log_file = self.__work_dir + '/disperse.log'
        self.__input_inv = None
        logging.basicConfig(filename=self.__log_file, format='%(ascime)s\n%(message)s')

    ##### Set functions area

    # Handles -dumpManifols of mse command. If None (default) this option is not activated
    def set_manifolds(self, in_str=None):
        self.__dump_manifolds = in_str

    # Handles -cut of mse command. If None (default) this option is not activated,
    # input must be a float
    def set_cut(self, cut_value=None):
        self.__cut = cut_value

    # Handles -nsig of mse command. If None (default) this option is not activated,
    # input must be a float
    def set_nsig_cut(self, nsig_value=None):
        self.__nsig = nsig_value

    # Handles -robustness of mse command. If True (default False) this option is activated
    def set_robustness(self, switch=False):
        self.__robust = switch

    # Handles -smooth of skelconv command
    def set_smooth(self, smooth=0):
        self.__smooth = smooth

    # Adds mask for mse command. If None (default) this option is not activated
    def set_mask(self, input_file=None):

        if not os.path.exists(input_file):
            error_msg = 'File %s not found.' % self.__mask
            raise pexceptions.PySegInputError(expr='set_mask DisPerSe', msg=error_msg)
        stem, ext = os.path.splitext(input_file)
        if ext != '.fits':
            error_msg = '%s is a non valid format. Use FITS image format.' % ext
            raise pexceptions.PySegInputError(expr='set_mask DisPerSe', msg=error_msg)

        # Get pad from mask
        mask = disperse_io.load_tomo(input_file)
        if len(mask.shape) == 2:
            mask = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))
        if len(mask.shape) != 3:
            error_msg = 'Input mask is not a 3D image.'
            raise pexceptions.PySegInputError(expr='set_mask DisPerSe', msg=error_msg)
        spadx, spady, spadz = np.where(mask == 0)
        pad = [np.min(spadx), np.min(spady), np.min(spadz),
               mask.shape[0] - np.max(spadx), mask.shape[1] - np.max(spady), mask.shape[2] - np.max(spadz)]
        if None in pad:
            error_msg = 'Not valid input mask.'
            raise pexceptions.PySegInputError(expr='set_mask DisPerSe', msg=error_msg)

        self.__pad = pad
        self.__mask = input_file

    # Handles -vertexAsMinima option of mse command. If False (default) this option is not activated
    def set_vertex_as_min(self, mode=False):
        self.__v_as_min = mode

    # Handles -dumpArcs option. If -1 (default) then arcs leading to minima, else if 0 arcs linking
    # saddle point and if 1 arcs linking to maxima.
    def set_dump_arcs(self, mode=-1):
        self.__dump_arcs = mode

    ##### Get functionality area

    # Return the skeleton as a vtkPolyData
    # force_no_mse: if True (default) an the mse command is not even if skelton file is not found, and error is
    #               reported instead
    def get_skel(self, force_no_mse=True):

        # Parse input parameters
        path, filename = os.path.split(self.__input)
        input_file_skl = self.__work_dir + '/' + filename
        if (self.__cut is not None) and (self.__cut > 0):
            if round(self.__cut) == self.__cut:
                input_file_skl += '_c' + str(int(math.floor(self.__cut)))
            else:
                input_file_skl += '_c' + str(self.__cut)
        elif (self.__nsig is not None) and (self.__nsig > 0):
            if round(self.__nsig) == self.__nsig:
                input_file_skl += '_s' + str(int(math.floor(self.__nsig)))
            else:
                input_file_skl += '_s' + str(self.__nsig)
        if self.__dump_arcs == -1:
            input_file_skl += '.down'
        elif self.__dump_arcs == 0:
            input_file_skl += '.inter'
        elif self.__dump_arcs == 1:
            input_file_skl += '.up'
        input_file_skl += '.NDskl'
        input_file_vtp = input_file_skl + '.S' + str(self.__smooth).zfill(3) + '.vtp'

        # Commands are only called if the input file does not exist
        if not os.path.exists(input_file_vtp):
            if (not force_no_mse) and (not os.path.exists(input_file_skl)):
                self.mse()
            if not os.path.exists(input_file_skl):
                error_msg = 'Skeleton file %s not found!' % input_file_skl
                raise pexceptions.PySegInputError(expr='get_skel (DisPerSe)', msg=error_msg)
            # Converting to VTK data
            skelconv_cmd = ('skelconv', input_file_skl,
                            '-outDir', self.__work_dir,
                            '-smooth', str(self.__smooth),
                            '-to', 'vtp')
            try:
                file_log = open(self.__log_file, 'a')
                file_log.write('\n[' + time.strftime("%c") + ']RUNNING COMMAND:-> ' + ' '.join(skelconv_cmd) + '\n')
                subprocess.call(skelconv_cmd, stdout=file_log, stderr=file_log)
            except subprocess.CalledProcessError:
                file_log.close()
                error_msg = 'Error running command %s. (See %s file for more information)' \
                            % (skelconv_cmd, self.__log_file)
                raise pexceptions.PySegInputError(expr='get_skel DisPerSe', msg=error_msg)
            except IOError:
                error_msg = 'Log file could not be written %s.' % self.__log_file
                raise pexceptions.PySegInputError(expr='get_skel (DisPerSe)', msg=error_msg)
            file_log.close()

        # Reading the vtp file
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(input_file_vtp)
        reader.Update()

        return reader.GetOutput()

    # Return the manifolds image as a numpy array
    # no_cut: if True (default False) no cut threshold is applied
    # inv: if True (default False) the manifolds are taken from the inverted image
    # force_no_mse: if True (default) an the mse command is not even if skelton file is not found, and error is
    #               reported instead
    def get_manifolds(self, no_cut=False, inv=False, force_no_mse=True):

        if self.__dump_manifolds is None:
            error_msg = 'dumpManifolds option must be configured.'
            raise pexceptions.PySegInputError(expr='get_manifolds (DisPerSe)', msg=error_msg)

        if not no_cut:
            # Parse input parameters
            if inv:
                self.gen_inv()
                input_file_net = self.__input_inv
            else:
                path, filename = os.path.split(self.__input)
                input_file_net = self.__work_dir + '/' + filename
            if (self.__cut is not None) and (self.__cut > 0):
                if round(self.__cut) == self.__cut:
                    input_file_net += '_c' + str(int(math.floor(self.__cut)))
                else:
                    input_file_net += '_c' + str(self.__cut)
            elif (self.__nsig is not None) and (self.__nsig > 0):
                if round(self.__nsig) == self.__nsig:
                    input_file_net += '_s' + str(int(math.floor(self.__nsig)))
                else:
                    input_file_net += '_s' + str(self.__nsig)
            input_file_net += '_manifolds_' + self.__dump_manifolds
            input_file_net += '.NDnet'
            input_file_vti = input_file_net + '.vti'

            # Commands are only called if the input file does not exist
            if not os.path.exists(input_file_vti):
                if (not force_no_mse) and (not os.path.exists(input_file_net)):
                    self.mse()
                if not os.path.exists(input_file_net):
                    error_msg = 'Network file %s not found!' % input_file_net
                    raise pexceptions.PySegInputError(expr='get_manifolds (DisPerSe)', msg=error_msg)
                # Converting to VTK data
                netconv_cmd = ('netconv', input_file_net, '-outDir', self.__work_dir, '-to', 'vtu')
                try:
                    file_log = open(self.__log_file, 'a')
                    file_log.write('\n[' + time.strftime("%c") + ']RUNNING COMMAND:-> ' + ' '.join(netconv_cmd) + '\n')
                    subprocess.call(netconv_cmd, stdout=file_log, stderr=file_log)
                except subprocess.CalledProcessError:
                    file_log.close()
                    error_msg = 'Error running command %s. (See %s file for more information)' \
                                % (netconv_cmd, self.__log_file)
                    raise pexceptions.PySegInputError(expr='get_manifolds DisPerSe', msg=error_msg)
                except IOError:
                    error_msg = 'Log file could not be written %s.' % self.__log_file
                    raise pexceptions.PySegInputError(expr='get_manifolds DisPerSe', msg=error_msg)
                file_log.close()
                # Converting to .vti format
                input_file_net += '.vtu'
                disperse_io.manifold3d_from_vtu_to_img(filename=input_file_net, outputdir=self.__work_dir,
                                                       format='vti', transpose=False, pad=self.__pad)
        else:
            # Parse input parameters
            if inv:
                self.gen_inv()
                input_file_net = self.__input_inv
            else:
                path, filename = os.path.split(self.__input)
                input_file_net = self.__work_dir + '/' + filename
            input_file_net += '_manifolds_' + self.__dump_manifolds
            input_file_net += '.NDnet'
            input_file_vti = input_file_net + '.vti'
            # Commands are only called if the input file does not exist
            if not os.path.exists(input_file_vti):
                if (not force_no_mse) and (not os.path.exists(input_file_net)):
                    self.mse(no_cut=True, inv=inv)
                if not os.path.exists(input_file_net):
                    error_msg = 'Network file %s not found!' % input_file_net
                    raise pexceptions.PySegInputError(expr='get_manifolds (DisPerSe)', msg=error_msg)
                # Converting to VTK data
                netconv_cmd = ('netconv', input_file_net, '-outDir', self.__work_dir, '-to', 'vtu')
                try:
                    file_log = open(self.__log_file, 'a')
                    file_log.write('\n[' + time.strftime("%c") + ']RUNNING COMMAND:-> ' + ' '.join(netconv_cmd) + '\n')
                    subprocess.call(netconv_cmd, stdout=file_log, stderr=file_log)
                except subprocess.CalledProcessError:
                    file_log.close()
                    error_msg = 'Error running command %s. (See %s file for more information)' \
                                % (netconv_cmd, self.__log_file)
                    raise pexceptions.PySegInputError(expr='get_manifolds DisPerSe', msg=error_msg)
                except IOError:
                    error_msg = 'Log file could not be written %s.' % self.__log_file
                    raise pexceptions.PySegInputError(expr='get_manifolds DisPerSe', msg=error_msg)
                file_log.close()
                # Converting to .vti format
                input_file_net += '.vtu'
                disperse_io.manifold3d_from_vtu_to_img(filename=input_file_net, outputdir=self.__work_dir,
                                                       format='vti', transpose=False, pad=self.__pad)

        # Reading the vti file
        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(input_file_vti)
        reader.Update()

        # Parse for avoiding negative values
        img = disperse_io.vti_to_numpy(reader.GetOutput(), transpose=False)
        img.astype(np.int64)
        img[img < 0] = 0
        img[img > np.prod(img.shape)] = 0

        return img.astype(np.int64)

    def get_working_dir(self):
        return self.__work_dir

    ##### Functions for calling DisPerSe commands

    # Handle how to call mse DisPerSe command
    # no_cut: if True (default False) no cut threshold is applied
    # inv: if True (default False) the manifolds are taken from the inverted image
    def mse(self, no_cut=False, inv=False):

        raise pexceptions.PySegInputError(expr='mse DisPerSe', msg='simulating an pexception')

        # Parsing input parameters
        mask_opt = None
        manifolds_opt = None
        cut_opt = None
        nsig_opt = None
        rob_opt = None
        v_as_min_opt = None
        if self.__mask is not None:
            mask_opt = ('-mask', self.__mask)
        if (self.__dump_manifolds is not None) or no_cut:
            manifolds_opt = ('-dumpManifolds', self.__dump_manifolds)
        if (self.__cut is not None) or no_cut:
            if self.__cut > 0:
                cut_opt = ('-cut', str(self.__cut))
        elif (self.__nsig is not None) and (self.__nsig > 0):
            nsig_opt = ('-nsig', str(self.__nsig))
        if self.__robust:
            rob_opt = '-robustness'
        if self.__v_as_min:
            v_as_min_opt = '-vertexAsMinima'
        if not no_cut:
            if self.__dump_arcs == -1:
                dump_arcs_opt = '-downSkl'
            elif self.__dump_arcs == 0:
                dump_arcs_opt = '-interSkl'
            elif self.__dump_arcs == 1:
                dump_arcs_opt = '-upSkl'
            else:
                error_msg = 'No valid -dumpArcs option, only -1, 0 and 1 are valid.'
                raise pexceptions.PySegInputError(expr='mse DisPerSe', msg=error_msg)
        out_dir_opt = ('-outDir', self.__work_dir)

        # Building the command
        mse_cmd = list()
        mse_cmd.append(DPS_MSE_CMD)
        if inv:
            if self.__input_inv is None:
                self.gen_inv()
            mse_cmd.append(self.__input_inv)
        else:
            mse_cmd.append(self.__input)
        if mask_opt is not None:
            mse_cmd.append(mask_opt[0])
            mse_cmd.append(mask_opt[1])
        if (manifolds_opt is not None) or no_cut:
            mse_cmd.append(manifolds_opt[0])
            mse_cmd.append(manifolds_opt[1])
        if (cut_opt is not None) and (not no_cut):
            mse_cmd.append(cut_opt[0])
            mse_cmd.append(cut_opt[1])
        elif nsig_opt is not None:
            mse_cmd.append(nsig_opt[0])
            mse_cmd.append(nsig_opt[1])
        if rob_opt is not None:
            mse_cmd.append(rob_opt)
        if v_as_min_opt is not None:
            mse_cmd.append(v_as_min_opt)
        if (not no_cut) and (dump_arcs_opt is not None):
            mse_cmd.append(dump_arcs_opt)
        mse_cmd.append(out_dir_opt[0])
        mse_cmd.append(out_dir_opt[1])

        # Command calling
        try:
            file_log = open(self.__log_file, 'a')
            file_log.write('\n[' + time.strftime("%c") + ']RUNNING COMMAND:-> ' + ' '.join(mse_cmd) + '\n')
            subprocess.call(mse_cmd, stdout=file_log, stderr=file_log)
        except subprocess.CalledProcessError:
            file_log.close()
            error_msg = 'Error running command %s.' % mse_cmd
            raise pexceptions.PySegInputError(expr='mse DisPerSe', msg=error_msg)
        except IOError:
            error_msg = 'Log file could not be written %s.' % self.__log_file
            raise pexceptions.PySegInputError(expr='mse DisPerSe', msg=error_msg)
        file_log.close()

    # Clean working directory
    def clean_work_dir(self):

        try:
            os.system("rm -rf %s" % self.__work_dir)
            # shutil.rmtree(self.__work_dir)
        except OSError as e:
            error_msg = 'WARNING: Error cleaning working directory (errno=' + str(e.errno) + ', filename=' + \
                        str(e.filename) + ', strerror=' + str(e.strerror) + ')'
            raise pexceptions.PySegInputWarning(expr='clean_work_dir DisPerSe', msg=error_msg)
        self.__create_work_dir()
        self.__input = self.__parse_input_image(self.__orig_input)
        logging.basicConfig(filename=self.__log_file, format='%(ascime)s\n%(message)s')
        if self.__mask is not None:
            self.set_mask(self.__mask)

    # Generates an inverted copy of the input image
    def gen_inv(self):

        stem, ext = os.path.splitext(self.__input)
        self.__input_inv = stem + '_inv.fits'
        warnings.resetwarnings()
        warnings.filterwarnings('ignore', category=UserWarning, append=True)
        fits.writeto(self.__input_inv, utils.lin_map(array=fits.getdata(self.__input), lb=1, ub=0),
                       overwrite=True, output_verify='silentfix')
        warnings.resetwarnings()
        warnings.filterwarnings('always', category=UserWarning, append=True)


    ##### Internal functionality area

    # Create working directory
    def __create_work_dir(self):
        if not os.path.exists(self.__work_dir):
            try:
                os.mkdir(self.__work_dir)
            except:
                error_msg = 'Error creating working directory.'
                raise pexceptions.PySegInputError(expr='__create_work_dir DisPerSe', msg=error_msg)

    # DisPerSe does not accept MRC or EM formats directly so it is necessary to convert input image
    # into FITS format
    def __parse_input_image(self, input_file):

        hold_file = input_file
        if not os.path.exists(hold_file):
            error_msg = 'File %s not found.' % hold_file
            raise pexceptions.PySegInputError(expr='__parse_input_image DisPerSe', msg=error_msg)
        path, filename = os.path.split(hold_file)
        stem, ext = os.path.splitext(filename)
        if (ext == '.mrc') or (ext == '.em'):
            image = ImageIO()
            image.read(hold_file)
            fits_image = self.__work_dir + '/' + stem + '.fits'
            warnings.resetwarnings()
            warnings.filterwarnings('ignore', category=UserWarning, append=True)
            fits.writeto(fits_image, image.data.transpose(), overwrite=True, output_verify='silentfix')
            warnings.resetwarnings()
            warnings.filterwarnings('always', category=UserWarning, append=True)
            return fits_image
        return hold_file