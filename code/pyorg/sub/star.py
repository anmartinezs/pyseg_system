"""
Set of classes for dealing with a STAR files (Relion's format)

# Author: Antonio Martinez-Sanchez (Max Planck Institute for Biochemistry)
# Date: 1.06.16
"""

__author__ = 'Antonio Martinez-Sanchez'


import copy
import pyto
import errno
from pyorg.globals import *
# from plist import TomoPeaks
from plist import *
from pyorg import disperse_io
import numpy as np

from pyorg.pexceptions import *

###########################################################################################
# Global functionality
###########################################################################################

# Relion tomogram normalization
# tomo: input tomogram
# mask: if None (default) the whole tomogram is used for computing the statistics otherwise just the masked region
def relion_norm(tomo, mask=None):

    # Input parsing
    if mask is None:
        mask = np.ones(shape=tomo.shape, dtype=np.bool)

    # Inversion
    hold_tomo = -1. * tomo

    # Statistics
    stat_tomo = hold_tomo[mask>0]
    mn, st = stat_tomo.mean(), stat_tomo.std()

    # Histogram equalization
    tomo_out = np.zeros(shape=tomo.shape, dtype=np.float32)
    if st > 0:
        tomo_out = (hold_tomo-mn) / st
    else:
        print 'WARNING (relion_norm): standard deviation=' + str(st)

    return tomo_out

###########################################################################################
# Class for converting data types for columns used by Relion
###########################################################################################

class RelionCols(object):

    def __init__(self):
        self.__cols = (# RELION
                       '_rlnMicrographName',
                       '_rlnCoordinateX',
                       '_rlnCoordinateY',
                       '_rlnCoordinateZ',
                       '_rlnImageName',
                       '_rlnCtfImage',
                       '_rlnGroupNumber',
                       '_rlnAngleRotPrior',
                       '_rlnAngleTiltPrior',
                       '_rlnAnglePsiPrior',
                       '_rlnAngleRot',
                       '_rlnAngleTilt',
                       '_rlnAnglePsi',
                       '_rlnOriginX',
                       '_rlnOriginY',
                       '_rlnClassNumber',
                       '_rlnNormCorrection',
                       '_rlnOriginZ',
                       '_rlnLogLikeliContribution',
                       '_rlnMaxValueProbDistribution',
                       '_rlnNrOfSignificantSamples',
                       '_rlnRandomSubset',
                       # PySeg: Graph analysis
                       '_psGhMCFPickle',
                       # PySeg: Segmentation
                       '_psPickleFile',
                       '_psStarFile',
                       '_psSegImage',
                       '_psSegLabel',
                       '_psSegScale',
                       '_psSegRot',
                       '_psSegTilt',
                       '_psSegPsi',
                       '_psSegOffX',
                       '_psSegOffY',
                       '_psSegOffZ',
                       '_psPixelSize',
                       '_psPixelSizeSvol',
                       # PySeg: Affinity Propagation
                       '_psAPClass',
                       '_psAPCenter',
                       '_psStartSurfIds',
                       '_psEndSurfIds',
                       '_psSurfacePID',
                       # PyOrg: Surface analysis
                       '_suTomoParticles',
                       '_suSurfaceVtp',
                       '_suPartShape',
                       '_suSurfaceVtpSim',
                       # PyOrg: Microtubules
                       '_mtCenterLine',
                       '_mtParticlesTomo',
                       # PyOrg: generic fibers
                       '_fbCurve',
                       '_fbTomoFilaments'
                       )
        self.__dtypes = (str,
                         float,
                         float,
                         float,
                         str,
                         str,
                         int,
                         float,
                         float,
                         float,
                         float,
                         float,
                         float,
                         float,
                         float,
                         int,
                         float,
                         float,
                         float,
                         float,
                         int,
                         int,
                         # PySeg: Graph analysis
                         str,
                         # PySeg: Segmentation
                         str,
                         str,
                         str,
                         int,
                         float,
                         float,
                         float,
                         float,
                         float,
                         float,
                         float,
                         float,
                         float,
                         # PySeg: Affinity Propagation
                         int,
                         int,
                         str,
                         str,
                         int,
                         # PyOrg: Microtubules
                         str,
                         str,
                         str,
                         str,
                         # PyOrg: generic fibers
                         str,
                         str
                         )
        assert len(self.__cols) == len(self.__dtypes)

    #### External functionality area

    # Returns data type used for this column in Relion, None if this is not a valid column key
    def get_dtype(self, key):
        if self.is_valid(key):
            idx = self.__cols.index(key)
            return self.__dtypes[idx]
        else:
            return None

    # Checks if a column name is valid for Relion
    def is_valid(self, key):
        try:
            self.__cols.index(key)
        except ValueError:
            return False
        return True

###########################################################################################
# Class for representing a Star file of a set of subvolumes usable by Relion
###########################################################################################

class Star(object):

    def __init__(self):
        self.__header_1 = ['\n', 'data_\n', '\n', 'loop_\n']
        self.__data = {}
        self.__dtypes = list()
        self.__rows = 0
        self.__cols = list()
        self.__checker = RelionCols()
        self.__root_dir, self.__fname = None, None

    #### Get/Set Area

    def set_root_dir(self, root_dir):
        self.__root_dir = str(root_dir)

    def get_root_dir(self):
        return self.__root_dir

    # Return the number of columns
    def get_ncols(self):
        return len(self.__cols)

    # Return the number of rows
    def get_nrows(self):
        return self.__rows

    # Returns column data in a list (if exists)
    # key: column key
    def get_column_data(self, key):
        if self.is_column(key):
            return self.__data[key]
        else:
            return None

    # Returns the existing differnt elements in a columh
    # key: column key
    def get_column_data_set(self, key):
        hold = self.get_column_data(key)
        if hold is None:
            return None
        else:
            return set(hold)

    # Returns a list with the keys of all columns
    def get_column_keys(self):
        return copy.copy(self.__cols)

    # Returns column type, None if the column is not found
    # key: column key
    def get_column_type(self, key):
        try:
            idx = self.__cols.index(key)
        except ValueError:
            return None
        return self.__dtypes[idx]

    def get_element(self, key, row):
        """
        Returns the row and column specified element, if the key is not present a KeyError is returned
        :param key: column key
        :param row: row index
        :return:
        """
        return self.__data[key][row]

    # Get the elements specified in a list of key columns and a row
    # keys: list column keys
    # row: row index
    # Returns: a list with with elements in the same order as keys
    def get_elements(self, keys, row):
        values = list()
        for key in keys:
            values.append(self.get_element(key, row))
        return values

    # Set the value of a specific column and row
    # key: column key
    # row: row index
    # val: value to be set
    def set_element(self, key, row, val):
        dtype = self.__checker.get_dtype(key)
        self.__data[key][row] = dtype(val)

    # Return geometrical information of specific particle: coordinates and rotation (optional) as 3-tuples
    # row: coordinate row
    # orig: scaling factor from pickled coordinates and subtomogram averaging (pick_angpix/sub_angpix) (default None not applied)
    # rots: if True (default False) the rotation angles are also returned in another list
    def get_particle_coords(self, row, orig=None, rots=False):
        x, y, z = self.get_element('_rlnCoordinateX', row), self.get_element('_rlnCoordinateY', row), \
                  self.get_element('_rlnCoordinateZ', row)
        if orig is not None:
            orig_inv = 1. / orig
            try:
                o_x = self.get_element('_rlnOriginX', row)
            except KeyError:
                o_x = 0.
            try:
                o_y = self.get_element('_rlnOriginY', row)
            except KeyError:
                o_y = 0.
            try:
                o_z = self.get_element('_rlnOriginZ', row)
            except KeyError:
                o_z = 0.
            # x -= (o_y*orig_inv)
            # y -= (o_x*orig_inv)
            x -= (o_x * orig_inv)
            y -= (o_y * orig_inv)
            z -= (o_z*orig_inv)
        if rots:
            try:
                rho = self.get_element('_rlnAngleRot', row)
            except KeyError:
                rho = 0.
            try:
                tilt = self.get_element('_rlnAngleTilt', row)
            except KeyError:
                tilt = 0.
            try:
                psi = self.get_element('_rlnAnglePsi', row)
            except KeyError:
                psi = 0.
            return (x, y, z), (rho, tilt, psi)
        else:
            return x, y, z

    # Return particles geometrical information: coordinates and rotation (optional)
    # orig: take into account origin shifting information (default False)
    # rots: if True (default False) the rotation angles are also returned in another list
    def get_particles_coords(self, orig=False, rots=False):
        coords, angs = list(), list()
        for row in range(self.get_nrows()):
            x, y, z = self.get_element('_rlnCoordinateX', row), self.get_element('_rlnCoordinateY', row), \
                      self.get_element('_rlnCoordinateZ', row)
            if orig:
                o_x, o_y, o_z = self.get_element('_rlnOriginX', row), self.get_element('_rlnOriginY', row), \
                                self.get_element('_rlnOriginZ', row)
                x -= o_x
                y -= o_y
                z -= o_z
            coords.append(np.asarray((x, y, z), dtype=np.float))
            if rots:
                rho, tilt, psi = self.get_element('_rlnAngleRot', row), self.get_element('_rlnAngleTilt', row), \
                                 self.get_element('_rlnAnglePsi', row)
                angs.append(np.asarray((rho, tilt, psi), dtype=np.float))
        if rots:
            return coords, angs
        else:
            return coords

    # Set the data of a column in one call
    # If the column does not exist of data does not fit the number of rows it does nothing
    # key: column key
    # data: column data
    def set_column_data(self, key, dat):
        if (self.is_column(key)) and (len(dat) == self.get_nrows()):
            if isinstance(dat, np.ndarray):
                self.__data[key] = dat.tolist()
            else:
                self.__data[key] = dat[:]

    # Find the index of the first element with a value in a column
    # key: column key
    # val: row value
    # Returns: if exists the element row index is returns, otherwise a ValueError exception is thrown
    def find_element(self, key, val):
        return self.__data[key].index(val)

    # Check if a column exists
    def has_column(self, key):
        return key in self.__cols

    #### External functionality

    # Scale particles coordinates and origins
    # sf: scale factor
    def scale_coords(self, sf):
        fsf = float(sf)
        hold = np.asarray(self.get_column_data('_rlnCoordinateX'), dtype=np.float) * fsf
        self.set_column_data('_rlnCoordinateX', hold)
        hold = np.asarray(self.get_column_data('_rlnCoordinateY'), dtype=np.float) * fsf
        self.set_column_data('_rlnCoordinateY', hold)
        hold = np.asarray(self.get_column_data('_rlnCoordinateZ'), dtype=np.float) * fsf
        self.set_column_data('_rlnCoordinateZ', hold)
        hold = np.asarray(self.get_column_data('_rlnOriginX'), dtype=np.float) * fsf
        self.set_column_data('_rlnOriginX', hold)
        hold = np.asarray(self.get_column_data('_rlnOriginY'), dtype=np.float) * fsf
        self.set_column_data('_rlnOriginY', hold)
        hold = np.asarray(self.get_column_data('_rlnOriginZ'), dtype=np.float) * fsf
        self.set_column_data('_rlnOriginZ', hold)

    # Checks if a column already exist
    def is_column(self, key):
        return key in self.__cols

    # Check if a column key is compatible with Relion
    def is_relion_compatible(self, key):
        return isinstance(key, str) and self.__checker.is_valid(key) and (len(key)>=4) and (key[:4] == '_rln')

    # Counts the number of elements which satisfied several (key, val) pairs
    # pairs: list of (key,val) conditions
    def count_elements(self, pairs):

        # Intialization
        count = 0

        # All rows loop
        for row in range(self.get_nrows()):

            # Pairs condition loop
            is_hit = True
            for (pkey, pval) in pairs:
                try:
                    val = self.get_element(pkey, row)
                except KeyError:
                    is_hit = False
                    break
                if val != pval:
                    is_hit = False
                    break

            # Count if all conditions are true
            if is_hit:
                count += 1

        return count


    # Add a new column, if it already existed their values are overwritten
    # key: string key for the column, only Relion accepted key strings are valid
    # val: (default -1) if scalar value set to all rows, if iterable and has the same size of the number of rows
    #       allows to set different values to every row
    # no_parse: if True (default False) the column name is not parsed
    def add_column(self, key, val=-1, no_parse=False):

        if (no_parse is True) or (self.__checker.is_valid(key)):
            dtype = self.__checker.get_dtype(key)
            if dtype is None:
                dtype = float

            # Fill up the new row
            row = list()
            if hasattr(val, '__len__'):
                if len(val) != self.__rows:
                    error_msg = 'For multiple input values their length, ' + str(len(val)) + ' must aggree' + \
                                ' with the current number of rows, ' + str(self.__rows)
                    raise PySegInputError(expr='add_column (Star)', msg=error_msg)
                for v in val:
                    row.append(dtype(v))
            else:
                for i in range(self.__rows):
                    row.append(dtype(val))

            # Add or overwrite the column
            self.__data[key] = row
            idx = None
            try:
                idx = self.__cols.index(key)
            except ValueError:
                self.__cols.append(key)
                self.__dtypes.append(dtype)
            if idx is not None:
                self.__cols[idx] = key
                self.__dtypes[idx] = dtype

        else:
            error_msg = 'Column name ' + key + ' not accepted'
            raise PySegInputError(expr='add_column (Star)', msg=error_msg)

    # Add a new row (data entry), since no default values are imposed this call must contain a value for
    # all columns, every column key and data pair is introduced via kwargs
    def add_row(self, **kwargs):
        if kwargs is not None:
            keys, values = kwargs.keys(), kwargs.values()
            if len(keys) != self.get_ncols():
                error_msg = 'Number of columns introduced for this row, ' + str(len(keys)) + ' does not ' + \
                    ' fit the current number of columns, ' + str(self.get_ncols())
                raise PySegInputError(expr='add_row (Star)', msg=error_msg)
            for key, value in zip(keys, values):
                if self.is_column(key):
                    dtype = self.get_column_type(key)
                    self.__data[key].append(dtype(value))
                else:
                    error_msg = 'Column name ' + key + ' not present.'
                    raise PySegInputError(expr='add_row (Star)', msg=error_msg)
            self.__rows += 1

    # Delete a column
    # key: key of the column to delete
    def del_column(self, key):
        if self.is_column(key):
            idx = self.__cols.index(key)
            self.__cols.pop(idx)
            self.__dtypes.pop(idx)
            del self.__data[key]

    # Delete a set of rows
    # ids: list with the indices of the rows to delete
    def del_rows(self, ids):

        # Input parsing
        if not hasattr(ids, '__len__'):
            error_msg = 'Input ids must be a list.'
            raise PySegInputError(expr='del_rows (Star)', msg=error_msg)
        if len(ids) == 0:
            return

        # Temporary data initialization
        hold_data, hold_rows = copy.copy(self.__data), self.__rows
        self.__rows = 0
        for key in hold_data:
            self.__data[key] = list()
        ids_lut = np.zeros(shape=hold_rows, dtype=np.bool)
        for idx in ids:
            try:
                ids_lut[idx] = True
            except IndexError:
                pass

        # Loop for deleting rows
        for i in range(hold_rows):
            if not ids_lut[i]:
                # Copy row in attributes
                for key in self.__data:
                    self.__data[key].append(hold_data[key][i])
                self.__rows += 1

    # Copy data from one column to one another, if the last one did exist it is created
    # key_in: key string for input column
    # key_out: key string for output column
    # func: function (func(colum_vals)) to apply to data (default None)
    def copy_data_columns(self, key_in, key_out, func=None):
        vals = self.get_column_data(key_in)
        if vals is None:
            error_msg = 'Input column with ' + key_in + ' does not exist'
            raise PySegInputError(expr='self (Star)', msg=error_msg)
        if func is not None:
            vals = func(vals)
        self.add_column(key_out, val=vals)

    # Randomize data in a column (data will be overwriten)
    # key_in: key string with the column to randomize
    # mn_val: minimum value
    # mx_val: maximum value
    def rnd_data_column(self, key_in, mn_val, mx_val):
        nrows = self.get_nrows()
        rnd_vals = (mx_val-mn_val)*np.random.random((nrows,)) + mn_val
        self.add_column(key_in, val=rnd_vals.tolist())

    # Generates a TomoPeaks object where columns will be associated to peak attributes (Peaks coordinates are
    # set to '_rlnCoordinate(i)')
    # tomo: reference tomogram full path (only the peaks of this tomogram will be considered)
    # klass: if not None (default), it allows to pick the elements of some classes, it can be a list
    # orig: if True (default False), peaks coordinates are shifted according '_rlnOrigin(i)' columns
    # full_path: if True (default) for searching particles in the tomogram the input full path is considered,
    #            otherwise just the file name
    # micro: if True (default) the particle micrograph is checked
    def gen_tomo_peaks(self, tomo, klass=None, orig=True, full_path=True, micro=True):

        # Input parsing
        in_tomo = str(tomo)
        try:
            hold_tomo = disperse_io.load_tomo(in_tomo, mmap=True)
        except pexceptions as pe:
            raise pe
        if not full_path:
            in_tomo = os.path.split(in_tomo)[1]
        if (klass is not None) and (not hasattr(klass, '__len__')):
            in_klass = (klass,)
        else:
            in_klass = klass
        tpeaks = TomoPeaks(shape=hold_tomo.shape, name=in_tomo)
        del hold_tomo

        # Rows loop
        pid = 0
        for i in range(self.__rows):

            # Check if particle in tomogram
            if full_path:
                hold_str = self.get_element('_rlnMicrographName', i)
            else:
                hold_str = os.path.split(self.get_element('_rlnMicrographName', i))[1]
            if (not micro) or (hold_str == in_tomo) :
                # Check if it is in classes list
                if (klass is None) or (self.get_element('_rlnClassNumber', i) in klass):

                    # Crate a new peak
                    x, y, z = self.get_element('_rlnCoordinateX', i), self.get_element('_rlnCoordinateY', i), \
                              self.get_element('_rlnCoordinateZ', i)
                    if orig:
                        try:
                            ox = self.get_element('_rlnOriginX', i)
                        except KeyError:
                            ox = 0.
                        try:
                            oy = self.get_element('_rlnOriginY', i)
                        except KeyError:
                            oy = 0.
                        try:
                            oz = self.get_element('_rlnOriginZ', i)
                        except KeyError:
                            oz = 0.
                        x -= ox
                        y -= oy
                        z -= oz

                    # Add the peak to TomoPeaks
                    tpeaks.add_peak((x, y, z))

                    # Add STAR file attributes
                    if tpeaks.get_num_peaks() == 1:
                        for key in self.get_column_keys():
                            tpeaks.add_prop(key, 1, dtype=self.get_column_type(key))
                    for key in self.get_column_keys():
                        tpeaks.set_peak_prop(pid, key, self.get_element(key, i))
                    pid += 1

        return tpeaks

    # fname:
    def load(self, fname, nocol_raise=True):
        """
        Loading and input STAR file
        :param fname: full path to the input Star file to read
        :param nocol_raise: if True (default) a PySegInputError is raised if a column is not known, otherwise it just
                            not read
        :return:
        """

        # File reading
        lines = None
        with open(fname, 'r') as ffile:
            lines = ffile.readlines()
            ffile.close()
        if lines is None:
            error_msg = 'File ' + fname + ' was empty.'
            raise PySegInputError(expr='load (Star)', msg=error_msg)
        lidx = 0

        # Parse Header 1
        self.__header_1 = list()
        while (lidx < len(lines)) and (lines[lidx][0] != '_'):
            self.__header_1.append(lines[lidx])
            lidx += 1

        # Parse Header 2
        self.__cols, col_ids, col_count = list(), dict(), 0
        while (lidx < len(lines)) and (lines[lidx][0] == '_'):
            col = lines[lidx].split()[0]
            if self.__checker.is_valid(col) and (not self.is_column(col)):
                self.__data[col] = list()
                self.__dtypes.append(self.__checker.get_dtype(col))
                self.__cols.append(col)
                col_ids[col] = col_count
            else:
                if nocol_raise:
                    error_msg = 'Unexpected error parsing star file ' + fname + ' header.'
                    raise PySegInputError(expr='load (Star)', msg=error_msg)
            col_count += 1
            lidx += 1

        # Parse data rows
        for line in lines[lidx:]:
            datas = line.split()
            # EOF condition
            if self.get_ncols() > len(datas):
                break
            for (dtype, col) in zip(self.__dtypes, self.__cols):
                try:
                    self.__data[col].append(dtype(datas[col_ids[col]]))
                except ValueError:
                    if (dtype is int):
                        self.__data[col].append(int(float(datas[col_ids[col]])))
                    else:
                        raise ValueError
            self.__rows += 1

        self.__root_dir, self.__fname = os.path.split(fname)

    # Parse class attributes into a STAR format
    # Returns: an string with the STAR file
    def to_string(self):
        hold = str()
        for line in self.__header_1:
            hold += line
        for i, key in enumerate(self.__cols):
            hold += (key + ' ' + '#' + str(i+1) + '\n')
        if self.get_ncols() > 0:
            keys = self.get_column_keys()
            for i in range(self.get_nrows()):
                for key in keys[:-1]:
                    hold += (str(self.__data[key][i]) + '\t')
                hold += (str(self.__data[keys[-1]][i]) + '\n')
        return hold + '\n'

    # Store in a STAR
    # fname: output file name
    # sv: if not None (default), it specified shape (all dimensions must be even) for subvolumes that will be stored in a sub-folder called 'sub'
    #     in the same directory as the STAR file from reference tomograms (density is inverted and normalized)
    # mask: mask applied for sub-volume normalization (default None)
    # swap_xy: swap XY coordinate only for subvolume extraction (only applies if sv is not None)
    def store(self, fname, sv=None, mask=None, swap_xy=False, del_ang=(0,0,0)):

        # Sub-volumes
        if sv is not None:

            # Parsing input shape
            if (not hasattr(sv, '__len__')) or (not(len(sv) == 3)) \
                or (sv[0]<=0) or (sv[1]<=0) or (sv[2]<=0):
                error_msg = 'Subvolume shape must be 3-tuple with non trivial values.'
                raise PySegInputError(expr='store (Star)', msg=error_msg)
            if ((sv[0]%2) != 0) or ((sv[1]%2) != 0) or ((sv[2]%2) != 0):
                error_msg = 'All subvolume dimensions must be even, current ' + str(sv)
                raise PySegInputError(expr='store (Star)', msg=error_msg)
            hl_x, hl_y, hl_z = int(sv[1]*.5), int(sv[0]*.5), int(sv[2]*.5)
            tomo_path = ''
            sv_path = os.path.split(fname)[0] + '/sub'
            try:
                os.makedirs(sv_path)
            except OSError as exception:
                if exception.errno != errno.EEXIST:
                    error_msg = 'Directory ' + sv_path + ' could not be created'
                    raise PySegInputError(expr='store (Star)', msg=error_msg)


            # Rows loop
            for i in range(self.__rows):

                # Particle coordinate
                x, y, z = self.__data['_rlnCoordinateX'][i], self.__data['_rlnCoordinateY'][i], \
                          self.__data['_rlnCoordinateZ'][i]


                # Read tomogram map (if required only)
                hold_path = self.__data['_rlnMicrographName'][i]
                if hold_path != tomo_path:
                    tomo_path = hold_path
                    try:
                        tomo = disperse_io.load_tomo(tomo_path, mmap=True)
                    except KeyError:
                        error_msg = 'Reference tomogram ' + tomo_path + ' could not be read'
                        raise PySegInputError(expr='save_subvolumes (ParticleList)', msg=error_msg)

                # Cropping
                x_l, y_l, z_l = x-hl_x+1, y-hl_y+1, z-hl_z+1
                if (x_l < 0) or (y_l < 0) or (z_l < 0):
                    hold_sub = np.zeros(shape=sv, dtype=np.float32)
                x_h, y_h, z_h = x+hl_x+1, y+hl_y+1, z+hl_z+1
                if (x_l >= tomo.shape[0]) or (y_l >= tomo.shape[1]) or (z_l >= tomo.shape[2]):
                    hold_sub = np.zeros(shape=sv, dtype=np.float32)
                if swap_xy:
                    hold_sub = tomo[x_l:x_h, y_l:y_h, z_l:z_h]
                else:
                    hold_sub = tomo[x_l:x_h, y_l:y_h, z_l:z_h]

                # Normalization
                hold_sub = relion_norm(hold_sub, mask=mask)

                # Storing the sub-volume and updating image name
                part_name = sv_path + '/' + os.path.split(self.__data['_rlnImageName'][i])[1]
                disperse_io.save_numpy(hold_sub, part_name)
                self.__data['_rlnImageName'][i] = part_name

        # Angles deletion
        if del_ang[0] > 0:
            self.__data['_rlnAngleRot'][i] = 0
        if del_ang[1] > 0:
            self.__data['_rlnAngleTilt'][i] = 0
        if del_ang[2] > 0:
            self.__data['_rlnAnglePsi'][i] = 0

        # STAR file
        with open(fname, 'w') as ffile:
            ffile.write(self.to_string())

    # Compute alignment against a reference star file
    # ref_star: reference STAR file, with the same (at least partially) particles
    # ref_vect: reference vector
    # Returns: two arrays with length equal to the number of particles, first angles miss-alignment, second shifting.
    #          Negative values represent unexpected events.
    def compute_malign(self, ref_star, ref_vect):

        # Initialization
        n_parts = self.get_nrows()
        angs, shifts = (-1.)*np.ones(shape=n_parts, dtype=np.float32), (-1.)*np.ones(shape=n_parts, dtype=np.float32)
        a_ref_vect = np.mat(np.asarray(ref_vect, dtype=np.float32)).transpose()

        # Particles loop
        for i in range(n_parts):

            # Getting particle index in reference star file
            p_name = self.get_element('_rlnImageName', i)
            try:
                j = ref_star.find_element('_rlnImageName', p_name)
            except ValueError:
                continue

            # Geting particle information
            rot, psi, tilt = self.get_element('_rlnAngleRot', i), self.get_element('_rlnAnglePsi', i), \
                             self.get_element('_rlnAngleTilt', i)
            r_rot, r_psi, r_tilt = ref_star.get_element('_rlnAngleRot', j), ref_star.get_element('_rlnAnglePsi', j), \
                                   ref_star.get_element('_rlnAngleTilt', j)

            # Computing angle miss-alignment
            # mat_1 = rot_mat_relion(rot, psi, tilt, deg=True)
            # mat_2 = rot_mat_relion(r_rot, r_psi, r_tilt, deg=True)
            mat_1 = rot_mat_relion(rot, tilt, psi, deg=True)
            mat_2 = rot_mat_relion(r_rot, r_tilt, r_psi, deg=True)
            rot_v_1 = mat_1.T * a_ref_vect
            rot_v_2 = mat_2.T * a_ref_vect
            angs[i] = math.degrees(angle_2vec_3D(rot_v_1, rot_v_2))

            # print 'P=' + str((rot, tilt, psi)) + ', R=' + str((r_rot, r_tilt, r_psi)) + ', A=' + str(angs[i]) + 'deg'

            # Compute shift miss-alignment
            c_x, c_y, c_z = self.get_element('_rlnCoordinateX', i), self.get_element('_rlnCoordinateY', i), \
                            self.get_element('_rlnCoordinateZ', i)
            try:
                o_x, o_y, o_z = self.get_element('_rlnOriginX', i), self.get_element('_rlnOriginY', i), \
                                self.get_element('_rlnOriginZ', i)
            except KeyError:
                o_x, o_y, o_z = 0., 0., 0.
            coord = np.asarray((c_x-o_x, c_y-o_y, c_z-o_z), dtype=np.float32)
            c_x, c_y, c_z = ref_star.get_element('_rlnCoordinateX', j), ref_star.get_element('_rlnCoordinateY', j), \
                            ref_star.get_element('_rlnCoordinateZ', j)
            try:
                o_x, o_y, o_z = ref_star.get_element('_rlnOriginX', j), ref_star.get_element('_rlnOriginY', j), \
                                ref_star.get_element('_rlnOriginZ', j)
            except KeyError:
                o_x, o_y, o_z = 0., 0., 0.
            coord_r = np.asarray((c_x-o_x, c_y-o_y, c_z-o_z), dtype=np.float32)
            hold = coord - coord_r
            shifts[i] = math.sqrt((hold * hold).sum())

        return angs, shifts

    # Group particles into groups
    # min_gp: (default None) first particles are splited into groups by the their micrograph, if they don't reach min_gp
    #         values their are gathered the smallest group size first criteria
    # Returns: the column '_rlnGroupNumber' is created or updated
    def particle_group(self, min_gp=None):

        # Initialization
        try:
            curr_gp = list(set(self.get_column_data('_rlnGroupNumber')))
            # print str(curr_gp)
            nparts_gp = np.zeros(shape=len(curr_gp), dtype=np.int).tolist()
            for row in range(self.get_nrows()):
                gp = self.get_element('_rlnGroupNumber', row)
                idx = curr_gp.index(gp)
                nparts_gp[idx] += 1
        except ValueError:
            try:
                mics = self.get_column_data('_rlnImageName')
                curr_gp = np.arange(len(mics)).tolist()
                nparts_gp = np.zeros(shape=len(curr_gp), dtype=np.int).tolist()
                for row in range(self.get_nrows()):
                    mic = self.get_element('_rlnImageName', row)
                    nparts_gp[mics.index(mic)] += 1
            except ValueError:
                curr_gp = [1,]
                nparts_gp = (self.get_nrows(),)
            self.add_column('_rlnGroupNumber', val=curr_gp[0])

        # Loop until the gathering is finished
        lut_gp = np.arange(0, np.asarray(curr_gp, dtype=np.int).max()+1)
        while len(set(curr_gp)) > 1:
            # Find the smallest group
            min_ids = np.argsort(np.asarray(nparts_gp, dtype=np.int))
            if (nparts_gp[min_ids[0]] > min_gp) or (len(min_ids) <= 1):
                break
            # Find the pair to gather
            for i in range(len(lut_gp)):
                if lut_gp[i] == curr_gp[min_ids[0]]:
                    lut_gp[i] = curr_gp[min_ids[1]]
            lut_gp[curr_gp[min_ids[0]]] = curr_gp[min_ids[1]]
            nparts_gp[min_ids[1]] += nparts_gp[min_ids[0]]
            # Deleting groups
            curr_gp.pop(min_ids[0])
            nparts_gp.pop(min_ids[0])

        # Setting the groups
        new_gp = np.zeros(shape=self.get_nrows(), dtype=np.int)
        for row in range(self.get_nrows()):
            old_gp = self.get_element('_rlnGroupNumber', row)
            new_gp[row] =  lut_gp[old_gp]
        self.set_column_data('_rlnGroupNumber', new_gp)

    # Check if two (self and input) are comparable, i.e. have the same particles in the same location
    def check_comparable(self, star):
        if self.get_nrows() != star.get_nrows():
            return False
        else:
            for i in range(self.get_nrows()):
                if self.get_element('_rlnImageName', i) != star.get_element('_rlnImageName', i):
                    return False
        return True

    # Returns a list of STAR file objects one for each class
    def split_class(self):

        if not self.has_column('_rlnClassNumber'):
            error_msg = 'No _rlnClassNumber column found!'
            raise PySegInputError(expr='split_class (Star)', msg=error_msg)

        else:
            classes = self.get_column_data('_rlnClassNumber')
            class_ids = set(classes)
            stars, classes = list(), np.asarray(classes, dtype=np.int)
            keys = self.get_column_keys()
            for class_id in class_ids:
                star = Star()
                for key in keys:
                    star.add_column(key)
                rows = np.where(classes == class_id)[0]
                for row in rows:
                    kwargs = dict()
                    for key in keys:
                        kwargs[key] = self.get_element(key, row)
                    star.add_row(**kwargs)
                stars.append(star)
            return stars

    # Returns a copy parsed for being Relion compatible
    def get_relion_parsed_copy(self):
        hold_star = copy.deepcopy(self)
        for key in self.get_column_keys():
            if not self.is_relion_compatible(key):
                hold_star.del_column(key)
        return hold_star

    def from_TomoPeaks(self, in_plist, svols_path=None, gray_inv=False, ctf_path=None, del_aln=False, ref=None,
                       keep_pytom=False):
        """
        Initializes the Star object from the information contained by a particle list
        # VERY IMPORTANT: previous information is forgotten
        :param in_plist: input ParticleList object
        :param svols_path: path to store the sub-volumes in MRC format, if None (default) then the sub-volume paths are
                           kept
        :param gray_inv: if True (default) and svol_path is not None, the the sub-volumes are inverted and renormalized
                         before being stored in the new path
        :param ctf_path: path to a external CTF sub-volume (default None)
        :param del_aln: if True the alignment information is deleted (default False)
        :param ref: if not None (default) it contains the reference model for all particles, it requiere svols_path not
                    None in order to know where to store the transformed particles
        :param keep_pytom: if True (default False) then rigid transformation are keep as PyTom forma instead of being
                           converted to Relion
        :return: the Star object created
        """

        # Input parsing
        svol_ref = None
        if ref is not None:
            if svols_path is None:
                error_msg = 'If ref is providen svols_path must be aslo provided'
                raise PySegInputError(expr='from_TomoPeaks (Star)', msg=error_msg)
            svol_ref = disperse_io.load_tomo(ref)
            if gray_inv:
                svol_ref = relion_norm(svol_ref, mask=None)
                ref_cent = .5 * np.asarray(svol_ref.shape, dtype=np.float32)
                ref_stem = os.path.splitext(os.path.split(ref)[1])[0]

        # Initialization (clean former information first)
        self.__data = {}
        self.__dtypes = list()
        self.__rows = 0
        self.__cols = list()

        # Creating expected colums
        self.add_column('_rlnImageName')
        if ctf_path is not None:
            self.add_column('_rlnCtfImage')
        self.add_column('_rlnCoordinateX')
        self.add_column('_rlnCoordinateY')
        self.add_column('_rlnCoordinateZ')
        self.add_column('_rlnOriginX')
        self.add_column('_rlnOriginY')
        self.add_column('_rlnOriginZ')
        self.add_column('_rlnAngleRot')
        self.add_column('_rlnAngleTilt')
        self.add_column('_rlnAnglePsi')

        # Add particles
        for i, part in enumerate(in_plist._ParticleList__elem.iter(tag='Particle')):

            # The filename will work as particle unique identifier so its presence is a must
            try:
                fname = part.attrib['Filename']
            except KeyError:
                fname = None
            if svols_path is None:
                if fname is None:
                    continue
            else:
                if ref is None:
                    svol = disperse_io.load_tomo(fname)
                    if gray_inv:
                        svol = relion_norm(svol, mask=None)
                    fname = svols_path + '/' + os.path.splitext(os.path.split(fname)[1])[0] + '.mrc'
                    disperse_io.save_numpy(svol, fname)

                # Parsing just peak useful information
            pos = part.find('PickPosition')
            if (pos is not None) and (fname is not None):
                try:
                    y, x, z = float(pos.attrib['X']), float(pos.attrib['Y']), float(pos.attrib['Z'])
                except KeyError:
                    print 'WARNING: gen_TomoPeaks() (ParticleList), a Particle without Pick position cannot be converted into a Peak'
                    continue
                rot_eu = part.find('Rotation')
                rot, tilt, psi = 0, 0, 0
                if rot_eu is not None:
                    try:
                        if del_aln:
                            rot, tilt, psi = 0, 0, 0
                        else:
                            eu_z1, eu_x, eu_z2 = float(rot_eu.attrib['Z1']), float(rot_eu.attrib['X']), \
                                                 float(rot_eu.attrib['Z2'])
                            if keep_pytom:
                                rot, tilt, psi = eu_z1, eu_x, eu_z2
                            else:
                                M_hold = rot_mat_pytom(phi=-eu_z2, psi=-eu_z1, the=-eu_x)
                                rot, tilt, psi = rot_mat_eu_relion(M_hold, deg=True)
                    except KeyError:
                        pass
                shift = part.find('Shift')
                shift_x, shift_y, shift_z = 0, 0, 0
                if shift is not None:
                    try:
                        if del_aln:
                            shift_x, shift_y, shift_z = 0, 0, 0
                        else:
                            if keep_pytom:
                                shift_x, shift_y, shift_z = -float(shift.attrib['X']), -float(shift.attrib['Y']), \
                                                            -float(shift.attrib['Z'])
                            else:
                                shift_x, shift_y, shift_z = -float(shift.attrib['X']), -float(shift.attrib['Y']), \
                                                            -float(shift.attrib['Z'])
                    except KeyError:
                        pass

                if svol_ref is not None:
                    # Rotation
                    r3d_a = pyto.geometry.Rigid3D()
                    if keep_pytom:
                        svol = rotate3d_pytom(svol_ref, phi=rot, psi=psi, the=tilt, center=ref_cent, order=2)
                    else:
                        r3d_a.q = r3d_a.make_r_euler(angles=np.radians(np.asarray((rot, tilt, psi), dtype=np.float)),
                                                     mode='zyz_in_passive')
                        try:
                            svol = r3d_a.transformArray(svol_ref, origin=ref_cent, order=3, prefilter=True, mode='reflect')
                        except np.linalg.linalg.LinAlgError:
                            print('WARNING: particle in row ' + str(i) + ', with angles' + str((rot, tilt, psi)) +
                                  ' could not be rotated.')
                            continue
                    # Translation
                    svol = tomo_shift(svol, (-shift_x, -shift_y, -shift_z))
                    # Storing
                    fname = svols_path + '/' + ref_stem + '_' + str(i) + '.mrc'
                    disperse_io.save_numpy(svol, fname)

                # Setting new element properties
                if ctf_path is None:
                    kwargs = {'_rlnImageName':fname,
                              '_rlnCoordinateX':x, '_rlnCoordinateY':y, '_rlnCoordinateZ':z,
                              '_rlnOriginX':shift_x, '_rlnOriginY':shift_y, '_rlnOriginZ':shift_z,
                              '_rlnAngleRot':rot, '_rlnAngleTilt':tilt, '_rlnAnglePsi':psi}
                else:
                    kwargs = {'_rlnImageName': fname, '_rlnCtfImage': ctf_path,
                              '_rlnCoordinateX': x, '_rlnCoordinateY': y, '_rlnCoordinateZ': z,
                              '_rlnOriginX': shift_x, '_rlnOriginY': shift_y, '_rlnOriginZ': shift_z,
                              '_rlnAngleRot': rot, '_rlnAngleTilt': tilt, '_rlnAnglePsi': psi}
                self.add_row(**kwargs)

    # Swap XY coordinates (also OriginX-Y)
    def swap_xy(self):
        for row in range(self.get_nrows()):
            try:
                hold_x, hold_y = self.get_element('_rlnCoordinateX', row), self.get_element('_rlnCoordinateY', row)
                self.set_element('_rlnCoordinateX', row, hold_y)
                self.set_element('_rlnCoordinateY', row, hold_x)
            except KeyError:
                pass
            try:
                hold_x, hold_y = self.get_element('_rlnOriginX', row), self.get_element('_rlnOriginY', row)
                self.set_element('_rlnOriginX', row, hold_y)
                self.set_element('_rlnOriginY', row, hold_x)
            except KeyError:
                pass

    def compute_avg(self, pytom=False):
        """
        Compute the average sub-volume (without considering any CTF correction)e
        :param pytom: if True then PyTom convention is used intead the default Relion
        :return: a 3D numpy array with the averaged volume, None if no particles for averging
        """
        # Initialization
        n_rows = self.get_nrows()
        if n_rows <= 0:
            return None
        svol = disperse_io.load_tomo(self.get_element('_rlnImageName', 0))
        avg = np.zeros(shape=svol.shape, dtype=np.float)

        # Particles loop
        count = 0
        for row in range(self.get_nrows()):

            # Loading the particle
            svol = disperse_io.load_tomo(self.get_element('_rlnImageName', row))
            svol_cent = .5 * np.asarray(svol.shape, dtype=np.float32)

            # Translation
            try:
                ox = self.get_element('_rlnOriginX', row)
            except KeyError:
                ox = 0
            try:
                oy = self.get_element('_rlnOriginY', row)
            except KeyError:
                oy = 0
            try:
                oz = self.get_element('_rlnOriginZ', row)
            except KeyError:
                oz = 0
            svol = tomo_shift(svol, (ox, oy, oz))

            # Rotation
            try:
                rot = self.get_element('_rlnAngleRot', row)
            except KeyError:
                rot = 0
            try:
                tilt = self.get_element('_rlnAngleTilt', row)
            except KeyError:
                tilt = 0
            try:
                psi = self.get_element('_rlnAnglePsi', row)
            except KeyError:
                psi = 0
            # svol = tomo_rot(svol, (rot, tilt, psi), conv='relion', active=False)
            r3d_a = pyto.geometry.Rigid3D()
            if pytom:
                svol = rotate3d_pytom(svol, phi=-psi, psi=-rot, the=-tilt, center=svol_cent, order=2)
            else:
                r3d_a.q = r3d_a.make_r_euler(angles=np.radians(np.asarray((rot, tilt, psi), dtype=np.float)),
                                             mode='zyz_in_active')
                # print(str(row) + ': ' + str((rot, tilt, psi)) + ' => ' + str(r3d_a.q))
                try:
                    svol = r3d_a.transformArray(svol, origin=svol_cent, order=2, prefilter=True, mode='reflect')

                except np.linalg.linalg.LinAlgError:
                    print('WARNING: particle in row ' + str(row) + ', with angles' + str((rot, tilt, psi)) +
                          ' could not be rotated.')
                    continue
            # Average
            avg += svol
            count += 1

        return avg / float(count)

