"""
Set of classes for dealing with a particle list

# Author: Antonio Martinez-Sanchez (Max Planck Institute for Biochemistry)
# Date: 1.06.16
"""

__author__ = 'Antonio Martinez-Sanchez'

import vtk
import csv
import errno
from variables import *
from pyorg.globals import *
import pyorg.disperse_io as disperse_io
import numpy as np
from abc import *
import xml.etree.ElementTree as ET
from xml.dom import minidom
# from star import Star
from plist import *

from pyorg import pexceptions


#### Global variables

DEG_EPS = 0.1 # degrees
V_EPS = 0.1

#### Helper functionality

# For tomograms which suffer transformations 1st (phi, psi, the) rotation around center and 2nd cropping from a reference
# tomogram, this function from a coordinate of the transformed tomogram remaps it into reference tomogram
# X: input coordinate from transform tomogram (3D array)
# tr_sh: shape of the reference tomogram
# phi|psi|the: Euler angles in degrees according TOM convention used for rotating the reference tomogram
# off: offset used for cropping on the rotated tomogram (3D array)
def trans_to_ref_remap(X, tr_sh, phi, psi, the, off):

    # Uncropping
    C = .5*tr_sh - .5
    # T = off - C
    # Xu = X - T
    Xu = X + off

    Xc = Xu - C

    # Back-rotation
    R = rot_mat(phi, psi, the, deg=True)
    R_i = R.T
    Xb = np.asarray(R_i*Xc.reshape(3,1), dtype=np.float32).reshape(3)
    # Xb = np.asarray(R*Xc.reshape(3,1), dtype=np.float32).reshape(3)

    # print 'X='+str(X)+', Xu='+str(Xu)+'Xc='+str(Xc)+', Xb='+str(Xb)+', Xd='+str(Xb+C)

    # Set references tomogram's corner as origin
    return Xb + C

# Buid and returns a Particle form the xml (ElementTree) information, if fails if the ElementTree is does not
# follow XML format
def from_ElementTree_to_Particle(elem):

    # Filename
    fname_rt = elem.attrib['Filename']

    # Rotation
    rot_eu, rot_para = [0, 0, 0], 'ZXZ'
    rot = elem.find('Rotation')
    if rot is not None:
        try:
            rot_para = rot.attrib['Paradigm']
        except KeyError:
            pass
        try:
            rot_eu[0], rot_eu[1], rot_eu[2] = float(rot.attrib['Z1']), float(rot.attrib['Z2']), float(rot.attrib['X'])
        except KeyError:
            pass

    # Shift
    shift = [0, 0, 0]
    shi = elem.find('Shift')
    if shi is not None:
        try:
            shift[0], shift[1], shift[2] = float(shi.attrib['X']), float(shi.attrib['Y']), float(shi.attrib['Z'])
        except KeyError:
            pass

    # Pick position
    pos, pos_orig = [0, 0, 0], ''
    ppos = elem.find('PickPosition')
    if ppos is not None:
        try:
            pos_orig = ppos.attrib['Origin']
        except KeyError:
            pass
        try:
            pos[0], pos[1], pos[2] = float(ppos.attrib['X']), float(ppos.attrib['Y']), float(ppos.attrib['Z'])
        except KeyError:
            pass

    # Class
    cl = 0
    cls = elem.find('Class')
    if cls is not None:
        try:
            cl = int(cls.attrib['Name'])
        except KeyError:
            pass

    # Wedge
    smooth_rt, angle, cofr, tilt_rot = 0.0, [30,30], 0, [0,0,0]
    wed = elem.find('Wedge')
    if wed is not None:
        swed = wed.find('SingleTiltWedge')
        if swed is not None:
            try:
                smooth_rt = float(wed.attrib['Smooth'])
            except KeyError:
                pass
            try:
                angle[0], angle[1] = float(wed.attrib['Angle1']), float(wed.attrib['Angle2'])
            except KeyError:
                pass
            try:
                cofr = float(wed.attrib['CutoffRadius'])
            except KeyError:
                pass
            trot = swed.find('TiltAxisRotation')
            if trot is None:
                try:
                    tilt_rot[0], tilt_rot[1], tilt_rot[2] = float(swed.attrib['Z1']), float(swed.attrib['Z2']), \
                                                            float(swed.attrib['X'])
                except KeyError:
                    pass
    wedge = SingleTiltWedge(smooth=smooth_rt, angles=angle, cofr=cofr, tilt_rot=tilt_rot)

    # Score
    stype, rem_auto, val, smooth, rad, fname = 'xcfScore', False, 0.5, -1, 0, ''
    sco = elem.find('Score')
    if sco is not None:
        try:
            stype = sco.attrib['Type']
        except KeyError:
            pass
        try:
            if sco.attrib['RemoveAutocorr'] == 'True':
                rem_auto = True
        except KeyError:
            pass
        try:
            val = float(sco.attrib['Value'])
        except KeyError:
            pass
        pprio = sco.find('PeakPrior')
        if pprio is None:
            try:
                fname = pprio.attrib['Filename']
            except KeyError:
                pass
            try:
                rad = float(pprio.attrib['Radius'])
            except KeyError:
                pass
            try:
                smooth = float(pprio.attrib['Smooth'])
            except KeyError:
                pass
            trot = swed.find('TiltAxisRotation')
            if trot is None:
                try:
                    tilt_rot[0], tilt_rot[1], tilt_rot[2] = swed.attrib['Z1'], swed.attrib['Z2'], swed.attrib['X']
                except KeyError:
                    pass
    score = Score(stype=stype, rem_auto=rem_auto, val=val, smooth=smooth, rad=rad, fname=fname)

    return Particle(fname=fname_rt,rot_para=rot_para, rot_eu=rot_eu, shift=shift, pos=pos, orig_pos=pos_orig,
                    p_class=cl, wedge=wedge, score=score)

#### File global variables

###### CLASSES ###########################################################################

###########################################################################################
# Class for modelling a peak (structure localization)
###########################################################################################

class Peak(object):

    # coords: 3D coordinates they are stored as 3D vector property with key PK_COORDS
    def __init__(self, coords):
        self.__props = {}
        self.add_prop(PK_COORDS, coords)


    #### Set/Get methods area

    # input key must be a string
    def add_prop(self, key, val):
        assert isinstance(key, str)
        if isinstance(val, str):
            self.__props[key] = val
            return
        if hasattr(val, '__len__'):
            if len(val) > 1:
                self.__props[key] = np.asarray(val)
            else:
                self.__props[key] = val[0]
        else:
            self.__props[key] = val

    def get_prop(self, key):
        assert isinstance(key, str)
        try:
            return self.__props[key]
        except KeyError:
            error_msg = 'Non-valid key ' + key
            raise pexceptions.PySegInputError(expr='get_prop (Peak)', msg=error_msg)

    # Return all property keys a list()
    def get_prop_keys(self):
        return self.__props.keys()

    # Return all property values a list()
    def get_prop_vals(self):
        return self.__props.values()

    def get_prop_val(self, key):
        return self.__props[key]

    def set_prop_val(self, key, val):
        assert isinstance(key, str)
        if isinstance(val, str):
            self.__props[key] = val
            return
        curr = self.__props[key]
        if hasattr(curr, '__len__'):
            if hasattr(val, '__len__'):
                if len(val) != len(curr):
                    error_msg = 'Dimensions of input value (' + str(len(val)) + ') and previous value (' + \
                                str(len(curr)) + ') does not match!'
                    raise pexceptions.PySegInputError(expr='set_prop_val (Peak)', msg=error_msg)
                if len(val) > 1:
                    self.__props[key] = np.asarray(val)
                else:
                    self.__props[key] = val[0]
        else:
            if hasattr(val, '__len__'):
                if len(val) > 1:
                    error_msg = 'Dimensions of input value (' + str(len(val)) + ') and previous value (1) does not match!'
                    raise pexceptions.PySegInputError(expr='set_prop_val (Peak)', msg=error_msg)
                self.__props[key] = val[0]
            else:
                self.__props[key] = val

    # Equivalent to get_prop_val() but now returning a tuple() object is forced
    def get_prop_val_tuple(self, key):
        val = self.__props[key]
        if hasattr(val, '__len__'):
            return tuple(val)
        else:
            return val,

    def coords_swapxy(self):
        coords = self.get_prop_val(PK_COORDS)
        self.set_prop_val(PK_COORDS, (coords[1], coords[0], coords[2]))

    #### External functionality area

    # Returns peak properties as a string (separated by commas)
    # mode: 'key'-> key of the properties only, 'value'-> values only, and 'full' (default)-> two rows header and values
    def to_string(self, mode='full'):

        delimiter = ', '
        msg = str()

        # Headers row
        if (mode == 'key') or (mode == 'full'):
            for (prop, val) in zip(self.__props.keys(), self.__props.values()):
                msg += (prop + len(val)*delimiter)
            msg += '\n'

        # Values row
        if (mode == 'value') or (mode == 'full'):
            for val in self.__props.values():
                msg += (val + delimiter)
            msg += '\n'

        return msg

    # Returns peak properties or keys as a sequence of strings
    # mode: 'key'-> key of the properties only (default), 'value'-> values only
    def to_strings(self, mode='key'):

        msg = list()

        # Headers row
        if mode == 'key':
            for (prop, val) in zip(self.__props.keys(), self.__props.values()):
                if hasattr(val, '__len__'):
                    for i in range(len(val)):
                        msg.append(prop+'_'+str(i))
                else:
                    msg.append(prop)

        # Values row
        if mode == 'value':
            for val in self.__props.values():
                if hasattr(val, '__len__'):
                    for s in val:
                        msg.append(str(s))
                else:
                    msg.append(str(val))

        return msg

    # Return a dictionary
    # fmt_csv: if True (default False) vector properties are slitted in several properties (each per dimension) in
    #           order to be compatible with csv format (the key for every vector property is name_x wher x is the dimension (1,2,...)
    def to_dictionary(self, fmt_csv=False):
        if fmt_csv:
            dic = {}
            for (prop, val) in zip(self.__props.keys(), self.__props.values()):
                if hasattr(val, '__len__'):
                    for i in range(len(val)):
                        dic[prop+'_'+str(i)] = val[i]
                else:
                    dic[prop] = val
            return dic
        else:
            return self.__props

    # Check weather a property already exists
    def has_prop(self, key):
        if key in self.__props:
            return True
        else:
            return True


###########################################################################################
# Base class for modelling a set peaks contained by a tomogram (structure localization)
###########################################################################################

class TomoPeaks(object):

    # shape: 3-tuple or array with the dimensions of the parent tomogram
    # name: (default '') string name that identify the tomoram
    # mask: (default None) binary tomogram (0-bg otherwise-fg) for masking valid regions for peaks in tomogram
    def __init__(self, shape, name='', mask=None):
        if (not hasattr(shape, '__len__')) or (len(shape) != 3):
            error_msg = 'Input tomgram must have 3 dimensions.'
            raise pexceptions.PySegInputError(expr='__init__ (TomoPeaks)', msg=error_msg)
        if mask is None:
            self.__mask = np.ones(shape=shape, dtype=np.bool)
        else:
            if (not isinstance(mask, np.ndarray)) or (mask.shape != shape):
                error_msg = 'Input tomgram must have 3 dimensions.'
                raise pexceptions.PySegInputError(expr='__init__ (TomoPeaks)', msg=error_msg)
            self.__mask = mask.astype(np.bool)
        self.__name = str(name)
        self.__peaks = list()

    #### Set/Get methods area

    def get_peaks_list(self):
        return self.__peaks

    # Add a peak form its coordinates
    def add_peak(self, coords):
        peak = Peak(coords)
        if len(self.__peaks) > 0:
            for (key, val) in zip(self.__peaks[0].get_prop_keys(), self.__peaks[0].get_prop_vals()):
                if key != PK_COORDS:
                    if isinstance(val, str):
                        peak.add_prop(key, '')
                    elif hasattr(val, '__len__'):
                        hold_val = -1 * np.ones(shape=len(val), dtype=val.dtype)
                        peak.add_prop(key, hold_val)
                    else:
                        peak.add_prop(key, -1)
        self.__peaks.append(peak)
        return

    # Add in one call a list of peaks form its coordinates
    # l_coords: iterable list or coords
    def add_peaks(self, l_coords):
        for coords in l_coords:
            peak = Peak(coords)
            if len(self.__peaks) > 0:
                for (key, val) in zip(self.__peaks[0].get_prop_keys(), self.__peaks[0].get_prop_vals()):
                    if key != PK_COORDS:
                        peak.add_prop(key, val)
            self.__peaks.append(peak)
        return

    # Add a new property to all peaks
    # key: key property
    # n_comp: number of components of the property (default 1)
    # vals: values assigned to all peaks (default -1) it can be an array equal to the number of peaks
    # dtype: if not None (default) forces to cast al values to this type
    def add_prop(self, key, n_comp=1, vals=-1, dtype=None):
        n_peaks = self.get_num_peaks()
        if not hasattr(vals, '__len__'):
            if dtype is str:
                values = list()
                for i in range(n_peaks):
                    values.append('')
            elif dtype is not None:
                values = vals * np.ones(shape=(n_peaks, n_comp), dtype=dtype)
            else:
                values = vals * np.ones(shape=(n_peaks, n_comp))
        else:
            if hasattr(vals, '__len__') and (len(vals)>0) and (not hasattr(vals[0], '__len__'))\
                    and (len(vals) == n_comp):
                if dtype is None:
                    values = np.zeros(shape=(n_peaks, n_comp))
                    hold_vals = np.asarray(vals)
                elif dtype is str:
                    error_msg = 'String type cannot have more than one component'
                    raise pexceptions.PySegInputError(expr='add_prop (TomoPeaks)', msg=error_msg)
                else:
                    values = np.zeros(shape=(n_peaks, n_comp), dtype=dtype)
                    hold_vals = np.asarray(vals, dtype=dtype)
                for i in range(n_peaks):
                    values[i, :] = hold_vals
            elif len(vals) != n_peaks:
                error_msg = 'Input array of values has different length (' + str(len(vals)) \
                            + ') than the number of peaks (' + str(n_peaks) + ')'
                raise pexceptions.PySegInputError(expr='add_prop (TomoPeaks)', msg=error_msg)
            else:
                if dtype is None:
                    values = vals
                elif dtype is str:
                    values = list()
                    for val in vals:
                        values.append(val)
                else:
                    values = np.asarray(vals, dtype=dtype)
        if n_comp == 1:
            for (peak, value) in zip(self.__peaks, values):
                peak.add_prop(key, value)
        else:
            for (peak, value) in zip(self.__peaks, values):
                peak.add_prop(key, value[:n_comp])

    # pid: peak id (unique id and index on peaks list)
    def set_peak_prop(self, pid, key, val):
        if key not in self.__peaks[pid].get_prop_keys():
            error_msg = 'Property key ' + key + ' has not been added previously (see add_prop())'
            raise pexceptions.PySegInputError(expr='set_peak_prop (TomoPeaks)', msg=error_msg)
        self.__peaks[pid].set_prop_val(key, val)

    # Applies operations op(prop[key],val) to al peaks, where op is an operator of Python operator package
    def peaks_prop_op(self, key, val, op):
        if hasattr(val, '__len__'):
            l_val = len(val)
            for peak in self.__peaks:
                hold = peak.get_prop_val(key)
                res = np.zeros(shape=l_val, dtype=hold.dtype)
                for i in range(l_val):
                    res[i] = op(hold[i], val[i])
                peak.set_prop_val(key, res)
        else:
            for peak in self.__peaks:
                hold = peak.get_prop_val(key)
                res = op(hold, val)
                peak.set_prop_val(key, res)

    def get_num_peaks(self):
        return len(self.__peaks)

    # Returns the number of components of a property
    # key: property key
    def get_prop_ncomp(self, key):
        if self.get_num_peaks() <= 0:
            return 0
        if not self.has_prop(key):
            error_msg = 'Propertiy ' + key + ' not found.'
            raise pexceptions.PySegInputError(expr='get_prop_ncomp (TomoPeaks)', msg=error_msg)
        return len(self.__peaks[0].get_prop_val_tuple(key))

    # Returns a property values for all peaks in an array
    # key: property key
    # dtype: so as to impose property data type
    def get_prop_vals(self, key, dtype=None):
        n_peaks = self.get_num_peaks()
        if n_peaks <= 0:
            return None
        hold_val = self.__peaks[0].get_prop_val(key)
        if isinstance(hold_val, str):
            arr = list()
            for peak in self.__peaks:
                arr.append(peak.get_prop_val(key))
        else:
            try:
                n_comp = len(hold_val)
            except TypeError:
                n_comp = 1
            if n_comp == 1:
                arr = np.zeros(shape=n_peaks, dtype=dtype)
                for i, peak in enumerate(self.__peaks):
                    arr[i] = peak.get_prop_val(key)
            else:
                arr = np.zeros(shape=(n_peaks, n_comp), dtype=dtype)
                for i, peak in enumerate(self.__peaks):
                    arr[i, :] = peak.get_prop_val(key)

        return arr

    #### External functionality

    # Sort peaks according to property
    # key: key string for property used for sorting (must be an scalar)
    # ascend: if True (default) ascend order used, otherwise ascend
    def sort_peaks(self, key, ascend=True):

        # Getting property values
        vals = self.get_prop_vals(key)
        if len(vals.shape) > 1:
            error_msg = 'Input property ' + key + ' must be scalar valued'
            raise pexceptions.PySegInputError(expr='sort_peaks (TomoPeaks)', msg=error_msg)

        # Sorting
        ids_sort = np.argsort(vals)
        if not ascend:
            ids_sort = ids_sort[::-1]

        # Moving peaks list()
        hold_peaks = self.__peaks
        self.__peaks = list()
        for idx in ids_sort:
            self.__peaks.append(hold_peaks[idx])

    # Swap X and Y coordinates for all peaks
    def peaks_coords_swapxy(self):
        for peak in self.__peaks:
            peak.coords_swapxy()

    # Rotate peaks coordinates according some Euler angles
    # phi|psi|the: Euler angles 3-tuple (phi, psi, the)
    # deg: if True (default) angles are in degrees, otherwise radians
    # center: center used for the rotation (default (0,0,0))
    # conv: Euler angles convention, valid: 'tom' (Default) and 'relion'
    def rotate_coords(self, phi, psi, the, deg=True, key=PK_COORDS, center=(0,0,0), conv='tom'):

        if (conv != 'tom') and (conv != 'relion'):
            error_msg = 'Input convention ' + conv + ' is not valid.'
            raise pexceptions.PySegInputError(expr='rotate_vect (TomoPeaks)', msg=error_msg)

        # Build rotation matrix
        if conv == 'relion':
            R = rot_mat_relion(phi, psi, the, deg=deg)
        else:
            R = rot_mat(phi, psi, the, deg=deg)
        C = np.asarray(center, dtype=np.float32)

        # Loop for coordinates rotation
        for peak in self.__peaks:
            X = peak.get_prop_val(key) - C
            Xr = np.asarray(R*X.reshape(3,1), dtype=np.float32).reshape(3)
            peak.set_prop_val(key, Xr+C)

    # Rotate a vector property (number of components 3) with according another property which encodes Euler angles in
    # degress
    # key_vect: key string for vector the vector property
    # key_eu: key string for Euler angles property (in degrees) or a rotations 3-tuple
    # key_out: output property key string, if None (default) key_vect will be used, a new property will be created if
    #          necesary
    # inv: if True (default False) for invert rotations
    # conv: rotation convention, valid: 'tom' (Matlab, default), 'zyz' intrinsic and 'relion'
    def rotate_vect(self, key_vect, key_eu, key_out=None, inv=False, conv='tom'):

        # Input parsing
        if key_out is None:
            key_out = key_vect
        if not self.has_prop(key_out):
            self.add_prop(key_out, n_comp=3, dtype=np.float32)
        if (conv != 'tom') and (conv != 'zyz') and (conv != 'relion'):
            error_msg = 'Input convention ' + conv + ' is not valid.'
            raise pexceptions.PySegInputError(expr='rotate_vect (TomoPeaks)', msg=error_msg)
        if not isinstance(key_eu, str):
            if (not hasattr(key_eu, '__len__')) or (len(key_eu) != 3):
                error_msg = 'Euler angles must be a key property or a 3-tuple.'
                raise pexceptions.PySegInputError(expr='rotate_vect (TomoPeaks)', msg=error_msg)

        # Loop for peaks
        for peak in self.__peaks:
            if isinstance(key_eu, str):
                eu_angs = peak.get_prop_val(key_eu)
            else:
                eu_angs = np.asarray(key_eu, dtype=np.float32)
            if conv == 'tom':
                phi, psi, the = eu_angs[0], eu_angs[1], eu_angs[2]
                R = rot_mat(phi, psi, the, deg=True)
            elif conv == 'zyz':
                R = rot_mat_zyz(eu_angs[0], eu_angs[1], eu_angs[2], deg=True)
            elif conv == 'relion':
                R = rot_mat_relion(eu_angs[0], eu_angs[1], eu_angs[2], deg=True)
            if inv:
                R = R.T
            X = peak.get_prop_val(key_vect)
            Xr = np.asarray(R*X.reshape(3,1), dtype=np.float32).reshape(3)
            peak.set_prop_val(key_out, Xr)

    # Computes rotation angles of from an input normal so as to fit reference [0,0,1] vector
    # First Euler angle (Rotation) is assumed 0
    # key_v: key for input vectors
    # key_r: key for output Euler angles
    # v_ref: reference vector (default [0, 0, 1])
    # conv: convention, valid: 'relion' (default)
    # key_vo: if None omitted (default), otherwise property were rotated vectors after fitting input reference are
    #         stored, only with Debugging purposes, additional debugging messages are printed
    def vect_rotation_ref(self, key_v, key_r, v_ref=(0, 0, 1), conv='relion', key_vo=None):

        # Input parsing
        if not self.has_prop(key_v):
            error_msg = 'Input property ' + key_v + ' is not found.'
            raise pexceptions.PySegInputError(expr='vect_rotation_ref (TomoPeaks)', msg=error_msg)
        if (not hasattr(v_ref, '__len__')) or (len(v_ref) != 3):
            error_msg = 'Reference vector must be a 3-tuple.'
            raise pexceptions.PySegInputError(expr='vect_rotation_ref (TomoPeaks)', msg=error_msg)
        if conv != 'relion':
            error_msg = 'Input convention ' + conv + ' is not valid.'
            raise pexceptions.PySegInputError(expr='vect_rotation_ref (TomoPeaks)', msg=error_msg)
        self.add_prop(key_r, 3, vals=0, dtype=np.float32)
        u = np.asarray(v_ref, dtype=np.float32)
        try:
            u /= math.sqrt((u*u).sum())
        except ZeroDivisionError:
            error_msg = 'Any vector cannot be rotated to another on with module 0.'
            raise pexceptions.PySegInputError(expr='vect_rotation_ref (TomoPeaks)', msg=error_msg)
        I = np.identity(3, dtype=np.float32)
        if key_vo is not None:
            self.add_prop(key_vo, 3, vals=0, dtype=np.float32)

        for peak in self.__peaks:
            v_in = peak.get_prop_val(key_v)
            # Normalization
            v_m = np.asarray((v_in[1], v_in[0], v_in[2]), dtype=np.float32)
            try:
                n = v_m / math.sqrt((v_m*v_m).sum())
            except ZeroDivisionError:
                print 'WARNING (vect_rotation_ref): vector with module 0 cannot be rotated!'
                peak.set_prop_val(key_r, (0., 0., 0.))
                continue
            # Computing rotation matrix
            # Implementation of
            v = np.cross(n, u)
            s = math.sqrt((v*v).sum())
            c = np.dot(n, u)
            # Already aligned condition
            if c == 0:
                peak.set_prop_val(key_r, (0., 0., 0.))
                continue
            try:
                b = (1.-c) / (s*s)
            except ZeroDivisionError:
                peak.set_prop_val(key_r, (0., 0., 0.))
            D = np.matrix([[0, -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]], dtype=np.float32)
            R = I + D + (D*D*b)
            if key_vo is not None:
                hold = np.asarray(R*v_m.reshape(3,1), dtype=np.float32).reshape(3)
                hold_m = math.sqrt((hold*hold).sum())
                if hold_m > 0:
                    hold /= hold_m
                    peak.set_prop_val(key_vo, hold)
                    hold_e = hold - u
                    err = math.sqrt((hold_e*hold_e).sum())
                    if err > V_EPS:
                        print 'WARNING (vect_rotation_ref): (1) rotated vector ' + str(hold) + ' does not fit reference vector ' + str(u) + ' (EPS=' + str(err) + ')'
            # Computing Euler angles from rotation matrix
            if conv == 'relion':
                alpha, beta, gamma = rot_mat_eu_relion(R, deg=True)
                # alpha = 0.
            if key_vo is not None:
                R2 = rot_mat_relion(alpha, beta, gamma, deg=True)
                # R2 = rot_mat_relion(gamma, beta, alpha, deg=True)
                hold_e = R - R2
                hold_m = (hold_e*hold_e).sum()
                if hold_m > 0:
                    if math.sqrt(hold_m) > V_EPS:
                        print 'WARNING (vect_rotation_ref): rotation matrices do not fit:'
                        print '\tR1= ' + str(R)
                        print '\tR2= ' + str(R2)
                    hold = np.asarray(R2*v_m.reshape(3,1), dtype=np.float32).reshape(3)
                    hold_m = math.sqrt((hold*hold).sum())
                    if hold_m > 0:
                        hold /= hold_m
                        peak.set_prop_val(key_vo, hold)
                        hold_e = hold - u
                        err = math.sqrt((hold_e*hold_e).sum())
                        if err > V_EPS:
                            print 'WARNING (vect_rotation_ref): (2) rotated vector ' + str(hold) +' does not fit reference vector ' + str(u) + ' (EPS=' + str(err) + ')'
                            print '\tRot=' + str(alpha) + ', Tilt=' + str(beta) + ', Psi=' + str(gamma)
            # peak.set_prop_val(key_r, (alpha, beta, gamma))
            peak.set_prop_val(key_r, (alpha, beta, gamma))

    # Computes rotation angles of from an input vector to fit reference [0,0,1] vector having a free Euler angle
    # in Relion format
    # First Euler angle (Rotation) is assumed 0
    # key_v: key for input vectors
    # key_r: key for output Euler angles
    def vect_rotation_zrelion(self, key_v, key_r):

        # Input parsing
        if not self.has_prop(key_v):
            error_msg = 'Input property ' + key_v + ' is not found.'
            raise pexceptions.PySegInputError(expr='vect_rotation_ref (TomoPeaks)', msg=error_msg)
        self.add_prop(key_r, 3, vals=0, dtype=np.float32)
        zv = np.asarray((0., 0., 1.), dtype=np.float32)

        for peak in self.__peaks:
            v_in = peak.get_prop_val(key_v)
            # Normalization
            v_m = np.asarray((v_in[1], v_in[0], v_in[2]), dtype=np.float32)
            try:
                n = v_m / math.sqrt((v_m*v_m).sum())
            except ZeroDivisionError:
                print 'WARNING (vect_rotation_ref): vector with module 0 cannot be rotated!'
                peak.set_prop_val(key_r, (0., 0., 0.))
                continue
            # Computing angles in Extrinsic ZYZ system
            alpha = np.arccos(n[2])
            beta = np.arctan2(n[1], n[0])
            # Transform to Relion system (intrinsic ZY'Z'' where rho is free)
            # rot, tilt, psi = math.degrees(beta), math.degrees(alpha), 180.-math.degrees(beta)
            rot, tilt, psi = 0., unroll_angle(math.degrees(alpha), deg=True), \
                             unroll_angle(180.-math.degrees(beta), deg=True)
            # print rot, tilt, psi
            peak.set_prop_val(key_r, (rot, tilt, psi))
            # Checking the rotation angles are correctly estimated in Relion format
            M = rot_mat_relion(rot, tilt, psi, deg=True)
            v_h = np.asarray(M * n.reshape(3, 1), dtype=np.float).reshape(3)
            err = math.sqrt(((v_h - zv)*(v_h - zv)).sum())
            if err > V_EPS:
                print 'WARNING (vect_rotation_ref): (2) rotated vector ' + str(v_h) +' does not fit reference vector ' + str(zv) + ' (EPS=' + str(err) + ')'
                print '\tRot=' + str(rot) + ', Tilt=' + str(tilt) + ', Psi=' + str(psi)
            # print 'Ang (' + str(rot) + ', ' + str(tilt) + ', ' + str(psi) + ') error = ' + str(math.sqrt(((v_h - zv)*(v_h - zv)).sum()))

    # Returns vector between two 3-tupled properties
    # key_p1|2: keys for the two vector properties (they must have 3 components)
    # key_v: output property key string
    # Result: a new property is added to all peaks with the vectors and indexed by key_v=key_p1-key_p2
    def vect_2pts(self, key_p1, key_p2, key_v):

        # Input parsing
        if (not self.has_prop(key_p1)) or (not self.has_prop(key_p1)):
            error_msg = 'One (or both) of the input properties is not found.'
            raise pexceptions.PySegInputError(expr='vect_2pts (TomoPeaks)', msg=error_msg)
        if (self.get_prop_ncomp(key_p1) != 3) != (self.get_prop_ncomp(key_p2) != 3):
            error_msg = 'Input properties must have 3 components'
            raise pexceptions.PySegInputError(expr='vect_2pts (TomoPeaks)', msg=error_msg)
        self.add_prop(key_v, 3, vals=0, dtype=np.float32)

        # Loop for peaks
        for peak in self.__peaks:
            p1, p2 = peak.get_prop_val(key_p1), peak.get_prop_val(key_p2)
            peak.set_prop_val(key_v, p1-p2)

    # Returns shortest angle in degrees between two vector (3 components) properties
    # key_v1|2: keys for the two vector properties (they must have 3 components)
    # key_a: output property key string
    # Result: a new property is added to all peaks with the angles and idexed by key_a
    def vects_angle(self, key_v1, key_v2, key_a):

        # Input parsing
        if (not self.has_prop(key_v1)) or (not self.has_prop(key_v1)):
            error_msg = 'One (or both) of the input properties is not found.'
            raise pexceptions.PySegInputError(expr='vects_angle (TomoPeaks)', msg=error_msg)
        if (self.get_prop_ncomp(key_v1) != 3) != (self.get_prop_ncomp(key_v2) != 3):
            error_msg = 'Input properties must have 3 components'
            raise pexceptions.PySegInputError(expr='vects_angle (TomoPeaks)', msg=error_msg)
        self.add_prop(key_a, 1, vals=0, dtype=np.float32)

        # Loop for peaks
        for peak in self.__peaks:
            v1, v2 = peak.get_prop_val(key_v1), peak.get_prop_val(key_v2)
            peak.set_prop_val(key_a, math.degrees(angle_2vec_3D(v1, v2)))

    # For peak find the vector defined by the shortest distance to a segmented structure
    # tomo_seg: input tomogram with the segmentation (>0-fg, 0=bg)
    # key_v: property key
    # pt_mode: if 'in' defaults vector points to segmentation, otherwise vector orientation is inverted
    def seg_shortest_normal(self, tomo_seg, key_v, pt_mode='in'):

        # Input parsing
        try:
            shape = tomo_seg.shape
        except AttributeError:
            error_msg = 'Input tomogram must be a numpy.ndarray or derived class.'
            raise pexceptions.PySegInputError(expr='tomo_seg (TomoPeaks)', msg=error_msg)
        if len(shape) != 3:
            error_msg = 'Input tomogram must be 3D.'
            raise pexceptions.PySegInputError(expr='tomo_seg (TomoPeaks)', msg=error_msg)

        # Get peaks coordinates
        coords = self.get_prop_vals(PK_COORDS)

        # Distance transform
        _, img_ids = sp.ndimage.morphology.distance_transform_edt(np.invert(tomo_seg>0), return_indices=True)

        # Compute vector for every peak
        vects = np.zeros(shape=coords.shape, dtype=np.float32)
        pts = np.zeros(shape=coords.shape, dtype=np.float32)
        for i, coord in enumerate(coords):
            try:
                ix, iy, iz = img_ids[0, coord[0], coord[1], coord[2]], img_ids[1, coord[0], coord[1], coord[2]], \
                             img_ids[2, coord[0], coord[1], coord[2]]
            except IndexError:
                continue
            pt = np.asarray((ix, iy, iz), dtype=np.float32)
            if pt_mode == 'in':
                vects[i, :] = coord - pt
            else:
                vects[i, :] = pt - coord
            pts[i, :] = pt

        # Set the property
        self.add_prop(key_v, 3, vects, np.float32)

    # For peak find the shortest distantant point in a segmented structure
    # tomo_seg: input tomogram with the segmentation (>0-fg, 0=bg)
    # key_v: output property key
    # pt_mode: if 'in' defaults vector points to segmentation, otherwise vector orientation is inverted
    def seg_shortest_pt(self, tomo_seg, key_p):

        # Input parsing
        try:
            shape = tomo_seg.shape
        except AttributeError:
            error_msg = 'Input tomogram must be a numpy.ndarray or derived class.'
            raise pexceptions.PySegInputError(expr='tomo_seg (TomoPeaks)', msg=error_msg)
        if len(shape) != 3:
            error_msg = 'Input tomogram must be 3D.'
            raise pexceptions.PySegInputError(expr='tomo_seg (TomoPeaks)', msg=error_msg)

        # Get peaks coordinates
        coords = self.get_prop_vals(PK_COORDS)

        # Distance transform
        _, img_ids = sp.ndimage.morphology.distance_transform_edt(np.invert(tomo_seg>0), return_indices=True)

        # Compute vector for every peak
        vects = np.zeros(shape=coords.shape, dtype=np.float32)
        pts = np.zeros(shape=coords.shape, dtype=np.float32)
        for i, coord in enumerate(coords):
            try:
                ix, iy, iz = img_ids[0, coord[0], coord[1], coord[2]], img_ids[1, coord[0], coord[1], coord[2]], \
                             img_ids[2, coord[0], coord[1], coord[2]]
            except IndexError:
                continue
            pts[i, :] = np.asarray((ix, iy, iz), dtype=np.float32)
            
        # Set the property
        self.add_prop(key_p, 3, pts, np.float32)

    # Filter peak according to a scalar property (number of components one)
    # key_s: property key string
    # cte: constant to use as second operator
    # op: operator function
    def filter_prop_scalar(self, key_s, cte, op):
        hold_list = self.__peaks
        self.__peaks = list()
        for peak in hold_list:
            val = peak.get_prop_val(key_s)
            if not op(val, cte):
                self.__peaks.append(peak)

    # Check weather a property already exists
    def has_prop(self, key):
        if self.get_num_peaks() < 1:
            return False
        return self.__peaks[0].has_prop(key)

    # Generates a vtp object from the set of peaks
    def to_vtp(self):

        # Initialization
        poly = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        verts = vtk.vtkCellArray()
        vtk_ps = list()
        # For peaks IDs
        vtk_id = vtk.vtkIntArray()
        vtk_id.SetName(STR_PEAK_ID)
        vtk_id.SetNumberOfComponents(1)

        # Getting peaks geometry and properties
        if len(self.__peaks) > 0:
            keys, vals = self.__peaks[0].get_prop_keys(), self.__peaks[0].get_prop_vals()
            # Coordinates (first property called PK_COORDS) are extracted because they contains Geometry information
            pk_id = keys.index(PK_COORDS)
            del keys[pk_id]
            del vals[pk_id]
            for (val, key) in zip(vals, keys):
                if not isinstance(val, str):
                    hold_p = disperse_io.TypesConverter().numpy_to_vtk(np.asarray(val).dtype)
                    hold_p.SetName(key)
                    if hasattr(val, '__len__'):
                        hold_p.SetNumberOfComponents(len(val))
                    else:
                        hold_p.SetNumberOfComponents(1)
                    vtk_ps.append(hold_p)

            # Adding geometry and topology
            for i, peak in enumerate(self.__peaks):
                points.InsertPoint(i, peak.get_prop_val(PK_COORDS))
                verts.InsertNextCell(1)
                verts.InsertCellPoint(i)
                vtk_id.InsertTuple(i, (i,))
                for vtk_p in vtk_ps:
                    val = peak.get_prop_val(vtk_p.GetName())
                    if hasattr(val, '__len__'):
                        vtk_p.InsertTuple(i, tuple(val))
                    else:
                        vtk_p.InsertTuple(i, (val,))

        # Finalizing poly
        poly.SetPoints(points)
        poly.SetVerts(verts)
        poly.GetPointData().AddArray(vtk_id)
        for vtk_p in vtk_ps:
            poly.GetPointData().AddArray(vtk_p)

        return poly

    # Stores a csv file with the peaks and their properties
    # fname: full path of the output .csv file
    def save_csv(self, fname):

        # Create the new file
        with open(fname, 'w') as ffile:
            if len(self.__peaks) == 0:
                return
            writer = csv.DictWriter(ffile, dialect=csv.excel_tab, fieldnames=self.__peaks[0].to_strings(mode='key'))
            writer.writeheader()
            for peak in self.__peaks:
                writer.writerow(peak.to_dictionary(fmt_csv=True))

    # Stores a files with the coordinates of each peak in three columns (X, Y, Z)
    # fname: full path of the output .coords file
    # swap_xy: if True X and Y coordinates are swapped
    # add_prop: addtional property to store in other columns (default None)
    # fmode: mode in which the file is opened (default 'w'), see python open file doc for more info
    def save_coords(self, fname, swap_xy=False, add_prop=None, fmode='w'):
        # Create the new file
        with open(fname, fmode) as ffile:
            if len(self.__peaks) == 0:
                return
            for peak in self.__peaks:
                if swap_xy:
                    y, x, z = peak.get_prop_val(PK_COORDS)
                else:
                    x, y, z = peak.get_prop_val(PK_COORDS)
                line = str(x) + ' ' + str(y) + ' ' + str(z)
                if add_prop is not None:
                    prop = peak.get_prop_val(add_prop)
                    for val in prop:
                        line += (' ' + str(val))
                ffile.write(line + '\n')

###########################################################################################
# Class for storing several TomoPeaks objects from different tomograms
###########################################################################################

class SetTomoPeaks(object):

    # shape: 3-tuple or array with the dimensions of the parent tomogram
    # name: (default '') string name that identify the tomoram
    # mask: (default None) binary tomogram (0-bg otherwise-fg) for masking valid regions for peaks in tomogram
    def __init__(self):
        self.__tpeaks = list()
        self.__in_ref_paths = list()
        self.__w_angs = list()
        self.__prop_angs = list()
        self.__swaps_xy = list()
        self.__sub_shape = None
        self.__ctfs = list()

    #### External functionality

    # tpeaks: a new TomoPeaks object
    # in_ref_path: input reference tomogram full path
    # See TomoPeak().save_particles() for the rest of input parameters
    # ctf: ctf correction tomogram full path (only valid and required for generating afterwards a STAR file)
    def add_tomo_peaks(self, tpeaks, in_ref_path, w_angs=(30.,30.), prop_ang=None,
                       swap_xy=False, ctf=None):
        self.__tpeaks.append(tpeaks)
        self.__in_ref_paths.append(in_ref_path)
        self.__w_angs.append(w_angs)
        self.__prop_angs.append(prop_ang)
        self.__swaps_xy.append(swap_xy)
        self.__ctfs.append(ctf)

    # Generate a STAR file compatible with Relion
    # n_key: key for normal (3-tuple peak property) to compute rotation according Relion's convention (ZYZ)
    def gen_star(self, n_key=None):

        # Creating the header
        star = Star()
        star.add_column('_rlnMicrographName')
        star.add_column('_rlnCoordinateX')
        star.add_column('_rlnCoordinateY')
        star.add_column('_rlnCoordinateZ')
        star.add_column('_rlnImageName')
        star.add_column('_rlnCtfImage')
        star.add_column('_rlnGroupNumber')
        star.add_column('_rlnAngleRot')
        star.add_column('_rlnAngleTilt')
        star.add_column('_rlnAnglePsi')

        # Adding a peaks entries
        group = 0
        for (tpeaks, ref_name, ctf, swap_xy) in \
            zip(self.__tpeaks, self.__in_ref_paths, self.__ctfs, self.__swaps_xy):
            group += 1
            for i, peak in enumerate(tpeaks.get_peaks_list()):
                # Peak data
                _, f_name = os.path.split(ref_name)
                s_name, _ = os.path.splitext(f_name)
                if swap_xy:
                    y, x, z = peak.get_prop_val(PK_COORDS)
                else:
                    x, y, z = peak.get_prop_val(PK_COORDS)
                i_name = './sub/' + s_name + '_' + str(i) + '.mrc'
                if ctf is None:
                    error_msg = 'A STAR file could be generated without ctf entries.'
                    raise pexceptions.PySegInputError(expr='gen_star (SetTomoPeaks)', msg=error_msg)
                rot, tilt, psi = 0, 0, 0
                if n_key is not None:
                    rot, tilt, psi = peak.get_prop_val(n_key)
                star.add_row(_rlnMicrographName=ref_name,
                             _rlnCoordinateX=x,
                             _rlnCoordinateY=y,
                             _rlnCoordinateZ=z,
                             _rlnImageName=i_name,
                             _rlnCtfImage=ctf,
                             _rlnAngleRot=rot,
                             _rlnAngleTilt=tilt,
                             _rlnAnglePsi=psi,
                             _rlnGroupNumber=group)

        return star

    # Generate a particle list
    # path: particle list path
    def gen_plist(self, path):
        count = 0
        plist = ParticleList(path)
        p_path, _ = os.path.split(path)
        for tpeaks, tomo_path in zip(self.__tpeaks, self.__in_ref_paths):
            for tpeak in tpeaks.get_peaks_list():
                _, tomo_fname = os.path.split(tomo_path)
                tomo_stem, _ = os.path.splitext(tomo_fname)
                tpeak.add_prop('Filename', p_path+'/sub/'+tomo_stem+'_p'+str(count)+'.em')
                tpeak.add_prop('Tomogram', tomo_path)
                plist.add_particle_peak(tpeak)
                count += 1
        return plist

###########################################################################################
# Abstract parent class for every class which represent and subelement in a XML tree
###########################################################################################

class ETsub(object):

    # For Abstract Base Classes in python
    __metaclass__ = ABCMeta
    #### Functionality area

    @abstractmethod
    def add_to_subET(self, elem):
        raise NotImplementedError('ETsub() (sub.plist). '
                                  'Abstract method, it requires an implementation.')

###########################################################################################
# Class for representing a SingleTiltWedge
###########################################################################################

class Score(ETsub):

    # stype: type of score (default 'xcfScore')
    # rem_auto: remove autocorrelation (default False)
    # val: value (default 0.5)
    # smooth: default -1
    # rad: Radius (default 0)
    # fname: filename path (default '')
    def __init__(self, stype='xcfScore', rem_auto=False, val=.5, smooth=-1, rad=0, fname=''):
        self.__stype = str(stype)
        self.__rem_auto = bool(rem_auto)
        self.__val = float(val)
        self.__smooth = float(smooth)
        self.__rad = float(rad)
        self.__fname = str(fname)

    #### External functionality

    # Insert the wedge into a tree (ElementTree)
    def add_to_subET(self, elem):
        dic_1 = {'Type':str(self.__stype), 'RemoveAutocorr':str(self.__rem_auto), 'Value':str(self.__val)}
        dic_2 = {'Smooth':str(self.__smooth), 'Radius':str(self.__rad), 'Filename':str(self.__fname)}
        score = ET.SubElement(elem, 'Score', dic_1)
        ET.SubElement(score, 'PeakPrior', dic_2)

###########################################################################################
# Class for representing a SingleTiltWedge
###########################################################################################

class SingleTiltWedge(ETsub):

    # smooth: default 0.0
    # angles: absolute wedge angles from Z-axis (default (30.0, 30.0))
    # cofr: cut off radius (default 0.0)
    # tilt_rot: tilt axis rotation euler angles (default (0,0,0) Z1, Z2 and X)
    def __init__(self, smooth=0.0, angles=(30,30), cofr=0, tilt_rot=(0,0,0)):
        self.__smooth = float(smooth)
        self.__angle_1, self.__angle_2 = float(angles[0]), float(angles[1])
        self.__cofr = float(cofr)
        self.__tilt_rot_x, self.__tilt_rot_z1, self.__tilt_rot_z2 = float(tilt_rot[0]), float(tilt_rot[1]), \
                                                                    float(tilt_rot[2])

    #### External functionality

    # Insert the wedge into a tree (ElementTree)
    def add_to_subET(self, elem):
        dic_1 = {'Smooth':str(self.__smooth), 'Angle1':str(self.__angle_1), 'CutoffRadius':str(self.__cofr),
                 'Angle2':str(self.__angle_2)}
        dic_2 = {'Z1':str(self.__tilt_rot_z1), 'Z2':str(self.__tilt_rot_z2), 'X':str(self.__tilt_rot_x)}
        wedge = ET.SubElement(elem, 'Wedge', {'Type':'SingleTiltWedge'})
        swedge = ET.SubElement(wedge, 'SingleTiltWedge', dic_1)
        ET.SubElement(swedge, 'TiltAxisRotation', dic_2)


###########################################################################################
# Class for representing a Particle
###########################################################################################

class Particle(object):

    # fname: path to subvolume
    # tomo_path: reference tomogram path
    # rot_para: rotation paradigm (default 'ZXZ')
    # rot_eu: rotation euler angles (default (0,0,0) psi, theta and phi)
    # shift: particle shift (default (0,0,0))
    # pos: picked position in reference tomogram (default (0,0,0))
    # orig_pos: origin for picked position (default '')
    # p_class: class name (default 0)
    # wedge: wedge subelement (default SingleTiltWedgde instance)
    # score: score subelement (default Score instance)
    def __init__(self, fname, tomo_path='', rot_para='ZXZ', rot_eu=(0,0,0), shift=(0,0,0), pos=(0,0,0), orig_pos='', p_class=0,
                 wedge=SingleTiltWedge(), score=Score()):
        self.__fname = str(fname)
        self.__rot_para = str(rot_para)
        self.__rot_eu_x, self.__rot_eu_z1, self.__rot_eu_z2 = str(rot_eu[2]), str(rot_eu[0]), str(rot_eu[1])
        self.__shift = shift
        self.__pos = pos
        self.__orig_pos = orig_pos
        self.__class = int(p_class)
        self.__wedge = wedge
        self.__score = score
        self.__tomo_path = tomo_path
        self.__sub_elem = None

    #### Set/Get functionality

    # Return the SubElementTree
    def get_sub_elem(self):
        return self.__sub_elem

    def get_Filename(self):
        return  self.__fname

    # Rotation angles in degrees in format psi, theta and phi
    def get_Rotation_Angs(self):
        return (self.__rot_eu_z1, self.__rot_eu_z2, self.__rot_eu_x)

    # Shifting in voxels (X, Y, Z)
    def get_Shifts(self):
        return self.__shift

    # Pick position in voxels of the reference tomogram (X, Y, Z)
    def get_PickPosition(self):
        return self.__pos

    # Tomogram reference path
    def get_Tomogram_Path(self):
        return self.__tomo_path

    #### External functionality

    # Load particle information from and XML subelement
    def set_from_subET(self, sub_elem):

        # The filename will work as particle unique identifier so its presence is a must
        try:
            fname = sub_elem.attrib['Filename']
        except KeyError:
            error_msg = 'Particle requires Filename attribute ' + fname
            raise pexceptions.PySegInputError(expr='set_from_subET (ParticleList)', msg=error_msg)

        # Parsing just peak useful information
        x, y, z = -1., -1., -1.
        pos = sub_elem.find('PickPosition')
        orig_pos = ''
        if pos is not None:
            try:
                orig_pos = pos.attrib['Origin']
            except KeyError:
                pass
            try:
                x, y, z = float(pos.attrib['X']), float(pos.attrib['Y']), float(pos.attrib['Z'])
            except KeyError:
                pass

        rot = sub_elem.find('Rotation')
        rot_para = 'ZXZ'
        phi, psi, the = 0., 0., 0.
        if rot is not None:
            try:
                rot_para = rot.attrib['Paradigm']
            except KeyError:
                pass
            try:
                psi, phi, the = float(rot.attrib['Z1']), float(rot.attrib['Z2']), float(rot.attrib['X'])
            except KeyError:
                pass
        shift = sub_elem.find('Shift')
        shift_x, shift_y, shift_z = 0., 0., 0.
        if shift is not None:
            try:
                shift_x, shift_y, shift_z = float(shift.attrib['X']), float(shift.attrib['Y']), \
                                            float(shift.attrib['Z'])
            except KeyError:
                pass
        wedge = sub_elem.find('Wedge')
        wedge_a1, wedge_a2 = 0, 0
        if wedge is not None:
            wedge_sing = wedge.find('SingleTiltWedge')
            if wedge_sing is not None:
                try:
                    wedge_a1, wedge_a2 = float(wedge_sing.attrib['Angle1']), float(wedge_sing.attrib['Angle2'])
                except KeyError:
                    pass
        score = sub_elem.find('Score')
        score_v, stype = -1, 'xcfScore'
        if score is not None:
            try:
                score_v = float(score.attrib['Value'])
            except KeyError:
                pass
        cla = sub_elem.find('Class')
        cla_v = -1
        if cla is not None:
            try:
                cla_v = int(cla.attrib['Name'])
            except KeyError:
                pass
        tomo_path = ''
        tomo = sub_elem.find('Tomogram')
        if tomo is not None:
            try:
                tomo_path = tomo.attrib['Path']
            except KeyError:
                pass

        # Setting class attributes
        self.__fname = fname
        self.__rot_para = rot_para
        self.__rot_eu_z1, self.__rot_eu_z2, self.__rot_eu_x = psi, the, phi
        self.__shift = (shift_x, shift_y, shift_z)
        self.__pos = (x, y, z)
        self.__orig_pos = orig_pos
        self.__class = cla_v
        self.__tomo_path = tomo_path
        self.__wedge = SingleTiltWedge(angles=(wedge_a1, wedge_a2))
        self.__score = Score(stype=stype, val=score_v)
        self.__sub_elem = sub_elem

    # Insert the particle into a tree (ElementTree)
    def add_to_subET(self, elem):
        dic_rot = {'Paradigm':str(self.__rot_para), 'X':str(self.__rot_eu_x),
                   'Z1':str(self.__rot_eu_z1), 'Z2':str(self.__rot_eu_z2)}
        dic_shift = {'X':str(self.__shift[0]), 'Y':str(self.__shift[1]), 'Z':str(self.__shift[2])}
        dic_pick = {'Origin':str(self.__orig_pos), 'X':str(self.__pos[0]), 'Y':str(self.__pos[1]),
                    'Z':str(self.__pos[2])}
        part = ET.SubElement(elem, 'Particle', {'Filename':self.__fname})
        ET.SubElement(part, 'Rotation', dic_rot)
        ET.SubElement(part, 'Shift', dic_shift)
        ET.SubElement(part, 'PickPosition', dic_pick)
        self.__wedge.add_to_subET(part)
        self.__score.add_to_subET(part)
        ET.SubElement(part, 'Class', {'Name':str(self.__class)})
        ET.SubElement(part, 'Tomogram', {'Path':str(self.__tomo_path)})
        self.__sub_elem = part


###########################################################################################
# Class for representing a Particle list
###########################################################################################

class ParticleList(object):

    # path: path to the subvolumes
    def __init__(self, path):
        self.__path = str(path)
        self.__elem = ET.Element('ParticleList', {'Path':self.__path})

    ###### Get/Set functionality

    def get_num_particles(self):
        return len(self.__elem.findall('Particle'))

    # do_shift: if True (default False) coordinates are modified according to particle shift information
    # rot: if True (default False) rotation angles are also returned in another list
    # Returns a list with particles coordinates, if a Particle has no PickPosition field it is skipped,
    #         two list if rots is activated
    def get_particles_coords(self, do_shift=True, rot=False):
        coords = list()
        if rot:
            rots = list()
        for part in self.__elem.iter(tag='Particle'):
            # Parsing just peak useful information
            pos = part.find('PickPosition')
            x, y, z = -1., -1., -1.
            try:
                x, y, z = float(pos.attrib['X']), float(pos.attrib['Y']), float(pos.attrib['Z'])
            except KeyError:
                print 'WARNING: get_particles_coords() (ParticleList), a Particle without PickPostion field found!'
                continue
            if do_shift:
                shift = part.find('Shift')
                shift_x, shift_y, shift_z = 0., 0., 0.
                if shift is not None:
                    try:
                        shift_x, shift_y, shift_z = float(shift.attrib['X']), float(shift.attrib['Y']), \
                                                    float(shift.attrib['Z'])
                    except KeyError:
                        pass
                x, y, z = x-shift_x, y-shift_y, z-shift_z
            coords.append(np.asarray((x,y,z), dtype=np.float32))
            if rot:
                angles = part.find('Rotation')
                phi, psi, the = 0., 0., 0.
                if angles is not None:
                    try:
                        psi, phi, the = float(angles.attrib['Z1']), float(angles.attrib['Z2']), \
                                        float(angles.attrib['X'])
                    except KeyError:
                        pass
                rots.append(np.asarray((phi,psi,the), dtype=np.float32))
        if rot:
            return coords, rots
        else:
            return coords

    ###### External functionality

    # name: file name for the subvolume
    # Reference tomogram path (default '')
    # rot_para: rotation paradigm (default 'ZXZ')
    # rot_eu: rotation euler angles (default (0,0,0) Z1, X and Z2)
    # shift: particle shift (default (0,0,0))
    # pos: picked position in reference tomogram (default (0,0,0))
    # orig_pos: origin for picked position (default '')
    # p_class: class name (default 0)
    # wedge: wedge subelement (default SingleTiltWedge instance)
    # score: score subelement (default Score instance)
    def add_particle(self, name, tomo_path='', rot_para='ZXZ', rot_eu=(0,0,0), shift=(0,0,0), pos=(0,0,0), orig_pos='', p_class=0,
                 wedge=SingleTiltWedge(), score=Score()):
        part = Particle(name, tomo_path, rot_para, rot_eu, shift, pos, orig_pos, p_class, wedge, score)
        part.add_to_subET(self.__elem)

    # Add particle from a Peak object
    # peak: input peak
    def add_particle_peak(self, peak):

        # Parsing particle information
        fname = peak.get_prop_val('Filename')
        try:
            rot = peak.get_prop_val('Rotation')
        except KeyError:
            rot = (0., 0., 0.)
        try:
            shift = peak.get_prop_val('Shift')
        except KeyError:
            shift = (0., 0., 0.)
        try:
            wedge = peak.get_prop_val('Wedge')
        except KeyError:
            wedge = (30., 30)
        try:
            score = peak.get_prop_val('Score')
        except KeyError:
            score = 0.5
        try:
            cla = peak.get_prop_val('Class')
        except KeyError:
            cla = 0
        try:
            tomo = peak.get_prop_val('Tomogram')
        except KeyError:
            tomo = ''
        x, y, z = peak.get_prop_val(PK_COORDS)

        # Adding the particle
        self.add_particle(name=fname, tomo_path=tomo, rot_eu=(rot[1], rot[2], rot[0]), shift=shift, pos=(x,y,z),
                          p_class=cla, wedge=SingleTiltWedge(angles=(wedge[0], wedge[1])), score=Score(val=score))

    # Imports an already created (by another ParticleList) Particle
    def import_particle(self, elem):
        if elem.tag == 'Particle':
            part = from_ElementTree_to_Particle(elem)
            part.add_to_subET(self.__elem)

    def get_elements(self):
        elements = list()
        for e in self.__elem:
            elements.append(e)
        return elements

    def to_string(self, pretty=True):
        rough_string = ET.tostring(self.__elem, 'utf-8')
        if pretty:
            reparsed = minidom.parseString(rough_string)
            return reparsed.toprettyxml(indent="  ")
        else:
            return rough_string

    # Store as xml file in the already specified path
    def store(self):
        with open(self.__path, 'w') as xfile:
            xfile.write(self.to_string())

    # Load the particle list from XML file
    # fname: path of the XML file with the particle list
    def load(self, fname):

        # Input parsing
        try:
            tree = ET.parse(fname)
            elements = tree.getroot()
        except ET.ParseError as e:
            error_msg = 'Parse Error: ' + str(e)
            raise pexceptions.PySegInputError(expr='load (ParticleList)', msg=error_msg)
        try:
            self.__path = self.__elem.attrib['Path']
        except KeyError:
            self.__path = '/'

        # Loop for particles
        for subel in elements:
            if subel.tag == 'Particle':
                self.import_particle(subel)

    # Generates a TomoPeaks object by taken PickPosition, Rotation, Score and Class information (if available)
    # shape: 3-tuple or array with the dimensions of the parent tomogram
    # name: (default '') string name that identify the tomogram
    # mask: for adding a mask with valid peaks positions (default None)
    # swap_xy: if True (default False) input peaks X and Y coordinates are swapped
    # do_shift: if True (default) peaks coordinates ar shifted accorded particle attributes
    # Returns: the TomoPeaks object created
    def gen_TomoPeaks(self, shape, name, mask=None, swap_xy=False, do_shift=True):

        # Build TomoPeaks object
        tpeaks = TomoPeaks(shape, name, mask)

        # Add particles
        pid = 0
        for part in self.__elem.iter(tag='Particle'):

            # The filename will work as particle unique identifier so its presence is a must
            try:
                fname = part.attrib['Filename']
            except KeyError:
                continue

            # Parsing just peak useful information
            pos = part.find('PickPosition')
            if (pos is not None) and (fname is not None):
                try:
                    x, y, z = float(pos.attrib['X']), float(pos.attrib['Y']), float(pos.attrib['Z'])
                except KeyError:
                    print 'WARNING: gen_TomoPeaks() (ParticleList), a Particle without Pick position cannot be converted into a Peak'
                    continue
                rot = part.find('Rotation')
                phi, psi, the = 0, 0, 0
                if rot is not None:
                    try:
                        psi, phi, the = float(rot.attrib['Z1']), float(rot.attrib['Z2']), float(rot.attrib['X'])
                    except KeyError:
                        pass
                shift = part.find('Shift')
                shift_x, shift_y, shift_z = 0, 0, 0
                if shift is not None:
                    try:
                        shift_x, shift_y, shift_z = float(shift.attrib['X']), float(shift.attrib['Y']), \
                                                    float(shift.attrib['Z'])
                    except KeyError:
                        pass
                wedge = part.find('Wedge')
                wedge_a1, wedge_a2 = 0, 0
                if wedge is not None:
                    wedge_sing = wedge.find('SingleTiltWedge')
                    if wedge_sing is not None:
                        try:
                            wedge_a1, wedge_a2 = float(wedge_sing.attrib['Angle1']), float(wedge_sing.attrib['Angle2'])
                        except KeyError:
                            pass
                score = part.find('Score')
                score_v = -1
                if score is not None:
                    try:
                        score_v = float(score.attrib['Value'])
                    except KeyError:
                        pass
                cla = part.find('Class')
                cla_v = -1
                if cla is not None:
                    try:
                        cla_v = int(cla.attrib['Name'])
                    except KeyError:
                        pass
                tomo_path = ''
                tomo = part.find('Tomogram')
                if tomo is not None:
                    try:
                        tomo_path = tomo.attrib['Path']
                    except KeyError:
                        pass

                # Insert coordinates
                if do_shift:
                    x, y, z = x-shift_x, y-shift_y, z-shift_z
                if swap_xy:
                    tpeaks.add_peak((y, x, z))
                else:
                    tpeaks.add_peak((x, y, z))

                # Setting peaks properties
                if pid == 0:
                    tpeaks.add_prop('Filename', 1, dtype=str)
                    tpeaks.add_prop('Rotation', 3, dtype=np.float32)
                    tpeaks.add_prop('Shift', 3, dtype=np.float32)
                    tpeaks.add_prop('Wedge', 2, dtype=np.float32)
                    tpeaks.add_prop('Score', 1, dtype=np.float32)
                    tpeaks.add_prop('Class', 1, dtype=np.int)
                    tpeaks.add_prop('Tomogram', 1, dtype=str)
                tpeaks.set_peak_prop(pid, 'Filename', fname)
                tpeaks.set_peak_prop(pid, 'Rotation', (phi, psi, the))
                tpeaks.set_peak_prop(pid, 'Shift', (shift_x, shift_y, shift_z))
                tpeaks.set_peak_prop(pid, 'Wedge', (wedge_a1, wedge_a2))
                tpeaks.set_peak_prop(pid, 'Score', score_v)
                tpeaks.set_peak_prop(pid, 'Class', cla_v)
                tpeaks.set_peak_prop(pid, 'Tomogram', tomo_path)
                pid += 1

        return tpeaks

    # Filter particles depending if they contain ref_stem string not
    # ref_stem: string to look at the begining of every particle filename attribute
    # keep: if True (default) particle which has the string are preserved, otherwise those are the discarded ones
    # cmp: set string comparison mode, valid: 'full' (default) the whole Filename attribute is compared, and
    #       'stem' only the first len(ref_stem) characters of filename stem is checked
    def filter_particle_fname(self, ref_stem, keep=True, cmp='full'):

        # Initialization
        if (cmp != 'full') and (cmp != 'stem'):
            error_msg = 'Non valid comparison mode ' + str(cmp)
            raise pexceptions.PySegInputError(expr='filter_particle_fname (ParticleList)', msg=error_msg)
        s_ref_stem = str(ref_stem)
        l_ref = len(ref_stem)

        for part in self.__elem.iter(tag='Particle'):
            full_path = part.attrib['Filename']
            if cmp == 'stem':
                _, fname = os.path.split(full_path)
                if (len(fname) >= l_ref) and (fname[:l_ref] == s_ref_stem):
                    if not keep:
                        self.__elem.remove(part)
                else:
                    if keep:
                        self.__elem.remove(part)
            elif cmp == 'full':
                if full_path == ref_stem:
                    if not keep:
                        self.__elem.remove(part)
                else:
                    if keep:
                        self.__elem.remove(part)

    # Delete all particles which are not in a file names list
    # fnames: list of filenames
    def filter_particle_nolist(self, fnames):
        hold_list = list()
        for part in self.__parts:
            fname = part.attrib['Filename']
            if fname in fnames:
                hold_list.append(part)
        self.__parts = hold_list

    # Generates and stores subvolumes in particles list directory
    # sub_shape: 3-tuple with the shape in voxels for subvolumes, all dimension must be even, Notice that peaks with subvolumes
    #            that are totally or partially out reference tomogram won't be considered
    # purge: if True (default) the particle where sub-volumes were not cropped are deleted
    # align: if True (default False) the particle are aligned according to their rotation and shifting information
    # mask: if not None (default) it is applied to all subvolumes
    # crop: output density region is cropped with a mask (default False), only applied if mask is None
    # Results: the particles stored as sub-volumes for every particle in a subdirectory called 'sub'
    def save_subvolumes(self, sub_shape, purge=True, align=True, mask=None, crop=False):

        # Parsing inputs
        if (not hasattr(sub_shape, '__len__')) or (not(len(sub_shape) == 3)) \
                or (sub_shape[0]<=0) or (sub_shape[1]<=0) or (sub_shape[2]<=0):
            error_msg = 'Subvolume shape must be 3-tuple with non trivial values.'
            raise pexceptions.PySegInputError(expr='save_subvolumes (ParticleList)', msg=error_msg)
        if ((sub_shape[0]%2) != 0) or ((sub_shape[1]%2) != 0) or ((sub_shape[2]%2) != 0):
            error_msg = 'All subvolume dimensions must be even, current ' + str(sub_shape)
            raise pexceptions.PySegInputError(expr='save_particles (SetTomoPeaks)', msg=error_msg)
        outdir = os.path.split(self.__path)[0] + '/sub'
        try:
            os.makedirs(outdir)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                error_msg = 'Directory ' + outdir + ' do not exist I could not be created.'
                raise pexceptions.PySegInputError(expr='save_subvolumes (ParticleList)', msg=error_msg)
        if mask is not None:
            if (not isinstance(mask, np.ndarray)) or (len(mask.shape) != 3) or \
                    (mask.shape[0] != sub_shape[0]) or (mask.shape[1] != sub_shape[1]) or (mask.shape[2] != sub_shape[2]):
                error_msg = 'Input mask must be a 32 numpy.ndarray with the same shape of the specified sub-volume.'
                raise pexceptions.PySegInputError(expr='save_particles (ParticleList)', msg=error_msg)
        ss_arr = np.asarray(sub_shape).astype(np.float32)
        h_len = int(math.ceil(1.5 * math.sqrt((ss_arr * ss_arr).sum())))
        if (h_len%2) != 0:
            h_len += 1
        hhl_d = int(h_len * .5)
        hl_x, hl_y, hl_z = int(sub_shape[1]*.5), int(sub_shape[0]*.5), int(sub_shape[2]*.5)

        # Particles loop
        if purge:
            purge_l = list()
        for i, sub_elem in enumerate(self.__elem.iter(tag='Particle')):

            part = Particle('')
            part.set_from_subET(sub_elem)

            # Getting particle properties
            part_fname = part.get_Filename()
            coord = part.get_PickPosition()
            if align:
                shifts = part.get_Shifts()
                eu_angs = part.get_Rotation_Angs()
                eu_angs = np.asarray((eu_angs[2], eu_angs[0], eu_angs[1]), dtype=np.float32)

            # Read tomogram map
            tomo_path = part.get_Tomogram_Path()
            try:
                tomo = disperse_io.load_tomo(tomo_path, mmap=True)
            except KeyError:
                error_msg = 'Reference tomogram ' + tomo_path + ' could not be read'
                raise pexceptions.PySegInputError(expr='save_subvolumes (ParticleList)', msg=error_msg)

            # Computing un-rotated cropping
            x, y, z = int(math.floor(coord[0])), int(math.floor(coord[1])), int(math.floor(coord[2]))
            x_l, y_l, z_l = x-hhl_d+1, y-hhl_d+1, z-hhl_d+1
            if (x_l < 0) or (y_l < 0) or (z_l < 0):
                continue
                if purge:
                    purge_l.append(part)
            x_h, y_h, z_h = x+hhl_d+1, y+hhl_d+1, z+hhl_d+1
            if (x_l >= tomo.shape[0]) or (y_l >= tomo.shape[1]) or (z_l >= tomo.shape[2]):
                continue
                if purge:
                    purge_l.append(part)
            hold_sub = tomo[x_l:x_h, y_l:y_h, z_l:z_h]
            hold_sub = np.asarray(hold_sub).astype(np.float32)

            # Alignment
            if align:
                hold_sub = tomo_shift(hold_sub, -1.*shifts)
                hold_sub = tomo_rot(hold_sub, -1.*eu_angs, deg=True, order=2)

            # Final cropping
            x, y, z = hhl_d-1, hhl_d-1, hhl_d-1
            x_l, y_l, z_l = x-hl_x+1, y-hl_y+1, z-hl_z+1
            x_h, y_h, z_h = x+hl_x+1, y+hl_y+1, z+hl_z+1
            hold_sub = hold_sub[x_l:x_h, y_l:y_h, z_l:z_h]

            # Normalization
            if mask is None:
                mn, std = hold_sub.mean(), hold_sub.std()
            else:
                h_hold_sub = hold_sub[mask < 1]
                mn, std = h_hold_sub.mean(), h_hold_sub.std()
            if std <= 0:
                continue
            hold_sub = (hold_sub-mn) / std
            if (mask is not None) and crop:
                hold_sub *= mask

            # Storing subvolume as an EM-file
            disperse_io.save_numpy(hold_sub, part_fname)

        # Particle purging
        if purge:
            for part in purge_l:
                se = part.get_sub_elem()
                if se is not None:
                    self.__elem.remove(se)

    # Save particles in plain text, where rows are particles, and columns (space separated): X, Y, Z, phi, psi, the
    # path: full path where the file is stored
    def save_plain_txt(self, path):

        with open(path, 'w') as xfile:

            for elem in self.get_elements():
                pos = elem.find('PickPosition')
                x, y, z = -1, -1, -1
                if pos is not None:
                    try:
                        x, y, z = float(pos.attrib['X']), float(pos.attrib['Y']), float(pos.attrib['Z'])
                    except KeyError:
                        print 'WARNING: save_plain_txt() (ParticleList), a Particle without Pick position cannot be converted into a Peak'
                        continue
                rot = elem.find('Rotation')
                phi, psi, the = 0, 0, 0
                if rot is not None:
                    try:
                        psi, phi, the = float(rot.attrib['Z1']), float(rot.attrib['Z2']), float(rot.attrib['X'])
                    except KeyError:
                        continue
                line = str(x) + ' ' + str(y) + ' ' + str(z) + ' ' + str(phi) + ' ' + str(psi) + ' ' + str(the) + '\n'
                xfile.write(line)

                # Generates a TomoPeaks object by taken PickPosition, Rotation, Score and Class information (if available)

        # shape: 3-tuple or array with the dimensions of the parent tomogram
        # name: (default '') string name that identify the tomogram
        # mask: for adding a mask with valid peaks positions (default None)
        # swap_xy: if True (default False) input peaks X and Y coordinates are swapped
        # do_shift: if True (default) peaks coordinates ar shifted accorded particle attributes
        # Returns: the TomoPeaks object created
        def gen_TomoPeaks(self, shape, name, mask=None, swap_xy=False, do_shift=True):

            # Build TomoPeaks object
            tpeaks = TomoPeaks(shape, name, mask)

            # Add particles
            pid = 0
            for part in self.__elem.iter(tag='Particle'):

                # The filename will work as particle unique identifier so its presence is a must
                try:
                    fname = part.attrib['Filename']
                except KeyError:
                    continue

                # Parsing just peak useful information
                pos = part.find('PickPosition')
                if (pos is not None) and (fname is not None):
                    try:
                        x, y, z = float(pos.attrib['X']), float(pos.attrib['Y']), float(pos.attrib['Z'])
                    except KeyError:
                        print 'WARNING: gen_TomoPeaks() (ParticleList), a Particle without Pick position cannot be converted into a Peak'
                        continue
                    rot = part.find('Rotation')
                    phi, psi, the = 0, 0, 0
                    if rot is not None:
                        try:
                            psi, phi, the = float(rot.attrib['Z1']), float(rot.attrib['Z2']), float(rot.attrib['X'])
                        except KeyError:
                            pass
                    shift = part.find('Shift')
                    shift_x, shift_y, shift_z = 0, 0, 0
                    if shift is not None:
                        try:
                            shift_x, shift_y, shift_z = float(shift.attrib['X']), float(shift.attrib['Y']), \
                                                        float(shift.attrib['Z'])
                        except KeyError:
                            pass
                    wedge = part.find('Wedge')
                    wedge_a1, wedge_a2 = 0, 0
                    if wedge is not None:
                        wedge_sing = wedge.find('SingleTiltWedge')
                        if wedge_sing is not None:
                            try:
                                wedge_a1, wedge_a2 = float(wedge_sing.attrib['Angle1']), float(
                                    wedge_sing.attrib['Angle2'])
                            except KeyError:
                                pass
                    score = part.find('Score')
                    score_v = -1
                    if score is not None:
                        try:
                            score_v = float(score.attrib['Value'])
                        except KeyError:
                            pass
                    cla = part.find('Class')
                    cla_v = -1
                    if cla is not None:
                        try:
                            cla_v = int(cla.attrib['Name'])
                        except KeyError:
                            pass
                    tomo_path = ''
                    tomo = part.find('Tomogram')
                    if tomo is not None:
                        try:
                            tomo_path = tomo.attrib['Path']
                        except KeyError:
                            pass

                    # Insert coordinates
                    if do_shift:
                        x, y, z = x - shift_x, y - shift_y, z - shift_z
                    if swap_xy:
                        tpeaks.add_peak((y, x, z))
                    else:
                        tpeaks.add_peak((x, y, z))

                    # Setting peaks properties
                    if pid == 0:
                        tpeaks.add_prop('Filename', 1, dtype=str)
                        tpeaks.add_prop('Rotation', 3, dtype=np.float32)
                        tpeaks.add_prop('Shift', 3, dtype=np.float32)
                        tpeaks.add_prop('Wedge', 2, dtype=np.float32)
                        tpeaks.add_prop('Score', 1, dtype=np.float32)
                        tpeaks.add_prop('Class', 1, dtype=np.int)
                        tpeaks.add_prop('Tomogram', 1, dtype=str)
                    tpeaks.set_peak_prop(pid, 'Filename', fname)
                    tpeaks.set_peak_prop(pid, 'Rotation', (phi, psi, the))
                    tpeaks.set_peak_prop(pid, 'Shift', (shift_x, shift_y, shift_z))
                    tpeaks.set_peak_prop(pid, 'Wedge', (wedge_a1, wedge_a2))
                    tpeaks.set_peak_prop(pid, 'Score', score_v)
                    tpeaks.set_peak_prop(pid, 'Class', cla_v)
                    tpeaks.set_peak_prop(pid, 'Tomogram', tomo_path)
                    pid += 1

            return tpeaks







































































































