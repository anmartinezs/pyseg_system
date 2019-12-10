"""
Clases with the parsers for I/O with xml files

# Author: Antonio Martinez-Sanchez (Max Planck Institute for Biochemistry)
# Date: 02.02.15
"""

__author__ = 'martinez'

from bs4 import BeautifulSoup
import operator
from pyseg.mb.graph import *


########################GLOBAL PARSING STRINGS######################################################

XML_THRESHOLDING = 'thresholding'
XML_KEY_PROP = 'key_prop'
XML_THRESHOLD = 'threshold'
XML_OPERATOR = 'operator'
XML_MBS = 'mb_slice'
XML_MBS_NAME = 'name'
XML_MBS_SIDE = 'side'
XML_MBS_EU_DST = 'eu_dst'
XML_MBS_GEO_DST = 'geo_dst'
XML_MBS_GEO_LEN = 'geo_len'
XML_MBS_SIN = 'sin'
XML_MBS_NV = 'clst_nv'
XML_MBS_SIGN = 'sign'
XML_MBS_LOW = 'low'
XML_MBS_HIGH = 'high'
XML_MBS_CONT = 'cont'
XML_MBL = 'linker'
XML_MBL_SNAME = 'source'
XML_MBL_TNAME = 'target'
XML_MBL_EU_DST = 'eu_dst'
XML_MBL_GEO_DST = 'geo_dst'
XML_MBL_SIN = 'sin'
XML_MBL_NN = 'nn'
XML_MBT = 'thres_slice'
XML_MBT_PROP = 'key_prop'
XML_MBT_LOW = 'low'
XML_MBT_HIGH = 'high'
XML_MBT_MODE = 'mode'
XML_MBT_VAL = 'value'
XML_MBT_MIN = 'in'
XML_MBT_MOUT = 'out'
XML_MBT_MPERI = 'per_in'
XML_MBT_MPERO = 'per_out'

XML_SYN = 'syn_slice'
XML_SYN_MB = 'mb'
XML_SYN_SEG = 'seg'

XML_RWR = 'rwr'
XML_RWR_NAME = 'name'
XML_RWR_SIGN = 'rwr_sign'
XML_RWR_SP = '+'
XML_RWR_SN = '-'
XML_RWR_C = 'rwr_c'

MB_SEG = 1

# ####################################################################################################
# Class for loading thresholding operations
#
#
class Threshold(object):

    #### Constructor Area

    # fil: file with the XML file
    def __init__(self, file_name):
        try:
            self.__soup = BeautifulSoup(open(file_name), "xml")
        except:
            error_msg = 'File %s could not be read.' % file_name
            raise pexceptions.PySegInputError(expr='(Threshold)', msg=error_msg)

    #### Get/Set methods

    def get_num_thresholdings(self):
        return len(self.__soup.find_all(XML_THRESHOLDING))

    def get_key_prop_list(self):
        keys = list()
        for thresholding in self.__soup.find_all(XML_THRESHOLDING):
            keys.append(thresholding.find(XML_KEY_PROP).get_text())
        return keys

    def get_threshold_list(self):
        thresholds = list()
        for thresholding in self.__soup.find_all(XML_THRESHOLDING):
            thresholds.append(float(thresholding.find(XML_THRESHOLD).get_text()))
        return thresholds

    def get_operator_list(self):
        operators = list()
        for thresholding in self.__soup.find_all(XML_THRESHOLDING):
            txt = thresholding.find(XML_OPERATOR).get_text()
            if txt == 'eq':
                operators.append(operator.eq)
            elif txt == 'ne':
                operators.append(operator.ne)
            elif txt == 'ge':
                operators.append(operator.ge)
            elif txt == 'gt':
                operators.append(operator.gt)
            elif txt == 'le':
                operators.append(operator.le)
            elif txt == 'lt':
                operators.append(operator.lt)
            else:
                error_msg = 'Operator %s is not valid.' % txt
                raise pexceptions.PySegInputError(expr='(Threshold)', msg=error_msg)
        return operators

###########################################################################################
# Class for modelling a membrane slice
###########################################################################################

class Slice(object):

    def __init__(self, name, side, cont, eu_dst_r, geo_dst_r, geo_len_r, sin_dst_r, cnv_dst_r):
        self.__name = name
        self.__side = side
        self.__cont = cont
        self.__eu_dst = eu_dst_r
        self.__geo_dst = geo_dst_r
        self.__geo_len = geo_len_r
        self.__sin = sin_dst_r
        self.__cnv = cnv_dst_r
        self.__ths = list()

    #### Set/Get methods area

    def get_list_th(self):
        return self.__ths

    def get_side(self):
        return self.__side

    def get_name(self):
        return str(self.__name)

    def get_cont(self):
        if self.__cont == '+':
            return True
        else:
            return False

    def get_fil(self):
        if self.__cont == '=':
            return True
        else:
            return False

    def get_eu_dst_low(self):
        return self.__eu_dst[0]

    def get_eu_dst_high(self):
        return self.__eu_dst[1]

    def get_eu_dst_sign(self):
        return self.__eu_dst[2]

    def get_geo_dst_low(self):
        return self.__geo_dst[0]

    def get_geo_dst_high(self):
        return self.__geo_dst[1]

    def get_geo_dst_sign(self):
        return self.__geo_dst[2]

    def get_geo_len_low(self):
        return self.__geo_len[0]

    def get_geo_len_high(self):
        return self.__geo_len[1]

    def get_geo_len_sign(self):
        return self.__geo_len[2]

    def get_sin_low(self):
        return self.__sin[0]

    def get_sin_high(self):
        return self.__sin[1]

    def get_sin_sign(self):
        return self.__sin[2]

    def get_cnv_low(self):
        return self.__cnv[0]

    def get_cnv_high(self):
        return self.__cnv[1]

    def get_cnv_sign(self):
        return self.__cnv[2]

    #### External methods area

    # Add additional threshold for vertices properties
    def add_threshold(self, th):
        self.__ths.append(th)

    # Test from MbGraphMCF vertex parameters if it fulfils slice settings
    # x: value for variable 'x', if None (default) this value is not used
    def test(self, side=None, eu_dst=None, geo_dst=None, geo_len=None, sin=None, cnv=None):

        if side is not None:
            if side != self.__side:
                return False

        if self.__side != MB_SEG:
            if eu_dst is not None:
                if (eu_dst < self.__eu_dst[0]) or (eu_dst > self.__eu_dst[1]):
                    if self.__eu_dst[2] == '+':
                        return False
            if geo_dst is not None:
                if (geo_dst < self.__geo_dst[0]) or (geo_dst > self.__geo_dst[1]):
                    if self.__geo_dst[2] == '+':
                        return False
            if geo_len is not None:
                if (geo_len < self.__geo_len[0]) or (geo_len > self.__geo_len[1]):
                    if self.__geo_len[2] == '+':
                        return False
            if sin is not None:
                if (sin < self.__sin[0]) or (sin > self.__sin[1]):
                    if self.__sin[2] == '+':
                        return False
        if cnv is not None:
            if (cnv < self.__cnv[0]) or (cnv > self.__cnv[1]):
                if self.__cnv[2] == '+':
                    return False

        # Default
        return True

###########################################################################################
# Class for modelling a synapse slice
###########################################################################################

class SynSlice(Slice):

    def __init__(self, name, mb, seg, cont, eu_dst_r, geo_dst_r, geo_len_r, sin_dst_r, cnv_dst_r):
        self.__mb = mb
        self.__seg = seg
        if self.__seg == SYN_CLF_LBL:
            hold_side = MB_OUT_LBL
        elif (self.__seg == SYN_PSD_LBL) or (self.__seg == SYN_AZ_LBL):
            hold_side = MB_IN_LBL
        else:
            hold_side = MB_LBL
        super(SynSlice, self).__init__(name, hold_side, cont, eu_dst_r, geo_dst_r, geo_len_r, sin_dst_r, cnv_dst_r)

    #### Set/Get methods area

    def get_mb(self):
        return self.__mb

    def get_seg(self):
        return self.__seg

    #### External methods area

    # Test from checking if a set of SynGraphMCF vertex parameters fulfils the slice settings
    # x: value for variable 'x', if None (default) this value is not used
    def test(self, seg=None, eu_dst=None, geo_dst=None, geo_len=None, sin=None, cnv=None):

        if seg is not None:
            if seg != self.__seg:
                return False

        if eu_dst is not None:
            if (eu_dst < self._Slice__eu_dst[0]) or (eu_dst > self._Slice__eu_dst[1]):
                if self._Slice__eu_dst[2] == '+':
                    return False
        if geo_dst is not None:
            if (geo_dst < self._Slice__geo_dst[0]) or (geo_dst > self._Slice__geo_dst[1]):
                if self._Slice__geo_dst[2] == '+':
                    return False
        if geo_len is not None:
            if (geo_len < self._Slice__geo_len[0]) or (geo_len > self._Slice__geo_len[1]):
                if self._Slice__geo_len[2] == '+':
                    return False
        if sin is not None:
            if (sin < self._Slice__sin[0]) or (sin > self._Slice__sin[1]):
                if self._Slice__sin[2] == '+':
                    return False
        if cnv is not None:
            if (cnv < self._Slice__cnv[0]) or (cnv > self._Slice__cnv[1]):
                if self._Slice__cnv[2] == '+':
                    return False

        # Default
        return True

###########################################################################################
# Class for loading and getting membrane slices settings
# idx: for all get or testing function identifies the slice
###########################################################################################

class SliceSet(object):

    # fname: path to the XML file with slice settings
    def __init__(self, fname):
        self.__fname = fname
        self.__soup = None
        self.__slices = list()
        self.__load()

    #### Set/Get methods area

    def get_file_name(self):
        return self.__fname

    def get_slices_list(self):
        return self.__slices

    def get_num_slices(self):
        return len(self.__slices)

    #### External functionality area

    #### Internal functionality area

    # Load XML file and fill up correctly object variables
    def __load(self):

        # Loading XML File
        try:
            self.__soup = BeautifulSoup(open(self.__fname), "xml")
        except:
            error_msg = 'File %s could not be read.' % self.__fname
            raise pexceptions.PySegInputError(expr='(SliceSet)', msg=error_msg)

        # Parsing XML file
        for slice in self.__soup.find_all(XML_MBS):
            name = slice.find(XML_MBS_NAME).get_text()
            side = int(slice.find(XML_MBS_SIDE).get_text())
            cont = slice.find(XML_MBS_CONT).get_text()
            eu_dst_low = float(slice.find(XML_MBS_EU_DST, id=XML_MBS_LOW).get_text())
            eu_dst_high = float(slice.find(XML_MBS_EU_DST, id=XML_MBS_HIGH).get_text())
            eu_dst_sign = slice.find(XML_MBS_EU_DST, id=XML_MBS_SIGN).get_text()
            eu_dst = [eu_dst_low, eu_dst_high, eu_dst_sign]
            geo_dst_low = float(slice.find(XML_MBS_GEO_DST, id=XML_MBS_LOW).get_text())
            geo_dst_high = float(slice.find(XML_MBS_GEO_DST, id=XML_MBS_HIGH).get_text())
            geo_dst_sign = slice.find(XML_MBS_GEO_DST, id=XML_MBS_SIGN).get_text()
            geo_dst = [geo_dst_low, geo_dst_high, geo_dst_sign]
            geo_len_low = float(slice.find(XML_MBS_GEO_LEN, id=XML_MBS_LOW).get_text())
            geo_len_high = float(slice.find(XML_MBS_GEO_LEN, id=XML_MBS_HIGH).get_text())
            geo_len_sign = slice.find(XML_MBS_GEO_LEN, id=XML_MBS_SIGN).get_text()
            geo_len = [geo_len_low, geo_len_high, geo_len_sign]
            sin_dst_low = float(slice.find(XML_MBS_SIN, id=XML_MBS_LOW).get_text())
            sin_dst_high = float(slice.find(XML_MBS_SIN, id=XML_MBS_HIGH).get_text())
            sin_dst_sign = slice.find(XML_MBS_SIN, id=XML_MBS_SIGN).get_text()
            sin_dst = [sin_dst_low, sin_dst_high, sin_dst_sign]
            nv_dst_low = float(slice.find(XML_MBS_NV, id=XML_MBS_LOW).get_text())
            nv_dst_high = float(slice.find(XML_MBS_NV, id=XML_MBS_HIGH).get_text())
            nv_dst_sign = slice.find(XML_MBS_NV, id=XML_MBS_SIGN).get_text()
            nv_dst = [nv_dst_low, nv_dst_high, nv_dst_sign]
            self.__slices.append(Slice(name, side, cont, eu_dst, geo_dst, geo_len, sin_dst, nv_dst))

###########################################################################################
# Class for loading and getting synapse slices settings
# idx: for all get or testing function identifies the slice
###########################################################################################

class SynSliceSet(object):

    # fname: path to the XML file with slice settings
    def __init__(self, fname):
        self.__fname = fname
        self.__soup = None
        self.__slices = list()
        self.__load()

    #### Set/Get methods area

    def get_file_name(self):
        return self.__fname

    def get_slices_list(self):
        return self.__slices

    def get_num_slices(self):
        return len(self.__slices)

    #### External functionality area

    #### Internal functionality area

    # Load XML file and fill up correctly object variables
    def __load(self):

        # Loading XML File
        try:
            self.__soup = BeautifulSoup(open(self.__fname), "xml")
        except:
            error_msg = 'File %s could not be read.' % self.__fname
            raise pexceptions.PySegInputError(expr='(SliceSet)', msg=error_msg)

        # Parsing XML file
        for slice in self.__soup.find_all(XML_SYN):
            name = slice.find(XML_MBS_NAME).get_text()
            mb = int(slice.find(XML_SYN_MB).get_text())
            seg = int(slice.find(XML_SYN_SEG).get_text())
            cont = slice.find(XML_MBS_CONT).get_text()
            eu_dst_low = float(slice.find(XML_MBS_EU_DST, id=XML_MBS_LOW).get_text())
            eu_dst_high = float(slice.find(XML_MBS_EU_DST, id=XML_MBS_HIGH).get_text())
            eu_dst_sign = slice.find(XML_MBS_EU_DST, id=XML_MBS_SIGN).get_text()
            eu_dst = [eu_dst_low, eu_dst_high, eu_dst_sign]
            geo_dst_low = float(slice.find(XML_MBS_GEO_DST, id=XML_MBS_LOW).get_text())
            geo_dst_high = float(slice.find(XML_MBS_GEO_DST, id=XML_MBS_HIGH).get_text())
            geo_dst_sign = slice.find(XML_MBS_GEO_DST, id=XML_MBS_SIGN).get_text()
            geo_dst = [geo_dst_low, geo_dst_high, geo_dst_sign]
            geo_len_low = float(slice.find(XML_MBS_GEO_LEN, id=XML_MBS_LOW).get_text())
            geo_len_high = float(slice.find(XML_MBS_GEO_LEN, id=XML_MBS_HIGH).get_text())
            geo_len_sign = slice.find(XML_MBS_GEO_LEN, id=XML_MBS_SIGN).get_text()
            geo_len = [geo_len_low, geo_len_high, geo_len_sign]
            sin_dst_low = float(slice.find(XML_MBS_SIN, id=XML_MBS_LOW).get_text())
            sin_dst_high = float(slice.find(XML_MBS_SIN, id=XML_MBS_HIGH).get_text())
            sin_dst_sign = slice.find(XML_MBS_SIN, id=XML_MBS_SIGN).get_text()
            sin_dst = [sin_dst_low, sin_dst_high, sin_dst_sign]
            nv_dst_low = float(slice.find(XML_MBS_NV, id=XML_MBS_LOW).get_text())
            nv_dst_high = float(slice.find(XML_MBS_NV, id=XML_MBS_HIGH).get_text())
            nv_dst_sign = slice.find(XML_MBS_NV, id=XML_MBS_SIGN).get_text()
            nv_dst = [nv_dst_low, nv_dst_high, nv_dst_sign]
            syn_slice = SynSlice(name, mb, seg, cont, eu_dst, geo_dst, geo_len, sin_dst, nv_dst)
            for thres_entry in slice.find_all(XML_MBT):
                name = thres_entry.find(XML_MBS_NAME).get_text()
                val_prop = thres_entry.find(XML_MBT_VAL, id=XML_MBT_PROP).get_text()
                val_low = float(thres_entry.find(XML_MBT_VAL, id=XML_MBT_LOW).get_text())
                val_high = float(thres_entry.find(XML_MBT_VAL, id=XML_MBT_HIGH).get_text())
                val_mod = thres_entry.find(XML_MBT_VAL, id=XML_MBT_MODE).get_text()
                val = (val_low, val_high)
                syn_slice.add_threshold(ThresSlice(name, val_prop, val, val_mod))
            self.__slices.append(syn_slice)


###########################################################################################
# Class for modelling a linker between two membrane slices
###########################################################################################

class Linker(object):

    def __init__(self, name, s_name, t_name, eu_dst_r, geo_dst_r, sin_r, nn_r):
        self.__name = name
        self.__s_name = s_name
        self.__t_name = t_name
        self.__eu_dst = eu_dst_r
        self.__geo_dst = geo_dst_r
        self.__sin = sin_r
        self.__nn = nn_r

    #### Set/Get methods area

    def get_name(self):
        return self.__name

    def get_source_name(self):
        return self.__s_name

    def get_target_name(self):
        return self.__t_name

    def get_eu_dst_low(self):
        return self.__eu_dst[0]

    def get_eu_dst_high(self):
        return self.__eu_dst[1]

    def get_eu_dst_sign(self):
        return self.__eu_dst[2]

    def get_geo_dst_low(self):
        return self.__geo_dst[0]

    def get_geo_dst_high(self):
        return self.__geo_dst[1]

    def get_geo_dst_sign(self):
        return self.__geo_dst[2]

    def get_sin_low(self):
        return self.__sin[0]

    def get_sin_high(self):
        return self.__sin[1]

    def get_sin_sign(self):
        return self.__sin[2]

    def get_neighs_low(self):
        return self.__nn[0]

    def get_neighs_high(self):
        return self.__nn[1]

    def get_neighs_sign(self):
        return self.__nn[2]

    #### External methods area

    def test(self, eu_dst=None, geo_dst=None, sin=None, neighs=None):

        if eu_dst is not None:
            if (eu_dst < self.__eu_dst[0]) or (eu_dst > self.__eu_dst[1]):
                if self.__eu_dst[2] == '+':
                    return False
        if geo_dst is not None:
            if (geo_dst < self.__geo_dst[0]) or (geo_dst > self.__geo_dst[1]):
                if self.__geo_dst[2] == '+':
                    return False
        if sin is not None:
            if (sin < self.__sin[0]) or (sin > self.__sin[1]):
                if self.__sin[2] == '+':
                    return False
        if neighs is not None:
            if (neighs < self.__nn[0]) or (neighs > self.__nn[1]):
                if self.__nn[2] == '+':
                    return False

        # Default
        return True

###########################################################################################
# Class for loading and getting membrane slices linkers
# idx: for all get or testing function identifies the slice
###########################################################################################

class LinkerSet(object):

    # fname: path to the XML file with slice settings
    def __init__(self, fname):
        self.__fname = fname
        self.__soup = None
        self.__linkers = list()
        self.__load()

    #### Set/Get methods area

    def get_linkers_list(self):
        return self.__linkers

    def get_num_linkers(self):
        return len(self.__linkers)

    #### Internal functionality area

    # Load XML file and fill up correctly object variables
    def __load(self):

        # Loading XML File
        try:
            self.__soup = BeautifulSoup(open(self.__fname), "xml")
        except:
            error_msg = 'File %s could not be read.' % self.__fname
            raise pexceptions.PySegInputError(expr='(LinkerSet)', msg=error_msg)

        # Parsing XML file
        # try:
        for link_entry in self.__soup.find_all(XML_SYN):
            name = link_entry.find(XML_MBS_NAME).get_text()
            s_name = link_entry.find(XML_MBL_SNAME).get_text()
            t_name = link_entry.find(XML_MBL_TNAME).get_text()
            eu_dst_low = float(link_entry.find(XML_MBL_EU_DST, id=XML_MBS_LOW).get_text())
            eu_dst_high = float(link_entry.find(XML_MBL_EU_DST, id=XML_MBS_HIGH).get_text())
            eu_dst_sign = link_entry.find(XML_MBL_EU_DST, id=XML_MBS_SIGN).get_text()
            eu_dst_r = [eu_dst_low, eu_dst_high, eu_dst_sign]
            geo_dst_low = float(link_entry.find(XML_MBL_GEO_DST, id=XML_MBS_LOW).get_text())
            geo_dst_high = float(link_entry.find(XML_MBL_GEO_DST, id=XML_MBS_HIGH).get_text())
            geo_dst_sign = link_entry.find(XML_MBL_GEO_DST, id=XML_MBS_SIGN).get_text()
            geo_dst_r = [geo_dst_low, geo_dst_high, geo_dst_sign]
            sin_low = float(link_entry.find(XML_MBL_SIN, id=XML_MBS_LOW).get_text())
            sin_high = float(link_entry.find(XML_MBL_SIN, id=XML_MBS_HIGH).get_text())
            sin_sign = link_entry.find(XML_MBL_SIN, id=XML_MBS_SIGN).get_text()
            sin_r = [sin_low, sin_high, sin_sign]
            nn_low = float(link_entry.find(XML_MBL_NN, id=XML_MBS_LOW).get_text())
            nn_high = float(link_entry.find(XML_MBL_NN, id=XML_MBS_HIGH).get_text())
            nn_sign = link_entry.find(XML_MBL_NN, id=XML_MBS_SIGN).get_text()
            nn_r = [nn_low, nn_high, nn_sign]
            self.__linkers.append(Linker(name, s_name, t_name, eu_dst_r, geo_dst_r, sin_r, nn_r))

###########################################################################################
# Class for modelling a slice thresholder
###########################################################################################

class ThresSlice(object):

    def __init__(self, name, prop_key, range_val, mode):
        self.__name = name
        self.__prop_key = prop_key
        self.__rval = range_val
        self.__mode = mode

    #### Set/Get methods area

    def get_name(self):
        return self.__name

    def get_prop_key(self):
        return self.__prop_key

    def get_range(self):
        return self.__rval

    def get_mode(self):
        return self.__mode

    def get_value_low(self):
        return self.__rval[0]

    def get_value_high(self):
        return self.__rval[1]

    #### External methods area

    def test(self, in_val):
        if (in_val >= self.__rval[0]) and (in_val <= self.__rval[1]):
            if self.__mode == 'in':
                return True
            else:
                return False

# ####################################################################################################
# Class for loading slice thresholding operations
#
#
class ThresSliceSet(object):

    #### Constructor Area

    # fname: file with the XML file
    def __init__(self, fname):
        self.__fname = fname
        self.__soup = None
        self.__thres = list()
        self.__load()

    #### Set/Get methods area

    def get_thres_list(self):
        return self.__thres

    def get_num_thres(self):
        return len(self.__thres)

    #### Internal functionality area

    # Load XML file and fill up correctly object variables
    def __load(self):

        # Loading XML File
        try:
            self.__soup = BeautifulSoup(open(self.__fname), "xml")
        except:
            error_msg = 'File %s could not be read.' % self.__fname
            raise pexceptions.PySegInputError(expr='(ThresSlice)', msg=error_msg)

        # Parsing XML file
        # try:
        for thres_entry in self.__soup.find_all(XML_MBT):
            name = thres_entry.find(XML_MBS_NAME).get_text()
            val_prop = thres_entry.find(XML_MBT_VAL, id=XML_MBT_PROP).get_text()
            val_low = float(thres_entry.find(XML_MBT_VAL, id=XML_MBT_LOW).get_text())
            val_high = float(thres_entry.find(XML_MBT_VAL, id=XML_MBT_HIGH).get_text())
            val_mod = thres_entry.find(XML_MBT_VAL, id=XML_MBT_MODE).get_text()
            val = (val_low, val_high)
            self.__thres.append(ThresSlice(name, val_prop, val, val_mod))

###########################################################################################
# Class for modelling a RWR group
###########################################################################################

class RwrGroup(object):

    def __init__(self, name, slices, signs, rwr_cs):
        self.__name = str(name)
        if (not isinstance(slices, list)) or (not isinstance(signs, list)) or (not isinstance(rwr_cs, list)):
            error_msg = 'Input slices must be a list %s.'
            raise pexceptions.PySegInputError(expr='__init__() (RwrGroup)', msg=error_msg)
        if (len(slices) != len(signs)) or (len(slices) != len(rwr_cs)):
            error_msg = 'Input slice is not well construcuted.'
            raise pexceptions.PySegInputError(expr='__init__() RwrGroup', msg=error_msg)
        self.__slices = list()
        for slice in slices:
            if not isinstance(slice, SynSlice):
                error_msg = 'Input slice is not well construcuted.'
                raise pexceptions.PySegInputError(expr='__init__() RwrGroup', msg=error_msg)
            self.__slices.append(slice)
        self.__signs = list()
        for sign in signs:
            if str(sign) == XML_RWR_SP:
                self.__signs.append(XML_RWR_SP)
            else:
                self.__signs.append(XML_RWR_SN)
        self.__rwr_cs = list()
        for rwr_c in rwr_cs:
            if rwr_c < .0:
                self.__rwr_cs.append(.0)
            elif rwr_c > 1.:
                self.__rwr_cs.append(1.)
            self.__rwr_cs.append(float(rwr_c))

    #### Set/Get methods area

    def get_name(self):
        return self.__name

    def get_num_slices(self):
        return len(self.__slices)

    def get_slice(self, idx):
        return self.__slices[idx]

    def get_sign(self, idx):
        return self.__signs[idx]

    def get_rwr_c(self, idx):
        return self.__rwr_cs[idx]

    #### External methods area

###########################################################################################
# Class for loading and getting RWR groups from a XML file
# idx: for all get or testing function identifies the slice
###########################################################################################

class RwrGroupSet(object):

    # fname: path to the XML file with slice settings
    def __init__(self, fname):
        self.__fname = fname
        self.__soup = None
        self.__groups = list()
        self.__load()

    #### Set/Get methods area

    def get_file_name(self):
        return self.__fname

    def get_groups_list(self):
        return self.__groups

    def get_num_groups(self):
        return len(self.__groups)

    #### External functionality area

    #### Internal functionality area

    # Load XML file and fill up correctly object variables
    def __load(self):

        # Loading XML File
        try:
            self.__soup = BeautifulSoup(open(self.__fname), "xml")
        except:
            error_msg = 'File %s could not be read.' % self.__fname
            raise pexceptions.PySegInputError(expr='__load() (RwrGroupSet)', msg=error_msg)

        # Parsing XML file
        for group in self.__soup.find_all(XML_RWR):

            name = group.find(XML_RWR_NAME).get_text()

            # Loop for slices
            slc_l, sgn_l, rwr_c_l = list(), list(), list()
            for slc in group.find_all(XML_SYN):
                sl_name = slc.find(XML_MBS_NAME).get_text()
                mb = int(slc.find(XML_SYN_MB).get_text())
                seg = int(slc.find(XML_SYN_SEG).get_text())
                sgn = slc.find(XML_RWR_SIGN).get_text()
                rwr_c = float(slc.find(XML_RWR_C).get_text())
                eu_dst_low = float(slc.find(XML_MBS_EU_DST, id=XML_MBS_LOW).get_text())
                eu_dst_high = float(slc.find(XML_MBS_EU_DST, id=XML_MBS_HIGH).get_text())
                eu_dst_sign = slc.find(XML_MBS_EU_DST, id=XML_MBS_SIGN).get_text()
                eu_dst = [eu_dst_low, eu_dst_high, eu_dst_sign]
                geo_dst_low = float(slc.find(XML_MBS_GEO_DST, id=XML_MBS_LOW).get_text())
                geo_dst_high = float(slc.find(XML_MBS_GEO_DST, id=XML_MBS_HIGH).get_text())
                geo_dst_sign = slc.find(XML_MBS_GEO_DST, id=XML_MBS_SIGN).get_text()
                geo_dst = [geo_dst_low, geo_dst_high, geo_dst_sign]
                geo_len_low = float(slc.find(XML_MBS_GEO_LEN, id=XML_MBS_LOW).get_text())
                geo_len_high = float(slc.find(XML_MBS_GEO_LEN, id=XML_MBS_HIGH).get_text())
                geo_len_sign = slc.find(XML_MBS_GEO_LEN, id=XML_MBS_SIGN).get_text()
                geo_len = [geo_len_low, geo_len_high, geo_len_sign]
                sin_dst_low = float(slc.find(XML_MBS_SIN, id=XML_MBS_LOW).get_text())
                sin_dst_high = float(slc.find(XML_MBS_SIN, id=XML_MBS_HIGH).get_text())
                sin_dst_sign = slc.find(XML_MBS_SIN, id=XML_MBS_SIGN).get_text()
                sin_dst = [sin_dst_low, sin_dst_high, sin_dst_sign]
                nv_dst_low = float(slc.find(XML_MBS_NV, id=XML_MBS_LOW).get_text())
                nv_dst_high = float(slc.find(XML_MBS_NV, id=XML_MBS_HIGH).get_text())
                nv_dst_sign = slc.find(XML_MBS_NV, id=XML_MBS_SIGN).get_text()
                nv_dst = [nv_dst_low, nv_dst_high, nv_dst_sign]
                slc_l.append(SynSlice(sl_name, mb, seg, '-', eu_dst, geo_dst, geo_len, sin_dst, nv_dst))
                sgn_l.append(sgn)
                rwr_c_l.append(rwr_c)

            self.__groups.append(RwrGroup(name, slc_l, sgn_l, rwr_c_l))