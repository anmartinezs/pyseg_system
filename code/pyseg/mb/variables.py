"""

    Global variables for the package

"""

__author__ = 'martinez'

import numpy as np

MAX_FLOAT = np.finfo(float).max
MB_SEG = 'mb_seg'
MB_LBL = 1
MB_IN_LBL = 2
MB_OUT_LBL = 3

STR_H = 1
STR_T = 2
STR_C = 3
STR_F = 4

STR_CELL = 'cell'
STR_STR = 'str'
STR_LEN = 'len'
STR_PEN = 'pen'
STR_PENT = 'pent'
STR_PEN_LEN = 'pen_len'
STR_DST = 'dst'
STR_FNESS = 'fil_fness'
STR_SNESS = 'fil_sness'
STR_MNESS = 'fil_mness'
STR_DNESS = 'fil_dness'
STR_CLST = 'clst'
STR_CLST_EU = 'clst_eu'
STR_SIDE = 'side'
STR_CT = 'fil_ct'
STR_MC = 'fil_mc'
STR_SIN = 'fil_sin'
STR_SMO = 'fil_smo'
STR_ALPHA = 'fil_alpha'
STR_BETA = 'fil_beta'
STR_NORM = 'cont_norm'
STR_CARD = 'cont_card'

# Membrane
MB_SEG = 'mb_seg'
MB_EU_DST = 'mb_eu_dst'
MB_GEO_DST = 'mb_geo_dst'
MB_GEO_LEN = 'mb_geo_len'
MB_GEO_SIN = 'mb_geo_sin'
MB_GEO_KT = 'mb_geo_kt'
MB_GEO_TT = 'mb_geo_tt'
MB_GEO_NS = 'mb_geo_ns'
MB_GEO_BS = 'mb_geo_bs'
MB_GEO_APL = 'mb_geo_apl'
MB_CONT_COORD = 'mb_cont_coords'

MB_VTP_STR = 'struct'
MB_VTP_STR_C = 1
MB_VTP_STR_F = 2
MB_VTP_STR_E = 3

# Synapse
SYN_SEG = 'syn_seg'
MB_PST_EU_DST = 'pst_eu_dst'
MB_PRE_EU_DST = 'pre_eu_dst'
MB_PST_GEO_DST = 'pst_geo_dst'
MB_PRE_GEO_DST = 'pre_geo_dst'
MB_PST_GEO_LEN = 'pst_geo_len'
MB_PRE_GEO_LEN = 'pre_geo_len'
MB_PST_GEO_SIN = 'pst_geo_sin'
MB_PRE_GEO_SIN = 'pre_geo_sin'
MB_PST_CONT_COORD = 'pst_cont_coords'
MB_PRE_CONT_COORD = 'pre_cont_coords'

SYN_PST_LBL = 1
SYN_PRE_LBL = 2
SYN_PSD_LBL = 3
SYN_AZ_LBL = 4
SYN_CLF_LBL = 5
