__author__ = 'martinez'

import os
import numpy as np

#####################################################################################################
# File with the set of the globals variables of PySeg package
#

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
DATA_DIR = ROOT_DIR + '/data'
DPS_DIR = '/home/martinez/apps/disperse/bin/'

# DisPerSe commands
DPS_MSE_CMD = 'mse'
DPS_NCONV_CMD = 'netconv'
DPS_SCONV_CMD = 'skelconv'

# DisPerSe ID
DPID_CRITICAL_MIN = 0
DPID_CRITICAL_SAD = 1

# DisPerSe strings
DPSTR_CELL = 'cell'
DPSTR_ROBUSTNESS = 'robustness'

# VERTICES ID
VERTEX_TYPE_ANCHOR_FRONT = 1
VERTEX_TYPE_ANCHOR_BACK = 2
VERTEX_TYPE_INNER = 3
VERTEX_TYPE_OUTER = 4
VERTEX_TYPE_NO_PROC = 5

# FOR IDENTIFYING CRITICAL POINTS
CRITICAL_MIN = 11
CRITICAL_MAX = 12
CRITICAL_SAD = 13

# Return states
NO_PROPERTY_INSERTED = 1001
ITEM_INSERTED = 1002
NO_ITEM_INSERTED = 1003
ITEM_DELETED = 1004
NO_ITEM_DELETED = 1005

# Static strings
STR_VERTEX_ID = 'vid'
STR_ARCS_ID = 'aid'
STR_FIL_ID = 'fid'
STR_ARCGRAPH_ID = 'agid'
STR_GRAPH_ID = 'gid'
STR_EDGE_LENGTH = 'length'
STR_SIGN_P = '+'
STR_SIGN_N = '-'
STR_NFIELD_ARRAY_NAME = 'Normals_'
STR_ATYPE_ARRAY_NAME = 'anchor_type'
STR_CID_ARRAY_NAME = 'ccell_id'
STR_CDIST_ARRAY_NAME = 'ccell_dist'
STR_CCENTER_ARRAY_NAME = 'ccell_center'
STR_CRITICAL_VERTEX = 'critical_point'
STR_DENSITY_VERTEX = 'density'
STR_PSD_SGRAPH = 'psd_sgraph'
STR_GRAPH_TOOL_ID = 'gt_id'
STR_GRAPH_TOOL_WEIGHT = 'gt_weight'
STR_ARC_MAX_DENSITY = 'arc_max_density'
STR_ARC_REDUNDANCY = 'arc_redundancy'
STR_VER_REDUNDANCY = 'vertex_redundancy'
STR_ARC_RELEVANCE = 'arc_relevance'
STR_GRAPH_DIAM = 'graph_diameter'
STR_CRITICAL_INDEX = 'critical_index'
STR_FIELD_VALUE = 'field_value'
STR_FIELD_VALUE_EQ = 'field_value_eq'
STR_FIELD_VALUE_INV = 'field_value_inv'
STR_VERTEX_RELEVANCE = 'vertex_relevance'
STR_GRAPH_RELEVANCE = 'graph_relevance'
STR_EXT_DIST = 'ext_dist'
STR_EXT_ID = 'ext_id'
STR_FIELD_VALUE_DISTANCE = 'field_dist'
STR_CLOUD_NORMALS = 'cloud_normals'
STR_RAND_ID = 'rand_id'
STR_RAND_WALKS = 'rand_walks'
STR_AVG_VDEN = 'avg_den'
STR_TOT_VDEN = 'tot_den'
STR_AH_CLUST = 'ah_clusters'
STR_AFF_CLUST = 'aff_clusters'
STR_AFF_CENTER = 'aff_centers'
STR_DBSCAN_CLUST = 'dbscan_clusters'
STR_BM_CLUST = 'bm_clusters'
STR_SP_CLUST = 'sp_clusters'
STR_PER = 'persistence'
STR_V_PER = 'v_persistence'
STR_V_FPER = 'f_persistence'
STR_V_PAIR = 'v_pair'
STR_HID = 'hold_id'
STR_VGPH = 'vgph'
STR_AGPH = 'agph'
STR_EDGE_INT = 'edge_int'
STR_EDGE_SIM = 'edge_sim'
STR_VERT_DST = 'vert_dst'
STR_MFLOW_EK = 'max_flow_ek'
STR_MFLOW_BK = 'max_flow_bk'
STR_MFLOW_PR = 'max_flow_pr'
STR_FLOW_SS = 'ss_flow'
STR_PSM_VBET_1 = 'vertex_bet_psm_1'
STR_PSM_VBET_2 = 'vertex_bet_psm_2'
STR_PSM_VBET_3 = 'vertex_bet_psm_3'
STR_PSM_EBET_1 = 'edge_bet_psm_1'
STR_PSM_EBET_2 = 'edge_bet_psm_2'
STR_PSM_EBET_3 = 'edge_bet_psm_3'
STR_EDGE_FNESS = 'edge_fness'
STR_EDGE_AFFINITY = 'edge_a'
STR_GGF = 'ggf'
STR_BIL = 'bilateral'
STR_SCC = 'scc'
STR_EDGE_UK = 'edge_uk'
STR_EDGE_K = 'edge_k'
STR_EDGE_UT = 'edge_ut'
STR_EDGE_T = 'edge_t'
STR_EDGE_SIN = 'edge_sin'
STR_EDGE_APL = 'edge_apl'
STR_EDGE_NS = 'edge_ns'
STR_EDGE_BNS = 'edge_bns'
STR_SIMP_MASK = 'simp_mask'
STR_EDGE_ZANG = 'edge_zang'
STR_EDGE_VECT = 'edge_vect'
STR_VERT_VECT = 'vert_vect'
STR_FWVERT_DST = 'fwvert_dst'
STR_MAX_LP = 'max_lp'
STR_MAX_LP_X = 'max_lp_x'

# SubGraphMCF
STR_SGM_VID = 'sgm_vid'
STR_SGM_EID = 'sgm_eid'

# GT static strings
SGT_EDGE_LENGTH = 'edge_length'
SGT_EDGE_LENGTH_W = 'edge_length_w'
SGT_EDGE_LENGTH_WTOTAL = 'edge_length_wtotal'
SGT_MIN_SP_TREE = 'min_sp_tree'
# GT centrality
SGT_PAGERANK = 'pagerank'
SGT_BETWEENNESS = 'betweenness'
SGT_CLOSENESS = 'closeness'
SGT_EIGENVECTOR = 'eigenvec'
SGT_KATZ = 'katz'
SGT_HITS_AUT = 'hits_aut'
SGT_HITS_HUB = 'hits_hub'
# GT clustering
SGT_LOCAL_CLUST = 'local_clust'
SGT_EXT_CLUST_1 = 'ext_clust_1'
SGT_EXT_CLUST_2 = 'ext_clust_2'
SGT_EXT_CLUST_3 = 'ext_clust_3'
SGT_NDEGREE = 'n_degree'
# Coordinates
SGT_COORDS = 'coords'
# Filament persitence
SGT_MAX_LP = 'max_lp'
SGT_MAX_LP_X = 'max_lp_x'

# PSD
PSD_SGRAPH_POST = 2001
PSD_SGRAPH_MBPOST = 2002
PSD_SGRAPH_PRE = 2003
PSD_SGRAPH_MBPRE = 2004
PSD_SGRAPH_CLFT = 2005

# PSD Network
PSD_NET_PSD_KEY = 'psd_net'
PSD_NET_PSMC_KEY = 'psmc_net'
CLFT_NET_KEY = 'clft_net'
PSD_NET_PSM_ANC_CITO = 1
PSD_NET_PSD_PATCH = 2
PSD_NET_PSM_INT_FIL = 3
PSD_NET_PSM_ED_INT_FIL = 4
PSD_NET_PSD_INT_FIL = 3
PSD_NET_PSD_ED_INT_FIL = 4
CLFT_NET_PSM_ANC_MB = 5
CLFT_NET_RSM_ANC_MB = 6
CLFT_NET_EXT_INT_FIL = 7
CLFT_NET_EXT_ED_INT_FIL = 8

# Membrane
MB_V_LBL = 'mb_lbl'
MB_V_TA = 'mb_ta'
MB_V_TT = 'mb_tt'
MB_V_TI = 'mb_ti'
MB_E_TA = 'edge_mb_ta'
MB_E_TT = 'edge_mb_tt'
MB_E_TI = 'edge_mb_ti'
MB_V_GEO = 'mb_geo'

# Parameters
NO_SET = -1

# Precision varibles
FLT_EPSILON = np.finfo(float).eps