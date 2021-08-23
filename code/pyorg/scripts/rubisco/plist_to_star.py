"""

    Convert an XML particle list into a STAR file

    Input:  - XML particle list
            - Reference tomogram

    Output: - A STAR file with particles

"""

################# Package import

import os
import numpy as np
import scipy as sp
import sys
import time
from pyorg import pexceptions, sub, disperse_io, surf

###### Global variables

__author__ = 'Antonio Martinez-Sanchez'


########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '/fs/pool/pool-engel/antonio/rubisco/org'

# Input XML file
in_xml = ROOT_PATH + '/in/xmls/Tomo6L2_pl_p_bin4.xml' # '/in/xmls/Tomo1_Lam6_pl_p_bin4.xml' # '/in/xmls/L6Tomo6_pl_p_bin4.xml' # '/in/xmls/L3Tomo3_tp_bin4.xml' # '/in/xmls/L2Tomo8_pl_p_bin4.xml'

# Input reference tomogram (_rlnMicrographName)
in_mic = ROOT_PATH + '/in/masks/Tomo6L2_bin4_40_410_MASKall_inv.mrc' # '/in/masks/Tomo1_Lam6_bin4_40_400_MASKall_inv.mrc' # '/in/masks/L6Tomo6_bin4_100_370_MASKall_inv.mrc' # '/in/masks/L3Tomo3_fixedtilt_bin4noCTF_100_360_MASKall_inv.mrc' # '/in/masks/L2Tomo8_bin4_50_430_MASKall_inv.mrc'

# Output STAR file
out_star = ROOT_PATH + '/in/stars/Tomo6L2_pl_p_bin4_rln.star' # '/in/stars/Tomo1_Lam6_pl_p_bin4_rln.star' # '/in/stars/L6Tomo6_pl_p_bin4_rln.star' # '/in/stars/L3Tomo3_tp_bin4_rln.star' # '/in/stars/L2Tomo8_pl_p_bin4_rln.star'
out_svol = ROOT_PATH + '/in/svols/Tomo6L2_4bin1' # '/in/svols/Tomo1_Lam6_4bin1' # '/in/svols/L6Tomo6_4bin1' # '/in/svols/L3Tomo3_4bin1' # '/in/svols/L2Tomo8_4bin1' # None
out_wedge = ROOT_PATH + '/in/rec/wedge_28_30.mrc' # None
out_ref = None # '/fs/pool/pool-engel/4rubisco_k2/project/4bin1class/L2Tomo8/avg_p.em'
out_avg = ROOT_PATH + '/in/rec/dr_avg_Tomo6L2.mrc' # '/in/rec/dr_avg_Tomo1_Lam6.mrc' # '/in/rec/dr_avg_L6Tomo6.mrc' # '/in/rec/dr_avg_L3Tomo3.mrc' # '/in/rec/dr_avg_L2Tomo8.mrc'

# Pre-processing
pr_swapxy = True
pr_inv = True

########################################################################################
# MAIN ROUTINE
########################################################################################

########## Print initial message

print('Convert a XML particle list into a STAR file.')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('Options:')
print('\tOutput file: ' + str(out_star))
in_xml_ext = os.path.splitext(in_xml)[1]
if in_xml_ext == '.xml':
    print('\tInput XML file of particles: ' + str(in_xml))
else:
    print('ERROR: No valid input format file for particles: ' + str(in_xml_ext))
    print('Terminated. (' + time.strftime("%c") + ')')
    sys.exit(-1)
if out_svol is not None:
    print('\tSub-volumes a stored as MRC in folder: ' + str(out_svol))
if pr_inv:
    print('\t\t+Gray values will be inverted.')
if out_wedge is not None:
    print('\tAdd the wedge in path: ' + str(out_wedge))
print('\t\t-Input reference tomogram: ' + str(in_mic))
print('\tPre-processing: ')
if pr_swapxy:
    print('\t\t-Swap X and Y coordinates.')
print('')

######### Process

print('Main Routine: ')

print('\tLoading input XML file...')
plist = sub.ParticleList(in_mic)
try:
    plist.load(in_xml)
except pexceptions.PySegInputError as e:
    print('ERROR: input XML file could not be loaded because of "' + e.get_message() + '"')
    print('Terminated. (' + time.strftime("%c") + ')')
    sys.exit(-1)

print('\tConverting to STAR file...')
star = sub.Star()
star.from_TomoPeaks(plist, svols_path=out_svol, gray_inv=pr_inv, ctf_path=out_wedge, del_aln=False, ref=out_ref,
                    keep_pytom=False)
star.add_column('_rlnMicrographName')
for row in range(star.get_nrows()):
    star.set_element('_rlnMicrographName', row, in_mic)

print('\tSaving output STAR file...')
try:
    star.store(out_star)
except pexceptions.PySegInputError as e:
    print('ERROR: output STAR file could not be saved because of "' + e.get_message() + '"')
    print('Terminated. (' + time.strftime("%c") + ')')
    sys.exit(-1)

if out_avg is not None:
    print('\tComputing the average (without considering CTF)...')
    disperse_io.save_numpy(star.compute_avg(pytom=False), out_avg)
    print('\t\t-Stored in: ' + str(out_avg))

print('Terminated. (' + time.strftime("%c") + ')')
