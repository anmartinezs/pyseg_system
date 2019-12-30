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

ROOT_PATH = '/fs/pool/pool-lucic2/antonio/tomograms/marion/Arp23complex'

# Input XML file
in_xml = ROOT_PATH + '/in/Waves/tomo1_135p.xml'

# Input reference tomogram (_rlnMicrographName)
in_mic = ROOT_PATH + '/in/Waves/tomo1_2bin_Inv_mp_bmask.em'

# Output STAR file
out_star = ROOT_PATH + '/in/Waves/tomo1_135p.star'

# Pre-processing
pr_swapxy = True

########################################################################################
# MAIN ROUTINE
########################################################################################

########## Print initial message

print 'Convert a XML particle list into a STAR file.'
print '\tAuthor: ' + __author__
print '\tDate: ' + time.strftime("%c") + '\n'
print 'Options:'
print '\tOutput file: ' + str(out_star)
in_xml_ext = os.path.splitext(in_xml)[1]
if in_xml_ext == '.xml':
    print '\tInput XML file of particles: ' + str(in_xml)
else:
    print 'ERROR: No valid input format file for particles: ' + str(in_xml_ext)
    print 'Terminated. (' + time.strftime("%c") + ')'
    sys.exit(-1)
print '\t\t-Input reference tomogram: ' + str(in_mic)
print '\tPre-processing: '
if pr_swapxy:
    print '\t\t-Swap X and Y coordinates.'
print ''

######### Process

print 'Main Routine: '

print '\tLoading input XML file...'
plist = sub.ParticleList(in_mic)
try:
    plist.load(in_xml)
except pexceptions.PySegInputError as e:
    print 'ERROR: input XML file could not be loaded because of "' + e.get_message() + '"'
    print 'Terminated. (' + time.strftime("%c") + ')'
    sys.exit(-1)

print '\tConverting to STAR file...'
star = sub.Star()
star.from_TomoPeaks(plist)
star.add_column('_rlnMicrographName')
for row in xrange(star.get_nrows()):
    star.set_element('_rlnMicrographName', row, in_mic)

print '\tSaving output STAR file...'
try:
    star.store(out_star)
except pexceptions.PySegInputError as e:
    print 'ERROR: output STAR file could not be saved because of "' + e.get_message() + '"'
    print 'Terminated. (' + time.strftime("%c") + ')'
    sys.exit(-1)

print 'Terminated. (' + time.strftime("%c") + ')'