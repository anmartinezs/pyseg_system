"""
Sorts series, fixes image headers, limits the data and calculates the dose.
All this works if all images compising the series are saved in separate
files.

For series saved as a stack, electron dose can be calculated

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id: sort_series.py 1367 2016-12-14 15:51:56Z vladan $
"""

__version__ = "$Revision: 1367 $"

import logging
import numpy
import pyto


############################################################
#
# Parameters
#
##############################################################

# input series pattern
# Example: in_path = '../em-series/neu.*em' matches all files in '../em-series' 
# directory that start with 'neu' and end with 'em'. 
in_path = '../em-series/neu.*em'  

# regular expression match mode for in_path, one of the following:
#  - 'search':  (like re.search) pattern somewhere in the target string
#  - 'match': (like re.match) target string begins with the pattern
#  - 'exact': whole target string is mathed 
mode = 'exact'

# flag indicating if the complete input series is in a single stack file 
# (True), or the images are saved in separate files (False)
in_stack = True

# name of the tilt file (used only if input series is a stack)
tilt_angles_file = 'ck01-03_bin-0.tlt'

# sorted series name (uses the same format as in_path)
out_path = 'neu_'

# number of digits of a projection number in the sorted series file name
out_pad = 2

# how the image headers are fixed: None (no fixing), 'auto' (determined from
# microscope, 'polara_fei-tomo' (for polara 1 and 2 with FEI software) or
# 'cm300' (cm 300). If 'auto', it is determined from microscope parameter. 
header_fix_mode = 'auto'

# microscope type, options: 
#  - 'cm300': CM300
#  - 'polara-1_01-07': Polara 1 from 01.2007 - 12.2008, FEI tomo
#  - 'polara-1_01-09': Polara 1 from 01.2009 - present, FEI tomo 
#  - 'polara-2_01-09': Polara 2 from 01.2009 - present, FEI tomo
#  - 'polara-2_k2-count_sem': Polara 2, K2, SerialEM
#  - 'krios-2_falcon_05-2011': Krios 2, Falcon detector from 2011 - present
#  - 'titan-2_k2-count_sem'
#  - 'f20_eagle'
#  -  None: other microscope 
microscope = None

# counts per pixel
# Needs to be specified only if microscope is None, otherwise it is ignored
counts_per_pixel = 1

# greyscale values limits are the average plus or minus this number of std's
limit = 4

# size of the subarray used to find the replacement value for a voxel that's out
# of the limits 
size = 5

# only calculate the dose (if True parameters between out_path and size are
# not used)
dose_only = False

# print info messages
print_info = True

# file name where tilt and dose are written for each projection (as needed 
# for Relion tomo processing); None for no file
projection_dose_file = 'ck01-01_bin-0.order'

# print lot of info messages
print_debug = False


###########################################################
#
# Work
#
###########################################################

def main():
    """
    Main function
    """

    # determine fix_mode
    if header_fix_mode is None:
        fix_mode = None

    elif header_fix_mode == 'auto':
        if ((microscope == 'polara-1_01-07') 
            or (microscope == 'polara-1_01-09') 
            or (microscope == 'polara-2_01-09')):
            fix_mode = 'polara_fei-tomo'
        elif microscope == 'krios-2_falcon_05-2011':
            fix_mode = 'krios_fei-tomo'
        elif microscope == 'cm300':
            fix_mode = 'cm300'
        elif microscope == 'polara-2_k2-count_sem':
            fix_mode = None
        elif microscope == 'titan-2_k2-count_sem':
            fix_mode = None
        else:
            raise ValueError(
                "In header_fix_mode_auto = 'auto', microscope: ", 
                microscope, " .")
    else:
        fix_mode = header_fix_mode

    # set logging
    if print_debug:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(levelname)s %(message)s')
    elif print_info:
        logging.basicConfig(level=logging.INFO,
                            format='%(levelname)s %(message)s')

    logging.debug('DEBUG')
    logging.info("INFO")

    # set conversion factor
    if microscope is None:
        conversion = counts_per_pixel
    else:
        conversion=pyto.io.microscope_db.conversion[microscope]

    if not dose_only:

        # just check the series
        in_ser = pyto.tomo.Series(path=in_path, mode=mode)
        logging.info("Checking the series:")
        logging.info('  Angle        Old name    ->     New name')
        in_ser.sort(test=True)

        # remove (manually) bad projections
        # ToDo: check the angles and complain if needed
        print('\nIf a tilt angle is repeated interrupt the procedure (Ctrl-C)')
        raw_input('and remove bad projection(s), otherwise press Return:')

        # sort and correct the series 
        logging.info("Sorting the series:")
        logging.info('  Angle        Old name    ->     New name     Mean  Std')
        in_ser.sort(
            out=out_path, pad=out_pad, fix_mode=fix_mode, 
            microscope=microscope, limit=limit, limit_mode='std', size=size)

        # get dose for the sorted (corrected) series
        logging.info("Calculating the dose:")
        corr_ser = pyto.tomo.Series(path=in_path, mode='match')
        #corr_ser.getDose(conversion=conversion)

    else:

        # dose only
        logging.info("Calculating the dose:")
        corr_ser = pyto.tomo.Series(
            path=in_path, stack=in_stack, mode=mode, tilt_file=tilt_angles_file)

    # get dose
    if projection_dose_file is not None:
        tot, mean, proj_dose = corr_ser.getDose(
            conversion=conversion, projection_dose=True)
        numpy.savetxt(projection_dose_file, proj_dose, fmt='%7.2f %8.3f')

    else:
        corr_ser.getDose(conversion=conversion)


# run if standalone
if __name__ == '__main__':
    main()
