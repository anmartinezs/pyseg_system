#!/usr/bin/env python
"""
Batch execution of scripts.

Modifies scripts is a specified way, executes them, and modifies files that
hold information about the results (catalogs). 

As an example of a possible usage existing scripts for segmentation and 
analysis of individual tomograms are first modified (to change parameters, for 
example) and saved under different names. Then the modified scripts are
executed. Finally, the information about the files holding the script execution 
results are added to the catalog files (contain metadata about individual 
tomograms and their segmentation and analysis).

$Id$
# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
"""
from __future__ import unicode_literals
#from builtins import str
from builtins import range

__version__ = "$Revision$"


import sys
import os
import re
import logging
import pickle
import runpy
import imp

import numpy 
import scipy 

import pyto

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s %(module)s.%(funcName)s():%(lineno)d %(message)s',
    datefmt='%d %b %Y %H:%M:%S')


##############################################################
#
# Parameters (need to be edited)
#
##############################################################

##############################################################
#
# General
#

# script directories (used for modifying and running the scripts) 
directories = (['ctrl-mouse_' + str(ind) for ind in range(1,8)]  
               + ['CAM_' + str(ind) for ind in range(1,8)] 
               + ['ctrl-rat_' + str(ind) for ind in range(1,6)]
               + ['egta_0' + sym for sym in ['a','b','c','d','e','f']] 
               + ['egta_1' + sym for sym in ['a','b','c','d','e']] 
               + ['egta_2']
               + ['egta_3' + sym for sym in ['a','b','c','d','e','f']]) 
directories_discard = ['syncam_ox_4', 'syncam_ox_7']
directories = [ident for ident in directories
               if ident not in directories_discard]

# list of script file paths
#scripts = ['../../all_tomograms/' + dir + '/cleft_test/cleft.py' 
#           for dir in ['CAM_1', 'CAM_2']]
scripts = ['../../all_tomograms/' + dir + '/cleft/cleft.py' 
           for dir in directories]

# identifiers (used only for catalogs)
identifiers = (['syncam_ox_ctrl_' + str(ind) for ind in range(1,8)]  
               + ['syncam_ox_' + str(ind) for ind in range(1,8)] 
               + ['egta_ctrl_' + str(ind) for ind in range(1,6)]
               + ['egta_0' + sym for sym in ['a','b','c','d','e','f']] 
               + ['egta_1' + sym for sym in ['a','b','c','d','e']] 
               + ['egta_2']
               + ['egta_3' + sym for sym in ['a','b','c','d','e','f']]) 
identifiers_discard = ['egta_ctrl_6']
identifiers = [ident for ident in identifiers
               if ident not in identifiers_discard]

##############################################################
#
# Modifying scripts
#

# flag indicating if scripts need to be modified
modify_scripts = True

# script name (path) pattern that is changed for the modified scripts
old_script_pattern = r'cleft\.py$'

# replacements for the script name pattern (above)
new_script_replace =  'cleft.py'

# rules for modifying scripts, all rules are applied to each line (from each 
# script), two forms are available:
#   - line_matching_pattern : new_line; each line that matches pattern
# is replaced by the specified line (new_line needs to end with \n)
#   - line_matching_pattern : (pattern, pattern_replacement); in each line 
# that matches line matching pattern, pattern is replaced by 
# pattern_replacement and the rest of the line is left unchanged  
script_rules = {
    r'^n_layers =' : 'n_layers = 4' + os.linesep,
    r'^n_extra_layers =' : 'n_extra_layers = 2' + os.linesep,
    r'^membrane_thickness' : 'membrane_thickness = 1' + os.linesep,
    r'^layers_only =' : 'layers_only = True' + os.linesep,
    r'^lay_suffix =' : {r'_layers\.em' : '_layers-4.em'},
    r'^cleft_res_suffix =' : {r'_cl\.dat' : '_cl_layer-4.dat'},
    r'^cl_suffix =' : {r'_cl\.pkl' : '_cl_layer-4.pkl'}
    }

##############################################################
#
# Running scripts
#

# flag indicating if scripts need to be run
run_scripts = True

##############################################################
#
# Modifying catalogs
#

# flag indicating if catalogs need to be modified
modify_catalogs = True

# catalog directory
catalog_dir = '../catalogs/'

# list of catalogs file names or name patterns 
#catalogs  = [r'[^_].*\.py$']   # extension 'py', doesn't start with '_'
#catalogs = [r'^syncam_ox_1.py$']
catalogs = [ident + '.py$' for ident in identifiers]

# catalog lines to copy, each matched lines is first copied unmodified and then
# modified by the catalog_rules and written again
#catalog_copy = [r'^cleft_segmentation_hi = ']
catalog_copy = [r'^cleft_layers =']

# rules for modifying lines of catalogs, same format as script_rules
catalog_rules = {
    r'^cleft_' : {r'cleft_vl' : 'cleft'},
    r'^cleft_layers =' : {r'_cl' : '_layers-thin'},
    r'^cleft_layers_4 =' : {r'_cl' : ''},
    r'^cleft_segmentation_hi_layers =' : {r'^cleft' : '#cleft'},
    r'^cleft_segmentation_hi_layers_4 =' : {r'_cl' : ''}
    }


################################################################
#
# Main function (edit if you know what you're doing)
#
###############################################################

def main():

    # write and run scripts
    for script_path in scripts:

        if modify_scripts:

            # open old and new scripts
            #old_script = open(script_path)
            new_path = re.sub(old_script_pattern, new_script_replace, 
                              script_path)
            #try:
            #    new_script = open(new_path, 'w')
            #except IOError:
            #    new_dir, new_base = os.path.split(new_path)
            #    os.makedirs(new_dir)
            #    new_script = open(new_path, 'w')
            # print "created by script ..."

            # write new (modified) script
            logging.info("Modifying script " + new_path)
            pyto.util.bulk.replace(old=script_path, new=new_path, 
                                   rules=script_rules)
            #new_script.close()

        else:

            # use existing script
            new_path = script_path
        
        # run script
        if run_scripts:
            logging.info("Runninging script " + new_path)
            pyto.util.bulk.run_path(path=new_path)

    # add entries to catalogs
    if modify_catalogs:

        # read directory
        # Note: can not put in the for loop statement below, because the
        # loop may modify the directory
        all_files = os.listdir(catalog_dir)

        # modify all catalogs
        for cat in all_files:

            # check all catalog patterns
            for cat_pattern in catalogs:

                # do all modifications for the current catalog
                if re.search(cat_pattern, cat) is not None:
                    cat_path = os.path.join(catalog_dir, cat)
                    logging.info("Modifying catalog " + cat_path)
                    pyto.util.bulk.replace(
                        old=cat_path, rules=catalog_rules, 
                        repeat=catalog_copy)
                    break


# run if standalone
if __name__ == '__main__':
    main()
