"""
Useful functions that can modify or execute many files (scripts) together.

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id: bulk.py 1311 2016-06-13 12:41:50Z vladan $
"""

__version__ = "$Revision: 1311 $"

import os
import sys
import imp
import runpy
import warnings
import logging
import re
from copy import copy, deepcopy


def replace(old, rules, new=None, repeat=None):
    """
    Writes a new file where lines of the old file are modified according
    to the arg rules. 

    Each line is modified by all rules that apply. The order of the rule
    application is not specified, the order is actually given by rules.items().

    A line that matches arg repeat is first copied and then (the second copy
    of the line is) modified according to the rules. 

    If new is None, the (old) file is replaced by the modified one. In that 
    case arg old can only be a (string) file name. Alternatively, if old and 
    new are the same file names the modified file will replace the old one.

    Arguments:
      - old, new: old and new files, either file name strings or open file 
      descriptors
      - rules: (dict) replacement rules
      - repeat: (list) specifies lines to repeat
    """

    # open files if needed
    if not isinstance(old, file):
        old_fd = open(old)
    else:
        old_fd = old

    # modify lines
    new_lines = []
    for line in old_fd:

        # copy line if needed
        if repeat is not None:
            for line_pattern in repeat:
                if re.search(line_pattern, line) is not None:
                    new_lines.append(line)

        # check all line match patterns for replacement
        line_written = False
        modif_line = copy(line)
        for pattern, value in rules.items():
            if re.search(pattern, modif_line) is not None:

                # modify the current line using the current rule
                if isinstance(value, dict):
                    for pat, sub_ in value.items():
                        modif_line = re.sub(pat, sub_, modif_line)
                else:
                    modif_line = value

        new_lines.append(modif_line)

        #if not line_written:

            # copy line
            #new_lines.append(line)

    old_fd.close()

    # write new file
    if new is None:
        new = old

    # open new file and make directories if needed
    if not isinstance(new, file):
        try:
            new_fd = open(new, 'w')
        except IOError:
            new_dir, new_base = os.path.split(new)
            os.makedirs(new_dir)
            new_script = open(new, 'w')
    else:
        new_fd = new

    # write 
    new_fd.writelines(new_lines)
    new_fd.flush()
    new_fd.close()

# internal, needed to insure that all loaded modules have unique names
_module_index = -1

def run_path(path, package=''):
    """
    Run one or more python files (modules).  

    Each module is executed from the directory where it resides, and the 
    current directory is afterwards reverted to the original one. 

    In the current version modules are first imported under unique names,
    which is useful when module files have the same names (but reside in 
    different directories).

    In Python <= 2.6 each file module need to be loaded. Its full module
    name is

      package + '.' + file_name + '_' + _module_index

    where arg package is an already existing package name and a current 
    value of variable _module_index. This variable starts at 0 and is increased
    after each module import.

    In Python >= 2.7 this should change so that modules do not need to be
    imported at all, but this isn't implemented yet.

    Argument:
      - path: (str or iterable) module path(s)
      - package: name of the package within which the module(s) is/are placed
    """

    global _module_index

    # get current dir
    if __name__ == '__main__':
        this_dir, this_base = os.path.split(os.path.abspath(__file__))
    else:
        this_dir = os.getcwd()

    # get current module name
    if package != '':
        package = package + '.'

    if isinstance(path, str):
        all_paths = [path]
    else:
        all_paths = path

    # run scrpts
    for script_path in all_paths:
        _module_index += 1

        # run
        try:

            # cd to the script directory
            dir_, name = os.path.split(script_path)
            os.chdir(dir_)

            if ((sys.version_info[0] >= 2) and (sys.version_info[1] >= 7) 
                and False):

                # new in 2.7
                # Need to check about mixing variables 
                runpy.run_path(name)

            else:

                # <= 2.6, currently working
                # import module under a unique name
                base, ext = os.path.splitext(name)
                mod_file, mod_dir, mod_desc = imp.find_module(base, ['.'])
                mod_name = package + base + '_' + str(_module_index)
                mod = imp.load_module(mod_name, mod_file, mod_dir, mod_desc)

                # run the module
                mod.main()

        finally:

            # back to original directory
            os.chdir(this_dir)


