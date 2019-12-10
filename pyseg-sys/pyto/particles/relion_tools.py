"""

Tools for manipulation starfiles used by Relion and Xmipp

Generic starfile functions:
  - get_array_data(): reads data from a star file
  - array_data_generator(): yields data lines from a star file
  - array_data_head(): reads header
  - write_table(): writes a star table
  - sort_particles(): sorts particles by file name and particle number
  - find_file(): finds files whose names match given criteria
  - add_priors(): copies rotations to the corresponding priors
 
Relion starfile processing functions:
  - print_class_summary(): prints number of particles 
  - extract_class(): extracts particles of a specified class
  - extract_random(): extracts random particles
  - get_data(): reads one particle property array (star file column)
  - get_class(): gets particle class membership
  - get_n_particle_class_change(): finds number of particles that changed 
  class between iterations
  - get_t_n_class_changes(): finds number of times a particle changed its
  class between specified iterations
  - classify_array(): classifies elements of an array according to a
  specified criterion

Statistics:
  - two_way_class(): comparison of two classifications
  - get_contingency(): makes contingency table for two-way data
  - cluster_similarity(): calculates similarity between two given clusterings

Transformations:
  - multi_transform(): multiple rigid3d transforms a particle (image)
  - write_ransformed(): rigid3d transforms a particle (image)
  - symmetrize_particle_data(): symmetrizes particle data  
  - symmetrize_structure(): symmetrizes a structure or a particle (image)

Plotting functions
  - plot_fsc(): plots fcs
  - plot_ssnr(): plots ssnr

Relion subtomo functions:
  - average_fixed(): makes average without alignment

Functions useful to convert relion data to xmipp:
  - convert_angle_distribution(): converts relion angular distribution to a 
  form that can be displayed by xmipp
  - convert_boxes(): convertsparticle box coordinates from relion to xmipp
  - convert_particle_selection(): selects particles from an xmipp particle 
  file that exsist in a specified relion particle file

# Author: Vladan Lucic
# $Id: relion_tools.py 1527 2019-04-10 13:47:34Z vladan $
"""

__version__ = "$Revision: 1527 $"


import os
import itertools
import subprocess
import random
import warnings
#from copy import copy, deepcopy
import re
import collections

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import pyto
from pyto.geometry.rigid_3d import Rigid3D
from pyto.geometry.affine_3d import Affine3D
from pyto.segmentation.cluster import Cluster


##############################################################
#
# Dealing with classes
#

def extract_class(
    basename, iters, class_=None, max_change=None, out=None, 
    suffix='_data.star', iter_format = '_it%03d', tablename='data_images'):
    """
    Finds particles that satisfy specified class-related conditions and writes
    them to a new data file.

    If arg class_ is not None, keeps only the particles that belong to the 
    specified class(es) at the last of the iterations given in arg iters. 

    If arg max_change is not None, keeps only those particles that changed
    class at most max_change times during all iterations specified in arg 
    iters. The iterations considered are those between the sucessive
    numbers from arg int.

    Examples

      Extract particles belonging to class 3 at iter 15: 

      extract_class(basename='in_basename', iters=15, class=3, out='out_file')

      Extracts particles that changed class at most once during iters 13-14
      and 14-15.

      extract_class(basename='in_basename', iters=[13,14,15], max_change=1, 
                    out='out_file')

    Arguments:
      - basename: basename of the data file (everything until '_it')
      - iters: list (or tuple) of iterations, or a single iteration
      - iter_format: format for the iteration number used for the name of the 
      data file
      - suffix: suffix of the data file (whatever comes after iteration number)
      - class_: (list or int) class number(es)
      - max_change: max number of class changes
      - out: name of the output (new data) file or an open stream

    Returns: number of particles found
    """
    
    # constants
    image_label = 'rlnImageName'
    #class_label = 'rlnClassNumber'

    # deal with single int iters
    if not isinstance(iters, (list, tuple)):
        iters = [iters]

    # in file name
    in_file = basename + (iter_format % iters[-1]) + suffix

    # class changes for each particles
    if max_change is not None:
        change = get_n_class_changes(basename=basename, iters=iters)
        change_bool = (change <= max_change)

    # classes
    if class_ is not None:
        classes = get_class(
            basename=basename, iters=iters[-1], iter_format=iter_format,
            suffix=suffix, tablename=tablename)
        if np.iterable(class_):
            for one_class in class_:
                try:
                    class_bool = class_bool | (classes == one_class) 
                except NameError:
                    class_bool = (classes == one_class) 
        else:
            class_bool = (classes == class_)

    #  particle names
    names_dict = get_array_data(starfile=in_file, tablename=tablename, 
                                labels=[image_label], types=[str])
    names = np.asarray(names_dict[image_label])

    # find particles that satisfy class and max change consitions
    total_bool = np.ones(len(names), dtype=bool)
    if class_ is not None:
        total_bool = class_bool & total_bool 
    if max_change is not None:
        total_bool = change_bool & total_bool
    good_names = names[total_bool]

    if out is None:
        return len(good_names)

    # copy header
    if isinstance(out, str):
        out_fd = open(out, 'w')
    else:
        out_fd = out
    out_fd.write(os.linesep)
    head = array_data_head(starfile=in_file, tablename=tablename)
    for line in head:
        out_fd.write(line)

    # find and save data for good particles
    line_gen = array_data_generator(
        starfile=in_file, tablename=tablename, split=False)
    for (line, label_indices), good in zip(line_gen, total_bool):

        if good:
            out_fd.write(line)

    out_fd.close()
    return len(good_names)

def extract_random(starfile, out, number, tablename='data_images'):
    """
    Extracts randomly (arg) number of particles form the particle file (arg
    starfile) and writes them in a separate file (arg out).

    The output particles are written in the same order as they appear in the 
    particle file.

    Arguments:
      - starfile: particle file name
      - out: output (random particles) 
      - number: number of particles to extract
      - tablename: name of the table containing particles 
    """

    # get number of particles
    n_particles = array_data_length(starfile=starfile, tablename=tablename)

    # pick and sort random positions
    particle_indices = np.asarray(random.sample(range(n_particles), number))
    particle_indices.sort()

    # copy header
    out_fd = open(out, 'w')
    out_fd.write(os.linesep)
    head = array_data_head(starfile=starfile, tablename=tablename)
    for line in head:
        out_fd.write(line)

    # read random lines
    line_ind = 0
    particle_ind = 0 
    random_lines = []
    line_gen = array_data_generator(starfile=starfile, tablename=tablename, 
                                    split=False, labels=False)
    for line in line_gen:
        if line_ind == particle_indices[particle_ind]:
            random_lines.append(line)
            particle_ind += 1
            if particle_ind >= number:
                break
        line_ind += 1

    # write random particle liness
    out_fd.writelines(random_lines)

def print_class_summary(
        basename, iters, class_=None, mode='change', fraction=True, 
        suffix='_data.star', tablename='data_', label='rlnClassNumber', 
        cont=True, iter_format='_it%03d', warn=False):
    """
    Prints class membership for all specifed iterations and the number / 
    fraction of particles that changed class between neighboring iterations.

    Based on get_class() and get_n_particle_class_change()

    If arg cont is True, takes also continuing iterations into account.

    Arguments:
      - basename: path of the star data file (everything until '_it')
      - iter: list of iterations, if None, all iterations are used
      - class_: list of classes, or None for all classes
      - mode: 'change', 'to_from'
      - suffix: data starfile name part after iteration (includes extension)
      - table_name: name of the table where particles are stored
      - label: label for class number 
      - cont: flag indication if continued executions are used
      - iter_format: format for iteration number in starfiles
      - warn: prints warning when starfile not found. Note that if iters is
      None and warn is True, there is always a warning after the last iteration
    """

    # prepare class_
    class_ = prepare_class_arg(class_=class_, mode='tuple')

    # get particle number for each class
    class_content = get_data(
        basename=basename, iters=iters, suffix=suffix, tablename=tablename, 
        label=label, iter_format=iter_format)

    # print particle number 
    print("Number of particles in each class")
    sorted_iters = np.sort(class_content.keys())
    for iter_ in sorted_iters:
        cc = class_content[iter_]

        # all classes separately
        if class_ is None:
            class_n = [
                (cc==class_ind).sum() for class_ind 
                in range(1, class_content[iter_].max()+1)]

        # classes (class groups) specified
        else:
            class_n = []
            for class_ind in class_:
                abs_one = np.sum([(cc==cl_ind).sum() for cl_ind in class_ind])
                class_n.append(abs_one)

        # print particle number per class (group)
        print(str(iter_) + ": " + str(class_n))
            
    # find particle change
    class_change = get_n_particle_class_change(
        basename=basename, iters=iters, class_=class_, mode=mode, 
        fraction=fraction, out='dict', suffix=suffix, 
        tablename=tablename, label=label, iter_format=iter_format)

    # print particle change
    if fraction: fr_str = 'Fraction'
    else: fr_str = 'Number'
    print(
        "\n" + fr_str + " of particles that changed class between "
        + "consecutive iterations")
    if mode == 'to_from':
        print(
            '(from classes are on the vertical and '
            + 'to classes on horisontal axis)')
    if iters is None:
        prev_iter = 0
    else:
        prev_iter = iters[0]
    sorted_iters = np.sort(class_content.keys())
    for iter_ in sorted_iters[1:]:
        cc = class_change[iter_]
        if class_ is None:
            print('%2d - %2d: %6.3f' % (prev_iter, iter_, cc))
        elif mode == 'change':
            print(
                ('%2d - %2d: ' % (prev_iter, iter_)) 
                + (' '.join(['%6.3f']*len(cc)) % tuple(cc)))
        elif mode == 'to_from':
            print('%2d - %2d: ' % (prev_iter, iter_))
            print(np.array_str(
                cc, precision=2, suppress_small=True, max_line_width=100))
        prev_iter = iter_

def get_class(
        basename, iters, suffix='_data.star', tablename='data_images', 
        cont=True, iter_format='_it%03d'):
    """
    Reads particle class membership from a particle star file for one or 
    more iterations, for each iteration separately. 

    Convenience function based on get_data().

    The particle star file name is determined from args basename, iters,
    iter_format and suffix. If arg cont is True, find the proper file name 
    even if the file was generated by the continuation of a previous run 
    (the file name contains the contionuation string).

    It uses get_array_data() to read the data, but provides the following
    advantages:
      - requires iteration(s) as an argument instead of the whole file name
      - finds continuation files automatically
      - reads data ar several iterations

    Arguments:
      - basename: part of the data file path until '_it' or '_ct', if 
      cont is True doesn not need the continuation string
      - iters: list of iterations, if None, all iterations are used
      - suffix: data starfile name part after iteration (includes extension)
      - table_name: name of the table where particles are stored
      - cont: flag indication if continued executions are used
      - iter_format: format for iteration number in starfiles

    Returns dictionary with iterations as keys and class membership
    as values, where the class mebership is a list of class ids in the 
    order of particles in the particle file.
    """

    return get_data(
        basename=basename, iters=iters, label='rlnClassNumber', suffix=suffix,
        tablename=tablename, type_=int, cont=cont, iter_format=iter_format)

def get_data(
        basename, iters, label, suffix='_data.star', tablename='data_', 
        type_=int, cont=True, iter_format='_it%03d'):
    """
    Reads particle data from one column in a particle star file for one or 
    more iterations, for each iteration separately. 

    The particle star file name is determined from args basename, iters,
    iter_format and suffix. If arg cont is True, find the proper file name 
    even if the file was generated by the continuation of a previous run 
    (the file name contains the contionuation string).

    It uses get_array_data() to read the data, but provides the following
    advantages:
      - requires iteration(s) as an argument instead of the whole file name
      - finds continuation files automatically
      - reads data ar several iterations

    Arguments:
      - basename: part of the data file path until '_it' or '_ct', if 
      cont is True doesn not need the continuation string
      - iters: list of iterations, if None, all iterations are used
      - suffix: data starfile name part after iteration (includes extension)
      - table_name: name of the table where particles are stored
      - label: column label
      - type_: function that converts a string read to the proper type, 
        such as int or float
      - cont: flag indication if continued executions are used
      - iter_format: format for iteration number in starfiles

    Returns dictionary with iterations as keys and particle data arrays
    as values, where the class mebership is a list of class ids in the 
    order of particles in the particle file.
    """
    
    # initialize
    class_dict = {}
    single_iter = False
    if iters is None:
        local_iter = itertools.count(0)
    elif not np.iterable(iters):
        single_iter = True
        local_iter = [iters]
    else:
        local_iter = iters

    for it in local_iter:

        # find files that match the criteria
        path = find_file(
            basename=basename, iter_=it, half=None, suffix=suffix, cont=cont,
            iter_format=iter_format)

        if path is None:

            # starfile not found
            if iters is not None:
                 warnings.warn(
                    "Starfile with basename " + basename + " for iteration "
                    + str(it) + " could not be found.")
            break
               
        elif len(path) == 1:
            starfile = path[0]
        elif len(path) == 2:
            starfile = path[0]
        elif len(path) >= 2:
            warnings.warn(
                "Skipping basename " + basename + " iteration " + str(it) + 
                "because >2 files matched.")
            continue

        # find data in file
        #if os.path.exists(starfile):
        label_dict = get_array_data(
                starfile=starfile, tablename=tablename, labels=[label], 
                types=[type_])
        class_dict[it] = label_dict[label]
        #else:
        #    break
        #    warnings.warn(
        #        "Skipping file " + starfile + " because it doesn't exist.")

    if single_iter:
        return class_dict[local_iter[0]]
    else:
        return class_dict
    
def classify_array(data, mode, pattern, start=1):
    """
    Classifies 1D array data according to a criterion specified by args 
    mode and pattern. 

    The classes are labeled by consecutive intgers (class ids) starting 
    from arg start. In order to pass this array directly to scipy, the 
    class ids should start from 0.

    If arg mode is 'class', the array elements (arg data) are expected to
    be integers, such as the particle class ids. The array elements are 
    (re)classified taking into account arg pattern. Each element of arg 
    pattern specifies a new class. In the simplest case, arg pattern is 
    a list of integers (such as the existing class ids produced by Relion). 
    If an element of arg pattern is a tuple, all elements of the tuple are 
    taken together. For example:

      classify_array(data=[1,2,3,2,1], mode='class', pattern=[3,2,1])

    returns: [3, 2, 1, 2, 3]

      classify_array(data=[1,2,3,4,5], mode='class', pattern=[1, (2,3), (4,5)])

    returns: [1, 2, 2, 3, 3]

    If arg mode is 'find' the arg data should be an array or strings. The
    classification is again based on elements of arg patter, but here 
    str.find() is used to determine the class. For example:

      classify_array(
        data=['tomo-3/part_01.mrc', 'tomo-1/part_02.mrc', 'tomo-3/part_03.mrc'],
        mode='find', pattern=['tomo-1', 'tomo-3'])

    returns: [2, 1, 2].

    Arguments:
      - data: array of data to be classified
      - mode: classification mode, 'class' or 'find'
      - pattern: pattern that determins the (new) classes
      - start: starting class index

    Returns: array corresponding to arg data that contains (new) class ids.
    """

    # calculate class ids for all particles
    class_ids = np.zeros(len(data), dtype='int') - 1
    for ind, pat in enumerate(pattern):

        if mode == 'class':
            if isinstance(pat, tuple):
                class_pat_flags = np.zeros(len(data), dtype='bool')
                for one_ind in pat:
                    class_pat_flags = (
                        class_pat_flags | (data == one_ind))
            else:
                class_pat_flags = (data == pat)

        elif mode == 'find':
            class_pat_flags = np.asarray(
                [cl_val.find(pat) >= 0 for cl_val in data])

        else:
            raise ValueError("Mode " + mode + " was not understood.")

        class_ids[class_pat_flags] = ind + start

    return class_ids

def get_n_particle_class_change(
        basename, iters, class_=None, mode='change', fraction=True, out='list', 
        suffix='_data.star', tablename='data_images', label='rlnClassNumber', 
        cont=True, iter_format='_it%03d', warn=True):
    """
    Calculates number of particles that changed class between all pairs of 
    consecutive iterations, where iterations are specified by arg iters.

    There are three different usage types, depending on args class_ and mode:

      1) class_=None, mode='change': Total number (or fraction) of particles
      that changed class is calculated.

      2) class_ is a list, mode='change': Number (or fraction) of particles
      that changed class is given for each class (specified by arg class_)
      separately. A particle changed class if in the previous iteration 
      it was in a class different from the one where it is in the current
      iteration.

      3) class_ is a list, mode='to_from': Number of particles that moved
      from one class to another is calculated for each class (specified by 
      arg class_) and for all pairs of classes separately. The result
      is given as a 2D ndarray in the form result[from_class, to_class],
      where from_ and to_class are indices of classes as specified in 
      arg class_ (list).

    For the above classes 2 and 3, arg class_ is a list. It can contain 
    individual ints (class ids) or tuples of class ids in which cases 
    the classes specified in a tuple are taken together. For example:

      class_ = [2, (1, 3, 4), 5]

    means that classes 1, 3, and 4 are taken together (merged) and 1 and 
    5 are separate.

    If arg fraction is True, the fraction of total particles (case 1 above)
    or of the number of the current (to) particles (cases 2 and 3) is 
    calculated.  

    If arg out is 'list', the numbers of changed particles are returned in 
    a list (of length len(iters)-1), so that the element i corresponds to 
    the change between iterations iter[i+1] and iter[i].

    Alternatively, if arg out is 'dict', the numbers of changed particles 
    are returned as values of a dictionary where keys are the final 
    iterations (those in iters[1:]). 

    If arg cont is True, takes also continuing iterations into account.

    Arguments:
      - basename: basename of the data file (everything until '_it')
      - iters: list of iterations, if None, all iterations are used
      - class_: 
      - mode: 
      - fraction: If True returns number of particles that changed class 
      divided by the total number of particles. If False the (absolute) 
      number of particles that changes class is returned
      - out: output format, 'list' ot 'dict'
      - suffix: data starfile name part after iteration (includes extension)
      - tablename: name of the table where particles are stored
      - label: label for class number
      - cont: flag indication if continued executions are used
      - iter_format: format for iteration number in starfiles
      - warn: prints warning when starfile not found. Note that if iters is
      None and warn is True, there is always a warning after the last iteration

    Return: dict (keys are iterations) or list (elements correspond to those
    in arg iters[1:]), where iterations are the final from each iteration pair
    (so the first iteration does not have a corresponding entry). The
    keys / list elements can be:
      - single number (case 1, above)
      - list (case 2, above)
      - 2D ndarray (case 3, above)
    """

    # prepare class_
    class_ = prepare_class_arg(class_=class_, mode='tuple')

    # initialize
    if iters is None:
        local_iter = itertools.count(0)
    else:
        local_iter = iters

    first_iter = True
    result_dict = {}
    for it in local_iter:

        # find files that match the criteria
        path = find_file(
            basename=basename, iter_=it, half=None, suffix=suffix, cont=cont,
            iter_format=iter_format) 

        if path is None:

            # starfile not found
            if iters is not None:
                 warnings.warn(
                    "Starfile with basename " + basename + " for iteration "
                    + str(it) + " could not be found.")
            break
               
        elif len(path) == 1:
            starfile = path[0]
        elif len(path) == 2:
            starfile = path[0]
        elif len(path) >= 2:
            warnings.warn(
                "Skipping basename " + basename + " iteration " + str(it) + 
                "because >2 files matched.")
            continue

        # find data in file
        classes_dict = get_array_data(
            starfile=starfile, tablename=tablename, labels=[label], 
            types=[int])
 
        if first_iter:

            # first iter, just read
            classes_prev = classes_dict[label]
            first_iter = False

        else:

            # read and compare with the previous iter
            classes = classes_dict[label]

            # find unchanged for each class or class group
            if class_ is None:

                # only total change
                result_abs = (classes != classes_prev).sum()
                result_frac = result_abs / float(len(classes))

            else:

                if mode == 'change':

                    # change for each specified class

                    result_abs = []
                    result_frac = []
                    for cl in class_:

                        # find particles that changed class
                        in_curr = np.zeros(shape=classes.shape, dtype=bool)
                        in_prev = np.zeros(shape=classes.shape, dtype=bool)
                        for cl_one in cl:
                            in_curr = in_curr | (classes == cl_one)
                            in_prev = in_prev | (classes_prev == cl_one)
                        changed = in_curr & ~in_prev
                        result_abs.append(changed.sum())
                        result_frac.append(
                            np.asarray(changed.sum()) / float(in_curr.sum()))

                elif mode == 'to_from':

                    # change for all pairs of specified classes

                    result_abs = np.zeros(shape=(len(class_), len(class_))) - 1
                    result_frac = np.zeros(shape=(len(class_), len(class_))) - 1
                    for to_ind, cl_to in enumerate(class_):
                            
                        # find particles in to class
                        in_curr = np.zeros(shape=classes.shape, dtype=bool) 
                        for cl_one in cl_to:
                            in_curr = in_curr | (classes == cl_one)

                        for from_ind, cl_from in enumerate(class_):

                            # find particles in from class
                            in_prev = np.zeros(shape=classes.shape, dtype=bool) 
                            for cl_one in cl_from:
                                in_prev = in_prev | (classes_prev == cl_one)

                            # find particles that moved from to
                            trans = in_curr & in_prev    
                            result_abs[from_ind, to_ind] = trans.sum()
                            result_frac[from_ind, to_ind] = (
                                result_abs[from_ind, to_ind] 
                                / float(in_curr.sum()))

            # save results
            if fraction:
                result_dict[it] = result_frac
            else:
                result_dict[it] = result_abs

            # for the next iteration
            classes_prev = classes

    # prepare return 
    if out == 'dict':
        return result_dict
    elif out == 'list':
        sorted_iters = np.sort(result_dict.keys())
        result_list = [result_dict[it] for it in sorted_iters]
        return result_list
    else:
        raise ValueError(
            "Can not understand argument out: " + out + "allowed values "
            + "are 'list' and 'dict'.") 

def get_n_class_changes(basename, iters):
    """
    Returns the number of times a particle has changed its class membership
    between the specified iterations.

    Arguments:
      - basename: basename of the data file (everything until '_it')
      - iters: list of iterations

    Returns: (ndarray) number of changes for each particle in the order 
    the particles are listed in the data file      
    """

    # get classes for all iterations
    class_dict = get_class(basename=basename, iters=iters)

    # initialize
    previous_classes = np.asarray(class_dict[iters[0]])
    change = np.zeros_like(previous_classes)

    for it in iters[1:]:
        
        current_classes = np.asarray(class_dict[it])
        current_change = (current_classes != previous_classes) * 1.
        change += current_change
        previous_classes = current_classes

    return change

def class_group_interact(
        method, basename, iters, group_pattern, classes, suffix='_data.star', 
        tablename='data_', class_label='rlnClassNumber', 
        group_label='rlnMicrographName', group_mode='find', group_type=str, 
        iter_format='_it%03d'):
    """
    A special case of two_way_class() where for the first classification 
    the classes from the particle star file is used. The secon 
    classification may but does not have to be based on an experimental
    group (specified in the micrograph name).

    Not sure if needed
    """

    result = two_way_class(
        method=method, basename=basename, iters=iters, suffix=suffix, 
        tablename=tablename, label=(class_label, group_label),
        mode=('class', group_mode), pattern=(classes, group_pattern),
        iter_format=iter_format, type_=(int, group_type))

    return result

def two_way_class(
        method, basename, iters, mode, pattern, label, suffix='_data.star', 
        tablename='data_', type_=str, iter_format='_it%03d'):
    """
    Gets data from two star files, classifies the data, calculates 
    contingency table between two classifications and optionally
    (in)dependence between them using G-test. 

    Regarding the method, see for example Sokal and Rolph, Biometry, pg 724).

    Arg method can be 'contingency' for contingency table only or
    'contingency_g' for contingency table and G-test.

    The G-test is calsulated using get_contingency(), which is based on:
      sp.stats.chi2_contingency(
        contingency, lambda_="log-likelihood", correction=True)

    The two clasifications are read from classification partcle (star) files, 
    The starfiles are specified by args basename, iters and suffix,
    the data to be read by args tablename, label and type. The 
    classification is performed using classify_array() where args mode and 
    pattern are passed directly (see classify_array() for more info about
    classification).

    Particles in the two particle star files have to be the same and to
    have the same order.

    All arguments except method and iter_format can be tuples of length two, 
    in which case the first and the second element specify the corresponding
    values for the first and the second classification, respectivly. If
    an argument is not a tuple, its value is used for for both 
    classifications.

    Arguments:
      - method: 'contingency' or 'contingency_g'
      - basename: star file basename (the whole path until 'iter') 
      - iters: one or more iterations for which data is read
      - suffix: star file suffix
      - tablename: table name within the star file
      - label: record label in the star file
      - mode: classification mode, 'class' or 'find'
      - pattern: classification pattern
      - type_: data type, str or int
      - iter_format: iteration number format

    Returns: 
      - collections.namedtuple object if iters is a single number or a tuple
      containing single numbers
      - dictionary where keys are iteration numbers (or the data specified 
      first) and values are the corresponding collections.namedtuple objects

    Each collections.namedtuple object returned contains the following 
    attributes:
      - contingency: contingency table
      - fraction: fractions corresponding to the contingency table
      - total: total number of elements for all classes for both classifications
      - total_fract: fractions corresponding to total
      - g, p, dof: G-value, p-value and degrees of freedom, if G-test
    """

    min_class_id = 1

    # parse arguments:
    if isinstance(basename, tuple):
        basename_class, basename_group = basename
    else:
        basename_class = basename
        basename_group = basename
    if isinstance(iters, tuple):
        iters_class, iters_group = iters
    else:
        iters_class = iters
        iters_group = iters
    if isinstance(iters_class, (int, long)):
        iters_class = [iters_class]
    if isinstance(iters_group, (int, long)):
        iters_group = [iters_group]
    if isinstance(suffix, tuple):
        suffix_class, suffix_group = suffix
    else:
        suffix_class = suffix
        suffix_group = suffix
    if isinstance(tablename, tuple):
        tablename_class, tablename_group = tablename
    else:
        tablename_class = tablename
        tablename_group = tablename
    if isinstance(label, tuple):
        label_class, label_group = label
    else:
        label_class = label
        label_group = label
    if isinstance(type_, tuple):
        type_class, type_group = type_
    else:
        type_class = type_
        type_group = type_
    if isinstance(mode, tuple):
        mode_class, mode_group = mode
    else:
        mode_class = mode
        mode_group = mode
    if isinstance(pattern, tuple):
        pattern_class, pattern_group = pattern
    else:
        pattern_class = pattern
        pattern_group = pattern

    # get class values and ids
    class_values = get_data(
        basename=basename_class, iters=iters_class, suffix=suffix_class,
        tablename=tablename_class, label=label_class, type_=type_class)

    # get group values and ids
    group_values = get_data(
        basename=basename_group, iters=iters_group, tablename=tablename_group, 
        label=label_group, suffix=suffix_group, type_=type_group)

    # compare classifications for all iters
    result = {}
    for it_class, it_group in zip(iters_class, iters_group):

        # data for this iter
        class_val = class_values[it_class]
        group_val = group_values[it_group]

        # classify data
        class_ids = classify_array(
            data=class_val, mode=mode_class, pattern=pattern_class, 
            start=min_class_id)
        group_ids = classify_array(
            data=group_val, mode=mode_group, pattern=pattern_group, 
            start=min_class_id)

        # analyze
        if (method == 'contingency') or (method == 'contingency_g'):

            # contingency
            if (method == 'contingency'): test = None
            else: test = 'g'
            result[it_class] = get_contingency(
                class_ids_1=class_ids, class_ids_2=group_ids,
                id_limits_1=(min_class_id, len(pattern_class)-1+min_class_id), 
                id_limits_2=(min_class_id, len(pattern_group)-1+min_class_id), 
                test=test)

        elif (method == 'vi') or (method == 'rand') or (method == 'b-flat'):
            
            # clustering comparison
            simil = cluster_similarity(
                class_ids_1=class_ids, class_ids_2=group_ids, method=method)
            result[it_class] = simil

        else:
            raise ValueError('Test ' + str(method) + ' was not understood.')

    # return
    if isinstance(iters, (tuple, list, np.ndarray)):
        return result
    else:
        return result[iters]

def get_contingency(
        class_ids_1, class_ids_2, id_limits_1=None, id_limits_2=None, 
        test=None):
    """
    Calculates contingency table between two classifications (args class_ids_1
    and class_ids_2) and optionally (in)dependence between them using G-test. 

    Class ids are expected to be all integers between the limits
    specified by args id_limits_1 and id_limits_2 (limits included).

    Similar methods:
      - Already implemented in pyto.segmentation.Cluster but wo G-test 
      - G test implemented in pyto.clustering.Contingency.g_test()

    ToDo: probably just wrap pyto.segmentation.Cluster.getContingency()

    Arguments:
      - class_ids_1, class_ids_2: classification arrays where elements
      correspond to data points and the value of an element denotes its
      class (same as data representation in pyto.segmentation.CLuster) 
      - id_limits_1, id_limits_2: (list or tuple of length 2), the lowest 
      and the highest class ids, if None the min and max values, respectivly,
      are used
      - test: 'g' for G-test or None for no testing

    Returns: 
      - collections.namedtuple object if iters is a single number or a tuple
      containing single numbers
      - dictionary where keys are iteration numbers (or the data specified 
      first) and values are the corresponding collections.namedtuple objects

    Each collections.namedtuple object returned contains the following 
    attributes:
      - contingency: contingency table
      - fraction: fractions corresponding to the contingency table
      - total: total number of elements for all classes for both classifications
      - total_fract: fractions corresponding to total
      - g, p, dof: G-value, p-value and degrees of freedom, if test='g' 
    """

    class_ids_1 = np.asarray(class_ids_1)
    class_ids_2 = np.asarray(class_ids_2)

    # set class id limits
    if id_limits_1 is not None:
        min_id_1 = id_limits_1[0]
        max_id_1 = id_limits_1[1]
    else:
        min_id_1 = np.min(class_ids_1)
        max_id_1 = np.max(class_ids_1)
    if id_limits_2 is not None:
        min_id_2 = id_limits_2[0]
        max_id_2 = id_limits_2[1]
    else:
        min_id_2 = np.min(class_ids_2)
        max_id_2 = np.max(class_ids_2)

    # initial arrays
    contig = np.zeros(
        (max_id_1-min_id_1+1, max_id_2-min_id_2+1), dtype='int') - 1
    fraction = np.zeros((max_id_1-min_id_1+1, max_id_2-min_id_2+1)) - 1

    # calculate contingency and fractions
    for cl_ind in range(min_id_1, max_id_1+1):
        cl_belong = (class_ids_1 == cl_ind)
        cl_freq_sum = float(np.count_nonzero(cl_belong))

        # iterate over groups
        for gr_ind in range(min_id_2, max_id_2+1):
            gr_belong = (class_ids_2 == gr_ind)

            # calculate contingency
            contig[cl_ind-min_id_1, gr_ind-min_id_2] = np.count_nonzero(
                cl_belong & gr_belong)

        # calculate frequencies
        fraction[cl_ind-min_id_1, :] = (
            contig[cl_ind-min_id_1, :] / cl_freq_sum)

        # calculate marginals
        total = (contig.sum(axis=0), contig.sum(axis=1))
        total_fract = (
            contig.sum(axis=0) / float(contig.sum()),
            contig.sum(axis=1) / float(contig.sum()))

    # test
    if test is None:

        # no test, return contingencly and fraction
        namtu = collections.namedtuple(
            'Struct', 'contingency fraction total total_fract')
        res = namtu._make((contig, fraction, total, total_fract))
      
    elif test == "g":

        # g statistics
        g_val, p_val, dof, expected = sp.stats.chi2_contingency(
            contig, lambda_="log-likelihood", correction=True)
        namtu = collections.namedtuple(
            'Struct', 
            'g p dof expected contingency fraction total total_fract')
        res = namtu._make(
                (g_val, p_val, dof, expected, contig, fraction, 
                 total, total_fract))

    else:
        raise ValueError("Test " + test + " was not understood.")

    return res

def cluster_similarity(class_ids_1, class_ids_2, method=None):
    """
    Calculates similarity between two given clusterings. The similarity
    can be calculated using the Rand, Fowlkes and Mallows, and Variation
    of information methods.

    A wrapper for pyto.segmentation.Cluster.findSimilarity(). See that
    method for more info about the cluster similarity methods

    Arguments:
      - class_ids_1, class_ids_2: classification arrays where elements
      correspond to data points and the value of an element denotes its
      class
      - method: similarity method, the same as arg method in 
      Cluster.findSimilarity(), that is 'rand', 'b-flat' or 'vi'.

    Returns:
      - simil_index: similarity index for the method requested
    """

    class_ids_1 = np.asarray(class_ids_1)
    class_ids_2 = np.asarray(class_ids_2)


    # clustering comparison
    clust_1 = Cluster(clusters=class_ids_1, form='scipy')
    clust_2 = Cluster(clusters=class_ids_2, form='scipy')
    result = clust_1.findSimilarity(reference=clust_2, method=method)

    return result

##############################################################
#
#  Xmipp and Relion
#

def convert_angle_distribution(
    instar, outstar, tablename='data_images', half=None, class_=None, 
    weight=True):
    """
    Reads particle angles from Relion data file (instar), orders them
    and writes them to a xmipp style star file which 
    xmipp_angular_distribution_show understands.

    Argument half alows chosing one or the both data set halves generated by
    refine3d.

    Argument class_ allows chosing one or more classes generated by class3d.

    Arguments:
      - instar: relion data file (input)
      - outstar: xmipp angular distribution file (output)
      - tablename: table name
      - half: 1 or 2 for one or the other half sets, None to use all (default)
      - class_: class number or a list of class numbers to pick one or more 
      classes, None to use all (default)
      - weight: if True the weight of a direction is proportional to the 
      number of particles, if False all directions that contain particles
      are labeled
    """

    # constants
    labels = {'rlnAngleRot' : 'angleRot', 
              'rlnAngleTilt' : 'angleTilt'}
    if half is not None:
        labels['rlnRandomSubset'] = 'randomSubset'
    if class_ is not None:
        labels['rlnClassNumber'] = 'classNumber'
    labels_out = ['angleRot', 'angleTilt', 'weight', 'anglePsi']
    format_ = {'angleRot' : '%10.5f', 'angleTilt' : '%10.5f',
              'anglePsi' : '%10.5f', 'randomSubset' : '%4i',
              'weight' : '%8.3f'}
    comment = "# XMIPP_STAR_1 *"
    
    # read data table
    data = get_array_data(starfile=instar, tablename=tablename, 
                            labels=labels)
    # pick correct half
    if half is not None:
        subset_cond = (data['rlnRandomSubset'] == half)
        for lbl in labels:
            data[lbl] = data[lbl][subset_cond]
        labels.pop('rlnRandomSubset')

    # pick class
    if class_ is not None:
        if isinstance(class_, (list, tuple)):
            class_cond = np.zeros(len(data['rlnClassNumber']), dtype='bool')
            for cls in class_:
               class_cond = class_cond | (data['rlnClassNumber'] == cls) 
        else:
            class_cond = (data['rlnClassNumber'] == class_)
        for lbl in labels:
            data[lbl] = data[lbl][class_cond]
        labels.pop('rlnClassNumber')

    # order and save rot and tilt in out_data dict with new (out) labels
    lexsort = np.lexsort((data['rlnAngleTilt'], data['rlnAngleRot']))
    sorted_data = {}
    for lbl in labels.keys(): 
        sorted_data[labels[lbl]] = data[lbl][lexsort]

    # add weight
    rot_prev = None
    tilt_prev = None
    rot_list = []
    tilt_list = []
    weight_list = []
    current_weight = 1
    for rot, tilt in zip(sorted_data['angleRot'], sorted_data['angleTilt']):
        if (rot_prev is not None): 

            if (rot_prev == rot) and (tilt_prev == tilt):
                
                # same, just increase weight
                if weight:
                    current_weight += 1

            else:

                # different, write previous
                rot_list.append(rot_prev)
                tilt_list.append(tilt_prev)
                weight_list.append(current_weight)
                current_weight = 1        

        # prepare next iter
        rot_prev = rot
        tilt_prev = tilt

    # finish the last iteration
    rot_list.append(rot_prev)
    tilt_list.append(tilt_prev)
    weight_list.append(current_weight)

    # convert to dict
    out_data = {}
    out_data['angleRot'] = np.asarray(rot_list)
    out_data['angleTilt'] = np.asarray(tilt_list)
    out_data['weight'] = np.asarray(weight_list, dtype='float')
     
    # add psi
    out_data['anglePsi'] = np.zeros(len(out_data['angleRot']))

    # write new file
    if outstar is not None:
        write_table(
            starfile=outstar, tablename=tablename, labels=labels_out, 
            data=out_data, format_=format_, comment=comment)

    return out_data

def convert_boxes(instar, outstar, verbose=True):
    """
    Converts particle box coordinates from relion (usually stored in
    Particles/Micrographs/*_particles.star files) to xmipp (usually in 
    ParticlePicking/Supervised/run_nnn/extra/*.pos). 
    
    The corresponding relion and xmipp files have the same names except that 
    relion file names end with  '_particles.star' and xmipp with '.pos'. 

    Relion labels 'rlnCoordinateX' and 'rlnCoordinateY' are converted to
    xmipp 'xcoor' and 'ycoor'. Also, xmipp files are given the header that
    they usuallu have, just to mimic 'normal' xmipp particle picking.

    Arguments:
      - instar: directory containing relion particle files for all 
      micrographs (should end with '/', todo: extend to use regexp)
      - outstar: directory where xmipp files are written (should end with '/')
    """

    # constants
    in_tablename = 'data_images'
    in_suffix = '_particles.star'
    in_labels = ['rlnCoordinateX', 'rlnCoordinateY']
    out_suffix = '.pos'
    out_head = [
        '# XMIPP_STAR_1 * ',
        '# ',
        'data_header',
        'loop_',
        '_pickingMicrographState',
        '_autopickPercent',
        '_cost',
        'Manual         50     0.000000 ',
        'data_particles',
        'loop_']
    out_labels = ['xcoor', 'ycoor']

    # figure out in and out directories
    if os.path.isdir(instar):
        in_dir = instar
        in_base_pattern = ''
    else:
        in_dir, in_base_pattern = os.path.split(instar) 
    if os.path.isdir(outstar):
        out_dir = outstar
        out_base_pattern = ''
    else:
        out_dir, out_base_pattern = os.path.split(outstar) 

    for in_base in os.listdir(in_dir):

        # read data
        in_file = os.path.join(in_dir, in_base)
        if not in_file.endswith(in_suffix):
            continue
        in_data = get_array_data(starfile=in_file, tablename=in_tablename, 
                                 labels=in_labels)

        # prepare data
        out_data = {}
        for in_lbl, out_lbl in zip(in_labels, out_labels):
            out_data[out_lbl] = in_data[in_lbl]

        # write data
        out_base = in_base.rstrip(in_suffix) + out_suffix
        out_file = os.path.join(out_dir, out_base)
        if verbose:
            print("Converting ", in_file, ' to ', out_file)
        write_table(starfile=out_file, labels=out_labels, data=out_data, 
                    format_='%10i %10i', header=out_head, comment=None,
                    labelindent=' ')

def convert_particle_selection(relionstar, xmippinstar, xmippoutstar,
                               disable=True, verbose=False):
    """
    Modifies data forom xmipp particle star file (arg xmippinstar; usually 
    named images.xmd) so that only the particles that are listed in the relion 
    particle star file (arg relioninstar) are retained. The modifieded data is
    written to a new file (arg xmippoutstar).

    Useful to select particles in xmipp that were previously selected in
    relion.

    It is necessary that particles have the same names in relion and xmipp.
    That is, particle numbers and the particle file names (without directory
    and extension; usually derived from micrograph names) need to correspond
    between relion and xmipp.

    Consequently, all particles should be in the same directory (relion 
    particle in one and xmipp in another). If that is not the case, it is 
    necessary that particle file names (without directory part and extension) 
    are unique.

    It is possible that some particles exist in the relio file and not in 
    xmipp. These particles are listed if arg verbose is True.

    If arg disable is True, xmipp particle lines corresponding to the disabled
    particles (those not found in the relion file) are written to the output
    xmipp file, but the enable value is converted to a negative value. 
    Otherwise, if disable is False, these lines are not written at all.

    Arguments:
      - xmippinstar: original (input) xmipp particle star file 
      - relionstar: relion patrticle star file
      - xmippoutstar: modified (output) xmipp particle star file
      - disable: Flag indicating if how are the non-selected particles 
      specified
      - verbose: if True, provides info about each particle from xmipp 
    """

    # constants
    relion_table = 'data_images'
    relion_image_label = 'rlnImageName'
    xmipp_table = 'data_noname'
    xmipp_image_label = 'image'
    enable_label = 'enabled'

    # read and sort relion particle names
    relion_data = get_array_data(
        starfile=relionstar, tablename=relion_table, 
        labels=[relion_image_label], types=str)
    relion_names = relion_data[relion_image_label]
    relion_sorted = sort_particles(
        particles=relion_names, indices=True, compact=True, type='relion')
    (relion_sorted_particles, relion_sorted_indices, 
     relion_compacts) = relion_sorted
    print 'relion_compacts: ' + str(relion_compacts)

    # get head
    out_lines = array_data_head(starfile=xmippinstar, 
                                tablename=xmipp_table, top=True)

    # ideally xmipp data should be read and sorted before comparing with 
    # the relion data, but then the data should be sorted back for writing

    # read and write data 
    for line, indices in array_data_generator(
            starfile=xmippinstar, tablename=xmipp_table, split=False):

        # get particle name
        xmipp_image_index = indices[xmipp_image_label]
        xmipp_file = line.split()[xmipp_image_index]
        xmipp_number, xmipp_path = xmipp_file.split('@', 1) 
        xmipp_dir, xmipp_name = os.path.split(xmipp_path)
        xmipp_base, xmipp_ext = xmipp_name.rsplit('.', 1)
        xmipp_compact = xmipp_number + xmipp_base
        print 'xmipp_compact: ' + str(xmipp_compact)
        if xmipp_compact in relion_compacts:

            # keep the line
            out_lines.append(line)
            if verbose:
                print "Particle " + xmipp_compact + " kept."

            # remove current from relion (to speed up further searches)
            index = (relion_compacts == xmipp_compact).nonzero()
            relion_compacts = np.delete(relion_compacts, index)

        elif disable:

            # find position of enable field in the line
            split_line = line.split()
            start_pos = 0
            previous_len = 0
            enable_index = indices[enable_label]
            for ind in range(enable_index+1):
                start_pos = line.find(split_line[ind], 
                                      start_pos+previous_len)
                previous_len = len(split_line[ind])

            # make enable negative and check if enable field correct 
            if not (line[start_pos] == '-'):
                line = line[:start_pos-1] + '-' + line[start_pos:]
            if not (line[start_pos : start_pos + previous_len] == 
                    split_line[ind]):
                raise ValueError(
                    "Could not parse line " + line + " because " +
                    line[start_pos : start_pos + previous_len] + 
                    " found at position " + str(previous_len) + " while " +
                    split_line[ind] + " expected.")

            # keep the (modified) line
            out_lines.append(line)               
            if verbose:
                print "Particle " + xmipp_compact + " disabled."

        else:

            # line skipped
            if verbose:
                print "Particle " + xmipp_compact + " omitted."

    # write out xmipp file
    if isinstance(xmippoutstar, str):
        xmippoutstar = open(xmippoutstar, 'w')
    for line in out_lines:
        xmippoutstar.write(line)
    xmippoutstar.flush()

    # 
    if verbose:
        print('relion particles not found in xmipp: ' 
              + str(relion_compacts))


#############################################################
#
# Tomo processing 
#

def average_fixed(starfile, mean_file):
    """
    Averages all images listed in the arg starfile (label '_rlnImageName')
    without translations and rotations.

    Arguments:
      - starfile: name of the starfile
      - name of the average image file 
    """
    
    # get all particle names
    particle_files = get_array_data(
        starfile=starfile, tablename='loop_', 
        labels=['rlnImageName'], types=str)['rlnImageName']

    # relative directory of the star file
    rel_dir = os.path.split(starfile)[0]

    for part_file in particle_files:

        rel_part_file = os.path.join(rel_dir, part_file)
        try:
            part_sum = part_sum + pyto.core.Image.read(rel_part_file).data
        except NameError:
            part_sum = pyto.core.Image.read(rel_part_file).data
        
    # get mean and save 
    mean = part_sum / len(particle_files)
    mean_obj = pyto.core.Image(data=mean)
    mean_obj.write(file=mean_file)


##############################################################
#
# Lower level functions for processing star files
#

def get_array_data(starfile, tablename, labels=None, types=float):
    """
    Gets data (for all particles) for specified labels. 

    Few notes about file parsing:
      - comment lines start with '#' and are always ignored
      - labels start with '_' and can have preceeding white characters (' ' in
      xmipp)
      - line srarting with arg tablename indicates start of the table
      - table data begins at the first line that doesn't start with '_'
      (optionally preceeded with whitespace)
      - table data ends when an empty line is reached
      - the order of data records is determined by indices written next to
      labels (relion) or by label order if no index is written next to labels

    Few notes about file parsing:
      - comment lines start with '#' and are always ignored
      - labels start with '_' and can have preceeding white characters (' ' in
      xmipp)
      - line srarting with arg tablename indicates start of the table
      - table data begins at the first line that doesn't start with '_'
      (optionally preceeded with whitespace)
      - table data ends when an empty line is reached
      - the order of data records is determined by indices written next to
      labels (relion) or by label order if no index is written next to labels

    Arguments:
      - starfile: name of the starfile that contains data
      - tablename: name of the table that contains data
      - labels: list of lables (variables) without leading '_', if None 
      all labels are used
      - types: list of functions (or a single function) that converts the data 
      into proper types (such as str, int, float), in the order of arg labels
      (default is float)

    Returns dictionary where keyes are labels and values are data arrays 
    """

    # get all labels if needed
    if labels is None:
        labels = array_data_head(
            starfile=starfile, tablename=tablename, top=False, labels=True)

    # initialization
    label_indices = {}
    data_str = {}
    for one_label in labels:
        data_str[one_label] = []

    # read file
    line_gen = array_data_generator(starfile=starfile, tablename=tablename)
    for line, label_indices in line_gen:

        # read data
        for one_label in labels:
            index = label_indices[one_label]
            data_str[one_label].append(line[index])
 
    # convert
    if not isinstance(types, (list, tuple)):
        types = [types] * len(labels)
    data = {}
    for one_label, one_type in zip(labels, types):
        data[one_label] = [one_type(x) for x in data_str[one_label]]
        if (one_type == float) or (one_type == int):
            data[one_label] = np.array(data[one_label])

    return data

def array_data_length(starfile, tablename):
    """
    Returns the number of entries in the specified table.

    Arguments:
      - starfile: name of the starfile that contains data
      - tablename: name of the table that contains data
    """

    n = 0
    line_gen = array_data_generator(starfile=starfile, tablename=tablename, 
                                    split=False)
    for line in line_gen:
        n += 1

    return n

def array_data_generator(
    starfile, tablename, split=True, labels=True, labels_dict=True):
    """
    Generator that yelds data lines from file (arg) starfile, table (arg) 
    tablename. 

    Few notes about file parsing:
      - comment lines start with '#' and are always ignored
      - labels start with '_' and can have preceeding white characters (' ' in
      xmipp)
      - line srarting with arg tablename indicates start of the table
      - table data begins at the first line that doesn't start with '_'
      (optionally preceeded with whitespace)
      - table data ends when an empty line is reached
      - the order of data records is determined by indices written next to
      labels (relion) or by label order if no index is written next to labels

    Arguments:
      - starfile: name of the starfile that contains data
      - tablename: name of the table that contains data
      - split: flag indicating if yielded data lines should be split in a list
      of strings
      - labels: flag indicating if labels are yielded also
      - labels_dict: flag indicating if the yielded labels should be in
      the dict form (label : index). If not, labels are returned as a list

    Yields: data_line, label_indices / labels_list
      - data_line: data line split in a list or one string
      - label_indices: dictionary where each item has a label (without leading 
      '_') as a key and the position (index) of that label in the data_line
      list as the corresponding value (only if args labels and labels_dict 
      are True).
      - label_list: list of labels (without leading '_'), only if arg labels 
      and labels_dict is True and arg labels_dist is False.
    """

    # initialization
    block_found = False
    data_found = False
    label_indices = {}
    labels_list = []

    # read file
    for line in open(starfile):
        if not block_found:

            # skip comments
            if line.startswith('#'):
                continue

            # look for block beginning
            if line.startswith(tablename):
                block_found = True
                continue

        else:

            # finish if passed data
            if data_found:
                if line.isspace() or (line.strip().startswith('_')):
                    return

            # skip these
            if (line.startswith('loop') or line.isspace() or 
                line.startswith('#')):
                continue

            # labels or data
            line_strip = line.strip()
            if line_strip.startswith('_'):

                # add label : index to label_indices
                line_split = line.split()
                labels_list.append(line_split[0].lstrip('_'))
                label = line_split[0][1:]
                if len(line_split) >= 2:
                    index = int(line_split[1][1:]) - 1
                else:
                    try:
                        index = index + 1
                    except NameError:
                        index = 0
                label_indices[label] = index                    

            else:

                # yield data
                data_found = True
                if split:
                    line_split = line.split()
                    if labels and labels_dict:
                        yield line_split, label_indices
                    elif labels:
                        yield line_split, labels_list
                    else:
                        yield line_split
                else:
                    if labels and labels_dict:
                        yield line, label_indices
                    elif labels:
                        yield line_split, labels_list
                    else:
                        yield line

def sort_particles(particles, indices=False, compact=False, type=None):
    """
    Sorts particle names given in spider format (used by relion and xmipp).

    Only the particle number and the file name are taken into account for
    sorting, the directory part and the extension are disregarded. Therefore,
    this function should be used if all particles are in the same directory,
    or if all particle file names (without directory part and extension) are 
    unique even if they reside in different directories. 

    Particle number and file name (without directory part and extension) are 
    concatenated to obtain the compact form of the particle name. In addition,
    if type is 'relion', the trailing '_particles' is removed from the compact
    form. If type is None or 'xmipp' compact form is not changed.

    Note: All particles 

    Arguments:
      - particles: array of particle names
      - indices: flag indicating if an ndarray of indices corresponding to the 
      sorting (what np.argsort returns) is returned
      - compact: Flag indicating if an ordered ndarray of compact particle 
      names (particle number + file name) is returned
      - type: indicates additional processing of the compact form

    Returns ndarray of sorted partcle names and optionally inices (for 
    sorting) and sorted compact particle names.  
    """

    # constants
    relion_remove_suffix = '_particles'

    # parse names
    all_numbers = []
    all_bases = []
    all_compact = []
    particles = np.asarray(particles)
    for full_name in particles:
        number, path = full_name.split('@', 1) 
        dir_, name = os.path.split(path)
        base, ext = name.rsplit('.', 1)
        all_numbers.append(number)
        if type is None:
            pass
        elif type == 'relion':
            base = base.rstrip(relion_remove_suffix)
        elif type == 'xmipp':
            pass
        all_bases.append(base)
        if compact:
            all_compact.append(number + base)

    # sort
    lexsorted = np.lexsort((all_numbers, all_bases))
    sorted_particles = particles[lexsorted]
    if compact:
        sorted_compact = np.asarray(all_compact)[lexsorted]

    # return
    result = [sorted_particles]
    if indices:
        result.append(lexsorted)
    if compact:
        result.append(sorted_compact)
    if len(result) ==  1:
        return result[0]
    else:
        return tuple(result)

def array_data_head(starfile, tablename, top=False, labels=False):
    """
    Returns header of the specified table in the specified file.

    If arg labels is True, returns only a list of labels (without leading '_').

    Arguments:
      - starfile: name of the starfile that contains data
      - tablename: name of the table that contains data
      - top: if True, the top of the file, before the first table is reached,
      is included in the header lines (default False)

    Returns: list of header lines
    """

    # initialization
    block_found = False
    data_found = False
    a_block_found = False
    head = []

    # check if starfile exists
    if not os.path.exists(starfile):
        raise ValueError("Starfile: " + str(starfile) + " doesn't exist.")

    # read file
    for line in open(starfile):

        # skip comments no matter what
        if line.startswith('#'):
            if top and not a_block_found:
                if not labels:
                    head.append(line)
            continue

        if not block_found:

            # look for block beginning
            if not (line.startswith(' ') or line.startswith('_')):
                a_block_found = True
                if line.startswith(tablename):
                    block_found = True
                    if not labels:
                        head.append(line)
                    continue

            # don't know what it is, copy if out of block and requested
            if top and not a_block_found:
                if not labels:
                    head.append(line)

        else:

            # copy these
            if (line.startswith('loop') or line.isspace()):
                if not labels:
                    head.append(line)
                continue

            # copy labels or return if reached data
            clean_line = line.lstrip()
            if clean_line.startswith('_'):
                if not labels:
                    head.append(line)
                else:
                    lab_line = clean_line.rstrip()
                    lab_line = lab_line.rsplit('#')[0].rstrip().lstrip('_')
                    head.append(lab_line)
                continue
            else:
                break
        
    return head

def write_table(starfile, labels, data, format_, comment='#', header=None, 
                tablename=None, labelindent=None, delimiter=' '):
    """
    Writes a data table to a star file.

    Typical use: args starfile, tablename, labels, data, and format_ are given.

    Alternatively, arg header is used in which case arg tablename is ignored. 

    Arguments:
      - starfile: name of the file or a file descriptor
      - comment: comment written at the file head
      - tablename: name of the table
      - labels: list of labels (with or without leading '_')
      - data: dictionary where keys are labels and values are data ndarrays
      - format_: data format, can be a list of format strings ordered 
      according to arg labels, or a dictionary where keys are labels and 
      values are format strings. If 'auto', floats are formated as '%12.6f' 
      and ints as '%12d'.
      - header: list of header lines, from the beginning to (not including) 
      the labels
      - labelindent: (str) label indentation (typically ' ' for xmipp, None
      for relion
    """

    # constants
    #column_separator = '  '
    int_format = '%12d'
    float_format = '%12.6f'
    str_format = '%s'

    # open file if needed
    if isinstance(starfile, str):
        fd = open(starfile, 'w')
    else:
        fd = starfile
    lines = []

    # comment
    if comment is not None:
        lines.append(comment)
        lines.append('')

    # data header
    if header is not None:
        lines.extend(header)
    else:
        lines.append(tablename)
        lines.append('')
        lines.append('loop_')

    # labels
    for label, ind in zip(labels, range(len(labels))):
        if not label.startswith('_'):
            label = '_' + label
        if labelindent is not None:
            label = labelindent + label
        lines.append(label + ' #' + str(ind+1))

    # write the header 
    for one_line in lines: 
        fd.write(one_line + os.linesep)

    # write data
    try:
        if isinstance(data, dict):

            # figure out format
            if isinstance(format_, str) and (format_ == 'auto'):

                # auto: convert string arrays to int or float, if possible
                fmt = []
                for lab in labels:
                    data_lab = np.asarray(data[lab])
                    if np.issubdtype(data_lab.dtype, 'str'):
                        new_fmt = str_format
                        try:
                            data[lab] = data_lab.astype('float')
                            new_fmt = float_format
                            data[lab] = data_lab.astype('int')
                            new_fmt = int_format
                        except ValueError:
                            pass
                    elif np.issubdtype(data_lab.dtype, 'float'):
                        new_fmt = float_format
                    elif np.issubdtype(data_lab.dtype, 'int'):
                        new_fmt = int_format
                    else:
                        raise ValueError(
                            "Could not do format_='auto' because array format"
                            + " was not understood.")
                    fmt.append(new_fmt)

            elif isinstance(format_, dict):
                # convert format to list
                fmt = [format_[lab] for lab in labels]
            else:
                fmt = format_

            # arrange data into a table and write
            data_nd = np.column_stack(
                [np.asarray(data[lab]).astype('object') for lab in labels])
            np.savetxt(fd, data_nd, fmt=fmt, delimiter=delimiter)

        else:
            raise ValueError("Argument data has to be dict type")
    finally:
        fd.flush()

def find_file(
        basename, suffix, iter_=None, iter_format='_it%03d', half=None, 
        half_format='_half%d', cont=True):
    """
    Returns paths of all existing files that can be composed as:
    
      basename + continuation_str + iteration_str + half_str + suffix

    from the specified arguments, where the continuation string may or may
    not exist. If it exists it is of the form '_ct' + number, for example
    '_ct21' or '_ct4'.
    
    If arg cont is False, the continuation string (above) is not allowed.

    If the path without the continuation is found, does not search for
    paths with continuation.

    Arguments:
      - basename: part of the path that contains everything until the part
      composed of continuation, iteration, half and suffix strings 
      - suffix: the last part of the path, everyhing after continuation, 
      iteration and half strings
      - iter_: iteration number, None for no iteration
      - iter_format: formatting of the interation
      - half: number of the half (1 or 2), None when no half is in the path
      - half_format: formatting of the half string
      - cont: flag indication if continued executions are used
    
    Returns:
      - None if no path found
      - list of paths that satisfy the arguments
    """

    # make the part of the path after continuation
    if iter_ is not None:
        iter_str = iter_format % iter_
    else:
        iter_str = ''
    if half is not None:
        half_str = half_format % half
    else:
        half_str = ''
    after_ct = iter_str + half_str + suffix

    starfile = basename + after_ct
    if os.path.exists(starfile):

        # file found wo continuation
        result = [starfile]

    else: 

        # no continuation allowed
        if not cont:
            return None

        # try to find continuation
        found_ct = []
        dir_, prefix = os.path.split(basename)
        for file_ in os.listdir(dir_):
            reg = re.search('^' + prefix + '_ct(.+)' + after_ct + '$', file_)
            if reg is not None:
                found_path = os.path.join(dir_, file_)
                found_ct.append(found_path)
        result = found_ct

    # return
    if len(result) == 0:
        return None
    else:
        return result

##############################################################
#
# Plots
#

def plot_fsc(
        basename, iter=None, half=None, class_id=1, legend=False, new=True):
    """
    Plots FSC from 3d refinement or 3d classification.

    FSC data is read from model star file determined by args basename, iter,
    and half. The table read data_model_class_ + class_id. The column labels 
    are 'rlnResolution' (x-axis) and 'rlnGoldStandardFsc' (y-axis).

    To plot the fsc of the final model set args iter and half to None.  

    Arguments:
      - basename: model file base name (directory and file name until '_it')
      - iter: (single int or list of ints) one or more iterations, or 'all'
      for all iterations
      - half: determine which data half is used (None if data is not separated
      in halves, 1, 2, or [1,2])
      - class_id: (single int or list of ints) one or more class ids
      - legend: flag indicating if legend is shown
      - new: flag indicating if a new plot is made
    """

    # constants
    label_x = 'rlnResolution'
    label_y = 'rlnGoldStandardFsc'

    # main plot
    plot_multi(
        basename=basename, label_x=label_x, label_y=label_y, iter=iter,
        half=half, class_id=class_id, new=new)

    # finish plot
    plt.axis((0, 0.2, -0.1, 1))
    plt.plot([0, 0.5], [0.5, 0.5], 'k--')
    plt.plot([0, 0.5], [0.143, 0.143], 'k--')
    if legend:
        plt.legend()
    plt.show()

def plot_post_fsc(starfile, legend=True):
    """
    """

    # constants:
    tablename='data_fsc'
    label_x='rlnResolution'

    # start figure
    plt.figure()

    # unmasked
    plot(starfile=starfile, tablename=tablename, label_x=label_x, 
         label_y='rlnFourierShellCorrelationUnmaskedMaps', one=False, 
         label='unmasked')
    
    # masked
    plot(starfile=starfile, tablename=tablename, label_x=label_x, 
         label_y='rlnFourierShellCorrelationMaskedMaps', one=False, 
         label='masked')
    
    # corrected
    plot(starfile=starfile, tablename=tablename, label_x=label_x, 
         label_y='rlnFourierShellCorrelationCorrected', one=False, 
         label='corrected')

    # finish plot
    plt.axis((0, 0.2, -0.1, 1))
    plt.plot([0, 0.5], [0.143, 0.143], 'k--')
    plt.plot([0, 0.5], [0.5, 0.5], 'k--')
    if legend:
        plt.legend()
    plt.show()

def plot_guinier(starfile, legend=True):
    """
    """

    # constants:
    tablename='data_guinier'
    label_x='rlnResolutionSquared'

    # start figure
    plt.figure()

    # original
    plot(starfile=starfile, tablename=tablename, label_x=label_x, 
         label_y='rlnLogAmplitudesOriginal', one=False, label='original')

    # weighted
    plot(starfile=starfile, tablename=tablename, label_x=label_x, 
         label_y='rlnLogAmplitudesWeighted', one=False, label='weighted')

    # sharpened
    plot(starfile=starfile, tablename=tablename, label_x=label_x, 
         label_y='rlnLogAmplitudesSharpened', one=False, label='sharpened')

    # finish plot
    if legend:
        plt.legend()
    plt.show()

def plot_ssnr(basename, iter=None, half=None, class_id=1, legend=True):
    """
    Plots SSNR.

    SSNR data is read from model star file determined by args basename, iter,
    and half. The table read data_model_class_ + class_id. The column labels 
    are 'rlnResolution' (x-axis) and 'rlnSsnrMap' (y-axis).

    Arguments:
      - basename: model file base name (directory and fine name until 'it')
      - iter: (single int or list of ints) one or more iterations, or 'all'
      for all iterations
      - half: determine which data half is used (None if data is not separated
      in halves, 1, 2, or [1,2])
      - class_id: (single int or list of ints) one or more class ids
      - legend: flag indicating if legend is shown
    """    

    # constants
    label_x = 'rlnResolution'
    label_y = 'rlnSsnrMap'

    # main plot
    plot_multi(
        basename=basename, label_x=label_x, label_y=label_y, iter=iter,
        half=half, class_id=class_id)

    # finish plot
    plt.axis((0, 0.2, -0.1, 5))
    plt.plot([0, 0.5], [1, 1], 'k--')
    if legend:
        plt.legend()
    plt.show()

def plot_multi(basename, label_x, label_y, iter=None, half=None, class_id=1, 
               new=True):
    """
    Plots data corresponding to args label_x and label_y from one or more 
    files. 

    File names are formed as all possible combinations that have the form:

      basename + '_it' + iteration + '_half' + one_half + '_model.star'

    where iteration and one_half are elements of (args) iter and and half, 
    respectively. If None is an element of iter or half, the preceeding string
    ('_it' or '_half') is also skipped. If such file doesn't exist, it is
    checked if a continuation file exist.

    If the file with such file name still doesn't exist a warning is printed 
    and the files are skipped.

    Furthermore, combinations of all file names obtained as above and 
    tablenames ('data_model_class_' + one_class_id, where one class_id is an 
    element of arg class_id) are considered. 

    Arguments:
      - basename: model file base name (directory and fine name until 'it')
      - iter: (single int or list) one or more iterations, None for no 
      iteration, 'all' for all iterations
      - half: (single int or list) denotes data half(s) used, None data is not
      separated in halfs
      - class_id: (single int or list) one or more class ids
      - label_x, label_y: data labels
      - new: flag indicating if a new plot is made
    """

    # constants, can be moved to args if needed
    iter_format = '_it%03d'
    half_format = '_half%d'
    table_format = 'data_model_class_%d'

    # start plot
    if new:
        plt.figure()

    # fugure out arguments
    if not np.iterable(iter):
        iter = [iter]
    elif iter == 'all':
        iter = list_iterations(basename=basename)
    if not np.iterable(class_id):
        class_id = [class_id]
    if not np.iterable(half):
        half = [half]

    for cid, it, ha in itertools.product(class_id, iter, half):  

        # make file name
        #if it is not None:
        #    iter_str = iter_format % it
        #else:
        #    iter_str = ''
        #if ha is not None:
        #    half_str = half_format % ha
        #else:
        #    half_str = ''
        #suffix = iter_str + half_str + '_model.star'
        #starfile = (basename + suffix)
        
        # check if file exists
        #if not os.path.exists(starfile):

            # try to find unique continuation
            #found_ct = []
            #dir_, prefix = os.path.split(basename)
            #for file_ in os.listdir(dir_):
            #    if file_.startswith(prefix + '_ct') and file_.endswith(suffix):
            #        found_ct.append(file_)
        path = find_file(
            basename=basename, iter_=it, half=ha, suffix='_model.star', 
            iter_format=iter_format)

        # find starfile
        if path is None:
            warnings.warn(
                "Skipping basename " + basename + " iteration " + str(it) + 
                " half" + str(half) +
                "because the corresponding file doesn't exist.")
            continue
        elif len(path) == 1:
            starfile = path[0]
        elif len(path) == 2:
            starfile = path[0]
        elif len(path) >= 2:
            warnings.warn(
                "Skipping basename " + basename + " iteration " + str(it) + 
                " half" + str(half) +
                "because >2 files matched.")
            continue

            #if len(path) == 1:
            #    starfile = os.path.join(dir_, found_ct[0])
            #elif len(found_ct) == 2:
                # if continuation here pick any 
            #    for f_ct in found_ct:
            #        if file_.startswith(prefix + '_ct'):
            #            starfile = os.path.join(dir_, f_ct)
            #else:
            #    warnings.warn(
            #        "Skipping file " + starfile + " because it doesn't exist.")
            #    continue

        # make label
        if it is not None:
            iter_label = ', iter %d' % it
        else:
            iter_label = ''
        if ha is not None:
            half_label = ', half %d' % ha
        else:
            half_label = ''    
        label = ('class %d' % cid) + iter_label + half_label

        # make table name
        tablename = table_format % cid

        # plot
        plot(starfile=starfile, tablename=tablename, label_x=label_x, 
             label_y=label_y, one=False, label=label)

def plot(starfile, tablename, label_x, label_y, one=True, label=None):
    """
    Plots data from (arg) starfile and datablock (arg) tablename, so that
    label_x is shown on x and (arg) label_y on the y axis.

    If arg one is True a plot is started and finished here. Otherwise, it is 
    expected that a plot is already open and that this function only adds 
    data to it.

    Arguments:
      - startfile: fine name
      - tablename: data block name
      - label_x, label_y: variable names (without leading '_'
      - one: Flag indication if only one plot is needed, or if this function
      is called multiple times to add data to the same plot
    """

    # initialization
    block_found = False
    header_found = False
    data_found = False
    label_x_found = False
    label_y_found = False
    data_x = []
    data_y = []

    # read file
    for line in open(starfile):

        if not block_found:

            # skip comments
            if line.startswith('#'):
                continue

            # look for block beginning
            if line.startswith(tablename):
                block_found = True
                continue

        else:

            # finish if passed data
            if data_found:
                if line.isspace() or line[0].isalpha() or (line[0] == '_'):
                    break

            # skip these
            if (line.startswith('loop') or line.isspace()):
                continue

            # find label indices
            if line.startswith('_'):
                line_split = line.split()
                if line_split[0] == ('_' + label_x):
                    label_x_index = line_split[1]
                    label_x_index = int(label_x_index[1:])
                elif line_split[0] == ('_' + label_y):
                    label_y_index = line_split[1]
                    label_y_index = int(label_y_index[1:])
                continue

            else:

                # read data
                line_split = line.split()
                data_x_one = float(line_split[label_x_index-1])
                data_x.append(data_x_one)
                data_y_one = float(line_split[label_y_index-1])
                data_y.append(data_y_one)

                data_found = True

    # plot
    if one:
        plt.figure()
    if label is None:
        plt.plot(data_x, data_y, '-')
    else:
        plt.plot(data_x, data_y, '-', label=label)
    plt.title(starfile)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    if one:
        plt.show()

def list_iterations(basename):
    """
    Makes a sorted ndarray of all iterations that matches basename.
    """

    iter_set = set()
    dir_, prefix = os.path.split(basename)
    for file_ in os.listdir(dir_):
        if not file_.startswith(prefix + '_it'):
            continue

        it = int(file_.lstrip(prefix + '_it')[:3])
        iter_set.add(it)

    iters = np.array(list(iter_set))
    iters = np.sort(iters)

    return iters

def prepare_class_arg(class_, mode='tuple'):
    """
    Converts arg class_ in a form useful for further processing:

      - If class_ is None, returns None
      - If class is a single number (not a list or ndarray) and arg mode 
      is 'list', returns class_ as a list (that is [class_])
      - If class is a single number (not a list or ndarray) and arg mode 
      is 'tuple', if first converts class_ to a list and then converts
      each non-tuple element to a tuple. For example:

      prepare_class_arg(3, mode-'tuple') -> [(3,)]
      prepare_class_arg([3, (4,5), 6] mode-'tuple') -> [(3,), (4,5), (6,)]

    Arguments:
      - class_: class(es) as a single int, list of ints or a list of ints and
      tuples of ints
      - mode: 'list' or 'tuple'      
    """
    
    if class_ is None:
        return None
        
    # make a list
    if not isinstance(class_, (list, np.ndarray)): 
        class_ = [class_]
        
    #
    if mode == 'list':
        return class_

    
    elif mode == 'tuple':

        class_new = []
        for cl in class_:
            if isinstance(cl, tuple): 
                class_new.append(cl)
            else:
                class_new.append((cl,))
        return class_new

    else:
        raise ValueError(
            "Argument mode: " + str(mode) + " was not understood. "
            + "Valid options are 'list' and 'tuple'.")


##############################################################
#
# Transformations
#

def write_transformed(
    array, angles, origin, scale=1., d=0., rot_mode='zyz_in_active', 
        dir_=None, suffix='', dtype='float32', pixel=1):
    """
    Transforms 3D array according to angles, scale and d and writes the 
    resulting array.

    The file name is formed as:

      r-<angle_0>-<angle_1>-<angle_2>_t-<d_0>-<d_1>-<d_2>_suffix.mrc

    Arguments:
      - array: 3D ndarray to be transformed
      - angles: euler angles in rad
      - origin: coordinates of the rotation origin
      - scale: scale factor
      - d: translations
      - rot_mode: Eauler angles convention
      - dir_: directory name where the transformed image is stored
      - suffix: file name suffix, if given, '_' is prepended 
      - dtype: data type (should correspond to the file format)
      - pixel: pixel size in nm

    Returns file_path
    """
    
    # make transformed images
    angles = np.array(angles) 
    q = Rigid3D.make_r_euler(angles=angles, mode=rot_mode)
    s_scalar = scale
    r3d = Rigid3D(q=q, scale=s_scalar, d=np.array(d))
    gg_tr = r3d.transformArray(array=array, origin=origin)

    # write images
    angles_deg = angles * 180 / np.pi
    if suffix != '':
        file_name = (
            'r-%d-%d-%d_t-%d-%d-%d_%s.mrc' 
            % (tuple(np.hstack([angles_deg, d])) + (suffix,)))
    else:
        file_name = (
            'r-%d-%d-%d_t-%d-%d-%d.mrc' 
            % tuple(np.hstack([angles_deg, d])))
    gg_im = pyto.core.Image(data=gg_tr)
    file_path = os.path.join(dir_, file_name)
    gg_im.write(file=file_path, dataType=dtype, pixel=pixel)

    return file_path     

def multi_transform(
    array, transform_iter, origin, part_file, part_dir, 
        rot_mode='zyz_in_active', star='default', number=False, 
        dtype='float32', pixel=1):
    """
    Makes multiple transformations of array (image) according to the 
    transformation iterator, saves the  files and writes a corresponding 
    star file.

    Arg star determines what values for angles and origin are entered
    into the star file. If star is 'default', all are set to 0. This is
    useful when the particles and the star file generated here are used
    for (global) alignment and averaging.

    If arg star is 'inverse', negative translations are used for star file 
    rlnOriginX/Y/Z and the inverse rotation angles (in deg) are used
    for relion rotation parameters. Provided that arg rot_mode is 
    'zyz_in_active', this gives relion-correct assignements, because
    the application of translation given by rlnOriginX/Y/Z and rotation
    specified by relion angles on a (transformed) array (particle) yields 
    (the non-transformed) arg array.

    In the star file, '_rlnMicrographName' is set to 'none', ImageName to 
    the correct name and and all
    angles and coordinates to 0.

    Arguments:
      - array: 3D ndarray (image) to be transformed
      - transform_iter: iterator composed of 6 iterators, the three Euler
      angles (in rad) and x, y and z translations (in pixels)
      - origin: coordinates of the rotation origin
      - rot_mode: Euler angles convention
      - part_dir_: directory name where the transformed images are stored
      - number: Flag indicating whether a unique index (counter) is appended 
      to file names (useful when multiple copies of the same transformed 
      image are written
      - dtype: data type (should correspond to the file format)
      - pixel: pixel size in nm

    """
    # labels
    labels = [
        '_rlnMicrographName', '_rlnCoordinateX', '_rlnCoordinateY', 
        '_rlnCoordinateZ',
        '_rlnImageName', '_rlnAngleRot', '_rlnAngleTilt', '_rlnAnglePsi',
        '_rlnOriginX', '_rlnOriginY', '_rlnOriginZ'
        ]
    #labels_full = [
    #    lab + ' #' + str(ind+1) for lab, ind in zip(labels, range(len(labels)))]
    #format_ = ['%s', '%d', '%d', '%d', '%s', '%d', '%d', '%d']
    format_ = '%s '

    # make particles directory
    if not os.path.exists(part_dir):
        os.makedirs(part_dir)
    
    # initialize data
    data = {}
    for lab in labels:
        data[lab] = []

    # make image and add to star file
    index = 0
    for angles, trans in transform_iter:

        # write transformed arrays
        index += 1
        if number:
            path = write_transformed(
                array=array, angles=angles, origin=origin, d=trans, 
                dir_=part_dir, rot_mode=rot_mode, suffix=('%02d' % index))
        else:
            path = write_transformed(
                array=array, angles=angles, origin=origin, d=trans, 
                dir_=part_dir)

        # save data to add to star file
        data['_rlnMicrographName'].append('none')
        data['_rlnImageName'].append(path)
        data['_rlnCoordinateX'].append(0)
        data['_rlnCoordinateY'].append(0)
        data['_rlnCoordinateZ'].append(0)

        if star == 'default':
            data['_rlnOriginX'].append(0)
            data['_rlnOriginY'].append(0)
            data['_rlnOriginZ'].append(0)
            data['_rlnAngleRot'].append(0)
            data['_rlnAngleTilt'].append(0)
            data['_rlnAnglePsi'].append(0)

        elif star == 'inverse':            
            data['_rlnOriginX'].append(-trans[0])
            data['_rlnOriginY'].append(-trans[1])
            data['_rlnOriginZ'].append(-trans[2])
            data['_rlnAngleRot'].append(-angles[2] * 180 / np.pi)
            data['_rlnAngleTilt'].append(-angles[1] * 180 / np.pi)
            data['_rlnAnglePsi'].append(-angles[0] * 180 / np.pi)

        else:
            raise ValueError(
                "Argument star ", + star + " was not understood. Currently "
                + "implemented are 'default' and 'inverse'.")

    # finish data
    for lab in labels:
        data[lab] = np.asarray(data[lab])

    # write star file
    write_table(
        starfile=part_file, comment=None, tablename='data_', 
        labels=labels, data=data, format_=format_)
 
def symmetrize_particle_data(
        symmetry, instar, outstar, relion_dir=None, copy=True, 
        particles_dir=None, format_='%s '):
    """
    Symmetrizes particle data given in the specified relion star file.  

    Each particle of the initial relion star file (arg instar), is copied 
    as many times as there are rotational symmetries in the specified 
    symmetry group (arg symmetry). The Euler angles are modified to reflect 
    these rotations.

    More precisely, symmetrized roatations are obtained as a composition of
    symmetery rotations and the initial rotations specifed by Euler
    angles in the initial star file. In this way, the initial rotation is 
    applied on particles first and then a symmetry rotation. This is because
    rotation specified in relion star files rotates particles to obtain
    an averaged structure.

    If arg copy is False, particles (files) are not changed, so each particle
    file name occurs multiple times (with different angles) in the 
    symmetrized (final) star file.

    If arg copy is True, particles are copied for each rotation, so that
    particle file names in the symmetrized star file are unique. (Not sure
    if this is needed for relion to work properly, just in case.)
    
    Arguments:
      - symmetry: symmetry type, currently cn and dn
      - instar: input star file
      - outstar: output (symmetrized) star file
      - copy: flag indicating if particles are copied so that each 
      orientation has a corresponding particle file
      - relion_dir: relative path to the relion directory (in respect to
      which all paths are written in star files)
      - particles_dir: directory where the symmetrized particles are written
      relative to the relion directory. If None symmetrized particles 
      are written in the same directory where the particles already are
      - format: star file entries format
    """

    pi = np.pi

    # make particles dir if needed
    if copy and (particles_dir is not None):
        particles_dir_real = os.path.join(relion_dir, particles_dir) 
        if not os.path.isdir(particles_dir_real):
            #try:
                os.makedirs(particles_dir_real)
            #except OSError:
            #    pass

    # get symmetry iterator
    sym_n = int(symmetry[1])
    if (symmetry[0] == 'c') or (symmetry[0] == 'C'):
        sym_phi = 2. * pi * np.arange(sym_n) / float(sym_n)
        angle_iter = [[phi, 0, 0] for phi in sym_phi]
        sym_multiplicity = sym_n
    
    elif (symmetry[0] == 'd') or (symmetry[0] == 'D'):   
        sym_phi = 2. * pi * np.arange(sym_n) / float(sym_n)
        sym_theta = [0, pi]
        phi_theta_iter = itertools.product(sym_phi, sym_theta, [0])
        angle_iter = [
            [phi_theta[0], phi_theta[1], 0] for phi_theta in phi_theta_iter]
        sym_multiplicity = 2 * sym_n

    else:
        raise ValueError("Sorry, only C and D symmetries are implemented")
    
    data = {}
    init_iter = array_data_generator(
        starfile=instar, tablename='data', split=True, 
        labels=True, labels_dict=False)
    for init_data, labels in init_iter:
        init_dict = dict(zip(labels, init_data))
        #print init_dict['rlnImageName']
    
        # make all star variables except angles
        for lab in labels:
            if ((lab != 'rlnAngleRot') and (lab != 'rlnAngleTilt') 
                and (lab != 'rlnAnglePsi') and lab != 'rlnImageName'):
                for mult in range(sym_multiplicity):
                    try:
                        data[lab].append(init_dict[lab])
                    except KeyError:
                         data[lab] = [init_dict[lab]]
                
        # make angles
        rel_init_euler = [
            float(init_dict[name]) for name 
            in ['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']]
        rel_init_euler = np.array(rel_init_euler) * np.pi / 180
        rel_init_q = Rigid3D.make_r_euler(rel_init_euler, mode='zyz_in_active')
        rel_init_r3d = Rigid3D(q=rel_init_q)
        for sym_euler, sym_ind in zip(angle_iter, range(len(angle_iter))):
        
            # find symmetrized angles
            sym_q = Rigid3D.make_r_euler(sym_euler, mode='zyz_in_active')
            sym_r3d = Rigid3D(q=sym_q)
            final_r3d = Rigid3D.compose(sym_r3d, rel_init_r3d) 
            final_euler = final_r3d.extract_euler(
                final_r3d.q, mode='zyz_in_active') * 180 / np.pi

            # make symmetryzed names and copy particles, if needed
            old_name_rel = init_dict['rlnImageName']
            if copy:

                # make symmetryzed particle names
                split_name = old_name_rel.rsplit('.', 1)
                sym_suffix = '_' + symmetry + '-' + str(sym_ind)
                if particles_dir is not None:
                    dir_, name = os.path.split(split_name[0])
                    sym_name_rel = os.path.join(
                        particles_dir, name + sym_suffix + '.' + split_name[1])
                else:
                    sym_name_rel = (
                        split_name[0] + sym_suffix + '.' + split_name[1])

                # copy
                old_name = os.path.join(relion_dir, old_name_rel)
                sym_name = os.path.join(relion_dir, sym_name_rel)
                subprocess.call(['cp', old_name, sym_name])

            else:

                # symmetrized names not needed
                sym_name_rel = old_name_rel

            # add symmetrized angles to data
            try:
                data['rlnAngleRot'].append(final_euler[0])
                data['rlnAngleTilt'].append(final_euler[1])
                data['rlnAnglePsi'].append(final_euler[2])
                data['rlnImageName'].append(sym_name_rel)
            except KeyError:
                data['rlnAngleRot'] = [final_euler[0]]
                data['rlnAngleTilt'] = [final_euler[1]]
                data['rlnAnglePsi'] = [final_euler[2]]
                data['rlnImageName'] = [sym_name_rel]
            
    # convert all data to ndarray
    for lab in labels:
        data[lab] = np.asarray(data[lab])

    # write symmetrized star file
    write_table(
        starfile=outstar, comment=None, tablename='data_', 
        labels=labels, data=data, format_=format_)      

def add_priors(instar, outstar, prior_labels=None, format_='auto'):
    """
    Adds prior angular values to a particle file by copying the values of the 
    corresponding angles. The data is taken from an existing particle 
    file (arg instar) and written to a new file (arg outstar). 

    If arg prior_labels is not specified, prior angle labels are 
    _rlnAngleRotPrior, _rlnAngleTiltPrior and _rlnAnglePsiPrior, and the
    corresponding angle labels are _rlnAngleRot, _rlnAngleTilt and 
    _rlnAnglePsi.

    All other values are copied unchanged.

    Arguments:
      - instar: input particle data star file
      - outstar: output particle data star file
      - prior labels: (dict)
      - format_: data format, leaving the default ('auto') makes floats
      '%12.6f' and ints '%12d'.
    """

    # 
    if prior_labels is None:
        prior_labels = {
            'rlnAngleRot' : 'rlnAngleRotPrior',
            'rlnAngleTilt' : 'rlnAngleTiltPrior',
            'rlnAnglePsi' : 'rlnAnglePsiPrior'}

    # read data line by line
    data = {}
    init_iter = array_data_generator(
        starfile=instar, tablename='data', split=True, 
        labels=True, labels_dict=False)
    for init_data, labels in init_iter:
        init_dict = dict(zip(labels, init_data))
        #print init_dict['rlnImageName']
    
        # make all star variables
        for lab in labels:

            # add all data
            try:
                data[lab].append(init_dict[lab])
            except KeyError:
                data[lab] = [init_dict[lab]]

            # copy selected to priors
            if lab in prior_labels.keys():
                prior_lab = prior_labels[lab]
                try:
                    data[prior_lab].append(init_dict[lab])
                except KeyError:
                    data[prior_lab] = [init_dict[lab]]

    # add priors to labels
    labels_new = []
    priors_added = False
    pls = prior_labels.keys()
    for lab in labels:

        # add preexisting
        labels_new.append(lab)

        # check if all bases for priors passed
        if not priors_added and (lab in pls):
            pls.remove(lab)
            if len(pls) == 0:

                # add prior labels
                for pri_lab in prior_labels.values():
                    labels_new.append(pri_lab)
                priors_added = True

    # convert all data to ndarray
    for lab in labels_new:
        data[lab] = np.asarray(data[lab])

    # write final star file
    write_table(
        starfile=outstar, comment=None, tablename='data_', 
        labels=labels_new, data=data, format_=format_)      

def symmetrize_structure(
        structure, symmetry, average=None, dtype=None, pixel=None,
        mask=None, origin=None):
    """
    Rotates a given image (arg iniitial) according to the specified 
    symmetry (arg symmetry) and returns the average.

    If arg mask is specified, the initial image is first masked and then
    rotated and averaged.

    The rotation is performed around the specified origin. If None,
    the origin is in the image center (image_len / 2).

    Data type and pixel size are taken from the arguments, if specified.
    If these arguments are not given, they are read from the image file 
    if arg structure is a file. Pixel size and dtype are needed if the
    average image is saved to a file

    Arguments:
      - structure: initial image (file name, pyto.core.Image or ndarray)
      - symmetry: symmetry type ('Cn' or 'Dn')
      - origin: (list or ndarray) rotation origin
      - mask: mask image (file name, pyto.core.Image or ndarray), if 
      None no masking is performed
      - average: averaged image, file name or None to return the image
      - dtype: data type of the average image, only in case it is saved 
      in a file
      - pixel: pixel size in nm (used only for mrc format 

    Returns (ndarray) averaged image only in case average is None 
    """

    # get symmetry iterator
    pi = np.pi
    sym_n = int(symmetry[1])
    if (symmetry[0] == 'c') or (symmetry[0] == 'C'):
        sym_phi = 2. * pi * np.arange(sym_n) / float(sym_n)
        angle_iter = [[phi, 0, 0] for phi in sym_phi]
        sym_multiplicity = sym_n
    
    elif (symmetry[0] == 'd') or (symmetry[0] == 'D'):   
        sym_phi = 2. * pi * np.arange(sym_n) / float(sym_n)
        sym_theta = [0, pi]
        phi_theta_iter = itertools.product(sym_phi, sym_theta, [0])
        angle_iter = [
            [phi_theta[0], phi_theta[1], 0] for phi_theta in phi_theta_iter]
        sym_multiplicity = 2 * sym_n

    else:
        raise ValueError("Sorry, only C and D symmetries are implemented")

    # read image
    image_obj_found = False
    if isinstance(structure, str):
        image_obj = pyto.core.Image.read(structure)
        image_obj_found = True
        array = image_obj.data
    elif isinstance(structure, pyto.core.Image):
        image_obj = structure
        image_obj_found = True
        array = structure.data 
    elif isinstance(structure, np.ndarray):
        array = structure
    else:
        raise ValueError(
            "Argument structure was not understood. It should be a file name, "
            + "pyto.core.Image object or ndarray.")
    init_dtype = array.dtype 

    # read mask
    if mask is not None:
        if isinstance(mask, str):
            mask_obj = pyto.core.Image.read(mask)
            mask_array = mask_obj.data
        elif isinstance(mask, pyto.core.Image):
            mask_obj = mask
            mask_array = mask.data 
        elif isinstance(mask, np.ndarray):
            mask_array = mask
        else:
            raise ValueError(
                "Argument mask was not understood. It should be a file name, "
                + "pyto.core.Image object or ndarray.")

    # mask if needed
    if mask is not None:
        array = mask_array * array

    # origin
    if origin is None:
        origin = np.asarray(array.shape) / 2 
    
    new_array = np.zeros_like(array)
    for sym_euler, sym_ind in zip(angle_iter, range(sym_multiplicity)):  

        q = Rigid3D.make_r_euler(angles=sym_euler, mode='zxz_ex_active')
        r3d = Rigid3D(q=q, scale=1)
        rotated = r3d.transformArray(array=array, origin=origin)
        new_array = new_array + rotated

    new_array = new_array / float(sym_multiplicity)
      
    # return
    if average is None:
        return new_array

    else:

        # set write parameters
        if not image_obj_found:
            image_obj = pyto.core.Image()
        if dtype is None:
            dtype = init_dtype
        if (pixel is None) and image_obj_found:
            pixel = image_obj.pixelsize

        # write
        image_obj.data = new_array
        image_obj.write(file=average, dataType=dtype, header=True, pixel=pixel)
        
def symmetrize_particles(
        instar, outstar, tablename, symmetry, mask, name):
    """
    """

    # make symmetries

    # start symmetrized particles starfile

    # read particle data

    # calculate transformation 

    # read particle

    # transform particle

    # make particle file name and save

    # add entry to symmetrized particles starfile

    pass
