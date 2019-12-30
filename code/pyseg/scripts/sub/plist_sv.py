"""

    Script for generate subvolumes from a particle list

    Input:  - path to the input XML file
            - parth to reference tomgrams
            - path to output subvolumes

    Output: - The filtered particle list, survivor particle are unaltered

"""

__author__ = 'Antonio Martinez-Sanchez'

# ################ Package import

import  os
import sys
import time
import getopt
import copy
import operator
import pyseg as ps

################# Main declarations

_, cmd_name = os.path.split(__file__)

usage_msg = 'Basic Usage: ' + cmd_name + ' -i <in_xml> -o <out_xml> -e <elem> -c <value>\n' + \
            'Run \'' + cmd_name + ' -h\' for help and see additional parameters.'
help_msg = '    -i <path>: input xml particle list. \n' + \
           '    -o <path>: output filtered xml particle list.\n' + \
           '    -e <elem>: particle element for filtering.\n' + \
           '    -a <attrib>(optional): element attribute' + \
           '    -c <float or str>: constant value used for filtering comparisons.' + \
           '    -t <type>(optional): type used for values, valid: \'float\' (default) and \'str\'' + \
           '    -p <string>(optional): operation, valid: \'ne\' (default), \'eq\', \'lt\', \'le\', \'gt\' and \'ge\'.\n' + \
           '    -v (optional): verbose mode activated.'

################# Global variables

################# Helper classes (Visitors)


################# Helper functions

def str_to_oper(op):
    if op == 'ne':
        return operator.ne
    elif op == 'eq':
        return operator.eq
    elif op == 'lt':
        return operator.lt
    elif op == 'le':
        return operator.le
    elif op == 'gt':
        return operator.gt
    elif op == 'ge':
        return operator.ge
    else:
        return None

################# Work routine

def do_plist_filter(in_file, out_file, elem, attrib, cte, dtype, op, verbose):

    if verbose:
        print '\tLoading the input file...'
    in_path, _ = os.path.split(in_file)
    plist_in = ps.sub.ParticleList(in_path)
    plist_in.load(in_file)
    out_path, _ = os.path.split(out_file)
    plist_out = ps.sub.ParticleList(out_path)
    print '\t\t-Number of particles of the input particle list: ' + str(plist_in.get_num_particles())

    if verbose:
        print '\tFilter loop...'
    for et in plist_in.get_elements():
        el = et.find(elem)
        if el is not None:
            if attrib is None:
                hold_cte = el.text
            else:
                try:
                    hold_cte = el.attrib[attrib]
                except KeyError:
                    if verbose:
                        print '\tWARNING: particle without attribute ' + attrib
                    continue
            if dtype == 'float':
                hold_cte = float(hold_cte)
            if op(cte, hold_cte):
                plist_out.import_particle(et)
        else:
            if verbose:
                print '\tWARNING: particle without element ' + elem
    print '\t\t-Number of particles for output particle list: ' + str(len(plist_out._ParticleList__parts))

    if verbose:
        print '\tStoring the output file...'
    plist_out.store(out_file)


################# Main call

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hvi:o:e:a:c:t:p:")
    except getopt.GetoptError:
        print usage_msg
        sys.exit(0)

    in_file = None
    out_file = None
    elem = None
    attrib = None
    cte = None
    dtype = 'float'
    op = 'ne'
    verbose = True
    for opt, arg in opts:
        if opt == '-h':
            print usage_msg
            print help_msg
            sys.exit()
        elif opt == "-i":
            in_file = arg
        elif opt == "-o":
            out_file = arg
        elif opt == "-e":
            elem = arg
        elif opt == "-a":
            attrib = arg
        elif opt == "-c":
            cte = arg
        elif opt == "-t":
            dtype = arg
        elif opt == "-p":
            op = arg
        elif opt == "-v":
            verbose = True
        else:
            print 'Unknown option ' + opt
            print usage_msg
            sys.exit(1)

    if (in_file is None) or (out_file is None) or (elem is None) or (cte is None):
        print 'Not all input requirement fulfilled.'
        print usage_msg
        sys.exit(2)
    elif (dtype != 'float') and (dtype != 'str'):
        print 'Unknown input for type \'-t\' option value ' + dtype
        print usage_msg
        sys.exit(3)
    elif (op != 'ne') and (op != 'eq') and (op != 'lt') and \
        (op != 'le') and (op != 'gt') and (op != 'ge'):
        print 'Unknown input for operation \'-p\' option value ' + op
        print usage_msg
        sys.exit(4)
    else:
        # Print init message
        if verbose:
            print 'Filter for XML particle list.'
            print '\tAuthor: ' + __author__
            print '\tDate: ' + time.strftime("%c") + '\n'
            print 'Options:'
            print '\tInput file: ' + in_file
            print '\tOutput file: ' + out_file
            print '\tFilter criterium: '
            print '\t\t-Particle element: ' + elem
            if attrib is not None:
                print '\t\t\t+Element attribute: ' + attrib
                print '\t\t\t+Attribute constant value: ' + cte
            else:
                print '\t\t-Element constant value: ' + cte
            print '\t\t-Data type: ' + dtype
            print '\t\t-Operation: ' + op
            print ''

        # Change input parameters format
        if dtype == 'float':
            cte = float(cte)
        op = str_to_oper(op)

        # Do the job
        if verbose:
            print 'Starting...'
        do_plist_filter(in_file, out_file, elem, attrib, cte, dtype, op, verbose)

        if verbose:
            print cmd_name + ' successfully executed. (' + time.strftime("%c") + ')'


if __name__ == "__main__":
    main(sys.argv[1:])
