"""

    Script for reconstructing a surface from a segmented membrane

    Input:  - Original tomogram with the segmentation (0-fg, otherwise-bg)

    Output: - vtkPolyData with the surface
            - vtkPolyData with the cloud of points (optional)
            - tomogram with the signed distance (optional)

"""

__author__ = 'Antonio Martinez-Sanchez'

################# Package import

import sys
import os
import time
import getopt
from pyseg import *

################# Main declarations

_, cmd_name = os.path.split(__file__)

usage_msg = 'Basic Usage: ' + cmd_name + ' -i <input_tomogram> -o <output_dir> \n' + \
            'Run \'' + cmd_name + ' -h\' for help and see additional parameters.'
help_msg = '    -i <file_name>: input 3D density image in MRC, EM, VTK or FITS formats, segmented structures are' \
           '                    labelled as 1. \n' + \
           '    -o <dir_name>: Name of the directory where output and intermediate data will be ' + \
           'stored. If it already exists then will be cleared before running the tool. \n' + \
           '    -c (optional): a vtkPolyData with the cloud of points and their normals is also stored. \n' + \
           '    -d (optional): a tomogram with the signed distance is alse stored. \n' + \
           '    -p <int>: ratio of segmented voxels discarded, if 1 (default) all segmented voxels are used. \n' + \
           '    -f <em, mrc, fits or vti>: distance output format (default mrc). \n' + \
           '    -v (optional): verbose mode activated.'

SIGNED_DIST_PATH = '/home/martinez/workspace/c/vtk/signed_dist/build/signed_dist'

################# Work routine

def do_surf_recon(input_file, output_dir, p_ratio=1, cloud=False,  dist=False, fmt='.mrc',
                  verbose=False):

    # Surface generation
    if verbose:
        print '\tSurface generation...'
    input_tomo = disperse_io.load_tomo(input_file)
    if cloud or dist:
        surf_poly, cloud_poly = disperse_io.gen_surface_cloud(input_tomo, 1, p_ratio, True, False)
    else:
        surf_poly = disperse_io.gen_surface(input_tomo, 1, p_ratio, False, False)
    path, stem = os.path.split(input_file)
    stem, _ = os.path.splitext(stem)
    output_surf_name = output_dir + '/' + stem + '_surf.vtp'
    disperse_io.save_vtp(surf_poly, output_surf_name)

    # Signed distance measurement
    dist_tomo = None
    if dist:
        print '\tSigned distance measurement...'
        dist_tomo = disperse_io.signed_dist_cloud(cloud_poly, input_tomo.shape)

    # Store the results
    if verbose:
        print '\tStoring the result...'
    if cloud_poly is not None:
        disperse_io.save_vtp(cloud_poly, output_dir + '/' + stem + '_cloud.vtp')
    if dist_tomo is not None:
        disperse_io.save_numpy(dist_tomo, output_dir + '/' + stem + '_dist' + fmt)

################# Main call

def main(argv):

    try:
        opts, args = getopt.getopt(argv, "hvdci:o:p:f:")
    except getopt.GetoptError:
        print usage_msg
        sys.exit(2)

    input_file = ''
    output_dir = ''
    cloud = False
    dist = False
    p_ratio = 1
    fmt = '.mrc'
    verbose = False
    for opt, arg in opts:
        if opt == '-h':
            print usage_msg
            print help_msg
            sys.exit()
        elif opt == "-i":
            input_file = arg
        elif opt == "-o":
            output_dir = arg
        elif opt == "-d":
            dist = True
        elif opt == "-c":
            cloud = True
        elif opt == "-p":
            p_ratio = int(arg)
        elif opt == "-f":
            fmt = '.' + arg
        elif opt == "-v":
            verbose = True

    if (input_file == '') or (output_dir == ''):
        print usage_msg
        sys.exit(2)
    else:
        # Print init message
        if verbose:
            print 'Running tool for generating a surface from a membrane segmentation.'
            print '\tAuthor: ' + __author__
            print '\tDate: ' + time.strftime("%c") + '\n'
            print 'Options:'
            print '\tInput file: ' + input_file
            print '\tOutput directory: ' + output_dir
            print '\tPurge ratio: ' + str(p_ratio)
            if cloud:
                print '\tCloud of points generation activated.'
            if dist:
                print '\tSigned distance generation activated.'
                print '\tOutput segmentation format ' + fmt
            print ''

        # Do the job
        if verbose:
            print 'Starting...'
        do_surf_recon(input_file, output_dir, p_ratio, cloud,  dist, fmt, verbose)

        if verbose:
            print cmd_name + ' successfully executed.'

if __name__ == "__main__":
    main(sys.argv[1:])
