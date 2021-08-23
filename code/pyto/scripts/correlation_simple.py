#!/usr/bin/env python
"""

Establishes a correlation between two arbitrary systems and correlates 
positions of objects of interest (targets) between these two systems. 

The correlation procedure used in this script is direct, meaning there are no 
intermediary systems. Also, both systems have to have the same dimensionality.

Conceptually, the correlation procedure consists of two steps:
  1) Establish a correlation between the two systems (initial and final), 
that is find a coordinate transformation between the two systems.
  2) Correlate positions of obects of interest (targets) from one system to 
the other.

Marker points are needed to determine the transformation (step 1). That is, 
one needs to identify one set of features that are visible in both systems.

Requires the following coordinates:
  - marker coordinates in the initial system
  - corresponding marker coordinates in the final system
  - (optional) target coordinates the initial and / or final systems

This is a very simple script where coordinates of markers and targets are 
entered directly. The calculated transformation parameters and the coordinates 
of the correlated target points are displayed on the standard output. 

Standard usage:

1) Edit this file to enter the desired parameters (coordinates of markers 
and objects of interest, and the transformation type)
2) Import and run this file (in IPython or another python shell):
  >>> import correlation_simple
  >>> correlation_simple.main()

Advanced usage:

1) Edit this file to enter the desired parameters (coordinates of markers 
and objects of interest, and the transformation type)
2) Import this file and execute commands from main() one by 
one. Check values of variables as needed:
  >>> import correlation_simple
  >>> from correlation_simple import *
  >>> corr = Basic()
  >>> corr.establish(
        markers_1=markers_initial, markers_2=markers_final, 
        type_=transform_type)
  >>> corr.transf_1_to_2.phiDeg
  >>> corr.transf_1_to_2.gl
      ...
Other attributes of corr are listed in the documentation for 
pyto.correlative.Basic. If the initial and final systems are 2D, object 
corr.transf_1_to_2 is an instance of pyto.geometry.Affine2D, if they are 3D
corr.transf_1_to_2 is an instance of pyto.geometry.AffinewD and in other cases
of pyto.geometry.Affine. Please check docs for these classes for other 
attributes. 

If you use this script, please consider citing: 

  Fukuda, Y., N. Schrod, M. Schaffer, L. R. Feng, W. Baumeister, and V. Lucic, 
  2014. Coordinate transformation based cryo-correlative methods for electron 
  tomography and focused ion beam milling. Ultramicroscopy 143:15– 23. 


# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id: correlation_simple.py 1237 2015-10-14 12:18:37Z vladan $
"""
from __future__ import unicode_literals
from __future__ import print_function

__version__ = "$Revision: 1237 $"

import pyto
from pyto.correlative.basic import Basic


##################################################################
#
# Parameters
#
#################################################################

##################################################################
#
# Coordinates entered directly
#

# marker coordinates in the initial system
markers_initial = [[1, 1], 
                   [3, 1],
                   [1, 2]]

# marker coordinates in the final system
markers_final = [[5, 6], 
                 [5, 10],
                 [3, 6]]

# coordinates of objects of interest in the initial system
target_initial = [[2, 1],
                  [1, 3]]

# coordinates of other obects of interest in the final system 
target_final = [[5, 4],
                [4, 6]]

# transformation type: 'gl' for general linear or 'rs' for rigid
transform_type = 'gl'


#####################################################################
#
# Main function
#
#####################################################################

def main():
    """
    Main function
    """

    # establish correlation
    corr = Basic()
    corr.establish(
        markers_1=markers_initial, markers_2=markers_final, 
        type_=transform_type)

    # print transformation parameters
    print("Transformation parameters: ")
    print("Rotation angle: ", corr.transf_1_to_2.phiDeg)
    print("Scales: ", corr.transf_1_to_2.scale)
    print("Parity: ", corr.transf_1_to_2.parity)
    print("Shear: ", corr.transf_1_to_2.shear)
    print("Translation: ", corr.transf_1_to_2.d)
    print("RME error: ", corr.transf_1_to_2.rmsError)
    print("Individual marker errors: ")
    print(corr.transf_1_to_2.error)

    corr.correlate(targets_1=target_initial)
    print(" ")
    print("System 1 targets correlated to system 2: ")
    print(corr.correlated_1_to_2)

    corr.correlate(targets_2=target_final)
    print(" ")
    print("System 2 targets correlated to system 1: ")
    print(corr.correlated_2_to_1)



# run if standalone
if __name__ == '__main__':
    main()

