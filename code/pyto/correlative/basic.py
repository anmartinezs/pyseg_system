"""
Contains class Basic for the basic correlative approach between two systems.

The two systems need to have the same dimensionality,

Classes for specific approaches should inherit from this class.

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id: basic.py 1258 2015-11-30 09:20:16Z vladan $
"""

__version__ = "$Revision: 1258 $"


import warnings
from copy import copy, deepcopy
import numpy as np
import scipy as sp

from ..geometry.affine import Affine
from ..geometry.affine_2d import Affine2D
from ..geometry.affine_3d import Affine3D

class Basic(object):
    """
    Two same dimensionality systems correlative approach.

    Usage:
    
      corr = Basic()
      corr.establish(marker_points ...)
      corr.correlate(target_points)
      corr.targets_1, corr_targets_2   # display correlated points
      corr.transf_1_to_2.rmsError      # display corelation error

    Main methods:
      - establish() establishes correlation (transformation) between two systems
      - decompose() decomposes the transformation
      - correlate() correlates points from one system to another (once the
      correlation is established)

    Attributes and associated methods:
      - markers_1, markers_2: (ndarray n_markers x n_dim) markers for 
        systems 1 and 2, respectively
      - get/setMarkers(): markers
      - transf_1_to_2, transf_2_to_1: (geometry.Affine, or one of its
        subclasses) coordinate transformation from system 1 to 2 
      - get/setTransformation(): transformation between two systems
      - targets_1, targets_2: (ndarray n_targets x n_dim) target points
      - correlated_1_to_2, correlated_2_to_1: (ndarray n_targets x n_dim) 
        correlated target points from system 1 to 2 and from target 2 to 1  
      - get/setTargets(): targets 

    Usage examples are given in:
      - examples/correlation_simple.py: coordinates specified directly
      - examples/correlation_simple.py: coordinates specified in a file
    """

    ###############################################################
    #
    # Initialization and data access
    #
    ##############################################################

    def __init__(self):
        """
        """
        pass

    ###############################################################
    #
    # Markers
    #
    def getMarkers(self, system, gl=False):
        """
        Returns markers. If markers not found returns None. 

        Reads attribute self._markers_(system), where system is given in the 
        arguments.
        """
        
        # make attribute name for this case 
        name = '_markers_' + str(system)
        if gl:
            name = name + '_gl'

        # get attribute value
        try:
            result = getattr(self, name)
        except AttributeError:
            result = None

        return result

    def setMarkers(self, value, system, gl=False):
        """
        Sets markers to (arg) value.

        Sets attribute self._markers_(system), where system is given in the 
        arguments.
        """
        
        # make attribute name for this case 
        name = '_markers_' + str(system)
        if gl:
            name = name + '_gl'

        # set attribute value
        value_arr = np.asarray(value)
        result = setattr(self, name, value_arr)

        return result

    def getMarkers1(self):
        """
        Get markers for system 1.
        """
        return self.getMarkers(system=1)
    def setMarkers1(self, value):
        """
        Set markers for system 1.
        """
        self.setMarkers(value=value, system=1)
    markers_1 = property(fget=getMarkers1, fset=setMarkers1, 
                         doc='Markers for system 1.')

    def getMarkers1Gl(self):
        """
        Get Gl markers for system 1.
        """
        return self.getMarkers(system=1, gl=True)
    def setMarkers1Gl(self, value):
        """
        Set Gl markers for system 1.
        """
        self.setMarkers(value=value, system=1)
    markers_1_gl = property(fget=getMarkers1Gl, fset=setMarkers1Gl,
                         doc='Gl markers for system 1.')

    def getMarkers2(self):
        """
        Get markers for system 2.
        """
        return self.getMarkers(system=2)
    def setMarkers2(self, value):
        """
        Set markers for system 2.
        """
        self.setMarkers(value=value, system=2)
    markers_2 = property(fget=getMarkers2, fset=setMarkers2, 
                         doc='Markers for system 2.')

    def getMarkers2Gl(self):
        """
        Get Gl markers for system 2.
        """
        return self.getMarkers(system=2, gl=True)
    def setMarkers2Gl(self, value):
        """
        Set Gl markers for system 2.
        """
        self.setMarkers(value=value, system=2)
    markers_2_gl = property(fget=getMarkers2Gl, fset=setMarkers2Gl,
                         doc='Gl markers for system 2.')

    ###############################################################
    #
    # Transformations
    #
    def getTransformation(self, from_, to):
        """
        Returns transformation from syetem given by arg 'from_' to system 'to',
        or None if not found.

        Reads attribute self._transf_(from_)_to_(to), where from_ and to_ are
        given in the arguments.
        """

        name = '_markers_' + str(from_) + '_to_' + str(to)
        try:
            return getattr(self, name)
        except AttributeError:
            return None

    def setTransformation(self, value, from_, to):
        """
        Sets transformation given by arg 'value' from system given by arg 
        'from_' to system 'to'.

        Sets attribute self._transf_(from_)_to_(to), where from_ and to_ are
        given in the arguments.
        """

        name = '_markers_' + str(from_) + '_to_' + str(to)
        setattr(self, name, value)

    @property
    def transf_1_to_2(self):
        """
        Transformation from system 1 to 2
        """
        return self.getTransformation(from_=1, to=2)
    @transf_1_to_2.setter
    def transf_1_to_2(self, value):
        self.setTransformation(value=value, from_=1, to=2)

    @property
    def transf_2_to_1(self):
        """
        Transformation from system 2 to 1
        """
        return self.getTransformation(from_=2, to=1)
    @transf_2_to_1.setter
    def transf_2_to_1(self, value):
        self.setTransformation(value=value, from_=2, to=1)

    ###############################################################
    #
    # Target points
    #
    def getTargets(self, from_, to=None, gl=False):
        """
        If arg to is None, returns target point coordinates of the system from_.
        If args from_ and to differ, returns points correlated from targets of
        system (arg) from_ to system (arg) to.

        In any case returns None if not found.

        Reads attribute self._transf_(from_)_to_(to), where from_ and to_ are
        given in the arguments.
        """
            
        if (to is None) or (to == from_):

            # make name for targets
            name = '_targets_' + str(from_)
            if gl:
                name = name + '_gl'

            # get attribute value
            try:
                result = getattr(self, name)
            except AttributeError:
                result = None
            return result

        else:

            # make name for correlated
            name = '_correlated_' + str(from_) + '_to_' + str(to)

            # set attribute
            try:
                result = getattr(self, name)
            except AttributeError:
                result = None
            return result

    def setTargets(self, value, from_, to=None, gl=False):
        """
        Sets target point coordinates.
        If arg to is None, sets target point coordinates of the system from_
        to (arg) value. If args from_ and to differ, sets points correlated 
        from targets of system (arg) from_ to system (arg) to.

        Sets attribute self._transf_(from_)_to_(to), where from_ and to_ are
        given in the arguments.
        """

        if (to is None) or (to == from_):

            # make name for targets
            name = '_targets_' + str(from_)
            if gl:
                name = name + '_gl'

            # set attribute
            setattr(self, name, value)

        else:

            # make name for correlated
            name = '_correlated_' + str(from_) + '_to_' + str(to)

            # set attribute
            setattr(self, name, value)

    @property
    def targets_1(self):
        """
        Targets specified in the system 1
        """
        return self.getTargets(from_=1)
    @targets_1.setter
    def targets_1(self, value):
        self.setTargets(value=value, from_=1)

    @property
    def targets_2(self):
        """
        Targets specified in the system 2
        """
        return self.getTargets(from_=2)
    @targets_2.setter
    def targets_2(self, value):
        self.setTargets(value=value, from_=2)

    @property
    def correlated_1_to_2(self):
        """
        Correlated from targets specified in the system 1 to system 2.
        """
        return self.getTargets(from_=1, to=2)
    @correlated_1_to_2.setter
    def correlated_1_to_2(self, value):
        self.setTargets(value=value, from_=1, to=2)

    @property
    def correlated_2_to_1(self):
        """
        Correlated from targets specified in the system 2 to system 1.
        """
        return self.getTargets(from_=2, to=1)
    @correlated_2_to_1.setter
    def correlated_2_to_1(self, value):
        self.setTargets(value=value, from_=2, to=1)

    ###############################################################
    #
    # Transformation error
    #

    def getRmsError(self, from_, to):
        """
        Returns error of the transfromation specified by arguments. 

        Returns (error, estimated): 
          - error: rms error of the transformation (if transformation.rmsError 
          exists), or the estimated rms error (transformation,rmsErrorEst)
          - flag indicating if the error is estimated
        """

        try:
            err = self.getTransformation(from_=from_, to=to).rmsError
            estimated = False
        except AttributeError:
            err = self.getTransformation(from_=from_, to=to).rmsErrorEst
            estimated = True
        
        return (err, estimated)
        
    @property
    def error_1_to_2(self):
        """
        Transformation (1 to 2) rms error and a flag indicating if the error is 
        estimated (and not exact).
        """
        return self.getRmsError(from_=1, to=2)
    @property
    def error_2_to_1(self):
        """
        Transformation (2 to 1) rms error and a flag indicating if the error is 
        estimated (and not exact).
        """
        return self.getRmsError(from_=2, to=1)


    ###############################################################
    #
    # Establishment of correlation and transformations
    #
    ##############################################################

    def establish(
        self, points=None, markers_1=None, markers_2=None, 
        markers_1_gl=None, markers_2_gl=None, type_='gl', order=None, 
        format_='imagej', columns=None, comments=' ', skiprows=1, 
        delimiter=' ', indexing=1):
        """
        Establishes correlation from markers_1 to markers_2, that is finds
        the affine transformation that minimizes mean square root error 
        (see class ..geometry.Affine for details about the transformation). 
        Also finds the inverse transformation.

        Marker points can be specified in by any of the following:
          - arg points
          - args markers_1 and markers_2
          - previously set attributes self.markers_1 and self.markers_2. 
        The markers are searched in the above order. If markers
        markers_[1,2]_gl are specified, they are used only for the Gl part
        of the transformation (again, see ..geometry.Affine). 

        If markers are specified by arg points (dict) it has to have 
        'markers_1' and 'markers_2' for names. See readPositions() argument
        points for details. In this case agrs format_, columns and indexing 
        need to be specified.

        Usual form of points is:

        points = {
          'markers_1' : ('markers1_file_name', markers1_row_indices),
          'markers_2' : ('markers2_file_name', markers2_row_indices),
          'targets_1' : ('targets1_file_name', targets1_row_indices),
          'targets_2' : ('targets2_file_name', targets2_row_indices)
          ... }

        where *_row_indices is an 1D ndarray.

        The transformation is established using ..geometry.Affine2D / Affine3D
        / Affine .find() method for markers in 2D / 3D / other dimensions. See  
        these for details.

        If args markers_1_gs and markers_2_gs are given, these are used for
        the Gl part of the transformation, while markers_1 and markers_2 are
        used for the determination of d, using ..geometry.Affine.findTwoSteps(),
        see that method for details.

         Arguments:
          - points: dictionary that specifies files and rows where markers_1
          and markers_2 are specified. 
          - markers_1 and markers_2: (n_markers x ndim ndarray) markers
          - type_: type of the optimization, 'gl' to find Gl transformation
          that optimizes the square error, or 'rs' to find the best rotation 
          and one scale (currently implemented for 2D transformations only)
          In any case the translation is also found.
          - order: decomposition (of gl) order 'qpsm' (same as 'qr'), 'psmq' 
          (same as 'rq'), or 'usv'
          - format_: format of the files specified in arg points
          - columns: (list or ndarray) columns that contain coordinates 
          [x, y, ...]
          - comments: indicate start of a comment
          - skiprows: number of top rows that are ignored
          - delimiter: separates fields (columns)
          - indexing: if 1 rows and columns are indexed from 1 up, otherwise 
          they are indexed from 0 up

        Sets attributes in this instance:
          - marker_1, marker_2: (n_markers x ndim ndarray) markers
          - marker_1_gl, marker_2_gl: (n_markers x ndim ndarray) markers used
          to find the Gl part of the transformation
          - all other points specified by arg points are saved as attributes of
          the same names as the corresponding b=names of (dict) points
          - transf_1_to_2: (..Geometry.Affine) transformation between the two 
          systems
          - transf_2_to_1: (..Geometry.Affine) the inverse transformation
        """

        # get markers directly from arguments (overrides previously set)
        if markers_1 is not None:
            self.markers_1 = markers_1
        if markers_1_gl is not None:
            self.markers_1 = markers_1_gl
        if markers_2 is not None:
            self.markers_2 = markers_2
        if markers_2_gl is not None:
            self.markers_2 = markers_2_gl

        # read markers from file(s) (overrides all from before)
        if points is not None:
            self.readPositions(
                points=points, format_=format_, columns=columns, 
                comments=comments, skiprows=skiprows, 
                delimiter=delimiter, indexing=indexing)

        # sanity check
        if (self.markers_1 is None) or (self.markers_2 is None):
            raise ValueError(
                "Markers need to be specified either by argument points, "
                + "arguments markers_1 and markers_2, or "
                + " attributes self.markers_1 and self.markers_2.")

        # figure out if two step correlation procedure
        if self.markers_1_gl is not None:
            two_step = True
        else:
            two_step = False

        # establish correlation, depending on dimensionality and if two-step
        ndim = self.markers_1.shape[1]
        if ndim == 2:
            if two_step:
                transf_1_to_2 = Affine2D.findTwoStep(
                    x=self.markers_1, y=self.markers_2, x_gl=self.markers_1_gl,
                    y_gl=self.markers_2_gl, type_=type_)
            else:
                transf_1_to_2 = Affine2D.find(
                    x=self.markers_1, y=self.markers_2, type_=type_)

        elif ndim == 3:
            if two_step:
                transf_1_to_2 = Affine3D.findTwoStep(
                    x=self.markers_1, y=self.markers_2, x_gl=self.markers_1_gl,
                    y_gl=self.markers_2_gl, type_=type_)
            else:
                transf_1_to_2 = Affine3D.find(
                    x=self.markers_1, y=self.markers_2, type_=type_)

        else:
            if two_step:
                transf_1_to_2 = Affine.findTwoStep(
                    x=self.markers_1, y=self.markers_2, x_gl=self.markers_1_gl,
                    y_gl=self.markers_2_gl, type_=type_)
            else:
                transf_1_to_2 = Affine.find(
                    x=self.markers_1, y=self.markers_2,  type_=type_)
        self.transf_1_to_2 = transf_1_to_2

        # find inverse
        self.transf_2_to_1 = self.transf_1_to_2.inverse()

        # desompose 
        if order is not None:
            self.decompose(order=order)

    def decompose(self, order):
        """
        Decomposes the transformations from 1 to 2 and the inverese. Uses
        ..geometry.Affine.decompose().
        """

        self.transf_1_to_2.decompose(order=order)
        self.transf_2_to_1.decompose(order=order)

    def correlate(self, points=None, targets_1=None, targets_2=None, 
                  format_='imagej', columns=None, indexing=1):
        """
        Correlates target points form one system to another. The transformatin 
        between the two systems has to be established already.

        Target points have to be specified by arg points, args targets_1 and 
        targets_2, or (previously set) attributes self.targets_1 and 
        self.targets_2. The targets are searched in the above order. 

        If targets are specified by arg points (dict) it has to have 
        'targets_1' and 'targets_2' for names. See readPositions() argument
        points for details. In this case agrs format_, columns and indexing 
        need to be specified.

        Arguments:
          - points: dictionary that specifies files and rows where targets_1
          and targets_2 are specified. 
          - targets_1 and targets_2: (n_targets x ndim ndarray) targets
          - format_: format of the files referenced in arg points
          - columns: (list or ndarray) columns that contain  coordinates 
          [x, y, ...] in files specified by arg points
          - indexing: if 1 rows and columns of files specified by arg points 
          are indexed from 1 up, otherwise they are indexed from 0 up. 

        Sets attributes in this instance:
          - target_1, target_2: (n_targets x ndim ndarray) target points
          - correlated_1_to_2, correlated_2_to_1: (n_targets x ndim ndarray) 
          points correlated from target points of one system to another system. 
        """

        # get targets directly from arguments (overrides previously set values)
        if (targets_1 is not None):
            self.targets_1 = targets_1
        if (targets_2 is not None):
            self.targets_2 = targets_2

        # get target coordinates (overrides previously set values)
        if points is not None:
            self.readPositions(points=points, format_=format_, columns=columns,
                               indexing=indexing)

        # sanity check
        if (self.targets_1 is None) and (self.targets_2 is None):
            raise ValueError(
                "Targets for at least one of the systems "
                + "need to be specified either by argument points, "
                + "arguments targets_1 and targets_2, or "
                + " attributes self.targets_1 and self.targets_2.")

        # correlate
        self.correlated_1_to_2 = self.transf_1_to_2.transform(x=self.targets_1)
        self.correlated_2_to_1 = self.transf_2_to_1.transform(x=self.targets_2)


    ###############################################################
    #
    # IO
    #
    ##############################################################

    def readPositions(
            self, points, format_='imagej', columns=None, comments=' ', 
            skiprows=1, delimiter=' ', indexing=1):
        """
        Reads positions of points specified by arg points from multiple files.

        Points is a dictionary where each name is the name of one group of 
        points that have a specific function. For example, 'markers_1' may 
        denote marker points of the first system, while 'targets_2' target 
        points (those that need to be correlated) of the second system. 

        Each value of points is a tuple of length 2, where the element 0 is the
        name of the file and the element 1 is an array of indices specifying
        rows that contain coordinates of points. Alternatively, if the element 1
        is None or it doesn't exist all rows are read.

        Usual form of points is:

        points = {
          'markers_1' : ('markers1_file_name', markers1_row_indices),
          'markers_2' : ('markers2_file_name', markers2_row_indices),
          'targets_1' : ('targets1_file_name', targets1_row_indices),
          'targets_2' : ('targets2_file_name', targets2_row_indices)
          ... }

        where *_row_indices is an 1D ndarray.

        If arg format_ is imagej, the first row as well as rows starting with 
        ' ' are ignored and the field are separated by '\t'. In this case args 
        comments, skiprows and delimiter are ignored. If arg format_ is 
        anything else, the the values of args comments and skiprows args 
        determine which rows contain data.

        In both cases arg columns specifies which columns contain coordinates.
        Args columns, comments and columns are passed directly to 
        numpy.loadtxt().

        If arg indexing is 1, rows that are read (not ignored) are indexed 
        from 1 up (just like the index which is shown in the first column in 
        the imagej format). The columns are also indexed from 1 up. Otherwise, 
        if arg indexing is 0 (or any other number), both rows and columns
        are indexed from 0.

        The point coordinates read are saved as attributes of the current 
        instance that have the same name as the corresponding group of points.
        In the above examples these would be self.markers_1 and 
        self.target_2.

        Arguments:
          - points: (dict) specifies files where data points are stored and
          the positions within these files
          - format_: format of the files
          - columns: (list or ndarray) columns that contain coordinates 
          [x, y, ...]
          - comments: indicate start of a comment
          - skiprows: number of top rows that are ignored
          - delimiter: separates fields (columns)
          - indexing: if 1 rows and columns are indexed from 1 up, otherwise 
          they are indexed from 0 up
        """

        # set file parsing parameters
        if format_ is 'imagej':

            # ImageJ definitions, ignore args delimiter, comments and skiprows
            comments = ' '
            skiprows = 1    # probably redundant with comments
            delimiter = '\t'

        else:

            # use args delimiter, comments and skiprows
            pass

        # adjust columns to be indexed from 0
        if (indexing == 1) and (columns is not None):
            columns = np.array(columns) - 1

        # read positions and save as attributes
        for name, val in points.items():

            # read columns from file
            try:
                file_ = val[0]
            except TypeError:
                continue
            table = np.loadtxt(
                file_, comments=comments, skiprows=skiprows, 
                delimiter=delimiter, usecols=columns)

            # pick specified rows
            try:

                # adjust rows to be indexed from 0
                rows = val[1]
                if indexing == 1:
                    rows = np.array(rows) - 1

                if rows is not None:
                    pos = table[rows]
                else:
                    pos = table
            except IndexError:
                raise

            # save as an attribute
            setattr(self, name, pos) 

 
