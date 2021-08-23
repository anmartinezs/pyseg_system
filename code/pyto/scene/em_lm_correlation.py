"""
Contains class EmLmCorrelation for the correlation between EM and LM

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id$
"""
from __future__ import unicode_literals
from builtins import object

__version__ = "$Revision$"


import logging
from copy import copy, deepcopy
import numpy
import scipy

from ..geometry.affine_2d import Affine2D 


class EmLmCorrelation(object):
    """
    """

    ###############################################################
    #
    # Initialization
    #
    ##############################################################

    def __init__(
        self, mode=None, lmMarkers=None, lmMarkersGl=None, lmMarkersD=None, 
        overviewMarkers=None, overviewMarkersGl=None, overviewMarkersD=None, 
        overviewDetail=None, searchDetail=None, 
        searchMain=None, overviewCenter=None):
        """
        Initializes attributes.

        Arguments:
          - mode: establishing LM to EM overview transformation mode: 
          'move search' or 'move overview'
          - lmMarkers, overviewMarkers (ndarrays of shape (n_markers, 2)): 
          positions of markers in LM and EM overview
          - lmMarkersGl, lmMarkersD, overviewMarkersGl, overviewMarkersD
          (ndarrays of shape (n_markers, 2)): positions of markers in LM and 
          EM overview for the case where the Gl and the translation parts of
          the transformation are calculated separately
          - overviewDetails, searchDetails: positions of detail in EM overview
          and search modes
          - searchMain: search coordinates of the overview image used for 
          LM - EM overview correlation. Needed for 'move overview' mode only.
          - overviewCenter: coordinates of a correlated spot at an image 
          taken at the overview magnification and at the corresponding 
          correlated stage position (typically the center of the 
          overview image). Needed for 'move overview' mode only.
        """
        self.mode = mode
        self.lmMarkers = lmMarkers
        self.lmMarkersGl = lmMarkersGl
        self.lmMarkersD = lmMarkersD
        self.overviewMarkers = overviewMarkers
        self.overviewMarkersGl = overviewMarkersGl
        self.overviewMarkersD = overviewMarkersD
        self.overviewDetail = overviewDetail
        self.searchDetail = searchDetail
        self.searchMain = searchMain
        self.overviewCenter = overviewCenter

    def establish(self, lm2overviewType='gl', overview2searchType='rs',
                  order='qpsm'):
        """
        Establishes transformation from LM to EM search mode.

        Reguires marker coordinates in LM and EM overview, coordinates of
        detail in EM overview and search, mode and index of the main detail (if
        mode is 'move overview' to be set as attributes, by __init__ for
        example.

        In the usual (direct) way the lm to overview transformation is 
        calculated directly from self.lmMarkers and self.overviewMarkers
        attributes. 

        Alternatively, if self.lmMarkersGl, self.lmMarkersD, overviewMarkersGl 
        and self.overviewMarkersD are specified, first the gl part of the 
        transformation is calculated from the above attributes ending in 'Gl'
        and then the translation is calculated from the attributes ending in
        'd'. The final transformation is the a composition of these two
        transformation. In this case error can not be calculated directly (as 
        in the direct case) so the individual transformation errors are saved 
        as errorGL and errorD attributes of the final transformation 
        (self.lm2overview). The rms error is calculated for the final
        transformation.

        Either self.lmMarkers and self.overviewMarkers or self.lmMarkersGl,
        self.lmMarkersD, overviewMarkersGl and self.overviewMarkersD have to
        be specified. 

        Argument:
          - lm2overviewType: type of transformation from LM to EM overview, 
          'gl' for general linear or 'rs' for rotation and sigle scaling
          - overview2searchType: type of transformation from EM overview to
          search, 'gl' for general linear or 'rs' for rotation and sigle scaling

        Sets:
          - lm2overview: (Affine2D) LM to EM overview transformation
          - overview2search: (Affine2D) EM overview to search transformation
          - lm2search: (Affine2D_ LM to EM search transformation
          - order: decomposition order, 'qpsm', 'psmq' or 'uqv' (see 
          ..geometry.Affine.decompose for details)
        """

        # establish LM to overview EM transformation
        if (self.lmMarkersGl is not None) and (self.lmMarkersD is not None):

            # find gl
            lm2overviewGl = Affine2D.find(
                x=self.lmMarkersGl, y=self.overviewMarkersGl, 
                type_=lm2overviewType)
            lm2overviewGl.d = numpy.zeros(shape=2)

            # find translation
            transformed_lmmd = lm2overviewGl.transform(x=self.lmMarkersD)
            lm2overviewD = Affine2D.findTranslation(
                x=transformed_lmmd, y=self.overviewMarkersD)

            # compose
            lm2overview = Affine2D.compose(t_1=lm2overviewD, t_2=lm2overviewGl)
            lm2overview.decompose(order=order)
 
            # save individual error
            lm2overview.errorGl = lm2overviewGl.error
            lm2overview.errorD = lm2overviewD.error

        elif self.lmMarkers is not None:

            # find affine directly
            lm2overview = Affine2D.find(
                x=self.lmMarkers, y=self.overviewMarkers, type_=lm2overviewType)

        else:
            raise ValueError("Either lmMarkers or lmMarkersGl and lmMarkersD" 
                             + " attributes have to be specified")

        # establish overview EM to search EM transformation
        if self.mode == 'move search':

            # move search mode
            overview2search = Affine2D.find(
                x=self.overviewDetail, y=self.searchDetail, 
                type_=overview2searchType)

        elif self.mode == 'move overview':

            # make transformation for the detail 
            overview2search_detail = Affine2D.find(
                x=self.overviewDetail, y=self.searchDetail, 
                type_=overview2searchType)

            # make transformation that leads to overviewCenter
            search_adjust = Affine2D(gl=overview2search_detail.gl, 
                                     d=self.searchMain)
            overview2search_d = search_adjust.transform(self.overviewCenter)
            overview2search = Affine2D(gl=-overview2search_detail.gl,
                                       d=overview2search_d)
            overview2search.error = overview2search_detail.error

            #overview2search.d = \
            #    2 * self.stageOverview - overview2search.d
            #overview2search.phi = numpy.pi + overview2search.phi

        else:
            raise ValueError("Attribute 'mode' (", self.mode, ") can be",
                             "'move search' or 'move overview'.")

        # decompose overview to search
        overview2search.decompose(order=order)

        # make final transformation
        lm2search = Affine2D.compose(overview2search, lm2overview)
        lm2search.decompose(order=order) 

        # save transformations
        self.lm2overview = lm2overview
        self.overview2search = overview2search
        if self.mode == 'move overview':
            self._overview2search_detail = overview2search_detail
        self.lm2search = lm2search

    def decompose(self, order):
        """
        Decomposes all three transformations.

        Arguments:
          - order: decomposition order, 'qpsm', 'psmq' or 'uqv' (see 
          ..geometry.Affine.decompose for details)
        """

        self.lm2overview.decompose(order=order)
        self.overview2search.decompose(order=order)
        self.lm2search.decompose(order=order)


    ###############################################################
    #
    # IO
    #
    ##############################################################

    def readPositions(self, specs, format='imagej', xyColumns=None):
        """
        Reads positions (specified by arg specs) and sets them as attributes 
        of this instance.

        Names of the attributes that are set by this method are specified as
        keys of arg specs (dictionary). Each value of specs is a tuple that
        contains the file name and the list of rows (line numbers) that 
        contains coordinates that should be read. 

        Note that rows are counted starting with 0, but disregarding comment 
        lines. If rows is None or it doesn't exist all rows are read.

        For example, if:

          specs = {'lmMarkers' : ('marker_file.txt', [7, 8]),
                   'overviewMarkers' : ('marker_file.txt', [1, 2])}

        and marker_file.txt (generated by ImageJ):

                Label	        X	Y	Z	Value	
          1	129+7_220_kink	724	647	0	32774	
          2	129+7_220_kink	828	608	0	32776	
          3	129+7_220_kink	776	804	0	32770	
          4	129+7_220_kink	256	839	0	32774	
          5	129+7_220_kink	36	351	0	32771	
          7	129+7_220_kink	190	460	0	32777	
          8	Stack:8	1.673	1.907	7	1399	
          9	Stack:8	1.680	1.787	7	993	
          10	Stack:8	1.867	1.913	7	1130	

          self.readPositions(specs=specs, format='imagej', xyColumns=[2, 3])

        will set the following attributes:

          self.lmMarkers = numpy.array([[1.680	1.787],
                                        [1.867	1.913]])
          self.overviewMarkers = numpy.array([[828	608],
                                              [776	804]])

        Arguments:
          - specs: (dict) specifies the file location of positions to be read
          and the names of attributes where the read positions should be stored
          - format: format of file(s) specified in arg specs; currently only
          'imagej' (for files generated by ImageJ
          - xyColumns: (list of length 2), columns containing x and y 
          coordinates (numbering starts with 0) 
        """

        # set file parsing parameters
        if format == 'imagej':

            # ImageJ definitions
            comments = '#'
            #comments = ' '
            skiprows = 1    # probably redundant with comments
            delimiter = '\t'

        else:
            raise ValueError('Argument format: ', format, 'was not understood.',
                             " Currently implemented format is 'imagej'.")

        # read positions and save as attributes
        for name, val in list(specs.items()):

            # read columns from file
            try:
                file_ = val[0]
            except TypeError:
                continue
            table = numpy.loadtxt(file_, comments=comments, delimiter=delimiter,
                                  skiprows=skiprows, usecols=xyColumns)

            # pick specified rows
            try:
                rows = val[1]
                if rows is not None:
                    pos = table[rows]
                else:
                    pos = table
            except IndexError:
                #pos = table
                raise

            # save as an attribute
            setattr(self, name, pos) 

 

            
