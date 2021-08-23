"""
Contains class Cleft for the analysis of a cleft-like region (a region
between two roughly parallel boundaries).

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id$
"""
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
from builtins import zip
from builtins import str
from past.utils import old_div

__version__ = "$Revision$"


import logging
#from copy import copy, deepcopy
import numpy
import scipy

from pyto.geometry.vector import Vector
from pyto.geometry.affine_3d import Affine3D
from pyto.geometry.coordinates import Coordinates
from .segment import Segment
from .grey import Grey


class Cleft(Segment):
    """
    Important attributes:
      - data: (ndarray) labeled cleft and boundary regions
      - cleftId/bound1Id/bound2Id: ids of cleft / boundary 1 / boundary 2,
      each saved as an ndarray
      - ids: all of the above ids

    Methods:
      - makeLayers(): makes layers in the cleft and possibly over boundaries
      - getWidth(): calculates width and the orientation of the cleft
      - getBoundaryDistances(): for each boundary edge voxel, finds the position
      and the distance to closest voxel on the other boundary
      - parametrizeLabels(): puts a coordinate system on labels
      
    """

    ###############################################################
    #
    # Initialization
    #
    ##############################################################

    def __init__(self, data, cleftId=[], bound1Id=[], bound2Id=[], copy=True, 
                 clean=False):
        """
        Sets attributes from arguments.

        Each of the id args (cleftId, bound1Id, bound2Id) can be a single int
        or a list (array) in which cases a (boundary or cleft) is formed form 
        all specified ids.

        Arguments:
          - data: ndarray with labeled cleft and boundary regions
          - bound1Id, bound2Id, cleftId: ids of boundaries 1, 2 and cleft
          - copy: flag indicating if data array is copied
          - clean: flag indicating if segmentss other than those specified
          by boundary and cleft ids are removed from tha data array
        """

        # set ids
        self.setCleftIds(cleftId=cleftId, bound1Id=bound1Id, bound2Id=bound2Id)
        all_ids = numpy.concatenate([self.cleftId, self.bound1Id, 
                                     self.bound2Id])

        # call super to set data and ids
        super(Cleft, self).__init__(data=data, copy=copy, ids=all_ids, 
                                    clean=clean)

    def setCleftIds(self, cleftId, bound1Id, bound2Id):
        """
        Sets boundary, cleft and all ids as attributes, in addition to 
        whatever else super does.
        """

        # convert args bound1Id, bound2Id and cleftId to ndarrays and save 
        # as attributes
        all_ids = []
        for ids in [cleftId, bound1Id, bound2Id]: 
            if isinstance(ids, int):
                all_ids.append(numpy.array([ids]))
            else:
                all_ids.append(numpy.asarray(ids))
        self.cleftId, self.bound1Id, self.bound2Id = all_ids

        # set self.ids and related
        super(Cleft, self).setIds(ids=all_ids)


    ###############################################################
    #
    # Geometry
    #
    ##############################################################

    def getWidth(self, mode='median', toBoundary=0, maxDistance=None):
        """
        Calculates the width and the orientation of the cleft (direction of the 
        width vector).

        The width is calculated by taking into account only the regions of 
        the two boundaries that border the cleft region. First, min distance to 
        the boundary specified by toBoundary is calculated for each
        element of a border region of the other boundary. Distance between 
        boundaries is then obtained as the mean/median/min/max (depending on the
        arg mode) of the distances for each element. Finally, the cleft width 
        is distance between boundaries - 1. For example, if boundaries touch
        exch other the distance between them is 1 and the cleft width is 0.

        If toBoundary=0, distances to both boundaries are calculated and the 
        width is obtained from all those distances taken together.

        The distance vector is calculated in a similar way, except that 
        distance vectors are calculated (instead of distances alone) and 
        that the average vector is returned. I toBoundary is 1 or 2, average
        of the distance vectors to boundary 1 or 2 is returned. If toBoundary
        is 0, distance vectors are combined and then averaged. In any case,
        the distance vector is oriended from boundary 1 (self.bound1Id) to
        boundary 2 (self.bound2Id).

        If arg maxDistance is specified, only those elements of boundaries
        borders (boundary elemnts that touch the cleft) whose distance to the 
        other boundary is not larger than maxDistance are taken into account.
        Note that arg MaxDistance here has different meaning from the one in
        Segment.makeLayersBetween().

        Argument:
          - mode: cleft width calculating mode, 'mean', 'median', 'min' or 
          'max' (actually can be any appropriate numpy function)
          - toBoundary: distances calculated to that boundary
          - maxDistance: max distance between boundaries

        Returns (width, width_vector) where:
          - width: cleft width calculated between boundary edges, that is if 
          the boundaries touch each other the width is 0
          - width_vector: (..geometry.Vector) average distance vector
        """

        # get distances between boundaries
        dis_to_1, pos_to_1, pos_2, dis_to_2, pos_to_2, pos_1 = \
            self.getBoundaryDistances()

        # remove elements that are too far from the other boundary
        if maxDistance is not None:
            pos_to_1 = [single_ind_ar.compress(dis_to_1<=maxDistance) 
                        for single_ind_ar in pos_to_1]
            pos_2 = [single_ind_ar.compress(dis_to_1<=maxDistance) 
                     for single_ind_ar in pos_2]
            dis_to_1 = dis_to_1.compress(dis_to_1<=maxDistance)
            pos_to_2 = [single_ind_ar.compress(dis_to_2<=maxDistance) 
                        for single_ind_ar in pos_to_2]
            pos_1 = [single_ind_ar.compress(dis_to_2<=maxDistance) 
                     for single_ind_ar in pos_1]
            dis_to_2 = dis_to_2.compress(dis_to_2<=maxDistance)

        # calculate mean/median ... distance between boundaries
        if toBoundary == 0:
            dis_to = numpy.concatenate([dis_to_1, dis_to_2])
        elif toBoundary == 1:
            dis_to = dis_to_1
        elif toBoundary == 2:
            dis_to = dis_to_2
        else:
            raise ValueError("Argument toBoundary ", str(toBoundary), 
                             "can be 0, 1 or 2.")
        distance = getattr(numpy, mode)(dis_to)
        
        # generate distance vectors between boundary surface elements
        # and their closest elements on the other boundary
        if toBoundary == 0:
            vec1 = numpy.array(pos_2) - numpy.array(pos_to_1)
            vec2 = numpy.array(pos_to_2) - numpy.array(pos_1)
            vectors = numpy.concatenate([vec1, vec2], axis=-1)
        elif toBoundary == 1:
            vectors = numpy.array(pos_2) - numpy.array(pos_to_1) 
        elif toBoundary == 2:
            vectors = numpy.array(pos_to_2) - numpy.array(pos_1)

        # average distance vector
        width_vector = vectors.mean(axis=1)
        width_vector = Vector(data=width_vector)

        return distance-1, width_vector

    def getBoundaryDistances(self):
        """
        For each boundary element contacting cleft (boundary edge), finds the 
        position of the closest element on the other boundary and calculates 
        the distance between them. 

        Uses the smallest inset of the data array to speed up the calculations.
        However, positions calculated are adjusted for self.inset, that is
        they are given in respect to the fully expanded self.data.

        Used to provide info to higher level methods that deal with distances 
        between boundaries, such as self.getWidth().

        Return tuple containing the following ndarrays:
          - distance to boundary 1 from each element of the edge of boundary 2
          - positions of elements of boundary 1 found above
          - positions of edge elements of boundary 2 
          - distance to boundary 2 from each element of the edge of boundary 1
          - positions of elements of boundary 2 found above
          - positions of edge elements of boundary 1 
        The first (last) three are indexed in the same way, that is according 
        to the the boundary 2 (boundary 1) edge elements. Positions are 
        returned in the form numpy uses for coordinates, that is as a list
        where element i contains array of coordinates along axis i.
        """

        # inset is extended by this amount to prevent segments touching 
        # image edges
        extend_inset = 1

        # find inset that contains boundaries and cleft
        inset = self.findInset(ids=self.ids, extend=extend_inset, mode='abs')
        data_inset = self.makeInset(
            ids=self.ids, extend=extend_inset, update=False)

        # make cleft mask
        cleft_mask = numpy.zeros(shape=data_inset.shape, dtype=bool)
        for id_ in self.cleftId:
            cleft_mask[data_inset==id_] = True

        # extract boundary surfaces that border cleft
        dilated = scipy.ndimage.binary_dilation(input=cleft_mask)
        bound_surf_1 = numpy.zeros(shape=data_inset.shape, dtype=bool)
        for id_ in self.bound1Id:
            bound_surf_1[dilated & (data_inset==id_)] = True
        bound_surf_2 = numpy.zeros(shape=data_inset.shape, dtype=bool)
        for id_ in self.bound2Id:
            bound_surf_2[dilated & (data_inset==id_)] = True

        # calculate min distances of all elements of boundary 2 to boundary 1
        if (~bound_surf_1 > 0).all():  # workaround for scipy bug 1089
            raise ValueError("Can't calculate distance_function " +
                             "(no background)")
        else:
            distance_to_1, position_to_1 = \
                scipy.ndimage.distance_transform_edt(~bound_surf_1, 
                                                      return_indices=True)
        dis_to_1 = distance_to_1[bound_surf_2] 
        pos_to_1 = [pos[bound_surf_2] for pos in position_to_1] 
        pos_2 = bound_surf_2.nonzero()

        # calculate min distances of all elements of boundary 1 to boundary 2
        if (~bound_surf_2 > 0).all():  # workaround for scipy bug 1089
            raise ValueError("Can't calculate distance_function ",
                             "(no background)")
        else:
            distance_to_2, position_to_2 = \
                scipy.ndimage.distance_transform_edt(~bound_surf_2, 
                                                      return_indices=True)

        dis_to_2 = distance_to_2[bound_surf_1] 
        pos_to_2 = [pos[bound_surf_1] for pos in position_to_2] 
        pos_1 = bound_surf_1.nonzero()

        # correct coordintates for inset
        pos_to_1 = [pos + ins.start for pos, ins in zip(pos_to_1, inset)]
        pos_2 = [pos + ins.start for pos, ins in zip(pos_2, inset)]
        pos_to_2 = [pos + ins.start for pos, ins in zip(pos_to_2, inset)]
        pos_1 = [pos + ins.start for pos, ins in zip(pos_1, inset)]

        return dis_to_1, pos_to_1, pos_2, dis_to_2, pos_to_2, pos_1


    ########################################################
    #
    # Layers
    #
    #########################################################

    def makeLayers(self, nLayers=None, width=None, widthMode='median', 
                   maxDistance=None, fill=True, nExtraLayers=0, 
                   extra_1=None, extra_2=None, offset=0.5):
        """
        Makes cleft and boundary layers.

        Attributes self.bound1Id, self.bound2Id and self.cleftId (defining 
        boundaries and the cleft region, respectively) need to be set. Cleft
        layers can be formed on (is a subset of) the cleft region only, while
        the boundary layers are formed on boundaries and extra regions 
        (subset again).

        If arg width is None, the cleft width is calculated using 
        self.getWidth(toBoundary=0). In short, for each element of boundary 
        borders (elements of boundaries that contact the cleft region) the 
        shortest distance to the other boundary is calculated. The distances
        greater than arg maxDistance are not taken into account, The 
        mean/median/min/max (according to the argument widthMode) of these 
        values -1 is then cleft width calculated edge-to-edge. Strictly 
        speaking, subtracting 1 is valid for relatively flat cleft. The only 
        reason for specifying arg width as an argument is to avoid computing 
        it twice. 

        Number of layers is the rounded value of the cleft width.

        The layers are then made using Segment.makeLayersBetween() (see for more
        detailed info). If there is no extra layers (nExtraLayers=0) layers of
        approximatelly same thickness are positioned parallel to the boundary 
        borders. Specifically, layers are calculated as:

        layer_no = floor{(d_1 - offset) * nLayers / (d_1 + d_2 - 2*offset) + 1}

        where d_1 and d_2 are the shortest distances to borders of boundary 1 
        and 2. Layers are made only in the cleft region. 

        If arg maxDistance is specified, the cleft region is restricted to 
        those elements that have sum of distances to both boundaries 
        <= maxDistance. If arg fill is True the holes created by the 
        restriction to maxDistance are filled.

        Number of layers is either given by arg nLayers (in which case layer
        thcikness is cleft_width / nLayers) or it equals the cleft width 
        (thickness = 1).

        In case nExtraLayers is > 0, extra layers are formed in addition to the
        cleft layers (explained above). The extra layers are formed on the
        boundaries and on the extra regions (args extra_1 and extra_2). The
        extra layers are formed based on their euclidean distance to the closest
        cleft layer, and they have the same thickness as the cleft layers. 
        This is done using Segment.makeLayersFrom() method. The
        additional layers formed over the first boundary and the first extra
        region are numbered from 1 to nExtraLayers, the ids of the cleft layers
        are shifted by nExtraLayers and the layers formed over the second
        boundary and the second extra region go from 
        nLayers_nExtraLayers+1 to nLayers+2*nExtraLayers.

        The layers are returned as a Segment object, where layers.data is an
        array that contains all layers. Layers.data can have smaller shape than 
        self.data, and layers.offset is set appropriatelly. To be precise, 
        layers.data is just big enough to contain all layers as well as regions
        occupied by boundaries.

        Arguments:
          - nLayers: number of layers, in None 1-pixel thick layers are formed
          - width: cleft width, calculated if None
          - widthMode: cleft width calculation mode ('mean', 'median', 'min', 
          'max', or any appropriate numpy function), used only if width is None
          - maxDistance: max allowed sum of (minimal) distances to the two 
          boundaries, if None no limit is imposed
          - fill: flag indicating if holes created by maxDistance procedure
          are filled (used only if maxDistance is not None)
          - nExtraLayers: (int) number of additional layers formed on each side
          of the cleft
          - extra_1, extra_2: (int, or a list of int's) ids of extra regions 1 
          and 2 respectivly
          - offset: numerical parameter that compensates for the disccrete
          nature of distances, should be 0.5
      
        Returns: layers, width
          - layers: (Segment) layers
          - width: cleft width
        """

        # find cleft width if needed
        if width is None:
            width, width_vector = self.getWidth(mode=widthMode, 
                                                maxDistance=maxDistance)

        # make layers
        cleft_id_list = list(self.cleftId)
        layers, width_other = self.makeLayersBetween(
            bound_1=self.bound1Id, bound_2=self.bound2Id, mask=cleft_id_list, 
            nLayers=nLayers, width=width, offset=offset, 
            maxDistance=maxDistance, fill=fill, 
            nExtraLayers=nExtraLayers, extra_1=extra_1, extra_2=extra_2)
            
        return layers, width

    def adjustToRegions(self, regions, ids=None, value=0):
        """
        Adjusts self.data to correspond to regions.data.

        Sets elements of self.data that have one of the specified ids and that
        correspond at 0-elements of arg regions to arg value.

        If arg ids is None, any element of self.data can be set to arg value. 

        Arguments:
          - regions: (Segment) regions
          - ids: list of ids of this instance where self.data can be adjusted
          - value: value to be adjusted to

        Modifies self.data.
        """

        # position regions to this instance
        reg_data = regions.useInset(inset=self.inset, mode='abs', 
                                   expand=True, update=False)

        # set elements
        if ids is None:
            self.data[reg_data==0] = value
        else:
            for id_ in ids:
                self.data[(reg_data==0) & (self.data==id_)] = value


    ########################################################
    #
    # Columns and layers parametrization
    #
    #########################################################

    def makeColumns(
        self, bins, ids=None, system='radial', normalize=False, 
        originMode='one', startId=None, metric='euclidean', connectivity=1, 
        rimId=0, rimLocation='out', rimConnectivity=1):
        """
        Segments cleft to columns, that is segments perpendicular to cleft 
        layers. 

        The attribute data of this instance needs to contain layers, preferably 
        of thickness 1. This can be done using makeLayers(nLayers=None).

        First, the layers are parametrized, that is layer centers are determined
        (see parametrizeLayers() and pickCenter()) and then a coordinate system
        with origin at the layer center is placed on each of the layers 
        according to args system and metric (see self.parametrizeLayers() for 
        details). 

        If arg normalize is False, each layer element is assigned a value
        that equals its distance to the center. Alternatively, if normalize
        is True and a radial distance needs to be calculated (system 'radial' or
        'polar'), radial values assigned to layer elements are caluclated as:

            distance_to_center / (distance_to_center + min_distance_to_rim)

        where min_distance_to_rim is the shortest distance to rim which 
        surrounds layers. Consequently, center gets value of 0 and layer
        elements that contact the rim are assigned value 1. 

        Then, each layer element is assigned to a segment based on its
        parametrization and arg bins (see Grey.labelByBins() for details). The
        arg bins can contain one set of binning values for each of the 
        coordinates of the specified coordinate system. If bins are not given 
        for all coordinates, those at the end are ignored. For example, given:

          bin=[[0,2,4], [-pi, 0, pi]], system='polar'

        in 3D, rho coordinate is binned according to [0,2,4], phi according
        to [-pi, 0, pi] and z is disregarded (the order of coordinates in the
        polar system is: rho, phi, z).

        Lower bin limits are inclsive, while the upper are exclusive except for
        the last one which is inclusive.

        Non-asigned array elements are set to 0. The resulting segmented image 
        is converted to a Segmentation object that has the same positioning 
        as this instance.

        For example, if self.data is:

            [[5, 5, 5, 5, 5, 5],
             [5, 1, 1, 1, 5, 5],
             [5, 1, 1, 1, 5, 5],
             [5, 1, 1, 1, 5, 5], 
             [5, 1, 1, 1, 5, 5],
             [5, 1, 1, 1, 5, 5],
             [5, 1, 1, 1, 5, 5],
             [5, 1, 1, 1, 5, 5],
             [5, 5, 5, 5, 5, 5]])
        
        self.makeColumns(
            ids=[1], system='radial', normalize=False, startId=1, 
            metric='euclidean', connectivity=1, rimId=5, bins=[0, 1, 2, 3, 4])

        will return:

            [[0, 0, 0, 0, 0, 0],
             [0, 4, 4, 4, 0, 0],
             [0, 3, 3, 3, 0, 0],
             [0, 2, 2, 2, 0, 0],
             [0, 2, 1, 2, 0, 0],
             [0, 2, 2, 2, 0, 0],
             [0, 3, 3, 3, 0, 0],
             [0, 4, 4, 4, 0, 0],
             [0, 0, 0, 0, 0, 0]])

        while: 

        self.makeColumns(
            ids=[1], system='radial', normalize=False, startId=1, 
            metric='euclidean', connectivity=1, rimId=5, bins=[0, 2, 4])

        will return:

            [[0, 0, 0, 0, 0, 0],
             [0, 2, 2, 2, 0, 0],
             [0, 2, 2, 2, 0, 0],
             [0, 1, 1, 1, 0, 0],
             [0, 1, 1, 1, 0, 0],
             [0, 1, 1, 1, 0, 0],
             [0, 2, 2, 2, 0, 0],
             [0, 2, 2, 2, 0, 0],
             [0, 0, 0, 0, 0, 0]])

        Note that there might be a global ambiquity in the parametrization. 
        Namely, because there is an ambiguity in the direction of the best
        fit line generated by scipy.linalg.svd(), cartesian axis directions
        and the sign of the polar angle are ambiguous.

        Arguments:
          - bins: binning values
          - ids: ids of layer where columns are made
          - system: coordinate syatem, can be 'radial', 'cartesian' or 'polar'
          - normalize: flag indicating if radial distance is normalized
          - metric: distance metric, can be 'euclidean', 'geodesic' or
          'euclidean-geodesic'
          - connectivity: connectivity of the structure element (as in
        scipy.ndimage.generate_binary_structure() where rank is self.ndim)
        for geodesic distance calculation (int). Not used for euclidean.
          - originMode: determines how centers are determined, currently only 
          'one' (see self.pickLayerCenters())
          - startId: layer where the center is first determined, used in mode 
          'one', None means start from the middle (see self.pickLayerCenters())
          - rimId: id or rim region
          - rimLocation: specifies if the rim is just outside of layers
          ('out') or on the layer boundary ('in')
          - rimConnectivity: (int) connectivity of the structure element
          that defines contact elements between a layer and the rim region.

        Returns column-segmented cleft as an instance of Segment where:
          - column.data: (ndarray) labeled array
          - column,binIds: (dict) where each id is a key and the corresponding 
          value is a list of lower and uper bin limits for each binning.
        """

        # get layer ids
        if ids is None:
            ids = self.cleftId

        # parametrize layers
        coordinates = self.parametrizeLayers(
            ids=ids, system=system, normalize=normalize, 
            originMode=originMode, startId=startId, metric=metric, 
            connectivity=connectivity, rimId=rimId, 
            rimLocation=rimLocation, rimConnectivity=rimConnectivity, 
            noDistance=-1)

        # make labels
        col_data, bin_ids = Grey.labelByBins(values=coordinates, bins=bins)

        # make self.column object and set as an attribute of this object
        columns = Segment(data=col_data, ids=list(bin_ids.keys()), copy=False)
        columns.copyPositioning(self, saveFull=False)
        columns.binIds = bin_ids

        return columns

    def parametrizeLayers(
        self, ids, system='radial', normalize=False, originMode='one', 
        startId=None, metric='euclidean', connectivity=1, rimId=0, 
        rimLocation='out', rimConnectivity=1, noDistance=-1):
        """
        Parametrizes layers, that is puts a coordinate system on layer-like
        segments. As a result, each element of self.data that belongs to one of 
        the layers (specified by arg ids) is associated with one or more
        coordinates of the new system.

        Radial coordinate is calculated as the distance of each layer element
        to the origin of that layer, using the specified distance metric. The 
        origin of a layer is an element that has the largest distance to the 
        rim surrounding the layer. In case there's more than one element having 
        the largest distance, the element closest to the layer center of mass 
        is chosen. See self.elementDistanceToRim() and self.pickLayerCenters()
        for the origin determination and self.distanceFromOrigin() for the
        distance from origin.

        If arg normalize is False, each layer element is assigned a value
        that equals its distance to the center. Alternatively, if normalize
        is True, the values assigned to layer elements is caluclated as:

            distance_to_center / (distance_to_center + min_distance_to_rim)

        where min_distance_to_rim is the shortest distance to rim which 
        surrounds layers. Consequently, center gets value of 0 and layer
        elements that contact the rim are assigned value 1. 

        In 2D and >3D only the radial parametrization is possible.

        To calculate cartesian coordinates in 3D the layers are rotated so that
        the vector perpendicular to layers (the best fit line for all
        layer origins) is aligned with the z-axis (theta=0), and that that the 
        direction towards the lowest z-postions of the original layers is 
        aligned with the x-axis (phi=0). The origin of this rotation is the 
        layer center, so this rotation is repeated for each layer separately. 
        The coordinates of the rotated cleft are taken as the cartesian 
        parametrization of the cleft. 

        If the metric is euclidean for 3D cartesian system, the 
        coordinates of the rotated cleft are the calculated coordinates. In
        addition, z coordinate is calculated in repect to the layer center
        (just like x and y). Also, it is not defined if layer ids increase 
        along positive or negative z-axis.

        Alternatively, if the metric is geodesic for 3D cartesian system,
        first the polar coordinate phi is calculated and the x and y 
        coordinates are calculated from the radial and phi coordinate. In
        this case z coordinate is not calculated (nor returned).

        Layers of this instance should be thin (of thickness 1 or so), so that
        center of each layer borders layers on both sides.

        Polar coordinates can be used in 3D. The angle (phi) is calculated from
        the cartesian coordinates (x and y). The radial coordinate calculated as
        explained above and the z coordinate from the cartesian coordinates
        complete the system.
        
        Note that there might be a global ambiquity in the parametrization. 
        Namely, because there is an ambiguity in the direction of the best
        fit line generated by scipy.linalg.svd(), cartesian axis directions
        and the sign of the polar angle are ambiguous.

        Arguments:
          - ids: layer ids
          - system: coordinate syatem, can be 'radial', 'cartesian' or 'polar'
          - normalize: flag indicating if radial distance is normalized
          - metric: distance metric, can be 'euclidean', 'geodesic' or
          'euclidean-geodesic'
          - connectivity: connectivity of the structure element (as in
        scipy.ndimage.generate_binary_structure() where rank is self.ndim)
        for geodesic distance calculation (int). Not used for euclidean.
          - originMode: determines how centers are determined, currently only 
          'one' (see self.pickLayerCenters())
          - startId: layer where the center is first determined, used in mode 
          'one', None means start from the middle (see self.pickLayerCenters())
          - rimId: id or rim region
          - rimLocation: specifies if the rim is just outside of layers
          ('out') or on the layer boundary ('in')
          - rimConnectivity: (int) connectivity of the structure element
          that defines contact elements between a layer and the rim region.
          - noDistance: value used for array elements where the distance is
          not calculated

        Returns: masked ndarray where axis 0 denotes a coordinate and the other
        dimensions are the same as those of self.data. Elements outside the
        layers are masked. The coordinates returned depend on the coordinate 
        system (arg system):
          - radial: rho
          - cartesian (3D only): [x, y, z]
          - polar (3D only): [rho, phi, z]
        """

        # set mask containing all segments from ids
        ids_mask = numpy.zeros(shape=self.data.shape, dtype=bool)
        for id_ in ids:
            ids_mask[self.data==id_] = True

        # distance to rim
        to_rim = self.elementDistanceToRim(
            ids=ids, metric=metric, connectivity=connectivity, rimId=rimId,
            rimLocation=rimLocation, rimConnectivity=rimConnectivity, 
            noDistance=noDistance)

        # origin based on max distance to rim
        centers = self.pickLayerCenters(ids=ids, distance=to_rim, 
                                        mode=originMode, startId=startId)

        # radial parametrization as distance from origin
        if not ((system == 'cartesian') and (metric == 'euclidean')):
            rho = self.distanceFromOrigin(
                origins=centers, metric=metric, connectivity=connectivity, 
                noDistance=noDistance)

            # normalize if needed
            if normalize:
                total = rho + to_rim
                rho = numpy.where(total > 0, rho / total.astype(float), 0)

        # wrap up the radial parametrization in a masked array
        if (system == 'radial') or (self.ndim == 2):
            #rho = numpy.expand_dims(rho, axis=0)
            rho = numpy.ma.masked_array(rho, mask=~ids_mask, 
                                        fill_value=noDistance)
            return rho

        elif self.ndim == 3:

            # cartesian parametrization (based on Euclidean metric)
            if (system == 'polar') or (system == 'cartesian'):

                # make best fit line for layer centers
                cent = Vector(list(centers.values()))
                cm, vector = cent.bestFitLine()
                
                # extract one point and angles for the centers line
                #origin = numpy.rint(cm.data).astype(int)
                theta = vector.theta
                phi = vector.phi

                # make a rotation that brings the cleft axis to the z axis
                # and the steepest z-descent to x-axis (phi=0)
                rot_phi = Affine3D(alpha=-phi, axis='z')
                rot_theta = Affine3D(alpha=-theta, axis='y')
                rot = Affine3D.compose(rot_theta, rot_phi)

                # get cartesian parametrization for Euclidean metric
                for id_, origin in list(centers.items()):                    
                    xyz_id = Coordinates.transform(
                        shape=self.data.shape, affine=rot, 
                        origin=origin, center=True)
                    id_mask = numpy.array(3 * [self.data == id_])
                    try:
                        xyz_euc[id_mask] = xyz_id[id_mask]
                    except NameError:
                        xyz_euc = xyz_id

            if (system == 'cartesian') and (metric == 'euclidean'):
                if normalize:
                    x_max = numpy.abs(xyz_euc[0]).max()
                    y_max = numpy.abs(xyz_euc[1]).max()
                    xyz_euc[0] = xyz_euc[0] / float(x_max)
                    xyz_euc[1] = xyz_euc[1] / float(y_max)
                xyz = numpy.ma.masked_array(
                    xyz_euc, mask=3 * [~ids_mask], fill_value=noDistance)
                return xyz
            
            # get phi from cartesian with Euclid
            phi = numpy.arctan2(xyz_euc[1], xyz_euc[0])

            if system == 'polar':

                # rho from radial, z from cartesian and mask
                rho_phi_z = numpy.zeros(shape=xyz_euc.shape)
                rho_phi_z[0] = rho
                rho_phi_z[1] = phi
                rho_phi_z[2] = xyz_euc[2]
                rho_phi_z = numpy.ma.masked_array(
                    rho_phi_z, mask=3 * [~ids_mask], fill_value=noDistance)

                return rho_phi_z

            elif system == 'cartesian':

                # get xy from rho and phi
                shape = [2] + list(self.data.shape)
                xy = numpy.zeros(shape=shape)
                xy[0] = rho * numpy.cos(phi)
                xy[1] = rho * numpy.sin(phi)
                xy = numpy.ma.masked_array(
                    xy, mask=2 * [~ids_mask], fill_value=noDistance)

                return xy

        else: 
            raise ValueError(
                "Sorry, only 'radial' coordinate can be calculated for Ndim"
                + "other than 2 or 3.") 

    def pickCenter(self, id_, distance, fromPosition=None):
        """
        Returns coordinates of the center of segment specified by arg id_.
        Center is defined as the point that has the highest distance (to the 
        rim of the segment). Distances of all points are specified by 
        arg distance.

        If arg fromPosition is None, the position of global maximum of the
        specified segment is returned. 

        Alternatively, if arg fromPosition is a coordinate, the position of 
        maximum on the specified segment within the neighborhood of the arg 
        fromPosition (both rank and connectivity of the structure element are 
        self.ndim). If no element with the specified id exists in the 
        neighborhood of fromPosition, None is returned.

        If more than one max position exist, the one closest to the center of
        mass of the whole segment (having id_) is chosen. If multiple (min
        position to cm) exist, one of them is chosen randomly (calculated by 
        scipy.ndimage.minimum_position).

        Arguments:
          - id_: segment id
          - distance: (ndarray of the same shape as self.data) distances
          - fromPosition: (1d ndarray) coordinates of a point in the 
          neighborhood of or on the specified segment

        Returns: (1d ndarray) coordinates of max position
        """

        # find max distance position(s)
        if fromPosition is None:

            # in the whole segment
            max_dist = scipy.ndimage.maximum(
                distance, labels=self.data, index=id_)
            max_indices = numpy.nonzero((distance==max_dist) & (self.data==id_))

        else:
            
            # find neighborhood of fromPosition on the id_
            se = scipy.ndimage.generate_binary_structure(self.ndim, self.ndim)
            hood = numpy.zeros(shape=self.data.shape, dtype=bool)
            hood[tuple(fromPosition)] = True
            hood = scipy.ndimage.binary_dilation(hood, structure=se)
            hood[self.data != id_] = False
            if not hood.any():
                return None

            # max distance(s) in the hood
            #center_pos = scipy.ndimage.maximum_position(distance, labels=hood)
            max_dist = scipy.ndimage.maximum(distance, labels=hood)
            max_indices = numpy.nonzero((distance==max_dist) & hood)

        # extract one max distance position
        if len(max_indices[0]) > 1:

            # more than one max, pick the closest to the cm
            max_indices = numpy.asarray(max_indices)
            cm = scipy.ndimage.center_of_mass(self.data==id_)
            cm = numpy.expand_dims(numpy.asarray(cm), axis=-1)
            sq_diff = numpy.square(max_indices - cm)
            cm_dist_sq = numpy.add.reduce(sq_diff, axis=0)
            center_index = scipy.ndimage.minimum_position(cm_dist_sq)[0]
            center_pos = max_indices[:, center_index]

        else:
            center_pos = numpy.array([x[0] for x in max_indices])

        return numpy.asarray(center_pos)

    def pickLayerCenters(self, ids, distance, mode='one', startId=None,
                         notFound='previous'):
        """
        Returns coordinates of 'centers'of all segments specified by arg ids.

        If arg mode is 'one', it starts by finding the max position (center) 
        on the distance array (arg distance) on the segment specified by arg
        startId. Afterwards, the center determination proceeds to neighboring 
        segments according to the positioning of ids in arg ids, in both 
        ascending (to the right) and descending (to the left) order. The 
        center of a next layer is determined as the max distance on the region
        of the next layer that is within the neighborhood of the previous  
        center. The neighborhood of an element is defined as all elements
        that have at least a vertex in common. If no layer element is found 
        within the neighborhood, the center of that layer is set to None 
        (arg notFound is None) or to the previous layer center (arg notFound
        is 'previous').

        It is necessary that segments are organized as thin layers,
        so that a center of each layer always lies in a neighborhood of the 
        layers that are next to it. Layers of thickness 1 most likely 
        satisfy this condition. 

        Max positions are calculated by self.pickCenter, that in turn calls
        scipy.ndimage.maximum_position, which returns only one value if 
        multiple max points exist.

        Arguments:
          - id_: segment id
          - distance: (ndarray of the same shape as self.data) distances
          - mode: determines how centers are determined, currently only 'one'
          - startId: Segment layer where the center is first determined, 
          used in mode 'one', None means start from the middle
          - notFound: determines the returned value for layers whose
          center wasn't found

        Returns: (dict) of coordinates, where keys are ids and values are 
        (1d ndarray) coordinates of centers, or None if the center was not 
        found.
        """

        centers = {}
        if mode == 'one':

            # start id
            if startId is None:
                startId = ids[int(old_div(len(ids), 2))] 

            # find start center
            start_center = self.pickCenter(id_=startId, distance=distance)
            centers[startId] = start_center

            # make ascending and descending sequences
            ids = numpy.asarray(ids)
            index = (ids == startId).nonzero()[0][0]
            ascend = ids[index+1:]
            if index == 0:
                descend = numpy.array([])
            else:
                descend = ids[slice(index-1, None, -1)]

            # acsend and descend part
            for id_range in (ascend, descend):
                last_center = start_center
                for id_ in id_range:
                    curr_center = self.pickCenter(id_=id_, distance=distance, 
                                                  fromPosition=last_center)
                    if curr_center is not None:
                        centers[id_] = curr_center
                        last_center = curr_center
                    else:
                        if notFound is None:
                            centers[id_] = curr_center
                        elif notFound == 'previous':
                            centers[id_] = last_center
                        else:
                            raise ValueError(
                                "Arg notFound: " + notFound + "was not "
                                + "understood. Valid values are None and "
                                + "'previous'.")

        else:
            raise ValueError(
                "Argument mode: " + mode + " not understood."
                + " Only 'one' is currently implemented.")

        return centers
        

    ########################################################
    #
    # Input / output
    #
    #########################################################

    @classmethod
    def read(cls, cleftId, bound1Id, bound2Id, file, clean=True, 
             byteOrder=None, dataType=None, arrayOrder=None, shape=None):
        """
        Reads segmented image (label filed) from a file and sets required ids.

        Each of the id args (cleftId, bound1Id, bound2Id) can be a single int
        or a list (array) in which cases a (boundary or cleft) is formed form 
        all specified ids.

        If file is in em or mrc format (file extension em or mrc) only the file
        argument is needed. If the file is in the raw data format (file
        extension raw) all arguments are required.

        Arguments:
          - bound1Id, bound2Id, cleftId: ids of boundaries 1, 2 and cleft
          - file: file name
          - clean: if true, only the segments corresponding to ids are kept
          - byteOrder: '<' (little-endian), '>' (big-endian)
          - dataType: any of the numpy types, e.g.: 'int8', 'int16', 'int32',
            'float32', 'float64'
          - arrayOrder: 'C' (z-axis fastest), or 'F' (x-axis fastest)
          - shape: (x_dim, y_dim, z_dim)

        Returns:
          - instance of Segment
        """

        # read file
        from pyto.io.image_io import ImageIO as ImageIO
        fi = ImageIO()
        fi.read(file=file, byteOrder=byteOrder, dataType=dataType, 
                arrayOrder=arrayOrder, shape=shape)

        # make new instance
        cleft = cls(data=fi.data, cleftId=cleftId, bound1Id=bound1Id, 
                    bound2Id=bound2Id, copy=False, clean=clean)

        return cleft

   
