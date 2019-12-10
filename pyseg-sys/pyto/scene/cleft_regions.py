"""
Contains class CleftRegions for the analysis of a cleft-like region (a region
between two roughly parallel boundaries) of an image segmented in regions.

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id: cleft_regions.py 1070 2014-11-06 14:07:41Z vladan $
"""

__version__ = "$Revision: 1070 $"


import logging
from copy import copy, deepcopy
import numpy
import scipy

from pyto.core.image import Image
from pyto.segmentation.density import Density
from pyto.segmentation.cleft import Cleft


class CleftRegions(object):
    """
    Formation and analysis of cleft regions. A cleft is defined by an 
    greyscale image (self.image) and a corresponding segmentation (self.cleft) 
    that defines two boundaries and cleft proper that is located between the 
    boundaries (self.cleft). 

    Cleft regions are segments organized as layers parallel to
    the boundaries or columns perpendicular to the boundaries .

    Contains methods for the geometrical analysis of the cleft and for the 
    basic statistical analysis of cleft regions density. 

    Important attributes:
      - self.regions: (..segmentation.Segment) cleft regions
      - self.image: density (greyscale) image
      - self.cleft (..segmentation.Cleft): labels main cleft parts (boundaries,
      cleft proper)
      - self.nLayers: number of layers (layers)
      - self.nBoundLayers: number of boundary layers (layers)

    Common usage to make layers / columns:
    
      cl = CleftRegions(image=image, cleft=cleft)
      cl.makeLayers() / cl.makeColumns()
      cl.findDensity(regions=cl.regions)

    Some of the attributes calculated and set by this example are:
      - self.regions: (..segmentation.Segment) regions
      - self.regionDensity (pyto.segmentation.Density) region density
      - self.width: cleft width (layers)
      - self.widthVector: average cleft width vector (layers)

    All methods respect image inset information.
    """

    ###############################################################
    #
    # Initialization
    #
    ##############################################################

    def __init__(self, image, cleft):
        """
        Initializes attributes.

        Arguments:
          - image: (core.Image) greyscale image
          - cleft: (segmentation.Cleft) segmented image that defines cleft 
          and boundaries
        """

        # set attributes from arguments
        self.image = image
        self.cleft = cleft

        # use inset
    
        # parameters that most likely don't need to be changed
        self.extendInset = 1

    def getParameterStrings(self, names=None):
        """
        """

        # set parameter names
        if names is None:
            names = self.parameterNames

        # format
        strings = [self.parameterFormats[name] % getattr(self, name) 
                   for name in names]

        return strings

    def getBinIdsStrings(self):
        """
        """

        ids = self.regions.binIds.keys()
        ids = numpy.asarray(ids)
        strings = ['%d : ' % id_ + str(self.regions.binIds[id_]) 
                   for id_ in ids]

        return strings

    ###############################################################
    #
    # Layer ids
    #
    ##############################################################

    def getCleftLayerIds(self, exclude=0):
        """
        Returns layer ids corresponding to the cleft.

        If arg exclude is a list (ndarray or tuple) exclude[0] cleft layers
        facing boundary1 and exclude[1] layers facing boundary2 are excluded
        from the returned ids. Otherwise, if arg exclude is a single int,
        exclude cleft layers are removed from each side.

        Requires attributes self.nLayers and self.nBoundLayers to be set.

        Argument:
          - exclude: (int, list, tuple, or ndarray) number of layers to exclude 
          from the cleft

        Returns: (ndarray) layer ids
        """
        if not isinstance(exclude, (list, numpy.ndarray, tuple)):
            exclude = [exclude, exclude]

        ids = range(self.nBoundLayers + 1 + exclude[0], 
                    self.nBoundLayers + 1 + self.nLayers - exclude[1])
        return numpy.array(ids)

    cleftLayerIds = property(fget=getCleftLayerIds, doc='Cleft layer ids')

    def getBoundLayerIds(self, thick=None):
        """
        Returns all boundary layer ids (ndarray). 
        
        Boundary thickness can be restricted to a certan number of layers.
        This can be done by (arg) thick if specified, or by self.boundThick 
        (provided it's neither None nor [None, None]). If neither of the two 
        is specified, the boundaries consist of all layers outside the cleft, 
        that is the boundary thickness is self.nBoundLayers. In this case, 
        only the specified number of layers are considered for each boundary, 
        starting from boundary layers facing the cleft. 

        Requires attributes self.nLayers and self.nBoundLayers to be set.

        Argument:
          - thick: (single int or list or array of two ints) boundary thickness
          in number of layers 
        """

        if not isinstance(thick, (list, numpy.ndarray, tuple)):
            thick = [thick, thick]

        ids = numpy.concatenate([self.getBound1LayerIds(thick=thick[0]), 
                                 self.getBound2LayerIds(thick=thick[1])])
        return ids

    boundLayerIds = property(fget=getBoundLayerIds, 
                             doc='Layer ids of both boundaries')
        
    def getBound1LayerIds(self, thick=None):
        """
        Returns layer ids corresponding to the boundary 1 (ndarray).

        Boundary thickness can be restricted to a certan number of layers.
        This can be done by (arg) thick if specified, or by self.boundThick 
        (provided it's not None. If neither of the two is specified, 
        the boundary consists of all layers outside the cleft, 
        that is the boundary thickness is self.nBoundLayers. In this case, 
        only the specified number of layers are considered for the boundary, 
        starting from boundary layers facing the cleft. 

        Requires attribute self.nBoundLayers to be set.
        """

        if (thick is None) or (thick == [None, None]):
            thick = getattr(self, 'boundThick', None)
        if (thick is None) or (thick == [None, None]):
            thick = self.nBoundLayers
        if isinstance(thick, (list, numpy.ndarray, tuple)):
            thick = thick[0]

        ids = numpy.arange(thick)
        ids += self.nBoundLayers + 1 - thick
        return ids
        
    bound1LayerIds = property(fget=getBound1LayerIds, 
                             doc='Layer ids of boundary 1')
        
    def getBound2LayerIds(self, thick=None):
        """
        Returns layer ids corresponding to the boundary 2 (ndarray).

        Boundary thickness can be restricted to a certan number of layers.
        This can be done by (arg) thick if specified, or by self.boundThick 
        (provided it's not None. If neither of the two is specified, 
        the boundary consists of all layers outside the cleft, 
        that is the boundary thickness is self.nBoundLayers. In this case, 
        only the specified number of layers are considered for the boundary, 
        starting from boundary layers facing the cleft. 

        Requires attributes self.nLayers and self.nBoundLayers to be set.
        """

        if (thick is None) or (thick == [None, None]):
            thick = getattr(self, 'boundThick', None)
        if (thick is None) or (thick == [None, None]):
            thick = self.nBoundLayers
        if isinstance(thick, (list, numpy.ndarray, tuple)):
            thick = thick[1]

        ids = numpy.arange(thick)        
        ids += self.nLayers + self.nBoundLayers + 1
        return ids
        
    bound2LayerIds = property(fget=getBound2LayerIds, 
                             doc='Layer ids of boundary 2')
        
    def getLayerIds(self):
        """
        Returns all layer ids
        """
        ids = numpy.concatenate([self.getBound1LayerIds(), 
                            self.getCleftLayerIds(), self.getBound2LayerIds()])
        return ids

    def getBoundEdgeLayerIds(self):
        """
        Returns a list containing ids of those boundary1 and boundary2 layers
        (in this order) that are right next to (contacting) cleft layers. 
        """

        id_1 = self.getBound1LayerIds(thick=1)
        id_2 = self.getBound2LayerIds(thick=1)
        return [id_1, id_2]

    boundEdgeLayerIds = property(fget=getBoundEdgeLayerIds,
                                 doc='Boundary edge layer ids')


    ###############################################################
    #
    # Main methods
    #
    ##############################################################

    def makeLayers(self, nLayers=None, widthMode='median', nBoundLayers=0, 
                   maxDistance=None, fill=True, adjust=False, refine=False):
        """
        Makes layers on the cleft and possibly on the boundary regions of a 
        cleft (self.cleft) and calculates cleft width.

        Width (magnitude and vector) is calculated using
        pyto.segmentation.Cleft.getWidth() method, according to arg widthMode
        (passed as arg mode), and by combining cleft distances from both 
        boundaries (Cleft.getWidth() argument toBoundary=0). 

        Layers are formed using pyto.segmentation.Cleft.makeLayers() method.
        In short, layers of the same thickness are first made on the cleft
        region based on the euclidean distance to the cleft boundaries (cleft 
        layers). Cleft layers are restricted to those elements of the cleft 
        region that are not further away from the boundaries than arg 
        maxDistance. If arg fill is True, the holes in the cleft region that 
        are formed by the maxDistance procedure are filled. Then, if arg 
        nBoundLayers > 0, additional layers are formed on cleft boundaries 
        based on the euclidean distance to the cleft layers (boundary layers).

        If arg nLayers is None, the number of cleft layers is calculated as 
        the rounded value of cleft width, so that layers are approximately 1 
        pixel size wide. 

        If arg adjust is True, the cleft and boundary regions (of self.cleft)
        are restricted to include only those elements that were assigned to
        cleft and boundary layers. If in additon arg refine is specified,
        cleft width is recalculated using the adjusted cleft and boundary 
        regions. In case arg nLayers is None and the recalculated cleft width 
        leads to a different number of cleft layers, the layers are
        recalculated also.

        Arguments:
          - widthMode: cleft width calculating mode, 'mean', 'median', 'min' or 
          'max' (actually can be any appropriate numpy function)
          - nLayers: number of layers
          - maxDistance: max allowed sum of (minimal) distances to the two 
          bounaries, if None no limit is imposed
          - fill: flag indicating if holes created by maxDistance procedure
          are filled (used only if maxDistance is not None)
          - nBoundLayers: (int) number of additional layers formed on each side
          of the cleft
          - adjust: flag indicating if the self.cleft is adjusted to layers
          (self.regions)
          - refine: flag indication if the layers are recalculated after 
          self.cleft was recalculated (has effect only if arg adjust is True
          and arg nLayers is None)

        Modifies: 
          - self.cleft: only if arg adjust is True

        Sets:
          - self.widthMode (from arguments)
          - self.nBoundLayers (from arguments)
          - self.maxDistance (from arguments)
          - self.width: cleft width
          - self.widthVector: (..geometry.Vector) cleft width vector
          - self.nLayers: number of layers
          - self.regions: (pyto.segmentation.Segment) layers
        """

        # set parameter attributes
        self.widthMode = widthMode
        self.nBoundLayers = nBoundLayers
        self.maxDistance = maxDistance

        # parameter names and formats
        self.parameterNames = ['widthMode', 'nBoundLayers', 'maxDistance']
        self.parameterFormats = {
            'widthMode' : 'width mode: %s',
            'nBoundLayers' : 'number of layers made on each boundary: %d'}
        if maxDistance is None:
            self.parameterFormats['maxDistance'] = 'maximal distance: %s'
        else:
            self.parameterFormats['maxDistance'] = 'maximal distance: %d'

        # geometry
        self.width, self.widthVector = self.cleft.getWidth(mode=widthMode)

        # n layers
        if nLayers is None:
            self.nLayers = numpy.round(self.width).astype(int)
        else:
            self.nLayers = nLayers
        
        # make layers
        self.regions, width = self.cleft.makeLayers(\
            nLayers=self.nLayers, width=self.width, nExtraLayers=nBoundLayers, 
            maxDistance=maxDistance, fill=fill)

        # adjyst cleft to layers
        if adjust:
            self.adjustCleft()

        # recalculate layers if needed
        if adjust and refine:

            # get width of adjusted layers
            self.width, self.widthVector = self.cleft.getWidth(mode=widthMode)

            if nLayers is None:

                # make layers again if n layers is different
                new_n_layers = numpy.round(self.width).astype(int)
                if self.nLayers != new_n_layers:
                    self.nLayers = new_n_layers
                    self.regions, width = self.cleft.makeLayers(
                        nLayers=self.nLayers, width=self.width, fill=fill,
                        nExtraLayers=nBoundLayers, maxDistance=maxDistance)
                    
    def makeColumns(
        self, bins, ids=None, system='radial', normalize=False, 
        originMode='one', startId=None, metric='euclidean', connectivity=1, 
        rimId=0, rimLocation='out', rimConnectivity=1):
        """
        Segments cleft to columns, that is segments perpendicular to cleft 
        layers. 

        This instance needs to contain layers (self.regions), preferably of
        thickness 1. This can be done using makeLayers().

        The columns are generated by makeColumn() method of (Cleft) 
        self.regions. See Cleft.makeColumns() for details. Returns a new
        instance of this class that has attribute regions set to the resulting 
        columns (Segment object). The positioning, image and cleft attributes 
        are the same as in this instance.

        Elements outside the current data array are considered to be 0, which
        is important if arg rimId is 0 and one or more of layer elements are
        on the array boundaries.

        Saves all parameters as attributes (with the same names) of the 
        returned object. Also sets:
          - parameterNames: list of parameter names
          - parameterFormats: dictionary where parameter names are keys and 
          more descriptive parameter names together with formating strings 
          are values.

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

        Returns column-segmented cleft as an instance of this object where:
          - column.regions: (Segment) coulumns 
          - column.regions.data: (ndarray) labeled array
          - column.regions.binIds: (dict) where each id is a key and the 
          corresponding value is a list of lower and uper bin limits for each 
          binning.
        """

        # set layer ids used to make columns 
        if ids is None:
            ids = self.cleftLayerIds

        # make an object of this class to hold columns and save parameters
        cleft_col = self.__class__(image=self.image, cleft=self.cleft)
        cleft_col.system = system
        cleft_col.normalize = normalize
        cleft_col.originMode = originMode
        cleft_col.startId = startId
        cleft_col.metric = metric
        cleft_col.connectivity = connectivity
        cleft_col.rimId = rimId
        cleft_col.rimLocation = rimLocation
        cleft_col.rimConnectivity = rimConnectivity

        # parameter names and formats
        cleft_col.parameterNames = [
            'system', 'normalize', 'originMode', 'startId', 'metric', 
            'connectivity', 'rimId', 'rimLocation', 'rimConnectivity']
        cleft_col.parameterFormats = {
            'system' : 'coordinate system: %s',
            'normalize' : 'normalize radial: %s',
            'originMode' : 'layer origin mode: %s',
            'startId' : 'layer origin start id: %d',
            'metric' : 'distance metric: %s',
            'connectivity' : 'geodesic metric connectivity: %d',
            'rimId' : 'rim id %d',
            'rimLocation' : 'rim location %s',
            'rimConnectivity' : 'rim connectivity %d'}
        if cleft_col.startId is None:
            cleft_col.parameterFormats['startId'] = 'layer origin start id: %s'

        # extend layers data by 1 in each dimension
        ext_shape = [x+2 for x in self.regions.data.shape]
        expanded = numpy.zeros(shape=ext_shape, dtype=self.regions.data.dtype)
        slice_1 = tuple([slice(1,-1)] * self.regions.ndim)
        expanded[slice_1] = self.regions.data
        
        # convert extended layers to Cleft object and extend
        # (clean=False important, otherwise rim appears where bound layes are)
        layers = Cleft(data=expanded, cleftId=ids, clean=False)

        # make columns
        columns = layers.makeColumns(
            bins=bins, ids=ids, system=system, normalize=normalize, 
            originMode=originMode, startId=startId, metric=metric,
            connectivity=connectivity, rimId=rimId, rimLocation=rimLocation,
            rimConnectivity=rimConnectivity)

        # return to original size
        columns.data = columns.data[slice_1]
        columns.copyPositioning(image=self.regions)

        cleft_col.regions = columns
        cleft_col.binIds = columns.binIds

        return cleft_col

    def adjustCleft(self, ids=None, value=0):
        """
        Adjusts the data of self.cleft to the data of self.regions.

        Specifically, sets to arg value all elements of self.cleft.data 
        specified by arg ids where the corresponding self.region.data elements
        are 0.

        If arg ids is None, any element of self.data can be set to arg value. 

        Arguments:
           - ids: cleft ids that can be adjusted
          - value: value to be adjusted to

        Modifies self.cleft.data.
        """

        self.cleft.adjustToRegions(regions=self.regions, ids=ids, value=value)

    def findDensity(self, regions, mode=None, groups=None, 
                    exclude=0, boundThick=None):
        """
        Calculates basic density statistics (mean, std, min, max, volume) of
        individual regions and region groups.  

        See getRegionDensity(), getGroupdensity() and minCleftDensity() methods
        for more detailed explanantion.

        Arguments:
          - regions: (pyto.segmentation.Segment) regions
          - mode: regions type 
          - groups: dictionary with group names (strings) as keys and region
          ids (lists or arrays) as values
          - boundThick: (int or list of ints) boundary thickness
          - exclude: number of cleft regions to be excluded 
          
        Sets:
          - self.regionDensity (pyto.segmentation.Density) region density
          - self.groupDensity: (dictionary) in the form {name:density, ...}
            - name: group name
            - density (pyto.segmentation.Density) group density stats in the
            non-array form (e.g. density.mean is a single number) 
          - self.groupIds: dictionary with group names as keys and ids as values
          - self.exclude: from argument
          - self.boundThick: from argument
          - self.minCleftDensity: minimum cleft region density
          - self.relativeMinCleftDensity: relative minimum cleft region density
          in respect to the cleft density and membrane density
          - self.minCleftDensityId: (numpy.ndarray) layer id (or ids in case 
          more than one layer has min denisty) of the minimum cleft layer 
          density
          - self.minCleftDensityPosition: (numpy.ndarray) position (or 
          positions in case more than one layer has min denisty) of the layer 
          with minimum density in respect to the cleft layers (with excluded)
        """

        # regions density
        self.regionDensity = self.getRegionDensity(regions=regions)

        # parse nBoundLayers and exclude arguments
        if mode == 'layers':
            if not isinstance(boundThick, (list, numpy.ndarray, tuple)):
                boundThick = [boundThick] * 2
            if not isinstance(exclude, (list, numpy.ndarray, tuple)):
                exclude = [exclude] * 2
            self.boundThick = boundThick
            self.exclude = exclude

        # layer related density calculations
        if mode == 'layers':

            # group density
            group_density, group_ids = self.getGroupDensity(
                regionDensity=self.regionDensity, groups=groups,
                exclude=exclude, boundThick=boundThick)
            self.groupDensity = group_density
            self.groupIds = group_ids

            # density profile
            if mode == 'layers':
                reference = self.getBoundLayerIds(thick=boundThick)
                min_density = self.getMinCleftDensity(
                    layerDensity=self.regionDensity,
                    exclude=exclude, reference=reference)
                (self.minCleftDensity, self.relativeMinCleftDensity, 
                 self.minCleftDensityId, 
                 self.minCleftDensityPosition) = min_density

    
    ###################################################################
    #
    # Density related methods
    #
    ####################################################################

    def getRegionDensity(self, image=None, regions=None):
        """
        Calculates density (mean, std, min, max, volume) for each region.

        Arguments:
          - image: (core.Image) grey-scale image, if None self.image used
          - regions: (segmentation.Segment) regions image, if None self.regions 
          used

        Returns: (segmentation.Density) regions density where attributes mean,
        std, min, max and volume are ndarrays indexed by region number
        """

        # arguments
        if image is None:
            image = self.image
        if regions is None:
            regions = self.regions

        # density
        dens = Density()
        dens.calculate(image=image, segments=regions)

        # set attributes
        return dens

    def getGroupDensity(self, regionDensity, groups=None, boundThick=None, 
                        exclude=0):
        """
        Calculates basic density statistics (mean, std, min, max, volume) for
        grupes of regions.

        Groups of regions are defined by arg groups, a dictionary having group
        names as kays and lists (arrays) of ids as values. If arg groups is
        not specified, the default groups are used. These are: 'cleft' with
        (arg) exclude number of regions excluded from each side, 'bound_1' and
        'bound_2' (boundary 1 and 2) containing (arg) boundThick regions from
        the cleft sides.

        Arguments boundThick and exclude can be specified as lists (arrays) 
        of two ints: first for boundary 1 / cleft regions facing boundary 1 
        and second for boundary 2 / cleft regions facing boundary 2, or as
        sigle ints in which case they're used for both boundaries / cleft
        sides.

        Either groups, or boundThick and exclude need to be specifed.

        Requires self.makeRegions() to be executed earlier, in order to use
        attributes set by it.

        Arguments:
          - regionDensity: (pyto.segmentation.Density): region density
          - groups: dictionary with group names (strings) as keys and region
          ids (lists or arrays) as values
          - boundThick: (int or a list of two ints) boundary thicknes
          - exclude: (int or a list of two ints) number of end regions to
          exclude from the cleft

        Returns:
          - group_density: (dictionary) in the form {name:density, ...}
            - name: group name
            - density (pyto.segmentation.Density) group density stats in the
            non-array form (e.g. density.mean is a single number) 
          - group_ids: dictionary of the same form as the argument groups
        """

        # form group ids
        if groups is None:
            groups = {
                'cleft' : self.getCleftLayerIds(exclude=exclude),
                'bound_1' : self.getBound1LayerIds(thick=boundThick[0]),
                'bound_2' : self.getBound2LayerIds(thick=boundThick[1]),
                'all' : self.getLayerIds()}
        group_ids = groups.values()

        # group density
        group_dens = regionDensity.aggregate(ids=group_ids)
        group_density = [(key, group_dens.extractOne(id_=id_, array_=False)) 
                         for key, id_ 
                         in zip(groups.keys(), range(1, len(groups)+1))] 
        group_density = dict(group_density)

        return group_density, groups

    def getMinCleftDensity(self, layerDensity, exclude=0, boundThick=0, 
                        reference=None):
        """
        Calculates minimum cleft layer density, the layer id of that layer, its
        position within the cleft and the relative minimum cleft layer
        density in respect to the cleft and reference layer densities.

        Excluded layers are not taken into account for finding min layer.

        If reference is not given, cleft density (not including excluded layers)
        and boundary density (using up to thick boundaries starting from those
        facing the cleft)

        Returns: (min_density, relative_min_density, min_layer_id, 
        min_layer_position):
          - min_density, relative_min_density: mean density of the layer with 
          lowest mean density, absolute and relative respectivly
          - min_layer_id, min_layer_position: id and relative postion of the 
          min density layer. Note that these two are arrays that contain more
          than one element in case there's more than one layer with min density.
        """

        # get cleft ids
        reduced_cleft_ids = self.getCleftLayerIds(exclude=exclude)
        cleft_ids = self.getCleftLayerIds(exclude=0)

        # get min density
        reduced_cleft_density = layerDensity.mean[reduced_cleft_ids]
        min_dens = reduced_cleft_density.min()

        # find position(s) of min density
        min_layer_reduced = numpy.flatnonzero(reduced_cleft_density == min_dens)
        min_layer_id = reduced_cleft_ids[min_layer_reduced]

        # find fractional position(s) of min density
        min_layer_frac = (min_layer_id - cleft_ids[0] + 0.5) \
            / (cleft_ids[-1] + 1 - cleft_ids[0])

        # get density
        if reference is None:
            reference = self.getBoundLayerIds(thick=boundThick)
        agreg_dens = layerDensity.aggregate(ids=[reduced_cleft_ids, 
                                                      reference])

        # get relative density of the min
        rel_min_dens = (min_dens - agreg_dens.mean[2]) \
            / (agreg_dens.mean[1] - agreg_dens.mean[2])

        return (min_dens, rel_min_dens, min_layer_id, min_layer_frac)
        
    ###################################################################
    #
    # Points
    #
    ####################################################################

    def getPoints(self, ids, mode='all', connectivity=1, distance=2,
                  format_='coordinates'):
        """
        Returns coordinates of selected elements (points) of cleft regions 
        identified by by arg ids. 

        Returns coordinates of selected elements (points) of segments labeled 
        by arg ids.

        If mode is 'all', coordinates of all points are returned.

        If mode is 'geodesic', the points are selected so that they are not
        closer than the argument distance.

        Respects inset attribute, that is the returned coordinates are given 
        for the full size array self.data. In addition, it works internally 
        with the smallest subarray of self.data that contains all ids.

        Calls ..segmentation.Labels.getPoints().

        Arguments:
          - ids: (list or ndarrays) ids, or a sigle (int) id 
          - mode: determines how the points are selected
          - distance: min distance between selected points (needed if mode is
          'geodesic')
          - format_: output format; 'numpy' for the format used by 
          numpy.nonzero(), or 'coordinates' for a 2D array where the first 
          index denotes different points, and the second the coordinates of the
          point.
        """

        points = self.regions.getPoints(
            ids=ids, mode=mode, connectivity=connectivity, 
            distance=distance, format_=format_)

        return points
