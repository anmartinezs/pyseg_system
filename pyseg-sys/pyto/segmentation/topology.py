"""

Contains class Topology for the calculation of topological properties of 
segmented images.


Basic usage:

# caclulate
topo = Topology(segments=segment_object)
topo.calculate()

# show some of the results
topo.euler
topo.nLoops
topo.nHoles

# Author: Vladan Lucic (MPI for Biochemistry)
# $Id: topology.py 1216 2015-08-12 16:40:17Z vladan $
"""

__version__ = "$Revision: 1216 $"
#__version__ = "$Revision: 1216 $"[1:-1].replace('Revision:', '').strip()


import sys
import logging
import inspect
import numpy
import scipy
import scipy.ndimage as ndimage

from features import Features
from labels import Labels
from segment import Segment
from hierarchy import Hierarchy

class Topology(Features):
    """
    Basic topological properties of one or more segments from a segmented image.

    Typical usage:

      topo = Topology(segments=segmented_image)
      topo.calculate()

    This calculates and sets the following attributes (N is the 
    dimensionality of the segmented image):

      - nFaces: (2-dim ndarray) number of 0-dim to N-dim faces (basic 
      simplexes), indexed by segment ids on axis 0 and face dimensionality 
      on axis 1        
      - euler: (ndarray, indexed by segment ids) Euler characteristics
      - homologyRank: (2-dim ndarray) rank of the homology for dimensionality 
      from 0 to N, indexed by segment ids on axis 0 and the dimensionality 
      on axis 1 
      - nObjects: (ndarray, indexed by segment ids) number of disconnected 
      objects, the same as homology rank for dimension 0
      - nLoops: (ndarray, indexed by segment ids) number of topologically 
      independent loops, the same as homology rank for dimension 1
      - nHoles: (ndarray, indexed by segment ids) number of holes, the same as
      homology rank for dimension 2

    """
    
    #############################################################
    #
    # Initialization
    #
    #############################################################

    def __init__(self, segments=None, ids=None):
        """
        Initializes attributes.

        Arguments:
          - segments: (Segment) segments
          - ids: segment ids (if None segments.ids used)
        """

        # call super
        super(Topology, self).__init__(segments=segments, ids=ids)

        # data names
        self.dataNames = ['nFaces', 'euler', 'homologyRank']

    def setSegments(self, segments, ids=None):
        """
        Sets attributes:
          - segments
          - ids
          - structEl: ndim structure element of connectivity 1
          - invStructEl: inverse structure element (ndim of rank ndim)
        """

        # call super
        super(Topology, self).setSegments(segments=segments, ids=ids)

        # set structuring elements
        if self.segments is not None:
            self.structEl = ndimage.generate_binary_structure(rank=self.ndim, 
                                                              connectivity=1)
            self.invStructEl = \
                ndimage.generate_binary_structure(rank=self.ndim, 
                                                  connectivity=self.ndim)
            # warn if segment.structEl is different?

    def initializeData(self):
        """
        Sets all data structures to length 0 ndarrays
        """

        # set everything to 1D float
        super(Topology, self).initializeData()

        # correct
        setattr(self, 'euler', numpy.zeros(shape=(0,), dtype=int))
        for name in ['nFaces', 'homologyRank']:
            setattr(self, name, numpy.zeros(shape=(0,0), dtype=int))


    ############################################################
    #
    # Data
    #
    #############################################################

    def getNFaces(self):
        """
        Returns number of faces as an ndarray indexed by ids (axis 0) and
        face dimensionality (axis 1).
        """
        return self._nFaces

    def setNFaces(self, nFaces):
        """
        Sets number of faces

        Argument:
          - nFaces: (ndarray) faces indexed by ids (axis 0) and face
          dimensionality (axis 1).
        """
        self._nFaces = numpy.asarray(nFaces)

    nFaces = property(
        fget=getNFaces, fset=setNFaces, 
        doc="Number of faces indexed by ids (axis 0) and dimension (axis 1)")
    
    def getHomologyRank(self):
        """
        Returns ranks of the Homology group as an ndarray indexed by ids
        (axis 0) and order (axis 1).
        """
        return self._homologyRank

    def setHomologyRank(self, homologyRank):
        """
        Sets ranks of the Homology group.

        Argument:
          - homologyRank: (ndarray) the Homology group ranks indexed by ids 
          (axis 0) and order (axis 1).
        """
        self._homologyRank = numpy.asarray(homologyRank)

    homologyRank = property(
        fget=getHomologyRank, fset=setHomologyRank,
        doc="Ranks of the Homology group indexed by ids (axis 0) and " 
        + "order (axis 1)")
    
    def getNObjects(self):
        """
        Returns number of objects as an ndarray indexed by ids. 
        """
        return self._homologyRank[:,0]

    def setNObjects(self, nObjects):
        """
        Sets number of objects.

        Argument:
          - nObjects: (ndarray) faces, 
        """
        self._homologyRank[:,0] = numpy.asarray(nObjects)

    nObjects = property(fget=getNObjects, fset=setNObjects, \
                         doc="Number of objects indexed by dimension")

    def getNLoops(self):
        """
        Returns number of independent closed loops as an ndarray indexed by ids.
        """

        if (self.ndim > 0) and (self.ndim <=3):
            return self._homologyRank[:,1]
        else:
            raise NotImplementedError, "Sorry, don't know how to caclulate the"\
                  + " number of independent loops for a " + str(self.ndim) \
                  + "-dimensional obect."

    def setNLoops(self, nLoops):
        """
        Sets number of independent closed loops .

        Argument:
          - nloops: (ndarray) number of loops, 
        """
        self._homologyRank[:,1] = numpy.asarray(nLoops)

    nLoops = property(fget=getNLoops, fset=setNLoops, \
                 doc="Number of closed independent loops, or the rank of the "\
                 + "fundamental group.")

    def getNHoles(self):
        """
        Returns number of holes (rank of ndim-1 Homology group) as an ndarray
        indexed by ids. 
        """

        if (self.ndim > 0) and (self.ndim <=3):
            return self._homologyRank[:,self.ndim-1]
        else:
            raise NotImplementedError, "Sorry, don't know how to caclulate the"\
                  + " number of holes for a " + str(self.ndim) \
                  + "-dimensional obect."

    def setNHoles(self, nHoles):
        """
        Sets number of holes (rank of ndim-1 Homology group).

        Argument:
          - nholes: (ndarray) number of holes, 
        """
        self._homologyRank[:,self.ndim-1] = numpy.asarray(nHoles)

    nHoles = property(fget=getNHoles, fset=setNHoles, \
                 doc="Number of holes, or the rank of ndim-1 Homology group.")


    #############################################################
    #
    # Calculations
    #
    ############################################################

    def calculate(self, ids=None):
        """
        Calculates basic topological and related properties.

        First the number of faces for all dimensions between 0 and N 
        (dimensionality of the segmented image, given by self.ndim) are 
        calculated (countFaces()). Euler characteristics is then calculated
        from the number of faces (getEuler()). Independently, homology ranks 
        for dim 0, self.N-1 and self.N are calculated. Finally, in 3D the
        number of topologically independent loops is calculated using 
        Euler-Poincare formula. All this is done for each segments separately.

        Sets self.euler (indexed by ids, index 0 is for total), self.nFaces and
        self.homology (axis 0 ids, axis i dimensions).
        """

        # set ids
        ids, max_id = self.findIds(ids=ids)

        # calculate everything that can be done directly
        self.getEuler(ids=ids)
        self.homologyRank = self.getHomologyRank(ids=ids)

        # calculate n loops for 3d
        if self.ndim == 3:
            n_other = numpy.zeros(shape=max_id+1, dtype='int')
            n_loops = numpy.zeros(shape=max_id+1, dtype='int')
            for i_dim in [0, 2, 3]:
                n_other += (-1)**i_dim * self.homologyRank[:,i_dim]
            n_loops[ids] = - self.euler[ids] + n_other[ids]
            n_loops[0] = sum(n_loops[id_] for id_ in ids)
            self.nLoops = n_loops

    def getEuler(self, ids=None):
        """
        Calculates Euler characteristics for segments specifed by ids.

        The Euler characteristics is calculated from the number of faces:

          euler = n_0-dim_faces - n_1-dim_faces + n_2-dim_faces - ...

        which in 3d becomes:

          euler = n_verteces - n_edges + n_surfaces - n_volumes

        Sets self.euler (indexed by segment ids) and self.nFaces (first index
        id, second index face dimensionality).
        """

        # set ids
        ids, max_id = self.findIds(ids=ids)

        # count faces
        self.nFaces = self.countFaces(ids=ids, dim=None)

        # calculate Euler
        euler = numpy.zeros(shape=max_id+1, dtype='int')
        for dim in range(0, self.ndim+1):
            euler += (-1)**dim * self.nFaces[:,dim]
        euler[0] = sum(euler[id_] for id_ in ids)
        self.euler = euler

        return euler

    def countFaces(self, ids=None, dim=None):
        """
        Calculates number of (arg) dim-dimensional faces.

        If dim is None, the number of faces is calculated for all dimensions 
        (from 0-faces to ndim-faces).

        Argument:
          - ids: segment ids
        """

        # set ids
        ids, max_id = self.findIds(ids=ids)

        # deal with no ids
        if max_id == 0:
            return numpy.zeros(shape=(max_id+1, self.ndim+1), dtype='int')
            
        if self.segments is not None:

            # single segments array
            if dim is None:

                # calculate all faces
                n_faces = numpy.zeros(shape=(max_id+1, self.ndim+1), 
                                      dtype='int')
                for i_dim in range(self.ndim+1):
                    n_faces_i = self.countFaces(ids=ids, dim=i_dim)
                    n_faces[:,i_dim] = n_faces_i
                    
            elif dim == 0:

                # 0-simplex
                n_faces = numpy.zeros(shape=max_id+1, dtype='int')
                n_faces_ids = ndimage.sum(input=1*(self.segments>0),  
                                          labels=self.segments, index=ids)
                n_faces[ids] = numpy.asarray(n_faces_ids).round().astype(int)
                n_faces[0] = sum(n_faces[id_] for id_ in ids)

            else:

                # loop over all dim>0 simplexes
                n_faces = numpy.zeros(shape=max_id+1, dtype='int')
                for simplex in self.generateBasicSimplexes(dim=dim):

                    # correlate
                    cc = ndimage.correlate(input=self.segments, 
                                           weights=simplex, mode='constant')

                    # extract number of faces
                    simp_size = simplex.size
                    n_faces[ids] += \
                        [((self.segments==id_) & (cc==id_*simp_size)).sum() \
                             for id_ in ids]
                n_faces[0] = sum(n_faces[id_] for id_ in ids)

        else:
            raise NotImplementedError("Sorry, dealing with Hierarchy object ",
                                      "hasn't been implemented yet.")

        return n_faces

    def generateBasicSimplexes(self, dim):
        """
        Generates basic simplexes of the specified dimension and yields them 
        one by one.

        Argument dim determines the dimensionality of simplexes, which can be 
        1 - N (dimensionality of the image). This generator is implemented 
        for 2 and 3 dimensional images. Maximal length of a simplex is 2. For 
        example, in 3D, the following simplexes are generated:
          - dim=1: line simplexes along major axes, each consisting of 2 
          elements, so 3 simplexes total 
          - dim=2: surface simplexes perpendicular to major axes, each 
          consisting of a 2x2 squares, so 3 simplexes total
          - dim=3: volume simplex, 2x2x2 cube

        Note that this method does not generate 0-dim simplex (single point).

        Yields (ndarray) simplex.
        """
            
        shape = [2] * self.ndim
        #sim = numpy.zeros(shape=shape, dtype='int32')

        # make permutations (should be done for any ndim)
        if self.ndim == 2:
            if dim == 1:
                combs = [[0,1], [1,0]]
            elif dim == 2:
                combs = [[1,1]]
        elif self.ndim == 3:
            if dim == 1:
                combs = [[0,0,1], [0,1,0], [1,0,0]]
            elif dim == 2:
                combs = [[0,1,1], [1,0,1], [1,1,0]]
            elif dim == 3:
                combs = [[1,1,1]]

        # generate simplexes
        for axes in combs:
            shape = numpy.array(axes) + 1
            #simplex = numpy.ones(shape=shape, dtype='int32')
            simplex = numpy.ones(shape=shape, dtype='int64')
            yield simplex

    def getHomologyRank(self, ids=None, dim=None):
        """
        Calculates the rank of the homology group for dimensionality dim.

        Currently implemented for dim = 0, self.ndim-1 and self.ndim (trivial).
        If dim is None, the ranks are calculated for the above three dim's 
        and the data is saved in self.homologyRank.

        Arguments:
          - ids: segment ids, if None self.ids is used
          - dim: dimensionality, 0 - self.ndim or None to calculate all

        Returns (ndarray) rank[id, dim].
        """

        # check ndim
        if dim is not None:
            if (dim > 0) and (dim < self.ndim-1):
                raise NotImplementedError("Sorry, don't know how to caclulate"\
                      + " rank of the " + str(dim) + "-Homology group for an "\
                      + str(self.ndim) + "-dimensional obect.")
                
        # ids
        ids, max_id = self.findIds(ids=ids)

        # deal with no ids
        if max_id == 0:
            return numpy.zeros(shape=(max_id+1, self.ndim+1), dtype='int')
            
        if self.segments is not None:

            # single segments array
            if dim is None:

                # recursively calculate all faces
                h_rank = numpy.zeros(shape=(max_id+1, self.ndim+1), dtype='int')
                for i_dim in [0, self.ndim-1, self.ndim]:
                    h_rank_i = self.getHomologyRank(ids=ids, dim=i_dim)
                    h_rank[:,i_dim] = h_rank_i
                return h_rank

            # get objects and expand them if a "non-existing" id > max id
            objects = ndimage.find_objects(self.segments)
            len_obj = len(objects)
            no_slice = self.ndim * [slice(0,0)]
            if len_obj <= max_id:
                for id_ in range(len_obj, max_id+1):
                    objects.append(tuple(no_slice))

            if dim == 0:

                # find separate segments
                h_rank = numpy.zeros(shape=(max_id+1), dtype='int')
                h_rank[ids] = \
                    [(ndimage.label(self.segments[objects[id_-1]]==id_, 
                                    structure=self.structEl))[1] \
                         for id_ in ids]
                h_rank[0] = sum(h_rank[id_] for id_ in ids) 

            elif dim == self.ndim-1:
                
                # find holes 
                h_rank = numpy.zeros(shape=(max_id+1), dtype='int')
                for id_ in ids:
                    data_inset = self.segments[objects[id_-1]]
                    filled = ndimage.binary_fill_holes(data_inset==id_, 
                                                 structure=self.invStructEl)
                    inter = (filled==True) & (data_inset==0) 
                    h_rank[id_] = (ndimage.label(input=inter, 
                                                 structure=self.structEl))[1]
                h_rank[0] = sum(h_rank[id_] for id_ in ids)
            
            elif dim == self.ndim:
                
                h_rank = numpy.zeros(shape=max_id+1, dtype='int')
            
            return h_rank

        else:
            raise NotImplementedError("Sorry, dealing with Hierarchy objects",
                                      " hasn't been implemented yet.")

    ######################################################
    #
    # Basic data manipulation 
    #
    #######################################################
    
    def restrict(self, ids):
        """
        Effectivly removes data that correspond to ids that are not specified 
        in the arg ids. Arg ids should not contain ids that are not in self.ids.

        Sets self.ids to ids and recalculates totals (index 0). Currently
        it doesn't actually remove non-id values from data.

        Argument:
          - ids: new ids
        """

        # set ids
        super(Topology, self).restrict(ids=ids)

        # remove non-id values?

        # set total
        self.setTotal()

    def setTotal(self):
        """
        Sets total values (index 0) for . The total value
        is the sum of all elements corresponding to self.ids.
        """

        ids = self.ids
        if (ids is None) or (len(ids) == 0):
            return

        for var in [self.euler, self.nFaces, self.homologyRank]:
            if var is not None:
                var[0] = var[ids].sum(axis=0)

