"""
 
Provides class Hiarrchy for the analysis of multiple segementations orgainized
in a hierarchy (each segmentation is a subset of the next one).

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id$
"""
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
from builtins import zip
from builtins import str
from builtins import range
#from past.utils import old_div
from past.builtins import basestring

__version__ = "$Revision$"


from copy import deepcopy
import logging
import numpy
import scipy
import scipy.ndimage as ndimage

from .labels import Labels
from .connected import Connected
from .statistics import Statistics
from .grey import Grey
import pyto.util.nested as nested

class Hierarchy(Labels):
    """
    A hierarchy contains segments that are orgainuzed in a strictly hierarchical
    manner, that is they satisfy the following requirements:
      1) Each segment is either superset (overlaps), subset or does not
      intersect any other segment. 
      2) Segments are organized in levels, all segments of a level taken
      together overlap all segmenents at one level below the level.   
    
    Adding and removing hierarchy levels:
      - addLevel: adds given segments to a specified level
      - extractLevelsGen: generator that yields levels
      - popLevel: removes top or bottom level and returns it 
      - extractLevel: returns specified level
      - removeLowerLevels: removes ids corresponding to all levels below a
      given level
      - removeHigherLevels: removes ids corresponding to all levels above a
      given level
      - remove: removes segments corresponding to the given ids

    Lower level id related methods:
      - getIdLevels: returns levels of specified ids
      - findHigherIds: finds ids directly above given ids 
      - findLowerIds: finds ids directly below given ids

    Analysis:
      - 
    
    Conversion:
      - toSegment(): converts this instance to Segment provided this instance
      is flat (no segment is above or below any other segment)

    Data structure attributes (should not be accessed directly):
      - self.ids: (numpy.ndarray) all ids
      eg: [2,4,5,6,12,16,19,22, ...]
      - self.levelIds: each element of this list is alist that contains all
      ids present at that level
      eg: [[2,4,5,6], [12,16,19], [22], ...]
      - self._higherIds: distionary of one level higher ids for each id
      eg: {2:12, 4:16, 5:16, 6:16, 12;22, 16:22, 22:None, ...} 
      - self._lowerIds: dictionary of (lists of) one level lower ids for each id
      inverse of self._higherIds)
      eg: {2:[], 3:[], 5:[], 12:[2], 16:[4,5,6], 19:[], 22:[12,16], ...}
      - self.data: ndarray of all segments

    There might be a number of level-defined properties, whose names are given
    in self.properties. 

    """

    ###############################################################
    #
    # Lower level manipulations (methods may access id structures directly)
    #
    ##############################################################

    def __init__(self, data=None, levelIds=[], higherIds={}):
        """
        Initializes id and data related attributes.

        Specifying args levelIds and higherIds is enough to make a functional 
        id-related attributes (data structures). In this case attributes
        ids, levelIds, _higherIds and _lowerIds are set. These arguments are 
        deepcopied, so they are not changed, nor they can be changed in this
        instance.

        Arguments:
          - levelIds: nested list of ids organized in levels
          - higherIds: dictionary of id : higher_id pairs
          - data: (ndarray) data array
        """

        # initialize super (in order to be able to call super.setDefaults, for 
        # example)
        super(Hierarchy, self).__init__(data)

        # initialize id-related structures
        self.levelIds = deepcopy(levelIds)
        self.ids = nested.flatten(self.levelIds)
        self.ids = numpy.asarray(self.ids, dtype='int')
        self._higherIds = deepcopy(higherIds)
        self.makeLowerIds()

        # initialize data
        self.data = data

        # initialize properties
        self.properties = []

    def makeLowerIds(self):
        """
        Makes a data structure that holds self._lowerIds using 
        self._higherIds.
        """
        
        self._lowerIds = {}
        for l_id in self._higherIds:
            h_id = self.getHigherId(l_id)
            if h_id > 0:
                try:
                    l_ids = self.getLowerIds(h_id)
                    l_ids.append(l_id)
                except KeyError:
                    l_ids = [l_id]
                self._lowerIds[h_id] = l_ids

    def orderLevelIds(self):
        """
        Orders ids in the self.levelIds so that on each level ids that have 
        higher ids precede those that do not. Also orders self.ids in the 
        same way.
        """

        # order the top level?

        # order levels below top
        for level in range(self.topLevel, 0, -1):

            # find ids at level-1 that have higher ids
            lower = self.findLowerIds(ids=self.getIds(level))
            lower = nested.flatten(lower)

            # find ids at level-1 that do not have higher ids
            other = set(self.getIds(level-1)).difference(lower)

            # put those that have higher first
            self.levelIds[level-1] = lower + list(other)

        # order ids in the same way
        self.ids = nested.flatten(self.levelIds)
        
    def getIds(self, level):
        """
        Returns ids that exist at the (argument) level in a flat list.

        Argument:
          - level: an int, or a slice (indexed by level)
        """
        if len(self.ids) == 0:
            return []
        if isinstance(level, slice):
            res = self.levelIds[level]
            return nested.flatten(res)
        else:
            # level should not be numpy.* type
            res =  self.levelIds[int(level)]
            return res
        
    def addIds(self, ids, level):
        """
        Adds ids at the level. If ids is None or an empty list (ndarray), 
        [] is added.
 
        Arguments:
          - ids: single or a list (ndarray) of ids to be added
          - level: id level
        """

        # convert ids to ndarray
        if ids is None:
            ids = numpy.array([])
        elif isinstance(ids, list) or isinstance(ids, numpy.ndarray):
            ids = numpy.asarray(ids)
        else:
            ids = numpy.array([ids])

        # add to id structures
        if self.ids is None:
            self.ids = numpy.array([], dtype='int')
        self.ids = numpy.append(self.ids, ids)
        if self.levelIds is None:
            self.levelIds = []
        self.levelIds.insert(level, ids.tolist())

    def getIdLevels(self, ids):
        """
        Returns levels of given ids.

        Argument:
          - ids: one or a list of ids
        """
        if isinstance(ids, list) or isinstance(ids, numpy.ndarray):
            levels = [self.getIdLevels(id_) for id_ in ids]                

        else:
            id_ = ids
            for level in range(len(self.levelIds)):
                if id_ in self.levelIds[level]:
                    return level
            else:
                return None
                
        return levels

    def checkIds(self):
        """
        Checks if the same id appears at different levels.
        """
        flat_l_ids = nested.flatten(self.levelIds)
        if len(flat_l_ids) == len(set(flat_l_ids)):
            return True
        else:
            return False

    def getHigherId(self, id_, check=False):
        """
        Returns id directly above given id_, or 0 if there is no higher
        level id.

        Raises KeyError if id_ is not in self.ids.

        Argument:
          - id: segment id
        """

        # check (prevents infinite loop in findHigherIds)
        if check and not self.checkIds():
            raise ValueError("Id-structures are not consistent")

        try:
            higher = self._higherIds[id_]
        except KeyError:
            if id_ in self.ids:
                # not sure what's better: 0 or None
                higher = 0
            else:
                raise

        return higher

    def getHighestBranchId(self, id_):
        """
        Finds the highest id on a branch to which argument id_ belongs.

        A branch is a part of the id hierarchy (tree) that contains no branching
        points, that is all ids on a branch have strict higher-lower relation.

        Argument:
          - id_: segment id
        """

        hi = self.getHigherId(id_=id_)
        if (hi is None) or (hi == 0):

            # no higher id exist
            return id_

        else:
            lo = self.getLowerIds(id_=hi)
            if len(lo) > 1:

                # higher id is a branching point
                return id_

            else:

                # higher id ok, try searching up
                return self.getHighestBranchId(id_=hi)

    def findHigherIds(self, ids, level=None, mode='single'):
        """
        Returns ids that are directly above (argument) ids.

        If mode is single, for each element of ids an id is found that is 
        directly above the given id, and it belongs either to the specified 
        level, or to the first higher level (if level is None). A single id is 
        returned if argument ids is a sigle id, or a list of (higher) ids that 
        correspond to argument ids is returned if argument ids is a list.

        If mode is all, for each element of argument ids all ids directly above 
        up to (including) the argument level are returned. If arg level is None,
        all ids above the specified arguments are returned. If argument ids
        is a single number a list of ids is returned, while if it is a list, a 
        nested list corresponding to the specified ids is returned.

        None is specified in the return for each id that has no higher ids (at
        the required level) in the single mode, and empty list in the all mode.
        Note that this differs slightly from the return of getHigherId where
        0 is used insted of None. 

        Argument:
          - ids: snigle id, or a list, of ids
          - level: None for one level up, or (int) level 
          - mode: 'single' or 'all'

        Return: higher id(s) as a single number a list or a nested list,
        depending on the argument ids and the mode.
        """

        # list of ids
        if (isinstance(ids, list) or isinstance(ids, tuple) 
            or isinstance(ids, numpy.ndarray)):
            return [self.findHigherIds(id_, level, mode) for id_ in ids]

        # single id, some initialization
        id_ = ids
        id_level = self.getIdLevels(id_)

        # find how many levels up from the id_ level we need to go
        if level is None:
            if mode == 'single':
                levels_up = 1
            elif mode == 'all':
                levels_up = self.topLevel - id_level
            else:
                raise ValueError(
                    "Argument mode has to be either 'all' or 'single'.")
        else:
            levels_up = level - id_level

        # find higher id(s)
        h_id = id_
        result = []
        for count in range(levels_up):
            h_id = self.getHigherId(h_id)

            if mode == 'single':
                if (h_id is None) or (h_id == 0):
                    result = None
                    break
                else:
                    result = h_id
            
            elif mode == 'all':
                if (h_id is None) or (h_id == 0):
                    break
                else:
                    result.append(h_id)

        return result

    def getLowerIds(self, id_, mode='single'):
        """
        Returns list of ids that are one level directly below given id if mode
        is 'single'. If there are no ids below id_, returns empty list. If mode
        is 'all', ids from all lower levels are returned in a (flat) list. 

        Raises KeyError if id_ is not in self.ids.

        If mode is 

        Argument:
          - id: segment id
        """        

        # one level lower
        try:
            lower = self._lowerIds[id_]
        except KeyError:
            if id_ in self.ids:
                return []
            else:
                raise

        if mode == 'single':

            # one level down
            return lower

        elif mode == 'all':
           
            # all levels down
            res = deepcopy(lower)
            curr_l = lower
            while len(curr_l) > 0:
                curr_l = [self.getLowerIds(lower_id) for lower_id in curr_l]
                curr_l = nested.flatten(curr_l)
                res.extend(curr_l)
            return res

        else:
            raise ValueError(
                "Argument mode has to be either 'all' or 'single'.")

    def findLowerIds(self, ids, exclude=None, mode='single'):
        """
        Returns ids that are one level lower than ids.

        For each element of ids, a list of ids that are directly below the ids
        is generated. The lower ids can not belong to exclude and belong to the
        highest level possible under these conditions. 

        If mode is 'single' only ids that are one level below the given ids 
        are returned. In this case, a nested list is returned where each sublist
        correspond to an element of ids. If is is a single int, a flat list is 
        returned.

        If mode is 'all', all ids below the given ids are returned in a 
        (flat) list.

        Argument:
          - ids: single id, or a list, of ids
          - exclude: list of ids that are not accepted for higher ids
          - mode: 'single' or 'all'

        Returns: nested list indexed by ids.
        """

        # check for single vs all
        if mode == 'single':

            if (exclude is None) or (len(exclude == 0)):
                if isinstance(ids, list) or isinstance(ids, numpy.ndarray):

                    # id list, nothing to exclude
                    return [self.getLowerIds(id_) for id_ in ids]

                else:

                    # single id, nothing to exclude
                    return self.getLowerIds(ids)

            else:
                if isinstance(ids, list):

                    # id list, exclude
                    return [self.findLowerIds(ids=id_, exclude=exclude)
                            for id_ in ids]

                else:

                    # single id, exclude, minimal number of levels down
                    res = []
                    while ((ids is not None) and isinstance(ids, list) 
                           and (len(ids) > 0)):
                        all_l_ids = self.findLowerIds(ids)
                        all_l_ids = nested.flatten(all_l_ids)
                        good_l_ids = [id_ for id_ in all_l_ids 
                                      if exclude.count(id_) == 0] 
                        res.extend(good_l_ids)
                        ids = [id_ for id_ in all_l_ids 
                               if exclude.count(id_) > 0] 
                    return res

        elif mode == 'all':

            # run this method in 'single' mode as many times as needed
            curr_ids = ids
            result = []
            while(True):
                curr_ids = self.findLowerIds(ids=curr_ids, exclude=exclude, 
                                             mode='single')
                curr_ids = nested.flatten(curr_ids)
                result = curr_ids + result
                if len(curr_ids) == 0:
                    return result

        else:
            raise ValueError("Mode " + mode + " is not understood. Acceptable "
                             + "values are 'single' and 'all'.") 

    def getRepresentedLowerIds(self, id_, mode='any'):
        """
        Returns an arbitrary / all id(s) that is (are) below id_ and is (are)
        represented. A id is represented if it exists in the data array, that 
        is it is larger than id(s) directly below it.

        Arguments:
          - id_: segment id
          - mode: determines how many ids are returned

        Returns an id if mode is 'any' or a list of ids if mode is 'all'

        In progress
        """

        lower_ids = self.getLowerIds(id_, mode='all')

        if mode == 'any':

            # return first id found or []
            #if (self.data == id_).sum() > 0:
            #    return id_
            for l_id in lower_ids:
                if (self.data == l_id).sum() > 0:
                    return l_id
            return None

        elif mode == 'all':

            # return all represented ids
            return [l_id for l_id in lower_ids if (self.data == l_id).sum() > 0]

        else:
            raise ValueError("Argument mode can be 'any' or 'all' but not " + mode) 
        
    def addHigherIds(self, lower, higher):
        """
        Sets higher to be higher ids of lower (ids) and adds them to
        the existing lower-higher id pairs.

        Arguments:
          - lower: single id or a list
          - higer: single id or a list
          """

        # initialize if needed
        if self._higherIds is None:
            self._higherIds = {}

        # add
        if isinstance(lower, list) or isinstance(lower, numpy.ndarray):
            if (not isinstance(higher, list)) \
                   and (not isinstance(higher, numpy.ndarray)):
                higher = [higher]*len(lower)
            self._higherIds.update(list(zip(lower, higher)))
        else:
            self._higherIds[lower] = higher

    def removeIds(self, ids=None, level=None, clean=False):
        """
        Removes given ids, or all ids belonging to a given level from all
        id-related structures. Also establishes lower-higher relationship
        between the remaining ids that were affected by the removal of ids. 

        Either ids or level has to be specified. If ids is given, level is
        ignored and the resulting instance keeps the same number of levels. 
        If level is specified, also the level(s) are removed from
        id structures, so that the resulting instance has smaller number
        of levels

        If clean is True, all levels with no ids are removed

        Note: unless whole levels are removed, the resulting lower-higher
        id relations between ids of non-adjacent levels are likely to appear.

        Arguments:
          - ids: list of ids to be removed
          - level: (int or a list of int's) level(s) to be removed
          - clean: flag indicating if levels with no ids are removed
        """

        # set variables from arguments
        if ids is not None:
            if isinstance(ids, list) or isinstance(ids, numpy.ndarray):
                pass
            else:
                ids = [ids]
            level = None
        elif level is not None:
            if isinstance(level, (list, numpy.ndarray)):
                ids = [self.getIds(lev) for lev in level]
                ids = nested.flatten(ids)
            else:
                ids = self.getIds(level)
        else:
            raise ValueError("Either ids ot level has to be given.")

        # in each item that has any of ids as a value, replace the value by the
        # corresponding higher id or remove the item if no higher id
        # Note: need to keep self._higherIds and self._lowerIds consistent
        for rm_id in ids:

            # get and remove higher and lower ids of the current id 
            high_id = self._higherIds.pop(rm_id, -1)
            low_ids = self._lowerIds.pop(rm_id, [])
            if high_id > 0:

                # higer ids exist, connect it with the lower ids
                for l_id in low_ids:
                    self._higherIds[l_id] = high_id
                self._lowerIds[high_id].remove(rm_id)
                self._lowerIds[high_id].extend(low_ids)

            else:

                # no higher id, just remove it
                [self._higherIds.pop(l_id) for l_id in low_ids]
                                    
        # generate self._lowerIds 
        # probably not needed, see above (25.09.08)
        self.makeLowerIds()

        # remove from self.ids
        self.ids = numpy.setdiff1d(self.ids, ids)

        # remove from self.levelIds
        if level is None:
            self.levelIds = \
                 [ [id_ for id_ in l_ids if not id_ in ids] \
                   for l_ids in self.levelIds]
        else:
            self.removeIdLevels(level)

        # remove empty levels
        if clean:
            self.levelIds = [curr_lev for curr_lev in self.levelIds \
                                 if len(curr_lev) > 0]

    def removeIdLevels(self, levels):
        """
        Removes levels from self.levelIds.

        Modifies levels if it is a list.

        Argument:
          - levels: single level, or a list of levels
        """

        if isinstance(levels, list):
            levels.sort(reverse=True)
            [self.levelIds.pop(lev) for lev in levels]
        else:
            self.levelIds.pop(levels)

    def getTopLevel(self):
        """
        Returns position of the top level (determined from self.levelIds), 
        or None if no level exists. 
        """
        if self.levelIds is None:
            return None
        top = len(self.levelIds) - 1
        if top == -1:
            return None
        else:
            return top
    topLevel = property(fget=getTopLevel, doc='Position of the top level')

    def removeData(self, ids=None, level=None):
        """
        Remove segments labeled by ids from data. Elements of each removed 
        segment are assigned to a segment directly obove.

        Either ids or level has to be specified. If ids is given, level is
        ignored.

        Arguments:
          - ids: list of ids to be removed
          - level: (int) level
        """

        # check ids
        if ids is None:
            ids = self.getIds(level)
        if (not isinstance(ids, list)) \
                and (not isinstance(ids, numpy.ndarray)):
            ids = [ids]

        # find replacing (higher) ids (0 if no higher id)
        higher_ids = self.findHigherIds(ids=ids)
        for ind in range(len(higher_ids)):
            if higher_ids[ind] is None:
                higher_ids[ind] = 0

        # make a self-consistent dictionary
        order = dict(list(zip(ids, higher_ids)))
        order = nested.resolve_dict(order)
        
        # replace and remove
        self.data = self.reorder(order=order, data=self.data)
        
    def addData(self, segment, level, check=False):
        """
        Adds data given by segment to (arg) level, so that the segment is
        placed above level-1 and below level (of self.data).

        Updates current data (self.data), but does not update id-related 
        structures. Positioning of segment and self (attribute inset) has to be
        the same. 

        Important: check can be used only if this instance forms a perfect
        herarchy, that is all segments are below the highest segment.

        Arguments:
          - segment: segment to be added, where segment.data is the 
          actual array
          - level: level at which segment is added
          - check: flag indicating if segment fits the current hierarchy at
          the specified level 
        """

        # no data
        if self.data is None:
            self.data = segment.data
            return
            
        # make hierarchy containing only levels below level 
        if level == 0:
            below = self.__class__()
            below.data = numpy.zeros_like(self.data)
        else:
            below = self.removeHigherLevels(level-1, new=True)

        # check
        if check:

            # make hierarchy containing only levels above level 
            if level == self.topLevel+1:
                above = self.__class__()
                above.data = numpy.zeros_like(self.data) + 1
            else:
                above = self.removeLowerLevels(level, new=True)

            # check
            if (((above.data<=0) & (segment.data>0)).any()) or \
                   (((below.data>0) & (segment.data<=0)).any()):
                raise ValueError(("Segment data do not fit this hierarchy at "
                                  + "level %d.") % level)

        # add data without overwriting lower levels
        mod_segment = numpy.where(below.data>0, 0, segment.data)
        self.data = numpy.where(mod_segment>0, mod_segment, self.data)

    def findOverlapIds(self, segment, level, mode='above', check=False):
        """
        Finds ids from segment that overlap ids at level-1 if mode is 'above',
        or ids from level that overlap segment ids if mode is 'below'.

        This object and segment have to be aligned (inset attribute should be
        the same). 

        Arguments:
          - segment: (Segment) segments
          - level: the level where segment is inserted
          - mode: 
          - check: flag indicating if consistency checks are preformed

        Return:
          - list of segment ids, ordered according to self.getIds(level-1) if
          mode is 'above', or a list of level-ids ordered by segment ids. 
        """
        
        # find overlap
        # Note: hudge preformance improvement using find_objects 
        if mode == 'above':
            
            # remove levels below level-1
            lower = self.removeLowerLevels(level=level-1, new=True)

            # find segment ids that overlap with extended level-1 segments
            level_minus_1_ids = self.getIds(level-1)
            slices = ndimage.find_objects(lower.data)
            overlap_ids = \
                [segment.data[slices[id_-1]][lower.data[slices[id_-1]]==id_] \
                     for id_ in level_minus_1_ids]

            # slower than above (19.09.08)
            # find ids at level-1
            #ll_ids = self.getIds(level=level-1)

            # replace each level-1 id that is not represented in data by any of
            # its lower ids that is represented
            #lower_ids = []            
            #for l_id in ll_ids:
            #    if (self.data == l_id).sum() > 0:
            #        lower_ids.append(l_id)
            #    else:
            #        rep_id = self.getRepresentedLowerIds(l_id, mode='any')
            #        lower_ids.append(rep_id) 

            # for each id get a corresponding segment id
            #try:
            #    max_lower_id = max(lower_ids)
            #except ValueError:
            #    max_label = 1
            #    overlap_ids = []
            #slices = ndimage.find_objects(self.data, max_label=max(lower_ids))
            #sl = slices[id_-1]
            #overlap_ids = \
            #    [segment.data[sl][self.data[sl]==id_] \
            #         for id_ in lower_ids]

        elif mode == 'below':

            # find level ids that overlap segment ids
            higher = self.removeLowerLevels(level=level, new=True)
            slices = ndimage.find_objects(segment.data)
            overlap_ids = \
                [higher.data[slices[s_id-1]]\
                     [segment.data[slices[s_id-1]]==s_id] \
                     for s_id in segment.ids]
            
        else:
            raise ValueError("Argument mode can be either 'above' or 'below'.")

        # check if segment fits this hierarchy
        if check:

            # check if there's exactly one overlaping id for each lower id
            overlap_ids_count = [len(numpy.unique(o_ids)) != 1 \
                                     for o_ids in overlap_ids]
            if True in overlap_ids_count:
                raise ValueError("Segments gien by the argument segment do ",
                                 "not fit with at level ", str(level), ".")

            # check if all overlaping ids belong to level
            if mode == 'below':
                overlap_id_set = set([o_id[0] for o_id in overlap_ids])
                if not overlap_id_set.issubset(set(self.getIds(level))):
                    raise ValueError("Segments gien by the argument segment " \
                          "do not fit at level " + str(level) + ".")

        # pick only 0-th elements 
        overlap_ids = [o_id[0] for o_id in overlap_ids] 

        return overlap_ids

    def findLevel(self, segment, mode='volume'):
        """
        Finds level at which segment can be inserted.

        Currently implemented mode='volume', determines level by comparing
        volumes at different levels with the total segment volume.

        Arguments:
          - segment: (Segment) segments
          - mode: currently only 'volume'

        Returns level
        """

        # no levels
        if self.topLevel is None:
            return 0
        
        if mode is None:

            # need to implement
            raise NotImplementedError("Sorry, mode=None not implemented yet.")

        elif mode == 'volume':
            
            # search for the highest level that have volume larger than
            # that of segment
            seg_vol = (segment.data > 0).sum()
            curr_hi = deepcopy(self)
            # workaround for bug #906 in numpy 1.1.1
            if numpy.__version__ == '1.1.1':
                try:
                    curr_hi.contacts._n._mask = \
                        deepcopy(self.contacts._n._mask)
                except AttributeError:
                    pass
            levels = list(range(self.topLevel+1))
            levels.reverse()            
            for lev in levels:

                # check if segment volume is larger that the current volume
                curr_vol = (curr_hi.data > 0).sum()
                if seg_vol > curr_vol:
                    return lev + 1 

                # remove current level segments
                curr_hi.remove(level=lev)

            else:
                return 0

        else:
            raise ValueError("Sorry mode has to be 'volume'.")

    ###############################################################
    #
    # Manipulation of level-associated properties
    #
    ##############################################################
   
    def addProperties(self, props, level):
        """
        Adds each of the specified level-associated properties to an
        appropriate (already existing or new) lists.

        If a list already exists, the new value is inserted at position level.

        Names (string) of variables and their values are given as key:value
        pairs in dictionary props. The variables are set to be attributes
        of this instance.

        For example:

        self.addProperties(self, {'me':22, 'you':33}, 0)
        self.addProperties(self, {'me':23, 'you':34}, 1)

        will create the following:

        self.me = [22, 23]
        self.you = [33, 34]

        Arguments:
          - props: dictionary containing property variable and value pairs
          - level: level
        """

        for name in props:

            # initialize if first time using var
            try:
                if getattr(self, name) is None:
                    setattr(self, name, [])
            except AttributeError:
                setattr(self, name, [])
            if name not in self.properties:
                self.properties.append(name)

            if name == 'contacts':

                # shift segment ids and insert contacts
                shift = props['shift']
                curr_contacts = props['contacts']
                try:
                    self.contacts.add(curr_contacts, shift=shift)
                except AttributeError:
                    self.contacts = curr_contacts

            else:

                # insert values of other variables
                var = getattr(self, name)
                var.insert(level, props[name])

    def removeProperties(self, level):
        """
        Removes all properites at given level(s).

        If more than one level need to be removed by repeated calls to this
        method, the order has to be from higher level to lower. Alternatively,
        if a multiple levels are specified, the given order doesn't matter 
        because it is internally set in descending order.

        Argument:
          - level: (list or numpy.ndarray) level(s)
        """

        # list of levels
        if isinstance(level, numpy.ndarray):
            level = level.tolist()
        if isinstance(level, list):
            level.sort()
            level.reverse()
            for curr_level in level:
                self.removeProperties(level=curr_level)
            return

        # single level
        for name in self.properties:

            if name == 'contacts':

                # contacts
                self.contacts.removeSegments(ids=self.getIds(level))

            else:

                # other properties
                var = getattr(self, name)
                new = [var[ind] for ind in range(len(var)) if ind != level]
                setattr(self, name, new)
            
    def extractProperties(self, ids=None, level=None):
        """
        Returns dictionary of properties, where keys are property names and
        values are property values for specified ids.

        Ids are specified by arg ids, or by arg level if ids are None.

        If contacts is one of the properties, the corresponding Contacts
        object is (deep)copied and saved in the resulting dictionary in 
        the original compactified / expanded state.

        Arguments:
          - ids: (list or array) ids
          - level: level

        Return: dictionary of properties
        """

        # ids
        if ids is None:
            ids = self.getIds(level)
        levels = self.getIdLevels(ids)

        #if len(ids) == 0: 
        #    return {}

        dict = {}
        for name in self.properties:

            if name == 'contacts':

                # contacts
                contacts = deepcopy(self.contacts)
                was_compactified = False
                if contacts.compactified:
                    contacts.expand()
                    was_compactified == True
                contacts.keepSegments(ids=self.ids)
                if was_compactified:
                    contacts.compactify()
                dict[name] = contacts

            else:

                # other properties
                var = getattr(self, name)
                var = numpy.asarray(var)
                if len(ids) > 0:
                    dict[name] = var[levels]
                else:
                    dict[name] = []

        return dict


    ###############################################################
    #
    # Adding and removing levels and individual segments (no direct access 
    # to id-structures)
    #
    ##############################################################

    def addLevel(self, segment, level=None, check=False, shift=None, props={}):
        """
        Inserts segment at the given level or at the appropriate level. Current
        levels higher or equal to level are pushed up.

        All id and data structures are updated to include added segments. 
        Respects positioning (attribute inset) of both this object and arg
        segment. Inset (data) of this object is enlarged to be able to
        fit the data af segment.

        The ids of segment are increased by shift before segment is added 
        to self.data. If shift is None the ids are increased by self.maxId,
        so that the added segments do not overlap with the current ones. If
        there's no data (self.maxId is 0) shift is set to 0.

        Modifies segment in the following ways. Removes all segments from 
        segment.data that are not listed in segment.ids. After that, shifts
        segment ids (both in segment.data and segment.ids) so that the shifted
        ids do not colide with the ids of this instance. Also, may change 
        positioning of segment so that it aligns with this object, 

        Adds 'shift' to self.properties and sets self.shift[level] to the
        current shift. Also adds other properties given in (arg) props
        using self.addProperties.

        Important: check can be used only if this instance forms a perfect
        herarchy, that is all segments are below the highest segment.

        Arguments:
          - segment: (instance of Segment) segments to be added
          - level: segment is added at this level 
          - check: flag for checking if segment fits this hierarchy at level 
          - shift: ids of segment are increased by this amount
          - props: dictionary of property names and values

        Returns lev.
        """

        # change ids in segment so that they differ from self.ids
        if shift is None:
            shift = self.maxId
        if self.maxId == 0:
            shift = 0
        if shift != 0:
            new_seg_ids = segment.ids + shift

            # check if ids collide
            if numpy.intersect1d(self.ids, new_seg_ids).size > 0:
                logging.warning("Ids of the segments to be added collide with "
                                + "the current ids.")
            # preformace problem
            #order = dict(zip(segment.ids, new_seg_ids))
            #segment.reorder(order, clean=True)
            segment.shiftIds(shift=shift)

        # clean segment
        segment.clean()

        # no current data
        if self.data is None:
            self.addIds(ids=segment.ids, level=0)
            self.addData(segment, level=0, check=check)
            self.inset = segment.inset
            props['shift'] = 0
            self.addProperties(level=0, props=props)
            return 0

        # adjust positioning
        big_inset = self.findEnclosingInset(inset=segment.inset)
        self.useInset(inset=big_inset, mode='abs', useFull=True, expand=True)
        segment.useInset(inset=big_inset, mode='abs', 
                         useFull=True, expand=True)

        # figure out level
        if level is None:
            level = self.findLevel(segment, mode='volume')
        elif isinstance(level, basestring):
            if level == 'top':
                level = self.topLevel+1
            elif level == 'bottom':
                level = 0
            else:
                raise ValueError("Argument level: " + level + " is not " \
                      "understood. Valid choices are 'top', 'bottom', and " \
                      " and integers.")
                
        # find segment ids that overlap with level-1 ids
        if level > 0:
            # preformace problem
            over_seg_ids = self.findOverlapIds(segment, level, check=check)

        # find level ids that overlap segment ids
        if level <= self.topLevel:
            over_level_ids = self.findOverlapIds(segment, level, mode='below',
                                                 check=check)

        # update data (needs to be done before updating id-related, but after
        # checks in findOverlapIds)
        # ToDo (perhaps): do everything on a new instance first
        self.addData(segment, level, check)

        # insert segment ids into the hierarchy at level (level ids are to be
        # pushed up to level+1)
        if level == 0:
            self.addHigherIds(segment.ids, over_level_ids)
        elif level <= self.topLevel: 
            self.addHigherIds(self.getIds(level-1), over_seg_ids)
            self.addHigherIds(segment.ids, over_level_ids)
        elif level == self.topLevel+1:
            self.addHigherIds(self.getIds(level-1), over_seg_ids)

        # update other id-structures
        self.makeLowerIds()
        self.addIds(ids=segment.ids, level=level)

        # update properties
        props['shift'] = shift
        self.addProperties(props=props, level=level)

        return level

    def remove(self, ids=None, level=None, new=False):
        """
        Removes both the ids (specified by ids or level) from id-related
        structures, and the segments labeled by those ids.  

        Elements of each removed segment are assigned to a segment directly
        above. Also establishes lower-higher relationship between the remaining
        ids that were affected by the removal of ids. 

        Either ids or level has to be specified. If ids is given, level is
        ignored and the resulting instance keeps the same number of levels. 
        If level is specified, the whole level is removed (data and ids), so
        the resulting instance has one level less.

        The ids are also removed from self.contacts.

        If new is True a new instance (with ids removed) is returned, while
        this instance is not modified. Otherwise this instance is modified.

        Arguments:
          - ids: list of ids to be removed
          - level: (int) level
          - new: flag indicating if new instance is generated
        """

        # make a copy of this instance if needed
        if new:
            inst = deepcopy(self)
            # workaround for bug #906 in numpy 1.1.1
            if numpy.__version__ == '1.1.1':
                try:
                    inst.contacts._n._mask = deepcopy(self.contacts._n._mask)
                except AttributeError:
                    pass
        else:
            inst = self

        # replace segments by higher (has to be done before ids are removed)
        inst.removeData(ids, level)

        # remove properties if removing a level (before removing ids)
        if (ids is None) and (level is not None):
            inst.removeProperties(level)

        # removes segments from contacts
        if (ids is not None) and ('contacts' in self.properties):
            inst.contacts.removeSegments(ids=ids)

        # remove ids from id structures
        inst.removeIds(ids, level)

        # return new instance if created
        if new:
            return inst
        
    def keep(self, ids, new=False):
        """
        Removes all segments other than those specified by arg ids. The
        segments are removed both from id-related structures and from the data
        array.

        Elements of each removed segment are assigned to a segment directly
        obove. Also establishes lower-higher relationship between the remaining
        ids that were affected by the removal of ids. 

        If new is True a new instance (with ids removed) is returned, while
        this instance is not modified. Otherwise this instance is modified.

        The ids are also removed from self.contacts.

        Based on remove().

        Arguments:
          - ids: list of ids to be removed
          - new: flag indicating if new instance is generated
        """

        # remove other (bad) ids
        bad_ids = numpy.setdiff1d(self.ids, ids)
        inst = self.remove(ids=bad_ids, new=new)

        if new: return inst
        
    def removeLowerLevels(self, level, new=False):
        """
        Removes segments that are below (argument) level. Segments, ids and
        properies (of the given level) are removed from data, id and property
        related structures.

        If new is True, a new instance is generated and returned and this
        instance is not modified. Otherwise, data structures of this instance
        are modified. 

        Arguments:
          - level: levels below this one are removed
          - new: flag indicating if a new instance is created or this one is
          modified

        Returns new instance if new is True
        """

        # make a copy if needed
        if new:
            inst = deepcopy(self)
            # workaround for bug #906 in numpy 1.1.1
            if numpy.__version__ == '1.1.1':
                try:
                    inst.contacts._n._mask = deepcopy(self.contacts._n._mask)
                except AttributeError:
                    pass
        else:
            inst = self

        # replace lower level ids by the appropriate level id
        # (faster, 18.09.08)
        if inst.topLevel > 0:
            rm_ids = []
            slices = ndimage.find_objects(inst.data)
            for h_id in inst.getIds(level=level):
                l_ids = inst.getLowerIds(id_=h_id, mode='all')
                inst.data = inst._remove(data=inst.data, remove=l_ids, 
                                         all=inst.ids, mode='remove', 
                                         value=h_id, slices=slices)

        # remove all other lower level ids
        rm_ids = []
        for l_level in range(level):
            rm_ids.extend(inst.getIds(level=l_level))
        inst.data = inst._remove(data=inst.data, remove=rm_ids, all=inst.ids)

        # slower, before 18.09.08
        # find all ids to be removed
        # rm_levels = range(level)
        #rm_ids = []
        #for curr_level in rm_levels:
        #    rm_ids.extend(inst.getIds(level=curr_level))
            
        # find higher ids of rm_ids on level
        #higher_ids = inst.findHigherIds(ids=rm_ids, level=level, mode='single')

        # remove levels in data
        #rm_order = dict(zip(rm_ids, higher_ids))
        #inst.data = inst.reorder(order=rm_order, data=inst.data)

        # remove properties and ids
        rm_levels = list(range(level))
        inst.removeProperties(level=rm_levels)
        inst.removeIds(level=rm_levels)

        # return if new
        if new: return inst

    def removeHigherLevels(self, level, new=False):
        """
        Removes segments that are above (argument) level. Segments, ids and
        properies (of the liven level) are removed from data, id and property
        related structures.

        If new is True, a new instance is generated and returned and this
        instance is not modified. Otherwise, data structures of this instance
        are modified. 

        Arguments:
          - level: levels above this one are removed
          - new: flag indicating if a new instance is created or this one is
          modified

        Returns new instance if new is True
        """
        
        # make a copy if needed
        if new:
            inst = deepcopy(self)
            # workaround for bug #906 in numpy 1.1.1
            if numpy.__version__ == '1.1.1':
                try:
                    inst.contacts._n._mask = deepcopy(self.contacts._n._mask)
                except AttributeError:
                    pass
        else:
            inst = self

        # remove levels from top down
        rm_levels = list(range(level+1, self.topLevel+1))
        rm_levels.reverse()

        # remove segments from data
        #rm_ids = []
        for cur_level in rm_levels:
            rm_ids = inst.getIds(level=cur_level)
            inst.data = self._remove(data=inst.data, remove=rm_ids,
                                     all=self.ids, mode='remove')

            # same as above but slower (16.09.08, r130)
            #rm_order = dict(zip(rm_ids, [0]*len(rm_ids)))
            #inst.data = inst.reorder(order=rm_order, data=inst.data)

            # remove properties
            #for cur_level in rm_levels:
            inst.removeProperties(cur_level)

            # remove ids
            inst.removeIds(level=cur_level)

        # return if new
        if new: return inst

    def extractLevel(self, level, new=False):
        """
        Extracts the specified level (arg level) from this instance.

        All data and id structures are updated accordingly.

        If new is True a new instance is returned and the current instance is 
        not modified. Otherwise, the current instance is modified to contain 
        one level only.

        The returned/modified instance is an instance of this (Hierarchy) class,
        but since it has only one level it is more like a Segment object,
        to which it can be converted using toSegment() method.

        Preformance note (09.09.08): When lot of segments are present
        getHigherId() is called many times and the check part of that method
        becomes a major preformance bottleneck (due to nested.flatten()).

        Arguments:
          - level: (int) level
          - new: flag that determines if a new instance is created
        """
        # make a copy if needed
        if new:
            inst = deepcopy(self)
            # workaround for bug #906 in numpy 1.1.1
            if numpy.__version__ == '1.1.1':
                try:
                    inst.contacts._n._mask = deepcopy(self.contacts._n._mask)
                except AttributeError:
                    pass
        else:
            inst = self

        # remove levels above and below this level
        inst.removeHigherLevels(level=level)
        inst.removeLowerLevels(level=level)

        # return if new
        if new: return inst

    def popLevel(self, level='top'):
        """
        Removes and returns the top or bottom level (depending on arg level) 
        from the current hierarchy.

        The current hierarchy is therefore left with one level less, all data 
        structures are adjusted. If the current hierarchy before calling this
        method has only one level, it will have no segments (levelIds = []) 
        after the call. 

        Argument:
          - level: 'top' for top level or 'bottom' for bottom level

        Returns: top or bottom level as a Segment object
        """

        if level == 'top':

            # extract top level
            extracted = self.removeLowerLevels(level=self.topLevel, new=True)

            # remove top level from this instance
            self.removeHigherLevels(level=self.topLevel-1, new=False)

        elif level == 'bottom':

            # extract bottom level data and ids (hopefully fast)
            extracted = self.removeHigherLevels(level=0, new=True)

            # left in case the above is too slow
            #extracted = deepcopy(self)
            #ids = extracted.getIds(level=0)
            #data = extracted._remove(data=extracted.data, keep=ids, 
            #                         all=extracted.ids)

            # set data, ids 
            #extracted.setData(data=data, copy=False, ids=ids)
            #extracted.levelIds = [extracted.ids.tolist()]
            #extracted._higherIds = {}
            #extracted.makeLowerIds()
            #extracted.removeProperties(level=range(1, self.topLevel+1))

            # remove bottom level from this instance
            self.removeLowerLevels(level=1, new=False)

        else:
            raise ValueError("Level ", level, " was not understood. Allowed ",
                             "values are 'top' and 'bottom'.")

        # convert to Segment
        seg = extracted.toSegment(copy=False)
        return seg

    def extractLevelsGen(self, order='<'):
        """
        Generator that removes and yields levels from the current object in the 
        order specified by arg order.

        Each extracted level is converted to Segment object. Each Segment object
        has attribute contacts (Contact object) containing info for segments of
        the Segment object. In addition, each Segment object has attributes
        named after properties of the current Hierarchy objects, only these 
        properties are converted from level-indexed (as in Hierarchy) to 
        ids-indexed. Notably, if the current object is an instance of 
        ThreshConn, it has (ids-indexed) attribute 'threshold'.

        After this method is executed, the current object has no segments and
        no levels.

        Argument:
          - order: 'ascend' ('<'), or 'descend' ('>')

        Yields: segment, level
          - segment: (Segment) extracted level
          - level: level that the segment had in this instance before this
          method was called
        """

        # get and order levels
        levels = list(range(self.topLevel+1))
        if (order == 'ascend') or (order == '<'):
            levels.sort()
            level_str = 'bottom'
        elif (order == 'descend') or (order == '>'):
            levels.sort()
            levels.reverse()
            level_str = 'top'
        else:
            raise ValueError("Order ", order, " was not understood. Allowed ",
                             "values are 'ascend' ('<') and 'descend ('>').")

        # adjust levels (an attempt to speed this up)
        #levels = numpy.asarray()
        #for ind in range(len(levels)):
        #    levels[ind+1:] = numpy.where(levels[ind+1:] > levels[ind], 
        #                                 levels[ind+1:] - 1, levels[ind+1:])

        # loop over levels
        for curr_level in levels:

            seg = self.popLevel(level=level_str)
            yield seg, curr_level 

        
    ##########################################################
    #
    # Segment analysis
    #
    ##########################################################

    def findNewIds(self, level=None, belowIds=None, below=None):
        """
        Finds segments that have no segment directly below them (new ids).
        This is subject to additional constraints defined by the arguments.

        If arg level is specified the returned segments are restricted to 
        those belonging to that level.

        If arg belowIds is specified, new ids that are below belowIds are 
        returned.

        If arg below is specified, the returned ids are restricted to
        those that have higher ids at the level given by arg below.
        Note that this may not be the same as taking all new segments at levels 
        below or at the arg below. For example, if a branch starts and
        finishes below level given by argument below, its new segments are
        not included.

        Note: only the first specified argument from belowIds and below is taken
        into account.

        Argument:
          - level: (int) level that returned ids belong to
          - belowIds: list of ids
          - below: (int) returned ids have to be below ids of this level or
          on this level.

        Returns (ndarray) segment ids.
        """
        
        if belowIds is not None:

            # restrict to those below belowIds
            below_ids = self.findLowerIds(ids=belowIds, mode='all')
            ids = below_ids + belowIds

        elif below is not None:

            # restrict to ids having a higher id at the below level
            at_below = self.getIds(level=below)
            below_ids = self.findLowerIds(ids=at_below, mode='all')
            ids = below_ids + at_below

        else:
            ids = self.ids

        # restrict to a level, if needed
        if level is not None:
            ids = numpy.intersect1d(ids, self.getIds(level))

        # pick only those that are not among higher ids
        higher_ids = list(self._higherIds.values())
        return numpy.setdiff1d(ids, higher_ids)

    def findNewBranchTops(self):
        """
        For each new id (id that has no ids below) finds the highest id that
        belongs to the same branch.

        A branch is a part of the id hierarchy (tree) that contains no branching
        points, that is all ids on a branch have strict higher-lower relation.

        Returns (ndarray) segment ids.
        """

        # ToDo: add arg below, just like in findNewIds()

        new = self.findNewIds()
        return [self.getHighestBranchId(id_=new_id) for new_id in new] 

    def isFlat(self):
        """
        Returns True if the segments are flat, that is if no segment
        is above (or below) any other segment.
        """

        h_ids = self.findHigherIds(ids=self.ids)
        for id_ in h_ids:
            if (id_ is not None) and (id_ > 0):
                return False
        return True

    def getNNew(self, level=None):
        """
        Returns number of segments that have no other segments derectly
        below at the given level, or at all levels if level is None.

        Argument:
          - level

        Returns: int if level is given, or a list of ints indexed by level
        if level is None.
        """

        if level is None:

            # all levels
            result = [self.getNNew(level=t_level) for t_level 
                      in range(self.topLevel+1)]
            return result

        else:

            # level specified
            return len(self.findNewIds(level=level))

    def findIds(self, func, kargs):
        """
        Returns ids for which function func applied on arguments arg retruns 
        True.

        Arguments:
          - func:
          - args: dictionary containing func arguments

        Returns: list of ids
        """

        # in progress
        return [id_ for id_ in self.ids if func(**kargs)] 

    def findIdsByVolume(self, min=None, max=None, volume=None):
        """
        Retturns ids of segments that have volume between min (inclusive) and
        max (exclusive).

        If min (max) argument is not specified min (max) limit is not imposed. 

        If (arg) volume is given segment volums are not calculated.

        Arguments:
          - min: minimum volume (inclusive)
          - max: maximum volume (exclusive)
          - volume: array of volumes indexed by ids
        """

        # get volume of all segments
        if volume is None:
            vol = self.getVolume()

        # impose limits
        ids = self.ids
        if min is not None:
            ids = [id_ for id_ in ids if vol[id_] >= min]
        if max is not None:
            ids = [id_ for id_ in ids if vol[id_] < max]

        return ids

    def findIdsBySVRatio(self, min=None, max=None, volume=None,
                         surface=None, thick=1):
        """
        Retturns ids of segments that have volube between min and max 
        (inclusive).

        Arguments:
          - min: minimum surface to volume ratio (inclusive)
          - max: maximum surface to volume ratio (exclusive)
          - volume: array of volumes indexed by ids
          - surface: array of surfaces indexed by ids
          - thick: thickness of surfaces
        """

        # get volume of all segments
        if volume is None:
            vol = self.getVolume()
        if surface is None:
            sur = self.getSurface(size=size)
        sv_ratio = sur / vol

        # impose limits
        ids = self.ids
        if min is not None:
            ids = [id_ for id_ in ids if sv_ratio[id_] >= min]
        if max is not None:
            ids = [id_ for id_ in ids if sv_ratio[id_] < max]

        return ids


    def getVolume(self, ids=None):
        """
        Finds volume of segments whose ids are specified, or of all segments
        id ids is None.

        Arguments:
          - ids: list of ids

        Returns:
          - if ids is None: ndarray of volumes indexed by ids (0-th element is 0)
          - if ids are given: list of volumes corresponding to ids
         """

        # get volumes of segment extensions over lower levels
        from .morphology import Morphology
        mor = Morphology(segments=self.data, ids=self.ids)
        vol = mor.getVolume()

        # for each segment add voulmes of all segments directly below the segment
        for level in range(1, self.topLevel+1):
            for id_ in self.getIds(level):
                for l_id in self.getLowerIds(id_):
                    vol[id_] += vol[l_id]

        # return
        if ids is None:
            vol[0] = 0
            return vol
        else:
            return vol[ids]

    def getSurface(self, ids=None, thick=1):
        """
        Finds surface of segments whose ids are specified, or of all segments
        id ids is None.

        Uses Segment.getSurface to obtain and calculate surfaces.

        Arguments:
          - ids: list of ids
          - thick: thickness of surface

        Returns:
          - if ids is None: ndarray of surfaces indexed by ids (0-th element is 0)
          - if ids a re given: list of surfaces corresponding to ids
         """

        # find levels to which ids belong
        if ids is None:
            levels = list(range(self.topLevel+1))
        else:
            levels = set(self.getIdLevels(ids))

        # get surfaces
        from .morphology import Morphology
        mor = Morphology()
        for level in levels:

            level_seg = self.extractLevel(level=level, new=True)
            curr_mor = Morphology(level_seg)
            curr_mor.getSurface(size=thick, copy=False)
            mor.merge(curr_mor)

        sur = mor.surface
        sur[0] = 0
        if ids is None:
            return sur
        else:
            return sur[ids]

    def findIdsByNBound(self, min=None, max=None, contacts=None):
        """
        Returns ids of segments that contact at least min and at most max
        boundaries.

        Arguments:
          - min/max: min/max number of contacted boundaries
          - contacts: (Contacts) contacts between segments and boundaries, if
          None self.contacts is used

        Returns: (list) segment ids
        """

        # count number of contacted boundaries for all segments
        n_bound = self.getNBound(contacts=contacts)

        # impose limits
        ids = self.ids
        if min is not None:
            ids = [id_ for id_ in ids if n_bound[id_] >= min]
        if max is not None:
            ids = [id_ for id_ in ids if n_bound[id_] <= max]

        return ids

    def getNBound(self, ids=None, contacts=None):
        """
        Finds number of contacted boundaries.

        Argument:
          - ids: (list, ndarray, or single int) segment id(s)
          - contacts: (Contacts) contacts between segments and boundaries, if
          None self.contacts is used

        Returns (int) number of boundary ids if arg ids is a sigle int, or a
        list if it is a list (or ndarray) 
        """

        # set contacts
        if contacts is None:
            contacts = self.contacts

        # get n bound
        if ids is None:
            n_bound = numpy.zeros(shape=self.maxId+1, dtype='int')
            for id_ in self.ids:
                n_bound[id_] = self.getBoundIds(id_, contacts=contacts).size
            return n_bound

        elif isinstance(ids, list) or isinstance(ids, numpy.ndarray):
            return [self.getNBound(id_, contacts=contacts) for id_ in ids]

        else:
            return self.getBoundIds(ids, contacts=contacts).size

    def getBoundIds(self, ids, contacts=None):
        """
        Return boundary ids that contact each of the specified segment ids.

        Argument:
          - ids: (list, ndarray, or single int) segment id(s)
          - contacts: (Contacts) contacts between segments and boundaries, if
          None self.contacts is used

        Returns ndarray of boundary ids if ids is a sigle int, or a list of 
        ndarrays if it is a list (or ndarray) 
        """
        if isinstance(ids, list) or isinstance(ids, numpy.ndarray):
            return [self.getBoundIds(id_, contacts=contacts) for id_ in ids]

        else:

            id_ = ids
            if contacts is None:
                contacts = self.contacts
            b_ids = contacts.findBoundaries(segmentIds=id_, nSegment=1,
                                            mode='at_least', update=False)
            return b_ids
        
    def getDensity(self, image, ids=None):
        """
        Calculates density-related statistics (mean, std, min, max).

        Only the values for segments whose ids are specified are calculated,
        but the returned object still have (wrong) values for other ids. If
        ids is None density for all ids is calculated. 

        Arguments:
          - image: ndarray of Image object containing a (grayscale) image
          - ids

        Returns Statistics objects with following attributes:
          - mean
          - std
          - min
          - max
        Each of these is an array indexed by ids (e.g. mean[3] is the mean value
        of segment 3). Elements at position 0 are set to 0. If ther is no ids, 
        all above attributes are None.
        """
        
        # figure out image
        if isinstance(image, Grey):
            image = image.data

        # find levels to which ids belong
        if ids is None:
            levels = list(range(self.topLevel+1))
        else:
            levels = set(self.getIdLevels(ids))

        # get density stats of segment extensions over all levels
        dens = Statistics()
        for level in levels:

            level_seg = self.extractLevel(level=level, new=True)
            dens.calculate(data=image, labels=level_seg.data, ids=level_seg.ids)

        # set elements at position 0 to 0
        try:
            dens.mean[0] = 0
            dens.std[0] = 0
            dens.min[0] = 0
            dens.max[0] = 0
        except TypeError:
            pass
        
        return dens
    

    ##################################################################
    #
    # Conversion
    #
    #################################################################
    
    def toSegment(self, copy=False):
        """
        Makes Segment object from this instance. Makes sense only of this
        instance is flat (no segment is above or below any other segment).

        Sets data, ids, threshold structEl and positional attributes. Data is 
        copied if argument copy is True. All other attributes are always copied.

        Also extracts values of all (level defined) properties corresponding
        to of this instance and saves them as attributes of the newly created
        Segment. These attributes are ndarrays indexed by ids (of the Segment
        object).

        Argument:
          - copy: if True a copy of self.data is made

        Returns Segment object.
        """
        
        # set data and ids
        from .segment import Segment
        seg = Segment(data=self.data, ids=self.ids, copy=copy)

        # set positionong
        seg.copyPositioning(self)

        # set thresholds
        #seg.thresh = self.thresh

        # set properties (indexed by ids)
        props = self.extractProperties(ids=seg.ids)
        seg.setProperties(props)

        # set structEl
        if self.structEl is not None:
            seg.structEl = deepcopy(self.structEl)

        # set contact and count se connectivities
        self.contactStructElConn = seg.contactStructElConn
        self.countStructElConn = seg.countStructElConn
        
        return seg
    

    ##################################################################
    #
    # Dendogram related
    #
    #################################################################
    
    def dendogram(
            self, mode='center', nodes=None, nodesize=2, ids=False,
            line_color='black', line_width=1.0, new_plot=True):
        """
        Plots dendogram of this hierarchy.

        Note: works only if package matplotlib.pyplot can be imported.

        ToDo:
          - show/label only new nodes
          - perhaps put the threshold part in ThreshConn
          - put constants to arguments
          - optimize v_lines

        Arguments:
          - mode: determines how an id (node) is positioned in respect to
          the ids that are directly below it, 'left', or 'center'
          - nodes: indicating if and which nodes (ids) are marked by circles
          - ids: flag indicating if nodes are labeled by ids 
        """

        # import matplotlib
        import matplotlib.pyplot as plt

        # make new instance and copy id-related structures
        ordered = self.__class__(levelIds=self.levelIds, 
                                 higherIds=self._higherIds)

        # order ids
        ordered.orderLevelIds()

        # calculate coordinates
        node, h_line, v_line = ordered.findDendogramCoords(mode=mode)

        # start new plot
        if new_plot: plt.figure()

        # plot nodes
        if nodes is None:
            node_ar = numpy.array(list(node.values()))
        elif nodes == 'all':
            node_ar = numpy.array(list(node.values()))
            plt.plot(node_ar[:,0], node_ar[:,1], linestyle='None', 
                     marker='o', markersize=nodesize)
        elif nodes == 'new':
            new_ids = self.findNewIds()
            node_ar = [value for key, value in list(node.items()) if key in new_ids]
            node_ar = numpy.array(node_ar)
            plt.plot(node_ar[:,0], node_ar[:,1], 'o', linestyle='None', 
                     markersize=nodesize)
        else:
            raise ValueError(
                'Argument nodes (' + nodes + ") was not understood."
                + " Available values are None, 'all' and 'new'.") 

        # label nodes by ids
        if ids:
            for (id_, pos) in node.items():
                plt.text(pos[0]+0.1, pos[1], str(id_)) 

        # plot lines
        for line in h_line:
            plt.plot(
                [line[0], line[1]], [line[2], line[3]], color=line_color, 
                linewidth=line_width)
        for line in v_line:
            plt.plot(
                [line[0], line[1]], [line[2], line[3]], color=line_color, 
                linewidth=line_width)

        # rest
        plt.axis([0, node_ar[:,0].max()+1, 0, ordered.topLevel+0.5])
        #plt.xticks([])
        #plt.yticks(range(ordered.topLevel+1))
        if not new_plot: return
        plt.ylabel('Level')
        try:
            thr = dict(list(zip(list(range(ordered.topLevel+1)), self.threshold)))
            thr_format = '%6.3f'
            tick_pos = numpy.asarray(plt.yticks()[0], dtype='int')
            tick_pos = [x for x in tick_pos if thr.get(x, None) is not None]
            tick_label = [(thr_format % thr[level_pos]) \
                              for level_pos in tick_pos] 
            plt.yticks(tick_pos, tick_label)
            plt.ylabel('Threshold')
        except AttributeError:
            pass

    def findDendogramCoords(self, mode='center'):
        """
        Assigns coordinates to all ids (segments) that are meant to be
        used for plotting dendrogram for this hierarchy.
        
        X coordinates start from 1 and have unit increment. Y coordinates are 
        simply the levels.

        Returns dictionary with ids as keys and corresponding positions 
        ([x_position, y_position]) as values. 

        ToDo:
          - Put constants to arguments
        """

        node = {}
                  
        # initial (left mode) top level nodes
        cur_pos = 1
        for id_ in self.getIds(self.topLevel):
            node[id_] = [cur_pos, self.topLevel]
            cur_pos += len(self.findNewIds(belowIds=[id_]))

        # initial (left mode) other level nodes
        for level in range(self.topLevel-1, -1, -1):

            # loop over ids at the current level
            cur_pos = 1
            for id_ in self.getIds(level):
                h_id = self.findHigherIds(ids=id_)

                # find position of the current id (node) 
                if h_id is not None:
                    cur_pos = max(cur_pos, node[h_id][0])
                node[id_] = [cur_pos, level]

                # update current position (for the next round)
                cur_pos += len(self.findNewIds(belowIds=[id_]))

        # adjust node coordinates to the mode
        if mode == 'left':
            pass

        elif mode == 'center':
            
            # center nodes in respect to the lower ones 
            for level in range(1, self.topLevel+1):
                for id_ in self.getIds(level):
                    l_ids = self.findLowerIds(ids=id_)
                    if (l_ids is not None) and len(l_ids)>0:
                        x_min = min(node[lid][0] for lid in l_ids)
                        x_max = max(node[lid][0] for lid in l_ids)
                        node[id_][0] = (x_min + x_max) / 2.
                    
        else:
            raise ValueError('Argmument mode(' + mode \
                                 + ") can be 'left' or 'center'.") 

        # find coordinates of lines
        v_line = []
        h_line = []
        for level in range(self.topLevel, -1, -1):
            old_id = None
            for id_ in self.getIds(level):
                h_id = self.findHigherIds(ids=id_)

                # vertical lines
                if h_id is None:
                    v_line.append(
                        [node[id_][0], node[id_][0], level, level+0.1])
                else:
                    v_line.append([node[id_][0], node[id_][0], level, level+1])

                # horizontal lines
                if (old_id is not None) and (h_id is not None) \
                        and (h_id == self.findHigherIds(ids=old_id)):
                    h_line.append([node[old_id][0], node[id_][0], 
                                   level+1, level+1])

                # update for the next round
                old_id = id_

        # convert lines to ndarray
        h_line = numpy.array(h_line)
        v_line = numpy.array(v_line)

        return node, h_line, v_line


