"""
Useful probability functions.
"""
# Author: Vladan Lucic, last modified 05.04.07

import numpy

def combinations(elements, size, repeat=False, order=False):
    """
    Generator for combinations of elements having size.

    Argument repeat determins if the elements are repeated, and argument
    order if they are ordered.

    Usage example:
      for comb in combinations(elements=[1,3,6,7], size=2): print comb

    Warning: order=True, repeat=False case is not implemented. Can be obtained
    by permuting each of the repeat=False, order=False combinations. 

    Arguments:
      - elements: list of elements used for combinations
      - size: number of elements in a combination
      - repeat: flag indicating if elements can be repeated (in a same
      combination), e.g. [1,3,3] is returned of repeat=True, but not if
      repeat=False
      - order: flag indicating if the order of elements is important, e.g [1,3,4]
      and [3,4,1] are different cobinations if order=True, but they are considered
      the same if order=False.
    """

    if order:
        if repeat:
            for comb in _combinationsRepeatOrder(elements, size): yield comb
        else:
            # could be generated as permutations on no repeat, no order comb's
            raise NotImplementedError, "Sorry, combinations with ordered, " \
                  + "nonrepeated elements are not implemented." 

    else:
        for comb in _combinationsNoOrder(elements, size, repeat): yield comb

def _combinationsNoOrder(elements, size, repeat=False):
    
    # initialize 
    elements = numpy.array(elements)

    if repeat: eInd = [0] * size
    else: eInd = range(size)
    eInd[size-1] -= 1
    
    sInd = size - 1
    _except = False

    while True:

        try: 

            # normal execution, increase rightmost index
            if not _except: eInd[sInd] += 1
            yield elements[eInd]
            sInd = size - 1
            _except = False

        except IndexError:

            # coud not increase value at sInd position, so move sInd to the left
            sInd -= 1

            # check if end of iterations
            if sInd < 0: raise StopIteration

            # increase element index at sInd and reset indices right of sInd 
            eInd[sInd] += 1
            for ind in range(sInd+1, size):
                eInd[ind] = eInd[ind-1] + 1 - repeat
            _except = True

def _combinationsRepeatOrder(elements, size):
    
    # initialize 
    elements = numpy.array(elements)

    eInd = [0] * size
    eInd[size-1] -= 1
    
    sInd = size - 1
    _except = False

    while True:

        try: 

            # normal execution, increase rightmost index
            if not _except: eInd[sInd] += 1
            yield elements[eInd]
            sInd = size - 1
            _except = False

        except IndexError:

            # coud not increase value at sInd, so move sInd to the left
            sInd -= 1

            # check if end of iterations
            if sInd < 0: raise StopIteration

            # increase element index at sInd and reset indices right of sInd 
            eInd[sInd] += 1
            for ind in range(sInd+1, size):
                eInd[ind] = 0
            _except = True

        
