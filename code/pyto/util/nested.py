"""
Functions dealing with nested lists and related.

# Author: Vladan Lucic
# $Id: nested.py 514 2009-06-30 13:54:21Z vladan $
"""

__version__ = "$Revision: 514 $"


from copy import copy, deepcopy
import numpy

def flatten(x):
    """flatten(sequence) -> list

    Returns a single, flat list which contains all elements retrieved
    from the sequence and all recursively contained sub-sequences
    (iterables).

    Examples:
    >>> [1, 2, [3,4], (5,6)]
    [1, 2, [3, 4], (5, 6)]
    >>> flatten([[[1,2,3], (42,None)], [4,5], [6], 7, MyVector(8,9,10)])
    [1, 2, 3, 42, None, 4, 5, 6, 7, 8, 9, 10]

    From http://kogs-www.informatik.uni-hamburg.de/~meine/python_tricks by
    Hans Meine.
    """

    result = []
    for el in x:
        #if isinstance(el, (list, tuple)):
        if hasattr(el, "__iter__") and not isinstance(el, basestring):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

def map(fun, lis):
    """
    Applies function fun to each non-iterable element of a (possibly nested)
    list lis at arbitrary nesting depth. Returns list with the same nested
    structure as lis.

    Example:

    >>> map(lambda x: x+5, [[2,3], 4])
    [[7, 8], 9]

    Arguments:
      - fun: function to be applied
      - lis: list or other iterable except basestring 
      """

    result = []
    for el in lis:
        if hasattr(el, "__iter__") and not isinstance(el, basestring):
            result.append(map(fun, el))
        else:
            result.append(fun(el))
    return result

def add(x, lis):
    """
    Adds x to each non-iterable element of lis at arbitrary depth.

    Example:

    >>> add(5, [[2,3], 4])
    >>> [[7, 8], 9]

    Arguments:
      - x: number to be added
      - lis: list or other iterable except basestring     
    """

    result = []
    for el in lis:
        if hasattr(el, "__iter__") and not isinstance(el, basestring):
            result.append(add(x, el))
        else:
            result.append(el + x)
    return result

def is_nested(lis):
    """
    Returns True if lis is a nested list.
    """

    result = False

    if isinstance(lis, (list, numpy.ndarray)):
        for item in lis:
            if isinstance(item, (list, numpy.ndarray)):
                result = True
                break

    return result

def resolve_dict(dict):
    """
    Applies mapping given by dictionary dict to its own values and to the
    result of this operation, as many times as needed to produce a dictionary 
    whose keys and values have no common element.

    Example:
      >>> selfmap_dict({1:11, 2:12, 3:13, 11:21, 12:22, 21:31})
      {1:31, 2:22, 3:13}

    Argument:
      dict: initial dictionary

    Returns: modified dictionary
    """

    result = deepcopy(dict)

    for key in dict.iterkeys():
        val = key
        try:
            while(True):
                val = dict[val]
        except KeyError:
            result.update({key:val})

    return result
