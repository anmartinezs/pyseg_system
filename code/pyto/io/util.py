"""
Useful io functions.

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id$
"""

__version__ = "$Revision$"


def arrayFormat(arrays, format, indices, prependIndex=False):
    """
    Makes a list of formated strings, where each string contains formated
    values of all vars for one of ids, that is:

    ['vars[0][ids[0]] vars[1][ids[0]] ... ',
     'vars[0][ids[1]] ...',
     ... ]

     Arguments:
       - arrays: list of arrays
       - format: format string
       - idices: list of indices
       - prependIndex: if True, the index is prepended to each line. Note that
       in this case format has to contain an entry for index. 
     """

    out = []
    for ind in indices:

        # make a tuple of values for this id
        if prependIndex: 
            row = [ind]
        else: 
            row = []
        for ar in arrays: 
            row.append( ar[ind] )

        # format the current list
        out.append( format % tuple(row) )

    return out
