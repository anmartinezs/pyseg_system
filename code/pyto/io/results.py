"""
Contains class Results for read/write or results in the table form.

Note: Depreciated because results are stored in a pickle file.

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id$
"""
from builtins import zip
from builtins import object

__version__ = "$Revision$"


import string
import re
import numpy
import scipy
import scipy.io

class Results(object):
    """
    Contains methods to read and write tables.

    """

    def __init__(self):
        """
        Initializes attributes.
        """
        self.names = None
        self.columns = None
        self.comment = '#'

    def read(self, file, names=None, columns=None, comment=None):
        """
        Reads results from file, parses the data, assignes it to local 
        arttributes and orders them according to the ids.

        Columns should be set so that only the part of results that forms 
        a table (2d array) is read.
        """

        # use attributes if arguments not set
        if names is None:
            names = self.names
        if columns is None:
            columns = self.columns
        if comment is None:
            comment = self.comment

        # read file and separate it in tables according to columns
        tables = self.readTable(file=file, columns=columns, comment=comment)

        # assign tables to attributes
        self.nameTables(tables=tables, names=names)

        # order tables
        self.reorder(ids=self.ids, names=self.names)

    def readTable(self, file, columns=None):
        """
        Reads contents of file and puts the data in a 2d array (table).
        """

        # use attributes if arguments not set
        if columns is None:
            columns = self.columns

        tab = scipy.io.read_array(file, columns=columns)
        return tab

    def nameTables(self, tables, names):
        """
        Assign each table to an attribute
        """
        if (names is not None) and isinstance(tables, tuple):
            for (nam, tab) in zip(names, tables):
                self.__dict__[nam] = tab
            return
        else:
            return tables
        
    def getVersion(self, file):
        """
        Finds version of the results file.
        
        First it reads file to find the svn revision number of the script that
        was used to write this results file. Then sets the version of the
        results file to be the highest element of self.versions that is still
        lower than the revision.

        Requires self.versions, which is normally not specified in this class.
        This method should be called from instances of classes that inherit from
        this class and have self.versions.set (such as Connections).

        Argument:
          - file: results file

        Retirns: (int) version
        """

        # open results file
        if isinstance(file, str):
            file = open(file)

        # find results file revision
        for line in file:
            revision = self.enclosedString(string=line, begin='$Revision:',
                                           end='$')
            if revision is None:
                continue
            else:
                revision = int(revision.strip())
                break
        else:
            revision = 0

        # find appropriate version
        reverse_vers = [ver for ver in self.versions]  # copy
        reverse_vers.sort()
        reverse_vers.reverse()
        for ver in reverse_vers:
            if revision >= ver: 
                version = ver
                break

        return version

    def reorder(self, ids, arrays=None, names=None):
        """
        Orders elements of arrays according to the ids array.
        
        Arrays can be specified as a list of ndarrays (argument arrays),
        or by names (argument names).

        Arguments:

        """

        # use attributes if arguments not set
        #if names is None:
        #    names = self.names

        # order ids
        ids = numpy.asarray(ids)
        order = ids.argsort()

        # reorder arrays given directly
        if arrays is not None:
            if isinstance(arrays, numpy.ndarray):
                arrays = [arrays]
            res = [ar[order] for ar in arrays]
                
        # reorder arrays given by names
        if names is not None:
            for nam in names:
                self.__dict__[nam] = self.__dict__[nam][order]

        # return in the same form as in the argument
        if arrays is not None:
            if isinstance(arrays, numpy.ndarray):
                return res[0]
            else:
                return res 

    def index(self, ids, names):
        """
        Arranges arrays so that they are indexed by ids.
        """

        # use attributes if arguments not set
        #if names is None:
        #    names = self.names

        # prepare ids
        ids = numpy.asarray(ids, dtype='int')
        max_id = ids.max()
        order = ids.argsort()
        ids.sort()

        # index all arrays
        for nam in names:
            tmp = numpy.zeros(max_id+1, dtype=self.__dict__[nam].dtype)
            tmp[ids] = self.__dict__[nam][order]
            self.__dict__[nam] = tmp

    def enclosedString(self, string, begin, end):
        """
        Returns part of string between strings begin and end.
        """

        # 
        begin = re.escape(begin)
        end = re.escape(end)

        # find 
        whole_re = re.compile(begin + '.*' + end)
        match = whole_re.findall(string)
        if len(match) == 0:
            return None

        begin_re = re.compile(begin)
        match = begin_re.sub('', match[0])
        end_re = re.compile(end)
        match = end_re.sub('', match)
        
        return match
