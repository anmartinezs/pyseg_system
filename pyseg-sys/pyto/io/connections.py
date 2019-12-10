"""
Contains class Connections for reading connections results from a file and
parsing the results.

After executing the following statements:

conn = pyto.io.Connections()
conn.read('connections_file')

all data from the connections is saved in the following attributes:

  - conn.ids: ndarray of connection ids
  - conn.densityMean
  - conn.densityStd
  - conn.densityMin
  - conn.densityMax
  - conn.volume
  - conn.surface
  - conn.boundaryIds: ndarray (dtype object) where each element is an array of
  boundary ids

All attributes are ndarrays ordered in the same way as ids in conn.ids.

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id: connections.py 130 2008-05-08 16:36:56Z vladan $
"""

__version__ = "$Revision: 130 $"


import numpy
import scipy
import scipy.io
from results import Results


class Connections(Results):
    """
    Reading connections results from a file and parses the results.

    """

    def __init__(self):
        """
        Initializes attributes.
        """

        # from super
        super(self.__class__, self).__init__()

        # local
        self.versions = [0, 100]

    def read(self, file):
        """
        Reads connections data from a connections file, parses the data,
        assignes it to local arttributes and orders them according to the ids.

        Argument:
          file: file name of the connections file
        """

        # check revision and set attributes
        version = self.getVersion(file)
        self.defineVersion(version)

        # read and assign to attributes everything except boundaries
        super(self.__class__, self).read(file=file)

        # read extra 
        self.readExtra(file, version)

    def defineVersion(self, version):
        """
        Sets attributes required for the read method according to the version.

        Argument:
          - version: (int) version, as returned from getVersion

        Attributes set:
          - self.names:
          - self.columns:
          - self.comment:
          - self.extraNames:
          - self.extraColumns
        """

        if version == 0:
            self.names = ['ids', 'densityMean', 'densityStd', 'densityMin',
                          'densityMax', 'volume', 'surface']
            self.columns = [(0,), (1,), (2,), (3,), (4,), (5,), (6,)]
            self.comment = '#'
            self.extraNames = ['extraIds', 'boundaryIds']
            self.extraColumns = [(0,),((7,None),)]

        elif version == 100:
            self.names = ['ids', 'densityMean', 'densityStd', 'densityMin',
                          'densityMax', 'volume', 'surface']
            self.columns = [(0,), (1,), (2,), (3,), (4,), (5,), (6,)]
            self.comment = '#'
            self.extraNames = ['extraIds', 'boundaryIds']
            self.extraColumns = [(0,),((7,None),)]

    def readExtra(self, file, version):
        """
        Reads connections data from a file that does not fit in the table
        (2d array format), according tho the connections file version.

        Arguments:
          - file: connections file name
          - version: connections file version
        """

        # read file and find lines that start with positive number
        file_2 = open(file)
        line_no = 0
        data_line_nos = []
        data_lines = []
        for line in file_2:
            line=line.strip()
            split_line = line.split()
            if len(split_line) > 0:
                id_perhaps = split_line[0]
                if id_perhaps.isdigit() and (int(id_perhaps) > 0):
                    data_line_nos.append(line_no)
                    data_lines.append(line)
            line_no += 1

        # initialize new arrays
        for nam in self.extraNames:
            if nam == 'extraIds':
                self.__dict__[nam] = numpy.zeros(len(data_line_nos)+1, 
                                                 dtype='int')
            else:
                self.__dict__[nam] = numpy.zeros(len(data_line_nos)+1, 
                                                 dtype='object')

        # enter data from positive id lines to arrays
        nam_ran = range(len(self.extraNames))
        line_ran = range(len(data_line_nos))
        for (line_ind, line_no, line) in \
                zip(line_ran, data_line_nos, data_lines):

            # read and parse line using scipy.io.array
            tab = scipy.io.read_array(file, columns=self.extraColumns, 
                                      lines=(line_no,), atype=['l','l'])
            tab = list(tab)
            
            # do different stuff for differnet versions
            if version == 0:

                # extract boundary ids between [ and ]
                boundary_str = self.enclosedString(line, begin='[', end=']')
                b_ids = \
                    [int(b_id) for b_id in boundary_str.strip().split()] 
                tab[1] = numpy.array(b_ids)

            elif version == 100:
                pass

            # add variables from this line to appropriate arrays
            for (nam_ind, nam) in zip(nam_ran, self.extraNames):
                self.__dict__[nam][line_ind+1] = tab[nam_ind]

        # reorder new arrays
        self.reorder(ids=self.extraIds, names=self.extraNames)
