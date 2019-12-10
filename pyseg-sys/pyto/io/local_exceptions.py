"""
Contains Exception classes for image io.

# Author: Vladan Lucic Max Planck Institute for Biochemistry
# $Id: local_exceptions.py 1194 2015-06-12 09:43:33Z vladan $
"""

__version__ = "$Revision: 1194 $"


class FileTypeError(IOError):
    """
    Exception reised when nonexistant file type is given.
    """
    def __init__(self, requested, defined):
        self.requested = requested
        self.defined = defined

    def __str__(self):
        msg = "Defined file formats are: \n\t" \
               + str(list(set(self.defined.values()))) \
               + "\nand defined extensions are: \n\t" \
               + str(set(self.defined.keys()))
        if self.requested is None:
            msg = msg + " File format not understood. "
        else:
            msg = msg + " File format: " + self.requested \
               + " doesn't exist. " 
        return msg
               

    
