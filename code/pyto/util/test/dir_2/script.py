"""

Help with tests for module util.runpy .

# Author: Vladan Lucic
# $Id: script.py 765 2010-10-05 17:15:02Z vladan $
"""

__version__ = "$Revision: 765 $"


import pyto
from pyto.util.test.test_bulk import _TestBulkError

two = 2

common = 2

def main():
    import os

    # check current directory, raise AsserError if thecheck fails
    wd = os.path.abspath(__file__)
    dir_ = os.path.split(os.path.split(wd)[0])[1]
    assert dir_ == 'dir_2'

    # check module variables, raise AsserError if any check fails
    assert two == 2
    try:
        one
        assert False
    except NameError:
        pass
    assert common == 2

    # signal that main was executed
    raise _TestBulkError()


if __name__ == '__main__':
    main()
