############################################################################################################
# Set classes for handling pexceptions in the PySeg package
#
#

__author__ = 'martinez'

import sys

################################################################################################
# Exceptions

class PySegError(Exception):
    """Base class for pexceptions in this module."""
    pass

class PySegInputError(PySegError):
    """Exception raised for errors in the input.

    Attributes:
        expr -- input expression in which the error occurred
        msg  -- explanation of the error
    """

    def __init__(self, expr, msg):
        self.expr = expr
        self.msg = msg
        print('pexceptions 67: ' + self.expr + ' - ' + self.msg, file=sys.stdout, flush=True)

    def get_message(self):
        return self.expr + ' - ' + self.msg

class PySegTransitionError(PySegError):
    """Raised when an operation attempts a state transition that's not
    allowed.

    Attributes:
        prev -- state at beginning of transition
        next -- attempted new state
        msg  -- explanation of why the specific transition is not allowed
    """

    def __init__(self, prev, next, msg):
        self.prev = prev
        self.next = next
        self.msg = msg

################################################################################################
# Warnings

class PySegWarning(Warning):
    """Base class for warnings pexceptions in this module."""
    pass

class PySegInputWarning(PySegWarning):
    """Exception raised for warnings in the input.

    Attributes:
        expr -- input expression in which the warning occurred
        msg  -- explanation of the warning
    """

    def __init__(self, expr, msg):
        self.expr = expr
        self.msg = msg

    def get_message(self):
        return self.expr + ' - ' + self.msg
