"""

Functions related to attribute get/set python built-ins. 

# Author: Vladan Lucic, Max Planck Institute for Biochemistry
# $Id$
"""
from __future__ import unicode_literals
from past.builtins import basestring

__version__ = "$Revision$"


def getattr_deep(object, name):
    """
    Like built-in getattr, but name can contain dots indicating that it is 
    an attribute of an attribte ... of object.

    Arguments:
      - object: objects
      - name: attribute (of an attribute ...) of object
    """
    
    # split name in attributes (list)
    if isinstance(name, basestring):
        attributes = name.split('.')
    else:
        attributes = name

    # get attribute
    for attr in attributes:
        object = getattr(object, attr)

    return object

def setattr_deep(object, name, value, mode='_'):
    """
    Like built-in setattr, but if name contains dots it is changed according
    to the mode. If mode is '_' dots are replaced by underscores, and if it
    is 'last', only the part after the rightmost dot is used as name.

    Arguments:
      - object: object
      - name: attribute name
      - value: value
      - mode: determines how a name containing dots is transformed 
    """
    name = get_deep_name(name=name, mode=mode)
    setattr(object, name, value)

def get_deep_name(name, mode='_'):
    """
    Returns name transformed by mode. If mode is '_' dots in name are 
    replaced by underscores, and if it is 'last', only the part after the 
    rightmost dot is used as name.

    Arguments:
      - name: attribute name
      - mode: determines how a name containing dots is transformed 
    """

    if mode == '_':
        name = name.replace('.', '_')

    elif mode == 'last':
        attributes = name.split('.')
        name = attributes.pop()

    else:
        raise ValueError("Argument mode can be '_', or 'last' but not " + mode)

    return name
    
