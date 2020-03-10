"""
Set of classes for dealing with a XML files for loading cellular objects (e.g. filaments)

# Author: Antonio Martinez-Sanchez (University of Goettingen Medical Center)
# Date: 1.02.20
"""

__author__ = 'Antonio Martinez-Sanchez'

import xml.etree.ElementTree as ET
import numpy as np

###########################################################################################
# Global functionality
###########################################################################################



###########################################################################################
# Class for converting data types for columns used by Relion
###########################################################################################



###########################################################################################
# Class for representing an Amira XML file to represent a filaments network
###########################################################################################

class XMLFilaments(object):

    def __init__(self):
        self.__nodes, self.__points, self.__segments = dict(), dict(), dict()

    # Get/Set functions

    def get_nfils(self):
        """
        Return the number of filaments
        :return: the number of filaments
        """
        return len(self.__segments['Segment ID'])

    def get_fil_coords(self, segment_id):
        """
        Return the filament by its 'Segment ID'
        :param segment_id: Segment ID
        :return: an array of coordinates
        """
        sidx = self.__segments['Segment ID'].find(segment_id)
        coords = np.zeros(shape=(len(self.__segments['Point IDs'][sidx]), 3), dtype=np.float32)
        for i, idx in enumerate(self.__segments['Point IDs']):
            pid = self.__points['Point ID'].find(idx)
            coords[i, :] = (self.__points['X Coord'][pid], self.__points['Y Coord'][pid], self.__points['Z Coord'][pid])
        return coords

    # External functionality

    def load(self, path):
        """
        Load the filament coordinates from an Amira XML file
        :param path: path to the input file
        :return: raises an exception if the XML cannot be read
        """

        # Load the input XML file
        tree = ET.parse(path)
        root = tree.getroot()

        # Parse the XML elements
        for child in root:
            if 'Worksheet' in child.tag:
                for key_i, val_i in zip(child.attrib.iterkeys(), child.attrib.itervalues()):
                    if ('Name' in key_i) and ('Nodes' in val_i):
                        sub_child = child[0]
                        if 'Table' in sub_child.attrib:
                            for row in sub_child:
                                if 'Row' in row.attrib:
                                    for cell in row:
                                        if 'Cell' in str(cell):
                                            for dat in cell:
                                                if 'Data' in dat.tag:
                                                    for key_ii, val_ii in zip(dat.attrib.keys(), dat.attrib.values()):
                                                        if ('Type' in key_ii) and ('String' in val_ii):
                                                            nodes[dat.tag] = list()
                                                        else:
                                                            nodes[dat.tag].append(float(dat.text))
                    if ('Name' in key_i) and ('Points' in val_i):
                        sub_child = child[0]
                        if 'Table' in sub_child.attrib:
                            for row in sub_child:
                                if 'Row' in row.attrib:
                                    for cell in row:
                                        if 'Cell' in str(cell):
                                            for dat in cell:
                                                if 'Data' in dat.tag:
                                                    for key_ii, val_ii in zip(dat.attrib.keys(), dat.attrib.values()):
                                                        if ('Type' in key_ii) and ('String' in val_ii):
                                                            points[dat.tag] = list()
                                                        else:
                                                            points[dat.tag].append(float(dat.text))
                    if ('Name' in key_i) and ('Segments' in val_i):
                        sub_child = child[0]
                        if 'Table' in sub_child.attrib:
                            for row in sub_child:
                                if 'Row' in row.attrib:
                                    for cell in row:
                                        if 'Cell' in str(cell):
                                            for dat in cell:
                                                if 'Data' in dat.tag:
                                                    for key_ii, val_ii in zip(dat.attrib.keys(), dat.attrib.values()):
                                                        if ('Type' in key_ii) and ('String' in val_ii):
                                                            if ',' in val_ii.text:
                                                                segments[dat.tag].append(dat.text.split(','))
                                                            else:
                                                                segments[dat.tag] = list()
                                                        else:
                                                            segments[dat.tag].append(float(dat.text))
