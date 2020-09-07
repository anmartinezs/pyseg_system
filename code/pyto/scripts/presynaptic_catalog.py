"""
Segmentation and analysis meta-data
"""
from __future__ import unicode_literals


#######################################################
#
# Experiment description and file names
#

# identifier
identifier = 'project_ctrl_5'

# experimental condition
treatment = 'ctrl'

# further info (optional)
animal_nr = 'a30p 2'


# synaptic vesicles
sv_file = 'path to segmentation and analysis /vesicles/ some name _vesicles.pkl'
sv_membrane_file = 'path to segmentation and analysis /vesicles/ some name _mem.pkl'
sv_lumen_file = 'path to segmentation and analysis /vesicles/ some name _lum.pkl'

# hierarchical segmentation of tethers and connectors
tethers_file = 'path to segmentation and analysis /conn/ connectors name .pkl'
connectors_file = 'path to segmentation and analysis /conn/ tethers name .pkl'
cluster_file = 'path to segmentation and analysis /cluster/ some name .pkl'

# layers
layers_file =  'path to segmentation and analysis /conn/ some name _layers.dat'


########################################################
#
# Observations
#

# mitochondria in the presyn terminal
mitochondria = True


######################################################
#
# Microscopy
#

# microscope
microscope = 'polara_1'

# pixel size [nm]
pixel_size = 1.888

# person who recorded the series
operator = 'Someone'

# person who did membrane segmentation
segmented = 'Someone else'
