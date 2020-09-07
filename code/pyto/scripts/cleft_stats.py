#!/usr/bin/env python
"""
Statstical analysis of the synaptic cleft

Common usage contains the following steps.

1) Setup pickles: Make pickles that contain data from all individual 
segmentation & analysis pickles:

  - Make catalog files for each tomogram in ../catalog/
  - cd to this directory, then in ipython (or similar) execute:
      > import work
      > from work import *
      > work.main(individual=True, save=True)

  This should create few pickle files in this directory (layers.pkl, 
  conn.pkl, ...)

2) Create profile: not required, but advantage keeps profile-specific history
 
  - Create ipython profile for your project: 
      $ ipython profile create <project_name>
  - Create startup file for this profile, it enough to have:
      import work
      work.main()
      from work import *

3) Analysis

  - Start ipython (qtconsole is optional, profile only if created):
      $ ipython qtconsole  --profile=<project_name> --pylab=qt
  - In ipython (not needed if profile used):
      > import work
      > work.main()
      > from work import *
  - Run individual analysis commands (from SV distribution section and below)

  - If work.py is edited during an ipython session:
      > reload(work)
      > from work import *


# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id$
"""
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from builtins import zip
#from builtins import str
from builtins import range
#from past.utils import old_div

__version__ = "$Revision$"


import sys
import os
import logging
import itertools
from copy import copy, deepcopy
import pickle

import numpy 
import scipy 
import scipy.stats
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

import pyto
from pyto.analysis.groups import Groups
from pyto.analysis.observations import Observations

# to debug replace INFO by DEBUG
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s %(module)s.%(funcName)s():%(lineno)d %(message)s',
    datefmt='%d %b %Y %H:%M:%S')


##############################################################
#
# Parameters (need to be edited)
#
##############################################################

###########################################################
#
# Analysis
#

# experiment categories
categories = ['syncam_ox_ctrl', 'syncam_ox', 'syncam_ko_ctrl', 'syncam_ko', 
              'ctrl_rat', 'egta_ctrl', 'egta_1', 'egta_3', 'egta_0']
#categories = ['syncam_ox_ctrl', 'syncam_ox', 'syncam_ko_ctrl', 'syncam_ko', 
#              'egta_ctrl', 'egta_1', 'egta_3', 'egta_0']
#categories = ['syncam_ko', 'syncam_ko_ctrl']
categories_syncam = ['syncam_ox_ctrl', 'syncam_ox', 
                     'syncam_ko_ctrl', 'syncam_ko'] 
categories_egta = ['ctrl_rat', 'egta_ctrl', 'egta_1', 'egta_3', 'egta_0']
#categories_egta = ['egta_ctrl', 'egta_1', 'egta_3', 'egta_0']
#categories = categories_egta

# experiment identifiers
identifiers = (['syncam_ox_ctrl_' + str(ind) for ind in range(1,8)]  
               + ['syncam_ox_' + str(ind) for ind in range(1,8)] 
               + ['syncam_ko_ctrl_' + str(ind) for ind in range(1,9)]  
               + ['syncam_ko_' + str(ind) for ind in range(1,10)] 
               + ['ctrl_rat_' + str(ind) for ind in range(1,6)]
               + ['egta_ctrl_' + str(ind) for ind in range(1,7)]
               + ['egta_0' + sym for sym in ['a','b','c','d','e','f']] 
               + ['egta_1' + sym for sym in ['a','b','c','d','e']] 
               + ['egta_2']
               + ['egta_3' + sym for sym in ['a','b','c','d','e','f']]) 
identifiers_discard = ['syncam_ox_4', 'syncam_ox_7', 
                       'syncam_ox_ctrl_6', 'syncam_ox_ctrl_7',
                       'syncam_ko_9', 'syncam_ko_ctrl_3',
                       'egta_0f', 'egta_3f', 'egta_ctrl_7']
identifiers = [ident for ident in identifiers 
               if ident not in identifiers_discard]
#identifiers = ['egta_ctrl_1', 'egta_ctrl_2', 'egta_3a', 'egta_3b']
#identifiers = None

# reference category
reference = {'syncam_ox' : 'syncam_ox_ctrl', 
             'syncam_ox_ctrl' : 'syncam_ox_ctrl',
             'syncam_ko' : 'syncam_ko_ctrl', 
             'syncam_ko_ctrl' : 'syncam_ox_ctrl',
             'egta_ctrl' : 'egta_ctrl',
             'egta_0' : 'egta_ctrl',
             'egta_1' : 'egta_ctrl',
             'egta_2' : 'egta_ctrl',
             'egta_3' : 'egta_ctrl',
             'ctrl_rat' : 'egta_ctrl'} 

# catalog directory
catalog_directory = '../../analysis/catalogs'

# catalogs file name pattern (can be a list of patterns) 
catalog_pattern = r'[^_].*\.py$'   # extension 'py', doesn't start with '_'

# catalog attribute names that specify locations of scripts-generated
# pickle files (pyto.scene.CleftRegions and pyto.segmentation.Hierarchy)
connections_name = 'cleft_segmentation_hi'
conn_col4_name = 'cleft_segmentation_hi_on_column_4'
#connections_norm_05_name = 'cleft_segmentation_norm_05'
layers_name = 'cleft_layers'
layers_4_name = 'cleft_layers_4'
layers_8_name = 'cleft_layers_8'
columns_name = 'cleft_columns'
layers_on_connections_name = 'cleft_segmentation_hi_layers'
layers_4_on_connections_name = 'cleft_segmentation_hi_layers_4'
layers_8_on_connections_name = 'cleft_segmentation_hi_layers_8'
layers_4_on_column_1_name = 'cleft_layers_4_on_column_1'
layers_4_on_column_2_name = 'cleft_layers_4_on_column_2'
layers_4_on_column_3_name = 'cleft_layers_4_on_column_3'
layers_4_on_column_4_name = 'cleft_layers_4_on_column_4'


# names of pyto.analysis.Connections and CleftRegions pickle files
connections_pkl = 'connections.pkl'
connections_norm_05_pkl = 'connections_norm_05.pkl'
connections_col4_pkl = 'conn_col4.pkl'
layers_pkl = 'layers.pkl'
layers_4_pkl = 'layers_4.pkl'
layers_8_pkl = 'layers_8.pkl'
columns_pkl = 'columns.pkl'
layers_on_connections_pkl = 'layers_on_connections.pkl'
layers_4_on_connections_pkl = 'layers_4_on_connections.pkl'
layers_8_on_connections_pkl = 'layers_8_on_connections.pkl'
layers_4_on_column_1_pkl = 'layers_4_on_column_1.pkl'
layers_4_on_column_2_pkl = 'layers_4_on_column_2.pkl'
layers_4_on_column_3_pkl = 'layers_4_on_column_3.pkl'
layers_4_on_column_4_pkl = 'layers_4_on_column_4.pkl'

###########################################################
#
# Printing
#

# print format
print_format = {

    # data
    'nSegments' : '   %5d ',
    'surfaceDensityContacts_1' : '   %7.4f ',
    'surfaceDensityContacts_2' : '   %7.4f ',

    # stats
    'mean' : '   %6.3f ',
    'std' : '   %6.3f ',
    'sem' : '   %6.3f ',
    'diff_bin_mean' : '   %6.3f ',
    'n' : '    %5d ',
    'testValue' : '   %7.4f ',
    'confidence' : '  %7.4f ',
    'fraction' : '    %5.2f ',
    'count' : '    %5d ',
    't_value' : '    %5.2f ',
    't' : '    %5.2f ',
    'h' : '    %5.2f ',
    'u' : '    %5.2f ',
    't_bin_value' : '    %5.2f ',
    't_bin' : '    %5.2f ',
    'chi_square' : '    %5.2f ',
    'chi_sq_bin' : '    %5.2f ',
    'confid_bin' : '  %7.4f '
    }

###########################################################
#
# Plotting
#
# Note: These parameters are used directly in the plotting functions of
# this module, they are not passed as function arguments.
#

# determines if data is plotted
plot_ = True 

# legend
legend = False

# data color
color = {
    'syncam_ox_ctrl' : 'blue',
    'syncam_ox' : 'turquoise',
    'syncam_ko_ctrl' : 'DarkGreen',
    'syncam_ko' : 'LightGreen',
    'egta_ctrl' : 'black',
    'egta_0' : 'yellow',
    'egta_1' : 'red',
    'egta_2' : 'violet',
    'egta_3' : 'orange',
    'ctrl_rat' : 'grey'
    }

# edge color
ecolor={}

# data labels
category_label = {
    'syncam_ko' : 'Syncam ko',
    'syncam_ko_ctrl' : 'Syncam ko ctrl',
    'syncam_ox' : 'Syncam ox',
    'syncam_ox_ctrl' : 'Syncam ox ctrl',
    'egta_ctrl' : 'egta ctrl',
    'egta_0' : 'egta 0',
    'egta_1' : 'egta 1',
    'egta_2' : 'egta 2',
    'egta_3' : 'egta 3',
    'ctrl_rat' : 'plain ruben'
    }

# data alpha
alpha = {
    'syncam_ox' : 1,
    'syncam_ox_ctrl' : 1,
    'egta_ctrl' : 1,
    'egta_0' : 1,
    'egta_1' : 1,
    'egta_2' : 1,
    'egta_3' : 1,
    'ctrl_rat' : 1
    }

# markers
marker =  {
    'syncam_ox' : 'o',
    'syncam_ox_ctrl' : 's',
    'all' : 'o',
    'tethered' : '+',
    'non-tethered' : 'o'
    }

# list of default markers
markers_default = ['o', 'v', 's', '*', '^', 'H', '<', 'D', '>']

# markersize
marker_size = 7

# line width
line_width = {
    'mean' : 4
    }

# line width
default_line_width = 2

# thick line width
thick_line_width = 4

# line style
default_line_style = '-'

# bar arrangement
bar_arrange = 'uniform'    # all bars in a row, single bar space between groups
bar_arrange = 'grouped'    # each group takes the same space 

# flag indicating if one-sided yerr
one_side_yerr = True

# confidence format 
confidence_plot_format = "%5.3f"

# confidence font size
confidence_plot_font_size = 7

# confidence levels for 1, 2, 3, ... stars
confidence_stars = [0.05, 0.01, 0.001]


##############################################################
#
# Functions (edit only if you know what you're doing)
#
##############################################################

##############################################################
#
# Higher level functions 
#

def stats(data, name, bins=None, bin_names=None, fraction=None, join=None, 
          groups=None, identifiers=None, test=None, reference=None, ddof=1, 
          out=sys.stdout, label=None, outNames=None, plot_=True, plot_name=None,
          yerr='sem', confidence='stars', title='', x_label=None, y_label=None):
    """
    Does statistical analysis of data specified by args data and name, prints
    and plots the results as a bar chart.

    In case arg bins is specified, data (arg name) of all observations 
    belonging to one group are first joined (arg join) and then converted to a
    histogram according to arg bins (property name 'histogram'). Also 
    calculated are probabilities for each histogram bin (property name 
    'probability') and the probability for bin indexed by arg fraction is
    saved separately as property 'fraction'. For example, fraction of connected
    vesicles is obtained for name='n_connection', bins=[0,1,100], fraction=1.

    In case arg join is None a value is printed and a bar is plotted for each 
    experiment. This value is either the value of the specified property if 
    scalar, or a mean of the property values if indexed.

    If arg join is 'join', the values of the specified property are pooled 
    accross all experiments belonging to one group, and the mean is 
    printed and plotted. 

    If arg join is 'mean', the mean of experiment means for each group 
    (indexed properties only) is printed and plotted. 

    If arg join is None and the data is indexed, both significance between 
    groups and between experiments are calculated.

    If specified, args groups and identifiers specify the order of groups
    and experiments on the x axis. 

    Arg plot_name specifies which statistical property to plot. If it is not 
    specified the property to plot is determined in the following way: if arg
    bins are not given 'mean' is plotted, otherwise if bin_names is specified
    'histogram' is plotted and if not 'fraction'. Therefore, most often 
    arg plot_name should not be given. Notable exception is for a histogram
    when instead of numer of occurences a probability (fraction) of occurences
    needs to be plotted, in which case 'fraction' should be specified.

     Arguments:
      - data: (Groups or Observations) data structure
      - bins: (list) bins for making histogram
      - fraction: bin index for which the fraction is calculated
      - name: name of the analyzed property
      - join: 'join' to join experiments, otherwise None
      - groups: list of group names
      - identifiers: list of identifiers
      - test: statistical inference test type
      - reference: specifies reference data
      - ddof: differential degrees of freedom used for std
      - out: output stream for printing data and results
      - outNames: list of statistical properties that are printed
      - yerr: name of the statistical property used for y error bars
      - plot_: flag indicating if the result are to be plotted
      - plot_name: name of the calculated property to plot
      - label: determines which color, alpha, ... is used, can be 'group' to 
      label by group or 'experiment' to label by experiment
      - confidence: determines how confidence is plotted
      - x_label: x axis label
      - y_label: y axis label, if not specified arg name used instead 
      - title: title

    ToDo: include stats_x in stats
    """

    # prepare for plotting
    if plot_:
        plt.figure()

    # makes sure elements of bin_names are different from those of 
    # category_label (appends space(s) to make bin_names different)
    if bin_names is not None:
        fixed_bin_names = []
        for bin_nam in bin_names:
            while(bin_nam in category_label):
                bin_nam = bin_nam + ' '
            fixed_bin_names.append(bin_nam)
        bin_names = fixed_bin_names

    # determine what property to plot
    if plot_name is None:
        if bins is None:
            plot_name = 'mean'
        else:
            if bin_names is not None:
                plot_name = 'fraction'
            else:
                plot_name='histogram'

    # figure out if indexed
    indexed = name in list(data.values())[0].indexed

    if isinstance(data, Groups):
        if not indexed:

            # not indexed
            if join is None:
                
                # groups, scalar property, no joining
                data.printStats(
                    out=out, names=[name], groups=groups, 
                    identifiers=identifiers, format_=print_format, title=title)
                if plot_:
                    plot_stats(
                        stats=data, name=name, groups=groups, 
                        identifiers=identifiers, yerr=None, confidence=None)

            elif join == 'join':

                # groups, scalar property, join
                stats = data.joinAndStats(
                    name=name, mode=join, groups=groups, 
                    identifiers=identifiers, test=test, reference=reference, 
                    ddof=ddof, out=out, outNames=outNames, 
                    format_=print_format, title=title)
                if plot_:
                    plot_stats(stats=stats, name=plot_name, 
                               yerr=yerr, confidence=confidence)

            else:
                raise ValueError(
                    "For Groups data and non-indexed (scalar) property "
                    + "argument join can be None or 'join'.")

        else:

            # indexed
            if (join is None) or (join == 'pair'):

                # stats between groups and  between observations
                if groups is None:
                    groups = list(data.keys())

                # between experiments
                exp_ref = {}
                for categ in groups:
                    exp_ref[categ] = reference
                if join is None:
                    exp_test = test
                elif join == 'pair':
                    exp_test = None
                stats = data.doStats(
                    name=name, bins=bins, fraction=fraction, groups=groups, 
                    test=exp_test, between='experiments', 
                    reference=exp_ref, ddof=ddof, identifiers=identifiers,
                    format_=print_format, out=None)

                # between groups
                if data.isTransposable() and (len(groups)>0):
                    group_ref = {}
                    for ident in data[groups[0]].identifiers:
                        group_ref[ident] = reference
                    try:
                        stats_x = data.doStats(
                            name=name, bins=bins, fraction=fraction, 
                            groups=groups, identifiers=identifiers, test=test, 
                            between='groups', reference=group_ref, ddof=ddof,
                            format_=print_format, out=None)
    # ToDo: include stats_x in stats
                        names_x = ['testValue', 'confidence']
                    except KeyError:
                        stats_x = None
                        names_x = None
                else:
                    stats_x = None
                    names_x = None

                # print and plot
                stats.printStats(
                    out=out, groups=groups, identifiers=identifiers, 
                    format_=print_format, title=title, 
                    other=stats_x, otherNames=names_x)
                if plot_:
                    plot_stats(
                        stats=stats, name=plot_name, groups=groups, 
                        identifiers=identifiers, yerr=yerr, label=label,
                        confidence=confidence, stats_between=stats_x)

            elif (join == 'join') or (join == 'mean'):

                # groups, indexed property, join or mean
                stats = data.joinAndStats(
                    name=name, bins=bins, fraction=fraction, mode=join, 
                    test=test, reference=reference, groups=groups, 
                    identifiers=identifiers,
                    ddof=ddof, out=out, format_=print_format, title=title)

                if ((plot_name != 'histogram') 
                    and (plot_name != 'probability')):

                    # just plot
                    if plot_:
                        plot_stats(
                            stats=stats, name=plot_name, identifiers=groups, 
                            yerr=yerr, confidence=confidence)

                else:
                    
                    # split histogram and plot
                    stats_split = stats.splitIndexed()
                    histo_groups = Groups()
                    histo_groups.fromList(groups=stats_split, names=bin_names)

                    if plot_:
                        plot_stats(
                            stats=histo_groups, name=plot_name, 
                            groups=bin_names, identifiers=groups, yerr=yerr, 
                            confidence=confidence, label='experiment')

            else:
                raise ValueError(
                    "For Groups data and indexed property "
                    + "argument join can be None, 'join', or 'mean'.")

    elif isinstance(data, list):    

        # list of groups
        raise ValueError("Please use stats_list() instead.")
    
    else:
        raise ValueError("Argument data has to be an instance of Groups "  
                         + "or a list of Groups objects.")

    # finish plotting
    if plot_:
        plt.title(title)
        if y_label is None:
            y_label = name
        plt.ylabel(y_label)
        if x_label is not None:
            plt.xlabel(x_label)
        if legend:
            plt.legend()
        plt.show()

    if indexed or (join is not None):
        return stats

def correlation(
    xData, xName, yName, yData=None, test=None, regress=True, 
    reference=reference, groups=None, identifiers=None, join=None, 
    out=sys.stdout, format_={}, title=''):
    """
    Correlates two properties and plots them as a 2d scatter graph.

    In case arg join is None a value is printed and a bar is plotted for each 
    experiment. This value is either the value of the specified property if 
    scalar, or a mean of the property values if indexed.

    If arg join is 'join', the values of the specified property are pooled 
    accross all experiments belonging to one group, and the mean is 
    printed and plotted. 

    If arg join is 'mean', the mean of experiment means for each group 
    (indexed properties only) is printed and plotted. 

    Arguments:
      - xData, yData: (Groups or Observations) structures containing data
      - xName, yName: names of the correlated properties
      - test: correlation test type
      - regress: flag indicating if regression (best fit) line is calculated
      - reference: 
      - groups: list of group names
      - identifiers: list of identifiers
      - join: None to correlate data from each experiment separately, or 'join'
      to join experiments belonging to the same group  
      - out: output stream for printing data and results
      - title: title
    """

    # combine data if needed
    if yData is not None:
        data = deepcopy(xData)
        data.addData(source=yData, names=[yName])
    else:
        data = xData

    # set regression paramets
    if regress:
        fit = ['aRegress', 'bRegress']
    else:
        fit = None

    # start plotting
    if plot_:
        fig = plt.figure()

    if isinstance(data, Groups):
        
        # do correlation and print
        corr = data.doCorrelation(
            xName=xName, yName=yName, test=test, regress=regress, 
            reference=reference, mode=join, groups=groups,
            identifiers=identifiers, out=out, format_=format_, 
            title=title)
        print("corr.properties: ", corr.properties)
        print("corr.xData: ", corr.xData)
        print("corr.yData: ", corr.yData)
        # plot
        if plot_:
            plot_2d(x_data=corr, x_name='xData', y_name='yData', groups=groups,
                    identifiers=identifiers, graph_type='scatter', fit=fit)

    elif isinstance(data, Observations):

        # do correlation and print
        corr = data.doCorrelation(
            xName=xName, yName=yName, test=test,  regress=regress, 
            reference=reference, mode=join, out=out, 
            identifiers=identifiers, format_=format_, title=title)

        # plot
        if plot_:
            plot_2d(x_data=corr, x_name='xData', y_name='yData', 
                    identifiers=identifiers, graph_type='scatter', fit=fit)

    else:
        raise ValueError("Argument data has to be an instance of " 
                         + "pyto.analysis.Groups or Observations.")

    # finish plotting
    if plot_:
        plt.title(title)
        plt.ylabel(yName)
        plt.xlabel(xName)
        if legend:
            plt.legend()
        plt.show()

    return corr

##############################################################
#
# Plot and save functions 
#

def plot_cleft_layers(
    data, yName, xName='ids', yerr=None, groups=None, identifiers=None, 
    mode='all', graphType='line',  ddof=1, 
    title='', x_label='Layers', y_label='Greyscale value', legend=False):
    """
    Plots values of an indexed property specified by arg yName vs. another
    indexed property specified by arg xName as a line plot. In addition,
    short vertical dashed lines are plotted at the position of the cleft
    edges.
 
   If mode is 'all' or 'all&mean' data from all experiments belonging to one 
   group is plotted on one figure. If mode is 'all&mean' the group mean is also 
   plotted. If mode is 'mean' all group means are plotted together.

    Arguments:
      - data: (Groups or Observations) data structure
      - xName, yName: name of the plotted properties
      - yerr: property used for y-error
      - groups: list of group names
      - identifiers: list of identifiers
      - mode: 'all', 'mean' or 'all&mean'
      - graphType: 'line' for line-graph or 'scatter' for a scatter-graph
      - title: title
      - ddof = difference degrees of freedom used for std
      - legend: flag indicating if legend should be added to plot(s)
    """

    # plot ot not
    if not plot_:
        return

    # if data is Groups, print a separate figure for each group
    if isinstance(data, Groups):
        if groups is None:
            groups = list(data.keys())

        if (mode == 'all') or (mode == 'all&mean'): 

            # a separate figure for each group
            for group_name in groups:
                title = category_label.get(group_name, group_name)
                plot_cleft_layers_one(
                    data=data[group_name], yName=yName, xName=xName, yerr=yerr,
                    identifiers=identifiers, mode=mode, graphType=graphType,
                    title=title, x_label=x_label, y_label=y_label, 
                    legend=legend)

        elif mode == 'mean':

            # calculate means and plot (one graph)
            data = data.joinAndStats(
                name=yName, mode='byIndex', groups=groups, 
                identifiers=identifiers, ddof=ddof, out=None, title=title)
            #data.printData(names=['mean', 'sem'], out=sys.stdout, format_=print_format)
            plot_cleft_layers_one(
                data=data, yName='mean', xName='ids', identifiers=None, 
                mode=mode, graphType=graphType, ddof=ddof, yerr=yerr, 
                title=title, x_label=x_label, y_label=y_label, legend=legend)

    elif isinstance(data, Observations):

        # Observations: plot one graph
        plot_cleft_layers_one(
            data=data, yName=yName, xName=xName, identifiers=identifiers, 
            mode=mode, graphType=graphType, ddof=ddof, yerr=yerr, 
            title=title, x_label=x_label, y_label=y_label, legend=legend)

    else:
        raise ValueError("Argument 'data' has to be either pyto.analysis.Groups"
                         + " or Observations.") 

    return data

def plot_cleft_layers_one(
    data, yName, xName='ids', yerr=None, identifiers=None, mode='all', 
    graphType='line', title='', ddof=1, x_label='Layers', 
    y_label='Greyscale value', legend=False):
    """
    Plots values of an indexed property specified by arg yName vs. another
    indexed property specified by arg xName as a line plot. In addition,
    short vertical dashed lines are plotted at the position of the cleft
    edges.
 
    All data is plotted on one graph.

    Arguments:
      - data: (Observations) data structure
      - xName, yName: name of the plotted properties
      - yerr: property used for y-error
      - groups: list of group names
      - identifiers: list of identifiers
      - mode: 'all', 'mean' or 'all&mean'
      - graphType: 'line' for line-graph or 'scatter' for a scatter-graph
      - title: title
      - ddof = difference degrees of freedom used for std
      - legend: flag indicating if legend should be added to plot(s)
    """
    # from here on plotting an Observations object
    fig = plt.figure()

    # don't plot yerr for individual plots in mode 'all&mean'
    if mode == 'all&mean':
        exp_yerr = None
    else:
        exp_yerr = yerr

    # set identifiers
    if identifiers is None:
        identifiers = data.identifiers
    identifiers = [ident for ident in identifiers if ident in data.identifiers]

    # plot data for each experiment
    for ident in identifiers:

        # plot data for the current experiment 
        line = plot_2d(x_data=data, x_name=xName, y_name=yName, yerr=exp_yerr, 
                       identifiers=[ident], graph_type=graphType)

        # mark the cleft ends
        if (line is not None) and ((mode == 'all') or (mode == 'all&mean')):
            try:
                cleft_ids = data.getValue(property='cleftIds', identifier=ident)
                x_value = data.getValue(property=xName, identifier=ident)
                for end_id in [cleft_ids[0], cleft_ids[-1]+1]:
                    x_end = (x_value[end_id-1] + x_value[end_id-2]) / 2.
                    y_value = data.getValue(property=yName, identifier=ident)
                    y_mid = (y_value[end_id-1] + y_value[end_id-2]) / 2.
                    y_diff = y_value[end_id-1] - y_value[end_id-2]
                    y_end = [y_mid + 0.3 * y_diff, y_mid - 0.3 * y_diff]
                    plt.plot(
                        [x_end, x_end], y_end, linestyle='--', 
                        linewidth=default_line_width/2., 
                        color=line[0].get_color())
            except AttributeError:
                pass

    # calculate and plot mean
    if mode == 'all&mean':
        exp = data.doStatsByIndex(
            name=yName, identifiers=identifiers, 
            identifier='mean', ddof=ddof)
        line = plot_2d(x_data=exp, x_name=xName, y_name='mean', yerr=yerr, 
                       graph_type=graphType, line_width_='thick')

    # finish plotting
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    ends = plt.axis()
    plt.axis([0, 1.3 * ends[1], ends[2], 1.5 * ends[3]])
    if legend:
        plt.legend()
    plt.show()

def plot_stats(stats, name, groups=None, identifiers=None, yerr='sem', 
               confidence='stars', stats_between=None, label=None):
    """
    Does main part of plotting property (arg) name of (arg) stats, in the
    form of a bar chart. 

    If specified, args groups and identifiers specify the order of groups
    and experiments on the x axis. 

    Plots on the current figure.     

    Arguments:
      - stats: (Groups, or Observations) object
      containing data
      - name: property name
      - groups: list of group names, None for all groups
      - identifiers: experiment identifier names, None for all identifiers 
      - stats_between: (Groups) Needs to contain confidence between 
      (experiments of) different groups having the same identifiers  
      - label: determines which color, alpha, ... is used, can be 'group' to 
      label by group or 'experiment' to label by experiment
    """

    # stats type
    if isinstance(stats, Groups):
        stats_type = 'groups'
    elif isinstance(stats, Observations):
        stats_type = 'observations'
        stats_obs = stats
        stats = Groups()
        stats[''] = stats_obs
    else:
        raise ValueError(
            "Argument stats has to be an instance of Groups or Observations.")

    # find rough range of y-axis values (to plot cinfidence)
    y_values = [group.getValue(identifier=ident, property=name)
                for group in list(stats.values())
                for ident in group.identifiers]
    if (y_values is not None) and (len(y_values) > 0):
        rough_y_min = min(y_values)
        rough_y_max = max(y_values)
    else:
        rough_y_min = 0
        rough_y_max = 1
    rough_y_range = rough_y_max - min(rough_y_min, 0)

    # set bar width if needed
    if bar_arrange == 'uniform':
        bar_width = 0.2
        left = -2 * bar_width
    elif bar_arrange == 'grouped':
        max_identifs = max(
            len(group.identifiers) for group in list(stats.values()))
        bar_width = numpy.floor(80 / max_identifs) / 100
    else:
        raise ValueError("bar_arrange has to be 'uniform' or 'grouped'.")

    # set group order
    if groups is None:
        group_names = list(stats.keys()) 
    else:
        group_names = groups

    # loop over groups
    y_min = 0
    y_max = 0
    group_left = []
    label_done = False
    for group_nam, group_ind in zip(group_names, list(range(len(group_names)))):
        group = stats[group_nam]

        # set experiment order
        if identifiers is None:
            loc_identifs = group.identifiers
        elif isinstance(identifiers, list):
            loc_identifs = [ident for ident in identifiers 
                            if ident in group.identifiers]
        elif isinstance(identifiers, dict):
            loc_identifs = identifiers[group_nam]

        # move bar position
        if bar_arrange == 'uniform':
            left += bar_width
            group_left.append(left + bar_width)

        # loop over experiments
        for ident, exp_ind in zip(loc_identifs, list(range(len(loc_identifs)))):

            # label
            if label is None:
                if stats_type == 'groups':
                    label_code = group_nam
                elif stats_type == 'observations':
                    label_code = ident
            elif label == 'group':
                label_code = group_nam
            elif label == 'experiment':
                label_code = ident

            # adjust alpha 
            loc_alpha = alpha.get(label_code, 1)

            # y values and y error
            value = group.getValue(identifier=ident, property=name)
            if ((yerr is not None) and (yerr in group.properties) 
                and (loc_alpha == 1)):
                yerr_num = group.getValue(identifier=ident, property=yerr)
                yerr_one = yerr_num
                y_max = max(y_max, value+yerr_num)
                if one_side_yerr:
                    yerr_num = ([0], [yerr_num])
            else:
                yerr_num = None
                yerr_one = 0
                y_max = max(y_max, value)
            y_min = min(y_min, value)

            # plot
            if bar_arrange == 'uniform':
                left += bar_width
            elif bar_arrange == 'grouped':
                left = group_ind + exp_ind * bar_width
            if label_done:
                bar = plt.bar(
                    left=left, height=value, yerr=yerr_num, width=bar_width, 
                    color=color[label_code], ecolor=color[label_code],
                    alpha=loc_alpha)[0]
            else:
                bar = plt.bar(
                    left=left, height=value, yerr=yerr_num, width=bar_width, 
                    label=category_label.get(label_code, ''), 
                    color=color[label_code], ecolor=color[label_code],
                    alpha=loc_alpha)[0]

            #
            plt.errorbar(left+bar_width/2, value,yerr=yerr_num,
                         ecolor=ecolor.get(label_code, 'k'), label='_nolegend_')
            
            # confidence within group
            if (confidence is not None) and ('confidence' in group.properties):

                # get confidence
                confid_num = group.getValue(identifier=ident, 
                                            property='confidence')
                # workaround for problem in Observations.getValue()
                if (isinstance(confid_num, (list, numpy.ndarray)) 
                    and len(confid_num) == 1):
                    confid_num = confid_num[0]
                if confidence == 'number':
                    confid = confidence_plot_format % confid_num
                    conf_size = confidence_plot_font_size
                elif confidence == 'stars':
                    confid = '*' * get_confidence_stars(
                        confid_num, limits=confidence_stars)
                    conf_size = 1.5 * confidence_plot_font_size

                # plot confidence
                x_confid = bar.get_x() + bar.get_width()/2.
                y_confid = 0.02 * rough_y_range + bar.get_height() + yerr_one
                ref_ident = group.getValue(identifier=ident,   
                                           property='reference')
                ref_color = color.get(ref_ident, label_code) 
                plt.text(x_confid, y_confid, confid, ha='center', va='bottom',
                         size=conf_size, color=ref_color)

            # confidence between groups
            if ((stats_type == 'groups') and (confidence is not None) 
                and (stats_between is not None) 
                and ('confidence' in stats_between[group_nam].properties)):
                other_group = stats_between[group_nam]

                # check
                other_ref = other_group.getValue(identifier=ident, 
                                                 property='reference')
                if other_ref != ident:
                    logging.warning(
                        "Confidence between groups calculated between " \
                            + "experiments with different identifiers: " \
                            + ident + " and " + other_ref + ".")

                # get confidence
                confid_num = other_group.getValue(identifier=ident, 
                                                  property='confidence')
                if confidence == 'number':
                    confid = confidence_plot_format % confid_num
                    conf_size = confidence_plot_font_size
                elif confidence == 'stars':
                    confid = '*' * get_confidence_stars(
                        confid_num, limits=confidence_stars)
                    conf_size = 1.5 * confidence_plot_font_size

                # plot
                x_confid = bar.get_x() + bar.get_width()/2.
                y_confid = 0.04 * rough_y_range + bar.get_height() + yerr_one
                ref_color = color[ident]
                plt.text(x_confid, y_confid, confid, ha='center', va='bottom',
                         size=conf_size, color=ref_color)

        # set flag that prevents adding further labels to legend
        label_done = True

    # adjust axes
    axis_limits = list(plt.axis())
    plt.axis([axis_limits[0]-bar_width, max(axis_limits[1], 4), 
              y_min, 1.1*y_max])
    if bar_arrange == 'uniform':
        group_left.append(left)
        x_tick_pos = [
            group_left[ind] + 
            (group_left[ind+1] - group_left[ind] - bar_width) / 2. 
            for ind in range(len(group_left) - 1)]
    elif bar_arrange == 'grouped':
        x_tick_pos = numpy.arange(len(group_names)) + bar_width*(exp_ind+1)/2.
    group_labels = [category_label.get(name, name) for name in group_names]
    plt.xticks(x_tick_pos, group_labels)

def plot_2d(x_data, x_name='x_data', y_data=None, y_name='y_data', yerr=None,
            groups=None, identifiers=None, graph_type='scatter', 
            line_width_=None, fit=None):
    """
    Min part for plottings a 2d graph.

    If specified, args groups and identifiers specify the order of groups
    and experiments on the x axis. 

    Plots on the current figure.     

    Arguments:
      - x_data, y_data: data objects, have to be instances of Groups, 
      Observations or Experiment and they both have to be instances of the 
      same class
      - x_name, y_name: names of properties of x_data and y_data that are 
      plotted on x and y axis  
    """

    # y data
    if y_data is None:
        y_data = x_data

    # determine data type and set group order
    if (isinstance(x_data, Groups) and isinstance(y_data, Groups)):
        data_type = 'groups'
        if groups is None:
            group_names = list(x_data.keys()) 
        else:
            group_names = groups
    elif (isinstance(x_data, Observations) 
          and isinstance(y_data, Observations)):
        data_type = 'observations'
        group_names = ['']
    elif (isinstance(x_data, pyto.analysis.Experiment)
          and isinstance(y_data, pyto.analysis.Experiment)):
        data_type = 'experiment'
        group_names = ['']
    else:
        raise ValueError(
            "Arguments x_data and y_data have to be instances of Groups, "
            + "Observations or Experiment and they need to be instances "
            + "of the same class.")

    # line style and width
    if graph_type == 'scatter':
        loc_line_style = ''
        loc_marker = marker
    elif graph_type == 'line':
        loc_line_style = default_line_style
        loc_marker = ''
    if line_width_ is None:
        loc_line_width = default_line_width
    elif line_width_ == 'thick':
        loc_line_width = thick_line_width

    # loop over groups
    figure = None
    markers_default_copy = copy(markers_default)
    for group_nam, group_ind in zip(group_names, list(range(len(group_names)))):

        # get data
        if data_type == 'groups':
            x_group = x_data[group_nam]
            y_group = y_data[group_nam]
        elif data_type == 'observations':
            x_group = x_data
            y_group = y_data
        elif data_type == 'experiment':
            x_group = Observations()
            x_group.addExperiment(experiment=x_data)
            y_group = Observations()
            y_group.addExperiment(experiment=y_data)

        # set experiment order
        if identifiers is None:
            loc_identifs = x_group.identifiers
        elif isinstance(identifiers, list):
            loc_identifs = [ident for ident in identifiers 
                            if ident in x_group.identifiers]
        elif isinstance(identifiers, dict):
            loc_identifs = identifiers[group_nam]

        # loop over experiments
        for ident, exp_ind in zip(loc_identifs, list(range(len(loc_identifs)))):

            # values
            if (data_type == 'groups') or (data_type == 'observations'):
                x_value = x_group.getValue(identifier=ident, property=x_name)
                y_value = y_group.getValue(identifier=ident, property=y_name)
            elif data_type == 'experiment':
                x_ident = x_data.identifier
                x_value = x_group.getValue(identifier=x_ident, property=x_name)
                y_ident = y_data.identifier
                y_value = y_group.getValue(identifier=y_ident, property=y_name)

            # cut data to min length
            if len(x_value) != len(y_value):
                min_len = min(x_value, y_value)
                x_value = x_value[:min_len]
                x_value = y_value[:min_len]

            # adjust colors
            loc_alpha = alpha.get(ident, 1)
            if graph_type == 'scatter':
                loc_marker = marker.get(ident, None)
                if loc_marker is None:
                    loc_marker = markers_default_copy.pop(0)
            loc_color = color.get(ident, None)

            # plot data points
            label = (group_nam + ' ' + ident).strip()
            if loc_color is not None:
                figure = plt.plot(
                    x_value, y_value, linestyle=loc_line_style, color=loc_color,
                    linewidth=loc_line_width, marker=loc_marker, 
                    markersize=marker_size, alpha=loc_alpha, label=ident)
            else:
                figure = plt.plot(
                    x_value, y_value, linestyle=loc_line_style,
                    linewidth=loc_line_width, marker=loc_marker, 
                    markersize=marker_size, alpha=loc_alpha, label=ident)

            # plot eror bars
            if yerr is not None:
                try:
                    yerr_value = y_group.getValue(
                        identifier=ident, property=yerr)
                    # arg color needed otherwise makes line with another color
                    if loc_color is not None:
                        plt.errorbar(
                            x_value, y_value, yerr=yerr_value, color=loc_color, 
                            ecolor=loc_color, label='_nolegend_')
                    else:
                        plt.errorbar(
                            x_value, y_value, yerr=yerr_value, 
                            label='_nolegend_')
                except AttributeError:
                    pass        

            # plot fit line
#            if fit is not None:
            if fit is not None:

                # data limits
                x_max = x_value.max()
                x_min = x_value.min()
                y_max = y_value.max()
                y_min = y_value.min()

                # fit line parameters
                a_reg = x_group.getValue(identifier=ident, property=fit[0])
                b_reg = y_group.getValue(identifier=ident, property=fit[1])

                # fit limits
                x_range = numpy.arange(x_min, x_max, (x_max - x_min) / 100.)
                poly = numpy.poly1d([a_reg, b_reg])
                y_range = numpy.polyval(poly, x_range)
                start = False
                x_fit = []
                y_fit = []
                for x, y in zip(x_range, y_range):
                    if (y >= y_min) and (y <= y_max):
                        x_fit.append(x)
                        y_fit.append(y)
                        start = True
                    else:
                        if start:
                            break
                # plot fit
                if loc_color is not None:
                    plt.plot(
                        x_fit, y_fit, linestyle=default_line_style, 
                        color=loc_color, linewidth=loc_line_width, marker='', 
                        alpha=loc_alpha)
                else:
                    plt.plot(
                        x_fit, y_fit, linestyle=default_line_style,
                        linewidth=loc_line_width, marker='', alpha=loc_alpha)

    return figure

def get_confidence_stars(value, limits):
    """
    Returns number of stars for a given confidence level(s).
    """

    # more than one value
    if isinstance(value, (numpy.ndarray, list)):
        result = [get_confidence_stars(x, limits) for x in value]
        return result

    # one value
    result = 0
    for lim in limits:
        if value <= lim:
            result += 1
        else:
            break

    return result

def save_data(object, base, name=['mean', 'sem'], categories=categories):
    """
    Saves indexed data in a file. If more than one property is specified, the 
    corresponding values are saved in separate files. Each row contains
    values for one index. Indices are saved in the first coulmn, while each 
    other column correspond to one of the identifiers.

    Arguments:
      - object: (Observations) object that contains data
      - base: file name is created as base_property_name
      - name: names of properties that need to be saved
      - categories: categories
    """

    ids = object.ids[0]
    result = numpy.zeros(shape=(len(ids), len(categories)+1))

    # loop over properties
    if not isinstance(name, (list, tuple)):
        name = [name]
    for one_name in name:
        result[:, 0] = ids

        # make array that contains all values for current property
        for group, group_ind in zip(categories, list(range(len(categories)))):
            values =  object.getValue(identifier=group, name=one_name)
            result[:, group_ind+1] = values

        # write current array
        format = ' %i '
        header = 'index'
        for categ in categories:
            format += '      %8.5f'
            header += '  ' + categ
        file_ = base + '_' + one_name
        numpy.savetxt(file_, result, fmt=format, header=header)

##############################################################
#
# Functions that calculate special properites 
#

def findSpecialThreshold(layers, segments, fraction, new, thresh_str='_thr-',
                         eps=1.e-9, groups=None, identifiers=None):
    """
    Calculates the threshold that is at the fraction between the boundary and 
    cleft densities and finds the segmentation file that is the closest to
    that threshold, for identifier.

    Requires that the hierarchy and individual threshold segmentation files 
    have the same names except for the '_thr-<threshold value>' string just
    in front of the extension (like the default in the cleft script).

    Arguments:
      - layers: cleft layers
      - segments: hierarchical segmentation
      - fraction: fraction
      - new: name used to save the calculated (fraction threshold) property
      - thresh_str: threshold string
      - eps: numerical error for floats
      - grous: categories
      - identifiers: identifiers

    Sets: Fraction threshold property, with name given by arg new

    Returns: dictionary with keys as identifiers and the special threshold 
    segmentation files. 
    """

    # find and save fraction of means
    layers.getRelative(
        fraction=fraction, name='mean', new=new, region=['bound', 'cleft'], 
        weight='volume', categories=groups)

    # get groups
    if groups is None:
        groups = list(cleft.keys())

    # loop over groups
    special_files = {}
    for categ in groups:

        # loop over experiments (identifiers)
        categ_identifiers = layers[categ].identifiers
        for ident in categ_identifiers:
 
            # skip identifiers that were not passed
            if identifiers is not None:
                if ident not in identifiers:
                    continue
 
            # find closest threshold
            thresholds = segments[categ].getValue(
                name='thresholds', identifier=ident) 
            fract_thresh = layers[categ].getValue(name=new, identifier=ident)
            start = True
            for thresh in thresholds:
                diff = numpy.abs(thresh - fract_thresh)
                if start:
                    best_thresh = thresh
                    best_diff = diff                    
                    start = False
                elif diff < best_diff:
                    best_thresh = thresh
                    best_diff = diff

            # hierarchy segmentation file
            hi_file = segments[categ].getValue(
                name='cleft_segmentation_hi', identifier=ident)
            hi_file_dir, hi_file_base = os.path.split(hi_file)
            [hi_file_root, hi_file_ext] = hi_file_base.rsplit('.', 1)

            # find file that matches the hierarchy file with best threshold
            found = False
            all_files = os.listdir(hi_file_dir)
            for file_ in all_files:

                # check if extensions the same
                [file_root, file_ext] = file_.rsplit('.', 1)
                if file_ext != hi_file_ext: continue

                # check if threshold string present
                if thresh_str not in file_root: continue
               
                # check if beginnings the same
                [file_root_main, file_root_thresh] = file_root.split(
                    thresh_str)
                if file_root_main != hi_file_root: continue

                # check if float after the threshold string
                try:
                    curr_thresh = float(file_root_thresh)
                except ValueError:
                    continue
               
                # check if thresholds agree
                if numpy.abs(curr_thresh - best_thresh) < eps:
                    new_file = os.path.join(hi_file_dir, file_)
                    try:
                        special_files[categ].update({ident : new_file})
                    except KeyError:
                        special_files[categ] = {ident : new_file}
                    found = True
                    break

            if not found:
                logging.warning(
                    "Could not find segmentation file for group " + categ
                    + ", identifier " + ident + " and threshold " 
                    + str(best_thresh) + ".")
    
    return special_files

def getSpecialThreshold(cleft, segments, fraction, 
                       groups=None, identifiers=None):
    """
    Return threshold closest to the 

    Arguments:
      - cleft: (Groups)
    """

    # get groups
    if groups is None:
        groups = list(cleft.keys())

    # loop over groups
    fract_thresholds = {}
    fract_densities = {}
    for categ in groups:

        # loop over experiments (identifiers)
        categ_identifiers = cleft[categ].identifiers
        for identif in categ_identifiers:
 
            # skip identifiers that were not passed
            if identifiers is not None:
                if identif not in identifiers:
                    continue
 
            # get boundary and cleft ids
            bound_ids = cleft[categ].getValue(identifier=identif, 
                                             property='boundIds')
            cleft_ids = cleft[categ].getValue(identifier=identif, 
                                             property='cleftIds')

            # get mean boundary and cleft and fractional densities
            bound_densities = cleft[categ].getValue(
                identifier=identif, property='mean', ids=bound_ids)
            bound_volume = cleft[categ].getValue(
                identifier=identif, property='volume', ids=bound_ids)
            bound_density = numpy.dot(
                bound_densities, bound_volume) / bound_volume.sum() 
            cleft_densities = cleft[categ].getValue(
                identifier=identif, property='mean', ids=cleft_ids)
            cleft_volume = cleft[categ].getValue(
                identifier=identif, property='volume', ids=cleft_ids)
            cleft_density = numpy.dot(
                cleft_densities, cleft_volume) / cleft_volume.sum() 
            fract_density = (
                bound_density + (cleft_density - bound_density) * fraction
            
            # get closest threshold
            # ERROR thresholds badly formated in segments
            all_thresh = segments[categ].getValue(
                identifier=identif, property='thresh')
            index = numpy.abs(all_thresh - fract_density).argmin()
            thresh = all_thresh[index]
            thresh_str = "%6.3f" % thresh
            try:
                fract_thresholds[categ][identif] = thresh_str
                fract_densities[categ][identif] = (bound_density, 
                                                   cleft_density, fract_density)
            except KeyError:
                fract_thresholds[categ] = {}
                fract_thresholds[categ][identif] = thresh_str
                fract_densities[categ] = {}
                fract_densities[categ][identif] = (bound_density, 
                                                   cleft_density, fract_density)

    return fract_densities

def get_occupancy(segments, layers, groups, name):
    """
    Occupancy is added to the segments object

    Arguments:
      - segments: (connections)
      - layers: (CleftRegions)
      - groups
      - name: name of the added (occupancy) property
    """

    for categ in groups:
        for ident in segments[categ].identifiers:

            seg_vol = segments[categ].getValue(identifier=ident, 
                                               property='volume').sum()
            total_vol = layers[categ].getValue(identifier=ident, 
                                               property='volume')
            cleft_ids = layers[categ].getValue(identifier=ident, 
                                               property='cleftIds')
            cleft_vol = total_vol[cleft_ids-1].sum()
            occup = seg_vol / float(cleft_vol)
            segments[categ].setValue(identifier=ident, property=name, 
                                     value=occup)

def get_cleft_layer_differences(data, name, ids, groups):
    """
    """

    #def abs_diff43(x):
    #    return x[3] - x[2]
    #def abs_diff65(x):
    #    return x[5] - x[4]

    # not good because apply mane the new property indexed
    #data.apply(funct=abs_diff43, args=[name], 
    #           name='diffNormalMean43', categories=groups)
    #data.apply(funct=abs_diff65, args=[name], 
    #           name='diffNormalMean65', categories=groups)

    for categ in groups:
        for ident in data[categ].identifiers:

            # 4 - 3
            val4 = data[categ].getValue(
                identifier=ident, property=name, ids=ids[0])
            val3 = data[categ].getValue(
                identifier=ident, property=name, ids=ids[1])
            difference = val4 - val3
            new_name = 'diff_' + name + '_' + str(ids[0]) + '_' +  str(ids[1])
            data[categ].setValue(
                identifier=ident, property=new_name, value=difference)

            # 6 - 5
            #val6 = data[categ].getValue(
            #    identifier=ident, property=name, ids=[6])[0]
            #val5 = data[categ].getValue(
            #    identifier=ident, property=name, ids=[5])[0]
            #diff65 = val6 - val5
            #data[categ].setValue(
            #    identifier=ident, property='diffNormalMean65', value=diff65)


################################################################
#
# Main function (edit to add additional analysis files and plot (calculate)
# other properties)
#
###############################################################

def main(individual=False, save=False, analyze=False):
    """
    Arguments:
      - individual: if True read data for each tomo separately, otherwise 
      read from pickles containing partailly analyzed data
      - save: if True save pickles containing partailly analyzed data
    """

    ##########################################################
    #
    # Read data and calculate few things
    #

    # make catalog and groups
    if __name__ == '__main__':
        try:
            curr_dir, base = os.path.split(os.path.abspath(__file__))
        except NameError:
            # needed for running from ipython
            curr_dir = os.getcwd()
    else:
        curr_dir = os.getcwd()

    cat_dir = os.path.normpath(os.path.join(curr_dir, catalog_directory))
    catalog = pyto.analysis.Catalog(catalog=catalog_pattern, dir=cat_dir,
                                    identifiers=identifiers)
    catalog.makeGroups(feature='treatment')

    # set global
    global layers, layers_4, layers_8, layers_conn, layers_4_conn, columns
    global layers_4_col_1, layers_4_col_2, layers_4_col_3, layers_4_col_4
    global conn, conn_05, layers_4_conn, conn_col4

    ##########################################################
    #
    # Read individual tomo data, calculate few things and save this 
    # preprocessed data, or just read the preprocessed data 
    #

    if individual:

        # read individual experiment data 
        cl_files = getattr(catalog, layers_name)
        logging.info("Reading layers")
        layers = pyto.analysis.CleftRegions.read(
            files=cl_files, mode='layers', categories=categories, 
            catalog=catalog) 
        cl_4_files = getattr(catalog, layers_4_name)
        logging.info("Reading layers-4")
        layers_4 = pyto.analysis.CleftRegions.read(
            files=cl_4_files, mode='layers', categories=categories, 
            catalog=catalog) 
        cl_8_files=getattr(catalog, layers_8_name)
        logging.info("Reading layers-8")
        layers_8 = pyto.analysis.CleftRegions.read(
	    files=cl_8_files, mode='layers', categories=categories,
	    catalog=catalog)
        cl_files = getattr(catalog, columns_name)
        logging.info("Reading columns")
        columns = pyto.analysis.CleftRegions.read(
            files=cl_files, mode='columns', reference=layers_4, 
            categories=categories, catalog=catalog) 
        files = getattr(catalog, layers_4_on_column_1_name)
        logging.info("Reading layers on columns 1")
        layers_4_col_1 = pyto.analysis.CleftRegions.read(
                files=files, mode='layers_on_columns', reference=layers_4, 
                categories=categories, catalog=catalog) 
        files = getattr(catalog, layers_4_on_column_2_name)
        logging.info("Reading layers on columns 2")
        layers_4_col_2 = pyto.analysis.CleftRegions.read(
                files=files, mode='layers_on_columns', reference=layers_4, 
                categories=categories, catalog=catalog) 
        files = getattr(catalog, layers_4_on_column_3_name)
        logging.info("Reading layers on columns 3")
        layers_4_col_3 = pyto.analysis.CleftRegions.read(
                files=files, mode='layers_on_columns', reference=layers_4, 
                categories=categories, catalog=catalog) 
        files = getattr(catalog, layers_4_on_column_4_name)
        logging.info("Reading layers on columns 4")
        layers_4_col_4 = pyto.analysis.CleftRegions.read(
                files=files, mode='layers_on_columns', reference=layers_4, 
                categories=categories, catalog=catalog) 

        conn_files = getattr(catalog, connections_name)
        logging.info("Reading connections")
        conn = pyto.analysis.Connections.read(
            files=conn_files, mode='cleft', catalog=catalog,
            categories=categories)
        #conn_files = getattr(catalog, connections_norm_05_name)
        conn_05_files = findSpecialThreshold(
            layers=layers_4, segments=conn, fraction=0.5, new='density_05',
            groups=categories)
        logging.info("Reading connections at 0.5 normalized threshold")
        conn_05 = pyto.analysis.Connections.read(
            files=conn_05_files, mode='cleft', catalog=catalog,
            categories=categories, single=True)
        #files = getattr(catalog, layers_on_connections_name)
        #layers_conn = pyto.analysis.CleftRegions.read(
        #    files=files, mode='layers_cleft', categories=categories, 
        #    catalog=catalog) 
        files = getattr(catalog, layers_4_on_connections_name)
        logging.info("Reading connections on layers-4")
        layers_4_conn = pyto.analysis.CleftRegions.read(
            files=files, mode='layers_cleft', categories=categories, 
            catalog=catalog) 

        # connections on columns
	conn_col4_files = getattr(catalog, conn_col4_name)
        logging.info("Reading connections on column 4")
        conn_col4 = pyto.analysis.Connections.read(
            files=conn_col4_files, mode='cleft', catalog=catalog,
            categories=categories)
	
        # keep only cleft elements
        #layers_4_conn = layers_4_conn.syncam_ox.split(name='ids', 
        #                                              bins=[2.5, 6.5])[0]

        # pickle
        if save:
            pickle.dump(conn, open(connections_pkl, 'wb'), -1)
            pickle.dump(conn_05, open(connections_norm_05_pkl, 'wb'), -1)
            pickle.dump(layers, open(layers_pkl, 'wb'), -1)
            pickle.dump(layers_4, open(layers_4_pkl, 'wb'), -1)
            pickle.dump(layers_4_col_1, 
                        open(layers_4_on_column_1_pkl, 'wb'), -1)
            pickle.dump(layers_4_col_2, 
                        open(layers_4_on_column_2_pkl, 'wb'), -1)
            pickle.dump(layers_4_col_3, 
                        open(layers_4_on_column_3_pkl, 'wb'), -1)
            pickle.dump(layers_4_col_4, 
                        open(layers_4_on_column_4_pkl, 'wb'), -1)
            pickle.dump(layers_8, open(layers_8_pkl, 'wb'), -1)
            pickle.dump(columns, open(columns_pkl, 'wb'), -1)
            #pickle.dump(layers_conn, open(layers_on_connections_pkl, 'wb'), -1)
            pickle.dump(layers_4_conn, 
                        open(layers_4_on_connections_pkl, 'wb'), -1)

    else:

        # unpickle
        conn = pickle.load(open(connections_pkl))
        conn_05 = pickle.load(open(connections_norm_05_pkl))
        conn_col4 = pickle.load(open(connections_col4_pkl))
        layers = pickle.load(open(layers_pkl))
        layers_4 = pickle.load(open(layers_4_pkl))
        layers_4_col_1 = pickle.load(open(layers_4_on_column_1_pkl))
        layers_4_col_2 = pickle.load(open(layers_4_on_column_2_pkl))
        layers_4_col_3 = pickle.load(open(layers_4_on_column_3_pkl))
        layers_4_col_4 = pickle.load(open(layers_4_on_column_4_pkl))
        layers_8 = pickle.load(open(layers_8_pkl))
        columns = pickle.load(open(columns_pkl))
        #layers_conn = pickle.load(open(layers_on_connections_pkl))
        layers_4_conn = pickle.load(open(layers_4_on_connections_pkl))
        #layers_8_conn = pickle.load(open(layers_8_on_connections_pkl))

        # keep only specified
        for obj in [
            conn, conn_05, conn_col4, layers, layers_4, layers_4_col_1, 
            layers_4_col_2, layers_4_col_3, layers_4_col_4, layers_8,
            columns, layers_4_conn]:
            obj.keep(identifiers=identifiers)


    ##########################################################
    #
    # Set other properties
    #

    # AZ surface for layers
    layers.getBoundarySurfaces(
        names=('psd_surface_um', 'az_surface_um'), surface='surface_nm', 
        categories=categories, factor=1.e-6)

    # membrane-only normalization
    layers_4.normalizeByMean(name='mean', normalName='absMean', mode='absolute',
                             region='bound', categories=categories)
    columns.normalizeByMean(
        name='mean', normalName='absMean', mode='absolute', region='bound', 
        categories=categories, reference=layers_4)
    layers_4_col_1.normalizeByMean(
        name='mean', normalName='absMean', mode='absolute', region='bound', 
        categories=categories, reference=layers_4)
    layers_4_col_2.normalizeByMean(
        name='mean', normalName='absMean', mode='absolute', region='bound', 
        categories=categories, reference=layers_4)
    layers_4_col_3.normalizeByMean(
        name='mean', normalName='absMean', mode='absolute', region='bound', 
        categories=categories, reference=layers_4)
    layers_4_col_4.normalizeByMean(
        name='mean', normalName='absMean', mode='absolute', region='bound', 
        categories=categories, reference=layers_4)

    # cleft layer density differences
    get_cleft_layer_differences(data=layers_4, name='normalMean', 
                                ids=[3,4], groups=categories)
    get_cleft_layer_differences(data=layers_4_col_1, name='normalMean', 
                                ids=[3,4], groups=categories)
    get_cleft_layer_differences(data=layers_4_col_2, name='normalMean', 
                                ids=[3,4], groups=categories)
    get_cleft_layer_differences(data=layers_4_col_3, name='normalMean', 
                                ids=[3,4], groups=categories)
    get_cleft_layer_differences(data=layers_4_col_4, name='normalMean', 
                                ids=[3,4], groups=categories)

    if not analyze: return


    ##########################################################
    #
    # AZ and psd
    #

    #  psd surface, all experiments
    stats(data=layers, name='psd_surface_um', join='join', groups=categories, 
          identifiers=identifiers, test='t', reference=reference,
          y_label=r'PSD surface area [${\mu}m^2$]', title="PSD surface")

    #  psd surface, individual experiments
    stats(data=layers, name='psd_surface_um', join=None, groups=categories, 
          identifiers=identifiers, test='t', reference=reference,
          y_label=r'PSD surface area [${\mu}m^2$]', title="PSD surface")

    #  az surface
    stats(data=layers, name='az_surface_um', join='join', groups=categories, 
          identifiers=identifiers, test='t', reference=reference,
          y_label=r'AZ area [${\mu}m^2$]', title="AZ surface")

    ##########################################################
    #
    # Analyze connectors
    #

    # n connectors, all experiments
    stats(data=conn, name='nSegments', groups=categories, 
          identifiers=identifiers, test='t', reference=reference,
          title="N connectors, experiment values") 

    # n connectors, average per treatment
    stats(data=conn, name='nSegments', join='join', groups=categories, 
          identifiers=identifiers, test='t', reference=reference,
          title="N segments, treatment means")
    
    # connector surface density, all experiments
    stats(data=conn, name='surfaceDensity_nm', groups=categories, 
          identifiers=identifiers, test='t', reference=reference,
          title="Connector surface density, experiment values") 

    # connector surface density, treatment average
    stats(data=conn, name='surfaceDensity_nm', join='join', groups=categories, 
          identifiers=identifiers, test='t', reference=reference,
          title="Connector surface density, treatment means") 

    # n connectors vs cleft angle
    correlation(
        xData=layers, xName='angleToYDeg', yData=conn, yName='nSegments', 
        test='r', join='join', groups=categories, identifiers=identifiers, 
        format_=print_format, title='N connectors dependence on missing wedge')
    plt.axis([0, 90, 0, plt.axis()[3]*1.1])

    # connector length, all experiments
    stats(data=conn, name='length_nm', join=None, groups=categories, 
          identifiers=identifiers, test=None, reference=None,
          title="Connector length, experiment means")

    # connector length, all experiment, test between experiments
    ref = {'syncam_ox_ctrl':'syncam_ox_ctrl_1', 'syncam_ox':'syncam_ox_1'}
    stats(data=conn, name='length_nm', join=None, groups=categories, 
          identifiers=identifiers, test='t', reference=ref,
          title="Connector length, experiment means")

    # connector length, treatment means
    stats(data=conn, name='length_nm', join='join', groups=categories, 
          identifiers=identifiers, test='t', reference=reference,
          title="Connector length, treatment means")

    # connector length, mean of experiment means
    stats(data=conn, name='length_nm', join='mean', groups=categories, 
          identifiers=identifiers, test='t', reference=reference,
          title="Connector length, means of experiment means")

    # n contacts to post side, all experiments
    stats(data=conn, name='nContacts_1', join=None, groups=categories, 
          identifiers=identifiers, test=None, reference=None,
          title="N contacts per segment, post side, experiment means")

    # n contacts to pre side, all experiments
    stats(data=conn, name='nContacts_2', join=None, groups=categories, 
          identifiers=identifiers, test=None, reference=None,
          title="N contacts per segment, pre side, experiment means")

    # contact surface density on post side, all experiments
    stats(data=conn, name='surfaceDensityContacts_1_nm', join=None, 
          groups=categories, identifiers=identifiers, test=None, reference=None,
          title="Contact surface density on post side, experiment means")

    # contact surface density on pre side, all experiments
    stats(data=conn, name='surfaceDensityContacts_2_nm', join=None, 
          groups=categories, identifiers=identifiers, test=None, reference=None,
          title="Contact surface density on pre side, experiment means")

    # contact surface density on post side, treatment means
    stats(data=conn, name='surfaceDensityContacts_1_nm', join='join', 
          groups=categories, identifiers=identifiers, test='t',
          reference=reference,
          title="Contact surface density on post side, treatment means")

    # contact surface density on pre side, treatment means
    stats(data=conn, name='surfaceDensityContacts_2_nm', join='join', 
          groups=categories, identifiers=identifiers, test='t',
          reference=reference,
          title="Contact surface density on pre side, treatment means")

    # topology all experiments
    stats(data=conn, name='nLoops', join=None, groups=categories, 
          identifiers=identifiers, test=None, reference=None,
          title="N loops, experiment means")

    # topology treatment means
    stats(data=conn, name='nLoops', join='join', groups=categories, 
          identifiers=identifiers, test='t', reference=reference,
          title="N loops, experiment means")

    ##########################################################
    #
    # Analyze cleft
    #

    # cleft width
    stats(data=layers, name='width_nm', join=None, groups=categories, 
          identifiers=identifiers, test=None, reference=None,
          title="Cleft width [nm], all experiments")
    stats(data=layers, name='width_nm', join='join', groups=categories, 
          identifiers=identifiers, test='t', reference=reference,
          title="Cleft width [nm], treatment means")

    # cleft density profile, normalized membrane and cleft 
    plot_cleft_layers(
        data=layers_4, xName='ids', yName='normalMean', yerr='sem', 
        groups=categories, identifiers=identifiers, mode='mean',
        title='Cleft density by layers')
    axis([0.8, 9.2, -0.1, 1.5])
    plot([2.5, 2.5], [0,1], linestyle='--', color='black', linewidth=4)
    plot([6.5, 6.5], [0,1], linestyle='--', color='black', linewidth=4)
    plot_cleft_layers(
        data=layers_4, xName='ids', yName='normalMean', 
        groups=categories, identifiers=identifiers, mode='all&mean')
    plot_cleft_layers(
        data=layers_4, xName='ids', yName='normalMean',
        groups=categories, identifiers=identifiers, mode='all')

    # cleft density profile, normalized membrane only
    plot_cleft_layers(
        data=layers_4, xName='ids', yName='absMean', yerr='sem', 
        groups=categories, identifiers=identifiers, mode='mean',
        title='Cleft density by layers')

    # cleft density profile, not normalized  
    plot_cleft_layers(data=layers_4, xName='ids', yName='mean', 
                      groups=categories, identifiers=identifiers, mode='mean')
    plot_cleft_layers(data=layers_4, xName='ids', yName='mean', 
                      groups=categories, identifiers=identifiers, mode='all')
    plot_cleft_layers(data=layers, xName='ids', yName='mean',
                      groups=categories, identifiers=identifiers, mode='all')

    # cleft density profile on segments, normalized  
    plot_cleft_layers(
        data=layers_4_conn, xName='ids', yName='normalMean', yerr='sem', 
        groups=categories, identifiers=identifiers, mode='mean')
    plot_cleft_layers(
        data=layers_4_conn, xName='ids', yName='normalVolume', yerr='sem', 
        groups=categories, identifiers=identifiers, mode='mean')

    # celft density differences
    stats(data=layers_4, name='diff_normalMean_3_4', join='join', 
          groups=categories, identifiers=identifiers, test='t', 
          reference=reference, title="Cleft density difference 3-4")
    stats(data=layers_4_col_4, name='diff_normalMean_3_4', join='join', 
          groups=categories, identifiers=identifiers, test='t', 
          reference=reference, title="Cleft density column 4 difference 3-4")
    stats(data=layers_4, name='diff_normalMean_6_5', join='join', 
          groups=categories, identifiers=identifiers, test='t', 
          reference=reference, title="Cleft density difference 6-5")

    # cleft layers on columns, normalized
    plot_cleft_layers(
        data=layers_4_col_1, xName='ids', yName='normalMean', yerr='sem', 
        groups=categories, identifiers=identifiers, mode='mean',
        title='Cleft density by layers on column 1')
    plot_cleft_layers(
        data=layers_4_col_2, xName='ids', yName='normalMean', yerr='sem', 
        groups=categories, identifiers=identifiers, mode='mean',
        title='Cleft density by layers on column 2')
    plot_cleft_layers(
        data=layers_4_col_3, xName='ids', yName='normalMean', yerr='sem', 
        groups=categories, identifiers=identifiers, mode='mean',
        title='Cleft density by layers on column 3')
    plot_cleft_layers(
        data=layers_4_col_4, xName='ids', yName='normalMean', yerr='sem', 
        groups=categories, identifiers=identifiers, mode='mean',
        title='Cleft density by layers on column 4')

    # cleft columns
    plot_cleft_layers(data=columns, xName='ids', yName='mean',
                      groups=categories, identifiers=identifiers, mode='all',
                      x_label='Radial columns', title='Cleft columns')
    plot_cleft_layers(data=columns, xName='ids', yName='mean', yerr='sem',
                      groups=categories, identifiers=identifiers, mode='mean',
                      x_label='Radial columns', title='Cleft columns')

    # cleft density
    stats(data=layers, name='relativeMinCleftDensity', join=None, 
          groups=categories, identifiers=identifiers, test=None, reference=None,
          title="Minumum cleft density (relative)")
    stats(data=layers, name='relativeMinCleftDensity', join='join', 
          groups=categories, identifiers=identifiers, 
          test='t', reference=reference,
          title="Minumum cleft density (relative), treatment means")
    stats(data=layers, name='minCleftDensityPosition', join=None, 
          groups=categories, identifiers=identifiers, test=None, reference=None,
          title="Minumum cleft density position (relative)")
    stats(data=layers, name='minCleftDensityPosition', join='join', 
          groups=categories, identifiers=identifiers, 
          test='t', reference=reference,
          title="Minumum cleft density position (relative), treatment means")

    # volume occupancy
    get_occupancy(segments=conn_n, layers=layers_4, groups=categories, 
                  name='occupancy')
    stats(data=conn_n, name='occupancy', join=None, groups=categories, 
          identifiers=identifiers)
    stats(data=conn_n, name='occupancy', join='join', groups=categories, 
          identifiers=identifiers, test='t', reference=reference, 
          title="Occupancy at 0.5 normalized threshold")
    get_occupancy(segments=conn, layers=layers_4, groups=categories, 
                  name='occupancy')
    stats(data=conn, name='occupancy', join=None, groups=categories, 
          identifiers=identifiers)
    stats(data=conn, name='occupancy', join='join', groups=categories, 
          identifiers=identifiers, test='t', reference=reference, 
          title="Occupancy of new segments")


    ########################################################
    #
    # Other 
    #
    

# run if standalone
if __name__ == '__main__':
    main(individual=True, save=True)
