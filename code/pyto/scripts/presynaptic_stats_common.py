#!/usr/bin/env python
"""
Functions used to display analysys results for presynaptic_stats.py script.

This was previously part of presynaptic_stats.py.

Work in progress (03.2018)

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id$
"""
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from builtins import zip
from builtins import range
#from past.utils import old_div
from past.builtins import basestring

__version__ = "$Revision$"

import sys
import logging
from copy import copy, deepcopy

import numpy 
import scipy 
import scipy.stats
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError:
    pass

import pyto
from pyto.analysis.groups import Groups
from pyto.analysis.observations import Observations

logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s %(module)s.%(funcName)s():%(lineno)d %(message)s',
    datefmt='%d %b %Y %H:%M:%S')


##############################################################
#
# Functions (edit only if you know what you're doing)
#
##############################################################

##############################################################
#
# Higher level functions 
#

def analyze_occupancy(
    layer, bins, bin_names, pixel_size, groups=None, identifiers=identifiers,
    test=None, reference=None, ddof=1, out=sys.stdout, 
    outNames=None, title='', yerr='sem', confidence='stars', y_label=None):
    """
    Statistical analysis of sv occupancy divided in bins according to the 
    distance to the AZ.

    Arguments:
      - layer: (Layers) layer data structure
      - bins: (list) distance bins
      - bin_names: (list) names of distance bins, has to correspond to bins
      - groups: list of group names
      - test: statistical inference test type
      - reference: specifies reference data
      - ddof: differential degrees of freedom used for std
      - out: output stream for printing data and results
      - outNames: list of statistical properties that are printed
      - title: title
      - yerr: name of the statistical property used for y error bars
      - confidence: determines how confidence is plotted
      - y_label: y axis label (default 'occupancy')
    """

    # Note: perhaps should be moved to pyto.alanysis.Layers

    # rearange data by bins
    layer_bin = layer.rebin(bins=bins, pixel=pixel_size, categories=groups)

    # make a separate Groups object for each bin
    layer_list = layer_bin.splitIndexed()

    # convert to Groups where group names are distance bins and identifiers
    # are treatment names
    converted = pyto.analysis.Groups.joinExperimentsList(
        groups=groups, identifiers=identifiers,
        list=layer_list, listNames=bin_names, name='occupancy')

    # do statistics and plot
    result = stats(
        data=converted, name='occupancy', join=None, groups=bin_names, 
        identifiers=groups, test=test, reference=reference, ddof=ddof, 
        out=out, outNames=outNames, title=title, yerr=yerr, 
        label='experiment', confidence=confidence, y_label=y_label)

    return result

def stats_list(
    data, dataNames, name, join='join', bins=None, fraction=1, groups=None, 
    identifiers=None, test=None, reference=None, ddof=1, out=sys.stdout, 
    label=None, outNames=None, plot_=True, yerr='sem', confidence='stars',
    title='', x_label=None, y_label=None):
    """
    Statistical analysis of data specified as a list of Groups objects.

    First, the data from idividual observations of each group are joined. In 
    this way each group (of arg data) becomes one observation and the elements
    of data list become groups. 

    Arguments:
      - data: (list of Groups) list of data structures
      - dataNames: (list of strs) names corrensponfing to elements of arg data, 
      have to be in the same order as the data
      - name: name of the analyzed property
      - join: 'join' to join experiments, otherwise None
      - bins: (list) bins for making histogram
      - fraction: bin index for which the fraction is calculated
      - groups: list of group names
      - identifiers: list of identifiers
      - test: statistical inference test type
      - reference: specifies reference data
      - ddof: differential degrees of freedom used for std
      - out: output stream for printing data and results
      - outNames: list of statistical properties that are printed
      - yerr: name of the statistical property used for y error bars
      - plot_: flag indicating if the result are to be plotted
      - label: determines which color, alpha, ... is used, can be 'group' to 
      label by group or 'experiment' to label by experiment
      - confidence: determines how confidence is plotted
      - x_label: x axis label
      - y_label: y axis label, if not specified arg name used instead 
      - title: title
    """

    # make one groups object by joining observations
    together = Groups.joinExperimentsList(
        list=data, listNames=dataNames, name=name, mode=join,
        groups=groups, identifiers=identifiers)

    # do stats 
    result = stats(
        data=together, name=name, join=None, bins=bins, fraction=fraction, 
        groups=dataNames, identifiers=groups, test=test, reference=reference, 
        ddof=ddof, out=out, outNames=outNames, yerr=yerr, label='experiment', 
        confidence=confidence, title=title, x_label=x_label, y_label=y_label)

    return result

def stats_list_pair(
    data, dataNames, name, groups=None, identifiers=None, 
    test='t_rel', reference=None,  out=sys.stdout, yerr='sem', ddof=1, 
    outNames=None, plot_=True,label=None, confidence='stars',
    title='', x_label=None, y_label=None):
    """
    Statistical analysis of paired data specified as a list of Groups objects. 

    Unlike in stats_list(), the data has to be paired so that all Groups
    objects (elements of arg data) have to have the same group names, and the 
    same identifiers.

    First, the means of the data (arg name) from all idividual observations of 
    each group are calculated. In this way each group (of arg data) becomes one
    observation and the elements of data list become groups. 

    Arguments:
      - data: (list of Groups) list of data structures
      - dataNames: (list of strs) names corrensponfing to elements of arg data, 
      have to be in the same order as the elements of data
      - name: name of the analyzed property
      - groups: list of group names
      - identifiers: list of identifiers
      - test: statistical inference test type (default 't_rel')
      - reference: specifies reference data
      - ddof: differential degrees of freedom used for std
      - out: output stream for printing data and results
      - outNames: list of statistical properties that are printed
      - yerr: name of the statistical property used for y error bars
      - plot_: flag indicating if the result are to be plotted
      - label: determines which color, alpha, ... is used, can be 'group' to 
      label by group or 'experiment' to label by experiment
      - confidence: determines how confidence is plotted
      - x_label: x axis label
      - y_label: y axis label, if not specified arg name used instead 
      - title: title
    """

    # make one groups object by joining observations
    together = Groups.joinExperimentsList(
        list=data, listNames=dataNames, name=name, mode='mean', 
        removeEmpty=False)

    # do stats 
    result = stats(
        data=together, name=name, join='pair', 
        groups=dataNames, identifiers=groups, test=test, reference=reference, 
        ddof=ddof, out=out, outNames=outNames, yerr=yerr, label='experiment', 
        confidence=confidence, title=title, x_label=x_label, y_label=y_label)

    return result

def stats(data, name, bins=None, bin_names=None, fraction=None, join=None, 
          groups=None, identifiers=None, test=None, reference=None, ddof=1, 
          out=sys.stdout, label=None, outNames=None, plot_=True, plot_name=None,
          yerr='sem', confidence='stars', title='', x_label=None, y_label=None):
    """
    Does statistical analysis of data specified by args data and name, prints
    and plots the results as a bar chart.

    Argument join determines how the data is pooled across experiments.
    If join is 'join', data of individual experiments (observations) are 
    joined (pooled)  together within a group to be used for further 
    analysis. If it is 'mean', the mean value for each experiment is 
    calculated and these means are used for further analysis.

    Argument bins determined how the above obtained data is further 
    processed. If arg bins is not specified, basic stats (mean, std, sem)
    are calculated for all groups and the data is statistically compared 
    among the groups. 

    Alternatively, if arg bins is specified, histograms of the data are 
    calculated for all groups (property name 'histogram'). Histograms are 
    normalized to 1 to get probabilities (property name 'probability'). The 
    probability for bin indexed by arg fraction is saved separately as 
    property 'fraction'. For example, fraction of connected vesicles is 
    obtained for name='n_connection', bins=[0,1,100], fraction=1. The 
    histograms are statistically compared between groups.

    Joins 'join_bins' and 'byIndex' are described below. Specifically,
    the following types of analysis are implemented: 

      - join is None: a value is printed and a bar is plotted for each 
      experiment. This value is either the value of the specified property if 
      scalar, or a mean of the property values if indexed. If the data is 
      indexed, both significance between groups and between experiments 
      are calculated.

      - join='join', bins=None: Data is pooled across experiments of
      the same group, basic stats are calculated within groups and 
      statistically compared between groups.

      - join='join', bins specified (not None): Data is pooled across 
      experiments of the same group, histograms (acording to arg bins)
      of the data values are calculated within group and statistically 
      compared among groups.

      - join='mean', bins=None: Mean values are calculated for all
      experiments, basic stats are calculated for means within groups 
      and statistically compared between groups.

      - join='mean', bins specified (not None): Mean values are 
      calculated for all experiment, histograms (acording to arg bins)
      of the means are calculated within groups and statistically 
      compared between groups.

      - join='mean_bin', bins have to be specified (not None): 
      Histograms of the data values are calculated for each experiment 
      (acording to arg bins) and normalized to 1, basic stats are 
      calculated for values of the bin specified by arg within groups, 
      and statistically compared between groups
    
      - join='byIndex', bins should not be specified: Basic stats 
      (mean, std, sem) are calculated for each index (position) 
      separately. Data has to be indexed, and all experiments within 
      one group have to have same ids.

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

    # determine which property to plot
    if plot_name is None:
        if (bins is None) or (join == 'mean_bin'):
            plot_name = 'mean'
        else:
            if bin_names is None:
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

            elif (join == 'join') or (join == 'mean') or (join == 'mean_bin'):

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
                    stats = histo_groups

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

def count_histogram(
    data, name='ids', dataNames=None, groups=None, identifiers=None, test=None,
    reference=None, out=sys.stdout, outNames=None, plot_=True, label=None, 
    plot_name='fraction', confidence='stars', title='', x_label=None, 
    y_label=None):
    """
    Analyses and plots the number of data items specified by arg name.

    If (arg) data is a list of Groups objects, makes a histogram of the number 
    of items for each group, so a histogram is calculated for each group. Bins 
    of one histogram corespond to Groups objects specified by (arg) data. The
    histograms are then compared statistically.

    Data from all experiemnts of a group are combined.

    If (arg) data is a Groups object, makes a histogram of the number 
    of items for each experiment identifier. It is expected that all groups 
    have the same identifiers. Bins of one histogram corespond to groups of the
    Groups objects specified by (arg) data. The histograms are then compared 
    statistically. 

    Arguments:
      - data: (list of Groups) list of data structures
      - name: name of the analyzed property
      - dataNames: (list of strs) names corrensponfing to elements of arg data, 
      have to be in the same order as the elements of data
      - groups: list of group names
      - identifiers: list of identifiers
      - test: statistical inference test type 
      - reference: specifies reference data
      - out: output stream for printing data and results
      - outNames: list of statistical properties that are printed
      - plot_: flag indicating if the result are to be plotted
      - plot_name: determines which values are plotted, can be 'count' for the
      number of elements in each bin, or 'fraction' for the fraction of 
      elements in respect to all bins belonging to the same histogram.
      - label: determines which color, alpha, ... is used, can be 'group' to 
      label by group or 'experiment' to label by experiment
      - confidence: determines how confidence is plotted
      - x_label: x axis label
      - y_label: y axis label, if not specified arg name used instead 
      - title: title
    """

    # make Groups object if data is a list
    if isinstance(data, list):

        # join list 
        class_ = data[0].__class__
        groups_data = class_.joinExperimentsList(
            list=data, name=name, listNames=dataNames,  mode='join',
            groups=groups, identifiers=identifiers)

        # do stats
        stats = groups_data.countHistogram(
            name=name, test=test, reference=reference, 
            groups=dataNames, identifiers=groups,
            out=out, outNames=outNames, format_=print_format, title=title)

        # adjust group and identifiers
        loc_groups = dataNames
        loc_identifiers = groups

    elif isinstance(data, Groups):

        # check
        if not data.isTransposable():
            raise ValueError(
                "Argument data has to be transposable, that is each group "
                + "has to contain the same experiment identifiers.")

        # do stats
        stats = data.countHistogram(
            name=name, groups=None, identifiers=groups, test=test, 
            reference=reference, out=out, outNames=outNames, 
            format_=print_format, title=title)

    else:
        raise ValueError("Argument data has to be a Groups instance or a list"
                         + " of Groups instances.")

    # plot
    if plot_:

        # prepare for plotting
        plt.figure()

        # plot
        plot_stats(
            stats=stats, name=plot_name, yerr=None, groups=loc_groups, 
            identifiers=loc_identifiers, label=label, confidence=confidence)

        # finish plotting
        plt.title(title)
        if y_label is None:
            y_label = name
        plt.ylabel(y_label)
        if x_label is not None:
            plt.xlabel(x_label)
        if legend:
            plt.legend()
        plt.show()

    return stats

def correlation(
    xData, xName, yName, yData=None, test=None, regress=True, 
    reference=reference, groups=None, identifiers=None, join=None, 
    out=sys.stdout, format_=print_format, title='', x_label=None, y_label=None):
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
      - x_label, y_label: x and y axis labels, if not specified args xName 
      and yName are used instead 
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

        # plot
        if plot_:
            plot_2d(x_data=corr, x_name='xData', y_name='yData', groups=None,
                    identifiers=groups, graph_type='scatter', fit=fit)

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
        if x_label is None:
            x_label = xName
        plt.xlabel(x_label)
        if y_label is None:
            y_label = yName
        plt.ylabel(y_label)
        if legend:
            plt.legend()
        plt.show()

    return corr

##############################################################
#
# Plot functions 
#

def plot_layers(
    data, yName='occupancy', xName='distance_nm', yerr=None, groups=None, 
    identifiers=None, mode='all', ddof=1, graphType='line', 
    x_label='Distance to the AZ [nm]', y_label='Vesicle occupancy', title=''):
    """
    Plots values of an indexed property specified by arg yName vs. another
    indexed property specified by arg xName as a line plot. Makes separate
    plots for each group of the arg groups.

    Plots sv occupancy by layer for if Layers object is given as arg data and 
    the default values of args xName and yName are used.

    If mode is 'all' or 'all&mean' data from all observations (experiments) of
    one group is plotted on one figure. If mode is 'all&mean' the group mean is 
    also plotted. If mode is 'mean' all group means are plotted together.

    Arguments:
      - data: (Groups or Observations) data structure
      - xName, yName: name of the plotted properties
      - yerr: property used for y-error
      - groups: list of group names
      - identifiers: list of identifiers
      - mode: 'all', 'mean' or 'all&mean'
      - ddof = difference degrees of freedom used for std
      - graphType: 'line' for line-graph or 'scatter' for a scatter-graph
      - x_label, y_label: labels for x and y axes
      - title: title (used only if mode is 'mean')
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
                plot_layers_one(
                    data=data[group_name], yName=yName, xName=xName, yerr=yerr,
                    identifiers=identifiers, mode=mode, graphType=graphType,
                    x_label=x_label, y_label=y_label, title=title)

        elif mode == 'mean':

            # calculate means, add distance_nm and plot (one graph)
            stats = data.joinAndStats(
                name=yName, mode='byIndex', groups=groups, 
                identifiers=identifiers, ddof=ddof, out=None, title=title)
            for group_name in groups:
                dist = data[group_name].getValue(
                    property=xName, identifier=data[group_name].identifiers[0])
                stats.setValue(property=xName, value=dist, 
                               identifier=group_name)
            plot_layers_one(
                data=stats, yName='mean', xName=xName, yerr=yerr, 
                identifiers=None, mode=mode, graphType=graphType, ddof=ddof,
                x_label=x_label, y_label=y_label, title='Mean')

    elif isinstance(data, Observations):

        # Observations: plot one graph
        plot_layers_one(
            data=data, yName=yName, xName=xName, yerr=yerr, 
            identifiers=identifiers, mode=mode, graphType=graphType, ddof=ddof,
            x_label=x_label, y_label=y_label)

    else:
        raise ValueError("Argument 'data' has to be either pyto.analysis.Groups"
                         + " or Observations.") 

    if mode == 'mean': return stats

def plot_layers_one(
    data, yName='occupancy', xName='distance_nm', yerr=None, identifiers=None, 
    mode='all', ddof=1, graphType='line', x_label='Distance to the AZ', 
    y_label='Vesicle occupancy', title=''):
    """
    Plots values of an indexed property specified by arg yName vs. another
    indexed property specified by arg xName as a line plot. 
    
    Only one group can be specified as arg data. Data for all observations 
    (experiments) of that group are plotted on one graph. 

    Arguments:
      - data: (Observations) data structure
      - xName, yName: name of the plotted properties
      - yerr: property used for y-error
      - groups: list of group names
      - identifiers: list of identifiers
      - mode: 'all', 'mean' or 'all&mean'
      - ddof = difference degrees of freedom used for std
      - graphType: 'line' for line-graph or 'scatter' for a scatter-graph
      - x_label, y_label: labels for x and y axes
      - title: title
    """
    # from here on plotting an Observations object
    fig = plt.figure()

    # set identifiers
    if identifiers is None:
        identifiers = data.identifiers
    identifiers = [ident for ident in identifiers if ident in data.identifiers]

    # plot data for each experiment
    for ident in identifiers:

        # plot data for the current experiment 
        line = plot_2d(x_data=data, x_name=xName, y_name=yName, yerr=yerr, 
                       identifiers=[ident], graph_type=graphType)

    # calculate and plot mean
    if mode == 'all&mean':
        exp = data.doStatsByIndex(
            name=yName, identifiers=identifiers, identifier='mean', ddof=ddof)
        if len(identifiers) > 0:
            #exp_dist = data.getExperiment(identifier=identifiers[0])

            # set x axis values
            x_values = data.getValue(identifier=identifiers[0], name=xName)
            if len(x_values) > len(exp.mean):
                x_values = x_values[:len(exp.mean)]
            exp.__setattr__(xName, x_values) 
            exp.properties.add(xName)
            exp.indexed.add(xName)

            # plot
            line = plot_2d(
                x_data=exp, x_name=xName, y_data=exp, y_name='mean', 
                yerr=yerr, graph_type=graphType, line_width_='thick')

    # finish plotting
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    ends = plt.axis()
    plt.axis([0, 250, 0, 0.3])
    if legend:
        plt.legend()
    plt.show()

def plot_histogram(data, name, bins, groups=None, identifiers=None, 
                   facecolor=None, edgecolor=None, x_label=None, title=None):
    """
    Plots data as a histogram.

    If more than one group is given (arg groups), data from all groups are 
    combined. Also data from all experiments are combined.

    Arguments:
      - data: (Groups or Observations) data
      - name: property name
      - bins: histogram bins
      - groups: list of group names, None for all groups
      - identifiers: experiment identifier names, None for all identifiers 
      - facecolor: histogram facecolor
      - edgecolor: histogram edgecolor
      - x_label: x axis label
      - title: title
    """

    # combine data
    if isinstance(data, Groups):
        obs = data.joinExperiments(name=name, mode='join', groups=groups,
                                   identifiers=identifiers)
    elif isinstance(data, Observations):
        obs = data
    exp = obs.joinExperiments(name=name, mode='join')
    combined_data = getattr(exp, name)

    # color
    if facecolor is None:
        if (groups is not None): 
            if len(groups)==1:
                facecolor = color.get(groups[0], None)
        else:
            if isinstance(groups, basestring):
                facecolor = color.get(groups, None)

    # plot
    plt.hist(combined_data, bins=bins, facecolor=facecolor, edgecolor=edgecolor)

    # finish plot
    if title is not None:
         plt.title(title)
    if x_label is None:
        x_label = name
    plt.xlabel(x_label)
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

    # set group order
    if groups is None:
        group_names = list(stats.keys()) 
    else:
        group_names = groups

    # find rough range of y-axis values (to plot cinfidence)
    y_values = [
        stats[group_nam].getValue(identifier=ident, property=name)
        for group_nam in group_names
        for ident in stats[group_nam].identifiers
        if ((identifiers is None) or (ident in identifiers))]
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
            len(stats[group_nam].identifiers) for group_nam in group_names)
        bar_width = numpy.floor(80 / max_identifs) / 100
    else:
        raise ValueError("bar_arrange has to be 'uniform' or 'grouped'.")

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

            # should be removed when Matplot lib 1.* not used anymore
            if mpl.__version__[0] == 1:
                plt.errorbar(
                    left+bar_width/2, value,yerr=yerr_num,
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
    group_labels = [category_label.get(g_name, g_name) 
                    for g_name in group_names]
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
                min_len = min(len(x_value), len(y_value))
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
            #label = (group_nam + ' ' + ident).strip()
            loc_label = category_label.get(ident, ident)
            if loc_color is not None:
                figure = plt.plot(
                    x_value, y_value, linestyle=loc_line_style, color=loc_color,
                    linewidth=loc_line_width, marker=loc_marker, 
                    markersize=marker_size, alpha=loc_alpha, label=loc_label)
            else:
                figure = plt.plot(
                    x_value, y_value, linestyle=loc_line_style,
                    linewidth=loc_line_width, marker=loc_marker, 
                    markersize=marker_size, alpha=loc_alpha, label=loc_label)

            # plot eror bars
            if yerr is not None:
                yerr_value = y_group.getValue(identifier=ident, property=yerr)
                # arg color needed otherwise makes line with another color
                plt.errorbar(
                    x_value, y_value, yerr=yerr_value, 
                    color=loc_color, ecolor=loc_color, label='_nolegend_')

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

    # find shortest ids
    if 'ids' in object.indexed:
        ids = object.ids[0]
        for group, group_ind in zip(categories, list(range(len(categories)))):
            current_ids = object.getValue(identifier=group, name='ids')
            if len(current_ids) < len(ids):
                ids = current_ids
        len_ids = len(ids)

    # loop over properties
    if not isinstance(name, (list, tuple)):
        name = [name]
    for one_name in name:

        # initialize results
        if one_name in object.indexed:
            result = numpy.zeros(shape=(len_ids, len(categories)+1))
            result[:, 0] = ids
        else:
            result = numpy.zeros(shape=(1, len(categories)+1))
            result[0,0] = 1

        # make array that contains all values for current property
        for group, group_ind in zip(categories, list(range(len(categories)))):
            values =  object.getValue(identifier=group, name=one_name)

            if one_name in object.indexed:
                len_values = len(values)
                if len_ids <= len_values:
                    result[:, group_ind+1] = values[:len_ids]
                else:
                    result[:len_values, group_ind+1] = values[:]

            else:
                result[0, group_ind+1] = values

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
# Functions that calculate certain properites 
#

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
            bound_density = (
                numpy.dot(bound_densities, bound_volume) / bound_volume.sum())
            cleft_densities = cleft[categ].getValue(
                identifier=identif, property='mean', ids=cleft_ids)
            cleft_volume = cleft[categ].getValue(
                identifier=identif, property='volume', ids=cleft_ids)
            cleft_density = (
                numpy.dot(cleft_densities, cleft_volume) / cleft_volume.sum()) 
            fract_density = (
                bound_density + (cleft_density - bound_density) * fraction)
            
            # get closest threshold
            # ERROR thresholds badly formated in segments
            all_thresh = segments[categ].getValue(identifier=identif, 
                                                  property='thresh')
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

def get_cleft_layer_differences(data, name, groups):
    """
    """

    def abs_diff43(x):
        return x[3] - x[2]
    def abs_diff65(x):
        return x[5] - x[4]

    # not good because apply mane the new property indexed
    #data.apply(funct=abs_diff43, args=[name], 
    #           name='diffNormalMean43', categories=groups)
    #data.apply(funct=abs_diff65, args=[name], 
    #           name='diffNormalMean65', categories=groups)

    for categ in groups:
        for ident in data[categ].identifiers:

            # 4 - 3
            val4 = data[categ].getValue(
                identifier=ident, property=name, ids=[4])[0]
            val3 = data[categ].getValue(
                identifier=ident, property=name, ids=[3])[0]
            diff43 = val4 - val3
            data[categ].setValue(
                identifier=ident, property='diffNormalMean43', value=diff43)

            # 6 - 5
            val6 = data[categ].getValue(
                identifier=ident, property=name, ids=[6])[0]
            val5 = data[categ].getValue(
                identifier=ident, property=name, ids=[5])[0]
            diff65 = val6 - val5
            data[categ].setValue(
                identifier=ident, property='diffNormalMean65', value=diff65)

def calculateVesicleProperties(data, layer=None, tether=None, categories=None):
    """
    Calculates additional vesicle related properties. 

    The properties calculated are:
      - 'n_vesicle'
      - 'az_surface_um' 
      - 'vesicle_per_area_um'
      - 'mean_tether_nm' (for non-tethered vesicles value set to numpy.nan) 
    """

    # calculate n vesicles per synapse
    data.getNVesicles(name='n_vesicle', categories=categories)

    # calculate az surface (defined as layer 1) in um
    if layer is not None:
        data.getNVesicles(
            layer=layer, name='az_surface_um', fixed=1, inverse=True,
            layer_name='surface_nm', layer_factor=1.e-6, categories=categories)

    # calculate N vesicles per unit az surface (defined as layer 1) in um
    if layer is not None:
        data.getNVesicles(
            layer=layer, name='vesicle_per_area_um', 
            layer_name='surface_nm', layer_factor=1.e-6, categories=categories)

    # calculate mean tether length for each sv
    if tether is not None:
        data.getMeanConnectionLength(conn=tether, name='mean_tether_nm', 
                                     categories=categories, value=numpy.nan)

def calculateTetherProperties(data, layer=None, categories=None):
    """
    Calculates additional vesicle related properties. 

    The properties calculated are:
      - 'n_tether'
      - 'tether_per_area_um'
    """

    # calculate n tethers per synapse (to be moved up before pickles are made)
    data.getN(name='n_tether', categories=categories)

    # calculate N tethers per unit az surface (defined as layer 1) in um
    if layer is not None:
        data.getN(
            layer=layer, name='tether_per_area_um', 
            layer_name='surface_nm', layer_factor=1.e-6, categories=categories)

def calculateConnectivityDistanceRatio(
        vesicles, initial, distances, name='n_tethered_ratio', categories=None):
    """
    """

    # calculate connectivity distances
    vesicles.getConnectivityDistance(
        initial=initial, name='conn_distance', distance=1,
        categories=categories)

    # shortcuts
    d0 = [distances[0], distances[0]]
    d1 = [distances[1], distances[1]]

    # find n vesicles at specified distances
    conndist_0_sv = vesicles.split(
        value=d0, name='conn_distance', categories=categories)[0]
    conndist_0_sv.getNVesicles(name='_n_conndist_0', categories=categories)
    vesicles.addData(
        source=conndist_0_sv, names={'_n_conndist_0':'_n_conndist_0'})
    conndist_1_sv = vesicles.split(
        value=d1, name='conn_distance', categories=categories)[0]
    conndist_1_sv.getNVesicles(name='_n_conndist_1', categories=categories)
    vesicles.addData(
        source=conndist_1_sv, names={'_n_conndist_1':'_n_conndist_1'})

    # calculate reatio
    vesicles.apply(
        funct=numpy.true_divide, args=['_n_conndist_1', '_n_conndist_0'], 
        name=name, categories=categories, indexed=False)

def str_attach(string, attach):
    """
    Inserts '_' followed by attach in front of the right-most '.' in string and 
    returns the resulting string.

    For example:
      str_attach(string='sv.new.pkl', attach='raw') -> 'sv.new_raw.pkl)
    """

    string_parts = list(string.rpartition('.'))
    string_parts.insert(-2, '_' + attach)
    res = ''.join(string_parts)

    return res

def connectivity_factorial(
        data, groups, identifiers=None, name='n_connection', mode='positive'):
    """
    Calculates interaction term for 4 sets of data obtained under two
    conditions.

    Uses property n_connection to calculate fraction connected for each 
    experiemnt. In other words, the data points for one condition consist 
    of individual values corresponding to experiments. 
    """
    
    # extract values
    #values = [
    #    numpy.array([len(x[x>0]) / float(len(x)) 
    #                 for x in getattr(data[group], name)]) 
    #    for group in groups]

    total_conn = []
    for group in groups:
        conn_values = []
        for ident in data[group].identifiers:
            if (identifiers is None) or (ident in identifiers):
                x = data[group].getValue(name=name, identifier=ident)
                if mode is None:
                    conn_values.extend(x)
                elif mode == 'join':
                    conn_values.append(x.sum() / float(len(x)))
                elif mode == 'positive':
                    conn_values.append(len(x[x>0]) / float(len(x)))
        total_conn.append(numpy.asarray(conn_values))

    # calculate
    anova_factorial(*total_conn)

def anova_factorial(data_11, data_12, data_21, data_22):
    """
    ANOVA analysis of 2x2 factorial experimental design.
    """

    # make sure ndarrays
    data_11 = numpy.asarray(data_11)
    data_12 = numpy.asarray(data_12)
    data_21 = numpy.asarray(data_21)
    data_22 = numpy.asarray(data_22)

    # all data
    tot = numpy.hstack((data_11, data_12, data_21, data_22))
    ss_tot = (tot**2).sum() - tot.sum()**2 / float(len(tot))

    # ss between columns
    ss_col = (
        numpy.hstack((data_11, data_21)).sum()**2 /
            (float(len(data_11) + len(data_21)))
        + numpy.hstack((data_12, data_22)).sum()**2 /
            (float(len(data_12) + len(data_22)))
        - tot.sum()**2 / float(len(tot)) )

    # ss between rows
    ss_row = (
        numpy.hstack((data_11, data_12)).sum()**2 /
            (float(len(data_11) + len(data_12)))
        + numpy.hstack((data_21, data_22)).sum()**2 /
            (float(len(data_21) + len(data_22)))
        - tot.sum()**2 / float(len(tot)) )
        
    # ss interaction
    ss_int = (
        data_11.sum()**2 / float(len(data_11))
        + data_12.sum()**2 / float(len(data_12)) 
        + data_21.sum()**2 / float(len(data_21)) 
        + data_22.sum()**2 / float(len(data_22)) 
        - tot.sum()**2 / float(len(tot))
        - (ss_col + ss_row) )

    # ss error
    ss_err = ss_tot - (ss_col + ss_row + ss_int)
    ms_err = ss_err / float(
        len(data_11) + len(data_12) + len(data_21) + len(data_22) - 4)

    # f values and significances
    f_col = ss_col / ms_err
    p_col = scipy.stats.f.sf(f_col, dfn=1, dfd=len(tot)-4)
    print("Columns (1&3 vs 2&4): f = %f6.2  p = %f7.5" % (f_col, p_col))

    f_row = ss_row / ms_err
    p_row = scipy.stats.f.sf(f_row, dfn=1, dfd=len(tot)-4)
    print("Rows (1&2 vs 3&4):    f = %f6.2  p = %f7.5" % (f_row, p_row))

    f_int = ss_int / ms_err
    p_int = scipy.stats.f.sf(f_int, dfn=1, dfd=len(tot)-4)
    print("Interaction:          f = %f6.2  p = %f7.5" % (f_int, p_int))
