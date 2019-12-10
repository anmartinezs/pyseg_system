"""
Contains class Contingency for compaing a clustering (classification with a
reference clustering.


# Author: Vladan Lucic
# $Id: cluster.py 1520 2019-03-11 17:00:20Z vladan $
"""

__version__ = "$Revision: 1520 $"


import collections

import numpy as np
import scipy as sp
import pandas as pd


class Contingency(object):
    """
    Holds and manipulates a contingency (confusion) matrix
    """

    def __init__(self, data=None):
        """
        Sets contingency table

        Argument:
          - data: contingency table specified as pandas.DataFrame or a 
          2D ndarray, where rows / axis 0 denote classes and columns / 
          axis 1 references

        Sets attribute:
          - data (Pandas.DataFrame)
        """

        if data is not None:

            # ndarray
            if isinstance(data, np.ndarray):
                shape = data.shape
                columns = ['ref_' + str(ref_ind) for ref_ind in range(shape[1])]
                self.data = pd.DataFrame(data=data, columns=columns)

            # DataFrame
            elif isinstance(data, pd.DataFrame):
                self.data = data

            else:
                raise ValueError(
                    "Argument data has to be of type numpy.ndarray " +
                    "or pandas.DataFrame and not " + type(data))

    def do_g_test(self, contingency=None):
        """
        G test on contingency table

        Argument:
          - contingency: contingency table, if None self.data is used

        Sets namedtuple object with following attributes:
          - 
        """

        # chose contingency
        if contingency is None:
            contingency = self.data

        # calculate g-test
        g_val, p_val, dof, expected = sp.stats.chi2_contingency(
            contingency, lambda_="log-likelihood", correction=True)
        namtu = collections.namedtuple('Struct', 'g p dof expected')
        res = namtu._make((g_val, p_val, dof, expected))

        self.g_test = res

    def match(self):
        """
        Associates each cluster with the reference that most of its 
        elements belongs to.

        Sets attribute:
          - self.matched
          - self.class_ref
        """

        # initialize
        matched_shape = (self.data.shape[1], self.data.shape[1])
        self.matched = pd.DataFrame(
            data=np.zeros(matched_shape, dtype=int), columns=self.data.columns)

        # define matched
        arg_max = np.array(self.data).argmax(axis=1)
        self.class_ref = dict(zip(range(len(arg_max)), arg_max))
        for clust_ind, ref_ind in enumerate(arg_max):
            self.matched.loc[ref_ind, :] += self.data.loc[clust_ind, :]

    def evaluate(self, match=True):
        """
        """

        # match clusters to references if needed
        try:
            self.matched
        except AttributeError:
            if match:
                self.match()
            else:
                self.matched = self.data
        shape_matched = self.matched.shape

        # sanity check
        if shape_matched[0] != shape_matched[1]:
            raise ValueError("Matched contingency matrix has to be square.")

        # marginals
        total_class = self.data.sum(axis=1)
        total_class_matched = self.matched.sum(axis=1)
        total_ref = self.matched.sum(axis=0)
            
        # true positives
        tp = [self.data.iloc[cl_ind, self.class_ref[cl_ind]]
              for cl_ind in range(self.data.shape[0])]
        tp_matched = np.diag(self.matched)

        # false positives
        fp_matched = total_class_matched - tp_matched
    
        # false negatives
        fn_matched = total_ref - tp_matched

        # precision and recall
        precision_matched_all = tp_matched / (tp_matched + fp_matched)
        precision_matched = (
            precision_matched_all.sum() / len(precision_matched_all))
        recall_matched_all = tp_matched / (tp_matched + fn_matched)
        recall_matched = recall_matched_all.sum() / len(recall_matched_all)
        recall_matched_adj = recall_matched_all.sum() / len(total_class)

        # F1
        f1_matched = (
            2 * precision_matched * recall_matched
            / (precision_matched + recall_matched))
        f1_matched_adj = (
            2 * precision_matched * recall_matched_adj
            / (precision_matched + recall_matched_adj))

        # set attributes
        self.class_margin = total_class
        self.class_margin_matched = total_class_matched
        self.ref_margin = total_ref
        self.tp = tp
        self.tp_matched = tp_matched
        self.fp_matched = fp_matched
        self.fn_matched = fn_matched
        self.precision_matched = precision_matched
        self.recall_matched = recall_matched
        self.recall_matched_adj = recall_matched_adj
        self.f1_matched = f1_matched
        self.f1_matched_adj = f1_matched_adj
