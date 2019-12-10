"""
Scipy related utility functions.

# Author: Vladan Lucic, Max Planck Institute for Biochemistry
# $Id: scipy_plus.py 1564 2019-06-04 09:37:47Z vladan $
"""

__version__ = "$Revision: 1564 $"


import numpy
import scipy
import scipy.stats as stats


def chisquare_2(f_obs_1, f_obs_2, yates=False):
    """
    Calculates chi-square between two arrays of observation frequencies and
    returns the result. 

    Differs from scipy.stats.chisquare() in that this function calculated 
    significance between two distributions and that the two distributions 
    can have different number of data points.
    
    Arguments:
      - f_obs_1, f_obs_2: frequencies of observations 1 and 2

    Returns: chi-square, associated p-value 
    """
    
    # prepare variables
    f_obs_1 = numpy.asarray(f_obs_1)
    f_obs_2 = numpy.asarray(f_obs_2)

    # chisquare
    if not yates:

        # no yates
        sum_1 = float(f_obs_1.sum())
        sum_2 = float(f_obs_2.sum())

        # calculate chi-square value
        chisq = [(el_1 * sum_2 - el_2 * sum_1) ** 2 / (el_1 + el_2) 
                 for el_1, el_2 in zip(f_obs_1, f_obs_2)]
        chisq = numpy.array(chisq, dtype='float').sum() / (sum_1 * sum_2)

    else:

        # with yates (easier to see what happens)
        data = numpy.vstack([f_obs_1, f_obs_2])

        # expectations
        sum_freq = data.sum(axis=1)
        sum_obs = data.sum(axis=0)
        expect = numpy.outer(sum_freq, sum_obs) / float(data.sum())

        # chisquare
        chisq = ((numpy.abs(data - expect) - 0.5)**2 / expect).sum()

    # probability (same as stats.chi2.sf())
    #p = stats.chisqprob(chisq, len(f_obs_1)-1)  depreciated
    p = stats.distributions.chi2.sf(chisq, len(f_obs_1)-1)

    return chisq, p

def ttest_ind_nodata(mean_1, std_1, n_1, mean_2, std_2, n_2):
    """
    Student's t-test between two independent samples. Unlike in ttest_ind(), 
    the samples (data) are not given, instead the basic statstical quantities 
    (mean, standard deviation and number of measurements) are used to do the 
    test. Returns t-value and a two-tailed confidence level.

    Arguments:
      - mean_1, mean_2: means of samples 1 and 2
      - std_1, std_2: standard deviations of samples 1 and 2 calculated using
      n-1 degrees of freedom where n is the number of measurements in a sample
      - n_1, n_2: number of measurements 
      Arguments can be ndarrays instead of single numbers. 

    Returns: (t_value, two_tail_confidence)
    """

    # sums of squares
    sum_squares_1 = std_1 ** 2 * (n_1 - 1)
    sum_squares_2 = std_2 ** 2 * (n_2 - 1)

    # std of the defference between means
    pooled_var = (sum_squares_1 + sum_squares_2) / (n_1 + n_2 - 2.)
    std_means = numpy.sqrt(pooled_var * (1. / n_1 + 1. / n_2))

    # t-value and confidence
    t = (mean_1 - mean_2) / std_means
    confidence = 2 * stats.t.sf(numpy.abs(t), n_1 + n_2 - 2)

    return t, confidence

