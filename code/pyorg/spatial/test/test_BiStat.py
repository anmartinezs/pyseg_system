__author__ = 'martinez'

from pyorg import disperse_io
import numpy as np
from unittest import TestCase
from pyorg.spatial.sparse import BiStat, PlotBi, gen_sin_points, gen_rand_in_mask

# ### Test variables

# Global
RES = 0.684 # nm/pixel

# Mask
MASK_R = 2000 # nm
SIZE = (200, 200, 1) # pixel

# Pattern 1 (Reference)
N_PST = 500
CYCLES = (4, 4, 1)
STD = 1.

# Pattern 2
SHIFT = .5 * np.pi # rad

# Analysis
MAX_D_1 = 50 # nm (G, F and J)
MAX_D_2 = 50 # nm (K, L and O)
N_BINS = 50
PER = 5 #%
N_SIM_1 = 1000 # Number of simulations for G
N_SIM_K = 20 # Number of simulations for K
N_SAMP_K = 50 # Number of samples for K
W_O = 1


class TestBiStat(TestCase):

  def test_Execute(self):

        print('\tRunning test for sparse point spatial analysis:')

        print('\t\tBuilding the mask...')
        size_nm = np.asarray(SIZE) * RES
        X, Y, Z = np.meshgrid(np.linspace(-.5*size_nm[1], .5*size_nm[1], SIZE[1]),
                              np.linspace(-.5*size_nm[0], .5*size_nm[0], SIZE[0]),
                              np.linspace(-.5*size_nm[2], .5*size_nm[2], SIZE[2]))
        mask = (X*X + Y*Y + Z*Z) < (MASK_R*MASK_R)
        disperse_io.save_numpy(mask, './results/mask.mrc')
        mod_2D = False
        if SIZE[2] == 1:
            mod_2D = True

        print('\t\tGenerating the random patterns:')
        pat_1 = gen_sin_points(N_PST, CYCLES, mask, STD)
        pat_2 = gen_sin_points(N_PST, CYCLES, mask, STD, phase=(SHIFT, SHIFT, SHIFT))
        pat_3 = gen_rand_in_mask(N_PST, mask)
        print('\tPatterns shifting: ' + str(SHIFT) + ' rad')

        print('\t\tCreating the objects for analysis...')
        bi1, bi2 = BiStat(pat_1, pat_2, mask, RES), BiStat(pat_1, pat_3, mask, RES)
        n_bins = N_BINS
        if n_bins > bi1.get_ns_points():
            print('\t\t\t-WARNING: Number of bins decreased to: ' + str(pat_1.shape[0]))
            n_bins = bi1.get_ns_points()
        if n_bins > bi2.get_ns_points():
            print('\t\t\t-WARNING: Number of bins decreased to: ' + str(pat_2.shape[0]))
            n_bins = bi2.get_n_points()
        bi1.set_name('1 vs. sh')
        bi2.set_name('1 vs. rand')
        analyzer = PlotBi()
        analyzer.insert_bi(bi1)
        analyzer.insert_bi(bi2)

        print('\t\tAnalysis:')
        print('\t\t\t-G')
        analyzer.analyze_G(MAX_D_1, n_bins, N_SIM_1, PER, out_file='./results/bi_G.png', block=False)
        print('\t\t\t-K')
        analyzer.analyze_K(MAX_D_2, N_SAMP_K, N_SIM_K, PER, out_file='./results/bi_K.png', block=False)
        print('\t\t\t-L')
        analyzer.analyze_L(out_file='./results/bi_L.png', block=False)
        print('\t\t\t-O')
        analyzer.analyze_O(W_O, out_file='./results/bi_O.png', block=False)
