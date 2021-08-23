__author__ = 'martinez'

import numpy as np
from unittest import TestCase
from pyorg.spatial.sparse import UniStat, PlotUni, gen_sin_points, gen_rand_in_mask
from pyorg import disperse_io

# ### Test variables

# Global
RES = 0.684 # nm/pixel

# Mask
MASK_R = 2000 # nm
SIZE = (200, 100, 5) # pixel

# Pattern 1
N_PST_1 = 338
CYCLES_1 = (4, 4, 4)
STD_1 = 1.

# Pattern 2
N_PST_2 = 1000
CYCLES_2 = (10, 10, 10)
STD_2 = 1.

# Analysis
MAX_D_1 = 10 # nm (G, F and J)
MAX_D_2 = 50 # nm (G, F and J)
N_BINS = 50
PER = 5 #%
N_SIM_1 = 100 # Number of simulations for G and F
N_SAMP_F = 1000 # Number of samples for F
N_SIM_K = 20 # Number of simulations for K
N_SAMP_K = 50 # Number of samples for K
W_O = 1
TCSR = True
N_SAMP_W = 20 # Number of samples for K
rads = [10, 20, 40] # nm

class TestUniStat(TestCase):

    def test_Execute(self):

        print('\tRunning test for sparse point spatial analysis:')

        print('\t\tBuilding the mask...')
        size_nm = np.asarray(SIZE) * RES
        X, Y, Z = np.meshgrid(np.linspace(-.5*size_nm[1], .5*size_nm[1], SIZE[1]),
                              np.linspace(-.5*size_nm[0], .5*size_nm[0], SIZE[0]),
                              np.linspace(-.5*size_nm[2], .5*size_nm[2], SIZE[2]))
        mask = (X*X + Y*Y + Z*Z) < (MASK_R*MASK_R)
        disperse_io.save_numpy(mask, './results/mask.mrc')

        print('\t\tGenerating the random patterns:')
        pat_1 = gen_sin_points(N_PST_1, CYCLES_1, mask, STD_1, phase=(.5*np.pi,.5*np.pi,.5*np.pi))
        pat_2 = gen_sin_points(N_PST_2, CYCLES_2, mask, STD_2, phase=(.5*np.pi,.5*np.pi,.5*np.pi))
        pat_3 = gen_rand_in_mask(N_PST_1, mask)
        pat_4 = gen_rand_in_mask(N_PST_2, mask)
        print('\t\t\t-Number of points for pattern 1: ' + str(pat_1.shape[0]))
        print('\t\t\t-Number of points for pattern 2: ' + str(pat_2.shape[0]))
        print('\t\t\t-Number of points for pattern 3: ' + str(pat_3.shape[0]))
        print('\t\t\t-Number of points for pattern 4: ' + str(pat_4.shape[0]))

        print('\t\tCreating the objects for analysis...')
        uni1, uni2 = UniStat(pat_1, mask, RES), UniStat(pat_2, mask, RES)
        uni3, uni4 = UniStat(pat_3, mask, RES), UniStat(pat_4, mask, RES)
        n_bins = N_BINS
        if n_bins > uni1.get_n_points():
            print('\t\t\t-WARNING: Number of bins decreased to: ' + str(pat_1.shape[0]))
            n_bins = uni1.get_n_points()
        if n_bins > uni2.get_n_points():
            print('\t\t\t-WARNING: Number of bins decreased to: ' + str(pat_2.shape[0]))
            n_bins = uni2.get_n_points()
        uni1.set_name('Pattern 1')
        uni2.set_name('Pattern 2')
        uni3.set_name('CSR 1')
        uni4.set_name('CSR 2')
        analyzer = PlotUni()
        analyzer.insert_uni(uni1)
        analyzer.insert_uni(uni2)
        analyzer.insert_uni(uni3)
        analyzer.insert_uni(uni4)
        uni1.save_sparse('./results/pat1.mrc')
        uni2.save_sparse('./results/pat2.mrc')
        uni3.save_sparse('./results/pat3.mrc')
        uni4.save_sparse('./results/pat4.mrc')
        uni1.plot_points(block=False, out_file='./results/pat1_pts.png')
        uni2.plot_points(block=False, out_file='./results/pat2_pts.png')
        uni3.plot_points(block=False, out_file='./results/pat3_pts.png')
        uni4.plot_points(block=False, out_file='./results/pat4_pts.png')

        print('\t\tAnalysis:')
        print('\t\t\t-G')
        analyzer.analyze_G(MAX_D_1, n_bins, N_SIM_1, PER, block=False, out_file='./results/G.png', legend=True)
        print('\t\t\t-F')
        analyzer.analyze_F(MAX_D_1, n_bins, N_SAMP_F, N_SIM_1, PER, block=False, out_file='./results/F.png')
        print('\t\t\t-J')
        analyzer.analyze_J(block=False, out_file='./results/J.png', p=PER)
        print('\t\t\t-K')
        analyzer.analyze_K(MAX_D_2, N_SAMP_K, N_SIM_K, PER, tcsr=TCSR, block=False, out_file='./results/K.png')
        print('\t\t\t-L')
        analyzer.analyze_L(block=False, out_file='./results/L.png', p=PER)
        print('\t\t\t-O')
        analyzer.analyze_O(W_O, block=False, out_file='./results/O.png', p=PER)
        for uni in (uni1, uni2, uni3, uni4):
            if uni.is_2D():
                print('\t\t\t-W ' + uni.get_name() + ':')
                uni.analyze_W(rads, N_SAMP_W, block=False, out_file='./results/W_'+uni.get_name()+'.png',
                              legend=True, pimgs=True)
