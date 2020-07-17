from unittest import TestCase
from pyseg.diff_geom import SpaceCurve
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Test parameters
N_SAMPLES = 100
# L_BOUND = 0.5 * np.pi
# U_BOUND = 1.5 * np.pi
L_BOUND = (-0.15) * np.pi
U_BOUND = 0.5 * np.pi
C_MODE = 2
EPS = 1e-1
COMP = False

class TestSpaceCurve(TestCase):

    def test_Execute(self):

        # Curve initialization (cos(t), sin(t), sqrt(t))
        t = np.linspace(L_BOUND, U_BOUND, N_SAMPLES)
        samp_3d = np.zeros(shape=(len(t), 3), dtype=np.float)
        # samp_3d[:, 0], samp_3d[:, 1], samp_3d[:, 2] = np.cos(t), np.sin(t), np.sqrt(t)
        # samp_3d[:, 0], samp_3d[:, 1], samp_3d[:, 2] = 3*np.cos(3*t)+2, 3*np.sin(t)+2, np.sqrt(t+1.)
        samp_3d[:, 0], samp_3d[:, 1], samp_3d[:, 2] = np.cos(t), np.sin(t), np.ones(shape=len(t))

        # Numerical estimator
        curve = SpaceCurve(samp_3d, mode=C_MODE)

        # Plot the space curve
        fig1 = plt.figure(1)
        plt.title('Space curve (cos(t), sin(t), sqrt(t))')
        ax = fig1.gca(projection='3d')
        ax.plot(samp_3d[:, 0], samp_3d[:, 1], samp_3d[:, 2])

        if COMP:

            # Parametric estimation
            t_2 = t * t
            t_3 = t_2 * t
            t_4 = t_3 * t
            t_5 = t_4 * t
            hold_1 = 16*t_3 + 4*t_2 + 1
            hold_2 = 1 + 4*t
            hold_3 = np.sqrt(t)
            k = (2.*np.sqrt(hold_1)) / (hold_2**1.5)
            ks = (8*(8*t_2+2*t-3)*hold_3) / (np.sqrt(hold_1)*hold_2*hold_2*hold_2)
            # I don't why sign must be te opposite to
            # Boutin M. "Numerically Invariant Signature Curves" Int. J. Comput. Vision, 40(3): 235-248, 2000
            tau = (2.*hold_3*(3+4*t_2)) / hold_1
            taus = ((-2)*(64*t_5-16*t_4+240*t_3+16*t_2-3)) / (np.sqrt(hold_2)*hold_1*hold_1)

            # Plot estimations vs parametric
            plt.figure(2)
            plt.title('Estimations vs Parametric: uk')
            plt.ylabel('uk')
            plt.xlabel('length')
            plt.plot(curve.get_lengths(), curve.get_uk())
            plt.plot(curve.get_lengths(), k, '--k')
            plt.figure(3)
            plt.title('Estimations vs Parametric: k')
            plt.ylabel('k')
            plt.xlabel('length')
            plt.plot(curve.get_lengths(), curve.get_k())
            plt.plot(curve.get_lengths(), ks, '--k')
            plt.figure(4)
            plt.title('Estimations vs Parametric: ut')
            plt.ylabel('ut')
            plt.xlabel('length')
            plt.plot(curve.get_lengths(), curve.get_ut())
            plt.plot(curve.get_lengths(), tau, '--k')
            plt.figure(5)
            plt.title('Estimations vs Parametric: t')
            plt.ylabel('t')
            plt.xlabel('length')
            plt.plot(curve.get_lengths(), curve.get_t())
            plt.plot(curve.get_lengths(), taus, '--k')

            plt.show(block=True)

            # Validation
            e_t_uk = curve.get_total_uk()
            p_t_uk = (k*curve.get_ds()).sum()
            diff_uk = math.fabs(e_t_uk - p_t_uk)
            print '\tEstimated total unsigned curvature: ' + str(e_t_uk) + ' rad'
            print '\tParametric total unsigned curvature: ' + str(p_t_uk) + ' rad'
            self.assertLess(diff_uk, EPS, 'Total unsigned curvature error bigger than ' + str(EPS))
            e_t_k = curve.get_total_k()
            p_t_k = (ks*curve.get_ds()).sum()
            diff_k = math.fabs(e_t_k - p_t_k)
            print '\tEstimated total signed curvature: ' + str(e_t_k) + ' rad'
            print '\tParametric total signed curvature: ' + str(p_t_k) + ' rad'
            self.assertLess(diff_k, EPS, 'Total signed curvature error bigger than ' + str(EPS))
            e_t_ut = curve.get_total_ut()
            p_t_ut = (tau*curve.get_ds()).sum()
            diff_ut = math.fabs(e_t_ut - p_t_ut)
            print '\tEstimated total unsigned torsion: ' + str(e_t_ut) + ' rad'
            print '\tParametric total unsigned torsion: ' + str(p_t_ut) + ' rad'
            self.assertLess(diff_ut, EPS, 'Total unsigned torsion error bigger than ' + str(EPS))
            e_t_t = curve.get_total_t()
            p_t_t = (taus*curve.get_ds()).sum()
            diff_t = math.fabs(e_t_t - p_t_t)
            print '\tEstimated total signed torsion: ' + str(e_t_t) + ' rad'
            print '\tParametric total signed torsion: ' + str(p_t_t) + ' rad'
            self.assertLess(diff_t, EPS, 'Total signed torsion error bigger than ' + str(EPS))

        else:

            # Plot estimations vs parametric
            plt.figure(2)
            plt.title('Estimation: uk')
            plt.ylabel('uk')
            plt.xlabel('length')
            plt.plot(curve.get_lengths(), curve.get_uk())
            plt.figure(3)
            plt.title('Estimation: ks')
            plt.ylabel('ks')
            plt.xlabel('length')
            plt.plot(curve.get_lengths(), curve.get_k())
            plt.figure(4)
            plt.title('Estimation: ut')
            plt.ylabel('ut')
            plt.xlabel('length')
            plt.plot(curve.get_lengths(), curve.get_ut())
            plt.figure(5)
            plt.title('Estimation: ts')
            plt.ylabel('ts')
            plt.xlabel('length')
            plt.plot(curve.get_lengths(), curve.get_t())\

            plt.show(block=True)

            print '\tEstimated total unsigned curvature: ' + str(curve.get_total_uk()) + ' rad'
            print '\tEstimated total ks: ' + str(curve.get_total_k()) + ' rad'
            print '\tEstimated total unsigned torsion: ' + str(curve.get_total_ut()) + ' rad'
            print '\tEstimated total ts: ' + str(curve.get_total_t()) + ' rad'
            print '\tEstimated normal symmetry: ' + str(curve.get_normal_symmetry())
            print '\tEstimated binormal symmetry: ' + str(curve.get_binormal_symmetry())
            print '\tEstimated apex length: ' + str(curve.get_apex_length())
            print '\tEstimated sinuosity: ' + str(curve.get_sinuosity())


