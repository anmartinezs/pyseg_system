"""

    Script for doing Univariate statistical spatial analysis for a 3D distributed point pattern

    Input:  - XML file with a particle list
            - Parameters for setting the statistical analysis
            - Tomogram mask (optional)

    Output: - Graphs with the results of the statistical analysis

"""

__author__ = 'Antonio Martinez-Sanchez'


###### Global variables

import gc
import pyseg as ps

########################################################################################
# PARAMETERS
########################################################################################

# ROOT_PATH = '/home/martinez/pool/pool-engel/pyrenoid_stat/bin2'
ROOT_PATH = '/home/martinez/workspace/stat/pyrenoid'

# Particles lists
in_plist_xml_l = (# ROOT_PATH + '/in/L3Tomo3_tp_bin4.xml',
                  # ROOT_PATH + '/in/L2Tomo8_pl_p_bin4.xml',
                  # ROOT_PATH + '/in/L6Tomo6_pl_p_bin4.xml',
                  ROOT_PATH + '/in/Tomo1_Lam6_pl_p_bin4.xml',
                  # ROOT_PATH + '/in/Tomo6L2_pl_p_bin4.xml'
                  )
# in_plist_xml_l = (# ROOT_PATH + '/pl/pl_full_part.xml',
#                   # ROOT_PATH + '/pl/pl_full.xml',
#                   )

# Mask tomogram
in_ref_l = (# ROOT_PATH+'/in/L3Tomo3_fixedtilt_bin4noCTF_100_360_MASKall_inv.mrc',
            # ROOT_PATH+'/in/L2Tomo8_bin4_50_430_MASKall_inv.mrc',
            # ROOT_PATH+'/in/L6Tomo6_bin4_100_370_MASKall_inv.mrc',
            ROOT_PATH+'/in/Tomo1_Lam6_bin4_40_400_MASKall_inv_crop.mrc',
            # ROOT_PATH+'/in/Tomo6L2_bin4_40_410_MASKall_inv.mrc',
            )
# in_ref_l = ((400, 400, 200),
            # ROOT_PATH + '/in/L3Tomo3_final_mask.mrc',
            # ROOT_PATH + '/in/L3Tomo3_final_mask_ccrop.mrc',
#             )

####### Output data

out_stem = 'agg_test'
out_dir = ROOT_PATH + '/agg_test/'
out_fmt = '.png'


###### Transformation data (from particle list to holding tomogram)

# Tomogram resolution
in_res = 1.368 # 0.684 # nm/piexel

####### Particle

in_temp = ROOT_PATH + '/in/tmp_def_5_size_24_bin_g1_sf1_er1.mrc'
# in_temp = ROOT_PATH + '/in/tmp_def_5_size_24_bin_g1_sf1.mrc'
in_con = 'tom'

###### Analysis

gather = True
gstd = True
do_ana_1 = False
do_ana_2 = False
do_ana_3 = True
do_ana_4 = True
an_mx_d1 = 35 # nm (first order)
an_mx_d2 = 50 # nm (second order)
an_nbins = 50
an_per = 1 #%
an_nsim_1 = 1 # Number of simulations for first order
an_samp_f = 2000 # Number of samples for F
an_nsim_2 = 1 # Number of simulations for second order
an_samp_k = 50 # Number of samples for K
an_wo = 1
an_tcsr = True
an_shell_thick = 1.4 # 0.7 # nm
db = False

###### Simulator

sim_nrot = 1000
do_linker = False
lin_l = 2 # nm
lin_sg = 1 # nm
lin_nc = 2
do_rubc = False
do_ccp = False
do_hcp = False
cp_rad = 6.75 # nm
do_agg = True
agg_rs = 0.05
agg_t = 20
agg_binds = ((15,15,15),
             (15,15,10),
             (15,10,15),
             (15,10,10),
             (10,15,15),
             (10,15,10),
             (10,10,15),
             (10,10,10),
             )

########################################################################################
# MAIN ROUTINE
########################################################################################

################# Package import

import os
import sys
import math
import time
import numpy as np
from pyseg.sub import ParticleList
from pyseg.spatial import UniStat, PlotUni, CSRSimulator, LinkerSimulator, RubCSimulation, CSRTSimulator, \
    CPackingSimulator, AggSimulator2
from pyseg.disperse_io import disperse_io
from skimage.transform import resize

########## Global variables

########## Print initial message

print 'Univariate Spatial 3D distribution.'
print '\tAuthor: ' + __author__
print '\tDate: ' + time.strftime("%c") + '\n'
print 'Options:'
print '\tInput particles lists: ' + str(in_plist_xml_l)
print '\tMask tomogram names: ' + str(in_ref_l)
print '\tOutput directory: ' + out_dir
print '\tTransformation properties (from reference to holding tomogram):'
print '\t\t-Tomogram resolution: ' + str(in_res) + ' nm/pixel'
if in_temp is not None:
    print '\tParticle properties: '
    print '\t\t-Template: ' + in_temp
    print '\t\t-Name: ' + out_stem
    print '\t\t-Angles convention: ' + in_con
    print '\t\t-Number or random rotations: ' + str(sim_nrot)
    if do_linker:
        print '\tLinker simulator:'
        print '\t\t-Linker length: ' + str(lin_l) + ' nm'
        print '\t\t-Linker length sigma: ' + str(lin_sg) + 'nm'
        print '\t\t-Number of copies: ' + str(lin_nc)
    elif do_rubc:
        print '\tRubisco Crystal simulator.'
    elif do_ccp:
        print '\tCube close packing simulator.'
    elif do_hcp:
        print '\tHexagonal close packing simulator.'
    elif do_agg:
        print '\tAggregates simulator:'
        print '\t\t-Linker length: ' + str(lin_l) + ' nm'
        print '\t\t-Linker length sigma: ' + str(lin_sg) + 'nm'
        print '\t\t-Restart probability: ' + str(agg_rs)
        print '\t\t-Maximum number of tries to link a particle: ' + str(agg_t)
        print '\t\t-Bind sites: ' + str(agg_binds)
    else:
        print '\tTemplate based CSR simulator.'
else:
    print '\tCSR simulator:'
if do_ana_1 or do_ana_2:
    if gather:
        print '\tAnalysis (composed):'
    else:
        print '\tAnalysis (isolated):'
    print '\t\t-Percentile for random simulations: ' + str(an_per) + ' %'
if do_ana_1:
    print '\t\t-First order metrics (G, F and J):'
    print '\t\t\t-Maximum scale: ' + str(an_mx_d1) + ' nm'
    print '\t\t\t-Number of bins: ' + str(an_nbins)
    print '\t\t\t-Number of simulations: ' + str(an_nsim_1)
    print '\t\t\t-Number of samples for F: ' + str(an_samp_f)
if do_ana_2:
    print '\t\t-Second order metrics (K and O):'
    print '\t\t\t-Maximum scale: ' + str(an_mx_d2)
    print '\t\t\t-Number of simulations: ' + str(an_nsim_2)
    print '\t\t\t-Number of samples: ' + str(an_samp_k)
    print '\t\t\t-Ring shell for O: ' + str(an_wo) + ' voxels'
    if an_tcsr:
        print '\t\t\t-Plotting bounds for random simulation.'
if do_ana_3:
    if db:
        print '\t\t-Normalized density analysis (dB):'
    else:
        print '\t\t-Normalized density analysis (linear):'
    print '\t\t\t-Maximum scale: ' + str(an_mx_d2)
    print '\t\t\t-Number of simulations: ' + str(an_nsim_2)
    print '\t\t\t-Number of samples: ' + str(an_samp_k)
    print '\t\t\t-Ring shell thickness: ' + str(an_shell_thick) + ' nm'
if do_ana_4:
    if db:
        print '\t\t-Normalized density analysis (dB):'
    else:
        print '\t\t-Normalized density analysis (linear):'
    print '\t\t\t-Maximum scale: ' + str(an_mx_d2)
    print '\t\t\t-Number of samples: ' + str(an_samp_k)
    print '\t\t\t-Ring shell thickness: ' + str(0.2*an_shell_thick) + ' nm'
print ''

######### Process

print 'Main Routine: '

analyzer = PlotUni()

print '\tParticles list loop: '
for in_plist_xml, in_ref in zip(in_plist_xml_l, in_ref_l):

    print '\tSetting up simulator...'
    res = float(in_res)
    temp, rots = None, None
    if in_temp is None:
        simulator = CSRSimulator()
    else:
        print '\t\t\t-Loading the template...'
        temp = disperse_io.load_tomo(in_temp)
        if do_linker:
            leng = lin_l / res
            simulator = LinkerSimulator(leng, lin_nc, temp, nrots=sim_nrot, no_over=True, len_sg=lin_sg)
        elif do_rubc:
            simulator = RubCSimulation(res, temp)
        elif do_ccp:
            simulator = CPackingSimulator(res, cp_rad, temp, packing=1)
        elif do_hcp:
            simulator = CPackingSimulator(res, cp_rad, temp, packing=2)
        elif do_agg:
            simulator = AggSimulator2(res, temp, agg_binds, lin_l, lin_sg, agg_rs, agg_t, sh_sg=5, nrots=sim_nrot)
        else:
            simulator = CSRTSimulator(temp, nrots=sim_nrot)

    if isinstance(in_ref, str):
        print '\t\tLoading the mask...'
        mask = disperse_io.load_tomo(in_ref)
    else:
        print '\t\tGenerating the mask...'
        mask = np.ones(shape=in_ref, dtype=np.bool)


    if in_plist_xml is None:
        cloud = simulator.gen_rand_in_mask(100, mask)
        rots = np.zeros(shape=cloud.shape, dtype=np.float32)
    else:
        print '\t\tParsing particle list: ' + in_plist_xml
        pl_path, pl_fname = os.path.split(in_plist_xml)
        plist = ParticleList(pl_path + '/sub')
        plist.load(in_plist_xml)
        if in_temp is not None:
            cloud, rots = plist.get_particles_coords(do_shift=True, rot=True)
        else:
            cloud = plist.get_particles_coords(do_shift=True, rot=False)
        print '\t\t\t-Number of particles found: ' + str(plist.get_num_particles())

    print '\t\tCreating the object for the univariate analysis...'
    uni = UniStat(np.asarray(cloud), mask, res, rots=np.asarray(rots), temp=temp, conv=in_con)
    uni.set_name(out_stem)
    uni.set_simulator(simulator)
    analyzer.insert_uni(uni)
    print '\t\t-Number of points in the pattern: ' + str(uni.get_n_points())
    if uni.is_2D():
        print '\t\t-Pattern total intensity: ' + str(uni.get_intensity()) + ' points/nm^2'
    else:
        print '\t\t-Pattern total intensity: ' + str(uni.get_intensity()) + ' points/nm^3'
    if (not do_linker) and (do_rubc or do_ccp or do_hcp):
        hold_points = simulator.gen_rand_in_mask(len(cloud), uni.get_mask())
        print '\t\t-Crystal lattice density: ' + str(float(hold_points.shape[0])/uni.get_volume()) \
              + ' particles/nm^3'

    print '\tStoring the patterns to analyze...'
    if in_plist_xml is not None:
        pl_stem = os.path.splitext(pl_fname)[0]
    else:
        pl_stem = 'whole'
    # uni.save_sparse(out_dir+'/'+out_stem+'_'+pl_stem+'_pts.mrc', mask=True)
    # uni.save_sparse(out_dir+'/'+out_stem+'_'+pl_stem+'_dense.mrc', mask=True)
    # ps.disperse_io.save_numpy(uni.get_mask(), out_dir+'/'+out_stem+'_'+pl_stem+'_mask.mrc')
    uni.save_random_instance(out_dir+'/'+out_stem+'_'+pl_stem+'_rnd.mrc')
    # hold_iso = simulator.gen_rand_in_mask_tomo(1, np.ones(shape=(400,400,400), dtype=np.bool))
    # ps.disperse_io.save_numpy(hold_iso, out_dir+'/'+out_stem+'_'+pl_stem+'_rnd_iso.mrc')
    uni.save_random_instance(out_dir+'/'+out_stem+'_'+pl_stem+'_rnd_pts.mrc', pts=True)

    gc.collect()

if do_ana_1:
    print '\t\tFirst order Analysis:'
    print '\t\t\t-G'
    analyzer.analyze_G(an_mx_d1, an_nbins, an_nsim_1, an_per, block=False,
                       out_file=out_dir+'/'+out_stem+'_G'+out_fmt, legend=False, gather=gather)
    print '\t\t\t-F'
    analyzer.analyze_F(an_mx_d1, an_nbins, an_samp_f, an_nsim_1, an_per, block=False,
                       out_file=out_dir+'/'+out_stem+'_F'+out_fmt, gather=gather)
    print '\t\t\t-J'
    analyzer.analyze_J(block=False, out_file=out_dir+'/'+out_stem+'_J'+out_fmt, p=an_per)

if do_ana_2:
    print '\t\tSecond order Analysis:'
    print '\t\t\t-K'
    analyzer.analyze_K(an_mx_d2, an_samp_k, an_nsim_2, an_per, tcsr=an_tcsr,
                       block=False, out_file=out_dir+'/'+out_stem+'_K'+out_fmt, gather=gather)
    print '\t\t\t-L'
    analyzer.analyze_L(block=False, out_file=out_dir+'/'+out_stem+'_L'+out_fmt, gather=gather, p=an_per, gstd=gstd)
    print '\t\t\t-O'
    analyzer.analyze_O(an_wo, block=False, out_file=out_dir+'/'+out_stem+'_O'+out_fmt, gather=gather, p=an_per, gstd=gstd)
    print '\t\t\t-I'
    analyzer.analyze_I(block=False, out_file=out_dir+'/'+out_stem+'_I'+out_fmt, gather=gather, p=an_per, gstd=gstd)
    print '\t\t\t-D'
    analyzer.analyze_D(an_wo, block=False, out_file=out_dir+'/'+out_stem+'_D'+out_fmt, gather=gather, p=an_per, gstd=gstd)

if do_ana_3:
    print '\t\tDensity analysis:'
    print '\t\t\t-Normalized Local Density:'
    analyzer.analyze_NLD(an_shell_thick, an_mx_d2, an_samp_k, an_nsim_2, an_per,
                       block=False, out_file=out_dir+'/'+out_stem+'_NLD'+out_fmt, gather=gather, gstd=gstd, db=db)

if do_ana_4:
    print '\t\tDensity analysis without border compensation:'
    print '\t\t\t-Normalized Local Density:'
    analyzer.analyze_RBF(0.2*an_shell_thick, an_mx_d2, 5*an_samp_k, p=an_per,
                       block=False, out_file=out_dir+'/'+out_stem+'_RBF'+out_fmt, gather=gather, gstd=gstd, db=db)

print '\tStoring the analyzer...'
analyzer.pickle(out_dir+'/'+out_stem+'_ana.pkl')

print 'Terminated. (' + time.strftime("%c") + ')'