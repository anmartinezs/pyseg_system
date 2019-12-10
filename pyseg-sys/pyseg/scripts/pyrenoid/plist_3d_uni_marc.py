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

ROOT_PATH = '/media/martinez/DATAPART1/stat/fas/gather'
# ROOT_PATH = '/home/martinez/workspace/stat/pyrenoid'

# Particles lists
in_plist_xml_l =   (ROOT_PATH+'/in/tomo6_FAS.xml',
                    ROOT_PATH+'/in/tomo56_FAS.xml',
                    ROOT_PATH+'/in/tomo55_FAS.xml',
                    ROOT_PATH+'/in/tomo54_FAS.xml',
                    ROOT_PATH+'/in/tomo53_FAS.xml',
                    ROOT_PATH+'/in/tomo4_FAS.xml',
                    ROOT_PATH+'/in/tomo48_FAS.xml',
                    ROOT_PATH+'/in/tomo40_FAS.xml',
                    ROOT_PATH+'/in/tomo3_FAS.xml',
                    ROOT_PATH+'/in/tomo38_FAS.xml',
                    ROOT_PATH+'/in/tomo37_FAS.xml',
                    ROOT_PATH+'/in/tomo36_FAS.xml',
                    ROOT_PATH+'/in/tomo35_FAS.xml',
                    ROOT_PATH+'/in/tomo34_FAS.xml',
                    ROOT_PATH+'/in/tomo33_FAS.xml',
                    ROOT_PATH+'/in/tomo32_FAS.xml',
                    ROOT_PATH+'/in/tomo31_FAS.xml',
                    ROOT_PATH+'/in/tomo2_FAS.xml',
                    ROOT_PATH+'/in/tomo29_FAS.xml',
                    ROOT_PATH+'/in/tomo28_FAS.xml',
                    ROOT_PATH+'/in/tomo27_FAS.xml',
                    ROOT_PATH+'/in/tomo22_FAS.xml',
                    ROOT_PATH+'/in/tomo1_FAS.xml',
                    ROOT_PATH+'/in/tomo18_FAS.xml',
                    ROOT_PATH+'/in/tomo17_FAS.xml',
                    ROOT_PATH+'/in/tomo13_FAS.xml',
                    )
# in_plist_xml_l = (ROOT_PATH + '/pl/pl_full_part.xml', )

# Mask tomogram
in_ref_l =         (ROOT_PATH+'/in/tomo6_cytoplasm.mrc',
                    ROOT_PATH+'/in/tomo56_cytoplasm.mrc',
                    ROOT_PATH+'/in/tomo55_cytoplasm.mrc',
                    ROOT_PATH+'/in/tomo54_cytoplasm.mrc',
                    ROOT_PATH+'/in/tomo53_cytoplasm.mrc',
                    ROOT_PATH+'/in/tomo4_cytoplasm.mrc',
                    ROOT_PATH+'/in/tomo48_cytoplasm.mrc',
                    ROOT_PATH+'/in/tomo40_cytoplasm.mrc',
                    ROOT_PATH+'/in/tomo3_cytoplasm.mrc',
                    ROOT_PATH+'/in/tomo38_cytoplasm.mrc',
                    ROOT_PATH+'/in/tomo37_cytoplasm.mrc',
                    ROOT_PATH+'/in/tomo36_cytoplasm.mrc',
                    ROOT_PATH+'/in/tomo35_cytoplasm.mrc',
                    ROOT_PATH+'/in/tomo34_cytoplasm.mrc',
                    ROOT_PATH+'/in/tomo33_cytoplasm.mrc',
                    ROOT_PATH+'/in/tomo32_cytoplasm.mrc',
                    ROOT_PATH+'/in/tomo31_cytoplasm.mrc',
                    ROOT_PATH+'/in/tomo2_cytoplasm.mrc',
                    ROOT_PATH+'/in/tomo29_cytoplasm.mrc',
                    ROOT_PATH+'/in/tomo28_cytoplasm.mrc',
                    ROOT_PATH+'/in/tomo27_cytoplasm.mrc',
                    ROOT_PATH+'/in/tomo22_cytoplasm.mrc',
                    ROOT_PATH+'/in/tomo1_cytoplasm.mrc',
                    ROOT_PATH+'/in/tomo18_cytoplasm.mrc',
                    ROOT_PATH+'/in/tomo17_cytoplasm.mrc',
                    ROOT_PATH+'/in/tomo13_cytoplasm.mrc',
                    )

####### Output data

out_stem = 'mmap_test_400'
out_dir = ROOT_PATH + '/results/mmap_test/'
out_fmt = '.png'


###### Transformation data (from particle list to holding tomogram)

# Offsets
in_off = (0, 0, 0)

# Particles list binning factor
in_bin = 1.

# Tomogram resolution
in_res = 1.368

# In not None Memory map active: useful for saving memory in many tomograms and few particles
mmap_dir = ROOT_PATH + '/mmap' # None

####### Particle

in_temp = ROOT_PATH + '/in/template_mask.mrc'
# in_temp = ROOT_PATH + '/in/tmp_def_5_size_24_bin_g1_sf1.mrc'
in_con = 'tom'

###### Analysis

gather = True
gstd = gather
do_ana_1 = False
do_ana_2 = True
an_mx_d1 = 200 # nm (first order)
an_mx_d2 = 80 # nm (second order)
an_nbins = 20
an_per = 5 #%
an_nsim_1 = 400 # Number of simulations for first order
an_samp_f = 2000 # Number of samples for F
an_nsim_2 = 400 # Number of simulations for second order
an_samp_k = 40 # Number of samples for K
an_wo = 1
an_tcsr = True

###### Simulator

sim_nrot = 1000
do_linker = False
lin_l = 13 # nm
lin_sg = 1 # nm
lin_nc = 2
do_rubc = False

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
from pyseg.spatial import UniStat, PlotUni, CSRSimulator, LinkerSimulator, RubCSimulation, CSRTSimulator
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
if mmap_dir is not None:
    print '\tIMPORTANT: memory maps option active!'
print '\tTransformation properties (from reference to holding tomogram):'
print '\t\t-Offset: ' + str(in_off) + ' voxels'
print '\t\t-Coordinates binning: ' + str(in_bin)
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
print ''

######### Process

print 'Main Routine: '

analyzer = PlotUni()

print '\tParticles list loop: '
for in_plist_xml, in_ref in zip(in_plist_xml_l, in_ref_l):

    print '\t\tParsing particle list: ' + in_plist_xml
    pl_path, pl_fname = os.path.split(in_plist_xml)
    plist = ParticleList(pl_path + '/sub')
    plist.load(in_plist_xml)
    print '\t\tLoading mask: ' + in_ref
    mask = disperse_io.load_tomo(in_ref)
    temp, rots = None, None
    if in_temp is not None:
        temp = disperse_io.load_tomo(in_temp)
        cloud, rots = plist.get_particles_coords(do_shift=True, rot=True)
    else:
        cloud = plist.get_particles_coords(do_shift=True, rot=False)
    print '\t\t\t-Number of particles found: ' + str(plist.get_num_particles())
    if plist.get_num_particles() <= 0:
        print '\t\t\t-WARNING: No particles found!, skipping this particle list...'
        continue

    if in_bin > 1:
        i_bin = 1. / in_bin
        res = in_res * in_bin
        print '\t\tChanging resolution to: ' + str(res) + ' nm/voxel\n'
        print '\t\tBinning coordinates...'
        hold_cloud = cloud
        cloud = list()
        for coords in hold_cloud:
            cloud.append(coords / in_bin)
        bin_sp = int(math.ceil(mask.shape[0]*i_bin)), int(math.ceil(mask.shape[1]*i_bin)), int(math.ceil(mask.shape[2]*i_bin))
        print '\t\tResizing the mask (from ' + str(mask.shape) + ' to ' + str(bin_sp) + ')...'
        mask = resize(mask.astype(np.float32), bin_sp) > 0.
        if temp is not None:
            bin_tsp = int(math.ceil(temp.shape[0]*i_bin)), int(math.ceil(temp.shape[1]*i_bin)), \
                      int(math.ceil(temp.shape[2]*i_bin))
            temp = resize(temp, bin_tsp) > 0.
            ps.disperse_io.save_numpy(temp, out_dir+'/'+out_stem+'_temp_bin.mrc')
    else:
        res = float(in_res)

    print '\t\tSetting up simulator...'
    if temp is None:
        simulator = CSRSimulator()
    else:
        if do_linker:
            leng = lin_l / res
            simulator = LinkerSimulator(leng, lin_nc, temp, nrots=sim_nrot, no_over=True, len_sg=lin_sg)
        elif do_rubc:
            simulator = RubCSimulation(res, temp)
        else:
            simulator = CSRTSimulator(temp, nrots=sim_nrot)

    print '\t\tCreating the object for the univariate analysis...'
    pl_stem = os.path.splitext(pl_fname)[0]
    uni = UniStat(np.asarray(cloud), mask, res, rots=np.asarray(rots), temp=temp, conv=in_con,
                  mmap=mmap_dir+'/'+pl_stem)
    uni.set_name(out_stem)
    uni.set_simulator(simulator)
    print '\t\t\t-Number of points in the pattern: ' + str(uni.get_n_points())
    if uni.get_n_points() <= 0:
        print '\t\t\t-WARNING: No points found!, skipping this particle list...'
        continue
    analyzer.insert_uni(uni)
    if uni.is_2D():
        print '\t\t\t-Pattern total intensity: ' + str(uni.get_intensity()) + ' points/nm^2'
    else:
        print '\t\t\t-Pattern total intensity: ' + str(uni.get_intensity()) + ' points/nm^3'
        if do_rubc:
            hold_points = simulator.gen_rand_in_mask(len(cloud), uni.get_mask())
            print '\t\t-Rubisco crystal lattice density: ' + str(float(hold_points.shape[0])/uni.get_volume()) \
                 + ' particles/nm^3'

    # print '\tStoring the patterns to analyze...'
    # uni.save_sparse(out_dir+'/'+pl_stem+'_pts.mrc', mask=True)
    # uni.save_sparse(out_dir+'/'+pl_stem+'_dense.mrc', mask=True)
    # ps.disperse_io.save_numpy(uni.get_mask(), out_dir+'/'+pl_stem+'_mask.mrc')
    # uni.save_random_instance(out_dir+'/'+pl_stem+'_rnd.mrc')
    # uni.save_random_instance(out_dir+'/'+pl_stem+'_rnd_pts.mrc', pts=True)

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

# print '\tStoring the analyzer...'
# analyzer.pickle(out_dir+'/'+out_stem+'_ana.pkl')

print 'Terminated. (' + time.strftime("%c") + ')'