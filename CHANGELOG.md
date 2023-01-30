* **13/02/2020**
    + Adding this CHANGELOG file
    + **clean_out_data.sh**: error correction for ab-inition data directories generation.
    + **code/tutorials/synth_sumb/pre/gen_microsomes.py**: error correction when the number of microsomes is greater than the number of processes.
    + Corrections in organizations analysis to fit synapse scripts.  
* **25/02/2020**
    + Adding statistical analysis for filaments
* **10/03/2020**
    + Updating synapses colocalization routines
    + Adding /code/tutorials/exp_sumb/tracing
    + Bugs correction in /code/tutorials/synth_sumb/tracing/rec_particles.py
* **16/03/2020**
    + Adding AG and Kmeans (with PCA) to /code/tutorials/synth_sumb/class/plane_align_class.py
    + Creating /code/pyorg/diff_geom module
* **17/03/2020**
    + Git Nightly branch created
* **20/03/2020**
    + Adding filament and segmentations analysis in PyOrg
* **21/03/2020**
    + Bug correction in /code/tutorials/exp_sumb/pre/pre_tomo_seg.py
    + Improvement of clean_out_data.sh
    + Adding nighly branch to README.md
* **26/03/2020**
    + Adding file code/tutorials/synth_sumb/class/mbz_cly_mask.m
* **30/03/2020**
    + Correcting bug in code/tutorials/exp_sumb/pre/pre_tomo_seg.m
* **08/04/2020**
    + Updates in rec_particles.py script
    + Synthetic model (ModelFilsRSR) for filaments
* **30/04/2020**
    + Adding /code/tutorials/exp_somb tutorial for processing single oriented membranes.
    + Commenting mask imsave in code/tutorials/synth_sumb/class/plane_align_class.py
* **06/05/2020**
    + Adding membrane signal suppression code/tutorials/exp_somb/post_rec_particles.py
    + Bug fixed for code/tutorials/exp_somb/post_rec_particles.py for non-splitting mode
* **26/05/2020**
    + Adding tutorial exp_domb (/{code,data}/tutorials/exp_domb)
* **17/07/2020**
    + Adding filament analysis scrits in code/pyorg/scripts/filaments
* **03/09/2020**
    + Correction bug in script /code/tutorials/exp_somb/pre_tomos_seg.py
* **11/09/2020**
    + Corrections for test in module /code/pyorg/surf
* **08/10/2020**
    + Corrections for the statistical analysis of filaments organization
* **26/12/2020**
    + Adding comments to explain XML files
* **27/12/2020**
    + Segmentation pre-processing scripts in tutorial using global_analysis function
* **08/01/2020**
    + Eliminating 'WARNING: filament with less than 3 points!' messages
    + Preparing scripts to build an independant repository for filaments quantitative analysis
* **02/09/2020**
    + Adding Gaussian low-pass filtering to script ``/code/tutorials/exp_somb/post_rec_particles.py``
* **26/12/2020**
    + Adding comments to explain XML file
* **27/12/2020**
    + Using global_analysis within pre-procesing segmentation scripts in PySeg tutorials
* **26/01/2021**
    + Adding particle surface overlapping fraction option
* **06/02/2021**
    + Using numpy histogram funtion to increase the speed for computational units in PyOrg.
* **01/03/2021**
    + Adding computation level option (out_level) to plane_align_class.py
* **15/04/2021**
    + Modifications to reduce memory occupancy for PyOrg.
* **30/08/2021**
    + Updating **pyto** to deal with scipy 1.6.
* **12/09/2021**
    + Updating script for pre-processing tomograms for single un-oriented membranes.
* **15/09/2021**
    + Bug correction: input parsing for Booleans introduced during integration with SCIPION.
* **02/10/2021**
    + Adding MRC I/O TOM-Toolbox functions
* **21/07/2022**
    + Adding system installers for Python3 on Ubuntu 20.04	
* **27/07/2022**
    + Adding models in MRC format for generating synthetic tomogras in /data/tutorials/synth_sumb/modes
* **30/01/2023**
    + Since numpy_1.24 types like np.int, np.float and np.bool are no longer used, they have been substituted Python native types int, float and bool respectively.
    + Dockefile with container for Ubuntu 20.04 was added by DimitriosBellos
