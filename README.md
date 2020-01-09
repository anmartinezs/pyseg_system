# pyseg_system (v 1.0)
De novo analysis for cryo-electron tomography.
The code and data here is also available in GitHub (https://github.com/anmartinezs/pyseg_system.git).

### CONTENTS

* **data**: input data for running some tests and tutorials, output and intermediate data stored here during tutorials and tests execution are cleaned by running **clean_out_data.sh**.
* **code**: the code (libraries and execution scripts) for PySeg, PyOrg and TomoSegMemTV.
* **sys**: third party software with their installers.
  + DisPerSe (v. 0.9.24)
    - Cfitsio (v. 3.380)
  + CGAL (v 4.7)
  + Graph-tool (v 2.2.44)
  + VTK (v 6.3.0)
* **doc**: documentation files
  + manual: a general manual, with installation, for PySeg (v 0.1)
  + tutorials: specific examples to introduce the users in the usage of the software
    - synth_sumb: tutorial for using PySeg for single unoriented membranes using synthetic data
  + tomosegmemtv: documentation for TomoSegMemTV (membrane segmentation for electron tomogramphy)
  + synapsegtools: documentation for SynapSegTools (some graphic extension for post-processing the outputs of TomoSegMemTV)

### INSTALLATION

A description of the requirements, auxiliary software, installation and functionality testing is available on **docs/manual/manual.pdf** file. 

### USAGE
In **docs/tutorials/synth_sumb/synth_sumb.pdf** there is a tutorial for de novo analysis of membrane proteins using self-generated synthetic data, it is strongly recomended to complete this tutorial before starting with your experimental data.

### LICENSE

Licensed under the Apache License, Version 2.0 (see LICENSE file)

### PUBLICATIONS

* Template-free particle picking and unsupervised classification (PySeg, PyOrg):

        [1] Martinez-Sanchez et al. "Template-free detection and classification of heterogeneous membrane-bound complexes in cryo-electron tomograms" Nature Methods (2020) doi:10.1038/s41592-019-0687-1


* Membrane segmentation (TomoSegMemTV):

        [2] Martinez-Sanchez et al. "Robust membrane detection based on tensor voting for electron tomography" J Struct Biol (2014) https://doi.org/10.1016/j.jsb.2014.02.015

