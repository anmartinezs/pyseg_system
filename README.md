# pyseg_system (v 2.0.0)
De novo analysis for cryo-electron tomography.

This GitHub repository have two branches (git checkout <feature_branch>):
* **master**: (default) stable Python 3 version.
* **python2**: old Python 2.7 (deprecated).
* **python3**: branch with the latest modifications and improved.

## What's new from v2.0.1
Python 3 transition completed and changes for being compatible with [Scipion](https://github.com/scipion-em/scipion-em-pyseg.git).
Now PySeg has been ugraded to run Ubunutu 22.04 LTS.

### CONTENTS

* **data**: input data for running some tests and tutorials, output and intermediate data stored here during tutorials and tests execution are cleaned by running **clean_out_data.sh**.
* **code**: the code (libraries and execution scripts) for PySeg, PyOrg and TomoSegMemTV.
* **sys**: third party software with their installers.
  + DisPerSe (v. 0.9.24)
    - Cfitsio (v. 3.380)
  + CGAL (v 4.7)
  + Graph-tool 
  + VTK 
* **doc**: documentation files
  + manual: a general manual, with installation instructions included, for PySeg 
  + tutorials: specific examples to introduce the users in the usage of the software
    - synth_sumb: basic tutorial for using PySeg for **s**ingle **u**noriented **m**em**b**ranes using **synth**etic data
    - synth_ssmb: Deprecated, just for testing.
    - exp_sumb: additional and modified scripts in respect to **synth_sumb** to process **s**ingle **u**oriented **m**em**b**ranes from **exp**erimental data.
    - exp_somb: additional and modified scripts in respect to **synth_sumb** to process **s**ingle **o**riented **m**em**b**ranes from **exp**erimental data.
    - exp_domb: additional and modified scripts in respect to **synth_sumb** to process **d**ouble **o**riented **m**em**b**ranes from **exp**erimental data.
  + tomosegmemtv: documentation for TomoSegMemTV (membrane segmentation for electron tomogramphy)
  + synapsegtools: documentation for SynapSegTools (some graphic extension for post-processing the outputs of TomoSegMemTV)

### INSTALLATION

A description of the requirements, auxiliary software, installation and functionality testing is available on **docs/manual/manual.pdf** file. 

### BUILDING AND RUNNING WITH DOCKER
You may want to build a docker container to run Pyseg. First, you have build docker image in **Dockerfile**:
```
docker build . -t pyseg:latest
```
Then you can run a terminal on image by:
```
docker run -it pyseg:latest bash
```
In this terminal, you can work like in any other Linux terminal having acces to the whole Pyseg funcitionality.
If you just want to run a specific script then (replace the <> placeholders accordingly, a typical location for the <mount-directory-in-container> is `/mnt`):
```
docker run --rm -it -v <data-directory-in-host-machine>:<mount-directory-in-container> pyseg:latest <command> <options>
```
For the available commands look at [USAGE](README.md#USAGE).

### USAGE
In **docs/tutorials/synth_sumb/synth_sumb.pdf** there is a tutorial for de novo analysis of membrane proteins using self-generated synthetic data, it is strongly recomended to complete this tutorial before starting with your experimental data.

### REPORTING BUGS

If you have found a bug or have an issue with the software, please open an issue [here](https://github.com/anmartinezs/pyseg_system/wiki).

### LICENSE

Licensed under the Apache License, Version 2.0 (see LICENSE file)

### PUBLICATIONS

* Template-free particle picking and unsupervised classification (PySeg):

        [1] Martinez-Sanchez et al. "Template-free detection and classification of heterogeneous membrane-bound complexes in cryo-electron tomograms" Nature Methods (2020) doi:10.1038/s41592-019-0687-1


* Membrane segmentation (TomoSegMemTV):

        [2] Martinez-Sanchez et al. "Robust membrane detection based on tensor voting for electron tomography" J Struct Biol (2014) https://doi.org/10.1016/j.jsb.2014.02.015
 
* Statistical spatial analysis (PyOrg):

        [3] Martinez-Sanchez, Lucic & Baumeister. "Statistical spatial analysis for cryo-electron tomography" Comput Methods Programs Biomed (2022) https://doi.org/10.1016/j.cmpb.2022.106693

