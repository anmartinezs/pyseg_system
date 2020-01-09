## PYSEG SYSTEM CODE (v 1.0)

The code of PySeg system has been developed for template-free macromolecule localization, structural and quantitative organization analysis 
for cryo-electron tomography (cryo-ET) data. The code and data here is also available in GitHub (https://github.com/anmartinezs/pyseg_system.git).

**IMPORTANT:** Proficiency in Python programming language is required to use all capabilities contained in PySeg, as GUI is not available at this point. PySeg is an open source project, contributions are welcome.

### CONTENTS

* **pyseg**: python package for template-free particle picking and usupervised structural classification
* **pyorg**: python package for quantitative organization analysis
* **tomosegmemtv**: matlab/C scripts for robust membrane segmentation 
* **tests**: contains scripts for functionality testing, output data stored here during test executing are cleaned by running **clean_out_data.sh** (one folder above).
* **tutorials**: scripts and data for learning how to use the software (see documentation one folder above)
* **run.sh** a script for testing the functiontionallity, only usable on Code Ocen capsule (https://codeocean.com/capsule/0526052/tree)

### LICENSE

Licensed under the Apache License, Version 2.0 (see LICENSE file)

### PUBLICATIONS

* Template-free particle picking and unsupervised classification (pyseg, pyorg):

        [1] Martinez-Sanchez et al. "Template-free detection and classification of heterogeneous membrane-bound complexes in cryo-electron tomograms" Nature Methods (2020) doi:10.1038/s41592-019-0687-1

* Membrane segmentation (tomosegmemtv):

        [2] Martinez-Sanchez et al. "Robust membrane detection based on tensor voting for electron tomography" J Struct Biol (2014)

