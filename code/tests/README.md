This folder contains a serial of test for ensuring the proper PySeg installation and its functionality. 

### CONTENTS

* **tracing_grid.sh** generates a serial of syntethic grids to estimate the precision of the spatially embedded graph to recover their geometry.
* **tracing.sh** computes the spatially embedded graph on a experimental microsme, traces the filaments and extract the membrane associated particles for both cytosolic and lumen sides.
* **classification.sh** unsupervised structural classification for membrane aligned particles.
* **org.sh** statistical analysis for the particles picked by the script **tracing.sh**
* **uni_2nd_speedup.py**: script for measuring the speed-up of computing the univariate second order metrics. 