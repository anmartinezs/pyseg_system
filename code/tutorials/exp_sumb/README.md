 Additional and modified scripts in respect to **synth_sumb** to process **s**ingle **u**oriented **m**em**b**ranes from **exp**erimental data.
 
 * **pre**:
    + **pre_tomo_seg.py**: pre-processing for mb_graph_batch.py of un-oriented and oriented membranes from TomoSegMemTV output.
    + **rec_particles.py**: script for reconstructing particles RELION compatible from full reconstructed tomograms.
    + **resize_particles.py**: script for resizing (upsizing) already reconstructed particles, the resizing can be selective so its allow to mix
    particles with different pixel sizes.
 
 If you are not familiar with PySeg scripts, please go to **/docs/tutorials/synth_sumb/synth_sumb.pdf** to learn the basis.