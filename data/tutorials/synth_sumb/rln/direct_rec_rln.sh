#!/bin/bash

for i in `seq 0 25`;
do
	printf -v cmd "%s" "relion_reconstruct --i ../../../../data/tutorials/synth_sumb/class/mbf_align_all/class_ap_r_ali_k$i""_split.star --o ../../../../data/tutorials/synth_sumb/class/mbf_align_all/dr_class_ap_r_ali_k$i"".mrc --angpix 2.62 --maxres 45"
	echo $cmd
    $cmd
done
