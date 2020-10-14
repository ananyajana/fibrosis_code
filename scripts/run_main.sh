#!/usr/bin/env bash

#base_path='../data/patient_folders_arranged/'
dest_path='../data/patient_folders_arranged_empty_deleted/'
rm -rf $dest_path
mkdir $dest_path
cp -rf $base_path/* $dest_path 
python delete_almost_empty_patches_seg.py --base_path $dest_path

seed=2
h5_filename='patient_details.h5'
src='../data/all_patients_images_resized/'
src2='../data/patient_folders_arranged/'
#for data_type in 'lbp' 'edt' 'img_lbp' 'edt_lbp' 'edt_img'
for data_type in 'lbp' 'img_lbp' 
do
data_path='../data/data_miccai_'$data_type
rm -rf $data_path
mkdir $data_path

python build_30_specific_pat.py --data_path $data_path --src $src --h5_filename $h5_filename --seed ${seed} --data_type ${data_type} --src2 $src2
done
