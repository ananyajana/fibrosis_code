#!/usr/bin/env bash

data_path='./data/lits_131_pat_all_train/'
sup_model='contextrestore2'
epochs=5000
num_feats=256
save_folder_name='checkpoint_'${sup_model}'_13oct'
save_dir='../fibrosis_self_supervision/chkpts/'${save_folder_name}
python self_supervision3.py --data_path ${data_path} --save_dir ${save_dir} --n_epochs ${epochs} --num_feats ${num_feats} --sup_model ${sup_model}
