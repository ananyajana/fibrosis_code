#!/usr/bin/env bash

data_path='./data/lits_50_pat/'
method='rotate'
sup_model='fea_ex2'
epochs=1
run=1
num_feats=128
save_folder_name=${sup_model}'_'${method}'_'${epochs}'_epoch_run_'${run}
save_dir='../fibrosis_self_supervision/chkpts/'${save_folder_name}
python self_supervision.py --data_path ${data_path} --save_dir ${save_dir} --method ${method} --n_epochs ${epochs} --num_feats ${num_feats} --sup_model ${sup_model}
