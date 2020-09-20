#!/usr/bin/env bash

data_path='./data/lits_50_pat/'
method='smooth'
sup_model='fea_ex2'
epochs=1
run=1
for method in 'smooth' 'rotate' 'downsize'
do
for run in {1..3}
do
save_folder_name=${sup_model}'_'${method}'_'${epochs}'_epoch_run_'${run}
save_dir='./chkpts/'${save_folder_name}
python self_supervision.py --data_path ${data_path} --save_dir ${save_dir} --method ${method} --n_epochs ${epochs}
done
done
