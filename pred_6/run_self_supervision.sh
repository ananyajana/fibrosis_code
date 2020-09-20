#!/usr/bin/env bash

data_path='./data/data_isbi_non_seg_3fold'
save_dir='./chkpts/checkpoint_fea_ex2_smooth_1_epoch'
#save_dir='./checkpoint_fea_ex2_smooth_1_epoch'
method='rotate'
for fold in 'fold_1' 'fold_2' 'fold_3'
do
fold_data=${data_path}'/'${fold}
fold_save_dir=${save_dir}'_'${fold}
echo $fold_data
echo $fold_save_dir
python self_supervision_fibrosis_data.py --data_path ${fold_data} --save_dir ${fold_save_dir} --method ${method}
done
