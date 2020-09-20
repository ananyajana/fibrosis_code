#!/usr/bin/env bash

for i in {1..6}
do
ngpus=6
epochs=30
exp_name='fib'
exp_num='baseline_1'
bool_use_resnet=0

pre_train_sup=1
method='rotate'
sup_model='fea_ex2'

chkpt_path='fibrosis_3fold'
#chkpt_path='fea_ex2_rotate_fibrosis_3_fold_1_epoch_run1'
data_dir='./data/data_miccai'
img_dir='./data/data_miccai'
for exp_name in 'fib' 'nas_stea' 'nas_lob' 'nas_balloon'
do
for exp_num in 'baseline_1' 'baseline_2' 'baseline_3'
do

if [ "$exp_num" = "baseline_1" ]; then
    echo "baseline_1, pass fold_1 checkpoint"
    sup_model_path='./chkpts/checkpoint_fea_ex2_smooth_1_epoch_fold_1/checkpoint_best.pth.tar'
elif [ "$exp_num" = "baseline_2" ]; then
    echo "baseline_2, pass fold_2 checkpoint"
    sup_model_path='./chkpts/checkpoint_fea_ex2_smooth_1_epoch_fold_2/checkpoint_best.pth.tar'
else
    echo "baseline_3, pass fold_3 checkpoint"
    sup_model_path='./chkpts/checkpoint_fea_ex2_smooth_1_epoch_fold_3/checkpoint_best.pth.tar'
fi
num_class=3
if [ "$exp_name" = "nas_stea" ]; then
    echo "steat. class num 2"
    num_class=2
else
    echo "not steat. class num 3"
    num_class=3
fi
python train.py  --random-seed -1 --epochs ${epochs}  --exp ${exp_name} --exp-num ${exp_num} --model Classifier --num-class ${num_class} --batch-size 4 \
  --save-dir ./experiments/${exp_name}/${exp_num} --gpus ${ngpus} --use_resnet ${bool_use_resnet} --sup_model_path ${sup_model_path} --data-dir ${data_dir} --pre_train_sup ${pre_train_sup}
python test.py --exp ${exp_name} --exp-num ${exp_num} --model Classifier --num-class ${num_class} --batch-size 4  \
  --model-path ./experiments/${exp_name}/${exp_num}/checkpoint_best.pth.tar  \
  --save-dir ./experiments/${exp_name}/${exp_num}/best --gpus ${ngpus} --use_resnet ${bool_use_resnet} --img-dir ${img_dir}
done
done

# collect the results
python parse_results.py
python create_excel.py --chkpt_path ${chkpt_path}
exps='./experiments'
all_exps='../exp_all'
#mv ${exps} ${all_exps}'/'${exps}'_'${chkpt_path}
excel='output.xlsx'
#mv ${exps} ${exps}'_'${chkpt_path}'_'$i
mv $excel $i'_'$excel
done
