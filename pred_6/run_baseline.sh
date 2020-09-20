#!/usr/bin/env bash

for i in {1..6}
do
ngpus=7
epochs=30
exp_name='fib'
exp_num='baseline_1'
bool_use_resnet=0

pre_train_sup=1
method='rotate'
#method='smooth'
sup_model='fea_ex2'
chkpt_path=${sup_model}'_'${method}'_1_epoch_run_1'
sup_model_path='../fibrosis_self_supervision/chkpts/'$chkpt_path'/checkpoint_best.pth.tar'

data_dir='./data/data_miccai'
img_dir=${data_dir}

for exp_name in 'fib' 'nas_stea' 'nas_lob' 'nas_balloon'
do
for exp_num in 'baseline_1' 'baseline_2' 'baseline_3'
do
num_class=3
if [ "$exp_name" = "nas_stea" ]; then
    echo "steat. class num 2"
    num_class=2
else
    echo "not steat. class num 3"
    num_class=3
fi
python train.py  --random-seed -1 --epochs ${epochs}  --exp ${exp_name} --exp-num ${exp_num} --model Classifier --num-class ${num_class} --batch-size 4 \
  --save-dir ./experiments/${exp_name}/${exp_num} --gpus ${ngpus} --use_resnet ${bool_use_resnet} --sup_model_path ${sup_model_path} --data-dir ${data_dir} \
  --pre_train_sup ${pre_train_sup}
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
mv ${exps} ${exps}'_'${chkpt_path}'_'$i
mv $excel $i'_'$excel
done
