#!/usr/bin/env bash

dirname='run_prev_20oct_contextrestore_1k_epoch'
mkdir $dirname
mv *.xlsx $dirname
mv experiments* $dirname
for i in {1..10}
do
ngpus=4
epochs=30
exp_name='fib'
exp_num='baseline_1'
bool_use_resnet=0

pre_train_sup=1
sup_model='context_restore2'
chkpt_path='checkpoint_contextrestore2_gan_loss_patch_sz_20'
#sup_model_path='../fibrosis_self_supervision/chkpts/'$chkpt_path'/checkpoint_best.pth.tar'
sup_model_path='../fibrosis_self_supervision/'$chkpt_path'/checkpoint_best.pth.tar'


#data_dir='../fibrosis_self_supervision/data/data_miccai'
data_dir='../fibrosis_self_supervision/data/data_miccai_224_lbp'
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
python train.py  --random-seed -1 --epochs ${epochs}  --exp ${exp_name} --exp-num ${exp_num} --model Classifier --num-class ${num_class} --batch-size 1 \
  --save-dir ./experiments/${exp_name}/${exp_num} --gpus ${ngpus} --use_resnet ${bool_use_resnet} --sup_model_path ${sup_model_path} --data-dir ${data_dir} \
  --pre_train_sup ${pre_train_sup}
python test.py --exp ${exp_name} --exp-num ${exp_num} --model Classifier --num-class ${num_class} --batch-size 1  \
  --model-path ./experiments/${exp_name}/${exp_num}/checkpoint_best.pth.tar  \
  --save-dir ./experiments/${exp_name}/${exp_num}/best --gpus ${ngpus} --use_resnet ${bool_use_resnet} --img-dir ${img_dir}
done
done

# collect the results
chkpt_path='context_restore2_'$i
python parse_results.py
python create_excel.py --chkpt_path ${chkpt_path}
exps='./experiments'
all_exps='../exp_all'
#mv ${exps} ${all_exps}'/'${exps}'_'${chkpt_path}
excel='output.xlsx'
mv ${exps} ${exps}'_'$i
mv $excel $i'_'$excel
done
