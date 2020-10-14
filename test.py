
import os
import random
from tqdm import tqdm
import h5py
from sklearn import metrics

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as torch_models
from torch.utils.data import DataLoader
import numpy as np

from options import Options
from dataset import LiverDataset
from models import BaselineNet, BaselineNet2
import utils


def main():
    opt = Options(isTrain=False)
    opt.parse()

    os.makedirs(opt.test['save_dir'], exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.train['gpus'])

    # opt.save_options()
    opt.print_options()

    # if not os.path.isfile('{:s}/test_prob_results.npy'.format(opt.test['save_dir'])):
    get_probs(opt)

    # compute accuracy
    save_dir = opt.test['save_dir']
    test_prob = np.load('{:s}/test_prob_results.npy'.format(save_dir), allow_pickle=True).item()

    txt_file = open('{:s}/test_results.txt'.format(save_dir), 'w')
    save_filepath = '{:s}/roc.png'.format(save_dir)
    acc, auc, auc_CIs = compute_metrics(test_prob, opt)

    if opt.exp in ['fib', 'nas_lob', 'nas_balloon']:
        ave_auc = np.mean(auc)
        ave_auc_CIs = np.mean(auc_CIs, axis=0)
        message = 'Acc: {:5.2f}\tAUC: {:5.2f} ({:5.2f}, {:5.2f})\t' \
                  'AUC0: {:5.2f} ({:5.2f}, {:5.2f})\tAUC1: {:5.2f} ({:5.2f}, {:5.2f})\tAUC2: {:5.2f} ({:5.2f}, {:5.2f})' \
            .format(acc * 100, ave_auc * 100, ave_auc_CIs[0] * 100, ave_auc_CIs[1] * 100,
                    auc[0] * 100, auc_CIs[0][0] * 100, auc_CIs[0][1] * 100,
                    auc[1] * 100, auc_CIs[1][0] * 100, auc_CIs[1][1] * 100,
                    auc[2] * 100, auc_CIs[2][0] * 100, auc_CIs[2][1] * 100)
    else:
        message = 'Acc: {:5.2f}\tAUC: {:5.2f} ({:5.2f}, {:5.2f})\n\n' \
            .format(acc * 100, auc * 100, auc_CIs[0] * 100, auc_CIs[1] * 100)

    print(message)
    txt_file.write(message)
    txt_file.close()


def get_probs(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.test['gpus'])

    model_path = opt.test['model_path']
    save_dir = opt.test['save_dir']

    # create model
    #print('in test file', opt.model['use_resnet'])
    if opt.model['use_resnet'] == 1:
        model = BaselineNet2(opt.model['out_c'], opt.model['resnet_layers'], opt.model['train_res4'])
    else:
        #print('using baseline net')
        #print('opt model use_resnet value', opt.model['use_resnet'])
        #print('opt model use_resnet ', type(opt.model['use_resnet']))
        model = BaselineNet(opt.model['in_c'], opt.model['out_c'])
    model = model.cuda()

    print("=> loading trained model")
    best_checkpoint = torch.load(model_path)
    model.load_state_dict(best_checkpoint['state_dict'])
    # model = model.module
    print('Model obtained in epoch: {:d}'.format(best_checkpoint['epoch']))

    data_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485],
                                         std=[0.229])
                ])
    fold_num = opt.exp_num.split('_')[-1]
    print('Fold number: {:s}'.format(fold_num))
    if opt.model['use_resnet'] == 1:
        test_set = LiverDataset('{:s}/test{:s}.h5'.format(opt.test['img_dir'], fold_num), data_transform, opt)
    else:
        test_set = LiverDataset('{:s}/test{:s}.h5'.format(opt.test['img_dir'], fold_num), data_transform)
 
    print("=> Test begins:")
    # switch to evaluate mode
    model.eval()
    
    prob_results = {}
    for i in tqdm(range(len(test_set))):
        slide_name = test_set.keys[i]
        ct_data, fib_label, nas_stea_label, nas_lob_label, nas_balloon_label = test_set[i]
        input_data = ct_data
        if opt.exp == 'fib':
            label = fib_label
        elif opt.exp == 'nas_stea':
            label = nas_stea_label
        elif opt.exp == 'nas_lob':
            label = nas_lob_label
        elif opt.exp == 'nas_balloon':
            label = nas_balloon_label
        else:
            raise ValueError('Wrong nas label name')
        with torch.no_grad():

            output = model(input_data.cuda())
            probs = nn.functional.softmax(output, dim=1).squeeze(0).cpu().numpy()
           
        prob_results[slide_name] = {'probs': probs, 'labels': label.item()}
  
    np.save('{:s}/test_prob_results.npy'.format(save_dir), prob_results)
 

def compute_metrics(slides_probs, opt):
    all_probs = []
    all_labels = []
    for slide_name, data in slides_probs.items():
        # print('{:s}\t{:.4f}\t{:.4f}\t{:.4f}\t{:d}'.format(slide_name, data['prob_nas'][0], data['prob_nas'][1],
        #                                                   data['prob_nas'][2], data['label_nas']))
        all_probs.append(data['probs'])
        all_labels.append(data['labels'])
        
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_pred = np.argmax(all_probs, axis=1).astype(np.float)

    acc = metrics.accuracy_score(all_labels, all_pred)
    if np.unique(np.array(all_labels)).size == 1:
        auc = -0.01
        auc_CIs = [-0.01, -0.01]
    else:
        if opt.exp in ['fib', 'nas_lob', 'nas_balloon']:
            auc = []
            auc_CIs = []
            for i in range(3):
                auc_i, auc_CIs_i = bootstrap_AUC_CIs(all_probs[:, i], (all_labels==i).astype(np.float))
                auc.append(auc_i)
                auc_CIs.append(auc_CIs_i)
            auc = np.array(auc)
            auc_CIs = np.array(auc_CIs)
        else:
            auc, auc_CIs = bootstrap_AUC_CIs(all_probs[:, 1], all_labels)
    return acc, auc, auc_CIs



def bootstrap_AUC_CIs(probs, labels):
    probs = np.array(probs)
    labels = np.array(labels)
    N_slide = len(probs)
    index_list = np.arange(0, N_slide)
    AUC_list = []
    i = 0
    while i < 1000:
        sampled_indices = random.choices(index_list, k=N_slide)
        sampled_probs = probs[sampled_indices]
        sampled_labels = labels[sampled_indices]

        if np.unique(sampled_labels).size == 1:
            continue

        auc_bs = metrics.roc_auc_score(sampled_labels, sampled_probs)
        AUC_list.append(auc_bs)
        i += 1

    assert len(AUC_list) == 1000
    AUC_list = np.array(AUC_list)
    auc_avg = np.mean(AUC_list)
    auc_CIs = [np.percentile(AUC_list, 2.5), np.percentile(AUC_list, 97.5)]
    return auc_avg, auc_CIs


if __name__ == '__main__':
    main()
