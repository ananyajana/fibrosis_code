import h5py
import numpy as np
from skimage import io
from tqdm import tqdm
from glob import glob
import argparse
import pandas as pd
from skimage.feature import local_binary_pattern
import scipy.ndimage as ndimage

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_path', type=str, default='', help='custom data path')
parser.add_argument('--src', type=str, default='', help='custom data path')
parser.add_argument('--src2', type=str, default='', help='custom data path')
parser.add_argument('--h5_filename', type=str, default='', help='custom data path')
parser.add_argument('--seed', type=int, default=0, help='custom data path')
parser.add_argument('--data_type', type=str, default='lbp', help='custom data path')
args = parser.parse_args()

# Deleting the record of patient 15 and patient 24, 
# because we do not have image data for them. the 
# segmented images folder is traversed for missing
# patient indixes and then those are deleted.


original_data_dir=args.src
original_data_dir1='../all_patients_images_resized_part2'
original_data_dir3=args.src2 # contains the path to the segmentation maps which is required in case we want to make edt
save_path = args.data_path
h5_filename = args.h5_filename
seed = args.seed
data_type = args.data_type


prev_N = 32
#N = 43
N = 32
patients = ['Patient_{}'.format(i) for i in range(1, N+1)]

import os

def load_labels():
    df = pd.read_excel('../data/info_file.xlsx', sheet_name='Sheet2')
    df = df.iloc[:, 1:]
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])
    df.drop(df.index[16], inplace=True) # drop the patient_id 17_1
    df.reset_index(drop=True, inplace=True) # reset the indices to take the deletion into account
    df= df[df.index < prev_N]   # our dataset consistsa of only first 32 patients from the first excel sheet

    patient_labels = {'ID': list(df['STUDY_ID'].values),
                      'Fibrosis': list(df['FIBROSIS'].values),
                      'NAS-total': list(df['NAS_TOTAL'].values),
                      'NAS-steatosis': list(df['NAS_STEATOSIS'].values),
                      'NAS-lob': list(df['NAS_LOB_INFL'].values),
                      'NAS-balloon': list(df['NAS_BALLOON'].values)}

    return patient_labels

patient_labels = load_labels()
# find the patient folders present in the given pathi and
# put them into contents list
contents = []
for dirpath, dirs, files in os.walk(original_data_dir):
    for pat in patients:
        for dirpath in dirs:
            if dirpath == pat:
                contents.append(pat)

for dirpath, dirs, files in os.walk(original_data_dir1):
    for pat in patients:
        for dirpath in dirs:
            if dirpath == pat:
                contents.append(pat)

print('contents: ', contents)
# check for the entries present in patients list but not in contents list
for pat in patients:
    if pat not in contents:
        print('Absent', pat)

dataset = h5py.File(h5_filename, 'w')
pat_cnt = 0
for i in range(len(patients)):
    pat = patients[i]
    if pat in contents:
        fibrosis_label = patient_labels['Fibrosis'][i]
        nas_label = patient_labels['NAS-total'][i]
        steatosis_label = patient_labels['NAS-steatosis'][i]
        lobular_label = patient_labels['NAS-lob'][i]
        ballooning_label = patient_labels['NAS-balloon'][i]
        assert nas_label == steatosis_label + lobular_label + ballooning_label

        dataset.create_group('{:s}/CT'.format(pat))
        dataset.create_dataset('{:s}/Fibrosis'.format(pat), data=fibrosis_label)
        dataset.create_dataset('{:s}/NAS'.format(pat), data=nas_label)
        dataset.create_dataset('{:s}/Steatosis'.format(pat), data=steatosis_label)
        dataset.create_dataset('{:s}/Lobular'.format(pat), data=lobular_label)
        dataset.create_dataset('{:s}/Ballooning'.format(pat), data=ballooning_label)

        #print('globbing the CT group')
        print('pat: ', pat)
        if i >= prev_N:
            # globbing the segmentation maps in case of edt
            if 'edt' in data_type:
                CT_seg_files = glob('{:s}/{:s}/masks/*'.format(original_data_dir3, pat))
            CT_files = glob('{:s}/{:s}/*'.format(original_data_dir1, pat))
        else:
            if 'edt' in data_type:
                CT_seg_files = glob('{:s}/{:s}/masks/*'.format(original_data_dir3, pat))
            CT_files = glob('{:s}/{:s}/*'.format(original_data_dir, pat))
        # get the patient number to access the HE and trichrome files for a particular patient
        # Patient17 has two different records and hence they are named as 17_1 and 17_2
        if i == 16:
            pat_num = '{}_2'.format(i+1)
        else:
            pat_num = '{}'.format(i+1)

        for ct in CT_files:
            print('creating dataset for {}'.format(pat))
            pat_cnt += 1
            #print('reading images for patient {}'.format(i+1))
            img_name = ct.split('/')[-1].split('.')[0]
            #print(img_name) 
            img = io.imread(ct)
            #print(img.shape)
            if data_type in 'lbp':
                radius = 1
                n_points = radius * 8
                METHOD = 'uniform'
                img = local_binary_pattern(img, n_points, radius, METHOD)
                dataset.create_dataset('{:s}/CT/{:s}'.format(pat, img_name+data_type), data=img)
            elif data_type in 'img':
                dataset.create_dataset('{:s}/CT/{:s}'.format(pat, img_name), data=img)
            elif data_type in 'edt_lbp':
                radius = 1
                n_points = radius * 8
                METHOD = 'uniform'
                img = local_binary_pattern(img, n_points, radius, METHOD)
                dataset.create_dataset('{:s}/CT/{:s}'.format(pat, img_name+'_lbp'), data=img)
            elif data_type in 'img_lbp':
                dataset.create_dataset('{:s}/CT/{:s}'.format(pat, img_name), data=img)
                radius = 1
                n_points = radius * 8
                METHOD = 'uniform'
                img = local_binary_pattern(img, n_points, radius, METHOD)
                dataset.create_dataset('{:s}/CT/{:s}'.format(pat, img_name+'_lbp'), data=img)
            elif data_type in 'edt_img':
                dataset.create_dataset('{:s}/CT/{:s}'.format(pat, img_name), data=img)
        if 'edt' in data_type:
            for ct in CT_seg_files:
                #print('reading images for patient {}'.format(i+1))
                img_name = ct.split('/')[-1].split('.')[0]
                #print(img_name) 
                img = io.imread(ct)
                #print(img.shape)
                img = ndimage.morphology.distance_transform_edt(img)
                dataset.create_dataset('{:s}/CT/{:s}'.format(pat, img_name+'_edt'), data=img)

            
        print(pat_cnt)

dataset.close()

def split_dataset_according_list(h5_filepath, train_list, test_list, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    h5_file = h5py.File(h5_filepath, 'r')

    keys = list(h5_file.keys())

    # cross validation: fold i
    for fold in range(len(train_list)):
        print('Fold {:d}'.format(fold+1))

        train_file = h5py.File('{:s}/train{:d}.h5'.format(save_dir, fold+1), 'w')
        test_file = h5py.File('{:s}/test{:d}.h5'.format(save_dir, fold+1), 'w')
        for id in train_list[fold]:
            print(id)
            pat_id = 'Patient_' + id
            h5_file.copy(pat_id, train_file)
        for id in test_list[fold]:
            pat_id = 'Patient_' + id
            h5_file.copy(pat_id, test_file)

        print(len(train_file))
        print(len(test_file))
        train_file.close()
        test_file.close()

    h5_file.close()


# data_miccai pat list
train_list =[['1', '11', '12', '13', '17', '18', '20', '22', '25', '26', '27', '28', '29', '3', '32', '4', '5', '7', '8', '9'], ['10', '11', '13', '14', '16', '17', '19', '2', '20', '21', '23', '25', '27', '28', '29', '3', '30', '31', '6', '8'], ['1', '10', '12', '14', '16', '18', '19', '2', '21', '22', '23', '26', '30', '31', '32', '4', '5', '6', '7', '9']]


test_list = [['10', '14', '16', '19', '2', '21', '23', '30', '31', '6'], ['1', '12', '18', '22', '26', '32', '4', '5', '7', '9'], ['11', '13', '17', '20', '25', '27', '28', '29', '3', '8']]


split_dataset_according_list(h5_filename, train_list, test_list, save_path)

