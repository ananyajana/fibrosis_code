import torch.utils.data as data
from PIL import Image
import h5py
import numpy as np
import torch
from skimage.transform import rotate
from glob import glob
from random import randint
import random
import math
import torchvision.transforms as transforms

def is_hdf5_file(filename):
    return filename.lower().endswith('.h5')


def get_keys(hdf5_path):
    with h5py.File(hdf5_path, 'r') as file:
        return list(file.keys())


def h5_loader(data, opt=None):
    ct_data = data['CT']
    fib_score = data['Fibrosis'][()]
    nas_stea_score = data['Steatosis'][()]
    nas_lob_score = data['Lobular'][()]
    nas_balloon_score = data['Ballooning'][()]

    ct_imgs = []
    for key in ct_data.keys():
        img = ct_data[key][()]
        if opt is not None and opt.model['use_resnet'] == 1:
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]
                img = np.repeat(img, 3, axis=2)
            ct_imgs.append(Image.fromarray(img.astype('uint8'), 'RGB'))
        else:
            ct_imgs.append(Image.fromarray(img.astype('uint8'), 'L'))

    if fib_score == 0:  # 0: 0
        fib_label = 0
    elif fib_score < 3:  # 1: [1, 2, 2.5]
        fib_label = 1
    else:               # 2: [3, 3.5, 4]
        fib_label = 2

    nas_stea_label = 0 if nas_stea_score < 2 else 1
    nas_lob_label = nas_lob_score if nas_lob_score < 2 else 2
    # nas_lob_label = 0 if nas_lob_score < 2 else 1
    nas_balloon_label = nas_balloon_score

    return ct_imgs, fib_label, nas_stea_label, nas_lob_label, nas_balloon_label


class LiverDataset(data.Dataset):
    def __init__(self, hdf5_path, data_transform, opt=None):
        super(LiverDataset, self).__init__()
        self.hdf5_path = hdf5_path
        self.data_transform = data_transform
        self.keys = get_keys(self.hdf5_path)
        self.opt = opt

    def __getitem__(self, index):
        hdf5_file = h5py.File(self.hdf5_path, "r")
        slide_data = hdf5_file[self.keys[index]]
        ct_imgs, fib_label, nas_stea_label, nas_lob_label, nas_balloon_label = h5_loader(slide_data, self.opt)
        ct_tensor = []
        for i in range(len(ct_imgs)):
            ct_tensor.append(self.data_transform(ct_imgs[i]).unsqueeze(0))

        return torch.cat(ct_tensor, dim=0), \
               torch.tensor(fib_label).unsqueeze(0).long(), torch.tensor(nas_stea_label).unsqueeze(0).long(), \
               torch.tensor(nas_lob_label).unsqueeze(0).long(), torch.tensor(nas_balloon_label).unsqueeze(0).long()

    def __len__(self):
        return len(self.keys)


def my_collate_fn(batch):
    data_list = []
    labels_list = []
    for i in range(len(batch)):
        data_list.append(batch[i][0])
        labels_list.append(batch[i][1])
        
    return torch.cat(data_list, dim=0), torch.cat(labels_list, dim=0).long()


class PretrainLiverDataset3(data.Dataset):
    def __init__(self, path, transform, num_class, is_resnet=False):
        super(PretrainLiverDataset3, self).__init__()
        self.path = path
        self.transform = transform
        self.num_class = num_class
        self.files = glob('{:s}*'.format(self.path))
        self.is_resnet = is_resnet


    def __getitem__(self, index):
        img0 = Image.open(self.files[index])
        img = np.array(img0)
        keys = np.arange(self.num_class)
        random.shuffle(keys)
        rotated_imgs = []
        for i in range(len(keys)):
            key = keys[i]
            if key == 0:
                temp_im = img
            else:
                temp_im = np.array(img0.rotate(90 * key))
            
            if self.is_resnet is True and len(temp_im.shape) == 2:
                temp_im = temp_im[:, :, np.newaxis]
                temp_im = np.repeat(temp_im, 3, axis=2)
                #ct_imgs.append(Image.fromarray(img.astype('uint8'), 'RGB'))
            rotated_imgs.append(self.transform(temp_im.reshape(temp_im.shape[0], temp_im.shape[1], -1)))
                

        rotation_labels = torch.LongTensor(keys)
        
        return torch.stack(rotated_imgs, dim=0), rotation_labels

    def __len__(self):
        return len(self.files)
