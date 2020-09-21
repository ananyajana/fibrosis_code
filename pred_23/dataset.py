import torch.utils.data as data
from PIL import Image
import h5py
import numpy as np
import torch
from skimage.util import random_noise
from skimage.transform import rotate
from skimage import exposure
from glob import glob
from random import randint
import random
from skimage.transform import resize
import math
from scipy.ndimage import gaussian_filter
import scipy.ndimage as ndimage

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
        #print(self.keys[index])
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

from skimage.feature import local_binary_pattern
def h5_loader2(data, opt=None):
    ct_data = data['CT']
    fib_score = data['Fibrosis'][()]
    nas_stea_score = data['Steatosis'][()]
    nas_lob_score = data['Lobular'][()]
    nas_balloon_score = data['Ballooning'][()]

    ct_imgs = []
    for key in ct_data.keys():
        img0 = ct_data[key][()]

        radius = 1
        n_points = radius * 8
        METHOD = 'uniform'
        img = local_binary_pattern(img0, n_points, radius, METHOD)
        
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

    
class LiverDataset2(data.Dataset):
    def __init__(self, hdf5_path, data_transform, opt=None):
        super(LiverDataset2, self).__init__()
        self.hdf5_path = hdf5_path
        self.data_transform = data_transform
        self.keys = get_keys(self.hdf5_path)
        self.opt = opt

    def __getitem__(self, index):
        #print(self.keys[index])
        hdf5_file = h5py.File(self.hdf5_path, "r")
        slide_data = hdf5_file[self.keys[index]]
        ct_imgs, fib_label, nas_stea_label, nas_lob_label, nas_balloon_label = h5_loader2(slide_data, self.opt)
        ct_tensor = []
        for i in range(len(ct_imgs)):
            ct_tensor.append(self.data_transform(ct_imgs[i]).unsqueeze(0))

        return torch.cat(ct_tensor, dim=0), \
               torch.tensor(fib_label).unsqueeze(0).long(), torch.tensor(nas_stea_label).unsqueeze(0).long(), \
               torch.tensor(nas_lob_label).unsqueeze(0).long(), torch.tensor(nas_balloon_label).unsqueeze(0).long()

    def __len__(self):
        return len(self.keys)
    
from skimage.feature import local_binary_pattern
def h5_loader3(data, opt=None):
    ct_data = data['CT']
    fib_score = data['Fibrosis'][()]
    nas_stea_score = data['Steatosis'][()]
    nas_lob_score = data['Lobular'][()]
    nas_balloon_score = data['Ballooning'][()]

    ct_imgs = []
    for key in ct_data.keys():
        img0 = ct_data[key][()]
        img1 = np.array(img0)

        img = ndimage.morphology.distance_transform_edt(img1)
        
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

class LiverDataset3(data.Dataset):
    def __init__(self, hdf5_path, data_transform, opt=None):
        super(LiverDataset3, self).__init__()
        self.hdf5_path = hdf5_path
        self.data_transform = data_transform
        self.keys = get_keys(self.hdf5_path)
        self.opt = opt

    def __getitem__(self, index):
        #print(self.keys[index])
        hdf5_file = h5py.File(self.hdf5_path, "r")
        slide_data = hdf5_file[self.keys[index]]
        ct_imgs, fib_label, nas_stea_label, nas_lob_label, nas_balloon_label = h5_loader3(slide_data, self.opt)
        ct_tensor = []
        for i in range(len(ct_imgs)):
            ct_tensor.append(self.data_transform(ct_imgs[i]).unsqueeze(0))

        return torch.cat(ct_tensor, dim=0), \
               torch.tensor(fib_label).unsqueeze(0).long(), torch.tensor(nas_stea_label).unsqueeze(0).long(), \
               torch.tensor(nas_lob_label).unsqueeze(0).long(), torch.tensor(nas_balloon_label).unsqueeze(0).long()

    def __len__(self):
        return len(self.keys)

def h5_loader4(data, opt=None):
    ct_data = data['CT']
    fib_score = data['Fibrosis'][()]
    nas_stea_score = data['Steatosis'][()]
    nas_lob_score = data['Lobular'][()]
    nas_balloon_score = data['Ballooning'][()]

    ct_imgs = []
    for key in ct_data.keys():
        img0 = ct_data[key][()]
        img1 = np.array(img0)

        radius = 1
        n_points = radius * 8
        METHOD = 'uniform'
        img = local_binary_pattern(img0, n_points, radius, METHOD)

        img = ndimage.morphology.distance_transform_edt(img1)
        
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

class LiverDataset4(data.Dataset):
    def __init__(self, hdf5_path, data_transform, opt=None):
        super(LiverDataset3, self).__init__()
        self.hdf5_path = hdf5_path
        self.data_transform = data_transform
        self.keys = get_keys(self.hdf5_path)
        self.opt = opt

    def __getitem__(self, index):
        #print(self.keys[index])
        hdf5_file = h5py.File(self.hdf5_path, "r")
        slide_data = hdf5_file[self.keys[index]]
        ct_imgs, fib_label, nas_stea_label, nas_lob_label, nas_balloon_label = h5_loader4(slide_data, self.opt)
        ct_tensor = []
        for i in range(len(ct_imgs)):
            ct_tensor.append(self.data_transform(ct_imgs[i]).unsqueeze(0))

        return torch.cat(ct_tensor, dim=0), \
               torch.tensor(fib_label).unsqueeze(0).long(), torch.tensor(nas_stea_label).unsqueeze(0).long(), \
               torch.tensor(nas_lob_label).unsqueeze(0).long(), torch.tensor(nas_balloon_label).unsqueeze(0).long()

    def __len__(self):
        return len(self.keys)

class PretrainLiverDataset2(data.Dataset):
    def __init__(self, path, patch_sz, num_swap, transform, random_mirror=False, scale=False):
        super(PretrainLiverDataset2, self).__init__()
        self.path = path
        self.transform = transform
        self.files = glob('{:s}*'.format(self.path))
        self.patch_sz = patch_sz
        self.length = None
        self.num_swap = num_swap
        self.random_mirror = random_mirror
        self.scale=scale
 

    # this funciton will return the original image as well as the context restore version
    def __getitem__(self, index):
        img0 = Image.open(self.files[index])
        img = np.array(img0)
        orig_img = img.copy()
        if self.random_mirror is True:
            flip_num = randint(0, 7)
            if flip_num == 1:
                img = np.flipud(img)
                orig_img = np.flipud(orig_img)
            elif flip_num == 2:
                img = np.fliplr(img)
                orig_img = np.fliplr(orig_img)
            elif flip_num == 3:
                img = np.rot90(img, k=1, axes=(1, 0))
                orig_img = np.rot90(orig_img, k=1, axes=(1, 0))
            elif flip_num == 4:
                img = np.rot90(img, k=3, axes=(1, 0))
                orig_img = np.rot90(orig_img, k=3, axes=(1, 0))
            elif flip_num == 5:
                img = np.fliplr(img)
                orig_img = np.fliplr(orig_img)
                img = np.rot90(img, k=1, axes=(1, 0))
                orig_img = np.rot90(orig_img, k=1, axes=(1, 0))
            elif flip_num == 6:
                img = np.fliplr(img)
                orig_img = np.fliplr(orig_img)
                img = np.rot90(img, k=3, axes=(1, 0))
                orig_img = np.rot90(orig_img, k=3, axes=(1, 0))
            elif flip_num == 7:
                img = np.flipud(img)
                orig_img = np.flipud(orig_img)
                img = np.fliplr(img)
                orig_img = np.fliplr(orig_img)
        if self.scale is True:
            #  randomly scale
            scale = np.random.uniform(0.8,1.2)
            h,w = img.shape
            h = int(h * scale)
            w = int(w * scale)
            img = resize(img, (h, w), order=3, mode='edge', cval=0, clip=True, preserve_range=True)
            orig_img = resize(orig_img, (h, w), order=3, mode='edge', cval=0, clip=True, preserve_range=True)
        
        patch_area = self.patch_sz * self.patch_sz
        img_sz = img.shape[0]
        self.length = img_sz - self.patch_sz
        for _ in range(self.num_swap):
            while(True):
                k=random.randint(0, self.length)
                l=random.randint(0, self.length)
                m=random.randint(0, self.length)
                n=random.randint(0, self.length)
                    
                # check the mask, because if both the patches are blank it will not make any sense
                # check the two patches from the segmentation image
                p1_seg = img[k:k+self.patch_sz, l:l+self.patch_sz].copy()
                p2_seg = img[m:m+self.patch_sz, n:n+self.patch_sz].copy()
                # count of black pixels,
                blank_pix_count = np.sum(p1_seg == 0) + np.sum(p2_seg == 0)
                #print('blank pix count: {}, k:{}, l:{}, m:{}, n:{}'.format(blank_pix_count, k, l, m, n))
                #if (blank_pix_count > int(2 * patch_area * 0.1)) and (blank_pix_count <= int(2 * patch_area * 0.95)):
                if (blank_pix_count <= int(2 * patch_area * 0.95)):
                    break

            #print('{}. img shape: {}'.format(self.files[index], img.shape))
            #plt.imshow(img)
            p1 = img[k:k+self.patch_sz, l:l+self.patch_sz].copy()
            p2 = img[m:m+self.patch_sz, n:n+self.patch_sz].copy()

            img[m:m+self.patch_sz, n:n+self.patch_sz] = p1
            img[k:k+self.patch_sz, l:l+self.patch_sz] = p2
        
        img0.close()
        # convert intoa proper shape so that it can be converted to PIL image
        img = self.transform(img.reshape(img.shape[0], img.shape[1], -1))
        orig_img = self.transform(orig_img.reshape(orig_img.shape[0], orig_img.shape[1], -1))
      
        return orig_img, img 


    def __len__(self):
        return len(self.files)
        

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

# this dataset returns an image and its downsamples version
class PretrainLiverDataset4(data.Dataset):
    def __init__(self, path, transform, num_class, is_resnet=False):
        super(PretrainLiverDataset4, self).__init__()
        self.path = path
        self.transform = transform
        self.num_class = num_class
        self.files = glob('{:s}*'.format(self.path))
        self.is_resnet = is_resnet


    def __getitem__(self, index):
        img0 = Image.open(self.files[index])
        img = np.array(img0)
        w, h = img.shape
        # anyway the 512x512 images will be downsized to 22 which is a factor of 0.5
        # hence to make the difference significant we donsize the corrup image bya  afactor of 0.25
        w_, h_ = math.floor(w/4), math.floor(h/4)
        keys = np.arange(self.num_class)
        random.shuffle(keys)
        rotated_imgs = []
        for i in range(len(keys)):
            key = keys[i]
            if key == 0:
                temp_im = img
            else:                
                new_img0 = img0.resize((w_, h_))
                temp_im = np.array(new_img0.resize((w, h)))
                
            if self.is_resnet is True and len(temp_im.shape) == 2:
                temp_im = temp_im[:, :, np.newaxis]
                temp_im = np.repeat(temp_im, 3, axis=2)
                #ct_imgs.append(Image.fromarray(img.astype('uint8'), 'RGB'))
            rotated_imgs.append(self.transform(temp_im.reshape(temp_im.shape[0], temp_im.shape[1], -1)))
                

        rotation_labels = torch.LongTensor(keys)
        
        return torch.stack(rotated_imgs, dim=0), rotation_labels

    def __len__(self):
        return len(self.files)
    
# this dataset returns an image and its downsamples version
class PretrainLiverDataset5(data.Dataset):
    def __init__(self, path, transform, num_class, is_resnet=False):
        super(PretrainLiverDataset5, self).__init__()
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
                temp_im = np.array(gaussian_filter(img0, sigma=5))
                
            if self.is_resnet is True and len(temp_im.shape) == 2:
                temp_im = temp_im[:, :, np.newaxis]
                temp_im = np.repeat(temp_im, 3, axis=2)
                #ct_imgs.append(Image.fromarray(img.astype('uint8'), 'RGB'))
            rotated_imgs.append(self.transform(temp_im.reshape(temp_im.shape[0], temp_im.shape[1], -1)))
                

        rotation_labels = torch.LongTensor(keys)
        
        return torch.stack(rotated_imgs, dim=0), rotation_labels

    def __len__(self):
        return len(self.files)
