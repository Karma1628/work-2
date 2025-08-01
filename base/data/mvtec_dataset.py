import os
import cv2
import glob
import random
import numpy as np
import imgaug.augmenters as iaa

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append('/home/gysj_cyc/workbench/base')

from data.perlin import create_perlin_mask

def read_rgb_image(path, size):
    image_bgr = cv2.imread(path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (size[0], size[1]))
    return image_rgb

class MVTecTrain(Dataset):
    '''
    Inputs:
        root_dir: root of mvtec
        class_name: the class to be trained
        resource_dir: root of extra dataset
        resize_shape: size of images

    Outputs:
        normal image: defect-free or good image
        pseudo image: pseudo defect on the normal image 
        mask: mask of pseudo or normal image
    '''
    def __init__(self, root_dir, resource_dir, resize_shape=256):
        
        print(os.path.basename(root_dir))

        self.image_paths = glob.glob(root_dir + '/train/good/*.png') 
        self.resource_paths = glob.glob(resource_dir + "/*/*.jpg")
        self.resize_shape = [resize_shape, resize_shape]
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        self.augmentations = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                              iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                              iaa.pillike.EnhanceSharpness(),
                              iaa.Solarize(0.5, threshold=(32,128)),
                              iaa.Posterize(),
                              iaa.Invert(),
                              iaa.pillike.Autocontrast(),
                              iaa.pillike.Equalize(),
                              iaa.Affine(rotate=(-45, 45))]
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        self.transform = transforms.Compose([transforms.ToPILImage(), 
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
        
    def __len__(self):
        return len(self.image_paths)
    
    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmentations)), 3, replace=False)
        aug = iaa.Sequential([self.augmentations[aug_ind[0]],
                              self.augmentations[aug_ind[1]],
                              self.augmentations[aug_ind[2]]])
        return aug
    
    def __getitem__(self, idx):

        normal_rgb_image = read_rgb_image(self.image_paths[idx], self.resize_shape) # np.uint8 (256, 256, 3)
        normal_lab_image = cv2.cvtColor(normal_rgb_image, cv2.COLOR_RGB2LAB)
        normal_mask = torch.zeros((self.resize_shape[0], self.resize_shape[1]), dtype=torch.long) # torch.float32 (1, 256, 256)
        
        # default rotation
        # if torch.rand(1).item() > 0.7:
        #     normal_rgb_image = self.rot(image=normal_rgb_image)
        
        k = random.randint(0, len(self.resource_paths)-1) # randomly choose the resource image
        aug = self.randAugmenter()
        resource_image = read_rgb_image(self.resource_paths[k], self.resize_shape) # np.uint8 (256, 256, 3)
        resource_image = aug(image=resource_image)

        pseudo_mask = create_perlin_mask(self.resize_shape) # binary np.float32 [256, 256, 1]
        pseudo_mask = self.rot(image=pseudo_mask) # rotate the mask

        beta = np.random.rand() * 0.8
        pseudo_rgb_image = normal_rgb_image * (1 - pseudo_mask) + (1 - beta) * resource_image * pseudo_mask + beta * normal_rgb_image * pseudo_mask # np.float64 [256, 256, 3]
        pseudo_rgb_image = np.uint8(pseudo_rgb_image)
        pseudo_lab_image = cv2.cvtColor(pseudo_rgb_image, cv2.COLOR_RGB2LAB)

        # path = '/home/gysj_cyc/workbench/base/data/'
        # cv2.imwrite(os.path.join(path, 'rgb_image.png'), pseudo_rgb_image)
        # cv2.imwrite(os.path.join(path, 'lab_image.png'), pseudo_lab_image)
        # cv2.imwrite(os.path.join(path, 'mask.png'), pseudo_mask * 255)
        # exit(0)

        if torch.rand(1).item() < 0.5:
            pseudo_rgb_image = self.transform(pseudo_rgb_image)
            pseudo_lab_image = self.transform(pseudo_lab_image)
            pseudo_mask = torch.from_numpy(pseudo_mask).squeeze().long()
        else:
            pseudo_rgb_image = self.transform(normal_rgb_image)
            pseudo_lab_image = self.transform(normal_lab_image)
            pseudo_mask = normal_mask

        return {'rgb_image': pseudo_rgb_image,
                'lab_image': pseudo_lab_image,
                'mask': pseudo_mask}

class MVTecTest(Dataset):
    '''
    Inputs:
        root_dir: root of mvtec
        class_name: the class to be trained
        resize_shape: size of images

    Outputs:
        image: image in test dataset
        mask: mask of image
    '''
    def __init__(self, root_dir, resize_shape=256):
        self.image_paths = glob.glob(root_dir + '/test/*/*.png')
        self.mask_path = root_dir + '/ground_truth/'
        self.resize_shape = [resize_shape, resize_shape]
        self.transform = transforms.Compose([transforms.ToPILImage(), 
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        rgb_image = read_rgb_image(image_path, self.resize_shape)
        lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
        
        rgb_image = self.transform(rgb_image)
        lab_image = self.transform(lab_image)
        
        image_name = os.path.basename(image_path) # 000.png
        image_dir = os.path.basename(os.path.dirname(image_path)) # thread or good
        if image_dir == 'good':
            mask = np.zeros((1, self.resize_shape[0], self.resize_shape[1]), dtype=np.float32)
        else:
            mask_name = image_name[:-4] + '_mask' + image_name[-4:]
            mask_path = self.mask_path + image_dir + '/' + mask_name
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.resize_shape[0], self.resize_shape[1])).astype(np.float32) / 255.0
            mask = mask[None, ...]

        return {'image_path': self.image_paths[idx], 
                'rgb_image': rgb_image,
                'lab_image': lab_image,
                'mask': mask}



if __name__ == '__main__':
    
    # path = '/home/gysj_cyc/workbench/base/data/006.png'
    # lab_image = read_lab_image(path, (256,256))
    # cv2.imwrite('/home/gysj_cyc/workbench/base/data/lab_image.png', lab_image)
    # exit(0)
    

    train_dataset = MVTecTrain(
        root_dir = '/home/gysj_cyc/workbench/data/mvtec/carpet',
        resource_dir = '/home/gysj_cyc/workbench/data/dtd/images'
    )

    test_dataset = MVTecTest(
        root_dir = '/home/gysj_cyc/workbench/data/mvtec/carpet',
    )

    # print(dataset[0])
    # exit(0)

    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=8, 
                                  shuffle=True, 
                                  num_workers=16)
    
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=8, 
                                 shuffle=True, 
                                 num_workers=16)
    
    for data in train_dataloader:
        rgb_image = data['rgb_image']
        lab_image = data['lab_image']
        mask = data['mask']
        print(rgb_image.shape, type(rgb_image), torch.max(rgb_image), torch.min(rgb_image)) # [8, 3, 256, 256]
        print(lab_image.shape, type(lab_image), torch.max(lab_image), torch.min(lab_image)) # [8, 3, 256, 256]
        print(mask.shape, type(mask), torch.max(mask), torch.min(mask)) # [8, 256, 256]
        print('-' * 100)

    '''
    torch.Size([8, 3, 256, 256]) <class 'torch.Tensor'> tensor(2.0609) tensor(-1.9638)
    torch.Size([8, 3, 256, 256]) <class 'torch.Tensor'> tensor(2.4308) tensor(-1.9980)
    '''

    for data in test_dataloader:
        image_path = data['image_path']
        rgb_image = data['rgb_image']
        lab_image = data['lab_image']
        mask = data['mask']
        print(image_path)
        print(rgb_image.shape, type(rgb_image), torch.max(rgb_image), torch.min(rgb_image))
        print(lab_image.shape, type(lab_image), torch.max(lab_image), torch.min(lab_image))
        print(mask.shape, type(mask), torch.max(mask), torch.min(mask))
        print('-' * 100)
    
    '''
    torch.Size([8, 3, 256, 256]) <class 'torch.Tensor'> tensor(2.2535) tensor(-1.8953)
    torch.Size([5, 3, 256, 256]) <class 'torch.Tensor'> tensor(2.3960) tensor(-2.1179)
    torch.Size([8, 1, 256, 256]) <class 'torch.Tensor'> tensor(1.) tensor(0.)
    '''