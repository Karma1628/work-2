import os
import cv2
import glob
import random
import numpy as np
import imgaug.augmenters as iaa

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from data.perlin import *
# from perlin import *

def read_RGB_image(path, size):
    image_bgr = cv2.imread(path)
    # image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_bgr, (size[0], size[1]))
    return image_rgb

class BTAD_Train(Dataset):

    def __init__(self,
                 root_dir,
                 resource_dir=None,
                 resize_shape=256):
        
        self.image_paths = glob.glob(root_dir + '/train/ok/*.bmp')
        self.resource_paths = glob.glob(resource_dir + '/*/*.jpg')
        self.resize_shape = [resize_shape, resize_shape]
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
        normal_image = read_RGB_image(self.image_paths[idx], self.resize_shape) # np.uint8 (256, 256, 3)
        normal_mask = torch.zeros((self.resize_shape[0], self.resize_shape[1]), dtype=torch.long) # torch.float32 (1, 256, 256)
                
        k = random.randint(0, len(self.resource_paths)-1) # randomly choose the resource image
        aug = self.randAugmenter()
        resource_image = read_RGB_image(self.resource_paths[k], self.resize_shape) # np.uint8 (256, 256, 3)
        resource_image = aug(image=resource_image)

        pseudo_mask = generate_perlin_noise(self.resize_shape) # binary np.float32 [256, 256, 1]
        pseudo_mask = self.rot(image=pseudo_mask) # rotate the mask

        beta = np.random.rand() * 0.8
        # np.float64 [256, 256, 3]
        pseudo_image = normal_image * (1 - pseudo_mask) + (1 - beta) * resource_image * pseudo_mask + beta * normal_image * pseudo_mask
        # cv2.imwrite('/home/karma1729/Desktop/My_DRAEM/temp/normal_image.png', normal_image)
        # cv2.imwrite('/home/karma1729/Desktop/My_DRAEM/temp/pseudo_mask.png', pseudo_mask * 255)
        # cv2.imwrite('/home/karma1729/Desktop/My_DRAEM/temp/pseudo_image.png', pseudo_image.astype(np.uint8))
        # exit(0)

        normal_image = self.transform(normal_image)

        if torch.rand(1).item() < 0.5:
            pseudo_image = self.transform(pseudo_image.astype(np.uint8))
            mask = torch.from_numpy(pseudo_mask).squeeze().long()
        else:
            pseudo_image = normal_image
            mask = normal_mask

        return {'normal_image': normal_image,
                'pseudo_image': pseudo_image,
                'mask': mask}
    
class BTAD_test(Dataset):
    '''
    Inputs:
        root_dir: root of mvtec
        class_name: the class to be trained
        resize_shape: size of images

    Outputs:
        image: image in test dataset
        mask: mask of image
    '''
    def __init__(self, 
                 root_dir=None, 
                 resize_shape=256):
        self.image_paths = glob.glob(root_dir + '/test/*/*.bmp')
        self.mask_path = root_dir + '/ground_truth/ko/'
        self.resize_shape = [resize_shape, resize_shape]
        self.transform = transforms.Compose([transforms.ToPILImage(), 
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = read_RGB_image(image_path, self.resize_shape)
        image = self.transform(image)
        image_name = os.path.basename(image_path) # 0000.bmp
        image_dir = os.path.basename(os.path.dirname(image_path)) # ok or ko
        if image_dir == 'ok':
            mask = np.zeros((1, self.resize_shape[0], self.resize_shape[1]), dtype=np.float32)
        else:
            mask_name = image_name[:-4] + '.png'
            mask_path = self.mask_path + '/' + mask_name
            print(mask_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.resize_shape[0], self.resize_shape[1])).astype(np.float32) / 255.0
            mask = mask[None, ...]

        return {'image_path': self.image_paths[idx], 
                'image':image, 
                'mask':mask}


if __name__ == '__main__':

    # dataset = BTAD_Train(
    #     root_dir='/home/karma1729/Desktop/data/BTAD/01/',
    #     resource_dir='/home/karma1729/Desktop/data/dtd/images/',
    # )

    dataset = BTAD_test(
        root_dir = '/home/karma1729/Desktop/data/BTAD/01/',
    )
    
    dataloader = DataLoader(dataset, 
                            batch_size=8, 
                            shuffle=True, 
                            num_workers=16)
    # print(len(dataset))
    # print(dataset[0])
    # exit(0)
    
    # Train 
    # for data in dataloader:
    #     normal_image = data['normal_image']
    #     pseudo_image = data['pseudo_image']
    #     mask = data['mask']
    #     print(normal_image.shape, type(normal_image), torch.max(normal_image), torch.min(normal_image)) # [8, 3, 256, 256]
    #     print(pseudo_image.shape, type(pseudo_image), torch.max(pseudo_image), torch.min(pseudo_image))
    #     print(mask.shape, type(mask), torch.max(mask), torch.min(mask))
    #     print('-' * 100)

    # Test
    for data in dataloader:
        image_path = data['image_path']
        image = data['image']
        mask = data['mask']
        print(image_path)
        print(image.shape, type(image), torch.max(image), torch.min(image))
        print(mask.shape, type(mask), torch.max(mask), torch.min(mask))
        print('-' * 100)