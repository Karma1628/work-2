import torch
import torch.nn as nn

import sys
sys.path.append('/home/gysj_cyc/workbench/base_hybrid')
import time

from model.extractor import TimmFeatureExtractor
from model.reconstructor import FeatureReconstructor
from model.segmentor import DiscriminativeSubNetwork
from utils import *


class MyModel(nn.Module):

    def __init__(self, 
                 weights_name='vit_small_patch14_dinov2',
                 weights_path='/home/gysj_cyc/workbench/ckpts/',
                 image_size=256,
                 feature_size=64,
                 ch_base=256, 
                 stages=[11],
                 ):
        super(MyModel, self).__init__()

        self.dino_feature_extractor = TimmFeatureExtractor(
            weights_name='vit_small_patch14_dinov2',
            weights_path=weights_path+'dinov2_vits14_pretrain.pth',
            pretrained=True,
            image_size=image_size,
            feature_size=feature_size,
            stages=stages)
        
        self.swin_feature_extractor = TimmFeatureExtractor(
            weights_name='swin_large_patch4_window12_384',
            weights_path=weights_path+'swin_large_patch4_window12_384_22kto1k.pth',
            pretrained=True,
            image_size=256,
            feature_size=feature_size,
            stages=[1])
        
        self.stage_channels = self.dino_feature_extractor.count() # 192, 384, 768
        self.target_channels = sum(self.stage_channels)
        
        self.rgb2lab = FeatureReconstructor(ch_in=self.target_channels, ch_base=ch_base)
        self.lab2rgb = FeatureReconstructor(ch_in=self.target_channels, ch_base=ch_base)
        self.feature_segmentor = DiscriminativeSubNetwork(in_channels=self.target_channels, out_channels=2)

    def forward(self, rgb_img, lab_img):

        #*------------------------------------ Feature Extraction ------------------------------------*#
        # start_time = time.time()
        
        with torch.no_grad():
            rgb_feat = self.dino_feature_extractor(rgb_img)
            lab_feat = self.dino_feature_extractor(lab_img)
            
            rgb_img = F.interpolate(rgb_img, size=256, mode='bilinear', align_corners=True)
            lab_img = F.interpolate(lab_img, size=256, mode='bilinear', align_corners=True)
            
            rgb_swin_feat = self.swin_feature_extractor(rgb_img)
            lab_swin_feat = self.swin_feature_extractor(lab_img)
            
            rgb_feat = rgb_feat + rgb_swin_feat
            lab_feat = lab_feat + lab_swin_feat
            
        # print(time.time() - start_time)   

        #*---------------------------------- Feature Reconstruction ----------------------------------*#
        # start_time = time.time()
        
        lab_feat_recon = self.rgb2lab(rgb_feat)
        rgb_feat_recon = self.lab2rgb(lab_feat)
        
        # print(time.time() - start_time)   
        #*-------------------------------------- Residual Feature --------------------------------------*#
        # start_time = time.time()
        
        rgb_feat_res = (rgb_feat_recon - rgb_feat) ** 2
        lab_feat_res = (lab_feat_recon - lab_feat) ** 2
        
        # print(time.time() - start_time)
        #*--------------------------------- Residual Feature Fusion ---------------------------------*#
        # start_time = time.time()
        
        fusion_feat = rgb_feat_res + lab_feat_res
        
        # print(time.time() - start_time)
        #*--------------------------------- Fusion Feature Segmentation---------------------------------*#
        # start_time = time.time()
        
        pred = self.feature_segmentor(fusion_feat)
        pred = torch.sigmoid(pred)
        
        # print(time.time() - start_time)
        #*------------------------------------ Output the Results ------------------------------------*#

        return rgb_feat_res, lab_feat_res, pred

if __name__ == "__main__":

    x = torch.randn(4, 3, 256, 256).to('cuda:0')
    mask = torch.randn(4, 64, 64).to('cuda:0')
    net = MyModel(ch_base=256, stages=[11]).to('cuda:0')
    net.train()
    out = net(x, x)
    for i in out:
        print(i.shape)