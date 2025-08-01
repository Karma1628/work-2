import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
from timm.models.vision_transformer import _cfg

class TimmFeatureExtractor(nn.Module):
    
    def __init__(self, 
                 weights_name="vit_small_patch14_dinov2", 
                 weights_path='/home/gysj_cyc/workbench/base/ckpts/dinov2_vits14_pretrain.pth', 
                 pretrained=True,
                 image_size=256,
                 feature_size=64, 
                 stages=[0, 1, 2]):
        super(TimmFeatureExtractor, self).__init__()
        
        
        self.model = timm.create_model(
            weights_name,
            img_size=image_size,
            pretrained=pretrained,
            features_only=True,
            pretrained_cfg_overlay=dict(file=weights_path),
            out_indices=stages,
        )
        
        self.image_size = image_size
        self.feature_size = feature_size
        self.stages = stages

        for param in self.model.parameters():
            param.requires_grad = False

    def count(self):
        fake_batch = torch.randn(1, 3, self.image_size, self.image_size).to(next(self.model.parameters()).device)
        fake_features = self.model(fake_batch)
        channels = []
        for fake_feat in fake_features:
            if fake_feat.shape[2] != fake_feat.shape[3]:
                channels.append(fake_feat.shape[3])
            else:
                channels.append(fake_feat.shape[1])
        return channels

    def forward(self, x):
        features = self.model(x) # tuple 
        resized_features = []
        for feat in features:
            if feat.shape[2] != feat.shape[3]:
                feat = feat.permute(0, 3, 1, 2).contiguous()
            resized_feat = F.interpolate(feat, size=self.feature_size, mode='bilinear', align_corners=True)
            resized_features.append(resized_feat)
        resized_features = torch.cat(resized_features, dim=1)
        return resized_features

if __name__ == '__main__':

    # import timm
    # timm_models = timm.list_models(pretrained=True)
    # for model in timm_models:
    #     if 'dinov2' in model:
    #         print(model)

    '''
    4, 3, 256, 256
    for the input size of 256x256:
        torch.Size([4, 192, 64, 64])
        torch.Size([4, 384, 32, 32])
        torch.Size([4, 768, 16, 16])
        torch.Size([4, 1536, 8, 8])
    '''

    '''
        swin-base_in21k-pre-3rdparty_in1k
        swin-large_in21k-pre-3rdparty_in1k
        wide-resnet50_3rdparty_8xb32_in1k
    '''

    # inputs = torch.randn(4, 3, 256, 256).to('cuda:0')
    # FESwin = FeatureExtractor(weights='swin-large_in21k-pre-3rdparty_in1k').to('cuda:0')
    # outputs = FESwin(inputs)
    # print(outputs.shape, outputs.requires_grad)
    # print(FESwin.count())
    
    image_size = 256
    inputs = torch.randn(4, 3, image_size, image_size).to('cuda:0')
    net = TimmFeatureExtractor(
        # weights_name="vit_small_patch14_dinov2",
        weights_name='swin_large_patch4_window12_384',
        weights_path='/home/gysj_cyc/workbench/ckpts/swin_large_patch4_window12_384_22kto1k.pth',
        pretrained=True,
        image_size=image_size,
        feature_size=64,
        stages=[0, 1, 2]).to('cuda:0')
    print(net.count())
    out = net(inputs)
    print(out.shape)
    
    # model = get_model('vit-small-p14_dinov2-pre_3rdparty', pretrained=True, backbone=dict(out_indices=[0,1,2,3,4,5,6,7,8,9,10,11]))
    # print(model)
    # x = torch.randn(4, 3, 256, 256)
    # out = model.extract_feat(x, stage='backbone')
    # for i in out:
    #     print(i.shape)
    # import timm
    # model = timm.create_model('vit_small_patch14_dinov2', pretrained=True, img_size=512, features_only=True, out_indices=(-3, -2,))
    # output = model(torch.randn(2, 3, 512, 512))

    # for o in output:    
    #     print(o.shape)   

    
