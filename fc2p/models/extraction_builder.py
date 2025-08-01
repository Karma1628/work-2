import torch
import torch.nn as nn

from mmpretrain import get_model

class FeatureExtractor(nn.Module):
    
    def __init__(self, weights=None, pre=True, device='cpu', stages=[0, 1]):
        super(FeatureExtractor, self).__init__()
        '''
            swin-base_in21k-pre-3rdparty_in1k
            swin-large_in21k-pre-3rdparty_in1k

            wide-resnet50_3rdparty_8xb32_in1k
        '''
        self.model = get_model(weights, 
                               pretrained=pre, 
                               device=device, 
                               backbone=dict(out_indices=(0, 1, 2, 3)))
        self.stages = stages
        
        for param in self.model.parameters():
            param.requires_grad = False

        # self.pad = nn.ReflectionPad2d(padding=1)
        
    def forward(self, x):
        '''
        4, 3, 256, 256
        for the input size of 256x256:
            torch.Size([4, 192, 64, 64])
            torch.Size([4, 384, 32, 32])
            torch.Size([4, 768, 16, 16])
            torch.Size([4, 1536, 8, 8])
        '''
        features = self.model.extract_feat(x, stage='backbone') # tuple 
        max_size = features[self.stages[0]].shape[2] # max size of choosen stage of feature
        resized_features = [nn.functional.interpolate(features[idx], 
                                                      size=max_size, 
                                                      mode='bilinear', 
                                                      align_corners=True) for idx in self.stages]
        features = torch.cat(resized_features, dim=1)
        # features = self.pad(f)
        # image = nn.functional.interpolate(x, size=max_size, mode='bilinear', align_corners=True)
        # features = torch.cat([features, image], dim=1)
        return features
    
class Extractor(nn.Module):
    
    def __init__(self, weights=None, pre=True, device='cpu', stages=[0, 1], size=256):
        super(Extractor, self).__init__()
        '''
            swin-base_in21k-pre-3rdparty_in1k
            swin-large_in21k-pre-3rdparty_in1k

            wide-resnet50_3rdparty_8xb32_in1k
        '''
        self.model = get_model(weights, 
                               pretrained=pre, 
                               device=device, 
                               backbone=dict(out_indices=(0, 1, 2, 3)))
        self.stages = stages
        self.size = size
        self.device = device
        self.channels = self.get_channels()
        
        for param in self.model.parameters():
            param.requires_grad = False

        # self.pad = nn.ReflectionPad2d(padding=1)

    def get_channels(self):
        fake_data = torch.randn(2, 3, self.size, self.size).to(self.device)
        fake_features = self.model.extract_feat(fake_data, stage='backbone')
        channels = 0
        for f in self.stages:
            channels += fake_features[f].shape[1]
        return channels
        
    def forward(self, x):
        '''
        4, 3, 256, 256
        for the input size of 256x256:
            torch.Size([4, 192, 64, 64])
            torch.Size([4, 384, 32, 32])
            torch.Size([4, 768, 16, 16])
            torch.Size([4, 1536, 8, 8])
        '''
        features = self.model.extract_feat(x, stage='backbone') # tuple 
        max_size = features[self.stages[0]].shape[2] # max size of choosen stage of feature
        resized_features = [nn.functional.interpolate(features[idx], 
                                                      size=max_size, 
                                                      mode='bilinear', 
                                                      align_corners=True) for idx in self.stages]
        features = torch.cat(resized_features, dim=1)
        # features = self.pad(f)
        # image = nn.functional.interpolate(x, size=max_size, mode='bilinear', align_corners=True)
        # features = torch.cat([features, image], dim=1)
        return features

if __name__ == '__main__':

    import timm
    timm_models = timm.list_models(pretrained=True)
    for model in timm_models:
        if 'swin' in model:
            print(model)

    inputs = torch.randn(4, 3, 256, 256).to('cuda:0')
    FESwin = FeatureExtractor(weights='swin-large_in21k-pre-3rdparty_in1k', device='cuda:0')
    outputs = FESwin(inputs)
    print(outputs.shape, outputs.requires_grad)