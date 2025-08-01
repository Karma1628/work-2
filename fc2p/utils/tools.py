import os
import torch
from torchvision import transforms

def add_noise(inputs):
    '''
        add gaussian noise [B,C,H,W]
    '''
    noise = torch.randn_like(inputs).to(inputs.device)
    inputs_noise = inputs + noise
    return inputs_noise

def l2_normalize(input, dim=1, eps=1e-12):
    denom = torch.sqrt(torch.sum(input**2, dim=dim, keepdim=True))
    return input / (denom + eps)

# reverse normalize and convert to PIL
# if normalize and want to display normalize image
# c h w -> c h w
def unnormalize(img):
    reverse_transform = transforms.Compose([
        transforms.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
        transforms.ToPILImage()])
    # mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    # std = torch.tensor([0.229, 0.256, 0.225], dtype=torch.float32)
    # normalize = transforms.Normalize(mean.tolist(), std.tolist())
    # unnormalize_trans = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    return reverse_transform(img)

def fusion(x1, x2):
    # return torch.cat((x1, x2), dim=1)
    return torch.pow(x1 - x2, 2)

