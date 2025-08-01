import cv2
import os
import numpy as np
from PIL import Image

import torch

def convert2heatmap(x):
    """
        x: [h, w, 3] np.uin8
    """
    x = cv2.applyColorMap(x, cv2.COLORMAP_JET)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    return x

def float2uint8(x):
    """
        x: [h, w], (0, 1) np.float
    """
    x = np.repeat(x[..., np.newaxis], 3, 2)
    x = np.uint8(x * 255)
    return x 

def visualize_results(save_path, image_path, mask, feat_res, seg_predict, file_name,
                      heat_map=True, size=(256, 256), stage_channel=[192, 384, 758]):

    save_img_path = os.path.join(save_path, file_name)
    os.makedirs(save_img_path, exist_ok=True)
    display_pil_images = []

    image_pil = Image.open(image_path).resize(size)
    display_pil_images.append(image_pil)

    mask_np = float2uint8(mask)
    mask_pil = Image.fromarray(mask_np)
    display_pil_images.append(mask_pil)

    feat_res_stages = torch.split(feat_res, stage_channel, dim=1)
    for feat in feat_res_stages:
        feat = feat.sum(1).squeeze().detach().cpu().numpy()
        feat = (feat - np.min(feat)) / (np.max(feat) - np.min(feat))
        feat = float2uint8(feat)
        if heat_map:
            feat = convert2heatmap(feat)
        feat_pil = Image.fromarray(feat)
        display_pil_images.append(feat_pil)

    feat_res_np = feat_res.sum(1).squeeze().detach().cpu().numpy()
    feat_res_np = (feat_res_np - np.min(feat_res_np)) / (np.max(feat_res_np) - np.min(feat_res_np))
    feat_res_np = float2uint8(feat_res_np)
    if heat_map:
        feat_res_np = convert2heatmap(feat_res_np)
    feat_res_pil = Image.fromarray(feat_res_np)
    display_pil_images.append(feat_res_pil)
    
    seg_predict = (seg_predict - np.min(seg_predict)) / (np.max(seg_predict) - np.min(seg_predict))
    seg_predict = ((np.repeat(seg_predict[..., None], 3, axis=2)) * 255).astype(np.uint8)
    if heat_map:
        seg_predict = convert2heatmap(seg_predict)
        # img_np = np.array(image_pil)
        # seg_predict = cv2.addWeighted(img_np, 0.7, seg_predict, 0.3, 0)
    seg_predict_pil = Image.fromarray(seg_predict)
    display_pil_images.append(seg_predict_pil)

    num_images = len(display_pil_images)
    combination = Image.new('RGB', (size[0]*num_images+5*num_images, size[0]), (255, 255, 255))
    for idx, img in enumerate(display_pil_images):
        img = img.resize(size)
        combination.paste(img, (idx*(size[0] + 5), 0))
    save_name = '_'.join(image_path.split('/')[-2:])
    combination.save(os.path.join(save_img_path, save_name))

def visualize_feature_map(feature, idx, path, size):
    os.makedirs(path, exist_ok=True)
    feature = feature.detach().cpu().numpy()
    feature = (feature - np.min(feature)) / (np.max(feature) - np.min(feature)) * 255
    feature = float2uint8(feature)
    feature = cv2.resize(feature, dsize=(size, size))
    feature = cv2.applyColorMap(feature, cv2.COLORMAP_VIRIDIS)
    feature = cv2.cvtColor(feature, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(path, f'{idx}.jpg'), feature)

def feature_map_np(fm):
    fm = fm.detach().cpu().numpy()
    fm = (fm - np.min(fm)) / (np.max(fm) - np.min(fm))
    fm = np.uint8(fm * 255)
    fm = cv2.applyColorMap(fm, cv2.COLORMAP_VIRIDIS)
    # fm = cv2.cvtColor(fm, cv2.COLOR_BGR2RGB)
    return fm
