import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

import torch
import torch.nn as nn
from torchvision import transforms 
import torch.nn.functional as F

def setup_seed(seed):
     random.seed(seed)
     np.random.seed(seed)
     torch.manual_seed(seed)
     torch.cuda.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
    #  torch.backends.cudnn.deterministic = True
    #  torch.backends.cudnn.benchmark = False

def log_info(log_path, file_name, info):
    print(info)  # 打印日志信息到控制台
    with open(os.path.join(log_path, file_name), mode='a') as f:
        f.write(info + '\n')

def calculate_metrics(masks, score_maps):
    
    masks = np.array(masks).astype(np.int32)
    score_maps = np.array(score_maps, dtype=np.float32)

    score_maps_tensor = torch.tensor(score_maps, dtype=torch.float32).unsqueeze(0)
    pooled_score_maps = nn.functional.avg_pool2d(score_maps_tensor, 32, stride=1, padding=16).squeeze(0)
    
    image_score = pooled_score_maps.amax(dim=(1, 2)).numpy()
    image_label = masks.any(axis=(1, 2))
    image_auc = roc_auc_score(image_label, image_score)

    masks_flat = masks.ravel()
    score_maps_flat = score_maps.ravel()
    pixel_auc = roc_auc_score(masks_flat, score_maps_flat)
    pixel_ap = average_precision_score(masks_flat, score_maps_flat)

    return {'image_auc': image_auc,'pixel_auc': pixel_auc,'pixel_ap': pixel_ap}

def summary_results_to_excel(work_dir, result_name, data_name, cls, categories, metrics):
    excel_path = os.path.join(work_dir, result_name + '_' + data_name + '_results.xlsx')

    if not os.path.exists(excel_path):
        num_categories = len(categories) + 1
        DataExcel = {
            'category': categories + ['average'],
            'image_aucroc':list(range(num_categories)),
            'pixel_aucroc':list(range(num_categories)),
            'pixel_ap':list(range(num_categories))}
        df = pd.DataFrame(DataExcel)
        df.to_excel(excel_path, engine='openpyxl', index=False)

    df = pd.read_excel(excel_path)
    columns_to_update = ['image_aucroc', 'pixel_aucroc', 'pixel_ap']
    values_to_update = [metrics["image_auc"], metrics["pixel_auc"], metrics["pixel_ap"]]
    df.loc[df['category'] == cls, columns_to_update] = values_to_update

    ave_values_to_update = [np.mean(df['image_aucroc'][:-1]), np.mean(df['pixel_aucroc'][:-1]), np.mean(df['pixel_ap'][:-1])]
    df.loc[df['category'] == 'average', columns_to_update] = ave_values_to_update

    df.to_excel(excel_path, index=False)

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

def batch_feat_score(feat_res, mask):
    feat_score = torch.sum(feat_res, dim=1).view(-1)
    feat_score_np = feat_score.detach().cpu().numpy()
    mask_resized = F.interpolate(mask, size=64, mode='bilinear', align_corners=True)
    mask_resized[mask_resized > 0.5] = 1
    mask_resized[mask_resized <= 0.5] = 0
    mask_np = mask_resized.view(-1).detach().cpu().numpy()
    return feat_score_np[mask_np == 0], feat_score_np[mask_np == 1]

def plot_feat_score_distribution(feat_score_0, feat_score_1, log_path, file_name):
    feat_score_0 = np.array(feat_score_0)
    feat_score_1 = np.array(feat_score_1)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(feat_score_0, bins=50, alpha=0.75, color='blue')
    plt.title('Pixel Value Distribution (mask == 0)')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.subplot(1, 2, 2)
    plt.hist(feat_score_1, bins=50, alpha=0.75, color='green')
    plt.title('Pixel Value Distribution (mask == 1)')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(log_path, file_name), dpi=500)
    plt.close()