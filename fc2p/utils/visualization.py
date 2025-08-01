import cv2
import os
import torch
import numpy as np
from PIL import Image

import torch.nn as nn

def visualizer(idx, image_path, save_path, mask, rec_diff, seg_target, heat_map=True, size=(256, 256)):
    save_img_path = save_path + '/images/'
    if not os.path.exists(save_img_path):
        os.makedirs(save_img_path)

    image = Image.open(image_path)
    image_pil = image.resize(size) # PIL

    mask_np = np.repeat(mask[..., None], 3, axis=2)
    mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))


    B, C, H, W = rec_diff.shape # 1, 576, 64, 64
    asm_rec = torch.sum(rec_diff, dim=1) # 1, 64, 64 anomaly score map
    asm_rec = nn.functional.interpolate(asm_rec.unsqueeze(0), 
                                        size=size[0], 
                                        mode='bilinear',
                                        align_corners=True) # 1, 1, 64, 64
    asm_rec = (asm_rec[0] - torch.min(asm_rec[0])) / (torch.max(asm_rec[0]) - torch.min(asm_rec[0]))
    asm_rec = asm_rec.data.cpu().numpy().astype(np.float32) # 1, 64, 64
    asm_rec = asm_rec.repeat(3, axis=0).transpose(1, 2, 0)

    img_np = np.array(image_pil).astype(np.uint8)

    asm_rec = (asm_rec * 255).astype(np.uint8)
    if heat_map:
        asm_rec = cv2.applyColorMap(asm_rec, cv2.COLORMAP_JET)
        asm_rec = cv2.cvtColor(asm_rec, cv2.COLOR_BGR2RGB)
        # asm_rec = cv2.addWeighted(img_np, 0.7, asm_rec, 0.3, 0)
    asm_rec = Image.fromarray(asm_rec)
    
    asm_seg = ((np.repeat(seg_target[..., None], 3, axis=2)) * 255).astype(np.uint8)
    if heat_map:
        asm_seg = cv2.applyColorMap(asm_seg, cv2.COLORMAP_JET)
        asm_seg = cv2.cvtColor(asm_seg, cv2.COLOR_BGR2RGB)
        # asm_seg = cv2.addWeighted(img_np, 0.7, asm_seg, 0.3, 0)
    asm_seg = Image.fromarray(asm_seg)

    asm = Image.new('RGB', (256*4+20, 256), (255, 255, 255))
    asm.paste(image_pil, (0, 0))
    asm.paste(mask_pil, (256+5, 0))
    asm.paste(asm_rec, (256*2+10, 0))
    asm.paste(asm_seg, (256*3+15, 0))
    asm.save(save_img_path + str(idx) + '.png')
