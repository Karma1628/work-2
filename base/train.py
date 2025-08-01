import os
import time
import argparse
import warnings
import torch.utils
import torch.utils
import torch.utils
from tqdm import tqdm
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *
from utils import *
from data.mvtec_dataset import MVTecTrain, MVTecTest

'''
    '''''': shift + alt + a
'''

# * Configurations
parser = argparse.ArgumentParser()
# ------------------------------------- data -------------------------------------
parser.add_argument('--data_name', type=str, default='mvtec', help='Name of the dataset')
parser.add_argument('--cls_ids', type=int, nargs='+', default=[0], help='Class indexes to be used')
parser.add_argument('--data_dir', type=str, default='/home/gysj_cyc/workbench/data/', help='Directory containing the data')
parser.add_argument('--work_dir', type=str, default='/home/gysj_cyc/workbench/base/', help='Working directory')
parser.add_argument('--bs', type=int, default=8, help='Batch size for training')
parser.add_argument('--nw', type=int, default=8, help='Number of workers for data loading')
parser.add_argument('--image_size', type=int, default=448, help='Image size of training datasets') # !
parser.add_argument('--test_size', type=int, default=256, help='Size of computing metrics')
parser.add_argument('--feature_size', type=int, default=64, help='Feature size of extractor') # !
# ------------------------------------- model -------------------------------------
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to be used')
parser.add_argument('--stages', type=int, nargs='+', default=[11], help='Stage of pre-trained backbone')
parser.add_argument('--base_ch', type=int, default=64, help='Base channel of networks') # !
# ------------------------------------- train -------------------------------------
parser.add_argument('--epochs', type=int, default=400, help='Total training epochs')
parser.add_argument('--log_epoch', type=int, default=20, help='Logging interval per epochs')
args = parser.parse_args()

for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")

# * Data class names
data_categories = {
    'mvtec': ['carpet', 'grid', 'leather', 'tile', 'wood', 
              'bottle','cable', 'capsule','hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper'],
    'visa': ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 
             'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'],
}

data_names = data_categories.get(args.data_name)
if args.cls_ids[0] != -1:
    clss = [data_names[i] for i in args.cls_ids]
    print('Categories to be train:', clss)
else:
    clss = data_names
    print('single GPU trains all categories!')

# * Define data paths
data_path = os.path.join(args.data_dir, args.data_name)
source_path = os.path.join(args.data_dir, 'dtd/images')

# * Choose device
device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

# * Start Training & Test
for cls in clss:

    torch.cuda.empty_cache()

    class_path = os.path.join(data_path, cls) # /home/karma1729/Desktop/data/mvtec/carpet
    log_path = os.path.join(args.work_dir, 'log', cls) #  /home/c1c/workbench/paper3/log/carpet
    os.makedirs(log_path, exist_ok=True)

    # * Define dataset & dataloader
    train_dataset = MVTecTrain(class_path, source_path, args.image_size)
    test_dataset = MVTecTest(class_path, args.image_size)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True, num_workers=args.nw)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=args.nw)
    num_dataset = len(train_dataset)

    # * Define the model
    # vit_small_patch14_dinov2       vit_base_patch14_dinov2       
    # swin_large_patch4_window12_384
    myADmodel = MyModel(
        weights_name='vit_small_patch14_dinov2',
        weights_path='/home/gysj_cyc/workbench/ckpts/',
        image_size=args.image_size,
        feature_size=args.feature_size,
        ch_base=args.base_ch, 
        stages=args.stages).to(device)
    stage_channel = myADmodel.stage_channels
    # stage_channel = [channel // 2 for channel in stage_channel] # if reduction

    # * Define loss functions 
    l2_loss = nn.MSELoss(reduction='mean')
    dicebce_loss = DiceBCELoss(bce_weight=0.5)

    # * Define optimizer
    optimizer = torch.optim.Adam([
        {'params': myADmodel.rgb2lab.parameters(), 'lr': 1e-4},
        {'params': myADmodel.lab2rgb.parameters(), 'lr': 1e-4},
        {'params': myADmodel.feature_segmentor.parameters(), 'lr': 1e-4},
    ])

    # * Define lr Scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                     [args.epochs * 0.8, args.epochs * 0.9], 
                                                     gamma=0.1, 
                                                     last_epoch=-1)
    
    # * Define tensorboard log
    tensorboard_writer = SummaryWriter(log_dir=log_path)

    # * Start training
    iter_num = 0
    

    # ! reduction realnet: afs
    # indices = myADmodel.reduction(reduction_train_loader, device)
    # indices = [0]

    torch.cuda.empty_cache()
    for epoch in range(1, args.epochs + 1):

        myADmodel.train()
        
        total_loss_epoch = 0.0
        rgb_loss_epoch = 0.0
        lab_loss_epoch = 0.0
        seg_loss_epoch = 0.0
        
        time_start = time.time()
        for _, batch in enumerate(tqdm(train_loader, desc='Train', leave=False)):

            # * Load batch of image and mask
            rgb_image = batch['rgb_image'].to(device)
            lab_image = batch['lab_image'].to(device)
            mask = batch['mask'].to(device)

            # * Foward the model
            rgb_feat_res, lab_feat_res, pred = myADmodel(rgb_image, lab_image)

            # * Resize mask to the size of feature
            mask = F.interpolate(mask.unsqueeze(1).float(), size=pred.shape[2], mode='bilinear', align_corners=True)
            mask = (mask >= 0.5).squeeze(1)
            
            # * Compute reconstruction loss
            loss_rgb = torch.mean(rgb_feat_res)
            loss_lab = torch.mean(lab_feat_res)
            loss_rec = loss_rgb + loss_lab
            # * Compute segmentation loss
            loss_seg = dicebce_loss(pred, mask.long())
            # * Sum losses
            loss_total = loss_rec + loss_seg

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            # * Log losses
            total_loss_epoch += loss_total.data
            rgb_loss_epoch += loss_rgb.data
            lab_loss_epoch += loss_lab.data
            seg_loss_epoch += loss_seg.data

            tensorboard_writer.add_scalar('Rec_loss', loss_rec, iter_num)
            tensorboard_writer.add_scalar('Seg_loss', loss_seg, iter_num)
            tensorboard_writer.add_scalar('All_loss', loss_total, iter_num)

            iter_num += 1

        scheduler.step()
        
        epoch_log = (
            f"Epoch[{epoch:03d}/{args.epochs}] "
            f"total_losses: {total_loss_epoch / num_dataset:.4f} "
            f"rgb_loss: {rgb_loss_epoch / num_dataset:.4f} "
            f"lab_loss: {lab_loss_epoch / num_dataset:.4f} "
            f"seg_loss: {seg_loss_epoch / num_dataset:.4f} "
            f"Time: {time.time()-time_start:.4f}s "
        )

        log_info(log_path, 'loss_log.txt', epoch_log)

        # * Start testing
        if epoch % args.log_epoch == 0:

            myADmodel.eval()

            masks = []
            preds = []
            rgb_preds = []
            lab_preds = []
            for _, batch in enumerate(tqdm(test_loader, desc='Validation', leave=False)):

                # * Load test image and mask
                image_path = batch['image_path'][0]
                rgb_image = batch['rgb_image'].to(device)
                lab_image = batch['lab_image'].to(device)
                mask = batch['mask'] # [1, 1, 256, 256]
                mask = F.interpolate(mask, size=args.test_size, mode='bilinear', align_corners=True)
                mask[mask >  0.5] = 1.0
                mask[mask <= 0.5] = 0.0
        
                # * Foward the model
                rgb_feat_res, lab_feat_res, pred = myADmodel(rgb_image, lab_image)

                # * Resize the reconstruction predict
                rgb_predict_feat = F.interpolate(rgb_feat_res, size=args.test_size, mode='bilinear', align_corners=True) # [1, C, 256, 256]
                rgb_predict_feat = torch.sum(rgb_predict_feat, dim=1).squeeze()
                rgb_predict_feat = rgb_predict_feat.detach().cpu().numpy()
                
                lab_predict_feat = F.interpolate(lab_feat_res, size=args.test_size, mode='bilinear', align_corners=True) # [1, C, 256, 256]
                lab_predict_feat = torch.sum(lab_predict_feat, dim=1).squeeze()
                lab_predict_feat = lab_predict_feat.detach().cpu().numpy()
                
                # * Resize the segementation predict
                pred_resized = F.interpolate(pred, size=args.test_size, mode='bilinear', align_corners=True) # [1, 2, 256, 256]
                pred_resized = pred_resized[0 ,1 ,: ,:].detach().cpu().numpy() # [256, 256]

                # * Visualize the image, ground_truth, feat_res, predict
                mask = mask.squeeze().numpy()
                
                visualize_results(log_path, image_path, mask, rgb_feat_res, pred_resized, 'rgb_images',
                                  True, (args.test_size, args.test_size), stage_channel)
                visualize_results(log_path, image_path, mask, lab_feat_res, pred_resized, 'lab_images',
                                  True, (args.test_size, args.test_size), stage_channel)

                masks.append(mask)
                preds.append(pred_resized)
                rgb_preds.append(rgb_predict_feat)
                lab_preds.append(lab_predict_feat)

            # * Calculate the metrics
            metrics = calculate_metrics(masks, preds)
            score_log = (
                f"feat "
                f"epoch[{epoch:03d}/{args.epochs}] "
                f"image_aucroc: {metrics['image_auc']:.4f}, "
                f"pixel_aucroc: {metrics['pixel_auc']:.4f}, "
                f"pixel_ap: {metrics['pixel_ap']:.4f}"
            )
            log_info(log_path, 'seg_score_log.txt', score_log)
            
            rgb_metrics = calculate_metrics(masks, rgb_preds)
            rgb_score_log = (
                f"rgb_feat "
                f"epoch[{epoch:03d}/{args.epochs}] "
                f"image_aucroc: {rgb_metrics['image_auc']:.4f}, "
                f"pixel_aucroc: {rgb_metrics['pixel_auc']:.4f}, "
                f"pixel_ap: {rgb_metrics['pixel_ap']:.4f}"
            )
            log_info(log_path, 'rgb_score_log.txt', rgb_score_log)
            
            lab_metrics = calculate_metrics(masks, lab_preds)
            lab_score_log = (
                f"lab_feat "
                f"epoch[{epoch:03d}/{args.epochs}] "
                f"image_aucroc: {lab_metrics['image_auc']:.4f}, "
                f"pixel_aucroc: {lab_metrics['pixel_auc']:.4f}, "
                f"pixel_ap: {lab_metrics['pixel_ap']:.4f}"
            )
            log_info(log_path, 'lab_score_log.txt', lab_score_log)

    summary_results_to_excel(os.path.join(args.work_dir, 'log'), 'product', args.data_name, cls, data_categories[args.data_name], metrics)
    summary_results_to_excel(os.path.join(args.work_dir, 'log'), 'rgb', args.data_name, cls, data_categories[args.data_name], rgb_metrics)
    summary_results_to_excel(os.path.join(args.work_dir, 'log'), 'lab', args.data_name, cls, data_categories[args.data_name], lab_metrics)
    