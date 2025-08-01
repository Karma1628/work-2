import os
import time
import argparse
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import *
from models import *
from data.visa_dataset import *
from data.btad_dataset import *
from data.mvtec_dataset import *

'''
    open terminal: ctrl + `
    '''''': shift + alt + a
    resnet18_8xb32_in1k

    # ! cls need rotation: carpet, wood, cable, capsule, toothbrush
'''

def setup_seed(seed):
     random.seed(seed)
     np.random.seed(seed)
     torch.manual_seed(seed)
     torch.cuda.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
    #  torch.backends.cudnn.deterministic = True
    #  torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--data_name', type=str, default='mvtec')
parser.add_argument('--cls_ids', type=int, nargs='+', default=[0])
parser.add_argument('--data_dir', type=str, default='/home/c1c/workbench/data/mvtec/')
parser.add_argument('--dtd_dir', type=str, default='/home/c1c/workbench/data/dtd/images/')
parser.add_argument('--save_root', type=str, default='/home/c1c/workbench/draem_siam/log/')
parser.add_argument('--bs', type=int, default=8, help='batch size')
parser.add_argument('--nw', type=int, default=8, help='num workers')
parser.add_argument('--base_ch', type=int, default=256, help='base channel of networs')
parser.add_argument('--image_size', type=int, default=256, help='size of image in datasets')
parser.add_argument('--epochs', type=int, default=400, help='trainng epochs')
parser.add_argument('--stages', type=int, nargs='+', default=[0, 1, 2], help='stage of pre-trained backbone')
parser.add_argument('--log_epoch', type=int, default=20)
args = parser.parse_args()

# * Data class names
mvtec = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle','cable', 'capsule','hazelnut', 
         'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper'] 
visa = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 
              'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
btad = ['01', '02', '03']

if args.data_name == 'mvtec':
    data_names = mvtec
elif args.data_name == 'visa':
    data_names = visa
elif args.data_name == 'btad':
    data_names = btad
else:
    print('There is no data!')
    exit(0)

if args.cls_ids[0] != -1:
    data_names = [data_names[i] for i in args.cls_ids]

# * Choose device
device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

for name in data_names:

    torch.cuda.empty_cache()
    setup_seed(0)

    class_path = os.path.join(args.data_dir, name) # /home/karma1729/Desktop/data/mvtec/carpet

    result_path = args.save_root + name
    os.makedirs(result_path, exist_ok=True)

    # * Define dataset
    if args.data_name == 'mvtec':
        train_dataset = MVTec_Train(class_path, args.dtd_dir, args.image_size)
        test_dataset = MVTec_test(class_path, args.image_size)
    elif args.data_name == 'visa':
        train_dataset = Visa_Train(class_path, args.dtd_dir, args.image_size)
        test_dataset = Visa_test(class_path, args.image_size)
    elif args.data_name == 'btad':
        train_dataset = BTAD_Train(class_path, args.dtd_dir, args.image_size)
        test_dataset = BTAD_test(class_path, args.image_size)

    # * Load data
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True, num_workers=args.nw)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=args.nw)

    # * Define models
    pre_model = 'swin-large_in21k-pre-3rdparty_in1k'
    feature_extractor = FeatureExtractor(weights=pre_model, device=device, stages=args.stages)
    feature_temp = feature_extractor(train_dataset[0]['normal_image'][None, ...].to(device))
    channel = feature_temp.shape[1] # 1344
    reconstructor_0 = GlobalNet(channel // 2, args.base_ch).to(device)
    reconstructor_1 = GlobalNet(channel // 2, args.base_ch).to(device)
    # feature_segmentor = DiscriminativeSubNetwork(in_channels=channel, out_channels=2).to(device)
    feature_segmentor = AnomalySeg(2).to(device)

    # * Define loss functions 
    L2_loss = nn.MSELoss(reduction='mean')

    # * Define optim
    optimizer = optim.Adam([
        {'params': reconstructor_0.parameters(), 'lr': 1e-4},
        {'params': reconstructor_1.parameters(), 'lr': 1e-4},
        {'params': feature_segmentor.parameters(), 'lr': 1e-4},
    ])

    # * Define lr Scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                                               [args.epochs * 0.8, args.epochs * 0.9], 
                                               gamma=0.1, 
                                               last_epoch=-1)
    writer = SummaryWriter(log_dir=result_path)

    iter_num = 0
    max_ap = 0.0
    # * Start training
    for epoch in range(1, args.epochs + 1):

        feature_extractor.eval()
        reconstructor_0.train()
        reconstructor_1.train()
        feature_segmentor.train()
        
        loss_epoch = 0.0
        rec_loss_epoch = 0.0
        seg_loss_epoch = 0.0
        time_start = time.time()

        for i, batch in enumerate(tqdm(train_loader, desc='Train', leave=False)):

            # * Load image and mask
            normal_image = batch['normal_image'].to(device) # torch.float32 [b, 3, 256, 256]
            pseudo_image = batch['pseudo_image'].to(device) # torch.float32 [b, 3, 256, 256]
            mask = batch['mask'].to(device) # torch.int64 [b, 256, 256]

            # * 1 Extract feature of normal image
            normal_feature = feature_extractor(normal_image) # [b, 576, 64, 64] 
            # * 2 Extract feature of pseudo image
            pseudo_feature = feature_extractor(pseudo_image) # [b, 576, 64, 64] 

            normal_feature_0 = normal_feature[:, ::2, ...]
            normal_feature_1 = normal_feature[:, 1::2, ...]

            pseudo_feature_0 = pseudo_feature[:, ::2, ...]
            pseudo_feature_1 = pseudo_feature[:, 1::2, ...]
            
            reconstruction_1 = reconstructor_0(pseudo_feature_0)
            reconstruction_0 = reconstructor_1(pseudo_feature_1)

            reconstruction = torch.empty_like(normal_feature).to(device)
            reconstruction[:, ::2, ...] = reconstruction_0
            reconstruction[:, 1::2, ...] = reconstruction_1
            
            # * Fuse the reconstruction and feature of pseudo
            reconstruction_diff = fusion(reconstruction, pseudo_feature)

            # * Segment the difference
            pseudo_predict = feature_segmentor(fusion(reconstruction_0, pseudo_feature_0), fusion(reconstruction_1, pseudo_feature_1))  # [b, 2, 64, 64] 

            # * Resize mask to the size of feature
            mask = nn.functional.interpolate(mask[:, None, :, :].float(), 
                                             size=pseudo_predict.shape[2], 
                                             mode='bilinear', 
                                             align_corners=False)
            mask = torch.where(
                mask < 0.5, torch.zeros_like(mask), torch.ones_like(mask)
            )
            mask = mask[:, 0, :, :] # b, 64, 64

            # * 1 Compute reconstruction loss
            # alpha = 1 / 3
            loss0 = L2_loss(reconstruction_0, normal_feature_0.detach().data)
            loss1 = L2_loss(reconstruction_1, normal_feature_1.detach().data)
            # loss2 = L2_loss(reconstruction, normal_feature.detach().data)
            # loss_rec_normal = alpha * loss0 + alpha * loss1 + (1 - 2 * alpha) * loss2
            loss_rec_normal = loss0 + loss1# + loss2
            # * 2 Compute segmentation loss
            loss_seg_pseudo = calc_loss(pseudo_predict, mask.long())
            # * 3 Sum losses
            loss_total = loss_rec_normal + loss_seg_pseudo

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            loss_epoch += loss_total.data.cpu().numpy()
            rec_loss_epoch += loss_rec_normal.data.cpu().numpy()
            seg_loss_epoch += loss_seg_pseudo.data.cpu().numpy()

            writer.add_scalar('Rec_loss', loss_rec_normal, iter_num)
            writer.add_scalar('Seg_loss', loss_seg_pseudo, iter_num)
            writer.add_scalar('All_loss', loss_total, iter_num)

            iter_num +=1
        
        scheduler.step()
        
        data_num = len(train_loader)
        epoch_log = '{} Epoch[{}/{}] losses: {:.4f} rec_loss: {:.4f} seg_loss: {:.4f} Time: {:.4f}s'.format(
            name, epoch, args.epochs, loss_epoch / data_num, rec_loss_epoch / data_num, seg_loss_epoch / data_num, time.time()-time_start)
        print(epoch_log)

        with open(result_path + '/loss_log.txt', mode='a') as f:
            f.write(epoch_log)
            f.write('\n')

        # * Start test
        display_image_path = []
        display_mask = []
        display_feature_diff = []
        display_segment_predict = []
        if epoch % args.log_epoch == 0:

            feature_extractor.eval()
            reconstructor_0.eval()
            reconstructor_1.eval()
            feature_segmentor.eval()

            masks = []
            scores = []
            ious = []
            for i, batch in enumerate(tqdm(test_loader, desc='Validation', leave=False)):

                # * Load test image and mask
                image_path = batch['image_path'][0]
                image = batch['image'].to(device)
                mask = batch['mask'].squeeze().numpy() # [256, 256]

                # * Extract feature
                feature = feature_extractor(image)
                # * Reconstruct feature
                feature_0 = feature[:, ::2, ...]
                feature_1 = feature[:, 1::2, ...]
                reconstruction_1 = reconstructor_0(feature_0)
                reconstruction_0 = reconstructor_1(feature_1)
                reconstruction = torch.empty_like(feature).to(device)
                reconstruction[:, ::2, ...] = reconstruction_0
                reconstruction[:, 1::2, ...] = reconstruction_1

                # * Fuse the difference of renconstruction and feature
                feature_diff = fusion(reconstruction, feature)
                # * Segment the difference
                feature_predict = feature_segmentor(fusion(reconstruction_0, feature_0), fusion(reconstruction_1, feature_1)) # [1, 2, 64, 64]
                # * Output the probability value
                feature_predict = torch.sigmoid(feature_predict)

                # * Resize the predict
                feature_predict = nn.functional.interpolate(feature_predict, 
                                                            size=args.image_size, 
                                                            mode='bilinear', 
                                                            align_corners=False) # [1, 2, 256, 256] [1, 1, 256, 256]
                segment_predict = feature_predict[0 ,1 ,: ,:].detach().cpu().numpy() # [256, 256]

                display_image_path.append(image_path)
                display_mask.append(mask)
                display_feature_diff.append(feature_diff.detach().cpu()) # cuda will locate the memory
                display_segment_predict.append(segment_predict)              

                masks.append(mask)
                scores.append(segment_predict)

            # * Calculate the metrics
            masks = np.array(masks).astype(np.int32)
            scores = np.array(scores)

            # * 1 Calulate Image-level metrics
            scores_ = torch.from_numpy(scores)
            scores_ = scores_.unsqueeze(0)
            scores_ = nn.functional.avg_pool2d(scores_, 32, 
                                               stride=1, padding=32//2).squeeze().numpy()
            image_pred = scores_.max(1).max(1)
            image_label = masks.any(axis=1).any(axis=1)
            image_auc = roc_auc_score(image_label, image_pred)
            image_ap = average_precision_score(image_label, image_pred)

            # * 2 Calulate Pixel-level metrics
            pixel_auc = roc_auc_score(masks.ravel(), scores.ravel())
            pixel_ap = average_precision_score(masks.ravel(), scores.ravel())

            bn_scores = (scores >= 0.5).astype(np.int32)
            iou01 = segment_iou_score(bn_scores, masks)
            iou = 0.0 if len(iou01) == 1 else iou01[1]

            # * Log metrics
            writer.add_scalar('image_auc', image_auc, epoch)
            writer.add_scalar('pixel_auc', pixel_auc, epoch)
            writer.add_scalar('pixel_ap', pixel_ap, epoch)

            score_log = 'epoch[{}/{}] image_aucroc: {:.4f}, pixel_aucroc: {:.4f}, pixel_ap: {:.4f}, IoU: {:.4f}'.format(
                    epoch, args.epochs, image_auc, pixel_auc, pixel_ap, iou)
            print(score_log)

            with open(result_path+'/score_log.txt', mode='a') as f:
                f.write(score_log)
                f.write('\n')

            # * Visualize the results
            for i in tqdm(range(len(display_image_path)), desc='Visualization', leave=False):
                visualizer(
                    idx=i,
                    image_path=display_image_path[i],
                    save_path=result_path,
                    mask=display_mask[i],
                    rec_diff=display_feature_diff[i],
                    seg_target=display_segment_predict[i],
                    heat_map=False)


    




                










            









            




