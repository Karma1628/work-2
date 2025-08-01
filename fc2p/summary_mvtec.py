import os 
import re
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='/home/c1c/workbench/draem_siam/log/')
parser.add_argument('--mode', type=str, default='last')
args = parser.parse_args()

DataExcel = {
        'category':['carpet', 'grid', 'leather', 'tile', 'wood', 'Avg. Tex.',
                    'bottle','cable', 'capsule','hazelnut', 'metal_nut', 'pill', 
                    'screw', 'toothbrush', 'transistor', 'zipper', 'Avg. Obj.', 'Avg.'],
        'image_aucroc':list(range(18)),
        'pixel_aucroc':list(range(18)),
        'pixel_ap':list(range(18)),
        'IoU(0.5)':list(range(18))
}

textures = ['carpet', 'grid', 'leather', 'tile', 'wood']
objects = ['bottle','cable', 'capsule','hazelnut', 'metal_nut', 
           'pill', 'screw', 'toothbrush', 'transistor', 'zipper'] 
data_names = textures + objects

for name in data_names:
    class_path = args.path + name
    if os.path.exists(class_path):
        iaucroc = []
        paucroc = []
        pap = []
        piou = []
        with open(class_path+'/score_log.txt', 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines, desc=name, leave=False):
                pattern = r'\b\d+\.\d+\b'
                numbers = re.findall(pattern, line)
                numbers = [float(i) for i in numbers]
                iaucroc.append(numbers[0])
                paucroc.append(numbers[1])
                pap.append(numbers[2])
                piou.append(numbers[3])

        if args.mode == 'max':
            log_iauc = max(iaucroc)
            log_pauc = max(paucroc)
            log_pap = max(pap)
            log_pi = max(piou)
            file_name = '/result_max.xlsx'
        elif args.mode == 'last':
            log_iauc = iaucroc[-1]
            log_pauc = paucroc[-1]
            log_pap = pap[-1]
            log_pi = piou[-1]
            file_name = '/result_last.xlsx'

        class_idx = DataExcel['category'].index(name)
        DataExcel['image_aucroc'][class_idx] = log_iauc
        DataExcel['pixel_aucroc'][class_idx] = log_pauc
        DataExcel['pixel_ap'][class_idx] = log_pap
        DataExcel['IoU(0.5)'][class_idx] = log_pi
    else:
        print('There is no path of [' + name + ']!')

avg_tex_idx = DataExcel['category'].index('Avg. Tex.')
avg_obj_idx = DataExcel['category'].index('Avg. Obj.')
avg_idx = DataExcel['category'].index('Avg.')

DataExcel['image_aucroc'][avg_tex_idx] = np.mean(DataExcel['image_aucroc'][:5])
DataExcel['image_aucroc'][avg_obj_idx] = np.mean(DataExcel['image_aucroc'][6:16])
DataExcel['image_aucroc'][avg_idx] = np.mean(DataExcel['image_aucroc'][:5] + DataExcel['image_aucroc'][6:16])

DataExcel['pixel_aucroc'][avg_tex_idx] = np.mean(DataExcel['pixel_aucroc'][:5])
DataExcel['pixel_aucroc'][avg_obj_idx] = np.mean(DataExcel['pixel_aucroc'][6:16])
DataExcel['pixel_aucroc'][avg_idx] = np.mean(DataExcel['pixel_aucroc'][:5] + DataExcel['pixel_aucroc'][6:16])

DataExcel['pixel_ap'][avg_tex_idx] = np.mean(DataExcel['pixel_ap'][:5])
DataExcel['pixel_ap'][avg_obj_idx] = np.mean(DataExcel['pixel_ap'][6:16])
DataExcel['pixel_ap'][avg_idx] = np.mean(DataExcel['pixel_ap'][:5] + DataExcel['pixel_ap'][6:16])

DataExcel['IoU(0.5)'][avg_tex_idx] = np.mean(DataExcel['IoU(0.5)'][:5])
DataExcel['IoU(0.5)'][avg_obj_idx] = np.mean(DataExcel['IoU(0.5)'][6:16])
DataExcel['IoU(0.5)'][avg_idx] = np.mean(DataExcel['IoU(0.5)'][:5] + DataExcel['IoU(0.5)'][6:16])

df = pd.DataFrame(DataExcel)
df.to_excel(args.path+file_name, index=False)
        
                