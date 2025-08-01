import os 
import re
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='/home/c1c/workbench/My_DRAEM/log/')
parser.add_argument('--mode', type=str, default='max')
args = parser.parse_args()

DataExcel = {
        'category':['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 
                    'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum', 'Avg.'],
        'image_aucroc':list(range(13)),
        'pixel_aucroc':list(range(13)),
        'pixel_ap':list(range(13)),
        'IoU(0.5)':list(range(13))
}
 
data_names = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 
              'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']

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

avg_idx = DataExcel['category'].index('Avg.')
DataExcel['image_aucroc'][avg_idx] = np.mean(DataExcel['image_aucroc'][:12])
DataExcel['pixel_aucroc'][avg_idx] = np.mean(DataExcel['pixel_aucroc'][:12])
DataExcel['pixel_ap'][avg_idx] = np.mean(DataExcel['pixel_ap'][:12])
DataExcel['IoU(0.5)'][avg_idx] = np.mean(DataExcel['IoU(0.5)'][:12])

df = pd.DataFrame(DataExcel)
df.to_excel(args.path+file_name, index=False)