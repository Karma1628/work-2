## *Easy Training*
```
python main.py --gpu_id 0 --class_ids 0 13 --epochs 300 --stages 0 1 2
```
## *Training and Testing*
```
python main.py --gpu_id 0 --class_ids -1 --mvtec_dir '/home/karma1729/Desktop/data/mvtec/' --dtd_dir '/home/karma1729/Desktop/data/dtd/images/' --save_root '/home/karma1729/Desktop/My_DRAEM/log/' --epohcs 300 --stages 0 1 2
```
## *Summarize metrics*
```
python summarizer.py --path '/home/karma1729/Desktop/My_DRAEM/test/'
```