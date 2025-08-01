import time
import math
import subprocess


def run_commands(commands):
    processes = []
    for command in commands:
        print(command)
        process = subprocess.Popen(command, shell=True)
        processes.append(process)

    for process in processes:
        process.wait()

if __name__ == '__main__':

    start_time = time.time()
    
    gpu_ids = list(range(4))
    class_ids = list(range(15)) 
    per = math.ceil(len(class_ids) / len(gpu_ids))

    cls = ['' for i in gpu_ids] 

    main_coms = []
    for gpu_id in gpu_ids:
        for i in range(per):
           num = gpu_id * per + i
           if num in class_ids:
              cls[gpu_id] += str(num)+' '
        task = f'python main.py --gpu_id {gpu_id} --class_ids ' + cls[gpu_id]
        main_coms.append(task.rstrip())
    
    main_coms = [
        'python main.py --gpu_id 0 --cls_ids 5 8 11', 
        'python main.py --gpu_id 1 --cls_ids 0 1 10 12',
        'python main.py --gpu_id 2 --cls_ids 2 4 7 13',
        'python main.py --gpu_id 3 --cls_ids 3 6 9 14',
    ]

    run_commands(main_coms)

    if len(class_ids) == 15:
        summary_coms = [
            'python summary_mvtec.py --mode max',
            'python summary_mvtec.py --mode last'
        ]
    elif len(class_ids) == 12:
        summary_coms = [
            'python summary_visa.py --mode max',
            'python summary_visa.py --mode last'
        ]

    run_commands(summary_coms)

    total_seconds = time.time() - start_time
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"一共耗时：{hours}小时 {minutes}分钟 {seconds}秒")
    print("All commands have completed.")