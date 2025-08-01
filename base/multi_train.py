import time
import math
import subprocess
import pynvml

def check_gpu_usage():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    used_memory = mem_info.used / 1024**2  # Used memory in MB
    total_memory = mem_info.total / 1024**2  # Total memory in MB
    usage_percentage = (used_memory / total_memory) * 100
    pynvml.nvmlShutdown()
    return usage_percentage

def run_commands(commands):
    processes = []
    for command in commands:
        print(command)
        process = subprocess.Popen(command, shell=True)
        processes.append(process)

    for process in processes:
        process.wait()

if __name__ == '__main__':

    while True:
        usage = check_gpu_usage()
        print(f"GPU 0 Memory Usage: {usage:.2f}%")
        if usage < 50:
            print("Memory usage is below 50%, executing the program...")
            break
        else:
            print("Memory usage is above 50%, waiting for the next check...")
            time.sleep(60)

    start_time = time.time()
    
    # 4 GPUs
    main_coms = [
        'python train.py --gpu_id 0 --cls_ids 5 8 11', 
        'python train.py --gpu_id 1 --cls_ids 0 1 10 12',
        'python train.py --gpu_id 2 --cls_ids 2 4 7 13',
        'python train.py --gpu_id 3 --cls_ids 3 6 9 14',
    ]

    # 8 GPUs
    main_coms = [
        'python train.py --gpu_id 0 --cls_ids 5 8', 
        'python train.py --gpu_id 1 --cls_ids 0 1',
        'python train.py --gpu_id 2 --cls_ids 2 4',
        'python train.py --gpu_id 3 --cls_ids 3 6',
        'python train.py --gpu_id 4 --cls_ids 9 14',
        'python train.py --gpu_id 5 --cls_ids 7 13',
        'python train.py --gpu_id 6 --cls_ids 10 12',
        'python train.py --gpu_id 7 --cls_ids 11',
    ]

    run_commands(main_coms)

    total_seconds = time.time() - start_time
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"一共耗时：{hours}小时 {minutes}分钟 {seconds}秒")
    print("All commands have completed.")