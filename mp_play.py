import multiprocessing as mp
import queue
import os
import time

def worker_target(input_queue: mp.Queue, result_queue: mp.Queue):
    while True:        
        try:
            input = input_queue.get(block=False)
        except queue.Empty:
            return
        
        time.sleep(0.5)
        output = input ** 2
        result_queue.put(output)
        print(f"pid: {os.getpid()}. input: {input}. output: {output}.")

if __name__ == "__main__":
    
    # The default on MacOS is spawn, not fork.
    # mp.set_start_method('spawn')
    # mp.set_start_method('fork')

    # The default mp.Queue() implementation does not work on MacOS! For whatever 
    # reason, the mp.Manager().Queue() does.
    input_q = mp.Manager().Queue()
    result_q = mp.Manager().Queue()

    for num in range(20):
        input_q.put(num)
    
    num_processes = 3
    processes = []
    for process_idx in range(num_processes):
        process = mp.Process(target=worker_target, args=(input_q, result_q))
        processes.append(process)
        process.start()

    while input_q.qsize() > 0:
        # print(f"Input queue size: {input_q.qsize()}")
        time.sleep(1.0)

    for process in processes:
        process.join()

    q_results = []
    while not result_q.empty():
        q_results.append(result_q.get())
    print(f"result queue: {q_results}.")