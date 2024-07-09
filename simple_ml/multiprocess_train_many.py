import multiprocessing as mp
import queue
import time

import tqdm
import torch
import pandas as pd

from simple_ml.train import train
from simple_ml.shallow_network import ShallowNetwork


def repeatedly_train(input_queue: mp.Queue, output_queue: mp.Queue, training_df: pd.DataFrame, min_loss_improvement: float, num_steps_to_improve_min_amount: int):
    
    while True:
        try:
            # The input doesn't really matter here, it's just a placeholder.
            _ = input_queue.get(block=False)
        except queue.Empty:
            # There's no more work in the queue(). Time for a holiday!
            break
        
        model = ShallowNetwork(num_hidden_units=1)
        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = torch.nn.BCELoss(reduction='mean')
        
        learned_model, best_loss, num_training_steps = train(
            model, loss_fn, optimizer, training_df, min_loss_improvement, num_steps_to_improve_min_amount
        )

        result = (learned_model, best_loss.item(), num_training_steps)
        output_queue.put(result)

def multiprocess_train(num_processes: int, num_training_runs: int, training_df: pd.DataFrame, min_loss_improvement: float, num_steps_to_improve_min_amount: int):
    
    # The default mp.Queue() implementation does not work on MacOS! For whatever 
    # reason, the mp.Manager().Queue() does.
    input_queue = mp.Manager().Queue()
    output_queue = mp.Manager().Queue()

    for _ in range(num_training_runs):
        input_queue.put(_)
    
    processes = []
    for _ in range(num_processes):
        process = mp.Process(target=repeatedly_train, args=(input_queue, output_queue, training_df, min_loss_improvement, num_steps_to_improve_min_amount))
        processes.append(process)
        process.start()
    
    with tqdm.tqdm(total=num_training_runs, smoothing=0.0, ncols=60) as progress_bar:
        
        prior_size = 0
        while output_queue.qsize() <= num_training_runs:
            
            current_size = output_queue.qsize()
            if current_size != prior_size:
                progress_bar.update(current_size - prior_size)
            prior_size = current_size
            time.sleep(1)
        
            # Intentionally check this condition here, rather than using a less-than above,
            # to ensure the progress-bar will fill up entirely upon completion.
            if current_size == num_training_runs:
                break
    
    for process in processes:
        process.join()

    outputs = []
    while not output_queue.empty():
        outputs.append(output_queue.get())
    
    return pd.DataFrame(outputs, columns=['model', 'best_loss', 'num_training_steps'])

