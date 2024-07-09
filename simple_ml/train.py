"""This module exposes a single function for training a model with a given loss function, optimizer
training_dataset and some other hyper-parameters. This function only stops training when the loss
doesn't improve by some minimum amount over some number of update steps.


"""

import torch
import pandas as pd

def train(
    model: torch.nn.Module,
    loss_fn: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    training_df: pd.DataFrame,
    min_improvement: float = 1e-4,
    num_steps_to_improve_min_amount: int = 10_000,
    device: torch.device = torch.device('cpu'),
):
    
    best_loss, best_step = float('inf'), -1
    step = 0
    
    # .to() moves the model (i.e. torch.Module) in-place, i.e. the "model" variable now references a model 
    # on the specified device.
    model.to(device)
    
    X = torch.tensor(data=training_df[['x1', 'x2']].values, device=device, dtype=torch.float32)
    Y = torch.tensor(data=training_df['y'].values, device=device, dtype=torch.float32)
    
    while True:
        predictions = model(X)
        per_element_loss = loss_fn(predictions, Y)
        total_loss = per_element_loss.sum()

        # Take the partial-derivative of the total_loss w.r.t. each 
        # model-parameter (i.e. torch.nn.Parameter object). And store that 
        # partial-derivative in each Parameter object.
        total_loss.backward()
        # Update each parameter, leveraging it's stored gradient.
        optimizer.step()
        step += 1
        # Clear the stored gradient to prevent accumulation.
        optimizer.zero_grad()
        
        if total_loss < (best_loss - min_improvement):
            best_loss = total_loss
            best_step = step
        
        if (step - best_step) > num_steps_to_improve_min_amount:
            break
    
    return model, best_loss, step