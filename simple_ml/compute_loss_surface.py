"""This module iterates over every possible parameter setting for a simple-model, computes the 
loss, then plots the results. 

A ShallowNetwork with just one hidden unit, an input dimensionality of two and output dimensionality of one
still has 5 parameters! Unfortunately, 6-D plotting is not something I can do. So, we choose to vary and 
plot only some parameters (or amalgams of parameters).

The 5 parameters are: hidden_slope_1, hidden_slope_2, hidden_offset, output_slope, output_offset. We begin 
by looking at the results when varying these two values: hidden_slope_1 and outputs_slope.

What happens when hidden_slope_2 is constant?
    The activation-boundary is determined by h1 = slope1 * x1 + slope2 * x2 + offset.
    Assuming offset is 0; x1/x2 = -slope2/slope1. By varying slope1, from -inf to inf, 
    we effectively also vary the slope of the activation-boundary from 0 to -inf to inf to 0.
    
What about when hidden_offset is constant?
    Previously, we figured any slope of the activation-boundary was achievable, however, we lose
    some ability to adjust the x2-intercept, which will be at -offset/hidden_slope_2.

What about when output_offset is constant?
    The activation-point (not line), is at -output_offset/output_slope. As we vary output_slope
    from -inf to inf we effectively also vary this point from 0 to -inf to inf to 0.

And, we set the unobserved model-parameters to fixed values.
"""

import numpy as np
import pandas as pd
import torch

from simple_ml.shallow_network import ShallowNetwork


simple_model = ShallowNetwork(num_hidden_units=1)
loss_fn = torch.nn.BCELoss(reduction='none')

# Set the model parameters which will not change.
hidden_offset = 0.6
hidden_slope_2_value = 0.1
output_offset = 1

hidden_slope_1_values = np.arange(-18, 18, 0.1)
output_slope_values = np.arange(-18, 18, 0.1)

def compute_all_possible_models_and_losses(dataset_df: pd.DataFrame):
    """
    Args:
        dataset_df: A pd.DataFrame to be used for computing the loss of a model, 
            with three columns: x1, x2, and y.
    """
    X = torch.Tensor(dataset_df[['x1', 'x2']].values)
    Y = torch.Tensor(dataset_df['y'].values)

    parameters_and_losses = []

    for hidden_slope_1_value in hidden_slope_1_values:
        for output_slope_value in output_slope_values:

            # Set the output-layer params.
            simple_model._set_params(hidden_unit_scales=[output_slope_value], offset=output_offset)
            # Set the hidden-layer params.
            simple_model.hidden_units[0]._set_params(slope1=hidden_slope_1_value, slope2=hidden_slope_2_value, offset=hidden_offset)

            predictions = simple_model(X)
            loss_per_element = loss_fn(predictions, Y)
            loss = loss_per_element.sum().item()


            parameters_and_loss = (hidden_slope_1_value, output_slope_value, loss)
            parameters_and_losses.append(parameters_and_loss)
    
    return parameters_and_losses



