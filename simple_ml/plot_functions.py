import math

import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
from matplotlib import cm as color_maps

from simple_ml.hidden_unit import HiddenUnit
from simple_ml.shallow_network import ShallowNetwork

def create_dataset_plot(dataset_df: pd.DataFrame, ax: plt.Axes = None, ndim=2):
    """Plot the dataset stored in the provided dataframe. The dataframe should have 
    three columns: x1, x2 and y. 
    """
    
    if ax is None and ndim == 2:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,6))
    
    if ax is None and ndim == 3:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(projection='3d')
    
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_xlim([dataset_df['x1'].min() - 3, dataset_df['x1'].max() + 3])
    ax.set_ylim([dataset_df['x2'].min() - 3, dataset_df['x2'].max() + 3])

    if ndim == 2:
        ax.scatter(dataset_df['x1'], dataset_df['x2'], c=dataset_df['y'].apply(lambda y: 'forestgreen' if y==1  else 'blue'))
    elif ndim == 3:
        ax.scatter(xs=dataset_df['x1'], ys=dataset_df['x2'], zs=dataset_df['y'], c=dataset_df['y'].apply(lambda y: 'forestgreen' if y==1  else 'blue'))
        ax.set_zlim([dataset_df['y'].min() - 1, dataset_df['y'].max() + 1])

    ax.grid(visible=True)

def create_model_performance_plot(model: ShallowNetwork, dataset_df: pd.DataFrame):
    
    fig = plt.figure(figsize=(18, 8))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    _create_model_performance_plot_2d(ax1, model, dataset_df)
    _create_model_performance_plot_3d(fig, ax2, model, dataset_df)


def _create_model_performance_plot_2d(ax: plt.Axes, model: ShallowNetwork, dataset_df: pd.DataFrame):
    """Plot the dataset and the activation-lines of each hidden-unit in the model.
    """

    create_dataset_plot(dataset_df, ax, ndim=2)
    
    for idx, hidden_unit in enumerate(model.hidden_units):
        scaled_hidden_unit_repr = model.get_scaled_hidden_unit_repr(idx, precision=2)
        _plot_hidden_units_activation_boundary(hidden_unit, idx+1, scaled_hidden_unit_repr, ax)
    
    ax.axis('equal')
    ax.set_xlim([dataset_df['x1'].min() - 3, dataset_df['x1'].max() + 3])
    ax.set_ylim([dataset_df['x2'].min() - 3, dataset_df['x2'].max() + 3])
    ax.set_title("Activation Boundaries")
    ax.legend()
    

def _create_model_performance_plot_3d(fig: plt.Figure, ax: plt.Axes, model: ShallowNetwork, dataset_df: pd.DataFrame):
    
    create_dataset_plot(dataset_df, ax, ndim=3)

    num_points_per_axis = 500

    x1_min, x1_max = dataset_df['x1'].min() - 5, dataset_df['x1'].max() + 5
    x2_min, x2_max = dataset_df['x2'].min() - 5, dataset_df['x2'].max() + 5
    
    x1s = np.linspace(x1_min, x1_max, num_points_per_axis)
    x2s = np.linspace(x2_min, x2_max, num_points_per_axis)
    x1s_grid, x2s_grid = np.meshgrid(x1s, x2s)
    # With two inputs np.meshgrid() returns 2-D arrays of the x1 values and x2 values.
    x1s_grid, x2s_grid = x1s_grid.flatten(), x2s_grid.flatten()
    x1_x2_grid = np.stack([x1s_grid, x2s_grid], axis=-1)

    y = model(torch.Tensor(x1_x2_grid)).detach().numpy()

    surface = ax.plot_surface(
        x1s_grid.reshape((num_points_per_axis, num_points_per_axis)),
        x2s_grid.reshape((num_points_per_axis, num_points_per_axis)),
        y.reshape((num_points_per_axis, num_points_per_axis)),
        cmap=color_maps.winter,
        alpha=0.4
    )
    fig.colorbar(surface, shrink=0.5, pad=0.1)
    ax.set(xlabel="x1", ylabel="x2", zlabel="y")
    ax.set_title("Model Prediction Surface")
    ax.axis('equal')
    ax.set_xlim([x1_min, x1_max])
    ax.set_ylim([x2_min, x2_max])


def _plot_hidden_units_activation_boundary(hidden_unit: HiddenUnit, hidden_unit_num: int, scaled_hidden_unit_repr: str, ax: plt.Axes):
    """Plot the activation-boundary line of a given hidden unit onto the provided axes object."""
    
    # -------- Plot the activation-boundary line -------- #
    
    x1s = np.linspace(-10, 10, num=21)
    
    # Solve x1 * slope1 + x2 * slope2 + offset = 0, to get the line
    # defining the hidden unit's activation, i.e.
    # x2 = (-offset - x1 * slope1) / slope2
    slope1, slope2, offset = [hidden_unit.slopes[0].item(), hidden_unit.slopes[1].item(), hidden_unit.offset.item()]
    x2s = -offset/slope2 - x1s * slope1/slope2

    plotted_lines = ax.plot(x1s, x2s, label=f"hidden unit {hidden_unit_num}'s activation: {scaled_hidden_unit_repr}", alpha=0.5)
    # Save the color, so we can draw the arrows with the same color.
    color = plotted_lines[0].get_color()

    # -------- Plot small arrows along the boundary line in the direction of activation -------- #
    
    # The line perpendicular to the activation line:
    # perpendicular_x2s = (x1s * slope2/slope1) - offset/slope2

    # Begin at a point along the activation-boundary line. Travel one-unit
    # in the positive x1 direction along the line perpendicular to the activation 
    # boundary, then calculate if y positive or negative at that point.

    # These are the coordinates of the point.
    x1, x2 = (1, -offset/slope2 + slope2/slope1)
    # This is the y-value at that point.
    y = x1 * slope1 + x2 * slope2 + offset
    if y > 0:
        # The unit activates in the direction of positive x1.
        dx1 = 1
        dx2 = slope2/slope1
    else:
        # The unit activates in the direction of negative x1.
        dx1 = -1
        dx2 = -slope2/slope1
    
    # Draw arrows in the direction of activation at evenly spaced intervals along the original
    # activation-boundary line.    
    # How much to increment x1 to travel {arrow_spacing} units of distance on the original line?
    # Solve this for x1, (x1)^2 + (x1*slope1/slope2)^2 = arrow_spacing^2:
    arrow_spacing = 1
    x1_increment = math.sqrt(arrow_spacing ** 2 / (1 + (slope1/slope2) ** 2))
    x1s = np.arange(-10, 10, x1_increment)
    x2s = -offset/slope2 - x1s * slope1/slope2
    
    # Normalize the magnitude of the arrows (i.e. the vector (dx1, dx2)) to be 0.25.
    arrow_length = 0.25
    scaling_factor = arrow_length / math.sqrt((dx1 ** 2 + dx2 ** 2))
    scaled_dx1, scaled_dx2 = scaling_factor * dx1, scaling_factor * dx2

    for x1, x2 in zip(x1s, x2s):
        ax.arrow(x=x1, y=x2, dx=scaled_dx1, dy=scaled_dx2, width=0.05, color=color)