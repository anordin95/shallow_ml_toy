"This module provides functions for producing/generating datasets. For example, by sampling from some distribution or function."

import math
import random
from typing import Tuple

import pandas as pd
import numpy as np

def _set_random_seeds():
    random.seed(7)
    np.random.seed(7)

def _generate_random_point_on_a_semi_circle(radius: int) -> Tuple[float, float]:
    """Returns the (x,y) coordinates of a random point that lies on a semi-circle (y > 0) of the given 
    radius centered at the origin: (0,0).  """
    
    x = radius * random.uniform(-1.0, 1.0)
    
    # This equation always provides a positive y-value.
    y = math.sqrt(radius ** 2 - x ** 2)
    return (x, y)

def _generate_random_point_on_a_circle(radius: int) -> Tuple[float, float]:
    """Returns the (x,y) coordinates of a random point that lies on a circle of the given 
    radius centered at the origin: (0,0).  """
    
    x, y = _generate_random_point_on_a_semi_circle(radius)
    # Flip a coin to decide if y is negative or positive.
    if random.random() > 0.5:
        y = -y

    return (x, y)

def generate_semi_circle_dataset(inner_radius: int, outer_radius: int, num_data_points_per_class: int) -> pd.DataFrame:
    _set_random_seeds()

    data_points = []
    for _ in range(num_data_points_per_class):
        
        # Add the labels to each point. The inner-circle is labelled 1 and the outer-circle 0.
        inner_circle_point = _generate_random_point_on_a_semi_circle(inner_radius) + (1,)
        outer_circle_point = _generate_random_point_on_a_semi_circle(outer_radius) + (0,)
        
        data_points.append(inner_circle_point)
        data_points.append(outer_circle_point)

    df = pd.DataFrame(data_points, columns=['x1', 'x2', 'y'])
    return df

def generate_circle_dataset(inner_radius: int, outer_radius: int, num_data_points_per_class: int) -> pd.DataFrame:
    _set_random_seeds()

    data_points = []
    for _ in range(num_data_points_per_class):
        
        # Add the labels to each point. The inner-circle is labelled 1 and the outer-circle 0.
        inner_circle_point = _generate_random_point_on_a_circle(inner_radius) + (1,)
        outer_circle_point = _generate_random_point_on_a_circle(outer_radius) + (0,)
        
        data_points.append(inner_circle_point)
        data_points.append(outer_circle_point)

    df = pd.DataFrame(data_points, columns=['x1', 'x2', 'y'])
    return df

def generate_lines_dataset(num_data_points_per_class: int):
    """This dataset has 3-D points. The first two dimensions of a point are real-values 
    and the third is a binary class. Points in the same class, lie along the same line
    in those first two dimensions. """
    _set_random_seeds()
    
    data_points = []

    x1s = np.linspace(1, 5, num_data_points_per_class)
    
    label_zero_func = lambda x1: x1 * 0.5 + 0.5
    label_one_func = lambda x1: x1 * 1 + 10
    
    for x1 in x1s:
        data_point = (x1, label_zero_func(x1), 0)
        data_points.append(data_point)
        
        data_point = (x1, label_one_func(x1), 1)
        data_points.append(data_point)

    df = pd.DataFrame(data_points, columns = ['x1', 'x2', 'y'])
    return df
    

def get_manually_created_data():
    df = pd.DataFrame([
        [1, 4, 0],
        [1, 5, 0],
        [1.5, 7, 0],
        [2.5, 6.5, 0],
        [2, 6, 0],
        [3, 7, 0],
        [3, 7.5, 0],
        [4, 5.5, 0],
        [4.5, 6, 0],
        [4.75, 7, 0],
        [5, 3, 0],
        [5, 4, 0],
        [6, 4.5, 0],
        [4.5, 2, 0],
        [3.5, 1, 0],
        [3, 0, 0],
        [4, 1, 0],
        [2, 0.5, 0],
        [1.5, 2, 0],
        [1.5, 3, 0],
        [3, 3, 1],
        [3, 4, 1],
        [2, 4, 1],
        [4, 3.5, 1],
        [3, 3.5, 1],
        [2.5, 4, 1],
        [3.5, 4.5, 1],
        [3.5, 4.25, 1],
        [3, 5, 1],
        ], 
        columns=['x1', 'x2', 'y']
    )
    return df