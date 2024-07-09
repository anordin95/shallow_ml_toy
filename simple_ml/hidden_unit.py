import torch

class HiddenUnit(torch.nn.Module):
    """This implementation strictly assumes the input dimensionality of the data is 2."""
    
    def __init__(self):
        super().__init__()
        self.slopes = torch.nn.Parameter(torch.rand(2))
        self.offset = torch.nn.Parameter(torch.rand(1))
        self.relu = torch.nn.ReLU()
    
    def _set_params(self, slope1: float, slope2: float, offset: float):
        """This function provides a way to manually override the randomly initialized or learned weights."""
        self.slopes = torch.nn.Parameter(torch.Tensor([slope1, slope2]))
        self.offset = torch.nn.Parameter(torch.Tensor([offset]))
        
    def forward(self, x: torch.Tensor) -> float:
        # Ensure the data's dimensionality is 2.
        if len(x.shape) == 1:
            assert x.shape[0] == 2, (
                f"Received a single data-point with shape: {x.shape}",
                f" The dimensionality of the data-point should be 2."
            )
        elif len(x.shape) == 2:
            assert x.shape[1] == 2, (
                f"Received a batch of data-points with shape: {x.shape}."
                " The dimensionality of each data-point should be 2."
            )
        else:
            raise ValueError(f"Received unexpected shaped input that has shape: {x.shape}")

        # Reduce from [Batch-size, Data-Dim] to [Batch-size, 1].
        value = torch.sum(self.slopes * x, dim=-1) + self.offset
        clipped_value = self.relu(value)
        
        return clipped_value

    def __repr__(self, precision=4, use_tex: bool = False):
        # Unicode \u208 puts the next character in a subscript. However, that's not supported
        # for matplotlib text objects. Instead, they support TeX-like notation.
        if use_tex:
            format_str = f"ReLU({self.slopes[0].item():.{precision}f}$x_1$"
            format_str += f" + {self.slopes[1].item():.{precision}f}$x_2$"
        else:
            format_str = f"ReLU({self.slopes[0].item():.{precision}f}x\u2081"
            format_str += f" + {self.slopes[1].item():.{precision}f}x\u2082"

        format_str += f" + {self.offset.item():.{precision}f})"
        return format_str
    
    def nearly_equals(self, obj, precision=4) -> bool:
        if not isinstance(obj, HiddenUnit):
            return False
        are_slopes_equal = torch.equal(self.slopes.round(decimals=precision), obj.slopes.round(decimals=precision))
        are_offsets_equal = torch.equal(self.offset.round(decimals=precision), obj.offset.round(decimals=precision))

        return (are_slopes_equal and are_offsets_equal)