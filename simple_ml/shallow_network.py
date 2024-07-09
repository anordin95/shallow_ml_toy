import torch

from simple_ml.hidden_unit import HiddenUnit

class ShallowNetwork(torch.nn.Module):
    def __init__(self, num_hidden_units: int):
        super().__init__()
        
        self.num_hidden_units = num_hidden_units
        # It's crucial to wrap these sub-modules (i.e. the HiddenUnits) in a ModuleList rather than a plain ol'
        # list. In a regular list, the sub-modules would not be recognized by PyTorch as being learnable 
        # components and any parameters in those sub-modules would not be updated during training! Ask me
        # how I found that out...
        self.hidden_units = torch.nn.ModuleList([HiddenUnit() for _ in range(num_hidden_units)])
        
        self.hidden_unit_scales = torch.nn.Parameter(torch.rand(num_hidden_units))
        self.offset = torch.nn.Parameter(torch.rand(1))
        self.sigmoid = torch.nn.Sigmoid()
    
    def _set_params(self, hidden_unit_scales: list[float], offset: float):
        """This function provides a way to manually override the randomly initialized or learned weights."""
        self.hidden_unit_scales = torch.nn.Parameter(torch.Tensor(hidden_unit_scales))
        self.offset = torch.nn.Parameter(torch.Tensor([offset]))
        
    def forward(self, x):
        hidden_unit_values = torch.stack([hidden_unit(x) for hidden_unit in self.hidden_units], dim=1)        
        scaled_hidden_unit_values = self.hidden_unit_scales * hidden_unit_values
        total_hidden_unit_value = torch.sum(scaled_hidden_unit_values, dim=1)
        
        y = self.sigmoid(self.offset + total_hidden_unit_value)

        
        return y

    def __repr__(self, precision=4):
        
        # Include/exclude new-line formatting.
        # format_str = f"Sigmoid(\n{self.offset.item():.{precision}f} + \n"
        format_str = f"Sigmoid({self.offset.item():.{precision}f} + "

        for i in range(self.num_hidden_units):
            
            format_str += f" {self.hidden_unit_scales[i].item():.{precision}f}{self.hidden_units[i]}"
            
            # Don't add a plus-sign after the final hidden-unit's equation.
            if i != self.num_hidden_units - 1:
                format_str += ' +'
            
            # Include/exclude new-line formatting.
            # format_str += '\n'

        format_str += ")"

        return format_str
    
    def nearly_equals(self, obj, precision=4) -> bool:
        
        if not isinstance(obj, ShallowNetwork):
            return False
        if self.num_hidden_units != obj.num_hidden_units:
            return False
        
        are_hidden_units_equal = all([self.hidden_units[idx].nearly_equals(obj.hidden_units[idx], precision) for idx in range(self.num_hidden_units)])
        are_hidden_unit_scales_equal = torch.equal(self.hidden_unit_scales.round(decimals=precision), obj.hidden_unit_scales.round(decimals=precision))
        are_offsets_equal = torch.equal(self.offset.round(decimals=precision), obj.offset.round(decimals=precision))
        
        return (
            are_hidden_units_equal and 
            are_hidden_unit_scales_equal and 
            are_offsets_equal
        )