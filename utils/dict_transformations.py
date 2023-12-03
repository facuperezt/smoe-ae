from typing import Dict, Union
import torch

__all__ = ["flatten_dict", "sum_nested_dicts"]

def flatten_dict(d: Dict[str, Union[Dict, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    flattened_dict = {}
    for key, value in d.items():
        if isinstance(value, Dict):
            nested_dict = flatten_dict(value)
            for nested_key, nested_value in nested_dict.items():
                flattened_dict[key + "/" + nested_key] = nested_value
        else:
            flattened_dict[key] = value
    return flattened_dict

def sum_nested_dicts(d: Dict[str, torch.Tensor]) -> torch.Tensor:
    total = 0
    for value in d.values():
        if isinstance(value, Dict):
            total += sum_nested_dicts(value)
        else:
            total += value.sum()
    return total