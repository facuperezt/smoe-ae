import copy
import json
from typing import Dict

def parse_cfg_file(path: str) -> Dict:
    with open(path, "rb") as f:
        unparsed_cfg = json.load(f)
    
    base_keys = {key: val for key, val in unparsed_cfg.items() if not isinstance(val, Dict)}
    if not len(base_keys) > 0:
        return unparsed_cfg

    parsed_cfg = replace_cfg_variables(unparsed_cfg, base_keys)
    return parsed_cfg
    

def replace_cfg_variables(_cfg_dict: Dict, base_keys: Dict) -> Dict:
    cfg_dict = copy.deepcopy(_cfg_dict)
    for base_key, base_val in base_keys.items():
        cfg_dict = replace_values_in_dict(cfg_dict, f"${base_key}", base_val)
        del cfg_dict[base_key]

    return cfg_dict

def replace_values_in_dict(d, old_value, new_value):
    """
    Recursively replace all occurrences of old_value with new_value in a nested dictionary.
    
    :param d: The dictionary to process
    :param old_value: The value to replace
    :param new_value: The value to replace with
    :return: The dictionary with replaced values
    """
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = replace_values_in_dict(value, old_value, new_value)
        elif value == old_value:
            d[key] = new_value
    return d
