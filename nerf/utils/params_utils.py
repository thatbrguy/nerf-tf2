import yaml
from box import Box

def load_params(path):
    """
    Loads a yaml file located at the given path and returns 
    the loaded parameters.
    """
    with open(path, 'r') as handle:
        params = yaml.safe_load(handle)

    params = Box(params)
    return params
