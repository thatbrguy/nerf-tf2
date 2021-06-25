import yaml
from box import Box

def load_params(path):
    """
    Loads a yaml file located at the given 
    path and returns a TODO object.

    TODO: Cleanup.
    """
    with open(path, 'r') as handle:
        params = yaml.safe_load(handle)

    ## TODO: Explore best practices for Box
    params = Box(params)
    return params
