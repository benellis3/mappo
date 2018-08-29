# map parameter registry
map_param_registry = {
    "m5v5_c_far": {
        "n_agents": 5,
        "n_enemies": 5,
        "limit": 200,
        "shield": False,
        "map_type": 'marines'},
    }

def get_map_params(map_name):
    return map_param_registry[map_name]

def map_present(map_name):
    return map_name in map_param_registry
