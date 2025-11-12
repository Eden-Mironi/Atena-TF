"""
A File that should contain only the (single) global_env_prop variables
This variable is shared by all files in the project.
OPTIMIZED: Lazy loading to avoid 9-minute startup delay
"""
import os

import Configuration.config as cfg
from arguments import SchemaName
from gym_atena.lib.flight_delays_helpers import create_flights_env_properties
from gym_atena.lib.networking_helpers import create_networking_env_properties
from gym_atena.lib.big_flights_helpers import create_big_flights_env_properties
from gym_atena.lib.wide_flights_helpers import create_wide_flights_env_properties
from gym_atena.lib.wide12_flights_helpers import create_wide12_flights_env_properties
from gym_atena.lib.netflix_helpers import create_netflix_env_properties

_global_env_prop = None  # Private variable for lazy init


def update_global_env_prop_from_cfg():
    """
    Changing the properties of the current schema based on cfg.schema and returning the new properties object
    Returns:

    """
    global _global_env_prop
    
    # Only load if not already loaded
    if _global_env_prop is not None:
        return _global_env_prop
    
    print(f"Loading datasets for schema: {cfg.schema}")
    schema_name = SchemaName(cfg.schema)
    if schema_name is SchemaName.NETWORKING:
        _global_env_prop = create_networking_env_properties()
    elif schema_name is SchemaName.FLIGHTS:
        _global_env_prop = create_flights_env_properties()
    elif schema_name is SchemaName.BIG_FLIGHTS:
        _global_env_prop = create_big_flights_env_properties()
    elif schema_name is SchemaName.WIDE_FLIGHTS:
        _global_env_prop = create_wide_flights_env_properties()
    elif schema_name is SchemaName.WIDE12_FLIGHTS:
        _global_env_prop = create_wide12_flights_env_properties()
    elif schema_name is SchemaName.NETFLIX:
        _global_env_prop = create_netflix_env_properties()
    else:
        raise NotImplementedError
    if cfg.dataset_number is not None and cfg.outdir:
        # Save dataset name
        with open(os.path.join(cfg.outdir, 'dataset.txt'), 'w') as f:
            f.write(str(_global_env_prop.env_dataset_prop.repo.file_list[cfg.dataset_number]))
    print(f"Datasets loaded successfully!")
    return _global_env_prop


# Property-based lazy access
class _GlobalEnvPropAccessor:
    """Lazy accessor for global_env_prop"""
    def __getattr__(self, name):
        global _global_env_prop
        if _global_env_prop is None:
            update_global_env_prop_from_cfg()
        return getattr(_global_env_prop, name)
    
    def __bool__(self):
        """Support truthiness checks"""
        global _global_env_prop
        return _global_env_prop is not None


# Export the lazy accessor as global_env_prop
global_env_prop = _GlobalEnvPropAccessor()

# LAZY LOADING: Don't call update_global_env_prop_from_cfg() at import time!
# It will be called automatically when first accessed via the property accessor
