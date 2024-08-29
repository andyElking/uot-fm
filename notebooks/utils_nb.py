import os
import sys
from importlib import util

def config_from_path(script_path):
    script_dir = os.path.dirname(script_path)
    sys.path.append(script_dir)

    module_name = os.path.basename(script_path).replace('.py', '')
    spec = util.spec_from_file_location(module_name, script_path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)

    config = module.get_config()
    return config