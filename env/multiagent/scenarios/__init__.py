# import imp
import os.path as osp
import importlib

# def load(name):
#     pathname = osp.join(osp.dirname(__file__), name)
#     return imp.load_source('', pathname)


def load(name):
    # remove ".py" if present
    module_name = name.replace(".py", "")
    return importlib.import_module(f"multiagent.scenarios.{module_name}")
