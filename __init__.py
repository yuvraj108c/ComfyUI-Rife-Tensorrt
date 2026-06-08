from .nodes.load_rife_tensorrt import LoadRifeTensorrtModel
from .nodes.rife_tensorrt import RifeTensorrt

NODE_CLASS_MAPPINGS = {
    "RifeTensorrt": RifeTensorrt,
    "LoadRifeTensorrtModel": LoadRifeTensorrtModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RifeTensorrt": "⚡ Rife Tensorrt",
    "LoadRifeTensorrtModel": "Load Rife Tensorrt Model",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
