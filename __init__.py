"""
Custom nodes for ComfyUI image editing.

This module provides custom nodes for scaling and compositing images.
"""

from .scale_node import ScaleImageNode
from .composite_node import CompositeImageNode

NODE_CLASS_MAPPINGS = {
    "ScaleImage": ScaleImageNode,
    "CompositeImage": CompositeImageNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ScaleImage": "Scale Image",
    "CompositeImage": "Composite Image",
}
