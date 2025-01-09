import funcnodes as fn
from .SEM import SEM_NODE_SHELF
from .segmentation import SEGMENTATION_NODE_SHELF

__version__ = "0.0.3"

NODE_SHELF = fn.Shelf(
    nodes=[],
    name="Microscopy",
    description="The nodes of Funcnodes Microscopy package",
    subshelves=[SEM_NODE_SHELF, SEGMENTATION_NODE_SHELF],
)
