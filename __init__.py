#from .Zho_layout import NODE_CLASS_MAPPINGS as NODE_CLASS_MAPPINGS_A
#from .PhotoMakerNode import NODE_CLASS_MAPPINGS as NODE_CLASS_MAPPINGS_B
#from .PM2 import NODE_CLASS_MAPPINGS as NODE_CLASS_MAPPINGS_B
#from .AnyTextNodeTest import NODE_CLASS_MAPPINGS as NODE_CLASS_MAPPINGS_C
from .ISTTID import NODE_CLASS_MAPPINGS as NODE_CLASS_MAPPINGS_E



# Combine the dictionaries
#NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS_A, **NODE_CLASS_MAPPINGS_B, **NODE_CLASS_MAPPINGS_C}
NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS_E}

__all__ = ['NODE_CLASS_MAPPINGS']
