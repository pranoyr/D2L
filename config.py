from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
from utils.collections import AttrDict

__C = AttrDict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

__C.DATASET = AttrDict()

# Default colours_per_class mapping
__C.DATASET.COLORS_PER_CLASS = {
                '0' : [254, 202, 87],
                '1' : [255, 107, 107]
        }
