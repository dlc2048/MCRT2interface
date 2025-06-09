from __future__ import annotations
"""
RT2 pencil beam group
"""

__author__ = "Chang-Min Lee"
__copyright__ = "Copyright 2022, Seoul National University"
__credits__ = ["Chang-Min Lee"]
__license__ = None
__maintainer__ = "Chang-Min Lee"
__email__ = "dlc2048@postech.ac.kr"
__status__ = "Production"

from enum import Enum
from copy import deepcopy

import numpy as np

from rt2.memory import _UnformattedVector
from rt2.print import fieldFormat, nameFormat


DTYPE_BEAMLET = [
    ('x',   'float32'),
    ('y',   'float32'),
    ('z',   'float32'),
    ('u',   'float32'),
    ('v',   'float32'),
    ('w',   'float32'),
    ('wee', 'float32'),
    ('e',   'float32'),
    ('sig', 'float32')
]


class Beamlet(_UnformattedVector):
    _dtype = np.dtype(DTYPE_BEAMLET)

    def summary(self):
        beamlet = self.data()
        total_weight = np.sum(beamlet['wee'])
        total_count  = len(beamlet)
        mean_std     = np.average(beamlet['sig'], weights=beamlet['wee']) if total_count else 0.0
        message = ""
        message += fieldFormat("Total counts",      total_count)
        message += fieldFormat("Total weights",     total_weight)
        message += fieldFormat("Mean Gauss std", mean_std)
        return message

