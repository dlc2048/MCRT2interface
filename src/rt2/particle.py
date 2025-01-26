from enum import Enum


class PTYPE(Enum):
    ELECTRON = -1
    PHOTON   = 0
    POSITRON = 1
    NEUTRON  = 6
    FNEUTRON = 7
    PROTON   = 21
    HEAVYION = 26
    HEAVYNUC = 27


PID_TO_PNAME = {
    -99: 'Mixed',
    -11: 'Hounsfield',
    -1: 'Electron',
    0: 'Photon',
    1: 'Positron',
    6: 'Neutron',
    7: 'Fast Neutron',
    21: 'Proton',
    26: 'Heavy Ion',
    27: 'Untransportable'
}


DTYPE_PHASE_SPACE = [
    ('pid', 'int32'),
    ('x', 'float32'),
    ('y', 'float32'),
    ('z', 'float32'),
    ('u', 'float32'),
    ('v', 'float32'),
    ('w', 'float32'),
    ('wee', 'float32'),
    ('e', 'float32')
]
