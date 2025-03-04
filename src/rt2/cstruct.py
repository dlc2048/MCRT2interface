from __future__ import annotations
"""
Bit field structure of RT2 C++ interface
"""

import ctypes
import numpy as np


class FLAGS_DEFAULT(ctypes.Structure):
    _fields_ = [
        ("region", ctypes.c_uint16, 16),
        ("fmask" , ctypes.c_uint16, 16)
    ]


class FLAGS_NEUTRON_P(ctypes.Structure):
    _fields_ = [
        ("region", ctypes.c_uint16, 16),
        ("group" , ctypes.c_short,  12),
        ("fmask" , ctypes.c_uint16, 4 )
    ]


class FLAGS_NEUTRON_S(ctypes.Structure):
    _fields_ = [
        ("iso_idx", ctypes.c_uint16, 16),
        ("rea_idx", ctypes.c_uint8,  8 ),
        ("sec_pos", ctypes.c_uint8,  8 )
    ]


class FLAGS_GENION(ctypes.Structure):
    _fields_ = [
        ("region" , ctypes.c_uint16, 16),
        ("fmask"  , ctypes.c_uint8,  8 ),
        ("ion_idx", ctypes.c_uint8,  8 )
    ]


class FLAGS_DEEX(ctypes.Structure):
    _fields_ = [
        ("region", ctypes.c_uint16, 16),
        ("z"     , ctypes.c_uint8,  8 ),
        ("a"     , ctypes.c_uint8,  8 )
    ]


class FLAGS_INCL(ctypes.Structure):
    _fields_ = [
        ("region"    , ctypes.c_uint16, 16),
        ("target_idx", ctypes.c_uint8,  8 ),
        ("proj_idx"  , ctypes.c_uint8,  8 )
    ]


def getBitMaskAndOffset(flags_struct, name : str) -> tuple:
    offset = 0
    for fname, ftype, bit_size in flags_struct._fields_:
        bit_mask = ((1 << bit_size) - 1) << offset
        if fname == name:
            return bit_mask, offset
        offset += bit_size
    assert False
