from __future__ import annotations

import numpy as np


class Fortran:
    def __init__(self, file_name: str, mode="r"):
        """universal binary I/O"""
        self._file = open(file_name, mode=mode+"b")
        
    def close(self):
        self._file.close()

    def init(self):
        self._file.seek(0)

    def read(self, dtype):
        seg = self._file.read(4)
        if not seg:
            return seg  # EOF
        blen1 = np.frombuffer(seg, dtype=np.int32)[0]
        buffer = self._file.read(blen1)
        seg = self._file.read(4)
        blen2 = np.frombuffer(seg, dtype=np.int32)[0]

        if blen1 != blen2:
            raise ValueError
        if dtype == str:
            return buffer.decode()
        else:
            return np.frombuffer(buffer, dtype=dtype)
    
    def write(self, ndarray: np.ndarray | str):
        """write ndarray binary segment

        ndarray: numpy array. dtype should be int32, float32 or float64
        """
        # write ndarray binary as below
        # check data type
        if type(ndarray) is str:  # string
            bs = ndarray.encode()
        else:  # ndarray
            bs = ndarray.flatten().tobytes()

        length = len(bs)
        blen = np.array([length], dtype=np.int32).tobytes()
        self._file.write(blen)
        self._file.write(bs)
        self._file.write(blen)


class UnformattedFortran:
    def __init__(self, file_name: str, mode="r", recl: int = 1):
        """universal binary I/O"""
        self._file = open(file_name, mode=mode+"b")
        self._recl = recl

    def close(self):
        self._file.close()

    def init(self):
        self._file.seek(0)

    def read(self, rec: int, size: int = -1):
        self._file.seek(self._recl * rec)
        return self._file.read(size)

    def write(self, ndarray: np.ndarray | str):
        # write ndarray binary as below
        # check data type
        if type(ndarray) is str:  # string
            bs = ndarray.encode()
        else:  # ndarray
            bs = ndarray.flatten().tobytes()
        self._file.write(bs)
