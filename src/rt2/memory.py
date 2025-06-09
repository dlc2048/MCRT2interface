from __future__ import annotations

from enum import Enum
import numpy as np

from rt2.fortran import Fortran, UnformattedFortran


class _UnformattedVector:
    _dtype = None

    def __init__(self,  file_name: str = '', max_counts=-1):
        self._capacity = 100
        self._size     = 0
        self._data     = np.empty(self._capacity, dtype=self._dtype)
        if file_name == '':
            pass
        else:
            stream = UnformattedFortran(file_name, recl=self._dtype.itemsize)
            bytes_array = stream.read(0, max_counts)
            stream.close()
            data_temp = np.frombuffer(bytes_array, dtype=self._dtype)
            self.append(data_temp)

    def __init_subclass__(cls, **kwargs):
        assert isinstance(cls._dtype, np.dtype)

    def reserve(self, size: int):
        if size > self._capacity:
            data_temp  = self._data
            self._data = np.empty(size, dtype=self._dtype)
            self._data[:self._capacity] = data_temp[:self._capacity]
            self._capacity = size

    def resize(self, size: int):
        if size > self._capacity:
            self.reserve(size)
        self._size = size

    def data(self):
        return self._data[:self._size]

    def append(self, arr: np.ndarray):
        assert arr.dtype == self._dtype
        arr_1d = arr.flatten()

        while self._capacity < self._size + len(arr_1d):
            self.reserve(self._capacity * 2)

        len_origin = self._size
        self.resize(len_origin + len(arr_1d))
        self._data[len_origin:self._size] = arr_1d

    def write(self, file_name: str):
        stream = UnformattedFortran(file_name, mode='w', recl=self._dtype.itemsize)
        stream.write(self.data())
        stream.close()

    def summary(self):
        return super.__repr__(self)

    def __repr__(self):
        return self.summary()
