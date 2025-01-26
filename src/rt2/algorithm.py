from __future__ import annotations
"""
RT2 algorithms
"""

__author__ = "Chang-Min Lee"
__copyright__ = "Copyright 2022, Seoul National University"
__credits__ = ["Chang-Min Lee"]
__license__ = None
__maintainer__ = "Chang-Min Lee"
__email__ = "dlc2048@postech.ac.kr"
__status__ = "Production"

from collections.abc import Iterable
from copy import deepcopy

import numpy as np


def probFromAlias(alias_table: np.ndarray, prob_table: np.ndarray) -> np.ndarray:
    """
    convert alias table to the probability density function

    :param alias_table: Alias table
    :param prob_table: probablility table

    :return: probability density function
    """
    prob = np.zeros(prob_table.shape)
    mean = 1 / len(prob_table)
    for i in range(len(prob_table)):
        if prob_table[i] >= 1:
            prob[i] += mean
        else:
            target = alias_table[i]
            prob[target] += mean * (1.e0 - prob_table[i])
            prob[i] += mean * prob_table[i]

    return prob


class AliasTableEGS:

    def __init__(self, xdata: np.ndarray, fdata: np.ndarray, idata: np.ndarray, wdata: np.ndarray):
        """
        EGSnrc internal (N,M) dimension alias table where N is the number of group and M is the
        number of interpolation bins

        :param xdata: Alias table domain. All group share same domain when 'xdata' is 1-D, have
                      their own domain otherwise. Dimension must be (M+1) or (N, M+1)
        :param fdata: Alias table value. Dimension must be (N, M+1)
        :param idata: Alias index. Dimension must be (N, M)
        :param wdata: Alias probability. Dimension must be (N, M)
        """

        dim_x = xdata.shape
        dim_f = fdata.shape
        dim_i = idata.shape
        dim_w = wdata.shape

        if len(dim_f) != 2:
            raise Exception("'AliasTableEGS' fdata must be 2-D")
        if len(dim_i) != 2:
            raise Exception("'AliasTableEGS' idata must be 2-D")
        if len(dim_w) != 2:
            raise Exception("'AliasTableEGS' wdata must be 2-D")
        if len(dim_x) not in (1, 2):
            raise Exception("'AliasTableEGS' xdata must be 1-D or 2-D")

        self._dim = dim_w
        self._wdata = np.copy(wdata)
        if len(dim_x) == 1:
            if dim_x[0] != self._dim[1] + 1:
                raise Exception("'AliasTableEGS' dimension exception")
            self._xdata = np.broadcast_to(np.expand_dims(xdata, axis=0), fdata.shape)
        else:
            if dim_x[0] != self._dim[0] or dim_x[1] != self._dim[1] + 1:
                raise Exception("'AliasTableEGS' dimension exception")
            self._xdata = np.copy(xdata)
        if dim_i[0] != self._dim[0] or dim_i[1] != self._dim[1]:
            raise Exception("'AliasTableEGS' dimension exception")
        self._idata = np.copy(idata)
        if dim_f[0] != self._dim[0] or dim_f[1] != self._dim[1] + 1:
            raise Exception("'AliasTableEGS' dimension exception")
        self._fdata = np.copy(fdata)

    def sample(self, g: int):
        """
        Sample random number from alias table

        :param g: Group index
        :return: Random number
        """
        r1, r2 = np.random.random(2)
        aj = r1 * self._dim[1]
        j = int(aj)
        aj -= j

        if aj > self._wdata[g, j]:
            j = self._idata[g, j]

        x = self._xdata[g, j]
        dx = self._xdata[g, j + 1] - x

        if self._fdata[g, j] > 0:
            a = self._fdata[g, j + 1] / self._fdata[g, j] - 1
            if np.abs(a) < 0.2:
                rnno1 = 0.5 * (1 - r2) * a
                out = x + r2 * dx * (1 + rnno1 * (1 - r2 * a))
            else:
                out = x - dx / a * (1 - np.sqrt(1 + r2 * a * (a + 2)))
        else:
            out = x + dx * np.sqrt(r2)

        return out


class LogInterpEntry:
    def __init__(self, coeffs: np.ndarray, entries: np.ndarray):
        """
        Log-log interpolation entry
        :param coeffs: (2,)
        :param entries:
        """
        sh = entries.shape
        size = 1
        for dim in sh:
            size *= dim
        self._coeffs = coeffs
        self._eval = np.reshape(entries, (size // 2, 2))

    def __getitem__(self, energy: float):
        ele = np.log(energy)
        if isinstance(energy, Iterable):
            lele = np.array(self._coeffs[0] + self._coeffs[1] * ele, dtype=np.int)
            if np.min(lele) < 0:
                raise Exception("Energy out of range")
        else:
            lele = np.int(self._coeffs[0] + self._coeffs[1] * ele)
            if lele < 0:
                raise Exception("Energy out of range")

        return self._eval[lele, 0] + self._eval[lele, 1] * ele

    def eval(self):
        return self._eval


class Slicer:
    def __init__(self, size: int):
        self._size = size

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index, )

        if len(index) > self._size:
            raise ValueError()
        else:
            index = list(index)
            for i in range(len(index), self._size):
                index += [slice(None, None, None)]

        slicer = []
        for item in index:
            if isinstance(item, slice):
                slicer += [item]
            else:
                slicer += [slice(item - 1, item)]
        return tuple(slicer)


class Affine:
    def __init__(self, matrix: np.ndarray | None = None):
        if matrix is None:
            self._matrix = np.identity(4, dtype=float)
        else:
            if matrix.shape != (4, 4):
                raise ValueError("matrix must have 4x4 dimension")
            self._matrix = np.array(matrix, dtype=float)

    def __repr__(self) -> str:
        msg = ""
        for i in range(4):
            msg += "|"
            for j in range(4):
                msg += "{: >12.4f} ".format(self._matrix[i, j])
            msg += "|\n"
        return msg

    def __mul__(self, other: Affine):
        if not isinstance(other, Affine):
            raise TypeError

        affine_new = deepcopy(self)
        affine_new.transform(other)
        return affine_new

    def affine(self) -> np.ndarray:
        return np.copy(self._matrix)

    def inverse(self) -> Affine:
        old_mat = self.affine()
        new_mat = np.identity(4)
        new_mat[:3, :3] = np.linalg.inv(old_mat[:3, :3])
        new_mat[:3, 3] = -(new_mat[:3, :3]).dot(old_mat[:3, 3])
        return Affine(new_mat)

    def transform(self, other: Affine) -> None:
        if not isinstance(other, Affine):
            raise TypeError

        m1 = self._matrix
        m2 = other.affine()

        # Affine rotation & translation matrix
        r1 = m1[:3, :3]
        r2 = m2[:3, :3]
        t1 = m1[:3, -1]
        t2 = m2[:3, -1]

        # Transform
        m1[:3, :3] = r2.dot(r1)
        m1[:3, -1] = r2.dot(t1) + t2

    def translate(self, x: float, y: float, z: float) -> None:
        tmat = np.identity(4)
        tmat[:3, -1] = [x, y, z]
        self.transform(Affine(tmat))

    def scale(self, x: float, y: float, z: float) -> None:
        tmat = np.identity(4)
        tmat[0, 0] = x
        tmat[1, 1] = y
        tmat[2, 2] = z
        self.transform(Affine(tmat))

    def rotate(self, theta: float, axis: int) -> None:
        if not 0 <= axis < 3:
            raise ValueError("Axis out of range")
        tmat = np.identity(4)

        rad = theta / 180 * np.pi
        cost = np.cos(rad)
        sint = np.sin(rad)

        rot = np.identity(3)
        mask = np.ones_like(rot, dtype=bool)
        mask[axis, :] = 0
        mask[:, axis] = 0
        rot[mask] = [+cost, -sint, +sint, +cost]
        tmat[:3, :3] = rot[:]

        if axis == 1:
            tmat = np.transpose(tmat)
        self.transform(Affine(tmat))

    def origin(self) -> tuple:
        mat = self.affine()
        return tuple(mat[:3, 3])

    def size(self) -> tuple:
        mat = self.affine()
        return tuple((mat[:3, :3]).dot(np.array([1, 1, 1])))

    def axisAligned(self) -> bool:
        """
        Check whether the affine is axis-aligned or not

        :return: True if aligned, false elsewhere
        """
        diag = np.sum(np.abs(self._matrix[:3, :3]) > 0, axis=1)
        return True if np.all(diag == 1) else False
