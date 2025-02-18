from __future__ import annotations
"""
RT2 tally I/O and interface
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
from typing import Union

import numpy as np

from rt2.fortran import Fortran, UnformattedFortran
from rt2.particle import PID_TO_PNAME, DTYPE_PHASE_SPACE
from rt2.algorithm import Slicer, Affine
from rt2.print import fieldFormat, nameFormat


class ENERGY_TYPE(Enum):
    LINEAR  = 0
    LOG     = 1


class DENSITY_TYPE(Enum):
    DEPO = 0
    DOSE = 1


class _TallyContext:
    def __init__(self):
        self.name = ""
        self.unit = ""
        self.data = None
        self.unc  = None

    def _readHeader(self, stream: Fortran):
        name_byte = stream.read(np.byte).tostring()
        self.name = name_byte.decode('utf-8')
        part_byte = stream.read(np.byte).tostring()
        self.unit = part_byte.decode('utf-8')

    def _writeHeader(self, stream: Fortran):
        stream.write(self.name)
        stream.write(self.unit)

    @staticmethod
    def _readData(stream: Fortran):
        data_1d = stream.read(np.float32)
        unc_1d  = stream.read(np.float32)
        return data_1d, unc_1d

    def _writeData(self, stream: Fortran):
        stream.write(self.data.flatten())
        stream.write(self.unc.flatten())

    def _add(self, other: Union[_TallyContext, float, int, np.ndarray]):
        if isinstance(other, _TallyContext):
            if self.unit != other.unit:
                raise TypeError()
            # Error propagation
            var = (self.unc * self.data) ** 2 + (other.unc * other.data) ** 2
            self.data += other.data
        elif isinstance(other, (float, int, np.ndarray)):
            var = (self.unc * self.data) ** 2
            self.data += other
        else:
            raise TypeError()
        self.unc = np.divide(np.sqrt(var), self.data, out=np.zeros_like(self.data), where=self.data != 0)

    def _sub(self, other: Union[_TallyContext, float, int, np.ndarray]):
        if isinstance(other, _TallyContext):
            if self.unit != other.unit:
                raise TypeError()
            # Error propagation
            var = (self.unc * self.data) ** 2 + (other.unc * other.data) ** 2
            self.data -= other.data  # Boundary check will be processed in child
        elif isinstance(other, (float, int, np.ndarray)):
            var = (self.unc * self.data) ** 2
            self.data -= other
            # Same unc
        else:
            raise TypeError()
        self.unc = np.divide(np.sqrt(var), self.data, out=np.zeros_like(self.data), where=self.data != 0)

    def _mul(self, other: Union[float, int, np.ndarray]):
        if isinstance(other, (float, int, np.ndarray)):
            self.data *= other
            # Same unc
        else:
            raise TypeError()

    def _truediv(self, other: Union[float, int, np.ndarray]):
        if isinstance(other, (float, int, np.ndarray)):
            self.data /= other
            # Same unc
        else:
            raise TypeError()

    def _summary(self):
        message = ""
        message += fieldFormat("Name", self.name)
        message += fieldFormat("Unit", self.unit)
        return message


class _FilterContext:
    def __init__(self):
        self.part = ""

    def _readFilter(self, stream: Fortran):
        name_byte = stream.read(np.byte).tostring()
        self.part = name_byte.decode('utf-8')

    def _writeFilter(self, stream: Fortran):
        stream.write(self.part)

    def _combine(self, other: Union[_FilterContext, float, int, np.ndarray]):
        if isinstance(other, _FilterContext):
            if self.part == other.part:
                pass
            else:
                self.part = "mixed"
        else:
            pass

    def _summary(self):
        message = ""
        message += fieldFormat("Part", self.part)
        return message


class _MeshContext(Affine):
    def __init__(self):
        Affine.__init__(self)
        self._shape = np.empty(3, dtype=int)

    def __repr__(self):
        message = ""
        message += fieldFormat("shape", "{},{},{}".format(self._shape[0], self._shape[1], self._shape[2]))
        message += "Affine matrix\n"
        message += Affine.__repr__(self)
        return message

    def _readGeometryInfo(self, stream: Fortran):
        shape = stream.read(np.int32)
        for i in range(3):
            self._shape[i] = shape[i]
        matrix_1d = stream.read(float)
        matrix = np.reshape(matrix_1d, (3, 4))
        matrix = np.append(matrix, np.array([[0, 0, 0, 1]]), axis=0)
        Affine.__init__(self, matrix)

    def _writeGeometryInfo(self, stream: Fortran):
        shape = np.empty(3, dtype=np.int32)
        for i in range(3):
            shape[i] = self._shape[i]
        stream.write(shape)
        matrix = self.affine()[:-1].flatten()
        stream.write(matrix)

    def _operatorCheck(self, other: Union[_MeshContext, float, int]):
        if isinstance(other, _MeshContext):
            if not (self.affine() == other.affine()).all():
                raise ValueError("Affine matrix must be same")
            if not (self._shape == other._shape).all():
                raise ValueError("Mesh shape must be same")

    def _setMeshBoundary(self, index: tuple):
        for i, s in enumerate(index):
            start = s.start
            if not start:  # None
                start = 0
            elif start < 0:
                start += self._shape[i]
            stop = s.stop
            if not stop:
                stop = self._shape[i]
            elif stop < 0:
                stop += self._shape[i]

            if s.step and s.step > 1:
                raise IndexError
            if not 0 <= start < stop <= self._shape[i]:
                raise IndexError("Index out of range")

            self._matrix[i, 3] += start * self._matrix[i, i]
            self._shape[i] = stop - start

    def _imageSlice(self, pos: float, axis):
        """
        Get the 2-D slice and extent from mesh

        :param pos: Slice coordinate (cm)
        :param axis: Slicing axis, must be in range [0,2]
        :return: 2-D image slice (tuple), extent (tuple)
        """
        if not isinstance(axis, int):
            raise ValueError("axis must be integer")
        if not 0 <= axis < 3:
            raise IndexError("axis out of range")

        pos_3 = np.array([self._matrix[0, 3], self._matrix[1, 3], self._matrix[2, 3]])
        pos_3[axis] = pos

        idx_3 = self.index(pos_3[0], pos_3[1], pos_3[2])
        idx = idx_3[axis]
        if not 0 <= idx < self._shape[axis]:
            raise IndexError("pos out of range")

        slicer = [slice(None, None, None), slice(None, None, None), slice(None, None, None)]
        slicer[axis] = idx
        slicer = tuple(slicer)

        extent = np.empty(4)
        origin = self.origin()
        size = self.size()
        for i in range(2):  # Axis permutation (aka Levi-Chivita)
            target = (-i + axis + 2) % 3
            extent[2 * i] = origin[target]
            extent[2 * i + 1] = origin[target] + size[target] * self._shape[target]

        return slicer, tuple(extent)

    def index(self, x: float, y: float, z: float) -> tuple:
        """
        Get the mesh index from point position

        :param x: x coordinate (cm)
        :param y: y coordinate (cm)
        :param z: z coordinate (cm)
        :return: Mesh index tuple (i,j,k)
        """
        inv_mat = self.inverse().affine()
        idx = (inv_mat[:3, :3]).dot(np.array([x, y, z])) + inv_mat[:3, 3]
        idx = idx.astype(int)
        for i in range(3):
            if not 0 <= idx[i] < self._shape[i]:
                raise IndexError("Point out of range")
        return tuple(idx)

    def where(self, i: float, j: float, k: float) -> tuple:
        """
        Get the voxel's absolute position (center of voxel) from mesh index

        :param i: x mesh index
        :param j: y mesh index
        :param k: z mesh index
        :return: Voxel center coordinate (cm)
        """
        idx = np.array([i, j, k])
        for i in range(3):
            if not 0 <= idx[i] < self._shape[i]:
                raise IndexError("Point out of range")
        mat = self.affine()
        pos = (mat[:3, :3]).dot(idx) + mat[:3, 3]
        return tuple(pos)

    def origin(self) -> tuple:
        """
        Get the mesh origin in absolute coordinates

        :return: Mesh origin (x, y, z)
        """
        mat = self.affine()
        pos_ref = (mat[:3, :3]).dot(np.array([-0.5, -0.5, -0.5])) + mat[:3, 3]
        return tuple(pos_ref)

    def size(self) -> tuple:
        """
        Get the mesh size in reference corrdinates

        :return: Mesh size (dx, dy, dz)
        """
        mat = self.affine()
        return tuple((mat[0, 0], mat[1, 1], mat[2, 2]))

    def shape(self) -> tuple:
        """
        Get the mesh shape

        :return: Mesh shape (nx, ny, nz)
        """
        return tuple(self._shape)

    def extent(self) -> tuple:
        """
        Get the mesh extent. Mesh must be axis-aligned

        :return: Mesh extent (xmin, xmax, ymin, ymax, zmin, zmax)
        """
        mat = self.affine()
        if not (np.abs(mat[:3, :3] - np.diag(np.diagonal(mat[:3, :3]))) < 1.e-6).all():  # 1e-6 FP error tolerance
            raise ValueError("Mesh is not axis-aligned")

        amin = np.array([-0.5, -0.5, -0.5])
        amax = np.array([
            -0.5 + self._shape[0],
            -0.5 + self._shape[1],
            -0.5 + self._shape[2]
        ])
        amin = (mat[:3, :3]).dot(amin) + mat[:3, 3]
        amax = (mat[:3, :3]).dot(amax) + mat[:3, 3]
        for i in range(3):
            if amin[i] > amax[i]:
                amin[i], amax[i] = amax[i], amin[i]

        return tuple((amin[0], amax[0], amin[1], amax[1], amin[2], amax[2]))


class _FluenceContext:
    def __init__(self):
        self._etype  = ENERGY_TYPE(0)
        self._erange = np.empty(2)
        self._nbin   = 0

    def _readEnergyStructure(self, stream: Fortran):
        etype = stream.read(np.int32)[0]
        self._etype  = ENERGY_TYPE(etype)
        self._erange = stream.read(float)
        self._nbin   = stream.read(np.int32)[0]

    def _writeEnergyStructure(self, stream: Fortran):
        etype = np.array([self._etype.value], dtype=np.int32)
        stream.write(etype)
        stream.write(self._erange)
        nbin = np.array([self._nbin], dtype=np.int32)
        stream.write(nbin)

    def _operatorCheck(self, other: Union[_FluenceContext, float, int]):
        if isinstance(other, _FluenceContext):
            if self._etype != other._etype:
                raise TypeError("Energy bin type must be same")
            if not (self._erange == other._erange).all():
                raise ValueError("Energy boundary must be same")
            if self._nbin != other._nbin:
                raise ValueError("Number of energy bin must be same")

    def _setEnergyBoundary(self, index: slice):
        start = index.start
        if not start:  # None
            start = 0
        elif start < 0:
            start += self._nbin
        stop = index.stop
        if not stop:  # None
            stop = self._nbin
        elif stop < 0:
            stop += self._nbin

        if index.step and index.step > 1:
            raise IndexError
        if not 0 <= start < stop <= self._nbin:
            raise IndexError("Index out of range")

        efrom = self._erange[0]
        eto   = self._erange[1]
        if self._etype == ENERGY_TYPE.LOG:
            efrom = np.log10(efrom)
            eto = np.log10(eto)
        estep = (eto - efrom) / self._nbin
        eto   = efrom + estep * stop
        efrom = efrom + estep * start
        self._nbin = stop - start
        if self._etype == ENERGY_TYPE.LOG:
            efrom = 10 ** efrom
            eto   = 10 ** eto
        self._erange[0] = efrom
        self._erange[1] = eto

    def _summary(self):
        message = ""
        message += fieldFormat("Ebin type", self.etype())
        message += fieldFormat("Ebin range", tuple(self._erange), "MeV")
        message += fieldFormat("# of ebin", self._nbin)
        return message

    def etype(self):
        return "Linear" if self._etype == ENERGY_TYPE.LINEAR else "Log"

    def ebin(self):
        if self._etype == ENERGY_TYPE.LINEAR:
            return np.linspace(self._erange[0], self._erange[1], self._nbin + 1)
        else:
            return np.logspace(np.log10(self._erange[0]), np.log10(self._erange[1]), self._nbin + 1)

    def eindex(self, energy: float):
        """
        Get the energy bin index from energy

        :param energy: Point energy (MeV)
        :return: Energy bin index
        """
        if not self._erange[0] <= energy < self._erange[1]:
            raise IndexError("Point out of range")
        return np.argmax(energy < self.ebin()) - 1


class MeshTrack(_TallyContext, _FilterContext, _MeshContext, _FluenceContext):
    def __init__(self, file_name: str):
        _TallyContext.__init__(self)
        _FilterContext.__init__(self)
        _MeshContext.__init__(self)
        _FluenceContext.__init__(self)

        stream = Fortran(file_name)
        _TallyContext._readHeader(self, stream)
        _FilterContext._readFilter(self, stream)
        _MeshContext._readGeometryInfo(self, stream)
        _FluenceContext._readEnergyStructure(self, stream)

        data_1d, err_1d = _TallyContext._readData(stream)
        # Get dimension info
        shape = (self._shape[0], self._shape[1], self._shape[2], self._nbin)
        self.data = data_1d.reshape(shape)
        self.unc = err_1d.reshape(shape)

        stream.close()

    def __add__(self, other: Union[MeshTrack, float, int]):
        if isinstance(other, _TallyContext) and not isinstance(other, MeshTrack):
            raise TypeError
        _MeshContext._operatorCheck(self, other)
        _FluenceContext._operatorCheck(self, other)
        new = deepcopy(self)
        new._add(other)
        new._combine(other)
        return new

    def __sub__(self, other: Union[MeshTrack, float, int]):
        if isinstance(other, _TallyContext) and not isinstance(other, MeshTrack):
            raise TypeError
        _MeshContext._operatorCheck(self, other)
        _FluenceContext._operatorCheck(self, other)
        new = deepcopy(self)
        new._sub(other)
        new._combine(other)
        return new

    def __mul__(self, other: Union[MeshTrack, float, int]):
        _MeshContext._operatorCheck(self, other)
        _FluenceContext._operatorCheck(self, other)
        new = deepcopy(self)
        new._mul(other)
        new._combine(other)
        return new

    def __truediv__(self, other: Union[MeshTrack, float, int]):
        _MeshContext._operatorCheck(self, other)
        _FluenceContext._operatorCheck(self, other)
        new = deepcopy(self)
        new._truediv(other)
        new._combine(other)
        return new

    def __getitem__(self, index):
        index = Slicer(4)[index]
        new = deepcopy(self)
        new._setMeshBoundary(index[:3])
        new._setEnergyBoundary(index[3])
        new.data = self.data[index]
        new.unc  = self.unc[index]
        return new

    def summary(self):
        message = ""
        message += _TallyContext._summary(self)
        message += _FilterContext._summary(self)
        message += _MeshContext.__repr__(self)
        message += _FluenceContext._summary(self)
        return message

    def write(self, file_name: str):
        stream = Fortran(file_name, mode='w')
        self._writeHeader(stream)
        self._writeFilter(stream)
        self._writeGeometryInfo(stream)
        self._writeEnergyStructure(stream)
        self._writeData(stream)
        stream.close()

    def extract(self) -> MeshDensity:
        """
        Get the merged MeshDensity tally
        All energy channels are collapsed to a single value

        :return: MeshDensity tally
        """
        new = MeshDensity('', mode='w')
        new.transform(self)

        new._shape[0] = self.shape()[0]
        new._shape[1] = self.shape()[1]
        new._shape[2] = self.shape()[2]

        shape = self.data.shape

        new.data = np.zeros((shape[0], shape[1], shape[2]), dtype=np.float32)
        new.unc  = np.zeros((shape[0], shape[1], shape[2]), dtype=np.float32)

        for i in range(self.data.shape[-1]):
            other = MeshDensity('', mode='w')
            other.transform(self)

            other._shape[0] = self.shape()[0]
            other._shape[1] = self.shape()[1]
            other._shape[2] = self.shape()[2]

            other.data = self.data[:, :, :, i]
            other.unc  = self.unc[:, :, :, i]

            new = new + other

        new.name = self.name
        new.part = self.part
        new.unit = self.unit
        return new

    def __repr__(self):
        return self.summary()


class MeshDensity(_TallyContext, _FilterContext, _MeshContext):
    def __init__(self, file_name: str, mode="r"):
        _TallyContext.__init__(self)
        _FilterContext.__init__(self)
        _MeshContext.__init__(self)

        if mode == "r":
            stream = Fortran(file_name)
            _TallyContext._readHeader(self, stream)
            _FilterContext._readFilter(self, stream)
            _MeshContext._readGeometryInfo(self, stream)

            data_1d, err_1d = super()._readData(stream)
            # Get dimension info
            mesh = self.shape()
            shape = (mesh[0], mesh[1], mesh[2])
            self.data = data_1d.reshape(shape)
            self.unc = err_1d.reshape(shape)
            stream.close()

    def __add__(self, other: Union[MeshDensity, float, int]):
        if isinstance(other, _TallyContext) and not isinstance(other, MeshDensity):
            raise TypeError
        _MeshContext._operatorCheck(self, other)
        new = deepcopy(self)
        new._add(other)
        new._combine(other)
        return new

    def __sub__(self, other: Union[MeshDensity, float, int]):
        if isinstance(other, _TallyContext) and not isinstance(other, MeshDensity):
            raise TypeError
        _MeshContext._operatorCheck(self, other)
        new = deepcopy(self)
        new._sub(other)
        new._combine(other)
        return new

    def __mul__(self, other: Union[MeshDensity, float, int]):
        _MeshContext._operatorCheck(self, other)
        new = deepcopy(self)
        new._mul(other)
        new._combine(other)
        return new

    def __truediv__(self, other: Union[MeshDensity, float, int]):
        _MeshContext._operatorCheck(self, other)
        new = deepcopy(self)
        new._truediv(other)
        new._combine(other)
        return new

    def __getitem__(self, index):
        index = Slicer(3)[index]
        new = deepcopy(self)
        new._setMeshBoundary(index[:])
        new.data = self.data[index]
        new.unc = self.unc[index]
        return new

    def summary(self):
        message = ""
        message += _TallyContext._summary(self)
        message += _FilterContext._summary(self)
        message += _MeshContext.__repr__(self)
        return message

    def setData(self, data: np.ndarray):
        self.data   = data
        self._shape = np.array(data.shape, dtype=int)
        if not len(self._shape) == 3:
            raise ValueError("'data' must be 3-D array")

    def setErr(self, unc: np.ndarray):
        self.unc = unc
        shape = tuple(unc.shape)
        if not len(self._shape) == 3:
            raise ValueError("'data' must be 3-D array")
        for i in range(3):
            if self._shape[i] != shape[i]:
                raise ValueError("'data' and 'unc' dimension must be same")

    def image(self, pos: float, axis: int = 2):
        """
        Get the 2-D image and extent from mesh

        :param pos: Slice coordinate (cm)
        :param axis: Slicing axis, must be in range [0,2]
        :return: 2-D value image (np.ndarray), 2-D uncertainty image (np.ndarray), extent (tuple)
        """
        slicer, extent = self._imageSlice(pos, axis)
        img_val = self.data[slicer]
        img_unc = self.unc[slicer]

        if axis == 1:  # Permutation
            img_val = np.transpose(img_val)
            img_unc = np.transpose(img_unc)

        return img_val, img_unc, extent

    def write(self, file_name: str):
        stream = Fortran(file_name, mode='w')
        self._writeHeader(stream)
        self._writeFilter(stream)
        self._writeGeometryInfo(stream)
        self._writeData(stream)
        stream.close()

    def __repr__(self):
        return self.summary()


class Cross(_TallyContext, _FilterContext, _FluenceContext):
    def __init__(self, file_name: str, mode="r"):
        _TallyContext.__init__(self)
        _FilterContext.__init__(self)
        _FluenceContext.__init__(self)

        if mode == "r":
            stream = Fortran(file_name)
            _TallyContext._readHeader(self, stream)
            _FilterContext._readFilter(self, stream)
            _FluenceContext._readEnergyStructure(self, stream)

            data_1d, err_1d = _TallyContext._readData(stream)
            # Get dimension info
            shape = self._nbin
            self.data = data_1d.reshape(shape)
            self.unc  = err_1d.reshape(shape)

            stream.close()

    def __add__(self, other: Union[Cross, float, int]):
        if isinstance(other, _TallyContext) and not isinstance(other, Cross):
            raise TypeError
        _FluenceContext._operatorCheck(self, other)
        new = deepcopy(self)
        new._add(other)
        new._combine(other)
        return new

    def __sub__(self, other: Union[Cross, float, int]):
        if isinstance(other, _TallyContext) and not isinstance(other, Cross):
            raise TypeError
        _FluenceContext._operatorCheck(self, other)
        new = deepcopy(self)
        new._sub(other)
        return new

    def __mul__(self, other: Union[Cross, float, int]):
        _FluenceContext._operatorCheck(self, other)
        new = deepcopy(self)
        new._mul(other)
        return new

    def __truediv__(self, other: Union[Cross, float, int]):
        _FluenceContext._operatorCheck(self, other)
        new = deepcopy(self)
        new._truediv(other)
        return new

    def __getitem__(self, index):
        index = Slicer(1)[index]
        new = deepcopy(self)
        new._setEnergyBoundary(index[-1])
        new.data = self.data[index]
        new.unc  = self.unc[index]
        return new

    def summary(self):
        message = ""
        message += _TallyContext._summary(self)
        message += _FilterContext._summary(self)
        message += _FluenceContext._summary(self)
        return message

    def write(self, file_name: str):
        stream = Fortran(file_name, mode='w')
        self._writeHeader(stream)
        self._writeFilter(stream)
        self._writeEnergyStructure(stream)
        self._writeData(stream)
        stream.close()

    def __repr__(self):
        return self.summary()


class Track(_TallyContext, _FilterContext, _FluenceContext):
    def __init__(self, file_name: str):
        _TallyContext.__init__(self)
        _FilterContext.__init__(self)
        _FluenceContext.__init__(self)

        stream = Fortran(file_name)
        _TallyContext._readHeader(self, stream)
        _FilterContext._readFilter(self, stream)
        _FluenceContext._readEnergyStructure(self, stream)

        data_1d, err_1d = _TallyContext._readData(stream)
        # Get dimension info
        shape = self._nbin
        self.data = data_1d.reshape(shape)
        self.unc  = err_1d.reshape(shape)

        stream.close()

    def __add__(self, other: Union[Track, float, int]):
        if isinstance(other, _TallyContext) and not isinstance(other, Track):
            raise TypeError
        _FluenceContext._operatorCheck(self, other)
        new = deepcopy(self)
        new._add(other)
        new._combine(other)
        return new

    def __sub__(self, other: Union[Track, float, int]):
        if isinstance(other, _TallyContext) and not isinstance(other, Track):
            raise TypeError
        _FluenceContext._operatorCheck(self, other)
        new = deepcopy(self)
        new._sub(other)
        new._combine(other)
        return new

    def __mul__(self, other: Union[Track, float, int]):
        _FluenceContext._operatorCheck(self, other)
        new = deepcopy(self)
        new._mul(other)
        new._combine(other)
        return new

    def __truediv__(self, other: Union[Track, float, int]):
        _FluenceContext._operatorCheck(self, other)
        new = deepcopy(self)
        new._truediv(other)
        new._combine(other)
        return new

    def __getitem__(self, index):
        index = Slicer(1)[index]
        new   = deepcopy(self)
        new._setEnergyBoundary(index[-1])
        new.data = self.data[index]
        new.unc  = self.unc[index]
        return new

    def summary(self):
        message = ""
        message += _TallyContext._summary(self)
        message += _FilterContext._summary(self)
        message += _FluenceContext._summary(self)
        return message

    def write(self, file_name: str):
        stream = Fortran(file_name, mode='w')
        self._writeHeader(stream)
        self._writeFilter(stream)
        self._writeEnergyStructure(stream)
        self._writeData(stream)
        stream.close()

    def __repr__(self):
        return self.summary()


class Density(_TallyContext, _FilterContext):
    def __init__(self, file_name: str):
        _TallyContext.__init__(self)
        _FilterContext.__init__(self)

        stream = Fortran(file_name)
        _TallyContext._readHeader(self, stream)
        _FilterContext._readFilter(self, stream)

        data_1d, err_1d = _TallyContext._readData(stream)
        # Get dimension info
        self.data = data_1d[0]
        self.unc  = err_1d[0]

        stream.close()

    def __add__(self, other: Union[Density, float, int]):
        if isinstance(other, _TallyContext) and not isinstance(other, Density):
            raise TypeError
        new = deepcopy(self)
        new._add(other)
        return new

    def __sub__(self, other: Union[Density, float, int]):
        if isinstance(other, _TallyContext) and not isinstance(other, Density):
            raise TypeError
        new = deepcopy(self)
        new._sub(other)
        return new

    def __mul__(self, other: Union[Density, float, int]):
        new = deepcopy(self)
        new._mul(other)
        return new

    def __truediv__(self, other: Union[Density, float, int]):
        new = deepcopy(self)
        new._truediv(other)
        return new

    def summary(self):
        message = ""
        message += _TallyContext._summary(self)
        message += _FilterContext._summary(self)
        return message

    def write(self, file_name: str):
        stream = Fortran(file_name, mode='w')
        self._writeHeader(stream)
        self._writeFilter(stream)
        self._writeData(stream)
        stream.close()

    def __repr__(self):
        return self.summary()


class Detector(Cross):
    def __init__(self, file_name: str):
        Cross.__init__(self, file_name)

    def __add__(self, other: Union[Detector, float, int]):
        if isinstance(other, _TallyContext) and not isinstance(other, Detector):
            raise TypeError
        _FluenceContext._operatorCheck(self, other)
        new = deepcopy(self)
        new._add(other)
        return new

    def __sub__(self, other: Union[Detector, float, int]):
        if isinstance(other, _TallyContext) and not isinstance(other, Detector):
            raise TypeError
        _FluenceContext._operatorCheck(self, other)
        new = deepcopy(self)
        new._sub(other)
        return new

    def __mul__(self, other: Union[Detector, float, int]):
        _FluenceContext._operatorCheck(self, other)
        new = deepcopy(self)
        new._mul(other)
        return new

    def __truediv__(self, other: Union[Detector, float, int]):
        _FluenceContext._operatorCheck(self, other)
        new = deepcopy(self)
        new._truediv(other)
        return new

    def summary(self):
        message = ""
        message += _TallyContext._summary(self)
        message += _FluenceContext._summary(self)
        return message


class PhaseSpace:
    def __init__(self, file_name: str = '', max_counts=-1):
        self._capacity = 100
        self._size     = 0
        self._ps       = np.empty(self._capacity, dtype=DTYPE_PHASE_SPACE)
        if file_name == '':
            pass
        else:
            stream = UnformattedFortran(file_name, recl=36)
            bytes_array = stream.read(0, max_counts)
            stream.close()
            ps_temp = np.frombuffer(bytes_array, dtype=DTYPE_PHASE_SPACE)
            self.append(ps_temp)

    def reserve(self, size: int):
        if size > self._capacity:
            ps_temp = self._ps
            self._ps = np.empty(size, dtype=DTYPE_PHASE_SPACE)
            self._ps[:self._capacity] = ps_temp[:self._capacity]
            self._capacity = size

    def resize(self, size: int):
        if size > self._capacity:
            self.reserve(size)
        if size > self._size:
            self._size = size

    def data(self):
        return self._ps[:self._size]

    def append(self, arr: np.ndarray):
        if arr.dtype != DTYPE_PHASE_SPACE:
            raise ValueError("'arr' dtype must be 'DTYPE_PHASE_SPACE'")
        arr_1d = arr.flatten()

        while self._capacity < self._size + len(arr_1d):
            self.reserve(self._capacity * 2)

        len_origin = self._size
        self.resize(len_origin + len(arr_1d))
        self._ps[len_origin:self._size] = arr_1d
        return

    def write(self, file_name: str):
        stream = UnformattedFortran(file_name, mode="w", recl=36)
        stream.write(self.data())
        stream.close()

    def summary(self):
        ps = self.data()
        total_weight = np.sum(ps['wee'])
        total_count = len(ps)
        message = ""
        message += fieldFormat("Total counts", total_count)
        message += fieldFormat("Total weights", total_weight)

        pid_list = np.unique(ps['pid'])
        for pid in pid_list:
            part = PID_TO_PNAME[pid]
            weight = np.sum(ps[ps['pid'] == pid]['wee'])
            count = len(ps[ps['pid'] == pid])
            message += nameFormat(part)
            message += fieldFormat("counts", count)
            message += fieldFormat("weights", weight)

        return message

    def __repr__(self):
        return self.summary()
