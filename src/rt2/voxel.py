from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy

import numpy as np
import nibabel as nib

from rt2.algorithm import Slicer, Affine
from rt2.scoring import _MeshContext
from rt2.fortran import Fortran


class Voxel(_MeshContext):

    def __init__(self, file_name: str | None = None,
                 image: nib.filebasedimages | None = None,
                 shape: np.ndarray | Iterable | None = None):

        _MeshContext.__init__(self)

        self._region = []
        self._data = np.empty(0, dtype=np.uint16)

        if file_name is not None:
            self._read(file_name)
        elif image is not None:
            self._data = image.get_fdata()
            self._shape = np.array(self._data.shape, dtype=int)
            self.transform(Affine(image.affine))
            self.scale(0.1, 0.1, 0.1)  # to cm
        else:
            for i in range(3):
                self._shape[i] = shape[i]
            self._data = np.ones(self._shape, dtype=np.uint16) * np.iinfo('uint16').max

    def __setitem__(self, key, value):
        if isinstance(value, str):
            for i, region in enumerate(self._region):
                if value == region:
                    self._data[key] = i
                    return
            raise ValueError("Entered region name '{}' is not found".format(value))
        elif isinstance(value, int):
            if value >= len(self._region):
                raise ValueError("Region name index out of range")
            self._data[key] = value
        else:
            raise TypeError

    def __getitem__(self, index):
        index = Slicer(3)[index]
        new = deepcopy(self)
        new._setMeshBoundary(index[:])
        new._data = self._data[index]
        return new

    def __repr__(self):
        ss = object.__repr__(self)
        ss += "\n"
        ss += _MeshContext.__repr__(self)
        return ss

    def _read(self, file_name: str):
        stream = Fortran(file_name, mode="r")
        _MeshContext._readGeometryInfo(self, stream)
        data_1d = stream.read(np.uint16)
        self._data = np.reshape(np.copy(data_1d), self._shape)
        n_region = stream.read(np.int32)[0]
        for i in range(n_region):
            name_byte = stream.read(np.byte).tostring()
            self._region += [name_byte.decode('utf-8')]
        stream.close()

    def data(self) -> np.ndarray:
        return np.copy(self._data)

    def shape(self) -> tuple:
        return tuple(self._shape)

    def region(self) -> tuple:
        return tuple(self._region)

    def addRegion(self, region_name: str) -> None:
        """
        Add string 'region_name' in region namelist

        :param region_name: Region name
        :return:
        """
        for region in self._region:
            if region_name == region:
                raise ValueError("Entered region name '{}' is duplicated".format(region_name))
        self._region += [region_name]

    def delRegion(self, region_name: str) -> None:
        """
        Delete string 'region_name' from region namelist
        and initialize every voxel assigned to this region as null

        :param region_name: Target region name to delete
        :return:
        """
        target_is_found = False
        for i, region in enumerate(self._region):
            if region_name == region:
                target_is_found = True
                mask_target = self.mask(region_name)
                mask_up = self._data > i
                self._data[mask_target] = np.iinfo('uint16').max
                self._data[mask_up] -= 1
                break

        if not target_is_found:
            raise ValueError("Entered region name '{}' is not found".format(region_name))
        else:
            self._region.remove(region_name)

    def mask(self, region_name: str) -> np.ndarray:
        """
        Get the mask array of target region

        :param region_name: Name of target region
        :return: Numpy mask ndarray
        """
        for i, region in enumerate(self._region):
            if region_name == region:
                return self._data == i
        raise ValueError("Entered region name '{}' is not found".format(region_name))

    def good(self) -> bool:
        """
        Check the completeness of region assignment

        :return: True if complete, false elsewhere
        """
        return (self._data < len(self._region)).all()

    def transpose(self, axes: Iterable | tuple | Iterable[int]) -> None:
        """
        Transpose axes

        :param axes: Axes
        :return:
        """
        if not self.axisAligned():
            raise AttributeError("Cannot transpose since voxel is not axis-aligned")

        self._data = np.transpose(self._data, axes=axes)
        self._shape = self._shape[np.array(axes)]

        mat = self.affine()
        mat_new = np.copy(mat)
        for i in range(3):
            mat_new[i] = mat[axes[i]]
        self._matrix = np.copy(mat_new)

    def flip(self, axis: int):
        """
        Flip axes

        :param axis: Target axis
        :return:
        """
        if not self.axisAligned():
            raise AttributeError("Cannot transpose since voxel is not axis-aligned")
        idx0 = [0, 0, 0]
        idx1 = [0, 0, 0]
        idx1[axis] += self._shape[axis] - 1
        pos0 = self.where(idx0[0], idx0[1], idx0[2])
        pos1 = self.where(idx1[0], idx1[1], idx1[2])
        self._data = np.flip(self._data, axis=axis)
        self._matrix[:, axis] = -self._matrix[:, axis]
        self.translate(pos1[0] - pos0[0], pos1[1] - pos0[1], pos1[2] - pos0[2])

    def align(self):
        """
        Align to reference coordinate

        :return:
        """
        mat = self.affine()
        axes = np.empty(3, dtype=int)
        for i in range(3):
            axes[i] = np.argmax(mat[:, i] != 0)
        self.transpose(axes)
        mat = self.affine()
        for i in range(3):
            if mat[i, i] < 0:
                self.flip(i)

    def image(self, pos: float, axis: int = 2):
        """
        Get the 2-D image and extent from mesh

        :param pos: Slice coordinate (cm)
        :param axis: Slicing axis, must be in range [0,2]
        :return: 2-D value image (np.ndarray), 2-D uncertainty image (np.ndarray), extent (tuple)
        """
        slicer, extent = self._imageSlice(pos, axis)
        img_val = self.data()[slicer]

        if axis == 1:  # Permutation
            img_val = np.transpose(img_val)

        return img_val, extent

    def write(self, file_name: str) -> None:
        """
        Write voxel data to Fortran formatted binary

        :param file_name: Output file name
        :return:
        """
        if not self.good():
            raise IndexError("Some voxels are not assigned by any region")

        # Erase all unused regions
        region_list = self.region()
        for region in region_list:
            mask = self.mask(region)
            if not mask.any():
                self.delRegion(region)

        stream = Fortran(file_name, mode="w")
        _MeshContext._writeGeometryInfo(self, stream)
        stream.write(self._data.flatten())
        stream.write(np.array([len(self._region)], dtype=np.int32))
        for region in self._region:
            stream.write(region)
        stream.close()

