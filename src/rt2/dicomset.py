"""
RT2 Dicom preprocessor and interface
"""

__author__ = "Chang-Min Lee"
__copyright__ = "Copyright 2022, Seoul National University"
__credits__ = ["Chang-Min Lee"]
__license__ = None
__maintainer__ = "Chang-Min Lee"
__email__ = "dlc2048@postech.ac.kr"
__status__ = "Production"

import pydicom
import numpy as np
import os
from tqdm import tqdm

from rt2.algorithm import Affine
from rt2.scoring import _MeshContext
from rt2.fortran import Fortran


def dicomAffine(dcm: pydicom.dataset.Dataset) -> np.ndarray:
    """
    Calculate 4x4 affine matrix from dicom slice

    :param dcm: pydicom slice object
    :return: 4x4 affine matrix, numpy ndarray form
    """
    f11, f21, f31 = dcm.ImageOrientationPatient[3:]
    f12, f22, f32 = dcm.ImageOrientationPatient[:3]

    dr, dc = dcm.PixelSpacing
    sx, sy, sz = dcm.ImagePositionPatient
    dh = dcm.SliceThickness

    return np.array(
        [
            [f11 * dr, f12 * dc, 0, sx],
            [f21 * dr, f22 * dc, 0, sy],
            [f31 * dr, f32 * dc, dh, sz],
            [0, 0, 0, 1]
        ]
    )


class DicomBinary(_MeshContext):
    def __init__(self, file_name: str, from_binary: bool = True):
        _MeshContext.__init__(self)
        self._proportionCoeff = np.empty(2, dtype=float)
        self._data = np.empty(0, dtype=np.uint16)
        self._houns = np.empty(0, dtype=np.int32)

        if not from_binary:
            return

        stream = Fortran(file_name, mode="r")
        self._proportionCoeff = stream.read(float)
        _MeshContext._readGeometryInfo(self, stream)
        arr_ct_1d = stream.read(np.uint16)
        self._data = arr_ct_1d.reshape(self.shape())
        stream.close()

        self._unpack()

    def __repr__(self):
        ss = object.__repr__(self)
        ss += "\n"
        ss += _MeshContext.__repr__(self)
        return ss

    def _unpack(self):
        self._houns = (self._data.astype(dtype=np.int32) + self._proportionCoeff[0]) / self._proportionCoeff[1]

    def write(self, file_name: str):
        stream = Fortran(file_name, mode="w")
        stream.write(self._proportionCoeff)
        _MeshContext._writeGeometryInfo(self, stream)
        arr_ct_1d = self._data.flatten()
        stream.write(arr_ct_1d)
        stream.close()

    '''
    def image(self, pos: float, axis: int = 2):
        """
        Get the 2-D image and extent from mesh

        :param pos: Slice coordinate (cm)
        :param axis: Slicing axis, must be in range [0,2]
        :return: 2-D image (np.ndarray), extent (tuple)
        """
        slicer, extent = self._imageSlice(pos, axis)
        img = self.data[slicer]

        if axis == 1:  # Permutation
            img = np.transpose(img)

        return img, extent
    '''

    # def shape(self):
    #     return tuple(self._data.shape)

    def data(self):
        return np.copy(self._data)


class DicomSet(DicomBinary):
    def __init__(self, path: str, verbose=True):
        """
        Import dicom dataset. Every CT image file in 'path' is imported.
        All file are sorted by pydicom '.SliceLocation' attribute.

        :param path: Dicom CT dataset path
        :param verbose: Verbose level
        """
        DicomBinary.__init__(self, "", False)
        self._indexSorting(path, verbose)
        self._open(path, verbose)
        self._unpack()
        
    def _indexSorting(self, path, verbose):
        if verbose:
            print("Sort DICOM dataset")
        dir_dataset = os.path.join(os.getcwd(), path)
        list_file = os.listdir(dir_dataset)
        index_list = []

        rows = None
        columns = None
        series_instance_uid = None

        rr = enumerate(tqdm(list_file)) if verbose else enumerate(list_file)
        for i, file_name in rr:
            try:
                file = pydicom.dcmread(os.path.join(dir_dataset, file_name), force=True)
                modality = file.Modality
            except:
                print("{} is not a dicom file".format(file_name))
            else:
                if modality != 'CT':
                    print("{} is not a CT file".format(file_name))
                else:
                    if series_instance_uid is None:  # First instance
                        series_instance_uid = file.SeriesInstanceUID
                        self._proportionCoeff = np.array(
                            [float(file.RescaleIntercept),
                             float(file.RescaleSlope)], dtype=float)
                        affine = dicomAffine(file) * 0.1  # mm to cm
                        affine[-1, -1] = 1.0  # Diagonal 1
                        Affine.__init__(self, affine)  # Initialize affine matrix
                    elif series_instance_uid != file.SeriesInstanceUID:
                        raise IndexError("CT file {} has different Series Instance UID (0020,000E)".format(file_name))
                    index_list += [[i, int(file.InstanceNumber)]]
                    rows = file.Rows
                    columns = file.Columns

        self._index_arr = np.array(index_list, dtype=np.uint16)
        ind = np.argsort(self._index_arr[:, 1])
        self._index_arr = self._index_arr[ind]  # index_file, index_slice
        n_layer = np.max(self._index_arr[:, 1]) - np.min(self._index_arr[:, 1]) + 1
        self._data = np.zeros((rows, columns, n_layer), dtype=np.uint16)
        self._shape[0] = rows
        self._shape[1] = columns
        self._shape[2] = n_layer
        
    def _open(self, path, verbose):
        if verbose:
            print("Import DICOM dataset")
            rr = tqdm(self._index_arr)
        else:
            rr = self._index_arr
        dir_dataset = os.path.join(os.getcwd(), path)
        layer_offset = np.min(self._index_arr[:, 1])
        for file_index, layer_index in rr:
            file = pydicom.dcmread(os.path.join(dir_dataset,
                                                os.listdir(dir_dataset)[file_index]), force=True)
            self._data[:, :, layer_index - layer_offset] = file.pixel_array[:]
