import os
import numpy as np

from rt2.photon import Photon
from rt2.particle import PID_TO_PNAME
from rt2.fortran import Fortran


class AtomList:
    __home  = os.path.join(os.environ['MCRT2_HOME'], 'resource\\material')
    __atoms = {}

    @staticmethod
    def symbol(z: int):
        if not AtomList.__atoms:
            AtomList.__getData()
        return AtomList.__atoms[z][0]

    @staticmethod
    def name(z: int):
        if not AtomList.__atoms:
            AtomList.__getData()
        return AtomList.__atoms[z][1]

    @staticmethod
    def __getData():
        AtomList.__atoms = {}
        with open(os.path.join(AtomList.__home, 'elements.dat')) as file:
            for line in file:
                items = line.split()
                z      = int(items[0])
                symbol = items[1]
                name   = items[2]
                AtomList.__atoms[z] = (symbol, name)


class Compound:

    UNIT_LENGTH = 1.0  # cm

    def __init__(self, is_vacuum: bool):

        self.is_vacuum = is_vacuum
        self.photon = None


def reader(file_name: str):
    compound_list = []
    stream = Fortran(file_name)
    stream.init()
    # unit length
    Compound.UNIT_LENGTH = stream.read(np.float64)[0]
    # material dimension
    n_mat, n_region = stream.read(np.int32)
    if not n_mat:
        return compound_list
    # read vacuum indicator
    is_vacuum = stream.read(np.int32)
    if len(is_vacuum) != n_mat:
        raise BufferError("Vacuum indicator dimension mismatched")
    for vac in is_vacuum:
        compound_list += [Compound(bool(vac))]
    # read assign information (skip)
    for i in range(n_region):
        stream.read(np.byte)
        stream.read(np.int32)
    # read photon data
    if not len(compound_list):
        return compound_list
    mode = stream.read(np.int32)[0]
    Photon.mode(bool(mode))
    if mode:
        rayleigh = stream.read(np.int32)[0]
        Photon.rayleighMode(bool(rayleigh))
        if rayleigh:
            ff_domain = stream.read(np.float64)
            Photon.formFactorDomain(ff_domain)
        nrc_pair = stream.read(np.int32)[0]
        Photon.nrcPairMode(bool(nrc_pair))
        if nrc_pair:
            nrcp_coeff = stream.read(np.float64)
            if len(nrcp_coeff) != 3:
                raise BufferError("Nrcp coefficient dimension mismatched")
            Photon.nrcpCoeff(nrcp_coeff)
        for compound in compound_list:
            if compound.is_vacuum:
                continue
            compound.photon = Photon(stream)

    return compound_list


def pidToName(pid: int) -> str:
    if pid >= 1000:  # heavy ion ZA
        z = pid // 1000
        a = pid  % 1000
        return '{}-{}'.format(AtomList.symbol(z), a)
    else:
        return PID_TO_PNAME[pid]
