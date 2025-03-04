import os
import numpy as np

import rt2.constant as const
from rt2.fortran import Fortran
from rt2.singleton import _SingletonTemplate


class NuclearMassData(_SingletonTemplate):
    __file = "resource\\hadron\\walletlifetime.dat"

    def __init__(self):
        self._mass_table = {}

        if 'MCRT2_HOME' not in os.environ:
            assert 'Environment variable "MCRT2_HOME" is missing'
        home  = os.environ['MCRT2_HOME']
        fname = os.path.join(home, NuclearMassData.__file)

        with open(fname) as file:
            for line in file:
                items  = line.split()
                a      = int(items[0])
                z      = int(items[1])
                excess = float(items[2])
                mass   = a * const.ATOMIC_MASS_UNIT + excess - self._electronMass(z)
                self._mass_table[z * 1000 + a] = mass

    @staticmethod
    def _electronMass(z: int) -> float:
        emass   = z * const.MASS_ELECTRON
        binding = 14.4381 * z**2.39 + 1.55468e-6 * z**5.35
        return emass - binding * 1e-6

    def getMass(self, z: int, a: int) -> float:
        za = z * 1000 + a
        if za in self._mass_table.keys():
            return self._mass_table[za]
        else:
            return self._getWeizsaeckerMass(z, a)

    @staticmethod
    def _getWeizsaeckerMass(z: int, a: int) -> float:
        npair   = (a - z) % 2
        zpair   = z % 2
        binding = (
            - 15.67 * a
            + 17.23 * a**(2/3)
            + 93.15 * (a/2 - z)**2 / a
            + 0.6984523 * z**2 * a**(-1/3)
        )
        if npair == zpair:
            binding += (npair + zpair - 1) * 12 / a**0.5
        return z * const.MASS_PROTON + (a - z) * const.MASS_NEUTRON + binding

