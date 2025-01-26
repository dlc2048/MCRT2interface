from __future__ import annotations

from collections.abc import Iterable
from enum import Enum
from copy import deepcopy

import numpy as np

from rt2.algorithm import AliasTableEGS
from rt2.fortran import Fortran
from rt2.particle import PTYPE
from rt2.transport import rotateVector, sampleAzimuthalAngle
from rt2 import constant


class ITYPE(Enum):
    COHE = 0
    COMPT = 1
    PAIR = 2
    PHOTO = 3


class Photon:
    def __init__(self, file_name: str):
        # material
        self._vacuum = False
        self._density = 0.0
        # energy
        self._nbin = 0
        self._ge = np.empty(2, dtype=np.float64)
        # cross section data
        self._gmfp = np.empty((0, 2), dtype=np.float64)
        self._gbr1 = np.empty((0, 2), dtype=np.float64)
        self._gbr2 = np.empty((0, 2), dtype=np.float64)
        self._cohe = np.empty((0, 2), dtype=np.float64)
        self._rela = np.empty((0, 2), dtype=np.float64)
        # rayleigh data
        self._do_rayleigh = True
        self._ff_domain = np.empty(0, dtype=np.float64)
        self._pmax = np.empty((0, 2), dtype=np.float64)
        self._fcum = np.empty(0, dtype=np.float64)
        self._b_array = np.empty(0, dtype=np.float64)
        self._c_array = np.empty(0, dtype=np.float64)
        self._i_array = np.empty(0, dtype=np.int32)
        # NRC pair alias data
        self._use_nrc_pair = True
        self._nrcp_coeff = np.empty(3, dtype=np.float64)
        self._pair_nrc = None
        # Bethe-Heitler pair data
        self._dl = np.empty((6, 4), dtype=np.float64)
        self._delcm = 0.e0
        self._bpar = np.empty(2, dtype=np.float64)
        # read data
        stream = Fortran(file_name, mode="r")
        self._read(stream)
        stream.close()

    def _read(self, stream: Fortran):
        # material
        self._vacuum = bool(stream.read(np.int32)[0])
        self._density = stream.read(np.float64)[0]
        # energy
        self._nbin = stream.read(np.int32)[0]
        ge = stream.read(np.float64)
        if len(ge) != 2:
            raise BufferError("GE dimension mismatched")
        self._ge = ge
        # cross section data
        out_list = [None, None, None, None, None]
        for i in range(len(out_list)):
            buffer = stream.read(np.float64)
            if len(buffer) % 2:
                raise BufferError("XS dimension shape error")
            nbin = len(buffer) // 2
            if nbin != self._nbin:
                raise BufferError("XS dimension mismatched")
            out_list[i] = np.reshape(buffer, (nbin, 2))
        self._gmfp = out_list[0]
        self._gbr1 = out_list[1]
        self._gbr2 = out_list[2]
        self._cohe = out_list[3]
        self._rela = out_list[4]

        # rayleigh data
        self._do_rayleigh = bool(stream.read(np.int32)[0])
        if self._do_rayleigh:
            self._ff_domain = stream.read(self._ff_domain.dtype)
            buffer = stream.read(np.float64)
            if len(buffer) % 2:
                raise BufferError("Rayleigh pmax dimension reshape error")
            nbin = len(buffer) // 2
            self._pmax = np.reshape(buffer, (nbin, 2))
            self._fcum = stream.read(self._fcum.dtype)
            self._b_array = stream.read(self._b_array.dtype)
            self._c_array = stream.read(self._c_array.dtype)
            self._i_array = stream.read(self._i_array.dtype)

        # nrcp pair data
        self._use_nrc_pair = bool(stream.read(np.int32)[0])
        if self._use_nrc_pair:
            self._nrcp_coeff = stream.read(np.float64)
            dy, dx = stream.read(np.int32)
            shape_list = (dx + 1, (dy, dx + 1), (dy, dx), (dy, dx))
            dtype_list = (np.float64, np.float64, np.int32, np.float64)
            out = [None] * 4
            for i in range(4):
                buffer = stream.read(dtype_list[i])
                out[i] = buffer.reshape(shape_list[i])
            self._pair_nrc = AliasTableEGS(*out)
        # Bethe-Heitler pair data
        buffer = stream.read(np.float64)
        self._dl = np.reshape(buffer, self._dl.shape)
        buffer = stream.read(np.float64)
        self._delcm = buffer[0]
        buffer = stream.read(np.float64)
        self._bpar[0] = buffer[0]
        self._bpar[1] = buffer[1]

    def _interp(self, energy: float | Iterable):
        if isinstance(energy, Iterable):
            energy = np.array(np.copy(energy))
        gle = np.log(energy)
        lgle = np.asarray((self._ge[0] + self._ge[1] * gle), dtype=int)
        if np.min(lgle) < 0:
            raise ValueError("Entered energy out of range")
        return gle, lgle

    def _gbr1Branch(self, energy: float | Iterable):
        gle, lgle = self._interp(energy)
        return self._gbr1[lgle, 0] + self._gbr1[lgle, 1] * gle

    def _gbr2Branch(self, energy: float | Iterable):
        gle, lgle = self._interp(energy)
        return self._gbr2[lgle, 0] + self._gbr2[lgle, 1] * gle

    def _coheBranch(self, energy: float | Iterable):
        gle, lgle = self._interp(energy)
        return self._cohe[lgle, 0] + self._cohe[lgle, 1] * gle

    # Cross Section #

    def xsTotal(self, energy: float | Iterable):
        """Calculate total macroscopic cross section"""
        gle, lgle = self._interp(energy)
        return self._gmfp[lgle, 0] + self._gmfp[lgle, 1] * gle

    def relaxation(self, energy: float | Iterable):
        """Calculate mean relaxation energy (MeV)"""
        gle, lgle = self._interp(energy)
        return self._rela[lgle, 0] + self._rela[lgle, 1] * gle

    def xsCohe(self, energy: float | Iterable):
        """Calculate Rayleigh scattering macroscopic cross section"""
        branch = (1.e0 - self._coheBranch(energy))
        return branch * self.xsTotal(energy)

    def xsPair(self, energy: float | Iterable):
        """Calculate pair production macroscopic cross section"""
        branch = self._coheBranch(energy) * self._gbr1Branch(energy)
        return branch * self.xsTotal(energy)

    def xsCompt(self, energy: float | Iterable):
        """Calculate Compton scattering macroscopic cross section"""
        branch = self._coheBranch(energy) * (self._gbr2Branch(energy) - self._gbr1Branch(energy))
        return branch * self.xsTotal(energy)

    def xsPhoto(self, energy: float | Iterable):
        """Calculate Photoelectric effect macroscopic cross section"""
        branch = self._coheBranch(energy) * (1.e0 - self._gbr2Branch(energy))
        return branch * self.xsTotal(energy)

    # Sampling Interaction Branch #

    def branch(self, energy: float | Iterable):
        """Sampling interaction type branch"""

        if isinstance(energy, Iterable):
            if isinstance(energy, np.ndarray):
                vfunc = np.vectorize(self.branch)
                return vfunc(energy)
            else:
                return list(map(self.branch, energy))

        cohfac = self._coheBranch(energy)
        gbr1 = self._gbr1Branch(energy)
        gbr2 = self._gbr2Branch(energy)

        rand = np.random.random()
        if rand < 1.0 - cohfac:
            return ITYPE.COHE

        rand = np.random.random()
        if rand <= gbr1:
            return ITYPE.PAIR
        elif rand < gbr2:
            return ITYPE.COMPT
        else:
            return ITYPE.PHOTO

    # Secondary Sampling #

    def comptonAngle(self, energy: float | Iterable):
        """Sampling direction cosine of secondary photon after the Compton scattering"""

        if isinstance(energy, Iterable):
            if isinstance(energy, np.ndarray):
                vfunc = np.vectorize(self.comptonAngle)
                return vfunc(energy)
            else:
                return list(map(self.comptonAngle, energy))

        ko = energy / constant.ELECTRON_REST_MASS
        broi = 1 + 2 * ko
        bro = 1 / broi

        while True:
            r1 = np.random.random()
            r2 = np.random.random()
            if ko > 2:
                r3 = np.random.random()
                broi2 = broi * broi
                alph1 = np.log(broi)
                alph2 = ko * (broi + 1) * bro * bro
                alpha = alph1 + alph2

                if r1 * alpha < alph1:
                    br = np.exp(alph1 * r2) * bro
                else:
                    br = np.sqrt(r2 * broi2 + 1 - r2) * bro

                costhe = (1 - br) / (ko * br)
                sinthe = max(0, costhe * (2 - costhe))
                aux = 1 + br ** 2
                rejf3 = aux - br * sinthe

                if r3 * aux < rejf3:
                    break
            else:
                bro1 = 1 - bro
                rejmax = broi + bro

                br = bro + bro1 * r1
                costhe = (1 - br) / (ko * br)
                sinthe = max(0, costhe * (2 - costhe))
                rejf3 = 1 + br ** 2 - br * sinthe
                if r2 * br * rejmax < rejf3:
                    break

        costhe = 1 - costhe
        return costhe

    def rayleighAngle(self, energy: float | Iterable):
        """Sampling rayleigh scattering angle"""

        if isinstance(energy, Iterable):
            if isinstance(energy, np.ndarray):
                vfunc = np.vectorize(self.rayleighAngle)
                return vfunc(energy)
            else:
                return list(map(self.rayleighAngle, energy))

        gle, lgle = self._interp(energy)
        pmax_val = self._pmax[lgle, 0] + self._pmax[lgle, 1] * gle
        xmax = constant.HC_I * energy
        dwi = len(self._fcum) - 1

        while True:
            while True:
                r1 = np.random.random()

                temp = r1 * pmax_val
                ibin = int(temp * dwi)
                ib = self._i_array[ibin]
                next_ib = self._i_array[ibin + 1]

                if next_ib > ib:
                    while True:
                        if temp < self._fcum[ib] or ib >= len(self._fcum) - 2:
                            break
                        ib += 1

                temp = (temp - self._fcum[ib]) * self._c_array[ib]
                xv = self._ff_domain[ib] * np.exp(np.log(1 + temp) * self._b_array[ib])
                if xv < xmax:
                    break

            r2 = np.random.random()
            xv /= energy
            costhe = 1 - constant.TWICE_HC2 * xv * xv
            if 2 * r2 < 1 + costhe * costhe:
                break

        return costhe

    def rayleigh(self, primary: Particle):
        cosp, sinp = sampleAzimuthalAngle()
        cost = self.rayleighAngle(primary.energy())
        sint = np.sqrt(1.0 - cost**2)
        direction = deepcopy(primary.direction())
        rotateVector(direction, sint, cost, sinp, cosp)
        return Particle(primary.position(), direction, primary.energy(), PTYPE.PHOTON)

    def pairEnergy(self, energy: float | Iterable):
        """Sampling energy of secondary electron of pair production"""

        if isinstance(energy, Iterable):
            if isinstance(energy, np.ndarray):
                vfunc = np.vectorize(self.pairEnergy)
                return vfunc(energy)
            else:
                return list(map(self.pairEnergy, energy))

        pese2 = 0.0
        k = energy / constant.ELECTRON_REST_MASS

        do_nrc_pair = False
        if self._use_nrc_pair:  # NRC pair
            nrcp_emin = self._nrcp_coeff[0]
            nrcp_emax = self._nrcp_coeff[1]
            nrcp_dlei = self._nrcp_coeff[2]
            if k < nrcp_emax:
                do_nrc_pair = True
                ibin = 0
                if k >= nrcp_emin:
                    abin = np.log((k - 2) / (nrcp_emin - 2)) * nrcp_dlei
                    ibin = int(abin)
                    abin -= ibin
                    rbin = np.random.random()
                    if rbin < abin:
                        ibin += 1
                xx = self._pair_nrc.sample(ibin)
                pese2 = constant.ELECTRON_REST_MASS * (1 + xx * (k - 2))

        if not do_nrc_pair:  # BH pair
            if energy <= 2.1:  # Low energy approximation
                rand = np.random.random()
                pese2 = constant.ELECTRON_REST_MASS + 0.5 * rand * (energy - 2 * constant.ELECTRON_REST_MASS)
            else:
                if energy < 50:  # BH without Coulomb correction
                    l0 = 0
                    l1 = l0 + 1
                    delta = 4 * self._delcm / energy
                    if delta < 1:
                        amax = self._dl[0, l0] + delta * (self._dl[1, l0] + delta * self._dl[2, l0])
                        bmax = self._dl[0, l1] + delta * (self._dl[1, l1] + delta * self._dl[2, l1])
                    else:
                        aux2 = np.log(delta + self._dl[5, l0])
                        amax = self._dl[3, l0] + self._dl[4, l0] * aux2
                        bmax = self._dl[3, l1] + self._dl[4, l1] * aux2
                    aux1 = 1 - 2 * constant.ELECTRON_REST_MASS / energy
                    aux1 *= aux1
                    aux1 *= amax / 3
                    aux1 /= (bmax + aux1)
                else:  # BH Coulomb-corrected
                    l0 = 2
                    amax = self._dl[0, l0]
                    bmax = self._dl[0, l0 + 1]
                    aux1 = self._bpar[1] * (1 - self._bpar[0] * constant.ELECTRON_REST_MASS / k)

                de10 = energy * self._delcm
                e_avail = energy - 2 * constant.ELECTRON_REST_MASS

                while True:
                    rand1, rand2, rand3 = np.random.random(3)
                    if rand1 > aux1:  # use the uniform part
                        br = 0.5 * rand2
                        rejmax = bmax
                        l1 = l0 + 1
                    else:  # use the (br-0.5)**2 part
                        br = 0.5 - 0.5 * rand2 ** (1 / 3)
                        rejmax = amax
                        l1 = l0

                    e_minus = br * e_avail + constant.ELECTRON_REST_MASS
                    e_plus = energy - e_minus
                    delta = de10 / (e_minus * e_plus)
                    if delta < 1:
                        rejf = self._dl[0, l1] + delta * (self._dl[1, l1] + delta * self._dl[2, l1])
                    else:
                        rejf = self._dl[3, l1] + self._dl[4, l1] * np.log(delta + self._dl[5, l1])

                    if rand3 * rejmax <= rejf:
                        break

                pese2 = e_minus

        # symmetry
        rand = np.random.random()
        if rand > 0.5:
            pese2 = energy - pese2

        return pese2
