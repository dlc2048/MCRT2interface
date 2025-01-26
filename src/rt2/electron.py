
import os
from enum import Enum

import numpy as np

from rt2.fortran import Fortran
from rt2.algorithm import AliasTableEGS, LogInterpEntry
from rt2 import constant


class BREM_XS_METHOD(Enum):
    BH = 0
    NIST = 1
    NRC = 2


class ElectronXS:

    _electron_home = os.path.join("resource", "electron")
    _msnew_file = "msnew.bin"

    def __init__(self, file_name: str):
        file = Fortran(file_name, mode="r")
        # material
        self._vacuum = bool(file.read(np.int32)[0])
        if self._vacuum:
            self._density = 0.0
            return
        self._is_gas = bool(file.read(np.int32)[0])
        self._density = file.read(np.float64)[0]

        # energy
        self._gpcut = file.read(np.float64)[0]
        self._log_ap = np.log(self._gpcut)
        self._epcut = file.read(np.float64)[0]
        self._eblim = file.read(np.float64)[0]
        self._eulim = file.read(np.float64)[0]
        self._nbin = file.read(np.int32)[0]
        eke = file.read(np.float64)
        self._eke = eke

        # xs & branches
        self.esig = LogInterpEntry(eke, file.read(np.float64))
        self.psig = LogInterpEntry(eke, file.read(np.float64))
        self.ebr1 = LogInterpEntry(eke, file.read(np.float64))
        self.pbr1 = LogInterpEntry(eke, file.read(np.float64))
        self.pbr2 = LogInterpEntry(eke, file.read(np.float64))

        # dedx (soft collision stopping power
        self.ededx = LogInterpEntry(eke, file.read(np.float64))
        self.pdedx = LogInterpEntry(eke, file.read(np.float64))

        # multiple scattering
        self._blcc = file.read(np.float64)[0]
        self._xcc = file.read(np.float64)[0]
        self._esig_max = file.read(np.float64)[0]
        self._psig_max = file.read(np.float64)[0]
        self._e_is_monoton = bool(file.read(np.int32)[0])
        self._p_is_monoton = bool(file.read(np.int32)[0])
        self._ree = file.read(np.float64)
        self._rep = file.read(np.float64)
        self._range_ee = file.read(np.float64)
        self._range_ep = file.read(np.float64)
        self._tmxs = LogInterpEntry(eke, file.read(np.float64))

        # Bethe-Heitler Bremsstrahlung energy data
        dl_linear = file.read(np.float64)
        self._dl = np.reshape(dl_linear, (6, 4))
        self._delcm = file.read(np.float64)[0]

        # NIST & NRC Bremsstrahlung energy data
        self._brem_xs_method = BREM_XS_METHOD(file.read(np.int32)[0])
        if self._brem_xs_method != BREM_XS_METHOD.BH:
            dim = file.read(np.int32)
            dim_pad = np.copy(dim)
            dim_pad[1] += 1
            xdata_1d = file.read(np.float64)
            if xdata_1d.size != dim_pad[0] * dim_pad[1]:
                xdata = np.copy(xdata_1d)
            else:
                xdata = np.reshape(xdata_1d, dim_pad)
            fdata = np.reshape(file.read(np.float64), dim_pad)
            idata = np.reshape(file.read(np.int32), dim)
            wdata = np.reshape(file.read(np.float64), dim)
            self._brem_nrc = AliasTableEGS(xdata, fdata, idata, wdata)
            self._nb_lemin = file.read(np.float64)[0]
            self._nb_dlei = file.read(np.float64)[0]

        # Bremsstrahlung angular distribution
        self._zbrang = file.read(np.float64)[0]
        self._lzbrang = file.read(np.float64)[0]
        self._brem_angle_km = False

        # Spin effect
        self._do_spin = bool(file.read(np.int32)[0])
        if self._do_spin:
            self.etae = LogInterpEntry(eke, file.read(np.float64))
            self.q1ce = LogInterpEntry(eke, file.read(np.float64))
            self.q2ce = LogInterpEntry(eke, file.read(np.float64))
            self.etap = LogInterpEntry(eke, file.read(np.float64))
            self.q1cp = LogInterpEntry(eke, file.read(np.float64))
            self.q2cp = LogInterpEntry(eke, file.read(np.float64))
            self.blcce = LogInterpEntry(eke, file.read(np.float64))
            temp = file.read(np.float64)
            self._b2spin_min = temp[0]
            self._dbeta2i = temp[1]
            self._dleneri = temp[2]
            self._dqq1i = temp[3]
            self._espml = temp[4]
            self._earray = file.read(np.float64)
            shape = (constant.MAXE_SPI1 + 1, constant.MAXQ_SPIN + 1, constant.MAXU_SPIN + 1)
            self._spin_rej_ele = file.read(np.float64).reshape(shape)
            self._spin_rej_pos = file.read(np.float64).reshape(shape)

        file.close()

        # Read Rutherford multiple scattering alias table
        if "MCRT2_HOME" not in os.environ.keys():
            raise Exception("Environment variable 'MCRT2_HOME' is not defined")
        home = os.environ["MCRT2_HOME"]
        self._initMsTable(os.path.join(home, self._electron_home, self._msnew_file))

    def _initMsTable(self, file_name: str):
        file = Fortran(file_name, mode="r")
        dllambi, dqmsi = file.read(np.float64)
        self._dllambi = dllambi
        self._dqmsi = dqmsi

        shape1 = (constant.MAXL_MS + 1, constant.MAXQ_MS + 1, constant.MAXU_MS + 1)
        shape2 = (constant.MAXL_MS + 1, constant.MAXQ_MS + 1, constant.MAXU_MS)
        self._ums = file.read(np.float64).reshape(shape1)
        self._fms = file.read(np.float64).reshape(shape1)
        self._wms = file.read(np.float64).reshape(shape2)
        self._ims = file.read(np.int32).reshape(shape2)
        file.close()
        return

    def vacuum(self):
        return self._vacuum

    def isGas(self):
        return self._is_gas

    def density(self):
        return self._density

    def gpcut(self):
        return self._gpcut

    def epcut(self):
        return self._epcut

    def eblim(self):
        return self._eblim

    def eulim(self):
        return self._eulim

    def bremKM(self, setup: bool):
        self._brem_angle_km = setup

    def brem(self, eke: float):
        """
        Sampling Bremsstrahlung secondary photon
        :param eke: Electron kinetic energy (MeV)
        :return: [x-ray energy (MeV), x-ray direction cosine]
        """
        ene = eke + constant.ELECTRON_REST_MASS
        elke = np.log(eke)

        i = 0 if ene < 50 else 2
        i1 = i + 1

        brmin = self._gpcut / eke
        waux = elke - self._log_ap

        tteie = ene / constant.ELECTRON_REST_MASS
        beta = np.sqrt((tteie - 1) * (tteie + 1)) / tteie
        y2max = 2 * beta * (1 + beta) * tteie * tteie
        y2maxi = 1 / y2max

        if self._brem_xs_method != BREM_XS_METHOD.BH:  # NRC & NIST alias table
            ajj = (waux + self._log_ap - self._nb_lemin) * self._nb_dlei
            jj = int(ajj)
            ajj -= jj

            rand = np.random.random()
            if rand < ajj:
                jj += 1
            br = self._brem_nrc.sample(jj)
            esg = self._gpcut * np.exp(br * waux)
            ese = ene - esg
        else:  # Bethe-Heitler
            while True:
                rand1, rand2 = np.random.random(2)
                br = brmin * np.exp(rand1 * waux)
                esg = eke * br
                ese = ene - esg
                delta = esg / eke / ese * self._delcm
                aux = ese / eke
                if delta < 1:
                    phi1 = self._dl[0, i] + delta * (self._dl[1, i] + delta * self._dl[2, i])
                    phi2 = self._dl[0, i1] + delta * (self._dl[1, i1] + delta * self._dl[3, i1])
                else:
                    phi1 = self._dl[3, i] + self._dl[4, i] * np.log(delta + self._dl[5, i])
                    phi2 = phi1
                rejf = (1 + aux * aux) * phi1 - 2 * aux * phi2 / 3
                if rand2 < rejf:
                    break

        # now 'egs' is Bremsstrahlung x-ray energy
        if self._brem_angle_km:
            z2max = y2max + 1
            z2maxi = np.sqrt(z2max)
            ttese = ese / constant.ELECTRON_REST_MASS
            esedei = ttese / tteie
            rjarg1 = 1 + esedei**2
            rjarg2 = rjarg1 + 2 * esedei
            aux = 2 * ese * tteie / esg
            aux *= aux
            aux1 = aux * self._zbrang
            if aux1 > 10:
                rjarg3 = self._lzbrang + (1 - aux1) / aux1**2
            else:
                rjarg3 = np.log(aux / (1 + aux1))
            rejmax = rjarg1 * rjarg3 - rjarg2
            while True:
                rand1, rand2 = np.random.random(2)
                aux3 = z2maxi / (rand1 + (1 - rand1) * z2maxi)
                rand2 *= aux3 * rejmax
                y2tst = aux3**2 - 1
                y2tst1 = esedei * y2tst / aux3**4
                aux4 = 16 * y2tst1 - rjarg2
                aux5 = rjarg1 - 4 * y2tst1
                if rand2 < aux4 + aux5 * rjarg3:
                    break
                aux2 = np.log(aux / (1 + aux1 / aux3 ** 4))
                rejtst = aux4 + aux5 * aux2
                if rand2 < rejtst:
                    break
        else:
            rand1 = np.random.random()
            y2tst = rand1 / (1 - rand1 + y2maxi)

        cost = 1 - 2 * y2tst * y2maxi

        return esg, cost

    def _range(self, eke: float, is_electron: bool):
        """
        Compute electron/positron range
        :param eke: Electron/positron kinetic energy (MeV)
        :return: Electron/positron CSDA range (cm), restricted stopping power
        """
        elke = np.log(eke)
        lelke = int(self._eke[0] + self._eke[1] * elke)

        if lelke < 0:  # less than eblim
            return 0.0

        # calculate electron range of last energy entry
        # lelkei == lelke
        elkei = (lelke - self._eke[0]) / self._eke[1]
        ekei = np.exp(elkei)

        ededx = self.ededx if is_electron else self.pdedx
        range_e = self._range_ee if is_electron else self._range_ep

        fedep = 1.e0 - ekei / eke
        elkem = 0.5 * (elke + elkei + 0.25 * fedep * fedep * (1.e0 + fedep * (1.0 + 0.875 * fedep)))
        ekem = np.exp(elkem)
        ededxi = 1.0 / ededx[ekem]
        aux = ededx.eval()[lelke, 1] * ededxi
        aux *= (1.0 + 2.0 * aux) * (fedep / (2.0 - fedep)) ** 2 / 6.0
        return range_e[lelke] + fedep * eke * ededxi * (1 + aux)

    def erange(self, eke: float):
        """
        Compute electron range
        :param eke: Electron kinetic energy (MeV)
        :return: Electron CSDA range (cm), restricted stopping power
        """
        return self._range(eke, True)

    def prange(self, eke: float):
        """
        Compute positron range
        :param eke: Positron kinetic energy (MeV)
        :return: Positron CSDA range (cm), restricted stopping power
        """
        return self._range(eke, False)

    def _lossSeg(self, path: float, eke: float, lelke: int, is_electron: bool):
        """
        Computes the electron/positron energy loss due to the restricted stopping power
        Assume that initial and final energy are in the same interpolation bin
        :param path: Electron/positron path length (cm)
        :param eke: Electron/positron kinetic energy (MeV)
        :param lelke: log interpolation index
        :return: Electron/positron energy loss (MeV)
        """
        ededx_entry = self.ededx if is_electron else self.pdedx
        ededx = ededx_entry[eke]
        aux = ededx_entry.eval()[lelke, 1] / ededx
        fedep = ededx * path / eke
        return ededx * path * (1.0 - 0.5 * fedep * aux *
                               (1.0 - 1.0 / 3.0 * fedep * (aux - 1.0 - 0.25 * fedep * (2.0 - aux * (4.0 - aux)))))

    def _loss(self, path: float, eke: float, is_electron: bool):
        """
        Computes the energy loss due to the restricted stopping power
        :param path: Electron/positron path length (cm)
        :param eke: Electron/positron kinetic energy (MeV)
        :return: Electron/positron energy loss (MeV)
        """
        elke = np.log(eke)
        lelke = int(self._eke[0] + self._eke[1] * elke)
        erange = self._range(eke, is_electron)
        range_e = self._range_ee if is_electron else self._range_ep
        tuss = erange - range_e[lelke]

        if tuss >= path:
            return self._lossSeg(path, eke, lelke, is_electron)
        else:
            tuss = erange - path
            if tuss <= 0:
                return eke - self._eblim
            else:
                lelktmp = lelke
                while True:
                    if tuss > range_e[lelktmp - 1]:
                        break
                    lelktmp -= 1
                elktmp = (lelktmp - self._eke[0]) / self._eke[1]
                eketmp = np.exp(elktmp)
                tuss = range_e[lelktmp] - tuss
                return eke - eketmp + self._lossSeg(tuss, eketmp, lelktmp, is_electron)

    def eloss(self, path: float, eke: float):
        """
        Computes the energy loss due to the restricted stopping power
        :param path: Electron path length (cm)
        :param eke: Electron kinetic energy (MeV)
        :return: Electron energy loss (MeV)
        """
        return self._loss(path, eke, True)

    def ploss(self, path: float, eke: float):
        """
        Computes the energy loss due to the restricted stopping power
        :param path: Positron path length (cm)
        :param eke: Positron kinetic energy (MeV)
        :return: Positron energy loss (MeV)
        """
        return self._loss(path, eke, False)

    def _spinRejection(self, elke: float, beta2: float, q1: float, cost: float, is_electron: bool):
        """
        Determine the rejection function due to spin effects for
        :param elke: Log(eke)
        :param beta2: speed (v/c)
        :param q1: 1st MS moment
        :param cost: multiple scattering angle direction cosine
        :param is_electron: true if electron
        :return: rejection function [0, 1]
        """
        if beta2 >= self._b2spin_min:
            ai = (beta2 - self._b2spin_min) * self._dbeta2i
            i = int(ai)
            ai -= i
            i += constant.MAXE_SPIN + 1
        elif elke > self._espml:
            ai = (elke - self._espml) * self._dleneri
            i = int(ai)
            ai -= i
        else:
            i = 0
            ai = -1.0

        rand1 = np.random.random()
        if rand1 < ai:
            i += 1
        qq1 = 2.0 * q1
        qq1 = qq1 / (1.0 + qq1)
        aj = qq1 * self._dqq1i
        j = int(aj)
        if j >= constant.MAXQ_SPIN:
            j = constant.MAXQ_SPIN
        else:
            aj -= j
            rand2 = np.random.random()
            if rand2 < aj:
                j += 1

        xi = np.sqrt(0.5 * (1.0 - cost))
        ak = xi * constant.MAXU_SPIN
        k = int(ak)
        ak -= k

        spin_rej = self._spin_rej_ele if is_electron else self._spin_rej_pos
        return (1.0 - ak) * spin_rej[i, j, k] + ak * spin_rej[i, j, k + 1]

    def _mscatCoeffs(self, path: float, eke: float, is_electron: bool):
        """
        Prepare multiple electron/positron scattering variables
        :param path: Electron/positron track-length
        :param eke: Electron/positron kinetic energy  at the beginning of step
        :return: (lmb, chia2, xi, beta2, xi_corr, elkem)
        """
        eloss = self._loss(path, eke, is_electron)
        ekem = eke - 0.5 * eloss  # mean energy of this step
        p2 = ekem * (ekem + constant.ELECTRON_REST_MASS_D)  # momentum square
        p2i = 1.0 / p2
        chia2 = self._xcc * p2i / (4.0 * self._blcc)  # screening angle
        beta2 = p2 / (p2 + constant.ELECTRON_REST_MASS_SQ)  # beta = v/c
        lmb = path * self._blcc / beta2  # number of elastic scattering

        # Account for energy loss in the MS distribution
        factor = 1.0 / (1.0 + ekem / constant.ELECTRON_REST_MASS_D)
        epsilon = eloss / eke
        epsilon = epsilon / (1.0 - 0.5 * epsilon)
        temp = 0.25 * (1.0 - factor * (1 - factor / 3.0)) * epsilon ** 2
        lmb *= 1.0 + temp

        # Account for spin effect
        if self._do_spin:
            # correction to the screening angle derived from PWA
            etae = self.etae[ekem] if is_electron else self.etap[ekem]
            # correction to the first MS moment due to spin
            xi_corr = self.q1ce[ekem] if is_electron else self.q1cp[ekem]
            ms_corr = self.blcce[ekem]
        else:
            etae = 1.0
            xi_corr = 1.0
            ms_corr = 1.0

        chia2 *= etae
        lmb = lmb / etae / (1.0 + chia2) * ms_corr
        chilog = np.log(1.0 + 1.0 / chia2)
        q1 = 2.0 * chia2 * (chilog * (1.0 + chia2) - 1.0)
        xi = q1 * lmb

        return lmb, chia2, xi, beta2, xi_corr, np.log(ekem)

    def _mscat(self, lmb: float, chia2: float, q1: float, elke: float, beta2: float, is_electron: bool):
        """
        Sample multiple electron/positron scattering angles from the
        exact distribution resulting from elastic scattering
        :param lmb: Number of elastic scattering
        :param chia2: Screening angle
        :param q1: First GS - moment
        :param elke: Log(eke)
        :param beta2: speed at eke in units of c (v/c)^2
        :return: scattering angle cosine
        """
        # Do multiple scattering
        if lmb <= 1:
            return 1.0
        elif lmb <= 13.8:
            rand = np.random.random()
            explambda = np.exp(-lmb)
            if rand < explambda:  # No scattering event
                return 1.0
            wsum = (1.0 + lmb) * explambda
            if rand < wsum:
                while True:
                    rand2 = np.random.random()
                    rand2 = 2.0 * chia2 * rand2 / (1.0 - rand2 + chia2)
                    cost = 1.0 - rand2
                    if not self._do_spin:
                        break
                    rejf = self._spinRejection(elke, beta2, q1, cost, is_electron)
                    if rand2 <= rejf:
                        break
                return cost
        elif lmb > 1e5:
            raise Exception("Electron track length exceeds tmxs limit!")

        # It was a multiple scattering event
        # Sample the angle from the q^(2+) surface
        llmbda = np.log(lmb)
        # first fix lambda bin
        ai = llmbda * self._dllambi
        i = int(ai)
        ai -= i
        rand = np.random.random()
        if rand < ai:
            i += 1

        # fix now q1 bin
        aj = q1 * self._dqmsi
        j = int(aj)
        aj -= j
        rand = np.random.random()
        if rand < aj:
            j += 1
        j = min(constant.MAXQ_MS, max(0, j))

        # calculate omega2
        omega2 = chia2 * (lmb + 4)
        if llmbda < 2.2299:
            omega2 *= (1.347006 + llmbda * (0.209364 - llmbda *
                                            (0.45525 - llmbda * (0.50142 - 0.081234 * llmbda))))
        else:
            omega2 *= (-2.77164 + llmbda * (2.94874 - llmbda * (0.1535754 - llmbda * 0.00552888)))

        while True:
            rand = np.random.random()
            ak = rand * constant.MAXU_MS
            k = int(ak)
            ak -= k
            if ak > self._wms[i, j, k]:
                k = self._ims[i, j, k]
            a = self._fms[i, j, k]
            u = self._ums[i, j, k]
            du = self._ums[i, j, k + 1] - u
            rand = np.random.random()
            if abs(a) < 0.2:
                u += rand * du * (1 + 0.5 * (1 - rand) * a * (1 - rand * a))
            else:
                u -= du / a * (1 - np.sqrt(1 + rand * a * (2 + a)))

            rand = omega2 * u / (1 + 0.5 * omega2 - u)
            rand = min(1.999999, rand)
            cost = 1 - rand
            if not self._do_spin:
                break
            rejf = self._spinRejection(elke, beta2, q1, cost, is_electron)
            rand = np.random.random()
            if rand <= rejf:
                break

        return cost

    def emscat(self, path: float, eke: float):
        """
        Sample multiple electron scattering angles from the
        exact distribution resulting from elastic scattering
        :param path: Electron track-length
        :param eke: Electron kinetic energy  at the beginning of step
        :return: scattering cosine
        """
        lmb, chia2, xi, beta2, _, elkem = self._mscatCoeffs(path, eke, True)
        return self._mscat(lmb, chia2, xi, elkem, beta2, True)

    def pmscat(self, path: float, eke: float):
        """
        Sample multiple positron scattering angles from the
        exact distribution resulting from elastic scattering
        :param path: Positron track-length
        :param eke: Positron kinetic energy  at the beginning of step
        :return: scattering cosine
        """
        lmb, chia2, xi, beta2, _, elkem = self._mscatCoeffs(path, eke, False)
        return self._mscat(lmb, chia2, xi, elkem, beta2, False)

    def _presta1(self, path: float, eke: float, is_electron: bool):
        """
        PRESTA-I model that simulates multiple elastic scattering and spatial
        deflections for a given path-length
        Assume that initial particle position is (0,0,0) and direction is (0,0,1)
        :param path: Electron/positron track-length
        :param eke: Electron/positron kinetic energy  at the beginning of step
        :return: (position x,y,z), (direction u,v,w)
        """
        lmb, chia2, xi, beta2, xi_corr, elkem = self._mscatCoeffs(path, eke, is_electron)

        # Sample multiple scattering angle
        cost = self._mscat(lmb, chia2, xi, elkem, beta2, False)
        sint = np.sqrt(max(0.0, 1.0 - cost**2))
        phi = np.random.random() * 2 * np.pi
        cosp = np.cos(phi)
        sinp = np.sin(phi)

        xi *= xi_corr  # Correct xi used for the PLC calc. for spin effects

        # Calculate PLC and lateral transport a la PRESTA-I
        if xi < 0.1:
            z = 1 - xi * (0.5 - xi * (0.166666667 - 0.041666667*xi))
        else:
            z = (1 - np.exp(-xi)) / xi

        r = 0.5 * sint
        r2 = r**2
        z2 = z**2
        r2max = 1 - z2
        if r2max < r2:
            r2 = r2max
            r = np.sqrt(r2)

        # Calculate final position vector
        pv = np.array((r * cosp, r * sinp, z))
        # Final direction vector
        dv = np.array((sint * cosp, sint * sinp, cost))

        return pv * path, dv

    def epresta1(self, path: float, eke: float):
        """
        PRESTA-I model that simulates multiple elastic scattering and spatial
        deflections for a given path-length
        Assume that initial particle position is (0,0,0) and direction is (0,0,1)
        Path length must be smaller than electron CSDA range & tmxs limit
        :param path: Electron track-length
        :param eke: Electron kinetic energy  at the beginning of step
        :return: (position x,y,z), (direction u,v,w)
        """
        erange = self._range(eke, True)
        if path > erange:
            raise Exception("Path length must be smaller than electron range, {:.5e}".format(erange))
        return self._presta1(path, eke, True)

    def ppresta1(self, path: float, eke: float):
        """
        PRESTA-I model that simulates multiple elastic scattering and spatial
        deflections for a given path-length
        Assume that initial particle position is (0,0,0) and direction is (0,0,1)
        Path length must be smaller than electron CSDA range & tmxs limit
        :param path: Positron track-length
        :param eke: Positron kinetic energy  at the beginning of step
        :return: (position x,y,z), (direction u,v,w)
        """
        erange = self._range(eke, False)
        if path > erange:
            raise Exception("Path length must be smaller than positron range, {:.5e}".format(erange))
        return self._presta1(path, eke, False)

    def _presta2(self, path: float, eke: float, is_electron: bool):
        """
        PRESTA-II model that simulates multiple elastic scattering and spatial
        deflections for a given path-length
        Assume that initial particle position is (0,0,0) and direction is (0,0,1)
        :param path: Electron/positron track-length
        :param eke: Electron/positron kinetic energy  at the beginning of step
        :return: (position x,y,z), (direction u,v,w)
        """
        eloss = self._loss(path, eke, is_electron)
        ekem = eke - 0.5 * eloss
        tau = ekem / constant.ELECTRON_REST_MASS
        tau2 = tau ** 2
        epsilon = eloss / eke
        epsilonp = eloss / ekem
        ekem *= 1 - epsilonp ** 2 * (6 + 10 * tau + 5 * tau2) / (24 * tau2 + 72 * tau + 48)
        p2 = ekem * (ekem + constant.ELECTRON_REST_MASS_D)
        beta2 = p2 / (p2 + constant.ELECTRON_REST_MASS_SQ)
        chia2 = self._xcc / (4 * p2 * self._blcc)
        lmb = path * self._blcc / beta2
        lmb *= 1 - 1 / 6 * (4 + tau * (6 + tau * (7 + tau * (4 + tau)))) * (epsilonp / ((tau + 1) * (tau + 2))) ** 2
        elkem = np.log(ekem)

        # Account for spin effect
        if self._do_spin:
            # correction to the screening angle derived from PWA
            etae = self.etae[ekem] if is_electron else self.etap[ekem]
            # correction to the first MS moment due to spin
            xi_corr = self.q1ce[ekem] if is_electron else self.q1cp[ekem]
            gamma = self.q2ce[ekem] if is_electron else self.q2cp[ekem]
            ms_corr = self.blcce[ekem]
        else:
            etae = 1.0
            xi_corr = 1.0
            gamma = 1.0
            ms_corr = 1.0

        chia2 *= etae
        lmb = lmb / etae / (1.0 + chia2) * ms_corr
        chilog = np.log(1.0 + 1.0 / chia2)
        q1 = 2.0 * chia2 * (chilog * (1.0 + chia2) - 1.0)
        gamma *= 6.0 * chia2 * (1.0 + chia2) * (chilog * (1.0 + 2.0 * chia2) - 2) / q1
        xi = q1 * lmb

        cost1 = self._mscat(lmb * 0.5, chia2, xi * 0.5, elkem, beta2, is_electron)
        cost2 = self._mscat(lmb * 0.5, chia2, xi * 0.5, elkem, beta2, is_electron)
        sint1 = np.sqrt(max(0.0, 1.0 - cost1**2))
        sint2 = np.sqrt(max(0.0, 1.0 - cost2**2))
        phi = np.random.random(2) * 2 * np.pi
        cosp1 = np.cos(phi[0])
        sinp1 = np.sin(phi[0])
        cosp2 = np.cos(phi[1])
        sinp2 = np.sin(phi[1])

        xi *= xi_corr

        u2 = sint2 * cosp2
        v2 = sint2 * sinp2
        u2p = cost1 * u2 + sint1 * cost2

        # Final direction vector
        dv = np.array((
            u2p * cosp1 - v2 * sinp1,
            u2p * sinp1 + v2 * cosp1,
            cost1 * cost2 - sint1 * u2
        ))

        # Calculate delta, b, c
        eta = np.sqrt(np.random.random())
        eta1 = 0.5 * (1 - eta)
        delta = 0.9082483 - (0.1020621 - 0.0263747 * gamma) * xi

        # Correct the coefficients for energy loss
        temp1 = 2.0 + tau
        temp = (2.0 + tau * temp1) / ((tau + 1.0) * temp1)
        # Take logarithmic dependence into account as well
        temp -= (tau + 1.0) / ((tau + 2.0) * (chilog * (1.0 + chia2) - 1.0))
        temp *= epsilonp
        temp1 = 1 - temp
        delta += 0.40824829 * (epsilon * (tau + 1) / ((tau + 2) * (chilog * (1 + chia2) - 1)
                                                      * (chilog * (1 + 2 * chia2) - 2)) - 0.25 * temp * temp)
        b = eta * delta
        c = eta * (1 - delta)

        pv = np.array((
            b * sint1 * cosp1 + c * (cosp1 * u2 - sinp1 * cost1 * v2) + eta1 * dv[0] * temp1,
            b * sint1 * sinp1 + c * (sinp1 * u2 + cosp1 * cost1 * v2) + eta1 * dv[1] * temp1,
            eta1 * (1 + temp) + b * cost1 + c * cost2 + eta1 * dv[2] * temp1
        ))

        return pv * path, dv

    def epresta2(self, path: float, eke: float):
        """
        PRESTA-II model that simulates multiple elastic scattering and spatial
        deflections for a given path-length
        Assume that initial particle position is (0,0,0) and direction is (0,0,1)
        Path length must be smaller than electron CSDA range & tmxs limit
        :param path: Electron track-length
        :param eke: Electron kinetic energy  at the beginning of step
        :return: (position x,y,z), (direction u,v,w)
        """
        erange = self._range(eke, True)
        if path > erange:
            raise Exception("Path length must be smaller than electron range, {:.5e}".format(erange))
        return self._presta2(path, eke, True)
