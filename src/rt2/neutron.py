from __future__ import annotations

import os

import numpy as np

from rt2.algorithm import probFromAlias
from rt2.fortran import Fortran
from rt2.endf import REACTION_TYPE, SECONDARY_TYPE, SECONDARY_TO_PID
from rt2.particle import DTYPE_PHASE_SPACE


# noinspection PyProtectedMember
class Reaction:
    def __init__(self, parent: Library, index: int, mt: int, res_za: int, stape: np.ndarray):
        self._mt     = mt
        self._res_za = res_za
        self._index  = index
        self._desc   = REACTION_TYPE[mt]
        self._stape  = stape
        self._parent = parent

    def __repr__(self):
        return self._desc

    def mt(self) -> int:
        return self._mt
    def resZA(self) -> int:
        return self._res_za

    def xs(self, group: int = -1) -> np.ndarray | float:
        if group < 0:
            ngn = len(Library.neutronGroupStructure()) - 1
            xs_  = np.empty(ngn)
            for i in range(ngn):
                xs_[i] = self.xs(i)
            return xs_
        else:
            xs_       = self._parent._xs[group]
            prob_list = probFromAlias(self._parent._ralias[group], self._parent._rprob[group])
            return xs_ * prob_list[self._index]

    def instruction(self, group: int) -> (np.ndarray, float):
        ptype_list   = self._stape & 31
        nhadron      = np.sum(ptype_list != 16)
        multiplicity = self.multiplicity(group)
        return ptype_list[:int(np.ceil(multiplicity)) + nhadron], multiplicity - int(multiplicity)

    def transition(self, group: int, mf: int, norm: bool = False) -> np.ndarray:
        cont = -1
        for tape in self._stape:
            if (tape & 31) == mf:
                cont = tape >> 5
                break
        assert cont >= 0, 'Secondary table for "{}" not exist in reaction {}'.format(
            SECONDARY_TYPE[mf], REACTION_TYPE[self.mt()]
        )
        egn = self._parent.neutronGroupStructure()
        egg = self._parent.gammaGroupStructure()
        ngn = len(egn) - 1
        ngg = len(egg) - 1

        gpos, glow, glen = self._parent._gcontrol[cont * ngn + group]

        assert glen >= 1, 'Energy of incident neutron is lower than threshold'

        alias = self._parent._galias[gpos:gpos + glen]
        aprob = self._parent._gprob[gpos:gpos + glen]
        prob  = probFromAlias(alias, aprob)

        trans = np.zeros(ngg if mf == 16 else ngn)
        trans[glow:glow + glen] = prob
        if norm:
            trans /= egg[1:] - egg[:-1] if mf == 16 else egn[1:] - egn[:-1]  # 1/MeV
        return trans

    def equiProbable(self, group_from: int, group_to: int, mf: int) -> np.ndarray:
        if mf == 16:  # photon -> isotropic
            return np.array([-1.0, 1.0])

        cont = -1
        for tape in self._stape:
            if (tape & 31) == mf:
                cont = tape >> 5
                break
        assert cont >= 0, 'Secondary table for "{}" not exist in reaction {}'.format(
            SECONDARY_TYPE[mf], REACTION_TYPE[self.mt()]
        )
        egn = self._parent.neutronGroupStructure()
        ngn = len(egn) - 1

        gpos, glow, glen = self._parent._gcontrol[cont * ngn + group_from]

        assert glen >= 1, 'Energy of incident neutron is lower than threshold'

        eabin = self._parent._eabin[gpos:gpos + glen]
        assert len(eabin) > group_to - glow >= 0, '"group_to" out of range'
        return eabin[group_to - glow]

    def angularDistribution(self, group: int, mf: int) -> (np.ndarray, np.ndarray):
        if mf == 16:  # photon -> isotropic
            return np.array([-1.0, 1.0]), np.array([1.0])

        cont = -1
        for tape in self._stape:
            if (tape & 31) == mf:
                cont = tape >> 5
                break
        assert cont >= 0, 'Secondary table for "{}" not exist in reaction {}'.format(
            SECONDARY_TYPE[mf], REACTION_TYPE[self.mt()]
        )
        egn = self._parent.neutronGroupStructure()
        ngn = len(egn) - 1

        gpos, glow, glen = self._parent._gcontrol[cont * ngn + group]

        assert glen >= 1, 'Energy of incident neutron is lower than threshold'

        alias = self._parent._galias[gpos:gpos + glen]
        aprob = self._parent._gprob[gpos:gpos + glen]
        prob  = probFromAlias(alias, aprob)

        eabin = self._parent._eabin[gpos:gpos + glen]

        eabin_t  = np.sort(np.unique(eabin))
        eadist_t = np.zeros(len(eabin_t) - 1)

        for p, equi in zip(prob, eabin):
            norm = len(equi) - 1
            for i in range(norm):
                bifrom = np.where(eabin_t == equi[i])[0][0]
                bito   = np.where(eabin_t == equi[i + 1])[0][0]
                bseg   = eabin_t[bifrom:bito + 1]
                width  = bseg[1:] - bseg[:-1]
                width  = width / np.sum(width)
                eadist_t[bifrom:bito] += p / norm * width
                pass

        return eabin_t, eadist_t

    def secondaries(self) -> dict:
        ptype_list    = self._stape & 31
        secs = {}
        for ptype, counts in zip(*np.unique(ptype_list, return_counts=True)):
            secs[SECONDARY_TYPE[ptype]] = counts
        return secs

    def multiplicity(self, group: int = -1) -> np.ndarray | float:
        ptype_list  = self._stape & 31
        nhadron     = np.sum(ptype_list != 16)
        ngn = len(Library.neutronGroupStructure()) - 1
        if 16 not in ptype_list:
            multiplicity_arr = np.zeros(ngn, dtype=np.float32)
        else:
            offset = self._index
            multiplicity_arr = self._parent._multiplicity[offset * ngn:(offset + 1) * ngn] - nhadron

        if group < 0:
            return multiplicity_arr
        else:
            return multiplicity_arr[group]

    def deposition(self, group: int = -1) -> np.ndarray | float:
        ngn      = len(Library.neutronGroupStructure()) - 1
        offset   = self._index
        depo_arr = self._parent._edepo[offset * ngn:(offset + 1) * ngn]

        if group < 0:
            return depo_arr
        else:
            return depo_arr[group]

    def sample(self, group: int) -> np.ndarray[DTYPE_PHASE_SPACE]:
        ptype_list = self._stape & 31
        cont_list  = self._stape >> 5
        nhadron    = np.sum(ptype_list != 16)
        mult       = nhadron + self.multiplicity(group)
        ps         = np.empty(int(np.ceil(mult)), dtype=DTYPE_PHASE_SPACE)
        counter    = 0
        while mult > 0:
            if np.random.random() > mult:
                break
            mult -= 1.0
            ptype = ptype_list[counter]
            pid   = SECONDARY_TO_PID[ptype]
            sgroup, cost = self._parent._sampleParticle(ptype, group, cont_list[counter])
            if sgroup < 0:  # invalid group (energy out of range maybe)
                continue
            e = Library.getEnergyFromGammaGroup(sgroup) if ptype == 16 else Library.getEnergyFromNeutronGroup(sgroup)
            # uniform in energy bin
            phi  = np.random.random() * np.pi * 2.0
            sint = np.sqrt(1.0 - cost * cost)
            ps[counter]['pid'] = pid
            ps[counter]['x']   = 0.0
            ps[counter]['y']   = 0.0
            ps[counter]['z']   = 0.0
            ps[counter]['u']   = sint * np.cos(phi)
            ps[counter]['v']   = sint * np.sin(phi)
            ps[counter]['w']   = cost
            ps[counter]['wee'] = 1.0
            ps[counter]['e']   = e
            counter += 1
        return ps[:counter]

    def checkIntegrity(self, verbose: bool) -> bool:
        xs = self.xs()
        ptype_list  = self._stape & 31
        cont_list   = self._stape >> 5
        ngn = len(Library.neutronGroupStructure()) - 1
        ngg = len(Library.gammaGroupStructure()) - 1
        if verbose:
            print("Check data integrity of {} channel".format(self.__repr__()))
            print("Lowest group    : {}".format(np.argmax(xs > 0)))
            print("Secondary action: ")
            for i, ptype in enumerate(ptype_list):
                print("{} - {}".format(i, SECONDARY_TYPE[ptype]))

        assert len(self._stape) < 30, 'Sampling tape length overflow, length={}'.format(len(self._stape))

        for group in range(len(xs)):
            if xs[group] <= 0.0:
                continue
            # check depo & multiplicity value
            depo = self.deposition(group)
            mult = self._parent._multiplicity[self._index * ngn + group]
            assert depo >= 0.0, 'Negative deposition detected, group={} E={}'.format(group, depo)
            assert len(ptype_list) >= mult, \
                'Secondary control tape is shorter than multiplicity {} < {}'.format(len(ptype_list), mult)
            cont_checked = set()
            for i, (ptype, cont) in enumerate(zip(ptype_list, cont_list)):
                if i >= np.ceil(mult):
                    continue
                if cont in cont_checked:
                    continue
                cont_checked.add(cont)
                cpos, gfloor, length = self._parent._gcontrol[group + cont * ngn]
                # check control card
                assert cpos >= 0, \
                    'Negative control position detected, group={} ptype={}'.format(group, ptype)

                if length < 0:  # table not exist -> empty secondary
                    continue

                if ptype == 16:  # gamma
                    assert 0 <= gfloor < ngg, \
                        'Gamma transition group out of range from={} to={}'.format(group, gfloor)
                    assert 0 <= gfloor + length <= ngg, \
                        'Gamma transition group out of range from={} to={}'.format(group, gfloor + length - 1)
                else:
                    assert 0 <= gfloor < ngn, \
                        'Neutron transition group out of range from={} to={}'.format(group, gfloor)
                    assert 0 <= gfloor + length <= ngn, \
                        'Neutron transition group out of range from={} to={}'.format(group, gfloor + length - 1)
                # check alias table
                # check length
                prob  = self._parent._gprob[cpos:cpos + length]
                alias = self._parent._galias[cpos:cpos + length]
                assert len(prob)  == length, 'gprob out of range at group={}, position={}'.format(group, cpos)
                assert len(alias) == length, 'galias out of range at group={}, position={}'.format(group, cpos)
                # check table integrity
                for p, a in zip(prob, alias):
                    assert p > 1.0 or 0 <= a < length, \
                        'galias transition out of range at group={}, alias={}'.format(group, a)
                # check cost
                if ptype != 16:  # hadron must have eabin
                    for j in range(cpos, cpos + length):
                        assert 0 <= j < len(self._parent._eabin), \
                            'eabin length mismatched j={}, length={}'.format(j, len(self._parent._eabin))
                        eabin = self._parent._eabin[j]
                        assert eabin[0] >= -1.01, 'eabin floor out of range mu={}'.format(eabin[0])
                        assert eabin[-1] <= 1.01, 'eabin ceil out of range mu={}'.format(eabin[-1])
        return True


class Library:
    __egn = None  # neutron group
    __egg = None  # gamma group
    __ngn = None

    @staticmethod
    def loadNeutronGroupStructure(file_name: str):
        file = Fortran(file_name, mode='r')
        Library.__egn = file.read(np.float32)
        file.close()
        Library.__ngn = len(Library.__egn) - 1

    @staticmethod
    def neutronGroupStructure() -> np.ndarray[float] | None:
        return Library.__egn

    @staticmethod
    def loadGammaGroupStructure(file_name: str):
        file = Fortran(file_name, mode='r')
        Library.__egg = file.read(np.float32)
        file.close()

    @staticmethod
    def gammaGroupStructure() -> np.ndarray[float] | None:
        return Library.__egg

    @staticmethod
    def getGroupFromEnergy(energy: float):
        return np.argmax(energy < Library.__egn) - 1

    @staticmethod
    def getEnergyFromNeutronGroup(group: int):
        rand = np.random.random()
        return rand * Library.__egn[group] + (1.0 - rand) * Library.__egn[group + 1]

    @staticmethod
    def getEnergyFromGammaGroup(group: int):
        rand = np.random.random()
        return rand * Library.__egg[group] + (1.0 - rand) * Library.__egg[group + 1]

    def __init__(self, file_name: str, read_header: bool = False):
        file = Fortran(file_name, mode='r')
        # header
        self._za   = file.read(np.int32)[0]
        self._isom = file.read(np.int32)[0]
        self._temp = file.read(np.float32)[0]
        self._sab  = file.read(np.uint8).tobytes().decode('utf-8')[:-1]
        # angle group
        agroup = file.read(np.int32)[0]

        if read_header:  # finish
            return

        # get group number
        egn = Library.neutronGroupStructure()
        egg = Library.gammaGroupStructure()
        assert egn is not None, 'Neutron group structure data must be loaded'
        assert egg is not None, 'Gamma group structure data must be loaded'

        ngn = len(egn) - 1

        # xs
        self._xs = file.read(np.float32)
        assert len(self._xs) == ngn, 'Length of loaded neutron group is not matched to this file'

        # reaction
        mt_list     = file.read(np.int32)
        res_za_list = file.read(np.int32)

        # reaction sampling tables
        rshape = (ngn, len(mt_list))
        self._ralias = file.read(np.int32).reshape(rshape)
        self._rprob  = file.read(np.float32).reshape(rshape)

        stape_len = file.read(np.int32)
        stape_off = file.read(np.int32)
        stape     = file.read(np.uint32)

        # reconstruct reaction
        self._reactions = []
        for i, (mt, res_za) in enumerate(zip(mt_list, res_za_list)):
            offset = stape_off[i]
            length = stape_len[i]
            self._reactions += [Reaction(self, i, mt, res_za, stape[offset:offset + length])]

        # group
        self._multiplicity = file.read(np.float32)
        self._edepo        = file.read(np.float32)
        self._gcontrol     = file.read(np.int32).reshape((-1, 3))
        self._galias       = file.read(np.int32)
        self._gprob        = file.read(np.float32)

        # analyse the length of equiprobable bin
        eabin          = file.read(np.float32)
        assert len(eabin) % agroup == 0, 'Equiprobable angle bin is misaligned'
        self._eabin = eabin.reshape((-1, agroup))
        file.close()

    def za(self) -> int:
        return self._za

    def isomeric(self) -> int:
        return self._isom

    def temperature(self) -> float:
        return self._temp

    def sab(self) -> str:
        return self._sab

    def xs(self, group: int = -1) -> np.ndarray | float:
        if group < 0:
            return self._xs
        else:
            return self._xs[group]

    def keys(self) -> np.ndarray:
        return np.array(list(map(lambda reaction: reaction.mt(), self._reactions)))

    def branch(self, group: int) -> dict:
        key_list  = self.keys()
        prob_list = probFromAlias(self._ralias[group], self._rprob[group])
        key_dict  = {}
        for key, prob in zip(key_list, prob_list):
            key_dict[key] = prob
        return key_dict

    def sample(self, group: int) -> Reaction:
        rand = np.random.random()
        aj   = rand * self._rprob.shape[1]
        j    = int(aj)
        aj  -= j
        if aj > self._rprob[group, j]:
            j = self._ralias[group, j]
        return self._reactions[j]

    def _sampleParticle(self, mf: int, group: int, cont_offset: int) -> (int, float):
        cpos, gfloor, length = self._gcontrol[group + cont_offset * Library.__ngn]
        if length < 0:
            return -1, -1.0

        # group
        rand = np.random.random()
        aj   = rand * length
        j    = int(aj)
        aj  -= j
        if aj > self._gprob[cpos + j]:
            j = self._galias[cpos + j]
        group = gfloor + j
        cpos += j

        # cost
        rand = np.random.random()
        if mf != 16:  # hadron
            ak  = rand * (self._eabin.shape[1] - 1)
            k   = int(ak)
            ak -= k
            mu  = self._eabin[cpos, k] * ak + self._eabin[cpos, k + 1] * (1.0 - ak)
        else:  # gamma
            mu  = rand * 2.0 - 1.0  # isotropic
        return group, mu

    def checkIntegrity(self, verbose: bool = False) -> bool:
        try:
            for reaction in self._reactions:
                if not reaction.checkIntegrity(verbose):
                    return False
        except AssertionError as e:
            print(e)
            return False
        else:
            return True

    def __getitem__(self, mt: int) -> Reaction | None:
        for reaction in self._reactions:
            if reaction.mt() == mt:
                return reaction
        return None


class LibraryFactory:

    __library   = ''
    __cdata_key = ''
    __home      = os.path.join(os.environ['MCRT2_HOME'], os.path.join('resource', 'neutron'))
    __header    = []
    __cache     = {}

    @staticmethod
    def setLibrary(lib: str = 'endf8R0_260'):
        LibraryFactory.clear()
        LibraryFactory.__library = lib
        Library.loadNeutronGroupStructure(os.path.join(LibraryFactory.__home, LibraryFactory.__library, 'egn.bin'))
        Library.loadGammaGroupStructure(os.path.join(LibraryFactory.__home, LibraryFactory.__library, 'egg.bin'))
        LibraryFactory.__initDataList()

    @staticmethod
    def __initDataList():
        lib_list = []
        home = os.path.join(LibraryFactory.__home, LibraryFactory.__library, 'library')
        for file in os.listdir(home):
            library   = Library(os.path.join(home, file), True)
            lib_list += [[
                library.za(),
                library.isomeric(),
                library.temperature(),
                library.sab(),
                file
            ]]
        # sort
        for i in range(2, -1, -1):
            lib_list = sorted(lib_list, key=lambda li: li[i])
        LibraryFactory.__header = lib_list

    @staticmethod
    def currentLibrary() -> str:
        return LibraryFactory.__library

    @staticmethod
    def getLibraryList() -> list:
        folder_list = os.listdir(os.path.join(LibraryFactory.__home))
        return folder_list

    @staticmethod
    def getDataList(z: int = -1) -> list:
        if z < 0:
            return LibraryFactory.__header
        lib_list  = []
        for lib in LibraryFactory.__header:
            if z == (lib[0] // 1000):
                lib_list += [lib]
        # sort
        for i in range(2, -1, -1):
            lib_list = sorted(lib_list, key=lambda li: li[i])
        return lib_list

    @staticmethod
    def clear():
        LibraryFactory.__cache = {}

    @staticmethod
    def choose(za: int, isomer: int = 0, temperature: float = 293.6, identifier: str = ''):

        lib_list = []
        for lib in LibraryFactory.__header:
            if za == lib[0] and isomer == lib[1] and identifier == lib[3]:  # temperature later
                lib_list += [lib]
        # error
        assert len(lib_list), 'Data not exist for za={}, isomer={}, sab={}'.format(za, isomer, identifier)

        # choose proximate temperature
        temp_dif_min = 1e30
        lib_target   = None
        for lib in lib_list:
            temp_dif = abs(temperature - lib[2])
            if temp_dif < temp_dif_min:
                temp_dif_min = temp_dif
                lib_target   = lib

        za_str    = '{}'.format(lib_target[0])
        if isomer:
            za_str += 'm{}'.format(lib_target[1])
        dict_key = '{}_{}K'.format(za_str, int(lib_target[2]))
        if identifier:
            dict_key += '_{}'.format(lib_target[3])

        if dict_key not in LibraryFactory.__cache.keys():
            LibraryFactory.__cache[dict_key] = Library(
                os.path.join(LibraryFactory.__home, LibraryFactory.__library, 'library', lib_target[4])
            )
        LibraryFactory.__cdata_key = dict_key

    @staticmethod
    def getCurrentData() -> Library:
        assert LibraryFactory.__cdata_key in LibraryFactory.__cache.keys(), \
            'Data must be specified before the extraction'
        return LibraryFactory.__cache[LibraryFactory.__cdata_key]

    @staticmethod
    def getCurrentKey() -> list:
        assert LibraryFactory.__cdata_key in LibraryFactory.__cache.keys(), \
            'Data must be specified before the extraction'
        current_lib = LibraryFactory.__cache[LibraryFactory.__cdata_key]
        za          = current_lib.za()
        isomer      = current_lib.isomeric()
        temperature = int(current_lib.temperature())
        identifier  = current_lib.sab()
        return [za, isomer, temperature, identifier]


lib_list = LibraryFactory.getLibraryList()
assert len(lib_list), 'At least one neutron library must be installed'
LibraryFactory.setLibrary(lib_list[0])
