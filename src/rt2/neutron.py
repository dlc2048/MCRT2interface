from __future__ import annotations

import os
import re

import numpy as np

from rt2.algorithm import probFromAlias
from rt2.fortran import Fortran
from rt2.endf import REACTION_TYPE, SECONDARY_TYPE, SECONDARY_TO_PID
from rt2.particle import DTYPE_PHASE_SPACE


# noinspection PyProtectedMember
class Reaction:
    def __init__(self, parent: Library, index: int, mt: int, stape: np.ndarray):
        self._mt     = mt
        self._index  = index
        self._desc   = REACTION_TYPE[mt]
        self._stape  = stape
        self._parent = parent

    def __repr__(self):
        return self._desc

    def mt(self) -> int:
        return self._mt

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
    def neutronGroupStructure() -> np.ndarray | None:
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

    def __init__(self, file_name: str):
        file = Fortran(file_name, mode='r')

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
        mt_list = file.read(np.int32)

        # reaction sampling tables
        rshape = (ngn, len(mt_list))
        self._ralias = file.read(np.int32).reshape(rshape)
        self._rprob  = file.read(np.float32).reshape(rshape)

        stape_len = file.read(np.int32)
        stape_off = file.read(np.int32)
        stape     = file.read(np.uint32)

        # reconstruct reaction
        self._reactions = []
        for i, mt in enumerate(mt_list):
            offset = stape_off[i]
            length = stape_len[i]
            self._reactions += [Reaction(self, i, mt, stape[offset:offset + length])]

        # group
        self._multiplicity = file.read(np.float32)
        self._edepo        = file.read(np.float32)
        self._gcontrol     = file.read(np.int32).reshape((-1, 3))
        self._galias       = file.read(np.int32)
        self._gprob        = file.read(np.float32)

        # analyse the length of equiprobable bin
        last_block_idx = np.max((stape >> 5)[stape & 31 != 16])
        last_control   = self._gcontrol[ngn * (last_block_idx + 1) - 1]
        eabin_length   = last_control[0] + last_control[2]
        eabin          = file.read(np.float32)
        assert len(eabin) % eabin_length == 0, 'Equiprobable angle bin is misaligned'
        self._eabin = eabin.reshape((-1, len(eabin) // eabin_length))
        file.close()

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

    def __getitem__(self, mt: int) -> Reaction | None:
        for reaction in self._reactions:
            if reaction.mt() == mt:
                return reaction
        return None


class LibraryFactory:
    __FILE_PATTERN = r'^(\d+)(?:m(\d+))?_(\d+)K(?:_(\w+))?\.(\w+)$'

    __library   = ''
    __cdata_key = ''
    __home      = os.path.join(os.environ['MCRT2_HOME'], 'resource\\neutron')
    __cache     = {}

    @staticmethod
    def setLibrary(lib: str = 'endf8R0_260'):
        LibraryFactory.clear()
        LibraryFactory.__library = lib
        Library.loadNeutronGroupStructure(os.path.join(LibraryFactory.__home, LibraryFactory.__library, 'egn.bin'))
        Library.loadGammaGroupStructure(os.path.join(LibraryFactory.__home, LibraryFactory.__library, 'egg.bin'))

    @staticmethod
    def currentLibrary() -> str:
        return LibraryFactory.__library

    @staticmethod
    def getLibraryList() -> list:
        folder_list = os.listdir(os.path.join(LibraryFactory.__home))
        return folder_list

    @staticmethod
    def getDataList(z: int = -1) -> list:
        file_list = os.listdir(os.path.join(LibraryFactory.__home, LibraryFactory.__library, 'library'))
        lib_list  = []
        for file in file_list:
            match = re.match(LibraryFactory.__FILE_PATTERN, file)
            if match:
                za, isomer, temperature, identifier, extension = match.groups()
                za          = int(za)
                isomer      = int(isomer) if isomer else 0
                temperature = int(temperature)
                identifier  = identifier if identifier else ''
                if z < 0 or z == (za // 1000):
                    lib_list += [[za, isomer, temperature, identifier]]
        # sort
        for i in range(2, -1, -1):
            lib_list = sorted(lib_list, key=lambda li: li[i])
        return lib_list

    @staticmethod
    def clear():
        LibraryFactory.__cache = {}

    @staticmethod
    def choose(za: int, isomer: int = 0, temperature: int = 294, identifier: str = ''):
        za_str    = '{}'.format(za)
        if isomer:
            za_str += 'm{}'.format(isomer)
        file_name = '{}_{}K'.format(za_str, temperature)
        if identifier:
            file_name += '_{}'.format(identifier)
        file_name += '.bin'
        if file_name not in LibraryFactory.__cache.keys():
            LibraryFactory.__cache[file_name] = Library(
                os.path.join(LibraryFactory.__home, LibraryFactory.__library, 'library', file_name)
            )
        LibraryFactory.__cdata_key = file_name

    @staticmethod
    def getCurrentData() -> Library:
        assert LibraryFactory.__cdata_key in LibraryFactory.__cache.keys(), \
            'Data must be specified before the extraction'
        return LibraryFactory.__cache[LibraryFactory.__cdata_key]

    @staticmethod
    def getCurrentKey() -> list:
        assert LibraryFactory.__cdata_key in LibraryFactory.__cache.keys(), \
            'Data must be specified before the extraction'
        match = re.match(LibraryFactory.__FILE_PATTERN, LibraryFactory.__cdata_key)
        assert match
        za, isomer, temperature, identifier, extension = match.groups()
        za          = int(za)
        isomer      = int(isomer) if isomer else 0
        temperature = int(temperature)
        identifier  = identifier if identifier else ''
        return [za, isomer, temperature, identifier]


LibraryFactory.setLibrary()
