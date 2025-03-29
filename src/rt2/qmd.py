import numpy as np
import enum

from rt2.fortran import Fortran


MAX_DIMENSION_CLUSTER = 256


DTYPE_QMD_MODEL  = np.dtype([
    ('particlpant_idx'         , np.uint8   , 256),
    ('initial_flags'           , np.uint32) ,
    ('offset_1d'               , np.int32)  ,
    ('offset_2d'               , np.int32)  ,
    ('initial_hid'             , np.int32)  ,
    ('initial_eke'             , np.float32),
    ('initial_momentum'        , np.float32),
    ('initial_weight'          , np.float32),
    ('initial_position'        , np.float32 , 3),
    ('initial_polar'           , np.float32 , 2),
    ('initial_azim'            , np.float32 , 2),
    ('maximum_impact_parameter', np.float32),
    ('za_nuc'                  , np.int16   , 4),
    ('mass'                    , np.float32 , 2),
    ('beta_lab_nn'             , np.float32),
    ('beta_nn_cm'              , np.float32),
    ('cc_gamma'                , np.float32 , 2),
    ('cc_rx'                   , np.float32 , 2),
    ('cc_rz'                   , np.float32 , 2),
    ('cc_px'                   , np.float32 , 2),
    ('cc_pz'                   , np.float32 , 2),
    ('cphi'                    , np.float32),
    ('sphi'                    , np.float32),
    ('za_system'               , np.int16   , 2),
    ('momentum_system'         , np.float32 , 3),
    ('mass_system'             , np.float32),
    ('ekinal'                  , np.float32),
    ('vtot'                    , np.float32),
    ('excitation'              , np.float32),
    ('n_collisions'            , np.int32)  ,
    ('n_cluster'               , np.int32)  ,
    ('current_field_size'      , np.int32)  ,
    ('current_filed_size_2'    , np.int32)  ,
    ('iter_2body'              , np.int32)  ,
    ('iter_1body'              , np.int32)  ,
    ('iter_gphase'             , np.int32)  ,
    ('iter_grh3d'              , np.int32)  ,
    ('iter_gwarp'              , np.int32)  ,
    ('iter_gblock'             , np.int32)  ,
    ('iter_glane'              , np.int32)
])


DTYPE_PARTICLE_DUMP = np.dtype([
    ('flags'   , np.int32     ),
    ('mass'    , np.float32   ),
    ('position', np.float32, 3),
    ('momentum', np.float32, 3)
])


class PARTICIPANT_FLAGS(enum.Enum):
    PARTICIPANT_IS_PROTON     = 1 << 0
    PARTICIPANT_IS_TARGET     = 1 << 1
    PARTICIPANT_IS_PROJECTILE = 1 << 2
    PARTICIPANT_IS_IN_CLUSTER = 1 << 3


class Participant:
    def __init__(self, data: np.ndarray):
        self._ps = data

    def isProton(self):
        return self._ps['flags'] & PARTICIPANT_FLAGS.PARTICIPANT_IS_PROTON.value

    def isTargetParticipant(self):
        return self._ps['flags'] & PARTICIPANT_FLAGS.PARTICIPANT_IS_TARGET.value

    def isProjectileParticipant(self):
        return self._ps['flags'] & PARTICIPANT_FLAGS.PARTICIPANT_IS_PROJECTILE.value

    def isInCluster(self):
        return self._ps['flags'] & PARTICIPANT_FLAGS.PARTICIPANT_IS_IN_CLUSTER.value

    def clusterID(self):
        return self._ps['flags'] >> 4

    def position(self):
        return self._ps['position']

    def momentum(self):
        return self._ps['momentum']

    def mass(self):
        return self._ps['mass']

    def direction(self):
        p = self.momentum()
        return p / np.linalg.norm(p)


class QMDSnapshot:
    def __init__(self, stream: Fortran):
        file = stream.read(np.uint8)
        self.file = file.view(f'S{file.shape[0]}')
        func = stream.read(np.uint8)
        self.func = func.view(f'S{func.shape[0]}')
        self.line = stream.read(np.int32)[0]

        self.model = stream.read(DTYPE_QMD_MODEL)[0]

        dim  = self.model['current_field_size']
        dim2 = (dim, dim)

        # mean field
        self.rr2  = stream.read(np.float32).reshape(dim2)
        self.pp2  = stream.read(np.float32).reshape(dim2)
        self.rbij = stream.read(np.float32).reshape(dim2)
        self.rha  = stream.read(np.float32).reshape(dim2)
        self.rhe  = stream.read(np.float32).reshape(dim2)
        self.rhc  = stream.read(np.float32).reshape(dim2)

        # 1-D gradient
        self.ffrx = stream.read(np.float32)
        self.ffry = stream.read(np.float32)
        self.ffrz = stream.read(np.float32)
        self.ffpx = stream.read(np.float32)
        self.ffpy = stream.read(np.float32)
        self.ffpz = stream.read(np.float32)
        self.f0rx = stream.read(np.float32)
        self.f0ry = stream.read(np.float32)
        self.f0rz = stream.read(np.float32)
        self.f0px = stream.read(np.float32)
        self.f0py = stream.read(np.float32)
        self.f0pz = stream.read(np.float32)
        self.rh3d = stream.read(np.float32)

        # phase-space
        self._ps = []
        particles = stream.read(DTYPE_PARTICLE_DUMP)
        for p in particles:
            self._ps += [Participant(p)]

    def __getitem__(self, i):
        return self._ps[i]


class QMDDump:
    def __init__(self, file_name: str):
        stream = Fortran(file_name, 'r')
        n      = stream.read(np.int32)[0]
        self._snapshot = []
        for i in range(n):
            self._snapshot += [QMDSnapshot(stream)]
        stream.close()

    def __getitem__(self, i):
        return self._snapshot[i]

    def size(self):
        return len(self._snapshot)
