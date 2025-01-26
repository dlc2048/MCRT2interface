from enum import Enum

import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from skspatial import objects

from rt2.fortran import Fortran


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


class AVATAR_BUFFER_TYPE(Enum):
    EVENT           = 0
    FINALIZE        = 1
    NN_COLLISION    = 2
    INIT_PROJECTILE = 3
    INIT_TARGET     = 4
    IDLE            = 5
    EOB             = 6


class PARTICLE_TYPE(Enum):
    PROTON          = 0
    NEUTRON         = 1
    PI_PLUS         = 2
    PI_ZERO         = 3
    PI_MINUS        = 4
    DELTA_PLUS_PLUS = 5
    DELTA_PLUS      = 6
    DELTA_ZERO      = 7
    DELTA_MINUS     = 8
    COMPOSITE       = 9
    PHOTON          = 10
    UNKNOWN         = 11


class MODEL_FLAGS(Enum):
    TARGET_SWAPPED     = (1 << 0)
    CLUSTER_PROJECTILE = (1 << 1)
    FORCED_TRANSPARENT = (1 << 2)
    TRY_CN             = (1 << 3)
    IS_NUCLEUS_NUCLEUS = (1 << 4)
    ENERGY_VIOLATION   = (1 << 5)


class PARTICLE_FLAGS(Enum):
    IS_PARTICIPANT = (1 << 0)
    EJECTED        = (1 << 1)
    FIRST_ENTRY    = (1 << 2)


class ParticleDump:
    _STRUCT = np.dtype([
        ('type'        , 'i4'),
        ('particle_z'  , 'i2'),
        ('particle_a'  , 'i2'),
        ('flags'       , 'i4'),
        ('n_collisions', 'i4'),
        ('mass'        , 'f4'),
        ('potential'   , 'f4'),
        ('x'           , 'f4'),
        ('y'           , 'f4'),
        ('z'           , 'f4'),
        ('e'           , 'f4'),
        ('px'          , 'f4'),
        ('py'          , 'f4'),
        ('pz'          , 'f4'),
        ('frozen_e'    , 'f4'),
        ('frozen_px'   , 'f4'),
        ('frozen_py'   , 'f4'),
        ('frozen_pz'   , 'f4'),
        ('umomentum'   , 'f4'),
        ('surface_time', 'f4'),
        ('binary_time' , 'f4'),
        ('binary_idx'  , 'i4')
    ])

    def __init__(self, struct: np.void):
        # read phase-space
        self._struct = struct

    def __getitem__(self, index):
        return self._struct[index]

    @staticmethod
    def struct():
        return ParticleDump._STRUCT

    def visualize(self, ax):
        p_type = PARTICLE_TYPE(self['type'])
        if p_type == PARTICLE_TYPE.UNKNOWN:
            return

        x    = self['x']
        y    = self['y']
        z    = self['z']
        pos = np.array([x, y, z])

        # propagation momentum
        fpx  = self['frozen_px']
        fpy  = self['frozen_py']
        fpz  = self['frozen_pz']
        fp   = np.array([fpx, fpy, fpz])
        fe   = self['frozen_e']
        norm = np.linalg.norm(fp)
        if norm > 0:
            direction = fp / fe * 5
            arr = Arrow3D([pos[0], pos[0] + direction[0]],
                          [pos[1], pos[1] + direction[1]],
                          [pos[2], pos[2] + direction[2]],
                          mutation_scale=5,
                          lw=1, arrowstyle="-|>", color="k")
            ax.add_artist(arr)

        # total momentum
        px  = self['px']
        py  = self['py']
        pz  = self['pz']
        p   = np.array([px, py, pz])
        e   = self['e']
        norm = np.linalg.norm(p)
        if norm > 0:
            direction = p / e * 5
            arr = Arrow3D([pos[0], pos[0] + direction[0]],
                          [pos[1], pos[1] + direction[1]],
                          [pos[2], pos[2] + direction[2]],
                          mutation_scale=5,
                          lw=1, arrowstyle="-|>", color="r")
            ax.add_artist(arr)
        # point
        point = objects.Point(pos)
        edgecolors = 'r' if self['flags'] & PARTICLE_FLAGS.IS_PARTICIPANT.value else 'b'
        facecolors = 'y' if p_type == PARTICLE_TYPE.PROTON else 'g'
        point.plot_3d(ax, edgecolors=edgecolors, facecolors=facecolors, s=100)


class SnapShot:
    _STRUCT = np.dtype([
        ('model_idx'                  , 'i4'),
        ('offset_particle'            , 'i4'),
        ('offset_projectile'          , 'i4'),
        ('offset_avatar_time'         , 'i4'),
        ('offset_projectile_backup'   , 'i4'),
        ('flags'                      , 'i4'),
        ('initial_projectile_idx'     , 'i4'),
        ('initial_target_idx'         , 'i4'),
        ('initial_lab_energy'         , 'f4'),
        ('initial_internal_energy'    , 'f4'),
        ('initial_weight'             , 'f4'),
        ('initial_position_x'         , 'f4'),
        ('initial_position_y'         , 'f4'),
        ('initial_position_z'         , 'f4'),
        ('initial_direction_x'        , 'f4'),
        ('initial_direction_y'        , 'f4'),
        ('initial_direction_z'        , 'f4'),
        ('max_impact_parameter'       , 'f4'),
        ('max_universe_radius'        , 'f4'),
        ('max_interaction_distance'   , 'f4'),
        ('n_accepted_collisions'      , 'i4'),
        ('n_accepted_decay'           , 'i4'),
        ('n_emitted_cluster'          , 'i4'),
        ('nucleus_initial_internal_e' , 'f4'),
        ('nucleus_z'                  , 'i2'),
        ('nucleus_a'                  , 'i2'),
        ('incoming_momentum_x'        , 'f4'),
        ('incoming_momentum_y'        , 'f4'),
        ('incoming_momentum_z'        , 'f4'),
        ('incoming_angular_momentum_x', 'f4'),
        ('incoming_angular_momentum_y', 'f4'),
        ('incoming_angular_momentum_z', 'f4'),
        ('remnant_energy'             , 'f4'),
        ('remnant_position_x'         , 'f4'),
        ('remnant_position_y'         , 'f4'),
        ('remnant_position_z'         , 'f4'),
        ('remnant_momentum_x'         , 'f4'),
        ('remnant_momentum_y'         , 'f4'),
        ('remnant_momentum_z'         , 'f4'),
        ('remnant_z'                  , 'i2'),
        ('remnant_a'                  , 'i2'),
        ('maximum_time'               , 'f4'),
        ('current_time'               , 'f4'),
        ('nucleus_excitation_energy'  , 'f4'),
    ])

    def __init__(self, stream: Fortran):
        self.file = stream.read(str)
        self.func = stream.read(str)
        self.ground_state_energy   = stream.read(np.float32)
        self.initial_energy_levels = stream.read(np.float32)
        gen = stream.read(np.int32)
        self.line        = gen[0]
        self.avatar_type = AVATAR_BUFFER_TYPE(gen[1])
        self.model       = stream.read(SnapShot._STRUCT)[0]
        self.particles   = []
        for particle in stream.read(ParticleDump.struct()):
            self.particles += [ParticleDump(particle)]
        return

    def visualize(self, ax):
        sphere = objects.Sphere((0, 0, 0), self.model['max_universe_radius'])
        sphere.plot_3d(ax, alpha=0.2)
        for p in self.particles:
            p.visualize(ax)


class AvatarDump:
    def __init__(self, file_name: str):
        self.snapshot = []
        stream = Fortran(file_name, mode='r')
        n_snapshot = stream.read(dtype=np.int32)[0]
        for i in range(n_snapshot):
            self.snapshot += [SnapShot(stream)]

    def size(self) -> int:
        return len(self.snapshot)
