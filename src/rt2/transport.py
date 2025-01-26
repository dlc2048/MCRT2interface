import numpy as np


def sampleAzimuthalAngle():
    rand = np.random.random()
    phi = np.pi * 2 * rand
    return np.sin(phi), np.cos(phi)


def rotateVector(vector: np.ndarray, sint: float, cost: float, sinp: float, cosp: float):
    if vector.shape != (3,):
        raise Exception("Dimension of 'vector' must be (3,)")
    sinpsi = vector[0] * vector[0] + vector[1] * vector[1]
    # Small polar angle, no need to rotate
    if sinpsi < 1e-10:
        vector[0] = sint * cosp
        vector[1] = sint * sinp
        vector[2] *= cost
    else:
        sinpsi = np.sqrt(sinpsi)
        us = sint * cosp
        vs = sint * sinp
        sindel = vector[1] / sinpsi
        cosdel = vector[0] / sinpsi

        vector[0] = vector[2] * cosdel * us - sindel * vs + vector[0] * cost
        vector[1] = vector[2] * sindel * us + cosdel * vs + vector[1] * cost
        vector[2] = -sinpsi * us + vector[2] * cost

    # Normalization
    vector /= np.linalg.norm(vector)
