"""Non-learned reference predictors (numpy, canonical frame: last observed point = origin)."""
import numpy as np

from .data import T_OBS, T_PRED


def _steps():
    return np.arange(1, T_PRED + 1, dtype=np.float64)[None, :, None]


def zero_motion(obs):
    """Pedestrian stays where it was last seen.  obs: [B, 8, 2]."""
    return np.repeat(obs[:, -1:, :], T_PRED, axis=1)


def constant_velocity_last(obs):
    """Extrapolate with the last-step velocity (noisy)."""
    v = obs[:, -1] - obs[:, -2]
    return obs[:, -1:, :] + v[:, None, :] * _steps()


def constant_velocity_mean(obs):
    """Extrapolate with the mean velocity over the observed window (strong, smooth)."""
    v = (obs[:, -1] - obs[:, 0]) / (T_OBS - 1)
    return obs[:, -1:, :] + v[:, None, :] * _steps()


BASELINES = {
    "zero_motion": zero_motion,
    "cv_last": constant_velocity_last,
    "cv_mean": constant_velocity_mean,
}
