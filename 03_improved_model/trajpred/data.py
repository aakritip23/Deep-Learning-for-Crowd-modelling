"""TrajNet data loading, scene construction and canonical-frame batching.

File format (one row per observation):  frame  ped_id  x  y     (metres; y/x order as in the files)
Every pedestrian has exactly 20 rows = 8 observed + 12 future steps (0.4 s apart).
In test files the 12 future rows contain '?' instead of coordinates.

A *scene* is the group of pedestrians of one file that share the same 20-frame window; they are
each other's neighbours.
"""
import glob
import math
import os
import re
from collections import defaultdict

import numpy as np

T_OBS = 8
T_PRED = 12
T_ALL = T_OBS + T_PRED


# ----------------------------------------------------------------------------- parsing
def read_file(path):
    """Return {ped_id: (frames, xy)} with xy shape [n, 2] (NaN where the file has '?')."""
    frames, coords = defaultdict(list), defaultdict(list)
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) != 4:
                continue
            fr, pid, x, y = p
            frames[pid].append(float(fr))
            coords[pid].append((float("nan") if x == "?" else float(x),
                                float("nan") if y == "?" else float(y)))
    return {pid: (np.asarray(frames[pid]), np.asarray(coords[pid], dtype=np.float64)) for pid in frames}


class Scene:
    """Pedestrians sharing one 20-frame window.  xy: [P, 20, 2] (NaN = unknown future)."""

    __slots__ = ("xy", "ids", "source", "video", "file", "frames")

    def __init__(self, xy, ids, source, video, file, frames=None):
        self.xy, self.ids, self.source, self.video, self.file = xy, ids, source, video, file
        self.frames = frames  # [P, 20] frame numbers


def video_name(stem, known_stems):
    """'bookstore_0_3' -> 'bookstore_0' if 'bookstore_0' is a known file stem, else the stem itself.

    The `<video>_<k>` files are rotated, noisy copies of the video `<video>` (the dataset's augmentation: rotation by a
    multiple of 30 degrees + Gaussian noise, same frames and pedestrian ids); knowing the video lets us build
    hold-out splits without scene leakage.
    """
    m = re.match(r"^(.*)_(\d+)$", stem)
    if m and m.group(1) in known_stems:
        return m.group(1)
    return stem


def scenes_from_file(path, source, video):
    peds = read_file(path)
    groups = defaultdict(list)
    for pid, (fr, xy) in peds.items():
        if len(fr) != T_ALL:
            continue
        order = np.argsort(fr)
        groups[tuple(fr[order])].append((pid, xy[order], fr[order]))
    scenes = []
    for members in groups.values():
        ids = [m[0] for m in members]
        xy = np.stack([m[1] for m in members])  # [P, 20, 2]
        frames = np.stack([m[2] for m in members])  # [P, 20]
        scenes.append(Scene(xy, ids, source, video, os.path.basename(path), frames))
    return scenes


def list_files(root, split):
    return sorted(glob.glob(os.path.join(root, split, "*", "*.txt")))


def load_scenes(root, split, keep_video=None, max_files_per_source=None):
    """Load all scenes of a split ('train' | 'validation' | 'test').

    keep_video: optional callable(video_name) -> bool.   max_files_per_source: for smoke tests.
    """
    known = {os.path.splitext(os.path.basename(p))[0] for p in list_files(root, "train")}
    per_source = defaultdict(int)
    scenes = []
    for path in list_files(root, split):
        source = os.path.basename(os.path.dirname(path))
        stem = os.path.splitext(os.path.basename(path))[0]
        video = video_name(stem, known)
        if keep_video is not None and not keep_video(video):
            continue
        if max_files_per_source is not None and per_source[source] >= max_files_per_source:
            continue
        per_source[source] += 1
        scenes += scenes_from_file(path, source, video)
    return scenes


def build_protocol(root, protocol="official", holdout=(), max_files_per_source=None):
    """Return (train_scenes, eval_scenes).

    official : train -> validation (the validation files are rotated, noisy copies of training files, i.e. in-sample;
               optimistic, matches the baseline repo).
    holdout  : pool train+validation; the videos listed in `holdout` form the eval set and are
               never seen in training (leave-videos-out; no scene leakage).
    """
    if protocol == "official":
        return (load_scenes(root, "train", max_files_per_source=max_files_per_source),
                load_scenes(root, "validation", max_files_per_source=max_files_per_source))
    if protocol == "holdout":
        hold = {h.lower() for h in holdout}
        if not hold:
            raise ValueError("protocol 'holdout' needs at least one video name, e.g. students003")
        train, ev = [], []
        for split in ("train", "validation"):
            train += load_scenes(root, split, keep_video=lambda v: v.lower() not in hold,
                                 max_files_per_source=max_files_per_source)
            ev += load_scenes(root, split, keep_video=lambda v: v.lower() in hold,
                              max_files_per_source=max_files_per_source)
        return train, ev
    if protocol == "full":
        # final fit: train on train + validation (all videos).  The eval set is the validation split, which is now
        # *in-sample* - use it only to monitor that training works; never report it as a result.
        return (load_scenes(root, "train", max_files_per_source=max_files_per_source)
                + load_scenes(root, "validation", max_files_per_source=max_files_per_source),
                load_scenes(root, "validation", max_files_per_source=max_files_per_source))
    raise ValueError(protocol)


def observed_velocity_rms(scenes):
    """RMS of per-step displacement over observed steps (used to standardise model inputs)."""
    s, n = 0.0, 0
    for sc in scenes:
        d = np.diff(sc.xy[:, :T_OBS], axis=1)
        s += float(np.sum(d ** 2))
        n += d.size
    return math.sqrt(s / max(n, 1))


# ----------------------------------------------------------------------------- canonical frame
def canonical_transform(obs_target):
    """Origin = last observed point; x-axis = heading over the observed window. Returns (origin, angle)."""
    origin = obs_target[-1].copy()
    d = obs_target[-1] - obs_target[0]
    angle = 0.0 if np.hypot(d[0], d[1]) < 1e-3 else math.atan2(d[1], d[0])
    return origin, angle


def to_canonical(xy, origin, angle):
    """World [..., 2] -> canonical frame (translate, then rotate by -angle)."""
    c, s = math.cos(angle), math.sin(angle)
    rel = xy - origin
    return np.stack([c * rel[..., 0] + s * rel[..., 1], -s * rel[..., 0] + c * rel[..., 1]], axis=-1)


def to_world(xy, origin, angle):
    """Inverse of to_canonical."""
    c, s = math.cos(angle), math.sin(angle)
    return np.stack([c * xy[..., 0] - s * xy[..., 1] + origin[0],
                     s * xy[..., 0] + c * xy[..., 1] + origin[1]], axis=-1)


# ----------------------------------------------------------------------------- dataset
SOURCES = ["biwi", "crowds", "mot", "stanford"]


class TrajDataset:
    """One sample per pedestrian.  Neighbours = the other pedestrians of the same scene (nearest max_nbrs)."""

    def __init__(self, scenes, max_nbrs=16):
        self.scenes = scenes
        self.max_nbrs = max_nbrs
        self.index = [(si, pi) for si, sc in enumerate(scenes) for pi in range(len(sc.ids))]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        si, pi = self.index[i]
        sc = self.scenes[si]
        obs_world = sc.xy[:, :T_OBS]  # [P, 8, 2]
        origin, angle = canonical_transform(obs_world[pi])
        obs = to_canonical(obs_world[pi], origin, angle)
        fut = to_canonical(sc.xy[pi, T_OBS:], origin, angle)  # NaN if unknown
        others = [j for j in range(len(sc.ids)) if j != pi]
        if others:
            nb = to_canonical(obs_world[others], origin, angle)  # [K, 8, 2]
            dist = np.linalg.norm(nb[:, -1], axis=-1)
            nb = nb[np.argsort(dist)[: self.max_nbrs]]
        else:
            nb = np.zeros((0, T_OBS, 2))
        return {
            "obs": obs.astype(np.float32), "fut": fut.astype(np.float32), "nbr": nb.astype(np.float32),
            "origin": origin, "angle": angle,
            "source": SOURCES.index(sc.source) if sc.source in SOURCES else -1,
            "scene": si, "ped": pi,
        }


def collate(samples):
    """Pad neighbours to the batch maximum -> tensors (torch imported lazily so numpy-only tools work)."""
    import torch
    B = len(samples)
    K = max(1, max(len(s["nbr"]) for s in samples))
    nbr = np.zeros((B, K, T_OBS, 2), dtype=np.float32)
    mask = np.zeros((B, K), dtype=bool)
    for b, s in enumerate(samples):
        k = len(s["nbr"])
        if k:
            nbr[b, :k] = s["nbr"]
            mask[b, :k] = True
    return {
        "obs": torch.from_numpy(np.stack([s["obs"] for s in samples])),
        "fut": torch.from_numpy(np.stack([s["fut"] for s in samples])),
        "nbr": torch.from_numpy(nbr),
        "nbr_mask": torch.from_numpy(mask),
        "origin": np.stack([s["origin"] for s in samples]),
        "angle": np.asarray([s["angle"] for s in samples]),
        "source": torch.tensor([s["source"] for s in samples]),
        "scene": [s["scene"] for s in samples],
        "ped": [s["ped"] for s in samples],
    }


def augment(batch, rng, flip=True, scale=(0.8, 1.2), noise=0.005):
    """In-place batch augmentation in the canonical frame (mirror, speed/scale jitter, observation noise)."""
    import torch
    B = batch["obs"].shape[0]
    device = batch["obs"].device
    sign = torch.ones(B, 1, 1, device=device)
    if flip:
        sign = torch.where(torch.rand(B, 1, 1, generator=rng, device=device) < 0.5, -sign, sign)
    s = torch.empty(B, 1, 1, device=device).uniform_(scale[0], scale[1], generator=rng)
    flipper = torch.stack([torch.ones_like(sign), sign], dim=-1).squeeze(-2)  # [B,1,2] -> (1, +-1)
    for k in ("obs", "fut"):
        batch[k] = batch[k] * flipper * s
    batch["nbr"] = batch["nbr"] * flipper.unsqueeze(1) * s.unsqueeze(1)
    if noise > 0:
        batch["obs"] = batch["obs"] + noise * torch.randn(batch["obs"].shape, generator=rng, device=device)
    return batch
