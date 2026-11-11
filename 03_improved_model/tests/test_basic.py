"""Fast sanity tests (no GPU needed).   python tests/test_basic.py [--data-root <data dir>]"""
import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from trajpred.baselines import constant_velocity_mean, zero_motion
from trajpred.data import (T_OBS, T_PRED, TrajDataset, augment, build_protocol, canonical_transform, collate,
                           observed_velocity_rms, to_canonical, to_world)
from trajpred.metrics import MetricAccumulator
from trajpred.models import TrajPredictor, displacement_loss


def test_canonical_roundtrip():
    rng = np.random.default_rng(0)
    obs = np.cumsum(rng.normal(size=(T_OBS, 2)), axis=0) + 5
    origin, angle = canonical_transform(obs)
    c = to_canonical(obs, origin, angle)
    assert np.allclose(c[-1], 0), "last observed point must be the origin"
    assert abs(c[-1, 1] - c[0, 1]) < 1e-9 and c[-1, 0] - c[0, 0] > 0, "heading must point along +x"
    assert np.allclose(to_world(c, origin, angle), obs), "to_world must invert to_canonical"


def test_metrics_and_baselines():
    obs = np.stack([np.stack([np.arange(-7, 1) * 0.5, np.zeros(8)], -1)])  # walks +x at 0.5 m/step, ends at 0
    gt = np.stack([np.stack([np.arange(1, 13) * 0.5, np.zeros(12)], -1)])
    assert np.allclose(constant_velocity_mean(obs), gt)
    acc = MetricAccumulator()
    acc.add(zero_motion(obs), gt, [3])
    r = acc.summary()["ALL"]
    assert math.isclose(r["FDE"], 6.0) and math.isclose(r["ADE"], 3.25) and math.isclose(r["Avg"], 4.625)


def fake_batch(B=5, K=3):
    obs = torch.cumsum(torch.randn(B, T_OBS, 2) * 0.3, 1)
    obs = obs - obs[:, -1:]
    nbr = torch.randn(B, K, T_OBS, 2)
    mask = torch.rand(B, K) < 0.6
    mask[0] = False  # one sample without any neighbour
    return obs, nbr, mask, torch.randn(B, T_PRED, 2)


def test_model():
    obs, nbr, mask, gt = fake_batch()
    for social in (True, False):
        model = TrajPredictor(vel_scale=0.3, social=social).eval()
        pred = model(obs, nbr, mask)
        assert pred.shape == (5, T_PRED, 2) and torch.isfinite(pred).all()
        cv = constant_velocity_mean(obs.numpy())
        assert np.allclose(pred.detach().numpy(), cv, atol=1e-5), "untrained model must equal constant velocity"
    model = TrajPredictor(vel_scale=0.3, social=True).train()
    pred = model(obs, nbr, torch.zeros_like(mask))  # all neighbours masked
    loss = displacement_loss(pred, gt)
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_source_embedding():
    obs, nbr, mask, gt = fake_batch()
    src = torch.tensor([0, 1, 2, 3, -1])
    model = TrajPredictor(vel_scale=0.3, social=True, source_emb=True).eval()
    base = model(obs, nbr, mask)  # no source given -> plain path
    with_src = model(obs, nbr, mask, src)
    assert with_src.shape == base.shape and torch.isfinite(with_src).all()
    # zero-initialised decoder => still equals constant velocity before training, whatever the source
    assert np.allclose(with_src.detach().numpy(), constant_velocity_mean(obs.numpy()), atol=1e-5)
    # old-style call sites (no source argument) keep working for models without the embedding
    TrajPredictor(vel_scale=0.3, social=False).eval()(obs)


def test_augment():
    obs, nbr, mask, gt = fake_batch()
    b = augment({"obs": obs.clone(), "fut": gt.clone(), "nbr": nbr.clone()}, torch.Generator().manual_seed(0))
    assert b["obs"].shape == obs.shape and b["fut"].shape == gt.shape and b["nbr"].shape == nbr.shape


def test_data(root):
    train, ev = build_protocol(root, "official", max_files_per_source=1)
    ds = TrajDataset(train, max_nbrs=8)
    assert len(ds) > 0
    batch = collate([ds[i] for i in range(min(16, len(ds)))])
    assert batch["obs"].shape[1:] == (T_OBS, 2) and batch["fut"].shape[1:] == (T_PRED, 2)
    assert torch.isfinite(batch["fut"]).all() and torch.isfinite(batch["obs"]).all()
    assert np.allclose(batch["obs"][:, -1].numpy(), 0, atol=1e-5), "canonical frame: last obs at origin"
    assert observed_velocity_rms(train) > 0 and len(TrajDataset(ev)) > 0
    # test split has '?' futures -> NaN, observed part complete
    from trajpred.data import load_scenes
    test = load_scenes(root, "test", max_files_per_source=1)
    assert np.isnan(test[0].xy[:, T_OBS:]).all() and np.isfinite(test[0].xy[:, :T_OBS]).all()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None)
    a = ap.parse_args()
    test_canonical_roundtrip(); test_metrics_and_baselines(); test_model(); test_source_embedding(); test_augment()
    if a.data_root:
        test_data(a.data_root)
    print("all tests passed")
