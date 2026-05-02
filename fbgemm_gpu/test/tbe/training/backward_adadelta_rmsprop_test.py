#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[56]

import unittest

import hypothesis.strategies as st
import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    EmbeddingLocation,
    PoolingMode,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    ComputeDevice,
    SplitTableBatchedEmbeddingBagsCodegen,
)
from hypothesis import given, HealthCheck, settings, Verbosity

from .. import common  # noqa E402

try:
    # pyre-ignore[21]
    from test_utils import gpu_unavailable
except ImportError:
    from fbgemm_gpu.test.test_utils import gpu_unavailable


def _build_emb(
    E: int, D: int, T: int, optimizer: OptimType, lr: float, eps: float, beta1: float
) -> SplitTableBatchedEmbeddingBagsCodegen:
    return SplitTableBatchedEmbeddingBagsCodegen(
        embedding_specs=[
            (E, D, EmbeddingLocation.DEVICE, ComputeDevice.CUDA) for _ in range(T)
        ],
        optimizer=optimizer,
        learning_rate=lr,
        eps=eps,
        beta1=beta1,  # rho (AdaDelta) / alpha (RMSProp)
        weight_decay=0.0,
        pooling_mode=PoolingMode.SUM,
    )


def _ref_adadelta_step(
    w: torch.Tensor,
    g: torch.Tensor,
    s: torch.Tensor,
    u: torch.Tensor,
    lr: float,
    rho: float,
    eps: float,
) -> None:
    # In-place: w, s, u are the per-row state slices being updated.
    s.mul_(rho).addcmul_(g, g, value=1.0 - rho)
    dx = g * torch.sqrt(u + eps) / torch.sqrt(s + eps)
    u.mul_(rho).addcmul_(dx, dx, value=1.0 - rho)
    w.sub_(dx, alpha=lr)


def _ref_rmsprop_step(
    w: torch.Tensor,
    g: torch.Tensor,
    v: torch.Tensor,
    lr: float,
    alpha: float,
    eps: float,
) -> None:
    v.mul_(alpha).addcmul_(g, g, value=1.0 - alpha)
    w.sub_(lr * g / (torch.sqrt(v) + eps))


class BackwardAdaDeltaRMSPropTest(unittest.TestCase):
    @given(
        D=st.integers(min_value=1, max_value=8).map(lambda d: d * 4),
        E=st.integers(min_value=8, max_value=64),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=4,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
    )
    @unittest.skipIf(*gpu_unavailable)
    def test_adadelta_matches_torch_optim(self, D: int, E: int) -> None:
        T = 1
        lr = 0.1
        rho = 0.9
        eps = 1e-6

        torch.manual_seed(0)
        emb = _build_emb(E, D, T, OptimType.ADADELTA, lr=lr, eps=eps, beta1=rho).cuda()
        # Snapshot baseline weights and zero-init optimizer state to match torch.
        ref_w = emb.weights_dev.detach().clone().view(E, D)
        ref_s = torch.zeros_like(ref_w)
        ref_u = torch.zeros_like(ref_w)

        indices = torch.tensor([0, 2, 5], dtype=torch.long, device="cuda")
        offsets = torch.tensor([0, 1, 2, 3], dtype=torch.long, device="cuda")
        out = emb(indices, offsets)
        out.sum().backward()

        # FBGEMM applies the update fused-in-backward; pull final state.
        opt_state = emb.get_optimizer_state()
        fb_s = opt_state[0]["square_avg"]
        fb_u = opt_state[0]["acc_delta"]
        fb_w = emb.weights_dev.detach().view(E, D)

        # Reference: out = sum over selected rows; grad w.r.t. each is ones(D).
        for idx in indices.tolist():
            g = torch.ones(D, device="cuda")
            _ref_adadelta_step(ref_w[idx], g, ref_s[idx], ref_u[idx], lr, rho, eps)

        torch.testing.assert_close(fb_w, ref_w, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(fb_s, ref_s, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(fb_u, ref_u, rtol=1e-4, atol=1e-5)

    @given(
        D=st.integers(min_value=1, max_value=8).map(lambda d: d * 4),
        E=st.integers(min_value=8, max_value=64),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=4,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
    )
    @unittest.skipIf(*gpu_unavailable)
    def test_rmsprop_matches_torch_optim(self, D: int, E: int) -> None:
        T = 1
        lr = 0.01
        alpha = 0.99
        eps = 1e-8

        torch.manual_seed(0)
        emb = _build_emb(E, D, T, OptimType.RMSPROP, lr=lr, eps=eps, beta1=alpha).cuda()
        ref_w = emb.weights_dev.detach().clone().view(E, D)
        ref_v = torch.zeros_like(ref_w)

        indices = torch.tensor([1, 3, 7], dtype=torch.long, device="cuda")
        offsets = torch.tensor([0, 1, 2, 3], dtype=torch.long, device="cuda")
        out = emb(indices, offsets)
        out.sum().backward()

        opt_state = emb.get_optimizer_state()
        fb_v = opt_state[0]["square_avg"]
        fb_w = emb.weights_dev.detach().view(E, D)

        for idx in indices.tolist():
            g = torch.ones(D, device="cuda")
            _ref_rmsprop_step(ref_w[idx], g, ref_v[idx], lr, alpha, eps)

        torch.testing.assert_close(fb_w, ref_w, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(fb_v, ref_v, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
