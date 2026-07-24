"""Seeded Gaussian random projection for TokenCut patch keys."""

import math

import torch


def project_features_random(feats, target_dim, seed=0):
    """Project patch keys with one reproducible Gaussian matrix.

    The same input dimension, target dimension, and seed produce the same
    projection matrix for every image. CLS is excluded; a zero placeholder is
    prepended only because the current ncut() API drops token 0.
    """
    if feats.ndim != 3 or feats.shape[0] != 1:
        raise ValueError(
            "Random projection expects features with shape "
            f"[1, tokens, channels], got {tuple(feats.shape)}"
        )
    if feats.shape[1] < 2:
        raise ValueError(
            "Random projection requires a CLS token and at least one patch token"
        )

    patches = feats[:, 1:, :].squeeze(0)
    channels = patches.shape[1]
    if target_dim <= 0 or target_dim > channels:
        raise ValueError(
            f"Requested --random-projection-dim {target_dim}, but valid "
            f"dimensions are from 1 to {channels}"
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    projection = torch.randn(
        channels,
        target_dim,
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )
    projection = projection / math.sqrt(target_dim)
    projection = projection.to(device=patches.device, dtype=patches.dtype)
    projected_patches = patches @ projection

    cls_placeholder = projected_patches.new_zeros((1, 1, target_dim))
    return torch.cat((cls_placeholder, projected_patches.unsqueeze(0)), dim=1)
