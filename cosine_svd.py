"""Cosine-oriented SVD projection for TokenCut patch keys."""

import torch

_LOW_RANK_OVERSAMPLING = 8
_LOW_RANK_POWER_ITERATIONS = 2
import torch.nn.functional as F


def project_features_cosine_svd(
    feats, target_dim, eps=1e-12, return_singular_values=False
):
    """Apply uncentered SVD to L2-normalized patch keys.

    The input follows the cached TokenCut contract [1, CLS + patches, C].
    CLS is excluded from normalization, SVD fitting, and projection. A zero
    placeholder is prepended only because the current ncut() API drops token 0.
    """
    if feats.ndim != 3 or feats.shape[0] != 1:
        raise ValueError(
            "Cosine-SVD expects features with shape [1, tokens, channels], "
            f"got {tuple(feats.shape)}"
        )
    if feats.shape[1] < 2:
        raise ValueError(
            "Cosine-SVD requires a CLS token and at least one patch token"
        )

    patches = feats[:, 1:, :].squeeze(0)
    normalized = F.normalize(patches, p=2, dim=1, eps=eps)
    max_dim = min(normalized.shape[0], normalized.shape[1])
    if target_dim <= 0 or target_dim > max_dim:
        raise ValueError(
            f"Requested --cosine-svd-dim {target_dim}, but this image supports "
            f"dimensions from 1 to {max_dim} "
            f"(patches={normalized.shape[0]}, channels={normalized.shape[1]})"
        )

    q = min(target_dim + _LOW_RANK_OVERSAMPLING, max_dim)
    _, singular_values, vectors = torch.svd_lowrank(
        normalized, q=q, niter=_LOW_RANK_POWER_ITERATIONS
    )
    directions = vectors[:, :target_dim]
    projected_patches = normalized @ directions

    cls_placeholder = projected_patches.new_zeros((1, 1, target_dim))
    projected = torch.cat(
        (cls_placeholder, projected_patches.unsqueeze(0)), dim=1
    )
    if return_singular_values:
        return projected, singular_values
    return projected
