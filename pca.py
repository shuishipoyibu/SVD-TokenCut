"""Feature reduction utilities for TokenCut ablations."""

import torch

_LOW_RANK_OVERSAMPLING = 8
_LOW_RANK_POWER_ITERATIONS = 2

def _lowrank_right_singular_vectors(matrix, target_dim):
    max_dim = min(matrix.shape)
    q = min(target_dim + _LOW_RANK_OVERSAMPLING, max_dim)
    _, singular_values, vectors = torch.svd_lowrank(
        matrix, q=q, niter=_LOW_RANK_POWER_ITERATIONS
    )
    return singular_values, vectors[:, :target_dim]


def _validate_features(feats, method_name):
    if feats.ndim != 3 or feats.shape[0] != 1:
        raise ValueError(
            f"{method_name} expects features with shape [1, tokens, channels], "
            f"got {tuple(feats.shape)}"
        )
    if feats.shape[1] < 2:
        raise ValueError(
            f"{method_name} requires a CLS token and at least one patch token"
        )


def center_features(feats):
    """Subtract the per-image patch mean without reducing dimensions."""
    _validate_features(feats, "Center-only")
    cls_token = feats[:, :1, :]
    patches = feats[:, 1:, :]
    mean = patches.mean(dim=1, keepdim=True)
    return torch.cat((cls_token - mean, patches - mean), dim=1)


def project_features_pca(feats, target_dim, return_singular_values=False):
    """Center patch features and project them onto top PCA directions."""
    _validate_features(feats, "PCA")
    cls_token = feats[:, :1, :]
    patches = feats[:, 1:, :].squeeze(0)
    mean = patches.mean(dim=0, keepdim=True)
    centered = patches - mean
    max_dim = min(centered.shape[0], centered.shape[1])
    if target_dim > max_dim:
        raise ValueError(
            f"Requested --pca-dim {target_dim}, but this image supports at most "
            f"{max_dim} PCA dimensions (patches={centered.shape[0]}, "
            f"channels={centered.shape[1]})"
        )

    singular_values, directions = _lowrank_right_singular_vectors(centered, target_dim)
    projected_patches = centered @ directions
    projected_cls = (cls_token.squeeze(0) - mean) @ directions
    projected = torch.cat(
        (projected_cls.unsqueeze(0), projected_patches.unsqueeze(0)), dim=1
    )
    if return_singular_values:
        return projected, singular_values
    return projected


def project_features_svd(feats, target_dim, return_singular_values=False):
    """Project uncentered patch features onto top right singular vectors."""
    _validate_features(feats, "Uncentered SVD")
    cls_token = feats[:, :1, :]
    patches = feats[:, 1:, :].squeeze(0)
    max_dim = min(patches.shape[0], patches.shape[1])
    if target_dim > max_dim:
        raise ValueError(
            f"Requested --svd-dim {target_dim}, but this image supports at most "
            f"{max_dim} SVD dimensions (patches={patches.shape[0]}, "
            f"channels={patches.shape[1]})"
        )

    singular_values, directions = _lowrank_right_singular_vectors(patches, target_dim)
    projected_patches = patches @ directions
    projected_cls = cls_token.squeeze(0) @ directions
    projected = torch.cat(
        (projected_cls.unsqueeze(0), projected_patches.unsqueeze(0)), dim=1
    )
    if return_singular_values:
        return projected, singular_values
    return projected
