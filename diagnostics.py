"""Core per-image diagnostics for TokenCut experiments."""

import math

import torch
import torch.nn.functional as F


def compute_cosine_similarity(feats):
    """Return the patch-to-patch cosine-similarity matrix."""
    if feats.ndim != 3 or feats.shape[0] != 1 or feats.shape[1] < 2:
        raise ValueError(
            "Expected features with shape [1, CLS + patches, channels], "
            f"got {tuple(feats.shape)}"
        )
    patches = F.normalize(feats[0, 1:, :], p=2, dim=1)
    return patches @ patches.transpose(1, 0)


def compare_cosine_graphs(reference, reduced, tau):
    """Return relative cosine error and upper-triangle edge agreement."""
    if reference.shape != reduced.shape:
        raise ValueError(
            "Cosine-similarity matrices must have matching shapes, got "
            f"{tuple(reference.shape)} and {tuple(reduced.shape)}"
        )

    denominator = torch.linalg.matrix_norm(reference)
    error = torch.linalg.matrix_norm(reduced - reference)
    if denominator.item() == 0:
        e_cos = 0.0 if error.item() == 0 else float("nan")
    else:
        e_cos = (error / denominator).item()

    patch_count = reference.shape[0]
    if patch_count < 2:
        q_r = 1.0
    else:
        upper = torch.triu_indices(
            patch_count, patch_count, offset=1, device=reference.device
        )
        reference_edges = reference[upper[0], upper[1]] > tau
        reduced_edges = reduced[upper[0], upper[1]] > tau
        q_r = (reference_edges == reduced_edges).float().mean().item()
    return e_cos, q_r


def compute_energy_retention(singular_values, target_dim, total_energy):
    """Return retained energy using the exact input Frobenius norm."""
    if total_energy.item() == 0:
        return 0.0
    return (singular_values[:target_dim].square().sum() / total_energy).item()


class CoreDiagnosticsAccumulator:
    """Accumulate macro averages without retaining per-image diagnostics."""

    _METRICS = (
        "mean_singular_energy_retention",
        "mean_relative_cosine_graph_error",
        "mean_edge_agreement",
        "mean_second_eigenvalue",
        "mean_spectral_gap",
    )

    def __init__(self):
        self._sums = {name: 0.0 for name in self._METRICS}
        self._counts = {name: 0 for name in self._METRICS}

    def add(self, energy_retention, e_cos, q_r, lambda2, spectral_gap):
        values = {
            "mean_singular_energy_retention": energy_retention,
            "mean_relative_cosine_graph_error": e_cos,
            "mean_edge_agreement": q_r,
            "mean_second_eigenvalue": lambda2,
            "mean_spectral_gap": spectral_gap,
        }
        for name, value in values.items():
            if value is not None:
                self._sums[name] += float(value)
                self._counts[name] += 1

    def result_lines(self):
        lines = []
        for name in self._METRICS:
            count = self._counts[name]
            if count == 0:
                value = "not_applicable"
            else:
                mean = self._sums[name] / count
                value = f"{mean:.8f}" if math.isfinite(mean) else "not_applicable"
            lines.append(f"{name},{value},,\n")
        return lines
