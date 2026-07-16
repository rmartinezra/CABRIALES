#!/usr/bin/env python3
"""Shared loading and tail-aware interpolation for empirical MCS kernels."""
from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from scipy.interpolate import RBFInterpolator
    from scipy.spatial import Delaunay
except Exception as exc:  # pragma: no cover - CABRIALES requires scipy
    raise RuntimeError("scipy is required for hybrid empirical-kernel interpolation") from exc

try:
    from tail_aware_transport import LocalQuantileTransport, estimate_t50, load_npz
except ModuleNotFoundError:  # pragma: no cover - package import
    from modulos.tail_aware_transport import LocalQuantileTransport, estimate_t50, load_npz


REQUIRED_KERNEL_KEYS = (
    "centers_mrad",
    "edges_mrad",
    "probabilities",
    "L_m",
    "E_in_GeV",
    "clean_for_kernel",
)

TAIL_AWARE_REQUIRED_KEYS = (
    "full_tail_edges_mrad",
    "full_tail_L_m",
    "full_tail_T50_GeV",
    "full_tail_transport_features",
    "full_tail_transport_quantile_levels",
    "full_tail_transport_abs_quantiles_mrad",
    "full_tail_transport_L_m",
    "full_tail_transport_E_in_GeV",
    "full_tail_transport_E_over_T50",
    "full_tail_transport_source_families",
    "full_tail_transport_probabilities",
)

MUON_MASS_GEV = 0.10565837


def mcs_momentum_scale(
    kinetic_query_GeV: float,
    kinetic_reference_GeV: float,
    mass_GeV: float = MUON_MASS_GEV,
) -> float:
    """Map a sampled MCS angle with the standard 1/(beta*p) dependence.

    The full empirical PDF is sampled first, including its measured tail. This
    factor only maps that angle from the nearest measured energy to an
    out-of-domain query energy.
    """
    def inverse_beta_p(kinetic_GeV: float) -> float:
        total = float(kinetic_GeV) + float(mass_GeV)
        momentum_sq = float(kinetic_GeV) * (float(kinetic_GeV) + 2.0 * float(mass_GeV))
        if total <= 0.0 or momentum_sq <= 0.0:
            raise ValueError("Kinetic energy and mass must define a positive momentum")
        return total / momentum_sq

    return inverse_beta_p(float(kinetic_query_GeV)) / inverse_beta_p(float(kinetic_reference_GeV))


@dataclass(frozen=True)
class EmpiricalKernelLibrary:
    path: Path
    family: str
    centers_mrad: np.ndarray
    edges_mrad: np.ndarray
    probabilities: np.ndarray
    L_m: np.ndarray
    E_in_GeV: np.ndarray
    clean_for_kernel: np.ndarray


def _pick_hybrid_family(files: set[str], requested: str | None) -> str:
    families = sorted({
        key[:-len("_centers_mrad")]
        for key in files
        if key.endswith("_centers_mrad")
    })
    if not families:
        return ""

    if requested is None:
        requested = os.environ.get("CABRIALES_KERNEL_FAMILY")
    if requested:
        requested = requested.strip()
        if requested in families:
            return requested
        raise KeyError(
            f"Kernel family {requested!r} is not available. "
            f"Available families: {families}"
        )

    if "full_tail" in families:
        return "full_tail"
    if "core" in families:
        return "core"
    return families[0]


def load_empirical_kernel_library(
    npz_path: str | Path,
    family: str | None = None,
) -> EmpiricalKernelLibrary:
    """Load either a flat empirical kernel or a prefixed hybrid kernel.

    Flat libraries expose REQUIRED_KERNEL_KEYS directly. Hybrid libraries expose
    the same keys with a family prefix, for example full_tail_centers_mrad.
    The family can be selected with the function argument or with the
    CABRIALES_KERNEL_FAMILY environment variable.
    """
    path = Path(npz_path)
    with np.load(path, allow_pickle=False) as data:
        files = set(data.files)
        if all(key in files for key in REQUIRED_KERNEL_KEYS):
            prefix = ""
            selected_family = "flat"
        else:
            selected_family = _pick_hybrid_family(files, family)
            if not selected_family:
                raise KeyError(
                    "Kernel library does not match the flat or hybrid format. "
                    f"Available keys: {sorted(files)}"
                )
            prefix = f"{selected_family}_"
            missing = [f"{prefix}{key}" for key in REQUIRED_KERNEL_KEYS if f"{prefix}{key}" not in files]
            if missing:
                raise KeyError(
                    f"Kernel family {selected_family!r} is missing keys: {missing}. "
                    f"Available keys: {sorted(files)}"
                )

        centers_mrad = np.asarray(data[f"{prefix}centers_mrad"], dtype=float)
        edges_mrad = np.asarray(data[f"{prefix}edges_mrad"], dtype=float)
        probabilities = np.asarray(data[f"{prefix}probabilities"], dtype=float)
        L_m = np.asarray(data[f"{prefix}L_m"], dtype=float)
        E_in_GeV = np.asarray(data[f"{prefix}E_in_GeV"], dtype=float)
        clean_for_kernel = np.asarray(data[f"{prefix}clean_for_kernel"], dtype=bool)

    if centers_mrad.ndim != 1:
        raise ValueError(f"centers_mrad must be 1D; got {centers_mrad.shape}")
    if edges_mrad.ndim != 1 or edges_mrad.size != centers_mrad.size + 1:
        raise ValueError(
            "edges_mrad must be 1D with one more element than centers_mrad; "
            f"got edges={edges_mrad.shape}, centers={centers_mrad.shape}"
        )
    if probabilities.shape != (L_m.size, centers_mrad.size):
        raise ValueError(
            "probabilities must have shape (n_kernels, n_centers); "
            f"got {probabilities.shape}, L={L_m.size}, centers={centers_mrad.size}"
        )
    if E_in_GeV.shape != L_m.shape or clean_for_kernel.shape != L_m.shape:
        raise ValueError(
            "L_m, E_in_GeV and clean_for_kernel must have matching shapes; "
            f"got L={L_m.shape}, E={E_in_GeV.shape}, clean={clean_for_kernel.shape}"
        )

    return EmpiricalKernelLibrary(
        path=path,
        family=selected_family,
        centers_mrad=centers_mrad,
        edges_mrad=edges_mrad,
        probabilities=probabilities,
        L_m=L_m,
        E_in_GeV=E_in_GeV,
        clean_for_kernel=clean_for_kernel,
    )


@dataclass(frozen=True)
class TailAwareKernelPrediction:
    centers_mrad: np.ndarray
    probability_per_bin: np.ndarray
    sampling_cdf: np.ndarray
    used_nearest_fallback: bool
    outside_domain: bool
    valid: bool
    interpolation_mode: str
    tail_policy: str


class _CoreKernelInterpolator:
    """Interpolate the broad-domain measured core exactly as built upstream."""

    def __init__(self, model: dict[str, np.ndarray]) -> None:
        self.centers = np.asarray(model["core_centers_mrad"], dtype=float)
        self.edges = np.asarray(model["core_edges_mrad"], dtype=float)
        probability = np.asarray(model["core_probabilities"], dtype=float)
        length = np.asarray(model["core_L_m"], dtype=float)
        energy = np.asarray(model["core_E_in_GeV"], dtype=float)
        clean = np.asarray(model["core_clean_for_kernel"], dtype=bool)
        valid = (
            clean
            & np.isfinite(length) & (length > 0.0)
            & np.isfinite(energy) & (energy > 0.0)
            & np.isfinite(probability).all(axis=1)
            & (probability.sum(axis=1) > 0.0)
        )
        if np.count_nonzero(valid) < 4:
            raise RuntimeError("Too few clean core kernels in hybrid library.")
        self.probability = probability[valid]
        self.length_m = length[valid]
        self.energy_GeV = energy[valid]
        self.features = np.column_stack([np.log10(self.length_m), np.log10(self.energy_GeV)])
        self.mean = self.features.mean(axis=0)
        self.std = self.features.std(axis=0)
        self.std[self.std == 0.0] = 1.0
        self.scaled = (self.features - self.mean) / self.std
        self.rbf = RBFInterpolator(self.scaled, self.probability, kernel="linear", smoothing=0.0)
        self.tri = Delaunay(self.features)

    def energy_bounds_at_length(self, L_m: float) -> tuple[float, float]:
        """Return measured core-energy bounds at the nearest native length."""
        unique_length = np.unique(self.length_m)
        nearest_length = float(unique_length[np.argmin(np.abs(np.log(unique_length / float(L_m))))])
        selected = np.isclose(self.length_m, nearest_length)
        return float(np.min(self.energy_GeV[selected])), float(np.max(self.energy_GeV[selected]))

    @staticmethod
    def _normalize(probability: np.ndarray) -> tuple[np.ndarray, bool]:
        probability = np.asarray(probability, dtype=float).copy()
        probability[~np.isfinite(probability)] = 0.0
        probability[probability < 0.0] = 0.0
        total = float(probability.sum())
        if total <= 0.0 or not np.isfinite(total):
            return probability, False
        probability /= total
        probability = 0.5 * (probability + probability[::-1])
        probability /= probability.sum()
        return probability, True

    def predict(self, L_m: float, E_GeV: float) -> tuple[np.ndarray, str, bool, bool, bool]:
        query = np.array([np.log10(float(L_m)), np.log10(float(E_GeV))], dtype=float)
        outside = bool(self.tri.find_simplex(query[None, :])[0] < 0)
        used_nearest = outside
        if outside:
            index = int(np.argmin(np.sum((self.features - query[None, :]) ** 2, axis=1)))
            raw = self.probability[index]
            mode = "core_nearest"
        else:
            scaled_query = (query - self.mean) / self.std
            raw = np.asarray(self.rbf(scaled_query[None, :])[0], dtype=float)
            mode = "core_rbf_linear"
        probability, valid = self._normalize(raw)
        if not valid and not outside:
            index = int(np.argmin(np.sum((self.features - query[None, :]) ** 2, axis=1)))
            probability, valid = self._normalize(self.probability[index])
            used_nearest = True
            mode = "core_nearest"
        return probability, mode, outside, used_nearest, valid


class TailAwareEmpiricalKernel:
    """Dispatch between broad core and near-threshold full-tail interpolation.

    Inside the measured full-tail domain, the body uses inverse-CDF transport
    and the hard-scattering tail uses local measured histograms. Outside that
    domain, the model's broad core family is interpolated in (L, E), as required
    by the model metadata, and embedded in the common +/-1600 mrad output grid.
    """

    method = "tail-aware"
    tail_start_mrad = 250.0
    tail_full_mrad = 300.0
    tail_interp = "linear"
    policy_description = "tail-aware_full-tail-domain__core-rbf_broad-domain"

    def __init__(
        self,
        npz_path: str | Path,
        *,
        k_nearest: int = 18,
        energy_cache_dlog: float = 0.0,
        max_cache_items: int = 512,
    ) -> None:
        self.path = Path(npz_path)
        model = load_npz(self.path)
        missing = [key for key in TAIL_AWARE_REQUIRED_KEYS if key not in model]
        if missing:
            raise KeyError(
                f"Kernel {self.path} does not contain the tail-aware transport model. "
                f"Missing keys: {missing}"
            )
        self._model = model
        self._transport = LocalQuantileTransport(model, k_nearest=k_nearest)
        self._core = _CoreKernelInterpolator(model)
        self.kernel_family = "hybrid_core_and_full_tail_transport"
        self.edges_mrad = self._transport.edges.copy()
        self.centers_mrad = self._transport.centers.copy()
        self.widths_mrad = np.diff(self.edges_mrad)
        self.transport_L_min_m = float(np.min(self._transport.L))
        self.transport_L_max_m = float(np.max(self._transport.L))
        self.transport_L_nodes_m = np.unique(np.asarray(self._transport.L, dtype=float))
        self.transport_E_min_GeV = float(np.min(self._transport.E))
        self.transport_E_max_GeV = float(np.max(self._transport.E))
        self.energy_cache_dlog = float(energy_cache_dlog)
        self.max_cache_items = max(0, int(max_cache_items))
        self._prediction_cache: OrderedDict[tuple[float, int], TailAwareKernelPrediction] = OrderedDict()
        core_indices = np.searchsorted(self.centers_mrad, self._core.centers)
        if np.any(core_indices >= self.centers_mrad.size) or not np.allclose(
            self.centers_mrad[core_indices], self._core.centers
        ):
            raise RuntimeError("Core and full-tail angular grids are not aligned.")
        self._core_indices = core_indices

    def _inside_full_tail_domain(self, L_m: float, E_GeV: float, t50_GeV: float) -> bool:
        feature = np.array([np.log10(float(L_m)), np.log10(float(E_GeV) / t50_GeV)], dtype=float)
        if self._transport.tri is not None:
            return bool(self._transport.tri.find_simplex(feature[None, :])[0] >= 0)
        minimum = self._transport.features.min(axis=0)
        maximum = self._transport.features.max(axis=0)
        return bool(np.all(feature >= minimum) and np.all(feature <= maximum))

    def core_energy_bounds(self, L_m: float) -> tuple[float, float]:
        """Measured broad-core energy interval at the nearest native length."""
        return self._core.energy_bounds_at_length(L_m)

    def _embed_core(self, probability: np.ndarray) -> np.ndarray:
        embedded = np.zeros_like(self.centers_mrad, dtype=float)
        embedded[self._core_indices] = probability
        return embedded

    def _cache_key_and_energy(self, L_m: float, E_GeV: float) -> tuple[tuple[float, int] | None, float]:
        if self.energy_cache_dlog <= 0.0 or self.max_cache_items <= 0:
            return None, float(E_GeV)
        energy_key = int(round(np.log(float(E_GeV)) / self.energy_cache_dlog))
        energy_use = float(np.exp(energy_key * self.energy_cache_dlog))
        return (round(float(L_m), 6), energy_key), energy_use

    def predict_kernel(self, L_m: float, E_GeV: float) -> TailAwareKernelPrediction:
        if not (np.isfinite(L_m) and np.isfinite(E_GeV) and L_m > 0.0 and E_GeV > 0.0):
            return TailAwareKernelPrediction(
                self.centers_mrad.copy(),
                np.zeros_like(self.centers_mrad),
                np.zeros_like(self.centers_mrad),
                False,
                False,
                False,
                "invalid",
                "none",
            )

        cache_key, energy_use = self._cache_key_and_energy(L_m, E_GeV)
        if cache_key is not None and cache_key in self._prediction_cache:
            prediction = self._prediction_cache.pop(cache_key)
            self._prediction_cache[cache_key] = prediction
            return prediction

        try:
            t50 = estimate_t50(self._model, float(L_m))
            if self._inside_full_tail_domain(float(L_m), energy_use, t50):
                probability, info = self._transport.predict_pdf(
                    float(L_m),
                    energy_use,
                    t50,
                    method=self.method,
                    tail_start_mrad=self.tail_start_mrad,
                    tail_full_mrad=self.tail_full_mrad,
                    tail_interp=self.tail_interp,
                )
                mode = str(info.get("mode", "unknown"))
                outside = False
                used_nearest = mode == "nearest"
                tail_policy = str(info.get("tail_policy", "unknown"))
            else:
                core_probability, mode, outside, used_nearest, core_valid = self._core.predict(
                    float(L_m), energy_use
                )
                if not core_valid:
                    raise RuntimeError("Core interpolation produced an empty PDF.")
                probability = self._embed_core(core_probability)
                tail_policy = "broad_domain_core_measured_support"
        except (FloatingPointError, RuntimeError, ValueError):
            return TailAwareKernelPrediction(
                self.centers_mrad.copy(),
                np.zeros_like(self.centers_mrad),
                np.zeros_like(self.centers_mrad),
                False,
                True,
                False,
                "failed",
                "none",
            )

        probability = np.asarray(probability, dtype=float)
        probability[~np.isfinite(probability)] = 0.0
        probability[probability < 0.0] = 0.0
        total = float(probability.sum())
        valid = total > 0.0 and np.isfinite(total)
        if valid:
            probability /= total
            probability = 0.5 * (probability + probability[::-1])
            probability /= probability.sum()
        sampling_cdf = np.cumsum(probability) if valid else np.zeros_like(probability)
        if valid:
            sampling_cdf /= sampling_cdf[-1]

        prediction = TailAwareKernelPrediction(
            self.centers_mrad.copy(),
            probability,
            sampling_cdf,
            used_nearest,
            outside,
            valid,
            mode,
            tail_policy,
        )
        if cache_key is not None:
            self._prediction_cache[cache_key] = prediction
            if len(self._prediction_cache) > self.max_cache_items:
                self._prediction_cache.popitem(last=False)
        return prediction
