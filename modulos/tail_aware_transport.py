#!/usr/bin/env python3
"""
Tail-preserving full-tail interpolation helper.

This is intended for later use of the empirical kernel.  It avoids interpolating
the whole histogram with a global RBF.  The default mode uses inverse-CDF
transport for the body and local measured-histogram interpolation for the hard
scattering tails.

Properties of the prediction:
  - non-negative
  - normalized
  - monotone CDF by construction
  - no global RBF overshoot
  - measured tail bins are preserved instead of being capped by the last saved
    quantile
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from scipy.spatial import Delaunay
except Exception:  # pragma: no cover - optional dependency fallback
    Delaunay = None


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def estimate_t50(model: dict[str, np.ndarray], L_m: float) -> float:
    L = np.asarray(model["full_tail_L_m"], dtype=float)
    T50 = np.asarray(model["full_tail_T50_GeV"], dtype=float)
    mask = np.isfinite(L) & np.isfinite(T50) & (L > 0.0) & (T50 > 0.0)
    if not np.any(mask):
        raise RuntimeError("No finite T50 table found in model.")

    unique_L = np.array(sorted(set(L[mask])), dtype=float)
    unique_T50 = []
    for value in unique_L:
        unique_T50.append(float(np.mean(T50[mask & (L == value)])))
    unique_T50 = np.asarray(unique_T50, dtype=float)

    logL = np.log10(unique_L)
    logT = np.log10(unique_T50)
    q = np.clip(np.log10(float(L_m)), logL[0], logL[-1])
    return float(10.0 ** np.interp(q, logL, logT))


class LocalQuantileTransport:
    def __init__(self, model: dict[str, np.ndarray], k_nearest: int = 18) -> None:
        self.edges = np.asarray(model["full_tail_edges_mrad"], dtype=float)
        self.centers = 0.5 * (self.edges[:-1] + self.edges[1:])
        self.levels = np.asarray(model["full_tail_transport_quantile_levels"], dtype=float)
        self.features = np.asarray(model["full_tail_transport_features"], dtype=float)
        self.quantiles = np.asarray(model["full_tail_transport_abs_quantiles_mrad"], dtype=float)
        self.L = np.asarray(model["full_tail_transport_L_m"], dtype=float)
        self.E = np.asarray(model["full_tail_transport_E_in_GeV"], dtype=float)
        self.F = np.asarray(model["full_tail_transport_E_over_T50"], dtype=float)
        self.source = np.asarray(model["full_tail_transport_source_families"]).astype(str)
        self.k_nearest = int(k_nearest)
        if len(self.features) < 3:
            raise RuntimeError("Need at least three transport points.")
        self.tri = Delaunay(self.features) if Delaunay is not None else None
        self._logL = self.features[:, 0]
        self._logF = self.features[:, 1]
        self._unique_logL = np.array(sorted(set(np.round(self._logL, 14))), dtype=float)
        if "full_tail_transport_probabilities" in model:
            self.transport_probabilities = np.asarray(model["full_tail_transport_probabilities"], dtype=float)
        else:
            self.transport_probabilities = self._match_transport_probabilities(model)
        if self.transport_probabilities.shape != (len(self.features), len(self.centers)):
            raise RuntimeError("Transport probabilities are not aligned with transport features.")

    def _match_transport_probabilities(self, model: dict[str, np.ndarray]) -> np.ndarray:
        all_prob = np.asarray(model["full_tail_probabilities"], dtype=float)
        all_L = np.asarray(model["full_tail_L_m"], dtype=float)
        all_E = np.asarray(model["full_tail_E_in_GeV"], dtype=float)
        all_source = np.asarray(model["full_tail_source_families"]).astype(str)
        rows = []
        for L, E, src in zip(self.L, self.E, self.source):
            match = np.where(np.isclose(all_L, L) & np.isclose(all_E, E) & (all_source == src))[0]
            if len(match) == 0:
                match = np.where(np.isclose(all_L, L) & np.isclose(all_E, E))[0]
            if len(match) == 0:
                raise RuntimeError(f"Could not match transport probability for L={L:g}, E={E:g}.")
            rows.append(all_prob[int(match[0])])
        return np.asarray(rows, dtype=float)

    @staticmethod
    def _combine_duplicate_weights(indices: list[int], weights: list[float]) -> tuple[np.ndarray, np.ndarray]:
        combined: dict[int, float] = {}
        for idx, weight in zip(indices, weights):
            combined[int(idx)] = combined.get(int(idx), 0.0) + float(weight)
        out_idx = np.array(list(combined.keys()), dtype=int)
        out_w = np.array(list(combined.values()), dtype=float)
        out_w = out_w / np.sum(out_w)
        order = np.argsort(out_idx)
        return out_idx[order], out_w[order]

    def _factor_weights_at_L(self, logL_value: float, target_logF: float) -> tuple[np.ndarray, np.ndarray] | None:
        row = np.where(np.isclose(self._logL, logL_value, atol=5e-13, rtol=0.0))[0]
        if len(row) == 0:
            return None

        order = np.argsort(self._logF[row])
        row = row[order]
        row_logF = self._logF[row]
        eps = 5e-13
        if target_logF < row_logF[0] - eps or target_logF > row_logF[-1] + eps:
            return None

        exact = np.where(np.isclose(row_logF, target_logF, atol=eps, rtol=0.0))[0]
        if len(exact):
            idx = row[exact]
            weights = np.full(len(idx), 1.0 / len(idx), dtype=float)
            return idx, weights

        hi = int(np.searchsorted(row_logF, target_logF, side="right"))
        lo = hi - 1
        if lo < 0 or hi >= len(row):
            return None
        span = row_logF[hi] - row_logF[lo]
        if span <= 0.0:
            return np.array([int(row[lo])], dtype=int), np.array([1.0], dtype=float)
        w_hi = float((target_logF - row_logF[lo]) / span)
        return np.array([int(row[lo]), int(row[hi])], dtype=int), np.array([1.0 - w_hi, w_hi], dtype=float)

    def _grid_weights(self, feature: np.ndarray) -> tuple[np.ndarray, np.ndarray, str] | None:
        target_logL = float(feature[0])
        target_logF = float(feature[1])
        rows: list[tuple[float, np.ndarray, np.ndarray]] = []
        for logL_value in self._unique_logL:
            factor_weights = self._factor_weights_at_L(float(logL_value), target_logF)
            if factor_weights is not None:
                idx, weights = factor_weights
                rows.append((float(logL_value), idx, weights))
        if not rows:
            return None

        row_logL = np.array([row[0] for row in rows], dtype=float)
        eps = 5e-13
        exact = np.where(np.isclose(row_logL, target_logL, atol=eps, rtol=0.0))[0]
        if len(exact):
            _, idx, weights = rows[int(exact[0])]
            return idx, weights, "grid_exact_L"

        if target_logL < row_logL[0] - eps or target_logL > row_logL[-1] + eps:
            nearest = int(np.argmin(np.abs(row_logL - target_logL)))
            _, idx, weights = rows[nearest]
            return idx, weights, "grid_clamped_L"

        hi = int(np.searchsorted(row_logL, target_logL, side="right"))
        lo = hi - 1
        if lo < 0 or hi >= len(rows):
            nearest = int(np.argmin(np.abs(row_logL - target_logL)))
            _, idx, weights = rows[nearest]
            return idx, weights, "grid_clamped_L"

        lo_logL, lo_idx, lo_weights = rows[lo]
        hi_logL, hi_idx, hi_weights = rows[hi]
        span = hi_logL - lo_logL
        if span <= 0.0:
            return lo_idx, lo_weights, "grid_exact_L"
        w_hi_L = float((target_logL - lo_logL) / span)

        indices = [int(i) for i in lo_idx] + [int(i) for i in hi_idx]
        weights = [float(w) * (1.0 - w_hi_L) for w in lo_weights] + [float(w) * w_hi_L for w in hi_weights]
        idx, out_weights = self._combine_duplicate_weights(indices, weights)
        return idx, out_weights, "grid_bilinear"

    def weights(self, feature: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
        grid = self._grid_weights(feature)
        if grid is not None:
            return grid

        simplex = int(self.tri.find_simplex(feature)) if self.tri is not None else -1
        if self.tri is not None and simplex >= 0:
            transform = self.tri.transform[simplex]
            bary = np.dot(transform[:2], feature - transform[2])
            weights = np.r_[bary, 1.0 - np.sum(bary)]
            idx = self.tri.simplices[simplex]
            weights = np.clip(weights, 0.0, 1.0)
            weights = weights / np.sum(weights)
            return idx, weights, "delaunay"

        dist = np.linalg.norm(self.features - feature[None, :], axis=1)
        k = min(self.k_nearest, len(dist))
        idx = np.argsort(dist)[:k]
        inv = 1.0 / np.maximum(dist[idx], 1e-12)
        weights = inv / np.sum(inv)
        return idx, weights, "nearest"

    def predict_quantiles(self, L_m: float, E_in_GeV: float, t50_GeV: float) -> tuple[np.ndarray, dict]:
        feature = np.array([np.log10(L_m), np.log10(E_in_GeV / t50_GeV)], dtype=float)
        idx, weights, mode = self.weights(feature)
        q_abs = np.sum(self.quantiles[idx] * weights[:, None], axis=0)
        q_abs = np.maximum.accumulate(np.clip(q_abs, 0.0, abs(self.edges[-1])))
        info = {
            "mode": mode,
            "indices": idx,
            "weights": weights,
            "neighbor_L_m": self.L[idx],
            "neighbor_E_GeV": self.E[idx],
            "neighbor_E_over_T50": self.F[idx],
            "neighbor_source": self.source[idx],
        }
        return q_abs, info

    def quantiles_to_pdf(self, q_abs: np.ndarray, n_samples: int = 20000) -> np.ndarray:
        max_abs = float(max(abs(self.edges[0]), abs(self.edges[-1])))
        levels = np.r_[0.0, self.levels, 1.0]
        radii = np.r_[0.0, q_abs, max_abs]
        radii = np.maximum.accumulate(radii)
        order = np.argsort(radii)
        radii = radii[order]
        levels = np.maximum.accumulate(levels[order])

        unique_radii: list[float] = []
        unique_levels: list[float] = []
        for radius, level in zip(radii, levels):
            if unique_radii and np.isclose(radius, unique_radii[-1], atol=1e-12, rtol=0.0):
                unique_levels[-1] = max(unique_levels[-1], float(level))
            else:
                unique_radii.append(float(radius))
                unique_levels.append(float(level))

        radii = np.asarray(unique_radii, dtype=float)
        levels = np.maximum.accumulate(np.asarray(unique_levels, dtype=float))
        if radii[0] > 0.0:
            radii = np.r_[0.0, radii]
            levels = np.r_[0.0, levels]
        if radii[-1] < max_abs:
            radii = np.r_[radii, max_abs]
            levels = np.r_[levels, 1.0]
        levels[-1] = 1.0

        bin_width = float(self.edges[1] - self.edges[0])
        abs_edges = np.arange(0.0, max_abs + 0.5 * bin_width, bin_width)
        if abs_edges[-1] < max_abs:
            abs_edges = np.r_[abs_edges, max_abs]
        cdf = np.interp(abs_edges, radii, levels, left=0.0, right=1.0)
        abs_prob = np.clip(np.diff(cdf), 0.0, None)
        if np.sum(abs_prob) <= 0.0:
            raise RuntimeError("Quantile transport produced an empty PDF.")
        abs_prob = abs_prob / np.sum(abs_prob)

        abs_idx = np.searchsorted(abs_edges, np.abs(self.centers), side="right") - 1
        abs_idx = np.clip(abs_idx, 0, len(abs_prob) - 1)
        prob = 0.5 * abs_prob[abs_idx]
        prob = prob / np.sum(prob)
        return prob

    def local_histogram_pdf(
        self,
        indices: np.ndarray,
        weights: np.ndarray,
        tail_interp: str = "linear",
        tail_floor: float = 1e-15,
    ) -> np.ndarray:
        selected = self.transport_probabilities[indices]
        if len(indices) == 1 or tail_interp == "linear":
            prob = np.sum(selected * weights[:, None], axis=0)
        elif tail_interp == "log":
            logp = np.sum(weights[:, None] * np.log(np.maximum(selected, tail_floor)), axis=0)
            prob = np.exp(logp)
        elif tail_interp == "envelope":
            prob = np.max(selected, axis=0)
        else:
            raise ValueError(f"Unknown tail interpolation mode: {tail_interp}")

        prob = np.clip(prob, 0.0, None)
        total = float(np.sum(prob))
        if total <= 0.0:
            raise RuntimeError("Local histogram interpolation produced an empty PDF.")
        return prob / total

    @staticmethod
    def _blend_alpha(centers: np.ndarray, tail_start_mrad: float, tail_full_mrad: float) -> np.ndarray:
        abs_centers = np.abs(centers)
        if tail_full_mrad <= tail_start_mrad:
            return (abs_centers >= tail_start_mrad).astype(float)
        return np.clip((abs_centers - tail_start_mrad) / (tail_full_mrad - tail_start_mrad), 0.0, 1.0)

    def predict_pdf(
        self,
        L_m: float,
        E_in_GeV: float,
        t50_GeV: float,
        method: str = "tail-aware",
        tail_start_mrad: float = 250.0,
        tail_full_mrad: float = 300.0,
        tail_interp: str = "linear",
        tail_floor: float = 1e-15,
        prefer_exact: bool = True,
    ) -> tuple[np.ndarray, dict]:
        q_abs, info = self.predict_quantiles(L_m, E_in_GeV, t50_GeV)
        info["method"] = method
        if method == "quantile":
            info["tail_policy"] = "quantile_only"
            return self.quantiles_to_pdf(q_abs), info

        local_pdf = self.local_histogram_pdf(info["indices"], info["weights"], tail_interp=tail_interp, tail_floor=tail_floor)
        if method == "histogram":
            info["tail_policy"] = f"full_local_histogram_{tail_interp}"
            return local_pdf, info

        if method != "tail-aware":
            raise ValueError(f"Unknown prediction method: {method}")

        if prefer_exact and len(info["indices"]) == 1 and np.isclose(float(info["weights"][0]), 1.0):
            info["tail_policy"] = "exact_measured_histogram"
            return local_pdf, info

        body_pdf = self.quantiles_to_pdf(q_abs)
        alpha = self._blend_alpha(self.centers, tail_start_mrad, tail_full_mrad)
        prob = (1.0 - alpha) * body_pdf + alpha * local_pdf
        prob = np.clip(prob, 0.0, None)
        prob = prob / np.sum(prob)
        info["tail_policy"] = f"body_quantile_tail_histogram_{tail_interp}"
        info["tail_start_mrad"] = float(tail_start_mrad)
        info["tail_full_mrad"] = float(tail_full_mrad)
        return prob, info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parent / "hybrid_empirical_kernel_library.npz",
    )
    parser.add_argument("--L", type=float, required=True, help="Thickness in m")
    parser.add_argument("--E", type=float, required=True, help="Muon kinetic energy in GeV")
    parser.add_argument("--T50", type=float, default=None, help="Optional T50 override in GeV")
    parser.add_argument("--k-nearest", type=int, default=18, help="Fallback convex-neighbor count when SciPy/Delaunay is unavailable")
    parser.add_argument("--method", choices=["tail-aware", "quantile", "histogram"], default="tail-aware")
    parser.add_argument("--tail-start", type=float, default=250.0, help="Start blending measured tail histogram at |theta| in mrad")
    parser.add_argument("--tail-full", type=float, default=300.0, help="Use fully measured tail histogram beyond |theta| in mrad")
    parser.add_argument("--tail-interp", choices=["linear", "log", "envelope"], default="linear")
    parser.add_argument("--tail-floor", type=float, default=1e-15)
    parser.add_argument("--out", type=Path, default=None, help="Optional CSV output with theta_mrad,probability")
    args = parser.parse_args()

    model = load_npz(args.model)
    t50 = float(args.T50) if args.T50 is not None else estimate_t50(model, args.L)
    interp = LocalQuantileTransport(model, k_nearest=args.k_nearest)
    prob, info = interp.predict_pdf(
        args.L,
        args.E,
        t50,
        method=args.method,
        tail_start_mrad=args.tail_start,
        tail_full_mrad=args.tail_full,
        tail_interp=args.tail_interp,
        tail_floor=args.tail_floor,
    )
    centers = interp.centers

    print(f"L_m={args.L:g}")
    print(f"E_in_GeV={args.E:g}")
    print(f"T50_GeV={t50:g}")
    print(f"E_over_T50={args.E / t50:g}")
    print(f"interpolation_mode={info['mode']}")
    print(f"method={info['method']}")
    print(f"tail_policy={info['tail_policy']}")
    print(f"normalization={np.sum(prob):.12g}")
    print(f"tail_gt_300mrad={np.sum(prob[np.abs(centers) > 300.0]):.12g}")
    print(f"tail_gt_500mrad={np.sum(prob[np.abs(centers) > 500.0]):.12g}")
    print(f"tail_gt_1000mrad={np.sum(prob[np.abs(centers) > 1000.0]):.12g}")
    print("neighbors:")
    for L, E, f, w, src in zip(
        info["neighbor_L_m"],
        info["neighbor_E_GeV"],
        info["neighbor_E_over_T50"],
        info["weights"],
        info["neighbor_source"],
    ):
        print(f"  L={L:g} m  E={E:g} GeV  E/T50={f:.4f}  weight={w:.4f}  source={src}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            args.out,
            np.column_stack([centers, prob]),
            delimiter=",",
            header="theta_mrad,probability",
            comments="",
        )
        print(f"wrote={args.out}")


if __name__ == "__main__":
    main()
