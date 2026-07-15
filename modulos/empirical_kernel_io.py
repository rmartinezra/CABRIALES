#!/usr/bin/env python3
"""Shared loader for empirical angular-migration kernel libraries."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REQUIRED_KERNEL_KEYS = (
    "centers_mrad",
    "edges_mrad",
    "probabilities",
    "L_m",
    "E_in_GeV",
    "clean_for_kernel",
)


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

    if "core" in families:
        return "core"
    if "full_tail" in families:
        return "full_tail"
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
