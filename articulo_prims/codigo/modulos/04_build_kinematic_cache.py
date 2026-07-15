#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a compact chunked kinematic cache from a SHW/tar input.

The cache stores muon-level kinematics that are independent of detector point:

  theta_deg, phi_abs_deg, total_E_GeV, pz_positive, pid_code

It is deliberately chunked so multi-day inputs can be cached without holding all
events in RAM. Downstream stages can iterate over the chunks and avoid parsing
the large text SHW again.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np

try:
    from shw_io import open_shw_bytes, parse_muon_parts, stream_size_hint, theta_phi_from_momentum
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from shw_io import open_shw_bytes, parse_muon_parts, stream_size_hint, theta_phi_from_momentum

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    class tqdm:
        def __init__(self, iterable=None, total=None, **kwargs):
            self.iterable = iterable
        def __iter__(self):
            return iter(self.iterable) if self.iterable is not None else iter(())
        def update(self, n=1):
            pass
        def close(self):
            pass


MUON_IDS_B = {b"0005", b"0006", b"5", b"6"}


def now_stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def pid_code(pid: bytes) -> int:
    if pid in {b"0005", b"5"}:
        return 5
    if pid in {b"0006", b"6"}:
        return 6
    return 0


def flush_chunk(
    chunks_dir: Path,
    chunk_index: int,
    theta: list[float],
    phi_abs: list[float],
    total_e: list[float],
    pz_positive: list[int],
    pid_codes: list[int],
    compress: bool,
) -> dict[str, object]:
    chunks_dir.mkdir(parents=True, exist_ok=True)
    path = chunks_dir / f"chunk_{chunk_index:06d}.npz"
    arrays = {
        "theta_deg": np.asarray(theta, dtype=np.float32),
        "phi_abs_deg": np.asarray(phi_abs, dtype=np.float32),
        "total_E_GeV": np.asarray(total_e, dtype=np.float32),
        "pz_positive": np.asarray(pz_positive, dtype=np.uint8),
        "pid_code": np.asarray(pid_codes, dtype=np.uint8),
    }
    if compress:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)
    return {
        "file": str(path.name),
        "n_events": int(arrays["theta_deg"].size),
        "bytes": int(path.stat().st_size),
    }


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Build a compact chunked muon kinematic cache from SHW/tar input.")
    ap.add_argument("--shw", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="Output cache directory.")
    ap.add_argument("--shw-format", choices=["auto", "arti12", "cnf9"], default="auto")
    ap.add_argument("--shw-member", default=None)
    ap.add_argument("--chunk-events", type=int, default=1_000_000)
    ap.add_argument("--progress-update-mb", type=float, default=64.0)
    ap.add_argument("--head", type=int, default=0, help="Debug: stop after N cached events.")
    ap.add_argument("--include-all", dest="only_muons", action="store_false", help="Cache all parsed particles; default only muons.")
    ap.set_defaults(only_muons=True)
    ap.add_argument("--uncompressed", action="store_true", help="Write .npz chunks without deflate compression.")
    ap.add_argument("--force", action="store_true", help="Overwrite output cache directory.")
    return ap


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if not args.shw.exists():
        raise FileNotFoundError(args.shw)
    if args.out.exists():
        if not args.force:
            raise FileExistsError(f"{args.out} ya existe. Usa --force para regenerarlo.")
        shutil.rmtree(args.out)
    chunks_dir = args.out / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    total_bytes = stream_size_hint(args.shw)
    update_bytes = max(1, int(args.progress_update_mb * 1024 * 1024))
    pending_update = 0

    theta: list[float] = []
    phi_abs: list[float] = []
    total_e: list[float] = []
    pz_positive: list[int] = []
    pid_codes: list[int] = []
    chunks: list[dict[str, object]] = []

    n_lines = 0
    n_particles = 0
    n_muons = 0
    n_bad_momentum = 0
    chunk_index = 0
    t0 = time.time()

    def flush_if_needed(force: bool = False) -> None:
        nonlocal chunk_index
        if not theta:
            return
        if (not force) and len(theta) < args.chunk_events:
            return
        info = flush_chunk(
            chunks_dir=chunks_dir,
            chunk_index=chunk_index,
            theta=theta,
            phi_abs=phi_abs,
            total_e=total_e,
            pz_positive=pz_positive,
            pid_codes=pid_codes,
            compress=not args.uncompressed,
        )
        chunks.append(info)
        chunk_index += 1
        theta.clear()
        phi_abs.clear()
        total_e.clear()
        pz_positive.clear()
        pid_codes.clear()

    with open_shw_bytes(args.shw, member_name=args.shw_member) as fin:
        pbar = tqdm(total=total_bytes, unit="B", unit_scale=True, desc="kinematic-cache")
        for raw in fin:
            n_lines += 1
            pending_update += len(raw)
            if pending_update >= update_bytes:
                pbar.update(pending_update)
                pending_update = 0

            s = raw.strip()
            if not s or s.startswith(b"#"):
                continue

            rec = parse_muon_parts(s.split(), shw_format=args.shw_format, only_muons=args.only_muons)
            if rec is None:
                continue
            n_particles += 1
            if rec.pid in MUON_IDS_B:
                n_muons += 1
            angles = theta_phi_from_momentum(rec.px, rec.py, rec.pz)
            if angles is None:
                n_bad_momentum += 1
                continue
            th, ph = angles
            theta.append(float(th))
            phi_abs.append(float(ph))
            total_e.append(float(rec.e_total_GeV))
            pz_positive.append(1 if rec.pz > 0.0 else 0)
            pid_codes.append(pid_code(rec.pid))

            flush_if_needed(force=False)
            if args.head and sum(int(c["n_events"]) for c in chunks) + len(theta) >= args.head:
                break

        if pending_update:
            pbar.update(pending_update)
        pbar.close()
    flush_if_needed(force=True)

    n_events = int(sum(int(c["n_events"]) for c in chunks))
    total_cache_bytes = int(sum(int(c["bytes"]) for c in chunks))
    elapsed = time.time() - t0
    manifest = {
        "version": 1,
        "created_at": now_stamp(),
        "source_shw": str(args.shw.resolve()),
        "shw_format": args.shw_format,
        "shw_member": args.shw_member,
        "only_muons": bool(args.only_muons),
        "chunk_events_requested": int(args.chunk_events),
        "compressed": not args.uncompressed,
        "arrays": {
            "theta_deg": "float32",
            "phi_abs_deg": "float32",
            "total_E_GeV": "float32",
            "pz_positive": "uint8",
            "pid_code": "uint8",
        },
        "n_lines_read": int(n_lines),
        "n_particles_read": int(n_particles),
        "n_muons_read": int(n_muons),
        "n_bad_momentum": int(n_bad_momentum),
        "n_events": n_events,
        "n_chunks": len(chunks),
        "cache_bytes": total_cache_bytes,
        "elapsed_s": elapsed,
        "chunks": chunks,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\n[OK] Kinematic cache finished")
    print(f"Cache: {args.out}")
    print(f"Events: {n_events}")
    print(f"Chunks: {len(chunks)}")
    print(f"Size: {total_cache_bytes / (1024**2):.1f} MiB")
    print(f"Elapsed: {elapsed:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
