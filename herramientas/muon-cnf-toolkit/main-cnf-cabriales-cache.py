#!/usr/bin/env python3
"""Generate a CABRIALES v1 kinematic cache directly from the CNF model."""

import argparse
import json
import random
import shutil
import time
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from nflows import distributions, flows, transforms


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = Path(__file__).resolve().parent / "model.pt"
OUTPUT_DIR = Path(__file__).resolve().parent / "generated_kinematic_cache"

AREA_M2 = 1.0
MUON_MASS_GEV = 0.1056583755
PZ_SIGN = 1

BATCH_SIZE = 65536
CHUNK_EVENTS = 1_000_000
WRITE_WORKERS = 6
WRITE_QUEUE_SIZE = WRITE_WORKERS * 2

Y_MIN = 0.0
Y_MAX = 0.87
MU_MIN = 1e-6
MU_MAX = 1.0 - 1e-6

SEEDS = [5, 503, 1001, 1501, 1999]

MUON_POSITIVE_ID = 5
MUON_NEGATIVE_ID = 6

MUON_POSITIVE_FRACTION = 0.55
MUON_NEGATIVE_FRACTION = 0.45

# Architecture parameters
HIDDEN_FEATURES = 192
NUM_LAYERS = 6
NUM_BINS = 4
TAIL_BOUND = 8
CONTEXT_FEATURES = 3

# Min-Max statistics
H_MIN = 0.0
H_RANGE = 5230.0

BX_MIN = 9.63
BX_RANGE = 26.488

BZ_MIN = -14.326
BZ_RANGE = 70.254

LOGE_MIN = -2.1570927646089166
LOGE_RANGE = 11.345521211095374


# ============================================================
# Seed
# ============================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Flux and number of particles
# ============================================================

def expected_flux_from_height(height: float) -> float:
    """
    Expected flux in particles/(m²·s).
    """
    return 100.4 * np.exp(0.0002119 * height)


def expected_number_of_particles(height: float, simulation_seconds: float) -> int:
    flux = expected_flux_from_height(height)

    number_of_particles = round(
        flux * simulation_seconds * AREA_M2
    )

    return max(int(number_of_particles), 1)


# ============================================================
# Context normalization
# ============================================================

def normalize_context(
    height: float,
    bx: float,
    bz: float,
    device: torch.device,
) -> torch.Tensor:
    height_normalized = (height - H_MIN) / H_RANGE
    bx_normalized = (bx - BX_MIN) / BX_RANGE
    bz_normalized = (bz - BZ_MIN) / BZ_RANGE

    return torch.tensor(
        [[height_normalized, bx_normalized, bz_normalized]],
        dtype=torch.float32,
        device=device,
    )


# ============================================================
# CNF architecture and model loading
# ============================================================

def build_flow() -> flows.Flow:
    layers = []

    for _ in range(NUM_LAYERS):
        layers.append(
            transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=2,
                hidden_features=HIDDEN_FEATURES,
                num_bins=NUM_BINS,
                tails="linear",
                tail_bound=float(TAIL_BOUND),
                context_features=CONTEXT_FEATURES,
                min_bin_width=1e-3,
                min_bin_height=1e-4,
                min_derivative=1e-3,
            )
        )

        layers.append(
            transforms.ReversePermutation(features=2)
        )

    transform = transforms.CompositeTransform(layers)
    base_distribution = distributions.StandardNormal(shape=[2])

    return flows.Flow(transform, base_distribution)


def load_model(model_path: Path, device: torch.device) -> flows.Flow:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at: {model_path}"
        )

    flow = build_flow()

    checkpoint = torch.load(
        model_path,
        map_location=device,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

    flow.load_state_dict(state_dict)
    flow = flow.to(device)
    flow.eval()

    return flow


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False."
        )

    return torch.device(device_name)


# ============================================================
# Kinematics
# ============================================================

def normalized_log_energy_to_energy(
    normalized_log_energy: np.ndarray,
) -> np.ndarray:
    log_energy = (
        normalized_log_energy * LOGE_RANGE
        + LOGE_MIN
    )

    return np.exp(log_energy)


def mu_to_theta(mu: np.ndarray) -> np.ndarray:
    mu = np.clip(mu, MU_MIN, MU_MAX)

    theta_radians = np.arccos(1.0 - mu)

    return np.rad2deg(theta_radians)


def build_kinematic_arrays(
    normalized_log_energy: np.ndarray,
    mu: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    energy = normalized_log_energy_to_energy(normalized_log_energy)
    theta = mu_to_theta(mu)
    phi_abs_deg = rng.uniform(
        0.0,
        360.0,
        size=len(energy),
    )

    theta_radians = np.deg2rad(theta)
    momentum_squared = energy**2 - MUON_MASS_GEV**2
    bad_momentum = int(np.count_nonzero(momentum_squared <= 0.0))
    momentum = np.sqrt(np.maximum(momentum_squared, 0.0))
    pz = PZ_SIGN * momentum * np.cos(theta_radians)

    pid_code = rng.choice(
        [MUON_POSITIVE_ID, MUON_NEGATIVE_ID],
        size=len(energy),
        p=[MUON_POSITIVE_FRACTION, MUON_NEGATIVE_FRACTION],
    )

    return (
        np.asarray(theta, dtype=np.float32),
        np.asarray(phi_abs_deg, dtype=np.float32),
        np.asarray(energy, dtype=np.float32),
        np.asarray(pz > 0.0, dtype=np.uint8),
        np.asarray(pid_code, dtype=np.uint8),
        bad_momentum,
    )


# ============================================================
# Chunk writing
# ============================================================

def write_cache_chunk(
    chunks_dir: Path,
    chunk_index: int,
    theta_deg: np.ndarray,
    phi_abs_deg: np.ndarray,
    total_e_gev: np.ndarray,
    pz_positive: np.ndarray,
    pid_code: np.ndarray,
) -> dict:
    file_name = f"chunk_{chunk_index:06d}.npz"
    chunk_path = chunks_dir / file_name

    np.savez_compressed(
        chunk_path,
        theta_deg=np.asarray(theta_deg, dtype=np.float32),
        phi_abs_deg=np.asarray(phi_abs_deg, dtype=np.float32),
        total_E_GeV=np.asarray(total_e_gev, dtype=np.float32),
        pz_positive=np.asarray(pz_positive, dtype=np.uint8),
        pid_code=np.asarray(pid_code, dtype=np.uint8),
    )

    return {
        "file": file_name,
        "n_events": int(len(theta_deg)),
        "bytes": int(chunk_path.stat().st_size),
    }


def drain_completed_writes(
    pending_writes: deque[Future],
    chunks: list[dict],
) -> None:
    while pending_writes and pending_writes[0].done():
        chunks.append(pending_writes.popleft().result())


def wait_for_write_slot(
    pending_writes: deque[Future],
    chunks: list[dict],
    write_queue_size: int,
) -> None:
    drain_completed_writes(pending_writes, chunks)

    while len(pending_writes) >= write_queue_size:
        chunks.append(pending_writes.popleft().result())
        drain_completed_writes(pending_writes, chunks)


def concatenate_buffers(
    buffers: dict[str, list[np.ndarray]],
) -> dict[str, np.ndarray]:
    return {
        name: np.concatenate(values)
        for name, values in buffers.items()
    }


def reset_buffers() -> dict[str, list[np.ndarray]]:
    return {
        "theta_deg": [],
        "phi_abs_deg": [],
        "total_E_GeV": [],
        "pz_positive": [],
        "pid_code": [],
    }


def append_to_buffers(
    buffers: dict[str, list[np.ndarray]],
    theta_deg: np.ndarray,
    phi_abs_deg: np.ndarray,
    total_e_gev: np.ndarray,
    pz_positive: np.ndarray,
    pid_code: np.ndarray,
) -> int:
    buffers["theta_deg"].append(theta_deg)
    buffers["phi_abs_deg"].append(phi_abs_deg)
    buffers["total_E_GeV"].append(total_e_gev)
    buffers["pz_positive"].append(pz_positive)
    buffers["pid_code"].append(pid_code)

    return len(theta_deg)


def submit_chunk(
    writer: ProcessPoolExecutor,
    pending_writes: deque[Future],
    chunks_dir: Path,
    chunk_index: int,
    chunk_data: dict[str, np.ndarray],
) -> None:
    pending_writes.append(
        writer.submit(
            write_cache_chunk,
            chunks_dir,
            chunk_index,
            np.ascontiguousarray(chunk_data["theta_deg"], dtype=np.float32),
            np.ascontiguousarray(chunk_data["phi_abs_deg"], dtype=np.float32),
            np.ascontiguousarray(chunk_data["total_E_GeV"], dtype=np.float32),
            np.ascontiguousarray(chunk_data["pz_positive"], dtype=np.uint8),
            np.ascontiguousarray(chunk_data["pid_code"], dtype=np.uint8),
        )
    )


def flush_ready_chunks(
    buffers: dict[str, list[np.ndarray]],
    buffered_events: int,
    chunk_events: int,
    writer: ProcessPoolExecutor,
    pending_writes: deque[Future],
    chunks: list[dict],
    chunks_dir: Path,
    chunk_index: int,
    write_queue_size: int,
) -> tuple[dict[str, list[np.ndarray]], int, int]:
    while buffered_events >= chunk_events:
        combined = concatenate_buffers(buffers)
        chunk_data = {
            name: values[:chunk_events]
            for name, values in combined.items()
        }
        remainder = {
            name: values[chunk_events:]
            for name, values in combined.items()
        }

        wait_for_write_slot(
            pending_writes,
            chunks,
            write_queue_size,
        )

        submit_chunk(
            writer,
            pending_writes,
            chunks_dir,
            chunk_index,
            chunk_data,
        )

        chunk_index += 1
        buffered_events -= chunk_events
        buffers = reset_buffers()

        if buffered_events > 0:
            for name, values in remainder.items():
                buffers[name].append(values)

    return buffers, buffered_events, chunk_index


def write_manifest(
    output_dir: Path,
    source_shw: str | None,
    chunk_events_requested: int,
    n_events: int,
    n_bad_momentum: int,
    elapsed_s: float,
    chunks: list[dict],
) -> None:
    cache_bytes = int(sum(chunk["bytes"] for chunk in chunks))

    manifest = {
        "version": 1,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_shw": source_shw,
        "shw_format": "cnf9",
        "shw_member": None,
        "only_muons": True,
        "chunk_events_requested": int(chunk_events_requested),
        "compressed": True,
        "arrays": {
            "theta_deg": "float32",
            "phi_abs_deg": "float32",
            "total_E_GeV": "float32",
            "pz_positive": "uint8",
            "pid_code": "uint8",
        },
        "n_lines_read": int(n_events),
        "n_particles_read": int(n_events),
        "n_muons_read": int(n_events),
        "n_bad_momentum": int(n_bad_momentum),
        "n_events": int(n_events),
        "n_chunks": int(len(chunks)),
        "cache_bytes": cache_bytes,
        "elapsed_s": round(float(elapsed_s), 2),
        "chunks": chunks,
    }

    manifest_path = output_dir / "manifest.json"

    while True:
        manifest_path.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

        total_cache_bytes = cache_bytes + manifest_path.stat().st_size
        if manifest["cache_bytes"] == total_cache_bytes:
            break

        manifest["cache_bytes"] = total_cache_bytes


# ============================================================
# Cache generation
# ============================================================

@torch.inference_mode()
def generate_kinematic_cache(
    flow: flows.Flow,
    context: torch.Tensor,
    number_of_particles: int,
    batch_size: int,
    chunk_events: int,
    write_workers: int,
    write_queue_size: int,
    selected_seed: int,
    output_dir: Path,
    source_shw: str | None,
) -> dict:
    start_time = time.perf_counter()
    rng = np.random.default_rng(selected_seed)

    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    generated_particles = 0
    saved_events = 0
    n_bad_momentum = 0
    buffered_events = 0
    chunk_index = 0

    buffers = reset_buffers()
    pending_writes: deque[Future] = deque()
    chunks: list[dict] = []

    with ProcessPoolExecutor(max_workers=write_workers) as writer:
        while generated_particles < number_of_particles:
            current_batch_size = min(
                batch_size,
                number_of_particles - generated_particles,
            )

            samples = flow.sample(
                current_batch_size,
                context=context,
            )

            samples = samples.reshape(-1, 2)
            samples = samples.detach().cpu().numpy()

            generated_particles += current_batch_size

            normalized_log_energy = samples[:, 0]
            mu = samples[:, 1]

            valid_mask = (
                (normalized_log_energy >= Y_MIN)
                & (normalized_log_energy <= Y_MAX)
                & (mu >= MU_MIN)
                & (mu <= MU_MAX)
            )

            normalized_log_energy = normalized_log_energy[valid_mask]
            mu = mu[valid_mask]

            if len(normalized_log_energy) == 0:
                continue

            (
                theta_deg,
                phi_abs_deg,
                total_e_gev,
                pz_positive,
                pid_code,
                bad_momentum,
            ) = build_kinematic_arrays(
                normalized_log_energy,
                mu,
                rng,
            )

            n_bad_momentum += bad_momentum
            saved_events += len(theta_deg)
            buffered_events += append_to_buffers(
                buffers,
                theta_deg,
                phi_abs_deg,
                total_e_gev,
                pz_positive,
                pid_code,
            )

            buffers, buffered_events, chunk_index = flush_ready_chunks(
                buffers,
                buffered_events,
                chunk_events,
                writer,
                pending_writes,
                chunks,
                chunks_dir,
                chunk_index,
                write_queue_size,
            )

        if buffered_events > 0:
            wait_for_write_slot(
                pending_writes,
                chunks,
                write_queue_size,
            )

            submit_chunk(
                writer,
                pending_writes,
                chunks_dir,
                chunk_index,
                concatenate_buffers(buffers),
            )

        while pending_writes:
            chunks.append(pending_writes.popleft().result())

    elapsed_s = time.perf_counter() - start_time

    write_manifest(
        output_dir=output_dir,
        source_shw=source_shw,
        chunk_events_requested=chunk_events,
        n_events=saved_events,
        n_bad_momentum=n_bad_momentum,
        elapsed_s=elapsed_s,
        chunks=chunks,
    )

    return {
        "n_events": saved_events,
        "n_chunks": len(chunks),
        "n_bad_momentum": n_bad_momentum,
        "elapsed_s": elapsed_s,
        "cache_bytes": sum(chunk["bytes"] for chunk in chunks)
        + (output_dir / "manifest.json").stat().st_size,
    }


# ============================================================
# Main execution
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate a CABRIALES v1 kinematic cache using a trained CNF."
    )

    time_group = parser.add_mutually_exclusive_group(required=True)

    time_group.add_argument(
        "--seconds",
        type=float,
        help="Simulation time in seconds.",
    )

    time_group.add_argument(
        "--days",
        type=float,
        help="Simulation time in days.",
    )

    parser.add_argument(
        "--height",
        type=float,
        required=True,
        help="Site altitude in meters.",
    )

    parser.add_argument(
        "--bx",
        type=float,
        required=True,
        help="Geomagnetic Bx component in microteslas.",
    )

    parser.add_argument(
        "--bz",
        type=float,
        required=True,
        help="Geomagnetic Bz component in microteslas.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        choices=SEEDS,
        default=None,
        help=(
            "Optional seed. If not specified, one is automatically "
            f"selected from: {SEEDS}"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output CABRIALES cache directory.",
    )

    parser.add_argument(
        "--chunk-events",
        type=int,
        default=CHUNK_EVENTS,
        help="Events per NPZ chunk.",
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help=(
            "Execution device. Use 'auto' to select CUDA when available."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=(
            "Number of particles sampled per model call. Larger batches "
            "usually improve GPU throughput but require more memory."
        ),
    )

    parser.add_argument(
        "--write-workers",
        type=int,
        default=WRITE_WORKERS,
        help="Parallel processes used to write compressed NPZ chunks.",
    )

    parser.add_argument(
        "--write-queue-size",
        type=int,
        default=WRITE_QUEUE_SIZE,
        help="Maximum number of generated chunks waiting for writing.",
    )

    parser.add_argument(
        "--torch-threads",
        type=int,
        default=None,
        help="Optional number of CPU threads for PyTorch operations.",
    )

    parser.add_argument(
        "--source-shw",
        type=str,
        default=None,
        help="Optional source SHW path recorded in manifest.json.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the output directory before generation if it already exists.",
    )

    args = parser.parse_args()

    if args.seconds is not None and args.seconds <= 0:
        parser.error("--seconds must be greater than zero.")

    if args.days is not None and args.days <= 0:
        parser.error("--days must be greater than zero.")

    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero.")

    if args.chunk_events <= 0:
        parser.error("--chunk-events must be greater than zero.")

    if args.write_workers <= 0:
        parser.error("--write-workers must be greater than zero.")

    if args.write_queue_size <= 0:
        parser.error("--write-queue-size must be greater than zero.")

    if args.torch_threads is not None and args.torch_threads <= 0:
        parser.error("--torch-threads must be greater than zero.")

    return args


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Use --overwrite to replace it."
            )

        shutil.rmtree(output_dir)

    (output_dir / "chunks").mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_arguments()

    simulation_seconds = (
        args.seconds
        if args.seconds is not None
        else args.days * 86400.0
    )
    height = args.height
    bx = args.bx
    bz = args.bz

    selected_seed = (
        args.seed
        if args.seed is not None
        else random.SystemRandom().choice(SEEDS)
    )

    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)

    output_dir = args.output_dir
    prepare_output_dir(
        output_dir=output_dir,
        overwrite=args.overwrite,
    )

    flux = expected_flux_from_height(height)

    number_of_particles = expected_number_of_particles(
        height=height,
        simulation_seconds=simulation_seconds,
    )

    device = select_device(args.device)

    print(f"Device: {device}")
    print(f"Simulated time: {simulation_seconds} s")
    print(f"Altitude: {height} m")
    print(f"Bx: {bx} µT")
    print(f"Bz: {bz} µT")
    print(f"Seed: {selected_seed}")
    print(f"Particles to generate: {number_of_particles:,}")
    print(f"Batch size: {args.batch_size:,}")
    print(f"Chunk events: {args.chunk_events:,}")
    print(f"Write workers: {args.write_workers}")
    print(f"Write queue size: {args.write_queue_size}")
    print(f"PyTorch threads: {torch.get_num_threads()}")
    print(f"Output cache: {output_dir.resolve()}")

    context = normalize_context(
        height=height,
        bx=bx,
        bz=bz,
        device=device,
    )

    flow = load_model(
        model_path=MODEL_PATH,
        device=device,
    )

    seed_everything(selected_seed)

    summary = generate_kinematic_cache(
        flow=flow,
        context=context,
        number_of_particles=number_of_particles,
        batch_size=args.batch_size,
        chunk_events=args.chunk_events,
        write_workers=args.write_workers,
        write_queue_size=args.write_queue_size,
        selected_seed=selected_seed,
        output_dir=output_dir,
        source_shw=args.source_shw,
    )

    print("\nCABRIALES cache completed.")
    print("Estimated flux: " f"{flux:.6g} particles/(m2*s)")
    print(f"Saved events: {summary['n_events']:,}")
    print(f"Chunks: {summary['n_chunks']:,}")
    print(f"Bad momentum events: {summary['n_bad_momentum']:,}")
    print(f"Cache size: {summary['cache_bytes'] / 1024**3:.3f} GiB")
    print(f"Elapsed time: {summary['elapsed_s']:.2f} s")
    print(f"Manifest: {(output_dir / 'manifest.json').resolve()}")


if __name__ == "__main__":
    main()
