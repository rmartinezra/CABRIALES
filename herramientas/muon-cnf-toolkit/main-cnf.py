#!/usr/bin/env python3
"""Generate a CNF muon flux and export it in SHW format."""

import random
from pathlib import Path
import argparse
from concurrent.futures import Future, ProcessPoolExecutor
from collections import deque
from tempfile import TemporaryDirectory
import shutil

import numpy as np
import torch
from nflows import distributions, flows, transforms


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = Path(__file__).resolve().parent / "model.pt"
OUTPUT_PATH = Path(__file__).resolve().parent / "generated_muons.shw"

AREA_M2 = 1.0
MUON_MASS_GEV = 0.1056583755
PZ_SIGN = 1

BATCH_SIZE = 65536
WRITE_WORKERS = 6
WRITE_QUEUE_SIZE = WRITE_WORKERS * 2
SHW_FORMAT = "%d %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e"

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
    device: torch.device
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
# Denormalization
# ============================================================

def normalized_log_energy_to_energy(
    normalized_log_energy: np.ndarray
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


# ============================================================
# Momentum components
# ============================================================

def calculate_momentum_components(
    energy: np.ndarray,
    theta_degrees: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    theta_radians = np.deg2rad(theta_degrees)

    phi_radians = rng.uniform(
        0.0,
        2.0 * np.pi,
        size=len(energy),
    )

    momentum = np.sqrt(
        np.maximum(
            energy**2 - MUON_MASS_GEV**2,
            0.0,
        )
    )

    px = momentum * np.sin(theta_radians) * np.cos(phi_radians)
    py = momentum * np.sin(theta_radians) * np.sin(phi_radians)
    pz = PZ_SIGN * momentum * np.cos(theta_radians)

    return px, py, pz


def write_shw_batch(output_file, output_data: np.ndarray) -> None:
    np.savetxt(
        output_file,
        output_data,
        fmt=SHW_FORMAT,
        delimiter=" ",
    )


def write_shw_part(part_path: Path, output_data: np.ndarray) -> int:
    with part_path.open("w", encoding="utf-8") as part_file:
        write_shw_batch(part_file, output_data)

    return len(output_data)


def assemble_shw_file(
    output_path: Path,
    part_paths: list[Path],
) -> None:
    with output_path.open("wb") as output_file:
        output_file.write(b"# # # shw\n")
        output_file.write(b"# # CNFId energy theta px py pz h bx bz\n")

        for part_path in part_paths:
            with part_path.open("rb") as part_file:
                shutil.copyfileobj(part_file, output_file, length=1024 * 1024)


def drain_completed_writes(pending_writes: deque[Future]) -> None:
    while pending_writes and pending_writes[0].done():
        pending_writes.popleft().result()


def wait_for_write_slot(
    pending_writes: deque[Future],
    write_queue_size: int,
) -> None:
    drain_completed_writes(pending_writes)

    while len(pending_writes) >= write_queue_size:
        pending_writes.popleft().result()
        drain_completed_writes(pending_writes)


# ============================================================
# Inference and SHW file creation
# ============================================================

@torch.inference_mode()
def generate_shw(
    flow: flows.Flow,
    context: torch.Tensor,
    number_of_particles: int,
    batch_size: int,
    async_write: bool,
    write_workers: int,
    write_queue_size: int,
    height: float,
    bx: float,
    bz: float,
    selected_seed: int,
    output_path: Path,
) -> int:

    rng = np.random.default_rng(selected_seed)

    generated_particles = 0
    saved_particles = 0

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_path.open("w", encoding="utf-8")
        if not async_write
        else None
    )
    writer = (
        ProcessPoolExecutor(max_workers=write_workers)
        if async_write
        else None
    )
    pending_writes: deque[Future] = deque()
    part_paths: list[Path] = []

    with TemporaryDirectory(
        prefix=f".{output_path.name}.parts-",
        dir=output_path.parent,
    ) as temp_dir_name:
        temp_dir = Path(temp_dir_name)

        try:
            if output_file is not None:
                output_file.write("# # # shw\n")
                output_file.write("# # CNFId energy theta px py pz h bx bz\n")

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

                # The model first generates all particles indicated by the formula
                # The cutoff is applied afterward
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

                energy = normalized_log_energy_to_energy(
                    normalized_log_energy
                )

                theta = mu_to_theta(mu)

                px, py, pz = calculate_momentum_components(
                    energy=energy,
                    theta_degrees=theta,
                    rng=rng,
                )

                height_column = np.full(
                    len(energy),
                    height,
                )

                bx_column = np.full(
                    len(energy),
                    bx,
                )

                bz_column = np.full(
                    len(energy),
                    bz,
                )

                cnf_id_column = rng.choice(
                    [MUON_POSITIVE_ID, MUON_NEGATIVE_ID],
                    size=len(energy),
                    p=[MUON_POSITIVE_FRACTION, MUON_NEGATIVE_FRACTION],
                )

                output_data = np.column_stack(
                    (
                        cnf_id_column,
                        energy,
                        theta,
                        px,
                        py,
                        pz,
                        height_column,
                        bx_column,
                        bz_column,
                    )
                )

                if writer is None:
                    write_shw_batch(output_file, output_data)
                else:
                    wait_for_write_slot(
                        pending_writes,
                        write_queue_size,
                    )

                    part_path = temp_dir / f"part_{len(part_paths):08d}.shw"
                    part_paths.append(part_path)

                    pending_writes.append(
                        writer.submit(
                            write_shw_part,
                            part_path,
                            output_data,
                        )
                    )

                saved_particles += len(output_data)

            while pending_writes:
                pending_writes.popleft().result()

            if writer is not None:
                assemble_shw_file(output_path, part_paths)

        finally:
            if writer is not None:
                writer.shutdown(wait=True)

            if output_file is not None:
                output_file.close()

    return saved_particles


# ============================================================
# Main execution
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate a muon SHW file using a trained CNF."
    )

    parser.add_argument(
        "--seconds",
        type=float,
        required=True,
        help="Simulation time in seconds.",
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
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Output SHW file path.",
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
        "--torch-threads",
        type=int,
        default=None,
        help=(
            "Optional number of CPU threads for PyTorch operations."
        ),
    )

    parser.add_argument(
        "--sync-write",
        action="store_true",
        help=(
            "Write each batch synchronously. Useful for debugging or "
            "benchmarking against the default overlapped writer."
        ),
    )

    parser.add_argument(
        "--write-workers",
        type=int,
        default=WRITE_WORKERS,
        help=(
            "Number of parallel processes used to write temporary SHW parts."
        ),
    )

    parser.add_argument(
        "--write-queue-size",
        type=int,
        default=WRITE_QUEUE_SIZE,
        help=(
            "Maximum number of generated batches waiting for the "
            "asynchronous writer."
        ),
    )

    args = parser.parse_args()

    if args.seconds <= 0:
        parser.error("--seconds must be greater than zero.")

    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero.")

    if args.torch_threads is not None and args.torch_threads <= 0:
        parser.error("--torch-threads must be greater than zero.")

    if args.write_workers <= 0:
        parser.error("--write-workers must be greater than zero.")

    if args.write_queue_size <= 0:
        parser.error("--write-queue-size must be greater than zero.")

    return args


def main() -> None:
    args = parse_arguments()

    simulation_seconds = args.seconds
    height = args.height
    bx = args.bx
    bz = args.bz

    selected_seed = (
        args.seed
        if args.seed is not None
        else random.SystemRandom().choice(SEEDS)
    )

    output_path = args.output
    batch_size = args.batch_size
    async_write = not args.sync_write
    write_workers = args.write_workers
    write_queue_size = args.write_queue_size

    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)

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
    print(f"Batch size: {batch_size:,}")
    print(f"Async write: {async_write}")
    print(f"Write workers: {write_workers}")
    print(f"Write queue size: {write_queue_size}")
    print(f"PyTorch threads: {torch.get_num_threads()}")

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

    saved_particles = generate_shw(
        flow=flow,
        context=context,
        number_of_particles=number_of_particles,
        batch_size=batch_size,
        async_write=async_write,
        write_workers=write_workers,
        write_queue_size=write_queue_size,
        height=height,
        bx=bx,
        bz=bz,
        selected_seed=selected_seed,
        output_path=output_path,
    )

    print("\nGeneration completed.")
    print("Estimated flux: " f"{flux:.6g} particles/(m2*s)")
    print(f"Saved particles: {saved_particles:,}")
    print(f"Generated file: {output_path.resolve()}")


if __name__ == "__main__":
    main()
