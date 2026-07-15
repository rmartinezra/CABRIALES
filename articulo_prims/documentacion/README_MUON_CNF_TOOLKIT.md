# Muon CNF Toolkit

Muon sample generator based on **Conditional Normalizing Flows (CNF)**. The
model generates muon energy and zenith angle conditioned on:

* Site altitude above sea level.
* Northward geomagnetic component `Bx`.
* Vertical geomagnetic component `Bz`.

The program automatically estimates how many particles to generate for the
requested simulation time, computes the momentum components `px`, `py`, and
`pz`, and exports the result in `.shw` format.

## Features

* Muon energy and zenith-angle generation.
* Conditioning by altitude, `Bx`, and `Bz`.
* Automatic expected particle-count calculation.
* Approximate charge ratio: 55% `mu+` and 45% `mu-`.
* Automatic CUDA GPU execution when available.
* Configurable batch size for memory and throughput control.
* Parallel `.shw` writing with 6 processes by default.
* Reproducible generation through predefined seeds.

## Project Structure

```text
muon-cnf-toolkit/
├── model.pt
├── main-cnf.py
├── main-cnf-cabriales-cache.py
├── requirements.txt
└── README.md
```

| File | Description |
| --- | --- |
| `model.pt` | Trained CNF model weights. |
| `main-cnf.py` | Inference, post-processing, and `.shw` export script. |
| `main-cnf-cabriales-cache.py` | Direct CABRIALES v1 kinematic-cache generator. |
| `requirements.txt` | Python dependencies. |
| `README.md` | Installation, usage, and examples. |

## Requirements

Recommended Python version:

```text
Python 3.11 or later
```

Main dependencies:

```text
numpy>=2.0.2
torch>=2.10.0
nflows>=0.14
```

To use GPU acceleration, install a CUDA-enabled PyTorch build and use a
compatible NVIDIA GPU. If CUDA is not available, the program can run on CPU.

## Installation

```bash
git clone https://github.com/rmartinezra/muon-cnf-toolkit.git
cd muon-cnf-toolkit
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows with WSL, run the commands inside Ubuntu/WSL.

## Quick Start

General command:

```bash
python main-cnf.py \
  --seconds TIME \
  --height ALTITUDE_M \
  --bx BX_MICROTESLA \
  --bz BZ_MICROTESLA \
  --output FILE.shw
```

Example for Bogota, simulating one hour:

```bash
python main-cnf.py \
  --seconds 3600 \
  --height 2640 \
  --bx 26.473 \
  --bz 13.360 \
  --device cuda \
  --batch-size 65536 \
  --output bogota_1h.shw
```

If you do not want to force GPU execution, use `--device auto` or omit the
argument.

## CABRIALES v1 Kinematic Cache

For long simulations, such as 30 days of muon flux, the text `.shw` format can
become very large. The repository includes a second generator that writes the
lighter **CABRIALES v1 kinematic cache** directly, without first creating a
massive `.shw` file.

Use:

```bash
python main-cnf-cabriales-cache.py \
  --days 30 \
  --height 2640 \
  --bx 26.473 \
  --bz 13.360 \
  --device cuda \
  --batch-size 65536 \
  --chunk-events 1000000 \
  --write-workers 6 \
  --output-dir machin30dia_kinematic_cache
```

If the output directory already exists and you want to replace it, add:

```bash
--overwrite
```

The generated cache is a directory:

```text
machin30dia_kinematic_cache/
├── manifest.json
└── chunks/
    ├── chunk_000000.npz
    ├── chunk_000001.npz
    └── ...
```

Each chunk contains exactly these arrays:

| Array | dtype | Unit | Description |
| --- | --- | --- | --- |
| `theta_deg` | `float32` | degrees | Muon zenith angle. |
| `phi_abs_deg` | `float32` | degrees | Absolute azimuth in the range 0 to 360. |
| `total_E_GeV` | `float32` | GeV | Total muon energy, not kinetic energy. |
| `pz_positive` | `uint8` | - | `1` if `pz > 0`, otherwise `0`. |
| `pid_code` | `uint8` | - | `5` for `mu+`, `6` for `mu-`. |

All arrays inside a chunk have the same length. The recommended chunk size is
1,000,000 events, which is the default for `--chunk-events`.

The manifest follows the CABRIALES v1 layout:

```json
{
  "version": 1,
  "created_at": "2026-07-08 16:07:03",
  "source_shw": null,
  "shw_format": "cnf9",
  "shw_member": null,
  "only_muons": true,
  "chunk_events_requested": 1000000,
  "compressed": true,
  "arrays": {
    "theta_deg": "float32",
    "phi_abs_deg": "float32",
    "total_E_GeV": "float32",
    "pz_positive": "uint8",
    "pid_code": "uint8"
  },
  "n_events": 0,
  "n_chunks": 0,
  "chunks": []
}
```

Additional counters such as `n_particles_read`, `n_muons_read`,
`n_bad_momentum`, `cache_bytes`, and `elapsed_s` are filled at the end of the
run.

## Parameters

| Argument | Required | Unit | Description |
| --- | ---: | --- | --- |
| `--seconds` | Yes | s | Physical simulation time. |
| `--height` | Yes | m | Site altitude above sea level. |
| `--bx` | Yes | µT | Northward geomagnetic component. |
| `--bz` | Yes | µT | Vertical geomagnetic component, positive downward. |
| `--output` | No | - | Output `.shw` file path. |
| `--seed` | No | - | Seed: `5`, `503`, `1001`, `1501`, or `1999`. |
| `--device` | No | - | `auto`, `cuda`, or `cpu`. Default: `auto`. |
| `--batch-size` | No | particles | Particles sampled per model call. Default: `65536`. |
| `--torch-threads` | No | threads | Number of CPU threads used by PyTorch. |
| `--sync-write` | No | - | Disable parallel writing. Useful for benchmarking or debugging. |
| `--write-workers` | No | processes | Parallel processes used to write temporary `.shw` parts. Default: `6`. |
| `--write-queue-size` | No | batches | Maximum number of generated batches waiting for writing. Default: `12`. |

## City Examples

The `Bx` and `Bz` values in this table were estimated with the
**NOAA/NCEI IGRF2025** model for **July 8, 2026**. The convention is:

* `Bx = X`: northward component, positive toward geographic north.
* `Bz = Z`: vertical component, positive downward.
* Unit conversion: `1 µT = 1000 nT`.

Altitudes are approximate values for typical city-center locations. For
precision work, recompute the geomagnetic field for the exact coordinates,
altitude, and date of the simulation.

| City | Country | Altitude m | `Bx` µT | `Bz` µT | Example output |
| --- | --- | ---: | ---: | ---: | --- |
| Bogota | Colombia | 2640 | 26.473 | 13.360 | `bogota_1h.shw` |
| Medellin | Colombia | 1495 | 26.641 | 15.114 | `medellin_1h.shw` |
| Cali | Colombia | 1018 | 26.562 | 12.686 | `cali_1h.shw` |
| Caracas | Venezuela | 900 | 26.573 | 16.656 | `caracas_1h.shw` |
| Panama City | Panama | 2 | 26.912 | 18.376 | `panama_1h.shw` |
| Quito | Ecuador | 2850 | 26.389 | 9.603 | `quito_1h.shw` |
| Lima | Peru | 154 | 24.359 | -0.650 | `lima_1h.shw` |
| La Paz | Bolivia | 3640 | 21.923 | -5.131 | `lapaz_1h.shw` |
| Mexico City | Mexico | 2240 | 26.874 | 28.843 | `mexico_1h.shw` |
| Santiago | Chile | 570 | 19.014 | -13.481 | `santiago_1h.shw` |

Ready-to-run one-hour examples:

```bash
python main-cnf.py --seconds 3600 --height 2640 --bx 26.473 --bz 13.360 --device cuda --output bogota_1h.shw
python main-cnf.py --seconds 3600 --height 2850 --bx 26.389 --bz 9.603 --device cuda --output quito_1h.shw
python main-cnf.py --seconds 3600 --height 154 --bx 24.359 --bz -0.650 --device cuda --output lima_1h.shw
python main-cnf.py --seconds 3600 --height 2240 --bx 26.874 --bz 28.843 --device cuda --output mexico_1h.shw
python main-cnf.py --seconds 3600 --height 570 --bx 19.014 --bz -13.481 --device cuda --output santiago_1h.shw
```

Useful sources for recomputing geomagnetic fields:

* NOAA/NCEI Magnetic Field Calculator:
  https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml
* NOAA/NCEI IGRF:
  https://www.ncei.noaa.gov/products/international-geomagnetic-reference-field

## Performance

The program uses CUDA automatically when available:

```bash
python main-cnf.py --seconds 3600 --height 2640 --bx 26.473 --bz 13.360 --device auto
```

To force GPU execution:

```bash
python main-cnf.py --seconds 3600 --height 2640 --bx 26.473 --bz 13.360 --device cuda
```

To force CPU execution:

```bash
python main-cnf.py --seconds 3600 --height 2640 --bx 26.473 --bz 13.360 --device cpu
```

### Batch Size

`--batch-size` controls how many particles are generated per model call. Larger
values usually improve GPU throughput but require more memory.

Recommended starting value:

```bash
--batch-size 65536
```

If CUDA runs out of memory, reduce the batch size:

```bash
--batch-size 32768
```

### Parallel Writing

By default, `.shw` writing is parallelized as follows:

1. Each valid batch is sent to a writer process.
2. Each process writes a temporary part file.
3. At the end, all parts are concatenated in order.
4. The final file keeps the standard `.shw` format.

Relevant parameters:

```bash
--write-workers 6
--write-queue-size 12
```

To compare against traditional synchronous writing:

```bash
python main-cnf.py ... --sync-write
```

Note: parallel writing uses temporary files. During execution, it may require
extra disk space close to the final output file size.

## Particle Count Calculation

The expected muon flux is estimated from altitude:

$$
F(h) = 100.4\,e^{0.0002119h}
$$

where:

* `h` is in meters.
* `F(h)` is in particles/(m² s).

The number of particles is computed as:

$$
N = F(h)\times t\times A
$$

with `A = 1 m²` and `t = --seconds`.

Time examples:

| Physical time | `--seconds` |
| --- | ---: |
| 1 minute | 60 |
| 1 hour | 3600 |
| 1 day | 86400 |
| 10 days | 864000 |

## Output Format

The `.shw` file starts with:

```text
# # # shw
# # CNFId energy theta px py pz h bx bz
```

Columns:

| Column | Unit | Description |
| --- | --- | --- |
| `CNFId` | - | Particle identifier: `5 = mu+`, `6 = mu-`. |
| `energy` | GeV | Total muon energy. |
| `theta` | degrees | Zenith angle. |
| `px` | GeV/c | Momentum x-component. |
| `py` | GeV/c | Momentum y-component. |
| `pz` | GeV/c | Momentum z-component. |
| `h` | m | Site altitude. |
| `bx` | µT | Northward geomagnetic component used as condition. |
| `bz` | µT | Vertical geomagnetic component used as condition. |

## Recommended Model Range

The model was developed within these ranges:

| Variable | Minimum | Maximum |
| --- | ---: | ---: |
| Altitude | 0 m | 5230 m |
| `Bx` | 9.63 µT | 36.118 µT |
| `Bz` | -14.326 µT | 55.928 µT |

Values outside these ranges are extrapolations. The program can still run, but
the physical reliability may decrease.

## Reproducibility

Available seeds:

```text
5, 503, 1001, 1501, 1999
```

Example:

```bash
python main-cnf.py --seconds 3600 --height 2640 --bx 26.473 --bz 13.360 --seed 5 --output bogota_seed5.shw
```

If `--seed` is not provided, the program randomly selects one of the available
seeds.

## Practical Tips

* Use `--device cuda` when you explicitly want GPU execution.
* If CUDA runs out of memory, lower `--batch-size`.
* If writing is slow, test `--write-workers 2`, `4`, `6`, or `8`.
* If you need to debug output differences, use `--sync-write`.
* Avoid committing large `.shw` output files. They are ignored by `.gitignore`.

## Authors

* Jhon Almanzar-Quintero
* Cristian Orduz-Carvajal
* Rafael Martínez-Rivero
* Christian Sarmiento-Cano
* Luis Núñez-Villavicencio

## License

To be defined.
