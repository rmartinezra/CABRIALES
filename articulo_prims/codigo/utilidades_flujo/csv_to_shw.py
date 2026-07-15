#!/usr/bin/env python3
"""
Convert a CSV of ground-level muons to an ARTI-like .shw file.

Expected CSV columns, at minimum:
  energy_GeV, theta_deg, phi_deg, px_GeV_c, py_GeV_c, pz_GeV_c, h_m

Output .shw columns:
  CorsikaId px py pz x y z shower_id prm_id prm_energy prm_theta prm_phi

Notes:
- Momentum is copied from the CSV in GeV/c.
- x,y are not present in the CSV, so they are generated or set to zero.
- z is taken from h_m unless --z is given.
- If charge is unknown, use --muon-id random for a simple mu+/mu- mixture.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from pathlib import Path
from typing import Iterable, Optional


def signed_phi(phi_deg: float) -> float:
    """Map phi from any degree convention to [-180, 180)."""
    return ((phi_deg + 180.0) % 360.0) - 180.0


def fmt_e(x: float) -> str:
    return f"{x:+.5e}"


def detect_observation_level(csv_path: Path, h_col: str, max_rows: int = 10000) -> Optional[float]:
    vals = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or h_col not in reader.fieldnames:
            return None
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            try:
                vals.append(float(row[h_col]))
            except (TypeError, ValueError):
                pass
    if not vals:
        return None
    return statistics.median(vals)


def choose_muon_id(mode: str, rng: random.Random, muplus_frac: float) -> str:
    if mode == "random":
        return "0005" if rng.random() < muplus_frac else "0006"
    # mode should already be a four digit string
    return f"{int(mode):04d}"


def main() -> None:
    p = argparse.ArgumentParser(
        description="Convert a muon CSV file to an ARTI-like .shw file."
    )
    p.add_argument("input_csv", help="Input CSV file")
    p.add_argument("output_shw", help="Output .shw file")

    p.add_argument("--area-m2", type=float, default=1.0,
                   help="Square generation area for x,y in m^2. Default: 1.0")
    p.add_argument("--xy-mode", choices=["random", "zero"], default="random",
                   help="How to assign x,y when the CSV has no lateral position. Default: random")
    p.add_argument("--x0", type=float, default=0.0, help="Center x position in m. Default: 0")
    p.add_argument("--y0", type=float, default=0.0, help="Center y position in m. Default: 0")
    p.add_argument("--z", type=float, default=None,
                   help="Fixed z position in m. If omitted, uses h_m per row.")
    p.add_argument("--observation-level", type=float, default=None,
                   help="Value written in the .shw header. If omitted, median h_m is used.")

    p.add_argument("--muon-id", default="random",
                   help="CORSIKA muon id: 0005=mu+, 0006=mu-, or random. Default: random")
    p.add_argument("--muplus-frac", type=float, default=0.56,
                   help="Fraction of mu+ when --muon-id random. Default: 0.56")
    p.add_argument("--seed", type=int, default=12345,
                   help="Random seed for x,y and random charge. Default: 12345")

    p.add_argument("--primary-id", default="0014",
                   help="Primary CORSIKA id written in prm_id. Default: 0014 = proton placeholder")
    p.add_argument("--start-shower-id", type=int, default=1,
                   help="First shower_id. Default: 1")
    p.add_argument("--phi-convention", choices=["signed", "csv"], default="signed",
                   help="Write prm_phi as [-180,180) or as in CSV. Default: signed")

    args = p.parse_args()

    input_csv = Path(args.input_csv)
    output_shw = Path(args.output_shw)

    if args.area_m2 <= 0:
        raise SystemExit("ERROR: --area-m2 must be positive.")
    if not (0.0 <= args.muplus_frac <= 1.0):
        raise SystemExit("ERROR: --muplus-frac must be between 0 and 1.")
    if args.muon_id != "random":
        try:
            mid = int(args.muon_id)
        except ValueError:
            raise SystemExit("ERROR: --muon-id must be 0005, 0006, another integer id, or random.")
        if mid not in (5, 6):
            print(f"WARNING: --muon-id {mid:04d} is not a standard muon id. 0005=mu+, 0006=mu-.")

    obs = args.observation_level
    if obs is None:
        obs = detect_observation_level(input_csv, "h_m")
    if obs is None:
        obs = args.z if args.z is not None else 0.0

    side = math.sqrt(args.area_m2)
    half = side / 2.0
    rng = random.Random(args.seed)

    required = ["energy_GeV", "theta_deg", "phi_deg", "px_GeV_c", "py_GeV_c", "pz_GeV_c"]

    n_written = 0
    with input_csv.open("r", newline="") as fin, output_shw.open("w", newline="") as fout:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            raise SystemExit("ERROR: input CSV has no header.")
        missing = [c for c in required if c not in reader.fieldnames]
        if args.z is None and "h_m" not in reader.fieldnames:
            missing.append("h_m or use --z")
        if missing:
            raise SystemExit("ERROR: missing required columns: " + ", ".join(missing))

        fout.write("# # # shw\n")
        fout.write(f"# # CURVED mode is ENABLED and observation level is {obs:g} m a.s.l.\n")
        fout.write("# # This is the Secondaries file - ARTI     v1r9\n")
        fout.write("# # 12 column format is:\n")
        fout.write("# # CorsikaId px py pz x y z shower_id prm_id prm_energy prm_theta prm_phi\n")

        shower_id = args.start_shower_id
        primary_id = f"{int(args.primary_id):04d}"

        for row in reader:
            try:
                energy = float(row["energy_GeV"])
                theta = float(row["theta_deg"])
                phi_csv = float(row["phi_deg"])
                px = float(row["px_GeV_c"])
                py = float(row["py_GeV_c"])
                pz = float(row["pz_GeV_c"])
                z = float(row["h_m"]) if args.z is None else args.z
            except (TypeError, ValueError) as exc:
                raise SystemExit(f"ERROR: bad numeric value near input row {n_written + 2}: {exc}")

            if args.xy_mode == "zero":
                x = args.x0
                y = args.y0
            else:
                x = args.x0 + rng.uniform(-half, half)
                y = args.y0 + rng.uniform(-half, half)

            phi_out = signed_phi(phi_csv) if args.phi_convention == "signed" else phi_csv
            corsika_id = choose_muon_id(args.muon_id, rng, args.muplus_frac)

            fout.write(
                f"{corsika_id} "
                f"{fmt_e(px)} {fmt_e(py)} {fmt_e(pz)} "
                f"{fmt_e(x)} {fmt_e(y)} {fmt_e(z)} "
                f"{shower_id:08d} "
                f"{primary_id} "
                f"{fmt_e(energy)} "
                f"{theta:+07.3f} "
                f"{phi_out:+08.3f}\n"
            )
            shower_id += 1
            n_written += 1

    print(f"Wrote {n_written} particles to {output_shw}")
    print(f"Header observation level: {obs:g} m a.s.l.")
    if args.xy_mode == "random":
        print(f"x,y generated uniformly over a {args.area_m2:g} m^2 square centered at ({args.x0:g}, {args.y0:g})")


if __name__ == "__main__":
    main()
