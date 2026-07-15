#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chequeo rapido de integridad para una corrida Machin."""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


ALERT_RE = re.compile(r"(ERROR|Traceback|WARN|No se generaron|No outputs|Missing|Fallo|Fall[oó])", re.I)
POINTS = ("P1", "P2", "P4", "P5")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def count_column_sum(path: Path, column: str = "count") -> int | None:
    total = 0
    seen = False
    for row in read_csv_rows(path):
        if column not in row:
            return None
        try:
            total += int(float(row[column]))
            seen = True
        except (TypeError, ValueError):
            return None
    return total if seen else None


def scan_logs(log_dir: Path) -> list[str]:
    alerts: list[str] = []
    if not log_dir.exists():
        return [f"No existe carpeta de logs: {log_dir}"]
    for log in sorted(log_dir.glob("*.log")):
        with log.open(encoding="utf-8", errors="replace") as f:
            for lineno, line in enumerate(f, start=1):
                if ALERT_RE.search(line):
                    alerts.append(f"{log}:{lineno}: {line.rstrip()}")
    return alerts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Valida una carpeta de salida de orquestador_machin.")
    ap.add_argument("outdir", type=Path, help="Carpeta de corrida, por ejemplo run_bariloche_smoke")
    ap.add_argument("--show-alerts", action="store_true", help="Imprime todas las lineas sospechosas de logs.")
    args = ap.parse_args(argv)

    outdir = args.outdir
    index = outdir / "pipeline_outputs.csv"
    rows = read_csv_rows(index)
    missing = [row for row in rows if not Path(row.get("path", "")).exists()]
    alerts = scan_logs(outdir / "logs")

    print(f"Corrida: {outdir}")
    print(f"Salidas indexadas: {len(rows)}")
    print(f"Salidas faltantes: {len(missing)}")
    print(f"Alertas en logs: {len(alerts)}")

    for source in ("raw", "filtered"):
        plot_dir = outdir / "05_plots" / source
        if not plot_dir.exists():
            continue
        counts = []
        for point in POINTS:
            total = count_column_sum(plot_dir / f"theta_phi_counts_{point}.csv")
            if total is not None:
                counts.append(f"{point}={total}")
        if counts:
            print(f"theta-phi {source}: " + ", ".join(counts))

    for source in ("raw", "filtered"):
        summary = outdir / "09_event_mc_empirical" / source / "event_mc_smearing_summary.csv"
        rows_summary = read_csv_rows(summary)
        if not rows_summary:
            continue
        parts = []
        for row in rows_summary:
            parts.append(
                f"{row.get('point')}={row.get('input_total')}->{row.get('smeared_total')}"
                f" rel={row.get('relative_total_change')}"
            )
        print(f"event-MC {source}: " + ", ".join(parts))

    if missing:
        print("\nPrimeras salidas faltantes:")
        for row in missing[:20]:
            print(f"- {row.get('stage')} {row.get('point')} {row.get('kind')}: {row.get('path')}")

    if alerts and args.show_alerts:
        print("\nAlertas:")
        for alert in alerts:
            print(alert)

    return 1 if missing or alerts else 0


if __name__ == "__main__":
    raise SystemExit(main())
