#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Streaming helpers for ARTI/CNF muon text files, compressed files, and tar archives."""
from __future__ import annotations

import bz2
import gzip
import io
import lzma
import math
import tarfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator


MUON_MASS_GEV = 0.10565837
MUON_IDS = {b"0005", b"0006", b"5", b"6"}
GZIP_FAST_COMPRESSLEVEL = 1
COMPRESSED_SUFFIXES = (".gz", ".xz", ".bz2")
TAR_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".tar.xz", ".txz", ".tar.bz2", ".tbz2")
SHW_SUFFIXES = (".shw", ".shw.gz", ".shw.xz", ".shw.bz2")


@dataclass(slots=True)
class MuonRecord:
    pid: bytes
    px: float
    py: float
    pz: float
    e_total_GeV: float


def is_muon_id(pid: bytes) -> bool:
    return pid in MUON_IDS


def _lower_name(path: str | Path) -> str:
    return str(path).lower()


def is_tar_path(path: str | Path) -> bool:
    name = _lower_name(path)
    return name.endswith(TAR_SUFFIXES)


def is_compressed_path(path: str | Path) -> bool:
    name = _lower_name(path)
    return name.endswith(COMPRESSED_SUFFIXES)


def _member_is_shw(member_name: str) -> bool:
    name = member_name.lower()
    return name.endswith(SHW_SUFFIXES) or name.endswith(".txt")


def _strip_known_suffixes(name: str) -> str:
    low = name.lower()
    for suffix in (".tar.gz", ".tar.xz", ".tar.bz2", ".tgz", ".txz", ".tbz2", ".gz", ".xz", ".bz2", ".shw"):
        if low.endswith(suffix):
            name = name[: -len(suffix)]
            low = name.lower()
    return name


def shw_stem(path: str | Path) -> str:
    """Return a stable stem for plain, compressed, or tar-contained SHW inputs."""
    path = Path(path)
    if is_tar_path(path) and path.exists():
        try:
            with tarfile.open(path, "r:*") as tf:
                member = find_shw_member(tf)
                if member is not None:
                    return _strip_known_suffixes(Path(member.name).name)
        except tarfile.TarError:
            pass
    return _strip_known_suffixes(path.name)


def find_shw_member(tf: tarfile.TarFile, member_name: str | None = None) -> tarfile.TarInfo | None:
    if member_name:
        try:
            member = tf.getmember(member_name)
            return member if member.isfile() else None
        except KeyError:
            return None
    first_file = None
    for member in tf:
        if not member.isfile():
            continue
        if first_file is None:
            first_file = member
        if _member_is_shw(member.name):
            return member
    return first_file


def _wrap_member_stream(member_name: str, raw: BinaryIO) -> BinaryIO:
    name = member_name.lower()
    if name.endswith(".gz"):
        return gzip.GzipFile(fileobj=raw, mode="rb")
    if name.endswith(".xz"):
        return lzma.LZMAFile(raw, mode="rb")
    if name.endswith(".bz2"):
        return bz2.BZ2File(raw, mode="rb")
    return raw


@contextmanager
def open_shw_bytes(path: str | Path, member_name: str | None = None) -> Iterator[BinaryIO]:
    """Open plain/compressed/tar-contained SHW data as a binary line stream."""
    path = Path(path)
    name = path.name.lower()
    if is_tar_path(path):
        with tarfile.open(path, "r:*") as tf:
            member = find_shw_member(tf, member_name=member_name)
            if member is None:
                raise FileNotFoundError(f"No encontre un archivo .shw dentro de {path}")
            raw = tf.extractfile(member)
            if raw is None:
                raise FileNotFoundError(f"No pude abrir {member.name} dentro de {path}")
            with raw:
                stream = _wrap_member_stream(member.name, raw)
                try:
                    yield stream
                finally:
                    if stream is not raw:
                        stream.close()
    elif name.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            yield f
    elif name.endswith(".xz"):
        with lzma.open(path, "rb") as f:
            yield f
    elif name.endswith(".bz2"):
        with bz2.open(path, "rb") as f:
            yield f
    else:
        with path.open("rb") as f:
            yield f


@contextmanager
def open_shw_text(path: str | Path, member_name: str | None = None, encoding: str = "utf-8") -> Iterator[io.TextIOBase]:
    with open_shw_bytes(path, member_name=member_name) as f:
        wrapper = io.TextIOWrapper(f, encoding=encoding, errors="ignore", newline="")
        try:
            yield wrapper
        finally:
            wrapper.detach()


@contextmanager
def open_output_bytes(path: str | Path) -> Iterator[BinaryIO]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = path.name.lower()
    if name.endswith(".gz"):
        with gzip.open(path, "wb", compresslevel=GZIP_FAST_COMPRESSLEVEL) as f:
            yield f
    elif name.endswith(".xz"):
        with lzma.open(path, "wb") as f:
            yield f
    elif name.endswith(".bz2"):
        with bz2.open(path, "wb") as f:
            yield f
    else:
        with path.open("wb") as f:
            yield f


def stream_size_hint(path: str | Path) -> int:
    """Compressed/archive byte size for progress bars; exact for plain files."""
    path = Path(path)
    if is_tar_path(path) and path.exists():
        try:
            with tarfile.open(path, "r:*") as tf:
                member = find_shw_member(tf)
                if member is not None and member.size > 0:
                    return int(member.size)
        except tarfile.TarError:
            pass
    try:
        return path.stat().st_size
    except OSError:
        return 0


def iter_shw_lines(path: str | Path, member_name: str | None = None) -> Iterator[bytes]:
    with open_shw_bytes(path, member_name=member_name) as f:
        for line in f:
            yield line


def parse_muon_parts(parts: list[bytes], shw_format: str = "auto", only_muons: bool = True) -> MuonRecord | None:
    """Parse ARTI 12-col or CNF 9-col lines into common kinematics.

    ARTI format used here:
        pid px py pz ...

    CNF format:
        CNFId energy theta px py pz h bx bz
    """
    if not parts:
        return None

    fmt = shw_format.lower()
    pid = parts[0]
    if only_muons and not is_muon_id(pid):
        return None

    try:
        if fmt == "cnf9" or (fmt == "auto" and len(parts) >= 9 and pid in {b"5", b"6"}):
            e_total = float(parts[1])
            px = float(parts[3])
            py = float(parts[4])
            pz = float(parts[5])
            return MuonRecord(pid=pid, px=px, py=py, pz=pz, e_total_GeV=e_total)

        if fmt == "arti12" or fmt == "auto":
            if len(parts) < 4:
                return None
            px = float(parts[1])
            py = float(parts[2])
            pz = float(parts[3])
            p2 = px * px + py * py + pz * pz
            if p2 <= 0.0:
                return None
            e_total = math.sqrt(p2 + MUON_MASS_GEV * MUON_MASS_GEV)
            return MuonRecord(pid=pid, px=px, py=py, pz=pz, e_total_GeV=e_total)
    except (ValueError, IndexError):
        return None

    return None


def theta_phi_from_momentum(px: float, py: float, pz: float) -> tuple[float, float] | None:
    p2 = px * px + py * py + pz * pz
    if p2 <= 0.0:
        return None
    p = math.sqrt(p2)
    theta = math.degrees(math.acos(max(-1.0, min(1.0, pz / p))))
    phi = math.degrees(math.atan2(py, px))
    if phi < 0.0:
        phi += 360.0
    return theta, phi


def output_template_for_compression(template: str, compression: str) -> str:
    compression = compression.lower()
    if compression == "none":
        return template
    if compression == "gz" and not template.lower().endswith(".gz"):
        return template + ".gz"
    if compression == "xz" and not template.lower().endswith(".xz"):
        return template + ".xz"
    if compression == "bz2" and not template.lower().endswith(".bz2"):
        return template + ".bz2"
    return template
