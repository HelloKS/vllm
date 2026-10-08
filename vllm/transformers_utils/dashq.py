# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DASH-Q checkpoint validation and TP slicing, independent of CUDA."""

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from safetensors import safe_open


def checkpoint_fingerprint(folder):
    """Bind reports to checkpoint metadata and tensor-to-shard mapping."""
    return {
        name: hashlib.sha256((Path(folder) / name).read_bytes()).hexdigest()
        for name in ("config.json", "dashq_config.json", "model.safetensors.index.json")
    }


@dataclass(frozen=True)
class DashQSlice:
    """Source [N, K] rectangle and destination output-column offset."""

    n_start: int
    n_stop: int
    k_start: int
    k_stop: int
    out_start: int = 0


def validate_metadata(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if config.get("format") != "dashq-packed-linear":
        raise ValueError("DASH-Q requires dashq-packed-linear format")
    if config.get("format_version") != 1:
        raise ValueError("DASH-Q requires format_version=1")
    modules = config.get("quantized_modules")
    if not isinstance(modules, dict) or not modules:
        raise ValueError("DASH-Q requires nonempty quantized_modules")
    for name, meta in modules.items():
        if meta.get("nbits") != 2 or meta.get("group_size") != 32:
            raise ValueError(f"{name}: only INT2/g32 is supported")
        if meta.get("packing") != "int2_packed_u32":
            raise ValueError(f"{name}: unsupported packing")
        if meta.get("scale_zero_dtype") != "float16":
            raise ValueError(f"{name}: scale/zero must be float16")
        if meta.get("linear_dtype") != "bfloat16":
            raise ValueError(f"{name}: only BF16 activations are supported")
        n, k = meta.get("out_features", 0), meta.get("in_features", 0)
        if not isinstance(n, int) or not isinstance(k, int) or min(n, k) <= 0:
            raise ValueError(f"{name}: invalid matrix dimensions")
        if k % 32 or meta.get("quant_in_features", k) != k:
            raise ValueError(f"{name}: padded/unaligned input is unsupported")
        if meta.get("num_groups") != n * (k // 32):
            raise ValueError(f"{name}: inconsistent group count")
    return modules


def tp_slices(
    n: int,
    k: int,
    rank: int,
    size: int,
    kind: str,
    output_sizes: list[int] | None = None,
) -> list[DashQSlice]:
    """Split logical projections before concatenation (including Mamba z/x/B/C/dt)."""
    if size not in (1, 2) or not 0 <= rank < size or k % 32:
        raise ValueError("DASH-Q requires TP=1/2 and group-aligned K")
    if kind == "replicated":
        return [DashQSlice(0, n, 0, k)]
    if kind == "row":
        if k % (size * 32):
            raise ValueError("Row TP split crosses a quantization group")
        return [DashQSlice(0, n, rank * (k // size), (rank + 1) * (k // size))]
    if kind != "column":
        raise ValueError(f"Unknown DASH-Q partition kind: {kind}")
    sizes = output_sizes or [n]
    if sum(sizes) != n or any(s <= 0 or s % size for s in sizes):
        raise ValueError("Output partitions must sum to N and divide TP")
    source = dest = 0
    result = []
    for width in sizes:
        local = width // size
        result.append(
            DashQSlice(source + rank * local, source + (rank + 1) * local, 0, k, dest)
        )
        source += width
        dest += local
    return result


def packed_rows_to_kmajor(rows):
    """Transpose whole packed words; INT2 packing already runs along K.

    rows is [N, K/16] int32. No unpacked integer matrix is needed.
    """
    return rows.transpose(0, 1).contiguous()


CHUNK_BYTES = 64 * 1024 * 1024


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate DASH-Q JSON key: {key}")
        result[key] = value
    return result


class DashQTensorReader:
    """Keep at most two lazy safetensors mappings open; never prefetch shards."""

    def __init__(self, folder):
        self.folder = Path(folder)
        with (self.folder / "model.safetensors.index.json").open() as f:
            self.index = json.load(f, object_pairs_hook=_unique_object)["weight_map"]
        self.handles = OrderedDict()
        for filename in set(self.index.values()):
            if Path(filename).name != filename:
                raise ValueError(f"Invalid checkpoint shard name: {filename}")

    def close(self):
        while self.handles:
            _, (context, _) = self.handles.popitem(last=False)
            context.__exit__(None, None, None)

    def _file(self, key):
        filename = self.index[key]
        if filename not in self.handles:
            if len(self.handles) == 2:
                _, (context, _) = self.handles.popitem(last=False)
                context.__exit__(None, None, None)
            context = safe_open(
                str(self.folder / filename), framework="pt", device="cpu"
            )
            self.handles[filename] = (context, context.__enter__())
        self.handles.move_to_end(filename)
        return self.handles[filename][1]

    def tensor(self, key):
        return self._file(key).get_tensor(key)

    def matrix(self, key, n, width, packed):
        source = self._file(key).get_slice(key)
        shape = source.get_shape()
        expected = [n * width] if packed else [n * width, 1]
        dtype = "I32" if packed else "F16"
        if shape != expected or source.get_dtype() != dtype:
            raise ValueError(
                f"{key}: expected {expected}/{dtype}, got {shape}/{source.get_dtype()}"
            )
        return source


def load_matrix(reader, source, meta, dest, suffix, slices, out_offset=0):
    """Load one packed matrix or metadata tensor without expanding INT2 codes."""
    packed = suffix == "W_q_packed"
    divisor = 16 if packed else 32
    n, k = meta["out_features"], meta["in_features"]
    width = k // divisor
    element_bytes = 4 if packed else 2
    rows_per_chunk = max(1, CHUNK_BYTES // (width * element_bytes))
    tensor = reader.matrix(f"{source}.{suffix}", n, width, packed)
    for part in slices:
        for start in range(part.n_start, part.n_stop, rows_per_chunk):
            stop = min(start + rows_per_chunk, part.n_stop)
            rows = tensor[start * width : stop * width].reshape(stop - start, width)
            rows = rows[:, part.k_start // divisor : part.k_stop // divisor]
            local = out_offset + part.out_start + start - part.n_start
            chunk = packed_rows_to_kmajor(rows)
            dest[:, local : local + stop - start].copy_(chunk)
