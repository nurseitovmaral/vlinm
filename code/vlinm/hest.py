from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import h5py
import numpy as np
from scipy import sparse
from torch.utils.data import Dataset

from .data import PairedRegions


def _strings(dataset: h5py.Dataset) -> np.ndarray:
    return np.asarray(dataset.asstr()[:]).reshape(-1).astype(str)


def _unique_index(values: np.ndarray, label: str) -> dict[str, int]:
    result = {str(value): i for i, value in enumerate(values)}
    if len(result) != len(values) or any(not key for key in result):
        raise ValueError(f"{label} must be unique and nonempty")
    return result


def _expression(group: h5py.File) -> sparse.spmatrix | np.ndarray:
    matrix = group["X"]
    if isinstance(matrix, h5py.Dataset):
        return np.asarray(matrix[:])
    kind = matrix.attrs.get("encoding-type", "")
    if isinstance(kind, bytes):
        kind = kind.decode()
    shape = tuple(int(x) for x in matrix.attrs["shape"])
    parts = (matrix["data"][:], matrix["indices"][:], matrix["indptr"][:])
    if kind == "csr_matrix":
        return sparse.csr_matrix(parts, shape=shape)
    if kind == "csc_matrix":
        return sparse.csc_matrix(parts, shape=shape)
    raise ValueError(f"unsupported expression encoding: {kind}")


@dataclass(frozen=True)
class HestPairs:
    patch_path: Path
    barcodes: np.ndarray
    expression: np.ndarray
    coordinates: np.ndarray
    genes: tuple[str, ...]
    patch_shape: tuple[int, ...]

    def region_features(self, feature_path: str | Path, patient_id: str, slide_id: str) -> PairedRegions:
        if not patient_id.strip() or not slide_id.strip():
            raise ValueError("patient and slide identifiers must be supplied")
        with np.load(feature_path, allow_pickle=False) as features:
            barcodes = np.asarray(features["barcodes"]).astype(str).reshape(-1)
            values = np.asarray(features["features"], dtype=np.float32)
        feature_index = _unique_index(barcodes, "feature barcodes")
        if values.ndim != 2 or len(values) != len(barcodes):
            raise ValueError("features must have one row for each feature barcode")
        missing = [barcode for barcode in self.barcodes if barcode not in feature_index]
        if missing:
            raise ValueError(f"missing feature embeddings for {len(missing)} paired spots")
        order = np.asarray([feature_index[barcode] for barcode in self.barcodes])
        data = PairedRegions(values[order], self.expression, self.coordinates, np.full(len(order), slide_id), np.full(len(order), patient_id))
        data.validate()
        return data


class HestPatchDataset(Dataset[tuple[np.ndarray, np.ndarray]]):
    def __init__(self, pairs: HestPairs) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs.barcodes)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        with h5py.File(self.pairs.patch_path, "r") as source:
            image = np.asarray(source["img"][index])
        return image, self.pairs.expression[index]


def load_hest_pairs(adata_path: str | Path, patch_path: str | Path, genes: list[str], expected_patch_size: int = 256) -> HestPairs:
    if not genes or len(set(genes)) != len(genes) or any(not gene for gene in genes):
        raise ValueError("a nonempty, unique training-selected gene list is required")
    with h5py.File(patch_path, "r") as patch:
        barcodes = _strings(patch["barcode"])
        _unique_index(barcodes, "patch barcodes")
        shape = tuple(patch["img"].shape)
        if len(shape) != 4 or shape[0] != len(barcodes) or shape[1:] != (expected_patch_size, expected_patch_size, 3):
            raise ValueError(f"patch dimensions {shape} do not match the requested size {expected_patch_size}")
    with h5py.File(adata_path, "r") as adata:
        obs = adata["obs"]
        var = adata["var"]
        obs_key = obs.attrs.get("_index", "_index")
        var_key = var.attrs.get("_index", "_index")
        if isinstance(obs_key, bytes):
            obs_key = obs_key.decode()
        if isinstance(var_key, bytes):
            var_key = var_key.decode()
        obs_index = _unique_index(_strings(obs[obs_key]), "expression barcodes")
        gene_index = _unique_index(_strings(var[var_key]), "gene names")
        missing_spots = [barcode for barcode in barcodes if barcode not in obs_index]
        missing_genes = [gene for gene in genes if gene not in gene_index]
        if missing_spots or missing_genes:
            raise ValueError(f"missing {len(missing_spots)} expression barcodes and {len(missing_genes)} selected genes")
        rows = np.array([obs_index[barcode] for barcode in barcodes])
        cols = np.array([gene_index[gene] for gene in genes])
        counts = _expression(adata)
        if counts.shape != (len(obs_index), len(gene_index)):
            raise ValueError("expression shape does not match identifiers")
        selected = counts[rows][:, cols]
        expression = selected.toarray() if sparse.issparse(selected) else np.asarray(selected)
        coordinates = np.asarray(adata["obsm/spatial"][:])[rows].astype(np.float64)
    if coordinates.shape != (len(barcodes), 2) or not np.isfinite(coordinates).all() or not np.isfinite(expression).all():
        raise ValueError("spatial coordinates and expression must be finite")
    return HestPairs(Path(patch_path), barcodes, expression.astype(np.float32), coordinates, tuple(genes), shape[1:])


@dataclass(frozen=True)
class HestSample:
    sample_id: str
    patch_path: Path
    expression_path: Path


def read_hest_split(split_path: str | Path, cohort_root: str | Path) -> list[HestSample]:
    root = Path(cohort_root)
    result = []
    seen: set[str] = set()
    with Path(split_path).open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            sample_id = row["sample_id"].strip()
            if not sample_id or sample_id in seen:
                raise ValueError("sample identifiers must be unique and nonempty")
            seen.add(sample_id)
            paths = [PurePosixPath(row[key]) for key in ("patches_path", "expr_path")]
            if any(path.is_absolute() or ".." in path.parts or not path.parts for path in paths):
                raise ValueError("split paths must be relative to the cohort root")
            result.append(HestSample(sample_id, root.joinpath(*paths[0].parts), root.joinpath(*paths[1].parts)))
    if not result:
        raise ValueError("split contains no samples")
    return result


def assert_hest_patient_disjoint(splits: list[list[HestSample]], patients: dict[str, str]) -> None:
    seen: set[str] = set()
    seen_samples: set[str] = set()
    for split in splits:
        current: set[str] = set()
        for sample in split:
            if sample.sample_id in seen_samples:
                raise ValueError("sample appears in multiple splits")
            seen_samples.add(sample.sample_id)
            patient = patients.get(sample.sample_id, "").strip()
            if not patient:
                raise ValueError("patient identifiers are required for every sample")
            current.add(patient)
        if current & seen:
            raise ValueError("patient appears in multiple splits")
        seen.update(current)
