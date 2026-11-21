from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .spatial import knn_indices


@dataclass(frozen=True)
class PairedRegions:
    features: np.ndarray
    omics: np.ndarray
    coordinates: np.ndarray
    slide: np.ndarray
    patient: np.ndarray

    def validate(self) -> None:
        n = len(self.features)
        if n == 0 or self.features.ndim != 2 or self.omics.ndim != 2:
            raise ValueError("features and omics must be nonempty matrices")
        if self.coordinates.shape != (n, 2) or any(len(x) != n for x in (self.omics, self.slide, self.patient)):
            raise ValueError("regions must have matching first dimensions")
        if self.slide.ndim != 1 or self.patient.ndim != 1:
            raise ValueError("slide and patient identifiers must be vectors")
        if any(not np.isfinite(x).all() for x in (self.features, self.omics, self.coordinates)):
            raise ValueError("numeric arrays must be finite")
        for slide_id in np.unique(self.slide):
            if len(np.unique(self.patient[self.slide == slide_id])) != 1:
                raise ValueError("each slide must belong to one patient")


def load_regions(path: str | Path) -> PairedRegions:
    with np.load(path, allow_pickle=False) as archive:
        data = PairedRegions(*(archive[name].copy() for name in ("features", "omics", "coordinates", "slide", "patient")))
    data.validate()
    return data


def assert_patient_disjoint(*partitions: PairedRegions) -> None:
    seen: set[str] = set()
    for part in partitions:
        current = set(map(str, part.patient))
        if seen & current:
            raise ValueError("patient identifiers overlap between partitions")
        seen.update(current)


class RegionDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, regions: PairedRegions, neighbours: int = 6) -> None:
        regions.validate()
        self.regions = regions
        self.groups: list[np.ndarray] = []
        self.group_for_index = np.empty(len(regions.features), dtype=np.int64)
        self.local_for_index = np.empty(len(regions.features), dtype=np.int64)
        self.neighbours: list[np.ndarray] = []
        for slide_id in np.unique(regions.slide):
            idx = np.flatnonzero(regions.slide == slide_id)
            if len(idx) <= neighbours:
                raise ValueError("each slide must contain more regions than neighbours")
            self.group_for_index[idx] = len(self.groups)
            self.local_for_index[idx] = np.arange(len(idx))
            self.groups.append(idx)
            self.neighbours.append(knn_indices(regions.coordinates[idx], neighbours))

    def __len__(self) -> int:
        return len(self.regions.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        group = self.group_for_index[index]
        local = self.local_for_index[index]
        slide_indices = self.groups[group]
        neighbour_indices = slide_indices[self.neighbours[group][local]]
        return {
            "spot": torch.as_tensor(self.regions.features[index], dtype=torch.float32),
            "neighbours": torch.as_tensor(self.regions.features[neighbour_indices], dtype=torch.float32),
            "slide": torch.as_tensor(self.regions.features[slide_indices], dtype=torch.float32),
            "omics": torch.as_tensor(self.regions.omics[index], dtype=torch.float32),
        }


def collate_regions(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_tiles = max(len(item["slide"]) for item in batch)
    dimension = batch[0]["spot"].numel()
    slides = torch.zeros(len(batch), max_tiles, dimension)
    mask = torch.zeros(len(batch), max_tiles, dtype=torch.bool)
    for i, item in enumerate(batch):
        n = len(item["slide"])
        slides[i, :n] = item["slide"]
        mask[i, :n] = True
    return {
        "spot": torch.stack([item["spot"] for item in batch]),
        "neighbours": torch.stack([item["neighbours"] for item in batch]),
        "slide": slides,
        "slide_mask": mask,
        "omics": torch.stack([item["omics"] for item in batch]),
    }
