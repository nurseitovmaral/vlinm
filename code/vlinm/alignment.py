from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class OmicsEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 512, hidden_dim: int = 512) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, expression: torch.Tensor) -> torch.Tensor:
        return self.network(expression)


class SpatialAlignment(nn.Module):
    def __init__(self, feature_dim: int, omics_dim: int, output_dim: int = 512) -> None:
        super().__init__()
        self.attention = nn.Linear(feature_dim, 1)
        self.vision_projection = nn.Sequential(nn.Linear(4 * feature_dim, output_dim), nn.LayerNorm(output_dim))
        self.omics = OmicsEncoder(omics_dim, output_dim)

    def forward(self, spot: torch.Tensor, neighbours: torch.Tensor, slide: torch.Tensor, slide_mask: torch.Tensor, omics: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not slide_mask.any(dim=1).all():
            raise ValueError("every slide must contain at least one tile")
        mean = neighbours.mean(dim=1)
        variance = neighbours.var(dim=1, unbiased=False)
        scores = self.attention(slide).squeeze(-1).masked_fill(~slide_mask, -torch.inf)
        weights = scores.softmax(dim=1)
        pooled = torch.sum(slide * weights.unsqueeze(-1), dim=1)
        vision = self.vision_projection(torch.cat((spot, mean, variance, pooled), dim=-1))
        return vision, self.omics(omics)


class AlignmentLoss(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature)))

    def forward(self, vision: torch.Tensor, omics: torch.Tensor) -> torch.Tensor:
        if vision.ndim != 2 or vision.shape != omics.shape or len(vision) < 2:
            raise ValueError("paired embeddings must have equal shape and at least two samples")
        logits = F.normalize(vision, dim=-1) @ F.normalize(omics, dim=-1).T / self.log_temperature.exp()
        target = torch.arange(len(vision), device=vision.device)
        return (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
