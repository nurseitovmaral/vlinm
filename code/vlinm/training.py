from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def training_step(model: nn.Module, objective: nn.Module, optimizer: torch.optim.Optimizer, batch: dict[str, torch.Tensor]) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    vision, omics = model(batch["spot"], batch["neighbours"], batch["slide"], batch["slide_mask"], batch["omics"])
    loss = objective(vision, omics)
    if not torch.isfinite(loss):
        raise ValueError("non-finite loss")
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def save_state(path: str | Path, model: nn.Module, objective: nn.Module, optimizer: torch.optim.Optimizer, seed: int, step: int) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = {"model": model.state_dict(), "objective": objective.state_dict(), "optimizer": optimizer.state_dict(), "seed": seed, "step": step, "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}
    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
    fd, temporary = tempfile.mkstemp(dir=destination.parent, prefix=".state-")
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_state(path: str | Path, model: nn.Module, objective: nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    objective.load_state_dict(state["objective"])
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])
    if "cuda_rng" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return {"seed": state["seed"], "step": state["step"]}
