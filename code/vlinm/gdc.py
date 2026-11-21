from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ClinicalCase:
    patient_id: str
    stage: str | None
    vital_status: str | None
    days_to_death: float | None
    days_to_last_follow_up: float | None


def _number(value: Any) -> float | None:
    if value is None or value in ("", "Not Reported", "not reported", "--"):
        return None
    result = float(value)
    if result < 0:
        raise ValueError("follow-up duration cannot be negative")
    return result


def _consistent(values: list[Any], label: str) -> Any:
    nonmissing = {value for value in values if value not in (None, "", "Not Reported", "not reported", "--")}
    if len(nonmissing) > 1:
        raise ValueError(f"conflicting {label} values")
    return next(iter(nonmissing), None)


def read_gdc_cases(path: str | Path) -> list[ClinicalCase]:
    with Path(path).open(encoding="utf-8") as stream:
        payload = json.load(stream)
    cases = payload["data"]["hits"]
    result = []
    seen: set[str] = set()
    for case in cases:
        patient = case["submitter_id"]
        if not patient or patient in seen:
            raise ValueError("case identifiers must be unique and nonempty")
        seen.add(patient)
        diagnoses = case.get("diagnoses", [])
        primary = [item for item in diagnoses if item.get("diagnosis_is_primary_disease") in (True, "true", "Yes", "yes")]
        selected = primary if primary else diagnoses
        stage = _consistent([item.get("ajcc_pathologic_stage") for item in selected], "stage")
        followup = _consistent([_number(item.get("days_to_last_follow_up")) for item in selected], "follow-up")
        demographic = case.get("demographic") or {}
        vital = demographic.get("vital_status")
        result.append(ClinicalCase(patient, stage, vital, _number(demographic.get("days_to_death")), followup))
    return result
