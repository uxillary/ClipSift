"""Frame-level response parsing and clip-level classification."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ClipStatus(str, Enum):
    PERSON_DETECTED = "Person Detected"
    NEEDS_REVIEW = "Needs Review"
    NO_PERSON_DETECTED = "No Person Detected"


CONFIDENCE_SCORES = {
    "low": 0.35,
    "medium": 0.65,
    "high": 0.9,
}


@dataclass(frozen=True)
class FrameAssessment:
    person_visible: bool
    confidence: str
    description: str
    review_required: bool
    timestamp_seconds: float
    raw_response: str = ""
    parse_error: str | None = None

    @property
    def score(self) -> float:
        base = CONFIDENCE_SCORES.get(self.confidence.lower(), 0.0)
        if not self.person_visible:
            return 0.0
        return base


@dataclass(frozen=True)
class ClipDecision:
    status: ClipStatus
    strongest_frame: FrameAssessment | None
    person_votes: int
    review_votes: int
    frames_analysed: int
    rationale: str


def _extract_json_object(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("model response did not contain a JSON object")
    return text[start : end + 1]


def parse_model_response(raw_response: str, timestamp_seconds: float) -> FrameAssessment:
    """Parse a model response without allowing malformed output to stop a scan."""

    try:
        data = json.loads(_extract_json_object(raw_response))
        return FrameAssessment(
            person_visible=bool(data["person_visible"]),
            confidence=str(data.get("confidence", "low")).lower(),
            description=str(data.get("description", "")).strip(),
            review_required=bool(data.get("review_required", False)),
            timestamp_seconds=timestamp_seconds,
            raw_response=raw_response,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return FrameAssessment(
            person_visible=False,
            confidence="low",
            description="Model response could not be parsed safely.",
            review_required=True,
            timestamp_seconds=timestamp_seconds,
            raw_response=raw_response,
            parse_error=str(exc),
        )


def decide_clip(frame_assessments: list[FrameAssessment]) -> ClipDecision:
    """Combine sampled frames into an aid-for-review clip decision."""

    if not frame_assessments:
        return ClipDecision(
            status=ClipStatus.NEEDS_REVIEW,
            strongest_frame=None,
            person_votes=0,
            review_votes=1,
            frames_analysed=0,
            rationale="No frames were analysed, so human review is required.",
        )

    strongest = max(frame_assessments, key=lambda assessment: assessment.score)
    person_votes = sum(1 for assessment in frame_assessments if assessment.person_visible)
    high_person_votes = sum(
        1
        for assessment in frame_assessments
        if assessment.person_visible and assessment.confidence.lower() == "high"
    )
    review_votes = sum(1 for assessment in frame_assessments if assessment.review_required or assessment.parse_error)

    if person_votes >= 2 or high_person_votes >= 1:
        status = ClipStatus.PERSON_DETECTED
        rationale = "Multiple likely detections or one high-confidence detection were found."
    elif person_votes == 1 or review_votes > 0:
        status = ClipStatus.NEEDS_REVIEW
        rationale = "The evidence is ambiguous or one sampled frame needs manual review."
    else:
        status = ClipStatus.NO_PERSON_DETECTED
        rationale = "No sampled frame produced a likely person detection."

    return ClipDecision(
        status=status,
        strongest_frame=strongest if strongest.score > 0 else None,
        person_votes=person_votes,
        review_votes=review_votes,
        frames_analysed=len(frame_assessments),
        rationale=rationale,
    )


def assessment_to_dict(assessment: FrameAssessment | None) -> dict[str, Any]:
    if assessment is None:
        return {}
    return {
        "person_visible": assessment.person_visible,
        "confidence": assessment.confidence,
        "description": assessment.description,
        "review_required": assessment.review_required,
        "timestamp_seconds": assessment.timestamp_seconds,
        "parse_error": assessment.parse_error,
    }
