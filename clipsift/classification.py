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
    person_status: str
    assessment_confidence: str
    description: str
    timestamp_seconds: float
    raw_response: str = ""
    parse_error: str | None = None

    @property
    def person_visible(self) -> bool:
        """Backwards-compatible boolean observation."""
        return self.person_status == "present"

    @property
    def confidence(self) -> str:
        """Backwards-compatible confidence name."""
        return self.assessment_confidence

    @property
    def review_required(self) -> bool:
        """ClipSift policy decision; never sourced from model output."""
        return (
            self.parse_error is not None
            or self.person_status in {"present", "uncertain"}
            or self.assessment_confidence == "low"
        )

    @property
    def score(self) -> float:
        base = CONFIDENCE_SCORES.get(self.assessment_confidence, 0.0)
        if self.person_status != "present":
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
        if "person_status" in data:
            person_status = data.get("person_status")
            confidence = data.get("assessment_confidence")
            if person_status not in {"present", "absent", "uncertain"}:
                raise ValueError("person_status must be present, absent, or uncertain")
        else:
            if not isinstance(data.get("person_visible"), bool):
                raise TypeError("person_visible must be a JSON boolean")
            person_status = "present" if data["person_visible"] else "absent"
            confidence = data.get("confidence")
        if not isinstance(confidence, str) or confidence.lower() not in CONFIDENCE_SCORES:
            raise ValueError("assessment confidence must be low, medium, or high")
        if not isinstance(data.get("description"), str):
            raise TypeError("description must be a string")
        return FrameAssessment(
            person_status=person_status,
            assessment_confidence=confidence.lower(),
            description=data["description"].strip(),
            timestamp_seconds=timestamp_seconds,
            raw_response=raw_response,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return FrameAssessment(
            person_status="uncertain",
            assessment_confidence="low",
            description="Model response could not be parsed safely.",
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
    review_votes = sum(1 for assessment in frame_assessments if assessment.review_required or assessment.parse_error)

    if person_votes >= 1:
        status = ClipStatus.PERSON_DETECTED
        rationale = "At least one frame reported a person present, so ClipSift requires human review."
    elif review_votes > 0:
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
        "person_status": assessment.person_status,
        "assessment_confidence": assessment.assessment_confidence,
        "person_visible": assessment.person_visible,
        "confidence": assessment.confidence,
        "description": assessment.description,
        "review_required": assessment.review_required,
        "timestamp_seconds": assessment.timestamp_seconds,
        "parse_error": assessment.parse_error,
    }
