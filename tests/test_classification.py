from clipsift.classification import ClipStatus, decide_clip, parse_model_response


def test_parse_valid_json_response() -> None:
    assessment = parse_model_response(
        '{"person_visible": true, "confidence": "high", "description": "A person near a gate.", "review_required": true}',
        timestamp_seconds=3.0,
    )

    assert assessment.person_visible is True
    assert assessment.confidence == "high"
    assert assessment.review_required is True
    assert assessment.parse_error is None


def test_parse_malformed_response_returns_review_assessment() -> None:
    assessment = parse_model_response("not json", timestamp_seconds=1.0)

    assert assessment.person_visible is False
    assert assessment.review_required is True
    assert assessment.parse_error is not None


def test_parse_realistic_fenced_gemma_response_safely() -> None:
    assessment = parse_model_response(
        'Here is the result:\n```json\n{"person_visible": false, "confidence": "medium", "description": "No clear person; a reflection is present.", "review_required": true}\n```',
        timestamp_seconds=1.5,
    )
    assert assessment.person_visible is False
    assert assessment.review_required is True
    assert assessment.parse_error is None


def test_string_boolean_is_rejected() -> None:
    assessment = parse_model_response(
        '{"person_visible": "false", "confidence": "high", "description": "Empty", "review_required": false}', 0.0
    )
    assert assessment.review_required is True
    assert assessment.parse_error is not None


def test_decide_clip_person_detected_from_high_confidence_frame() -> None:
    assessment = parse_model_response(
        '{"person_visible": true, "confidence": "high", "description": "Likely person visible.", "review_required": true}',
        timestamp_seconds=0.0,
    )

    decision = decide_clip([assessment])

    assert decision.status == ClipStatus.PERSON_DETECTED
    assert decision.strongest_frame == assessment


def test_decide_clip_needs_review_for_single_medium_detection() -> None:
    assessment = parse_model_response(
        '{"person_visible": true, "confidence": "medium", "description": "Possible person.", "review_required": true}',
        timestamp_seconds=2.0,
    )

    decision = decide_clip([assessment])

    assert decision.status == ClipStatus.NEEDS_REVIEW


def test_decide_clip_no_person_detected() -> None:
    assessment = parse_model_response(
        '{"person_visible": false, "confidence": "high", "description": "Empty driveway.", "review_required": false}',
        timestamp_seconds=0.0,
    )

    decision = decide_clip([assessment])

    assert decision.status == ClipStatus.NO_PERSON_DETECTED
