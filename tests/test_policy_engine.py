from moderation.policy_engine import evaluate_events


def test_empty_evidence_never_approves():
    result = evaluate_events([], {"policy_id": "test", "country": "IN", "status": "test", "rules": []})
    assert result["decision"] == "hold_for_review"
    assert result["requires_human_review"] is True
    assert result["auto_approved"] is False


def test_unmatched_object_detection_requires_review():
    policy = {"policy_id": "test", "country": "IN", "status": "test", "rules": []}
    result = evaluate_events([{"category": "object_detected", "label": "person", "start_time": 0}], policy)
    assert result["decision"] == "human_review"
    assert result["auto_approved"] is False


def test_matching_event_requires_review():
    policy = {"policy_id": "test", "country": "IN", "status": "test", "rules": [{"category": "potential_physical_violence", "enabled": True}]}
    result = evaluate_events([{"category": "potential_physical_violence", "start_time": 1}], policy)
    assert result["decision"] == "human_review"
    assert result["matched_event_count"] == 1


def test_uncertain_event_is_held():
    policy = {"policy_id": "test", "country": "IN", "status": "test", "rules": []}
    result = evaluate_events([{"category": "uncertain"}], policy)
    assert result["decision"] == "hold_for_review"


def test_analysis_error_forces_hold():
    result = evaluate_events([], {"policy_id": "test", "country": "IN", "rules": []}, ["model unavailable"])
    assert result["decision"] == "hold_for_review"
    assert result["analysis_errors"] == ["model unavailable"]
