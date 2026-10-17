from __future__ import annotations
from pathlib import Path
import yaml

POLICY_DIR = Path(__file__).resolve().parents[1] / "config" / "policies"
POLICY_FILES = {"IN": "india_demo_v1.yaml", "US": "us_demo_v1.yaml", "BD": "bd_demo_v1.yaml", "CUSTOM": "custom_demo_v1.yaml"}


def load_policy(country: str) -> dict:
    filename = POLICY_FILES.get(str(country).upper(), POLICY_FILES["IN"])
    path = POLICY_DIR / filename
    with path.open("r", encoding="utf-8") as stream:
        policy = yaml.safe_load(stream) or {}
    if not isinstance(policy, dict):
        raise ValueError(f"Invalid policy file: {path}")
    policy.setdefault("rules", [])
    return policy


def evaluate_events(events: list[dict], policy: dict, analysis_errors: list[str] | None = None) -> dict:
    """Conservative demo: no automatic approval; analysis errors force a review hold."""
    errors = list(analysis_errors or [])
    enabled_rules = [rule for rule in policy.get("rules", []) if rule.get("enabled", True)]
    active_categories = {rule.get("category") for rule in enabled_rules}
    relevant = [event for event in events if event.get("category") in active_categories]

    if errors:
        decision = "hold_for_review"
        summary = "One or more analysis modules failed or were unavailable. Do not publish until the media is checked manually."
    elif any(event.get("category") == "uncertain" for event in events):
        decision = "hold_for_review"
        summary = "Uncertain evidence was recorded. Human review is required before publication."
    elif relevant:
        decision = "human_review"
        summary = f"{len(relevant)} event(s) matched configured review categories. A reviewer must assess context before publication."
    elif events:
        decision = "human_review"
        summary = "Evidence was detected but no configured rule matched it. The prototype cannot establish that this content is safe; review is required."
    else:
        decision = "hold_for_review"
        summary = "No evidence was returned. This may mean the model was disabled, unavailable, or missed content. This is not an approval; review is required."

    return {
        "decision": decision, "summary": summary,
        "policy_id": policy.get("policy_id"), "country": policy.get("country"),
        "policy_status": policy.get("status", "unknown"),
        "matched_event_count": len(relevant), "matched_events": relevant,
        "analysis_errors": errors, "requires_human_review": True,
        "auto_approved": False,
    }
