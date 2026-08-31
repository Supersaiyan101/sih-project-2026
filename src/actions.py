"""Rule-based factor -> recommended action mapping for SIH26017.

Maps a parcel's raw (human-readable) features to corrective actions, each with a
priority. Used by predict.py to enrich the prediction contract and by the dashboard
"recommended actions" panel.

Priorities: 1 = high, 2 = medium, 3 = low.
"""

from __future__ import annotations

from typing import Any

# Statutory/default values reused across rules
PRIORITY_LABELS = {1: "high", 2: "medium", 3: "low"}


def recommend_actions(features: dict[str, Any]) -> list[dict]:
    """Return a list of recommended actions [{factor, action, priority, priority_label}]."""
    actions: list[dict] = []

    def add(factor: str, action: str, priority: int) -> None:
        actions.append({
            "factor": factor,
            "action": action,
            "priority": priority,
            "priority_label": PRIORITY_LABELS[priority],
        })

    court_stay = int(features.get("court_stay", 0))
    compensation = features.get("compensation_status", "paid")
    pending_mutations = int(features.get("pending_mutations", 0))
    owner_count = int(features.get("owner_count", 1))
    land_class = features.get("land_class", "agri")
    rehab = float(features.get("rehab_progress_pct", 100))
    families = int(features.get("affected_families", 0))
    encumbrances = int(features.get("encumbrances", 0))
    responsiveness = float(features.get("stakeholder_responsiveness", 1.0))
    hist_perf = float(features.get("historical_performance_score", 1.0))
    project_type = features.get("project_type", "road")

    if court_stay == 1:
        add("court_stay", "Prioritize court-stay resolution: engage counsel, expedite the "
                          "legal matter, and escalate to the District Collector.", 1)
    if compensation != "paid":
        add("compensation_status", "Fast-track compensation disbursement and release pending "
                                   "award amounts to unblock possession.", 1)
    if rehab < 40 and families > 100:
        add("rehab_progress_pct", "Accelerate Rehabilitation & Resettlement; complete "
                                  "resettlement before possession to avoid displacement delays.", 1)
    if pending_mutations > 2:
        add("pending_mutations", "Expedite mutation records and resolve pending ownership "
                                 "transfers to clear the title.", 2)
    if owner_count > 4:
        add("owner_count", "Facilitate joint-owner consent; hold consolidated hearings to "
                           "reduce consent-related objections.", 2)
    if land_class == "orchard":
        add("land_class", "Begin valuation/appeal process early for orchard land (higher "
                          "compensation disputes expected).", 2)
    if encumbrances > 0:
        add("encumbrances", "Clear encumbrances/liens on titles before the award stage.", 2)
    if responsiveness < 0.5:
        add("stakeholder_responsiveness", "Strengthen inter-departmental coordination and "
                                          "assign a dedicated liaison officer.", 3)
    if project_type in ("dam", "irrigation"):
        add("project_type", "Buffer extra lead time for large-scale displacement and R&R.", 3)
    if hist_perf < 0.4:
        add("historical_performance_score", "Apply past-performance oversight and escalate "
                                            "monitoring for this department/region.", 3)

    if not actions:
        add("none", "No major risk factors identified; continue standard monitoring.", 3)

    actions.sort(key=lambda a: a["priority"])
    return actions
