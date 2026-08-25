"""Three-way listing policy and evidence-aware threshold selection."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from math import sqrt
from typing import Any

import numpy as np


AUTO_MATCH = "auto_match"
AUTO_NO_MATCH = "auto_no_match"
MANUAL_REVIEW = "manual_review"
VALID_ACTIONS = (AUTO_MATCH, AUTO_NO_MATCH, MANUAL_REVIEW)

_WILSON_95_Z = 1.959963984540054


def _normalized_threshold(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    threshold = float(value)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"{name} must be None or a finite value in [0, 1].")
    return threshold


def _validate_thresholds(
    match_threshold: float | None,
    no_match_threshold: float | None,
) -> tuple[float | None, float | None]:
    match = _normalized_threshold(match_threshold, "match_threshold")
    no_match = _normalized_threshold(no_match_threshold, "no_match_threshold")
    if match is not None and no_match is not None and no_match >= match:
        raise ValueError(
            "Enabled thresholds must satisfy no_match_threshold < match_threshold."
        )
    return match, no_match


def apply_policy(
    scores: float | Sequence[float] | np.ndarray,
    match_threshold: float | None,
    no_match_threshold: float | None,
) -> str | np.ndarray:
    """Apply enabled automatic actions and review everything else.

    Passing ``None`` for an action threshold disables that action. When both
    automatic actions are enabled, their thresholds must leave a review region.
    """

    match, no_match = _validate_thresholds(match_threshold, no_match_threshold)
    score_array = np.asarray(scores, dtype=float)
    if np.any(~np.isfinite(score_array)):
        raise ValueError("Scores must be finite.")
    if np.any((score_array < 0.0) | (score_array > 1.0)):
        raise ValueError("Scores must lie in [0, 1].")

    actions = np.full(score_array.shape, MANUAL_REVIEW, dtype=object)
    if match is not None:
        actions[score_array >= match] = AUTO_MATCH
    if no_match is not None:
        actions[score_array <= no_match] = AUTO_NO_MATCH
    if score_array.ndim == 0:
        return str(actions.item())
    return actions.astype(str)


def _as_binary_labels(y_true: Sequence[int] | np.ndarray, name: str) -> np.ndarray:
    labels = np.asarray(y_true, dtype=int).reshape(-1)
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError(f"{name} must contain only binary 0/1 labels.")
    return labels


def _validated_inputs(
    y_true: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    no_match_correct: Sequence[int] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = _as_binary_labels(y_true, "y_true")
    score_array = np.asarray(scores, dtype=float).reshape(-1)
    if labels.size != score_array.size:
        raise ValueError("Labels and scores must have equal lengths.")
    if labels.size == 0:
        raise ValueError("At least one validation example is required.")
    if np.any(~np.isfinite(score_array)) or np.any(
        (score_array < 0.0) | (score_array > 1.0)
    ):
        raise ValueError("Scores must be finite and lie in [0, 1].")

    if no_match_correct is None:
        no_match_labels = 1 - labels
    else:
        no_match_labels = _as_binary_labels(no_match_correct, "no_match_correct")
        if no_match_labels.size != labels.size:
            raise ValueError("no_match_correct must align with labels.")
    return labels, score_array, no_match_labels


def _count(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-negative integer.")
    integer = int(value)
    if integer != value or integer < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return integer


def wilson_interval(
    correct: int,
    support: int,
) -> tuple[float | None, float | None]:
    """Return a two-sided 95% Wilson interval for a binomial proportion."""

    correct = _count(correct, "correct")
    total = _count(support, "support")
    if correct > total:
        raise ValueError("correct_count cannot exceed support.")
    if total == 0:
        return None, None

    proportion = correct / total
    z_squared = _WILSON_95_Z**2
    denominator = 1.0 + z_squared / total
    center = (proportion + z_squared / (2.0 * total)) / denominator
    half_width = (
        _WILSON_95_Z
        * sqrt(
            proportion * (1.0 - proportion) / total
            + z_squared / (4.0 * total**2)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _action_statistics(
    selected: np.ndarray,
    correct_labels: np.ndarray,
    total: int,
) -> dict[str, float | int | None]:
    support = int(selected.sum())
    correct_count = int((correct_labels.astype(bool) & selected).sum())
    error_count = support - correct_count
    precision = correct_count / support if support else None
    wilson_low, wilson_high = wilson_interval(correct_count, support)
    return {
        "support": support,
        "correct_count": correct_count,
        "error_count": error_count,
        "empirical_precision": precision,
        "precision_wilson_95_low": wilson_low,
        "precision_wilson_95_high": wilson_high,
        "coverage": support / total,
    }


def policy_metrics(
    y_true: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    match_threshold: float | None,
    no_match_threshold: float | None,
    no_match_correct: Sequence[int] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute operational metrics for one selected candidate per listing.

    ``y_true`` marks a correct top-candidate link. A low-scored wrong candidate
    is not necessarily a correct no-match decision when the listing has another
    gold partner, so callers can supply the listing-level ``no_match_correct``
    labels. The default keeps ordinary binary examples convenient.
    """

    labels, score_array, no_match_labels = _validated_inputs(
        y_true, scores, no_match_correct
    )
    match, no_match = _validate_thresholds(match_threshold, no_match_threshold)
    actions = np.asarray(apply_policy(score_array, match, no_match), dtype=str)
    auto_match_mask = actions == AUTO_MATCH
    auto_no_match_mask = actions == AUTO_NO_MATCH
    manual_review_mask = actions == MANUAL_REVIEW
    auto_match = {
        "enabled": match is not None,
        "threshold": match,
        **_action_statistics(auto_match_mask, labels, labels.size),
    }
    auto_no_match = {
        "enabled": no_match is not None,
        "threshold": no_match,
        **_action_statistics(auto_no_match_mask, no_match_labels, labels.size),
    }
    automatic_support = auto_match["support"] + auto_no_match["support"]
    automatic_correct = (
        auto_match["correct_count"] + auto_no_match["correct_count"]
    )
    return {
        AUTO_MATCH: auto_match,
        AUTO_NO_MATCH: auto_no_match,
        "automatic_coverage": automatic_support / labels.size,
        "accuracy_on_auto_decisions": (
            automatic_correct / automatic_support if automatic_support else None
        ),
        "manual_review_rate": float(manual_review_mask.mean()),
        "n_manual_review": int(manual_review_mask.sum()),
        "n_total": int(labels.size),
    }


def _threshold_grid(step: float) -> np.ndarray:
    if not 0.0 < step <= 1.0:
        raise ValueError("grid_step must lie in (0, 1].")
    count = int(np.floor(1.0 / step))
    values = np.arange(count + 1, dtype=float) * step
    # Keep decimal grid boundaries stable for the inclusive policy comparisons.
    rounded = np.round(np.append(values, 1.0), decimals=12)
    return np.unique(np.clip(rounded, 0.0, 1.0))


def _validate_minimum_support(value: int, name: str) -> int:
    minimum = _count(value, name)
    if minimum == 0:
        raise ValueError(f"{name} must be a positive integer.")
    return minimum


def _threshold_diagnostics(
    scores: np.ndarray,
    correct_labels: np.ndarray,
    thresholds: np.ndarray,
    *,
    action: str,
    precision_target: float,
    min_support: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = scores.size
    for threshold_value in thresholds:
        threshold = float(threshold_value)
        selected = (
            scores <= threshold if action == AUTO_NO_MATCH else scores >= threshold
        )
        statistics = _action_statistics(selected, correct_labels, total)
        precision = statistics["empirical_precision"]
        support_met = statistics["support"] >= min_support
        precision_target_met = (
            precision is not None and precision >= precision_target
        )
        rows.append(
            {
                "action": action,
                "threshold": threshold,
                **statistics,
                "precision_target": precision_target,
                "min_support": min_support,
                "support_met": support_met,
                "precision_target_met": precision_target_met,
                "feasible": support_met and precision_target_met,
                "selected": False,
            }
        )
    return rows


def _compatible_pair_key(
    pair: tuple[dict[str, Any], dict[str, Any]],
) -> tuple[float, float, float]:
    """Prefer coverage, then lower no-match and higher match boundaries."""
    no_match, match = pair
    return (
        float(no_match["coverage"]) + float(match["coverage"]),
        -float(no_match["threshold"]),
        float(match["threshold"]),
    )


def _standalone_key(row: dict[str, Any]) -> tuple[float, float, int, float]:
    """Apply the documented coverage, precision, action, numeric tie-breaks."""
    action = str(row["action"])
    threshold = float(row["threshold"])
    action_priority = 1 if action == AUTO_NO_MATCH else 0
    conservative_threshold = -threshold if action == AUTO_NO_MATCH else threshold
    return (
        float(row["coverage"]),
        float(row["empirical_precision"]),
        action_priority,
        conservative_threshold,
    )


def _selected_action_result(
    operational: dict[str, Any],
    *,
    feasible: bool,
) -> dict[str, Any]:
    return {
        "feasible": feasible,
        "enabled": bool(operational["enabled"]),
        "threshold": operational["threshold"],
        "support": int(operational["support"]),
        "correct_count": int(operational["correct_count"]),
        "error_count": int(operational["error_count"]),
        "empirical_precision": operational["empirical_precision"],
        "precision_wilson_95_low": operational["precision_wilson_95_low"],
        "precision_wilson_95_high": operational["precision_wilson_95_high"],
        "coverage": float(operational["coverage"]),
    }


def select_thresholds(
    y_true: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    match_precision_target: float = 0.95,
    no_match_precision_target: float = 0.95,
    grid_step: float = 0.01,
    *,
    no_match_correct: Sequence[int] | np.ndarray | None = None,
    min_auto_match_support: int = 20,
    min_auto_no_match_support: int = 20,
    warn: bool = True,
) -> dict[str, Any]:
    """Select only automatic actions with adequate precision and support.

    Each action is screened independently. Compatible feasible actions are
    combined for maximum automatic coverage; an action is never enabled by
    relaxing its evidence requirements. If the feasible regions overlap, the
    best feasible standalone action is enabled and the other remains review.
    """

    for name, target in (
        ("match_precision_target", match_precision_target),
        ("no_match_precision_target", no_match_precision_target),
    ):
        if not 0.0 <= target <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1].")
    match_minimum = _validate_minimum_support(
        min_auto_match_support, "min_auto_match_support"
    )
    no_match_minimum = _validate_minimum_support(
        min_auto_no_match_support, "min_auto_no_match_support"
    )
    labels, score_array, no_match_labels = _validated_inputs(
        y_true, scores, no_match_correct
    )
    grid = _threshold_grid(grid_step)

    no_match_rows = _threshold_diagnostics(
        score_array,
        no_match_labels,
        grid,
        action=AUTO_NO_MATCH,
        precision_target=float(no_match_precision_target),
        min_support=no_match_minimum,
    )
    match_rows = _threshold_diagnostics(
        score_array,
        labels,
        grid,
        action=AUTO_MATCH,
        precision_target=float(match_precision_target),
        min_support=match_minimum,
    )
    diagnostics = [*no_match_rows, *match_rows]
    feasible_no_match = [row for row in no_match_rows if row["feasible"]]
    feasible_match = [row for row in match_rows if row["feasible"]]

    compatible_pairs = [
        (no_match, match)
        for no_match in feasible_no_match
        for match in feasible_match
        if no_match["threshold"] < match["threshold"]
    ]
    selected_rows: list[dict[str, Any]] = []
    if compatible_pairs:
        selected_rows.extend(max(compatible_pairs, key=_compatible_pair_key))
        selection_mode = "both"
    else:
        standalone_candidates = [*feasible_no_match, *feasible_match]
        if standalone_candidates:
            selected = max(standalone_candidates, key=_standalone_key)
            selected_rows.append(selected)
            selection_mode = f"{selected['action']}_only"
        else:
            selection_mode = "manual_review_only"

    for row in selected_rows:
        row["selected"] = True
    selected_by_action = {row["action"]: row for row in selected_rows}
    match_threshold = (
        float(selected_by_action[AUTO_MATCH]["threshold"])
        if AUTO_MATCH in selected_by_action
        else None
    )
    no_match_threshold = (
        float(selected_by_action[AUTO_NO_MATCH]["threshold"])
        if AUTO_NO_MATCH in selected_by_action
        else None
    )
    operational = policy_metrics(
        labels,
        score_array,
        match_threshold,
        no_match_threshold,
        no_match_correct=no_match_labels,
    )
    both_constraints_met = (
        match_threshold is not None and no_match_threshold is not None
    )

    if both_constraints_met:
        warning = None
    elif selected_rows:
        warning = (
            "Both automatic actions could not be enabled together while meeting "
            "their precision and support requirements. Only "
            f"{selected_rows[0]['action']} is enabled; all other listings require "
            "manual review."
        )
    else:
        warning = (
            "No automatic action met its precision and support requirements; all "
            "listings require manual review."
        )
    if warning is not None and warn:
        warnings.warn(warning, RuntimeWarning, stacklevel=2)

    return {
        "match_threshold": match_threshold,
        "no_match_threshold": no_match_threshold,
        AUTO_MATCH: _selected_action_result(
            operational[AUTO_MATCH], feasible=bool(feasible_match)
        ),
        AUTO_NO_MATCH: _selected_action_result(
            operational[AUTO_NO_MATCH], feasible=bool(feasible_no_match)
        ),
        "both_constraints_met": both_constraints_met,
        "selection_mode": selection_mode,
        "automatic_coverage": operational["automatic_coverage"],
        "accuracy_on_auto_decisions": operational["accuracy_on_auto_decisions"],
        "manual_review_rate": operational["manual_review_rate"],
        "n_manual_review": operational["n_manual_review"],
        "n_total": operational["n_total"],
        "warning": warning,
        "threshold_diagnostics": diagnostics,
    }
