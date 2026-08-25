"""Three-way abstention policy and validation threshold selection."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np


AUTO_MATCH = "auto_match"
AUTO_REJECT = "auto_reject"
MANUAL_REVIEW = "manual_review"
VALID_ACTIONS = (AUTO_MATCH, AUTO_REJECT, MANUAL_REVIEW)


def _validate_thresholds(match_threshold: float, reject_threshold: float) -> None:
    if not 0.0 <= reject_threshold < match_threshold <= 1.0:
        raise ValueError(
            "Thresholds must satisfy 0 <= reject_threshold < "
            "match_threshold <= 1."
        )


def apply_policy(
    probabilities: float | Sequence[float] | np.ndarray,
    match_threshold: float,
    reject_threshold: float,
) -> str | np.ndarray:
    """Map calibrated probabilities to auto-match, auto-reject, or review."""
    _validate_thresholds(match_threshold, reject_threshold)
    probability_array = np.asarray(probabilities, dtype=float)
    if np.any(~np.isfinite(probability_array)):
        raise ValueError("Probabilities must be finite.")
    if np.any((probability_array < 0.0) | (probability_array > 1.0)):
        raise ValueError("Probabilities must lie in [0, 1].")

    actions = np.full(probability_array.shape, MANUAL_REVIEW, dtype=object)
    actions[probability_array >= match_threshold] = AUTO_MATCH
    actions[probability_array <= reject_threshold] = AUTO_REJECT
    if probability_array.ndim == 0:
        return str(actions.item())
    return actions.astype(str)


def _as_binary_labels(y_true: Sequence[int] | np.ndarray) -> np.ndarray:
    labels = np.asarray(y_true, dtype=int).reshape(-1)
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("y_true must contain only binary 0/1 labels.")
    return labels


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def policy_metrics(
    y_true: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    match_threshold: float,
    reject_threshold: float,
    reject_correct: Sequence[int] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute operational metrics for one decision row per listing.

    ``y_true`` says whether the selected candidate is a gold match. In the
    open-world listing decision, rejecting a wrong candidate is not necessarily
    correct: a different gold candidate may exist. Pass ``reject_correct`` as
    true only for listings with no gold partner. Its default, ``1 - y_true``,
    preserves ordinary binary-pair behavior for small tests and other callers.
    """
    labels = _as_binary_labels(y_true)
    scores = np.asarray(probabilities, dtype=float).reshape(-1)
    if labels.size != scores.size:
        raise ValueError("Labels and probabilities must have equal lengths.")
    if labels.size == 0:
        raise ValueError("At least one validation example is required.")
    if reject_correct is None:
        reject_labels = 1 - labels
    else:
        reject_labels = _as_binary_labels(reject_correct)
        if reject_labels.size != labels.size:
            raise ValueError("reject_correct must align with labels.")

    actions = np.asarray(
        apply_policy(scores, match_threshold, reject_threshold), dtype=str
    )
    accepted = actions == AUTO_MATCH
    rejected = actions == AUTO_REJECT
    reviewed = actions == MANUAL_REVIEW
    auto_decided = accepted | rejected

    accepted_count = int(accepted.sum())
    rejected_count = int(rejected.sum())
    reviewed_count = int(reviewed.sum())
    correct_accepts = int(((labels == 1) & accepted).sum())
    correct_rejects = int(((reject_labels == 1) & rejected).sum())
    total = int(labels.size)

    return {
        "auto_match_precision": _ratio(correct_accepts, accepted_count),
        "auto_match_coverage": accepted_count / total,
        "auto_reject_precision": _ratio(correct_rejects, rejected_count),
        "auto_reject_coverage": rejected_count / total,
        "manual_review_rate": reviewed_count / total,
        "automatic_coverage": int(auto_decided.sum()) / total,
        "accuracy_on_auto_decisions": _ratio(
            correct_accepts + correct_rejects, int(auto_decided.sum())
        ),
        "n_auto_match": accepted_count,
        "n_auto_reject": rejected_count,
        "n_manual_review": reviewed_count,
        "n_total": total,
    }


def _threshold_grid(step: float) -> np.ndarray:
    if not 0.0 < step <= 1.0:
        raise ValueError("grid_step must lie in (0, 1].")
    count = int(np.floor(1.0 / step))
    values = np.arange(count + 1, dtype=float) * step
    return np.unique(np.clip(np.append(values, 1.0), 0.0, 1.0))


def select_thresholds(
    y_true: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    match_precision_target: float = 0.95,
    reject_precision_target: float = 0.95,
    grid_step: float = 0.01,
    *,
    reject_correct: Sequence[int] | np.ndarray | None = None,
    warn: bool = True,
) -> dict[str, Any]:
    """Select thresholds that maximize coverage under precision constraints.

    Only threshold pairs producing at least one match and one reject are treated
    as normally feasible. If no pair reaches both targets, the deterministic
    fallback minimizes summed precision shortfall, then maximizes coverage and
    achieved precision. The returned warning records that fallback explicitly.
    """
    for name, target in (
        ("match_precision_target", match_precision_target),
        ("reject_precision_target", reject_precision_target),
    ):
        if not 0.0 <= target <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1].")

    labels = _as_binary_labels(y_true)
    scores = np.asarray(probabilities, dtype=float).reshape(-1)
    if labels.size != scores.size:
        raise ValueError("Labels and probabilities must have equal lengths.")
    if labels.size == 0:
        raise ValueError("At least one validation example is required.")
    if np.any(~np.isfinite(scores)) or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("Probabilities must be finite and lie in [0, 1].")
    if reject_correct is None:
        reject_labels = 1 - labels
    else:
        reject_labels = _as_binary_labels(reject_correct)
        if reject_labels.size != labels.size:
            raise ValueError("reject_correct must align with labels.")

    feasible: list[tuple[tuple[float, ...], dict[str, Any]]] = []
    fallback: list[tuple[tuple[float, ...], dict[str, Any]]] = []
    grid = _threshold_grid(grid_step)

    for reject_threshold in grid[:-1]:
        for match_threshold in grid[1:]:
            if reject_threshold >= match_threshold:
                continue
            metrics = policy_metrics(
                labels,
                scores,
                float(match_threshold),
                float(reject_threshold),
                reject_correct=reject_labels,
            )
            match_precision = metrics["auto_match_precision"]
            reject_precision = metrics["auto_reject_precision"]
            has_both_actions = (
                metrics["n_auto_match"] > 0 and metrics["n_auto_reject"] > 0
            )
            match_value = 0.0 if match_precision is None else match_precision
            reject_value = 0.0 if reject_precision is None else reject_precision
            shortfall = max(0.0, match_precision_target - match_value) + max(
                0.0, reject_precision_target - reject_value
            )

            result = {
                "match_threshold": float(match_threshold),
                "reject_threshold": float(reject_threshold),
                **metrics,
            }
            conservative_width = float(match_threshold - reject_threshold)
            if (
                has_both_actions
                and match_value >= match_precision_target
                and reject_value >= reject_precision_target
            ):
                score = (
                    float(metrics["automatic_coverage"]),
                    min(match_value, reject_value),
                    conservative_width,
                )
                feasible.append((score, result))

            if has_both_actions:
                fallback_score = (
                    -shortfall,
                    float(metrics["automatic_coverage"]),
                    min(match_value, reject_value),
                    conservative_width,
                )
                fallback.append((fallback_score, result))

    if feasible:
        _, selected = max(feasible, key=lambda item: item[0])
        selected["constraints_met"] = True
        selected["precision_shortfall"] = 0.0
        selected["warning"] = None
        return selected

    if not fallback:
        # A tiny validation set may make it impossible to emit both actions.
        reject_threshold = float(grid[0])
        match_threshold = float(grid[-1])
        selected = {
            "match_threshold": match_threshold,
            "reject_threshold": reject_threshold,
            **policy_metrics(
                labels,
                scores,
                match_threshold,
                reject_threshold,
                reject_correct=reject_labels,
            ),
        }
    else:
        _, selected = max(fallback, key=lambda item: item[0])

    match_value = selected["auto_match_precision"] or 0.0
    reject_value = selected["auto_reject_precision"] or 0.0
    selected["constraints_met"] = False
    selected["precision_shortfall"] = max(
        0.0, match_precision_target - match_value
    ) + max(0.0, reject_precision_target - reject_value)
    selected["warning"] = (
        "Validation precision targets were not jointly achievable; using the "
        "threshold pair with the smallest total precision shortfall. Achieved "
        f"auto-match precision={match_value:.3f} and auto-reject "
        f"precision={reject_value:.3f}."
    )
    if warn:
        warnings.warn(selected["warning"], RuntimeWarning, stacklevel=2)
    return selected


choose_thresholds = select_thresholds
decide = apply_policy
