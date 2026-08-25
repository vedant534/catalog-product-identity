"""Evaluation and compact report helpers for catalog identity matching."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.model import select_top_candidates
from src.policy import (
    AUTO_MATCH,
    AUTO_NO_MATCH,
    MANUAL_REVIEW,
    apply_policy,
    group_action_statistics,
    wilson_interval,
)


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _listing_recall(
    candidates: pd.DataFrame,
    mask: pd.Series | np.ndarray,
    gold_by_listing: Mapping[str, set[str]],
) -> float:
    retrieved = (
        candidates.loc[mask, ["google_id", "amazon_id"]]
        .groupby("google_id")["amazon_id"]
        .agg(set)
        .to_dict()
    )
    hits = sum(
        bool(gold_ids & retrieved.get(google_id, set()))
        for google_id, gold_ids in gold_by_listing.items()
    )
    return _rate(hits, len(gold_by_listing))


def compute_retrieval_metrics(
    candidates: pd.DataFrame,
    diagnostic_pool: pd.DataFrame,
    gold_pairs: pd.DataFrame,
    ks: Sequence[int] = (5, 10, 20),
    all_listing_ids: Iterable[str] | None = None,
    latency_seconds: float | Sequence[float] | None = None,
    *,
    per_channel_candidate_depth: int | None = None,
) -> dict[str, float | int | str]:
    """Compare lexical, dense, fixed-budget RRF, and per-channel union recall."""
    required = {"google_id", "amazon_id"}
    if not required.issubset(candidates.columns):
        raise ValueError(f"candidates must contain {sorted(required)}")
    if not required.issubset(diagnostic_pool.columns):
        raise ValueError(f"diagnostic_pool must contain {sorted(required)}")
    if not required.issubset(gold_pairs.columns):
        raise ValueError(f"gold_pairs must contain {sorted(required)}")

    if per_channel_candidate_depth is None:
        raise ValueError("per_channel_candidate_depth is required")
    if isinstance(per_channel_candidate_depth, (bool, np.bool_)):
        raise ValueError("per_channel_candidate_depth must be a positive integer")
    candidate_depth = int(per_channel_candidate_depth)
    if candidate_depth != per_channel_candidate_depth or candidate_depth <= 0:
        raise ValueError("per_channel_candidate_depth must be a positive integer")
    requested_ks: list[int] = []
    for value in ks:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("retrieval K values must be positive integers")
        k = int(value)
        if k != value:
            raise ValueError("retrieval K values must be positive integers")
        requested_ks.append(k)
    if any(k <= 0 for k in requested_ks):
        raise ValueError("retrieval K values must be positive")
    too_deep = [k for k in requested_ks if k > candidate_depth]
    if too_deep:
        raise ValueError(
            "Requested retrieval Recall@K exceeds the generated per-channel "
            f"candidate depth {candidate_depth}: {too_deep}"
        )

    primary = candidates.copy()
    pool = diagnostic_pool.copy()
    for frame in (primary, pool):
        frame[["google_id", "amazon_id"]] = frame[["google_id", "amazon_id"]].astype(str)
        if "gold_injected" in frame:
            frame.drop(
                frame.index[frame["gold_injected"].fillna(False).astype(bool)],
                inplace=True,
            )

    gold = gold_pairs[["google_id", "amazon_id"]].astype(str).drop_duplicates()
    gold_by_listing = gold.groupby("google_id")["amazon_id"].agg(set).to_dict()
    metrics: dict[str, float | int | str] = {}
    for k in requested_ks:
        lexical = pd.to_numeric(pool["lexical_rank"], errors="coerce").le(k)
        dense = pd.to_numeric(pool["dense_rank"], errors="coerce").le(k)
        rrf = pd.to_numeric(primary["rrf_rank"], errors="coerce").le(k)
        metrics[f"lexical_recall_at_{k}"] = _listing_recall(
            pool, lexical, gold_by_listing
        )
        metrics[f"dense_recall_at_{k}"] = _listing_recall(pool, dense, gold_by_listing)
        metrics[f"rrf_recall_at_{k}"] = _listing_recall(primary, rrf, gold_by_listing)
        metrics[f"union_per_channel_recall_at_{k}"] = _listing_recall(
            pool, lexical | dense, gold_by_listing
        )

    if all_listing_ids is None:
        listing_ids = set(primary["google_id"])
    else:
        listing_ids = {str(value) for value in all_listing_ids}
    counts = primary.groupby("google_id")["amazon_id"].nunique().to_dict()
    metrics["mean_primary_candidate_count"] = (
        float(np.mean([counts.get(value, 0) for value in listing_ids]))
        if listing_ids
        else 0.0
    )
    metrics["listing_count"] = len(listing_ids)
    if latency_seconds is not None:
        latency = np.atleast_1d(np.asarray(latency_seconds, dtype=float))
        metrics["retrieval_latency_seconds"] = float(latency.sum())
        if latency.size:
            mean_seconds = (
                float(latency[0]) / len(listing_ids)
                if latency.size == 1 and listing_ids
                else float(latency.mean())
            )
            metrics["amortized_batch_retrieval_ms_per_listing"] = (
                mean_seconds * 1000.0
            )
            if latency.size > 1:
                metrics["p95_latency_ms"] = float(np.percentile(latency, 95) * 1000.0)
        metrics["latency_scope"] = (
            "Query dense encoding plus exact lexical/dense retrieval and RRF; "
            "catalog precomputation and model loading excluded."
        )
    return metrics


def compute_pair_metrics(
    y_true: Sequence[int],
    probabilities: Sequence[float],
    classification_threshold: float = 0.5,
) -> dict[str, float | int]:
    """Return pair discrimination, classification, and calibration diagnostics."""
    y = np.asarray(y_true, dtype=int).reshape(-1)
    probability = np.asarray(probabilities, dtype=float).reshape(-1)
    if y.shape != probability.shape or y.size == 0:
        raise ValueError("y_true and probabilities must be non-empty aligned arrays")
    threshold = float(classification_threshold)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("classification_threshold must be finite and lie in [0, 1]")
    predicted = probability >= threshold
    return {
        "count": int(y.size),
        "positive_count": int(y.sum()),
        "classification_threshold": threshold,
        "pr_auc": float(average_precision_score(y, probability)),
        "roc_auc": (
            float(roc_auc_score(y, probability)) if np.unique(y).size == 2 else 0.0
        ),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "brier_score": float(brier_score_loss(y, probability)),
    }


def compute_calibration_metrics(
    y_true: Sequence[int], probabilities: Sequence[float]
) -> dict[str, float | int]:
    labels = np.asarray(y_true, dtype=int).reshape(-1)
    scores = np.asarray(probabilities, dtype=float).reshape(-1)
    if labels.shape != scores.shape or labels.size == 0:
        raise ValueError("Calibration labels and scores must be non-empty and aligned")
    return {
        "count": int(labels.size),
        "positive_count": int(labels.sum()),
        "observed_match_rate": float(labels.mean()),
        "mean_score": float(scores.mean()),
        "brier_score": float(brier_score_loss(labels, scores)),
    }


def _validated_duplicate_group_mapping(
    listing_ids: set[str],
    duplicate_group_by_listing: Mapping[str, str],
) -> dict[str, str]:
    """Return one non-empty duplicate-group ID for every requested listing."""
    if not isinstance(duplicate_group_by_listing, Mapping):
        raise ValueError("duplicate_group_by_listing must be a listing-to-group mapping")

    normalized: dict[str, str] = {}
    for raw_listing_id, raw_group_id in duplicate_group_by_listing.items():
        listing_id = str(raw_listing_id).strip()
        if not listing_id or listing_id in normalized:
            raise ValueError(
                "duplicate_group_by_listing must contain unique non-empty listing IDs"
            )
        if raw_group_id is None or bool(pd.isna(raw_group_id)):
            raise ValueError("Duplicate-group IDs must be non-empty and non-missing")
        group_id = str(raw_group_id).strip()
        if not group_id:
            raise ValueError("Duplicate-group IDs must be non-empty and non-missing")
        normalized[listing_id] = group_id

    missing = sorted(listing_ids - set(normalized))
    extra = sorted(set(normalized) - listing_ids)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={len(missing)}")
        if extra:
            details.append(f"extra={len(extra)}")
        raise ValueError(
            "duplicate_group_by_listing must align exactly with requested listings ("
            + ", ".join(details)
            + ")"
        )
    return normalized


def build_listing_predictions(
    candidate_predictions: pd.DataFrame,
    gold_pairs: pd.DataFrame,
    all_listing_ids: Iterable[str],
    policy: Mapping[str, Any],
    *,
    duplicate_group_by_listing: Mapping[str, str],
) -> pd.DataFrame:
    """Build the single authoritative decision row for every listing."""
    frame = candidate_predictions.copy()
    frame[["google_id", "amazon_id"]] = frame[["google_id", "amazon_id"]].astype(str)
    frame["probability"] = pd.to_numeric(frame["probability"], errors="raise")
    top = select_top_candidates(frame)
    listing_ids = {str(value) for value in all_listing_ids}
    group_by_listing = _validated_duplicate_group_mapping(
        listing_ids,
        duplicate_group_by_listing,
    )
    missing = listing_ids - set(top["google_id"])
    if missing:
        raise ValueError(f"No retrieved candidates for {len(missing)} listing(s)")
    top = top.loc[top["google_id"].isin(listing_ids)].copy()
    top["duplicate_group_id"] = top["google_id"].map(group_by_listing)

    ranked = frame.sort_values(
        ["google_id", "probability", "amazon_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    second_scores = (
        ranked.groupby("google_id", sort=False)["probability"]
        .apply(lambda values: float(values.iloc[1]) if len(values) > 1 else 0.0)
        .to_dict()
    )
    top["top1_top2_margin"] = [
        float(score - second_scores.get(google_id, 0.0))
        for google_id, score in zip(top["google_id"], top["probability"])
    ]

    gold = gold_pairs[["google_id", "amazon_id"]].astype(str).drop_duplicates()
    gold_set = set(gold.itertuples(index=False, name=None))
    gold_by_listing = gold.groupby("google_id")["amazon_id"].agg(set).to_dict()
    retrieved_pairs = set(
        frame[["google_id", "amazon_id"]].itertuples(index=False, name=None)
    )
    top["has_gold_listing"] = top["google_id"].isin(gold_by_listing)
    top["is_gold_pair"] = [
        pair in gold_set
        for pair in top[["google_id", "amazon_id"]].itertuples(index=False, name=None)
    ]
    top["gold_retrieved"] = [
        any(
            (google_id, amazon_id) in retrieved_pairs
            for amazon_id in gold_by_listing.get(google_id, set())
        )
        for google_id in top["google_id"]
    ]
    top["ranking_outcome"] = np.select(
        [
            ~top["has_gold_listing"],
            top["has_gold_listing"] & ~top["gold_retrieved"],
            top["has_gold_listing"] & top["gold_retrieved"] & ~top["is_gold_pair"],
        ],
        ["assumed_no_match", "retrieval_miss", "reranking_miss"],
        default="top_gold_match",
    )

    match_threshold = (
        policy["auto_match"]["threshold"] if policy["auto_match"]["enabled"] else None
    )
    no_match_threshold = (
        policy["auto_no_match"]["threshold"]
        if policy["auto_no_match"]["enabled"]
        else None
    )
    top["action"] = apply_policy(
        top["probability"].to_numpy(),
        match_threshold=match_threshold,
        no_match_threshold=no_match_threshold,
    )
    top["action_correct"] = pd.Series(pd.NA, index=top.index, dtype="boolean")
    match_mask = top["action"].eq(AUTO_MATCH)
    no_match_mask = top["action"].eq(AUTO_NO_MATCH)
    top.loc[match_mask, "action_correct"] = top.loc[match_mask, "is_gold_pair"]
    top.loc[no_match_mask, "action_correct"] = ~top.loc[
        no_match_mask, "has_gold_listing"
    ]
    return top.sort_values("google_id", kind="mergesort").reset_index(drop=True)


def _listing_duplicate_group_ids(listings: pd.DataFrame) -> np.ndarray:
    if "duplicate_group_id" not in listings:
        raise ValueError(
            "Listing predictions must contain duplicate_group_id for group-aware evidence"
        )
    values = listings["duplicate_group_id"]
    if values.isna().any() or values.astype(str).str.strip().eq("").any():
        raise ValueError("Listing predictions contain missing duplicate_group_id values")
    return values.astype(str).str.strip().to_numpy()


def _action_metrics(
    listings: pd.DataFrame,
    action: str,
    correct_labels: np.ndarray,
    duplicate_group_ids: np.ndarray,
) -> dict[str, Any]:
    selected = listings["action"].eq(action)
    support = int(selected.sum())
    selected_array = selected.to_numpy(dtype=bool)
    correct = int((selected_array & correct_labels).sum())
    low, high = wilson_interval(correct, support)
    listing_operation = {
        "support": support,
        "correct_count": correct,
        "error_count": support - correct,
        "empirical_precision": _rate(correct, support) if support else None,
        "precision_wilson_95_low": low,
        "precision_wilson_95_high": high,
        "coverage": _rate(support, len(listings)),
    }
    group_evidence = group_action_statistics(
        selected_array,
        correct_labels.astype(int),
        duplicate_group_ids,
    )
    return {
        "support": listing_operation["support"],
        "correct_count": listing_operation["correct_count"],
        "error_count": listing_operation["error_count"],
        "empirical_precision": listing_operation["empirical_precision"],
        "wilson_95_low": low,
        "wilson_95_high": high,
        "listing_coverage": listing_operation["coverage"],
        "group_evidence": group_evidence,
        "listing_operation": listing_operation,
    }


def _manual_review_metrics(
    listings: pd.DataFrame,
    duplicate_group_ids: np.ndarray,
) -> dict[str, Any]:
    """Report review support while leaving action correctness undefined."""
    selected = listings["action"].eq(MANUAL_REVIEW).to_numpy(dtype=bool)
    support = int(selected.sum())
    group_evidence = group_action_statistics(
        selected,
        np.ones(len(listings), dtype=int),
        duplicate_group_ids,
    )
    undefined_fields = {
        "correct_count": None,
        "error_count": None,
        "empirical_precision": None,
        "precision_wilson_95_low": None,
        "precision_wilson_95_high": None,
    }
    group_evidence = {**group_evidence, **undefined_fields}
    listing_operation = {
        "support": support,
        **undefined_fields,
        "coverage": _rate(support, len(listings)),
    }
    return {
        "support": support,
        "correct_count": None,
        "error_count": None,
        "empirical_precision": None,
        "wilson_95_low": None,
        "wilson_95_high": None,
        "listing_coverage": listing_operation["coverage"],
        "group_evidence": group_evidence,
        "listing_operation": listing_operation,
        "correctness_definition": (
            "Undefined: manual review is an abstention, not an automatic decision."
        ),
    }


def compute_listing_metrics(listing_predictions: pd.DataFrame) -> dict[str, Any]:
    """Evaluate enabled operational actions on one row per listing."""
    actions = set(listing_predictions["action"].dropna().astype(str))
    unknown = actions - {AUTO_MATCH, AUTO_NO_MATCH, MANUAL_REVIEW}
    if unknown:
        raise ValueError(f"Unknown policy actions: {sorted(unknown)}")
    reviewed = listing_predictions["action"].eq(MANUAL_REVIEW)
    matched = listing_predictions["has_gold_listing"].astype(bool)
    duplicate_group_ids = _listing_duplicate_group_ids(listing_predictions)
    match_correct = listing_predictions["is_gold_pair"].astype(bool).to_numpy()
    no_match_correct = (~matched).to_numpy()
    automatic = ~reviewed
    correct_automatic = listing_predictions.loc[automatic, "action_correct"].fillna(False)
    matched_count = int(matched.sum())
    correct_auto_matches = int(
        (
            listing_predictions["action"].eq(AUTO_MATCH)
            & listing_predictions["is_gold_pair"]
        ).sum()
    )
    return {
        "listing_count": len(listing_predictions),
        "duplicate_group_count": int(len(set(duplicate_group_ids))),
        "auto_match": _action_metrics(
            listing_predictions,
            AUTO_MATCH,
            match_correct,
            duplicate_group_ids,
        ),
        "auto_no_match": _action_metrics(
            listing_predictions,
            AUTO_NO_MATCH,
            no_match_correct,
            duplicate_group_ids,
        ),
        "manual_review": _manual_review_metrics(
            listing_predictions,
            duplicate_group_ids,
        ),
        "manual_review_count": int(reviewed.sum()),
        "manual_review_rate": _rate(int(reviewed.sum()), len(listing_predictions)),
        "automatic_coverage": _rate(int(automatic.sum()), len(listing_predictions)),
        "accuracy_on_automatic_decisions": (
            float(correct_automatic.mean()) if len(correct_automatic) else None
        ),
        "matched_listing_count": matched_count,
        "assumed_no_match_listing_count": int((~matched).sum()),
        "matched_review_count": int((reviewed & matched).sum()),
        "matched_review_rate": _rate(int((reviewed & matched).sum()), matched_count),
        "assumed_no_match_review_count": int((reviewed & ~matched).sum()),
        "assumed_no_match_review_rate": _rate(
            int((reviewed & ~matched).sum()), int((~matched).sum())
        ),
        "end_to_end_auto_match_rate": _rate(correct_auto_matches, matched_count),
    }


def compute_no_match_metrics(listing_predictions: pd.DataFrame) -> dict[str, Any]:
    """Evaluate closed-world listing no-match detection."""
    labels = (~listing_predictions["has_gold_listing"].astype(bool)).astype(int)
    scores = 1.0 - listing_predictions["probability"].to_numpy(dtype=float)
    predicted = listing_predictions["action"].eq(AUTO_NO_MATCH).to_numpy()
    policy_support = int(predicted.sum())
    return {
        "assumption": "Official mapping absence is treated as no-match.",
        "count": int(len(labels)),
        "positive_count": int(labels.sum()),
        "average_precision": float(average_precision_score(labels, scores)),
        "precision_at_policy": (
            float(precision_score(labels, predicted, zero_division=0))
            if policy_support
            else None
        ),
        "recall_at_policy": float(recall_score(labels, predicted, zero_division=0)),
    }


def normalized_title_signature(value: object) -> str:
    text = "" if pd.isna(value) else str(value).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text)).strip()


def exact_title_ambiguity_ids(
    google: pd.DataFrame,
    amazon: pd.DataFrame,
    gold_pairs: pd.DataFrame,
    listing_ids: Iterable[str],
) -> set[str]:
    """Return officially unmapped listings colliding with a catalog title."""
    selected_ids = {str(value) for value in listing_ids}
    mapped_ids = set(gold_pairs["google_id"].astype(str))
    catalog_titles = {
        signature
        for signature in amazon["title"].map(normalized_title_signature)
        if signature
    }
    selected = google.loc[google["product_id"].astype(str).isin(selected_ids)]
    return {
        str(row.product_id)
        for row in selected.itertuples(index=False)
        if str(row.product_id) not in mapped_ids
        and normalized_title_signature(row.title) in catalog_titles
        and normalized_title_signature(row.title)
    }


def compute_sensitivity_metrics(
    listing_predictions: pd.DataFrame, ambiguous_listing_ids: set[str]
) -> dict[str, Any]:
    retained = listing_predictions.loc[
        ~listing_predictions["google_id"].astype(str).isin(ambiguous_listing_ids)
    ].copy()
    return {
        "definition": (
            "Officially unmapped listings with an exact normalized-title Amazon "
            "collision are excluded, not relabelled."
        ),
        "excluded_listing_count": len(listing_predictions) - len(retained),
        "listing_policy": compute_listing_metrics(retained),
        "no_match_detection": compute_no_match_metrics(retained),
    }


def extract_error_examples(
    candidate_predictions: pd.DataFrame,
    listing_predictions: pd.DataFrame,
    gold_pairs: pd.DataFrame,
    max_per_type: int = 10,
    google_records: pd.DataFrame | None = None,
    amazon_records: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Keep listing-policy errors separate from pair-score diagnostics."""
    if max_per_type <= 0:
        raise ValueError("max_per_type must be positive")
    pairs = candidate_predictions.copy()
    pairs[["google_id", "amazon_id"]] = pairs[["google_id", "amazon_id"]].astype(str)
    gold_set = set(
        gold_pairs[["google_id", "amazon_id"]]
        .astype(str)
        .itertuples(index=False, name=None)
    )
    pairs["is_gold_pair"] = [
        pair in gold_set
        for pair in pairs[["google_id", "amazon_id"]].itertuples(index=False, name=None)
    ]
    pair_sections: list[pd.DataFrame] = []
    pair_fp = pairs.loc[(pairs["probability"] >= 0.5) & ~pairs["is_gold_pair"]].nlargest(
        max_per_type, "probability"
    )
    if not pair_fp.empty:
        pair_fp = pair_fp.copy()
        pair_fp.insert(0, "error_type", "pair_false_positive_at_0_5")
        pair_sections.append(pair_fp)
    pair_fn = pairs.loc[(pairs["probability"] < 0.5) & pairs["is_gold_pair"]].nsmallest(
        max_per_type, "probability"
    )
    if not pair_fn.empty:
        pair_fn = pair_fn.copy()
        pair_fn.insert(0, "error_type", "pair_false_negative_at_0_5")
        pair_sections.append(pair_fn)

    listing_rules = (
        (
            "false_automatic_match",
            listing_predictions["action"].eq(AUTO_MATCH)
            & ~listing_predictions["is_gold_pair"],
        ),
        (
            "false_automatic_no_match",
            listing_predictions["action"].eq(AUTO_NO_MATCH)
            & listing_predictions["has_gold_listing"],
        ),
        (
            "reviewed_top_gold_match",
            listing_predictions["action"].eq(MANUAL_REVIEW)
            & listing_predictions["is_gold_pair"],
        ),
        ("retrieval_miss", listing_predictions["ranking_outcome"].eq("retrieval_miss")),
        ("reranking_miss", listing_predictions["ranking_outcome"].eq("reranking_miss")),
    )
    listing_sections: list[pd.DataFrame] = []
    for name, mask in listing_rules:
        section = listing_predictions.loc[mask].head(max_per_type).copy()
        if not section.empty:
            section.insert(0, "error_type", name)
            listing_sections.append(section)

    sections = pair_sections + listing_sections
    result = pd.concat(sections, ignore_index=True, sort=False) if sections else pd.DataFrame()
    result = _merge_product_context(result, google_records, "google_id", "google")
    return _merge_product_context(result, amazon_records, "amazon_id", "amazon")


def _merge_product_context(
    examples: pd.DataFrame,
    records: pd.DataFrame | None,
    pair_id_column: str,
    prefix: str,
) -> pd.DataFrame:
    if records is None or examples.empty or pair_id_column not in examples:
        return examples
    record_id_column = "product_id" if "product_id" in records else "id"
    fields = [
        column
        for column in (record_id_column, "title", "manufacturer", "price")
        if column in records
    ]
    context = records[fields].drop_duplicates(record_id_column).rename(
        columns={
            record_id_column: pair_id_column,
            **{
                column: f"{prefix}_{column}"
                for column in fields
                if column != record_id_column
            },
        }
    )
    context[pair_id_column] = context[pair_id_column].astype(str)
    return examples.merge(context, on=pair_id_column, how="left")


def save_evaluation_plots(
    pair_y_true: Sequence[int],
    pair_probabilities: Sequence[float],
    top_y_true: Sequence[int],
    top_probabilities: Sequence[float],
    retrieval_metrics: Mapping[str, Any],
    output_dir: str | Path,
    split_label: str,
) -> list[Path]:
    """Save retrieval, pair PR, and separately labelled calibration plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    ks = sorted(
        {
            int(str(key).rsplit("_", 1)[-1])
            for key in retrieval_metrics
            if "recall_at_" in str(key) and str(key).rsplit("_", 1)[-1].isdigit()
        }
    )
    path = output / f"{split_label}_retrieval_recall.png"
    fig, axis = plt.subplots(figsize=(8, 4))
    positions = np.arange(len(ks), dtype=float)
    width = 0.2
    channels = (
        ("lexical", "Lexical"),
        ("dense", "Dense"),
        ("rrf", "RRF fixed-budget"),
        ("union_per_channel", "Union per channel"),
    )
    for offset, (key_prefix, label) in enumerate(channels):
        values = [float(retrieval_metrics[f"{key_prefix}_recall_at_{k}"]) for k in ks]
        axis.bar(positions + (offset - 1.5) * width, values, width, label=label)
    axis.set(
        xticks=positions,
        xticklabels=[f"Recall@{k}" for k in ks],
        ylabel="Gold-bearing listings retrieved",
        title=f"{split_label.title()} candidate retrieval",
        ylim=(0, 1.05),
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    pair_y = np.asarray(pair_y_true, dtype=int)
    pair_scores = np.asarray(pair_probabilities, dtype=float)
    precision, recall, _ = precision_recall_curve(pair_y, pair_scores)
    path = output / f"{split_label}_pair_precision_recall_curve.png"
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.plot(recall, precision)
    axis.axhline(pair_y.mean(), color="gray", linestyle="--", label="prevalence")
    axis.set(
        xlabel="Recall",
        ylabel="Precision",
        title=f"{split_label.title()} pair PR curve",
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    for prefix, title, labels, scores in (
        ("pair", f"{split_label.title()} pair-level reliability", pair_y, pair_scores),
        (
            "top_candidate",
            f"{split_label.title()} top-candidate reliability",
            np.asarray(top_y_true, dtype=int),
            np.asarray(top_probabilities, dtype=float),
        ),
    ):
        observed, predicted = calibration_curve(labels, scores, n_bins=10, strategy="uniform")
        path = output / f"{split_label}_{prefix}_reliability_plot.png"
        fig, axis = plt.subplots(figsize=(6, 4))
        axis.plot([0, 1], [0, 1], color="gray", linestyle="--", label="ideal")
        axis.plot(predicted, observed, marker="o", label="model")
        axis.set(
            xlabel="Mean pair match score",
            ylabel="Observed match rate",
            title=title,
            xlim=(0, 1),
            ylim=(0, 1),
        )
        axis.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def save_stage_reports(
    reports_dir: str | Path,
    split_label: str,
    metrics: Mapping[str, Any],
    model_comparison: pd.DataFrame,
    threshold_diagnostics: Sequence[Mapping[str, Any]],
    listing_predictions: pd.DataFrame,
    error_examples: pd.DataFrame,
    pair_y_true: Sequence[int],
    pair_probabilities: Sequence[float],
    top_y_true: Sequence[int],
    top_probabilities: Sequence[float],
    retrieval_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Write the validation-only development report bundle."""
    _listing_duplicate_group_ids(listing_predictions)
    output = Path(reports_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics": output / "metrics.json",
        "model_comparison": output / "model_comparison.csv",
        "precision_coverage": output / "validation_precision_coverage.csv",
        "listing_predictions": output / f"{split_label}_listing_predictions.csv",
        "error_examples": output / f"{split_label}_error_examples.csv",
    }
    paths["metrics"].write_text(
        json.dumps(_json_ready(metrics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    model_comparison.to_csv(paths["model_comparison"], index=False)
    pd.DataFrame(threshold_diagnostics).to_csv(paths["precision_coverage"], index=False)
    listing_predictions.to_csv(paths["listing_predictions"], index=False)
    error_examples.to_csv(paths["error_examples"], index=False)
    plot_paths = save_evaluation_plots(
        pair_y_true,
        pair_probabilities,
        top_y_true,
        top_probabilities,
        retrieval_metrics,
        output,
        split_label,
    )
    return {**paths, "plots": plot_paths}


def save_corrected_resplit_reports(
    output_dir: str | Path,
    metrics: Mapping[str, Any],
    listing_predictions: pd.DataFrame,
    error_examples: pd.DataFrame,
    pair_y_true: Sequence[int],
    pair_probabilities: Sequence[float],
    top_y_true: Sequence[int],
    top_probabilities: Sequence[float],
    retrieval_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Write an additive corrected-resplit bundle in a fresh directory."""
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(
            f"Corrected-resplit output already exists: {output}. "
            "Choose a separate --output-dir for a deliberate rerun."
        )
    _listing_duplicate_group_ids(listing_predictions)
    output.mkdir(parents=True, exist_ok=False)
    paths = {
        "metrics": output / "metrics.json",
        "listing_predictions": output / "listing_predictions.csv",
        "error_examples": output / "error_examples.csv",
    }
    paths["metrics"].write_text(
        json.dumps(_json_ready(metrics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    listing_predictions.to_csv(paths["listing_predictions"], index=False)
    error_examples.to_csv(paths["error_examples"], index=False)
    plot_paths = save_evaluation_plots(
        pair_y_true,
        pair_probabilities,
        top_y_true,
        top_probabilities,
        retrieval_metrics,
        output,
        "corrected_resplit",
    )
    return {**paths, "plots": plot_paths}


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value
