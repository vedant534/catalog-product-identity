"""Evaluation and report helpers for the catalog identity pipeline."""

from __future__ import annotations

import json
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


VALID_ACTIONS = {"auto_match", "auto_reject", "manual_review"}


def compute_retrieval_metrics(
    candidates: pd.DataFrame,
    gold_pairs: pd.DataFrame,
    ks: Sequence[int] = (5, 10, 20),
    all_listing_ids: Iterable[str] | None = None,
    latency_seconds: float | Sequence[float] | None = None,
) -> dict[str, float | int]:
    """Compute gold-bearing-listing recall at K, counts, and simple latency.

    A candidate is in the union top K when either its lexical rank or dense
    rank is at most K. A listing is recalled when any of its possible gold
    catalog partners is retrieved.
    """

    required_candidates = {"google_id", "amazon_id"}
    required_gold = {"google_id", "amazon_id"}
    if not required_candidates.issubset(candidates.columns):
        raise ValueError(f"candidates must contain {sorted(required_candidates)}")
    if not required_gold.issubset(gold_pairs.columns):
        raise ValueError(f"gold_pairs must contain {sorted(required_gold)}")

    candidate_pairs = candidates.copy()
    if "gold_injected" in candidate_pairs:
        candidate_pairs = candidate_pairs.loc[
            ~candidate_pairs["gold_injected"].fillna(False).astype(bool)
        ].copy()
    candidate_pairs[["google_id", "amazon_id"]] = candidate_pairs[
        ["google_id", "amazon_id"]
    ].astype(str)
    gold = gold_pairs[["google_id", "amazon_id"]].astype(str).drop_duplicates()
    gold_by_listing = gold.groupby("google_id")["amazon_id"].agg(set).to_dict()
    metrics: dict[str, float | int] = {}

    rank_columns = [
        column
        for column in ("lexical_rank", "dense_rank", "rank")
        if column in candidate_pairs.columns
    ]
    if not rank_columns:
        raise ValueError(
            "candidates must contain lexical_rank/dense_rank or a combined rank"
        )

    for k in ks:
        if k <= 0:
            raise ValueError("retrieval K values must be positive")
        channel_masks: dict[str, np.ndarray] = {}
        for column in rank_columns:
            ranks = pd.to_numeric(candidate_pairs[column], errors="coerce")
            channel = column.removesuffix("_rank")
            channel_masks[channel] = ranks.le(k).fillna(False).to_numpy()
        union_mask = np.logical_or.reduce(list(channel_masks.values()))

        def _listing_recall(mask: np.ndarray) -> float:
            retrieved = (
                candidate_pairs.loc[mask, ["google_id", "amazon_id"]]
                .groupby("google_id")["amazon_id"]
                .agg(set)
                .to_dict()
            )
            hits = sum(
                bool(gold_ids & retrieved.get(google_id, set()))
                for google_id, gold_ids in gold_by_listing.items()
            )
            return float(hits / len(gold_by_listing)) if gold_by_listing else 0.0

        union_recall = _listing_recall(union_mask)
        metrics[f"recall_at_{k}"] = union_recall
        metrics[f"union_recall_at_{k}"] = union_recall
        for channel in ("lexical", "dense"):
            if channel in channel_masks:
                metrics[f"{channel}_recall_at_{k}"] = _listing_recall(
                    channel_masks[channel]
                )

    if all_listing_ids is None:
        listing_ids = set(candidate_pairs["google_id"].astype(str))
    else:
        listing_ids = {str(value) for value in all_listing_ids}
    counts = candidate_pairs.groupby("google_id")["amazon_id"].nunique().to_dict()
    metrics["mean_candidate_count"] = (
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
            metrics["mean_latency_ms"] = mean_seconds * 1000.0
            if latency.size > 1:
                metrics["p95_latency_ms"] = float(
                    np.percentile(latency, 95) * 1000.0
                )
    return metrics


def compute_pair_metrics(
    y_true: Sequence[int],
    probabilities: Sequence[float],
    classification_threshold: float = 0.5,
) -> dict[str, float]:
    """Return discrimination, classification, and calibration metrics."""

    y = np.asarray(y_true, dtype=int)
    probability = np.asarray(probabilities, dtype=float)
    if y.shape != probability.shape or y.ndim != 1 or y.size == 0:
        raise ValueError("y_true and probabilities must be non-empty 1D arrays")
    predicted = probability >= classification_threshold
    metrics = {
        "pr_auc": float(average_precision_score(y, probability)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "brier_score": float(brier_score_loss(y, probability)),
    }
    metrics["roc_auc"] = (
        float(roc_auc_score(y, probability)) if np.unique(y).size == 2 else 0.0
    )
    return metrics


def compute_listing_metrics(
    candidate_predictions: pd.DataFrame,
    gold_pairs: pd.DataFrame,
    all_listing_ids: Iterable[str] | None = None,
) -> dict[str, float | int]:
    """Evaluate the action on each listing's highest-probability candidate.

    The end-to-end successful-match rate is the fraction of listings with a
    gold catalog match that are both ranked first and automatically matched.
    Gold retrieval failures remain in its denominator.
    """

    required_predictions = {"google_id", "amazon_id", "probability", "action"}
    if not required_predictions.issubset(candidate_predictions.columns):
        raise ValueError(
            f"candidate_predictions must contain {sorted(required_predictions)}"
        )
    if not {"google_id", "amazon_id"}.issubset(gold_pairs.columns):
        raise ValueError("gold_pairs must contain google_id and amazon_id")

    predictions = candidate_predictions.copy()
    predictions[["google_id", "amazon_id"]] = predictions[
        ["google_id", "amazon_id"]
    ].astype(str)
    unknown_actions = set(predictions["action"].dropna().unique()) - VALID_ACTIONS
    if unknown_actions:
        raise ValueError(f"unknown policy actions: {sorted(unknown_actions)}")

    predictions["probability"] = pd.to_numeric(
        predictions["probability"], errors="coerce"
    ).fillna(0.0)
    predictions = predictions.sort_values(
        ["google_id", "probability", "amazon_id"],
        ascending=[True, False, True],
    )
    best = predictions.drop_duplicates("google_id", keep="first").copy()

    gold = gold_pairs[["google_id", "amazon_id"]].astype(str).drop_duplicates()
    gold_set = set(gold.itertuples(index=False, name=None))
    gold_by_listing = gold.groupby("google_id")["amazon_id"].agg(set).to_dict()

    if all_listing_ids is None:
        listing_ids = set(best["google_id"]) | set(gold["google_id"])
    else:
        listing_ids = {str(value) for value in all_listing_ids}

    best = best.set_index("google_id").reindex(sorted(listing_ids)).reset_index()
    best["action"] = best["action"].fillna("manual_review")
    best["has_gold_listing"] = best["google_id"].isin(gold_by_listing)
    best["is_gold_pair"] = [
        (google_id, amazon_id) in gold_set
        for google_id, amazon_id in zip(best["google_id"], best["amazon_id"])
    ]

    retrieved_pairs = set(
        predictions[["google_id", "amazon_id"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    gold_retrieved = {
        google_id: any(
            (google_id, amazon_id) in retrieved_pairs
            for amazon_id in amazon_ids
        )
        for google_id, amazon_ids in gold_by_listing.items()
    }

    total = len(best)
    auto_match = best["action"].eq("auto_match")
    auto_reject = best["action"].eq("auto_reject")
    manual_review = best["action"].eq("manual_review")
    correct_match = auto_match & best["is_gold_pair"]
    correct_reject = auto_reject & ~best["has_gold_listing"]
    automatically_decided = auto_match | auto_reject
    matched_listing_count = len(gold_by_listing)

    def _rate(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else 0.0

    metrics: dict[str, float | int] = {
        "auto_match_precision": _rate(int(correct_match.sum()), int(auto_match.sum())),
        "auto_match_coverage": _rate(int(auto_match.sum()), total),
        "auto_reject_precision": _rate(int(correct_reject.sum()), int(auto_reject.sum())),
        "auto_reject_coverage": _rate(int(auto_reject.sum()), total),
        "manual_review_rate": _rate(int(manual_review.sum()), total),
        "accuracy_on_auto_decisions": _rate(
            int((correct_match | correct_reject).sum()),
            int(automatically_decided.sum()),
        ),
        "end_to_end_successful_match_rate": _rate(
            int(correct_match.sum()), matched_listing_count
        ),
        "matched_listing_count": matched_listing_count,
        "gold_retrieval_failure_count": int(
            sum(not was_retrieved for was_retrieved in gold_retrieved.values())
        ),
        "listing_count": total,
    }
    return metrics


def model_comparison_rows(
    y_true: Sequence[int],
    predictions_by_model: Mapping[str, Sequence[float]],
    split: str = "validation",
    classification_threshold: float = 0.5,
    operational_metrics_by_model: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build rows that can be passed directly to ``pandas.DataFrame``."""

    rows: list[dict[str, Any]] = []
    for model_name, probabilities in predictions_by_model.items():
        row: dict[str, Any] = {"model": model_name, "split": split}
        row.update(
            compute_pair_metrics(y_true, probabilities, classification_threshold)
        )
        if operational_metrics_by_model and model_name in operational_metrics_by_model:
            row.update(operational_metrics_by_model[model_name])
        rows.append(row)
    return rows


def extract_error_examples(
    candidate_predictions: pd.DataFrame,
    gold_pairs: pd.DataFrame,
    max_per_type: int = 10,
    classification_threshold: float = 0.5,
    google_records: pd.DataFrame | None = None,
    amazon_records: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Extract compact, ranked examples for qualitative error analysis."""

    required = {"google_id", "amazon_id", "probability"}
    if not required.issubset(candidate_predictions.columns):
        raise ValueError(f"candidate_predictions must contain {sorted(required)}")
    if max_per_type <= 0:
        raise ValueError("max_per_type must be positive")

    frame = candidate_predictions.copy()
    frame[["google_id", "amazon_id"]] = frame[
        ["google_id", "amazon_id"]
    ].astype(str)
    frame["probability"] = pd.to_numeric(frame["probability"], errors="coerce")
    gold = gold_pairs[["google_id", "amazon_id"]].astype(str).drop_duplicates()
    gold_set = set(gold.itertuples(index=False, name=None))
    frame["is_gold_pair"] = [
        pair in gold_set
        for pair in frame[["google_id", "amazon_id"]].itertuples(
            index=False, name=None
        )
    ]
    frame["predicted_match"] = frame["probability"].ge(classification_threshold)
    wrong = frame[frame["predicted_match"] != frame["is_gold_pair"]].copy()
    examples: list[pd.DataFrame] = []

    false_positive = wrong[~wrong["is_gold_pair"]].nlargest(
        max_per_type, "probability"
    )
    if not false_positive.empty:
        false_positive = false_positive.copy()
        false_positive.insert(0, "error_type", "high_confidence_false_positive")
        examples.append(false_positive)

    false_negative = wrong[wrong["is_gold_pair"]].nsmallest(
        max_per_type, "probability"
    )
    if not false_negative.empty:
        false_negative = false_negative.copy()
        false_negative.insert(0, "error_type", "high_confidence_false_negative")
        examples.append(false_negative)

    conflict_column = next(
        (
            column
            for column in ("numeric_token_conflict", "numeric_conflict")
            if column in wrong.columns
        ),
        None,
    )
    if conflict_column:
        numeric_conflicts = wrong[wrong[conflict_column].fillna(0).astype(bool)]
        numeric_conflicts = numeric_conflicts.nlargest(max_per_type, "probability")
        if not numeric_conflicts.empty:
            numeric_conflicts = numeric_conflicts.copy()
            numeric_conflicts.insert(0, "error_type", "numeric_model_conflict")
            examples.append(numeric_conflicts)

    if "title_fuzzy_similarity" in wrong:
        similar_variant_mask = (
            ~wrong["is_gold_pair"]
            & wrong["title_fuzzy_similarity"].fillna(0).ge(0.80)
        )
        variant_signals = pd.Series(False, index=wrong.index)
        if conflict_column:
            variant_signals |= wrong[conflict_column].fillna(0).astype(bool)
        if "relative_price_difference" in wrong:
            variant_signals |= wrong["relative_price_difference"].fillna(0).ge(0.20)
        likely_variants = wrong[similar_variant_mask & variant_signals].nlargest(
            max_per_type, "probability"
        )
        if not likely_variants.empty:
            likely_variants = likely_variants.copy()
            likely_variants.insert(0, "error_type", "likely_similar_variant")
            examples.append(likely_variants)

    missing_columns = [
        column
        for column in (
            "manufacturer_missing",
            "missing_manufacturer",
            "query_manufacturer_missing",
            "candidate_manufacturer_missing",
            "price_missing",
            "missing_price",
            "query_price_missing",
            "candidate_price_missing",
        )
        if column in wrong.columns
    ]
    if missing_columns:
        missing_mask = wrong[missing_columns].fillna(0).astype(bool).any(axis=1)
        missing = wrong[missing_mask].nlargest(max_per_type, "probability")
        if not missing.empty:
            missing = missing.copy()
            missing.insert(0, "error_type", "missing_manufacturer_or_price")
            examples.append(missing)

    retrieved_by_listing = (
        frame.groupby("google_id")["amazon_id"].agg(set).to_dict()
    )
    gold_by_listing = gold.groupby("google_id")["amazon_id"].agg(set).to_dict()
    missed_listing_ids = {
        google_id
        for google_id, amazon_ids in gold_by_listing.items()
        if not (amazon_ids & retrieved_by_listing.get(google_id, set()))
    }
    missed_gold = (
        gold.loc[gold["google_id"].isin(missed_listing_ids)]
        .drop_duplicates("google_id")
        .head(max_per_type)
    )
    if not missed_gold.empty:
        missed_gold = missed_gold.copy()
        missed_gold.insert(0, "error_type", "gold_match_missed_by_retrieval")
        missed_gold["probability"] = np.nan
        missed_gold["is_gold_pair"] = True
        missed_gold["predicted_match"] = False
        examples.append(missed_gold)

    if not examples:
        result = pd.DataFrame(
            columns=[
                "error_type",
                "google_id",
                "amazon_id",
                "probability",
                "is_gold_pair",
                "predicted_match",
            ]
        )
    else:
        result = pd.concat(examples, ignore_index=True, sort=False)
    result = _merge_product_context(result, google_records, "google_id", "google")
    return _merge_product_context(result, amazon_records, "amazon_id", "amazon")


def _merge_product_context(
    examples: pd.DataFrame,
    records: pd.DataFrame | None,
    pair_id_column: str,
    prefix: str,
) -> pd.DataFrame:
    if records is None:
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
    y_true: Sequence[int],
    probabilities: Sequence[float],
    retrieval_metrics: Mapping[str, Any],
    output_dir: str | Path,
) -> list[Path]:
    """Save retrieval recall, PR, and reliability plots."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y_true, dtype=int)
    probability = np.asarray(probabilities, dtype=float)
    paths: list[Path] = []

    recall_keys = list(retrieval_metrics)
    for channel in ("lexical", "dense", "union"):
        channel_metrics = retrieval_metrics.get(channel)
        if isinstance(channel_metrics, Mapping):
            recall_keys.extend(channel_metrics)
    recall_ks = sorted(
        {
            int(str(key).rsplit("_", 1)[-1])
            for key in recall_keys
            if "recall" in str(key) and str(key).rsplit("_", 1)[-1].isdigit()
        }
    )
    if not recall_ks:
        recall_ks = [5, 10, 20]
    path = output / "retrieval_recall.png"
    fig, axis = plt.subplots(figsize=(7, 4))
    positions = np.arange(len(recall_ks), dtype=float)
    width = 0.25
    for offset, (channel, label) in enumerate(
        (("lexical", "Lexical"), ("dense", "Dense"), ("union", "Union"))
    ):
        values = []
        for k in recall_ks:
            key = f"{channel}_recall_at_{k}"
            fallback = f"recall_at_{k}" if channel == "union" else key
            channel_metrics = retrieval_metrics.get(channel, {})
            nested = (
                channel_metrics.get(f"recall_at_{k}")
                if isinstance(channel_metrics, Mapping)
                else None
            )
            value = retrieval_metrics.get(
                key, retrieval_metrics.get(fallback, nested if nested is not None else 0.0)
            )
            values.append(float(value))
        axis.bar(positions + (offset - 1) * width, values, width, label=label)
    axis.set(
        xticks=positions,
        xticklabels=[f"Recall@{k}" for k in recall_ks],
        ylabel="Gold-bearing listings retrieved",
        title="Test candidate retrieval",
        ylim=(0, 1.05),
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    precision, recall, _ = precision_recall_curve(y, probability)
    path = output / "precision_recall_curve.png"
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.plot(recall, precision)
    axis.axhline(y.mean(), color="gray", linestyle="--", label="prevalence")
    axis.set(xlabel="Recall", ylabel="Precision", title="Test precision-recall curve")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    observed, predicted = calibration_curve(y, probability, n_bins=10, strategy="uniform")
    path = output / "reliability_plot.png"
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.plot([0, 1], [0, 1], color="gray", linestyle="--", label="ideal")
    axis.plot(predicted, observed, marker="o", label="model")
    axis.set(
        xlabel="Mean predicted probability",
        ylabel="Observed match rate",
        title="Test reliability",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    return paths


def save_evaluation_reports(
    reports_dir: str | Path,
    retrieval: Mapping[str, Any],
    pair_matching: Mapping[str, Any],
    abstention: Mapping[str, Any],
    model_comparison: Sequence[Mapping[str, Any]] | pd.DataFrame,
    error_examples: pd.DataFrame,
    y_true: Sequence[int],
    probabilities: Sequence[float],
    match_threshold: float,
    reject_threshold: float,
) -> dict[str, Any]:
    """Write the compact metrics, CSV, and plot bundle used by the README."""

    output = Path(reports_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.json"
    comparison_path = output / "model_comparison.csv"
    errors_path = output / "error_examples.csv"

    metrics = {
        "retrieval": dict(retrieval),
        "pair_matching": dict(pair_matching),
        "abstention": dict(abstention),
        "thresholds": {
            "match_threshold": float(match_threshold),
            "reject_threshold": float(reject_threshold),
        },
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(metrics), handle, indent=2, sort_keys=True)
        handle.write("\n")

    comparison = (
        model_comparison.copy()
        if isinstance(model_comparison, pd.DataFrame)
        else pd.DataFrame(model_comparison)
    )
    comparison.to_csv(comparison_path, index=False)
    error_examples.to_csv(errors_path, index=False)
    plot_paths = save_evaluation_plots(
        y_true,
        probabilities,
        retrieval,
        output,
    )
    return {
        "metrics": metrics_path,
        "model_comparison": comparison_path,
        "error_examples": errors_path,
        "plots": plot_paths,
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    return value
