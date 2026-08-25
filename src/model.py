"""Small calibrated logistic models and deterministic validation selection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from src.features import (
    DENSE_FEATURE_COLUMNS,
    HYBRID_FEATURE_COLUMNS,
    HYBRID_PLUS_RANK_FEATURE_COLUMNS,
    LEXICAL_FEATURE_COLUMNS,
)


def _feature_matrix(
    features: pd.DataFrame | np.ndarray,
    feature_columns: Sequence[str],
) -> np.ndarray:
    if isinstance(features, pd.DataFrame):
        missing = set(feature_columns) - set(features.columns)
        if missing:
            raise KeyError(f"Missing model features: {sorted(missing)}")
        matrix = features.loc[:, list(feature_columns)].to_numpy(dtype=float)
    else:
        matrix = np.asarray(features, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(feature_columns):
            raise ValueError(
                "Array feature width must equal the supplied feature-column count."
            )
    if np.any(~np.isfinite(matrix)):
        raise ValueError("Model features must be finite.")
    return matrix


def _labels_and_groups(
    y: Sequence[int] | np.ndarray,
    groups: Sequence[Any] | np.ndarray,
    expected_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y, dtype=int).reshape(-1)
    group_array = np.asarray(groups).reshape(-1)
    if labels.size != expected_size or group_array.size != expected_size:
        raise ValueError("Features, labels, and groups must have equal lengths.")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Training labels must contain both binary classes.")
    return labels, group_array


def grouped_calibration_splits(
    y: Sequence[int] | np.ndarray,
    groups: Sequence[Any] | np.ndarray,
    cv_folds: int = 3,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create class-valid calibration folds while keeping query groups intact."""
    labels = np.asarray(y, dtype=int).reshape(-1)
    group_array = np.asarray(groups).reshape(-1)
    if labels.size != group_array.size:
        raise ValueError("Labels and groups must have equal lengths.")

    maximum_folds = min(int(cv_folds), len(np.unique(group_array)))
    for folds in range(maximum_folds, 1, -1):
        splitter = StratifiedGroupKFold(
            n_splits=folds, shuffle=True, random_state=seed
        )
        splits = list(splitter.split(np.zeros(labels.size), labels, group_array))
        if all(
            len(np.unique(labels[train])) == 2
            and len(np.unique(labels[validation])) == 2
            for train, validation in splits
        ):
            return splits
    raise ValueError(
        "Could not form at least two grouped calibration folds containing both "
        "classes. Add more query groups or use a larger synthetic fixture."
    )


def fit_calibrated_logistic(
    features: pd.DataFrame | np.ndarray,
    y: Sequence[int] | np.ndarray,
    groups: Sequence[Any] | np.ndarray,
    feature_columns: Sequence[str],
    *,
    class_weight: str | Mapping[int, float] | None = None,
    seed: int = 42,
    cv_folds: int = 3,
    max_iter: int = 1_000,
    regularization_c: float = 1.0,
) -> CalibratedClassifierCV:
    """Fit sigmoid-calibrated logistic regression using grouped train folds."""
    columns = list(feature_columns)
    matrix = _feature_matrix(features, columns)
    labels, group_array = _labels_and_groups(y, groups, len(matrix))
    splits = grouped_calibration_splits(labels, group_array, cv_folds, seed)
    estimator = LogisticRegression(
        C=regularization_c,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=seed,
        solver="liblinear",
    )
    calibrated = CalibratedClassifierCV(
        estimator=estimator,
        method="sigmoid",
        cv=splits,
    )
    calibrated.fit(matrix, labels)
    return calibrated


def predict_probabilities(
    model_bundle: Mapping[str, Any] | CalibratedClassifierCV,
    features: pd.DataFrame | np.ndarray,
    feature_columns: Sequence[str] | None = None,
) -> np.ndarray:
    """Return positive-class probabilities from a model or a small model bundle."""
    if isinstance(model_bundle, Mapping):
        estimator = model_bundle["estimator"]
        columns = list(model_bundle["feature_columns"])
    else:
        estimator = model_bundle
        if feature_columns is None:
            raise ValueError("feature_columns are required for a bare estimator.")
        columns = list(feature_columns)
    matrix = _feature_matrix(features, columns)
    return np.asarray(estimator.predict_proba(matrix)[:, 1], dtype=float)


def fit_model_variants(
    train_features: pd.DataFrame,
    y_train: Sequence[int] | np.ndarray,
    query_groups: Sequence[Any] | np.ndarray,
    *,
    class_weight: str | Mapping[int, float] | None = None,
    seed: int = 42,
    cv_folds: int = 3,
    max_iter: int = 1_000,
    regularization_c: float = 1.0,
) -> dict[str, dict[str, Any]]:
    """Fit the four predeclared lightweight logistic variants."""
    variants: dict[str, dict[str, Any]] = {}
    for name, columns in (
        ("lexical_logistic", LEXICAL_FEATURE_COLUMNS),
        ("dense_logistic", DENSE_FEATURE_COLUMNS),
        ("hybrid_logistic", HYBRID_FEATURE_COLUMNS),
        ("hybrid_rank_logistic", HYBRID_PLUS_RANK_FEATURE_COLUMNS),
    ):
        estimator = fit_calibrated_logistic(
            train_features,
            y_train,
            query_groups,
            columns,
            class_weight=class_weight,
            seed=seed,
            cv_folds=cv_folds,
            max_iter=max_iter,
            regularization_c=regularization_c,
        )
        variants[name] = {
            "estimator": estimator,
            "feature_columns": list(columns),
            "class_weight": class_weight,
        }
    return variants


def top_candidate_indices(
    probabilities: Sequence[float] | np.ndarray,
    query_ids: Sequence[Any] | np.ndarray,
    candidate_ids: Sequence[Any] | np.ndarray,
) -> np.ndarray:
    """Return top-pair positions using score descending, then candidate ID."""
    scores = np.asarray(probabilities, dtype=float).reshape(-1)
    groups = np.asarray(query_ids).reshape(-1)
    candidates = np.asarray(candidate_ids).astype(str).reshape(-1)
    if not (scores.size == groups.size == candidates.size):
        raise ValueError("Probabilities, query IDs, and candidate IDs must align.")
    frame = pd.DataFrame(
        {
            "query_id": groups,
            "candidate_id": candidates,
            "probability": scores,
            "input_order": np.arange(scores.size),
        }
    )
    top = (
        frame.sort_values(
            ["query_id", "probability", "candidate_id", "input_order"],
            ascending=[True, False, True, True],
            kind="mergesort",
        )
        .drop_duplicates("query_id", keep="first")
        .sort_values("input_order", kind="mergesort")
    )
    return top["input_order"].to_numpy(dtype=int)


def top_candidate_rows(
    probabilities: Sequence[float] | np.ndarray,
    y_true: Sequence[int] | np.ndarray,
    query_ids: Sequence[Any] | np.ndarray,
    candidate_ids: Sequence[Any] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Select one highest-probability pair per query with the shared tie-break."""
    scores = np.asarray(probabilities, dtype=float).reshape(-1)
    labels = np.asarray(y_true, dtype=int).reshape(-1)
    groups = np.asarray(query_ids).reshape(-1)
    if not (scores.size == labels.size == groups.size):
        raise ValueError("Probabilities, labels, and query IDs must align.")

    indices = top_candidate_indices(scores, groups, candidate_ids)
    return labels[indices], scores[indices]


def select_top_candidates(candidate_predictions: pd.DataFrame) -> pd.DataFrame:
    """Return exactly one deterministic top-scoring row per represented listing."""
    required = {"google_id", "amazon_id", "probability"}
    missing = required - set(candidate_predictions.columns)
    if missing:
        raise ValueError(f"candidate_predictions missing columns: {sorted(missing)}")
    frame = candidate_predictions.copy()
    frame[["google_id", "amazon_id"]] = frame[["google_id", "amazon_id"]].astype(str)
    frame["probability"] = pd.to_numeric(frame["probability"], errors="raise")
    frame = frame.sort_values(
        ["google_id", "probability", "amazon_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    return frame.drop_duplicates("google_id", keep="first").reset_index(drop=True)


def compute_ranking_metrics(
    candidate_predictions: pd.DataFrame,
    gold_pairs: pd.DataFrame,
    all_listing_ids: Sequence[Any] | np.ndarray,
) -> dict[str, float | int]:
    """Compute listing ranking metrics with gold-bearing listings as denominator."""
    required = {"google_id", "amazon_id", "probability"}
    if not required.issubset(candidate_predictions.columns):
        raise ValueError(f"candidate_predictions must contain {sorted(required)}")
    if not {"google_id", "amazon_id"}.issubset(gold_pairs.columns):
        raise ValueError("gold_pairs must contain google_id and amazon_id")

    listing_ids = {str(value) for value in all_listing_ids}
    frame = candidate_predictions.copy()
    frame[["google_id", "amazon_id"]] = frame[["google_id", "amazon_id"]].astype(str)
    frame = frame.loc[frame["google_id"].isin(listing_ids)].copy()
    frame["probability"] = pd.to_numeric(frame["probability"], errors="raise")
    frame = frame.sort_values(
        ["google_id", "probability", "amazon_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    frame["model_rank"] = frame.groupby("google_id", sort=False).cumcount() + 1

    gold = gold_pairs[["google_id", "amazon_id"]].astype(str).drop_duplicates()
    gold = gold.loc[gold["google_id"].isin(listing_ids)]
    gold_set = set(gold.itertuples(index=False, name=None))
    gold_listing_ids = set(gold["google_id"])
    frame["is_gold_pair"] = [
        pair in gold_set
        for pair in frame[["google_id", "amazon_id"]].itertuples(index=False, name=None)
    ]

    top = frame.drop_duplicates("google_id", keep="first")
    hit_count = int(top["is_gold_pair"].sum())
    best_gold_rank = frame.loc[frame["is_gold_pair"]].groupby("google_id")[
        "model_rank"
    ].min()
    retrieved_count = int(best_gold_rank.size)
    gold_count = len(gold_listing_ids)
    reciprocal_rank_sum = float((1.0 / best_gold_rank).sum())

    return {
        "gold_listing_count": gold_count,
        "gold_retrieved_count": retrieved_count,
        "hit_at_1_count": hit_count,
        "overall_hit_at_1": float(hit_count / gold_count) if gold_count else 0.0,
        "conditional_hit_at_1": (
            float(hit_count / retrieved_count) if retrieved_count else 0.0
        ),
        "mrr": float(reciprocal_rank_sum / gold_count) if gold_count else 0.0,
        "retrieval_miss_count": gold_count - retrieved_count,
        "reranking_miss_count": retrieved_count - hit_count,
    }


def select_model(
    candidates: Mapping[str, Mapping[str, Any]],
    validation_features: pd.DataFrame,
    validation_candidates: pd.DataFrame,
    validation_gold: pd.DataFrame,
    all_validation_listing_ids: Sequence[Any] | np.ndarray,
) -> tuple[str, dict[str, Any], pd.DataFrame]:
    """Select by validation Hit@1, MRR, fewer features, then declared order."""
    if not candidates:
        raise ValueError("At least one model candidate is required.")
    if len(validation_features) != len(validation_candidates):
        raise ValueError("Validation features and candidates must align.")

    gold_set = set(
        validation_gold[["google_id", "amazon_id"]]
        .astype(str)
        .itertuples(index=False, name=None)
    )
    labels = np.asarray(
        [
            pair in gold_set
            for pair in validation_candidates[["google_id", "amazon_id"]]
            .astype(str)
            .itertuples(index=False, name=None)
        ],
        dtype=int,
    )

    diagnostics: list[dict[str, Any]] = []
    enriched: dict[str, dict[str, Any]] = {}
    declared_order = {name: index for index, name in enumerate(candidates)}
    for name, candidate in candidates.items():
        probabilities = predict_probabilities(candidate, validation_features)
        predictions = validation_candidates[["google_id", "amazon_id"]].copy()
        predictions["probability"] = probabilities
        ranking = compute_ranking_metrics(
            predictions,
            validation_gold,
            all_validation_listing_ids,
        )
        pr_auc = float(average_precision_score(labels, probabilities))
        roc_auc = (
            float(roc_auc_score(labels, probabilities))
            if np.unique(labels).size == 2
            else 0.0
        )
        bundle = dict(candidate)
        enriched[name] = bundle
        diagnostics.append(
            {
                "model": name,
                "validation_overall_hit_at_1": ranking["overall_hit_at_1"],
                "validation_conditional_hit_at_1": ranking["conditional_hit_at_1"],
                "validation_mrr": ranking["mrr"],
                "validation_pr_auc": pr_auc,
                "validation_roc_auc": roc_auc,
                "class_weight": str(candidate.get("class_weight")),
                "feature_count": len(candidate["feature_columns"]),
                "declared_order": declared_order[name],
            }
        )

    comparison = pd.DataFrame(diagnostics)
    ordered = comparison.sort_values(
        [
            "validation_overall_hit_at_1",
            "validation_mrr",
            "feature_count",
            "declared_order",
        ],
        ascending=[False, False, True, True],
        kind="mergesort",
    )
    selected_name = str(ordered.iloc[0]["model"])
    comparison["selected"] = comparison["model"] == selected_name
    comparison = comparison.drop(columns="declared_order").reset_index(drop=True)
    return selected_name, enriched[selected_name], comparison
