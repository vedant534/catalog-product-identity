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

from src.features import HYBRID_FEATURE_COLUMNS, LEXICAL_FEATURE_COLUMNS
from src.policy import select_thresholds


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
    """Fit the two allowed learned variants: lexical and final hybrid."""
    variants: dict[str, dict[str, Any]] = {}
    for name, columns in (
        ("lexical_logistic", LEXICAL_FEATURE_COLUMNS),
        ("hybrid_logistic", HYBRID_FEATURE_COLUMNS),
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


def fit_balanced_variant(
    train_features: pd.DataFrame,
    y_train: Sequence[int] | np.ndarray,
    query_groups: Sequence[Any] | np.ndarray,
    feature_columns: Sequence[str],
    *,
    seed: int = 42,
    cv_folds: int = 3,
    max_iter: int = 1_000,
    regularization_c: float = 1.0,
) -> dict[str, Any]:
    """Fit one balanced alternative after validation shows it is warranted."""
    estimator = fit_calibrated_logistic(
        train_features,
        y_train,
        query_groups,
        feature_columns,
        class_weight="balanced",
        seed=seed,
        cv_folds=cv_folds,
        max_iter=max_iter,
        regularization_c=regularization_c,
    )
    return {
        "estimator": estimator,
        "feature_columns": list(feature_columns),
        "class_weight": "balanced",
    }


def top_candidate_indices(
    probabilities: Sequence[float] | np.ndarray,
    query_ids: Sequence[Any] | np.ndarray,
) -> np.ndarray:
    """Return stable input positions for each query's highest-scoring pair."""
    scores = np.asarray(probabilities, dtype=float).reshape(-1)
    groups = np.asarray(query_ids).reshape(-1)
    if scores.size != groups.size:
        raise ValueError("Probabilities and query IDs must align.")
    frame = pd.DataFrame(
        {
            "query_id": groups,
            "probability": scores,
            "input_order": np.arange(scores.size),
        }
    )
    top = (
        frame.sort_values(
            ["query_id", "probability", "input_order"],
            ascending=[True, False, True],
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
) -> tuple[np.ndarray, np.ndarray]:
    """Select one highest-probability pair per query, stably on input order."""
    scores = np.asarray(probabilities, dtype=float).reshape(-1)
    labels = np.asarray(y_true, dtype=int).reshape(-1)
    groups = np.asarray(query_ids).reshape(-1)
    if not (scores.size == labels.size == groups.size):
        raise ValueError("Probabilities, labels, and query IDs must align.")

    indices = top_candidate_indices(scores, groups)
    return labels[indices], scores[indices]


def _top_reject_correct(
    top_indices: np.ndarray,
    query_ids: np.ndarray,
    *,
    listing_has_gold: Sequence[int] | np.ndarray | Mapping[Any, bool] | None,
    reject_correct: Sequence[int] | np.ndarray | Mapping[Any, bool] | None,
) -> np.ndarray | None:
    if listing_has_gold is not None and reject_correct is not None:
        raise ValueError(
            "Pass validation_listing_has_gold or validation_reject_correct, not both."
        )
    supplied = reject_correct if reject_correct is not None else listing_has_gold
    if supplied is None:
        return None

    if isinstance(supplied, Mapping):
        try:
            values = np.asarray(
                [bool(supplied[query_id]) for query_id in query_ids[top_indices]],
                dtype=int,
            )
        except KeyError as error:
            raise KeyError(f"Missing listing policy label for query {error.args[0]!r}.")
    else:
        aligned = np.asarray(supplied, dtype=int).reshape(-1)
        if aligned.size != query_ids.size:
            raise ValueError(
                "Listing policy labels must align with validation candidate rows."
            )
        values = aligned[top_indices]

    if not set(np.unique(values)).issubset({0, 1}):
        raise ValueError("Listing policy labels must contain only 0/1 values.")
    return 1 - values if listing_has_gold is not None else values


def select_model(
    candidates: Mapping[str, Mapping[str, Any]],
    validation_features: pd.DataFrame,
    y_validation: Sequence[int] | np.ndarray,
    validation_query_ids: Sequence[Any] | np.ndarray,
    *,
    validation_listing_has_gold: (
        Sequence[int] | np.ndarray | Mapping[Any, bool] | None
    ) = None,
    validation_reject_correct: (
        Sequence[int] | np.ndarray | Mapping[Any, bool] | None
    ) = None,
    match_precision_target: float = 0.95,
    reject_precision_target: float = 0.95,
    threshold_grid_step: float = 0.01,
) -> tuple[str, dict[str, Any], pd.DataFrame]:
    """Select a deployable variant using validation ranking and policy metrics.

    Models satisfying both precision targets rank ahead of infeasible models;
    within that set validation pair PR-AUC is primary, followed by automatic
    coverage and then fewer features. If no model is feasible, the same ranking
    selects the highest-PR-AUC model with its documented fallback thresholds.
    """
    labels = np.asarray(y_validation, dtype=int).reshape(-1)
    query_ids = np.asarray(validation_query_ids).reshape(-1)
    if labels.size != query_ids.size:
        raise ValueError("Validation labels and query IDs must align.")
    if not candidates:
        raise ValueError("At least one model candidate is required.")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Validation labels must contain both classes.")

    diagnostics: list[dict[str, Any]] = []
    enriched: dict[str, dict[str, Any]] = {}
    for name, candidate in candidates.items():
        probabilities = predict_probabilities(candidate, validation_features)
        top_indices = top_candidate_indices(probabilities, query_ids)
        top_labels = labels[top_indices]
        top_probabilities = probabilities[top_indices]
        reject_correct = _top_reject_correct(
            top_indices,
            query_ids,
            listing_has_gold=validation_listing_has_gold,
            reject_correct=validation_reject_correct,
        )
        thresholds = select_thresholds(
            top_labels,
            top_probabilities,
            match_precision_target=match_precision_target,
            reject_precision_target=reject_precision_target,
            grid_step=threshold_grid_step,
            reject_correct=reject_correct,
            warn=False,
        )
        pr_auc = float(average_precision_score(labels, probabilities))
        roc_auc = float(roc_auc_score(labels, probabilities))
        bundle = dict(candidate)
        bundle["thresholds"] = {
            "match_threshold": thresholds["match_threshold"],
            "reject_threshold": thresholds["reject_threshold"],
        }
        enriched[name] = bundle
        diagnostics.append(
            {
                "model": name,
                "validation_pr_auc": pr_auc,
                "validation_roc_auc": roc_auc,
                "constraints_met": bool(thresholds["constraints_met"]),
                "automatic_coverage": thresholds["automatic_coverage"],
                "auto_match_precision": thresholds["auto_match_precision"],
                "auto_reject_precision": thresholds["auto_reject_precision"],
                "match_threshold": thresholds["match_threshold"],
                "reject_threshold": thresholds["reject_threshold"],
                "precision_shortfall": thresholds["precision_shortfall"],
                "class_weight": str(candidate.get("class_weight")),
                "feature_count": len(candidate["feature_columns"]),
                "warning": thresholds["warning"],
            }
        )

    comparison = pd.DataFrame(diagnostics)
    feasible_exists = bool(comparison["constraints_met"].any())
    pool = comparison[comparison["constraints_met"]] if feasible_exists else comparison
    ordered = pool.sort_values(
        ["validation_pr_auc", "automatic_coverage", "feature_count", "model"],
        ascending=[False, False, True, True],
        kind="mergesort",
    )
    selected_name = str(ordered.iloc[0]["model"])
    comparison["selected"] = comparison["model"] == selected_name
    comparison = comparison.sort_values("model", kind="mergesort").reset_index(drop=True)
    return selected_name, enriched[selected_name], comparison


def should_try_balanced(model_comparison: pd.DataFrame) -> bool:
    """Return true only when no unweighted validation candidate meets policy."""
    if "constraints_met" not in model_comparison:
        raise KeyError("model_comparison needs a constraints_met column.")
    return not bool(model_comparison["constraints_met"].any())


train_matcher = fit_calibrated_logistic
