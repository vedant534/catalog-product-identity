"""Run development or the separately authorized corrected-resplit evaluation."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy import sparse

from src.data import (
    canonicalize_split_assignments,
    load_dataset,
    make_entity_splits,
    official_raw_csv_sha256,
    split_assignment_metadata,
    validate_assignment_ids,
    validate_exact_split_assignments,
    validate_split_assignments,
)
from src.evaluate import (
    build_listing_predictions,
    compute_calibration_metrics,
    compute_listing_metrics,
    compute_no_match_metrics,
    compute_pair_metrics,
    compute_retrieval_metrics,
    compute_sensitivity_metrics,
    exact_title_ambiguity_ids,
    extract_error_examples,
    save_corrected_resplit_reports,
    save_stage_reports,
)
from src.features import build_pair_features
from src.model import (
    compute_ranking_metrics,
    fit_model_variants,
    predict_probabilities,
    select_model,
    select_top_candidates,
)
from src.policy import select_thresholds
from src.retrieval import (
    encode_products,
    fit_lexical_retriever,
    load_sentence_encoder,
    retrieve_candidates_with_diagnostics,
)


DEVELOP_STAGE = "develop"
CORRECTED_EVAL_STAGE = "corrected-eval"
CORRECTED_RESPLIT_PROTOCOL = "predeclared_corrected_resplit"
CORRECTED_ASSIGNMENT_SPLIT = "test"

DEVELOPMENT_REPORT_FILENAMES = (
    "metrics.json",
    "model_comparison.csv",
    "validation_precision_coverage.csv",
    "validation_listing_predictions.csv",
    "validation_error_examples.csv",
    "validation_retrieval_recall.png",
    "validation_pair_precision_recall_curve.png",
    "validation_pair_reliability_plot.png",
    "validation_top_candidate_reliability_plot.png",
)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("config.yaml must contain a mapping")
    return config


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def configure_caches(config: Mapping[str, Any]) -> None:
    model_cache = Path(str(config["model_cache_dir"])).resolve()
    model_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(model_cache))
    matplotlib_cache = model_cache.parent / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def listings_for_split(
    google: pd.DataFrame,
    assignments: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    """Materialize only the explicitly requested split."""
    validate_assignment_ids(google, assignments)
    listing_ids = set(
        assignments.loc[assignments["split"].eq(split_name), "google_id"].astype(str)
    )
    return google.loc[google["product_id"].astype(str).isin(listing_ids)].reset_index(
        drop=True
    )


def _development_assignments(
    artifacts_dir: Path,
    google: pd.DataFrame,
    gold: pd.DataFrame,
    *,
    seed: int,
    ratios: tuple[float, float, float],
) -> pd.DataFrame:
    """Regenerate and validate the seed-defined corrected-resplit assignments."""
    regenerated = validate_split_assignments(
        google,
        make_entity_splits(google, gold, seed=seed, ratios=ratios),
    )
    path = artifacts_dir / "split_assignments.csv"
    if path.exists():
        existing = pd.read_csv(
            path,
            dtype={"google_id": str, "component_id": str, "duplicate_group_id": str},
        )
        existing = validate_split_assignments(google, existing)
        validate_exact_split_assignments(regenerated, existing)
    return regenerated


def gold_for_listing_ids(gold: pd.DataFrame, listing_ids: Iterable[str]) -> pd.DataFrame:
    selected = {str(value) for value in listing_ids}
    return gold.loc[gold["google_id"].astype(str).isin(selected)].reset_index(drop=True)


def _duplicate_groups_for_listing_ids(
    assignments: pd.DataFrame,
    listing_ids: Iterable[str],
) -> dict[str, str]:
    selected_ids = {str(value) for value in listing_ids}
    selected = assignments.loc[
        assignments["google_id"].astype(str).isin(selected_ids),
        ["google_id", "duplicate_group_id"],
    ].copy()
    return dict(selected.astype(str).itertuples(index=False, name=None))


def label_candidates(candidates: pd.DataFrame, gold: pd.DataFrame) -> np.ndarray:
    gold_pairs = set(
        gold[["google_id", "amazon_id"]]
        .astype(str)
        .itertuples(index=False, name=None)
    )
    return np.asarray(
        [
            pair in gold_pairs
            for pair in candidates[["google_id", "amazon_id"]]
            .astype(str)
            .itertuples(index=False, name=None)
        ],
        dtype=int,
    )


def feature_candidates(
    candidates: pd.DataFrame,
    google: pd.DataFrame,
    amazon: pd.DataFrame,
) -> pd.DataFrame:
    return build_pair_features(
        candidates,
        google,
        amazon,
        lexical_scores=candidates["lexical_score"].to_numpy(),
        dense_scores=candidates["dense_score"].to_numpy(),
        record_id_column="product_id",
    )


def prediction_frame(
    candidates: pd.DataFrame,
    features: pd.DataFrame,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    frame = pd.concat(
        [candidates.reset_index(drop=True), features.reset_index(drop=True)], axis=1
    )
    frame["probability"] = probabilities
    return frame


def _score_baselines(
    candidates: pd.DataFrame,
    gold: pd.DataFrame,
    listing_ids: Iterable[str],
) -> dict[str, Any]:
    results: dict[str, Any] = {
        "definition": (
            "Score-only reranking diagnostics over the shared fixed-budget "
            "RRF candidate set; these are not standalone channel retrieval results."
        )
    }
    for name, column in (
        ("lexical_score", "lexical_score"),
        ("dense_score", "dense_score"),
        ("rrf_score", "rrf_score"),
    ):
        frame = candidates[["google_id", "amazon_id"]].copy()
        frame["probability"] = candidates[column].to_numpy(dtype=float)
        results[name] = compute_ranking_metrics(frame, gold, list(listing_ids))
    return results


def _top_policy_labels(
    candidates: pd.DataFrame,
    gold: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    top = select_top_candidates(candidates)
    gold_set = set(
        gold[["google_id", "amazon_id"]]
        .astype(str)
        .itertuples(index=False, name=None)
    )
    gold_listing_ids = set(gold["google_id"].astype(str))
    match_correct = np.asarray(
        [
            pair in gold_set
            for pair in top[["google_id", "amazon_id"]]
            .astype(str)
            .itertuples(index=False, name=None)
        ],
        dtype=int,
    )
    no_match_correct = np.asarray(
        [google_id not in gold_listing_ids for google_id in top["google_id"].astype(str)],
        dtype=int,
    )
    return top, match_correct, no_match_correct


def _action_policy_snapshot(selection: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "selection_evidence_unit": "unique_duplicate_group_id",
        "selection_group_count": int(selection["n_groups"]),
        "listing_count": int(selection["n_total"]),
        "auto_match": dict(selection["auto_match"]),
        "auto_no_match": dict(selection["auto_no_match"]),
        "both_constraints_met": bool(selection["both_constraints_met"]),
        "selection_mode": str(selection["selection_mode"]),
        "warning": selection["warning"],
    }


def _frozen_artifact_paths(artifacts_dir: Path) -> dict[str, Path]:
    return {
        "snapshot": artifacts_dir / "run_config.json",
        "assignments": artifacts_dir / "split_assignments.csv",
        "vectorizer": artifacts_dir / "tfidf_vectorizer.joblib",
        "catalog": artifacts_dir / "amazon_catalog.csv",
        "tfidf": artifacts_dir / "catalog_tfidf.npz",
        "dense": artifacts_dir / "catalog_dense.npy",
        "matcher": artifacts_dir / "matcher.joblib",
    }


def _development_report_paths(reports_dir: Path) -> dict[str, Path]:
    return {
        filename: reports_dir / filename for filename in DEVELOPMENT_REPORT_FILENAMES
    }


def _require_nonempty_files(paths: Mapping[str, Path], label: str) -> None:
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {label}(s): " + ", ".join(missing))
    empty = [str(path) for path in paths.values() if path.stat().st_size == 0]
    if empty:
        raise ValueError(f"Empty {label}(s): " + ", ".join(empty))


def _frozen_inputs(
    config: Mapping[str, Any],
    raw_csv_sha256: Mapping[str, str],
    assignments: pd.DataFrame,
    amazon: pd.DataFrame,
    catalog_tfidf: sparse.spmatrix,
    catalog_embeddings: np.ndarray,
) -> dict[str, Any]:
    return {
        "raw_csv_sha256": dict(raw_csv_sha256),
        "sentence_encoder": {
            "model_name": str(config["sentence_model"]),
            "revision": str(config["sentence_model_revision"]),
        },
        "split_assignments": split_assignment_metadata(assignments),
        "catalog": {
            "row_count": int(len(amazon)),
            "tfidf_shape": [int(value) for value in catalog_tfidf.shape],
            "dense_shape": [int(value) for value in catalog_embeddings.shape],
        },
    }


def _save_frozen_artifacts(
    output_dir: Path,
    config: Mapping[str, Any],
    frozen_inputs: Mapping[str, Any],
    assignments: pd.DataFrame,
    vectorizer: Any,
    amazon: pd.DataFrame,
    catalog_tfidf: sparse.spmatrix,
    catalog_embeddings: np.ndarray,
    matcher_bundle: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_assignments = canonicalize_split_assignments(assignments)
    joblib.dump(vectorizer, output_dir / "tfidf_vectorizer.joblib")
    amazon.to_csv(output_dir / "amazon_catalog.csv", index=False)
    sparse.save_npz(output_dir / "catalog_tfidf.npz", catalog_tfidf)
    np.save(output_dir / "catalog_dense.npy", catalog_embeddings)
    joblib.dump(dict(matcher_bundle), output_dir / "matcher.joblib")
    canonical_assignments.to_csv(
        output_dir / "split_assignments.csv",
        index=False,
        lineterminator="\n",
    )
    snapshot = {
        "stage": DEVELOP_STAGE,
        "evaluation_protocol": CORRECTED_RESPLIT_PROTOCOL,
        "corrected_resplit_status_at_development": "not_run",
        "config": dict(config),
        "frozen_inputs": dict(frozen_inputs),
        "split_assignments": dict(frozen_inputs["split_assignments"]),
        "selected_model": matcher_bundle["selected_model"],
        "feature_columns": list(matcher_bundle["feature_columns"]),
        "policy": matcher_bundle["policy"],
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _load_frozen_artifacts(
    artifacts_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame, Any, pd.DataFrame, Any, np.ndarray, Any]:
    required = _frozen_artifact_paths(artifacts_dir)
    _require_nonempty_files(required, "frozen development artifact")
    snapshot = json.loads(required["snapshot"].read_text(encoding="utf-8"))
    assignments = pd.read_csv(
        required["assignments"],
        dtype={"google_id": str, "component_id": str, "duplicate_group_id": str},
    )
    return (
        snapshot,
        assignments,
        joblib.load(required["vectorizer"]),
        pd.read_csv(required["catalog"], dtype={"product_id": str}),
        sparse.load_npz(required["tfidf"]),
        np.load(required["dense"]),
        joblib.load(required["matcher"]),
    )


def _validated_split_assignment_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Frozen snapshot is missing split-assignment metadata.")
    sha256 = str(value.get("sha256", "")).lower()
    if len(sha256) != 64 or any(
        character not in "0123456789abcdef" for character in sha256
    ):
        raise ValueError("Frozen split-assignment SHA-256 metadata is invalid.")

    row_count_value = value.get("row_count")
    if isinstance(row_count_value, (bool, np.bool_)):
        raise ValueError("Frozen split-assignment row count is invalid.")
    try:
        row_count = int(row_count_value)
    except (TypeError, ValueError) as error:
        raise ValueError("Frozen split-assignment row count is invalid.") from error
    if row_count < 0 or row_count != row_count_value:
        raise ValueError("Frozen split-assignment row count is invalid.")

    raw_counts = value.get("rows_by_split")
    expected_splits = {"train", "validation", CORRECTED_ASSIGNMENT_SPLIT}
    if not isinstance(raw_counts, Mapping) or set(raw_counts) != expected_splits:
        raise ValueError("Frozen split-assignment split counts are invalid.")
    rows_by_split: dict[str, int] = {}
    for split_name in ("train", "validation", CORRECTED_ASSIGNMENT_SPLIT):
        count_value = raw_counts[split_name]
        if isinstance(count_value, (bool, np.bool_)):
            raise ValueError("Frozen split-assignment split counts are invalid.")
        try:
            count = int(count_value)
        except (TypeError, ValueError) as error:
            raise ValueError("Frozen split-assignment split counts are invalid.") from error
        if count < 0 or count != count_value:
            raise ValueError("Frozen split-assignment split counts are invalid.")
        rows_by_split[split_name] = count
    if sum(rows_by_split.values()) != row_count:
        raise ValueError("Frozen split-assignment counts do not sum to its row count.")
    return {
        "sha256": sha256,
        "row_count": row_count,
        "rows_by_split": rows_by_split,
    }


def _validate_corrected_snapshot(
    current_config: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    matcher: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if snapshot.get("stage") != DEVELOP_STAGE:
        raise ValueError("Frozen snapshot must come from the development stage.")
    if snapshot.get("evaluation_protocol") != CORRECTED_RESPLIT_PROTOCOL:
        raise ValueError("Frozen snapshot has an incompatible evaluation protocol.")
    config = snapshot.get("config")
    frozen_inputs = snapshot.get("frozen_inputs")
    if not isinstance(config, dict) or not isinstance(frozen_inputs, dict):
        raise ValueError("Frozen snapshot is missing configuration or input metadata.")

    for name, converter in (
        ("split_seed", int),
        ("model_seed", int),
        ("top_k", int),
        ("rrf_constant", float),
    ):
        try:
            current_value = converter(current_config[name])
            snapshot_value = converter(config[name])
            matcher_value = converter(matcher[name])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Frozen {name} metadata is missing or invalid.") from error
        if current_value != snapshot_value:
            raise ValueError(f"Current and frozen configurations disagree on {name}.")
        if matcher_value != snapshot_value:
            raise ValueError(f"Frozen matcher and snapshot disagree on {name}.")

    try:
        current_ratios = tuple(float(value) for value in current_config["split_ratios"])
        snapshot_ratios = tuple(float(value) for value in config["split_ratios"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Frozen split-ratio metadata is missing or invalid.") from error
    if current_ratios != snapshot_ratios:
        raise ValueError("Current and frozen configurations disagree on split_ratios.")

    snapshot_features = snapshot.get("feature_columns")
    matcher_features = matcher.get("feature_columns")
    if (
        not isinstance(snapshot_features, (list, tuple))
        or not isinstance(matcher_features, (list, tuple))
        or list(snapshot_features) != list(matcher_features)
    ):
        raise ValueError(
            "Frozen matcher and snapshot disagree on feature column order."
        )
    if "feature_columns" in current_config:
        current_features = current_config["feature_columns"]
        if not isinstance(current_features, (list, tuple)) or list(
            current_features
        ) != list(snapshot_features):
            raise ValueError(
                "Current configuration and frozen snapshot disagree on feature "
                "column order."
            )

    snapshot_selected_model = snapshot.get("selected_model")
    matcher_selected_model = matcher.get("selected_model")
    if (
        not isinstance(snapshot_selected_model, str)
        or not snapshot_selected_model
        or matcher_selected_model != snapshot_selected_model
    ):
        raise ValueError("Frozen matcher and snapshot disagree on the selected model.")
    if (
        "selected_model" in current_config
        and current_config["selected_model"] != snapshot_selected_model
    ):
        raise ValueError(
            "Current configuration and frozen snapshot disagree on the selected model."
        )

    snapshot_assignment_metadata = _validated_split_assignment_metadata(
        snapshot.get("split_assignments")
    )
    frozen_assignment_metadata = _validated_split_assignment_metadata(
        frozen_inputs.get("split_assignments")
    )
    if snapshot_assignment_metadata != frozen_assignment_metadata:
        raise ValueError(
            "Frozen snapshot split-assignment metadata is internally inconsistent."
        )

    encoder = frozen_inputs.get("sentence_encoder")
    if not isinstance(encoder, dict):
        raise ValueError("Frozen snapshot is missing sentence encoder metadata.")
    expected_name = str(config.get("sentence_model", ""))
    expected_revision = str(config.get("sentence_model_revision", ""))
    if not expected_name or not expected_revision:
        raise ValueError("Frozen snapshot must pin the sentence encoder and revision.")
    if encoder.get("model_name") != expected_name or encoder.get("revision") != expected_revision:
        raise ValueError("Frozen sentence encoder metadata does not match configuration.")
    if (
        str(current_config.get("sentence_model", "")) != expected_name
        or str(current_config.get("sentence_model_revision", ""))
        != expected_revision
    ):
        raise ValueError(
            "Current and frozen configurations disagree on sentence encoder metadata."
        )
    if (
        matcher.get("sentence_model_name") != expected_name
        or matcher.get("sentence_model_revision") != expected_revision
    ):
        raise ValueError("Frozen matcher uses incompatible sentence encoder metadata.")
    if matcher.get("policy") != snapshot.get("policy"):
        raise ValueError("Frozen matcher and snapshot disagree on the policy.")
    if snapshot.get("corrected_resplit_status_at_development") != "not_run":
        raise ValueError(
            "Frozen development snapshot is not awaiting corrected evaluation."
        )
    return config, frozen_inputs


def _validate_frozen_split_assignments(
    frozen_inputs: Mapping[str, Any],
    assignments: pd.DataFrame,
    google: pd.DataFrame,
    gold: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Validate artifact metadata, invariants, and deterministic membership."""
    expected_metadata = _validated_split_assignment_metadata(
        frozen_inputs.get("split_assignments")
    )
    canonical_artifact = validate_split_assignments(google, assignments)
    actual_metadata = split_assignment_metadata(canonical_artifact)
    if actual_metadata != expected_metadata:
        raise ValueError(
            "Frozen split-assignment artifact does not match snapshot metadata."
        )

    try:
        split_seed = int(config["split_seed"])
        split_ratios = tuple(float(value) for value in config["split_ratios"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Frozen split configuration is missing or invalid.") from error
    regenerated = validate_split_assignments(
        google,
        make_entity_splits(
            google,
            gold,
            seed=split_seed,
            ratios=split_ratios,
        ),
    )
    validate_exact_split_assignments(regenerated, canonical_artifact)
    return canonical_artifact


def _validate_frozen_catalog(
    frozen_inputs: Mapping[str, Any],
    vectorizer: Any,
    amazon: pd.DataFrame,
    catalog_tfidf: sparse.spmatrix,
    catalog_embeddings: np.ndarray,
) -> None:
    metadata = frozen_inputs.get("catalog")
    if not isinstance(metadata, Mapping):
        raise ValueError("Frozen snapshot is missing catalog dimension metadata.")
    if getattr(catalog_tfidf, "ndim", None) != 2:
        raise ValueError("Frozen TF-IDF catalog matrix must be two-dimensional.")
    if np.asarray(catalog_embeddings).ndim != 2:
        raise ValueError("Frozen dense catalog matrix must be two-dimensional.")

    try:
        expected_rows = int(metadata["row_count"])
        expected_tfidf_shape = tuple(int(value) for value in metadata["tfidf_shape"])
        expected_dense_shape = tuple(int(value) for value in metadata["dense_shape"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Frozen catalog dimension metadata is invalid.") from error
    if len(expected_tfidf_shape) != 2 or len(expected_dense_shape) != 2:
        raise ValueError("Frozen catalog matrix shapes must have two dimensions.")

    actual_tfidf_shape = tuple(int(value) for value in catalog_tfidf.shape)
    actual_dense_shape = tuple(int(value) for value in catalog_embeddings.shape)
    if len(amazon) != expected_rows:
        raise ValueError("Frozen catalog row count does not match the development snapshot.")
    if actual_tfidf_shape != expected_tfidf_shape:
        raise ValueError("Frozen TF-IDF matrix shape does not match the development snapshot.")
    if actual_dense_shape != expected_dense_shape:
        raise ValueError("Frozen dense matrix shape does not match the development snapshot.")
    if expected_tfidf_shape[0] != expected_rows or expected_dense_shape[0] != expected_rows:
        raise ValueError("Frozen catalog matrices must have one row per catalog product.")
    if "product_id" not in amazon or not amazon["product_id"].astype(str).is_unique:
        raise ValueError("Frozen catalog product IDs must be present and unique.")

    vectorizer_width = int(vectorizer.transform([""]).shape[1])
    if vectorizer_width != expected_tfidf_shape[1]:
        raise ValueError("Frozen vectorizer and TF-IDF matrix feature counts differ.")


def _verify_raw_csv_digests(
    raw_dir: Path,
    frozen_inputs: Mapping[str, Any],
) -> None:
    expected = frozen_inputs.get("raw_csv_sha256")
    if not isinstance(expected, Mapping) or not expected:
        raise ValueError("Frozen snapshot is missing raw CSV SHA-256 digests.")
    actual = official_raw_csv_sha256(raw_dir)
    expected_digests = {str(name): str(digest) for name, digest in expected.items()}
    if actual != expected_digests:
        mismatched = sorted(set(actual) | set(expected_digests))
        mismatched = [
            name for name in mismatched if actual.get(name) != expected_digests.get(name)
        ]
        raise ValueError(
            "Official raw CSV SHA-256 mismatch: " + ", ".join(mismatched)
        )


def _stage_evaluation(
    *,
    split_label: str,
    candidates: pd.DataFrame,
    diagnostic_pool: pd.DataFrame,
    features: pd.DataFrame,
    probabilities: np.ndarray,
    listings: pd.DataFrame,
    gold: pd.DataFrame,
    google: pd.DataFrame,
    amazon: pd.DataFrame,
    policy: Mapping[str, Any],
    config: Mapping[str, Any],
    latency_seconds: float | np.ndarray,
    duplicate_group_by_listing: Mapping[str, str],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    listing_ids = listings["product_id"].astype(str).tolist()
    pair_labels = label_candidates(candidates, gold)
    predictions = prediction_frame(candidates, features, probabilities)
    listing_predictions = build_listing_predictions(
        predictions,
        gold,
        listing_ids,
        policy,
        duplicate_group_by_listing=duplicate_group_by_listing,
    )
    ranking = compute_ranking_metrics(predictions, gold, listing_ids)
    retrieval = compute_retrieval_metrics(
        candidates,
        diagnostic_pool,
        gold,
        ks=[int(value) for value in config["recall_ks"]],
        all_listing_ids=listing_ids,
        latency_seconds=latency_seconds,
        per_channel_candidate_depth=min(int(config["top_k"]), len(amazon)),
    )
    retrieval["primary_retriever"] = "fixed_budget_rrf"
    retrieval["rrf_constant"] = float(config["rrf_constant"])
    retrieval["candidate_budget"] = int(config["top_k"])
    retrieval["union_per_channel_definition"] = (
        "Diagnostic only: lexical top-K union dense top-K, up to 2K rows."
    )
    ambiguous_title_ids = exact_title_ambiguity_ids(
        google, amazon, gold, listing_ids
    )
    top_labels = listing_predictions["is_gold_pair"].astype(int).to_numpy()
    top_scores = listing_predictions["probability"].to_numpy(dtype=float)
    metrics = {
        "retrieval": retrieval,
        "common_rrf_pool_score_ranking_diagnostics": _score_baselines(
            candidates, gold, listing_ids
        ),
        "selected_matcher_ranking": ranking,
        "pair_matching": compute_pair_metrics(pair_labels, probabilities),
        "calibration": {
            "pair_level": compute_calibration_metrics(pair_labels, probabilities),
            "top_candidate": compute_calibration_metrics(top_labels, top_scores),
        },
        "listing_policy": compute_listing_metrics(listing_predictions),
        "no_match_detection": compute_no_match_metrics(listing_predictions),
        "exact_title_ambiguity_sensitivity": compute_sensitivity_metrics(
            listing_predictions, ambiguous_title_ids
        ),
    }
    errors = extract_error_examples(
        predictions,
        listing_predictions,
        gold,
        max_per_type=int(config["error_examples_per_type"]),
        google_records=google,
        amazon_records=amazon,
    )
    return metrics, listing_predictions, errors, pair_labels, top_labels


def run_develop(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    split_seed = int(config["split_seed"])
    model_seed = int(config["model_seed"])
    set_reproducible_seed(model_seed)
    configure_caches(config)

    raw_dir = Path(str(config["raw_dir"]))
    artifacts_dir = Path(str(config["artifacts_dir"]))
    reports_dir = Path(str(config["reports_dir"]))
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("Loading benchmark and regenerating duplicate-aware assignments...")
    amazon, google, gold = load_dataset(
        raw_dir, download_if_missing=True, url=str(config["dataset_url"])
    )
    raw_csv_sha256 = official_raw_csv_sha256(raw_dir)
    assignments = _development_assignments(
        artifacts_dir,
        google,
        gold,
        seed=split_seed,
        ratios=tuple(float(value) for value in config["split_ratios"]),
    )
    # Intentionally materialize only train and validation. Corrected-resplit
    # membership remains only in assignments until separately authorized.
    train = listings_for_split(google, assignments, "train")
    validation = listings_for_split(google, assignments, "validation")
    train_gold = gold_for_listing_ids(gold, train["product_id"].astype(str))
    validation_gold = gold_for_listing_ids(gold, validation["product_id"].astype(str))

    print("Fitting retrieval on the fixed catalog plus training text only...")
    vectorizer, catalog_tfidf = fit_lexical_retriever(
        train,
        amazon,
        ngram_range=tuple(int(value) for value in config["tfidf_ngram_range"]),
        min_df=int(config["tfidf_min_df"]),
        max_features=int(config["tfidf_max_features"]),
    )
    encoder = load_sentence_encoder(
        str(config["sentence_model"]),
        str(config["sentence_model_revision"]),
    )
    batch_size = int(config["dense_batch_size"])
    catalog_embeddings = encode_products(amazon, encoder, batch_size=batch_size)
    train_embeddings = encode_products(train, encoder, batch_size=batch_size)
    top_k = int(config["top_k"])
    rrf_constant = float(config["rrf_constant"])
    train_candidates, _ = retrieve_candidates_with_diagnostics(
        train,
        amazon,
        vectorizer,
        catalog_tfidf,
        train_embeddings,
        catalog_embeddings,
        top_k=top_k,
        gold_matches=train_gold,
        inject_gold=True,
        rrf_constant=rrf_constant,
    )
    started = time.perf_counter()
    validation_embeddings = encode_products(validation, encoder, batch_size=batch_size)
    validation_candidates, validation_pool = retrieve_candidates_with_diagnostics(
        validation,
        amazon,
        vectorizer,
        catalog_tfidf,
        validation_embeddings,
        catalog_embeddings,
        top_k=top_k,
        rrf_constant=rrf_constant,
    )
    validation_latency = time.perf_counter() - started

    train_labels = label_candidates(train_candidates, train_gold)
    train_features = feature_candidates(train_candidates, train, amazon)
    validation_features = feature_candidates(
        validation_candidates, validation, amazon
    )
    ambiguous_train_ids = set(
        assignments.loc[
            assignments["split"].eq("train") & assignments["ambiguous_label"].astype(bool),
            "google_id",
        ].astype(str)
    )
    train_keep = ~train_candidates["google_id"].astype(str).isin(ambiguous_train_ids)
    component_by_google = assignments.set_index("google_id")["component_id"]
    training_groups = train_candidates.loc[train_keep, "google_id"].map(component_by_google)

    print("Fitting four predeclared logistic variants...")
    variants = fit_model_variants(
        train_features.loc[train_keep].reset_index(drop=True),
        train_labels[train_keep.to_numpy()],
        training_groups.to_numpy(),
        seed=model_seed,
        cv_folds=int(config["calibration_folds"]),
        max_iter=int(config["logreg_max_iter"]),
    )
    ambiguous_validation_ids = set(
        assignments.loc[
            assignments["split"].eq("validation")
            & assignments["ambiguous_label"].astype(bool),
            "google_id",
        ].astype(str)
    )
    validation_selection_mask = ~validation_candidates["google_id"].astype(str).isin(
        ambiguous_validation_ids
    )
    validation_selection_ids = [
        value
        for value in validation["product_id"].astype(str)
        if value not in ambiguous_validation_ids
    ]
    selected_name, selected_bundle, comparison = select_model(
        variants,
        validation_features.loc[validation_selection_mask].reset_index(drop=True),
        validation_candidates.loc[validation_selection_mask].reset_index(drop=True),
        validation_gold,
        validation_selection_ids,
    )
    validation_probabilities = predict_probabilities(selected_bundle, validation_features)
    validation_predictions = validation_candidates[["google_id", "amazon_id"]].copy()
    validation_predictions["probability"] = validation_probabilities

    ambiguous_validation_groups = set(
        assignments.loc[
            assignments["split"].eq("validation")
            & assignments["ambiguous_label"].astype(bool),
            "duplicate_group_id",
        ].astype(str)
    )
    policy_excluded_ids = set(
        assignments.loc[
            assignments["split"].eq("validation")
            & assignments["duplicate_group_id"].astype(str).isin(
                ambiguous_validation_groups
            ),
            "google_id",
        ].astype(str)
    )
    policy_candidates = validation_predictions.loc[
        ~validation_predictions["google_id"].astype(str).isin(policy_excluded_ids)
    ].reset_index(drop=True)
    policy_top, match_correct, no_match_correct = _top_policy_labels(
        policy_candidates, validation_gold
    )
    duplicate_group_by_google = assignments.set_index("google_id")[
        "duplicate_group_id"
    ]
    policy_duplicate_group_ids = policy_top["google_id"].map(
        duplicate_group_by_google
    )
    if policy_duplicate_group_ids.isna().any():
        raise ValueError("Every policy-selection listing must have a duplicate group.")
    threshold_selection = select_thresholds(
        match_correct,
        policy_top["probability"].to_numpy(dtype=float),
        match_precision_target=float(config["match_precision_target"]),
        no_match_precision_target=float(config["no_match_precision_target"]),
        grid_step=float(config["threshold_grid_step"]),
        no_match_correct=no_match_correct,
        duplicate_group_ids=policy_duplicate_group_ids.to_numpy(dtype=str),
        min_auto_match_support=int(config["min_auto_match_support"]),
        min_auto_no_match_support=int(config["min_auto_no_match_support"]),
        warn=False,
    )
    policy = _action_policy_snapshot(threshold_selection)
    selected_bundle = {
        **selected_bundle,
        "selected_model": selected_name,
        "sentence_model_name": str(config["sentence_model"]),
        "sentence_model_revision": str(config["sentence_model_revision"]),
        "top_k": top_k,
        "rrf_constant": rrf_constant,
        "split_seed": split_seed,
        "model_seed": model_seed,
        "policy": policy,
    }
    frozen_inputs = _frozen_inputs(
        config,
        raw_csv_sha256,
        assignments,
        amazon,
        catalog_tfidf,
        catalog_embeddings,
    )

    stage_metrics, listing_predictions, errors, pair_labels, top_labels = _stage_evaluation(
        split_label="validation",
        candidates=validation_candidates,
        diagnostic_pool=validation_pool,
        features=validation_features,
        probabilities=validation_probabilities,
        listings=validation,
        gold=validation_gold,
        google=validation,
        amazon=amazon,
        policy=policy,
        config=config,
        latency_seconds=validation_latency,
        duplicate_group_by_listing=_duplicate_groups_for_listing_ids(
            assignments,
            validation["product_id"].astype(str),
        ),
    )
    injected = train_candidates["gold_injected"].fillna(False).astype(bool)
    metrics = {
        "stage": DEVELOP_STAGE,
        "evaluation_protocol": CORRECTED_RESPLIT_PROTOCOL,
        "frozen_inputs": frozen_inputs,
        "split_assignments": frozen_inputs["split_assignments"],
        "corrected_resplit": {
            "status": "not_run",
            "reserved_listings_encoded": False,
            "reserved_candidates_retrieved": False,
            "reserved_features_constructed": False,
            "reserved_pairs_scored": False,
            "reserved_metrics_computed": False,
            "reserved_sensitivity_computed": False,
            "reserved_examples_inspected": False,
        },
        "run": {
            "split_seed": split_seed,
            "model_seed": model_seed,
            "split_ratios": [float(value) for value in config["split_ratios"]],
            "top_k": top_k,
            "rrf_constant": rrf_constant,
            "recall_ks": [int(value) for value in config["recall_ks"]],
        },
        "dataset": {
            "amazon_records": len(amazon),
            "google_records": len(google),
            "gold_pairs": len(gold),
            "signature_groups_total": int(assignments["duplicate_group_id"].nunique()),
            "exact_duplicate_groups": int(
                assignments.loc[assignments["duplicate_group_size"].gt(1), "duplicate_group_id"].nunique()
            ),
            "exact_duplicate_rows": int(assignments["duplicate_group_size"].gt(1).sum()),
            "ambiguous_unmapped_rows": int(assignments["ambiguous_label"].astype(bool).sum()),
        },
        "splits": {
            "train": {"google_records": len(train), "gold_pairs": len(train_gold)},
            "validation": {
                "google_records": len(validation),
                "gold_pairs": len(validation_gold),
            },
            "corrected_resplit": {
                "google_records": int(
                    assignments["split"].eq(CORRECTED_ASSIGNMENT_SPLIT).sum()
                ),
                "evaluation_status": "not_run",
            },
        },
        "training": {
            "ambiguous_negative_listings_excluded": len(ambiguous_train_ids),
            "gold_injected_positive_rows": int(injected.sum()),
            "gold_injected_listings": int(
                train_candidates.loc[injected, "google_id"].nunique()
            ),
        },
        "model_selection": {
            "basis": ["validation_overall_hit_at_1", "validation_mrr", "feature_count", "declared_order"],
            "selected": selected_name,
            "feature_columns": list(selected_bundle["feature_columns"]),
            "ambiguous_validation_listings_excluded": len(ambiguous_validation_ids),
        },
        "validation": {
            **stage_metrics,
            "policy_selection": policy,
        },
    }
    _save_frozen_artifacts(
        artifacts_dir,
        config,
        frozen_inputs,
        assignments,
        vectorizer,
        amazon,
        catalog_tfidf,
        catalog_embeddings,
        selected_bundle,
    )
    report_paths = save_stage_reports(
        reports_dir,
        "validation",
        metrics,
        comparison,
        threshold_selection["threshold_diagnostics"],
        listing_predictions,
        errors,
        pair_labels,
        validation_probabilities,
        top_labels,
        listing_predictions["probability"].to_numpy(dtype=float),
        stage_metrics["retrieval"],
    )
    print(f"Selected model: {selected_name}")
    print(f"Policy mode: {policy['selection_mode']}")
    print("Corrected-resplit evaluation status: NOT RUN")
    print(f"Validation reports written to {reports_dir.resolve()}")
    print(f"Frozen artifacts written to {artifacts_dir.resolve()}")
    return {"metrics": metrics, "reports": report_paths}


def run_corrected_eval(
    config_path: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate the predeclared corrected resplit without retraining."""
    locator_config = load_config(config_path)
    artifacts_dir = Path(str(locator_config["artifacts_dir"]))
    (
        snapshot,
        assignments,
        vectorizer,
        amazon,
        catalog_tfidf,
        catalog_embeddings,
        matcher,
    ) = _load_frozen_artifacts(artifacts_dir)
    config, frozen_inputs = _validate_corrected_snapshot(
        locator_config,
        snapshot,
        matcher,
    )

    reports_dir = Path(str(config["reports_dir"]))
    development_reports = _development_report_paths(reports_dir)
    _require_nonempty_files(development_reports, "development report")
    development_metrics = json.loads(
        development_reports["metrics.json"].read_text(encoding="utf-8")
    )
    if (
        development_metrics.get("stage") != DEVELOP_STAGE
        or development_metrics.get("evaluation_protocol")
        != CORRECTED_RESPLIT_PROTOCOL
    ):
        raise ValueError("Development metrics use an incompatible evaluation protocol.")
    if development_metrics.get("frozen_inputs") != frozen_inputs:
        raise ValueError("Development metrics and frozen snapshot disagree on inputs.")
    if _validated_split_assignment_metadata(
        development_metrics.get("split_assignments")
    ) != _validated_split_assignment_metadata(frozen_inputs.get("split_assignments")):
        raise ValueError(
            "Development metrics and frozen snapshot disagree on split assignments."
        )

    corrected_output = (
        Path(output_dir)
        if output_dir is not None
        else reports_dir / "corrected_resplit"
    )
    if corrected_output.exists():
        raise FileExistsError(
            "Corrected-resplit output already exists; choose a new --output-dir "
            "for a deliberate rerun: "
            + str(corrected_output)
        )

    _validate_frozen_catalog(
        frozen_inputs,
        vectorizer,
        amazon,
        catalog_tfidf,
        catalog_embeddings,
    )
    raw_dir = Path(str(config["raw_dir"]))
    _verify_raw_csv_digests(raw_dir, frozen_inputs)
    set_reproducible_seed(int(config["model_seed"]))
    configure_caches(config)
    source_amazon, google, gold = load_dataset(
        raw_dir,
        download_if_missing=False,
        url=str(config["dataset_url"]),
    )
    if len(source_amazon) != len(amazon):
        raise ValueError("Normalized raw catalog row count differs from frozen catalog.")
    assignments = _validate_frozen_split_assignments(
        frozen_inputs,
        assignments,
        google,
        gold,
        config,
    )
    evaluation = listings_for_split(google, assignments, CORRECTED_ASSIGNMENT_SPLIT)
    evaluation_gold = gold_for_listing_ids(
        gold, evaluation["product_id"].astype(str)
    )
    encoder = load_sentence_encoder(
        str(matcher["sentence_model_name"]),
        str(matcher["sentence_model_revision"]),
        local_files_only=True,
    )
    started = time.perf_counter()
    evaluation_embeddings = encode_products(
        evaluation, encoder, batch_size=int(config["dense_batch_size"])
    )
    if (
        evaluation_embeddings.ndim != 2
        or evaluation_embeddings.shape[1] != catalog_embeddings.shape[1]
    ):
        raise ValueError("Corrected-resplit embeddings do not match frozen catalog dimensions.")
    evaluation_candidates, evaluation_pool = retrieve_candidates_with_diagnostics(
        evaluation,
        amazon,
        vectorizer,
        catalog_tfidf,
        evaluation_embeddings,
        catalog_embeddings,
        top_k=int(matcher["top_k"]),
        rrf_constant=float(matcher["rrf_constant"]),
    )
    latency = time.perf_counter() - started
    evaluation_features = feature_candidates(evaluation_candidates, evaluation, amazon)
    probabilities = predict_probabilities(matcher, evaluation_features)
    stage_metrics, listings, errors, pair_labels, top_labels = _stage_evaluation(
        split_label="corrected_resplit",
        candidates=evaluation_candidates,
        diagnostic_pool=evaluation_pool,
        features=evaluation_features,
        probabilities=probabilities,
        listings=evaluation,
        gold=evaluation_gold,
        google=evaluation,
        amazon=amazon,
        policy=matcher["policy"],
        config=config,
        latency_seconds=latency,
        duplicate_group_by_listing=_duplicate_groups_for_listing_ids(
            assignments,
            evaluation["product_id"].astype(str),
        ),
    )
    corrected_metrics = {
        "stage": CORRECTED_EVAL_STAGE,
        "evaluation_protocol": CORRECTED_RESPLIT_PROTOCOL,
        "evidence_role": "transparent_secondary_confirmation",
        "frozen_inputs": frozen_inputs,
        "split_assignments": frozen_inputs["split_assignments"],
        "run": {
            "split_seed": int(config["split_seed"]),
            "model_seed": int(config["model_seed"]),
            "top_k": int(matcher["top_k"]),
            "rrf_constant": float(matcher["rrf_constant"]),
            "recall_ks": [int(value) for value in config["recall_ks"]],
        },
        "development_checkpoint": {
            "selected_model": str(matcher["selected_model"]),
            "feature_columns": list(matcher["feature_columns"]),
            "policy": matcher["policy"],
        },
        "split": {
            "assignment_label": CORRECTED_ASSIGNMENT_SPLIT,
            "google_records": len(evaluation),
            "gold_pairs": len(evaluation_gold),
        },
        "corrected_resplit": {"status": "completed", **stage_metrics},
    }
    report_paths = save_corrected_resplit_reports(
        corrected_output,
        corrected_metrics,
        listings,
        errors,
        pair_labels,
        probabilities,
        top_labels,
        listings["probability"].to_numpy(dtype=float),
        stage_metrics["retrieval"],
    )
    print("Corrected resplit evaluated from frozen development artifacts.")
    print(f"Corrected-resplit reports written to {corrected_output.resolve()}")
    return {"metrics": corrected_metrics, "reports": report_paths}


def run(
    stage: str,
    config_path: str | Path = "config.yaml",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    if stage == DEVELOP_STAGE:
        if output_dir is not None:
            raise ValueError("--output-dir is valid only for corrected-eval.")
        return run_develop(config_path)
    if stage == CORRECTED_EVAL_STAGE:
        return run_corrected_eval(config_path, output_dir=output_dir)
    raise ValueError(f"Unknown stage: {stage}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=(DEVELOP_STAGE, CORRECTED_EVAL_STAGE), required=True
    )
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    parser.add_argument(
        "--output-dir",
        help=(
            "Fresh corrected-resplit bundle directory; valid only for corrected-eval. "
            "Defaults to reports/corrected_resplit from the frozen configuration."
        ),
    )
    args = parser.parse_args()
    if args.output_dir is not None and args.stage != CORRECTED_EVAL_STAGE:
        parser.error("--output-dir is valid only for corrected-eval")
    run(args.stage, args.config, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
