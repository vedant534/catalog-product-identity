"""Run validation development or the separately authorized final holdout."""

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

from src.data import load_dataset, make_entity_splits
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
FINAL_TEST_STAGE = "final-test"


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
    listing_ids = set(
        assignments.loc[assignments["split"].eq(split_name), "google_id"].astype(str)
    )
    return google.loc[google["product_id"].astype(str).isin(listing_ids)].reset_index(
        drop=True
    )


def gold_for_listing_ids(gold: pd.DataFrame, listing_ids: Iterable[str]) -> pd.DataFrame:
    selected = {str(value) for value in listing_ids}
    return gold.loc[gold["google_id"].astype(str).isin(selected)].reset_index(drop=True)


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
        "auto_match": dict(selection["auto_match"]),
        "auto_no_match": dict(selection["auto_no_match"]),
        "both_constraints_met": bool(selection["both_constraints_met"]),
        "selection_mode": str(selection["selection_mode"]),
        "warning": selection["warning"],
    }


def _save_frozen_artifacts(
    output_dir: Path,
    config: Mapping[str, Any],
    assignments: pd.DataFrame,
    vectorizer: Any,
    amazon: pd.DataFrame,
    catalog_tfidf: sparse.spmatrix,
    catalog_embeddings: np.ndarray,
    matcher_bundle: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, output_dir / "tfidf_vectorizer.joblib")
    amazon.to_csv(output_dir / "amazon_catalog.csv", index=False)
    sparse.save_npz(output_dir / "catalog_tfidf.npz", catalog_tfidf)
    np.save(output_dir / "catalog_dense.npy", catalog_embeddings)
    joblib.dump(dict(matcher_bundle), output_dir / "matcher.joblib")
    assignments.to_csv(output_dir / "split_assignments.csv", index=False)
    snapshot = {
        "stage": DEVELOP_STAGE,
        "final_test_status": "not_run",
        "config": dict(config),
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
    required = {
        "snapshot": artifacts_dir / "run_config.json",
        "assignments": artifacts_dir / "split_assignments.csv",
        "vectorizer": artifacts_dir / "tfidf_vectorizer.joblib",
        "catalog": artifacts_dir / "amazon_catalog.csv",
        "tfidf": artifacts_dir / "catalog_tfidf.npz",
        "dense": artifacts_dir / "catalog_dense.npy",
        "matcher": artifacts_dir / "matcher.joblib",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing frozen development artifact(s): " + ", ".join(missing))
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
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    listing_ids = listings["product_id"].astype(str).tolist()
    pair_labels = label_candidates(candidates, gold)
    predictions = prediction_frame(candidates, features, probabilities)
    listing_predictions = build_listing_predictions(
        predictions, gold, listing_ids, policy
    )
    ranking = compute_ranking_metrics(predictions, gold, listing_ids)
    retrieval = compute_retrieval_metrics(
        candidates,
        diagnostic_pool,
        gold,
        ks=[int(value) for value in config["recall_ks"]],
        all_listing_ids=listing_ids,
        latency_seconds=latency_seconds,
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

    print("Loading benchmark and creating duplicate-aware assignments...")
    amazon, google, gold = load_dataset(
        raw_dir, download_if_missing=True, url=str(config["dataset_url"])
    )
    assignments = make_entity_splits(
        google,
        gold,
        seed=split_seed,
        ratios=tuple(float(value) for value in config["split_ratios"]),
    )
    # Intentionally materialize only train and validation. Test membership exists
    # only in assignments until the separately authorized final-test stage.
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
    encoder = load_sentence_encoder(str(config["sentence_model"]))
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

    policy_candidates = validation_predictions.loc[
        ~validation_predictions["google_id"].astype(str).isin(ambiguous_validation_ids)
    ].reset_index(drop=True)
    policy_top, match_correct, no_match_correct = _top_policy_labels(
        policy_candidates, validation_gold
    )
    threshold_selection = select_thresholds(
        match_correct,
        policy_top["probability"].to_numpy(dtype=float),
        match_precision_target=float(config["match_precision_target"]),
        no_match_precision_target=float(config["no_match_precision_target"]),
        grid_step=float(config["threshold_grid_step"]),
        no_match_correct=no_match_correct,
        min_auto_match_support=int(config["min_auto_match_support"]),
        min_auto_no_match_support=int(config["min_auto_no_match_support"]),
        warn=False,
    )
    policy = _action_policy_snapshot(threshold_selection)
    selected_bundle = {
        **selected_bundle,
        "selected_model": selected_name,
        "sentence_model_name": str(config["sentence_model"]),
        "top_k": top_k,
        "rrf_constant": rrf_constant,
        "split_seed": split_seed,
        "model_seed": model_seed,
        "policy": policy,
    }

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
    )
    injected = train_candidates["gold_injected"].fillna(False).astype(bool)
    metrics = {
        "stage": DEVELOP_STAGE,
        "final_test": {
            "status": "not_run",
            "test_listings_encoded": False,
            "test_candidates_retrieved": False,
            "test_features_constructed": False,
            "test_pairs_scored": False,
            "test_metrics_computed": False,
            "test_sensitivity_computed": False,
            "test_examples_inspected": False,
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
            "duplicate_groups": int(assignments["duplicate_group_id"].nunique()),
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
            "test": {
                "google_records": int(assignments["split"].eq("test").sum()),
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
    print("Final test status: NOT RUN")
    print(f"Validation reports written to {reports_dir.resolve()}")
    print(f"Frozen artifacts written to {artifacts_dir.resolve()}")
    return {"metrics": metrics, "reports": report_paths}


def run_final_test(config_path: str | Path) -> dict[str, Any]:
    """Evaluate the frozen test split without retraining or reselection."""
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
    if snapshot.get("final_test_status") != "not_run":
        raise ValueError("Frozen development snapshot is not awaiting final evaluation.")
    config = snapshot["config"]
    set_reproducible_seed(int(config["model_seed"]))
    configure_caches(config)
    _, google, gold = load_dataset(
        Path(str(config["raw_dir"])),
        download_if_missing=True,
        url=str(config["dataset_url"]),
    )
    test = listings_for_split(google, assignments, "test")
    test_gold = gold_for_listing_ids(gold, test["product_id"].astype(str))
    encoder = load_sentence_encoder(str(matcher["sentence_model_name"]))
    started = time.perf_counter()
    test_embeddings = encode_products(
        test, encoder, batch_size=int(config["dense_batch_size"])
    )
    test_candidates, test_pool = retrieve_candidates_with_diagnostics(
        test,
        amazon,
        vectorizer,
        catalog_tfidf,
        test_embeddings,
        catalog_embeddings,
        top_k=int(matcher["top_k"]),
        rrf_constant=float(matcher["rrf_constant"]),
    )
    latency = time.perf_counter() - started
    test_features = feature_candidates(test_candidates, test, amazon)
    probabilities = predict_probabilities(matcher, test_features)
    stage_metrics, listings, errors, pair_labels, top_labels = _stage_evaluation(
        split_label="test",
        candidates=test_candidates,
        diagnostic_pool=test_pool,
        features=test_features,
        probabilities=probabilities,
        listings=test,
        gold=test_gold,
        google=test,
        amazon=amazon,
        policy=matcher["policy"],
        config=config,
        latency_seconds=latency,
    )
    reports_dir = Path(str(config["reports_dir"]))
    metrics_path = reports_dir / "metrics.json"
    development_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    final_metrics = {
        **development_metrics,
        "stage": FINAL_TEST_STAGE,
        "final_test": {"status": "completed_once", **stage_metrics},
    }
    final_metrics["splits"] = {
        **development_metrics["splits"],
        "test": {
            **development_metrics["splits"]["test"],
            "gold_pairs": len(test_gold),
            "evaluation_status": "completed_once",
        },
    }
    comparison = pd.read_csv(reports_dir / "model_comparison.csv")
    diagnostics = pd.read_csv(reports_dir / "validation_precision_coverage.csv").to_dict(
        orient="records"
    )
    report_paths = save_stage_reports(
        reports_dir,
        "test",
        final_metrics,
        comparison,
        diagnostics,
        listings,
        errors,
        pair_labels,
        probabilities,
        top_labels,
        listings["probability"].to_numpy(dtype=float),
        stage_metrics["retrieval"],
    )
    snapshot["final_test_status"] = "completed_once"
    (artifacts_dir / "run_config.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("Final test evaluated from frozen development artifacts.")
    print(f"Final reports written to {reports_dir.resolve()}")
    return {"metrics": final_metrics, "reports": report_paths}


def run(stage: str, config_path: str | Path = "config.yaml") -> dict[str, Any]:
    if stage == DEVELOP_STAGE:
        return run_develop(config_path)
    if stage == FINAL_TEST_STAGE:
        return run_final_test(config_path)
    raise ValueError(f"Unknown stage: {stage}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=(DEVELOP_STAGE, FINAL_TEST_STAGE), required=True
    )
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    args = parser.parse_args()
    run(args.stage, args.config)


if __name__ == "__main__":
    main()
