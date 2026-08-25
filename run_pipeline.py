"""Run the complete Amazon-Google catalog identity experiment."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data import load_dataset, make_entity_splits, split_google_listings
from src.evaluate import (
    compute_listing_metrics,
    compute_pair_metrics,
    compute_retrieval_metrics,
    extract_error_examples,
    save_evaluation_reports,
)
from src.features import HYBRID_FEATURE_COLUMNS, build_pair_features
from src.model import (
    fit_balanced_variant,
    fit_model_variants,
    predict_probabilities,
    select_model,
    should_try_balanced,
    top_candidate_indices,
)
from src.policy import apply_policy, select_thresholds
from src.retrieval import (
    encode_products,
    fit_lexical_retriever,
    load_sentence_encoder,
    retrieve_candidates,
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


def gold_for_listings(gold: pd.DataFrame, listings: pd.DataFrame) -> pd.DataFrame:
    listing_ids = set(listings["product_id"].astype(str))
    return gold.loc[gold["google_id"].astype(str).isin(listing_ids)].reset_index(
        drop=True
    )


def label_candidates(candidates: pd.DataFrame, gold: pd.DataFrame) -> np.ndarray:
    gold_pairs = set(
        gold[["google_id", "amazon_id"]]
        .astype(str)
        .itertuples(index=False, name=None)
    )
    return np.asarray(
        [
            (str(google_id), str(amazon_id)) in gold_pairs
            for google_id, amazon_id in candidates[
                ["google_id", "amazon_id"]
            ].itertuples(index=False, name=None)
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


def listing_has_gold_map(
    listings: pd.DataFrame, gold: pd.DataFrame
) -> dict[str, bool]:
    gold_ids = set(gold["google_id"].astype(str))
    return {
        str(product_id): str(product_id) in gold_ids
        for product_id in listings["product_id"]
    }


def retrieve_test_one_by_one(
    listings: pd.DataFrame,
    amazon: pd.DataFrame,
    vectorizer: Any,
    catalog_tfidf: sparse.spmatrix,
    encoder: Any,
    catalog_embeddings: np.ndarray,
    top_k: int,
    batch_size: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Measure warmed, serving-like latency while producing test candidates."""
    if listings.empty:
        return pd.DataFrame(), np.asarray([], dtype=float)

    warm_listing = listings.iloc[[0]]
    warm_embedding = encode_products(warm_listing, encoder, batch_size=1)
    retrieve_candidates(
        warm_listing,
        amazon,
        vectorizer,
        catalog_tfidf,
        warm_embedding,
        catalog_embeddings,
        top_k=top_k,
    )

    candidate_frames: list[pd.DataFrame] = []
    latencies: list[float] = []
    for index in range(len(listings)):
        listing = listings.iloc[[index]]
        started = time.perf_counter()
        embedding = encode_products(listing, encoder, batch_size=min(batch_size, 1))
        candidates = retrieve_candidates(
            listing,
            amazon,
            vectorizer,
            catalog_tfidf,
            embedding,
            catalog_embeddings,
            top_k=top_k,
        )
        latencies.append(time.perf_counter() - started)
        candidate_frames.append(candidates)
    return pd.concat(candidate_frames, ignore_index=True), np.asarray(latencies)


def lexical_baseline_row(
    candidates: pd.DataFrame,
    labels: np.ndarray,
    has_gold: dict[str, bool],
    config: dict[str, Any],
) -> dict[str, Any]:
    scores = candidates["lexical_score"].to_numpy(dtype=float)
    query_ids = candidates["google_id"].to_numpy()
    top_indices = top_candidate_indices(scores, query_ids)
    top_labels = labels[top_indices]
    reject_correct = np.asarray(
        [not has_gold[str(query_ids[index])] for index in top_indices], dtype=int
    )
    thresholds = select_thresholds(
        top_labels,
        scores[top_indices],
        match_precision_target=float(config["match_precision_target"]),
        reject_precision_target=float(config["reject_precision_target"]),
        grid_step=float(config["threshold_grid_step"]),
        reject_correct=reject_correct,
        warn=False,
    )
    return {
        "model": "lexical_similarity_baseline",
        "validation_pr_auc": float(average_precision_score(labels, scores)),
        "validation_roc_auc": float(roc_auc_score(labels, scores)),
        "constraints_met": bool(thresholds["constraints_met"]),
        "automatic_coverage": thresholds["automatic_coverage"],
        "auto_match_precision": thresholds["auto_match_precision"],
        "auto_reject_precision": thresholds["auto_reject_precision"],
        "match_threshold": thresholds["match_threshold"],
        "reject_threshold": thresholds["reject_threshold"],
        "precision_shortfall": thresholds["precision_shortfall"],
        "class_weight": "not_applicable",
        "feature_count": 1,
        "warning": thresholds["warning"],
        "selected": False,
    }


def save_artifacts(
    output_dir: Path,
    vectorizer: Any,
    amazon: pd.DataFrame,
    catalog_tfidf: sparse.spmatrix,
    catalog_embeddings: np.ndarray,
    matcher_bundle: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, output_dir / "tfidf_vectorizer.joblib")
    amazon.to_csv(output_dir / "amazon_catalog.csv", index=False)
    sparse.save_npz(output_dir / "catalog_tfidf.npz", catalog_tfidf)
    np.save(output_dir / "catalog_dense.npy", catalog_embeddings)
    joblib.dump(matcher_bundle, output_dir / "matcher.joblib")


def run(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    config = load_config(config_path)
    seed = int(config["seed"])
    set_reproducible_seed(seed)

    model_cache = Path(config["model_cache_dir"]).resolve()
    model_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(model_cache))
    matplotlib_cache = model_cache.parent / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    raw_dir = Path(config["raw_dir"])
    artifacts_dir = Path(config["artifacts_dir"])
    reports_dir = Path(config["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Amazon-Google Products benchmark...")
    amazon, google, gold = load_dataset(
        raw_dir,
        download_if_missing=True,
        url=str(config["dataset_url"]),
    )
    assignments = make_entity_splits(
        google,
        gold,
        seed=seed,
        ratios=tuple(float(value) for value in config["split_ratios"]),
    )
    splits = split_google_listings(google, assignments)
    split_gold = {
        name: gold_for_listings(gold, listings)
        for name, listings in splits.items()
    }

    print("Fitting lexical retriever on catalog and training listings...")
    vectorizer, catalog_tfidf = fit_lexical_retriever(
        splits["train"],
        amazon,
        ngram_range=tuple(int(value) for value in config["tfidf_ngram_range"]),
        min_df=int(config["tfidf_min_df"]),
        max_features=int(config["tfidf_max_features"]),
    )

    print(f"Loading frozen dense encoder: {config['sentence_model']}")
    encoder = load_sentence_encoder(str(config["sentence_model"]))
    batch_size = int(config["dense_batch_size"])
    catalog_embeddings = encode_products(amazon, encoder, batch_size=batch_size)
    train_embeddings = encode_products(splits["train"], encoder, batch_size=batch_size)
    validation_embeddings = encode_products(
        splits["validation"], encoder, batch_size=batch_size
    )

    top_k = int(config["top_k"])
    train_candidates = retrieve_candidates(
        splits["train"],
        amazon,
        vectorizer,
        catalog_tfidf,
        train_embeddings,
        catalog_embeddings,
        top_k=top_k,
        gold_matches=split_gold["train"],
        inject_gold=True,
    )
    validation_candidates = retrieve_candidates(
        splits["validation"],
        amazon,
        vectorizer,
        catalog_tfidf,
        validation_embeddings,
        catalog_embeddings,
        top_k=top_k,
    )
    test_candidates, test_latencies = retrieve_test_one_by_one(
        splits["test"],
        amazon,
        vectorizer,
        catalog_tfidf,
        encoder,
        catalog_embeddings,
        top_k,
        batch_size,
    )

    train_labels = label_candidates(train_candidates, split_gold["train"])
    validation_labels = label_candidates(
        validation_candidates, split_gold["validation"]
    )
    test_labels = label_candidates(test_candidates, split_gold["test"])
    train_features = feature_candidates(train_candidates, google, amazon)
    validation_features = feature_candidates(validation_candidates, google, amazon)
    test_features = feature_candidates(test_candidates, google, amazon)

    component_by_google = assignments.set_index("google_id")["component_id"]
    training_groups = train_candidates["google_id"].map(component_by_google)
    variants = fit_model_variants(
        train_features,
        train_labels,
        training_groups,
        seed=seed,
        cv_folds=int(config["calibration_folds"]),
        max_iter=int(config["logreg_max_iter"]),
    )
    validation_has_gold = listing_has_gold_map(
        splits["validation"], split_gold["validation"]
    )
    selected_name, selected_bundle, comparison = select_model(
        variants,
        validation_features,
        validation_labels,
        validation_candidates["google_id"].to_numpy(),
        validation_listing_has_gold=validation_has_gold,
        match_precision_target=float(config["match_precision_target"]),
        reject_precision_target=float(config["reject_precision_target"]),
        threshold_grid_step=float(config["threshold_grid_step"]),
    )

    if should_try_balanced(comparison):
        print("Unweighted models missed the policy targets; trying balanced hybrid.")
        variants["hybrid_logistic_balanced"] = fit_balanced_variant(
            train_features,
            train_labels,
            training_groups,
            HYBRID_FEATURE_COLUMNS,
            seed=seed,
            cv_folds=int(config["calibration_folds"]),
            max_iter=int(config["logreg_max_iter"]),
        )
        selected_name, selected_bundle, comparison = select_model(
            variants,
            validation_features,
            validation_labels,
            validation_candidates["google_id"].to_numpy(),
            validation_listing_has_gold=validation_has_gold,
            match_precision_target=float(config["match_precision_target"]),
            reject_precision_target=float(config["reject_precision_target"]),
            threshold_grid_step=float(config["threshold_grid_step"]),
        )

    baseline = lexical_baseline_row(
        validation_candidates,
        validation_labels,
        validation_has_gold,
        config,
    )
    comparison = pd.concat(
        [pd.DataFrame([baseline]), comparison], ignore_index=True, sort=False
    )

    thresholds = selected_bundle["thresholds"]
    test_probabilities = predict_probabilities(selected_bundle, test_features)
    test_actions = apply_policy(
        test_probabilities,
        match_threshold=float(thresholds["match_threshold"]),
        reject_threshold=float(thresholds["reject_threshold"]),
    )
    prediction_frame = pd.concat(
        [test_candidates.reset_index(drop=True), test_features.reset_index(drop=True)],
        axis=1,
    )
    prediction_frame["label"] = test_labels
    prediction_frame["probability"] = test_probabilities
    prediction_frame["action"] = test_actions

    retrieval_metrics = compute_retrieval_metrics(
        test_candidates,
        split_gold["test"],
        ks=[int(value) for value in config["recall_ks"]],
        all_listing_ids=splits["test"]["product_id"].astype(str),
        latency_seconds=test_latencies,
    )
    retrieval_metrics["latency_definition"] = (
        "Warm per-listing query text transforms, dense encoding, and both exact "
        "searches; excludes model load and catalog precomputation."
    )
    pair_metrics = compute_pair_metrics(test_labels, test_probabilities)
    listing_metrics = compute_listing_metrics(
        prediction_frame,
        split_gold["test"],
        all_listing_ids=splits["test"]["product_id"].astype(str),
    )
    errors = extract_error_examples(
        prediction_frame,
        split_gold["test"],
        max_per_type=int(config["error_examples_per_type"]),
        google_records=google,
        amazon_records=amazon,
    )

    report_paths = save_evaluation_reports(
        reports_dir,
        retrieval_metrics,
        pair_metrics,
        listing_metrics,
        comparison,
        errors,
        test_labels,
        test_probabilities,
        float(thresholds["match_threshold"]),
        float(thresholds["reject_threshold"]),
    )

    metrics_path = Path(report_paths["metrics"])
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    selected_row = comparison.loc[comparison["model"].eq(selected_name)].iloc[0]
    metrics["dataset"] = {
        "amazon_records": len(amazon),
        "google_records": len(google),
        "gold_pairs": len(gold),
        "matched_google_records": int(gold["google_id"].nunique()),
    }
    metrics["splits"] = {
        name: {
            "google_records": len(listings),
            "gold_pairs": len(split_gold[name]),
            "matched_google_records": int(split_gold[name]["google_id"].nunique()),
        }
        for name, listings in splits.items()
    }
    metrics["model"] = {
        "selected": selected_name,
        "sentence_encoder": str(config["sentence_model"]),
        "class_weight": selected_bundle.get("class_weight"),
        "feature_columns": list(selected_bundle["feature_columns"]),
        "validation_constraints_met": bool(selected_row["constraints_met"]),
        "validation_warning": (
            None if pd.isna(selected_row["warning"]) else str(selected_row["warning"])
        ),
    }
    metrics["run"] = {
        "seed": seed,
        "split_ratios": [float(value) for value in config["split_ratios"]],
        "top_k": top_k,
        "recall_ks": [int(value) for value in config["recall_ks"]],
    }
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    matcher_bundle = {
        **selected_bundle,
        "selected_model": selected_name,
        "sentence_model_name": str(config["sentence_model"]),
        "top_k": top_k,
    }
    save_artifacts(
        artifacts_dir,
        vectorizer,
        amazon,
        catalog_tfidf,
        catalog_embeddings,
        matcher_bundle,
    )

    print(f"Selected model: {selected_name}")
    print(
        "Thresholds: "
        f"reject <= {thresholds['reject_threshold']:.2f}, "
        f"match >= {thresholds['match_threshold']:.2f}"
    )
    if not bool(selected_row["constraints_met"]):
        print(f"WARNING: {selected_row['warning']}")
    print(f"Reports written to {reports_dir.resolve()}")
    print(f"Demo artifacts written to {artifacts_dir.resolve()}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
