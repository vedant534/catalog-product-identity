"""Small, fast checks for the core product-identity workflow."""

from __future__ import annotations

import warnings

import joblib
import numpy as np
import pandas as pd
import pytest

from run_pipeline import (
    CORRECTED_RESPLIT_PROTOCOL,
    DEVELOP_STAGE,
    _validate_corrected_snapshot,
    _validate_frozen_split_assignments,
)
from src.data import (
    make_entity_splits,
    split_assignment_metadata,
    validate_assignment_ids,
    validate_split_assignments,
)
from src.evaluate import (
    build_listing_predictions,
    compute_listing_metrics,
    compute_no_match_metrics,
    compute_pair_metrics,
    compute_retrieval_metrics,
    save_corrected_resplit_reports,
)
from src.features import HYBRID_FEATURE_COLUMNS, build_pair_features, pair_feature
from src.model import fit_calibrated_logistic, predict_probabilities
from src.policy import apply_policy, select_thresholds
from src.retrieval import _cosine_scores, fit_lexical_retriever, retrieve_candidates


def _product(product_id: str, title: str, manufacturer: str, price: float) -> dict:
    return {
        "product_id": product_id,
        "title": title,
        "manufacturer": manufacturer,
        "description": f"Details for {title}",
        "price": price,
    }


def test_pair_features_handle_missing_values_and_model_tokens() -> None:
    query = {
        "title": "Acme Camera X100 64GB",
        "manufacturer": None,
        "description": "",
        "price": None,
    }
    candidate = {
        "title": "Acme Camera X200 128GB",
        "manufacturer": "Acme",
        "description": None,
        "price": 0,
    }

    features = pair_feature(query, candidate, lexical_similarity=0.7, dense_similarity=0.8)

    assert set(HYBRID_FEATURE_COLUMNS) == set(features)
    assert features["numeric_token_conflict"] == 1.0
    assert features["query_manufacturer_missing"] == 1.0
    assert features["query_description_missing"] == 1.0
    assert features["query_price_missing"] == 1.0
    assert features["candidate_price_missing"] == 1.0
    assert features["relative_price_difference"] == 0.0
    assert np.isfinite(list(features.values())).all()


def test_entity_components_never_cross_splits() -> None:
    google = pd.DataFrame(
        [
            _product(f"g{index}", f"Product {index}", "Maker", 10.0 + index)
            for index in range(16)
        ]
    )
    gold = pd.DataFrame(
        [
            ("g0", "a_shared"),
            ("g1", "a_shared"),
            *[(f"g{index}", f"a{index}") for index in range(2, 12)],
        ],
        columns=["google_id", "amazon_id"],
    )

    assignments = make_entity_splits(
        google, gold, seed=17, ratios=(0.70, 0.15, 0.15)
    )

    assert assignments["google_id"].is_unique
    assert set(assignments["google_id"]) == set(google["product_id"])
    assert assignments.groupby("component_id")["split"].nunique().max() == 1
    split_components = {
        split: set(rows["component_id"])
        for split, rows in assignments.groupby("split")
    }
    names = list(split_components)
    assert all(
        split_components[left].isdisjoint(split_components[right])
        for index, left in enumerate(names)
        for right in names[index + 1 :]
    )
    shared_component_splits = assignments.loc[
        assignments["google_id"].isin(["g0", "g1"]), "split"
    ]
    assert shared_component_splits.nunique() == 1


def test_assignment_ids_must_match_normalized_google_ids_exactly() -> None:
    google = pd.DataFrame({"product_id": ["g1", "g2"]})
    valid = pd.DataFrame({"google_id": ["g1", "g2"], "split": ["train", "test"]})
    validate_assignment_ids(google, valid)

    duplicate = pd.DataFrame({"google_id": ["g1", "g1"]})
    with pytest.raises(ValueError, match="must be unique"):
        validate_assignment_ids(google, duplicate)

    missing_and_extra = pd.DataFrame({"google_id": ["g1", "g3"]})
    with pytest.raises(ValueError, match=r"missing=1, extra=1"):
        validate_assignment_ids(google, missing_and_extra)


def test_corrected_preflight_rejects_same_count_membership_swap() -> None:
    google = pd.DataFrame(
        [
            _product(f"g{index:02d}", f"Unique Product {index}", "Maker", 10 + index)
            for index in range(24)
        ]
    )
    gold = pd.DataFrame(columns=["google_id", "amazon_id"])
    ratios = (0.5, 0.25, 0.25)
    seed = 20260825
    expected = validate_split_assignments(
        google,
        make_entity_splits(google, gold, seed=seed, ratios=ratios),
    )
    tampered = expected.copy()
    validation_index = tampered.index[tampered["split"].eq("validation")][0]
    test_index = tampered.index[tampered["split"].eq("test")][0]
    tampered.loc[validation_index, "split"] = "test"
    tampered.loc[test_index, "split"] = "validation"

    # The swap preserves IDs, counts, and singleton group isolation. The
    # deterministic regeneration check must still reject changed membership.
    tampered = validate_split_assignments(google, tampered)
    assert split_assignment_metadata(tampered)["rows_by_split"] == (
        split_assignment_metadata(expected)["rows_by_split"]
    )
    with pytest.raises(ValueError, match="canonical membership"):
        _validate_frozen_split_assignments(
            {"split_assignments": split_assignment_metadata(tampered)},
            tampered,
            google,
            gold,
            {"split_seed": seed, "split_ratios": ratios},
        )


def test_policy_threshold_boundaries_are_inclusive() -> None:
    actions = apply_policy(
        np.array([0.0, 0.2, 0.2001, 0.7999, 0.8, 1.0]),
        match_threshold=0.8,
        no_match_threshold=0.2,
    )
    assert actions.tolist() == [
        "auto_no_match",
        "auto_no_match",
        "manual_review",
        "manual_review",
        "auto_match",
        "auto_match",
    ]


def test_synthetic_pipeline_and_joblib_round_trip(tmp_path) -> None:
    amazon = pd.DataFrame(
        [
            _product("a1", "Acme Camera X100", "Acme", 100.0),
            _product("a2", "Acme Camera X200", "Acme", 150.0),
            _product("a3", "Beta Printer P10", "Beta", 80.0),
            _product("a4", "Gamma Keyboard K2", "Gamma", 20.0),
        ]
    )
    listing_specs = [
        ("g1", "Acme Camera X100", "Acme", 99.0, "a1"),
        ("g2", "Acme Camera X200", "Acme", 151.0, "a2"),
        ("g3", "Beta Printer P10", "Beta", 81.0, "a3"),
        ("g4", "Gamma Keyboard K2", "Gamma", 19.0, "a4"),
        ("g5", "X100 Acme Digital Camera", "Acme", 101.0, "a1"),
        ("g6", "X200 Acme Digital Camera", "Acme", 149.0, "a2"),
        ("g7", "P10 Beta Office Printer", "Beta", 79.0, "a3"),
        ("g8", "K2 Gamma USB Keyboard", "Gamma", 21.0, "a4"),
    ]
    google = pd.DataFrame(
        [_product(gid, title, maker, price) for gid, title, maker, price, _ in listing_specs]
    )
    gold = pd.DataFrame(
        [(gid, amazon_id) for gid, _, _, _, amazon_id in listing_specs],
        columns=["google_id", "amazon_id"],
    )

    vectorizer, catalog_tfidf = fit_lexical_retriever(google, amazon)
    catalog_embeddings = np.eye(len(amazon), dtype=np.float32)
    match_index = {amazon_id: index for index, amazon_id in enumerate(amazon["product_id"])}
    query_embeddings = np.vstack(
        [catalog_embeddings[match_index[amazon_id]] for *_, amazon_id in listing_specs]
    )
    candidates = retrieve_candidates(
        google,
        amazon,
        vectorizer,
        catalog_tfidf,
        query_embeddings,
        catalog_embeddings,
        top_k=2,
        gold_matches=gold,
        inject_gold=True,
    )

    gold_pairs = set(gold.itertuples(index=False, name=None))
    labels = np.fromiter(
        (
            (google_id, amazon_id) in gold_pairs
            for google_id, amazon_id in candidates[["google_id", "amazon_id"]].itertuples(
                index=False, name=None
            )
        ),
        dtype=int,
        count=len(candidates),
    )
    features = build_pair_features(
        candidates,
        google,
        amazon,
        lexical_scores=candidates["lexical_score"].to_numpy(),
        dense_scores=candidates["dense_score"].to_numpy(),
        record_id_column="product_id",
    )
    groups = candidates["google_id"].to_numpy()
    estimator = fit_calibrated_logistic(
        features,
        labels,
        groups,
        HYBRID_FEATURE_COLUMNS,
        seed=23,
        cv_folds=2,
    )
    bundle = {
        "estimator": estimator,
        "feature_columns": list(HYBRID_FEATURE_COLUMNS),
    }
    probabilities = predict_probabilities(bundle, features)

    artifact_path = tmp_path / "synthetic_model.joblib"
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Setting the shape on a NumPy array has been deprecated.*",
            category=DeprecationWarning,
        )
        joblib.dump(bundle, artifact_path)
        loaded = joblib.load(artifact_path)
    reloaded_probabilities = predict_probabilities(loaded, features)

    assert len(candidates) >= len(google) * 2
    assert labels.min() == 0 and labels.max() == 1
    assert probabilities.shape == (len(candidates),)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    np.testing.assert_allclose(reloaded_probabilities, probabilities)


def test_exact_duplicate_groups_do_not_cross_splits_and_flag_ambiguity() -> None:
    google = pd.DataFrame(
        [
            _product("g_mapped", "Same Product", "Maker", 25.0),
            _product("g_unmapped", "  same   product ", "maker", 25.0),
            *[
                _product(f"g_unique_{index}", f"Unique {index}", "Maker", 30.0 + index)
                for index in range(18)
            ],
        ]
    )
    # Make the duplicate descriptions identical after whitespace normalization.
    google.loc[google["product_id"].eq("g_unmapped"), "description"] = (
        google.loc[google["product_id"].eq("g_mapped"), "description"].iloc[0].upper()
    )
    gold = pd.DataFrame(
        [("g_mapped", "a_shared")], columns=["google_id", "amazon_id"]
    )

    assignments = make_entity_splits(
        google, gold, seed=20260825, ratios=(0.70, 0.15, 0.15)
    )
    duplicate_rows = assignments.loc[
        assignments["google_id"].isin(["g_mapped", "g_unmapped"])
    ].set_index("google_id")

    assert duplicate_rows["component_id"].nunique() == 1
    assert duplicate_rows["duplicate_group_id"].nunique() == 1
    assert duplicate_rows["split"].nunique() == 1
    assert duplicate_rows["duplicate_group_size"].eq(2).all()
    assert not bool(duplicate_rows.loc["g_mapped", "ambiguous_label"])
    assert bool(duplicate_rows.loc["g_unmapped", "ambiguous_label"])
    assert assignments.groupby("duplicate_group_id")["split"].nunique().max() == 1


def test_threshold_support_is_required_independently_per_action() -> None:
    unsupported_match = select_thresholds(
        y_true=np.array([1, *([0] * 20)]),
        scores=np.array([0.99, *([0.01] * 20)]),
        no_match_correct=np.array([0, *([1] * 20)]),
        min_auto_match_support=20,
        min_auto_no_match_support=20,
        warn=False,
    )
    assert unsupported_match["auto_match"]["feasible"] is False
    assert unsupported_match["auto_match"]["enabled"] is False
    assert unsupported_match["auto_match"]["threshold"] is None
    assert unsupported_match["auto_no_match"]["enabled"] is True
    assert unsupported_match["both_constraints_met"] is False

    supported = select_thresholds(
        y_true=np.array([*([1] * 20), *([0] * 20)]),
        scores=np.array([*([0.99] * 20), *([0.01] * 20)]),
        no_match_correct=np.array([*([0] * 20), *([1] * 20)]),
        min_auto_match_support=20,
        min_auto_no_match_support=20,
        warn=False,
    )
    assert supported["auto_match"]["enabled"] is True
    assert supported["auto_no_match"]["enabled"] is True
    assert supported["both_constraints_met"] is True
    assert supported["auto_match"]["support"] >= 20
    assert supported["auto_no_match"]["support"] >= 20


def test_duplicate_groups_drive_threshold_selection_but_report_listings() -> None:
    scores: list[float] = []
    no_match_correct: list[int] = []
    group_ids: list[str] = []
    for group_id in ("correct_1", "correct_2", "correct_3", "correct_4"):
        scores.extend([0.02] * 4)
        no_match_correct.extend([1] * 4)
        group_ids.extend([group_id] * 4)
    # This exact-duplicate group is label-inconsistent. Even though one listing
    # is correct, the whole group must be conservative incorrect evidence.
    scores.extend([0.02, 0.02])
    no_match_correct.extend([1, 0])
    group_ids.extend(["inconsistent", "inconsistent"])
    scores.extend([0.03, 0.03])
    no_match_correct.extend([0, 0])
    group_ids.extend(["wrong_1", "wrong_2"])

    no_match = np.asarray(no_match_correct, dtype=int)
    selection = select_thresholds(
        y_true=1 - no_match,
        scores=np.asarray(scores),
        no_match_correct=no_match,
        duplicate_group_ids=np.asarray(group_ids),
        match_precision_target=0.75,
        no_match_precision_target=0.75,
        min_auto_match_support=8,
        min_auto_no_match_support=5,
        grid_step=0.01,
        warn=False,
    )

    assert selection["no_match_threshold"] == 0.02
    assert selection["auto_match"]["enabled"] is False
    group = selection["auto_no_match"]["group_evidence"]
    listings = selection["auto_no_match"]["listing_operation"]
    assert (group["correct_count"], group["support"]) == (4, 5)
    assert group["empirical_precision"] == 0.8
    assert (listings["correct_count"], listings["support"]) == (17, 18)
    assert listings["coverage"] == 0.9
    assert listings["precision_wilson_95_low"] is not None

    loose = next(
        row
        for row in selection["threshold_diagnostics"]
        if row["action"] == "auto_no_match" and row["threshold"] == 0.03
    )
    assert loose["group_empirical_precision"] == 4 / 7
    assert loose["listing_empirical_precision"] == 17 / 20
    assert loose["listing_empirical_precision"] >= 0.75
    assert loose["feasible"] is False


def test_listing_policy_uses_one_top_row_and_shared_tie_break() -> None:
    candidates = pd.DataFrame(
        [
            ("g_tie", "a2", 0.9),
            ("g_tie", "a1", 0.9),
            ("g_retrieval", "a2", 0.7),
            ("g_retrieval", "a3", 0.6),
            ("g_rerank", "a2", 0.7),
            ("g_rerank", "a3", 0.6),
            ("g_no_match", "a2", 0.1),
            ("g_no_match", "a3", 0.05),
        ],
        columns=["google_id", "amazon_id", "probability"],
    )
    gold = pd.DataFrame(
        [("g_tie", "a1"), ("g_retrieval", "a9"), ("g_rerank", "a3")],
        columns=["google_id", "amazon_id"],
    )
    policy = {
        "auto_match": {"enabled": True, "threshold": 0.8},
        "auto_no_match": {"enabled": True, "threshold": 0.2},
    }
    listings = build_listing_predictions(
        candidates,
        gold,
        ["g_tie", "g_retrieval", "g_rerank", "g_no_match"],
        policy,
        duplicate_group_by_listing={
            "g_tie": "group_tie",
            "g_retrieval": "group_retrieval",
            "g_rerank": "group_rerank",
            "g_no_match": "group_no_match",
        },
    ).set_index("google_id")

    assert len(listings) == 4
    assert listings.loc["g_tie", "amazon_id"] == "a1"
    assert listings.loc["g_tie", "action"] == "auto_match"
    assert listings.loc["g_retrieval", "ranking_outcome"] == "retrieval_miss"
    assert listings.loc["g_rerank", "ranking_outcome"] == "reranking_miss"
    assert listings.loc["g_no_match", "action"] == "auto_no_match"
    assert bool(listings.loc["g_no_match", "action_correct"])
    assert listings.loc["g_no_match", "duplicate_group_id"] == "group_no_match"
    assert "action" not in candidates.columns


def test_listing_policy_reports_conservative_group_and_listing_evidence() -> None:
    listings = pd.DataFrame(
        {
            "google_id": ["g1", "g2", "g3", "g4", "g5"],
            "duplicate_group_id": ["dup_a", "dup_a", "dup_b", "dup_b", "single"],
            "action": [
                "auto_no_match",
                "auto_no_match",
                "auto_no_match",
                "auto_no_match",
                "manual_review",
            ],
            "has_gold_listing": [False, False, False, True, False],
            "is_gold_pair": [False, False, False, False, False],
            "action_correct": pd.array([True, True, True, False, pd.NA], dtype="boolean"),
        }
    )

    metrics = compute_listing_metrics(listings)
    no_match = metrics["auto_no_match"]
    assert no_match["group_evidence"]["support"] == 2
    assert no_match["group_evidence"]["correct_count"] == 1
    assert no_match["group_evidence"]["error_count"] == 1
    assert no_match["group_evidence"]["empirical_precision"] == 0.5
    assert no_match["group_evidence"]["coverage"] == pytest.approx(2 / 3)
    assert no_match["group_evidence"]["precision_wilson_95_low"] is not None
    assert no_match["listing_operation"]["support"] == 4
    assert no_match["listing_operation"]["correct_count"] == 3
    assert no_match["listing_operation"]["empirical_precision"] == 0.75
    assert no_match["listing_operation"]["coverage"] == 0.8
    assert metrics["auto_match"]["group_evidence"]["empirical_precision"] is None
    assert metrics["auto_match"]["listing_operation"]["empirical_precision"] is None
    assert metrics["manual_review"]["group_evidence"]["support"] == 1
    assert metrics["manual_review"]["group_evidence"]["coverage"] == pytest.approx(
        1 / 3
    )
    assert metrics["manual_review"]["listing_operation"]["support"] == 1
    assert metrics["manual_review"]["listing_operation"]["coverage"] == 0.2
    assert metrics["manual_review"]["listing_operation"]["correct_count"] is None


def test_frozen_metadata_mismatches_fail_clearly() -> None:
    config = {
        "split_seed": 20260825,
        "model_seed": 42,
        "split_ratios": [0.7, 0.15, 0.15],
        "top_k": 20,
        "rrf_constant": 60.0,
        "sentence_model": "encoder",
        "sentence_model_revision": "revision",
    }
    assignment_metadata = {
        "sha256": "0" * 64,
        "row_count": 4,
        "rows_by_split": {"train": 2, "validation": 1, "test": 1},
    }
    policy = {
        "auto_match": {"enabled": False, "threshold": None},
        "auto_no_match": {"enabled": True, "threshold": 0.02},
    }
    snapshot = {
        "stage": DEVELOP_STAGE,
        "evaluation_protocol": CORRECTED_RESPLIT_PROTOCOL,
        "corrected_resplit_status_at_development": "not_run",
        "config": dict(config),
        "frozen_inputs": {
            "sentence_encoder": {
                "model_name": "encoder",
                "revision": "revision",
            },
            "split_assignments": dict(assignment_metadata),
        },
        "split_assignments": dict(assignment_metadata),
        "selected_model": "hybrid_logistic",
        "feature_columns": ["lexical_similarity", "dense_similarity"],
        "policy": policy,
    }
    matcher = {
        "split_seed": 20260825,
        "model_seed": 42,
        "top_k": 20,
        "rrf_constant": 60.0,
        "sentence_model_name": "encoder",
        "sentence_model_revision": "revision",
        "selected_model": "hybrid_logistic",
        "feature_columns": ["lexical_similarity", "dense_similarity"],
        "policy": policy,
    }

    _validate_corrected_snapshot(config, snapshot, matcher)
    with pytest.raises(ValueError, match="top_k"):
        _validate_corrected_snapshot(config, snapshot, {**matcher, "top_k": 21})
    with pytest.raises(ValueError, match="feature column order"):
        _validate_corrected_snapshot(
            config,
            snapshot,
            {**matcher, "feature_columns": list(reversed(matcher["feature_columns"]))},
        )


def test_corrected_report_bundle_is_additive_and_refuses_overwrite(tmp_path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    development_files = {
        "metrics.json": b"development metrics\n",
        "model_comparison.csv": b"development comparison\n",
        "validation_precision_coverage.csv": b"development thresholds\n",
    }
    for name, content in development_files.items():
        (reports / name).write_bytes(content)

    corrected = reports / "corrected_resplit"
    retrieval = {
        "lexical_recall_at_1": 1.0,
        "dense_recall_at_1": 1.0,
        "rrf_recall_at_1": 1.0,
        "union_per_channel_recall_at_1": 1.0,
    }
    save_corrected_resplit_reports(
        corrected,
        {"stage": "corrected-eval"},
        pd.DataFrame(
            {
                "google_id": ["g1"],
                "duplicate_group_id": ["group_g1"],
                "action": ["manual_review"],
            }
        ),
        pd.DataFrame({"error_type": ["none"]}),
        [0, 1],
        [0.1, 0.9],
        [0, 1],
        [0.1, 0.9],
        retrieval,
    )

    assert {path.name for path in corrected.iterdir()} == {
        "metrics.json",
        "listing_predictions.csv",
        "error_examples.csv",
        "corrected_resplit_retrieval_recall.png",
        "corrected_resplit_pair_precision_recall_curve.png",
        "corrected_resplit_pair_reliability_plot.png",
        "corrected_resplit_top_candidate_reliability_plot.png",
    }
    assert "duplicate_group_id" in pd.read_csv(
        corrected / "listing_predictions.csv"
    ).columns
    for name, content in development_files.items():
        assert (reports / name).read_bytes() == content
    with pytest.raises(FileExistsError, match="already exists"):
        save_corrected_resplit_reports(
            corrected,
            {},
            pd.DataFrame(),
            pd.DataFrame(),
            [0, 1],
            [0.1, 0.9],
            [0, 1],
            [0.1, 0.9],
            retrieval,
        )


def test_small_correctness_helpers_fail_clearly_and_name_metrics_honestly() -> None:
    pair_metrics = compute_pair_metrics(
        [0, 1],
        [0.4, 0.6],
        classification_threshold=0.7,
    )
    assert pair_metrics["classification_threshold"] == 0.7
    assert {"precision", "recall", "f1"}.issubset(pair_metrics)
    assert not any(key.endswith("_at_0_5") for key in pair_metrics)

    no_actions = pd.DataFrame(
        {
            "has_gold_listing": [True, False],
            "probability": [0.8, 0.2],
            "action": ["manual_review", "manual_review"],
        }
    )
    assert compute_no_match_metrics(no_actions)["precision_at_policy"] is None

    minimal = pd.DataFrame({"google_id": ["g1"], "amazon_id": ["a1"]})
    with pytest.raises(ValueError, match="candidate depth"):
        compute_retrieval_metrics(
            minimal,
            minimal,
            minimal,
            ks=[2],
            per_channel_candidate_depth=1,
        )
    with pytest.raises(ValueError, match="one- or two-dimensional"):
        _cosine_scores(np.asarray(1.0), np.eye(2))
