"""Small, fast checks for the core product-identity workflow."""

from __future__ import annotations

import warnings

import joblib
import numpy as np
import pandas as pd

from src.data import make_entity_splits
from src.evaluate import build_listing_predictions
from src.features import HYBRID_FEATURE_COLUMNS, build_pair_features, pair_feature
from src.model import fit_calibrated_logistic, predict_probabilities
from src.policy import apply_policy, select_thresholds
from src.retrieval import fit_lexical_retriever, retrieve_candidates


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
    ).set_index("google_id")

    assert len(listings) == 4
    assert listings.loc["g_tie", "amazon_id"] == "a1"
    assert listings.loc["g_tie", "action"] == "auto_match"
    assert listings.loc["g_retrieval", "ranking_outcome"] == "retrieval_miss"
    assert listings.loc["g_rerank", "ranking_outcome"] == "reranking_miss"
    assert listings.loc["g_no_match", "action"] == "auto_no_match"
    assert bool(listings.loc["g_no_match", "action_correct"])
    assert "action" not in candidates.columns
