"""Interpretable pair features for catalog product identity matching."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover - requirements install RapidFuzz.
    from difflib import SequenceMatcher

    class _FallbackFuzz:
        @staticmethod
        def ratio(left: str, right: str) -> float:
            return 100.0 * SequenceMatcher(None, left, right).ratio()

        token_set_ratio = ratio

    fuzz = _FallbackFuzz()


TOKEN_RE = re.compile(r"[a-z0-9]+")
MODEL_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")

LEXICAL_FEATURE_COLUMNS = [
    "lexical_similarity",
    "title_token_jaccard",
    "title_fuzzy_similarity",
    "manufacturer_exact",
    "manufacturer_fuzzy_similarity",
]

DENSE_FEATURE_COLUMNS = ["dense_similarity"]

HYBRID_FEATURE_COLUMNS = [
    "lexical_similarity",
    "dense_similarity",
    "title_token_jaccard",
    "title_fuzzy_similarity",
    "manufacturer_exact",
    "manufacturer_fuzzy_similarity",
    "model_token_jaccard",
    "numeric_token_conflict",
    "relative_price_difference",
    "query_manufacturer_missing",
    "candidate_manufacturer_missing",
    "query_description_missing",
    "candidate_description_missing",
    "query_price_missing",
    "candidate_price_missing",
]

RANK_FEATURE_COLUMNS = [
    "retrieval_source_agreement",
    "reciprocal_lexical_rank",
    "reciprocal_dense_rank",
]

HYBRID_PLUS_RANK_FEATURE_COLUMNS = [
    *HYBRID_FEATURE_COLUMNS,
    *RANK_FEATURE_COLUMNS,
]

FEATURE_COLUMNS = HYBRID_FEATURE_COLUMNS


def normalize_text(value: Any) -> str:
    """Lowercase text and collapse whitespace; missing values become empty."""
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return " ".join(str(value).lower().split())


def text_tokens(value: Any) -> set[str]:
    """Return simple lowercase alphanumeric tokens."""
    return set(TOKEN_RE.findall(normalize_text(value)))


def model_tokens(value: Any) -> set[str]:
    """Return title tokens containing a digit, including model-like tokens."""
    return {
        token
        for token in MODEL_TOKEN_RE.findall(normalize_text(value))
        if any(character.isdigit() for character in token)
    }


extract_model_tokens = model_tokens


def jaccard(left: set[str], right: set[str]) -> float:
    """Set Jaccard similarity, defined as zero when both sets are empty."""
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _valid_price(value: Any) -> float | None:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(price) or price <= 0:
        return None
    return price


def relative_price_difference(left: Any, right: Any) -> float:
    """Relative price distance in [0, 1], or zero if either price is absent."""
    left_price = _valid_price(left)
    right_price = _valid_price(right)
    if left_price is None or right_price is None:
        return 0.0
    return abs(left_price - right_price) / max(abs(left_price), abs(right_price))


def _record_value(record: Mapping[str, Any] | pd.Series, field: str) -> Any:
    return record.get(field, "")


def pair_feature(
    query: Mapping[str, Any] | pd.Series,
    candidate: Mapping[str, Any] | pd.Series,
    lexical_similarity: float = 0.0,
    dense_similarity: float = 0.0,
) -> dict[str, float]:
    """Create the complete feature dictionary for one aligned product pair."""
    query_title = normalize_text(_record_value(query, "title"))
    candidate_title = normalize_text(_record_value(candidate, "title"))
    query_manufacturer = normalize_text(_record_value(query, "manufacturer"))
    candidate_manufacturer = normalize_text(_record_value(candidate, "manufacturer"))
    query_description = normalize_text(_record_value(query, "description"))
    candidate_description = normalize_text(_record_value(candidate, "description"))

    query_models = model_tokens(query_title)
    candidate_models = model_tokens(candidate_title)
    query_price = _valid_price(_record_value(query, "price"))
    candidate_price = _valid_price(_record_value(candidate, "price"))

    return {
        "lexical_similarity": float(lexical_similarity),
        "dense_similarity": float(dense_similarity),
        "title_token_jaccard": jaccard(
            text_tokens(query_title), text_tokens(candidate_title)
        ),
        "title_fuzzy_similarity": (
            float(fuzz.token_set_ratio(query_title, candidate_title)) / 100.0
            if query_title and candidate_title
            else 0.0
        ),
        "manufacturer_exact": float(
            bool(query_manufacturer)
            and bool(candidate_manufacturer)
            and query_manufacturer == candidate_manufacturer
        ),
        "manufacturer_fuzzy_similarity": (
            float(fuzz.ratio(query_manufacturer, candidate_manufacturer)) / 100.0
            if query_manufacturer and candidate_manufacturer
            else 0.0
        ),
        "model_token_jaccard": jaccard(query_models, candidate_models),
        "numeric_token_conflict": float(
            bool(query_models)
            and bool(candidate_models)
            and query_models != candidate_models
        ),
        "relative_price_difference": relative_price_difference(
            query_price, candidate_price
        ),
        "query_manufacturer_missing": float(not query_manufacturer),
        "candidate_manufacturer_missing": float(not candidate_manufacturer),
        "query_description_missing": float(not query_description),
        "candidate_description_missing": float(not candidate_description),
        "query_price_missing": float(query_price is None),
        "candidate_price_missing": float(candidate_price is None),
    }


def _score_array(values: Sequence[float] | np.ndarray | None, size: int) -> np.ndarray:
    if values is None:
        return np.zeros(size, dtype=float)
    scores = np.asarray(values, dtype=float).reshape(-1)
    if scores.size != size:
        raise ValueError(f"Expected {size} similarity scores, received {scores.size}.")
    return scores


def build_feature_frame(
    query_records: pd.DataFrame,
    candidate_records: pd.DataFrame,
    lexical_scores: Sequence[float] | np.ndarray | None = None,
    dense_scores: Sequence[float] | np.ndarray | None = None,
) -> pd.DataFrame:
    """Build features for two equally sized, row-aligned product frames."""
    if len(query_records) != len(candidate_records):
        raise ValueError("Query and candidate frames must have equal lengths.")

    size = len(query_records)
    lexical = _score_array(lexical_scores, size)
    dense = _score_array(dense_scores, size)
    query_rows = query_records.to_dict(orient="records")
    candidate_rows = candidate_records.to_dict(orient="records")
    rows = [
        pair_feature(query, candidate, lexical[index], dense[index])
        for index, (query, candidate) in enumerate(zip(query_rows, candidate_rows))
    ]
    return pd.DataFrame(rows, columns=HYBRID_FEATURE_COLUMNS, index=query_records.index)


def build_retrieval_rank_features(pairs: pd.DataFrame) -> pd.DataFrame:
    """Create lightweight rank signals from a retrieval candidate table.

    Missing or invalid channel ranks contribute zero. Source agreement is one
    only when the candidate occurs in both the lexical and dense top-k lists.
    """

    def _ranks(column: str) -> np.ndarray:
        if column not in pairs:
            return np.full(len(pairs), np.nan, dtype=float)
        return pd.to_numeric(pairs[column], errors="coerce").to_numpy(dtype=float)

    lexical_ranks = _ranks("lexical_rank")
    dense_ranks = _ranks("dense_rank")
    lexical_present = np.isfinite(lexical_ranks) & (lexical_ranks > 0)
    dense_present = np.isfinite(dense_ranks) & (dense_ranks > 0)
    reciprocal_lexical = np.zeros(len(pairs), dtype=float)
    reciprocal_dense = np.zeros(len(pairs), dtype=float)
    reciprocal_lexical[lexical_present] = 1.0 / lexical_ranks[lexical_present]
    reciprocal_dense[dense_present] = 1.0 / dense_ranks[dense_present]

    return pd.DataFrame(
        {
            "retrieval_source_agreement": (lexical_present & dense_present).astype(
                float
            ),
            "reciprocal_lexical_rank": reciprocal_lexical,
            "reciprocal_dense_rank": reciprocal_dense,
        },
        index=pairs.index,
        columns=RANK_FEATURE_COLUMNS,
    )


def build_pair_features(
    pairs: pd.DataFrame,
    google: pd.DataFrame,
    amazon: pd.DataFrame,
    lexical_scores: Sequence[float] | np.ndarray | None = None,
    dense_scores: Sequence[float] | np.ndarray | None = None,
    *,
    query_id_column: str = "google_id",
    candidate_id_column: str = "amazon_id",
    record_id_column: str = "product_id",
) -> pd.DataFrame:
    """Join candidate-pair IDs to source records and return aligned features.

    If explicit score arrays are omitted, the retrieval columns
    ``lexical_score`` and ``dense_score`` (or their ``*_similarity`` aliases)
    are read from ``pairs`` when present.
    """
    required_pair_columns = {query_id_column, candidate_id_column}
    missing_pair_columns = required_pair_columns - set(pairs.columns)
    if missing_pair_columns:
        raise KeyError(f"Missing pair columns: {sorted(missing_pair_columns)}")
    if record_id_column not in google or record_id_column not in amazon:
        raise KeyError(f"Both source tables need a {record_id_column!r} column.")

    google_index = google.set_index(record_id_column, drop=False)
    amazon_index = amazon.set_index(record_id_column, drop=False)
    query_ids = pairs[query_id_column]
    candidate_ids = pairs[candidate_id_column]
    unknown_queries = set(query_ids) - set(google_index.index)
    unknown_candidates = set(candidate_ids) - set(amazon_index.index)
    if unknown_queries or unknown_candidates:
        raise KeyError(
            "Candidate pairs reference unknown IDs: "
            f"{len(unknown_queries)} Google, {len(unknown_candidates)} Amazon."
        )

    query_records = google_index.loc[query_ids].reset_index(drop=True)
    candidate_records = amazon_index.loc[candidate_ids].reset_index(drop=True)
    if lexical_scores is None:
        for column in ("lexical_score", "lexical_similarity"):
            if column in pairs:
                lexical_scores = pairs[column].to_numpy()
                break
    if dense_scores is None:
        for column in ("dense_score", "dense_similarity"):
            if column in pairs:
                dense_scores = pairs[column].to_numpy()
                break

    features = build_feature_frame(
        query_records,
        candidate_records,
        lexical_scores=lexical_scores,
        dense_scores=dense_scores,
    )
    features.index = pairs.index
    return pd.concat([features, build_retrieval_rank_features(pairs)], axis=1)


make_pair_features = build_pair_features
