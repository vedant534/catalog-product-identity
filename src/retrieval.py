"""Exact lexical and dense candidate retrieval for the small benchmark."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


TEXT_FIELDS = ("title", "manufacturer", "description")
CANDIDATE_COLUMNS = [
    "google_id",
    "amazon_id",
    "lexical_score",
    "dense_score",
    "lexical_rank",
    "dense_rank",
    "retrieval_sources",
    "gold_injected",
]


def combine_text_fields(records: pd.DataFrame) -> pd.Series:
    """Concatenate available retrieval text without leaking ``nan`` strings."""
    pieces = []
    for field in TEXT_FIELDS:
        if field in records:
            values = records[field].astype("string").fillna("")
        else:
            values = pd.Series("", index=records.index, dtype="string")
        pieces.append(values.str.replace(r"\s+", " ", regex=True).str.strip())

    combined = pieces[0]
    for piece in pieces[1:]:
        combined = combined.str.cat(piece, sep=" ")
    return combined.str.replace(r"\s+", " ", regex=True).str.strip()


def fit_lexical_retriever(
    train_listings: pd.DataFrame,
    catalog: pd.DataFrame,
    ngram_range: tuple[int, int] = (3, 5),
    min_df: int = 1,
    max_features: int | None = None,
) -> tuple[TfidfVectorizer, sparse.csr_matrix]:
    """Fit char TF-IDF on the fixed catalog plus training Google listings only."""
    catalog_text = combine_text_fields(catalog)
    training_text = combine_text_fields(train_listings)
    fit_text = pd.concat([catalog_text, training_text], ignore_index=True)
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=tuple(ngram_range),
        min_df=min_df,
        max_features=max_features,
        lowercase=True,
        norm="l2",
        sublinear_tf=True,
    )
    vectorizer.fit(fit_text)
    return vectorizer, vectorizer.transform(catalog_text).tocsr()


def load_sentence_encoder(model_name: str, device: str | None = None) -> Any:
    """Load the frozen encoder from cache first, downloading only when absent."""
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(
            model_name,
            device=device,
            local_files_only=True,
        )
    except (OSError, ValueError):
        return SentenceTransformer(model_name, device=device)


def encode_products(
    records: pd.DataFrame,
    model: Any,
    batch_size: int = 64,
) -> np.ndarray:
    """Encode product text as normalized float32 embeddings."""
    if hasattr(model, "eval"):
        model.eval()
    embeddings = model.encode(
        combine_text_fields(records).tolist(),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def _cosine_scores(query_matrix: Any, catalog_matrix: Any) -> np.ndarray:
    if query_matrix.shape[1] != catalog_matrix.shape[1]:
        raise ValueError("Query and catalog matrices must have the same feature count")
    if catalog_matrix.shape[0] == 0:
        raise ValueError("Catalog must contain at least one product")

    if sparse.issparse(query_matrix) or sparse.issparse(catalog_matrix):
        query_normalized = normalize(sparse.csr_matrix(query_matrix), norm="l2", copy=True)
        catalog_normalized = normalize(
            sparse.csr_matrix(catalog_matrix), norm="l2", copy=True
        )
        return (query_normalized @ catalog_normalized.T).toarray()

    query_array = np.asarray(query_matrix, dtype=np.float32)
    catalog_array = np.asarray(catalog_matrix, dtype=np.float32)
    if query_array.ndim == 1:
        query_array = query_array.reshape(1, -1)
    if catalog_array.ndim == 1:
        catalog_array = catalog_array.reshape(1, -1)
    query_normalized = normalize(query_array, norm="l2", copy=True)
    catalog_normalized = normalize(catalog_array, norm="l2", copy=True)
    return np.asarray(query_normalized @ catalog_normalized.T)


def _top_k(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    if k <= 0:
        raise ValueError("k must be positive")
    effective_k = min(k, scores.shape[1])
    indices = np.argsort(-scores, axis=1, kind="stable")[:, :effective_k]
    values = np.take_along_axis(scores, indices, axis=1)
    return indices.astype(int), values.astype(float)


def exact_top_k_cosine(
    query_matrix: Any,
    catalog_matrix: Any,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact cosine top-k catalog row indices and scores per query."""
    return _top_k(_cosine_scores(query_matrix, catalog_matrix), k)


def _rank_lookup(indices: np.ndarray) -> list[dict[int, int]]:
    return [
        {int(catalog_index): rank for rank, catalog_index in enumerate(row, start=1)}
        for row in indices
    ]


def retrieve_candidates(
    listings: pd.DataFrame,
    catalog: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    catalog_tfidf: sparse.spmatrix,
    query_embeddings: np.ndarray,
    catalog_embeddings: np.ndarray,
    top_k: int = 10,
    gold_matches: pd.DataFrame | None = None,
    inject_gold: bool = False,
) -> pd.DataFrame:
    """Return the union of lexical and dense exact top-k candidates.

    Gold candidates are optionally injected only for training. Injected rows are
    explicitly marked so they cannot inflate retrieval-recall measurements.
    """
    if len(listings) != len(query_embeddings):
        raise ValueError("query_embeddings must have one row per listing")
    if (
        len(catalog) != catalog_tfidf.shape[0]
        or len(catalog) != catalog_embeddings.shape[0]
    ):
        raise ValueError("Catalog matrices must have one row per catalog product")
    if inject_gold and gold_matches is None:
        raise ValueError("gold_matches are required when inject_gold=True")
    if listings.empty:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)

    query_tfidf = vectorizer.transform(combine_text_fields(listings)).tocsr()
    lexical_scores = _cosine_scores(query_tfidf, catalog_tfidf)
    dense_scores = _cosine_scores(query_embeddings, catalog_embeddings)
    lexical_indices, _ = _top_k(lexical_scores, top_k)
    dense_indices, _ = _top_k(dense_scores, top_k)
    lexical_ranks = _rank_lookup(lexical_indices)
    dense_ranks = _rank_lookup(dense_indices)

    catalog_ids = catalog["product_id"].astype(str).tolist()
    catalog_positions = {product_id: index for index, product_id in enumerate(catalog_ids)}
    gold_by_google: dict[str, list[str]] = {}
    if gold_matches is not None:
        gold_by_google = (
            gold_matches.groupby("google_id", sort=False)["amazon_id"]
            .agg(lambda values: list(dict.fromkeys(values.astype(str))))
            .to_dict()
        )

    rows: list[dict[str, object]] = []
    for query_index, listing in enumerate(listings.itertuples(index=False)):
        google_id = str(getattr(listing, "product_id"))
        candidate_positions = set(map(int, lexical_indices[query_index]))
        candidate_positions.update(map(int, dense_indices[query_index]))
        injected_positions: set[int] = set()
        if inject_gold:
            for amazon_id in gold_by_google.get(google_id, []):
                position = catalog_positions.get(amazon_id)
                if position is not None and position not in candidate_positions:
                    candidate_positions.add(position)
                    injected_positions.add(position)

        def sort_key(position: int) -> tuple[float, int]:
            ranks = [
                rank
                for rank in (
                    lexical_ranks[query_index].get(position),
                    dense_ranks[query_index].get(position),
                )
                if rank is not None
            ]
            return (min(ranks) if ranks else float("inf"), position)

        for catalog_index in sorted(candidate_positions, key=sort_key):
            lexical_rank = lexical_ranks[query_index].get(catalog_index)
            dense_rank = dense_ranks[query_index].get(catalog_index)
            sources = []
            if lexical_rank is not None:
                sources.append("lexical")
            if dense_rank is not None:
                sources.append("dense")
            if catalog_index in injected_positions:
                sources.append("gold_injected")
            rows.append(
                {
                    "google_id": google_id,
                    "amazon_id": catalog_ids[catalog_index],
                    "lexical_score": float(lexical_scores[query_index, catalog_index]),
                    "dense_score": float(dense_scores[query_index, catalog_index]),
                    "lexical_rank": lexical_rank,
                    "dense_rank": dense_rank,
                    "retrieval_sources": "+".join(sources),
                    "gold_injected": catalog_index in injected_positions,
                }
            )

    candidates = pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)
    candidates["lexical_rank"] = candidates["lexical_rank"].astype("Int64")
    candidates["dense_rank"] = candidates["dense_rank"].astype("Int64")
    return candidates
