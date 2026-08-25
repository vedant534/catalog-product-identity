"""Exact lexical and dense candidate retrieval for the small benchmark."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


TEXT_FIELDS = ("title", "manufacturer", "description")
DEFAULT_RRF_CONSTANT = 60.0
CANDIDATE_COLUMNS = [
    "google_id",
    "amazon_id",
    "lexical_score",
    "dense_score",
    "lexical_rank",
    "dense_rank",
    "rrf_score",
    "rrf_rank",
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


def _candidate_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    candidates = pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)
    for column in ("lexical_rank", "dense_rank", "rrf_rank"):
        candidates[column] = candidates[column].astype("Int64")
    return candidates


def _rrf_score(
    lexical_rank: int | None,
    dense_rank: int | None,
    constant: float,
) -> float:
    return sum(
        1.0 / (constant + rank)
        for rank in (lexical_rank, dense_rank)
        if rank is not None
    )


def retrieve_candidates_with_diagnostics(
    listings: pd.DataFrame,
    catalog: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    catalog_tfidf: sparse.spmatrix,
    query_embeddings: np.ndarray,
    catalog_embeddings: np.ndarray,
    top_k: int = 10,
    gold_matches: pd.DataFrame | None = None,
    inject_gold: bool = False,
    rrf_constant: float = DEFAULT_RRF_CONSTANT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return fixed-budget RRF candidates and the raw per-channel union.

    The first frame contains exactly ``min(top_k, len(catalog))`` fused rows per
    listing before optional gold injection. The second frame is the unforced raw
    union of lexical top-k and dense top-k rows (up to ``2 * top_k``), intended
    for per-channel and union-per-channel retrieval diagnostics.

    Gold candidates are optionally injected into the first frame only. This is a
    training-only caller choice: injected rows are explicitly marked, while the
    diagnostic union always remains unforced.
    """
    if not np.isfinite(rrf_constant) or rrf_constant < 0:
        raise ValueError("rrf_constant must be a finite non-negative number")
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
        empty = _candidate_frame([])
        return empty.copy(), empty

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
        gold = gold_matches[["google_id", "amazon_id"]].astype(str)
        gold_by_google = (
            gold.groupby("google_id", sort=False)["amazon_id"]
            .agg(lambda values: list(dict.fromkeys(values)))
            .to_dict()
        )

    fused_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    for query_index, listing in enumerate(listings.itertuples(index=False)):
        google_id = str(getattr(listing, "product_id"))
        candidate_positions = set(map(int, lexical_indices[query_index]))
        candidate_positions.update(map(int, dense_indices[query_index]))

        listing_union: list[dict[str, object]] = []
        for catalog_index in candidate_positions:
            lexical_rank = lexical_ranks[query_index].get(catalog_index)
            dense_rank = dense_ranks[query_index].get(catalog_index)
            sources = []
            if lexical_rank is not None:
                sources.append("lexical")
            if dense_rank is not None:
                sources.append("dense")
            listing_union.append(
                {
                    "google_id": google_id,
                    "amazon_id": catalog_ids[catalog_index],
                    "lexical_score": float(lexical_scores[query_index, catalog_index]),
                    "dense_score": float(dense_scores[query_index, catalog_index]),
                    "lexical_rank": lexical_rank,
                    "dense_rank": dense_rank,
                    "rrf_score": _rrf_score(
                        lexical_rank, dense_rank, rrf_constant
                    ),
                    "rrf_rank": None,
                    "retrieval_sources": "+".join(sources),
                    "gold_injected": False,
                }
            )

        listing_union.sort(
            key=lambda row: (-float(row["rrf_score"]), str(row["amazon_id"]))
        )
        for rank, row in enumerate(listing_union, start=1):
            row["rrf_rank"] = rank
        diagnostic_rows.extend(listing_union)

        fused = [dict(row) for row in listing_union[: min(top_k, len(catalog))]]
        fused_ids = {str(row["amazon_id"]) for row in fused}
        union_by_id = {str(row["amazon_id"]): row for row in listing_union}
        if inject_gold:
            for amazon_id in sorted(gold_by_google.get(google_id, [])):
                if amazon_id in fused_ids:
                    continue
                position = catalog_positions.get(amazon_id)
                if position is None:
                    continue
                if amazon_id in union_by_id:
                    injected = dict(union_by_id[amazon_id])
                else:
                    injected = {
                        "google_id": google_id,
                        "amazon_id": amazon_id,
                        "lexical_score": float(
                            lexical_scores[query_index, position]
                        ),
                        "dense_score": float(dense_scores[query_index, position]),
                        "lexical_rank": None,
                        "dense_rank": None,
                        "rrf_score": 0.0,
                        "rrf_rank": None,
                        "retrieval_sources": "",
                        "gold_injected": False,
                    }
                injected["gold_injected"] = True
                sources = str(injected["retrieval_sources"])
                injected["retrieval_sources"] = "+".join(
                    value for value in (sources, "gold_injected") if value
                )
                fused.append(injected)
                fused_ids.add(amazon_id)
        fused_rows.extend(fused)

    return _candidate_frame(fused_rows), _candidate_frame(diagnostic_rows)


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
    rrf_constant: float = DEFAULT_RRF_CONSTANT,
) -> pd.DataFrame:
    """Return fixed-budget reciprocal-rank-fused candidates.

    Validation and test callers should leave ``inject_gold=False``. Training may
    inject missing gold rows; every such row is marked by ``gold_injected=True``.
    Use :func:`retrieve_candidates_with_diagnostics` when the raw top-k-per-
    channel union is also needed for retrieval reporting.
    """
    candidates, _ = retrieve_candidates_with_diagnostics(
        listings,
        catalog,
        vectorizer,
        catalog_tfidf,
        query_embeddings,
        catalog_embeddings,
        top_k=top_k,
        gold_matches=gold_matches,
        inject_gold=inject_gold,
        rrf_constant=rrf_constant,
    )
    return candidates
