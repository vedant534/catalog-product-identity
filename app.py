"""Minimal Streamlit demo for catalog product identity matching."""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import yaml
from scipy import sparse

from src.features import build_pair_features
from src.model import predict_probabilities, select_top_candidates
from src.policy import apply_policy
from src.retrieval import encode_products, load_sentence_encoder, retrieve_candidates


def read_config() -> dict:
    with Path("config.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@st.cache_resource
def load_demo_assets(config: dict):
    artifact_dir = Path(config["artifacts_dir"])
    required = {
        "vectorizer": artifact_dir / "tfidf_vectorizer.joblib",
        "catalog": artifact_dir / "amazon_catalog.csv",
        "tfidf": artifact_dir / "catalog_tfidf.npz",
        "dense": artifact_dir / "catalog_dense.npy",
        "matcher": artifact_dir / "matcher.joblib",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing pipeline artifacts: "
            + ", ".join(missing)
            + ". Run python run_pipeline.py --stage develop first."
        )

    vectorizer = joblib.load(required["vectorizer"])
    catalog = pd.read_csv(required["catalog"], dtype={"product_id": str})
    catalog_tfidf = sparse.load_npz(required["tfidf"])
    catalog_dense = np.load(required["dense"])
    matcher = joblib.load(required["matcher"])
    encoder = load_sentence_encoder(matcher["sentence_model_name"])
    return vectorizer, catalog, catalog_tfidf, catalog_dense, matcher, encoder


def parse_optional_price(value: str) -> float:
    if not value.strip():
        return float("nan")
    price = float(value)
    if not np.isfinite(price) or price <= 0:
        raise ValueError("Price must be a positive number or left blank.")
    return price


def main() -> None:
    config = read_config()
    cache_dir = Path(config["model_cache_dir"]).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    st.title("Catalog Product Identity")
    st.caption("Retrieve Amazon candidates, estimate identity, and abstain when uncertain.")

    try:
        assets = load_demo_assets(config)
    except Exception as error:
        st.error(str(error))
        st.stop()
    vectorizer, catalog, catalog_tfidf, catalog_dense, matcher, encoder = assets

    with st.form("listing_form"):
        title = st.text_input("Product title")
        manufacturer = st.text_input("Manufacturer (optional)")
        description = st.text_area("Description (optional)")
        price_text = st.text_input("Price (optional)")
        submitted = st.form_submit_button("Find catalog identity")

    if not submitted:
        return
    if not any(value.strip() for value in (title, manufacturer, description)):
        st.warning("Enter at least a title, manufacturer, or description.")
        return
    try:
        price = parse_optional_price(price_text)
    except ValueError as error:
        st.error(str(error))
        return

    query = pd.DataFrame(
        [
            {
                "product_id": "user_query",
                "title": title,
                "manufacturer": manufacturer,
                "description": description,
                "price": price,
                "source": "google",
            }
        ]
    )
    query_embedding = encode_products(query, encoder, batch_size=1)
    candidates = retrieve_candidates(
        query,
        catalog,
        vectorizer,
        catalog_tfidf,
        query_embedding,
        catalog_dense,
        top_k=int(matcher["top_k"]),
        rrf_constant=float(matcher["rrf_constant"]),
    )
    features = build_pair_features(
        candidates,
        query,
        catalog,
        lexical_scores=candidates["lexical_score"].to_numpy(),
        dense_scores=candidates["dense_score"].to_numpy(),
        record_id_column="product_id",
    )
    probabilities = predict_probabilities(matcher, features)
    results = candidates.copy()
    results["probability"] = probabilities
    results = results.merge(
        catalog[
            ["product_id", "title", "manufacturer", "description", "price"]
        ].rename(columns={"product_id": "amazon_id"}),
        on="amazon_id",
        how="left",
    )

    best = select_top_candidates(results).iloc[0]
    policy = matcher["policy"]
    decision = apply_policy(
        float(best["probability"]),
        match_threshold=(
            policy["auto_match"]["threshold"]
            if policy["auto_match"]["enabled"]
            else None
        ),
        no_match_threshold=(
            policy["auto_no_match"]["threshold"]
            if policy["auto_no_match"]["enabled"]
            else None
        ),
    )
    st.subheader(f"Decision: {decision}")
    st.write(
        f"Best candidate: **{best['title']}**  \n"
        f"Pair match score: **{best['probability']:.3f}**"
    )
    results = results.sort_values(
        ["probability", "amazon_id"], ascending=[False, True], kind="mergesort"
    )
    display_columns = [
        "amazon_id",
        "title",
        "manufacturer",
        "price",
        "probability",
        "lexical_score",
        "dense_score",
    ]
    st.dataframe(results[display_columns].reset_index(drop=True), width="stretch")


if __name__ == "__main__":
    main()
