# Cold-Start Catalog Product Identity with Abstention

This compact entity-resolution project treats Google product rows as incoming
seller listings and Amazon rows as an existing canonical catalog. It retrieves
plausible candidates, estimates whether each candidate is the same underlying
product, and chooses one of three operational actions: `auto_match`,
`auto_reject`, or `manual_review`.

The project is intentionally interview-sized. It demonstrates retrieval,
interpretable pair matching, calibrated probabilities, and validation-selected
abstention without an API, database, model registry, or deployment stack.

## Business problem

Product identity asks whether two records describe the same sellable product,
including the same model or variant. This differs from search relevance: a
compatible accessory or a newer edition can be highly relevant to a query but
must still be rejected as a catalog identity match.

Abstention makes that distinction operational. Confident matches can be linked,
confident no-match listings can be rejected, and ambiguous variants can be sent
to a human reviewer.

## Dataset and task construction

The project uses the public
[Amazon–Google Products entity-resolution benchmark](https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution).
The pipeline downloads the two source tables and the perfect mapping from the
official archive. It uses all Google rows, including listings with no gold
Amazon partner, so automatic rejection is evaluated on natural no-match cases.

Gold links form a bipartite graph and are not one-to-one. Google listings in the
same connected component are kept in one deterministic 70%/15%/15% split. The
Amazon table is a fixed reference catalog available to every split, matching the
cold-start setting. The character TF-IDF vectorizer is fitted on the known
catalog and training Google text only; validation and test Google text do not
influence fitted preprocessing.

## Approach

Candidate retrieval unions two exact-search methods:

- Character n-gram TF-IDF cosine similarity.
- Frozen `sentence-transformers/all-MiniLM-L6-v2` cosine similarity.

Training candidates use retrieved hard negatives and inject missing gold
partners only for training. Validation and test retrieval are never forced.

The matcher is sigmoid-calibrated logistic regression over a small feature set:
lexical and dense cosine scores, title overlap and fuzzy similarity,
manufacturer agreement, model-number overlap/conflict, relative price
difference, and missing-value indicators. A validation grid chooses match and
reject thresholds that maximize automatic coverage while targeting 95% precision
for both automatic actions.

## Verified results

These values come from the successful real-data run on 2026-08-25 with seed 42.
The held-out test split contains 484 Google listings, including 192 with at
least one gold Amazon match. The authoritative full output is
[`reports/metrics.json`](reports/metrics.json).

### Retrieval

Recall is the fraction of gold-bearing listings for which any correct Amazon
partner was retrieved. “Union” combines each retriever's top-K results, so it
can contain up to `2K` distinct candidates.

| Retriever | Recall@5 | Recall@10 | Recall@20 |
|---|---:|---:|---:|
| Character TF-IDF | 0.880 | 0.938 | 0.974 |
| MiniLM dense | 0.875 | 0.943 | 0.964 |
| Union | **0.953** | **0.974** | **0.995** |

At the configured top-20 cutoff, the deduplicated union averaged 32.93
candidates per listing. Warm per-listing retrieval averaged 13.94 ms with a
15.16 ms p95 in this local run; this includes query text transformation, dense
encoding, and both exact searches, but excludes model loading and catalog
precomputation. One of 192 matched test listings was missed by the union.

### Matcher comparison and held-out pair metrics

Validation PR-AUC increased from 0.316 for raw lexical cosine to 0.471 for the
lexical logistic model and 0.560 for the hybrid logistic model. The hybrid was
selected. Its validation policy met both configured precision constraints with
thresholds of 0.97 for matching and 0.01 for rejection.

| Test pair metric | Value |
|---|---:|
| PR-AUC | 0.543 |
| ROC-AUC | 0.988 |
| Precision at 0.5 | 0.677 |
| Recall at 0.5 | 0.438 |
| F1 at 0.5 | 0.532 |
| Brier score | 0.0074 |

PR-AUC is reported as scikit-learn average precision, the standard non-
interpolated summary of the precision-recall curve.

Pair metrics are conditional on the unforced retrieved candidate set. The
end-to-end metric below separately counts retrieval failures.

### Abstention policy

| Held-out listing metric | Value |
|---|---:|
| Auto-match precision | 0.667 |
| Auto-match coverage | 0.006 |
| Auto-reject precision | 0.974 |
| Auto-reject coverage | 0.318 |
| Manual-review rate | 0.676 |
| Accuracy on automatic decisions | 0.968 |
| End-to-end successful-match rate | 0.010 |

The 95% targets are validation constraints, not guarantees. On test, only three
listings crossed the very conservative match threshold and two were correct, so
auto-match precision fell to 66.7%. Most listings were reviewed, and only 1.0%
of gold-bearing listings ended as correct automatic matches despite 99.5%
retrieval recall@20. This is an important negative result: the model ranking is
useful, but this small validation set did not support a high-coverage,
high-precision automatic-match policy.

## Error analysis

[`reports/error_examples.csv`](reports/error_examples.csv) contains ten examples
each for high-confidence false positives, low-scored gold pairs, likely similar
variants, numeric/model conflicts, and errors involving missing manufacturer or
price, plus the one retrieval miss.

- The highest-scoring false positive paired two `macbackup` records with the
  same title and price but no benchmark gold link. The Google manufacturer is
  missing while Amazon says `macware`; this illustrates both missing-field risk
  and possible ambiguity in old offline labels.
- Similar-title products can still be different variants. An observed example
  paired `norton antivirus 2004` with `norton antivirus 2007`; the explicit model
  conflict kept it in manual review rather than auto-match.
- Several gold pairs have large naming and price shifts. `tinyterm v.4.3x` and an
  Acrobat 8 upgrade example received probabilities below 0.004 and were
  automatically rejected.
- Google manufacturer is absent for most source records, so the matcher often
  cannot use brand agreement to resolve otherwise identical titles.
- The sole retrieval miss was `simply magazine sales skills` versus Amazon's
  `sales skills 2.0 ages 10+`, where both lexical and dense retrieval failed to
  place the gold item in the union top 20.

These categories are diagnostic slices rather than statistically complete root
causes. The similar-variant category is a heuristic based on high title
similarity plus a model-token conflict or substantial price difference.

## Install and run

Python 3.12 is recommended. The following `uv` commands were used for the
verified run and keep the package cache inside the project:

```bash
UV_CACHE_DIR=.uv-cache uv venv --python 3.12
source .venv/bin/activate
UV_CACHE_DIR=.uv-cache uv pip install --python .venv/bin/python -r requirements.txt
python run_pipeline.py
pytest -q
streamlit run app.py
```

If Python 3.12 and `pip` are already installed, a standard virtual environment
works too: replace the first and third commands with `python3.12 -m venv .venv`
and `python -m pip install -r requirements.txt`.

The first pipeline run needs network access for the dataset and frozen sentence
encoder. Later runs reuse the downloaded data and normal model cache. The demo
expects artifacts produced by `python run_pipeline.py`.

## Outputs

- `reports/metrics.json`: held-out retrieval, pair, and abstention metrics.
- `reports/model_comparison.csv`: validation model comparison.
- `reports/error_examples.csv`: selected real test errors.
- `reports/*.png`: retrieval, precision-recall, and reliability plots.
- `artifacts/`: the small set of files required by the Streamlit demo.

## Limitations and production extensions

- The benchmark is small and old, so its language, catalog mix, and noise do not
  represent a modern marketplace.
- Offline pair labels do not reproduce every seller-listing problem, open-world
  catalog change, or downstream business cost.
- Precision targets and manual-review capacity are illustrative rather than
  estimated from a real operation.
- Exact cosine search is appropriate here but would need an approximate nearest
  neighbour index at large catalog scale.
- The model uses text and price only. A production matcher could add category,
  brand normalization, stronger model-number parsing, images, and reviewer
  feedback.
- The system is an interview project, not a production catalog service.
