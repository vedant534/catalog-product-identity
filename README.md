# Cold-Start Catalog Product Identity with Abstention

This compact entity-resolution project treats Google product rows as incoming
seller listings and Amazon rows as a fixed reference catalog. It retrieves a
fixed number of catalog candidates, scores each candidate pair with a small
logistic matcher, selects one top candidate per listing, and emits one of three
listing-level actions: `auto_match`, `auto_no_match`, or `manual_review`.

The project is intentionally interview-sized. It keeps exact retrieval, a
transparent feature set, grouped calibration, and a simple Streamlit demo. It
does not add an API, database, model registry, approximate index, deployment
stack, or broader model family.

## Development checkpoint

The duplicate-aware correction pass has completed its development stage only.
All values below are training/validation results from `split_seed: 20260825`
and `model_seed: 42`. The complete final-test membership has been saved, but
during this corrected development run no test listing has been encoded,
retrieved, featured, scored, evaluated, or inspected. The final holdout remains
unrun pending separate approval. Previously published pre-correction test
results are treated only as development diagnostics, not final evidence.

The authoritative machine-readable development output is
[`reports/metrics.json`](reports/metrics.json).

## Dataset, split, and label assumptions

The project uses the public
[Amazon–Google Products entity-resolution benchmark](https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution).
The normalized data contains 1,363 Amazon products, 3,226 Google listings, and
1,300 official gold pairs. The Amazon catalog remains available in every split.

Before gold-link components are assigned, Google records are grouped by an
exact normalized signature of title, manufacturer, description, and parsed
price. No fuzzy duplicate clustering is used. The resulting split has 2,258
training listings, 484 validation listings, and 484 reserved final-test
listings. It contains 93 exact-duplicate groups covering 234 rows; no duplicate
group crosses a split. One unmapped member of a mixed mapped/unmapped duplicate
group is flagged as ambiguous and excluded as a definite training negative.

The official benchmark is the primary evaluation. Under its closed-world
assumption, a Google listing absent from the mapping is treated as no-match.
That assumption is imperfect: an unmapped listing can still collide with an
Amazon title. A secondary sensitivity result therefore excludes exact
normalized-title collisions without relabelling them as positives. Fuzzy
collisions are not silently relabelled.

## Method

### Retrieval

Two exact-search channels retrieve candidates:

- Character n-gram TF-IDF cosine similarity. The vectorizer is fitted on the
  fixed Amazon catalog plus training Google text only.
- Frozen `sentence-transformers/all-MiniLM-L6-v2` cosine similarity.

Reciprocal-rank fusion with constant 60 combines the two channel rankings and
truncates them to exactly 20 candidates per listing. This fixed-budget `RRF@20`
set is the primary retriever. The lexical-top-20 plus dense-top-20 union, which
can contain up to 40 distinct candidates, remains clearly labelled as a
diagnostic only. Missing gold candidates are injected only into training; 13
training listings required injection. Validation retrieval is unforced.

### Matcher and top-candidate decision

Four predeclared sigmoid-calibrated logistic variants are compared on
validation: lexical, dense, the current hybrid feature set, and hybrid plus
three simple retrieval-rank signals. Selection uses validation overall Hit@1,
then MRR, then fewer features, then declared model order. The final test is not
used for model selection.

Every candidate receives a pair match score. Exactly one top candidate is then
selected per listing by score descending and Amazon ID ascending. Thresholds
and operational actions apply only to that row; non-top candidates retain
scores but receive no action. The same rule is used in validation, reporting,
error analysis, final-test code, and Streamlit.

### Ranking metric definitions

- **Overall Hit@1** is evaluated only over gold-bearing listings. Retrieval and
  reranking misses contribute zero.
- **Conditional Hit@1** is evaluated only over gold-bearing listings for which
  at least one gold Amazon partner was retrieved.
- **MRR** is evaluated only over gold-bearing listings. If a listing has
  multiple gold partners, the highest-ranked gold partner is used; retrieval
  misses contribute zero.

## Validation results

### Fixed-budget retrieval

Recall is the fraction of validation gold-bearing listings for which at least
one gold Amazon partner is present.

| Retriever | Recall@5 | Recall@10 | Recall@20 |
|---|---:|---:|---:|
| Character TF-IDF | 0.9158 | 0.9474 | 0.9684 |
| MiniLM dense | 0.9421 | 0.9632 | 0.9947 |
| Fixed-budget RRF | 0.9474 | 0.9737 | 0.9947 |
| Union@K-per-channel diagnostic | 0.9842 | 0.9895 | 1.0000 |

The primary RRF set contains exactly 20 candidates for each of the 484
validation listings. It retrieved a gold partner for 189 of 190 gold-bearing
listings.

### Model comparison and ranking

| Logistic variant | Features | Overall Hit@1 | Conditional Hit@1 | MRR |
|---|---:|---:|---:|---:|
| Lexical | 5 | 0.8526 | 0.8571 | 0.9062 |
| Dense | 1 | 0.6737 | 0.6772 | 0.7902 |
| **Current hybrid (selected)** | **15** | **0.8684** | **0.8730** | **0.9254** |
| Hybrid plus rank signals | 18 | 0.8474 | 0.8519 | 0.9162 |

For the selected hybrid, 165 of 190 gold-bearing listings place a gold partner
first. Of the 25 remaining gold-bearing listings, one is a retrieval miss and
24 are reranking misses. The validation pair PR-AUC is 0.6441; pair-level
metrics remain matcher diagnostics rather than listing-policy results.

### Pair-level and top-candidate calibration

The model is calibrated on candidate pairs, but the operational policy acts on
the maximum score per listing. The two diagnostics are therefore reported
separately and the UI calls the value a **pair match score**, not a calibrated
match probability.

| Calibration diagnostic | Rows | Positive rate | Mean score | Brier score |
|---|---:|---:|---:|---:|
| Candidate pairs | 9,680 | 0.0197 | 0.0183 | 0.0103 |
| Top candidate per listing | 484 | 0.3409 | 0.2321 | 0.1453 |

### Closed-world no-match detection

| Evaluation | Listings | No-match AP | Policy precision | Policy recall |
|---|---:|---:|---:|---:|
| Primary official labels | 484 | 0.9483 | 0.9515 | 0.7347 |
| Exact-title sensitivity | 478 | 0.9542 | 0.9515 | 0.7500 |

The sensitivity row excludes six officially unmapped validation listings with
an exact normalized-title Amazon collision. It does not convert them to
positives.

### Independent automatic-action feasibility

Each action is screened independently on validation top-candidate rows. A
threshold is feasible only when its action region has at least 20 decisions and
empirical precision of at least 0.95. Wilson intervals are uncertainty
diagnostics, not feasibility constraints or population guarantees. A disabled
action uses a null threshold and cannot emit decisions.

When both independently selected actions overlap, compatible feasible pairs are
compared by combined coverage. Exact ties prefer the lower no-match threshold,
then the higher match threshold. If only one action can be enabled, coverage
and empirical precision are compared before the stable action order
`auto_no_match`, then `auto_match`.

| Action | Enabled | Threshold | Support | Correct | Errors | Empirical precision | 95% Wilson interval | Coverage |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `auto_match` | No | null | 0 | 0 | 0 | — | — | 0.0000 |
| `auto_no_match` | Yes | 0.03 | 227 | 216 | 11 | 0.9515 | [0.9153, 0.9727] | 0.4690 |

`both_constraints_met` is false because no auto-match threshold met both the
precision and support requirements. The policy therefore enables only
`auto_no_match`; all other score regions become `manual_review`. Overall, 257
of 484 listings are reviewed. Review rates are 179/190 (0.9421) for mapped
listings and 78/294 (0.2653) for assumed-no-match listings.

The compact threshold diagnostic is
[`reports/validation_precision_coverage.csv`](reports/validation_precision_coverage.csv).

## Reports and artifacts

- [`reports/metrics.json`](reports/metrics.json): complete development metrics
  and explicit final-test access flags.
- [`reports/model_comparison.csv`](reports/model_comparison.csv): the four
  validation model variants and selection result.
- [`reports/validation_listing_predictions.csv`](reports/validation_listing_predictions.csv):
  one authoritative top-candidate/action row per validation listing.
- [`reports/validation_error_examples.csv`](reports/validation_error_examples.csv):
  listing-policy error categories kept separate from pair-score diagnostics.
- [`reports/validation_precision_coverage.csv`](reports/validation_precision_coverage.csv):
  supported and unsupported threshold points with the selected row marked.
- `reports/validation_*.png`: fixed-budget retrieval, pair PR, pair-level
  reliability, and top-candidate reliability plots.
- `artifacts/`: frozen vectorizer, catalog matrices, matcher, split assignment,
  policy, and configuration snapshot for the separately authorized final stage
  and Streamlit demo.

## Install and run

Python 3.12 is recommended. Using `uv`:

```bash
UV_CACHE_DIR=.uv-cache uv venv --python 3.12
source .venv/bin/activate
UV_CACHE_DIR=.uv-cache uv pip install --python .venv/bin/python -r requirements.txt
pytest -q
python run_pipeline.py --stage develop
streamlit run app.py
```

The first development run needs network access only if the benchmark or frozen
sentence encoder is not already cached. The project also implements the frozen
holdout command:

```bash
python run_pipeline.py --stage final-test
```

Do not invoke that command until the development implementation and validation
outputs have received explicit approval. The final test is intended to be
evaluated exactly once without changing retrieval, features, model, thresholds,
ambiguity handling, or reporting afterward.

## Limitations

- The benchmark is small and old, and mapping absence is only a closed-world
  no-match assumption.
- High validation retrieval recall does not imply a high-coverage automatic
  match policy; the current validation evidence supports no automatic matches.
- Pair-level calibration does not establish calibration of the selected maximum
  score, so both views are reported separately.
- The precision thresholds and minimum support are development criteria, not
  claims of 95% population precision.
- Exact cosine search and the small Streamlit UI are appropriate for this
  portfolio-scale project, not a production catalog service.
