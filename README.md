# Cold-Start Catalog Product Identity with Abstention

This compact entity-resolution project treats Google product rows as incoming
seller listings and Amazon rows as a fixed reference catalog. It retrieves a
fixed number of catalog candidates, scores each candidate pair with a small
logistic matcher, selects one top candidate per listing, and emits one of three
listing-level actions: `auto_match`, `auto_no_match`, or `manual_review`.

The project is intentionally interview-sized. It keeps exact retrieval, a
transparent feature set, grouped calibration, and a simple Streamlit demo. The
Streamlit app is a local portfolio demo, not a hardened public service. The
project does not add an API, database, model registry, approximate index,
deployment stack, or broader model family.

## Development checkpoint

The duplicate-aware correction pass has completed its development stage only.
All values below are training/validation results from `split_seed: 20260825`
and `model_seed: 42`. The corrected-resplit assignments are frozen, but during
this development run no reserved evaluation listing was encoded, retrieved,
featured, scored, evaluated, or inspected. The predeclared corrected resplit
evaluation remains unrun pending separate approval.

> Because this benchmark was previously evaluated under seed 42 before the leakage issue was discovered, the reserved evaluation is a predeclared corrected resplit, not a historically untouched holdout. Validation results are tuning-set evidence, and the corrected-resplit result is a transparent secondary confirmation rather than pristine independent evidence.

Previously published pre-correction results are development diagnostics only.
Repartitioning does not erase that history or restore historical independence.

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
training listings, 484 validation listings, and 484 reserved corrected-resplit
evaluation listings. It contains 93 exact-duplicate groups covering 234 rows;
no duplicate group crosses a split. One unmapped member of a mixed
mapped/unmapped duplicate group is flagged as ambiguous and excluded as a
definite training negative.

The official benchmark mapping is the primary label view. Under its closed-world
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
- Frozen `sentence-transformers/all-MiniLM-L6-v2` cosine similarity at revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.

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
then MRR, then fewer features, then declared model order. The corrected-resplit
evaluation is not used for model selection. Abstention is a downstream
risk-control layer; it does not replace the predeclared ranking objective.

Every candidate receives a pair match score. Exactly one top candidate is then
selected per listing by score descending and Amazon ID ascending. Thresholds
and operational actions apply only to that row; non-top candidates retain
scores but receive no action. The same rule is used in validation, reporting,
error analysis, corrected-evaluation code, and Streamlit.

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
| Primary official labels | 484 | 0.9483 | 0.9619 | 0.6871 |
| Exact-title sensitivity | 478 | 0.9542 | 0.9619 | 0.7014 |

The sensitivity row excludes six officially unmapped validation listings with
an exact normalized-title Amazon collision. It does not convert them to
positives.

### Independent automatic-action feasibility

After ambiguous listings are excluded, each action is screened independently
on validation top-candidate rows collapsed by exact `duplicate_group_id`. Each
unique signature group is one threshold-selection evidence unit. A
label-inconsistent group is counted conservatively as incorrect. A threshold
is feasible only when its group-level action region has at least 20 groups and
empirical precision of at least 0.95. The selected threshold is then applied to
every individual listing. Wilson intervals are uncertainty diagnostics, not
feasibility constraints or population guarantees. A disabled action uses a
null threshold and cannot emit decisions.

When both independently selected actions overlap, compatible feasible pairs are
compared by combined coverage. Exact ties prefer the lower no-match threshold,
then the higher match threshold. If only one action can be enabled, coverage
and empirical precision are compared before the stable action order
`auto_no_match`, then `auto_match`.

Group-level evidence used for threshold selection:

| Action | Enabled | Threshold | Groups | Correct | Errors | Empirical precision | 95% Wilson interval | Group coverage |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `auto_match` | No | null | 0 | 0 | 0 | — | — | 0.0000 |
| `auto_no_match` | Yes | 0.02 | 201 | 193 | 8 | 0.9602 | [0.9234, 0.9797] | 0.4258 |

Listing-level operational results after applying the group-selected threshold:

| Action | Support | Correct | Errors | Empirical precision | 95% Wilson interval | Listing coverage |
|---|---:|---:|---:|---:|---:|---:|
| `auto_match` | 0 | 0 | 0 | — | — | 0.0000 |
| `auto_no_match` | 210 | 202 | 8 | 0.9619 | [0.9266, 0.9806] | 0.4339 |

`both_constraints_met` is false because no auto-match threshold met both the
precision and support requirements. The policy therefore enables only
`auto_no_match`; all other score regions become `manual_review`. Overall, 274
of 484 listings are reviewed. Review rates are 182/190 (0.9579) for mapped
listings and 92/294 (0.3129) for assumed-no-match listings.

The compact threshold diagnostic is
[`reports/validation_precision_coverage.csv`](reports/validation_precision_coverage.csv).

## Reports and artifacts

- [`reports/metrics.json`](reports/metrics.json): complete development metrics
  and explicit corrected-resplit access flags.
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
  policy, and development snapshot for the separately authorized corrected
  evaluation and Streamlit demo. The snapshot records the encoder revision and
  SHA-256 digest of each official raw CSV.

The frozen input metadata records these exact development inputs:

| Input | SHA-256 or revision |
|---|---|
| MiniLM encoder | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| `Amazon.csv` | `cabc0379070c595eca65c4a69c77b3267b97f8a81b303e77115b52e6e534e65f` |
| `GoogleProducts.csv` | `18fbec453670e40e0969fdaffe71e92ba62a24c49b0c8341fe4621e70e402e3f` |
| `Amzon_GoogleProducts_perfectMapping.csv` | `885eda7da34ed00809975d34452800e7fbde9ef0540fb60474a18dca15fa4fe7` |

The future corrected evaluation is additive and writes only to its own bundle:

- `reports/corrected_resplit/metrics.json`
- `reports/corrected_resplit/listing_predictions.csv`
- `reports/corrected_resplit/error_examples.csv`
- `reports/corrected_resplit/corrected_resplit_*.png`

It does not rewrite development metrics, model comparison, threshold evidence,
validation predictions, validation errors, or validation plots. Before writing,
the stage checks the frozen artifacts, raw inputs, assignments, and development
reports. It refuses to overwrite an existing corrected-resplit bundle. A
deliberate rerun must use a different output directory.

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

The first development run needs network access only if the benchmark or pinned
sentence encoder revision is not already cached. The project also implements
the predeclared corrected resplit command:

```bash
python run_pipeline.py --stage corrected-eval
```

Do not invoke that command until the development implementation and validation
outputs have received explicit approval. Corrected evaluation loads the pinned
encoder revision offline and writes only the separate bundle above. To make a
deliberate rerun without overwriting that bundle, specify another directory:

```bash
python run_pipeline.py --stage corrected-eval --output-dir reports/corrected_resplit_rerun
```

The Streamlit app loads locally generated `joblib` artifacts from `artifacts/`.
It must never load model artifacts supplied through uploads or from untrusted
sources.

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
