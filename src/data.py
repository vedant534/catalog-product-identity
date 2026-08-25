"""Download, normalize, and split the Amazon-Google Products benchmark."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DATASET_URL = "https://dbs.uni-leipzig.de/files/datasets/Amazon-GoogleProducts.zip"
ARCHIVE_NAME = "Amazon-GoogleProducts.zip"
DATASET_FILES = {
    "amazon": "Amazon.csv",
    "google": "GoogleProducts.csv",
    "gold": "Amzon_GoogleProducts_perfectMapping.csv",
}
PRODUCT_COLUMNS = [
    "product_id",
    "title",
    "manufacturer",
    "description",
    "price",
    "source",
]
SPLIT_NAMES = ("train", "validation", "test")
SPLIT_ASSIGNMENT_COLUMNS = (
    "google_id",
    "component_id",
    "split",
    "duplicate_group_id",
    "duplicate_group_size",
    "ambiguous_label",
)
DUPLICATE_TEXT_COLUMNS = ("title", "manufacturer", "description")
MISSING_PRICE_SIGNATURE = "<missing-price>"


def _normalized_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _locate_dataset_files(raw_dir: Path) -> dict[str, Path]:
    """Find the three benchmark CSVs, including the gold file's original typo."""
    found: dict[str, Path] = {}
    for key, filename in DATASET_FILES.items():
        path = raw_dir / filename
        if path.exists():
            found[key] = path

    for path in raw_dir.glob("*.csv") if raw_dir.exists() else []:
        name = _normalized_name(path.name)
        if name == "amazoncsv":
            found.setdefault("amazon", path)
        elif name == "googleproductscsv":
            found.setdefault("google", path)
        elif "perfectmapping" in name:
            found.setdefault("gold", path)
    return found


def official_raw_csv_sha256(raw_dir: str | Path) -> dict[str, str]:
    """Return SHA-256 digests for the three official benchmark CSV files."""
    raw_path = Path(raw_dir)
    paths = _locate_dataset_files(raw_path)
    missing = set(DATASET_FILES) - set(paths)
    if missing:
        names = ", ".join(DATASET_FILES[key] for key in sorted(missing))
        raise FileNotFoundError(f"Missing dataset file(s): {names}")

    digests: dict[str, str] = {}
    for key, official_name in DATASET_FILES.items():
        digest = hashlib.sha256()
        with paths[key].open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digests[official_name] = digest.hexdigest()
    return digests


def _archive_members(archive: zipfile.ZipFile) -> dict[str, str]:
    members: dict[str, str] = {}
    for member in archive.namelist():
        name = _normalized_name(Path(member).name)
        if name == "amazoncsv":
            members["amazon"] = member
        elif name == "googleproductscsv":
            members["google"] = member
        elif "perfectmapping" in name:
            members["gold"] = member
    return members


def download_dataset(
    raw_dir: str | Path,
    url: str = DATASET_URL,
) -> dict[str, Path]:
    """Download and extract the official ZIP only when a benchmark CSV is missing."""
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    existing = _locate_dataset_files(raw_path)
    if set(existing) == set(DATASET_FILES):
        return existing

    archive_path = raw_path / ARCHIVE_NAME
    if not archive_path.exists():
        partial_path = archive_path.with_suffix(".zip.part")
        request = urllib.request.Request(url, headers={"User-Agent": "catalog-identity/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                with partial_path.open("wb") as output:
                    shutil.copyfileobj(response, output)
            partial_path.replace(archive_path)
        except Exception:
            partial_path.unlink(missing_ok=True)
            raise

    with zipfile.ZipFile(archive_path) as archive:
        members = _archive_members(archive)
        missing_members = set(DATASET_FILES) - set(members)
        if missing_members:
            names = ", ".join(sorted(missing_members))
            raise FileNotFoundError(f"Official archive is missing expected file(s): {names}")

        for key, filename in DATASET_FILES.items():
            destination = raw_path / filename
            if not destination.exists():
                with archive.open(members[key]) as source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)

    return _locate_dataset_files(raw_path)


def _column(frame: pd.DataFrame, aliases: Iterable[str], table_name: str) -> pd.Series:
    columns = {_normalized_name(name): name for name in frame.columns}
    for alias in aliases:
        source_name = columns.get(_normalized_name(alias))
        if source_name is not None:
            return frame[source_name]
    expected = ", ".join(aliases)
    raise ValueError(f"{table_name} table is missing a required column ({expected})")


def _clean_text(values: pd.Series) -> pd.Series:
    return (
        values.astype("string")
        .fillna("")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def _clean_ids(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("").str.strip()


def _clean_prices(values: pd.Series) -> pd.Series:
    text = values.astype("string").fillna("").str.strip()
    # The benchmark mixes plain USD-like numbers with a small number of explicit
    # GBP prices. Cross-currency relative differences would be misleading, so
    # accept only plain numbers or explicit USD notation and leave the rest missing.
    amount_pattern = (
        r"(?:usd\s*|us\$\s*|\$\s*)?"
        r"([+-]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+))"
        r"(?:\s*usd)?"
    )
    valid = text.str.fullmatch(amount_pattern, case=False, na=False)
    amount = text.str.extract(amount_pattern, flags=re.IGNORECASE, expand=False)
    numeric = pd.to_numeric(amount.str.replace(",", "", regex=False), errors="coerce")
    return numeric.where(valid & numeric.gt(0)).astype(float)


def normalize_product_table(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """Map either source table to the shared, lightweight product schema."""
    source_key = source.lower().strip()
    if source_key not in {"amazon", "google"}:
        raise ValueError("source must be 'amazon' or 'google'")

    title_aliases = ("title", "name") if source_key == "amazon" else ("name", "title")
    normalized = pd.DataFrame(
        {
            "product_id": _clean_ids(_column(frame, ("id",), source_key)),
            "title": _clean_text(_column(frame, title_aliases, source_key)),
            "manufacturer": _clean_text(
                _column(frame, ("manufacturer",), source_key)
            ),
            "description": _clean_text(_column(frame, ("description",), source_key)),
            "price": _clean_prices(_column(frame, ("price",), source_key)),
            "source": source_key,
        }
    )
    normalized = normalized.loc[normalized["product_id"].ne("")]
    return normalized.drop_duplicates("product_id", keep="first").reset_index(drop=True)


def normalize_gold_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize the perfect-mapping columns to google_id and amazon_id."""
    normalized = pd.DataFrame(
        {
            "google_id": _clean_ids(
                _column(frame, ("idGoogleBase", "google_id", "idGoogle"), "gold")
            ),
            "amazon_id": _clean_ids(
                _column(frame, ("idAmazon", "amazon_id"), "gold")
            ),
        }
    )
    present = normalized["google_id"].ne("") & normalized["amazon_id"].ne("")
    return normalized.loc[present].drop_duplicates().reset_index(drop=True)


def load_dataset(
    raw_dir: str | Path,
    download_if_missing: bool = True,
    url: str = DATASET_URL,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load normalized Amazon catalog, Google listings, and gold match pairs."""
    raw_path = Path(raw_dir)
    paths = _locate_dataset_files(raw_path)
    if set(paths) != set(DATASET_FILES):
        if not download_if_missing:
            missing = ", ".join(sorted(set(DATASET_FILES) - set(paths)))
            raise FileNotFoundError(f"Missing dataset file(s): {missing}")
        paths = download_dataset(raw_path, url=url)

    read_options = {"encoding": "latin-1", "dtype": str, "low_memory": False}
    amazon_raw = pd.read_csv(paths["amazon"], **read_options)
    google_raw = pd.read_csv(paths["google"], **read_options)
    gold_raw = pd.read_csv(paths["gold"], **read_options)
    return (
        normalize_product_table(amazon_raw, "amazon"),
        normalize_product_table(google_raw, "google"),
        normalize_gold_table(gold_raw),
    )


class _UnionFind:
    """Minimal union-find used only to keep matched components in one split."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, item: str) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: str) -> str:
        self.add(item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        smaller, larger = sorted((left_root, right_root))
        self.parent[larger] = smaller


def _target_counts(total: int, ratios: tuple[float, float, float]) -> np.ndarray:
    raw = np.asarray(ratios, dtype=float) * total
    targets = np.floor(raw).astype(int)
    remainder = total - int(targets.sum())
    if remainder:
        order = np.argsort(-(raw - targets), kind="stable")
        targets[order[:remainder]] += 1
    return targets


def _duplicate_signature_frame(google: pd.DataFrame) -> pd.DataFrame:
    """Build exact, normalized signatures for the first row of each Google ID."""
    normalized_text = {
        column: _clean_text(google[column]).str.lower()
        for column in DUPLICATE_TEXT_COLUMNS
    }
    parsed_prices = _clean_prices(google["price"])
    price_signatures = parsed_prices.map(
        lambda value: (
            MISSING_PRICE_SIGNATURE
            if pd.isna(value)
            else float(value).hex()
        )
    )
    signatures = pd.DataFrame(
        {
            "google_id": _clean_ids(google["product_id"]),
            **normalized_text,
            "price_signature": price_signatures,
        }
    ).drop_duplicates("google_id", keep="first")
    signatures["duplicate_signature"] = list(
        zip(
            signatures["title"],
            signatures["manufacturer"],
            signatures["description"],
            signatures["price_signature"],
        )
    )
    return signatures[["google_id", "duplicate_signature"]]


def _stable_duplicate_group_id(signature: tuple[str, str, str, str]) -> str:
    """Return an order-independent identifier derived only from the signature."""
    payload = json.dumps(signature, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"duplicate_{digest[:16]}"


def make_entity_splits(
    google: pd.DataFrame,
    gold: pd.DataFrame,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> pd.DataFrame:
    """Assign every Google listing to a deterministic connected-component split.

    Amazon remains a fixed catalog. Exact duplicate Google rows and gold-linked
    Google records sharing any Amazon record stay together. Unmapped members of a
    mixed mapped/unmapped duplicate group are flagged as label-ambiguous.
    """
    if len(ratios) != len(SPLIT_NAMES) or any(value < 0 for value in ratios):
        raise ValueError("ratios must contain three non-negative values")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("split ratios must sum to 1")

    google_ids = _clean_ids(google["product_id"]).drop_duplicates().tolist()
    google_id_set = set(google_ids)
    links = _UnionFind()
    for google_id in google_ids:
        links.add(f"g:{google_id}")

    signature_groups: defaultdict[tuple[str, str, str, str], list[str]] = (
        defaultdict(list)
    )
    signature_frame = _duplicate_signature_frame(google)
    for row in signature_frame.itertuples(index=False):
        if row.google_id in google_id_set:
            signature_groups[row.duplicate_signature].append(row.google_id)

    # Exact duplicate edges are added before gold edges so records with missing
    # mappings cannot be separated from otherwise identical mapped records.
    for duplicate_ids in signature_groups.values():
        ordered_ids = sorted(duplicate_ids)
        for duplicate_id in ordered_ids[1:]:
            links.union(f"g:{ordered_ids[0]}", f"g:{duplicate_id}")

    mapped_google_ids: set[str] = set()
    for pair in gold[["google_id", "amazon_id"]].itertuples(index=False):
        google_id = str(pair.google_id).strip()
        amazon_id = str(pair.amazon_id).strip()
        if google_id in google_id_set and amazon_id:
            mapped_google_ids.add(google_id)
            links.union(f"g:{google_id}", f"a:{amazon_id}")

    duplicate_metadata: dict[str, tuple[str, int, bool]] = {}
    for signature, duplicate_ids in signature_groups.items():
        ordered_ids = sorted(duplicate_ids)
        group_id = _stable_duplicate_group_id(signature)
        group_size = len(ordered_ids)
        has_mapped = any(item in mapped_google_ids for item in ordered_ids)
        has_unmapped = any(item not in mapped_google_ids for item in ordered_ids)
        for google_id in ordered_ids:
            ambiguous_label = (
                has_mapped and has_unmapped and google_id not in mapped_google_ids
            )
            duplicate_metadata[google_id] = (
                group_id,
                group_size,
                ambiguous_label,
            )

    grouped: defaultdict[str, list[str]] = defaultdict(list)
    for google_id in google_ids:
        grouped[links.find(f"g:{google_id}")].append(google_id)

    components = [sorted(ids) for ids in grouped.values()]
    components.sort(key=lambda ids: tuple(ids))
    component_labels = {
        tuple(ids): f"component_{number:05d}"
        for number, ids in enumerate(components)
    }

    rng = np.random.default_rng(seed)
    shuffled = [components[index] for index in rng.permutation(len(components))]
    targets = _target_counts(len(google_ids), ratios)
    counts = np.zeros(len(SPLIT_NAMES), dtype=int)
    assignments: dict[str, tuple[str, str]] = {}
    for ids in shuffled:
        remaining = targets - counts
        split_index = int(np.argmax(remaining))
        split_name = SPLIT_NAMES[split_index]
        component_id = component_labels[tuple(ids)]
        counts[split_index] += len(ids)
        for google_id in ids:
            assignments[google_id] = (component_id, split_name)

    rows = [
        {
            "google_id": google_id,
            "component_id": assignments[google_id][0],
            "split": assignments[google_id][1],
            "duplicate_group_id": duplicate_metadata[google_id][0],
            "duplicate_group_size": duplicate_metadata[google_id][1],
            "ambiguous_label": duplicate_metadata[google_id][2],
        }
        for google_id in google_ids
    ]
    return pd.DataFrame(
        rows,
        columns=list(SPLIT_ASSIGNMENT_COLUMNS),
    )


def _canonical_ambiguous_labels(values: pd.Series) -> pd.Series:
    normalized: list[bool] = []
    for value in values:
        if value is None or pd.isna(value):
            raise ValueError("Split assignment ambiguous_label values must be non-null")
        if isinstance(value, (bool, np.bool_)):
            normalized.append(bool(value))
            continue
        if isinstance(value, (int, float, np.integer, np.floating)) and float(
            value
        ) in {0.0, 1.0}:
            normalized.append(bool(value))
            continue
        text = str(value).strip().lower()
        if text in {"true", "1"}:
            normalized.append(True)
        elif text in {"false", "0"}:
            normalized.append(False)
        else:
            raise ValueError(
                "Split assignment ambiguous_label values must be boolean"
            )
    return pd.Series(normalized, index=values.index, dtype=bool)


def canonicalize_split_assignments(assignments: pd.DataFrame) -> pd.DataFrame:
    """Normalize a split-assignment table into one deterministic representation."""
    missing = set(SPLIT_ASSIGNMENT_COLUMNS) - set(assignments.columns)
    if missing:
        raise ValueError(
            "Split assignments are missing required columns: "
            + ", ".join(sorted(missing))
        )

    canonical = assignments.loc[:, list(SPLIT_ASSIGNMENT_COLUMNS)].copy()
    for column in ("google_id", "component_id", "split", "duplicate_group_id"):
        if canonical[column].isna().any():
            raise ValueError(f"Split assignment {column} values must be non-null")
        canonical[column] = _clean_ids(canonical[column])
        if canonical[column].eq("").any():
            raise ValueError(f"Split assignment {column} values must be non-blank")
    canonical["split"] = canonical["split"].str.lower()
    unknown_splits = sorted(set(canonical["split"]) - set(SPLIT_NAMES))
    if unknown_splits:
        raise ValueError(
            "Split assignments contain unknown split labels: "
            + ", ".join(unknown_splits)
        )

    sizes = pd.to_numeric(canonical["duplicate_group_size"], errors="coerce")
    if sizes.isna().any() or (sizes < 1).any() or (sizes % 1 != 0).any():
        raise ValueError("Split assignment duplicate_group_size must be positive integers")
    canonical["duplicate_group_size"] = sizes.astype("int64")
    canonical["ambiguous_label"] = _canonical_ambiguous_labels(
        canonical["ambiguous_label"]
    )
    return canonical.sort_values("google_id", kind="mergesort").reset_index(drop=True)


def _canonical_split_assignment_bytes(assignments: pd.DataFrame) -> bytes:
    canonical = canonicalize_split_assignments(assignments)
    return canonical.to_csv(index=False, lineterminator="\n").encode("utf-8")


def split_assignment_sha256(assignments: pd.DataFrame) -> str:
    """Hash canonical split-assignment rows independent of input row order."""
    return hashlib.sha256(_canonical_split_assignment_bytes(assignments)).hexdigest()


def split_assignment_metadata(assignments: pd.DataFrame) -> dict[str, object]:
    """Return compact frozen metadata for one canonical assignment table."""
    canonical = canonicalize_split_assignments(assignments)
    counts = canonical["split"].value_counts()
    payload = canonical.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "row_count": int(len(canonical)),
        "rows_by_split": {
            split_name: int(counts.get(split_name, 0)) for split_name in SPLIT_NAMES
        },
    }


def validate_split_assignments(
    google: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    """Validate complete membership and group isolation, returning canonical rows."""
    canonical = canonicalize_split_assignments(assignments)
    if "product_id" not in google:
        raise ValueError("Google records must contain product_id")
    if google["product_id"].isna().any():
        raise ValueError("Normalized Google product IDs must be non-null")
    google_ids = _clean_ids(google["product_id"])
    if google_ids.eq("").any():
        raise ValueError("Normalized Google product IDs must be non-blank")
    if google_ids.duplicated().any():
        raise ValueError("Normalized Google product IDs must be unique")
    if canonical["google_id"].duplicated().any():
        raise ValueError("Split assignment IDs must be unique")

    google_id_set = set(google_ids)
    assignment_id_set = set(canonical["google_id"])
    missing = google_id_set - assignment_id_set
    extra = assignment_id_set - google_id_set
    if missing or extra:
        raise ValueError(
            "Split assignment IDs must exactly equal normalized Google IDs "
            f"(missing={len(missing)}, extra={len(extra)})"
        )

    component_splits = canonical.groupby("component_id")["split"].nunique()
    if component_splits.gt(1).any():
        raise ValueError("Connected components must not cross split assignments")
    duplicate_splits = canonical.groupby("duplicate_group_id")["split"].nunique()
    if duplicate_splits.gt(1).any():
        raise ValueError("Exact duplicate groups must not cross split assignments")
    return canonical


def validate_exact_split_assignments(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
) -> None:
    """Require exact equality after canonical normalization and row ordering."""
    expected_canonical = canonicalize_split_assignments(expected)
    actual_canonical = canonicalize_split_assignments(actual)
    if not expected_canonical.equals(actual_canonical):
        raise ValueError(
            "Split assignments differ from the frozen canonical membership"
        )


def validate_assignment_ids(
    google: pd.DataFrame,
    assignments: pd.DataFrame,
) -> None:
    """Require one split assignment for every normalized Google product ID."""
    if "product_id" not in google:
        raise ValueError("Google records must contain product_id")
    if "google_id" not in assignments:
        raise ValueError("Split assignments must contain google_id")

    google_ids = _clean_ids(google["product_id"])
    assignment_text = assignments["google_id"].astype("string").fillna("")
    assignment_ids = _clean_ids(assignments["google_id"])
    if google_ids.eq("").any():
        raise ValueError("Normalized Google product IDs must be non-empty")
    if assignment_ids.eq("").any():
        raise ValueError("Split assignment IDs must be non-empty")
    if not assignment_text.equals(assignment_ids):
        raise ValueError("Split assignment IDs must already be normalized")
    if google_ids.duplicated().any():
        raise ValueError("Normalized Google product IDs must be unique")
    if assignment_ids.duplicated().any():
        raise ValueError("Split assignment IDs must be unique")

    google_id_set = set(google_ids)
    assignment_id_set = set(assignment_ids)
    missing = sorted(google_id_set - assignment_id_set)
    extra = sorted(assignment_id_set - google_id_set)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={len(missing)}")
        if extra:
            details.append(f"extra={len(extra)}")
        raise ValueError(
            "Split assignment IDs must exactly equal normalized Google IDs ("
            + ", ".join(details)
            + ")"
        )


def split_google_listings(
    google: pd.DataFrame,
    assignments: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Materialize train/validation/test Google frames from split assignments."""
    validate_assignment_ids(google, assignments)
    merged = google.merge(
        assignments[["google_id", "split"]],
        left_on="product_id",
        right_on="google_id",
        how="left",
        validate="one_to_one",
    )
    if merged["split"].isna().any():
        raise ValueError("Every Google product must have a split assignment")
    merged = merged.drop(columns="google_id")
    return {
        split_name: merged.loc[merged["split"].eq(split_name)]
        .drop(columns="split")
        .reset_index(drop=True)
        for split_name in SPLIT_NAMES
    }
