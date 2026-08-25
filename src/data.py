"""Download, normalize, and split the Amazon-Google Products benchmark."""

from __future__ import annotations

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


def make_entity_splits(
    google: pd.DataFrame,
    gold: pd.DataFrame,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> pd.DataFrame:
    """Assign every Google listing to a deterministic connected-component split.

    Amazon remains a fixed catalog. Gold-linked Google records sharing any Amazon
    record stay together; unmatched Google listings are singleton components.
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

    for pair in gold[["google_id", "amazon_id"]].itertuples(index=False):
        google_id = str(pair.google_id).strip()
        amazon_id = str(pair.amazon_id).strip()
        if google_id in google_id_set and amazon_id:
            links.union(f"g:{google_id}", f"a:{amazon_id}")

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
        }
        for google_id in google_ids
    ]
    return pd.DataFrame(rows, columns=["google_id", "component_id", "split"])


def split_google_listings(
    google: pd.DataFrame,
    assignments: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Materialize train/validation/test Google frames from split assignments."""
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
