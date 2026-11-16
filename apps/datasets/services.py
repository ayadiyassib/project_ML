"""
Ingestion, validation and profiling services for datasets.

Everything here is pure Python/pandas: Django views stay thin and the same
functions are reused by the tests.  Supported formats: ``.csv``, ``.tsv``,
``.txt``, ``.json``, ``.xlsx`` and ``.xls`` (legacy Excel requires ``xlrd``).

Design notes
------------
* Encoding and delimiter are sniffed instead of assumed, because French CSV
  exports are very often ``cp1252`` + semicolon separated.
* Nothing is ever written back to the original sample files: samples are copied
  into ``MEDIA_ROOT`` so a dataset can be deleted without touching
  ``sample_data/``.
* ``profile_dataframe`` is the profiling contract consumed by the EDA module
  (column kinds) and by preprocessing (numeric vs categorical split).
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Sequence
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pandas as pd
from django.conf import settings
from django.core.files import File
from django.db import transaction

from apps.datasets.cache import get_dataframe, invalidate
from apps.datasets.models import Dataset, human_size

MAX_UPLOAD_BYTES = getattr(settings, "DATA_UPLOAD_MAX_MEMORY_SIZE", 50 * 1024 * 1024)
MAX_PREVIEW_PAGE_SIZE = 100
DEFAULT_PREVIEW_PAGE_SIZE = 25
SAMPLE_DIR = Path(settings.BASE_DIR) / "sample_data"

SUPPORTED_EXTENSIONS = (".csv", ".tsv", ".txt", ".json", ".xlsx", ".xls")
_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
_DELIMITERS = ",;\t|"
_MISSING_TOKENS = ["?", "NA", "N/A", "na", "null", "NULL", "None", "-", "--"]

_ID_NAME_PATTERN = re.compile(
    r"^(id|index|uuid|guid|unnamed:?\s*\d+)$|(_id|_uuid|_key|Id|ID)$"
)
_TARGET_NAMES = (
    "target",
    "survived",
    "medhouseval",
    "price",
    "label",
    "class",
    "outcome",
    "default",
    "churn",
)


class DatasetError(Exception):
    """Raised when a file cannot be ingested, with a user-facing message."""


# ---------------------------------------------------------------------------
# Sample datasets
# ---------------------------------------------------------------------------

SAMPLE_DATASETS: tuple[dict[str, Any], ...] = (
    {
        "slug": "iris",
        "filename": "iris.csv",
        "title": "Iris",
        "description": "150 fleurs, 4 mesures morphologiques, 3 espèces à séparer.",
        "task": "classification",
        "accent": "purple",
    },
    {
        "slug": "titanic",
        "filename": "titanic.csv",
        "title": "Titanic",
        "description": "891 passagers, survie à prédire, vraies valeurs manquantes et colonnes texte.",
        "task": "classification",
        "accent": "indigo",
    },
    {
        "slug": "wine",
        "filename": "wine.csv",
        "title": "Wine Quality",
        "description": "178 vins, 13 mesures chimiques, 3 cépages.",
        "task": "classification",
        "accent": "purple",
    },
    {
        "slug": "california_housing",
        "filename": "california_housing.csv",
        "title": "California Housing",
        "description": "20 640 districts, 8 variables explicatives, prix médian à estimer.",
        "task": "regression",
        "accent": "emerald",
    },
    {
        "slug": "mall_customers",
        "filename": "mall_customers.csv",
        "title": "Mall Customers",
        "description": "200 clients, revenu annuel et score de dépense, idéal pour K-Means.",
        "task": "clustering",
        "accent": "amber",
    },
)

_row_cache: dict[str, tuple[int, int, int]] = {}


def available_samples() -> list[dict[str, Any]]:
    """Return the sample catalog enriched with row/column counts when available."""
    catalog: list[dict[str, Any]] = []
    for sample in SAMPLE_DATASETS:
        path = SAMPLE_DIR / sample["filename"]
        if not path.exists():
            continue
        entry = dict(sample)
        rows, cols = _fast_shape(path)
        entry.update(rows=rows, cols=cols, available=True)
        catalog.append(entry)
    return catalog


def _fast_shape(path: Path) -> tuple[int, int]:
    """Cheap shape estimate for CSV samples (shown on the catalog cards)."""
    cached = _row_cache.get(str(path))
    if cached and cached[0] == int(path.stat().st_mtime):
        return cached[1], cached[2]

    rows = cols = 0
    if path.suffix.lower() in (".csv", ".tsv", ".txt"):
        try:
            with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                sample = handle.read(64 * 1024)
                handle.seek(0)
                reader = csv.reader(handle, delimiter=_sniff_delimiter(sample))
                header = next(reader, [])
                cols = len(header)
                rows = sum(1 for _ in reader)
        except (OSError, csv.Error, StopIteration):
            rows = cols = 0
    _row_cache[str(path)] = (int(path.stat().st_mtime), rows, cols)
    return rows, cols


# ---------------------------------------------------------------------------
# Low-level readers
# ---------------------------------------------------------------------------


def _sniff_delimiter(sample: str) -> str:
    """Guess the CSV delimiter from a text sample."""
    if not sample.strip():
        return ","
    try:
        return csv.Sniffer().sniff(sample[: 64 * 1024], delimiters=_DELIMITERS).delimiter
    except csv.Error:
        first_line = sample.splitlines()[0]
        counts = {delimiter: first_line.count(delimiter) for delimiter in _DELIMITERS}
        best = max(counts, key=lambda key: counts[key])
        return best if counts[best] else ","


def _detect_encoding(raw: bytes) -> str:
    """Return the first encoding able to decode the sample."""
    for encoding in _ENCODINGS:
        try:
            raw.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return "latin-1"


def _read_delimited(path: Path) -> pd.DataFrame:
    raw = path.read_bytes()[: 512 * 1024]
    encoding = _detect_encoding(raw)
    text_sample = raw.decode(encoding, errors="replace")
    separator = "\t" if path.suffix.lower() == ".tsv" else _sniff_delimiter(text_sample)

    try:
        frame = pd.read_csv(
            path,
            sep=separator,
            encoding=encoding,
            na_values=_MISSING_TOKENS,
            keep_default_na=True,
            low_memory=False,
        )
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise DatasetError(f"Fichier tabulaire illisible ({exc}).") from exc

    if frame.empty:
        raise DatasetError("Le fichier ne contient aucune ligne de données.")
    return frame


def _read_json(path: Path) -> pd.DataFrame:
    raw = path.read_bytes()
    encoding = _detect_encoding(raw)
    payload: Any
    try:
        payload = json.loads(raw.decode(encoding))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise DatasetError(f"JSON invalide : {exc}") from exc

    if isinstance(payload, dict):
        for key in ("data", "records", "rows", "items", "results"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break

    if isinstance(payload, list):
        if not payload:
            raise DatasetError("Le JSON ne contient aucun enregistrement.")
        if isinstance(payload[0], dict):
            try:
                frame = pd.json_normalize(payload)
            except (TypeError, ValueError) as exc:
                raise DatasetError(f"Structure JSON non tabulaire : {exc}") from exc
        else:
            frame = pd.DataFrame(payload)
    elif isinstance(payload, dict):
        try:
            frame = pd.DataFrame.from_dict(payload, orient="index").reset_index()
        except (TypeError, ValueError) as exc:
            raise DatasetError(f"Structure JSON non tabulaire : {exc}") from exc
    else:
        raise DatasetError("Le JSON doit être une liste d'objets ou un objet de listes.")

    if frame.empty:
        raise DatasetError("Le JSON ne contient aucune ligne exploitable.")
    return frame


def _read_excel(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_excel(path)
    except ImportError as exc:
        raise DatasetError(
            "Le format Excel nécessite le paquet 'openpyxl' (installé) ; "
            "les fichiers .xls anciens exigent 'xlrd'."
        ) from exc
    except (ValueError, OSError) as exc:
        raise DatasetError(f"Classeur Excel illisible : {exc}") from exc

    if frame.empty:
        raise DatasetError("Le classeur ne contient aucune ligne de données.")
    return frame


def read_tabular(path: str | Path) -> pd.DataFrame:
    """Dispatch to the right reader based on the file extension."""
    file_path = Path(path)
    if not file_path.exists():
        raise DatasetError("Le fichier est introuvable sur le disque.")
    if file_path.stat().st_size == 0:
        raise DatasetError("Le fichier est vide.")

    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise DatasetError(
            f"Format non supporté ({suffix or 'sans extension'}). "
            f"Formats acceptés : {', '.join(SUPPORTED_EXTENSIONS)}."
        )

    if suffix in (".xlsx", ".xls"):
        frame = _read_excel(file_path)
    elif suffix == ".json":
        frame = _read_json(file_path)
    else:
        frame = _read_delimited(file_path)

    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def read_from_url(url: str) -> tuple[bytes, str]:
    """Download a remote file. Returns ``(content, filename)``."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise DatasetError("L'URL doit commencer par http:// ou https://.")

    request = Request(url, headers={"User-Agent": "ML-Studio/1.0 (+Django)"})
    try:
        with urlopen(request, timeout=20) as response:  # noqa: S310 - scheme checked above
            declared = response.headers.get("Content-Length")
            if declared and int(declared) > MAX_UPLOAD_BYTES:
                raise DatasetError(
                    f"Fichier distant trop volumineux (max {human_size(MAX_UPLOAD_BYTES)})."
                )
            content = response.read(MAX_UPLOAD_BYTES + 1)
    except DatasetError:
        raise
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        raise DatasetError(f"Téléchargement impossible : {exc}") from exc

    if len(content) > MAX_UPLOAD_BYTES:
        raise DatasetError(f"Fichier distant trop volumineux (max {human_size(MAX_UPLOAD_BYTES)}).")

    filename = Path(parsed.path).name or "jeu_de_donnees.csv"
    if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise DatasetError("L'URL ne pointe pas vers un fichier CSV, JSON ou Excel.")
    return content, filename


# ---------------------------------------------------------------------------
# Import orchestration
# ---------------------------------------------------------------------------


def load_dataframe(dataset: Dataset) -> pd.DataFrame:
    """Return the parsed DataFrame for a dataset (cached, returned as a copy)."""
    if not dataset.file:
        raise DatasetError("Ce jeu de données n'a pas de fichier associé.")
    try:
        path = Path(dataset.file.path)
    except (ValueError, NotImplementedError) as exc:
        raise DatasetError("Le stockage du fichier n'est pas accessible.") from exc
    return get_dataframe(path, read_tabular)


def _finalize(dataset: Dataset) -> Dataset:
    """Read the file back, store row/column counts and column metadata."""
    frame = read_tabular(dataset.file.path)
    metadata = describe_columns(frame)

    dataset.n_rows = int(frame.shape[0])
    dataset.n_cols = int(frame.shape[1])
    dataset.columns_meta = metadata
    dataset.size_bytes = dataset.size_bytes or Path(dataset.file.path).stat().st_size
    dataset.save(update_fields=["n_rows", "n_cols", "columns_meta", "size_bytes"])
    invalidate(dataset.file.path)
    return dataset


@transaction.atomic
def create_dataset_from_file(
    *,
    file_obj: BinaryIO,
    filename: str,
    source: str = Dataset.Source.UPLOAD,
    name: str | None = None,
    size_bytes: int | None = None,
) -> Dataset:
    """Persist an uploaded or downloaded file and profile it."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise DatasetError(
            f"Format non supporté ({suffix or 'sans extension'}). "
            f"Formats acceptés : {', '.join(SUPPORTED_EXTENSIONS)}."
        )

    dataset = Dataset(
        name=name or Path(filename).name,
        original_filename=Path(filename).name,
        source=source,
        size_bytes=size_bytes or 0,
    )
    dataset.save()
    dataset.file.save(Path(filename).name, File(file_obj), save=True)
    try:
        return _finalize(dataset)
    except DatasetError:
        dataset.delete()
        raise


def create_dataset_from_upload(uploaded_file) -> Dataset:
    """Validate an ``UploadedFile`` and import it."""
    if uploaded_file is None:
        raise DatasetError("Aucun fichier reçu.")
    if uploaded_file.size > MAX_UPLOAD_BYTES:
        raise DatasetError(
            f"Fichier trop volumineux ({human_size(uploaded_file.size)}). "
            f"Limite : {human_size(MAX_UPLOAD_BYTES)}."
        )
    return create_dataset_from_file(
        file_obj=uploaded_file,
        filename=uploaded_file.name,
        source=Dataset.Source.UPLOAD,
        size_bytes=uploaded_file.size,
    )


def create_dataset_from_url(url: str) -> Dataset:
    content, filename = read_from_url(url)
    return create_dataset_from_file(
        file_obj=io.BytesIO(content),
        filename=filename,
        source=Dataset.Source.URL,
        size_bytes=len(content),
    )


def create_dataset_from_sample(slug: str) -> Dataset:
    """Copy a bundled sample into ``MEDIA_ROOT`` (reused if already imported)."""
    sample = next((item for item in SAMPLE_DATASETS if item["slug"] == slug), None)
    if sample is None:
        raise DatasetError(f"Jeu de démonstration inconnu : {slug}.")

    path = SAMPLE_DIR / sample["filename"]
    if not path.exists():
        raise DatasetError(f"Le fichier de démonstration {sample['filename']} est absent.")

    existing = Dataset.objects.filter(
        source=Dataset.Source.SAMPLE, original_filename=sample["filename"]
    ).first()
    if existing is not None:
        return existing

    with path.open("rb") as handle:
        return create_dataset_from_file(
            file_obj=handle,
            filename=sample["filename"],
            source=Dataset.Source.SAMPLE,
            name=sample["title"],
            size_bytes=path.stat().st_size,
        )


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------


def _is_datetime_like(series: pd.Series) -> bool:
    if (
        pd.api.types.is_numeric_dtype(series)
        or pd.api.types.is_bool_dtype(series)
        or pd.api.types.is_datetime64_any_dtype(series)
    ):
        return False
    sample = series.dropna().head(50)
    if sample.empty:
        return False
    try:
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        return False
    return bool(parsed.notna().mean() > 0.9)


def classify_column(series: pd.Series) -> str:
    """Return the functional kind of a column: numeric, boolean, datetime, categorical, text."""
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    non_null = series.dropna()
    if non_null.empty:
        return "empty"

    if _is_datetime_like(series):
        return "datetime"

    cardinality = non_null.nunique() / max(len(non_null), 1)
    if cardinality > 0.5 and non_null.astype(str).str.len().mean() > 12:
        return "text"
    return "categorical"


def _suggested_target(frame: pd.DataFrame, kinds: dict[str, str]) -> str | None:
    """Guess the target column so the preprocessing form can pre-select it."""
    lowered = {column.lower(): column for column in frame.columns}

    for candidate in _TARGET_NAMES:
        if candidate in lowered:
            return lowered[candidate]

    if frame.shape[1] == 0:
        return None

    last = frame.columns[-1]
    last_series = frame[last].dropna()
    unique = last_series.nunique()
    if kinds.get(last) == "numeric":
        if 2 <= unique <= 10:
            return last
        return last
    if kinds.get(last) in ("categorical", "boolean") and 2 <= unique <= 50:
        return last
    return None


def describe_columns(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Per-column metadata stored on the ``Dataset`` row."""
    rows = max(len(frame), 1)
    kinds = {str(name): classify_column(frame[name]) for name in frame.columns}
    suggested = _suggested_target(frame, kinds)

    columns: list[dict[str, Any]] = []
    for name in frame.columns:
        series = frame[name]
        kind = kinds[str(name)]
        non_null = int(series.notna().sum())
        unique = int(series.nunique(dropna=True))
        missing = int(len(series) - non_null)
        lowered = str(name).strip().lower()
        name_looks_like_id = bool(_ID_NAME_PATTERN.search(str(name))) or lowered in {
            "id",
            "index",
            "uuid",
            "guid",
        }
        is_id_like = name_looks_like_id or (
            rows > 20
            and unique >= rows * 0.95
            and kind in ("categorical", "text")
        )
        columns.append(
            {
                "name": str(name),
                "dtype": str(series.dtype),
                "kind": kind,
                "missing": missing,
                "missing_pct": round(missing / rows * 100, 2),
                "unique": unique,
                "unique_pct": round(unique / rows * 100, 2),
                "is_id_like": is_id_like,
                "is_constant": unique <= 1,
                "suggested_target": str(name) == suggested,
                "sample_values": [
                    _format_cell(value) for value in series.dropna().head(3).tolist()
                ],
            }
        )
    return columns


def profile_dataframe(frame: pd.DataFrame) -> dict[str, Any]:
    """Global + per-column profile used by the ingestion page."""
    rows = int(frame.shape[0])
    cols = int(frame.shape[1])
    total_cells = max(rows * cols, 1)
    missing_cells = int(frame.isna().sum().sum())
    duplicates = int(frame.duplicated().sum())
    memory_bytes = int(frame.memory_usage(deep=True).sum())

    columns = describe_columns(frame)
    kind_counts: dict[str, int] = {}
    for column in columns:
        kind_counts[column["kind"]] = kind_counts.get(column["kind"], 0) + 1

    return {
        "rows": rows,
        "cols": cols,
        "rows_display": f"{rows:,}".replace(",", "\u202f"),
        "cols_display": str(cols),
        "memory_bytes": memory_bytes,
        "memory_human": human_size(memory_bytes),
        "duplicates": duplicates,
        "missing_cells": missing_cells,
        "missing_pct": round(missing_cells / total_cells * 100, 2),
        "completeness": round(100 - missing_cells / total_cells * 100, 2),
        "columns": columns,
        "numeric_columns": [c["name"] for c in columns if c["kind"] == "numeric"],
        "categorical_columns": [c["name"] for c in columns if c["kind"] in ("categorical", "boolean", "text")],
        "kind_counts": kind_counts,
        "numeric_count": kind_counts.get("numeric", 0),
        "categorical_count": sum(
            kind_counts.get(kind, 0) for kind in ("categorical", "text", "boolean")
        ),
        "datetime_count": kind_counts.get("datetime", 0),
        "empty_count": kind_counts.get("empty", 0),
        "high_missing_columns": [c for c in columns if c["missing_pct"] >= 40],
        "id_like_columns": [c["name"] for c in columns if c["is_id_like"]],
        "constant_columns": [c["name"] for c in columns if c["is_constant"]],
        "suggested_target": next(
            (c["name"] for c in columns if c["suggested_target"]), None
        ),
    }


# ---------------------------------------------------------------------------
# Preview pagination
# ---------------------------------------------------------------------------


def _format_cell(value: Any) -> str:
    """Render a cell as a compact string for the HTML table."""
    try:
        if value is None or pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    if isinstance(value, (pd.Timestamp, datetime)):
        return value.strftime("%Y-%m-%d") if value.hour == 0 and value.minute == 0 else value.strftime(
            "%Y-%m-%d %H:%M"
        )
    if isinstance(value, bool):
        return "Oui" if value else "Non"
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return str(int(value))
        return f"{value:.4g}"
    if isinstance(value, int):
        return str(value)
    text = str(value)
    return text if len(text) <= 80 else text[:77] + "…"


def preview_page(
    frame: pd.DataFrame,
    page: int = 1,
    page_size: int = DEFAULT_PREVIEW_PAGE_SIZE,
) -> dict[str, Any]:
    """Slice a paginated, display-ready preview of the dataframe."""
    page_size = max(1, min(int(page_size), MAX_PREVIEW_PAGE_SIZE))
    total_rows = int(frame.shape[0])
    total_pages = max(1, math.ceil(total_rows / page_size))
    page = max(1, min(int(page), total_pages))
    start = (page - 1) * page_size
    end = min(start + page_size, total_rows)
    chunk = frame.iloc[start:end]

    return {
        "columns": [
            {"name": str(name), "kind": classify_column(frame[name])} for name in frame.columns
        ],
        "rows": [[_format_cell(value) for value in record] for record in chunk.itertuples(index=False)],
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "total_rows": total_rows,
        "rows_display": f"{total_rows:,}".replace(",", "\u202f"),
        "start": start + 1 if total_rows else 0,
        "end": end,
        "has_prev": page > 1,
        "has_next": page < total_pages,
        "prev_page": page - 1,
        "next_page": page + 1,
        "page_sizes": (10, 25, 50, 100),
    }
