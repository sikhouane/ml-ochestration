"""Nettoyage du jeu de données médical avant entraînement.

Usage:
    uv run python scripts/prepare.py
    uv run python scripts/prepare.py --input data/dataset.csv \
        --output data/dataset_clean.csv
"""

from __future__ import annotations

import argparse
import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)

DEFAULT_INPUT = Path("data/dataset.csv")
DEFAULT_OUTPUT = Path("data/dataset_clean.csv")

IDENTIFIER_COLUMNS = ("name", "doctor", "hospital")
DATE_COLUMNS = ("date_of_admission", "discharge_date")


def normalize_column_name(value: str) -> str:
    """Convertit un nom de colonne en snake_case ASCII."""
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    snake_case = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_value.strip())
    return snake_case.strip("_").lower()


def clean_text(series: pd.Series) -> pd.Series:
    """Nettoie les espaces et convertit les chaînes vides en valeurs manquantes."""
    result = series.astype("string").str.strip()
    result = result.str.replace(r"\s+", " ", regex=True)
    return result.replace("", pd.NA)


def normalize_binary_target(series: pd.Series) -> pd.Series:
    """Normalise une cible binaire vers des entiers 0/1."""
    numeric = pd.to_numeric(series, errors="coerce")

    text = clean_text(series).str.lower()
    mapped = text.map(
        {
            "0": 0,
            "0.0": 0,
            "false": 0,
            "no": 0,
            "non": 0,
            "negative": 0,
            "1": 1,
            "1.0": 1,
            "true": 1,
            "yes": 1,
            "oui": 1,
            "positive": 1,
        }
    )

    result = numeric.where(numeric.isin([0, 1]), mapped)
    return result.astype("Int64")


def add_date_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Parse les dates et ajoute des variables temporelles utiles."""
    df = dataframe.copy()

    for column in DATE_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")

    if "date_of_admission" in df.columns:
        admission = df["date_of_admission"]
        df["admission_year"] = admission.dt.year.astype("Int64")
        df["admission_month"] = admission.dt.month.astype("Int64")
        df["admission_day_of_week"] = admission.dt.dayofweek.astype("Int64")

    if "discharge_date" in df.columns:
        discharge = df["discharge_date"]
        df["discharge_year"] = discharge.dt.year.astype("Int64")
        df["discharge_month"] = discharge.dt.month.astype("Int64")
        df["discharge_day_of_week"] = discharge.dt.dayofweek.astype("Int64")

    if {"date_of_admission", "discharge_date"}.issubset(df.columns):
        stay = (df["discharge_date"] - df["date_of_admission"]).dt.days
        df["length_of_stay_days"] = stay.where(stay >= 0).astype("Int64")

    return df


def clean_dataset(
    dataframe: pd.DataFrame,
    *,
    keep_identifiers: bool = False,
    keep_dates: bool = False,
) -> pd.DataFrame:
    """Nettoie et valide le jeu de données."""
    df = dataframe.copy()
    initial_rows = len(df)

    df.columns = [normalize_column_name(str(column)) for column in df.columns]

    duplicated_columns = df.columns[df.columns.duplicated()].tolist()
    if duplicated_columns:
        raise ValueError(
            "Noms de colonnes dupliqués après normalisation : "
            + ", ".join(duplicated_columns)
        )

    for column in df.select_dtypes(include=["object", "string"]).columns:
        df[column] = clean_text(df[column])

    title_case_columns = (
        "name",
        "gender",
        "medical_condition",
        "doctor",
        "hospital",
        "insurance_provider",
        "admission_type",
        "medication",
        "test_results",
    )
    for column in title_case_columns:
        if column in df.columns:
            df[column] = df[column].str.title()

    if "blood_type" in df.columns:
        df["blood_type"] = df["blood_type"].str.upper()

    if "gender" in df.columns:
        df["gender"] = df["gender"].replace(
            {
                "M": "Male",
                "F": "Female",
                "Man": "Male",
                "Woman": "Female",
            }
        )

    numeric_columns = ("age", "billing_amount")
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "room_number" in df.columns:
        room_number = pd.to_numeric(df["room_number"], errors="coerce")
        df["room_number"] = room_number.where(room_number > 0).astype("Int64")

    if "age" in df.columns:
        df["age"] = df["age"].where(df["age"].between(0, 120))

    if "billing_amount" in df.columns:
        df["billing_amount"] = df["billing_amount"].where(
            df["billing_amount"] >= 0
        )
        df["billing_amount"] = df["billing_amount"].round(2)

    df = add_date_features(df)

    if "target" not in df.columns:
        raise ValueError(
            "La colonne cible 'target' est absente. "
            f"Colonnes disponibles : {', '.join(df.columns)}"
        )

    df["target"] = normalize_binary_target(df["target"])
    invalid_target_rows = int(df["target"].isna().sum())
    if invalid_target_rows:
        LOGGER.warning(
            "%d ligne(s) supprimée(s), car la cible n'est pas égale à 0 ou 1.",
            invalid_target_rows,
        )
        df = df.dropna(subset=["target"])

    df["target"] = df["target"].astype("int8")

    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows:
        LOGGER.info("%d ligne(s) dupliquée(s) supprimée(s).", duplicate_rows)
        df = df.drop_duplicates()

    if not keep_identifiers:
        columns_to_drop = [
            column for column in IDENTIFIER_COLUMNS if column in df.columns
        ]
        df = df.drop(columns=columns_to_drop)

    if not keep_dates:
        columns_to_drop = [
            column for column in DATE_COLUMNS if column in df.columns
        ]
        df = df.drop(columns=columns_to_drop)

    df = df.reset_index(drop=True)

    LOGGER.info(
        "Nettoyage terminé : %d lignes avant, %d lignes après, %d colonnes.",
        initial_rows,
        len(df),
        len(df.columns),
    )

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if not missing.empty:
        LOGGER.info("Valeurs manquantes restantes :\n%s", missing.to_string())

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"CSV source (défaut : {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"CSV nettoyé (défaut : {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--keep-identifiers",
        action="store_true",
        help="Conserve les colonnes name, doctor et hospital.",
    )
    parser.add_argument(
        "--keep-dates",
        action="store_true",
        help="Conserve les colonnes de dates brutes en plus des variables dérivées.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Dataset introuvable : {args.input}")

    LOGGER.info("Lecture de %s", args.input)
    dataframe = pd.read_csv(args.input)

    cleaned = clean_dataset(
        dataframe,
        keep_identifiers=args.keep_identifiers,
        keep_dates=args.keep_dates,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(args.output, index=False)
    LOGGER.info("Dataset nettoyé écrit dans %s", args.output)


if __name__ == "__main__":
    main()
