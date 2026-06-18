"""Client de prediction pour l'API FastAPI.

Utilisation :
    uv run python predict.py
    uv run python predict.py --csv data/healthcare_dataset.csv --n-samples 5
    uv run python predict.py --api-url http://127.0.0.1:8000

L'API doit etre lancee avant :
    uv run uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import requests


DEFAULT_API_URL = "http://127.0.0.1:8000"

# Correspondance entre les colonnes du CSV et les champs attendus par l'API.
COLUMN_MAPPING = {
    "Age": "Age",
    "Billing Amount": "Billing_Amount",
    "Gender": "Gender",
    "Blood Type": "Blood_Type",
    "Medical Condition": "Medical_Condition",
    "Admission Type": "Admission_Type",
    "Test Results": "Test_Results",
    "Medication": "Medication",
}

DEFAULT_SAMPLE = {
    "Age": 30.0,
    "Billing_Amount": 18856.281305978155,
    "Gender": "Male",
    "Blood_Type": "B-",
    "Medical_Condition": "Cancer",
    "Admission_Type": "Urgent",
    "Test_Results": "Normal",
    "Medication": "Paracetamol",
}


def check_health(api_url: str, timeout: float) -> None:
    """Verifier que l'API est accessible."""
    try:
        response = requests.get(f"{api_url}/health", timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Impossible de joindre l'API sur {api_url}. "
            "Verifie que FastAPI est bien lancee."
        ) from exc

    data = response.json()
    if data.get("status") != "ok":
        raise RuntimeError(f"Reponse inattendue de /health : {data}")


def row_to_payload(row: pd.Series) -> dict[str, Any]:
    """Convertir une ligne du dataset vers le JSON attendu par l'API."""
    missing_columns = [
        csv_column
        for csv_column in COLUMN_MAPPING
        if csv_column not in row.index
    ]

    if missing_columns:
        raise ValueError(
            "Colonnes absentes du CSV : " + ", ".join(missing_columns)
        )

    payload = {
        api_field: row[csv_column]
        for csv_column, api_field in COLUMN_MAPPING.items()
    }

    # Conversion explicite des types pour obtenir un JSON propre.
    payload["Age"] = float(payload["Age"])
    payload["Billing_Amount"] = float(payload["Billing_Amount"])

    for field in (
        "Gender",
        "Blood_Type",
        "Medical_Condition",
        "Admission_Type",
        "Test_Results",
        "Medication",
    ):
        payload[field] = str(payload[field]).strip()

    return payload


def predict_one(
    payload: dict[str, Any],
    api_url: str,
    timeout: float,
) -> dict[str, Any]:
    """Envoyer une prediction a l'API."""
    try:
        response = requests.post(
            f"{api_url}/predict",
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Erreur de connexion pendant l'appel a {api_url}/predict"
        ) from exc

    if response.status_code != 200:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text

        raise RuntimeError(
            f"Erreur API {response.status_code} : {detail}"
        )

    return response.json()


def predict_from_csv(
    csv_path: Path,
    api_url: str,
    n_samples: int,
    timeout: float,
) -> pd.DataFrame:
    """Lire un CSV et appeler l'API pour plusieurs lignes."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {csv_path}")

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError("Le fichier CSV est vide.")

    sample_size = min(n_samples, len(df))

    sample_df = df.sample(
        n=sample_size,
        random_state=42,
    ).copy()

    results: list[dict[str, Any]] = []

    for index, row in sample_df.iterrows():
        payload = row_to_payload(row)

        try:
            result = predict_one(payload, api_url, timeout)
            results.append(
                {
                    "row_index": int(index),
                    "prediction": result["prediction"],
                    "probability": result["probability"],
                    "target": row.get("target"),
                    "status": "ok",
                }
            )
        except Exception as exc:
            results.append(
                {
                    "row_index": int(index),
                    "prediction": None,
                    "probability": None,
                    "target": row.get("target"),
                    "status": str(exc),
                }
            )

    return pd.DataFrame(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tester l'endpoint /predict de l'API FastAPI."
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"Adresse de l'API, par defaut : {DEFAULT_API_URL}",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Chemin optionnel vers le dataset CSV.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5,
        help="Nombre de lignes du CSV a tester.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Timeout HTTP en secondes.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_url = args.api_url.rstrip("/")

    if args.n_samples <= 0:
        print("Erreur : --n-samples doit etre superieur a 0.", file=sys.stderr)
        return 1

    try:
        check_health(api_url, args.timeout)
        print(f"API disponible : {api_url}")

        if args.csv is None:
            result = predict_one(DEFAULT_SAMPLE, api_url, args.timeout)

            print("\nDonnees envoyees :")
            print(DEFAULT_SAMPLE)

            print("\nPrediction :")
            print(f"  classe      : {result['prediction']}")
            print(f"  probabilite : {result['probability']}")
        else:
            results = predict_from_csv(
                csv_path=args.csv,
                api_url=api_url,
                n_samples=args.n_samples,
                timeout=args.timeout,
            )

            print("\nResultats :")
            print(results.to_string(index=False))

    except Exception as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
