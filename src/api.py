"""API d'inference d'un modele de classification (FastAPI).

Lancement :
    uv run uvicorn mlproject.api:app --reload
"""
from __future__ import annotations

import logging
import os
import re
from contextlib import asynccontextmanager
from typing import AsyncIterator, Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mlproject.config import CATEGORICAL_FEATURES, MODEL_DIR, NUMERIC_FEATURES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

ml: dict[str, Any] = {}


def _normalise_name(name: str) -> str:
    """Normaliser un nom pour faire correspondre espaces et underscores.

    Exemples :
        ``Billing Amount`` -> ``billingamount``
        ``Billing_Amount`` -> ``billingamount``
    """
    return re.sub(r"[^a-z0-9]", "", name.lower())


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Charger le modele au demarrage et le liberer a l'arret."""
    model_path = MODEL_DIR / "model.joblib"

    if not model_path.exists():
        raise RuntimeError(f"Modele introuvable : {model_path}")

    model = joblib.load(model_path)
    ml["model"] = model

    configured_features = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)
    trained_features = list(getattr(model, "feature_names_in_", configured_features))

    if trained_features != configured_features:
        logger.warning(
            "Les colonnes du modele different de config.py. "
            "Modele=%s ; config=%s",
            trained_features,
            configured_features,
        )

    logger.info("Modele charge depuis %s", model_path)
    logger.info("Colonnes attendues : %s", configured_features)

    try:
        yield
    finally:
        ml.clear()
        logger.info("Modele decharge")


app = FastAPI(
    title="Classification API",
    version="0.1.0",
    lifespan=lifespan,
)


class Features(BaseModel):
    """Schema d'entree pour l'endpoint /predict."""

    Age: float = Field(..., ge=0, description="Age du patient")
    Billing_Amount: float = Field(..., ge=0, description="Montant de la facturation")
    Gender: str = Field(..., description="Genre du patient")
    Blood_Type: str = Field(..., description="Groupe sanguin")
    Medical_Condition: str = Field(..., description="Condition medicale")
    Admission_Type: str = Field(..., description="Type d'admission")
    Test_Results: str = Field(..., description="Resultats des tests")
    Medication: str = Field(..., description="Medicament")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Age": 45.0,
                    "Billing_Amount": 15000.0,
                    "Gender": "Male",
                    "Blood_Type": "O+",
                    "Medical_Condition": "Diabetes",
                    "Admission_Type": "Emergency",
                    "Test_Results": "Abnormal",
                    "Medication": "Aspirin",
                }
            ]
        }
    }


class PredictionOut(BaseModel):
    """Schema de sortie pour l'endpoint /predict."""

    prediction: int = Field(..., description="Classe predite (0 ou 1)")
    probability: float = Field(..., ge=0, le=1, description="Probabilite de la classe 1")


def _build_input_row(features: Features) -> pd.DataFrame:
    """Construire une ligne avec exactement les noms/types attendus."""
    payload = features.model_dump()
    payload_by_normalised_name = {
        _normalise_name(name): value for name, value in payload.items()
    }

    expected_features = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)
    missing_features = [
        name
        for name in expected_features
        if _normalise_name(name) not in payload_by_normalised_name
    ]

    if missing_features:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Le schema de l'API ne correspond pas aux colonnes du modele.",
                "colonnes_absentes": missing_features,
                "colonnes_attendues": expected_features,
                "champs_recus": list(payload),
            },
        )

    # Les clés du dictionnaire utilisent les noms EXACTS de config.py.
    ordered_values = {
        expected_name: payload_by_normalised_name[_normalise_name(expected_name)]
        for expected_name in expected_features
    }
    row = pd.DataFrame([ordered_values], columns=expected_features)

    # Eviter que les nombres arrivent au preprocesseur sous forme de texte/object.
    for column in NUMERIC_FEATURES:
        try:
            row[column] = pd.to_numeric(row[column], errors="raise").astype(float)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail=f"La colonne numerique '{column}' contient une valeur invalide.",
            ) from exc

    # Utiliser des chaînes Python classiques pour les variables categorielles.
    for column in CATEGORICAL_FEATURES:
        row[column] = row[column].astype(object)

    if row.isna().any().any():
        null_columns = row.columns[row.isna().any()].tolist()
        raise HTTPException(
            status_code=422,
            detail=f"Valeurs manquantes detectees dans : {null_columns}",
        )

    return row


@app.get("/health")
def health() -> dict[str, str]:
    """Verifier que l'API fonctionne."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOut)
def predict(features: Features) -> PredictionOut:
    """Predire une classe a partir des caracteristiques fournies."""
    model = ml.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Modele non charge")

    row = _build_input_row(features)

    logger.info("Valeurs envoyees au modele : %s", row.to_dict(orient="records"))
    logger.info("Types envoyes au modele : %s", row.dtypes.astype(str).to_dict())

    try:
        probabilities = model.predict_proba(row)[0]
        classes = list(getattr(model, "classes_", [0, 1]))

        if 1 in classes:
            positive_index = classes.index(1)
        elif "1" in classes:
            positive_index = classes.index("1")
        elif len(classes) == 2:
            positive_index = 1
        else:
            raise ValueError(f"Classes inattendues : {classes}")

        probability = float(probabilities[positive_index])
        prediction = int(probability >= 0.5)

    except Exception as exc:
        logger.exception(
            "Erreur de prediction. Colonnes=%s, types=%s",
            row.columns.tolist(),
            row.dtypes.astype(str).to_dict(),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Erreur pendant la prediction : {type(exc).__name__}: {exc}",
        ) from exc

    logger.info("Prediction=%d, probabilite=%.4f", prediction, probability)
    return PredictionOut(
        prediction=prediction,
        probability=round(probability, 4),
    )


@app.get("/model-info")
def model_info() -> dict[str, object]:
    """Retourner la version et les colonnes du modele servi."""
    model = ml.get("model")
    configured_features = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)

    return {
        "version": os.environ.get("MODEL_VERSION", "unknown"),
        "model_loaded": model is not None,
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "trained_features": list(
            getattr(model, "feature_names_in_", configured_features)
        )
        if model is not None
        else [],
    }
