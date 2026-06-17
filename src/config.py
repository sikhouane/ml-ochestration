"""Configuration centrale du projet de classification.

C'est le SEUL fichier a adapter pour brancher votre propre jeu de donnees :
data.py, features.py et les scripts d'entrainement lisent toutes leurs
colonnes via ces constantes. Voir tp/TP_S0_projet_personnel.md.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

# TODO (S0-1) : chemin vers votre fichier de donnees (CSV) place dans data/
DATA_PATH = ROOT / "data" / "dataset.csv"
MODEL_DIR = ROOT / "models"

# TODO (S0-2) : nom de la colonne cible binaire (valeurs 0/1)
TARGET = "target"

# TODO (S0-3) : colonnes numeriques de votre dataset
NUMERIC_FEATURES: list[str] = ["Age", "Billing Amount"]

# TODO (S0-4) : colonnes categorielles (peut rester vide : [])
CATEGORICAL_FEATURES: list[str] = ["Gender", "Blood Type", "Medical Condition", "Admission Type", "Test Results", "Medication"]

RANDOM_STATE = 42

# Surcouche via variables d'environnement (principe 12-factor)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "classification-baseline")
MODEL_NAME = os.getenv("MODEL_NAME", "classifier")

MLFLOW_EXPERIMENT_DESCRIPTION = os.getenv(
    "MLFLOW_EXPERIMENT_DESCRIPTION",
    "Projet de classification binaire.",
)

MLFLOW_EXPERIMENT_TAGS: dict[str, str] = {
    "project": "ml-classification",
    "task": "binary-classification",
    "target": TARGET,
    "dataset": "Healthcare",
    "model_registry_name": MODEL_NAME,
}

# Evaluation thresholds
EVAL_F1_MIN = float(os.getenv("EVAL_F1_MIN", "0.7"))
EVAL_ROC_AUC_MIN = float(os.getenv("EVAL_ROC_AUC_MIN", "0.8"))
