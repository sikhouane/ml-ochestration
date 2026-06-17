"""Interface Streamlit simple pour le modèle de classification médical.

Lancement :
    uv run streamlit run app.py
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"

CATEGORICAL_OPTIONS = {
    "gender": ["Male", "Female"],
    "blood_type": ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"],
    "medical_condition": [
        "Arthritis",
        "Asthma",
        "Cancer",
        "Diabetes",
        "Hypertension",
        "Obesity",
    ],
    "insurance_provider": [
        "Aetna",
        "Blue Cross",
        "Cigna",
        "Medicare",
        "UnitedHealthcare",
    ],
    "admission_type": ["Elective", "Emergency", "Urgent"],
    "medication": [
        "Aspirin",
        "Ibuprofen",
        "Lipitor",
        "Paracetamol",
        "Penicillin",
    ],
    "test_results": ["Abnormal", "Inconclusive", "Normal"],
}


@st.cache_resource
def load_model(path: str) -> Any:
    """Charge le modèle entraîné et le garde en cache."""
    return joblib.load(path)


def build_features() -> pd.DataFrame:
    """Construit une ligne de variables compatible avec le dataset nettoyé."""
    left, right = st.columns(2)

    with left:
        age = st.number_input("Âge", min_value=0, max_value=120, value=45)
        gender = st.selectbox("Genre", CATEGORICAL_OPTIONS["gender"])
        blood_type = st.selectbox(
            "Groupe sanguin",
            CATEGORICAL_OPTIONS["blood_type"],
        )
        medical_condition = st.selectbox(
            "Condition médicale",
            CATEGORICAL_OPTIONS["medical_condition"],
        )
        admission_type = st.selectbox(
            "Type d'admission",
            CATEGORICAL_OPTIONS["admission_type"],
        )
        admission_date = st.date_input(
            "Date d'admission",
            value=date.today(),
        )

    with right:
        insurance_provider = st.selectbox(
            "Assurance",
            CATEGORICAL_OPTIONS["insurance_provider"],
        )
        billing_amount = st.number_input(
            "Montant facturé",
            min_value=0.0,
            value=10000.0,
            step=100.0,
        )
        room_number = st.number_input(
            "Numéro de chambre",
            min_value=1,
            value=101,
            step=1,
        )
        medication = st.selectbox(
            "Médicament",
            CATEGORICAL_OPTIONS["medication"],
        )
        test_results = st.selectbox(
            "Résultat du test",
            CATEGORICAL_OPTIONS["test_results"],
        )
        discharge_date = st.date_input(
            "Date de sortie",
            value=date.today(),
        )

    length_of_stay = (discharge_date - admission_date).days

    return pd.DataFrame(
        [
            {
                "age": age,
                "gender": gender,
                "blood_type": blood_type,
                "medical_condition": medical_condition,
                "insurance_provider": insurance_provider,
                "billing_amount": billing_amount,
                "room_number": int(room_number),
                "admission_type": admission_type,
                "medication": medication,
                "test_results": test_results,
                "admission_year": admission_date.year,
                "admission_month": admission_date.month,
                "admission_day_of_week": admission_date.weekday(),
                "discharge_year": discharge_date.year,
                "discharge_month": discharge_date.month,
                "discharge_day_of_week": discharge_date.weekday(),
                "length_of_stay_days": max(length_of_stay, 0),
            }
        ]
    )


def show_prediction(model: Any, features: pd.DataFrame) -> None:
    """Affiche la prédiction et la probabilité quand elle est disponible."""
    prediction = model.predict(features)[0]

    st.subheader("Résultat")
    if int(prediction) == 1:
        st.error("Classe prédite : 1")
    else:
        st.success("Classe prédite : 0")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        if len(probabilities) >= 2:
            st.metric(
                "Probabilité de la classe 1",
                f"{probabilities[1] * 100:.1f} %",
            )

    with st.expander("Variables envoyées au modèle"):
        st.dataframe(features, use_container_width=True)


def batch_prediction(model: Any) -> None:
    """Permet d'effectuer des prédictions sur un fichier CSV."""
    st.subheader("Prédiction par lot")
    uploaded_file = st.file_uploader(
        "Dépose un CSV déjà nettoyé",
        type=["csv"],
    )

    if uploaded_file is None:
        return

    dataframe = pd.read_csv(uploaded_file)
    st.dataframe(dataframe.head(), use_container_width=True)

    if "target" in dataframe.columns:
        features = dataframe.drop(columns=["target"])
    else:
        features = dataframe.copy()

    if st.button("Lancer les prédictions du CSV"):
        try:
            results = dataframe.copy()
            results["prediction"] = model.predict(features)

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(features)
                if probabilities.shape[1] >= 2:
                    results["probability_class_1"] = probabilities[:, 1]

            st.success(f"{len(results)} prédiction(s) générée(s).")
            st.dataframe(results, use_container_width=True)

            csv_data = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Télécharger les résultats",
                data=csv_data,
                file_name="predictions.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.exception(exc)


def main() -> None:
    st.set_page_config(
        page_title="Prédiction médicale",
        page_icon="🏥",
        layout="wide",
    )

    st.title("🏥 Prédiction médicale")
    st.caption("Interface simple pour tester le modèle entraîné.")

    model_path = st.sidebar.text_input(
        "Chemin du modèle",
        value=str(DEFAULT_MODEL_PATH),
    )

    try:
        model = load_model(model_path)
        st.sidebar.success("Modèle chargé")
    except FileNotFoundError:
        st.error(
            "Modèle introuvable. Entraîne d'abord le modèle ou vérifie "
            f"le chemin : {model_path}"
        )
        st.stop()
    except Exception as exc:
        st.error("Impossible de charger le modèle.")
        st.exception(exc)
        st.stop()

    tab_single, tab_batch = st.tabs(
        ["Prédiction individuelle", "Prédiction CSV"]
    )

    with tab_single:
        with st.form("prediction_form"):
            features = build_features()
            submitted = st.form_submit_button("Prédire")

        if submitted:
            try:
                show_prediction(model, features)
            except Exception as exc:
                st.error(
                    "La prédiction a échoué. Vérifie que les colonnes du "
                    "formulaire correspondent aux variables utilisées "
                    "pendant l'entraînement."
                )
                st.exception(exc)

    with tab_batch:
        batch_prediction(model)


if __name__ == "__main__":
    main()
