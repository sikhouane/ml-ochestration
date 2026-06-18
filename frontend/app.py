"""Interface Streamlit enrichie pour le modèle de classification médical.

Onglets :
- Prédiction individuelle
- Prédiction CSV
- Évaluation du modèle
- Wiki documentation
- Airflow DAGs disponibles

Lancement local :
    uv run streamlit run frontend/app.py

Dans Docker :
    streamlit run frontend/app.py --server.address=0.0.0.0 --server.port=8501
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import requests
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"

AIRFLOW_API_URL = os.environ.get(
    "AIRFLOW_API_URL",
    "http://airflow-webserver:8080/api/v1/dags",
)
AIRFLOW_USERNAME = os.environ.get("AIRFLOW_USERNAME", "admin")
AIRFLOW_PASSWORD = os.environ.get("AIRFLOW_PASSWORD", "admin")

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


def inject_css() -> None:
    """Ajoute un style visuel simple au frontend."""
    st.markdown(
        """
        <style>
        .main {
            background: #f8fafc;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .app-header {
            padding: 1.5rem 1.8rem;
            border-radius: 22px;
            background: linear-gradient(135deg, #0f172a 0%, #2563eb 100%);
            color: white;
            margin-bottom: 1.5rem;
            box-shadow: 0 15px 40px rgba(15, 23, 42, 0.20);
        }

        .app-header h1 {
            margin: 0;
            font-size: 2.2rem;
        }

        .app-header p {
            margin-top: 0.5rem;
            color: #dbeafe;
            font-size: 1rem;
        }

        .wiki-card {
            padding: 1.2rem;
            border-radius: 18px;
            background: white;
            border: 1px solid #e5e7eb;
            margin-bottom: 1rem;
            box-shadow: 0 10px 25px rgba(15, 23, 42, 0.06);
        }

        .success-pill {
            display: inline-block;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            background: #dcfce7;
            color: #166534;
            font-weight: 700;
        }

        .danger-pill {
            display: inline-block;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            background: #fee2e2;
            color: #991b1b;
            font-weight: 700;
        }

        section[data-testid="stSidebar"] {
            background: #0f172a;
        }

        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span {
            color: #e5e7eb;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_header() -> None:
    st.markdown(
        """
        <div class="app-header">
            <h1>🏥 Prédiction médicale</h1>
            <p>
                Interface Streamlit pour tester le modèle, évaluer ses performances,
                consulter la documentation et suivre les DAGs Airflow.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model(path: str) -> Any:
    """Charge le modèle entraîné et le garde en cache."""
    return joblib.load(path)


@st.cache_data
def load_test_data() -> tuple[pd.DataFrame, pd.Series]:
    """Recharge le dataset puis récupère le jeu de test."""
    from mlproject.data import load_data, split

    df = load_data()
    _, x_test, _, y_test = split(df)
    return x_test, y_test


def build_features() -> pd.DataFrame:
    """Construit une ligne de variables compatible avec le dataset nettoyé."""
    left, right = st.columns(2)

    with left:
        age = st.number_input("Âge", min_value=0, max_value=120, value=45)
        gender = st.selectbox("Genre", CATEGORICAL_OPTIONS["gender"])
        blood_type = st.selectbox("Groupe sanguin", CATEGORICAL_OPTIONS["blood_type"])
        medical_condition = st.selectbox(
            "Condition médicale",
            CATEGORICAL_OPTIONS["medical_condition"],
        )
        admission_type = st.selectbox(
            "Type d'admission",
            CATEGORICAL_OPTIONS["admission_type"],
        )
        admission_date = st.date_input("Date d'admission", value=date.today())

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
        medication = st.selectbox("Médicament", CATEGORICAL_OPTIONS["medication"])
        test_results = st.selectbox("Résultat du test", CATEGORICAL_OPTIONS["test_results"])
        discharge_date = st.date_input("Date de sortie", value=date.today())

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
    prediction = int(model.predict(features)[0])

    st.subheader("Résultat de la prédiction")

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.markdown(
                '<span class="danger-pill">Classe prédite : 1</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="success-pill">Classe prédite : 0</span>',
                unsafe_allow_html=True,
            )

    with col2:
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            if len(probabilities) >= 2:
                st.metric("Probabilité de la classe 1", f"{probabilities[1] * 100:.2f} %")

    with st.expander("Variables envoyées au modèle"):
        st.dataframe(features, use_container_width=True)


def batch_prediction(model: Any) -> None:
    """Permet d'effectuer des prédictions sur un fichier CSV."""
    st.subheader("Prédiction par lot")

    uploaded_file = st.file_uploader("Dépose un CSV déjà nettoyé", type=["csv"])

    if uploaded_file is None:
        st.info("Dépose un fichier CSV pour générer des prédictions en lot.")
        return

    dataframe = pd.read_csv(uploaded_file)

    st.write("Aperçu du fichier :")
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
            st.error("La prédiction CSV a échoué.")
            st.exception(exc)


def evaluation_tab(model: Any) -> None:
    """Affiche les métriques, la matrice de confusion et le classification report."""
    st.subheader("Évaluation du modèle")

    st.write(
        "Cette page recharge le dataset, applique le même découpage que l'entraînement "
        "et évalue le modèle sauvegardé sur le jeu de test."
    )

    try:
        x_test, y_test = load_test_data()
        predictions = model.predict(x_test)

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy_score(y_test, predictions):.3f}")
        col2.metric("F1-score", f"{f1_score(y_test, predictions):.3f}")

        probabilities = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(x_test)[:, 1]
            col3.metric("ROC AUC", f"{roc_auc_score(y_test, probabilities):.3f}")
        else:
            col3.metric("ROC AUC", "N/A")

        st.divider()

        left, right = st.columns([1, 1])

        with left:
            st.markdown("### Matrice de confusion")
            cm = confusion_matrix(y_test, predictions)
            cm_df = pd.DataFrame(
                cm,
                index=["Réel 0", "Réel 1"],
                columns=["Prédit 0", "Prédit 1"],
            )
            st.dataframe(cm_df, use_container_width=True)

            st.markdown("### Répartition des prédictions")
            pred_counts = pd.DataFrame(
                {
                    "classe": ["Prédit 0", "Prédit 1"],
                    "nombre": [
                        int((predictions == 0).sum()),
                        int((predictions == 1).sum()),
                    ],
                }
            ).set_index("classe")
            st.bar_chart(pred_counts)

        with right:
            st.markdown("### Classification report")
            report = classification_report(
                y_test,
                predictions,
                output_dict=True,
                zero_division=0,
            )
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)

        with st.expander("Voir quelques probabilités"):
            if probabilities is not None:
                preview = pd.DataFrame(
                    {
                        "target": list(y_test),
                        "prediction": predictions,
                        "probability_class_1": probabilities,
                    }
                ).head(50)
                st.dataframe(preview, use_container_width=True)
            else:
                st.info("Le modèle ne fournit pas predict_proba.")

    except Exception as exc:
        st.error(
            "Impossible d'évaluer le modèle. Vérifie que le dataset, "
            "mlproject.data.load_data et mlproject.data.split sont disponibles."
        )
        st.exception(exc)


def wiki_tab() -> None:
    """Affiche des liens utiles vers les documentations."""
    st.subheader("Wiki du projet")

    docs = [
        {
            "name": "Apache Airflow",
            "description": "Orchestration des DAGs, scheduling et suivi des tâches.",
            "url": "https://airflow.apache.org/docs/",
            "emoji": "🌬️",
        },
        {
            "name": "MLflow",
            "description": "Tracking des expériences, modèles, métriques et artefacts.",
            "url": "https://mlflow.org/docs/latest/index.html",
            "emoji": "🧪",
        },
        {
            "name": "FastAPI",
            "description": "Création de l'API REST pour servir le modèle.",
            "url": "https://fastapi.tiangolo.com/",
            "emoji": "⚡",
        },
        {
            "name": "Streamlit",
            "description": "Interface web pour tester le modèle facilement.",
            "url": "https://docs.streamlit.io/",
            "emoji": "🎨",
        },
        {
            "name": "Scikit-learn",
            "description": "Pipelines, preprocessing et modèles de machine learning.",
            "url": "https://scikit-learn.org/stable/user_guide.html",
            "emoji": "🤖",
        },
    ]

    for doc in docs:
        st.markdown(
            f"""
            <div class="wiki-card">
                <h3>{doc["emoji"]} {doc["name"]}</h3>
                <p>{doc["description"]}</p>
                <a href="{doc["url"]}" target="_blank">Ouvrir la documentation</a>
            </div>
            """,
            unsafe_allow_html=True,
        )


@st.cache_data(ttl=30)
def fetch_airflow_dags(api_url: str, username: str, password: str) -> pd.DataFrame:
    """Récupère les DAGs disponibles depuis l'API REST Airflow."""
    response = requests.get(api_url, auth=(username, password), timeout=10)
    response.raise_for_status()

    payload = response.json()
    dags = payload.get("dags", [])

    rows = []
    for dag in dags:
        rows.append(
            {
                "dag_id": dag.get("dag_id"),
                "is_paused": dag.get("is_paused"),
                "is_active": dag.get("is_active"),
                "description": dag.get("description"),
                "timetable_description": dag.get("timetable_description"),
            }
        )

    return pd.DataFrame(rows)


def airflow_tab() -> None:
    """Affiche les DAGs disponibles dans Airflow."""
    st.subheader("DAGs Airflow disponibles")

    st.write(
        "Cette page interroge l'API REST d'Airflow. Dans Docker Compose, l'URL "
        "utilisée est généralement `http://airflow-webserver:8080/api/v1/dags`."
    )

    with st.expander("Configuration Airflow"):
        api_url = st.text_input("AIRFLOW_API_URL", value=AIRFLOW_API_URL)
        username = st.text_input("AIRFLOW_USERNAME", value=AIRFLOW_USERNAME)
        password = st.text_input("AIRFLOW_PASSWORD", value=AIRFLOW_PASSWORD, type="password")

    if st.button("Rafraîchir la liste des DAGs"):
        st.cache_data.clear()

    try:
        dags_df = fetch_airflow_dags(api_url, username, password)

        if dags_df.empty:
            st.warning("Aucun DAG trouvé dans Airflow.")
            return

        total = len(dags_df)
        active = int(dags_df["is_active"].fillna(False).sum())
        paused = int(dags_df["is_paused"].fillna(False).sum())

        col1, col2, col3 = st.columns(3)
        col1.metric("DAGs", total)
        col2.metric("Actifs", active)
        col3.metric("En pause", paused)

        st.dataframe(dags_df, use_container_width=True)

    except Exception as exc:
        st.error(
            "Impossible de récupérer les DAGs Airflow. Vérifie que le webserver "
            "Airflow tourne et que l'API basic_auth est activée."
        )
        st.exception(exc)


def sidebar(model_path: str) -> None:
    """Affiche les informations de la sidebar."""
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.write("Chemin du modèle :")
    st.sidebar.code(model_path)

    st.sidebar.divider()
    st.sidebar.write("Services utiles :")
    st.sidebar.markdown("- Streamlit : `8501`")
    st.sidebar.markdown("- FastAPI : `8000`")
    st.sidebar.markdown("- MLflow : `5000`")
    st.sidebar.markdown("- Airflow : `8080`")


def main() -> None:
    st.set_page_config(
        page_title="Prédiction médicale - MLOps",
        page_icon="🏥",
        layout="wide",
    )

    inject_css()
    show_header()

    model_path = st.sidebar.text_input("Chemin du modèle", value=str(DEFAULT_MODEL_PATH))
    sidebar(model_path)

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

    tab_single, tab_batch, tab_eval, tab_wiki, tab_airflow = st.tabs(
        [
            "🔮 Prédiction individuelle",
            "📁 Prédiction CSV",
            "📊 Évaluation",
            "📚 Wiki",
            "🌬️ Airflow",
        ]
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
                    "La prédiction a échoué. Vérifie que les colonnes du formulaire "
                    "correspondent aux variables utilisées pendant l'entraînement."
                )
                st.exception(exc)

    with tab_batch:
        batch_prediction(model)

    with tab_eval:
        evaluation_tab(model)

    with tab_wiki:
        wiki_tab()

    with tab_airflow:
        airflow_tab()


if __name__ == "__main__":
    main()
