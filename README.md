# MACHINE LEARNING ORCHESTRATION : Healthcare

Ce projet vise à construire un pipeline MLOps complet de classification binaire dans un contexte où les données sont sensibles et réglementées. Les objectifs sont :

## Modèle de Prédiction
Développer un modèle capable de classer des observations en deux catégories (diagnostic sain/malade, fraude/légitime, churn/retention, etc.)

## Dataset

Le dataset contient des informations sur les hospitalisations des patients.

## Caractéristiques
- **Source** : `data/dataset.csv`
- **Type** : Classification binaire
- **Domaine** : Santé
- **Format** : Données tabulaires (CSV)

- Données de patients hospitalisés avec conditions médicales variées (Cancer, Obesity, Diabetes, Asthma, etc.)
- Mix de features numériques (Age, Billing Amount) et catégoriques (Gender, Blood Type, Medical Condition, etc.)
- Données réelles avec variations de formats (casse, espaces)
- **Cible binaire** : prédiction d'un résultat médical ou administrative (0/1)

## Pipeline d'apprentissage automatisé
- Charger et prétraiter les données de manière reproductible
- Entraîner des modèles avec optimisation des hyperparamètres (Optuna)
- Évaluer les performances avec métriques pertinentes (F1, ROC-AUC)
- Générer des explications des prédictions (SHAP)
- Automatiser le réentraînement via Airflow

## Déploiement
- **API REST** (FastAPI) pour les prédictions en temps réel
- **Interface frontend** (Streamlit) pour la visualisation et l'exploration
- **Infrastructure containerisée** (Docker) pour la portabilité et la scalabilité

## Gestion du cycle de vie
- Tracking des expériences avec MLflow
- Versionning des modèles et des artefacts
- Reproductibilité et traçabilité complètes
