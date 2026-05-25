# Breast Cancer Classifier — MLflow Experiment Tracking

Projet de comparaison de modèles ML avec tracking complet via MLflow.
Extension du projet [Breast Cancer API](https://github.com/Bokassa09/breast-cancer-api)
on passe du notebook au MLOps professionnel.

## Objectif

Comparer 4 familles de modèles dans un pipeline Semi-Supervisé (Self-Training)
avec seulement **2% de données étiquetées**, et tracker toutes les expériences
avec MLflow pour choisir le meilleur modèle de façon rigoureuse et reproductible.

## Ce que ce projet démontre

- Pourquoi on ne choisit jamais un modèle au feeling en entreprise
- Comment MLflow rend les expériences ML reproductibles et comparables
- L'impact des hyperparamètres sur les performances (visualisé dans MLflow)
- Le cycle de vie complet d'un modèle : Tracking → Registry → Production

## Pipeline MLflow complet

## Expériences réalisées : 7 Runs

NB: Pour voir le tableau de comparaison des 7 runs, allez dans le dossier figures et regardez l’image mlflow_ui_comparaison_des_3_modeles.png, qui provient directement de l’interface MLflow UI

## Meilleur modèle

**Run 2 — Self-Training + Logistic Regression ElasticNet**

Choisi selon le **Recall** (métrique prioritaire en médecine) :
- Recall = 100% → zéro cancer raté
- Enregistré en Production dans le MLflow Model Registry

> En médecine, rater un vrai cancer (faux négatif) est bien plus grave
> qu'une fausse alarme (faux positif). Le choix de la métrique dépend
> toujours du contexte métier.

## Leçons MLOps apprises

- XGBoost nécessite une **calibration des probabilités** avec peu de données
- La Régression Logistique avec ElasticNet surpasse des modèles plus complexes
  dans un contexte semi-supervisé avec très peu de labels
- Le **Recall** prime sur l'Accuracy dans les applications médicales

## Stack technique

## Lancer le projet

```bash
# Cloner le repo
git clone https://github.com/Bokassa09/Projet_MLflow.git
cd Projet_MLflow

# Créer l'environnement
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt

# Lancer les expériences
python3 run1_baseline.py
python3 run2_selftrain_lr.py
python3 run3_selftrain_xgboost.py
python3 run4_selftrain_mlp.py  # modifier les hyperparamètres pour runs 5,6,7

# Enregistrer le meilleur modèle
python3 registry.py

# Lancer l'interface MLflow
mlflow ui
# → http://127.0.0.1:5000
```

##  Structure du projet

## Projet lié

Ce projet est la suite naturelle de :

[Breast Cancer API](https://github.com/Bokassa09/breast-cancer-api)

déploiement FastAPI + Docker + SQLite + Hugging Face Spaces