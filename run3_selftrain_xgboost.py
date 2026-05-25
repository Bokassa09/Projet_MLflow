import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

# Données 
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Self-training 
rng = np.random.RandomState(42)
mask_unlabeled = rng.rand(len(X_train)) < 0.90

X_labeled = X_train[~mask_unlabeled]
y_labeled = y_train[~mask_unlabeled]

X_unlabeled = X_train[mask_unlabeled]

SEUIL = 0.70
MAX_ITER = 20

X_lab = X_labeled.copy()
y_lab = y_labeled.copy()
X_unlab = X_unlabeled.copy()

iterations_faites = 0
exemples_ajoutes = 0


for i in range(MAX_ITER):

    xgb = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )

    clf = CalibratedClassifierCV(xgb, cv=2)
    clf.fit(X_lab, y_lab)

    if len(X_unlab) == 0:
        break

    probas = clf.predict_proba(X_unlab)
    confiance = probas.max(axis=1)
    pseudo_labels = probas.argmax(axis=1)

    idx = np.where(confiance >= SEUIL)[0]

    if len(idx) == 0:
        break

    X_lab = np.vstack([X_lab, X_unlab[idx]])
    y_lab = np.concatenate([y_lab, pseudo_labels[idx]])
    X_unlab = np.delete(X_unlab, idx, axis=0)

    iterations_faites += 1
    exemples_ajoutes += len(idx)


mlflow.set_experiment("breast-cancer-classification")

with mlflow.start_run(run_name="XGBoost_SelfTraining_Calibrated"):

    # Params
    mlflow.log_params({
        "model": "XGBoost",
        "approche": "self_training",
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3,
        "calibration": "sigmoid",
        "seuil_confiance": SEUIL,
        "iterations": iterations_faites,
        "exemples_ajoutes": exemples_ajoutes
    })

    # Prediction
    y_pred = clf.predict(X_test)

    # Metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }

    mlflow.log_metrics(metrics)

    # Classification report 
    report = classification_report(y_test, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()

    plt.figure(figsize=(8,5))
    plt.table(
        cellText=df.values,
        colLabels=df.columns,
        rowLabels=df.index,
        loc='center'
    )
    plt.axis('off')

    plt.savefig("figures/xgb_selftraining_report.png")
    mlflow.log_artifact("figures/xgb_selftraining_report.png")
    plt.close()

    # Model logging
    mlflow.sklearn.log_model(clf, "model")

    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1 Score  : {metrics['f1_score']:.4f}")

    print("\n Run MLflow terminé")