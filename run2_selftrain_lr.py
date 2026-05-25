import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# Données
data = load_breast_cancer()
X, y = data.data, data.target


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rng = np.random.RandomState(42)
mask_unlabeled = rng.rand(len(X_train)) < 0.98
X_labeled   = X_train[~mask_unlabeled]
y_labeled   = y_train[~mask_unlabeled]
X_unlabeled = X_train[mask_unlabeled]

# Self-Training
SEUIL = 0.90
MAX_ITER = 20

X_lab   = X_labeled.copy()
y_lab   = y_labeled.copy()
X_unlab = X_unlabeled.copy()

iterations_faites = 0
exemples_ajoutes  = 0

for i in range(MAX_ITER):
    clf = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(
        max_iter=1000,
        random_state=42,
        penalty='elasticnet',
        solver='saga',
        l1_ratio=0.5
    ))])



    clf.fit(X_lab, y_lab)

    if len(X_unlab) == 0:
        break

    probas        = clf.predict_proba(X_unlab)
    confiance     = probas.max(axis=1)
    pseudo_labels = probas.argmax(axis=1)
    idx_confiants = np.where(confiance >= SEUIL)[0]

    if len(idx_confiants) == 0:
        break

    X_lab   = np.vstack([X_lab, X_unlab[idx_confiants]])
    y_lab   = np.concatenate([y_lab, pseudo_labels[idx_confiants]])
    X_unlab = np.delete(X_unlab, idx_confiants, axis=0)

    iterations_faites += 1
    exemples_ajoutes  += len(idx_confiants)

# MLflow
mlflow.set_experiment("breast-cancer-classification")

with mlflow.start_run(run_name="SelfTraining_LogisticRegression"):

    
    params = {
        "model":          "LogisticRegression",
        "approche":        "self_training",
        "labels_percent":  0.02,
        "seuil_confiance": SEUIL,
        "max_iter":        MAX_ITER,
        "iterations_faites": iterations_faites,
        "exemples_ajoutes":  exemples_ajoutes,
        "penalty":         "elasticnet",
        "l1_ratio":        0.5
    }
    mlflow.log_params(params)

    
    y_pred = clf.predict(X_test)

    report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

    df = pd.DataFrame(report).transpose()

    plt.figure(figsize=(8,5))
    plt.table(
    cellText=df.values,
    colLabels=df.columns,
    rowLabels=df.index,
    loc='center'
)

    plt.axis('off')

    

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1_score":  f1_score(y_test, y_pred)
    }
    mlflow.log_metrics(metrics)

    # save le modèle

    mlflow.sklearn.log_model(
    clf,
    "model"
)


    plt.savefig("figures/classification_report_selftrain_lr.png")

    mlflow.log_artifact(
    "figures/classification_report_selftrain_lr.png"
)



    print(f"Iterations    : {iterations_faites}")
    print(f"Exemples ajoutés : {exemples_ajoutes}")
    print(f"Accuracy      : {metrics['accuracy']:.4f}")
    print(f"Precision     : {metrics['precision']:.4f}")
    print(f"Recall        : {metrics['recall']:.4f}")
    print(f"F1 Score      : {metrics['f1_score']:.4f}")
    print("\nTRACKING  Run enregistré dans MLflow ")
    print("MODELS  Modèle sauvegardé dans MLflow ")

    plt.show()
    plt.close()