import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#  Simulation 2% labels
rng = np.random.RandomState(42)
mask_unlabeled = rng.rand(len(X_train)) < 0.98

X_labeled = X_train[~mask_unlabeled]
y_labeled = y_train[~mask_unlabeled]


clf = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

# MLflow
mlflow.set_experiment("breast-cancer-classification")

with mlflow.start_run(run_name="LogReg_Baseline"):

    # Params
    mlflow.log_params({
        "model": "LogisticRegression",
        "approche": "baseline_supervisee",
        "labels_percent": 0.02,
        "scaler": "StandardScaler",
        "max_iter": 1000,
        "random_state": 42
    })

    # Training
    clf.fit(X_labeled, y_labeled)
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

    plt.savefig("figures/logreg_baseline_report.png")
    mlflow.log_artifact("figures/logreg_baseline_report.png")
    plt.close()

    #  Model logging
    mlflow.sklearn.log_model(clf, "model")

    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1 Score  : {metrics['f1_score']:.4f}")

    print("\n Run MLflow terminé")