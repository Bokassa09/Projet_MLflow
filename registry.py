import mlflow
from mlflow.tracking import MlflowClient

# Connexion au client MLflow 
client = MlflowClient()

experiment = client.get_experiment_by_name("breast-cancer-classification")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.recall DESC"],
    max_results=1
)

meilleur_run = runs[0]
run_id       = meilleur_run.info.run_id
recall     = meilleur_run.data.metrics["recall"]
model       = meilleur_run.data.params["model"]

print(f"Meilleur Run    : {meilleur_run.info.run_name}")
print(f"Run ID          : {run_id}")
print(f"Modèle          : {model}")
print(f"Recall        : {recall:.4f}")

# Enregistrer le modèle 
model_uri  = f"runs:/{run_id}/model"
model_name = "breast-cancer-classifier"

print(f"\n REGISTRY — Enregistrement du modèle")

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name=model_name
)

print(f"Modèle enregistré : {model_name}")
print(f"Version           : {registered_model.version}")

# une description 
client.update_registered_model(
    name=model_name,
    description="Pipeline Semi-Supervisé Self-Training + LogisticRegression ElasticNet — Breast Cancer"
)


client.set_registered_model_tag(
    name=model_name,
    key="framework",
    value="sklearn"
)

client.set_registered_model_tag(
    name=model_name,
    key="dataset",
    value="breast-cancer"
)

print(f"\nModèle prêt dans le Registry")
print(f"   Nom     : {model_name}")
print(f"   Version : {registered_model.version}")

# Workflow 
client.set_registered_model_alias(
    name=model_name,
    alias="staging",
    version=registered_model.version
)
print("Modèle en STAGING : en cours de validation")

# Après validation 
client.set_registered_model_alias(
    name=model_name,
    alias="production", 
    version=registered_model.version
)
print("Modèle validé : PRODUCTION")