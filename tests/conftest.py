import warnings

import pytest

warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def champion_model():
    import mlflow.lightgbm

    from common.config import settings
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    return mlflow.lightgbm.load_model(f"models:/{settings.model_name}@{settings.model_alias}")


@pytest.fixture(scope="session")
def explainer(champion_model):
    from services.scoring.explain import ReasonCodeExplainer
    return ReasonCodeExplainer(champion_model)
