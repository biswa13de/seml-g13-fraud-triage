import warnings

import pytest

warnings.filterwarnings("ignore")

MODEL_FIXTURES = {"champion_model", "explainer"}


def pytest_collection_modifyitems(items):
    """Tag every test that uses the registered champion model, so CI (which has
    no real dataset and an empty MLflow registry) can exclude them with
    `-m "not requires_model"`. Tagging by fixture means a new model-dependent
    test is picked up automatically instead of needing a CI filter update."""
    for item in items:
        if MODEL_FIXTURES & set(getattr(item, "fixturenames", ())):
            item.add_marker(pytest.mark.requires_model)


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
