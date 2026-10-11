import pandas as pd

from fastapi.testclient import TestClient
from sklearn.datasets import load_breast_cancer

from app.main import app


client = TestClient(
    app
)

data = (
    load_breast_cancer()
)

sample = pd.Series(
    data.data[0],
    index=data.feature_names
)

valid_payload = {
    feature.replace(" ", "_"):
        float(
            sample[feature]
        )
    for feature
    in data.feature_names
}


def test_health():

    response = client.get(
        "/health"
    )

    assert (
        response.status_code
        == 201
    )

    assert (
        response.json()[
            "status"
        ]
        == "healthy"
    )


def test_model_info():

    response = client.get(
        "/model-info"
    )

    assert (
        response.status_code
        == 200
    )

    result = (
        response.json()
    )

    assert (
        result[
            "model_version"
        ]
        == "1.0.0"
    )

    assert (
        result[
            "number_of_features"
        ]
        == 30
    )


def test_valid_prediction():

    response = client.post(
        "/predict",
        json=valid_payload
    )

    assert (
        response.status_code
        == 200
    )

    result = (
        response.json()
    )

    assert (
        result[
            "prediction"
        ]
        in [0, 1]
    )

    assert (
        result[
            "class_name"
        ]
        in [
            "malignant",
            "benign"
        ]
    )

    assert (
        0.0
        <= result[
            "probability"
        ]
        <= 1.0
    )


def test_missing_feature():

    payload = (
        valid_payload.copy()
    )

    payload.pop(
        "mean_radius"
    )

    response = client.post(
        "/predict",
        json=payload
    )

    assert (
        response.status_code
        == 422
    )


def test_wrong_data_type():

    payload = (
        valid_payload.copy()
    )

    payload[
        "mean_radius"
    ] = "invalid"

    response = client.post(
        "/predict",
        json=payload
    )

    assert (
        response.status_code
        == 422
    )


def test_extra_feature():

    payload = (
        valid_payload.copy()
    )

    payload[
        "unexpected_feature"
    ] = 100

    response = client.post(
        "/predict",
        json=payload
    )

    assert (
        response.status_code
        == 422
    )
