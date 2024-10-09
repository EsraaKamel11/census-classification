import pytest
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from ml import model as model
from ml import process_data as process_data


@pytest.fixture
def rf_model():
    rf_model = joblib.load('../model/rfc_model.pkl')
    return rf_model


@pytest.fixture
def full_data():
    df_path = "../data/census.csv"
    df = pd.read_csv(df_path)
    return df


@pytest.fixture
def test_data():
    df_path = "../data/census.csv"
    df = pd.read_csv(df_path)
    return df.sample(20, random_state=42)


@pytest.fixture
def encoder():
    encoder = joblib.load('../model/encoder.pkl')
    return encoder


@pytest.fixture
def categorical_features():
    return ["workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country"
            ]


def test_fitting(test_data, categorical_features):
    """
    Tests whether fitting model returns RandomForestClassifier
    """
    processed_data = process_data.process_data(test_data, categorical_features)
    fitted_model = model.train_rf_model(
        X_train=processed_data[0],
        y_train=processed_data[1],
        save_model=False,
        custom_params={
            'n_estimators': [20],
            'max_features': ['sqrt'],
            'max_depth': [5],
            'criterion': ['gini', 'entropy']
        }
    )
    assert isinstance(fitted_model, RandomForestClassifier)


def test_encoder(categorical_features, full_data, test_data):
    """
    Tests OneHotEncoder returns same number of binary features with reduced feature input
    """
    processed_full = process_data.process_data(
        full_data, categorical_features=categorical_features)
    encoder = processed_full[2]
    processed_sample = process_data.process_data(
        test_data,
        categorical_features=categorical_features,
        encoder=encoder,
        training=False)
    assert processed_full[0].shape[1] == processed_sample[0].shape[1]


def test_inference(encoder, rf_model, test_data, categorical_features):
    """
    Tests whether random sample of census data produces binary output
    """
    processed_data_test = process_data.process_data(
        test_data,
        categorical_features=categorical_features,
        encoder=encoder,
        training=False
    )
    pred = rf_model.predict(processed_data_test[0])
    assert set(np.unique(pred)) == {0, 1}