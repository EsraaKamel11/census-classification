from fastapi.testclient import TestClient
import pandas as pd
import pytest
from main import app

client = TestClient(app)


@pytest.fixture
def test_df():
    """
    Pandas df with two test cases for inferences
    First row should return '<=50K' or 0
    Second row should return '>50K' or 1
    """
    test_dict = {
        'age': [18, 42],
        'workclass': ['Never-worked', 'Private'],
        'fnlgt': [189778, 159449],
        'education': ['Preschool', 'Bachelors'],
        'education_num': [2, 13],
        'marital_status': ['Never-married', 'Married-civ-spouse'],
        'occupation': ['Priv-house-serv', 'Exec-managerial'],
        'relationship': ['Own-child', 'Husband'],
        'race': ['Amer-Indian-Eskimo', 'White'],
        'sex': ['Female', 'Male'],
        'capital_gain': [0, 5178],
        'capital_loss': [0, 0],
        'hours_per_week': [3, 40],
        'native_country': ['Mexico', 'United-States'],
    }
    test_df = pd.DataFrame(test_dict)
    return test_df


# Test GET on root
def test_root_status():
    r = client.get("/")
    assert r.status_code == 200


def test_root_request_contents():
    r = client.get("/")
    assert r.json() == {
        "message": "Welcome to EK's API. This is a greeting only."}


# Test POST method with test data
def test_post_low_salary(test_df):
    test_json = test_df.head(1).to_dict(orient="records")[0]
    r = client.post(
        '/prediction',
        json=test_json,
        headers={'Content-Type': 'application/json'}
    )
    print(r.json())
    assert r.status_code == 200
    assert r.json() == {'results': '>50K'}


def test_post_high_salary(test_df):
    test_json = test_df.tail(1).to_dict(orient="records")[0]
    print(test_json)
    r = client.post(
        '/prediction',
        json=test_json,
        headers={'Content-Type': 'application/json'}
    )

    assert r.status_code == 200
    assert r.json() == {'results': '>50K'}