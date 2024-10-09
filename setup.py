from setuptools import setup, find_packages

setup(
    name="census_income_prediction",
    version="0.1",
    description="A Python package for census income prediction",
    author="Esraa Kamel",
    packages=find_packages(),
    install_requires=[
        "pandas==2.2.3",
        "scikit-learn==1.5.2",
        "joblib==1.4.2",
        "fastapi==0.115.0",
        "uvicorn==0.31.0",
        "pydantic==2.9.2"
    ],
)
