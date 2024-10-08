import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def train_rf_model(X_train, y_train, seed=42,
                   custom_params=None, save_model=False):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    # Fit random forest classifier and tune parameters using grid search
    # during CV
    rfc = RandomForestClassifier(random_state=seed)
    if custom_params is not None:
        param_grid = custom_params
    else:
        param_grid = {
            'n_estimators': [50],  # 'n_estimators': [50, 100, 200],
            'max_features': ['sqrt'],
            'max_depth': [5, 10],  # 'max_depth': [5, 10, 100],
            'criterion': ['gini', 'entropy']
        }
    cv_rfc = GridSearchCV(
        estimator=rfc,
        param_grid=param_grid,
        cv=5,
        scoring='f1')
    cv_rfc.fit(X_train, y_train)

    if save_model is True:
        joblib.dump(cv_rfc.best_estimator_, './model/rfc_model.pkl')

    return cv_rfc.best_estimator_
