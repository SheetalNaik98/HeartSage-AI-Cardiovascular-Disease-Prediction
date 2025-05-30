from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

def train_models(X_train, y_train):
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'KNN': KNeighborsClassifier()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"{name} trained successfully.")

    return models

def hyperparameter_tuning(preprocessor, X_train, y_train):
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ])

    param_grid = {'classifier__C': [0.1, 1.0, 10.0, 100.0]}
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    print("Best parameters from GridSearchCV:", grid_search.best_params_)
    print("Best CV score:", grid_search.best_score_)

    return grid_search.best_estimator_
