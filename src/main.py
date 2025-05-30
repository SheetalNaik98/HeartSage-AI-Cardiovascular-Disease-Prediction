import os
from data_preprocessing import load_and_preprocess_data
from model_training import train_models, hyperparameter_tuning
from evaluation import evaluate_model
from visualization import exploratory_data_analysis

def main():
    data_path = "data/heart_disease_data.csv"
    results_dir = "results"
    os.makedirs(f"{results_dir}/figures", exist_ok=True)

    print("Performing EDA...")
    exploratory_data_analysis(data_path, output_dir=f"{results_dir}/figures")

    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data(data_path)

    print("Training models...")
    models = train_models(X_train, y_train)

    print("Evaluating models...")
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, name, results_dir=results_dir)
        print(metrics)

    print("Hyperparameter tuning with GridSearchCV for Logistic Regression...")
    best_model = hyperparameter_tuning(preprocessor, X_train, y_train)
    best_metrics = evaluate_model(best_model, X_test, y_test, "Best_LogisticRegression", results_dir=results_dir)
    print(best_metrics)

if __name__ == "__main__":
    main()
