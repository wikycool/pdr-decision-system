import pandas as pd
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from typing import Dict, Any, Optional, Tuple
import numpy as np
import logging
from pathlib import Path

def objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
    """
    Objective function for Optuna hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        X: Feature matrix
        y: Target variable
        
    Returns:
        Cross-validation score (negative MSE)
    """
    try:
        # Define hyperparameter search space
        model_name = trial.suggest_categorical('model', ['rf', 'gb', 'svr'])
        
        if model_name == 'rf':
            # Random Forest parameters
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            max_depth = trial.suggest_int('max_depth', 3, 20)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
            
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
            
        elif model_name == 'gb':
            # Gradient Boosting parameters
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
            max_depth = trial.suggest_int('max_depth', 3, 10)
            subsample = trial.suggest_float('subsample', 0.6, 1.0)
            
            model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                random_state=42
            )
            
        else:  # svr
            # Support Vector Regression parameters
            C = trial.suggest_float('C', 0.1, 10.0, log=True)
            gamma = trial.suggest_float('gamma', 1e-5, 1e-1, log=True)
            kernel = trial.suggest_categorical('kernel', ['rbf', 'linear'])
            
            model = SVR(C=C, gamma=gamma, kernel=kernel)
        
        # Perform cross-validation
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        
        return scores.mean()
        
    except Exception as e:
        print(f"Error in objective function: {e}")
        return -float('inf')  # Return worst possible score

def tune_hyperparameters(X: pd.DataFrame, y: pd.Series, n_trials: int = 100) -> Dict[str, Any]:
    """
    Tune hyperparameters using Optuna.
    
    Args:
        X: Feature matrix
        y: Target variable
        n_trials: Number of optimization trials
        
    Returns:
        Dictionary containing best parameters and study results
    """
    try:
        # Create study
        study = optuna.create_study(direction='maximize')
        
        # Optimize
        study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)
        
        # Get best results
        best_params = study.best_params
        best_score = study.best_value
        
        print(f"Best score: {best_score:.4f}")
        print(f"Best parameters: {best_params}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'study': study,
            'n_trials': n_trials
        }
        
    except Exception as e:
        print(f"Error in hyperparameter tuning: {e}")
        return {}

def create_optimized_model(X: pd.DataFrame, y: pd.Series, n_trials: int = 100) -> RandomForestRegressor:
    """
    Create an optimized model using hyperparameter tuning.
    
    Args:
        X: Feature matrix
        y: Target variable
        n_trials: Number of optimization trials
        
    Returns:
        Optimized model
    """
    try:
        # Perform hyperparameter tuning
        results = tune_hyperparameters(X, y, n_trials)
        
        if not results:
            print("Hyperparameter tuning failed, using default model")
            return RandomForestRegressor(random_state=42)
        
        best_params = results['best_params']
        model_name = best_params.get('model', 'rf')
        
        # Create optimized model
        if model_name == 'rf':
            model = RandomForestRegressor(
                n_estimators=best_params.get('n_estimators', 100),
                max_depth=best_params.get('max_depth', 10),
                min_samples_split=best_params.get('min_samples_split', 2),
                min_samples_leaf=best_params.get('min_samples_leaf', 1),
                random_state=42
            )
        elif model_name == 'gb':
            model = GradientBoostingRegressor(
                n_estimators=best_params.get('n_estimators', 100),
                learning_rate=best_params.get('learning_rate', 0.1),
                max_depth=best_params.get('max_depth', 3),
                subsample=best_params.get('subsample', 1.0),
                random_state=42
            )
        else:  # svr
            model = SVR(
                C=best_params.get('C', 1.0),
                gamma=best_params.get('gamma', 'scale'),
                kernel=best_params.get('kernel', 'rbf')
            )
        
        # Fit the model
        model.fit(X, y)
        
        print(f"Created optimized {model_name} model")
        return model
        
    except Exception as e:
        print(f"Error creating optimized model: {e}")
        # Return default model
        return RandomForestRegressor(random_state=42)

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Dictionary containing evaluation metrics
    """
    try:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'rmse': rmse
        }
        
        print("Model Evaluation Results:")
        for metric, value in metrics.items():
            print(f"{metric.upper()}: {value:.4f}")
        
        return metrics
        
    except Exception as e:
        print(f"Error evaluating model: {e}")
        return {}

def save_model(model, file_path: str) -> None:
    """
    Save trained model to file.
    
    Args:
        model: Trained model to save
        file_path: Path to save the model
    """
    try:
        import joblib
        
        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(model, file_path)
        print(f"Model saved to {file_path}")
        
    except Exception as e:
        print(f"Error saving model: {e}")

def load_model(file_path: str):
    """
    Load trained model from file.
    
    Args:
        file_path: Path to the saved model
        
    Returns:
        Loaded model
    """
    try:
        import joblib
        
        if not Path(file_path).exists():
            print(f"Error: Model file {file_path} not found")
            return None
        
        model = joblib.load(file_path)
        print(f"Model loaded from {file_path}")
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def main():
    """Example usage of hyperparameter tuning functions."""
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                     columns=[f'feature_{i}' for i in range(n_features)])
    y = pd.Series(X.iloc[:, 0] * 2 + X.iloc[:, 1] * 1.5 + np.random.randn(n_samples) * 0.1)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Dataset created:")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}")
    
    # Create optimized model
    print("\nStarting hyperparameter optimization...")
    optimized_model = create_optimized_model(X_train, y_train, n_trials=50)
    
    # Evaluate model
    print("\nEvaluating optimized model...")
    metrics = evaluate_model(optimized_model, X_test, y_test)
    
    # Save model
    save_model(optimized_model, "models/optimized_model.pkl")
    
    # Load and test model
    loaded_model = load_model("models/optimized_model.pkl")
    if loaded_model is not None:
        print("\nTesting loaded model...")
        evaluate_model(loaded_model, X_test, y_test)

if __name__ == "__main__":
    main()