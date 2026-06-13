"""
Model training module.
Placeholder for model training and evaluation functionality.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer

# Import sklearn models
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


# Model family classification
LINEAR_DISTANCE_MODELS = {
    "LogisticRegression", "LinearRegression",
    "KNeighborsClassifier", "KNeighborsRegressor"
}

TREE_MODELS = {
    "DecisionTreeClassifier", "DecisionTreeRegressor",
    "RandomForestClassifier", "RandomForestRegressor",
    "GradientBoostingClassifier", "GradientBoostingRegressor"
}

# Model mapping dictionary
MODEL_MAP = {
    "LogisticRegression": LogisticRegression,
    "LinearRegression": LinearRegression,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "DecisionTreeRegressor": DecisionTreeRegressor,
    "RandomForestClassifier": RandomForestClassifier,
    "RandomForestRegressor": RandomForestRegressor,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "KNeighborsClassifier": KNeighborsClassifier,
    "KNeighborsRegressor": KNeighborsRegressor,
}


def _prepare_data(df: pd.DataFrame, target_column: str, task_type: str = "classification") -> Tuple[pd.DataFrame, np.ndarray, List[str], List[str]]:
    """
    Prepare features and target with basic cleaning only.
    Returns clean DataFrame with column type information for model-specific preprocessing.
    
    Args:
        df: DataFrame with features and target
        target_column: Name of target column
        task_type: "classification" or "regression"
        
    Returns:
        Tuple of (X_clean_df, y_encoded, numeric_cols, categorical_cols)
        - X_clean_df: Clean DataFrame with imputed values (no encoding or scaling)
        - y_encoded: Encoded target (LabelEncoded for classification, raw for regression)
        - numeric_cols: List of numeric column names
        - categorical_cols: List of categorical column names
        
    Raises:
        ValueError: If dataset has critical issues (too few samples, constant target, etc.)
    """
    # EDGE CASE 1: Remove rows with missing target values
    df_clean = df.dropna(subset=[target_column]).copy()
    
    if len(df_clean) == 0:
        raise ValueError(f"All rows have missing values in target column '{target_column}'")
    
    if len(df_clean) < len(df):
        print(f"Warning: Removed {len(df) - len(df_clean)} rows with missing target values")
    
    # EDGE CASE 2: Validate minimum samples
    if len(df_clean) < 3:
        raise ValueError(f"Dataset has only {len(df_clean)} samples after removing missing targets. Minimum 3 samples required for training.")
    
    X = df_clean.drop(columns=[target_column]).copy()
    y = df_clean[target_column].copy()
    
    # EDGE CASE 3: Validate target has sufficient unique values
    unique_targets = y.nunique()
    if task_type == "classification" and unique_targets < 2:
        raise ValueError(f"Target column '{target_column}' has only {unique_targets} unique value(s). Classification requires at least 2 classes.")
    
    # EDGE CASE 4: Drop features that are all NaN
    cols_before = len(X.columns)
    X = X.dropna(axis=1, how='all')
    if len(X.columns) < cols_before:
        print(f"Warning: Dropped {cols_before - len(X.columns)} feature(s) with all missing values")
    
    if len(X.columns) == 0:
        raise ValueError("No valid features remain after dropping all-NaN columns")
    
    # Detect column types BEFORE any encoding
    numeric_cols = []
    categorical_cols = []
    high_cardinality_to_drop = []
    
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            numeric_cols.append(col)
        else:
            unique_count = X[col].nunique()
            total_count = len(X[col].dropna())
            # High-cardinality string columns (e.g. Date, ID strings) cannot be
            # OneHotEncoded safely — they produce thousands of sparse columns and
            # crash distance-based models (KNN). Try to parse as datetime first;
            # if that works, convert to a numeric timestamp. Otherwise drop.
            if total_count > 0 and unique_count / total_count > 0.5 and unique_count > 20:
                try:
                    parsed = pd.to_datetime(X[col], infer_datetime_format=True, errors='raise')
                    X[col] = parsed.astype('int64') // 10**9  # Unix timestamp (seconds)
                    numeric_cols.append(col)
                    print(f"Info: Converted date-like column '{col}' to numeric timestamp.")
                except Exception:
                    high_cardinality_to_drop.append(col)
                    print(f"Warning: Dropped high-cardinality string column '{col}' ({unique_count} unique values) — too many unique values for OneHotEncoding.")
            else:
                categorical_cols.append(col)
    
    # Drop high-cardinality columns that couldn't be converted
    if high_cardinality_to_drop:
        X = X.drop(columns=high_cardinality_to_drop)
    
    if len(X.columns) == 0:
        raise ValueError("No valid features remain after dropping high-cardinality/date columns")
    
    # Handle missing values - basic imputation
    for col in X.columns:
        if col in numeric_cols:
            median_val = X[col].median()
            # Handle case where all values are NaN (median is NaN)
            if pd.isna(median_val):
                X[col] = X[col].fillna(0)
            else:
                X[col] = X[col].fillna(median_val)
        else:
            # Categorical columns - impute with mode
            mode_val = X[col].mode()[0] if len(X[col].mode()) > 0 else "missing"
            X[col] = X[col].fillna(mode_val)
    
    # Handle target encoding ONLY (not features)
    if task_type == "classification":
        if not pd.api.types.is_numeric_dtype(y):
            le_target = LabelEncoder()
            y_encoded = le_target.fit_transform(y.astype(str))
        else:
            y_encoded = y.values
    else:
        # Regression - keep target as-is
        y_encoded = y.values
    
    return X, y_encoded, numeric_cols, categorical_cols


def _preprocess_linear_distance_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocessing for linear and distance-based models.
    Applies OneHotEncoding for categorical features and StandardScaler for numeric features.
    
    These models require scaling because they rely on:
    - Dot products (linear models)
    - Distance calculations (KNN)
    - Gradient descent optimization
    
    Args:
        X_train: Training features (DataFrame)
        X_test: Testing features (DataFrame)
        numeric_cols: List of numeric column names
        categorical_cols: List of categorical column names
        
    Returns:
        Tuple of (X_train_processed, X_test_processed) as NumPy arrays
    """
    # Build preprocessing pipeline
    transformers = []
    
    if numeric_cols:
        # Numeric features: apply StandardScaler
        transformers.append(('num', StandardScaler(), numeric_cols))
    
    if categorical_cols:
        # Categorical features: apply OneHotEncoder
        transformers.append(('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols))
    
    if not transformers:
        # No features to transform (edge case)
        return X_train.values, X_test.values
    
    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    return X_train_processed, X_test_processed


def _preprocess_tree_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocessing for tree-based models.
    Applies OneHotEncoding for categorical features, NO scaling for numeric features.
    
    Tree models are scale-invariant because they split by thresholds.
    Scaling provides no benefit and adds computational cost.
    
    Args:
        X_train: Training features (DataFrame)
        X_test: Testing features (DataFrame)
        numeric_cols: List of numeric column names
        categorical_cols: List of categorical column names
        
    Returns:
        Tuple of (X_train_processed, X_test_processed) as NumPy arrays
    """
    # Build preprocessing pipeline
    transformers = []
    
    if numeric_cols:
        # Numeric features: pass through (no scaling)
        transformers.append(('num', 'passthrough', numeric_cols))
    
    if categorical_cols:
        # Categorical features: apply OneHotEncoder
        transformers.append(('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols))
    
    if not transformers:
        # No features to transform (edge case)
        return X_train.values, X_test.values
    
    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    return X_train_processed, X_test_processed


def train_and_evaluate_models_adaptive(
    dataset: pd.DataFrame,
    target_column: str,
    model_names: List[str],
    task_type: str = "classification",
    test_size: float = 0.2,
    enable_two_stage: str = "auto",
    light_sample_size: float = 0.25,
    full_sample_size: float = 0.7,
    top_k_models: int = 2
) -> Dict[str, Any]:
    """
    Adaptive training with smart auto-detection of optimal strategy.
    
    Auto-detects whether to use:
    - Direct training (1-stage): For small datasets or few models
    - 2-stage training: For large datasets with many models
    
    Args:
        dataset: DataFrame containing features and target
        target_column: Name of target column
        model_names: List of model names to train
        task_type: "classification" or "regression"
        test_size: Proportion of data to use for testing
        enable_two_stage: "auto", "true", or "false"
        light_sample_size: Stage 1 sample size (0-1)
        full_sample_size: Stage 2 sample size (0-1)
        top_k_models: Number of finalists for stage 2
        
    Returns:
        Dictionary with results and training strategy info
    """
    n_samples = len(dataset)
    n_features = len(dataset.columns) - 1
    n_models = len(model_names)
    
    # Auto-detection logic (simple rules)
    if enable_two_stage == "auto":
        use_two_stage = (
            n_models > 2 and  # Need at least 3 models
            n_samples >= 5000 and  # Dataset is large enough
            (n_samples * n_features) >= 200000  # Complexity threshold
        )
    else:
        use_two_stage = enable_two_stage == "true"
    
    # Build strategy info
    if use_two_stage:
        strategy_mode = "two_stage"
        strategy_reason = f"Large dataset ({n_samples} rows, {n_features} features) with {n_models} models - using 2-stage optimization"
    else:
        strategy_mode = "direct"
        strategy_reason = f"Small dataset ({n_samples} rows) or few models ({n_models}) - using direct training"
    
    # Execute appropriate strategy
    if use_two_stage:
        results = _train_two_stage(
            dataset, target_column, model_names, task_type,
            light_sample_size, full_sample_size, top_k_models
        )
        results["training_strategy"] = {
            "mode": strategy_mode,
            "reason": strategy_reason,
            "stage_1_models": n_models,
            "stage_2_models": results.get("finalists_count", top_k_models)
        }
    else:
        results = train_and_evaluate_models(
            dataset, target_column, model_names, task_type, test_size
        )
        results["training_strategy"] = {
            "mode": strategy_mode,
            "reason": strategy_reason
        }
    
    return results


def _train_two_stage(
    dataset: pd.DataFrame,
    target_column: str,
    model_names: List[str],
    task_type: str,
    light_sample_size: float,
    full_sample_size: float,
    top_k_models: int
) -> Dict[str, Any]:
    """
    Internal 2-stage training implementation with model-specific preprocessing.
    
    Stage 1: Quick training on small sample to rank models
    Stage 2: Full training on top performers
    """
    # Prepare full data (returns DataFrame + column types)
    X_clean, y, numeric_cols, categorical_cols = _prepare_data(dataset, target_column, task_type)
    
    # Stage 1: Light training with small sample
    sample_size_stage1 = int(len(X_clean) * light_sample_size)
    X_stage1_df, _, y_stage1, _ = train_test_split(
        X_clean, y, train_size=sample_size_stage1, random_state=42,
        stratify=y if task_type == "classification" else None
    )
    
    # Quick train-test split for stage 1
    X_train_s1_df, X_test_s1_df, y_train_s1, y_test_s1 = train_test_split(
        X_stage1_df, y_stage1, test_size=0.2, random_state=42,
        stratify=y_stage1 if task_type == "classification" else None
    )
    
    stage1_results = []
    for model_name in model_names:
        if model_name not in MODEL_MAP:
            continue
        try:
            # Apply model-specific preprocessing
            if model_name in LINEAR_DISTANCE_MODELS:
                X_train_processed, X_test_processed = _preprocess_linear_distance_models(
                    X_train_s1_df, X_test_s1_df, numeric_cols, categorical_cols
                )
            elif model_name in TREE_MODELS:
                X_train_processed, X_test_processed = _preprocess_tree_models(
                    X_train_s1_df, X_test_s1_df, numeric_cols, categorical_cols
                )
            else:
                X_train_processed, X_test_processed = _preprocess_tree_models(
                    X_train_s1_df, X_test_s1_df, numeric_cols, categorical_cols
                )
            
            model_class = MODEL_MAP[model_name]
            model = model_class()
            model.fit(X_train_processed, y_train_s1)
            y_pred = model.predict(X_test_processed)
            
            if task_type == "classification":
                score = accuracy_score(y_test_s1, y_pred)
            else:
                score = r2_score(y_test_s1, y_pred)
            
            stage1_results.append({"model": model_name, "score": score})
        except Exception:
            continue
    
    # Select top-K models
    stage1_results.sort(key=lambda x: x["score"], reverse=True)
    top_models = [r["model"] for r in stage1_results[:top_k_models]]
    
    # Stage 2: Full training on finalists
    sample_size_stage2 = int(len(X_clean) * full_sample_size)
    X_stage2_df, _, y_stage2, _ = train_test_split(
        X_clean, y, train_size=sample_size_stage2, random_state=42,
        stratify=y if task_type == "classification" else None
    )
    
    X_train_s2_df, X_test_s2_df, y_train_s2, y_test_s2 = train_test_split(
        X_stage2_df, y_stage2, test_size=0.2, random_state=42,
        stratify=y_stage2 if task_type == "classification" else None
    )
    
    final_results = []
    for model_name in top_models:
        try:
            # Apply model-specific preprocessing
            if model_name in LINEAR_DISTANCE_MODELS:
                X_train_processed, X_test_processed = _preprocess_linear_distance_models(
                    X_train_s2_df, X_test_s2_df, numeric_cols, categorical_cols
                )
            elif model_name in TREE_MODELS:
                X_train_processed, X_test_processed = _preprocess_tree_models(
                    X_train_s2_df, X_test_s2_df, numeric_cols, categorical_cols
                )
            else:
                X_train_processed, X_test_processed = _preprocess_tree_models(
                    X_train_s2_df, X_test_s2_df, numeric_cols, categorical_cols
                )
            
            model_class = MODEL_MAP[model_name]
            model = model_class()
            model.fit(X_train_processed, y_train_s2)
            y_pred = model.predict(X_test_processed)
            
            if task_type == "classification":
                accuracy = accuracy_score(y_test_s2, y_pred)
                f1 = f1_score(y_test_s2, y_pred, average='weighted')
                metrics = {"accuracy": round(accuracy, 4), "f1_score": round(f1, 4)}
                primary_value = accuracy
            else:
                rmse = np.sqrt(mean_squared_error(y_test_s2, y_pred))
                r2 = r2_score(y_test_s2, y_pred)
                metrics = {"rmse": round(rmse, 4), "r2_score": round(r2, 4)}
                primary_value = r2
            
            final_results.append({
                "model": model_name,
                "metrics": metrics,
                "primary_value": primary_value
            })
        except Exception:
            continue
    
    # Select best
    best_result = max(final_results, key=lambda x: x["primary_value"])
    
    if task_type == "classification":
        justification = f"Selected {best_result['model']} (accuracy: {best_result['metrics']['accuracy']:.4f}, F1: {best_result['metrics']['f1_score']:.4f})"
    else:
        justification = f"Selected {best_result['model']} (R²: {best_result['metrics']['r2_score']:.4f}, RMSE: {best_result['metrics']['rmse']:.4f})"
    
    return {
        "task_type": task_type,
        "models_evaluated": final_results,
        "best_model": {
            "model": best_result["model"],
            "metrics": best_result["metrics"],
            "justification": justification
        },
        "train_samples": len(X_train_s2_df),
        "test_samples": len(X_test_s2_df),
        "finalists_count": len(top_models)
    }



def train_and_evaluate_models(
    dataset: pd.DataFrame,
    target_column: str,
    model_names: List[str],
    task_type: str = "classification",
    test_size: float = 0.2
) -> Dict[str, Any]:
    """
    Train and evaluate multiple models with model-specific preprocessing.
    
    Args:
        dataset: DataFrame containing features and target
        target_column: Name of target column
        model_names: List of model names to train
        task_type: "classification" or "regression"
        test_size: Proportion of data to use for testing
        
    Returns:
        Dictionary containing evaluation results and best model
    """
    # Prepare data (returns DataFrame + column types)
    X_clean, y, numeric_cols, categorical_cols = _prepare_data(dataset, target_column, task_type)
    
    # EDGE CASE: Adjust test_size for very small datasets
    # For stratified split, we need at least 2 samples per class in both train and test
    if task_type == "classification":
        # Check minimum class count
        unique, counts = np.unique(y, return_counts=True)
        min_class_count = counts.min()
        
        # If smallest class has < 5 samples, adjust test_size or disable stratification
        if min_class_count < 5:
            # Use simpler split without stratification for very small datasets
            print(f"Warning: Smallest class has only {min_class_count} samples. Using simple train/test split without stratification.")
            X_train_df, X_test_df, y_train, y_test = train_test_split(
                X_clean, y, test_size=test_size, random_state=42, stratify=None
            )
        else:
            X_train_df, X_test_df, y_train, y_test = train_test_split(
                X_clean, y, test_size=test_size, random_state=42, stratify=y
            )
    else:
        # Regression: no stratification needed
        X_train_df, X_test_df, y_train, y_test = train_test_split(
            X_clean, y, test_size=test_size, random_state=42
        )
    
    results = []
    
    # Train and evaluate each model with appropriate preprocessing
    for model_name in model_names:
        if model_name not in MODEL_MAP:
            continue
        
        try:
            # Determine preprocessing strategy based on model family
            if model_name in LINEAR_DISTANCE_MODELS:
                # Strategy A: Linear/Distance models need scaling
                X_train_processed, X_test_processed = _preprocess_linear_distance_models(
                    X_train_df, X_test_df, numeric_cols, categorical_cols
                )
            elif model_name in TREE_MODELS:
                # Strategy B: Tree models don't need scaling
                X_train_processed, X_test_processed = _preprocess_tree_models(
                    X_train_df, X_test_df, numeric_cols, categorical_cols
                )
            else:
                # Fallback: unknown model, use tree strategy (safer default)
                X_train_processed, X_test_processed = _preprocess_tree_models(
                    X_train_df, X_test_df, numeric_cols, categorical_cols
                )
            
            # Instantiate model with default parameters
            model_class = MODEL_MAP[model_name]
            model = model_class()
            
            # Train model
            model.fit(X_train_processed, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_processed)
            
            # Calculate metrics
            if task_type == "classification":
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                metrics = {
                    "accuracy": round(accuracy, 4),
                    "f1_score": round(f1, 4)
                }
                primary_metric = "accuracy"
                primary_value = accuracy
            else:  # regression
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                metrics = {
                    "rmse": round(rmse, 4),
                    "r2_score": round(r2, 4)
                }
                primary_metric = "r2_score"
                primary_value = r2
            
            results.append({
                "model": model_name,
                "metrics": metrics,
                "primary_metric": primary_metric,
                "primary_value": primary_value
            })
            
        except Exception as e:
            # Skip models that fail to train
            print(f"Warning: Model {model_name} failed to train: {str(e)}")
            continue
    
    if not results:
        raise ValueError("No models were successfully trained")
    
    # Select best model based on primary metric
    if task_type == "classification":
        # Higher is better for accuracy
        best_result = max(results, key=lambda x: x["primary_value"])
    else:
        # Higher is better for R²
        best_result = max(results, key=lambda x: x["primary_value"])
    
    # Generate justification
    if task_type == "classification":
        justification = f"Selected {best_result['model']} as the best model with accuracy of {best_result['metrics']['accuracy']:.4f} and F1-score of {best_result['metrics']['f1_score']:.4f}."
    else:
        justification = f"Selected {best_result['model']} as the best model with R² score of {best_result['metrics']['r2_score']:.4f} and RMSE of {best_result['metrics']['rmse']:.4f}."
    
    return {
        "task_type": task_type,
        "models_evaluated": results,
        "best_model": {
            "model": best_result["model"],
            "metrics": best_result["metrics"],
            "justification": justification
        },
        "train_samples": len(X_train_df),
        "test_samples": len(X_test_df)
    }



def train_model(model_config, dataset, hyperparameters: dict = None):
    """
    Train a model with the given configuration and dataset.
    
    Args:
        model_config: Model configuration object
        dataset: Dataset object for training
        hyperparameters: Optional dictionary of hyperparameters
        
    Returns:
        Trained model object
    """
    # TODO: Implement model training logic
    pass


def evaluate_model(model, dataset):
    """
    Evaluate a trained model on a dataset.
    
    Args:
        model: Trained model object
        dataset: Dataset object for evaluation
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # TODO: Implement model evaluation logic
    pass
