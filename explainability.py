"""
Model explainability module.
Provides SHAP (global) and LIME (local) explanations for trained models.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from trainer import MODEL_MAP, _prepare_data, LINEAR_DISTANCE_MODELS, TREE_MODELS


def _is_tree_based(model_name: str) -> bool:
    """Check if model is tree-based."""
    return model_name in TREE_MODELS


def _is_linear(model_name: str) -> bool:
    """Check if model is linear."""
    return model_name in {"LogisticRegression", "LinearRegression"}


def _get_feature_names_after_encoding(
    preprocessor: ColumnTransformer,
    numeric_cols: List[str],
    categorical_cols: List[str]
) -> List[str]:
    """
    Derive feature names after ColumnTransformer encoding.
    Numeric columns keep their names; OneHot columns are expanded.
    """
    feature_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == 'num':
            feature_names.extend(cols if isinstance(cols, list) else list(cols))
        elif name == 'cat':
            if hasattr(transformer, 'get_feature_names_out'):
                feature_names.extend(transformer.get_feature_names_out(cols).tolist())
            else:
                feature_names.extend(cols)
    return feature_names


def get_global_feature_importance(
    model,
    X_train: np.ndarray,
    feature_names: List[str],
    model_name: str,
    task_type: str = "classification"
) -> Dict[str, Any]:
    """
    Compute global feature importance using SHAP.

    Args:
        model: Trained model object (trained on preprocessed numpy array)
        X_train: Preprocessed training features (numpy array)
        feature_names: Feature names after encoding
        model_name: Name of the model
        task_type: "classification" or "regression"

    Returns:
        Dictionary containing feature importance scores
    """
    background_size = min(100, len(X_train))
    background_data = X_train[:background_size]

    try:
        if _is_tree_based(model_name):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(background_data)
        elif _is_linear(model_name):
            explainer = shap.LinearExplainer(model, background_data)
            shap_values = explainer.shap_values(background_data)
        else:
            raise ValueError(f"Model {model_name} not supported for SHAP explanation")

        # Normalise SHAP value format across model types
        if isinstance(shap_values, list):
            shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        else:
            shap_values = np.abs(shap_values)

        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(1, -1)

        feature_importance = np.mean(shap_values, axis=0)
        if feature_importance.ndim > 1:
            feature_importance = feature_importance.flatten()

        importance_dict = {
            feature_names[i]: float(feature_importance[i])
            for i in range(min(len(feature_names), len(feature_importance)))
        }

        sorted_importance = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )

        return {
            "method": "SHAP",
            "feature_importance": sorted_importance,
            "model_type": "tree_based" if _is_tree_based(model_name) else "linear"
        }

    except Exception as e:
        raise ValueError(f"Error computing SHAP values: {str(e)}")


def explain_single_prediction(
    model,
    instance: np.ndarray,
    X_train: np.ndarray,
    feature_names: List[str],
    model_name: str,
    task_type: str = "classification"
) -> Dict[str, Any]:
    """
    Explain a single prediction using LIME.

    Args:
        model: Trained model object
        instance: Single preprocessed instance (1-D or 2-D array)
        X_train: Preprocessed training features for LIME background
        feature_names: Feature names after encoding
        model_name: Name of the model
        task_type: "classification" or "regression"

    Returns:
        Dictionary containing LIME explanation
    """
    try:
        if instance.ndim == 1:
            instance = instance.reshape(1, -1)

        explainer = LimeTabularExplainer(
            X_train,
            feature_names=feature_names,
            mode=task_type,
            discretize_continuous=True
        )

        explanation = explainer.explain_instance(
            instance[0],
            model.predict_proba if task_type == "classification" else model.predict,
            num_features=min(len(feature_names), 10)
        )

        explanation_dict = sorted(
            [{"feature": item[0], "contribution": float(item[1])} for item in explanation.as_list()],
            key=lambda x: abs(x["contribution"]),
            reverse=True
        )

        if task_type == "classification":
            return {
                "method": "LIME",
                "prediction": int(model.predict(instance)[0]),
                "prediction_probability": model.predict_proba(instance)[0].tolist(),
                "explanation": explanation_dict
            }
        else:
            return {
                "method": "LIME",
                "prediction": float(model.predict(instance)[0]),
                "explanation": explanation_dict
            }

    except Exception as e:
        raise ValueError(f"Error computing LIME explanation: {str(e)}")


def explain_model(
    dataset: pd.DataFrame,
    target_column: str,
    model_name: str,
    task_type: str = "classification",
    instance_to_explain: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Generate global (SHAP) and optionally local (LIME) explanations for a model.

    Args:
        dataset: DataFrame containing features and target
        target_column: Name of target column
        model_name: Name of the model to train and explain
        task_type: "classification" or "regression"
        instance_to_explain: Optional dict of feature values for a single prediction explanation

    Returns:
        Dictionary containing global and (optionally) local explanations
    """
    # Validate model support
    if not (_is_tree_based(model_name) or _is_linear(model_name)):
        raise ValueError(
            f"Model '{model_name}' not supported. Only linear and tree-based models are supported."
        )

    if model_name not in MODEL_MAP:
        raise ValueError(f"Model '{model_name}' not found in model map.")

    # ── Phase 1: Basic cleaning ──────────────────────────────────────────────
    # _prepare_data returns: (X_clean_df, y_encoded, numeric_cols, categorical_cols)
    X_clean, y, numeric_cols, categorical_cols = _prepare_data(dataset, target_column, task_type)

    # ── Phase 2: Model-specific preprocessing ───────────────────────────────
    # Build preprocessor matching the same strategy used in trainer.py
    transformers = []
    if numeric_cols:
        if model_name in LINEAR_DISTANCE_MODELS:
            transformers.append(('num', StandardScaler(), numeric_cols))
        else:
            # Tree models: pass numeric features through unchanged
            transformers.append(('num', 'passthrough', numeric_cols))
    if categorical_cols:
        transformers.append((
            'cat',
            OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'),
            categorical_cols
        ))

    if not transformers:
        # Edge case: no columns to transform
        X_processed = X_clean.values
        feature_names = list(X_clean.columns)
        preprocessor = None
    else:
        preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
        X_processed = preprocessor.fit_transform(X_clean)
        feature_names = _get_feature_names_after_encoding(preprocessor, numeric_cols, categorical_cols)

    # ── Train model on preprocessed data ─────────────────────────────────────
    model = MODEL_MAP[model_name]()
    model.fit(X_processed, y)

    # ── Global explanation (SHAP) ─────────────────────────────────────────────
    global_explanation = get_global_feature_importance(
        model, X_processed, feature_names, model_name, task_type
    )

    result = {
        "model": model_name,
        "task_type": task_type,
        "global_explanation": global_explanation
    }

    # ── Local explanation (LIME) ──────────────────────────────────────────────
    if instance_to_explain:
        # Build instance row aligned with the cleaned training columns
        instance_row = {col: instance_to_explain.get(col, np.nan) for col in X_clean.columns}
        instance_df = pd.DataFrame([instance_row])

        # Impute missing values using training data statistics
        for col in instance_df.columns:
            if col in numeric_cols:
                instance_df[col] = pd.to_numeric(instance_df[col], errors='coerce')
                instance_df[col] = instance_df[col].fillna(X_clean[col].median())
            elif col in categorical_cols:
                mode_val = X_clean[col].mode()[0] if len(X_clean[col].mode()) > 0 else "missing"
                instance_df[col] = instance_df[col].fillna(mode_val).astype(str)

        # Apply the SAME fitted preprocessor — guarantees identical transformation
        if preprocessor is None:
            instance_array = instance_df.values
        else:
            instance_array = preprocessor.transform(instance_df)

        local_explanation = explain_single_prediction(
            model, instance_array, X_processed, feature_names, model_name, task_type
        )
        result["local_explanation"] = local_explanation

    return result


# ── Legacy placeholder stubs (unused) ────────────────────────────────────────

def explain_prediction(model, input_data):
    """Legacy stub — not implemented."""
    pass


def explain_model_global(model, dataset):
    """Legacy stub — not implemented."""
    pass


def get_feature_importance(model, dataset):
    """Legacy stub — not implemented."""
    pass
