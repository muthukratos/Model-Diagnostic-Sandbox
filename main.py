"""
FastAPI main application entry point for ML backend.
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import numpy as np
import io

from profiling import profile_data
from model_selector import filter_models
from trainer import train_and_evaluate_models, train_and_evaluate_models_adaptive
from explainability import explain_model

app = FastAPI(
    title="ML Backend API",
    description="Minimal FastAPI backend for ML operations",
    version="1.0.0"
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


def sanitize_for_json(obj):
    """
    Recursively replace NaN, Inf, and numpy scalar types with JSON-safe equivalents.
    Starlette's JSONResponse uses Python's stdlib json.dumps which raises ValueError
    on float('nan') and float('inf') — this utility prevents those crashes.
    """
    import math
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    # Convert numpy scalar types to native Python types
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    return obj


def auto_detect_task_type(df: pd.DataFrame, target_column: str) -> Tuple[str, str]:
    """
    Auto-detect task type based on target column characteristics.
    
    Args:
        df: DataFrame to analyze
        target_column: Name of target column
        
    Returns:
        Tuple of (task_type, justification)
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")
    
    target = df[target_column].dropna()
    
    if len(target) == 0:
        raise ValueError(f"Target column '{target_column}' has no valid values")
    
    # Check if numeric
    is_numeric = pd.api.types.is_numeric_dtype(target)
    unique_count = target.nunique()
    total_count = len(target)
    
    # Decision logic (simple heuristics)
    if not is_numeric:
        # Non-numeric = classification
        task_type = "classification"
        justification = f"Auto-detected CLASSIFICATION: target '{target_column}' is categorical ({unique_count} classes)"
    elif unique_count <= 20 or unique_count / total_count < 0.05:
        # Few unique values = classification
        task_type = "classification"
        justification = f"Auto-detected CLASSIFICATION: target '{target_column}' has {unique_count} unique values (discrete categories)"
    else:
        # Many unique continuous values = regression
        task_type = "regression"
        justification = f"Auto-detected REGRESSION: target '{target_column}' is continuous numeric ({unique_count} unique values, {unique_count/total_count:.1%} unique ratio)"
    
    return task_type, justification


def auto_detect_target_column(df: pd.DataFrame, task_type: str = "classification") -> Tuple[str, str]:
    """
    Auto-detect the most likely target column using heuristic rules.
    
    Args:
        df: DataFrame to analyze
        task_type: "classification" or "regression"
        
    Returns:
        Tuple of (target_column_name, justification)
    """
    if len(df.columns) == 0:
        raise ValueError("Dataset has no columns")
    
    # Exclude identifier-like columns
    identifier_keywords = ['id', 'index', 'uuid', 'key', 'pk']
    candidate_columns = [
        col for col in df.columns
        if not any(keyword in col.lower() for keyword in identifier_keywords)
    ]
    
    if not candidate_columns:
        # Fallback: use all columns if all are identifier-like
        candidate_columns = list(df.columns)
    
    if task_type == "classification":
        # Prefer columns with low cardinality (few unique values)
        # Good classification targets have 2-50 unique values typically
        best_column = None
        best_score = float('inf')
        
        for col in candidate_columns:
            unique_count = df[col].nunique()
            total_count = len(df[col].dropna())
            
            if total_count == 0:
                continue
            
            # Prefer columns with moderate cardinality (not too many unique values)
            # Score: lower is better (fewer unique values = better for classification)
            # But avoid columns with only 1 unique value
            if unique_count > 1:
                score = unique_count
                # Prefer columns with 2-50 unique values, but also consider others as fallback
                if 2 <= unique_count <= 50:
                    if score < best_score:
                        best_score = score
                        best_column = col
                elif best_column is None:
                    # If no ideal candidate found yet, use this as temporary best
                    best_column = col
                    best_score = score
        
        if best_column:
            unique_count = df[best_column].nunique()
            if 2 <= unique_count <= 50:
                justification = f"Auto-selected '{best_column}' for classification: {unique_count} unique values (low cardinality suitable for classification)."
            else:
                justification = f"Auto-selected '{best_column}' for classification: {unique_count} unique values (best available candidate)."
        else:
            # Fallback: use last column
            best_column = candidate_columns[-1]
            justification = f"Auto-selected '{best_column}' as fallback (last column in dataset)."
    
    else:  # regression
        # Prefer continuous numeric columns with higher variance
        best_column = None
        best_score = -1
        justification_parts = []
        
        for col in candidate_columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                variance = df[col].var()
                if pd.notna(variance) and variance > best_score:
                    best_score = variance
                    best_column = col
        
        if best_column:
            variance_val = df[best_column].var()
            justification = f"Auto-selected '{best_column}' for regression: continuous numeric column with variance {variance_val:.4f} (higher variance indicates good regression target)."
        else:
            # Fallback: use last column
            best_column = candidate_columns[-1]
            justification = f"Auto-selected '{best_column}' as fallback (last column in dataset)."
    
    return best_column, justification


def determine_target_column(
    df: pd.DataFrame,
    target_column: Optional[str],
    task_type: str = "classification"
) -> Tuple[str, str, bool]:
    """
    Determine target column: validate if provided, or auto-detect if not.
    
    Args:
        df: DataFrame to analyze
        target_column: Optional target column name
        task_type: "classification" or "regression"
        
    Returns:
        Tuple of (selected_target_column, justification, was_auto_detected)
    """
    if target_column:
        # Validate provided target column
        if target_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Target column '{target_column}' not found in dataset. Available columns: {list(df.columns)}"
            )
        justification = f"Using provided target column '{target_column}'."
        return target_column, justification, False
    else:
        # Auto-detect target column
        detected_column, justification = auto_detect_target_column(df, task_type)
        return detected_column, justification, True


def parse_uploaded_file(file: UploadFile, max_size_mb: int = 50) -> pd.DataFrame:
    """
    Parse uploaded CSV or Excel file into pandas DataFrame.
    
    Args:
        file: Uploaded file object
        max_size_mb: Maximum file size in MB (default: 50MB)
        
    Returns:
        Parsed pandas DataFrame
    """
    # Validate file size
    file_content = file.file.read()
    file_size_mb = len(file_content) / (1024 * 1024)
    
    if file_size_mb > max_size_mb:
        raise HTTPException(
            status_code=400,
            detail=f"File size ({file_size_mb:.2f} MB) exceeds maximum allowed size ({max_size_mb} MB)"
        )
    
    # Validate file type
    filename = file.filename.lower()
    is_csv = filename.endswith('.csv')
    is_excel = filename.endswith(('.xlsx', '.xls'))
    
    if not (is_csv or is_excel):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported formats: CSV (.csv), Excel (.xlsx, .xls). Received: {file.filename}"
        )
    
    # Reset file pointer
    file.file.seek(0)
    
    try:
        # Parse file based on type, with fallback for double-extension files (e.g. .csv.xlsx)
        if is_csv:
            df = pd.read_csv(io.BytesIO(file_content))
        else:  # Excel — try Excel first, fall back to CSV if content is actually CSV
            try:
                df = pd.read_excel(io.BytesIO(file_content))
            except Exception:
                # Fallback: file may have .xlsx extension but actually contain CSV content
                # (common with double-extension names like "data.csv.xlsx")
                try:
                    df = pd.read_csv(io.BytesIO(file_content))
                except Exception as csv_err:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Could not parse '{file.filename}' as Excel or CSV. "
                            f"The file extension is '.xlsx' but the content does not match either format. "
                            f"Details: {str(csv_err)}"
                        )
                    )
        
        if df.empty:
            raise HTTPException(status_code=400, detail="Uploaded file is empty or could not be parsed")
        
        return df
    
    except HTTPException:
        raise
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"Error parsing file: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")


class ProfileRequest(BaseModel):
    """Request model for dataset profiling."""
    data: List[Dict[str, Any]]
    target_column: Optional[str] = None


class FilterModelsRequest(BaseModel):
    """Request model for model filtering."""
    profile: Dict[str, Any]
    task_type: str = "classification"


class TrainModelsRequest(BaseModel):
    """Request model for training and evaluation."""
    data: List[Dict[str, Any]]
    target_column: Optional[str] = None
    model_names: List[str]
    task_type: str = "classification"
    test_size: float = 0.2
    enable_two_stage: str = "auto"
    light_sample_size: float = 0.25
    full_sample_size: float = 0.7
    top_k_models: int = 2


class ExplainModelRequest(BaseModel):
    """Request model for model explanation."""
    data: List[Dict[str, Any]]
    target_column: Optional[str] = None
    model_name: str
    task_type: str = "classification"
    instance_to_explain: Optional[Dict[str, Any]] = None


class AnalysisStartRequest(BaseModel):
    """Request model for complete ML pipeline orchestration."""
    data: List[Dict[str, Any]]
    target_column: Optional[str] = None
    task_type: str = "classification"
    test_size: float = 0.2
    instance_to_explain: Optional[Dict[str, Any]] = None
    enable_two_stage: str = "auto"
    light_sample_size: float = 0.25
    full_sample_size: float = 0.7
    top_k_models: int = 2


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "ML Backend API is running"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/profile")
async def profile_dataset(request: ProfileRequest):
    """
    Profile a dataset and extract basic meta-features.
    
    Expected fields:
    - total_samples
    - total_features
    - numerical_features
    - categorical_features
    - missing_value_percentage (per column)
    - target_class_count (if classification target provided)
    - target_range (if regression target provided)
    """
    try:
        # Convert JSON data to pandas DataFrame
        df = pd.DataFrame(request.data)
        
        # Determine target column if task type can be inferred (for profiling, target is optional)
        # For profiling endpoint, we can work without target, but if provided, use it
        selected_target = None
        target_info = None
        
        if request.target_column:
            if request.target_column not in df.columns:
                raise HTTPException(
                    status_code=400,
                    detail=f"Target column '{request.target_column}' not found in dataset. Available columns: {list(df.columns)}"
                )
            selected_target = request.target_column
            target_info = {
                "selected": selected_target,
                "was_auto_detected": False,
                "justification": f"Using provided target column '{selected_target}'."
            }
        
        # Generate profile
        profile = profile_data(df, target_column=selected_target)
        
        # Add target column info if available
        if target_info:
            profile["target_column"] = target_info
        
        return profile
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error profiling dataset: {str(e)}")


@app.post("/filter-models")
async def filter_models_endpoint(request: FilterModelsRequest):
    """
    Filter lightweight ML models based on dataset meta-features.
    
    Uses rule-based filtering considering:
    - Dataset size
    - Number of features (dimensionality)
    - Feature types (numerical vs categorical)
    - Class imbalance (for classification)
    - Missing value ratio
    
    Returns list of selected models with justifications.
    """
    try:
        # Validate task_type
        if request.task_type not in ["classification", "regression"]:
            raise HTTPException(status_code=400, detail="task_type must be 'classification' or 'regression'")
        
        # Filter models based on profile
        selected_models = filter_models(request.profile, task_type=request.task_type)
        
        return {
            "task_type": request.task_type,
            "selected_models": selected_models,
            "count": len(selected_models)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error filtering models: {str(e)}")


@app.post("/train-evaluate")
async def train_evaluate_endpoint(request: TrainModelsRequest):
    """
    Train and evaluate multiple models on a dataset.
    
    Performs train-test split and evaluates models using:
    - Classification: accuracy and F1-score
    - Regression: RMSE and R²
    
    Returns evaluation results for all models and selects the best one.
    """
    try:
        # Validate task_type
        if request.task_type not in ["classification", "regression"]:
            raise HTTPException(status_code=400, detail="task_type must be 'classification' or 'regression'")
        
        # Validate test_size
        if not 0 < request.test_size < 1:
            raise HTTPException(status_code=400, detail="test_size must be between 0 and 1")
        
        # Convert JSON data to pandas DataFrame
        df = pd.DataFrame(request.data)
        
        # Determine target column
        selected_target, target_justification, was_auto_detected = determine_target_column(
            df, request.target_column, request.task_type
        )
        
        # Train and evaluate models (with adaptive 2-stage)
        results = train_and_evaluate_models_adaptive(
            dataset=df,
            target_column=selected_target,
            model_names=request.model_names,
            task_type=request.task_type,
            test_size=request.test_size,
            enable_two_stage=request.enable_two_stage,
            light_sample_size=request.light_sample_size,
            full_sample_size=request.full_sample_size,
            top_k_models=request.top_k_models
        )
        
        # Add target column info to response
        results["target_column"] = {
            "selected": selected_target,
            "was_auto_detected": was_auto_detected,
            "justification": target_justification
        }
        
        return results
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error training/evaluating models: {str(e)}")


@app.post("/explain")
async def explain_model_endpoint(request: ExplainModelRequest):
    """
    Generate explanations for a model using SHAP (global) and LIME (local).
    
    Supports linear and tree-based models only.
    Returns JSON-serializable explanation results.
    """
    try:
        # Validate task_type
        if request.task_type not in ["classification", "regression"]:
            raise HTTPException(status_code=400, detail="task_type must be 'classification' or 'regression'")
        
        # Convert JSON data to pandas DataFrame
        df = pd.DataFrame(request.data)
        
        # Determine target column
        selected_target, target_justification, was_auto_detected = determine_target_column(
            df, request.target_column, request.task_type
        )
        
        # Generate explanations
        explanations = explain_model(
            dataset=df,
            target_column=selected_target,
            model_name=request.model_name,
            task_type=request.task_type,
            instance_to_explain=request.instance_to_explain
        )
        
        # Add target column info to response
        explanations["target_column"] = {
            "selected": selected_target,
            "was_auto_detected": was_auto_detected,
            "justification": target_justification
        }
        
        return explanations
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error generating explanations: {str(e)}")


@app.post("/scan")
async def scan_dataset_endpoint(file: UploadFile = File(...)):
    """
    Scan a dataset to auto-detect target column and task type without training.
    """
    try:
        # Debug logging: print incoming request metadata for troubleshooting uploads
        try:
            # 'request' is not in the signature originally; attempt to get from context if available
            pass
        except Exception:
            pass
        # Log file info
        try:
            print(f"[scan] Received upload: filename={file.filename}, content_type={file.content_type}")
        except Exception:
            print("[scan] Received upload: could not read file metadata")

        df = parse_uploaded_file(file)
        
        # Profile data
        profile = profile_data(df)
        
        # Initial guess assuming classification
        suggested_target, target_reason = auto_detect_target_column(df, "classification")
        suggested_task, task_reason = auto_detect_task_type(df, suggested_target)
        
        # If regression detected, refine target guess
        if suggested_task == "regression":
             suggested_target_reg, target_reason_reg = auto_detect_target_column(df, "regression")
             # If the regression logic picks a different column that looks better, swap
             if suggested_target_reg != suggested_target:
                 suggested_target = suggested_target_reg
                 target_reason = target_reason_reg
                 # Re-verify task type just in case
                 suggested_task, task_reason = auto_detect_task_type(df, suggested_target)

        return sanitize_for_json({
            "filename": file.filename,
            "columns": list(df.columns),
            "count": len(df),
            "recommendations": {
                "target_column": suggested_target,
                "task_type": suggested_task,
                "target_justification": target_reason,
                "task_justification": task_reason
            },
            "preview": df.head(5).to_dict(orient='records')
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error scanning file: {str(e)}")


@app.post("/upload-and-analyze")
async def upload_and_analyze_endpoint(
    file: UploadFile = File(...),
    target_column: Optional[str] = Form(None),
    task_type: str = Form("auto"),
    test_size: float = Form(0.2),
    enable_two_stage: str = Form("auto"),
    light_sample_size: float = Form(0.25),
    full_sample_size: float = Form(0.7),
    top_k_models: int = Form(2)
):
    """
    Upload CSV/Excel file and run complete ML pipeline.
    
    Accepts file upload and runs:
    1. Parse and load dataset
    2. Auto-detect target column and/or task type if needed
    3. Profile dataset
    4. Filter models
    5. Train and evaluate (with adaptive 2-stage)
    6. Generate explanations
    
    Form parameters:
    - file: CSV or Excel file
    - target_column: Optional target column (auto-detected if not provided)
    - task_type: "auto", "classification", or "regression" (auto-detected if "auto")
    - test_size: Train/test split ratio
    - enable_two_stage: "auto", "true", or "false"
    - light_sample_size: Stage 1 sample size (0-1)
    - full_sample_size: Stage 2 sample size (0-1)
    - top_k_models: Number of finalists
    """
    try:
        # Validate test_size
        if not 0 < test_size < 1:
            raise HTTPException(status_code=400, detail="test_size must be between 0 and 1")
        
        # Parse uploaded file
        df = parse_uploaded_file(file)
        
        # Step 1: Determine target column (may need task_type for heuristics)
        # If task_type is auto, we'll refine it after knowing the target
        temp_task_type = "classification" if task_type == "auto" else task_type
        
        # Validate task_type if not auto
        if task_type not in ["auto", "classification", "regression"]:
            raise HTTPException(status_code=400, detail="task_type must be 'auto', 'classification', or 'regression'")
        
        selected_target, target_justification, was_auto_detected_target = determine_target_column(
            df, target_column, temp_task_type
        )
        
        # Step 2: Auto-detect task type if needed
        if task_type == "auto":
            detected_task_type, task_type_justification = auto_detect_task_type(df, selected_target)
            was_auto_detected_task = True
        else:
            detected_task_type = task_type
            task_type_justification = f"Using provided task type '{task_type}'"
            was_auto_detected_task = False
        
        # Profile dataset
        profile = profile_data(df, target_column=selected_target)
        
        # Filter models using detected task type
        selected_models_list = filter_models(profile, task_type=detected_task_type)
        model_names = [model["model"] for model in selected_models_list]
        
        if not model_names:
            raise HTTPException(status_code=400, detail="No models were selected by the filtering stage")
        
        # Train and evaluate with adaptive 2-stage
        training_results = train_and_evaluate_models_adaptive(
            dataset=df,
            target_column=selected_target,
            model_names=model_names,
            task_type=detected_task_type,
            test_size=test_size,
            enable_two_stage=enable_two_stage,
            light_sample_size=light_sample_size,
            full_sample_size=full_sample_size,
            top_k_models=top_k_models
        )
        # Merge per-model justifications from the filtering stage into training results
        try:
            justification_map = {m["model"]: m.get("justification") for m in selected_models_list}
            for item in training_results.get("models_evaluated", []):
                if "justification" not in item or not item.get("justification"):
                    item["justification"] = justification_map.get(item.get("model"))
            # Ensure best_model has a justification (prefer trainer-provided, fallback to filter justification)
            best = training_results.get("best_model", {})
            if best is not None and not best.get("justification"):
                best_model_name = best.get("model")
                if best_model_name:
                    best["justification"] = justification_map.get(best_model_name)
        except Exception:
            # Non-critical: if merging fails, continue without breaking the pipeline
            pass
        
        # Get best model for explanations
        best_model_name = training_results["best_model"]["model"]
        
        # Generate explanations
        explanations = None
        try:
            explanations = explain_model(
                dataset=df,
                target_column=selected_target,
                model_name=best_model_name,
                task_type=detected_task_type,
                instance_to_explain=None
            )
        except Exception:
            explanations = {
                "error": "Explanations not available for this model type"
            }
        
        # Return consolidated response
        return sanitize_for_json({
            "pipeline_status": "completed",
            "filename": file.filename,
            "task_type": {
                "detected": detected_task_type,
                "was_auto_detected": was_auto_detected_task,
                "justification": task_type_justification
            },
            "target_column": {
                "selected": selected_target,
                "was_auto_detected": was_auto_detected_target,
                "justification": target_justification
            },
            "profiling": profile,
            "model_filtering": {
                "selected_models": selected_models_list,
                "count": len(selected_models_list)
            },
            "training_evaluation": {
                "models_evaluated": training_results["models_evaluated"],
                "best_model": training_results["best_model"],
                "train_samples": training_results["train_samples"],
                "test_samples": training_results["test_samples"],
                "training_strategy": training_results.get("training_strategy", {})
            },
            "explanations": explanations
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in ML pipeline: {str(e)}")


@app.post("/analysis/start")
async def analysis_start_endpoint(request: AnalysisStartRequest):
    """
    Orchestrate the complete ML pipeline:
    1. Load and profile dataset
    2. Apply rule-based model filtering
    3. Train and evaluate shortlisted models
    4. Select best-performing model
    5. Generate SHAP and LIME explanations for best model
    
    Returns consolidated JSON response with all pipeline results.
    """
    try:
        # Validate task_type
        if request.task_type not in ["classification", "regression"]:
            raise HTTPException(status_code=400, detail="task_type must be 'classification' or 'regression'")
        
        # Validate test_size
        if not 0 < request.test_size < 1:
            raise HTTPException(status_code=400, detail="test_size must be between 0 and 1")
        
        # Step 1: Load dataset
        df = pd.DataFrame(request.data)
        
        # Step 1.5: Determine and lock target column
        selected_target, target_justification, was_auto_detected = determine_target_column(
            df, request.target_column, request.task_type
        )
        
        # Step 2: Perform dataset profiling
        profile = profile_data(df, target_column=selected_target)
        
        # Step 3: Apply rule-based model filtering
        selected_models_list = filter_models(profile, task_type=request.task_type)
        
        # Extract model names from filtered results
        model_names = [model["model"] for model in selected_models_list]
        
        if not model_names:
            raise HTTPException(status_code=400, detail="No models were selected by the filtering stage")
        
        # Step 4: Train and evaluate shortlisted models (with adaptive 2-stage)
        training_results = train_and_evaluate_models_adaptive(
            dataset=df,
            target_column=selected_target,
            model_names=model_names,
            task_type=request.task_type,
            test_size=request.test_size,
            enable_two_stage=request.enable_two_stage,
            light_sample_size=request.light_sample_size,
            full_sample_size=request.full_sample_size,
            top_k_models=request.top_k_models
        )

        # Merge per-model justifications from filter stage into training results
        try:
            justification_map = {m["model"]: m.get("justification") for m in selected_models_list}
            for item in training_results.get("models_evaluated", []):
                if "justification" not in item or not item.get("justification"):
                    item["justification"] = justification_map.get(item.get("model"))
            best = training_results.get("best_model", {})
            if best is not None and not best.get("justification"):
                best_name = best.get("model")
                if best_name:
                    best["justification"] = justification_map.get(best_name)
        except Exception:
            pass
        
        # Step 5: Get best model name
        best_model_name = training_results["best_model"]["model"]
        
        # Step 6: Generate explanations for best model
        # Only generate explanations if best model is linear or tree-based
        explanations = None
        try:
            explanations = explain_model(
                dataset=df,
                target_column=selected_target,
                model_name=best_model_name,
                task_type=request.task_type,
                instance_to_explain=request.instance_to_explain
            )
        except ValueError as e:
            # If model doesn't support explanations (e.g., KNeighbors), skip
            explanations = {
                "error": f"Explanations not available: {str(e)}",
                "note": "Only linear and tree-based models support SHAP/LIME explanations"
            }
        except Exception as e:
            explanations = {
                "error": f"Error generating explanations: {str(e)}"
            }
        
        # Step 7: Return consolidated response
        return sanitize_for_json({
            "pipeline_status": "completed",
            "task_type": request.task_type,
            "target_column": {
                "selected": selected_target,
                "was_auto_detected": was_auto_detected,
                "justification": target_justification
            },
            "profiling": profile,
            "model_filtering": {
                "selected_models": selected_models_list,
                "count": len(selected_models_list)
            },
            "training_evaluation": {
                "models_evaluated": training_results["models_evaluated"],
                "best_model": training_results["best_model"],
                "train_samples": training_results["train_samples"],
                "test_samples": training_results["test_samples"],
                "training_strategy": training_results.get("training_strategy", {})
            },
            "explanations": explanations
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in ML pipeline: {str(e)}")
