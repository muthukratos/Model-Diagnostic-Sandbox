# File Upload Usage Guide

## Overview

The `/upload-and-analyze` endpoint now allows you to upload CSV or Excel files directly and get complete ML analysis results.

## Supported File Formats

- **CSV** (`.csv`)
- **Excel** (`.xlsx`, `.xls`)
- Maximum file size: **50 MB**

## Using the File Upload Endpoint

### 1. Start the Server

```bash
uvicorn main:app --reload
```

### 2. Upload File via API

**Endpoint:** `POST /upload-and-analyze`

**Parameters (all as form data):**
- `file` (required): CSV or Excel file to analyze
- `target_column` (optional): Target column name (auto-detected if not provided)
- `task_type` (optional): "classification" or "regression"(auto-detected if not provided)
- `test_size` (default: 0.2): Train/test split ratio
- `enable_two_stage` (default: "auto"): "auto", "true", or "false"
- `light_sample_size` (default: 0.25): Stage 1 sample percentage
- `full_sample_size` (default: 0.7): Stage 2 sample percentage
- `top_k_models` (default: 2): Number of finalists

### 3. Using Python (requests)

```python
import requests

url = "http://localhost:8000/upload-and-analyze"

with open("your_dataset.csv", "rb") as f:
    files = {"file": ("your_dataset.csv", f, "text/csv")}
    data = {
        "task_type": "classification",
        "target_column": "target",  # Optional
        "enable_two_stage": "auto"
    }
    response = requests.post(url, files=files, data=data)

result = response.json()
print(result)
```

### 4. Using cURL

```bash
curl -X POST "http://localhost:8000/upload-and-analyze" \
  -F "file=@sample_titanic.csv" \
  -F "task_type=classification" \
  -F "target_column=Survived" \
  -F "enable_two_stage=auto"
```

### 5. Using Postman/Thunder Client

1. Set method to **POST**
2. URL: `http://localhost:8000/upload-and-analyze`
3. Go to **Body** → **form-data**
4. Add key `file` (type: File) and select your CSV/Excel
5. Add other parameters as text:
   - `task_type`: classification
   - `target_column`: (leave empty for auto-detection)
   - `enable_two_stage`: auto

## Response Format

```json
{
  "pipeline_status": "completed",
  "filename": "sample_titanic.csv",
  "task_type": "classification",
  "target_column": {
    "selected": "Survived",
    "was_auto_detected": false,
    "justification": "Using provided target column 'Survived'."
  },
  "profiling": {
    "total_samples": 30,
    "total_features": 10,
    "numerical_features": 6,
    "categorical_features": 3
  },
  "model_filtering": {
    "selected_models": [...],
    "count": 5
  },
  "training_evaluation": {
    "training_strategy": {
      "mode": "direct",
      "reason": "Small dataset (30 rows) or few models (5) - using direct training"
    },
    "models_evaluated": [...],
    "best_model": {
      "model": "LogisticRegression",
      "metrics": {
        "accuracy": 0.8333,
        "f1_score": 0.8000
      }
    }
  },
  "explanations": {...}
}
```

## Test It Out

Run the test script:

```bash
python test_file_upload.py
```

Make sure the server is running first!

## Available Endpoints Summary

| Endpoint | Input Type | Purpose |
|----------|-----------|---------|
| `/upload-and-analyze` | **File Upload** | Upload CSV/Excel, run full pipeline |
| `/analysis/start` | JSON | Run pipeline with JSON data |
| `/profile` | JSON | Profile dataset only |
| `/filter-models` | JSON | Filter models based on profile |
| `/train-evaluate` | JSON | Train and evaluate models |
| `/explain` | JSON | Generate model explanations |

## Tips

- **Auto-detection**: Leave `target_column` empty to automatically detect the target
- **Smart training**: Use `enable_two_stage="auto"` for automatic optimization
- **Large files**: Files over 50MB will be rejected
- **Mixed data**: Works with both numerical and categorical features
