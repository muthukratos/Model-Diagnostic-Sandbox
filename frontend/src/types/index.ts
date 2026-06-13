export interface DatasetProfile {
    total_samples: number;
    total_features: number;
    numerical_features: number;
    categorical_features: number;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    missing_value_percentage?: Record<string, number>;
    target_class_count?: Record<string, number>;
    target_range?: { min: number; max: number };
    target_column?: {
        selected: string;
        was_auto_detected: boolean;
        justification: string;
    };
}

export interface ModelSelection {
    model: string;
    type: string;
    reason: string;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    params?: Record<string, any>;
}

export interface ModelFilteringResult {
    task_type: string;
    selected_models: ModelSelection[];
    count: number;
}

export interface ModelMetrics {
    accuracy?: number;
    f1_score?: number;
    precision?: number;
    recall?: number;
    rmse?: number;
    r2_score?: number;
    mae?: number;
    [key: string]: number | undefined;
}

export interface TrainedModel {
    model: string;
    metrics: ModelMetrics;
    training_time?: number;
    // Optional human-readable justification or explanation provided by the backend
    justification?: string;
}

export interface TrainingEvaluationResult {
    models_evaluated: TrainedModel[];
    best_model: TrainedModel;
    train_samples: number;
    test_samples: number;
    training_strategy?: {
        mode: string;
        reason: string;
    };
}

export interface FeatureImportance {
    [feature: string]: number;
}

export interface ExplanationResult {
    global_explanation?: {
        method: string;
        feature_importance: FeatureImportance;
    };
    local_explanation?: {
        method: string;
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        prediction: any;
        contributions: Record<string, number>;
    };
    error?: string;
    note?: string;
}

export interface ScanResult {
    filename: string;
    columns: string[];
    count: number;
    recommendations: {
        target_column: string;
        task_type: string;
        target_justification: string;
        task_justification: string;
    };
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    preview: Record<string, any>[];
}

export interface AnalysisResult {
    pipeline_status: string;
    filename?: string;
    task_type: {
        detected: string;
        was_auto_detected: boolean;
        justification: string;
    };
    target_column: {
        selected: string;
        was_auto_detected: boolean;
        justification: string;
    };
    profiling: DatasetProfile;
    model_filtering: ModelFilteringResult;
    training_evaluation: TrainingEvaluationResult;
    explanations?: ExplanationResult;
}
