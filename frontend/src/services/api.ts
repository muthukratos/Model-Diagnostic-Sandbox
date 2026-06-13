import axios from 'axios';
import type { AnalysisResult, ScanResult } from '../types';

const API_URL = 'http://localhost:8000';

const api = axios.create({
    baseURL: API_URL,
    // Do not set default Content-Type here so FormData requests can let the browser set the
    // correct multipart/form-data boundary automatically.
});

export interface AnalyzeOptions {
    target_column?: string;
    task_type?: string;
    enable_two_stage?: string;
    test_size?: number;
}

export const scanFile = async (file: File): Promise<ScanResult> => {
    const formData = new FormData();
    formData.append('file', file);
    try {
        const response = await api.post<ScanResult>('/scan', formData);
        return response.data;
    } catch (err: unknown) {
        console.error('scanFile error', err);
        if (axios.isAxiosError(err)) {
            const detail = err.response?.data?.detail ?? err.response?.data ?? err.message;
            throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
        }
        throw new Error(err instanceof Error ? err.message : 'Network error while uploading file');
    }
};

export const uploadAndAnalyze = async (
    file: File,
    options: AnalyzeOptions = {}
): Promise<AnalysisResult> => {
    const formData = new FormData();
    formData.append('file', file);

    if (options.target_column) formData.append('target_column', options.target_column);
    if (options.task_type) formData.append('task_type', options.task_type);
    if (options.enable_two_stage) formData.append('enable_two_stage', options.enable_two_stage);
    if (options.test_size) formData.append('test_size', options.test_size.toString());

    try {
        const response = await api.post<AnalysisResult>('/upload-and-analyze', formData);
        return response.data;
    } catch (err: unknown) {
        console.error('uploadAndAnalyze error', err);
        if (axios.isAxiosError(err)) {
            const detail = err.response?.data?.detail ?? err.response?.data ?? err.message;
            throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
        }
        throw new Error(err instanceof Error ? err.message : 'Network error while uploading file');
    }
};

export const checkHealth = async (): Promise<boolean> => {
    try {
        await api.get('/health');
        return true;
    } catch (error) {
        console.error("Health check failed:", error);
        return false;
    }
};

export default api;
