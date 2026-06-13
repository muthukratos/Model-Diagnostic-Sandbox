import { useState } from 'react';
import { Layout } from './components/layout/Layout';
import { UploadSection } from './components/upload/UploadSection';
import { ConfigPanel } from './components/upload/ConfigPanel';
import { Dashboard } from './components/dashboard/Dashboard';
import type { ScanResult, AnalysisResult } from './types';
import { uploadAndAnalyze } from './services/api';

function App() {
    const [scanResult, setScanResult] = useState<ScanResult | null>(null);
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleAnalysisStart = (file: File, result: ScanResult) => {
        setError(null);
        setSelectedFile(file);
        setScanResult(result);
    };

    const handleConfigConfirm = async (config: { targetColumn: string; taskType: string; enableTwoStage: string }) => {
        if (!selectedFile) return;

        setIsAnalyzing(true);
        setError(null);
        try {
            const result = await uploadAndAnalyze(selectedFile, {
                target_column: config.targetColumn,
                task_type: config.taskType,
                enable_two_stage: config.enableTwoStage
            });
            setAnalysisResult(result);
        } catch (err: unknown) {
            if (err instanceof Error) {
                setError(err.message || "Analysis failed");
            } else {
                setError("Analysis failed");
            }
        } finally {
            setIsAnalyzing(false);
        }
    };

    const handleReset = () => {
        setScanResult(null);
        setSelectedFile(null);
        setAnalysisResult(null);
        setError(null);
    };

    return (
        <Layout>
            {/* Error Toast */}
            {error && (
                <div className="fixed top-24 right-6 bg-red-500/10 border border-red-500/20 text-red-500 px-4 py-3 rounded-lg backdrop-blur-md shadow-xl z-50 animate-in slide-in-from-right">
                    {error}
                    <button onClick={() => setError(null)} className="ml-4 hover:text-white">&times;</button>
                </div>
            )}

            <div className="max-w-7xl mx-auto">
                {!scanResult && !analysisResult && (
                    <div className="flex flex-col items-center justify-center min-h-[60vh] text-center space-y-8 animate-in fade-in duration-500">
                        <h1 className="text-5xl md:text-7xl font-bold tracking-tight">
                            <span className="bg-clip-text text-transparent bg-gradient-to-b from-white to-slate-400">
                                ML Diagnostics
                            </span>
                            <br />
                            <span className="text-indigo-500">Sandbox</span>
                        </h1>
                        <p className="text-lg text-slate-400 max-w-2xl mx-auto">
                            Upload your dataset and get instant insights. Automated model selection,
                            training, and explainability in one premium dashboard.
                        </p>

                        <UploadSection onAnalysisStart={handleAnalysisStart} />
                    </div>
                )}

                {scanResult && !analysisResult && (
                    <ConfigPanel
                        scanResult={scanResult}
                        onConfirm={handleConfigConfirm}
                        onCancel={handleReset}
                        isAnalyzing={isAnalyzing}
                    />
                )}

                {analysisResult && (
                    <Dashboard result={analysisResult} onReset={handleReset} />
                )}
            </div>
        </Layout>
    );
}

export default App;
