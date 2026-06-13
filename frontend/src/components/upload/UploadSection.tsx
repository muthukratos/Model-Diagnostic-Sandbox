import { useState } from 'react';
import { FileDropzone } from './FileDropzone';
import { Card } from '../common/Card';
import { scanFile } from '../../services/api';
import type { ScanResult } from '../../types';
import { Loader2 } from 'lucide-react';

interface UploadSectionProps {
    onAnalysisStart: (file: File, scanResult: ScanResult) => void;
}

export const UploadSection = ({ onAnalysisStart }: UploadSectionProps) => {
    const [isScanning, setIsScanning] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [dropzoneKey, setDropzoneKey] = useState(0);

    const handleFileSelect = async (file: File) => {
        setIsScanning(true);
        setError(null);
        try {
            const result = await scanFile(file);
            onAnalysisStart(file, result);
        } catch (err: unknown) {
            if (err instanceof Error) {
                setError(err.message || "Failed to scan file");
            } else {
                setError("Failed to scan file");
            }
            // Reset the dropzone so the user can immediately pick a new file
            setDropzoneKey(k => k + 1);
        } finally {
            setIsScanning(false);
        }
    };

    return (
        <div className="w-full max-w-4xl mx-auto space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
            <div className="text-center space-y-4">
                <h2 className="text-3xl font-bold text-white">Upload Your Dataset</h2>
                <p className="text-slate-400">
                    Support for CSV and Excel files. We'll automatically detect your target column.
                </p>
            </div>

            <Card glass className="p-8 bg-slate-900/40 border-slate-700/50">
                {isScanning ? (
                    <div className="flex flex-col items-center justify-center py-12 space-y-4">
                        <Loader2 className="w-12 h-12 text-indigo-500 animate-spin" />
                        <p className="text-xl font-medium text-slate-200">Scanning dataset...</p>
                        <p className="text-sm text-slate-500">Profiling columns and types</p>
                    </div>
                ) : (
                    <FileDropzone key={dropzoneKey} onFileSelect={handleFileSelect} isUploading={isScanning} />
                )}

                {error && (
                    <div className="mt-4 p-4 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400 text-center">
                        {error}
                    </div>
                )}
            </Card>
        </div>
    );
};
