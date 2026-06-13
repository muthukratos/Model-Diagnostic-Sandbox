import React, { useCallback, useState } from 'react';
import { UploadCloud, FileType, X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { clsx } from 'clsx';

interface FileDropzoneProps {
    onFileSelect: (file: File) => void;
    onFileClear?: () => void;
    isUploading?: boolean;
}

export const FileDropzone = ({ onFileSelect, onFileClear, isUploading }: FileDropzoneProps) => {
    const [isDragging, setIsDragging] = useState(false);
    const [file, setFile] = useState<File | null>(null);
    const [error, setError] = useState<string | null>(null);

    const handleDrag = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === 'dragenter' || e.type === 'dragover') {
            setIsDragging(true);
        } else if (e.type === 'dragleave') {
            setIsDragging(false);
        }
    }, []);

    const validateFile = (file: File) => {
        const validExtensions = ['.csv', '.xls', '.xlsx'];
        const hasValidExtension = validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
        if (!hasValidExtension) {
            setError('Please upload a CSV or Excel file.');
            return false;
        }
        if (file.size > 50 * 1024 * 1024) {
            setError('File size must be less than 50MB.');
            return false;
        }
        return true;
    };

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);
        setError(null);

        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            const droppedFile = e.dataTransfer.files[0];
            if (validateFile(droppedFile)) {
                setFile(droppedFile);
                onFileSelect(droppedFile);
            }
        }
    }, [onFileSelect]);

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        e.preventDefault();
        if (e.target.files && e.target.files[0]) {
            const selectedFile = e.target.files[0];
            if (validateFile(selectedFile)) {
                setFile(selectedFile);
                onFileSelect(selectedFile);
            }
        }
    };

    const clearFile = (e: React.MouseEvent) => {
        e.stopPropagation();
        e.preventDefault();
        setFile(null);
        setError(null);
        if (onFileClear) onFileClear();
    };

    return (
        <div className="w-full max-w-xl mx-auto">
            <div
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                className={clsx(
                    "relative flex flex-col items-center justify-center w-full h-64 border-2 border-dashed rounded-xl transition-all duration-300 ease-in-out cursor-pointer overflow-hidden group",
                    isDragging ? "border-indigo-500 bg-indigo-500/10" : "border-slate-700 bg-slate-900/50 hover:bg-slate-800/50 hover:border-indigo-500/50",
                    file ? "border-indigo-500/50 bg-indigo-500/5" : ""
                )}
            >
                <input
                    type="file"
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                    onChange={handleChange}
                    accept=".csv,.xlsx,.xls"
                    disabled={!!file || isUploading}
                    title=""
                    onClick={(e) => { (e.target as HTMLInputElement).value = ''; }}
                />

                <AnimatePresence mode="wait">
                    {file ? (
                        <motion.div
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -10 }}
                            className="flex flex-col items-center z-20"
                        >
                            <div className="p-4 bg-indigo-500/20 rounded-full mb-3">
                                <FileType className="w-8 h-8 text-indigo-400" />
                            </div>
                            <p className="font-medium text-slate-200 text-lg">{file.name}</p>
                            <p className="text-sm text-slate-400 mt-1">{(file.size / 1024).toFixed(1)} KB</p>

                            {!isUploading && (
                                <button
                                    onClick={clearFile}
                                    className="mt-4 p-2 bg-slate-800 rounded-full hover:bg-red-500/20 hover:text-red-400 transition-colors z-30 pointer-events-auto relative"
                                >
                                    <X className="w-4 h-4" />
                                </button>
                            )}
                        </motion.div>
                    ) : (
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            className="flex flex-col items-center text-center p-6 space-y-4"
                        >
                            <div className={clsx(
                                "p-4 rounded-full transition-colors duration-300",
                                isDragging ? "bg-indigo-500 text-white shadow-lg shadow-indigo-500/50" : "bg-slate-800 text-slate-400 group-hover:bg-indigo-500/20 group-hover:text-indigo-400"
                            )}>
                                <UploadCloud className="w-8 h-8" />
                            </div>
                            <div>
                                <p className="text-lg font-medium text-slate-200">
                                    {isDragging ? "Drop it like it's hot" : "Click to upload or drag and drop"}
                                </p>
                                <p className="text-sm text-slate-500 mt-2">
                                    CSV or Excel files (max 50MB)
                                </p>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>

            {error && (
                <motion.p
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mt-3 text-sm text-red-400 text-center"
                >
                    {error}
                </motion.p>
            )}
        </div>
    );
};
