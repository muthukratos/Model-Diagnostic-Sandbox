import { useState } from 'react';
import type { ScanResult } from '../../types';
import { Card } from '../common/Card';
import { Button } from '../common/Button';
import { ListFilter, Target, BarChart3, Settings2, PlayCircle } from 'lucide-react';
import { clsx } from 'clsx';


interface ConfigPanelProps {
    scanResult: ScanResult;
    onConfirm: (config: { targetColumn: string; taskType: string; enableTwoStage: string }) => void;
    onCancel: () => void;
    isAnalyzing?: boolean;
}

export const ConfigPanel = ({ scanResult, onConfirm, onCancel, isAnalyzing }: ConfigPanelProps) => {
    const [targetColumn, setTargetColumn] = useState(scanResult.recommendations.target_column);
    const [taskType, setTaskType] = useState(scanResult.recommendations.task_type);
    const [enableTwoStage, setEnableTwoStage] = useState('auto');

    // Filter out columns that are likely not targets (if needed) or just show all
    const columns = scanResult.columns;

    const handleSubmit = () => {
        onConfirm({ targetColumn, taskType, enableTwoStage });
    };

    return (
        <div className="w-full max-w-4xl mx-auto animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="flex items-center justify-between mb-6">
                <div>
                    <h2 className="text-2xl font-bold">Analysis Configuration</h2>
                    <p className="text-slate-400 text-sm mt-1">
                        Review the auto-detected settings for <span className="text-indigo-400 font-mono">{scanResult.filename}</span>
                    </p>
                </div>
                <Button variant="ghost" onClick={onCancel} disabled={isAnalyzing}>Change File</Button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Main Configuration Config */}
                <Card glass className="p-6 space-y-6 bg-slate-900/40 border-slate-700/50">

                    {/* Target Column Selection */}
                    <div className="space-y-3">
                        <div className="flex items-center gap-2 text-indigo-400 font-medium">
                            <Target className="w-5 h-5" />
                            <label>Target Column</label>
                        </div>
                        <p className="text-xs text-slate-500">
                            The column you want to predict.
                            {targetColumn === scanResult.recommendations.target_column && (
                                <span className="ml-1 text-emerald-500">(Auto-detected)</span>
                            )}
                        </p>
                        <select
                            value={targetColumn}
                            onChange={(e) => setTargetColumn(e.target.value)}
                            disabled={isAnalyzing}
                            className="w-full bg-slate-800 border border-slate-700 rounded-lg p-2.5 text-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition-all"
                        >
                            {columns.map(col => (
                                <option key={col} value={col}>{col}</option>
                            ))}
                        </select>
                        {targetColumn === scanResult.recommendations.target_column && (
                            <p className="text-xs text-slate-500 italic bg-slate-800/50 p-2 rounded border border-slate-700/50">
                                {scanResult.recommendations.target_justification}
                            </p>
                        )}
                    </div>

                    {/* Task Type Gen */}
                    <div className="space-y-3 pt-4 border-t border-slate-700/50">
                        <div className="flex items-center gap-2 text-purple-400 font-medium">
                            <ListFilter className="w-5 h-5" />
                            <label>Task Type</label>
                        </div>
                        <div className="grid grid-cols-2 gap-4">
                            <button
                                onClick={() => setTaskType('classification')}
                                disabled={isAnalyzing}
                                className={clsx(
                                    "p-3 rounded-lg border text-sm font-medium transition-all text-left",
                                    taskType === 'classification'
                                        ? "bg-purple-500/20 border-purple-500/50 text-white shadow-lg shadow-purple-900/20"
                                        : "bg-slate-800/50 border-slate-700 text-slate-400 hover:bg-slate-800"
                                )}
                            >
                                <div className="flex items-center gap-2 mb-1">
                                    <span className={clsx("w-2 h-2 rounded-full", taskType === 'classification' ? "bg-purple-400" : "bg-slate-500")} />
                                    Classification
                                </div>
                                <span className="text-xs opacity-70">Predict categories (e.g. Yes/No, Red/Blue)</span>
                            </button>

                            <button
                                onClick={() => setTaskType('regression')}
                                disabled={isAnalyzing}
                                className={clsx(
                                    "p-3 rounded-lg border text-sm font-medium transition-all text-left",
                                    taskType === 'regression'
                                        ? "bg-blue-500/20 border-blue-500/50 text-white shadow-lg shadow-blue-900/20"
                                        : "bg-slate-800/50 border-slate-700 text-slate-400 hover:bg-slate-800"
                                )}
                            >
                                <div className="flex items-center gap-2 mb-1">
                                    <span className={clsx("w-2 h-2 rounded-full", taskType === 'regression' ? "bg-blue-400" : "bg-slate-500")} />
                                    Regression
                                </div>
                                <span className="text-xs opacity-70">Predict numbers (e.g. Price, Age)</span>
                            </button>

                        </div>
                    </div>

                    {/* Advanced Settings */}
                    <div className="space-y-3 pt-4 border-t border-slate-700/50">
                        <div className="flex items-center gap-2 text-slate-400 font-medium text-sm">
                            <Settings2 className="w-4 h-4" />
                            <label>Advanced</label>
                        </div>
                        <div className="flex items-center justify-between p-3 bg-slate-800/30 rounded-lg border border-slate-700/50">
                            <div>
                                <p className="text-sm font-medium text-slate-200">Adaptive Training</p>
                                <p className="text-xs text-slate-500">Automatically optimize for speed on large datasets.</p>
                            </div>
                            <select
                                value={enableTwoStage}
                                onChange={(e) => setEnableTwoStage(e.target.value)}
                                disabled={isAnalyzing}
                                className="bg-slate-900 border border-slate-700 rounded p-1.5 text-xs focus:ring-1 focus:ring-indigo-500 outline-none"
                            >
                                <option value="auto">Auto</option>
                                <option value="true">Force Enable</option>
                                <option value="false">Force Disable</option>
                            </select>
                        </div>
                    </div>

                </Card>

                {/* Dataset Stats / Preview */}
                <div className="space-y-6">
                    <Card glass className="p-6 bg-slate-900/40 border-slate-700/50 h-full flex flex-col">
                        <div className="flex items-center gap-2 text-emerald-400 font-medium mb-4">
                            <BarChart3 className="w-5 h-5" />
                            <h2>Dataset "DNA"</h2>
                        </div>

                        <div className="grid grid-cols-2 gap-4 mb-6">
                            <div className="p-4 bg-slate-800/50 rounded-lg text-center">
                                <p className="text-3xl font-bold text-white">{scanResult.count}</p>
                                <p className="text-xs text-slate-400 uppercase tracking-wider mt-1">Rows</p>
                            </div>
                            <div className="p-4 bg-slate-800/50 rounded-lg text-center">
                                <p className="text-3xl font-bold text-white">{scanResult.columns.length}</p>
                                <p className="text-xs text-slate-400 uppercase tracking-wider mt-1">Columns</p>
                            </div>
                        </div>

                        <div className="flex-1 overflow-hidden flex flex-col">
                            <p className="text-sm font-medium text-slate-300 mb-2">Data Preview</p>
                            <div className="overflow-x-auto relative flex-1 text-xs border border-slate-700 rounded-lg">
                                <table className="w-full text-left bg-slate-800/30 whitespace-nowrap">
                                    <thead className="bg-slate-800/80 sticky top-0">
                                        <tr>
                                            {scanResult.columns.map((col, i) => (
                                                <th key={i} className="p-2 font-medium text-slate-300 border-b border-slate-700">{col}</th>
                                            ))}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {scanResult.preview.map((row, i) => (
                                            <tr key={i} className="border-b border-slate-700/50 hover:bg-slate-700/20">
                                                {scanResult.columns.map((col, j) => (
                                                    <td key={j} className="p-2 text-slate-400">
                                                        {String(row[col] ?? '')}
                                                    </td>
                                                ))}
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </Card>
                </div>
            </div>

            <div className="flex justify-end mt-6">
                <Button
                    onClick={handleSubmit}
                    isLoading={isAnalyzing}
                    size="lg"
                    className="w-full md:w-auto px-8"
                >
                    {!isAnalyzing && <PlayCircle className="w-5 h-5 mr-2" />}
                    {isAnalyzing ? 'Training Models...' : 'Start Analysis'}
                </Button>
            </div>
        </div>
    );
};
