import type { TrainingEvaluationResult } from '../../types';
import { Card } from '../common/Card';
import { Trophy, Timer, LineChart, Info } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from 'recharts';
import { clsx } from 'clsx';
import { useState } from 'react';

interface ModelComparisonProps {
    data: TrainingEvaluationResult;
}

export const ModelComparison = ({ data }: ModelComparisonProps) => {
    const [openJustification, setOpenJustification] = useState<null | { title: string; text?: string }>(null);

    const isRegression = data.models_evaluated.some(m => 'r2_score' in m.metrics);

    // Sort by primary metric descending so rankings reflect actual performance
    const sortedModels = [...data.models_evaluated].sort((a, b) => {
        const aVal = isRegression ? (a.metrics.r2_score ?? -Infinity) : (a.metrics.accuracy ?? 0);
        const bVal = isRegression ? (b.metrics.r2_score ?? -Infinity) : (b.metrics.accuracy ?? 0);
        return bVal - aVal;
    });

    const chartData = sortedModels.map(m => ({
        name: m.model,
        Accuracy: (m.metrics.accuracy || 0) * 100,
        F1: (m.metrics.f1_score || 0) * 100,
        R2: (m.metrics.r2_score || 0) * 100
    }));

    const metricKeys = isRegression ? ['R2'] : ['Accuracy', 'F1'];

    const bestModel = data.best_model;

    return (
        <div className="space-y-6">
            {/* Winner's Circle */}
            <Card className="p-6 bg-gradient-to-r from-indigo-900/40 to-slate-900/40 border-indigo-500/30 relative overflow-hidden">
                <div className="absolute top-0 right-0 p-4 opacity-10">
                    <Trophy className="w-32 h-32 text-indigo-400" />
                </div>

                <div className="relative z-10 flex flex-col md:flex-row items-start md:items-center gap-6">
                    <div className="p-4 bg-indigo-500/20 rounded-full border border-indigo-500/30">
                        <Trophy className="w-8 h-8 text-indigo-400" />
                    </div>
                    <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                            <span className="text-xs font-bold uppercase tracking-wider text-indigo-400 px-2 py-0.5 bg-indigo-500/10 rounded">Best Performing Model</span>
                        </div>
                        <h3 className="text-2xl font-bold text-white mb-2">{bestModel.model}</h3>
                        <div className="flex flex-wrap gap-4 text-sm text-slate-300">
                            {Object.entries(bestModel.metrics).slice(0, 3).map(([key, val]) => (
                                <div key={key} className="flex items-center gap-1 bg-slate-800/50 px-3 py-1.5 rounded-lg border border-slate-700">
                                    <span className="opacity-60 capitalize">{key.replace('_', ' ')}:</span>
                                    <span className="font-mono text-white">
                                        {typeof val === 'number' ? (val < 1 && val > -1 ? (val * 100).toFixed(1) + '%' : val.toFixed(4)) : val}
                                    </span>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </Card>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Chart */}
                <Card className="lg:col-span-2 p-6 bg-slate-800/30 border-slate-700">
                    <h3 className="text-sm font-medium text-slate-300 mb-6 flex items-center gap-2">
                        <LineChart className="w-4 h-4" />
                        Model Performance Comparison
                    </h3>
                    <div className="h-80">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                                <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} interval={0} angle={0} textAnchor="middle" height={50} />
                                <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} domain={[0, 100]} />
                                <Tooltip
                                    contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155' }}
                                    itemStyle={{ color: '#f8fafc' }}
                                    cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                                    formatter={(val: number | undefined) => [val ? val.toFixed(1) + '%' : '0%', '']}
                                />
                                <Legend wrapperStyle={{ paddingTop: '20px' }} />
                                {metricKeys.map((key, index) => (
                                    <Bar key={key} dataKey={key} fill={['#6366f1', '#10b981'][index % 2]} radius={[4, 4, 0, 0]} />
                                ))}
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </Card>

                {/* List */}
                <Card className="p-0 bg-slate-800/30 border-slate-700 overflow-hidden flex flex-col">
                    <div className="p-4 border-b border-slate-700 bg-slate-800/50">
                        <h3 className="text-sm font-medium text-slate-300">Detailed Rankings</h3>
                    </div>
                    <div className="overflow-auto max-h-[400px]">
                        <div className="divide-y divide-slate-700/50">
                            {sortedModels.map((m, i) => (
                                <div key={m.model} className={clsx("p-4 hover:bg-white/5 transition-colors", m.model === bestModel.model && "bg-indigo-500/5")}>
                                    <div className="flex items-center justify-between mb-2">
                                        <div className="flex items-center gap-2">
                                            <span className={clsx(
                                                "flex items-center justify-center w-5 h-5 rounded-full text-xs font-bold",
                                                i === 0 ? "bg-amber-400 text-slate-900" : "bg-slate-700 text-slate-400"
                                            )}>
                                                {i + 1}
                                            </span>
                                            <span className={clsx("font-medium", m.model === bestModel.model ? "text-indigo-400" : "text-slate-200")}>
                                                {m.model}
                                            </span>
                                        </div>
                                        <div className="flex items-center gap-3">
                                            <button
                                                title="Why this model was chosen"
                                                onClick={() => setOpenJustification({ title: m.model, text: m.justification })}
                                                className="p-1 rounded hover:bg-white/5"
                                            >
                                                <Info className="w-4 h-4 text-slate-300" />
                                            </button>
                                            {m.training_time && (
                                                <span className="text-xs text-slate-500 flex items-center gap-1">
                                                    <Timer className="w-3 h-3" />
                                                    {m.training_time.toFixed(2)}s
                                                </span>
                                            )}
                                        </div>
                                    </div>
                                    <div className="grid grid-cols-2 gap-2">
                                        {Object.entries(m.metrics).slice(0, 2).map(([key, val]) => (
                                            <div key={key} className="flex justify-between text-xs">
                                                <span className="text-slate-500 capitalize">{key.replace('_', ' ')}</span>
                                                <span className="text-slate-300 font-mono">
                                                    {typeof val === 'number' ? (val < 1 && val > -1 ? (val * 100).toFixed(1) + '%' : val.toFixed(4)) : val}
                                                </span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </Card>
            </div>

            {/* Justification Modal */}
            {openJustification && (
                <div className="fixed inset-0 z-50 flex items-center justify-center">
                    <div className="absolute inset-0 bg-black/50" onClick={() => setOpenJustification(null)} />
                    <div className="relative z-10 w-full max-w-lg bg-slate-900 rounded shadow-lg border border-slate-700 p-6">
                        <div className="flex items-start justify-between gap-4">
                            <h4 className="text-lg font-bold text-white">Why {openJustification.title}?</h4>
                            <button onClick={() => setOpenJustification(null)} className="text-slate-400 hover:text-slate-200">Close</button>
                        </div>
                        <div className="mt-4 text-sm text-slate-300">
                            {openJustification.text ? openJustification.text : <em>No justification provided by backend.</em>}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};
