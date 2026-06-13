import type { DatasetProfile } from '../../types';
import { Card } from '../common/Card';
import { Layers, Hash, Type, Activity } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

interface ProfilingViewProps {
    data: DatasetProfile;
}

export const ProfilingView = ({ data }: ProfilingViewProps) => {
    // Prepare data for Missing Values Chart
    const missingData = Object.entries(data.missing_value_percentage || {})
        .filter(([_, val]) => val > 0)
        .map(([name, val]) => ({ name, value: val }));

    // Prepare data for Target Distribution
    const targetData = Object.entries(data.target_class_count || {})
        .map(([name, val]) => ({ name, value: val }));

    return (
        <div className="space-y-6">
            {/* Stats Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <Card className="p-4 bg-slate-800/50 border-slate-700">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <Layers className="w-4 h-4" />
                        <span className="text-xs uppercase tracking-wider">Samples</span>
                    </div>
                    <p className="text-2xl font-bold text-white">{data.total_samples}</p>
                </Card>
                <Card className="p-4 bg-slate-800/50 border-slate-700">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <Hash className="w-4 h-4" />
                        <span className="text-xs uppercase tracking-wider">Features</span>
                    </div>
                    <p className="text-2xl font-bold text-white">{data.total_features}</p>
                </Card>
                <Card className="p-4 bg-slate-800/50 border-slate-700">
                    <div className="flex items-center gap-2 text-blue-400 mb-1">
                        <Hash className="w-4 h-4" />
                        <span className="text-xs uppercase tracking-wider">Numerical</span>
                    </div>
                    <p className="text-2xl font-bold text-white">{data.numerical_features}</p>
                </Card>
                <Card className="p-4 bg-slate-800/50 border-slate-700">
                    <div className="flex items-center gap-2 text-purple-400 mb-1">
                        <Type className="w-4 h-4" />
                        <span className="text-xs uppercase tracking-wider">Categorical</span>
                    </div>
                    <p className="text-2xl font-bold text-white">{data.categorical_features}</p>
                </Card>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Target Distribution */}
                {data.target_class_count && (
                    <Card className="p-6 bg-slate-800/30 border-slate-700">
                        <h3 className="text-sm font-medium text-slate-300 mb-4">Target Distribution</h3>
                        <div className="h-64">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={targetData} layout="vertical" margin={{ left: 20 }}>
                                    <XAxis type="number" hide />
                                    <YAxis dataKey="name" type="category" width={100} tick={{ fill: '#94a3b8', fontSize: 12 }} />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155' }}
                                        itemStyle={{ color: '#f8fafc' }}
                                        cursor={{ fill: 'transparent' }}
                                    />
                                    <Bar dataKey="value" fill="#6366f1" radius={[0, 4, 4, 0]}>
                                        {targetData.map((_, index) => (
                                            <Cell key={`cell-${index}`} fill={['#6366f1', '#8b5cf6', '#d946ef'][index % 3]} />
                                        ))}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </Card>
                )}

                {/* Missing Values */}
                {missingData.length > 0 ? (
                    <Card className="p-6 bg-slate-800/30 border-slate-700">
                        <h3 className="text-sm font-medium text-slate-300 mb-4">Missing Values (%)</h3>
                        <div className="h-64">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={missingData}>
                                    <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                                    <YAxis hide />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155' }}
                                        itemStyle={{ color: '#f8fafc' }}
                                        formatter={(val: number | undefined) => [val ? `${val.toFixed(1)}%` : '0%', 'Missing']}
                                    />
                                    <Bar dataKey="value" fill="#f43f5e" radius={[4, 4, 0, 0]} />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </Card>
                ) : (
                    <Card className="p-6 bg-slate-800/30 border-slate-700 flex flex-col items-center justify-center text-center">
                        <div className="p-3 bg-emerald-500/10 rounded-full mb-3">
                            <Activity className="w-6 h-6 text-emerald-500" />
                        </div>
                        <p className="text-emerald-400 font-medium">Clean Dataset</p>
                        <p className="text-sm text-slate-500 mt-1">No missing values detected.</p>
                    </Card>
                )}
            </div>
        </div>
    );
};
