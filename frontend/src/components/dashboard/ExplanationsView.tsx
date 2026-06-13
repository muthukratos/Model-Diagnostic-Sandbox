import type { ExplanationResult } from '../../types';
import { Card } from '../common/Card';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { AlertCircle, HelpCircle } from 'lucide-react';

interface ExplanationsViewProps {
    data?: ExplanationResult;
}

export const ExplanationsView = ({ data }: ExplanationsViewProps) => {
    if (!data || data.error) {
        return (
            <Card className="p-8 border-dashed border-slate-700 bg-slate-800/30 flex flex-col items-center justify-center text-center">
                <div className="p-3 bg-amber-500/10 rounded-full mb-3">
                    <AlertCircle className="w-6 h-6 text-amber-500" />
                </div>
                <p className="text-slate-200 font-medium">No Explanations Available</p>
                <p className="text-sm text-slate-400 mt-1 max-w-md">
                    {data?.error || "The selected model does not support SHAP/LIME explanations (e.g. KNN or SVM in some cases), or an error occurred."}
                </p>
                {data?.note && <p className="text-xs text-slate-500 mt-2">{data.note}</p>}
            </Card>
        );
    }

    // Prepare Global SHAP Data
    const globalData = Object.entries(data.global_explanation?.feature_importance || {})
        .map(([feature, importance]) => ({ feature, importance }))
        .sort((a, b) => b.importance - a.importance)
        .slice(0, 10); // Top 10

    return (
        <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="md:col-span-2">
                    <Card glass className="p-6 bg-slate-800/30 border-slate-700">
                        <div className="flex items-center justify-between mb-6">
                            <div>
                                <h3 className="text-lg font-bold text-white">Global Feature Importance</h3>
                                <p className="text-sm text-slate-400">Which features impact the model the most?</p>
                            </div>
                            <div className="px-2 py-1 bg-slate-800 rounded text-xs text-slate-400 border border-slate-700">
                                Method: {data.global_explanation?.method || 'SHAP'}
                            </div>
                        </div>

                        <div className="h-[400px]">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={globalData} layout="vertical" margin={{ top: 5, right: 30, left: 40, bottom: 5 }}>
                                    <XAxis type="number" hide />
                                    <YAxis
                                        dataKey="feature"
                                        type="category"
                                        width={100}
                                        tick={{ fill: '#94a3b8', fontSize: 12 }}
                                        interval={0}
                                    />
                                    <Tooltip
                                        cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                                        contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155' }}
                                        itemStyle={{ color: '#f8fafc' }}
                                    />
                                    <Bar dataKey="importance" fill="#818cf8" radius={[0, 4, 4, 0]}>
                                        {globalData.map((_, index) => (
                                            <Cell key={`cell-${index}`} fill={['#818cf8', '#a78bfa', '#c084fc'][index % 3]} />
                                        ))}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </Card>
                </div>

                <div>
                    <Card className="p-6 bg-slate-800/30 border-slate-700 h-full">
                        <div className="flex items-center gap-2 mb-4">
                            <HelpCircle className="w-5 h-5 text-indigo-400" />
                            <h3 className="font-bold text-white">Understanding SHAP</h3>
                        </div>
                        <div className="space-y-4 text-sm text-slate-400">
                            <p>
                                <span className="text-slate-200 font-medium">SHAP (SHapley Additive exPlanations)</span> assigns a value to each feature for a particular prediction.
                            </p>
                            <p>
                                Higher bars indicate features that have a stronger influence on the model's output.
                            </p>
                            <ul className="list-disc list-inside space-y-1 pl-1">
                                <li>Longer bars = More important</li>
                                <li>Use this to verify model logic</li>
                                <li>Check for data leakage</li>
                            </ul>
                        </div>
                    </Card>
                </div>
            </div>
        </div>
    );
};
