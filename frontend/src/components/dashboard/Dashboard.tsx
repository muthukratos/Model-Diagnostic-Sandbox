import type { AnalysisResult } from '../../types';
import { Card } from '../common/Card';
import { Activity, BarChart2, Lightbulb } from 'lucide-react';
import { useState, useRef, useEffect } from 'react';
import { ProfilingView } from './ProfilingView';
import { ModelComparison } from './ModelComparison';
import { ExplanationsView } from './ExplanationsView';

interface DashboardProps {
    result: AnalysisResult;
    onReset: () => void;
}

type TabId = 'overview' | 'models' | 'explain';

const TABS: { id: TabId; label: string; sublabel: string; icon: React.ElementType; accent: string; glow: string }[] = [
    {
        id: 'overview',
        label: 'Data DNA',
        sublabel: 'Profile',
        icon: Activity,
        accent: 'from-cyan-500 to-blue-500',
        glow: 'shadow-cyan-500/30',
    },
    {
        id: 'models',
        label: 'Model',
        sublabel: 'Leaderboard',
        icon: BarChart2,
        accent: 'from-violet-500 to-indigo-500',
        glow: 'shadow-violet-500/30',
    },
    {
        id: 'explain',
        label: 'Why?',
        sublabel: 'Explainability',
        icon: Lightbulb,
        accent: 'from-amber-400 to-orange-500',
        glow: 'shadow-amber-500/30',
    },
];

export const Dashboard = ({ result, onReset }: DashboardProps) => {
    const [activeTab, setActiveTab] = useState<TabId>('overview');
    const [pillStyle, setPillStyle] = useState({ left: 0, width: 0 });
    const navRef = useRef<HTMLDivElement>(null);
    const buttonRefs = useRef<(HTMLButtonElement | null)[]>([]);

    // Update the sliding pill position whenever activeTab changes
    useEffect(() => {
        const idx = TABS.findIndex(t => t.id === activeTab);
        const btn = buttonRefs.current[idx];
        const nav = navRef.current;
        if (btn && nav) {
            const navRect = nav.getBoundingClientRect();
            const btnRect = btn.getBoundingClientRect();
            setPillStyle({
                left: btnRect.left - navRect.left,
                width: btnRect.width,
            });
        }
    }, [activeTab]);

    const activeTabDef = TABS.find(t => t.id === activeTab)!;

    return (
        <div className="space-y-6 animate-in fade-in zoom-in-95 duration-500">
            {/* Header */}
            <div className="flex flex-col md:flex-row items-center justify-between gap-4">
                <div>
                    <h1 className="text-3xl font-bold text-white mb-2">Analysis Results</h1>
                    <div className="flex flex-wrap items-center gap-2 text-sm text-slate-400">
                        <span className="bg-slate-800 px-2 py-1 rounded border border-slate-700">
                            File: {result.filename || 'Unknown'}
                        </span>
                        <span className="bg-slate-800 px-2 py-1 rounded border border-slate-700 capitalize">
                            Task: {result.task_type.detected}
                        </span>
                        <span className="bg-slate-800 px-2 py-1 rounded border border-slate-700">
                            Target: {result.target_column.selected}
                        </span>
                    </div>
                </div>
                <button
                    onClick={onReset}
                    className="px-4 py-2 text-sm text-white bg-indigo-600 hover:bg-indigo-700 transition-colors rounded-md shadow-md shrink-0"
                >
                    Start New Analysis
                </button>
            </div>

            {/* ── Pill Navigation ── */}
            <div
                ref={navRef}
                className="relative flex items-center gap-1 bg-slate-900/70 backdrop-blur-md p-1.5 rounded-2xl border border-white/10 w-fit shadow-xl"
            >
                {/* Animated sliding pill */}
                <span
                    className={`absolute top-1.5 bottom-1.5 rounded-xl bg-gradient-to-r ${activeTabDef.accent} shadow-lg ${activeTabDef.glow} transition-all duration-300 ease-[cubic-bezier(0.34,1.56,0.64,1)]`}
                    style={{ left: pillStyle.left, width: pillStyle.width }}
                    aria-hidden="true"
                />

                {TABS.map((tab, idx) => {
                    const Icon = tab.icon;
                    const isActive = activeTab === tab.id;
                    return (
                        <button
                            key={tab.id}
                            ref={el => { buttonRefs.current[idx] = el; }}
                            onClick={() => setActiveTab(tab.id)}
                            className={`relative z-10 flex items-center gap-2 pl-3 pr-4 py-2.5 rounded-xl text-sm font-semibold transition-colors duration-200 select-none ${isActive
                                    ? 'text-white'
                                    : 'text-slate-400 hover:text-slate-200'
                                }`}
                        >
                            <Icon className={`w-4 h-4 shrink-0 transition-transform duration-200 ${isActive ? 'scale-110' : ''}`} />
                            <span className="flex flex-col leading-none">
                                <span>{tab.label}</span>
                                <span className={`text-[10px] font-normal mt-0.5 transition-opacity duration-200 ${isActive ? 'opacity-80' : 'opacity-0'}`}>
                                    {tab.sublabel}
                                </span>
                            </span>
                        </button>
                    );
                })}
            </div>

            {/* Content Area */}
            <div className="min-h-[500px]">
                {activeTab === 'overview' && (
                    <Card glass className="p-8 animate-in fade-in slide-in-from-bottom-2 duration-400">
                        <h2 className="text-xl font-bold mb-4">Dataset Profile</h2>
                        <ProfilingView data={result.profiling} />
                    </Card>
                )}

                {activeTab === 'models' && (
                    <div className="animate-in fade-in slide-in-from-bottom-2 duration-400">
                        <h2 className="text-xl font-bold mb-4 sr-only">Model Comparison</h2>
                        <ModelComparison data={result.training_evaluation} />
                    </div>
                )}

                {activeTab === 'explain' && (
                    <div className="animate-in fade-in slide-in-from-bottom-2 duration-400">
                        <h2 className="text-xl font-bold mb-4 sr-only">Model Explanations</h2>
                        <ExplanationsView data={result.explanations} />
                    </div>
                )}
            </div>
        </div>
    );
};
