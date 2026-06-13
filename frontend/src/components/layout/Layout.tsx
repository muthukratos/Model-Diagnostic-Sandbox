import React from 'react';
import { Navbar } from './Navbar';
import { NeuralBackground } from './NeuralBackground';

interface LayoutProps {
    children: React.ReactNode;
}

export const Layout = ({ children }: LayoutProps) => {
    return (
        <div className="min-h-screen bg-[#07060E] text-slate-100 selection:bg-violet-500/30">
            <NeuralBackground />

            <Navbar />

            <main className="relative z-10 container mx-auto px-6 pt-24 pb-12">
                {children}
            </main>
        </div>
    );
};
