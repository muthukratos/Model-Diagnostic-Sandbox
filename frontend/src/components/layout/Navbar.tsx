
export const Navbar = () => {
    return (
        <header className="fixed top-0 left-0 right-0 z-50 pointer-events-none">
            <div className="container mx-auto px-6 h-24 flex items-center">
                <div className="flex items-center gap-4 pointer-events-auto select-none">
                    <img
                        src="/logo.png"
                        alt="Model Diagnostic Sandbox logo"
                        className="h-16 w-auto object-contain mix-blend-screen drop-shadow-[0_0_15px_rgba(34,213,238,0.2)]"
                    />
                    <div className="flex flex-col leading-none">
                        <span className="text-xl md:text-2xl font-black bg-clip-text text-transparent bg-gradient-to-br from-white via-white to-slate-500 tracking-tight">
                            Model Diagnostic
                        </span>
                        <div className="flex items-center gap-2 mt-1">
                            <div className="h-[2px] w-6 bg-gradient-to-r from-indigo-500 to-transparent" />
                            <span className="text-[10px] md:text-xs font-black text-indigo-400 uppercase tracking-[0.3em]">
                                Sandbox
                            </span>
                        </div>
                    </div>
                </div>
            </div>
        </header>
    );
};
