import { useEffect, useRef } from 'react';

// Parallax depth factor per ribbon — positive moves WITH cursor, negative AGAINST
const RIBBONS = [
    { id: 'silk-1', factor: 28 },
    { id: 'silk-2', factor: -18 },
    { id: 'silk-3', factor: 22 },
    { id: 'silk-4', factor: -10 },
];

export const SilkBackground = () => {
    const wrapperRefs = useRef<(HTMLDivElement | null)[]>([]);

    useEffect(() => {
        let rafId: number;

        const handleMouseMove = (e: MouseEvent) => {
            cancelAnimationFrame(rafId);
            rafId = requestAnimationFrame(() => {
                const mx = (e.clientX / window.innerWidth - 0.5) * 2; // -1 → 1
                const my = (e.clientY / window.innerHeight - 0.5) * 2; // -1 → 1

                wrapperRefs.current.forEach((el, i) => {
                    if (!el) return;
                    const f = RIBBONS[i].factor;
                    el.style.setProperty('--sx', `${mx * f}px`);
                    el.style.setProperty('--sy', `${my * f}px`);
                });
            });
        };

        window.addEventListener('mousemove', handleMouseMove, { passive: true });
        return () => {
            window.removeEventListener('mousemove', handleMouseMove);
            cancelAnimationFrame(rafId);
        };
    }, []);

    return (
        <div className="fixed inset-0 z-0 overflow-hidden pointer-events-none">
            {RIBBONS.map((ribbon, i) => (
                // Outer div — mouse translate only
                <div
                    key={ribbon.id}
                    ref={el => { wrapperRefs.current[i] = el; }}
                    className="silk-wrapper"
                >
                    {/* Inner div — rotation animation only */}
                    <div className={`silk ${ribbon.id}`} />
                </div>
            ))}

            {/* Subtle grain overlay for fabric texture */}
            <div className="silk-grain" />
        </div>
    );
};
