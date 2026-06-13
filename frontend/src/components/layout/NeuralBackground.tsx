import { useEffect, useRef } from 'react';

const COLORS: [number, number, number][] = [
    [167, 100, 255],  // bright violet
    [220, 60, 240],  // neon fuchsia
    [50, 230, 255],  // bright cyan
    [40, 220, 150],  // bright emerald
    [255, 210, 50],  // bright amber
    [255, 100, 160],  // hot pink
];

const CFG = {
    nodeCount: 80,
    maxDist: 190,
    nodeSpeed: 0.45,
    particleCount: 65,
    mouseForce: 0.028,   // repulsion strength
    mouseRadius: 170,    // repulsion radius
};

export const NeuralBackground = () => {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d')!;

        let W = 0, H = 0;
        const mouse = { x: -9999, y: -9999 };

        // ── types ───────────────────────────────────────────────────────
        interface INode {
            x: number; y: number; vx: number; vy: number;
            r: number; col: [number, number, number];
            pulse: number; pulseSpeed: number;
        }
        interface IParticle {
            src: INode; dst: INode; t: number; spd: number;
            col: [number, number, number]; active: boolean;
        }

        let nodes: INode[] = [];
        let particles: IParticle[] = [];

        // ── helpers ──────────────────────────────────────────────────────
        function makeNode(initial: boolean): INode {
            return {
                x: initial ? Math.random() * W : -10,
                y: Math.random() * H,
                vx: (Math.random() - 0.5) * CFG.nodeSpeed,
                vy: (Math.random() - 0.5) * CFG.nodeSpeed,
                r: Math.random() * 2.5 + 1.8,
                col: COLORS[Math.floor(Math.random() * COLORS.length)],
                pulse: Math.random() * Math.PI * 2,
                pulseSpeed: 0.02 + Math.random() * 0.025,
            };
        }

        function makeParticle(): IParticle {
            const src = nodes[Math.floor(Math.random() * nodes.length)];
            const candidates = nodes.filter(n => {
                if (n === src) return false;
                const dx = n.x - src.x, dy = n.y - src.y;
                return Math.hypot(dx, dy) < CFG.maxDist;
            });
            if (!candidates.length) return { src, dst: src, t: 0, spd: 0, col: src.col, active: false };
            const dst = candidates[Math.floor(Math.random() * candidates.length)];
            return { src, dst, t: 0, spd: 0.005 + Math.random() * 0.007, col: src.col, active: true };
        }

        function init() {
            nodes = Array.from({ length: CFG.nodeCount }, (_, i) => makeNode(i < CFG.nodeCount));
            particles = Array.from({ length: CFG.particleCount }, () => makeParticle());
        }

        function resize() {
            W = canvas!.width = canvas!.offsetWidth;
            H = canvas!.height = canvas!.offsetHeight;
            init();
        }

        // ── update node ──────────────────────────────────────────────────
        function updateNode(n: INode) {
            const dx = mouse.x - n.x, dy = mouse.y - n.y, d = Math.hypot(dx, dy);
            if (d < CFG.mouseRadius && d > 1) {
                // Repulsion — nodes scatter AWAY from cursor
                const push = (1 - d / CFG.mouseRadius) * CFG.mouseForce;
                n.vx -= (dx / d) * push;
                n.vy -= (dy / d) * push;
            }
            const spd = Math.hypot(n.vx, n.vy);
            const cap = CFG.nodeSpeed * 3;
            if (spd > cap) { n.vx = (n.vx / spd) * cap; n.vy = (n.vy / spd) * cap; }
            n.x += n.vx; n.y += n.vy; n.pulse += n.pulseSpeed;
            if (n.x < -30 || n.x > W + 30 || n.y < -30 || n.y > H + 30) Object.assign(n, makeNode(false));
        }

        // ── draw node ────────────────────────────────────────────────────
        function drawNode(n: INode) {
            const glow = 0.65 + Math.sin(n.pulse) * 0.35;
            const [r, g, b] = n.col;

            // outer halo — dimmed
            let grad = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, n.r * 8);
            grad.addColorStop(0, `rgba(${r},${g},${b},${0.16 * glow})`);
            grad.addColorStop(0.4, `rgba(${r},${g},${b},${0.06 * glow})`);
            grad.addColorStop(1, `rgba(${r},${g},${b},0)`);
            ctx.beginPath(); ctx.arc(n.x, n.y, n.r * 8, 0, Math.PI * 2);
            ctx.fillStyle = grad; ctx.fill();

            // mid glow — dimmed
            grad = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, n.r * 3.5);
            grad.addColorStop(0, `rgba(${r},${g},${b},${0.30 * glow})`);
            grad.addColorStop(1, `rgba(${r},${g},${b},0)`);
            ctx.beginPath(); ctx.arc(n.x, n.y, n.r * 3.5, 0, Math.PI * 2);
            ctx.fillStyle = grad; ctx.fill();

            // coloured ring — dimmed
            ctx.beginPath(); ctx.arc(n.x, n.y, n.r * 1.6, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(${r},${g},${b},${0.25 * glow})`; ctx.fill();

            // soft coloured core (no blinding white)
            ctx.beginPath(); ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(${r},${g},${b},${0.65 * glow})`; ctx.fill();
        }

        // ── draw particle ────────────────────────────────────────────────
        function updateAndDrawParticle(p: IParticle) {
            if (!p.active) { Object.assign(p, makeParticle()); return; }
            p.t += p.spd;
            if (p.t >= 1) { Object.assign(p, makeParticle()); return; }
            const x = p.src.x + (p.dst.x - p.src.x) * p.t;
            const y = p.src.y + (p.dst.y - p.src.y) * p.t;
            const trail = Math.min(p.t, 1 - p.t) * 4;
            const [r, g, b] = p.col;
            const g3 = ctx.createRadialGradient(x, y, 0, x, y, 8);
            g3.addColorStop(0, `rgba(${r},${g},${b},${0.8 * trail})`);
            g3.addColorStop(1, `rgba(${r},${g},${b},0)`);
            ctx.beginPath(); ctx.arc(x, y, 8, 0, Math.PI * 2);
            ctx.fillStyle = g3; ctx.fill();
            ctx.beginPath(); ctx.arc(x, y, 2.2, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(255,255,255,${trail})`; ctx.fill();
        }

        // ── main frame ───────────────────────────────────────────────────
        let rafId: number;
        function frame() {
            ctx.fillStyle = 'rgba(7,6,14,0.30)';
            ctx.fillRect(0, 0, W, H);

            // mouse aura
            if (mouse.x > -100) {
                const ga = ctx.createRadialGradient(mouse.x, mouse.y, 0, mouse.x, mouse.y, CFG.mouseRadius);
                ga.addColorStop(0, 'rgba(139,92,246,0.07)');
                ga.addColorStop(1, 'rgba(139,92,246,0)');
                ctx.beginPath(); ctx.arc(mouse.x, mouse.y, CFG.mouseRadius, 0, Math.PI * 2);
                ctx.fillStyle = ga; ctx.fill();
            }

            // edges
            for (let i = 0; i < nodes.length; i++) {
                for (let j = i + 1; j < nodes.length; j++) {
                    const dx = nodes[j].x - nodes[i].x, dy = nodes[j].y - nodes[i].y;
                    const d = Math.hypot(dx, dy);
                    if (d > CFG.maxDist) continue;
                    const alpha = (1 - d / CFG.maxDist) * 0.14;  // dimmer edges
                    const [r, g, b] = nodes[i].col;
                    ctx.beginPath(); ctx.moveTo(nodes[i].x, nodes[i].y);
                    ctx.lineTo(nodes[j].x, nodes[j].y);
                    ctx.strokeStyle = `rgba(${r},${g},${b},${alpha})`;
                    ctx.lineWidth = 0.9; ctx.stroke();
                }
            }

            nodes.forEach(n => { updateNode(n); drawNode(n); });
            particles.forEach(p => updateAndDrawParticle(p));

            rafId = requestAnimationFrame(frame);
        }

        // ── events ───────────────────────────────────────────────────────
        const onMove = (e: MouseEvent) => { mouse.x = e.clientX; mouse.y = e.clientY; };
        const onLeave = () => { mouse.x = -9999; mouse.y = -9999; };
        const ro = new ResizeObserver(resize);

        window.addEventListener('mousemove', onMove, { passive: true });
        window.addEventListener('mouseleave', onLeave);
        ro.observe(canvas);
        resize();
        frame();

        return () => {
            cancelAnimationFrame(rafId);
            window.removeEventListener('mousemove', onMove);
            window.removeEventListener('mouseleave', onLeave);
            ro.disconnect();
        };
    }, []);

    return (
        <canvas
            ref={canvasRef}
            className="fixed inset-0 z-0 w-full h-full pointer-events-none"
            aria-hidden="true"
        />
    );
};
