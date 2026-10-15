/** Charts of the Research workspace (SVG / canvas, dark-well style of `ui/plots.tsx`): Sobol' bars with CIs and the MC-noise
 *  floor, Morris μ*–σ plot, replicate scatter with error bars, parallel coordinates with brushing, the excitation × emission
 *  channel map with design picks, and a raster viewer for abundance / parameter maps. */
import { useEffect, useMemo, useRef, useState } from 'react'
import { useWidth } from '../ui/charts'
import { fmtTick, fmtVal, niceTicks } from '../ui/plots'
import { colorAt, lut, type CmapName } from '../ui/colormaps'
import { wavelengthToHex } from '../lib/color'
import type { MorrisMetric, SobolMetric } from '../api/researchTypes'
import type { StudyRunRow } from '../api/researchTypes'
import { axisT, numOf, type Axis, type PointAgg } from './logic'

const AX = '#6b7788', AXL = '#566174', GRID = 'rgba(255,255,255,.07)'
export const C_S1 = '#35d8ff', C_ST = '#9b8cff', C_NOISE = '#ffb547', C_OK = '#4ef28a'
const MONO = 'var(--mono)'

/** CSS colour from a colormap at t in [0, 1]. */
export const cmapCss = (name: CmapName, t: number) => { const [r, g, b] = colorAt(name, Math.max(0, Math.min(1, t))); return `rgb(${r},${g},${b})` }

// ================================================================================================ Sobol bars
/** Horizontal grouped bars S1 / ST per parameter with 95 % CI whiskers; the dashed amber line is the Monte Carlo noise floor
 *  (fraction of output variance explained by replicate noise, so indices below it are not distinguishable from zero). */
export function SobolChart({ metric, params }: { metric: SobolMetric; params: string[] }) {
  const [ref, w] = useWidth<HTMLDivElement>()
  const rows = useMemo(() => params.map((p) => ({ p, ...metric.indices[p] })).sort((a, b) => (b.ST ?? -1) - (a.ST ?? -1)), [metric, params])
  const labW = Math.min(170, Math.max(96, w * 0.28))
  const m = { l: labW, r: 40, t: 22, b: 30 }
  const rowH = 44
  const h = m.t + m.b + rows.length * rowH
  const iw = Math.max(40, w - m.l - m.r)
  // domain from the point estimates (CIs of small designs can be huge; whiskers are clipped with an arrowhead)
  let hi = 1, lo = 0
  for (const r of rows) { hi = Math.max(hi, r.ST ?? 0, r.S1 ?? 0); lo = Math.min(lo, r.S1 ?? 0, r.ST ?? 0) }
  hi = Math.min(1.5, Math.ceil(hi * 10) / 10); lo = Math.max(-1, Math.floor(lo * 10) / 10)
  const sx = (v: number) => m.l + ((Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * iw
  const nf = metric.noise.noise_floor
  const ticks = niceTicks(lo, hi, 5)
  const [tip, setTip] = useState<{ x: number; y: number; r: (typeof rows)[number] } | null>(null)
  return (
    <div ref={ref} style={{ position: 'relative', width: '100%' }}>
      <svg width={w} height={h} style={{ display: 'block', overflow: 'visible' }} role="img" aria-label="Sobol sensitivity indices per parameter">
        {ticks.map((v) => <g key={v}><line x1={sx(v)} x2={sx(v)} y1={m.t} y2={h - m.b} stroke={GRID} /><text x={sx(v)} y={h - m.b + 14} textAnchor="middle" fontSize="9.5" fill={AX} fontFamily={MONO}>{fmtTick(v)}</text></g>)}
        <line x1={sx(0)} x2={sx(0)} y1={m.t} y2={h - m.b} stroke="#3a4350" />
        {nf != null && nf > 0 && <>
          <rect x={sx(0)} y={m.t} width={Math.max(0, sx(Math.min(nf, hi)) - sx(0))} height={h - m.t - m.b} fill={C_NOISE} opacity={0.07} />
          <line x1={sx(nf)} x2={sx(nf)} y1={m.t - 4} y2={h - m.b} stroke={C_NOISE} strokeDasharray="4 3" strokeWidth={1.3} />
          <text x={sx(nf) + 4} y={m.t - 8} fontSize="9.5" fill={C_NOISE} fontFamily={MONO}>MC noise floor {(nf * 100).toFixed(nf < 0.01 ? 2 : 1)} %</text>
        </>}
        {rows.map((r, i) => {
          const y0 = m.t + i * rowH
          const bar = (v: number | null, ci: [number, number] | null, dy: number, c: string) => v == null ? null : (
            <g>
              <rect x={sx(Math.min(0, v))} y={y0 + dy} width={Math.max(1, Math.abs(sx(v) - sx(0)))} height={13} rx={3} fill={c} opacity={0.9} />
              {ci && Number.isFinite(ci[0]) && Number.isFinite(ci[1]) && <g stroke="#e8ecf2" strokeWidth={1.3} opacity={0.85}>
                <line x1={sx(ci[0])} x2={sx(ci[1])} y1={y0 + dy + 6.5} y2={y0 + dy + 6.5} />
                {ci[0] >= lo ? <line x1={sx(ci[0])} x2={sx(ci[0])} y1={y0 + dy + 2} y2={y0 + dy + 11} /> : <path d={`M${sx(lo) + 5},${y0 + dy + 2.5}L${sx(lo)},${y0 + dy + 6.5}L${sx(lo) + 5},${y0 + dy + 10.5}`} fill="none" />}
                {ci[1] <= hi ? <line x1={sx(ci[1])} x2={sx(ci[1])} y1={y0 + dy + 2} y2={y0 + dy + 11} /> : <path d={`M${sx(hi) - 5},${y0 + dy + 2.5}L${sx(hi)},${y0 + dy + 6.5}L${sx(hi) - 5},${y0 + dy + 10.5}`} fill="none" />}</g>}
              <text x={sx(Math.min(hi, Math.max(v, ci?.[1] ?? v))) + 6} y={y0 + dy + 10.5} fontSize="10" fill="#cdd6e2" fontFamily={MONO}>{v.toFixed(2)}</text>
            </g>)
          return (
            <g key={r.p} onMouseMove={(e) => { const b = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect(); setTip({ x: e.clientX - b.left, y: e.clientY - b.top, r }) }} onMouseLeave={() => setTip(null)}>
              <rect x={0} y={y0} width={w} height={rowH} fill={tip?.r.p === r.p ? 'rgba(255,255,255,.03)' : 'transparent'} />
              <text x={m.l - 10} y={y0 + rowH / 2 + 4} textAnchor="end" fontSize="11.5" fill="#dfe6f0" style={{ font: '500 11.5px var(--font)' }}>{r.p.length > 24 ? `${r.p.slice(0, 23)}…` : r.p}</text>
              {bar(r.S1, r.S1_ci, 6, C_S1)}
              {bar(r.ST, r.ST_ci, 23, C_ST)}
            </g>)
        })}
        <text x={m.l + iw} y={h - 2} textAnchor="end" fontSize="9.5" fill={AXL}>fraction of output variance</text>
      </svg>
      <div className="legend" style={{ padding: '2px 0 0' }}><span><i style={{ background: C_S1 }} />S1 first order</span><span><i style={{ background: C_ST }} />ST total order</span><span><i style={{ background: C_NOISE, height: 2, borderRadius: 0, verticalAlign: 3 }} />Monte Carlo noise floor</span></div>
      {tip && <div className="plot-tip" style={{ left: Math.min(tip.x + 14, w - 190), top: Math.max(0, tip.y - 52) }}>
        <b>{tip.r.p}</b>
        <div><i style={{ background: C_S1 }} />S1<span>{tip.r.S1?.toFixed(3)} {tip.r.S1_ci ? `[${tip.r.S1_ci[0].toFixed(2)}, ${tip.r.S1_ci[1].toFixed(2)}]` : ''}</span></div>
        <div><i style={{ background: C_ST }} />ST<span>{tip.r.ST?.toFixed(3)} {tip.r.ST_ci ? `[${tip.r.ST_ci[0].toFixed(2)}, ${tip.r.ST_ci[1].toFixed(2)}]` : ''}</span></div>
      </div>}
    </div>
  )
}

// ================================================================================================ Morris
/** Morris screening plot: μ* (mean |elementary effect|, importance) against σ (sd of the effects, non-linearity / interactions). */
export function MorrisChart({ metric, params }: { metric: MorrisMetric; params: string[] }) {
  const [ref, w] = useWidth<HTMLDivElement>()
  const h = 300
  const m = { l: 50, r: 16, t: 14, b: 36 }
  const iw = Math.max(40, w - m.l - m.r), ih = h - m.t - m.b
  const pts = params.map((p) => ({ p, ...metric.indices[p] })).filter((r) => r.mu_star != null && r.sigma != null)
  const xmax = Math.max(1e-12, ...pts.map((q) => q.mu_star_ci?.[1] ?? q.mu_star ?? 0)) * 1.12
  const ymax = Math.max(1e-12, ...pts.map((q) => q.sigma ?? 0), xmax * 0.6) * 1.12
  const sx = (v: number) => m.l + (v / xmax) * iw, sy = (v: number) => m.t + ih - (v / ymax) * ih
  const xt = niceTicks(0, xmax, 5), yt = niceTicks(0, ymax, 4)
  const hue = (i: number) => `hsl(${(i * 67 + 190) % 360} 80% 66%)`
  const [hov, setHov] = useState<string | null>(null)
  return (
    <div ref={ref} style={{ position: 'relative', width: '100%' }}>
      <svg width={w} height={h} style={{ display: 'block', overflow: 'visible' }} role="img" aria-label="Morris screening mu-star against sigma">
        {yt.map((v) => <g key={v}><line x1={m.l} x2={m.l + iw} y1={sy(v)} y2={sy(v)} stroke={GRID} /><text x={m.l - 6} y={sy(v) + 3} textAnchor="end" fontSize="9.5" fill={AX} fontFamily={MONO}>{fmtTick(v)}</text></g>)}
        {xt.map((v) => <g key={v}><line x1={sx(v)} x2={sx(v)} y1={m.t} y2={m.t + ih} stroke={GRID} /><text x={sx(v)} y={m.t + ih + 14} textAnchor="middle" fontSize="9.5" fill={AX} fontFamily={MONO}>{fmtTick(v)}</text></g>)}
        <line x1={m.l} x2={m.l + iw} y1={m.t + ih} y2={m.t + ih} stroke="#3a4350" /><line x1={m.l} x2={m.l} y1={m.t} y2={m.t + ih} stroke="#2a313b" />
        {/* σ = μ* separates monotone-linear effects (below) from non-linear / interacting ones (above) */}
        <line x1={sx(0)} y1={sy(0)} x2={sx(Math.min(xmax, ymax))} y2={sy(Math.min(xmax, ymax))} stroke="#4b5666" strokeDasharray="4 4" />
        <text x={sx(Math.min(xmax, ymax)) - 4} y={sy(Math.min(xmax, ymax)) + 14} textAnchor="end" fontSize="9.5" fill="#6b7788">σ = μ*</text>
        <text x={m.l + 8} y={m.t + 12} fontSize="9.5" fill="#6b7788">non-linear / interacting ↑</text>
        {pts.map((q, i) => (
          <g key={q.p} onMouseEnter={() => setHov(q.p)} onMouseLeave={() => setHov(null)} opacity={hov && hov !== q.p ? 0.4 : 1}>
            {q.mu_star_ci && <line x1={sx(Math.max(0, q.mu_star_ci[0]))} x2={sx(q.mu_star_ci[1])} y1={sy(q.sigma!)} y2={sy(q.sigma!)} stroke={hue(i)} strokeWidth={1.4} opacity={0.6} />}
            <circle cx={sx(q.mu_star!)} cy={sy(q.sigma!)} r={7} fill={hue(i)} opacity={0.22} /><circle cx={sx(q.mu_star!)} cy={sy(q.sigma!)} r={4.2} fill="#0b0e12" stroke={hue(i)} strokeWidth={2} />
            <text x={sx(q.mu_star!) + 10} y={sy(q.sigma!) + 3.5} fontSize="11" fill="#dfe6f0" style={{ font: '500 11px var(--font)' }}>{q.p.length > 26 ? `${q.p.slice(0, 25)}…` : q.p}</text>
          </g>))}
        <text x={m.l + iw} y={h - 3} textAnchor="end" fontSize="9.5" fill={AXL}>μ* (mean |elementary effect|)</text>
        <text x={4} y={9} fontSize="9.5" fill={AXL}>σ</text>
      </svg>
      {hov && (() => { const q = pts.find((x) => x.p === hov); return q ? <div className="plot-tip" style={{ right: 10, top: 6 }}><b>{q.p}</b><div>μ*<span>{fmtVal(q.mu_star!)}</span></div><div>μ<span>{fmtVal(q.mu!)}</span></div><div>σ<span>{fmtVal(q.sigma!)}</span></div><div>effects<span>{q.n_effects}</span></div></div> : null })()}
    </div>
  )
}

// ================================================================================================ scatter with replicate error bars
export interface ScatterProps {
  points: PointAgg[]
  xLabel: string; yLabel: string
  logX?: boolean
  color?: (p: PointAgg) => string
  selectedPoint?: number | null
  onSelect?: (point: number) => void
  height?: number
  /** run rows to draw individual replicates as faint dots */
  showReplicates?: boolean
  metricKey: string
  yFmt?: (v: number) => string
}
export function ScatterReps({ points, xLabel, yLabel, logX, color, selectedPoint, onSelect, height = 280, showReplicates = true, metricKey }: ScatterProps) {
  const [ref, w] = useWidth<HTMLDivElement>()
  const m = { l: 56, r: 14, t: 12, b: 34 }
  const iw = Math.max(40, w - m.l - m.r), ih = height - m.t - m.b
  const [hov, setHov] = useState<PointAgg | null>(null)
  const { x0, x1, y0, y1 } = useMemo(() => {
    let a = Infinity, b = -Infinity, c = Infinity, d = -Infinity
    for (const p of points) { a = Math.min(a, p.x); b = Math.max(b, p.x); c = Math.min(c, p.y - p.std); d = Math.max(d, p.y + p.std) }
    if (!Number.isFinite(a)) { a = 0; b = 1; c = 0; d = 1 }
    if (a === b) { a -= 0.5; b += 0.5 }
    if (c === d) { c -= 0.5; d += 0.5 }
    const pad = (d - c) * 0.08, xp = logX && a > 0 ? 0 : (b - a) * 0.025
    return { x0: logX && a > 0 ? a / 1.05 : a - xp, x1: logX && a > 0 ? b * 1.05 : b + xp, y0: c - pad, y1: d + pad }
  }, [points, logX])
  const lx = logX && x0 > 0
  const tx = (v: number) => (lx ? (Math.log(v) - Math.log(x0)) / (Math.log(x1) - Math.log(x0)) : (v - x0) / (x1 - x0))
  const sx = (v: number) => m.l + tx(v) * iw
  const sy = (v: number) => m.t + ih - ((v - y0) / (y1 - y0)) * ih
  const xt = lx ? [] : niceTicks(x0, x1, Math.max(3, Math.floor(iw / 70)))
  const yt = niceTicks(y0, y1, Math.max(3, Math.floor(ih / 40)))
  // marginal trend: mean of the points in 8 equal-count bins of x
  const trend = useMemo(() => {
    if (points.length < 12) return []
    const s = [...points].sort((a, b) => a.x - b.x), nb = Math.min(10, Math.floor(s.length / 4)), out: { x: number; y: number }[] = []
    for (let i = 0; i < nb; i++) { const seg = s.slice(Math.floor((i * s.length) / nb), Math.floor(((i + 1) * s.length) / nb)); if (seg.length) out.push({ x: seg.reduce((a, p) => a + p.x, 0) / seg.length, y: seg.reduce((a, p) => a + p.y, 0) / seg.length }) }
    return out
  }, [points])
  return (
    <div ref={ref} style={{ position: 'relative', width: '100%' }}>
      <svg width={w} height={height} style={{ display: 'block', overflow: 'visible' }} role="img" aria-label={`${yLabel} against ${xLabel}`}>
        <defs><clipPath id="sc-clip"><rect x={m.l} y={m.t - 4} width={iw} height={ih + 8} /></clipPath></defs>
        {yt.map((v) => <g key={v}><line x1={m.l} x2={m.l + iw} y1={sy(v)} y2={sy(v)} stroke={GRID} /><text x={m.l - 6} y={sy(v) + 3} textAnchor="end" fontSize="9.5" fill={AX} fontFamily={MONO}>{fmtTick(v)}</text></g>)}
        {(lx ? niceTicks(x0, x1, 5) : xt).map((v) => <g key={v}><line x1={sx(v)} x2={sx(v)} y1={m.t + ih} y2={m.t + ih + 3} stroke="#4b5666" /><text x={sx(v)} y={m.t + ih + 14} textAnchor="middle" fontSize="9.5" fill={AX} fontFamily={MONO}>{fmtTick(v)}</text></g>)}
        <line x1={m.l} x2={m.l + iw} y1={m.t + ih} y2={m.t + ih} stroke="#3a4350" /><line x1={m.l} x2={m.l} y1={m.t} y2={m.t + ih} stroke="#2a313b" />
        <g clipPath="url(#sc-clip)">
          {trend.length > 1 && <path d={trend.map((p, i) => `${i ? 'L' : 'M'}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join('')} fill="none" stroke="rgba(255,255,255,.28)" strokeWidth={1.6} strokeDasharray="5 4" />}
          {showReplicates && points.some((p) => p.n > 1) && points.flatMap((p) => p.rows.map((r) => { const v = numOf(r, metricKey); return Number.isFinite(v) ? <circle key={r.run_id} cx={sx(p.x)} cy={sy(v)} r={1.8} fill="rgba(255,255,255,.35)" /> : null }))}
          {points.map((p) => {
            const c = color?.(p) ?? C_S1, sel = selectedPoint === p.point
            return (
              <g key={p.point} style={{ cursor: onSelect ? 'pointer' : undefined }} onMouseEnter={() => setHov(p)} onMouseLeave={() => setHov(null)} onClick={() => onSelect?.(p.point)}>
                {p.std > 0 && <g stroke={c} strokeWidth={1.3} opacity={0.75}><line x1={sx(p.x)} x2={sx(p.x)} y1={sy(p.y - p.std)} y2={sy(p.y + p.std)} /><line x1={sx(p.x) - 3} x2={sx(p.x) + 3} y1={sy(p.y - p.std)} y2={sy(p.y - p.std)} /><line x1={sx(p.x) - 3} x2={sx(p.x) + 3} y1={sy(p.y + p.std)} y2={sy(p.y + p.std)} /></g>}
                <circle cx={sx(p.x)} cy={sy(p.y)} r={sel ? 6 : hov === p ? 5.5 : 4} fill={sel ? c : '#0b0e12'} stroke={sel ? '#fff' : c} strokeWidth={sel ? 2 : 1.7} />
              </g>)
          })}
        </g>
        <text x={m.l + iw} y={height - 3} textAnchor="end" fontSize="10" fill={AXL}>{xLabel}</text>
        <text x={4} y={10} fontSize="10" fill={AXL}>{yLabel}</text>
      </svg>
      {hov && <div className="plot-tip" style={{ left: Math.min(sx(hov.x) + 12, w - 190), top: Math.max(0, sy(hov.y) - 54) }}>
        <b>{xLabel.split(' (')[0]} = {fmtVal(hov.x)}</b>
        <div>{yLabel.split(' (')[0]}<span>{fmtVal(hov.y)}{hov.n > 1 ? ` ± ${fmtVal(hov.std)}` : ''}</span></div>
        {hov.n > 1 && <div>replicates<span>{hov.n}</span></div>}
      </div>}
    </div>
  )
}

// ================================================================================================ parallel coordinates
export interface ParallelProps {
  rows: StudyRunRow[]
  axes: Axis[]
  colorKey: string
  cmap?: CmapName
  brush: Record<string, [number, number]>
  onBrush: (b: Record<string, [number, number]>) => void
  selected?: string | null
  onSelect?: (runId: string) => void
  height?: number
}
/** Polylines through one vertical axis per parameter and the metric, coloured by the metric. Drag on an axis to brush a range;
 *  the brush filters the table and fades the other lines. */
export function ParallelCoords({ rows, axes, colorKey, cmap = 'viridis', brush, onBrush, selected, onSelect, height = 300 }: ParallelProps) {
  const [ref, w] = useWidth<HTMLDivElement>()
  const cv = useRef<HTMLCanvasElement>(null)
  const m = { l: 70, r: 70, t: 36, b: 22 }
  const iw = Math.max(40, w - m.l - m.r), ih = height - m.t - m.b
  const ax = (i: number) => m.l + (axes.length > 1 ? (i / (axes.length - 1)) * iw : iw / 2)
  const ay = (t: number) => m.t + (1 - t) * ih
  const colAxis = axes.find((a) => a.key === colorKey) ?? axes[axes.length - 1]
  const passes = (r: StudyRunRow) => axes.every((a) => { const b = brush[a.key]; if (!b) return true; const t = axisT(a, numOf(r, a.key)); return t >= b[0] - 1e-9 && t <= b[1] + 1e-9 })
  const [drag, setDrag] = useState<{ key: string; t0: number; t1: number } | null>(null)
  const [hoverRun, setHoverRun] = useState<string | null>(null)

  const lines = useMemo(() => rows.filter((r) => r.status === 'ok').map((r) => ({ r, t: axes.map((a) => axisT(a, numOf(r, a.key))), c: axisT(colAxis, numOf(r, colorKey)) })).filter((l) => l.t.every(Number.isFinite)), [rows, axes, colAxis, colorKey])
  useEffect(() => {
    const c = cv.current; if (!c) return
    const dpr = window.devicePixelRatio || 1
    c.width = Math.round(w * dpr); c.height = Math.round(height * dpr)
    const g = c.getContext('2d'); if (!g) return
    g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, height)
    const draw = (l: (typeof lines)[number], alpha: number, lw: number, col?: string) => {
      g.beginPath(); l.t.forEach((t, i) => { const x = ax(i), y = ay(t); if (i) g.lineTo(x, y); else g.moveTo(x, y) })
      g.strokeStyle = col ?? cmapCss(cmap, l.c); g.globalAlpha = alpha; g.lineWidth = lw; g.stroke()
    }
    const alpha = Math.max(0.12, Math.min(0.7, 60 / Math.max(1, lines.length)))
    const hasBrush = Object.keys(brush).length > 0
    lines.filter((l) => !passes(l.r)).forEach((l) => draw(l, hasBrush ? 0.05 : alpha, 1, hasBrush ? '#8895a8' : undefined))
    lines.filter((l) => passes(l.r)).forEach((l) => draw(l, hasBrush ? Math.min(0.85, alpha * 2.2) : alpha, 1.2))
    const sel = lines.find((l) => l.r.run_id === (hoverRun ?? selected)); if (sel) { draw(sel, 1, 4, '#0b0e12'); draw(sel, 1, 2.2, '#ffffff') }
    g.globalAlpha = 1
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lines, brush, w, height, selected, hoverRun, cmap, axes])

  const tOf = (e: React.PointerEvent) => { const r = (e.currentTarget as SVGElement).ownerSVGElement!.getBoundingClientRect(); return Math.max(0, Math.min(1, 1 - (e.clientY - r.top - m.t) / ih)) }
  return (
    <div ref={ref} style={{ position: 'relative', width: '100%', height }}>
      <canvas ref={cv} style={{ position: 'absolute', inset: 0, width: w, height }} />
      {/* hover / click pick: nearest polyline at the pointer */}
      <div style={{ position: 'absolute', left: m.l, top: m.t, width: iw, height: ih }}
        onMouseMove={(e) => {
          const bb = e.currentTarget.getBoundingClientRect(); const px = e.clientX - bb.left + m.l, py = e.clientY - bb.top + m.t
          const seg = axes.length > 1 ? Math.min(axes.length - 2, Math.max(0, Math.floor(((px - m.l) / iw) * (axes.length - 1)))) : 0
          const u = axes.length > 1 ? ((px - m.l) / iw) * (axes.length - 1) - seg : 0
          let best: string | null = null, bd = 12
          for (const l of lines) { if (!passes(l.r)) continue; const y = ay(l.t[seg] * (1 - u) + l.t[Math.min(axes.length - 1, seg + 1)] * u); const d = Math.abs(y - py); if (d < bd) { bd = d; best = l.r.run_id } }
          setHoverRun(best)
        }}
        onMouseLeave={() => setHoverRun(null)} onClick={() => { if (hoverRun) onSelect?.(hoverRun) }} />
      <svg width={w} height={height} style={{ position: 'absolute', inset: 0, overflow: 'visible', pointerEvents: 'none' }} role="img" aria-label="Parallel coordinates of the runs">
        {axes.map((a, i) => {
          const b = drag?.key === a.key ? [Math.min(drag.t0, drag.t1), Math.max(drag.t0, drag.t1)] as [number, number] : brush[a.key]
          const ticks = a.log ? [a.lo, Math.sqrt(a.lo * a.hi), a.hi] : [a.lo, (a.lo + a.hi) / 2, a.hi]
          return (
            <g key={a.key}>
              <line x1={ax(i)} x2={ax(i)} y1={m.t} y2={m.t + ih} stroke={a.key === colorKey ? '#8b7bff' : '#4b5666'} strokeWidth={a.key === colorKey ? 2 : 1.2} />
              {ticks.map((v, k) => <text key={k} x={ax(i) + (i === axes.length - 1 ? 7 : -7)} y={ay(axisT(a, v)) + 3} textAnchor={i === axes.length - 1 ? 'start' : 'end'} fontSize="9" fill={AX} fontFamily={MONO}>{fmtTick(v)}</text>)}
              <text x={ax(i)} y={m.t - 16} textAnchor="middle" fontSize="10.5" fill={a.key === colorKey ? '#b7acff' : '#c3ccd9'} style={{ font: '600 10.5px var(--font)' }}>{a.label.length > 18 ? `${a.label.slice(0, 17)}…` : a.label}</text>
              {b && <rect x={ax(i) - 6} y={ay(b[1])} width={12} height={Math.max(2, ay(b[0]) - ay(b[1]))} rx={3} fill="rgba(53,216,255,.22)" stroke={C_S1} strokeWidth={1.3} />}
              <rect x={ax(i) - 12} y={m.t - 4} width={24} height={ih + 8} fill="transparent" style={{ cursor: 'ns-resize', pointerEvents: 'all' }}
                onPointerDown={(e) => { (e.currentTarget as Element).setPointerCapture(e.pointerId); const t = tOf(e); setDrag({ key: a.key, t0: t, t1: t }) }}
                onPointerMove={(e) => { if (drag?.key === a.key) setDrag({ ...drag, t1: tOf(e) }) }}
                onPointerUp={() => { if (!drag) return; const lo = Math.min(drag.t0, drag.t1), hi = Math.max(drag.t0, drag.t1); const nb = { ...brush }; if (hi - lo < 0.02) delete nb[drag.key]; else nb[drag.key] = [lo, hi]; setDrag(null); onBrush(nb) }}
                onDoubleClick={() => { const nb = { ...brush }; delete nb[a.key]; onBrush(nb) }} />
            </g>)
        })}
        <rect x={m.l} y={m.t} width={iw} height={ih} fill="transparent" style={{ pointerEvents: 'none' }} />
      </svg>
      {Object.keys(brush).length > 0 && <button className="btn sm ghost" style={{ position: 'absolute', right: 0, top: 0 }} onClick={() => onBrush({})}>Clear brush</button>}
    </div>
  )
}

// ================================================================================================ channel map
export interface ChannelMapProps {
  z: number[][]
  xs: number[]
  ys: number[]
  index: number[][]
  /** channel index → rank (1-based) of the greedy design picks */
  picks?: Map<number, number>
  cmap?: CmapName
  log?: boolean
  height?: number
  unit?: string
  onPick?: (channel: number) => void
  title?: string
}
/** Excitation (rows) × emission (columns) map of a per-channel quantity; design picks are outlined and numbered by selection order. */
export function ChannelMap({ z, xs, ys, index, picks, cmap = 'magma', log = true, height = 260, unit, onPick }: ChannelMapProps) {
  const [ref, w] = useWidth<HTMLDivElement>()
  const cv = useRef<HTMLCanvasElement>(null)
  const m = { l: 46, r: 58, t: 12, b: 32 }
  const iw = Math.max(40, w - m.l - m.r), ih = height - m.t - m.b
  const [hov, setHov] = useState<{ r: number; c: number } | null>(null)
  const nR = ys.length, nC = xs.length
  const { lo, hi } = useMemo(() => {
    let a = Infinity, b = -Infinity
    for (const r of z) for (const v of r) if (Number.isFinite(v) && v > 0) { a = Math.min(a, v); b = Math.max(b, v) }
    if (!Number.isFinite(a)) return { lo: 0, hi: 1 }
    return { lo: log ? Math.max(a, b * 1e-4) : 0, hi: b }
  }, [z, log])
  const tn = (v: number) => (!Number.isFinite(v) || v <= 0 ? 0 : log ? Math.max(0, Math.min(1, (Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo) || 1))) : Math.max(0, Math.min(1, v / hi)))
  const cw = iw / Math.max(1, nC), rh = ih / Math.max(1, nR)
  useEffect(() => {
    const c = cv.current; if (!c) return
    const dpr = window.devicePixelRatio || 1
    c.width = Math.round(w * dpr); c.height = Math.round(height * dpr)
    const g = c.getContext('2d'); if (!g) return
    g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, height)
    const L = lut(cmap)
    for (let r = 0; r < nR; r++) for (let k = 0; k < nC; k++) {
      const v = z[r][k]
      const x = m.l + k * cw, y = m.t + (nR - 1 - r) * rh
      if (!Number.isFinite(v)) { g.fillStyle = 'rgba(255,255,255,.025)'; g.fillRect(x, y, cw + 0.5, rh - 1); continue }
      const i = Math.round(tn(v) * 255) * 3
      g.fillStyle = `rgb(${L[i]},${L[i + 1]},${L[i + 2]})`; g.fillRect(x, y, cw + 0.5, rh - 1)
    }
    g.strokeStyle = '#3a4350'; g.strokeRect(m.l + 0.5, m.t + 0.5, iw, ih)
    g.font = '9.5px "JetBrains Mono Variable", monospace'; g.fillStyle = AX; g.textAlign = 'center'
    const step = Math.max(1, Math.ceil(nC / Math.max(3, Math.floor(iw / 56))))
    for (let k = 0; k < nC; k += step) g.fillText(String(Math.round(xs[k])), m.l + (k + 0.5) * cw, m.t + ih + 13)
    g.textAlign = 'right'
    const ystep = Math.max(1, Math.ceil(nR / Math.max(3, Math.floor(ih / 15))))
    for (let r = 0; r < nR; r += ystep) { const y = m.t + (nR - 1 - r + 0.5) * rh; g.fillStyle = wavelengthToHex(ys[r]); g.fillRect(m.l - 6, y - 3, 3, 6); g.fillStyle = AX; g.fillText(String(ys[r]), m.l - 10, y + 3) }
    g.fillStyle = AXL; g.textAlign = 'right'; g.fillText('emission (nm)', m.l + iw, height - 2); g.textAlign = 'left'; g.fillText('excitation (nm)', 2, 9)
    // colour bar
    const bx = m.l + iw + 12
    for (let i = 0; i < ih; i++) { const t = 1 - i / (ih - 1); const j = Math.round(t * 255) * 3; g.fillStyle = `rgb(${L[j]},${L[j + 1]},${L[j + 2]})`; g.fillRect(bx, m.t + i, 8, 1.2) }
    g.strokeStyle = '#3a4350'; g.strokeRect(bx + 0.5, m.t + 0.5, 8, ih)
    g.fillStyle = AX; g.textAlign = 'left'; g.fillText(fmtTick(hi), bx + 12, m.t + 8); g.fillText(fmtTick(lo), bx + 12, m.t + ih)
    if (unit) { g.fillStyle = AXL; g.fillText(unit, bx - 2, m.t + ih + 14) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [z, xs, ys, cmap, log, w, height, lo, hi, unit])
  const locate = (e: React.MouseEvent) => {
    const r = e.currentTarget.getBoundingClientRect(); const mx = e.clientX - r.left - m.l, my = e.clientY - r.top - m.t
    if (mx < 0 || my < 0 || mx > iw || my > ih) return null
    return { r: nR - 1 - Math.min(nR - 1, Math.floor(my / rh)), c: Math.min(nC - 1, Math.floor(mx / cw)) }
  }
  return (
    <div ref={ref} style={{ position: 'relative', width: '100%' }}>
      <canvas ref={cv} style={{ width: w, height, display: 'block', cursor: onPick ? 'pointer' : 'default' }}
        onMouseMove={(e) => setHov(locate(e))} onMouseLeave={() => setHov(null)}
        onClick={(e) => { const h = locate(e); if (h && onPick && index[h.r][h.c] >= 0) onPick(index[h.r][h.c]) }} />
      <svg width={w} height={height} style={{ position: 'absolute', left: 0, top: 0, pointerEvents: 'none' }}>
        {picks && [...picks.entries()].map(([ch, rank]) => {
          let rr = -1, kk = -1
          for (let r = 0; r < nR && rr < 0; r++) for (let k = 0; k < nC; k++) if (index[r][k] === ch) { rr = r; kk = k; break }
          if (rr < 0) return null
          const x = m.l + kk * cw, y = m.t + (nR - 1 - rr) * rh
          return <g key={ch}><rect x={x} y={y} width={cw} height={rh - 1} fill="none" stroke="#fff" strokeWidth={1.6} /><rect x={x - 1} y={y - 1} width={cw + 2} height={rh + 1} fill="none" stroke="#0b0e12" strokeWidth={0.8} opacity={0.7} />
            {cw > 11 && rh > 11 && <text x={x + cw / 2} y={y + rh / 2 + 3} textAnchor="middle" fontSize="9" fontWeight="700" fill="#fff" stroke="#0b0e12" strokeWidth={2.2} paintOrder="stroke" fontFamily={MONO}>{rank}</text>}</g>
        })}
      </svg>
      {hov && Number.isFinite(z[hov.r]?.[hov.c]) && (() => {
        const ch = index[hov.r][hov.c]; const rank = picks?.get(ch)
        return <div className="plot-tip" style={{ left: Math.min(m.l + (hov.c + 1) * cw + 8, w - 170), top: Math.max(0, m.t + (nR - 1 - hov.r) * rh - 40) }}>
          <b>ex {ys[hov.r]} → em {Math.round(xs[hov.c])} nm</b><div>value<span>{fmtVal(z[hov.r][hov.c])}</span></div>{rank != null && <div>design pick<span>#{rank}</span></div>}</div>
      })()}
    </div>
  )
}

// ================================================================================================ raster
export function MapCanvas({ data, w, h, cmap = 'viridis', vmin, vmax, maxHeight = 200, label, unit }: { data: ArrayLike<number>; w: number; h: number; cmap?: CmapName; vmin?: number; vmax?: number; maxHeight?: number; label?: string; unit?: string }) {
  const cv = useRef<HTMLCanvasElement>(null)
  const [hov, setHov] = useState<{ x: number; y: number; v: number } | null>(null)
  const range = useMemo(() => {
    let lo = Infinity, hi = -Infinity
    for (let i = 0; i < data.length; i++) { const v = data[i]; if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v) } }
    if (!Number.isFinite(lo)) { lo = 0; hi = 1 }
    return { lo: vmin ?? lo, hi: vmax ?? hi }
  }, [data, vmin, vmax])
  useEffect(() => {
    const c = cv.current; if (!c || !w || !h) return
    c.width = w; c.height = h
    const g = c.getContext('2d'); if (!g) return
    const img = g.createImageData(w, h), L = lut(cmap), span = range.hi - range.lo || 1
    for (let i = 0; i < w * h; i++) {
      const v = data[i], o = i * 4
      if (!Number.isFinite(v)) { img.data[o + 3] = 0; continue }
      const k = Math.round(Math.max(0, Math.min(1, (v - range.lo) / span)) * 255) * 3
      img.data[o] = L[k]; img.data[o + 1] = L[k + 1]; img.data[o + 2] = L[k + 2]; img.data[o + 3] = 255
    }
    g.putImageData(img, 0, 0)
  }, [data, w, h, cmap, range])
  return (
    <div style={{ position: 'relative' }}>
      <canvas ref={cv} style={{ display: 'block', width: '100%', maxHeight, objectFit: 'contain', imageRendering: 'pixelated', borderRadius: 6, background: '#06080b', border: '1px solid var(--line)' }} aria-label={label}
        onMouseMove={(e) => { const r = e.currentTarget.getBoundingClientRect(); const s = Math.min(r.width / w, r.height / h); const ox = (r.width - w * s) / 2, oy = (r.height - h * s) / 2; const x = Math.floor((e.clientX - r.left - ox) / s), y = Math.floor((e.clientY - r.top - oy) / s); setHov(x >= 0 && y >= 0 && x < w && y < h ? { x, y, v: data[y * w + x] } : null) }}
        onMouseLeave={() => setHov(null)} />
      <div style={{ display: 'flex', justifyContent: 'space-between', font: '10px var(--mono)', color: 'var(--faint)', paddingTop: 3 }}>
        <span>{fmtVal(range.lo)}</span><span style={{ color: 'var(--dim)' }}>{hov ? `(${hov.x}, ${hov.y}) ${fmtVal(hov.v)}${unit ? ` ${unit}` : ''}` : unit ?? ''}</span><span>{fmtVal(range.hi)}</span>
      </div>
      <div style={{ height: 4, borderRadius: 2, background: `linear-gradient(90deg, ${[0, 0.25, 0.5, 0.75, 1].map((t) => cmapCss(cmap, t)).join(',')})` }} />
    </div>
  )
}
