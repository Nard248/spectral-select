import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { wavelengthToHex } from '../lib/color'
import { lut, type CmapName } from './colormaps'

export function useWidth<T extends HTMLElement>(): [React.RefObject<T | null>, number] {
  const ref = useRef<T>(null)
  const [w, setW] = useState(300)
  useLayoutEffect(() => {
    const el = ref.current; if (!el) return
    const ro = new ResizeObserver(() => setW(Math.max(80, Math.floor(el.clientWidth))))
    ro.observe(el); setW(Math.max(80, Math.floor(el.clientWidth)))
    return () => ro.disconnect()
  }, [])
  return [ref, w]
}

function niceTicks(lo: number, hi: number, n = 5): number[] {
  const span = hi - lo || 1
  const raw = span / n, mag = 10 ** Math.floor(Math.log10(raw))
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? mag * 10
  const out: number[] = []
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10))
  return out
}
const short = (v: number) => (Math.abs(v) >= 10000 || (Math.abs(v) < 0.01 && v !== 0) ? v.toExponential(0) : String(+v.toPrecision(3)))

export interface Series { name?: string; x: ArrayLike<number>; y: ArrayLike<number>; color: string; fill?: boolean; dash?: string; width?: number; opacity?: number }
export interface Marker { x: number; color?: string; label?: string; dash?: string }

interface LineProps {
  series: Series[]
  height?: number
  xDomain?: [number, number]
  yDomain?: [number, number]
  markers?: Marker[]
  xLabel?: string
  yLabel?: string
  logY?: boolean
  strip?: boolean   // wavelength-coloured strip under the x axis
  shade?: { x0: number; x1: number; color: string }[]
  onClickX?: (x: number) => void
  legend?: boolean
  yTicks?: number
}

export function LineChart({ series, height = 120, xDomain, yDomain, markers = [], xLabel, yLabel, logY, strip, shade = [], onClickX, legend, yTicks = 3 }: LineProps) {
  const [ref, w] = useWidth<HTMLDivElement>()
  const m = { l: 34, r: 8, t: 6, b: strip ? 30 : 22 }
  const iw = w - m.l - m.r, ih = height - m.t - m.b
  const [hover, setHover] = useState<number | null>(null)
  const dom = useMemo(() => {
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity
    for (const s of series) for (let i = 0; i < s.x.length; i++) {
      x0 = Math.min(x0, s.x[i]); x1 = Math.max(x1, s.x[i])
      const y = logY ? Math.log10(Math.max(s.y[i], 1e-12)) : s.y[i]
      y0 = Math.min(y0, y); y1 = Math.max(y1, y)
    }
    if (!Number.isFinite(x0)) { x0 = 0; x1 = 1; y0 = 0; y1 = 1 }
    const xd = xDomain ?? [x0, x1]
    let yd = yDomain ?? [logY ? y0 : Math.min(0, y0), y1 * (logY ? 1 : 1.08) || 1]
    if (yd[1] === yd[0]) yd = [yd[0], yd[0] + 1]
    return { x: xd, y: yd }
  }, [series, xDomain, yDomain, logY])
  const sx = (v: number) => m.l + ((v - dom.x[0]) / (dom.x[1] - dom.x[0] || 1)) * iw
  const sy = (v: number) => m.t + ih - (((logY ? Math.log10(Math.max(v, 1e-12)) : v) - dom.y[0]) / (dom.y[1] - dom.y[0] || 1)) * ih
  const xt = niceTicks(dom.x[0], dom.x[1], Math.max(3, Math.floor(iw / 70)))
  const yt = logY ? [] : niceTicks(dom.y[0], dom.y[1], yTicks)
  const path = (s: Series, close: boolean) => {
    let d = ''
    for (let i = 0; i < s.x.length; i++) d += `${i ? 'L' : 'M'}${sx(s.x[i]).toFixed(1)},${sy(s.y[i]).toFixed(1)}`
    if (close && s.x.length) d += `L${sx(s.x[s.x.length - 1]).toFixed(1)},${sy(logY ? 10 ** dom.y[0] : Math.max(0, dom.y[0])).toFixed(1)}L${sx(s.x[0]).toFixed(1)},${sy(logY ? 10 ** dom.y[0] : Math.max(0, dom.y[0])).toFixed(1)}Z`
    return d
  }
  const gid = useRef(`g${Math.random().toString(36).slice(2, 8)}`).current
  const stripStops = useMemo(() => {
    const n = 24; return Array.from({ length: n + 1 }, (_, i) => ({ o: i / n, c: wavelengthToHex(dom.x[0] + ((dom.x[1] - dom.x[0]) * i) / n) }))
  }, [dom.x])
  return (
    <div ref={ref} style={{ width: '100%' }}>
      <svg width={w} height={height} style={{ display: 'block', overflow: 'visible' }}
        onMouseMove={(e) => { const r = e.currentTarget.getBoundingClientRect(); setHover(dom.x[0] + ((e.clientX - r.left - m.l) / iw) * (dom.x[1] - dom.x[0])) }}
        onMouseLeave={() => setHover(null)}
        onClick={(e) => { if (!onClickX) return; const r = e.currentTarget.getBoundingClientRect(); onClickX(dom.x[0] + ((e.clientX - r.left - m.l) / iw) * (dom.x[1] - dom.x[0])) }}>
        <defs>
          <linearGradient id={`${gid}s`} x1="0" x2="1">{stripStops.map((s) => <stop key={s.o} offset={s.o} stopColor={s.c} />)}</linearGradient>
          {series.map((s, i) => s.fill && <linearGradient key={i} id={`${gid}f${i}`} x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor={s.color} stopOpacity=".38" /><stop offset="1" stopColor={s.color} stopOpacity=".02" /></linearGradient>)}
        </defs>
        {shade.map((b, i) => <rect key={i} x={sx(b.x0)} width={Math.max(0, sx(b.x1) - sx(b.x0))} y={m.t} height={ih} fill={b.color} opacity={0.14} />)}
        {yt.map((v) => <g key={v}><line x1={m.l} x2={w - m.r} y1={sy(v)} y2={sy(v)} stroke="rgba(255,255,255,.07)" /><text x={m.l - 5} y={sy(v) + 3} textAnchor="end" fontSize="9.5" fill="#6b7788" fontFamily="var(--mono)">{short(v)}</text></g>)}
        {xt.map((v) => <g key={v}><line x1={sx(v)} x2={sx(v)} y1={m.t + ih} y2={m.t + ih + 3} stroke="#4b5666" /><text x={sx(v)} y={m.t + ih + 13} textAnchor="middle" fontSize="9.5" fill="#6b7788" fontFamily="var(--mono)">{short(v)}</text></g>)}
        <line x1={m.l} x2={w - m.r} y1={m.t + ih} y2={m.t + ih} stroke="#3a4350" />
        {strip && <rect x={m.l} y={m.t + ih + 17} width={iw} height={4} rx={2} fill={`url(#${gid}s)`} opacity={0.9} />}
        {series.map((s, i) => s.fill && <path key={`f${i}`} d={path(s, true)} fill={`url(#${gid}f${i})`} />)}
        {series.map((s, i) => <path key={i} d={path(s, false)} fill="none" stroke={s.color} strokeWidth={s.width ?? 1.6} strokeDasharray={s.dash} opacity={s.opacity ?? 1} strokeLinejoin="round" />)}
        {markers.map((mk, i) => mk.x >= dom.x[0] && mk.x <= dom.x[1] && (
          <g key={i}><line x1={sx(mk.x)} x2={sx(mk.x)} y1={m.t} y2={m.t + ih} stroke={mk.color ?? '#fff'} strokeWidth={1.3} strokeDasharray={mk.dash ?? '3 3'} opacity={0.85} />
            {mk.label && <text x={sx(mk.x) + 3} y={m.t + 9} fontSize="9" fill={mk.color ?? '#fff'} fontFamily="var(--mono)">{mk.label}</text>}</g>
        ))}
        {hover != null && hover >= dom.x[0] && hover <= dom.x[1] && <g><line x1={sx(hover)} x2={sx(hover)} y1={m.t} y2={m.t + ih} stroke="rgba(255,255,255,.3)" /><text x={Math.min(sx(hover) + 4, w - 40)} y={m.t + ih - 4} fontSize="9.5" fill="#cfd8e3" fontFamily="var(--mono)">{hover.toFixed(0)}</text></g>}
        {xLabel && <text x={w - m.r} y={height - (strip ? 0 : 2)} textAnchor="end" fontSize="9.5" fill="#566174">{xLabel}</text>}
        {yLabel && <text x={m.l + 4} y={m.t + 8} fontSize="9.5" fill="#566174">{yLabel}</text>}
      </svg>
      {legend && <div className="legend" style={{ padding: '4px 0 0' }}>{series.filter((s) => s.name).map((s, i) => <span key={i}><i style={{ background: s.color }} />{s.name}</span>)}</div>}
    </div>
  )
}

/* ---------- stacked area ---------- */
export interface Layer { name: string; color: string; y: ArrayLike<number> }
export function StackedArea({ x, layers, height = 150, overlay, xLabel, yLabel, yUnit, scaleBar }: { x: number[]; layers: Layer[]; height?: number; overlay?: { y: ArrayLike<number>; color: string; name: string }; xLabel?: string; yLabel?: string; yUnit?: string; scaleBar?: boolean }) {
  const [ref, w] = useWidth<HTMLDivElement>()
  const [hi, setHi] = useState<number | null>(null)
  const m = { l: 40, r: 8, t: 6, b: 28 }
  const iw = w - m.l - m.r, ih = height - m.t - m.b
  const n = x.length
  const tot = Array.from({ length: n }, (_, i) => layers.reduce((a, l) => a + Math.max(0, l.y[i]), 0))
  const ymax = Math.max(1e-9, ...tot, ...(overlay ? Array.from(overlay.y) : [0])) * 1.05
  const sx = (i: number) => m.l + (n > 1 ? (i / (n - 1)) * iw : 0)
  const sy = (v: number) => m.t + ih - (v / ymax) * ih
  const acc = new Array<number>(n).fill(0)
  const polys = layers.map((l) => {
    const lo = acc.slice()
    for (let i = 0; i < n; i++) acc[i] += Math.max(0, l.y[i])
    const up = acc.slice()
    let d = ''
    for (let i = 0; i < n; i++) d += `${i ? 'L' : 'M'}${sx(i).toFixed(1)},${sy(up[i]).toFixed(1)}`
    for (let i = n - 1; i >= 0; i--) d += `L${sx(i).toFixed(1)},${sy(lo[i]).toFixed(1)}`
    return d + 'Z'
  })
  const xt = niceTicks(x[0], x[n - 1], Math.max(3, Math.floor(iw / 70)))
  const xs = (v: number) => m.l + ((v - x[0]) / (x[n - 1] - x[0] || 1)) * iw
  const yt = niceTicks(0, ymax, 3)
  return (
    <div ref={ref}>
      <svg width={w} height={height} style={{ display: 'block' }}>
        {yt.map((v) => <g key={v}><line x1={m.l} x2={w - m.r} y1={sy(v)} y2={sy(v)} stroke="rgba(255,255,255,.07)" /><text x={m.l - 5} y={sy(v) + 3} textAnchor="end" fontSize="9.5" fill="#6b7788" fontFamily="var(--mono)">{short(v)}</text></g>)}
        {xt.map((v) => <g key={v}><text x={xs(v)} y={m.t + ih + 13} textAnchor="middle" fontSize="9.5" fill="#6b7788" fontFamily="var(--mono)">{short(v)}</text></g>)}
        {polys.map((d, i) => <path key={i} d={d} fill={layers[i].color} opacity={hi == null || hi === i ? 0.88 : 0.25} stroke="rgba(0,0,0,.35)" strokeWidth={0.6}
          onMouseEnter={() => setHi(i)} onMouseLeave={() => setHi(null)} />)}
        {overlay && <path d={Array.from(overlay.y).map((v, i) => `${i ? 'L' : 'M'}${sx(i).toFixed(1)},${sy(v).toFixed(1)}`).join('')} fill="none" stroke={overlay.color} strokeWidth={1.4} strokeDasharray="4 3" />}
        <line x1={m.l} x2={w - m.r} y1={m.t + ih} y2={m.t + ih} stroke="#3a4350" />
        {xLabel && <text x={w - m.r} y={height - 2} textAnchor="end" fontSize="9.5" fill="#566174">{xLabel}</text>}
        {yLabel && <text x={m.l + 4} y={m.t + 8} fontSize="9.5" fill="#566174">{yLabel}{yUnit ? ` · ${yUnit}` : ''}</text>}
        {scaleBar && yt.length > 1 && (() => {
          // vertical scale bar = one major tick step in data units
          const step = yt[1] - yt[0], bx = w - m.r - 10
          return <g><line x1={bx} x2={bx} y1={sy(step)} y2={sy(0)} stroke="#e8eef7" strokeWidth={2} />
            <line x1={bx - 3} x2={bx + 3} y1={sy(step)} y2={sy(step)} stroke="#e8eef7" /><line x1={bx - 3} x2={bx + 3} y1={sy(0)} y2={sy(0)} stroke="#e8eef7" />
            <text x={bx - 6} y={(sy(step) + sy(0)) / 2 + 3} textAnchor="end" fontSize="9.5" fill="#e8eef7" fontFamily="var(--mono)">{short(step)}</text></g>
        })()}
      </svg>
      <div className="legend">{layers.map((l, i) => <span key={i} style={{ opacity: hi == null || hi === i ? 1 : 0.4 }} onMouseEnter={() => setHi(i)} onMouseLeave={() => setHi(null)}><i style={{ background: l.color }} />{l.name}</span>)}
        {overlay && <span><i style={{ background: overlay.color, borderRadius: 0, height: 2, verticalAlign: 3 }} />{overlay.name}</span>}</div>
    </div>
  )
}

/* ---------- heatmap (EEM) ---------- */
export function Heatmap({ data, xs, ys, cmap = 'magma', height = 150, xLabel, yLabel, marks, log }: { data: number[][]; xs: number[]; ys: number[]; cmap?: CmapName; height?: number; xLabel?: string; yLabel?: string; marks?: { x?: number; y?: number; color: string }[]; log?: boolean }) {
  const [ref, w] = useWidth<HTMLDivElement>()
  const cv = useRef<HTMLCanvasElement>(null)
  const m = { l: 36, r: 6, t: 4, b: 22 }
  const iw = w - m.l - m.r, ih = height - m.t - m.b
  useEffect(() => {
    const c = cv.current; if (!c) return
    const dpr = window.devicePixelRatio || 1
    c.width = w * dpr; c.height = height * dpr
    const g = c.getContext('2d'); if (!g) return
    g.scale(dpr, dpr); g.clearRect(0, 0, w, height)
    const rows = data.length, cols = data[0]?.length ?? 0
    if (!rows || !cols) return
    let mx = 1e-12; for (const r of data) for (const v of r) mx = Math.max(mx, v)
    const off = document.createElement('canvas'); off.width = cols; off.height = rows
    const og = off.getContext('2d')!; const img = og.createImageData(cols, rows); const L = lut(cmap)
    for (let r = 0; r < rows; r++) for (let k = 0; k < cols; k++) {
      let t = data[r][k] / mx; if (log) t = Math.log10(1 + 99 * t) / 2
      const i = Math.round(Math.min(1, Math.max(0, t)) * 255) * 3, o = (( rows - 1 - r) * cols + k) * 4   // y up: excitation increases upward
      img.data[o] = L[i]; img.data[o + 1] = L[i + 1]; img.data[o + 2] = L[i + 2]; img.data[o + 3] = 255
    }
    og.putImageData(img, 0, 0)
    g.imageSmoothingEnabled = true
    g.drawImage(off, m.l, m.t, iw, ih)
    g.strokeStyle = '#3a4350'; g.strokeRect(m.l + 0.5, m.t + 0.5, iw, ih)
    g.fillStyle = '#6b7788'; g.font = '9.5px "JetBrains Mono Variable", monospace'
    const x0 = xs[0], x1 = xs[xs.length - 1], y0 = ys[0], y1 = ys[ys.length - 1]
    for (const v of niceTicks(x0, x1, Math.max(3, Math.floor(iw / 70)))) { const px = m.l + ((v - x0) / (x1 - x0 || 1)) * iw; g.textAlign = 'center'; g.fillText(short(v), px, m.t + ih + 12) }
    for (const v of niceTicks(y0, y1, 4)) { const py = m.t + ih - ((v - y0) / (y1 - y0 || 1)) * ih; g.textAlign = 'right'; g.fillText(short(v), m.l - 4, py + 3) }
    g.fillStyle = '#566174'; g.textAlign = 'right'
    if (xLabel) g.fillText(xLabel, w - m.r, height - 2)
    if (yLabel) { g.textAlign = 'left'; g.fillText(yLabel, 2, m.t + 8) }
    for (const mk of marks ?? []) {
      g.strokeStyle = mk.color; g.setLineDash([3, 3]); g.beginPath()
      if (mk.x != null) { const px = m.l + ((mk.x - x0) / (x1 - x0 || 1)) * iw; g.moveTo(px, m.t); g.lineTo(px, m.t + ih) }
      if (mk.y != null) { const py = m.t + ih - ((mk.y - y0) / (y1 - y0 || 1)) * ih; g.moveTo(m.l, py); g.lineTo(m.l + iw, py) }
      g.stroke(); g.setLineDash([])
    }
  }, [data, xs, ys, cmap, w, height, iw, ih, xLabel, yLabel, marks, log, m.l, m.t])
  return <div ref={ref}><canvas ref={cv} style={{ width: w, height, display: 'block' }} /></div>
}
