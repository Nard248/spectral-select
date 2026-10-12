/** Analysis plot primitives (added for the v1.1 workspace): an XY chart with ±std bands, a secondary axis, scatter, hover
 *  read-out and click handlers, and an EEM heatmap with contour lines, log scale and a diverging mode for differences. */
import { useEffect, useMemo, useRef, useState } from 'react'
import { lut, type CmapName } from './colormaps'
import { useWidth } from './charts'
import { wavelengthToHex } from '../lib/color'

export function niceTicks(lo: number, hi: number, n = 5): number[] {
  const span = hi - lo || 1
  const raw = span / Math.max(1, n), mag = 10 ** Math.floor(Math.log10(raw))
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? mag * 10
  const out: number[] = []
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-6; v += step) out.push(+v.toFixed(10))
  return out
}
export const fmtTick = (v: number) => {
  if (v === 0) return '0'
  const a = Math.abs(v)
  if (a >= 1e5 || a < 1e-2) return v.toExponential(0).replace('e+', 'e')
  return String(+v.toPrecision(3))
}
export const fmtVal = (v: number) => {
  if (!Number.isFinite(v)) return '—'
  if (v === 0) return '0'
  const a = Math.abs(v)
  return a >= 1e5 || a < 1e-3 ? v.toExponential(2).replace('e+', 'e') : String(+v.toPrecision(4))
}

const AX = '#6b7788', AXL = '#566174', GRID = 'rgba(255,255,255,.07)'

/* ============================================================ XY chart ============================================================ */
export interface XYSeries {
  name: string
  x: ArrayLike<number>
  y: ArrayLike<number>
  color: string
  std?: ArrayLike<number> | null
  dash?: string
  width?: number
  opacity?: number
  axis?: 'l' | 'r'
  points?: boolean
  fill?: boolean
}
export interface XYScatter { x: ArrayLike<number>; y: ArrayLike<number>; color: string; r?: number; opacity?: number }
export interface XYMarker { x: number; label?: string; color?: string; dash?: string }

interface XYProps {
  series: XYSeries[]
  scatter?: XYScatter[]
  height?: number
  xLabel?: string
  yLabel?: string
  yLabelR?: string
  xDomain?: [number, number]
  yDomain?: [number, number]
  yDomainR?: [number, number]
  zeroBase?: boolean
  logY?: boolean
  markers?: XYMarker[]
  onClickX?: (x: number) => void
  onClickPoint?: (series: number, index: number) => void
  legend?: boolean
  tipTitle?: (x: number) => string
  xFmt?: (x: number) => string
  /** wavelength-coloured strip under the x axis (nm axes) */
  strip?: boolean
}

export function XYChart({ series, scatter = [], height = 180, xLabel, yLabel, yLabelR, xDomain, yDomain, yDomainR, zeroBase = true, logY, markers = [], onClickX, onClickPoint, legend, tipTitle, xFmt, strip }: XYProps) {
  const [ref, w] = useWidth<HTMLDivElement>()
  const hasR = series.some((s) => s.axis === 'r')
  const m = { l: 46, r: hasR ? 46 : 10, t: yLabel || yLabelR ? 18 : 10, b: strip ? 34 : 26 }
  const iw = Math.max(10, w - m.l - m.r), ih = Math.max(10, height - m.t - m.b)
  const [hover, setHover] = useState<{ x: number; px: number; py: number } | null>(null)

  const dom = useMemo(() => {
    const ext = (pred: (s: XYSeries) => boolean, sc: boolean) => {
      let lo = Infinity, hi = -Infinity
      for (const s of series) if (pred(s)) for (let i = 0; i < s.y.length; i++) {
        const v = s.y[i], e = s.std?.[i] ?? 0
        if (!Number.isFinite(v)) continue
        lo = Math.min(lo, v - (Number.isFinite(e) ? e : 0)); hi = Math.max(hi, v + (Number.isFinite(e) ? e : 0))
      }
      if (sc) for (const q of scatter) for (let i = 0; i < q.y.length; i++) if (Number.isFinite(q.y[i])) { lo = Math.min(lo, q.y[i]); hi = Math.max(hi, q.y[i]) }
      return [lo, hi] as [number, number]
    }
    let x0 = Infinity, x1 = -Infinity
    for (const s of series) for (let i = 0; i < s.x.length; i++) if (Number.isFinite(s.x[i])) { x0 = Math.min(x0, s.x[i]); x1 = Math.max(x1, s.x[i]) }
    for (const q of scatter) for (let i = 0; i < q.x.length; i++) if (Number.isFinite(q.x[i])) { x0 = Math.min(x0, q.x[i]); x1 = Math.max(x1, q.x[i]) }
    if (!Number.isFinite(x0)) { x0 = 0; x1 = 1 }
    if (x0 === x1) { x0 -= 0.5; x1 += 0.5 }
    const fix = (e: [number, number], forced?: [number, number], zero = zeroBase): [number, number] => {
      if (forced) return forced
      let lo = e[0]
      const hi = e[1]
      if (!Number.isFinite(lo)) return [0, 1]
      if (logY) return [Math.log10(Math.max(lo, hi * 1e-4, 1e-12)), Math.log10(Math.max(hi, 1e-12)) + 0.05]
      if (zero && lo > 0) lo = 0
      const pad = (hi - lo || Math.abs(hi) || 1) * 0.06
      return [zero && lo === 0 ? 0 : lo - pad, hi + pad]
    }
    return { x: xDomain ?? ([x0, x1] as [number, number]), yl: fix(ext((s) => s.axis !== 'r', true), yDomain), yr: fix(ext((s) => s.axis === 'r', false), yDomainR, false) }
  }, [series, scatter, xDomain, yDomain, yDomainR, zeroBase, logY])

  const sx = (v: number) => m.l + ((v - dom.x[0]) / (dom.x[1] - dom.x[0] || 1)) * iw
  const dy = (axis: 'l' | 'r' | undefined) => (axis === 'r' ? dom.yr : dom.yl)
  const sy = (v: number, axis?: 'l' | 'r') => {
    const d = dy(axis)
    const t = logY && axis !== 'r' ? Math.log10(Math.max(v, 1e-12)) : v
    return m.t + ih - ((t - d[0]) / (d[1] - d[0] || 1)) * ih
  }
  const xt = niceTicks(dom.x[0], dom.x[1], Math.max(3, Math.floor(iw / 64)))
  const ylt = logY ? [] : niceTicks(dom.yl[0], dom.yl[1], Math.max(2, Math.floor(ih / 38)))
  const yrt = hasR ? niceTicks(dom.yr[0], dom.yr[1], Math.max(2, Math.floor(ih / 38))) : []
  const gid = useRef(`x${Math.random().toString(36).slice(2, 8)}`).current

  const path = (s: XYSeries) => {
    let d = '', pen = false
    for (let i = 0; i < s.x.length; i++) {
      if (!Number.isFinite(s.y[i]) || !Number.isFinite(s.x[i])) { pen = false; continue }
      d += `${pen ? 'L' : 'M'}${sx(s.x[i]).toFixed(1)},${sy(s.y[i], s.axis).toFixed(1)}`; pen = true
    }
    return d
  }
  const bandPath = (s: XYSeries) => {
    if (!s.std) return ''
    const top: string[] = [], bot: string[] = []
    for (let i = 0; i < s.x.length; i++) {
      const e = s.std[i]
      if (!Number.isFinite(s.y[i]) || !Number.isFinite(e)) continue
      top.push(`${sx(s.x[i]).toFixed(1)},${sy(s.y[i] + e, s.axis).toFixed(1)}`)
      bot.push(`${sx(s.x[i]).toFixed(1)},${sy(s.y[i] - e, s.axis).toFixed(1)}`)
    }
    return top.length > 1 ? `M${top.join('L')}L${bot.reverse().join('L')}Z` : ''
  }
  const areaPath = (s: XYSeries) => {
    const d = path(s)
    if (!d) return ''
    const base = sy(Math.max(dy(s.axis)[0], 0), s.axis)
    const first = s.x[0], last = s.x[s.x.length - 1]
    return `${d}L${sx(last).toFixed(1)},${base.toFixed(1)}L${sx(first).toFixed(1)},${base.toFixed(1)}Z`
  }

  const tip = hover && (() => {
    const rows: { name: string; color: string; v: number; e?: number }[] = []
    let xs = hover.x
    const ref0 = series.find((s) => s.x.length)
    if (ref0) { // snap to the nearest sample of the first series
      let bi = 0
      for (let i = 1; i < ref0.x.length; i++) if (Math.abs(ref0.x[i] - hover.x) < Math.abs(ref0.x[bi] - hover.x)) bi = i
      xs = ref0.x[bi]
    }
    for (const s of series) {
      let bi = 0
      for (let i = 1; i < s.x.length; i++) if (Math.abs(s.x[i] - xs) < Math.abs(s.x[bi] - xs)) bi = i
      if (s.x.length && Number.isFinite(s.y[bi])) rows.push({ name: s.name, color: s.color, v: s.y[bi], e: s.std?.[bi] })
    }
    return { xs, rows }
  })()

  return (
    <div ref={ref} style={{ width: '100%', position: 'relative' }}>
      <svg width={w} height={height} style={{ display: 'block', overflow: 'visible', cursor: onClickX || onClickPoint ? 'crosshair' : 'default' }}
        onMouseMove={(e) => { const r = e.currentTarget.getBoundingClientRect(); const px = e.clientX - r.left; if (px < m.l || px > m.l + iw) { setHover(null); return } setHover({ x: dom.x[0] + ((px - m.l) / iw) * (dom.x[1] - dom.x[0]), px, py: e.clientY - r.top }) }}
        onMouseLeave={() => setHover(null)}
        onClick={(e) => {
          const r = e.currentTarget.getBoundingClientRect(); const px = e.clientX - r.left
          if (px < m.l || px > m.l + iw) return
          const x = dom.x[0] + ((px - m.l) / iw) * (dom.x[1] - dom.x[0])
          if (onClickPoint) {
            let best: [number, number, number] | null = null
            series.forEach((s, si) => { for (let i = 0; i < s.x.length; i++) { const d = Math.abs(sx(s.x[i]) - px) + Math.abs(sy(s.y[i], s.axis) - (e.clientY - r.top)) * 0.4; if (!best || d < best[2]) best = [si, i, d] } })
            if (best && (best as [number, number, number])[2] < 26) onClickPoint((best as [number, number, number])[0], (best as [number, number, number])[1])
          } else onClickX?.(x)
        }}>
        <defs>
          {strip && <linearGradient id={`${gid}s`} x1="0" x2="1">{Array.from({ length: 25 }, (_, i) => <stop key={i} offset={i / 24} stopColor={wavelengthToHex(dom.x[0] + ((dom.x[1] - dom.x[0]) * i) / 24)} />)}</linearGradient>}
          {series.map((s, i) => s.fill && <linearGradient key={i} id={`${gid}f${i}`} x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor={s.color} stopOpacity=".34" /><stop offset="1" stopColor={s.color} stopOpacity=".02" /></linearGradient>)}
          <clipPath id={`${gid}c`}><rect x={m.l} y={m.t - 2} width={iw} height={ih + 4} /></clipPath>
        </defs>
        {ylt.map((v) => <g key={v}><line x1={m.l} x2={m.l + iw} y1={sy(v)} y2={sy(v)} stroke={GRID} /><text x={m.l - 6} y={sy(v) + 3} textAnchor="end" fontSize="9.5" fill={AX} fontFamily="var(--mono)">{fmtTick(v)}</text></g>)}
        {logY && niceTicks(dom.yl[0], dom.yl[1], 4).map((p) => <g key={p}><line x1={m.l} x2={m.l + iw} y1={sy(10 ** p)} y2={sy(10 ** p)} stroke={GRID} /><text x={m.l - 6} y={sy(10 ** p) + 3} textAnchor="end" fontSize="9.5" fill={AX} fontFamily="var(--mono)">{fmtTick(10 ** p)}</text></g>)}
        {yrt.map((v) => <text key={v} x={m.l + iw + 6} y={sy(v, 'r') + 3} fontSize="9.5" fill={AX} fontFamily="var(--mono)">{fmtTick(v)}</text>)}
        {xt.map((v) => <g key={v}><line x1={sx(v)} x2={sx(v)} y1={m.t + ih} y2={m.t + ih + 3} stroke="#4b5666" /><text x={sx(v)} y={m.t + ih + 14} textAnchor="middle" fontSize="9.5" fill={AX} fontFamily="var(--mono)">{xFmt ? xFmt(v) : fmtTick(v)}</text></g>)}
        <line x1={m.l} x2={m.l + iw} y1={m.t + ih} y2={m.t + ih} stroke="#3a4350" />
        <line x1={m.l} x2={m.l} y1={m.t} y2={m.t + ih} stroke="#2a313b" />
        {hasR && <line x1={m.l + iw} x2={m.l + iw} y1={m.t} y2={m.t + ih} stroke="#2a313b" />}
        {strip && <rect x={m.l} y={m.t + ih + 19} width={iw} height={4} rx={2} fill={`url(#${gid}s)`} opacity={0.9} />}
        <g clipPath={`url(#${gid}c)`}>
          {scatter.map((q, k) => <g key={k}>{Array.from({ length: q.x.length }, (_, i) => Number.isFinite(q.x[i]) && Number.isFinite(q.y[i]) && <circle key={i} cx={sx(q.x[i])} cy={sy(q.y[i])} r={q.r ?? 1.5} fill={q.color} opacity={q.opacity ?? 0.2} />)}</g>)}
          {series.map((s, i) => s.std && <path key={`b${i}`} d={bandPath(s)} fill={s.color} opacity={0.16} />)}
          {series.map((s, i) => s.fill && <path key={`f${i}`} d={areaPath(s)} fill={`url(#${gid}f${i})`} />)}
          {markers.map((mk, i) => mk.x >= dom.x[0] && mk.x <= dom.x[1] && (
            <g key={`m${i}`}><line x1={sx(mk.x)} x2={sx(mk.x)} y1={m.t} y2={m.t + ih} stroke={mk.color ?? '#fff'} strokeWidth={1.2} strokeDasharray={mk.dash ?? '3 3'} opacity={0.8} />
              {mk.label && <text x={sx(mk.x) + 3} y={m.t + 9} fontSize="9" fill={mk.color ?? '#fff'} fontFamily="var(--mono)">{mk.label}</text>}</g>))}
          {series.map((s, i) => <path key={i} d={path(s)} fill="none" stroke={s.color} strokeWidth={s.width ?? 1.7} strokeDasharray={s.dash} opacity={s.opacity ?? 1} strokeLinejoin="round" strokeLinecap="round" />)}
          {series.map((s, i) => s.points && Array.from({ length: s.x.length }, (_, k) => Number.isFinite(s.y[k]) && (
            <circle key={`${i}-${k}`} cx={sx(s.x[k])} cy={sy(s.y[k], s.axis)} r={3.6} fill="#0b0e12" stroke={s.color} strokeWidth={1.8} style={{ cursor: onClickPoint ? 'pointer' : undefined }} />)))}
          {hover && <line x1={sx(tip ? tip.xs : hover.x)} x2={sx(tip ? tip.xs : hover.x)} y1={m.t} y2={m.t + ih} stroke="rgba(255,255,255,.28)" />}
        </g>
        {xLabel && <text x={m.l + iw} y={height - (strip ? 0 : 2)} textAnchor="end" fontSize="9.5" fill={AXL}>{xLabel}</text>}
        {yLabel && <text x={m.l - 40} y={9} fontSize="9.5" fill={AXL}>{yLabel}</text>}
        {yLabelR && <text x={m.l + iw + 40} y={9} textAnchor="end" fontSize="9.5" fill={AXL}>{yLabelR}</text>}
      </svg>
      {tip && tip.rows.length > 0 && hover && (
        <div className="plot-tip" style={{ left: Math.min(hover.px + 12, w - 170), top: Math.max(0, hover.py - 10 - tip.rows.length * 8) }}>
          <b>{tipTitle ? tipTitle(tip.xs) : `${xLabel ? xLabel.split(' ')[0] + ' ' : ''}${fmtVal(tip.xs)}`}</b>
          {tip.rows.slice(0, 7).map((r, i) => <div key={i}><i style={{ background: r.color }} />{r.name}<span>{fmtVal(r.v)}{r.e != null && Number.isFinite(r.e) ? ` ± ${fmtVal(r.e)}` : ''}</span></div>)}
        </div>)}
      {legend && <div className="legend" style={{ padding: '4px 0 0' }}>{series.filter((s) => s.name).map((s, i) => <span key={i}><i style={{ background: s.color, ...(s.dash ? { height: 2, borderRadius: 0 } : {}) }} />{s.name}</span>)}</div>}
    </div>
  )
}

/* ============================================================ EEM heatmap ============================================================ */
export type HeatCmap = CmapName | 'diverging'

let divLut: Uint8ClampedArray | null = null
/** Dark-friendly diverging map: blue ← neutral → orange; index 0..255 (128 = zero). */
function divergingLut(): Uint8ClampedArray {
  if (divLut) return divLut
  const out = new Uint8ClampedArray(256 * 3)
  const neg: [number, number, number] = [74, 160, 255], mid: [number, number, number] = [22, 26, 33], pos: [number, number, number] = [255, 122, 61]
  for (let i = 0; i < 256; i++) {
    const t = (i - 127.5) / 127.5
    const a = t < 0 ? neg : pos, k = Math.abs(t) ** 0.85
    for (let c = 0; c < 3; c++) out[i * 3 + c] = mid[c] + (a[c] - mid[c]) * k
  }
  divLut = out
  return out
}

interface HeatProps {
  /** [n_ex][n_em] — NaN where there is no data */
  z: number[][]
  xs: number[]
  ys: number[]
  vmin?: number
  vmax?: number
  cmap?: HeatCmap
  log?: boolean
  contours?: boolean
  height?: number
  unit?: string
  xLabel?: string
  yLabel?: string
  marks?: { ex?: number | null; em?: number | null }
  onPick?: (ex: number, em: number) => void
}

export function EemHeatmap({ z, xs, ys, vmin, vmax, cmap = 'magma', log, contours, height = 230, unit, xLabel = 'emission (nm)', yLabel = 'excitation (nm)', marks, onPick }: HeatProps) {
  const [ref, w] = useWidth<HTMLDivElement>()
  const cv = useRef<HTMLCanvasElement>(null)
  const [hov, setHov] = useState<{ r: number; c: number; px: number; py: number } | null>(null)
  const m = { l: 44, r: 54, t: 16, b: 28 }
  const iw = Math.max(10, w - m.l - m.r), ih = Math.max(10, height - m.t - m.b)
  const diverging = cmap === 'diverging'

  const range = useMemo(() => {
    let lo = Infinity, hi = -Infinity
    for (const r of z) for (const v of r) if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v) }
    if (!Number.isFinite(lo)) { lo = 0; hi = 1 }
    if (diverging) { const a = Math.max(Math.abs(lo), Math.abs(hi), 1e-12); return { lo: vmin ?? -a, hi: vmax ?? a } }
    if (lo < 0 && -lo < 0.02 * hi) lo = 0   // numerical dust below zero (e.g. dark-subtracted noise) should not move the colour-bar floor
    return { lo: vmin ?? Math.min(0, lo), hi: vmax ?? hi }
  }, [z, vmin, vmax, diverging])

  // row geometry (y axis = excitation nm, nearest-row bands because excitation is sampled sparsely)
  const rows = useMemo(() => {
    const n = ys.length
    const y0 = ys[0], y1 = ys[n - 1]
    const lo = n > 1 ? y0 - (ys[1] - y0) / 2 : y0 - 10, hi = n > 1 ? y1 + (y1 - ys[n - 2]) / 2 : y1 + 10
    const edges = ys.map((_, i) => (i === 0 ? lo : (ys[i - 1] + ys[i]) / 2))
    edges.push(hi)
    return { edges, lo, hi }
  }, [ys])
  const x0 = xs[0], x1 = xs[xs.length - 1]
  const px = (v: number) => m.l + ((v - x0) / (x1 - x0 || 1)) * iw
  const py = (v: number) => m.t + ih - ((v - rows.lo) / (rows.hi - rows.lo || 1)) * ih

  const tnorm = (v: number) => {
    if (diverging) { const a = Math.max(Math.abs(range.lo), Math.abs(range.hi)) || 1; return 0.5 + 0.5 * Math.max(-1, Math.min(1, v / a)) }
    let t = (v - range.lo) / (range.hi - range.lo || 1)
    t = Math.min(1, Math.max(0, t))
    return log ? Math.log10(1 + 99 * t) / 2 : t
  }

  useEffect(() => {
    const c = cv.current; if (!c) return
    const dpr = window.devicePixelRatio || 1
    c.width = Math.round(w * dpr); c.height = Math.round(height * dpr)
    const g = c.getContext('2d'); if (!g) return
    g.setTransform(dpr, 0, 0, dpr, 0, 0)
    g.clearRect(0, 0, w, height)
    const L = diverging ? divergingLut() : lut(cmap as CmapName)
    const nR = z.length, nC = xs.length
    // rasterise cells to an offscreen bitmap (x: smooth interpolation, y: nearest row)
    const off = document.createElement('canvas'); off.width = nC; off.height = nR
    const og = off.getContext('2d')!; const img = og.createImageData(nC, nR)
    for (let r = 0; r < nR; r++) for (let k = 0; k < nC; k++) {
      const v = z[r][k], o = ((nR - 1 - r) * nC + k) * 4
      if (!Number.isFinite(v)) { img.data[o + 3] = 0; continue }
      const i = Math.round(tnorm(v) * 255) * 3
      img.data[o] = L[i]; img.data[o + 1] = L[i + 1]; img.data[o + 2] = L[i + 2]; img.data[o + 3] = 255
    }
    og.putImageData(img, 0, 0)
    g.imageSmoothingEnabled = true
    // draw row by row so uneven excitation spacing keeps its true height
    for (let r = 0; r < nR; r++) {
      const yTop = py(rows.edges[r + 1]), yBot = py(rows.edges[r])
      g.drawImage(off, 0, nR - 1 - r, nC, 1, m.l, yTop, iw, Math.max(1, yBot - yTop + 0.6))
    }
    // contours
    if (contours && nR > 1 && nC > 1) {
      g.save(); g.beginPath(); g.rect(m.l, m.t, iw, ih); g.clip()
      g.lineWidth = 0.9
      const levels = diverging ? [-0.5, -0.2, 0.2, 0.5].map((f) => f * Math.max(Math.abs(range.lo), Math.abs(range.hi))) : [0.1, 0.25, 0.5, 0.75, 0.9].map((f) => range.lo + f * (range.hi - range.lo))
      const X = (k: number) => px(xs[k]), Y = (r: number) => py(ys[r])
      levels.forEach((lv, li) => {
        g.strokeStyle = diverging ? 'rgba(255,255,255,.35)' : `rgba(255,255,255,${0.22 + 0.1 * (li % 3)})`
        g.beginPath()
        for (let r = 0; r < nR - 1; r++) for (let k = 0; k < nC - 1; k++) {
          const a = z[r][k], b = z[r][k + 1], cc = z[r + 1][k + 1], d = z[r + 1][k]
          if (![a, b, cc, d].every(Number.isFinite)) continue
          const idx = (a > lv ? 1 : 0) | (b > lv ? 2 : 0) | (cc > lv ? 4 : 0) | (d > lv ? 8 : 0)
          if (idx === 0 || idx === 15) continue
          const lerp = (p: number, q: number) => (lv - p) / (q - p || 1e-12)
          const top: [number, number] = [X(k) + lerp(a, b) * (X(k + 1) - X(k)), Y(r)]
          const right: [number, number] = [X(k + 1), Y(r) + lerp(b, cc) * (Y(r + 1) - Y(r))]
          const bottom: [number, number] = [X(k) + lerp(d, cc) * (X(k + 1) - X(k)), Y(r + 1)]
          const left: [number, number] = [X(k), Y(r) + lerp(a, d) * (Y(r + 1) - Y(r))]
          const seg = (p: [number, number], q: [number, number]) => { g.moveTo(p[0], p[1]); g.lineTo(q[0], q[1]) }
          switch (idx) {
            case 1: case 14: seg(left, top); break
            case 2: case 13: seg(top, right); break
            case 3: case 12: seg(left, right); break
            case 4: case 11: seg(right, bottom); break
            case 5: seg(left, top); seg(right, bottom); break
            case 6: case 9: seg(top, bottom); break
            case 7: case 8: seg(left, bottom); break
            case 10: seg(top, right); seg(left, bottom); break
          }
        }
        g.stroke()
      })
      g.restore()
    }
    g.strokeStyle = '#3a4350'; g.lineWidth = 1; g.strokeRect(m.l + 0.5, m.t + 0.5, iw, ih)
    g.font = '9.5px "JetBrains Mono Variable", monospace'
    g.fillStyle = AX
    g.textAlign = 'center'
    for (const v of niceTicks(x0, x1, Math.max(3, Math.floor(iw / 64)))) g.fillText(fmtTick(v), px(v), m.t + ih + 13)
    g.textAlign = 'right'
    if (ys.length <= 14) ys.forEach((v, i) => { g.fillStyle = wavelengthToHex(v); g.fillRect(m.l - 5, py(v) - 3, 3, 6); g.fillStyle = AX; if (i % Math.ceil(ys.length / 9) === 0 || ys.length < 10) g.fillText(String(v), m.l - 9, py(v) + 3) })
    else for (const v of niceTicks(rows.lo, rows.hi, 6)) g.fillText(fmtTick(v), m.l - 6, py(v) + 3)
    g.fillStyle = AXL; g.textAlign = 'right'; g.fillText(xLabel, m.l + iw, height - 2)
    g.textAlign = 'left'; g.fillText(yLabel, 2, 10)
    // colour bar
    const bx = m.l + iw + 12, bh = ih
    for (let i = 0; i < bh; i++) {
      const t = 1 - i / (bh - 1)
      const j = Math.round((diverging ? t : (log ? Math.log10(1 + 99 * t) / 2 : t)) * 255) * 3
      g.fillStyle = `rgb(${L[j]},${L[j + 1]},${L[j + 2]})`
      g.fillRect(bx, m.t + i, 8, 1.2)
    }
    g.strokeStyle = '#3a4350'; g.strokeRect(bx + 0.5, m.t + 0.5, 8, bh)
    g.fillStyle = AX; g.textAlign = 'left'
    g.fillText(fmtTick(range.hi), bx + 12, m.t + 8)
    g.fillText(fmtTick(diverging ? 0 : range.lo), bx + 12, m.t + bh)
    if (diverging) g.fillText(fmtTick(range.lo), bx + 12, m.t + bh - 14)
    if (unit) { g.fillStyle = AXL; g.fillText(unit.length > 9 ? unit.slice(0, 8) + '…' : unit, bx - 2, m.t + bh + 12) }
    // marks
    g.save(); g.beginPath(); g.rect(m.l, m.t, iw, ih); g.clip()
    g.setLineDash([3, 3]); g.lineWidth = 1.2
    if (marks?.ex != null) { g.strokeStyle = wavelengthToHex(marks.ex); g.beginPath(); g.moveTo(m.l, py(marks.ex)); g.lineTo(m.l + iw, py(marks.ex)); g.stroke() }
    if (marks?.em != null) { g.strokeStyle = wavelengthToHex(marks.em); g.beginPath(); g.moveTo(px(marks.em), m.t); g.lineTo(px(marks.em), m.t + ih); g.stroke() }
    g.restore()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [z, xs, ys, range, cmap, log, contours, w, height, marks?.ex, marks?.em, unit, rows])

  const locate = (e: React.MouseEvent) => {
    const r = e.currentTarget.getBoundingClientRect()
    const mx = e.clientX - r.left, my = e.clientY - r.top
    if (mx < m.l || mx > m.l + iw || my < m.t || my > m.t + ih) return null
    const em = x0 + ((mx - m.l) / iw) * (x1 - x0)
    const ex = rows.lo + (1 - (my - m.t) / ih) * (rows.hi - rows.lo)
    let ri = 0
    for (let i = 0; i < ys.length; i++) if (ex >= rows.edges[i]) ri = i
    let ci = 0
    for (let k = 1; k < xs.length; k++) if (Math.abs(xs[k] - em) < Math.abs(xs[ci] - em)) ci = k
    return { r: ri, c: ci, px: mx, py: my }
  }
  return (
    <div ref={ref} style={{ position: 'relative', width: '100%' }}>
      <canvas ref={cv} style={{ width: w, height, display: 'block', cursor: onPick ? 'crosshair' : 'default' }}
        onMouseMove={(e) => setHov(locate(e))} onMouseLeave={() => setHov(null)}
        onClick={(e) => { const h = locate(e); if (h && onPick) onPick(ys[h.r], xs[h.c]) }} />
      {hov && Number.isFinite(z[hov.r]?.[hov.c]) && (
        <div className="plot-tip" style={{ left: Math.min(hov.px + 12, w - 150), top: Math.max(0, hov.py - 46) }}>
          <b>ex {ys[hov.r]} nm → em {xs[hov.c]} nm</b>
          <div><span style={{ marginLeft: 0 }}>{fmtVal(z[hov.r][hov.c])}</span>{unit && <small style={{ marginLeft: 6, color: 'var(--faint)' }}>{unit}</small>}</div>
        </div>)}
    </div>
  )
}
