import { useEffect, useMemo, useRef, useState } from 'react'
import type { Cube, Raster } from '../api/client'
import { paintComposite, paintScalar, percentiles, sliceBand } from './render'
import { lut, type CmapName } from '../ui/colormaps'
import { bounds, centroid, dragHandle, handles, hit, isDegenerate, lineLength, nearestHandle, translateClamped, type Pt } from '../analysis/roi'
import type { AnyRoi, NamedRoi } from '../analysis/types'
import type { Tool } from '../analysis/store'

export type ViewMode = 'band' | 'true' | 'false'
export interface ImageStyle { mode: ViewMode; cmap: CmapName; pLo: number; pHi: number; log: boolean; globalScale: boolean }

interface Props {
  cube: Cube
  emission: number[]
  band: number
  style: ImageStyle
  gt?: Raster | null
  gtAlpha?: number
  pixel: { x: number; y: number } | null
  rois: NamedRoi[]
  selectedId: string | null
  tool: Tool
  /** a second view of the same ROIs that cannot be edited */
  readOnly?: boolean
  onPixel: (x: number, y: number) => void
  onCreate: (spec: AnyRoi) => void
  onUpdate: (id: string, spec: AnyRoi) => void
  onSelect: (id: string | null) => void
  onToolDone: () => void
  onHover?: (x: number, y: number, v: number) => void
  label?: string
  onRange?: (lo: number, hi: number) => void
}

type Drag =
  | { type: 'draw'; start: Pt }
  | { type: 'move'; id: string; start: Pt; orig: AnyRoi }
  | { type: 'handle'; id: string; handle: string }
  | { type: 'pixel' }

const snap = (v: number) => Math.round(v)
const common = { vectorEffect: 'non-scaling-stroke' as const }

export function ImageView({ cube, emission, band, style, gt, gtAlpha = 0.5, pixel, rois, selectedId, tool, readOnly, onPixel, onCreate, onUpdate, onSelect, onToolDone, onHover, label, onRange }: Props) {
  const cv = useRef<HTMLCanvasElement>(null)
  const stage = useRef<HTMLDivElement>(null)
  const [box, setBox] = useState({ w: 400, h: 300 })
  const drag = useRef<Drag | null>(null)
  const [draft, setDraft] = useState<AnyRoi | null>(null)
  const [polyDraft, setPolyDraft] = useState<Pt[] | null>(null)
  const [cursorPt, setCursorPt] = useState<Pt | null>(null)
  const [cursor, setCursor] = useState('default')
  const { width: W, height: H } = cube

  useEffect(() => {
    const el = stage.current; if (!el) return
    const ro = new ResizeObserver(() => setBox({ w: el.clientWidth, h: el.clientHeight }))
    ro.observe(el); setBox({ w: el.clientWidth, h: el.clientHeight })
    return () => ro.disconnect()
  }, [])

  const composite = useMemo(() => {
    if (style.mode === 'band') return null
    const img = new ImageData(W, H)
    const n = emission.length
    paintComposite(img, cube, emission, style.mode === 'false' ? { falseColor: [Math.round(n * 0.15), Math.round(n * 0.45), Math.round(n * 0.8)] } : {})
    return img
  }, [cube, emission, style.mode, W, H])

  const globalRange = useMemo(() => (style.globalScale ? percentiles(cube.data, style.pLo, style.pHi, 53) : null), [cube, style.globalScale, style.pLo, style.pHi])
  const bandData = useMemo(() => sliceBand(cube, Math.min(band, cube.bands - 1)), [cube, band])
  const range = useMemo(() => globalRange ?? percentiles(bandData, style.pLo, style.pHi, 5), [globalRange, bandData, style.pLo, style.pHi])

  useEffect(() => { onRange?.(range[0], range[1]) }, [range, onRange])

  useEffect(() => {
    const c = cv.current; if (!c) return
    c.width = W; c.height = H
    const g = c.getContext('2d')!
    const img = composite ? new ImageData(new Uint8ClampedArray(composite.data), W, H) : g.createImageData(W, H)
    if (!composite) paintScalar(img, bandData, { cmap: style.cmap, lo: range[0], hi: range[1], log: style.log })
    if (gt && gt.data.length === W * H) {
      let mn = Infinity, mx = -Infinity
      for (let k = 0; k < gt.data.length; k += 3) { const v = gt.data[k]; if (Number.isFinite(v)) { if (v < mn) mn = v; if (v > mx) mx = v } }
      const L = lut('turbo'), sp = mx - mn || 1
      for (let k = 0; k < gt.data.length; k++) {
        const v = gt.data[k]; if (!Number.isFinite(v)) continue
        const i = Math.round(((v - mn) / sp) * 255) * 3, o = k * 4
        for (let ch = 0; ch < 3; ch++) img.data[o + ch] = img.data[o + ch] * (1 - gtAlpha) + L[i + ch] * gtAlpha
      }
    }
    g.putImageData(img, 0, 0)
  }, [composite, bandData, range, style.cmap, style.log, gt, gtAlpha, W, H])

  const scale = Math.max(0.2, Math.min((box.w - 24) / W, (box.h - 24) / H))
  const s = scale >= 1 ? Math.floor(scale * 4) / 4 : scale
  const cw = W * s, ch = H * s
  const tol = 7 / s

  /** pointer position in image coordinates where pixel (i, j) is centred at (i, j) */
  const toImg = (e: React.PointerEvent | React.MouseEvent): Pt => {
    const r = cv.current!.getBoundingClientRect()
    return [((e.clientX - r.left) / r.width) * W - 0.5, ((e.clientY - r.top) / r.height) * H - 0.5]
  }
  const clampPt = (p: Pt): Pt => [Math.min(W - 1, Math.max(0, p[0])), Math.min(H - 1, Math.max(0, p[1]))]
  const snapPt = (p: Pt): Pt => clampPt([snap(p[0]), snap(p[1])])
  const px = (p: Pt) => ({ x: Math.min(W - 1, Math.max(0, Math.round(p[0]))), y: Math.min(H - 1, Math.max(0, Math.round(p[1]))) })

  const visible = rois.filter((r) => r.visible)
  const selected = rois.find((r) => r.id === selectedId) ?? null

  const finishPolygon = (pts: Pt[]) => {
    setPolyDraft(null)
    if (pts.length >= 3) onCreate({ shape: 'polygon', points: pts })
    onToolDone()
  }

  // keyboard control of an in-progress polygon
  useEffect(() => {
    if (!polyDraft) return
    const h = (e: KeyboardEvent) => {
      if (e.key === 'Enter') { e.preventDefault(); finishPolygon(polyDraft) }
      else if (e.key === 'Escape') { setPolyDraft(null); onToolDone() }
      else if (e.key === 'Backspace') { e.preventDefault(); e.stopPropagation(); setPolyDraft(polyDraft.length > 1 ? polyDraft.slice(0, -1) : null) }
    }
    window.addEventListener('keydown', h, true)
    return () => window.removeEventListener('keydown', h, true)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [polyDraft])
  useEffect(() => { if (tool !== 'polygon') setPolyDraft(null); setDraft(null) }, [tool])

  const makeDraft = (t: Tool, a: Pt, b: Pt): AnyRoi | null => {
    switch (t) {
      case 'disk': return { shape: 'disk', x: a[0], y: a[1], radius_px: Math.max(0.5, Math.round(Math.hypot(b[0] - a[0], b[1] - a[1]) * 2) / 2) }
      case 'rect': return { shape: 'rect', x0: Math.min(a[0], b[0]), y0: Math.min(a[1], b[1]), x1: Math.max(a[0], b[0]), y1: Math.max(a[1], b[1]) }
      case 'ellipse': return { shape: 'ellipse', cx: (a[0] + b[0]) / 2, cy: (a[1] + b[1]) / 2, rx: Math.abs(b[0] - a[0]) / 2, ry: Math.abs(b[1] - a[1]) / 2 }
      case 'line': return { shape: 'line', p0: a, p1: b, width_px: 1 }
      default: return null
    }
  }

  const onDown = (e: React.PointerEvent) => {
    if (e.button !== 0 || readOnly) return
    ;(e.currentTarget as Element).setPointerCapture(e.pointerId)
    const raw = toImg(e)
    if (tool === 'pointer') {
      if (selected?.visible) {
        const hnd = nearestHandle(selected.spec, raw[0], raw[1], tol * 1.5)
        if (hnd) { drag.current = { type: 'handle', id: selected.id, handle: hnd.id }; return }
      }
      for (let i = visible.length - 1; i >= 0; i--) {
        if (hit(visible[i].spec, raw[0], raw[1], tol)) {
          onSelect(visible[i].id)
          drag.current = { type: 'move', id: visible[i].id, start: raw, orig: visible[i].spec }
          return
        }
      }
      onSelect(null)
      drag.current = { type: 'pixel' }
      const p = px(raw); onPixel(p.x, p.y)
      return
    }
    const p = snapPt(raw)
    if (tool === 'point') { onCreate({ shape: 'point', x: p[0], y: p[1] }); onToolDone(); return }
    if (tool === 'polygon') {
      if (!polyDraft) { setPolyDraft([p]); return }
      const first = polyDraft[0]
      if (polyDraft.length >= 3 && Math.hypot(first[0] - raw[0], first[1] - raw[1]) <= tol * 1.4) { finishPolygon(polyDraft); return }
      setPolyDraft([...polyDraft, p]); return
    }
    drag.current = { type: 'draw', start: p }
    setDraft(makeDraft(tool, p, p))
  }

  const onMove = (e: React.PointerEvent) => {
    const raw = toImg(e)
    const pp = px(raw)
    onHover?.(pp.x, pp.y, bandData[pp.y * W + pp.x])
    setCursorPt(snapPt(raw))
    const d = drag.current
    if (d) {
      if (d.type === 'draw') setDraft(makeDraft(tool, d.start, snapPt(raw)))
      else if (d.type === 'move') onUpdate(d.id, translateClamped(d.orig, raw[0] - d.start[0], raw[1] - d.start[1], W, H))
      else if (d.type === 'handle') {
        const sp = rois.find((r) => r.id === d.id)?.spec
        if (sp) {
          const q = snapPt(raw), free = sp.shape === 'disk' || sp.shape === 'ellipse'
          onUpdate(d.id, dragHandle(sp, d.handle, free ? raw[0] : q[0], free ? raw[1] : q[1]))
        }
      } else if (d.type === 'pixel' && e.buttons === 1) onPixel(pp.x, pp.y)
      return
    }
    if (readOnly) { setCursor('default'); return }
    if (tool !== 'pointer') { setCursor('crosshair'); return }
    if (selected?.visible) {
      const hnd = nearestHandle(selected.spec, raw[0], raw[1], tol * 1.5)
      if (hnd) { setCursor(handles(selected.spec).find((q) => q.id === hnd.id)?.cursor ?? 'move'); return }
    }
    setCursor(visible.some((r) => hit(r.spec, raw[0], raw[1], tol)) ? 'move' : 'crosshair')
  }

  const onUp = () => {
    const d = drag.current
    drag.current = null
    if (d?.type === 'draw' && draft) {
      if (!isDegenerate(draft)) onCreate(draft)
      setDraft(null)
      onToolDone()
    }
  }

  const shapeEl = (spec: AnyRoi, color: string, opts: { selected?: boolean; dashed?: boolean; key?: string; fillOpacity?: number }) => {
    const dash = opts.dashed ? '5 4' : undefined
    const stroke = { stroke: color, strokeWidth: opts.selected ? 2.2 : 1.6, strokeDasharray: dash, fill: color, fillOpacity: opts.fillOpacity ?? (opts.selected ? 0.2 : 0.1), ...common }
    const halo = { stroke: 'rgba(0,0,0,.65)', strokeWidth: opts.selected ? 4.4 : 3.6, fill: 'none', strokeDasharray: dash, ...common }
    const o = 0.5
    switch (spec.shape) {
      case 'point': return <g key={opts.key}><rect x={spec.x} y={spec.y} width={1} height={1} {...halo} /><rect x={spec.x} y={spec.y} width={1} height={1} {...stroke} fillOpacity={0.35} /></g>
      case 'disk': return <g key={opts.key}><circle cx={spec.x + o} cy={spec.y + o} r={spec.radius_px} {...halo} /><circle cx={spec.x + o} cy={spec.y + o} r={spec.radius_px} {...stroke} /></g>
      case 'rect': { const b = bounds(spec); return <g key={opts.key}><rect x={b.x0} y={b.y0} width={b.x1 - b.x0 + 1} height={b.y1 - b.y0 + 1} {...halo} /><rect x={b.x0} y={b.y0} width={b.x1 - b.x0 + 1} height={b.y1 - b.y0 + 1} {...stroke} /></g> }
      case 'ellipse': return <g key={opts.key}><ellipse cx={spec.cx + o} cy={spec.cy + o} rx={spec.rx} ry={spec.ry} {...halo} /><ellipse cx={spec.cx + o} cy={spec.cy + o} rx={spec.rx} ry={spec.ry} {...stroke} /></g>
      case 'polygon': { const d = spec.points.map((p, i) => `${i ? 'L' : 'M'}${p[0] + o},${p[1] + o}`).join('') + 'Z'; return <g key={opts.key}><path d={d} {...halo} /><path d={d} {...stroke} /></g> }
      case 'line': {
        const w = Math.max(1, spec.width_px)
        const ln = { x1: spec.p0[0] + o, y1: spec.p0[1] + o, x2: spec.p1[0] + o, y2: spec.p1[1] + o }
        return <g key={opts.key}>
          {w > 1 && <line {...ln} stroke={color} strokeOpacity={0.18} strokeWidth={w * s} {...common} />}
          <line {...ln} {...halo} />
          <line {...ln} stroke={color} strokeWidth={opts.selected ? 2.2 : 1.6} strokeDasharray={dash} {...common} />
          {[spec.p0, spec.p1].map((p, i) => <circle key={i} cx={p[0] + o} cy={p[1] + o} r={3.2 / s} fill={color} stroke="rgba(0,0,0,.7)" strokeWidth={1} {...common} />)}
        </g>
      }
    }
  }

  const hs = 3.6 / s
  return (
    <div ref={stage} className="imgstage">
      <div style={{ position: 'relative', width: cw, height: ch }}>
        <canvas ref={cv} style={{ width: cw, height: ch, display: 'block', imageRendering: s >= 2 ? 'pixelated' : 'auto' }} />
        <svg width={cw} height={ch} viewBox={`0 0 ${W} ${H}`} style={{ position: 'absolute', inset: 0, cursor, touchAction: 'none' }}
          onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onPointerLeave={() => setCursorPt(null)}
          onDoubleClick={() => { if (tool === 'polygon' && polyDraft && polyDraft.length >= 3) finishPolygon(polyDraft) }}>
          {pixel && <g pointerEvents="none"><circle cx={pixel.x + 0.5} cy={pixel.y + 0.5} r={10 / s} fill="none" stroke="rgba(0,0,0,.7)" strokeWidth={4} {...common} /><circle cx={pixel.x + 0.5} cy={pixel.y + 0.5} r={10 / s} fill="none" stroke="#35d8ff" strokeWidth={1.8} {...common} /><rect x={pixel.x} y={pixel.y} width={1} height={1} fill="none" stroke="#35d8ff" strokeWidth={1.6} {...common} />
            <path d={`M${pixel.x + 0.5},0V${pixel.y - 1.5}M${pixel.x + 0.5},${pixel.y + 2.5}V${H}M0,${pixel.y + 0.5}H${pixel.x - 1.5}M${pixel.x + 2.5},${pixel.y + 0.5}H${W}`} stroke="rgba(53,216,255,.35)" strokeWidth={1} {...common} /></g>}
          <g pointerEvents="none">{visible.map((r) => shapeEl(r.spec, r.color, { selected: r.id === selectedId, key: r.id }))}</g>
          {draft && <g pointerEvents="none">{shapeEl(draft, '#ffffff', { dashed: true, fillOpacity: 0.08 })}</g>}
          {polyDraft && <g pointerEvents="none">
            <path d={polyDraft.map((p, i) => `${i ? 'L' : 'M'}${p[0] + 0.5},${p[1] + 0.5}`).join('') + (cursorPt ? `L${cursorPt[0] + 0.5},${cursorPt[1] + 0.5}` : '')} fill="rgba(255,255,255,.07)" stroke="#fff" strokeWidth={1.6} strokeDasharray="5 4" {...common} />
            {polyDraft.map((p, i) => <circle key={i} cx={p[0] + 0.5} cy={p[1] + 0.5} r={(i === 0 ? 4.6 : 3) / s} fill={i === 0 ? '#35d8ff' : '#fff'} stroke="rgba(0,0,0,.7)" strokeWidth={1} {...common} />)}
          </g>}
          {!readOnly && selected?.visible && tool === 'pointer' && <g>{handles(selected.spec).map((h) => (
            <rect key={h.id} x={h.x + 0.5 - hs} y={h.y + 0.5 - hs} width={hs * 2} height={hs * 2} rx={hs * 0.3} fill="#0b0e12" stroke={selected.color} strokeWidth={1.8} {...common} />))}</g>}
        </svg>
        {visible.map((r) => {
          const c = centroid(r.spec), b = bounds(r.spec)
          const top = r.spec.shape === 'line' ? c[1] : b.y0
          const x0 = (r.spec.shape === 'line' ? c[0] : (b.x0 + b.x1) / 2) + 0.5
          return <div key={r.id} className={`roi-tag${r.id === selectedId ? ' sel' : ''}`} style={{ left: x0 * s, top: Math.max(2, (top + 0.5) * s - 22), ['--c' as string]: r.color }}>
            <i />{r.name}{r.spec.shape === 'line' && <small>{lineLength(r.spec).toFixed(0)} px</small>}</div>
        })}
        {label && <div className="readout" style={{ position: 'absolute', left: 6, top: 6, transform: 'none', pointerEvents: 'none' }}>{label}</div>}
      </div>
    </div>
  )
}
