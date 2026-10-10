import { useEffect, useMemo, useRef, useState } from 'react'
import type { Cube, Raster } from '../api/client'
import { paintComposite, paintScalar, percentiles, sliceBand } from './render'
import { lut, type CmapName } from '../ui/colormaps'

export type ViewMode = 'band' | 'true' | 'false'
export interface Roi { x0: number; y0: number; x1: number; y1: number }
export interface ImageStyle { mode: ViewMode; cmap: CmapName; pLo: number; pHi: number; log: boolean; globalScale: boolean }

interface Props {
  cube: Cube
  emission: number[]
  band: number
  style: ImageStyle
  gt?: Raster | null
  gtAlpha?: number
  pixel: { x: number; y: number } | null
  roi: Roi | null
  tool: 'pixel' | 'roi'
  onPixel: (x: number, y: number) => void
  onRoi: (r: Roi | null) => void
  onHover?: (x: number, y: number, v: number) => void
  label?: string
  onRange?: (lo: number, hi: number) => void
}

export function ImageView({ cube, emission, band, style, gt, gtAlpha = 0.5, pixel, roi, tool, onPixel, onRoi, onHover, label, onRange }: Props) {
  const cv = useRef<HTMLCanvasElement>(null)
  const stage = useRef<HTMLDivElement>(null)
  const [box, setBox] = useState({ w: 400, h: 300 })
  const drag = useRef<{ x: number; y: number } | null>(null)
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
  const toPx = (e: React.PointerEvent) => {
    const r = cv.current!.getBoundingClientRect()
    return { x: Math.min(W - 1, Math.max(0, Math.floor(((e.clientX - r.left) / r.width) * W))), y: Math.min(H - 1, Math.max(0, Math.floor(((e.clientY - r.top) / r.height) * H))) }
  }
  return (
    <div ref={stage} className="imgstage">
      <div style={{ position: 'relative', width: cw, height: ch }}>
        <canvas ref={cv} style={{ width: cw, height: ch, display: 'block' }}
          onPointerDown={(e) => { (e.target as Element).setPointerCapture(e.pointerId); const p = toPx(e); if (tool === 'roi') { drag.current = p; onRoi({ x0: p.x, y0: p.y, x1: p.x, y1: p.y }) } else onPixel(p.x, p.y) }}
          onPointerMove={(e) => {
            const p = toPx(e)
            onHover?.(p.x, p.y, bandData[p.y * W + p.x])
            if (tool === 'roi' && drag.current) onRoi({ x0: Math.min(drag.current.x, p.x), y0: Math.min(drag.current.y, p.y), x1: Math.max(drag.current.x, p.x), y1: Math.max(drag.current.y, p.y) })
            else if (tool === 'pixel' && e.buttons === 1) onPixel(p.x, p.y)
          }}
          onPointerUp={() => { drag.current = null }} />
        <svg width={cw} height={ch} viewBox={`0 0 ${W} ${H}`} style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}>
          {pixel && <g><circle cx={pixel.x + 0.5} cy={pixel.y + 0.5} r={10 / s} fill="none" stroke="rgba(0,0,0,.7)" strokeWidth={4} vectorEffect="non-scaling-stroke" /><circle cx={pixel.x + 0.5} cy={pixel.y + 0.5} r={10 / s} fill="none" stroke="#35d8ff" strokeWidth={1.8} vectorEffect="non-scaling-stroke" /><rect x={pixel.x} y={pixel.y} width={1} height={1} fill="none" stroke="#35d8ff" strokeWidth={Math.max(0.5, 1.6 / s)} vectorEffect="non-scaling-stroke" />
            <path d={`M${pixel.x + 0.5},0V${pixel.y - 1.5}M${pixel.x + 0.5},${pixel.y + 2.5}V${H}M0,${pixel.y + 0.5}H${pixel.x - 1.5}M${pixel.x + 2.5},${pixel.y + 0.5}H${W}`} stroke="rgba(53,216,255,.35)" strokeWidth={1} vectorEffect="non-scaling-stroke" /></g>}
          {roi && <rect x={roi.x0} y={roi.y0} width={roi.x1 - roi.x0 + 1} height={roi.y1 - roi.y0 + 1} fill="rgba(255,181,71,.12)" stroke="#ffb547" strokeWidth={1.4} strokeDasharray="4 3" vectorEffect="non-scaling-stroke" />}
        </svg>
        {label && <div className="readout" style={{ position: 'absolute', left: 6, top: 6, transform: 'none' }}>{label}</div>}
      </div>
    </div>
  )
}
