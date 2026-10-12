/** ROI geometry in capture pixel coordinates: pixel (i, j) has its centre at (i, j); x = column, y = row (origin top-left).
 *  Rectangles are inclusive of pixels whose centres lie within [x0, x1] × [y0, y1]. */
import type { AnyRoi, LineSpec, RegionSpec } from './types'
import { isLine } from './types'

export const ROI_COLORS = ['#ffb547', '#ff6bb0', '#7bf26a', '#b48cff', '#35d8ff', '#ff7a5e', '#e8e35a', '#6b9dff']

export type Pt = [number, number]

export function pointInPolygon(x: number, y: number, pts: readonly Pt[]): boolean {
  let inside = false
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const [xi, yi] = pts[i], [xj, yj] = pts[j]
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi || 1e-12) + xi) inside = !inside
  }
  return inside
}

export function contains(spec: RegionSpec, i: number, j: number): boolean {
  switch (spec.shape) {
    case 'point': return Math.round(spec.x) === i && Math.round(spec.y) === j
    case 'disk': return (i - spec.x) ** 2 + (j - spec.y) ** 2 <= spec.radius_px ** 2 + 1e-9
    case 'rect': {
      const x0 = Math.min(spec.x0, spec.x1), x1 = Math.max(spec.x0, spec.x1), y0 = Math.min(spec.y0, spec.y1), y1 = Math.max(spec.y0, spec.y1)
      return i >= x0 - 1e-9 && i <= x1 + 1e-9 && j >= y0 - 1e-9 && j <= y1 + 1e-9
    }
    case 'ellipse': return spec.rx > 0 && spec.ry > 0 && ((i - spec.cx) / spec.rx) ** 2 + ((j - spec.cy) / spec.ry) ** 2 <= 1 + 1e-9
    case 'polygon': return spec.points.length >= 3 && pointInPolygon(i, j, spec.points)
  }
}

export function bounds(spec: AnyRoi): { x0: number; y0: number; x1: number; y1: number } {
  switch (spec.shape) {
    case 'point': return { x0: spec.x, y0: spec.y, x1: spec.x, y1: spec.y }
    case 'disk': return { x0: spec.x - spec.radius_px, y0: spec.y - spec.radius_px, x1: spec.x + spec.radius_px, y1: spec.y + spec.radius_px }
    case 'rect': return { x0: Math.min(spec.x0, spec.x1), y0: Math.min(spec.y0, spec.y1), x1: Math.max(spec.x0, spec.x1), y1: Math.max(spec.y0, spec.y1) }
    case 'ellipse': return { x0: spec.cx - spec.rx, y0: spec.cy - spec.ry, x1: spec.cx + spec.rx, y1: spec.cy + spec.ry }
    case 'polygon': {
      const xs = spec.points.map((p) => p[0]), ys = spec.points.map((p) => p[1])
      return { x0: Math.min(...xs), y0: Math.min(...ys), x1: Math.max(...xs), y1: Math.max(...ys) }
    }
    case 'line': return { x0: Math.min(spec.p0[0], spec.p1[0]), y0: Math.min(spec.p0[1], spec.p1[1]), x1: Math.max(spec.p0[0], spec.p1[0]), y1: Math.max(spec.p0[1], spec.p1[1]) }
  }
}

/** Indices (row-major, y·W + x) of the pixels inside a region ROI, clipped to the W×H image. */
export function maskIndices(spec: RegionSpec, W: number, H: number): Int32Array {
  const b = bounds(spec)
  const i0 = Math.max(0, Math.floor(b.x0)), i1 = Math.min(W - 1, Math.ceil(b.x1)), j0 = Math.max(0, Math.floor(b.y0)), j1 = Math.min(H - 1, Math.ceil(b.y1))
  const out: number[] = []
  for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) if (contains(spec, i, j)) out.push(j * W + i)
  return Int32Array.from(out)
}

export const lineLength = (s: LineSpec) => Math.hypot(s.p1[0] - s.p0[0], s.p1[1] - s.p0[1])

export function centroid(spec: AnyRoi): Pt {
  const b = bounds(spec)
  if (spec.shape === 'polygon') {
    const n = spec.points.length || 1
    return [spec.points.reduce((a, p) => a + p[0], 0) / n, spec.points.reduce((a, p) => a + p[1], 0) / n]
  }
  return [(b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2]
}

export function translate(spec: AnyRoi, dx: number, dy: number): AnyRoi {
  switch (spec.shape) {
    case 'point': return { ...spec, x: spec.x + dx, y: spec.y + dy }
    case 'disk': return { ...spec, x: spec.x + dx, y: spec.y + dy }
    case 'rect': return { ...spec, x0: spec.x0 + dx, x1: spec.x1 + dx, y0: spec.y0 + dy, y1: spec.y1 + dy }
    case 'ellipse': return { ...spec, cx: spec.cx + dx, cy: spec.cy + dy }
    case 'polygon': return { ...spec, points: spec.points.map(([x, y]) => [x + dx, y + dy] as Pt) }
    case 'line': return { ...spec, p0: [spec.p0[0] + dx, spec.p0[1] + dy], p1: [spec.p1[0] + dx, spec.p1[1] + dy] }
  }
}

/** Translate but keep the whole ROI inside the image. */
export function translateClamped(spec: AnyRoi, dx: number, dy: number, W: number, H: number): AnyRoi {
  const b = bounds(spec)
  const cx = Math.min(Math.max(dx, -b.x0), W - 1 - b.x1)
  const cy = Math.min(Math.max(dy, -b.y0), H - 1 - b.y1)
  return translate(spec, cx, cy)
}

export interface Handle { id: string; x: number; y: number; cursor: string }

export function handles(spec: AnyRoi): Handle[] {
  switch (spec.shape) {
    case 'point': return []
    case 'disk': return [{ id: 'r', x: spec.x + spec.radius_px, y: spec.y, cursor: 'ew-resize' }]
    case 'rect': return [
      { id: 'nw', x: spec.x0, y: spec.y0, cursor: 'nwse-resize' }, { id: 'ne', x: spec.x1, y: spec.y0, cursor: 'nesw-resize' },
      { id: 'se', x: spec.x1, y: spec.y1, cursor: 'nwse-resize' }, { id: 'sw', x: spec.x0, y: spec.y1, cursor: 'nesw-resize' },
    ]
    case 'ellipse': return [{ id: 'rx', x: spec.cx + spec.rx, y: spec.cy, cursor: 'ew-resize' }, { id: 'ry', x: spec.cx, y: spec.cy + spec.ry, cursor: 'ns-resize' }]
    case 'polygon': return spec.points.map((p, i) => ({ id: `v${i}`, x: p[0], y: p[1], cursor: 'move' }))
    case 'line': return [{ id: 'p0', x: spec.p0[0], y: spec.p0[1], cursor: 'move' }, { id: 'p1', x: spec.p1[0], y: spec.p1[1], cursor: 'move' }]
  }
}

/** Move one handle to (x, y) and return the new spec. */
export function dragHandle(spec: AnyRoi, id: string, x: number, y: number): AnyRoi {
  switch (spec.shape) {
    case 'point': return spec
    case 'disk': return { ...spec, radius_px: Math.max(0.5, Math.hypot(x - spec.x, y - spec.y)) }
    case 'rect': {
      const n = { ...spec }
      if (id.includes('n')) n.y0 = y; else n.y1 = y
      if (id.includes('w')) n.x0 = x; else n.x1 = x
      return n
    }
    case 'ellipse': return id === 'rx' ? { ...spec, rx: Math.max(0.5, Math.abs(x - spec.cx)) } : { ...spec, ry: Math.max(0.5, Math.abs(y - spec.cy)) }
    case 'polygon': { const i = Number(id.slice(1)); return { ...spec, points: spec.points.map((p, k) => (k === i ? ([x, y] as Pt) : p)) } }
    case 'line': return id === 'p0' ? { ...spec, p0: [x, y] } : { ...spec, p1: [x, y] }
  }
}

function distSeg(px: number, py: number, a: Pt, b: Pt): number {
  const dx = b[0] - a[0], dy = b[1] - a[1]
  const t = Math.min(1, Math.max(0, ((px - a[0]) * dx + (py - a[1]) * dy) / (dx * dx + dy * dy || 1)))
  return Math.hypot(px - (a[0] + t * dx), py - (a[1] + t * dy))
}

/** Hit test (image px). `tol` = pick tolerance in px. Region ROIs hit inside; lines and points within `tol`. */
export function hit(spec: AnyRoi, x: number, y: number, tol: number): boolean {
  if (spec.shape === 'line') return distSeg(x, y, spec.p0, spec.p1) <= Math.max(tol, spec.width_px / 2)
  if (spec.shape === 'point') return Math.hypot(x - spec.x, y - spec.y) <= tol
  if (spec.shape === 'polygon') return pointInPolygon(x, y, spec.points) || spec.points.some((p, i) => distSeg(x, y, p, spec.points[(i + 1) % spec.points.length]) <= tol)
  const b = bounds(spec)
  if (x < b.x0 - tol || x > b.x1 + tol || y < b.y0 - tol || y > b.y1 + tol) return false
  if (spec.shape === 'rect') return true
  if (spec.shape === 'disk') return Math.hypot(x - spec.x, y - spec.y) <= spec.radius_px + tol
  return ((x - spec.cx) / (spec.rx + tol)) ** 2 + ((y - spec.cy) / (spec.ry + tol)) ** 2 <= 1
}

export function nearestHandle(spec: AnyRoi, x: number, y: number, tol: number): Handle | null {
  let best: Handle | null = null, bd = tol
  for (const h of handles(spec)) { const d = Math.hypot(h.x - x, h.y - y); if (d <= bd) { best = h; bd = d } }
  return best
}

export function polygonArea(pts: readonly Pt[]): number {
  let a = 0
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) a += pts[j][0] * pts[i][1] - pts[i][0] * pts[j][1]
  return Math.abs(a) / 2
}

export function describe(spec: AnyRoi): string {
  const f = (v: number) => (Number.isInteger(v) ? String(v) : v.toFixed(1))
  switch (spec.shape) {
    case 'point': return `point (${f(spec.x)}, ${f(spec.y)})`
    case 'disk': return `disk (${f(spec.x)}, ${f(spec.y)}) r ${f(spec.radius_px)} px`
    case 'rect': { const b = bounds(spec); return `rect ${f(b.x1 - b.x0 + 1)}×${f(b.y1 - b.y0 + 1)} px @ (${f(b.x0)}, ${f(b.y0)})` }
    case 'ellipse': return `ellipse (${f(spec.cx)}, ${f(spec.cy)}) ${f(spec.rx)}×${f(spec.ry)} px`
    case 'polygon': return `polygon · ${spec.points.length} vertices`
    case 'line': return `line ${f(lineLength(spec))} px · width ${spec.width_px}`
  }
}

export const KIND_LABEL: Record<AnyRoi['shape'], string> = { point: 'Point', disk: 'Disk', rect: 'Rectangle', ellipse: 'Ellipse', polygon: 'Polygon', line: 'Line' }

export const isDegenerate = (s: AnyRoi): boolean => {
  if (s.shape === 'rect') return s.x0 === s.x1 && s.y0 === s.y1
  if (s.shape === 'disk') return s.radius_px < 0.5
  if (s.shape === 'ellipse') return s.rx < 0.5 || s.ry < 0.5
  if (s.shape === 'polygon') return s.points.length < 3
  if (isLine(s)) return lineLength(s) < 1
  return false
}

/** Rescale an ROI drawn on a W0×H0 image to a W1×H1 image (swept captures may have a different resolution). */
export function scaleRoi<T extends AnyRoi>(spec: T, sx: number, sy: number): T {
  const m = (x: number, s: number) => (x + 0.5) * s - 0.5
  const r = (sx + sy) / 2
  const o = spec as AnyRoi
  switch (o.shape) {
    case 'point': return { ...o, x: Math.round(m(o.x, sx)), y: Math.round(m(o.y, sy)) } as T
    case 'disk': return { ...o, x: m(o.x, sx), y: m(o.y, sy), radius_px: Math.max(0.5, o.radius_px * r) } as T
    case 'rect': return { ...o, x0: m(o.x0, sx), x1: m(o.x1, sx), y0: m(o.y0, sy), y1: m(o.y1, sy) } as T
    case 'ellipse': return { ...o, cx: m(o.cx, sx), cy: m(o.cy, sy), rx: Math.max(0.5, o.rx * sx), ry: Math.max(0.5, o.ry * sy) } as T
    case 'polygon': return { ...o, points: o.points.map(([x, y]) => [m(x, sx), m(y, sy)] as Pt) } as T
    case 'line': return { ...o, p0: [m(o.p0[0], sx), m(o.p0[1], sy)], p1: [m(o.p1[0], sx), m(o.p1[1], sy)] } as T
  }
}
