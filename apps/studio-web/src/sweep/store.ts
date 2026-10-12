/** Parameter-sweep state: configuration, run progress and per-point results. */
import { create } from 'zustand'
import type { MetricSpec, NormSpec, RoiSpec, SweepEvent } from '../api/types'

export type SweepValueMode = 'range' | 'list'

export interface SweepConfig {
  path: string | null
  label: string
  from: number
  to: number
  steps: number
  spacing: 'linear' | 'log'
  mode: SweepValueMode
  list: string
  runMode: 'preview' | 'sample'
  roiId: string | null
  metric: MetricSpec | null
}

export interface SweepPoint {
  index: number
  value: number | string | boolean
  captureId: string | null
  metric: number | null
  std: number | null
  /** why the metric is unavailable (capture lacks the band, request failed…) */
  err?: string | null
}

export interface SweepRun {
  id: string
  path: string
  label: string
  values: (number | string | boolean)[]
  points: SweepPoint[]
  state: 'running' | 'done' | 'error' | 'cancelled'
  fraction: number
  message: string
  error: string | null
  experimentName: string
  metric: MetricSpec | null
  roiId: string | null
  /** ROI in the reference capture's pixel grid, plus that grid's size (rescaled for every swept capture) */
  roi: RoiSpec | null
  refSize: [number, number]
  norm: NormSpec
  stat: 'mean' | 'median'
  unit: string
  note: string | null
}

interface SweepState {
  cfg: SweepConfig
  run: SweepRun | null
  setCfg: (p: Partial<SweepConfig>) => void
  startRun: (r: SweepRun) => void
  onEvent: (e: SweepEvent) => void
  setPoint: (index: number, p: Partial<SweepPoint>) => void
  patchRun: (p: Partial<SweepRun>) => void
  clear: () => void
}

export const useSweep = create<SweepState>()((set) => ({
  cfg: { path: null, label: '', from: 0, to: 2, steps: 8, spacing: 'linear', mode: 'range', list: '', runMode: 'preview', roiId: null, metric: null },
  run: null,
  setCfg: (p) => set((s) => ({ cfg: { ...s.cfg, ...p } })),
  startRun: (run) => set({ run }),
  patchRun: (p) => set((s) => (s.run ? { run: { ...s.run, ...p } } : {})),
  setPoint: (index, p) => set((s) => {
    if (!s.run) return {}
    const points = s.run.points.slice()
    const k = points.findIndex((q) => q.index === index)
    if (k >= 0) points[k] = { ...points[k], ...p }
    else points.push({ index, value: s.run.values[index], captureId: null, metric: null, std: null, ...p })
    points.sort((a, b) => a.index - b.index)
    return { run: { ...s.run, points } }
  }),
  onEvent: (e) => set((s) => {
    if (!s.run) return {}
    switch (e.type) {
      case 'progress': return { run: { ...s.run, fraction: e.fraction, message: e.message } }
      case 'point': return {}
      case 'done': return { run: { ...s.run, state: 'done', fraction: 1, message: 'Done' } }
      case 'error': return { run: { ...s.run, state: 'error', error: e.message, message: e.message } }
      case 'cancelled': return { run: { ...s.run, state: 'cancelled', message: 'Cancelled' } }
    }
  }),
  clear: () => set({ run: null }),
}))

/** Parameter values for a range specification. */
export function rangeValues(from: number, to: number, steps: number, spacing: 'linear' | 'log'): number[] {
  const n = Math.max(1, Math.round(steps))
  if (n === 1) return [from]
  const out: number[] = []
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1)
    out.push(spacing === 'log' && from > 0 && to > 0 ? from * (to / from) ** t : from + (to - from) * t)
  }
  return out.map((v) => +v.toPrecision(10))
}

/** Parse "0, 0.25 0.5; 1" into numbers (non-numeric tokens are ignored). */
export function parseList(text: string): number[] {
  return text.split(/[\s,;]+/).map((t) => t.trim()).filter(Boolean).map(Number).filter((v) => Number.isFinite(v))
}

if (import.meta.env.DEV && typeof window !== 'undefined') (window as unknown as { sweep: typeof useSweep }).sweep = useSweep
