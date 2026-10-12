import type { CaptureMeta } from '../api/types'
import { gtUnit } from '../lib/format'

const CATEGORICAL = new Set(['material_index', 'object_index', 'material', 'object', 'block_face'])
const RANK = ['layer_thickness', 'layer_depth', 'origin_depth_mm', 'origin_layer_fraction', 'depth_mm', 'thickness_mm', 'path_mm', 'irradiance']
const rank = (n: string) => { const i = RANK.findIndex((r) => n === r || n.startsWith(`${r}:`) || n.startsWith(`${r}@`)); return i < 0 ? RANK.length : i }
/** Numeric ground-truth layers, the physically interesting ones (layer thickness / depth) first. */
export const numericGt = (m: CaptureMeta): string[] => m.ground_truth_layers.filter((n) => !CATEGORICAL.has(n)).sort((a, b) => rank(a) - rank(b))

/** `layer_thickness:cover` → `layer thickness · cover` */
export function gtLabel(name: string): string {
  const [head, tail] = name.split(':')
  const h = head.replace(/_(mm|deg|nm|mW)$/, '').replace(/_/g, ' ')
  return tail ? `${h} · ${tail}` : h
}
export function gtUnitOf(name: string): string {
  const head = name.split(':')[0]
  if (head === 'layer_thickness' || head === 'layer_depth' || head === 'origin_depth_mm') return 'mm'
  if (head.startsWith('origin_layer_fraction')) return ''
  return gtUnit(head)
}
/** The most interesting default for a response / line plot: a layer thickness, else depth. */
export function defaultGt(m: CaptureMeta): string | null {
  const g = numericGt(m)
  return g.find((n) => n.startsWith('layer_thickness:') && /cover|collagen/i.test(n)) ?? g.find((n) => n.startsWith('layer_thickness:')) ?? g.find((n) => n === 'depth_mm') ?? g[0] ?? null
}
/** Brightest (excitation, band) of a capture by its 99th percentile ADU — a sensible default metric. */
export function brightestOf(m: CaptureMeta, preferEx?: number | null): { ex: number; band_nm: number } {
  let best: { ex: number; band_nm: number; v: number } | null = null
  const exs = preferEx != null && m.excitations_nm.includes(preferEx) ? [preferEx] : m.excitations_nm
  for (const ex of exs) {
    const st = m.stats[String(ex)], em = m.emission_nm[String(ex)] ?? []
    st?.forEach((s, i) => { if (!best || s.p99 > best.v) best = { ex, band_nm: em[i], v: s.p99 } })
  }
  const b = best as { ex: number; band_nm: number } | null
  return b ? { ex: b.ex, band_nm: b.band_nm } : { ex: exs[0], band_nm: m.emission_nm[String(exs[0])]?.[0] ?? 500 }
}
