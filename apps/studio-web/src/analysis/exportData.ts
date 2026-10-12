/** CSV / JSON export of every analysis result. CSV files start with `#`-prefixed header lines (units, ROI definition,
 *  normalisation, capture id, experiment name) so the file is self-describing. */
import type { EemResult, LineProfile, NormSpec, ResponseCurve, RoiSpec, RoiStats } from '../api/types'
import { metricLabel } from './metrics'
import type { AnyRoi } from './types'

export interface ExportContext {
  captureId: string
  experimentName: string
  norm: NormSpec
  roiName?: string
  roi?: AnyRoi | RoiSpec | null
  extra?: Record<string, string | number>
}

const esc = (v: unknown): string => {
  if (v == null) return ''
  if (typeof v === 'number') return Number.isFinite(v) ? String(+v.toPrecision(8)) : ''
  const s = String(v)
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
}
export const csvRow = (cells: unknown[]) => cells.map(esc).join(',')

export function csvHeader(title: string, unit: string, c: ExportContext): string[] {
  const norm = [c.norm.mode, c.norm.dark_capture_id ? `dark=${c.norm.dark_capture_id}` : '', c.norm.white_capture_id ? `white=${c.norm.white_capture_id}` : ''].filter(Boolean).join(' ')
  const lines = [
    `# SpectralForge Studio — ${title}`,
    `# experiment: ${c.experimentName}`,
    `# capture: ${c.captureId}`,
    `# normalisation: ${norm}`,
    `# unit: ${unit}`,
  ]
  if (c.roi) lines.push(`# roi${c.roiName ? ` (${c.roiName})` : ''}: ${JSON.stringify(c.roi)}`)
  for (const [k, v] of Object.entries(c.extra ?? {})) lines.push(`# ${k}: ${v}`)
  return lines
}

/** Long-format EEM table: one row per (excitation, band). */
export function eemToCsv(r: EemResult, c: ExportContext): string {
  const rows = [...csvHeader('EEM (ROI statistic per band)', r.unit, c), `# n_pixels: ${r.n_pixels}`, csvRow(['excitation_nm', 'emission_nm', `value_${r.unit}`, `std_${r.unit}`])]
  for (const ex of r.excitations_nm) {
    const k = String(ex), em = r.emission_nm[k] ?? []
    em.forEach((nm, i) => rows.push(csvRow([ex, nm, r.value[k]?.[i], r.std[k]?.[i]])))
  }
  return rows.join('\n') + '\n'
}

export function roiStatsToCsv(r: RoiStats, c: ExportContext): string {
  const rows = [...csvHeader('ROI ground-truth statistics', 'see column', c), `# n_pixels: ${r.n_pixels}`, csvRow(['ground_truth', 'mean', 'std', 'min', 'max'])]
  for (const [name, s] of Object.entries(r.ground_truth)) rows.push(csvRow([name, s.mean, s.std, s.min, s.max]))
  return rows.join('\n') + '\n'
}

/** Wide table: one row per sample along the line, one column per band of the chosen excitation(s) + ground-truth traces. */
export function lineToCsv(p: LineProfile, c: ExportContext): string {
  const exs = Object.keys(p.value)
  const gts = Object.keys(p.ground_truth)
  const head = ['distance_px', ...(p.distance_mm ? ['distance_mm'] : [])]
  for (const ex of exs) (p.emission_nm[ex] ?? []).forEach((nm) => head.push(`ex${ex}_em${nm}`))
  head.push(...gts)
  const rows = [...csvHeader('line profile', p.unit, c), csvRow(head)]
  p.distance_px.forEach((d, i) => {
    const cells: unknown[] = [d, ...(p.distance_mm ? [p.distance_mm[i]] : [])]
    for (const ex of exs) cells.push(...(p.value[ex]?.[i] ?? []))
    for (const g of gts) cells.push(p.ground_truth[g]?.[i])
    rows.push(csvRow(cells))
  })
  return rows.join('\n') + '\n'
}

export function responseToCsv(r: ResponseCurve, c: ExportContext): string {
  const rows = [...csvHeader(`response: ${metricLabel(r.metric)} vs ${r.x_ground_truth}`, r.unit, c), csvRow([r.x_ground_truth, 'mean', 'std', 'count'])]
  r.bin_centres.forEach((x, i) => rows.push(csvRow([x, r.mean[i], r.std[i], r.count[i]])))
  rows.push('', '# scatter (subsample)', csvRow([r.x_ground_truth, 'metric']))
  r.scatter_x.forEach((x, i) => rows.push(csvRow([x, r.scatter_y[i]])))
  return rows.join('\n') + '\n'
}

export interface SweepRow { index: number; value: number | string | boolean; metric: number | null; std?: number | null; captureId: string | null }
export function sweepToCsv(rows: SweepRow[], opt: { parameterPath: string; parameterLabel: string; metricName: string; unit: string }, c: ExportContext): string {
  const out = [...csvHeader(`sweep of ${opt.parameterLabel}`, opt.unit, { ...c, extra: { ...c.extra, parameter_path: opt.parameterPath, metric: opt.metricName } }),
    csvRow(['index', opt.parameterPath, opt.metricName, 'capture_id'])]
  rows.forEach((r) => out.push(csvRow([r.index, r.value, r.metric, r.captureId])))
  return out.join('\n') + '\n'
}

export function toJson(kind: string, c: ExportContext, data: unknown): string {
  return JSON.stringify({ kind, spectraforge_studio: 'v1.1', experiment: c.experimentName, capture: c.captureId, normalisation: c.norm, roi: c.roi ?? null, roi_name: c.roiName ?? null, ...(c.extra ? { extra: c.extra } : {}), data }, (_k, v) => (typeof v === 'number' && !Number.isFinite(v) ? null : v), 1)
}

/** Blob download (no server round trip). */
export function downloadText(filename: string, text: string, mime = 'text/csv'): void {
  const blob = new Blob([text], { type: `${mime};charset=utf-8` })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob); a.download = filename
  document.body.appendChild(a); a.click(); a.remove()
  setTimeout(() => URL.revokeObjectURL(a.href), 4000)
}

export const safeName = (s: string) => s.replace(/[^\w.-]+/g, '_').replace(/^_+|_+$/g, '') || 'export'
