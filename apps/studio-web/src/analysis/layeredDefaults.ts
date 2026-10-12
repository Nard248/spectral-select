/**
 * Sensible analysis defaults for captures that contain a layered block.
 *
 * Physics QA finding: the first camera band (ex + 20 nm) is dominated by elastic excitation leakage, while
 * ex + 40 nm is clean fluorescence.  For a buried fluorophore the useful readout is therefore
 *   - the excitation closest to the fluorophore's absorption peak,
 *   - the emission band at ≈ ex + 40 nm,
 *   - irradiance-normalised values (the lamp's spatial gradient divided out).
 */
import type { Experiment } from '../generated/experiment'
import type { CaptureMeta, MetricSpec } from '../api/types'
import type { CaptureMetaX } from './types'
import { fluorophoreMap } from '../state/presets'

/** Offset of the default readout band above the excitation line (nm). */
export const BAND_OFFSET_NM = 40
/** A band closer than this to its excitation is flagged as "contains scattered excitation light". */
export const LEAK_WINDOW_NM = 30

export const isLayeredCapture = (meta: CaptureMeta | null | undefined): boolean => !!(meta as CaptureMetaX | null | undefined)?.blocks?.length

const nearest = (arr: number[], v: number) => arr.reduce((b, e) => (Math.abs(e - v) < Math.abs(b - v) ? e : b), arr[0])

/**
 * Absorption peak (nm) of the buried fluorophore of the first layered block: the dominant dye of the layer named
 * "...fluor..." if there is one, else of the most concentrated dyed layer below the top (cover) layer.
 */
export function buriedFluorophorePeak(exp: Experiment | null | undefined): number | null {
  if (!exp) return null
  const mats = new Map(exp.materials.map((m) => [m.id, m]))
  const fl = new Map(Object.entries(fluorophoreMap(exp)))
  for (const o of exp.objects) {
    if (o.shape.kind !== 'layered_block') continue
    const layers = o.shape.layers
    const cands = layers.map((layer, i) => {
      const dyes = (mats.get(layer.material_id)?.dyes ?? []).filter((d) => d.concentration_uM > 0 && fl.has(d.fluorophore))
      const top = dyes.slice().sort((x, y) => y.concentration_uM - x.concentration_uM)[0]
      const total = dyes.reduce((acc, d) => acc + d.concentration_uM, 0)
      const score = !top ? -1 : (/fluor/i.test(layer.name) ? 1e9 : 0) + (i < layers.length - 1 || layers.length === 1 ? total : 0)
      return { top, score }
    }).filter((c) => c.top && c.score > 0)
    cands.sort((x, y) => y.score - x.score)
    if (cands.length) return fl.get(cands[0].top!.fluorophore)!.ex_peak_nm
  }
  return null
}

/** Default (excitation, ≈ex+40 nm band) metric for a layered capture; null if the capture has no block. */
export function layeredMetric(meta: CaptureMeta, exp?: Experiment | null): MetricSpec | null {
  if (!isLayeredCapture(meta) || !meta.excitations_nm.length) return null
  const peak = buriedFluorophorePeak(exp)
  const ex = peak != null ? nearest(meta.excitations_nm, peak) : meta.excitations_nm[0]
  const bands = meta.emission_nm[String(ex)] ?? []
  if (!bands.length) return null
  return { kind: 'band', ex, band_nm: nearest(bands, ex + BAND_OFFSET_NM) }
}

/** Does this metric read a band so close to its excitation that scattered excitation light dominates it? */
export function readsLeakBand(m: MetricSpec | null | undefined): boolean {
  if (!m) return false
  if (m.kind === 'band') return m.band_nm - m.ex < LEAK_WINDOW_NM
  if (m.kind === 'band_ratio') return m.a.band_nm - m.a.ex < LEAK_WINDOW_NM || m.b.band_nm - m.b.ex < LEAK_WINDOW_NM
  return false
}

export const LEAK_HINT = 'first band near the excitation contains scattered excitation light'
