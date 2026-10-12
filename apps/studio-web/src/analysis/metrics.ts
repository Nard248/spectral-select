import type { EemResult, MetricSpec } from '../api/types'

/** Linear interpolation of `y(x)` at `q` (x ascending). NaN when `q` is farther than one sample spacing outside the span. */
export function interp(x: ArrayLike<number>, y: ArrayLike<number>, q: number): number {
  const n = x.length
  if (!n) return NaN
  if (q <= x[0]) return n > 1 && x[0] - q > x[1] - x[0] ? NaN : y[0]
  if (q >= x[n - 1]) return n > 1 && q - x[n - 1] > x[n - 1] - x[n - 2] ? NaN : y[n - 1]
  let lo = 0, hi = n - 1
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (x[m] <= q) lo = m; else hi = m }
  const t = (q - x[lo]) / (x[hi] - x[lo] || 1)
  return y[lo] * (1 - t) + y[hi] * t
}

const keyOf = (r: EemResult, ex: number): string | null => {
  const k = String(ex)
  if (k in r.value) return k
  const near = r.excitations_nm.find((e) => Math.abs(e - ex) < 0.5)
  return near != null ? String(near) : null
}

/** Scalar metric from an ROI-averaged EEM (ratio of ROI means for band ratios). */
export function metricFromEem(r: EemResult, m: MetricSpec): number {
  const at = (ex: number, nm: number) => {
    const k = keyOf(r, ex); if (!k) return NaN
    return interp(r.emission_nm[k], r.value[k], nm)
  }
  switch (m.kind) {
    case 'band': return at(m.ex, m.band_nm)
    case 'band_ratio': { const b = at(m.b.ex, m.b.band_nm); return b === 0 ? NaN : at(m.a.ex, m.a.band_nm) / b }
    case 'band_sum': {
      const k = keyOf(r, m.ex); if (!k) return NaN
      const x = r.emission_nm[k], y = r.value[k]
      let s = 0
      for (let i = 0; i < x.length; i++) if (x[i] >= (m.from_nm ?? -Infinity) && x[i] <= (m.to_nm ?? Infinity)) s += y[i]
      return s
    }
    case 'peak_wavelength': {
      const k = keyOf(r, m.ex); if (!k) return NaN
      const x = r.emission_nm[k], y = r.value[k]
      let bi = 0
      for (let i = 1; i < y.length; i++) if (y[i] > y[bi]) bi = i
      if (bi > 0 && bi < y.length - 1) {   // parabolic refinement
        const a = y[bi - 1], b = y[bi], c = y[bi + 1], d = a - 2 * b + c
        if (d < 0) return x[bi] + (0.5 * (a - c)) / d * (x[bi + 1] - x[bi])
      }
      return x[bi]
    }
  }
}

export function metricLabel(m: MetricSpec): string {
  switch (m.kind) {
    case 'band': return `I(${m.ex} → ${m.band_nm} nm)`
    case 'band_ratio': return `I(${m.a.ex} → ${m.a.band_nm}) / I(${m.b.ex} → ${m.b.band_nm})`
    case 'band_sum': return `Σ I(${m.ex} nm, ${m.from_nm ?? 'min'}–${m.to_nm ?? 'max'} nm)`
    case 'peak_wavelength': return `peak λ (${m.ex} nm)`
  }
}

/** Unit of a metric given the unit of the underlying values. */
export function metricUnit(m: MetricSpec, unit: string): string {
  if (m.kind === 'band_ratio') return 'ratio'
  if (m.kind === 'peak_wavelength') return 'nm'
  return unit
}

/** Std over the ROI pixels of a single-band metric (other metrics: null). */
export function metricStdFromEem(r: EemResult, m: MetricSpec): number | null {
  if (m.kind !== 'band') return null
  const k = keyOf(r, m.ex); if (!k) return null
  const v = interp(r.emission_nm[k], r.std[k], m.band_nm)
  return Number.isFinite(v) ? v : null
}

/** Distinct excitation wavelengths a metric reads. */
export function metricExcitations(m: MetricSpec): number[] {
  return m.kind === 'band_ratio' ? [...new Set([m.a.ex, m.b.ex])] : [m.ex]
}
