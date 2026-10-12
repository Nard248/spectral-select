/** Precision-aware number formatting for physical / ground-truth values: 3 significant figures,
 *  exponential notation for |x| < 1e-3 or |x| > 1e5 (so 6.5e-5 never displays as "0"). */
export function fmtSig(v: number | null | undefined, unit = ''): string {
  if (v == null) return '—'
  if (!Number.isFinite(v)) return String(v)
  let s: string
  if (v === 0) s = '0'
  else {
    const a = Math.abs(v)
    s = a < 1e-3 || a > 1e5 ? v.toExponential(2).replace('e+', 'e') : String(+v.toPrecision(3))
  }
  return unit ? `${s} ${unit}` : s
}

/** Unit suffix inferred from a ground-truth field name (`depth_mm`, `incidence_deg`, `irradiance@310`, …). */
export function gtUnit(name: string): string {
  if (name.endsWith('_mm')) return 'mm'
  if (name.endsWith('_deg')) return '°'
  if (name.endsWith('_nm')) return 'nm'
  if (name.endsWith('_mW')) return 'mW'
  if (name.startsWith('irradiance')) return 'mW mm⁻²'
  return ''
}
