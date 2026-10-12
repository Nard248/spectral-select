import type { CaptureMeta, NormSpec } from '../api/types'

export const NORM_MODES: { value: NormSpec['mode']; label: string; short: string; tip: string }[] = [
  { value: 'adu', label: 'Raw ADU', short: 'ADU', tip: 'Camera counts exactly as recorded (offset, dark current and noise included).' },
  { value: 'radiance', label: 'Radiance', short: 'radiance', tip: 'Calibrated at-sensor radiance mW mm⁻² sr⁻¹ nm⁻¹ (offset removed, gain and optics undone).' },
  { value: 'electrons', label: 'Electrons', short: 'e⁻', tip: 'Photo-electrons: (ADU − offset) × gain.' },
  { value: 'dark_subtracted', label: 'Dark-subtracted', short: 'ADU − dark', tip: 'ADU minus a dark capture of the same geometry — removes offset, dark current and fixed pattern.' },
  { value: 'exposure_power', label: 'Exposure × power', short: '/ (ms·mW)', tip: 'ADU − offset divided by exposure time and source power per excitation: makes excitations comparable (the lab pipeline).' },
  { value: 'white_normalized', label: 'White-normalised', short: 'reflectance-like', tip: '(sample − dark) / (white − dark): divides out illumination and optics using a white reference capture.' },
  { value: 'irradiance_normalized', label: 'Irradiance-normalised', short: '/ (s·E_ex)', tip: '(ADU − offset) / (exposure × surface irradiance E_ex at the pixel). Divides out the lamp\u2019s spatial gradient, so use it for signal-vs-structure studies (e.g. fluorescence vs cover thickness) on obliquely lit samples. Needs the capture\u2019s ground-truth irradiance map; pixels where E_ex is below 1e-3 of its maximum are blank (so a narrow beam\u2019s tail is not divided by ~0).' },
]

export const normLabel = (m: NormSpec['mode']) => NORM_MODES.find((x) => x.value === m)?.label ?? m
export const normShort = (m: NormSpec['mode']) => NORM_MODES.find((x) => x.value === m)?.short ?? m

export function normUnit(mode: NormSpec['mode']): string {
  switch (mode) {
    case 'adu': return 'ADU'
    case 'radiance': return 'mW mm⁻² sr⁻¹ nm⁻¹'
    case 'electrons': return 'e⁻'
    case 'dark_subtracted': return 'ADU (dark-sub.)'
    case 'exposure_power': return 'ADU ms⁻¹ mW⁻¹'
    case 'white_normalized': return 'a.u. (white-norm.)'
    case 'irradiance_normalized': return 'ADU s⁻¹ (mW mm⁻²)⁻¹'
  }
}

/** Does `ref` (dark / white) have the same geometry and spectral sampling as `sample`? Returns the list of problems. */
export function compatibility(sample: CaptureMeta, ref: CaptureMeta, role: 'dark' | 'white'): string[] {
  const out: string[] = []
  if (ref.width !== sample.width || ref.height !== sample.height) out.push(`${role} is ${ref.width}×${ref.height}, sample is ${sample.width}×${sample.height}`)
  const missing = sample.excitations_nm.filter((e) => !ref.excitations_nm.includes(e))
  if (missing.length) out.push(`${role} lacks excitation${missing.length > 1 ? 's' : ''} ${missing.join(', ')} nm`)
  for (const e of sample.excitations_nm) {
    const a = sample.emission_nm[String(e)], b = ref.emission_nm[String(e)]
    if (b && a && (a.length !== b.length || a.some((v, i) => Math.abs(v - b[i]) > 0.5))) { out.push(`${role} band set differs at ${e} nm`); break }
  }
  return out
}

/** All reasons the requested normalisation cannot be applied to `sample` ([] = fine). */
export function normProblems(norm: NormSpec, sample: CaptureMeta, all: CaptureMeta[]): string[] {
  const out: string[] = []
  const find = (id: string | null | undefined) => (id ? all.find((c) => c.id === id) ?? null : null)
  if (norm.mode === 'dark_subtracted' || norm.mode === 'white_normalized') {
    const d = find(norm.dark_capture_id)
    if (!norm.dark_capture_id) out.push('choose a dark capture')
    else if (!d) out.push('the chosen dark capture is no longer available')
    else out.push(...compatibility(sample, d, 'dark'))
  }
  if (norm.mode === 'white_normalized') {
    const w = find(norm.white_capture_id)
    if (!norm.white_capture_id) out.push('choose a white reference capture')
    else if (!w) out.push('the chosen white capture is no longer available')
    else out.push(...compatibility(sample, w, 'white'))
  }
  if (norm.mode === 'irradiance_normalized' && sample.ground_truth_layers.length && !sample.ground_truth_layers.some((n) => n.startsWith('irradiance'))) {
    out.push('this capture has no irradiance ground-truth map (re-run the capture)')
  }
  return out
}

/** Compact, stable cache key of a norm spec (references that the mode does not use are dropped). */
export function normSpecClean(n: NormSpec): NormSpec {
  const r: NormSpec = { mode: n.mode }
  if (n.mode === 'dark_subtracted' || n.mode === 'white_normalized') r.dark_capture_id = n.dark_capture_id ?? null
  if (n.mode === 'white_normalized') r.white_capture_id = n.white_capture_id ?? null
  return r
}
