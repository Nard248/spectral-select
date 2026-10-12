/** Client-side spectral math, kept consistent with the Python engine
 *  (`spectraforge.fluorophore.Fluorophore`, `studio/optics.py`).
 *  ε(λ) = extinction · 55000 · exc(λ)  [M⁻¹cm⁻¹];  μa = ln10 · ε · c[M] / 10  [1/mm]. */
import type { AbsorberSpec, FluorophoreSpec, MaterialSpec, ScatteringSpec } from '../generated/experiment'

export const EXTINCTION_REFERENCE = 55000
export const FWHM_TO_SIGMA = 1 / 2.3548200450309493

export const range = (a: number, b: number, step: number): number[] => {
  const out: number[] = []
  for (let x = a; x <= b + 1e-9; x += step) out.push(x)
  return out
}

/** Peak-normalised Gaussian. */
export const gaussian = (x: number, mu: number, fwhm: number): number => {
  const s = fwhm * FWHM_TO_SIGMA
  return Math.exp(-0.5 * ((x - mu) / s) ** 2)
}

export const excitationProfile = (f: FluorophoreSpec, wl: number): number => gaussian(wl, f.ex_peak_nm, f.ex_fwhm_nm)

/** Un-normalised emission shape, peak ≈ 1 (skew + vibronic shoulder as in Python). */
export function emissionShape(f: FluorophoreSpec, wl: number): number {
  const sigma = f.em_fwhm_nm * FWHM_TO_SIGMA
  const d = wl - f.em_peak_nm
  if (f.em_skew === 0 && f.vibronic === 0) return Math.exp(-0.5 * (d / sigma) ** 2)
  const sRed = sigma * (1 + f.em_skew)
  const sBlue = sigma * Math.max(0.2, 1 - 0.4 * f.em_skew)
  const s = d >= 0 ? sRed : sBlue
  let g = Math.exp(-0.5 * (d / s) ** 2)
  if (f.vibronic > 0) {
    const spacing = f.em_peak_nm ** 2 * 1300e-7
    g += f.vibronic * Math.exp(-0.5 * ((wl - (f.em_peak_nm + spacing)) / sRed) ** 2)
  }
  return g
}

/** Molar extinction ε(λ) in M⁻¹cm⁻¹. */
export const epsilon = (f: FluorophoreSpec, wl: number): number =>
  f.extinction * EXTINCTION_REFERENCE * excitationProfile(f, wl)

/** Dye absorption coefficient μa (1/mm) for concentration in µM. */
export const muaDye = (f: FluorophoreSpec, wl: number, c_uM: number): number =>
  (Math.LN10 * epsilon(f, wl) * c_uM * 1e-6) / 10

/** Reduced scattering μs′(λ) = a·(λ/500)^-b (1/mm). */
export const musPrime = (s: ScatteringSpec, wl: number): number =>
  s.model === 'none' ? 0 : s.mus_prime_500_per_mm * (wl / 500) ** -s.power_b

/** Approximate tabulated absorbers (1/mm) — shape-only stand-ins for the engine tables. */
export function absorberMua(a: AbsorberSpec, wl: number): number {
  switch (a.preset) {
    case 'none': return 0
    case 'water': {
      // coarse pure-water absorption: low in blue/green, rises beyond 600 nm, large peaks at 980 nm
      const base = 0.00005 + 0.00045 * Math.exp((wl - 600) / 42) * (wl > 560 ? 1 : 0.15)
      return a.scale * (base + 0.04 * gaussian(wl, 975, 40))
    }
    case 'hemoglobin':
      return a.scale * (0.1 + 3.2 * gaussian(wl, 415, 30) + 0.9 * gaussian(wl, 542, 22) + 0.85 * gaussian(wl, 576, 18)) / 4
        * (wl > 620 ? Math.exp(-(wl - 620) / 45) : 1) * 1.0
    case 'melanin': return a.scale * (wl / 500) ** -3.4
    case 'carbon_black': return a.scale * (wl / 500) ** -0.2
    case 'custom': {
      const pts = a.custom
      if (!pts.length) return 0
      if (wl <= pts[0][0]) return pts[0][1]
      for (let i = 1; i < pts.length; i++) {
        if (wl <= pts[i][0]) {
          const t = (wl - pts[i - 1][0]) / (pts[i][0] - pts[i - 1][0] || 1)
          return pts[i - 1][1] * (1 - t) + pts[i][1] * t
        }
      }
      return pts[pts.length - 1][1]
    }
  }
}

export function materialMua(m: MaterialSpec, fl: Record<string, FluorophoreSpec>, wl: number): number {
  let mua = absorberMua(m.background_absorption, wl)
  for (const d of m.dyes) {
    const f = fl[d.fluorophore]
    if (f) mua += muaDye(f, wl, d.concentration_uM)
  }
  return mua
}

/** Excitation line profile (Gaussian, FWHM `fwhm`) averaged against a dye's absorption — what a
 *  monochromator-limited line actually excites. */
export function lineAveragedExcitation(f: FluorophoreSpec, exNm: number, fwhm: number): number {
  const s = Math.max(fwhm * FWHM_TO_SIGMA, 1e-6)
  let num = 0, den = 0
  for (let k = -3; k <= 3; k += 0.5) {
    const w = Math.exp(-0.5 * k * k)
    num += w * excitationProfile(f, exNm + k * s)
    den += w
  }
  return num / den
}

/** Excitation–emission matrix (rows = excitation, cols = emission), peak-normalised, for a material. */
export function eem(m: MaterialSpec, fl: Record<string, FluorophoreSpec>, exs: number[], ems: number[]): number[][] {
  const out = exs.map(() => new Array<number>(ems.length).fill(0))
  for (const d of m.dyes) {
    const f = fl[d.fluorophore]
    if (!f) continue
    const amp = f.extinction * f.quantum_yield * d.concentration_uM
    exs.forEach((ex, i) => {
      const e = excitationProfile(f, ex) * amp
      ems.forEach((em, j) => { out[i][j] += e * emissionShape(f, em) })
    })
  }
  const mx = Math.max(1e-12, ...out.flat())
  return out.map((r) => r.map((v) => v / mx))
}

/** Camera passband centres for a spectral filter spec at a given excitation. */
export function bandCentres(start: number, step: number, n: number, policy: 'fixed' | 'excitation_offset',
  offset: number, exNm: number | null): number[] {
  const s0 = policy === 'excitation_offset' && exNm != null ? Math.max(start, exNm + offset) : start
  return Array.from({ length: n }, (_, i) => s0 + i * step)
}

/** Vertical FOV (rad) of an HSI camera: 2·atan(sensor_h / (2 v)), v = f·s/(s−f). */
export function verticalFovRad(focalMm: number, focusMm: number, heightPx: number, pitchUm: number, binning: number): number {
  const s = Math.max(focusMm, focalMm * 1.01)
  const v = (focalMm * s) / (s - focalMm)
  const sensorH = (heightPx * pitchUm * binning) / 1000
  return 2 * Math.atan(sensorH / (2 * v))
}
export const sensorAspect = (w: number, h: number) => w / h
