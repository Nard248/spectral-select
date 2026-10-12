/** Approximate fluorescence "glow" of a material under one excitation, for the live preview shading:
 *  raw = Σ_dyes (1 − 10^(−ε_line·c·L)) · Φ · quench, colour = emission spectrum integrated against RGB. */
import type { Experiment, FluorophoreSpec, LightSourceSpec, MaterialSpec } from '../generated/experiment'
import { beamAt } from './lighting'
import { wavelengthToRGB } from './color'
import { EXTINCTION_REFERENCE, emissionShape, lineAveragedExcitation } from './spectral'

export interface Glow { color: string | null; intensity: number }

const rgbCache = new Map<string, [number, number, number]>()
/** Colour of a fluorophore's emission (area-integrated against the display RGB of each wavelength). */
export function emissionRGB(f: FluorophoreSpec): [number, number, number] {
  const key = `${f.name}|${f.em_peak_nm}|${f.em_fwhm_nm}|${f.em_skew}|${f.vibronic}`
  let c = rgbCache.get(key)
  if (c) return c
  let r = 0, g = 0, b = 0
  for (let wl = f.em_peak_nm - 3 * f.em_fwhm_nm; wl <= f.em_peak_nm + 6 * f.em_fwhm_nm; wl += 4) {
    if (wl < 380 || wl > 780) continue
    const w = emissionShape(f, wl), [cr, cg, cb] = wavelengthToRGB(wl)
    r += w * cr; g += w * cg; b += w * cb
  }
  const m = Math.max(r, g, b, 1e-9)
  c = [r / m, g / m, b / m]
  rgbCache.set(key, c)
  return c
}

const toHex = (c: [number, number, number]) => '#' + c.map((v) => Math.round(Math.min(1, Math.max(0, v)) * 255).toString(16).padStart(2, '0')).join('')

export function materialGlowRaw(m: MaterialSpec, fl: Record<string, FluorophoreSpec>, exNm: number, lineFwhm: number, withAbsorption = true) {
  let raw = 0; const rgb: [number, number, number] = [0, 0, 0]
  for (const d of m.dyes) {
    const f = fl[d.fluorophore]; if (!f) continue
    const eps = f.extinction * EXTINCTION_REFERENCE * lineAveragedExcitation(f, exNm, lineFwhm)
    const absP = withAbsorption ? 1 - Math.pow(10, -eps * d.concentration_uM * 1e-6 * 0.3) : 0.05
    const w = absP * f.quantum_yield / (1 + m.quenching_ksv_per_uM * d.concentration_uM)
    raw += w
    const c = emissionRGB(f)
    rgb[0] += w * c[0]; rgb[1] += w * c[1]; rgb[2] += w * c[2]
  }
  return { raw, rgb }
}

/** Glow for an object: material fluorescence × Gaussian beam footprint at the object centre. */
export function objectGlow(exp: Experiment, obj: Experiment['objects'][number], mat: MaterialSpec | undefined, src: LightSourceSpec | undefined,
  fl: Record<string, FluorophoreSpec>, exNm: number | null, powerMW = 3): Glow {
  if (!mat || !src || exNm == null || !exp.acquisition.physics.fluorescence) return { color: null, intensity: 0 }
  const { raw, rgb } = materialGlowRaw(mat, fl, exNm, src.line_fwhm_nm, exp.acquisition.physics.absorption)
  if (raw < 1e-5) return { color: null, intensity: 0 }
  const p = obj.transform.position_mm
  const irr = beamAt(src, [p[0], p[1] + 2, p[2]], [0, 1, 0]).irradiance
  const m = Math.max(...rgb, 1e-9)
  const intensity = 1 - Math.exp(-raw * irr * powerMW * 5.5)
  return { color: toHex([rgb[0] / m, rgb[1] / m, rgb[2] / m]), intensity }
}
