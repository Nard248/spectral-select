/** Client-side tissue optics: a coarse-grained copy of the chromophore spectra behind `MaterialSpec.tissue`, used only for the
 *  live optical-properties chart and the mock engine. The authoritative tables live in the Python engine
 *  (`spectraforge.studio.tissue_optics`); here every spectrum is tabulated on a 15-25 nm grid and interpolated linearly in
 *  log10, which keeps the curves within roughly 10-15 % of the engine over 350-900 nm (shape-faithful, not a data source).
 *
 *  Absorption of the composed tissue (Jacques 2013, Phys. Med. Biol. 58, R37, eq. 1; all in 1/mm):
 *    μa(λ) = B·S·μHbO2 + B·(1-S)·μHb + W·μwater + F·μfat + M·μmel + ln10·(ε_bili·c_bili + ε_βcar·c_βcar)/10
 *  with μHb(O2) = ln10·ε(λ)·[Hb]/(64 500 g/mol)/10 for whole blood at `hemoglobin_g_per_L`.
 *  Sources: haemoglobin ε (cm⁻¹/M) Prahl compilation (Takatani & Graham 1979; Gratzer & Kollias), omlc.org/spectra/hemoglobin;
 *  water Hale & Querry 1973, Appl. Opt. 12, 555 / Pope & Fry 1997, Appl. Opt. 36, 8710; lipid van Veen et al. 2005,
 *  J. Biomed. Opt. 10, 054004; melanosome μa = 6.6e11·λ^-3.33 cm⁻¹ (Jacques 1998, omlc.org/spectra/melanin); bilirubin and
 *  β-carotene as Gaussian band models of Prahl's solvent spectra.
 *  Scattering (Jacques 2013 eq. 2): μs′ = a′[f_ray (λ/500)^-4 + (1-f_ray)(λ/500)^-b_Mie]. */
import type { ScatteringSpec, TissueComposition } from '../generated/experiment'

const GRID: number[] = [350, 375, 400, 410, 420, 430, 440, 450, 475, 500, 525, 542, 560, 576, 590, 600, 625, 650, 675, 700, 725, 750, 775, 800, 825, 850, 875, 900]
/** molar extinction ε, cm⁻¹/M */
const E_HBO2 = [60000, 110000, 266000, 460000, 480000, 360000, 200000, 62000, 28000, 20900, 39000, 53200, 33000, 55000, 14000, 3200, 900, 368, 300, 290, 380, 518, 680, 816, 940, 1058, 1130, 1198]
const E_HB = [70000, 130000, 223000, 340000, 420000, 528000, 470000, 103000, 40000, 20900, 40000, 50500, 53000, 38000, 28000, 14700, 7000, 3750, 2300, 1794, 1500, 1405, 1100, 761, 700, 691, 720, 762]
/** pure water absorption, cm⁻¹ */
const W_WATER = [0.0003, 0.0004, 0.0006, 0.0005, 0.0005, 0.0004, 0.0004, 0.0004, 0.0003, 0.0003, 0.0004, 0.0005, 0.0006, 0.0009, 0.0015, 0.0022, 0.0028, 0.0034, 0.0045, 0.006, 0.012, 0.026, 0.022, 0.020, 0.030, 0.043, 0.055, 0.068]
/** pure lipid absorption, cm⁻¹ (approximate envelope of van Veen 2005) */
const W_FAT = [3.0, 2.2, 1.5, 1.3, 1.1, 1.0, 0.85, 0.6, 0.4, 0.25, 0.17, 0.13, 0.1, 0.085, 0.075, 0.07, 0.055, 0.045, 0.043, 0.045, 0.055, 0.06, 0.055, 0.05, 0.06, 0.08, 0.13, 0.2]

export const HB_G_PER_MOL = 64500
export const WL_MIN = 350, WL_MAX = 900

function logInterp(xs: number[], ys: number[], x: number): number {
  if (x <= xs[0]) return ys[0]
  if (x >= xs[xs.length - 1]) return ys[ys.length - 1]
  let i = 1
  while (xs[i] < x) i++
  const t = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
  return 10 ** (Math.log10(ys[i - 1]) * (1 - t) + Math.log10(ys[i]) * t)
}

/** ε of oxy- / deoxy-haemoglobin (cm⁻¹/M) */
export const epsHbO2 = (wl: number) => logInterp(GRID, E_HBO2, wl)
export const epsHb = (wl: number) => logInterp(GRID, E_HB, wl)
/** μa of pure water / lipid / melanosome (1/mm) */
export const muaWater = (wl: number) => logInterp(GRID, W_WATER, wl) / 10
export const muaFat = (wl: number) => logInterp(GRID, W_FAT, wl) / 10
/** Jacques melanosome: 6.6e11·λ^-3.33 cm⁻¹ → 1/mm */
export const muaMelanosome = (wl: number) => (6.6e11 * wl ** -3.33) / 10

const g = (x: number, mu: number, fwhm: number) => Math.exp(-2.772588722 * ((x - mu) / fwhm) ** 2)
/** bilirubin ε (M⁻¹cm⁻¹): band at ~452 nm */
export const epsBilirubin = (wl: number) => 55000 * g(wl, 452, 62) + 9000 * g(wl, 400, 50)
/** β-carotene ε (M⁻¹cm⁻¹): vibronic triplet 425 / 452 / 480 nm */
export const epsBetaCarotene = (wl: number) => 50000 * g(wl, 425, 26) + 140000 * g(wl, 452, 30) + 120000 * g(wl, 480, 30)

const LN10_10 = Math.LN10 / 10

export interface TissueMuaTerms {
  hbo2: number; hb: number; water: number; lipid: number; melanosome: number; bilirubin: number; carotene: number; total: number
}
/** Each chromophore's μa contribution (1/mm) at `wl`, and their sum. */
export function tissueMuaTerms(t: TissueComposition, wl: number): TissueMuaTerms {
  const hbM = t.hemoglobin_g_per_L / HB_G_PER_MOL
  const B = t.blood_volume_fraction, S = t.oxygen_saturation
  const hbo2 = B * S * LN10_10 * epsHbO2(wl) * hbM
  const hb = B * (1 - S) * LN10_10 * epsHb(wl) * hbM
  const water = t.water_fraction * muaWater(wl)
  const lipid = t.lipid_fraction * muaFat(wl)
  const melanosome = t.melanosome_fraction * muaMelanosome(wl)
  const bilirubin = LN10_10 * epsBilirubin(wl) * t.bilirubin_uM * 1e-6
  const carotene = LN10_10 * epsBetaCarotene(wl) * t.beta_carotene_uM * 1e-6
  return { hbo2, hb, water, lipid, melanosome, bilirubin, carotene, total: hbo2 + hb + water + lipid + melanosome + bilirubin + carotene }
}
export const tissueMua = (t: TissueComposition, wl: number): number => tissueMuaTerms(t, wl).total

/** Jacques 2013 eq. 2 (and the plain power law as the f_ray = 0 case), 1/mm. */
export function musPrimeAt(s: Pick<ScatteringSpec, 'model' | 'mus_prime_500_per_mm' | 'power_b' | 'f_rayleigh'>, wl: number): number {
  if (s.model === 'none') return 0
  const x = wl / 500
  if (s.model === 'mie_rayleigh') return s.mus_prime_500_per_mm * (s.f_rayleigh * x ** -4 + (1 - s.f_rayleigh) * x ** -s.power_b)
  return s.mus_prime_500_per_mm * x ** -s.power_b
}

/** Diffusion-theory 1/e penetration depth δ = 1/√(3 μa (μa + μs′)) in mm (Jacques 2013 eq. 3 region); ∞ for a clear medium. */
export function penetrationDepth(mua: number, musp: number): number {
  const d = 3 * mua * (mua + musp)
  return d > 0 ? 1 / Math.sqrt(d) : Infinity
}

export const DEFAULT_TISSUE: TissueComposition = {
  blood_volume_fraction: 0.02, oxygen_saturation: 0.75, hemoglobin_g_per_L: 150,
  water_fraction: 0.65, lipid_fraction: 0.1, melanosome_fraction: 0, bilirubin_uM: 0, beta_carotene_uM: 0,
}

/** Sanity warnings of a composition (empty = fine). */
export function tissueWarnings(t: TissueComposition): string[] {
  const w: string[] = []
  if (t.water_fraction + t.lipid_fraction > 1 + 1e-9) w.push(`Water + lipid = ${((t.water_fraction + t.lipid_fraction) * 100).toFixed(0)} % exceeds 100 % of the volume.`)
  if (t.water_fraction + t.lipid_fraction + t.blood_volume_fraction + t.melanosome_fraction > 1 + 1e-9 && t.water_fraction + t.lipid_fraction <= 1)
    w.push('Water + lipid + blood + melanosome fractions exceed 100 % of the volume.')
  if (t.blood_volume_fraction > 0.3) w.push('Blood volume fraction above 30 % is unusual outside a vessel lumen.')
  if (t.melanosome_fraction > 0.43) w.push('Melanosome fraction above 43 % (darkest skin) is outside the physiological range.')
  return w
}
