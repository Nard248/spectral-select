import { describe, expect, it } from 'vitest'
import { DEFAULT_TISSUE, muaMelanosome, musPrimeAt, penetrationDepth, tissueMua, tissueMuaTerms, tissueWarnings } from './tissueSpectra'
import { materialMua, musPrime } from './spectral'
import type { MaterialSpec, TissueComposition } from '../generated/experiment'

const T = (o: Partial<TissueComposition>): TissueComposition => ({ ...DEFAULT_TISSUE, blood_volume_fraction: 0, water_fraction: 0, lipid_fraction: 0, ...o })

describe('tissue μa anchors', () => {
  it('melanosome follows Jacques 6.6e11·λ^-3.33 cm⁻¹ (= 68.0 /mm at 500 nm, 22.7 /mm at 694 nm)', () => {
    expect(muaMelanosome(500)).toBeCloseTo(6.6e11 * 500 ** -3.33 / 10, 10)
    expect(muaMelanosome(500)).toBeGreaterThan(65); expect(muaMelanosome(500)).toBeLessThan(71)
    expect(muaMelanosome(694)).toBeGreaterThan(21); expect(muaMelanosome(694)).toBeLessThan(25)
    expect(tissueMua(T({ melanosome_fraction: 0.03 }), 500)).toBeCloseTo(0.03 * muaMelanosome(500), 12)
  })
  it('oxygenated whole blood (150 g/L) absorbs ≈ 270 cm⁻¹ = 27 /mm at the 576 nm α band; Hb ≫ HbO2 at 660 nm; isosbestic near 800 nm', () => {
    const ox = tissueMua(T({ blood_volume_fraction: 1, oxygen_saturation: 1 }), 576)
    expect(ox).toBeGreaterThan(24); expect(ox).toBeLessThan(31)
    const hb = (s: number, wl: number) => tissueMua(T({ blood_volume_fraction: 1, oxygen_saturation: s }), wl)
    expect(hb(0, 660) / hb(1, 660)).toBeGreaterThan(5)
    expect(Math.abs(hb(0, 800) - hb(1, 800)) / hb(1, 800)).toBeLessThan(0.15)
  })
  it('water: ≈ 0.0006 /mm at 500 nm… rising to ≈ 0.0068 /mm at 900 nm; sums linearly with its fraction', () => {
    expect(tissueMua(T({ water_fraction: 1 }), 900)).toBeCloseTo(0.0068, 4)
    expect(tissueMua(T({ water_fraction: 0.5 }), 900)).toBeCloseTo(0.5 * tissueMua(T({ water_fraction: 1 }), 900), 12)
  })
  it('terms add up to the total and bilirubin / β-carotene absorb in the blue', () => {
    const t = T({ blood_volume_fraction: 0.02, water_fraction: 0.6, lipid_fraction: 0.1, melanosome_fraction: 0.01, bilirubin_uM: 100, beta_carotene_uM: 5 })
    const x = tissueMuaTerms(t, 450)
    expect(x.hbo2 + x.hb + x.water + x.lipid + x.melanosome + x.bilirubin + x.carotene).toBeCloseTo(x.total, 12)
    expect(x.bilirubin).toBeGreaterThan(tissueMuaTerms(t, 650).bilirubin * 50)
  })
  it('materialMua adds the tissue term to the background absorber', () => {
    const m = { tissue: T({ water_fraction: 1 }), background_absorption: { preset: 'none', scale: 1, custom: [] }, dyes: [] } as unknown as MaterialSpec
    expect(materialMua(m, {}, 700)).toBeCloseTo(tissueMua(m.tissue!, 700), 12)
  })
})

describe('scattering + penetration depth', () => {
  const base = { model: 'mie_rayleigh' as const, mus_prime_500_per_mm: 4, power_b: 1, f_rayleigh: 0.4 }
  it('Mie+Rayleigh follows Jacques eq. 2 and reduces to the power law at f = 0', () => {
    expect(musPrimeAt(base, 500)).toBeCloseTo(4, 12)
    expect(musPrimeAt(base, 1000)).toBeCloseTo(4 * (0.4 * 2 ** -4 + 0.6 * 2 ** -1), 12)
    expect(musPrimeAt({ ...base, f_rayleigh: 0 }, 1000)).toBeCloseTo(4 * 0.5, 12)
    expect(musPrime({ ...base, g: 0.9 }, 700)).toBeCloseTo(musPrimeAt(base, 700), 12)
    expect(musPrimeAt({ ...base, model: 'none' }, 700)).toBe(0)
  })
  it('δ = 1/√(3 μa (μa + μs′)) (μa = 0.1, μs′ = 10 → 0.574 mm) and ∞ without absorption', () => {
    expect(penetrationDepth(0.1, 10)).toBeCloseTo(1 / Math.sqrt(3 * 0.1 * 10.1), 12)
    expect(penetrationDepth(0.1, 10)).toBeCloseTo(0.5745, 3)
    expect(penetrationDepth(0, 10)).toBe(Infinity)
  })
  it('warns when water + lipid exceed 100 %', () => {
    expect(tissueWarnings(T({ water_fraction: 0.7, lipid_fraction: 0.5 })).length).toBeGreaterThan(0)
    expect(tissueWarnings(T({ water_fraction: 0.6, lipid_fraction: 0.2 }))).toEqual([])
  })
})
