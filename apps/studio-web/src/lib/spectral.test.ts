import { describe, expect, it } from 'vitest'
import { bandCentres, eem, emissionShape, epsilon, excitationProfile, gaussian, lineAveragedExcitation, muaDye, musPrime, verticalFovRad, EXTINCTION_REFERENCE } from './spectral'
import { BUNDLE, fluorophoreMap } from '../state/presets'
import { wavelengthToRGB } from './color'

const fl = fluorophoreMap({ fluorophores: [] })

describe('spectral math (consistent with the Python engine)', () => {
  it('gaussian has unit peak and half-maximum at ±FWHM/2', () => {
    expect(gaussian(500, 500, 30)).toBe(1)
    expect(gaussian(515, 500, 30)).toBeCloseTo(0.5, 6)
    expect(gaussian(485, 500, 30)).toBeCloseTo(0.5, 6)
  })
  it('ε = extinction · 55000 · excitation profile', () => {
    const f = fl['fluorescein']
    expect(EXTINCTION_REFERENCE).toBe(55000)
    expect(epsilon(f, f.ex_peak_nm)).toBeCloseTo(f.extinction * 55000, 6)
    expect(epsilon(f, f.ex_peak_nm + f.ex_fwhm_nm / 2)).toBeCloseTo(f.extinction * 55000 * 0.5, 4)
    expect(excitationProfile(f, f.ex_peak_nm)).toBe(1)
  })
  it('μa = ln10 · ε · c / 10 (1/mm) with c in µM', () => {
    const f = fl['fluorescein']
    const c = 10 // µM
    expect(muaDye(f, f.ex_peak_nm, c)).toBeCloseTo((Math.LN10 * f.extinction * 55000 * 10e-6) / 10, 9)
    expect(muaDye(f, f.ex_peak_nm, 0)).toBe(0)
    expect(muaDye(f, f.ex_peak_nm, 2 * c)).toBeCloseTo(2 * muaDye(f, f.ex_peak_nm, c), 12)
  })
  it('μs′ follows a·(λ/500)^-b and vanishes for the clear model', () => {
    const s = { model: 'power_law' as const, mus_prime_500_per_mm: 2, power_b: 1.5, f_rayleigh: 0, g: 0.9 }
    expect(musPrime(s, 500)).toBe(2)
    expect(musPrime(s, 1000)).toBeCloseTo(2 * 2 ** -1.5, 9)
    expect(musPrime({ ...s, model: 'none' }, 500)).toBe(0)
  })
  it('emission shape peaks at em_peak for symmetric dyes and is red-skewed otherwise', () => {
    const f = { ...fl['fluorescein'], em_skew: 0, vibronic: 0 }
    expect(emissionShape(f, f.em_peak_nm)).toBe(1)
    const sk = { ...f, em_skew: 0.5 }
    expect(emissionShape(sk, sk.em_peak_nm + 30)).toBeGreaterThan(emissionShape(sk, sk.em_peak_nm - 30))
  })
  it('line-averaged excitation approaches the point value as the line narrows', () => {
    const f = fl['EGFP']
    expect(lineAveragedExcitation(f, f.ex_peak_nm, 0.1)).toBeCloseTo(1, 3)
    expect(lineAveragedExcitation(f, f.ex_peak_nm, 60)).toBeLessThan(0.95)
  })
  it('band centres follow the excitation_offset (Nuance) policy', () => {
    expect(bandCentres(420, 10, 3, 'excitation_offset', 20, 400)).toEqual([420, 430, 440])
    expect(bandCentres(420, 10, 3, 'excitation_offset', 20, 430)).toEqual([450, 460, 470])
    expect(bandCentres(420, 10, 3, 'fixed', 20, 430)).toEqual([420, 430, 440])
  })
  it('vertical FOV = 2 atan(sensor_h / 2v), v = f s / (s − f)', () => {
    const f = 35, s = 300, h = 256, pitch = 6.45, bin = 4
    const v = (f * s) / (s - f), sh = (h * pitch * bin) / 1000
    expect(verticalFovRad(f, s, h, pitch, bin)).toBeCloseTo(2 * Math.atan(sh / (2 * v)), 12)
  })
  it('EEM is peak-normalised and peaks near the dye excitation/emission', () => {
    const m = BUNDLE.materials.find((x) => x.id === 'fluorescein_10uM')!
    const exs = [400, 450, 490, 540], ems = [500, 520, 540, 600, 650]
    const e = eem(m, fl, exs, ems)
    expect(Math.max(...e.flat())).toBeCloseTo(1, 9)
    expect(e[2].indexOf(Math.max(...e[2]))).toBeLessThan(4)   // 490 nm excites around the green emission
    expect(e[2][1]).toBeGreaterThan(e[0][1])
  })
  it('wavelength colours: UV is violet, 530 is green, 650 is red', () => {
    const [ur, , ub] = wavelengthToRGB(365); expect(ub).toBeGreaterThan(ur)
    const g = wavelengthToRGB(530); expect(g[1]).toBeGreaterThan(g[0]); expect(g[1]).toBeGreaterThan(g[2])
    const r = wavelengthToRGB(650); expect(r[0]).toBeGreaterThan(r[1])
  })
})
