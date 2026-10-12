import { describe, expect, it } from 'vitest'
import type { CaptureMetaX as CaptureMeta } from './types'
import { presetExperiment } from '../state/presets'
import { NORM_MODES, normProblems, normUnit } from './norm'
import { LEAK_HINT, buriedFluorophorePeak, isLayeredCapture, layeredMetric, readsLeakBand } from './layeredDefaults'

const meta = (over: Partial<CaptureMeta> = {}): CaptureMeta => ({
  id: 'c', created: '2026-01-01T00:00:00Z', name: 'x', mode: 'sample', width: 10, height: 8, excitations_nm: [340, 450, 500],
  emission_nm: { '340': [360, 380, 400], '450': [470, 490, 510, 530], '500': [520, 540, 560, 580] }, exposure_ms: {}, power_mW: {}, stats: {},
  saturated_fraction: {}, ground_truth_layers: [], warnings: [], timings_s: {}, camera_id: 'c', source_id: 's',
  blocks: [{ object_id: 'b', layer_names: ['a', 'b'], footprint_mm: [4, 3] }], ...over,
})

describe('irradiance_normalized mode', () => {
  it('is selectable, explained and has a unit', () => {
    const m = NORM_MODES.find((x) => x.value === 'irradiance_normalized')
    expect(m).toBeTruthy()
    expect(m!.tip).toMatch(/lamp/i)
    expect(normUnit('irradiance_normalized')).toMatch(/mW/)
    expect(normProblems({ mode: 'irradiance_normalized' }, meta({ ground_truth_layers: ['irradiance@450'] }), [])).toEqual([])
    expect(normProblems({ mode: 'irradiance_normalized' }, meta({ ground_truth_layers: ['depth_mm'] }), [])[0]).toMatch(/irradiance/)
  })
})

describe('layered-block defaults', () => {
  it('picks the ex nearest the buried fluorophore peak and the band at ≈ ex + 40 nm', () => {
    const exp = presetExperiment('layered_wedge_phantom')
    const peak = buriedFluorophorePeak(exp)!
    expect(peak).toBeGreaterThan(300)
    const m = layeredMetric(meta(), exp)!
    expect(m.kind).toBe('band')
    if (m.kind !== 'band') return
    const exs = meta().excitations_nm
    expect(m.ex).toBe(exs.reduce((b, e) => (Math.abs(e - peak) < Math.abs(b - peak) ? e : b)))
    const bands = meta().emission_nm[String(m.ex)]
    expect(m.band_nm).toBe(bands.reduce((b, e) => (Math.abs(e - (m.ex + 40)) < Math.abs(b - (m.ex + 40)) ? e : b)))
    expect(readsLeakBand(m)).toBe(false)
  })
  it('does nothing for captures without a block, and flags near-excitation bands', () => {
    expect(layeredMetric(meta({ blocks: [] }), presetExperiment('layered_wedge_phantom'))).toBeNull()
    expect(isLayeredCapture(meta({ blocks: undefined }))).toBe(false)
    expect(readsLeakBand({ kind: 'band', ex: 450, band_nm: 470 })).toBe(true)
    expect(readsLeakBand({ kind: 'band', ex: 450, band_nm: 490 })).toBe(false)
    expect(LEAK_HINT).toMatch(/scattered excitation light/)
  })
})
