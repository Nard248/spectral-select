import { describe, expect, it } from 'vitest'
import { estimateVolumeCost, fmtDuration, qualityOf, ASSUMED_PHOTONS_PER_S } from './volumeEstimate'
import { presetExperiment } from './presets'

describe('volume MC cost model', () => {
  it('counts photons: N_ex·steps + N_em·nodes per block', () => {
    const e = presetExperiment('layered_wedge_phantom')
    const c = estimateVolumeCost(e)
    const v = e.acquisition.volume_mc
    expect(c.blocks).toBe(1)
    expect(c.totalPhotons).toBe(v.excitation_photons * c.excitationSteps + v.emission_photons * c.emissionNodes)
    expect(c.seconds).toBeCloseTo(c.totalPhotons / ASSUMED_PHOTONS_PER_S, 9)
    expect(c.previewSeconds).toBeCloseTo(c.seconds * v.preview_fraction, 9)
  })
  it('voxel count follows the layer-conforming grid', () => {
    const e = presetExperiment('layered_wedge_phantom')
    const c = estimateVolumeCost(e)
    expect(c.voxels).toBe(80 * 80 * 3 * 8)   // 40 mm / 0.5 mm cells, 3 layers, 8 sublayers
  })
  it('is zero without a block', () => {
    expect(estimateVolumeCost(presetExperiment('generic_macro_bench')).totalPhotons).toBe(0)
  })
  it('format + quality', () => {
    expect(fmtDuration(0.2)).toBe('< 1 s'); expect(fmtDuration(30)).toBe('30 s'); expect(fmtDuration(600)).toBe('10 min'); expect(fmtDuration(7200)).toBe('2.0 h')
    expect(qualityOf({ excitation_photons: 2_000_000, emission_photons: 2_000_000 })).toBe('standard')
    expect(qualityOf({ excitation_photons: 5, emission_photons: 2_000_000 })).toBe('custom')
  })
})
