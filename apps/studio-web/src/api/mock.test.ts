import { describe, expect, it } from 'vitest'
import { mockEngine } from './mock'
import { presetExperiment } from '../state/presets'

describe('mock engine honours the API contract', () => {
  it('previews, then serves band / cube / pixel / ground truth consistently', async () => {
    const exp = presetExperiment('generic_macro_bench')
    exp.sources[0].sequence = exp.sources[0].sequence.slice(0, 2)
    const meta = await mockEngine.preview({ experiment: exp, scale: 0.2 })
    expect(meta.mode).toBe('preview')
    expect(meta.excitations_nm).toHaveLength(1)
    const ex = meta.excitations_nm[0]
    const nb = meta.emission_nm[String(ex)].length
    const cube = await mockEngine.cube(meta.id, ex)
    expect(cube.width).toBe(meta.width); expect(cube.bands).toBe(nb)
    expect(cube.data.length).toBe(meta.width * meta.height * nb)
    const band = await mockEngine.band(meta.id, ex, 3)
    expect(band.data[10]).toBe(cube.data[10 * nb + 3])
    const px = await mockEngine.pixel(meta.id, 5, 5)
    expect(px.adu[String(ex)]).toHaveLength(nb)
    expect(px.ground_truth).toHaveProperty('material')
    const gt = await mockEngine.groundTruth(meta.id, meta.ground_truth_layers[0])
    expect(gt.data.length).toBe(meta.width * meta.height)
  })

  it('reports validation errors for dangling material references', async () => {
    const exp = presetExperiment('generic_macro_bench')
    exp.objects[0].material_id = 'nope'
    const r = await mockEngine.validate(exp)
    expect(r.ok).toBe(false)
    expect(r.errors[0].loc).toEqual(['objects', 0, 'material_id'])
  })

  it('runs a capture job with progress events and ends in done', async () => {
    const exp = presetExperiment('laser_line')
    const { job_id } = await mockEngine.startCapture({ experiment: exp })
    const events: string[] = []
    const done = new Promise<string>((resolve) => mockEngine.watchJob(job_id, (e) => { events.push(e.type); if (e.type === 'done') resolve(e.capture_id) }))
    const id = await done
    expect(events.filter((e) => e === 'progress').length).toBeGreaterThanOrEqual(exp.sources[0].sequence.length - 1)
    const m = await mockEngine.getCapture(id)
    expect(m.excitations_nm).toHaveLength(exp.sources[0].sequence.length)
  }, 15000)

  // 457 nm is the sequence's high-dose step (20 mW x 100 ms); the 488 nm step is deliberately 1 mW x 30 ms for bright fluorescein.
  it('a dye region is brighter than the dark table at its excitation', async () => {
    const exp = presetExperiment('laser_line')
    const meta = await mockEngine.preview({ experiment: exp, scale: 0.5, excitation_nm: 457 })
    const cube = await mockEngine.cube(meta.id, 457)
    let mx = 0, mn = Infinity
    for (const v of cube.data) { mx = Math.max(mx, v); mn = Math.min(mn, v) }
    expect(mx).toBeGreaterThan(mn * 10)
  })
})

describe('mock engine renders every preset sensibly', () => {
  for (const name of ['generic_macro_bench', 'mehsi_lab_replica', 'led_multispectral', 'laser_line']) {
    it(`${name}: dye signal visible, little saturation`, async () => {
      const exp = presetExperiment(name)
      const meta = await mockEngine.preview({ experiment: exp, scale: 0.25 })
      for (const ex of meta.excitations_nm) {
        const c = await mockEngine.cube(meta.id, ex)
        const s = Float32Array.from(c.data).sort()
        if (ex === meta.excitations_nm[0]) expect(s[s.length - 1]).toBeGreaterThan(s[0] * 5)
        expect(meta.saturated_fraction[String(ex)]).toBeLessThan(0.05)
      }
    })
  }
})
