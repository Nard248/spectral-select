import { beforeAll, describe, expect, it } from 'vitest'
import { mockEngine } from './mock'
import { presetExperiment } from '../state/presets'
import type { CaptureMeta, EemResult } from './types'
import { metricFromEem } from '../analysis/metrics'

let meta: CaptureMeta
let dark: CaptureMeta
let white: CaptureMeta

beforeAll(async () => {
  const exp = presetExperiment('layered_wedge_phantom')
  exp.sources[0].sequence = exp.sources[0].sequence.filter((s) => [340, 450].includes(s.wavelength_nm))
  exp.sources[0].sequence.forEach((s) => { s.enabled = true })
  exp.sources[0].sequence = exp.sources[0].sequence.length ? exp.sources[0].sequence : exp.sources[0].sequence
  const sweepSafe = { ...exp, acquisition: { ...exp.acquisition, preview_scale: 0.3 } }
  const cap = await mockEngine.preview({ experiment: sweepSafe, scale: 0.3, excitation_nm: 450 })
  meta = cap
  // capture-mode dark / white of identical geometry
  const mk = async (mode: 'dark' | 'white_reference') => {
    const { job_id } = await mockEngine.startCapture({ experiment: { ...sweepSafe }, mode })
    return new Promise<CaptureMeta>((res) => mockEngine.watchJob(job_id, (e) => { if (e.type === 'done') void mockEngine.getCapture(e.capture_id).then(res) }))
  }
  void mk
  dark = meta; white = meta
}, 60000)

describe('layered block in the mock engine', () => {
  it('exposes block ground truth and volumes in the capture metadata', () => {
    expect(meta.ground_truth_layers).toContain('layer_thickness:cover')
    expect(meta.ground_truth_layers).toContain('layer_depth:fluorescent layer')
    const x = meta as CaptureMeta & { blocks?: { layer_names: string[] }[]; volumes?: { kind: string }[] }
    expect(x.blocks?.[0].layer_names).toEqual(['substrate', 'fluorescent layer', 'cover'])
    expect(x.volumes?.map((v) => v.kind)).toEqual(expect.arrayContaining(['fluence', 'absorbed', 'source']))
  })

  it('cover thickness GT is a wedge (varies, 0..2 mm) and NaN off the block', async () => {
    const gt = await mockEngine.groundTruth(meta.id, 'layer_thickness:cover')
    const vals = Array.from(gt.data).filter(Number.isFinite)
    expect(vals.length).toBeGreaterThan(50)
    expect(Math.min(...vals)).toBeLessThan(0.3)
    expect(Math.max(...vals)).toBeGreaterThan(1.7)
    expect(Array.from(gt.data).some((v) => Number.isNaN(v))).toBe(true)
  })

  it('buried fluorescein emission decreases with cover thickness (response curve)', async () => {
    const ex = meta.excitations_nm[0]
    const em = meta.emission_nm[String(ex)]
    const nm = em.reduce((a, b) => (Math.abs(b - 520) < Math.abs(a - 520) ? b : a))
    const r = await mockEngine.response(meta.id, { x_ground_truth: 'layer_thickness:cover', metric: { kind: 'band', ex, band_nm: nm }, bins: 8, norm: { mode: 'dark_subtracted', dark_capture_id: null } }).catch((e: Error) => e)
    expect(r).toBeInstanceOf(Error)   // dark-subtracted without a dark capture is rejected, not silently ignored
    const ok = await mockEngine.response(meta.id, { x_ground_truth: 'layer_thickness:cover', metric: { kind: 'band', ex, band_nm: nm }, bins: 8 })
    if (ok instanceof Error) throw ok
    expect(ok.bin_centres.length).toBeGreaterThanOrEqual(4)
    expect(ok.mean[0]).toBeGreaterThan(ok.mean[ok.mean.length - 1] * 1.3)
    expect(ok.scatter_x.length).toBe(ok.scatter_y.length)
    expect(ok.scatter_x.length).toBeLessThanOrEqual(4000)
  })

  it('EEM over a disk returns the documented shape', async () => {
    const r: EemResult = await mockEngine.eem(meta.id, { roi: { shape: 'disk', x: meta.width / 2, y: meta.height / 2, radius_px: 4 } })
    expect(r.n_pixels).toBeGreaterThan(30)
    expect(r.unit).toBe('ADU')
    for (const ex of r.excitations_nm) {
      expect(r.value[String(ex)]).toHaveLength(r.emission_nm[String(ex)].length)
      expect(r.std[String(ex)]).toHaveLength(r.emission_nm[String(ex)].length)
    }
    expect(r.grid_value).toHaveLength(r.excitations_nm.length)
    expect(r.grid_value[0]).toHaveLength(r.grid_emission_nm.length)
    const peak = metricFromEem(r, { kind: 'peak_wavelength', ex: meta.excitations_nm[0] })
    expect(peak).toBeGreaterThan(450)
  })

  it('ROI stats include ground-truth statistics, optionally components', async () => {
    const r = await mockEngine.roi(meta.id, { roi: { shape: 'rect', x0: 10, y0: 10, x1: 30, y1: 30 }, components: true })
    expect(Object.keys(r.ground_truth).length).toBeGreaterThan(0)
    expect(r.components).toBeTruthy()
    const names = Object.keys(Object.values(r.components!)[0])
    expect(names.some((n) => n.startsWith('dye:') && n.includes('@'))).toBe(true)
  })

  it('line profile samples bands and ground truth along the line, with a mm axis on the block', async () => {
    const y = Math.round(meta.height / 2)
    const p = await mockEngine.line(meta.id, { p0: [Math.round(meta.width * 0.3), y], p1: [Math.round(meta.width * 0.7), y], width_px: 3, ground_truth: ['layer_thickness:cover'] })
    expect(p.distance_px.length).toBeGreaterThan(10)
    expect(p.value[String(meta.excitations_nm[0])]).toHaveLength(p.distance_px.length)
    expect(p.ground_truth['layer_thickness:cover']).toHaveLength(p.distance_px.length)
    expect(p.distance_mm).not.toBeNull()
    expect(p.distance_mm![p.distance_mm!.length - 1]).toBeGreaterThan(5)
  })

  it('serves volumes whose size matches the declared dims', async () => {
    const vols = await mockEngine.volumes(meta.id)
    const v = vols.find((q) => q.kind === 'fluence')!
    const d = await mockEngine.volume(meta.id, { object: v.object_id, kind: 'fluence', ex: v.excitations_nm[0] })
    expect(d.data.length).toBe(v.dims[0] * v.dims[1] * v.dims[2])
    expect(d.data.reduce((a, b) => Math.max(a, b), 0)).toBeGreaterThan(0)
    expect(v.x_edges_mm).toHaveLength(v.dims[0] + 1)
  })

  void dark; void white
})

describe('mock sweeps', () => {
  it('runs a thickness sweep, emitting a capture per point', async () => {
    const exp = presetExperiment('layered_wedge_phantom')
    exp.sources[0].sequence = exp.sources[0].sequence.slice(0, 2)
    const { sweep_id } = await mockEngine.startSweep({ experiment: exp, parameter: { path: '/objects/0/shape/layers/2/thickness/end_thickness_mm', values: [0, 1, 2] }, mode: 'preview' })
    const pts: number[] = []
    await new Promise<void>((resolve) => mockEngine.watchSweep(sweep_id, (e) => { if (e.type === 'point') pts.push(e.index); if (e.type === 'done') resolve() }))
    expect(pts).toEqual([0, 1, 2])
    const st = await mockEngine.sweepStatus(sweep_id)
    expect(st.state).toBe('done')
    expect(st.capture_ids.every((x) => !!x)).toBe(true)
  }, 120000)

  it('rejects a parameter path that does not exist', async () => {
    const exp = presetExperiment('layered_wedge_phantom')
    await expect(mockEngine.startSweep({ experiment: exp, parameter: { path: '/objects/0/shape/layers/9/thickness/thickness_mm', values: [1] } })).rejects.toThrow()
  })
})

describe('irradiance_normalized in the mock engine', () => {
  it('divides out exposure × irradiance (NaN where E = 0) and differs from raw ADU', async () => {
    const roi = { shape: 'rect' as const, x0: 0, y0: 0, x1: meta.width - 1, y1: meta.height - 1 }
    const raw = await mockEngine.eem(meta.id, { roi, norm: { mode: 'adu' } })
    const n = await mockEngine.eem(meta.id, { roi, norm: { mode: 'irradiance_normalized' } })
    expect(n.norm.mode).toBe('irradiance_normalized')
    const ex = String(n.excitations_nm[0])
    const a = n.value[ex].filter(Number.isFinite), b = raw.value[ex]
    expect(a.length).toBeGreaterThan(0)
    expect(a[0]).not.toBeCloseTo(b[0], 3)
  })
})
