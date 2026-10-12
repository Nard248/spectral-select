import { describe, expect, it } from 'vitest'
import type { EemResult } from '../api/types'
import { alignEem, atEmission, brightestBand, componentLabel, diffGrid, log2RatioGrid } from './eemMath'
import { interp, metricFromEem, metricLabel } from './metrics'
import { NORM_MODES, compatibility, normProblems, normSpecClean } from './norm'
import { csvRow, eemToCsv, lineToCsv, responseToCsv, sweepToCsv, toJson, roiStatsToCsv, safeName } from './exportData'
import { useAnalysis, DEFAULT_NORM } from './store'
import { parseList, rangeValues } from '../sweep/store'
import { enumerateParams, getByPointer, prettyKey, setByPointer } from '../sweep/paths'
import { presetExperiment } from '../state/presets'
import type { CaptureMeta } from '../api/types'

const eem = (offset = 0, scale = 1): EemResult => ({
  roi: { shape: 'point', x: 1, y: 1 }, n_pixels: 4, norm: { mode: 'adu' }, unit: 'ADU', excitations_nm: [340, 400],
  emission_nm: { '340': [400, 420, 440, 460], '400': [440, 460, 480] },
  value: { '340': [1, 4, 9, 4].map((v) => v * scale + offset), '400': [2, 8, 3].map((v) => v * scale + offset) },
  std: { '340': [0.1, 0.2, 0.3, 0.2], '400': [0.2, 0.3, 0.1] },
  grid_emission_nm: [400, 410, 420, 430, 440, 450, 460, 470, 480],
  grid_value: [
    [1, 2.5, 4, 6.5, 9, 6.5, 4, NaN, NaN],
    [NaN, NaN, NaN, NaN, 2, 5, 8, 5.5, 3],
  ].map((r) => r.map((v) => v * scale + offset)),
})

describe('metrics', () => {
  it('interpolates inside the span and refuses to extrapolate far outside', () => {
    expect(interp([400, 420], [1, 3], 410)).toBe(2)
    expect(interp([400, 420], [1, 3], 395)).toBe(1)    // within one spacing: clamp
    expect(Number.isNaN(interp([400, 420], [1, 3], 300))).toBe(true)
  })
  it('computes band, ratio, sum and peak metrics from an EEM', () => {
    const r = eem()
    expect(metricFromEem(r, { kind: 'band', ex: 340, band_nm: 440 })).toBe(9)
    expect(metricFromEem(r, { kind: 'band_ratio', a: { ex: 340, band_nm: 440 }, b: { ex: 340, band_nm: 420 } })).toBeCloseTo(9 / 4)
    expect(metricFromEem(r, { kind: 'band_sum', ex: 340, from_nm: 420, to_nm: 440 })).toBe(13)
    const pk = metricFromEem(r, { kind: 'peak_wavelength', ex: 340 })
    expect(pk).toBeGreaterThan(430); expect(pk).toBeLessThan(450)
    expect(metricLabel({ kind: 'band', ex: 340, band_nm: 440 })).toContain('440')
  })
})

describe('eem math', () => {
  it('aligns two EEMs, differences and ratios them', () => {
    const A = eem(), B = eem(0, 0.5)
    const al = alignEem(A, B)
    expect(al.exs).toEqual([340, 400])
    const d = diffGrid(al), q = log2RatioGrid(al)
    expect(d[0][4]).toBeCloseTo(4.5)
    expect(q[0][4]).toBeCloseTo(1)        // A is twice B → log2 = 1
    expect(Number.isNaN(d[0][7])).toBe(true)
  })
  it('finds values at an emission wavelength and the brightest band', () => {
    const r = eem()
    expect(atEmission(r, 340, 430).v).toBe(6.5)
    expect(brightestBand(r, 340)).toBe(440)
    expect(brightestBand(r, 400)).toBe(460)
  })
  it('renders component names readably', () => {
    expect(componentLabel('dye:NADH@tissue')).toBe('NADH · tissue')
    expect(componentLabel('raman@cover')).toBe('water Raman · cover')
    expect(componentLabel('elastic')).toBe('elastic scatter')
  })
})

const capMeta = (id: string, over: Partial<CaptureMeta> = {}): CaptureMeta => ({
  id, created: '2026-01-01T00:00:00Z', name: 'x', mode: 'sample', width: 10, height: 8, excitations_nm: [340, 400],
  emission_nm: { '340': [400, 420], '400': [440, 460] }, exposure_ms: {}, power_mW: {}, stats: {}, saturated_fraction: {},
  ground_truth_layers: [], warnings: [], timings_s: {}, camera_id: 'c', source_id: 's', ...over,
})

describe('normalisation validation', () => {
  it('requires and validates dark / white captures', () => {
    const s = capMeta('s'), d = capMeta('d', { mode: 'dark' }), bad = capMeta('b', { mode: 'dark', width: 12 }), w = capMeta('w', { mode: 'white_reference' })
    expect(normProblems({ mode: 'adu' }, s, [s])).toEqual([])
    expect(normProblems({ mode: 'dark_subtracted' }, s, [s, d])[0]).toMatch(/choose a dark/)
    expect(normProblems({ mode: 'dark_subtracted', dark_capture_id: 'd' }, s, [s, d])).toEqual([])
    expect(normProblems({ mode: 'dark_subtracted', dark_capture_id: 'b' }, s, [s, bad])[0]).toMatch(/12×8/)
    expect(normProblems({ mode: 'white_normalized', dark_capture_id: 'd' }, s, [s, d, w])[0]).toMatch(/white/)
    expect(normProblems({ mode: 'white_normalized', dark_capture_id: 'd', white_capture_id: 'w' }, s, [s, d, w])).toEqual([])
    expect(compatibility(s, capMeta('x', { emission_nm: { '340': [400, 430], '400': [440, 460] } }), 'dark').join()).toMatch(/band set/)
  })
  it('drops irrelevant references from the request key', () => {
    expect(normSpecClean({ mode: 'adu', dark_capture_id: 'd', white_capture_id: 'w' })).toEqual({ mode: 'adu' })
    expect(normSpecClean({ mode: 'dark_subtracted', dark_capture_id: 'd', white_capture_id: 'w' })).toEqual({ mode: 'dark_subtracted', dark_capture_id: 'd' })
    expect(NORM_MODES).toHaveLength(7)
  })
})

describe('CSV / JSON export', () => {
  const ctx = { captureId: 'cap_001', experimentName: 'Wedge, "demo"', norm: { mode: 'dark_subtracted' as const, dark_capture_id: 'cap_000' }, roiName: 'Disk 1', roi: { shape: 'disk' as const, x: 5, y: 6, radius_px: 3 } }
  it('escapes cells', () => { expect(csvRow(['a,b', 'q"t', 3, NaN, null])).toBe('"a,b","q""t",3,,') })
  it('writes self-describing EEM CSV with units, ROI, norm, capture and experiment', () => {
    const csv = eemToCsv(eem(), ctx)
    const lines = csv.trim().split('\n')
    expect(lines[0]).toMatch(/^# SpectralForge Studio/)
    expect(csv).toContain('# experiment: Wedge, "demo"')
    expect(csv).toContain('# capture: cap_001')
    expect(csv).toContain('# normalisation: dark_subtracted dark=cap_000')
    expect(csv).toContain('# unit: ADU')
    expect(csv).toContain('"shape":"disk"')
    const header = lines.find((l) => !l.startsWith('#'))!
    expect(header).toBe('excitation_nm,emission_nm,value_ADU,std_ADU')
    expect(lines.filter((l) => !l.startsWith('#')).length).toBe(1 + 4 + 3)
  })
  it('exports line profiles, response curves, sweeps and GT stats', () => {
    const line = lineToCsv({ distance_px: [0, 1], distance_mm: [0, 0.4], value: { '340': [[1, 2], [3, 4]] }, emission_nm: { '340': [440, 460] }, ground_truth: { 'layer_thickness:cover': [2, 1.9] }, unit: 'ADU' }, ctx)
    expect(line).toContain('distance_px,distance_mm,ex340_em440,ex340_em460,layer_thickness:cover')
    const resp = responseToCsv({ x_ground_truth: 'layer_thickness:cover', metric: { kind: 'band', ex: 340, band_nm: 440 }, bin_centres: [0.1], mean: [5], std: [1], count: [9], scatter_x: [0.1], scatter_y: [4], unit: 'ADU' }, ctx)
    expect(resp).toContain('layer_thickness:cover,mean,std,count')
    const sw = sweepToCsv([{ index: 0, value: 0.5, metric: 12, captureId: 'cap_5' }], { parameterPath: '/a/b', parameterLabel: 'B', metricName: 'I', unit: 'ADU' }, ctx)
    expect(sw).toContain('# parameter_path: /a/b')
    expect(sw).toContain('0,0.5,12,cap_5')
    expect(roiStatsToCsv({ ...eem(), ground_truth: { t: { mean: 1, std: 0.1, min: 0, max: 2 } } }, ctx)).toContain('t,1,0.1,0,2')
    const js = JSON.parse(toJson('eem', ctx, { v: [1, NaN] }))
    expect(js.capture).toBe('cap_001'); expect(js.data.v).toEqual([1, null]); expect(js.normalisation.mode).toBe('dark_subtracted')
    expect(safeName('a b/c')).toBe('a_b_c')
  })
})

describe('analysis store', () => {
  it('adds, names, colours, selects and removes ROIs', () => {
    const st = useAnalysis.getState()
    st.clearRois()
    const a = st.addRoi({ shape: 'disk', x: 3, y: 3, radius_px: 2 })
    const b = useAnalysis.getState().addRoi({ shape: 'disk', x: 9, y: 9, radius_px: 2 })
    let s = useAnalysis.getState()
    expect(s.rois.map((r) => r.name)).toEqual(['Disk 1', 'Disk 2'])
    expect(new Set(s.rois.map((r) => r.color)).size).toBe(2)
    expect(s.selectedId).toBe(b)
    s.updateRoi(a, { name: 'tissue', visible: false })
    s.setEem({ roiB: a })
    s.removeRoi(a)
    s = useAnalysis.getState()
    expect(s.rois).toHaveLength(1); expect(s.eem.roiB).toBeNull()
    s.setNorm({ mode: 'dark_subtracted', dark_capture_id: 'd' })
    expect(useAnalysis.getState().norm).toMatchObject({ mode: 'dark_subtracted', dark_capture_id: 'd' })
    useAnalysis.getState().setNorm({ ...DEFAULT_NORM })
    useAnalysis.getState().clearRois()
  })
})

describe('sweep helpers', () => {
  it('builds linear and log ranges and parses lists', () => {
    expect(rangeValues(0, 2, 5, 'linear')).toEqual([0, 0.5, 1, 1.5, 2])
    const lg = rangeValues(1, 100, 3, 'log'); expect(lg[1]).toBeCloseTo(10)
    expect(rangeValues(3, 9, 1, 'linear')).toEqual([3])
    expect(parseList('0, 0.5; 1 x 2')).toEqual([0, 0.5, 1, 2])
  })
  it('enumerates layer thicknesses, dye concentrations, scattering, power and camera fields with readable labels', () => {
    const exp = presetExperiment('layered_wedge_phantom')
    const opts = enumerateParams(exp)
    const cover = opts.find((o) => o.path === '/objects/0/shape/layers/2/thickness/thickness_mm')!
    expect(cover.label).toMatch(/cover › start thickness/); expect(cover.unit).toBe('mm'); expect(cover.group).toBe('Layer thickness')
    expect(opts.some((o) => o.path === '/objects/0/shape/layers/2/thickness/end_thickness_mm')).toBe(true)
    expect(opts.some((o) => o.path === '/objects/0/shape/layers/1/thickness/end_thickness_mm')).toBe(false)   // uniform layer: no end thickness
    expect(opts.some((o) => /concentration/.test(o.label) && o.unit === 'µM')).toBe(true)
    expect(opts.some((o) => /scattering/.test(o.label))).toBe(true)
    expect(opts.some((o) => /power/.test(o.label) && o.unit === 'mW')).toBe(true)
    expect(opts.some((o) => o.group === 'Camera' && /exposure|gain|read noise/i.test(o.label))).toBe(true)
    opts.forEach((o) => expect(getByPointer(exp, o.path)).toBe(o.value))
    expect(new Set(opts.map((o) => o.path)).size).toBe(opts.length)
  })
  it('reads and writes JSON pointers without mutating the source', () => {
    const exp = presetExperiment('layered_wedge_phantom')
    const p = '/objects/0/shape/layers/2/thickness/thickness_mm'
    const e2 = setByPointer(exp, p, 1.25)
    expect(getByPointer(e2, p)).toBe(1.25); expect(getByPointer(exp, p)).toBe(2)
    expect(() => setByPointer(exp, '/objects/0/shape/layers/9/thickness/thickness_mm', 1)).toThrow()
    expect(prettyKey('end_thickness_mm')).toEqual({ label: 'end thickness', unit: 'mm' })
    expect(prettyKey('mus_prime_500_per_mm').unit).toBe('mm⁻¹')
  })
})

import { depthProfiles, geometry, locate, renderXY, renderXZ, autoScale, colorIndex } from './volumeGeom'
import { profileAt, stackAt, topPlane, layerDepthTop } from './blockGeometry'
import { bounds, contains, dragHandle, handles, hit, maskIndices, nearestHandle, polygonArea, translate, translateClamped, describe as roiDescribe } from './roi'
import type { LayeredBlockShape } from '../generated/experiment'
import type { VolumeInfo } from '../api/types'

const wedge: LayeredBlockShape = {
  kind: 'layered_block', footprint_mm: [40, 40],
  layers: [
    { name: 'substrate', material_id: 'm', thickness: { kind: 'uniform', thickness_mm: 4, end_thickness_mm: 4, axis: 'x' } },
    { name: 'fluor', material_id: 'm', thickness: { kind: 'uniform', thickness_mm: 0.5, end_thickness_mm: 0.5, axis: 'x' } },
    { name: 'cover', material_id: 'm', thickness: { kind: 'linear', thickness_mm: 2, end_thickness_mm: 0, axis: 'x' } },
  ],
} as unknown as LayeredBlockShape

describe('block geometry', () => {
  it('evaluates the wedge thickness and the exact top plane', () => {
    expect(profileAt(wedge.layers[2]!.thickness, -20, 0, 40, 40)).toBeCloseTo(2)
    expect(profileAt(wedge.layers[2]!.thickness, 0, 0, 40, 40)).toBeCloseTo(1)
    expect(profileAt(wedge.layers[2]!.thickness, 20, 0, 40, 40)).toBeCloseTo(0)
    const tp = topPlane(wedge)
    expect(tp.a).toBeCloseTo(5.5); expect(tp.b).toBeCloseTo(-0.05); expect(tp.c).toBe(0); expect(tp.max).toBeCloseTo(6.5)
    const st = stackAt(wedge, -10, 3)
    expect(st.top).toBeCloseTo(tp.a + tp.b * -10)
    expect(layerDepthTop(st, 1)).toBeCloseTo(st.t[2])      // depth of the fluor layer top = cover thickness
    expect(layerDepthTop(st, 2)).toBe(0)
  })
})

describe('light-field rasterisation', () => {
  const nx = 20, nz = 20, S = 4, L = 3
  const info: VolumeInfo = {
    object_id: 'b', kind: 'fluence', excitations_nm: [340], channels: [], dims: [nx, nz, L * S],
    x_edges_mm: Array.from({ length: nx + 1 }, (_, i) => -20 + (40 * i) / nx), z_edges_mm: Array.from({ length: nz + 1 }, (_, i) => -20 + (40 * i) / nz),
    layer_names: ['substrate', 'fluor', 'cover'], sublayers: S, unit: 'mW mm⁻²',
  }
  const g = geometry(info, wedge)
  // value = 100 × layer index + sublayer → every cell identifies itself
  const data = new Float32Array(nx * nz * L * S)
  for (let c = 0; c < nx * nz; c++) for (let d = 0; d < L * S; d++) data[c * L * S + d] = 1 + d
  it('maps physical points to the right layer, thinning the cover along the wedge', () => {
    expect(locate(g, -15, 5.2, 0)!.layer).toBe(2)       // cover is thick at x = -15
    expect(locate(g, 18, 4.3, 0)!.layer).toBe(1)        // at x = +18 the cover is only 0.1 mm: y = 4.3 is already in the fluor layer
    expect(locate(g, 18, 4.8, 0)).toBeNull()            // above the sloped surface
    expect(locate(g, 0, 0.1, 0)!.layer).toBe(0)
    expect(locate(g, 0, 0.1, 0)!.d).toBe(0)
    expect(locate(g, 0, 4.4, 0)!.d % S).toBeGreaterThanOrEqual(0)
  })
  it('renders an XZ slice whose top edge follows the wedge and an XY slice with layer edges', () => {
    const sc = autoScale(data, true, 3, 'inferno')
    const w = 200, h = 80
    const xz = renderXZ(g, data, 0, w, h, sc)
    const bgAt = (u: number, v: number) => { const o = (v * w + u) * 4; return xz.data[o] === 10 && xz.data[o + 1] === 12 }
    // topmost non-background row is lower (further down the image) on the right where the cover vanished
    const topRow = (u: number) => { for (let v = 0; v < h; v++) if (!bgAt(u, v)) return v; return h }
    expect(topRow(w - 5)).toBeGreaterThan(topRow(5) + 4)
    const xy = renderXY(g, data, 5.0, 100, 100, sc)
    expect(xy.data.length).toBe(100 * 100 * 4)
    expect(colorIndex(sc.vmax, sc)).toBe(255); expect(colorIndex(0, sc)).toBe(0)
  })
  it('computes per-layer lateral-average depth profiles (cover thinner on average than its nominal 2 mm)', () => {
    const p = depthProfiles(g, data)
    expect(p.map((q) => q.layer)).toEqual([0, 1, 2])
    const cover = p[2]
    expect(cover.top).toBe(0)
    expect(cover.bottom).toBeGreaterThan(0.8); expect(cover.bottom).toBeLessThan(1.2)   // mean thickness ≈ 1 mm
    expect(cover.depth[0]).toBeGreaterThan(cover.depth[S - 1])  // sub-layer 0 is the deepest (bottom-to-top storage)
  })
})

describe('ROI geometry', () => {
  it('counts pixels of each shape', () => {
    expect(maskIndices({ shape: 'point', x: 3, y: 4 }, 10, 10).length).toBe(1)
    expect(maskIndices({ shape: 'rect', x0: 2, y0: 2, x1: 4, y1: 5 }, 10, 10).length).toBe(3 * 4)
    const disk = maskIndices({ shape: 'disk', x: 5, y: 5, radius_px: 2 }, 11, 11).length
    expect(disk).toBe(13)                               // classic radius-2 digital disk
    expect(maskIndices({ shape: 'ellipse', cx: 5, cy: 5, rx: 3, ry: 1 }, 11, 11).length).toBe(7 + 1 + 1)
    expect(maskIndices({ shape: 'polygon', points: [[1, 1], [6, 1], [6, 6], [1, 6]] }, 10, 10).length).toBeGreaterThanOrEqual(25)
    expect(maskIndices({ shape: 'disk', x: 0, y: 0, radius_px: 3 }, 10, 10).length).toBeLessThan(disk * 2)  // clipped at the image edge
    expect(contains({ shape: 'rect', x0: 5, y0: 5, x1: 1, y1: 1 }, 3, 3)).toBe(true)   // corners may be given in any order
  })
  it('hit-tests, moves and resizes through handles', () => {
    const rect = { shape: 'rect' as const, x0: 2, y0: 2, x1: 8, y1: 6 }
    expect(hit(rect, 5, 4, 1)).toBe(true); expect(hit(rect, 20, 4, 1)).toBe(false)
    expect(hit({ shape: 'line', p0: [0, 0], p1: [10, 0], width_px: 1 }, 5, 1, 1.5)).toBe(true)
    expect(handles(rect)).toHaveLength(4)
    expect(nearestHandle(rect, 8.2, 6.1, 1)!.id).toBe('se')
    const grown = dragHandle(rect, 'se', 12, 9)
    expect(bounds(grown)).toEqual({ x0: 2, y0: 2, x1: 12, y1: 9 })
    expect(dragHandle({ shape: 'disk', x: 5, y: 5, radius_px: 1 }, 'r', 9, 5)).toMatchObject({ radius_px: 4 })
    expect(translate({ shape: 'point', x: 1, y: 1 }, 2, 3)).toEqual({ shape: 'point', x: 3, y: 4 })
    const clamped = translateClamped(rect, 100, 100, 20, 20)
    expect(bounds(clamped).x1).toBe(19); expect(bounds(clamped).y1).toBe(19)
    expect(polygonArea([[0, 0], [4, 0], [4, 3], [0, 3]])).toBe(12)
    expect(roiDescribe({ shape: 'disk', x: 5, y: 6, radius_px: 3 })).toMatch(/disk/)
  })
})

import { scaleRoi } from './roi'
import { defaultGt, gtLabel, gtUnitOf, numericGt, brightestOf } from './gt'
import { nanify, parseLooseJson } from '../api/client'

describe('ground-truth helpers', () => {
  const m = capMeta('g', { ground_truth_layers: ['bleached:NADH', 'block_face', 'depth_mm', 'material', 'object', 'layer_thickness:cover', 'layer_depth:cover', 'conc:NADH', 'irradiance@340'] })
  it('lists numeric layers with the layer thickness first and drops categorical maps', () => {
    const g = numericGt(m)
    expect(g[0]).toBe('layer_thickness:cover')
    expect(g).not.toContain('material'); expect(g).not.toContain('block_face'); expect(g).not.toContain('object')
    expect(defaultGt(m)).toBe('layer_thickness:cover')
  })
  it('labels and units', () => {
    expect(gtLabel('layer_thickness:cover')).toBe('layer thickness · cover')
    expect(gtUnitOf('layer_thickness:cover')).toBe('mm'); expect(gtUnitOf('depth_mm')).toBe('mm'); expect(gtUnitOf('incidence_deg')).toBe('°')
  })
  it('finds the brightest excitation / band from capture statistics', () => {
    const s = (p99: number) => ({ min: 0, max: p99, p01: 0, p99, mean: p99 / 2 })
    const b = brightestOf(capMeta('b', { stats: { '340': [s(10), s(20)], '400': [s(5), s(90)] } }))
    expect(b).toEqual({ ex: 400, band_nm: 460 })
    expect(brightestOf(capMeta('b', { stats: { '340': [s(10), s(20)], '400': [s(5), s(90)] } }), 340).ex).toBe(340)
  })
})

describe('engine JSON quirks', () => {
  it('parses Python NaN / Infinity tokens and maps null back to NaN in numeric arrays', () => {
    const raw = '{"a": [1, NaN, 3], "b": {"c": [[NaN, 2], [-Infinity, 4]]}, "name": "NaN stays in strings"}'
    const o = nanify(parseLooseJson<{ a: number[]; b: { c: number[][] }; name: string }>(raw))
    expect(o.a[0]).toBe(1); expect(Number.isNaN(o.a[1])).toBe(true)
    expect(Number.isNaN(o.b.c[0][0])).toBe(true); expect(o.b.c[1][1]).toBe(4)
    expect(o.name).toBe('NaN stays in strings')
  })
  it('rescales ROIs between image sizes', () => {
    const r = scaleRoi({ shape: 'disk', x: 100, y: 60, radius_px: 8 }, 0.25, 0.25)
    expect(r.x).toBeCloseTo(24.625); expect(r.y).toBeCloseTo(14.625); expect(r.radius_px).toBeCloseTo(2)
    const l = scaleRoi({ shape: 'line', p0: [0, 0], p1: [100, 0], width_px: 3 }, 0.5, 0.5)
    expect(l.p1[0]).toBeCloseTo(49.75)   // pixel centres map to pixel centres: (x + 0.5)·s − 0.5
  })
})
