import { describe, expect, it } from 'vitest'
import {
  aggregatePoints, channelGrid, crlbSdFor, designBaselines, designSize, estimateRuntime, fmtDuration, invertSym, parseCsv, roiToFraction, runAssignments,
  runsToCsv, secondsPerRun, sortRows, specProblems, sobolUnreliable, parseMdBlocks, suggestLinks, parameterValid, defaultRange, axisOf, axisT,
} from './logic'
import type { ParameterSpec, StudyRunRow } from '../api/researchTypes'

const cont = (label: string, lo = 0, hi = 1): ParameterSpec => ({ pointer: `/x/${label}`, label, kind: 'continuous', low: lo, high: hi, scale: 'linear', also_set: [] })

describe('design size', () => {
  it('Saltelli n(k+2), Morris n(k+1), sample designs n', () => {
    const ps = [cont('a'), cont('b'), cont('c')]
    expect(designSize({ kind: 'saltelli', n: 64, levels: 4 }, ps, 1)).toEqual({ points: 320, runs: 320 })
    expect(designSize({ kind: 'saltelli', n: 64, levels: 4 }, ps, 3)).toEqual({ points: 320, runs: 960 })
    expect(designSize({ kind: 'morris', n: 10, levels: 4 }, ps, 2)).toEqual({ points: 40, runs: 80 })
    for (const kind of ['lhs', 'sobol', 'random'] as const) expect(designSize({ kind, n: 25, levels: 4 }, ps, 1).points).toBe(25)
  })
  it('grid multiplies the levels; choices use their values, integers are capped by their range', () => {
    const choice: ParameterSpec = { pointer: '/c', label: 'c', kind: 'choice', values: [1, 2, 3], scale: 'linear', also_set: [] }
    const int: ParameterSpec = { pointer: '/i', label: 'i', kind: 'integer', low: 1, high: 3, scale: 'linear', also_set: [] }
    expect(designSize({ kind: 'grid', n: 1, levels: 5 }, [cont('a'), choice, int], 1).points).toBe(5 * 3 * 3)
    expect(designSize({ kind: 'grid', n: 1, levels: 5 }, [], 1).points).toBe(0)
  })
})

describe('runtime estimate', () => {
  const t = { previewS: 2, nExcitations: 8, previewScale: 0.25 }
  it('preview = 1 capture, fast = one per excitation, full is scaled by 1/scale²', () => {
    expect(secondsPerRun('preview', t).s).toBe(2)
    expect(secondsPerRun('fast', t).s).toBe(16)
    expect(secondsPerRun('full', t).s).toBeCloseTo(16 / 0.0625)
    expect(secondsPerRun('full', { ...t, fullS: 90 }).s).toBe(90)
  })
  it('a measured time wins; with no timing at all a default is flagged', () => {
    expect(estimateRuntime(10, 'fast', { ...t, observed: { fast: 3 } })).toEqual({ seconds: 30, basis: 'observed' })
    expect(estimateRuntime(10, 'preview', { nExcitations: 1, previewScale: 0.3 }).basis).toBe('default')
  })
  it('formats durations', () => { expect(fmtDuration(0.2)).toBe('<1 s'); expect(fmtDuration(45)).toBe('45 s'); expect(fmtDuration(600)).toBe('10 min'); expect(fmtDuration(7200)).toBe('2.0 h') })
})

describe('parameters and ROIs', () => {
  it('links the end thickness of a wedge layer', () => {
    const all = ['/objects/0/shape/layers/2/thickness/thickness_mm', '/objects/0/shape/layers/2/thickness/end_thickness_mm']
    expect(suggestLinks(all[0], all)).toEqual([all[1]])
    expect(suggestLinks('/materials/0/dyes/0/concentration_uM', all)).toEqual([])
  })
  it('validates ranges', () => {
    expect(parameterValid(cont('a', 1, 0))).toMatch(/exceed/)
    expect(parameterValid({ ...cont('a', 0, 1), scale: 'log' })).toMatch(/log/)
    expect(parameterValid(cont('a'))).toBeNull()
    expect(defaultRange(2, 'mm', 'cover › thickness')).toMatchObject({ low: 0.05, high: 4 })
  })
  it('converts pixel ROIs to fractions with the engine convention x / (w - 1)', () => {
    expect(roiToFraction({ shape: 'rect', x0: 0, y0: 0, x1: 99, y1: 49 }, 100, 50)).toEqual({ shape: 'rect', x0: 0, y0: 0, x1: 1, y1: 1 })
    const d = roiToFraction({ shape: 'disk', x: 49.5, y: 24.5, radius_px: 9.9 }, 100, 50)
    expect(d).toMatchObject({ shape: 'disk', x: 0.5, y: 0.5, radius_px: 0.1 })
  })
  it('requires n >= 8 for Saltelli', () => {
    const m = [{ name: 'm', kind: 'band_sum' as const, roi_units: 'fraction' as const, norm: { mode: 'adu' as const }, stat: 'mean' as const, contrast: 'michelson' as const }]
    const s = { design: { kind: 'saltelli' as const, n: 4, levels: 4, seed: 0 }, replicates: 1, max_runs: 20000, parameters: [cont('a')], metrics: m }
    expect(specProblems(s)[0]).toMatch(/at least 8/)
    expect(specProblems({ ...s, design: { ...s.design, n: 8 } })).toEqual([])
  })
  it('collects spec problems', () => {
    const base = { design: { kind: 'saltelli' as const, n: 10, levels: 4, seed: 0 }, replicates: 1, max_runs: 20000 }
    expect(specProblems({ ...base, parameters: [], metrics: [] }).length).toBe(2)
    expect(specProblems({ ...base, parameters: [cont('a')], metrics: [{ name: 'm', kind: 'band_sum', roi_units: 'fraction', norm: { mode: 'adu' }, stat: 'mean', contrast: 'michelson' }] })).toEqual([])
    expect(specProblems({ ...base, design: { ...base.design, n: 9000 }, parameters: [cont('a')], metrics: [{ name: 'm', kind: 'band_sum', roi_units: 'fraction', norm: { mode: 'adu' }, stat: 'mean', contrast: 'michelson' }] })[0]).toMatch(/exceeds/)
    expect(specProblems({ ...base, parameters: [cont('a')], metrics: [{ name: 'c', kind: 'cnr', roi_units: 'fraction', norm: { mode: 'adu' }, stat: 'mean', contrast: 'michelson' }] })[0]).toMatch(/two ROIs/)
  })
})

const row = (point: number, replicate: number, x: number, y: number, status = 'ok'): StudyRunRow => ({ run_id: `${point}_${replicate}`, point, replicate, seed: replicate, status, 'p:x': x, 'm:y': y, t_capture_s: 1 })

describe('runs', () => {
  it('aggregates replicates: mean and sample sd, failed runs excluded', () => {
    const agg = aggregatePoints([row(0, 0, 1, 10), row(0, 1, 1, 12), row(1, 0, 2, 5), row(2, 0, 3, 99, 'error')], 'x', 'y')
    expect(agg).toHaveLength(2)
    expect(agg[0]).toMatchObject({ point: 0, x: 1, y: 11, n: 2 })
    expect(agg[0].std).toBeCloseTo(Math.SQRT2)
    expect(agg[1]).toMatchObject({ y: 5, std: 0, n: 1 })
  })
  it('sorts numerically with missing values last', () => {
    const rows = [row(0, 0, 3, 1), { ...row(1, 0, 1, 1), 'p:x': null }, row(2, 0, 2, 1)]
    expect(sortRows(rows, 'p:x', 'asc').map((r) => r.point)).toEqual([2, 0, 1])
    expect(sortRows(rows, 'p:x', 'desc').map((r) => r.point)).toEqual([0, 2, 1])
  })
  it('CSV round trip with quoting, comments and empty cells', () => {
    const rows: StudyRunRow[] = [{ ...row(0, 0, 0.5, 1e-7), error: 'bad, "quoted"\nline' }, { ...row(1, 0, 2, NaN), error: null }]
    const csv = runsToCsv(['run_id', 'p:x', 'm:y', 'error'], rows, ['study s1'])
    expect(csv.startsWith('# study s1\n')).toBe(true)
    const p = parseCsv(csv)
    expect(p.columns).toEqual(['run_id', 'p:x', 'm:y', 'error'])
    expect(p.rows[0]).toMatchObject({ run_id: '0_0', 'p:x': 0.5, 'm:y': 1e-7, error: 'bad, "quoted"\nline' })
    expect(p.rows[1]['m:y']).toBeNull()
  })
  it('applies a run to the pointers including linked ones', () => {
    const spec = { parameters: [{ ...cont('cover'), pointer: '/a/th', also_set: ['/a/end'] }, cont('d')] }
    expect(runAssignments(spec, { ...row(0, 0, 0, 0), 'p:cover': 0.7, 'p:d': 3 })).toEqual([{ pointer: '/a/th', value: 0.7 }, { pointer: '/a/end', value: 0.7 }, { pointer: '/x/d', value: 3 }])
  })
  it('axes normalise linearly and logarithmically', () => {
    const a = axisOf([row(0, 0, 1, 0), row(1, 0, 100, 0)], 'p:x', 'x', true)
    expect(a.log).toBe(true)
    expect(axisT(a, 10)).toBeCloseTo(0.5)
    expect(axisT({ ...a, log: false }, 50.5)).toBeCloseTo(0.5)
  })
})

describe('Cramér–Rao algebra', () => {
  it('inverts a symmetric matrix', () => {
    const inv = invertSym([[4, 1], [1, 3]])!
    expect(inv[0][0]).toBeCloseTo(3 / 11); expect(inv[0][1]).toBeCloseTo(-1 / 11); expect(inv[1][1]).toBeCloseTo(4 / 11)
    expect(invertSym([[0, 0], [0, 0]])).toBeNull()
  })
  it('one parameter: sd = 1 / sqrt(sum J² / var)', () => {
    const jac = [[2], [3], [1]], v = [4, 9, 1]
    expect(crlbSdFor(jac, v, [0, 1, 2])[0]).toBeCloseTo(1 / Math.sqrt(1 + 1 + 1))
    expect(crlbSdFor(jac, v, [2])[0]).toBeCloseTo(1)
  })
  it('two parameters: collinear channels give an infinite / huge bound, adding an independent one fixes it', () => {
    const jac = [[1, 1], [2, 2], [1, -1]], v = [1, 1, 1]
    const bad = crlbSdFor(jac, v, [0, 1])
    expect(bad.every((s) => !Number.isFinite(s) || s > 1e5)).toBe(true)
    const ok = crlbSdFor(jac, v, [0, 2])
    expect(ok[0]).toBeCloseTo(Math.sqrt(0.5)); expect(ok[1]).toBeCloseTo(Math.sqrt(0.5))
  })
  it('baselines: random draws collapse onto the all-channel bound at k = C and uniform never beats the full set', () => {
    const jac = Array.from({ length: 12 }, (_, c) => [1 + Math.sin(c), Math.cos(c * 0.7)]), v = jac.map(() => 1)
    const b = designBaselines({ jac_e: jac, var_e: v }, 0, 12, 40)
    const all = crlbSdFor(jac, v, jac.map((_, i) => i))[0]
    expect(b.random.mean[11]).toBeCloseTo(all); expect(b.random.lo[11]).toBeCloseTo(all); expect(b.uniform[11]).toBeCloseTo(all)
    for (let i = 3; i < 12; i++) expect(b.uniform[i]).toBeGreaterThanOrEqual(all - 1e-9)
    expect(b.random.lo[5]).toBeLessThanOrEqual(b.random.hi[5])
  })
  it('lays channels out on an excitation × emission grid', () => {
    const g = channelGrid({ ex_nm: [340, 340, 365, 365], em_nm: [450, 470, 470, 490] }, [1, 2, 3, 4])
    expect(g.ys).toEqual([340, 365]); expect(g.xs).toEqual([450, 470, 490])
    expect(g.z[0]).toEqual([1, 2, NaN]); expect(g.z[1]).toEqual([NaN, 3, 4]); expect(g.index[1][2]).toBe(3)
  })
  it('flags unreliable Sobol estimates', () => {
    expect(sobolUnreliable({ sum_S1: -9, n_rows: 64 })).toBe(true)
    expect(sobolUnreliable({ sum_S1: 0.9, n_rows: 8 })).toBe(true)
    expect(sobolUnreliable({ sum_S1: 0.95, n_rows: 64 })).toBe(false)
  })
  it('parses headings, paragraphs and pipe tables', () => {
    const b = parseMdBlocks('## Title\n\nHello\nworld\n| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n\nTail')
    expect(b[0]).toEqual({ kind: 'h', text: 'Title' })
    expect(b[1]).toEqual({ kind: 'p', text: 'Hello world' })
    expect(b[2]).toEqual({ kind: 'table', head: ['a', 'b'], rows: [['1', '2'], ['3', '4']] })
    expect(b[3]).toEqual({ kind: 'p', text: 'Tail' })
  })
})
