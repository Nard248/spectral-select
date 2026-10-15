import { describe, expect, it } from 'vitest'
import { haltonPoint, morrisIndices, sobolBootstrap, sobolIndices, unitDesign, unitToValue } from './designs'
import type { ParameterSpec } from '../api/researchTypes'

const P = (label: string, lo = 0, hi = 1): ParameterSpec => ({ pointer: `/${label}`, label, kind: 'continuous', low: lo, high: hi, scale: 'linear', also_set: [] })

describe('designs', () => {
  it('LHS puts exactly one point in each of the n strata of every parameter', () => {
    const n = 20, d = unitDesign('lhs', [P('a'), P('b'), P('c')], n, 4, 3)
    for (let j = 0; j < 3; j++) expect(new Set(d.map((p) => Math.floor(p.u[j] * n))).size).toBe(n)
  })
  it('is reproducible for a seed and different for another', () => {
    const a = unitDesign('lhs', [P('a')], 8, 4, 1), b = unitDesign('lhs', [P('a')], 8, 4, 1), c = unitDesign('lhs', [P('a')], 8, 4, 2)
    expect(a).toEqual(b); expect(a).not.toEqual(c)
  })
  it('Saltelli has n(k+2) rows; AB_i equals A except in column i, which is B', () => {
    const n = 16, k = 3, d = unitDesign('saltelli', [P('a'), P('b'), P('c')], n, 4, 0)
    expect(d).toHaveLength(n * (k + 2))
    const A = d.filter((p) => p.block === 'A'), B = d.filter((p) => p.block === 'B')
    for (let i = 0; i < k; i++) d.filter((p) => p.block === `AB:${i}`).forEach((p, r) => p.u.forEach((v, j) => expect(v).toBe(j === i ? B[r].u[j] : A[r].u[j])))
  })
  it('Morris trajectories move one parameter by Δ per step', () => {
    const d = unitDesign('morris', [P('a'), P('b'), P('c')], 5, 4, 0), delta = 4 / 6
    expect(d).toHaveLength(5 * 4)
    for (let r = 1; r < d.length; r++) if (d[r].step) { const diff = d[r].u.map((v, j) => v - d[r - 1].u[j]); expect(diff.filter((x) => Math.abs(x) > 1e-12)).toHaveLength(1); expect(Math.max(...diff.map(Math.abs))).toBeCloseTo(delta) }
    d.forEach((p) => p.u.forEach((v) => { expect(v).toBeGreaterThanOrEqual(-1e-12); expect(v).toBeLessThanOrEqual(1 + 1e-12) }))
  })
  it('grid enumerates the full factorial including both ends', () => {
    const d = unitDesign('grid', [P('a'), P('b')], 1, 3, 0)
    expect(d).toHaveLength(9); expect(d[0].u).toEqual([0, 0]); expect(d[8].u).toEqual([1, 1])
  })
  it('maps unit coordinates to values (linear, log, integer, choice)', () => {
    expect(unitToValue(P('a', 2, 4), 0.25)).toBeCloseTo(2.5)
    expect(unitToValue({ ...P('a', 1, 100), scale: 'log' }, 0.5)).toBeCloseTo(10)
    expect(unitToValue({ ...P('i', 1, 3), kind: 'integer' }, 0.99)).toBe(3)
    expect(unitToValue({ ...P('c'), kind: 'choice', values: ['x', 'y'] }, 0.6)).toBe('y')
  })
  it('Halton points lie in the unit cube', () => { for (let i = 0; i < 50; i++) haltonPoint(i, 4, [0.1, 0.2, 0.3, 0.4]).forEach((v) => { expect(v).toBeGreaterThanOrEqual(0); expect(v).toBeLessThan(1) }) })
})

describe('estimators against analytic results', () => {
  // Ishigami (a = 7, b = 0.1): S1 = [0.3139, 0.4424, 0], ST = [0.5576, 0.4424, 0.2437] (Saltelli et al. 2010)
  const ishigami = (u: number[]) => { const x = u.map((v) => -Math.PI + 2 * Math.PI * v); return Math.sin(x[0]) + 7 * Math.sin(x[1]) ** 2 + 0.1 * x[2] ** 4 * Math.sin(x[0]) }
  it('Saltelli / Jansen reproduce the Ishigami Sobol′ indices', () => {
    const n = 4096, d = unitDesign('saltelli', [P('a'), P('b'), P('c')], n, 4, 1), f = d.map((p) => ishigami(p.u))
    const fA = f.slice(0, n), fB = f.slice(n, 2 * n), fAB = [0, 1, 2].map((i) => f.slice(2 * n + i * n, 2 * n + (i + 1) * n))
    const est = sobolIndices(fA, fB, fAB)
    const S1 = [0.3139, 0.4424, 0], ST = [0.5576, 0.4424, 0.2437]
    est.forEach((e, i) => { expect(Math.abs(e.S1 - S1[i])).toBeLessThan(0.06); expect(Math.abs(e.ST - ST[i])).toBeLessThan(0.06) })
    const ci = sobolBootstrap(fA.slice(0, 1024), fB.slice(0, 1024), fAB.map((a) => a.slice(0, 1024)), 60)
    expect(ci.ST[0][0]).toBeLessThan(ci.ST[0][1])
  })
  it('an additive model has ΣS1 = 1 and ST = S1', () => {
    const n = 2048, d = unitDesign('saltelli', [P('a'), P('b')], n, 4, 2), f = d.map((p) => 3 * p.u[0] + p.u[1])
    const est = sobolIndices(f.slice(0, n), f.slice(n, 2 * n), [0, 1].map((i) => f.slice(2 * n + i * n, 2 * n + (i + 1) * n)))
    expect(est[0].S1 + est[1].S1).toBeCloseTo(1, 1); expect(est[0].S1).toBeCloseTo(0.9, 1); expect(est[0].ST).toBeCloseTo(est[0].S1, 1)
  })
  it('Morris recovers the slopes of a linear model with σ = 0', () => {
    const labels = ['a', 'b', 'c'], d = unitDesign('morris', labels.map((l) => P(l)), 8, 4, 0), f = d.map((p) => 3 * p.u[0] + 1 * p.u[1] + 0 * p.u[2])
    const m = morrisIndices(f, d, labels, 4 / 6)
    expect(m.map((x) => x.mu_star)).toEqual([expect.closeTo(3, 8), expect.closeTo(1, 8), expect.closeTo(0, 8)])
    m.forEach((x) => { expect(x.sigma).toBeCloseTo(0, 8); expect(x.n).toBe(8) })
  })
})
