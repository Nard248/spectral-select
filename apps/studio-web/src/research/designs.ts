/** Offline implementation of the study designs (grid, random, LHS, low-discrepancy, Saltelli radial, Morris trajectories) and
 *  of the Sobol' / Morris estimators, used by the in-browser mock engine and unit-tested against analytic results.
 *  The live engine does the same in Python (`spectraforge.studio.study`, `sensitivity`).
 *
 *  Sobol' estimators: first order Saltelli et al. (2010, Comput. Phys. Commun. 181, 259), S1 = mean(f(B)(f(AB_i) − f(A))) / V;
 *  total order Jansen (1999), ST = ½ mean((f(A) − f(AB_i))²) / V. Morris (1991, Technometrics 33, 161), Campolongo et al. (2007):
 *  μ* = mean |EE|, σ = sd(EE). */
import type { DesignKind, ParameterSpec } from '../api/researchTypes'
import { gridLevels, rng } from './logic'

export interface UnitPoint { u: number[]; block?: string; row?: number; traj?: number; step?: number; stepParam?: string; direction?: number }

const PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
function radicalInverse(i: number, base: number): number {
  let f = 1, r = 0
  for (let n = i; n > 0; n = Math.floor(n / base)) { f /= base; r += f * (n % base) }
  return r
}
/** Halton point i in d dimensions (Cranley-Patterson shifted by the seed): a low-discrepancy stand-in for scrambled Sobol. */
export function haltonPoint(i: number, d: number, shift: number[]): number[] {
  return Array.from({ length: d }, (_, j) => (radicalInverse(i + 1, PRIMES[j % PRIMES.length]) + shift[j]) % 1)
}

export function unitDesign(kind: DesignKind, params: ParameterSpec[], n: number, levels: number, seed: number): UnitPoint[] {
  const k = params.length
  const rand = rng(seed + 1)
  switch (kind) {
    case 'grid': {
      const ls = params.map((p) => gridLevels(p, levels))
      const out: UnitPoint[] = []
      const idx = new Array<number>(k).fill(0)
      const total = ls.reduce((a, b) => a * b, 1)
      for (let c = 0; c < total; c++) {
        out.push({ u: idx.map((v, j) => (ls[j] > 1 ? v / (ls[j] - 1) : 0.5)) })
        for (let j = k - 1; j >= 0; j--) { idx[j]++; if (idx[j] < ls[j]) break; idx[j] = 0 }
      }
      return out
    }
    case 'random': return Array.from({ length: n }, () => ({ u: Array.from({ length: k }, () => rand()) }))
    case 'lhs': {
      const cols = Array.from({ length: k }, () => {
        const perm = Array.from({ length: n }, (_, i) => i)
        for (let i = n - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [perm[i], perm[j]] = [perm[j], perm[i]] }
        return perm.map((p) => (p + rand()) / n)
      })
      return Array.from({ length: n }, (_, i) => ({ u: cols.map((c) => c[i]) }))
    }
    case 'sobol': { const sh = Array.from({ length: k }, () => rand()); return Array.from({ length: n }, (_, i) => ({ u: haltonPoint(i, k, sh) })) }
    case 'saltelli': {
      const sh = Array.from({ length: 2 * k }, () => rand())
      const AB = Array.from({ length: n }, (_, i) => haltonPoint(i, 2 * k, sh))
      const A = AB.map((r) => r.slice(0, k)), B = AB.map((r) => r.slice(k))
      const out: UnitPoint[] = []
      A.forEach((u, row) => out.push({ u, block: 'A', row }))
      B.forEach((u, row) => out.push({ u, block: 'B', row }))
      for (let i = 0; i < k; i++) A.forEach((u, row) => out.push({ u: u.map((v, j) => (j === i ? B[row][j] : v)), block: `AB:${i}`, row }))
      return out
    }
    case 'morris': {
      const p = Math.max(2, levels)
      const delta = p / (2 * (p - 1))
      const grid = Array.from({ length: p / 2 }, (_, i) => i / (p - 1))
      const out: UnitPoint[] = []
      for (let t = 0; t < n; t++) {
        const x = Array.from({ length: k }, () => grid[Math.floor(rand() * grid.length)])
        const order = Array.from({ length: k }, (_, i) => i)
        for (let i = k - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [order[i], order[j]] = [order[j], order[i]] }
        out.push({ u: [...x], traj: t, step: 0 })
        order.forEach((pi, s) => { x[pi] += delta; out.push({ u: [...x], traj: t, step: s + 1, stepParam: params[pi].label ?? String(pi), direction: 1 }) })
      }
      return out
    }
  }
}

/** Map a unit-cube coordinate to a parameter value. */
export function unitToValue(p: ParameterSpec, u: number): number | string | boolean {
  if (p.kind === 'choice') { const v = p.values ?? []; return v[Math.min(v.length - 1, Math.floor(u * v.length))] }
  const lo = p.low ?? 0, hi = p.high ?? 1
  if (p.kind === 'integer') return Math.min(Math.floor(hi), Math.ceil(lo) + Math.floor(u * (Math.floor(hi) - Math.ceil(lo) + 1)))
  if (p.scale === 'log') return Math.exp(Math.log(lo) + u * (Math.log(hi) - Math.log(lo)))
  return lo + u * (hi - lo)
}

// ---------------------------------------------------------------------------------------------- estimators
const mean = (a: number[]) => a.reduce((x, y) => x + y, 0) / (a.length || 1)
const variance = (a: number[]) => { const m = mean(a); return a.reduce((s, v) => s + (v - m) ** 2, 0) / Math.max(1, a.length - 1) }

export interface SobolEst { S1: number; ST: number }
/** Sobol' indices from the Saltelli blocks: fA, fB (n) and fAB[i] (n each). */
export function sobolIndices(fA: number[], fB: number[], fAB: number[][]): SobolEst[] {
  const V = variance([...fA, ...fB])
  const n = fA.length
  return fAB.map((f) => {
    let s1 = 0, st = 0
    for (let j = 0; j < n; j++) { s1 += fB[j] * (f[j] - fA[j]); st += (fA[j] - f[j]) ** 2 }
    return { S1: V > 0 ? s1 / n / V : NaN, ST: V > 0 ? st / (2 * n) / V : NaN }
  })
}
/** Percentile bootstrap (95 %) of the Sobol' indices over the base samples. */
export function sobolBootstrap(fA: number[], fB: number[], fAB: number[][], nBoot = 200, seed = 3): { S1: [number, number][]; ST: [number, number][] } {
  const rand = rng(seed), n = fA.length, k = fAB.length
  const s1: number[][] = Array.from({ length: k }, () => []), st: number[][] = Array.from({ length: k }, () => [])
  for (let b = 0; b < nBoot; b++) {
    const ix = Array.from({ length: n }, () => Math.floor(rand() * n))
    const e = sobolIndices(ix.map((i) => fA[i]), ix.map((i) => fB[i]), fAB.map((f) => ix.map((i) => f[i])))
    e.forEach((v, i) => { s1[i].push(v.S1); st[i].push(v.ST) })
  }
  const ci = (a: number[]): [number, number] => { const s = a.filter(Number.isFinite).sort((x, y) => x - y); return s.length ? [s[Math.floor(0.025 * s.length)], s[Math.min(s.length - 1, Math.floor(0.975 * s.length))]] : [NaN, NaN] }
  return { S1: s1.map(ci), ST: st.map(ci) }
}
/** Elementary effects of a Morris design: per trajectory, EE_i = (f(x + Δ e_i) − f(x)) / Δ. */
export function morrisIndices(f: number[], pts: UnitPoint[], paramLabels: string[], delta: number): { mu: number; mu_star: number; sigma: number; n: number }[] {
  const ee: number[][] = paramLabels.map(() => [])
  for (let r = 1; r < pts.length; r++) {
    const p = pts[r]
    if (p.step && p.step > 0 && pts[r - 1].traj === p.traj) { const i = paramLabels.indexOf(p.stepParam ?? ''); if (i >= 0) ee[i].push((f[r] - f[r - 1]) / delta) }
  }
  return ee.map((e) => ({ mu: mean(e), mu_star: mean(e.map(Math.abs)), sigma: Math.sqrt(variance(e)), n: e.length }))
}
export const varianceOf = variance
export const meanOf = mean
