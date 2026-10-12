import type { EemResult } from '../api/types'
import { interp } from './metrics'

export interface Aligned { exs: number[]; grid: number[]; a: number[][]; b: number[][] }

/** Put two EEMs on A's emission grid, keeping only excitations present in both (NaN stays NaN). */
export function alignEem(A: EemResult, B: EemResult): Aligned {
  const exs = A.excitations_nm.filter((e) => B.excitations_nm.some((f) => Math.abs(f - e) < 0.5))
  const grid = A.grid_emission_nm
  const row = (r: EemResult, e: number): number[] => {
    const i = r.excitations_nm.findIndex((f) => Math.abs(f - e) < 0.5)
    const src = r.grid_value[i]
    if (r.grid_emission_nm.length === grid.length && r.grid_emission_nm.every((g, k) => Math.abs(g - grid[k]) < 1e-6)) return src
    return grid.map((g) => {
      if (g < r.grid_emission_nm[0] || g > r.grid_emission_nm[r.grid_emission_nm.length - 1]) return NaN
      return interp(r.grid_emission_nm, src, g)
    })
  }
  return { exs, grid, a: exs.map((e) => row(A, e)), b: exs.map((e) => row(B, e)) }
}

export const diffGrid = (al: Aligned): number[][] => al.a.map((r, i) => r.map((v, k) => v - al.b[i][k]))
/** log2(A/B): symmetric around 0, NaN when either value is non-positive. */
export const log2RatioGrid = (al: Aligned): number[][] => al.a.map((r, i) => r.map((v, k) => (v > 0 && al.b[i][k] > 0 ? Math.log2(v / al.b[i][k]) : NaN)))

/** Value (and std) of one excitation row at an emission wavelength. */
export function atEmission(r: EemResult, ex: number, em: number): { v: number; s: number } {
  const key = String(r.excitations_nm.find((e) => Math.abs(e - ex) < 0.5) ?? ex)
  const x = r.emission_nm[key]
  if (!x) return { v: NaN, s: NaN }
  return { v: interp(x, r.value[key], em), s: interp(x, r.std[key], em) }
}

/** Emission wavelength of the brightest band of excitation `ex`. */
export function brightestBand(r: EemResult, ex: number): number | null {
  const key = String(r.excitations_nm.find((e) => Math.abs(e - ex) < 0.5) ?? ex)
  const x = r.emission_nm[key], y = r.value[key]
  if (!x || !y?.length) return null
  let bi = 0
  for (let i = 1; i < y.length; i++) if (Number.isFinite(y[i]) && (y[i] > y[bi] || !Number.isFinite(y[bi]))) bi = i
  return x[bi]
}

/** "dye:NADH@tissue" → "NADH · tissue", "raman@cover" → "Raman · cover" */
export function componentLabel(name: string): string {
  const [head, layer] = name.split('@')
  let base = head
  if (head.startsWith('dye:')) base = head.slice(4)
  else if (head.startsWith('secondary:')) base = `${head.slice(10)} (re-emitted)`
  else if (head === 'raman') base = 'water Raman'
  else if (head === 'elastic') base = 'elastic scatter'
  else if (head === 'glint') base = 'specular glint'
  else if (head === 'leak_source') base = 'excitation leak'
  else if (head === 'room_light') base = 'room light'
  else if (head.startsWith('background:')) base = `background (${head.slice(11).replace(/_/g, ' ')})`
  return layer ? `${base} · ${layer}` : base
}
