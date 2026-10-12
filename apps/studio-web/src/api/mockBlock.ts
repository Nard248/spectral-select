/** Mock-engine physics for `layered_block` objects: plausible (not rigorous) emission of buried layers seen through
 *  an absorbing / scattering cover, ground truth maps and fake 3D light-field volumes.
 *
 *  Model per layer i with thickness t_i, attenuation k_i(λ) = μa_i(λ) + 0.3·μs′_i(λ) (a cheap stand-in for the diffusion
 *  decay rate) and the layers above summing to the optical depth A(λ) = Σ_{j>i} k_j t_j:
 *     S_i(λ_em) = κ_dye(λ_ex) · Φ · e^{−A(λ_ex)} · e^{−A(λ_em)} · (1 − e^{−(k_i(λ_ex)+k_i(λ_em)) t_i}) / (k_i(λ_ex)+k_i(λ_em))
 *  i.e. excitation reaching the layer × emission escaping upward, integrated over the layer. */
import * as THREE from 'three'
import type { Experiment, LayeredBlockShape, MaterialSpec } from '../generated/experiment'
import type { BlockInfo, VolumeInfo } from './types'
import type { VolumeData } from './client'
import { emissionShape, EXTINCTION_REFERENCE, FWHM_TO_SIGMA, lineAveragedExcitation, materialMua, musPrime } from '../lib/spectral'
import { fluorophoreMap } from '../state/presets'
import { beamAt } from '../lib/lighting'
import { layerDepthTop, stackAt } from '../analysis/blockGeometry'
import { activeSource } from '../state/store'

const SCAT_ATT = 0.3
/** cosmetic brightness of buried dyes relative to the flat-sample mock so thin-layer signals clear the 100 ADU offset */
const DYE_BOOST = 8

export function blocksOf(exp: Experiment): { index: number; id: string; name: string; shape: LayeredBlockShape }[] {
  const out: { index: number; id: string; name: string; shape: LayeredBlockShape }[] = []
  exp.objects.forEach((o, index) => { if (o.shape.kind === 'layered_block') out.push({ index, id: o.id, name: o.name, shape: o.shape }) })
  return out
}

export function blockInfos(exp: Experiment): BlockInfo[] {
  return blocksOf(exp).map((b) => ({ object_id: b.id, layer_names: b.shape.layers.map((l) => l.name), footprint_mm: [b.shape.footprint_mm[0], b.shape.footprint_mm[1]] }))
}

export function blockGtNames(exp: Experiment): string[] {
  const names = new Set<string>()
  for (const b of blocksOf(exp)) for (const l of b.shape.layers) { names.add(`layer_thickness:${l.name}`); names.add(`layer_depth:${l.name}`); names.add(`origin_layer_fraction:${l.name}`) }
  if (names.size) { names.add('block_face'); names.add('origin_depth_mm') }
  return [...names]
}

interface Optics { mua: number; musp: number; att: number }
function optics(m: MaterialSpec, fl: ReturnType<typeof fluorophoreMap>, wl: number, ex: Experiment): Optics {
  const ph = ex.acquisition.physics
  const mua = ph.absorption ? materialMua(m, fl, wl) : 0
  const musp = ph.scattering ? musPrime(m.scattering, wl) : 0
  return { mua, musp, att: mua + SCAT_ATT * musp }
}
const integral = (a: number, t: number) => (a > 1e-9 ? (1 - Math.exp(-a * t)) / a : t)

export interface StepInfo { ex: number; powerMW: number; exposureMs: number }

/** Per-component spectra (arrays over `bands`) of one block pixel at local footprint position (lx, lz), WITHOUT beam shading. */
export function blockPixelComponents(exp: Experiment, shape: LayeredBlockShape, lx: number, lz: number, step: StepInfo, bands: number[],
  opts: { gain: number; bandStep: number; sigmaBand: number; lineFwhm: number }): Record<string, Float32Array> {
  const fl = fluorophoreMap(exp)
  const mats = new Map(exp.materials.map((m) => [m.id, m] as const))
  const ph = exp.acquisition.physics
  const st = stackAt(shape, lx, lz)
  const L = shape.layers.length
  const mat = shape.layers.map((l) => mats.get(l.material_id))
  const optEx = mat.map((m) => (m ? optics(m, fl, step.ex, exp) : { mua: 0, musp: 0, att: 0 }))
  const optEm = bands.map((wl) => mat.map((m) => (m ? optics(m, fl, wl, exp) : { mua: 0, musp: 0, att: 0 })))
  const out: Record<string, Float32Array> = {}
  const scale = opts.gain
  for (let i = 0; i < L; i++) {
    const m = mat[i]; const t = st.t[i]
    if (!m || t < 1e-6) continue
    let aEx = 0
    for (let j = i + 1; j < L; j++) aEx += optEx[j].att * st.t[j]
    const exUp = Math.exp(-aEx)
    if (ph.fluorescence) for (const d of m.dyes) {
      const f = fl[d.fluorophore]; if (!f) continue
      const eps = f.extinction * EXTINCTION_REFERENCE * lineAveragedExcitation(f, step.ex, opts.lineFwhm)
      const kappa = (Math.LN10 * eps * d.concentration_uM * 1e-6) / 10        // 1/mm
      const quench = ph.quenching ? 1 / (1 + m.quenching_ksv_per_uM * d.concentration_uM) : 1
      const sig = f.em_fwhm_nm * FWHM_TO_SIGMA * (1 + f.em_skew)
      const arr = new Float32Array(bands.length)
      bands.forEach((wl, b) => {
        let aEm = 0
        for (let j = i + 1; j < L; j++) aEm += optEm[b][j].att * st.t[j]
        const I = integral(optEx[i].att + optEm[b][i].att, t)
        arr[b] = DYE_BOOST * scale * kappa * f.quantum_yield * quench * exUp * Math.exp(-aEm) * I * (emissionShape(f, wl) / (sig * Math.sqrt(2 * Math.PI))) * opts.bandStep
      })
      out[`dye:${d.fluorophore}@${shape.layers[i].name}`] = arr
    }
    if (ph.raman && m.aqueous) {
      const wr = 1e7 / (1e7 / step.ex - 3400)
      const arr = new Float32Array(bands.length)
      bands.forEach((wl, b) => {
        let aEm = 0
        for (let j = i + 1; j < L; j++) aEm += optEm[b][j].att * st.t[j]
        arr[b] = scale * 4e-5 * m.raman_strength * exUp * Math.exp(-aEm) * integral(optEx[i].att + optEm[b][i].att, t) * Math.exp(-0.5 * ((wl - wr) / (6.4 + opts.sigmaBand)) ** 2)
      })
      out[`raman@${shape.layers[i].name}`] = arr
    }
  }
  if (ph.elastic_leak) {
    let top = L - 1
    while (top > 0 && st.t[top] < 1e-6) top--
    const refl = 0.02 + 0.2 * Math.min(1, optEx[top].musp * 0.1)
    const arr = new Float32Array(bands.length)
    bands.forEach((wl, b) => { arr[b] = scale * refl * 4e-4 * Math.exp(-0.5 * ((wl - step.ex) / (opts.sigmaBand * 1.2)) ** 2) })
    out.elastic = arr
  }
  return out
}

/** Ground-truth scalars of one block pixel (names as in the engine contract). */
export function blockGround(exp: Experiment, shape: LayeredBlockShape, lx: number, lz: number, face: number): Record<string, number> {
  const mats = new Map(exp.materials.map((m) => [m.id, m] as const))
  const fl = fluorophoreMap(exp)
  const st = stackAt(shape, lx, lz)
  const out: Record<string, number> = { block_face: face }
  const w: number[] = []
  shape.layers.forEach((l, i) => {
    const m = mats.get(l.material_id)
    const depth = layerDepthTop(st, i)
    out[`layer_thickness:${l.name}`] = st.t[i]
    out[`layer_depth:${l.name}`] = depth
    const k500 = m ? optics(m, fl, 500, exp).att : 0
    const fluorescent = m && m.dyes.length ? 1 : 0.02
    w.push(fluorescent * (1 - Math.exp(-0.4 * st.t[i])) * Math.exp(-depth * (k500 + 0.1)))
  })
  const sw = w.reduce((a, b) => a + b, 0) || 1
  let od = 0
  shape.layers.forEach((l, i) => { out[`origin_layer_fraction:${l.name}`] = w[i] / sw; od += (w[i] / sw) * (layerDepthTop(st, i) + st.t[i] / 2) })
  out.origin_depth_mm = od
  return out
}

// ---- 3D light field -----------------------------------------------------------------------------

export function volumeInfos(exp: Experiment, excitations: number[]): VolumeInfo[] {
  const out: VolumeInfo[] = []
  const vm = exp.acquisition.volume_mc
  for (const b of blocksOf(exp)) {
    const [Lx, Lz] = b.shape.footprint_mm
    const nx = Math.max(4, Math.round(Lx / vm.lateral_cell_mm)), nz = Math.max(4, Math.round(Lz / vm.lateral_cell_mm))
    const nL = b.shape.layers.length, S = vm.sublayers
    const edges = (n: number, Ltot: number) => Array.from({ length: n + 1 }, (_, i) => -Ltot / 2 + (Ltot * i) / n)
    const base = { object_id: b.id, excitations_nm: excitations, dims: [nx, nz, nL * S] as [number, number, number], x_edges_mm: edges(nx, Lx), z_edges_mm: edges(nz, Lz), layer_names: b.shape.layers.map((l) => l.name), sublayers: S }
    out.push({ ...base, kind: 'fluence', channels: [], unit: 'mW mm⁻²' })
    out.push({ ...base, kind: 'absorbed', channels: [], unit: 'mW' })
    const mats = new Map(exp.materials.map((m) => [m.id, m] as const))
    const channels: string[] = []
    b.shape.layers.forEach((l) => mats.get(l.material_id)?.dyes.forEach((d) => channels.push(`${l.name}|${d.fluorophore}`)))
    if (channels.length) out.push({ ...base, kind: 'source', channels, unit: 'mW (emitted)' })
  }
  return out
}

export function volumeData(exp: Experiment, q: { object: string; kind: VolumeInfo['kind']; ex?: number; channel?: string }): VolumeData {
  const b = blocksOf(exp).find((x) => x.id === q.object)
  if (!b) throw new Error(`unknown block object "${q.object}"`)
  const info = volumeInfos(exp, []).find((v) => v.object_id === q.object)!
  const [nx, nz, nd] = info.dims
  const S = info.sublayers, L = b.shape.layers.length
  const fl = fluorophoreMap(exp)
  const mats = new Map(exp.materials.map((m) => [m.id, m] as const))
  const src = activeSource(exp)
  const ex = q.ex ?? src.sequence[0].wavelength_nm
  const obj = exp.objects[b.index]
  const R = new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(...obj.transform.rotation_deg.map((x) => (x * Math.PI) / 180) as [number, number, number], 'XYZ'))
  const optEx = b.shape.layers.map((l) => { const m = mats.get(l.material_id); return m ? optics(m, fl, ex, exp) : { mua: 0, musp: 0, att: 0 } })
  const [chLayer, chDye] = (q.channel ?? '').split('|')
  const chIdx = b.shape.layers.findIndex((l) => l.name === chLayer)
  const data = new Float32Array(nx * nz * nd)
  const dx = (info.x_edges_mm[1] - info.x_edges_mm[0]), dz = (info.z_edges_mm[1] - info.z_edges_mm[0])
  for (let ix = 0; ix < nx; ix++) {
    const x = (info.x_edges_mm[ix] + info.x_edges_mm[ix + 1]) / 2
    for (let iz = 0; iz < nz; iz++) {
      const z = (info.z_edges_mm[iz] + info.z_edges_mm[iz + 1]) / 2
      const st = stackAt(b.shape, x, z)
      const p = new THREE.Vector3(x, st.top, z).applyMatrix4(R)
      const irr = beamAt(src, [p.x + obj.transform.position_mm[0], p.y + obj.transform.position_mm[1], p.z + obj.transform.position_mm[2]], null).irradiance
      const base = (ix * nz + iz) * nd
      for (let i = 0; i < L; i++) {
        const t = st.t[i]; if (t < 1e-9) continue
        let aUp = 0
        for (let j = i + 1; j < L; j++) aUp += optEx[j].att * st.t[j]
        const m = mats.get(b.shape.layers[i].material_id)
        const dye = m?.dyes.find((d) => d.fluorophore === chDye)
        for (let s = 0; s < S; s++) {
          const u = ((s + 0.5) / S) * t                       // height within the layer (bottom → top sublayers)
          const depthInLayer = t - u
          const buildUp = 1 + 1.2 * Math.exp(-(aUp + depthInLayer) * (optEx[0].musp * 0.5 + 0.3))
          const phi = irr * buildUp * Math.exp(-aUp - optEx[i].att * depthInLayer)
          const dV = dx * dz * (t / S)
          let v = phi
          if (q.kind === 'absorbed') v = optEx[i].mua * phi * dV
          else if (q.kind === 'source') {
            if (i !== chIdx || !dye) v = 0
            else { const f = fl[dye.fluorophore]; const eps = f.extinction * EXTINCTION_REFERENCE * lineAveragedExcitation(f, ex, src.line_fwhm_nm); v = (Math.LN10 * eps * dye.concentration_uM * 1e-6 / 10) * f.quantum_yield * phi * dV }
          }
          data[base + i * S + s] = v
        }
      }
    }
  }
  return { dims: [nx, nz, nd], data }
}

