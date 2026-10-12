/** Light-field overlay: fetch a block's 3D tally volume from the engine and paint a log-coloured slice on the cutaway plane.
 *
 *  Volume layout (types.ts): float32 LE, dims (nx, nz, nd = n_layers·sublayers), C-order, local block frame, layer-
 *  conforming — sublayer s of layer l sits at fraction (s + ½)/S across that layer's local thickness (bottom → top).
 *  The slice texture stacks the layers vertically, `ROWS` rows each, so it maps 1:1 onto `buildSectionStrips`. */
import * as THREE from 'three'
import { useEffect, useMemo, useRef, useState } from 'react'
import { lut, type CmapName } from '../ui/colormaps'
import type { VolumeInfo } from '../api/types'
import { useStudio } from '../state/store'
import { getEngine } from '../api/client'
import { useLayerView, type LightKind } from '../state/layerView'
import type { LayeredBlockShape } from '../generated/experiment'
import { interfaceHeights } from '../three/layers/profile'

export const ROWS = 160

export interface Volume { info: VolumeInfo; data: Float32Array; dims: [number, number, number]; vmax: number; demo?: boolean }
export interface Legend { vmax: number; vmin: number; unit: string; label: string; cmap: CmapName; demo?: boolean }

/* ----------------------------------------------------------------- fetching */
/** Volumes go through the active engine object (real engine = plain HTTP, mock engine = in-browser) with a raw-fetch fallback. */
export async function fetchVolumeInfos(captureId: string): Promise<VolumeInfo[]> {
  const eng = getEngine()
  if (eng) return eng.volumes(captureId)
  const r = await fetch(`/api/captures/${captureId}/volumes`)
  if (!r.ok) throw new Error(`GET volumes → ${r.status}`)
  return (await r.json()) as VolumeInfo[]
}

export async function fetchVolume(captureId: string, info: VolumeInfo, kind: LightKind, ex: number | null, channel: string | null): Promise<Volume> {
  const eng = getEngine()
  if (eng) {
    const v = await eng.volume(captureId, { object: info.object_id, kind, ex: ex ?? undefined, channel: channel ?? undefined })
    return { info, data: v.data, dims: v.dims, vmax: robustMax(v.data) }
  }
  const q = new URLSearchParams({ object: info.object_id, kind })
  if (ex != null) q.set('ex', String(ex))
  if (channel) q.set('channel', channel)
  const r = await fetch(`/api/captures/${captureId}/volume?${q}`)
  if (!r.ok) throw new Error(`GET volume → ${r.status} ${(await r.text().catch(() => '')).slice(0, 160)}`)
  const dims = (r.headers.get('X-Dims') ?? '').split(',').map(Number) as [number, number, number]
  const data = new Float32Array(await r.arrayBuffer())
  if (dims.length !== 3 || dims.some((d) => !(d > 0)) || dims[0] * dims[1] * dims[2] !== data.length) throw new Error(`volume has inconsistent X-Dims "${dims.join(',')}" for ${data.length} values`)
  return { info, data, dims, vmax: robustMax(data) }
}

/** 99.9th percentile of the positive values (a robust maximum that ignores single hot entry voxels). */
export function robustMax(a: Float32Array): number {
  const step = Math.max(1, Math.floor(a.length / 200_000))
  const s: number[] = []
  for (let i = 0; i < a.length; i += step) if (a[i] > 0 && Number.isFinite(a[i])) s.push(a[i])
  if (!s.length) return 0
  s.sort((x, y) => x - y)
  return s[Math.min(s.length - 1, Math.floor(0.999 * s.length))]
}

/** The requested source channel if it exists, else the first non-Raman one (the dyes are what people look for). */
export const defaultChannel = (channels: string[], want: string | null): string | null =>
  want && channels.includes(want) ? want : channels.find((c) => !/raman/i.test(c)) ?? channels[0] ?? null

/* ----------------------------------------------------------------- slicing → texture */
const findCell = (edges: number[], c: number) => {
  let lo = 0, hi = edges.length - 2
  while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (edges[mid] <= c) lo = mid; else hi = mid - 1 }
  return Math.min(Math.max(lo, 0), edges.length - 2)
}

export interface SliceOptions { axis: 'x' | 'z'; position_mm: number; decades: number; cmap: CmapName }

/** Build / update the RGBA slice texture for the section plane. Pass the previous texture to reuse its buffer. */
export function sliceTexture(vol: Volume, opt: SliceOptions, prev?: THREE.DataTexture | null): THREE.DataTexture {
  const { info, data, dims } = vol
  const [nx, nz, nd] = dims
  const L = info.layer_names.length, S = info.sublayers
  const horizontal = opt.axis === 'x' ? nz : nx
  const W = Math.min(768, Math.max(64, horizontal * 6))
  const H = ROWS
  const rgba = prev && prev.image.width === W && prev.image.height === L * H ? (prev.image.data as Uint8Array) : new Uint8Array(W * L * H * 4)
  const cell = opt.axis === 'x' ? findCell(info.x_edges_mm, opt.position_mm) : findCell(info.z_edges_mm, opt.position_mm)
  const val = (h: number, d: number) => (opt.axis === 'x' ? data[(cell * nz + h) * nd + d] : data[(h * nz + cell) * nd + d])
  const vmax = Math.max(vol.vmax, 1e-30)
  const lmax = Math.log10(vmax), floor = vmax * Math.pow(10, -(opt.decades + 1))
  const cm = lut(opt.cmap)
  // log10 of the slice once (floor-clamped), then cheap bilinear lookups per pixel
  const lg = new Float32Array(horizontal * nd)
  for (let h = 0; h < horizontal; h++) for (let d = 0; d < nd; d++) lg[h * nd + d] = Math.log10(Math.max(val(h, d), floor))
  const lvb = new Float32Array(W * L * H)
  for (let l = 0; l < L; l++) {
    for (let py = 0; py < H; py++) {
      const gs = Math.min(S - 1, Math.max(0, ((py + 0.5) / H) * S - 0.5))
      const s0 = Math.floor(gs), s1 = Math.min(S - 1, s0 + 1), fs = gs - s0
      const d0 = l * S + s0, d1 = l * S + s1
      for (let px = 0; px < W; px++) {
        const gh = Math.min(horizontal - 1, Math.max(0, ((px + 0.5) / W) * horizontal - 0.5))
        const h0 = Math.floor(gh), h1 = Math.min(horizontal - 1, h0 + 1), fh = gh - h0
        lvb[(l * H + py) * W + px] = (lg[h0 * nd + d0] * (1 - fh) + lg[h1 * nd + d0] * fh) * (1 - fs) + (lg[h0 * nd + d1] * (1 - fh) + lg[h1 * nd + d1] * fh) * fs
      }
    }
  }
  // colour map + one faint iso-line per decade (marks where the light has dropped by another factor of 10)
  for (let l = 0; l < L; l++) for (let py = 0; py < H; py++) for (let px = 0; px < W; px++) {
    const i = (l * H + py) * W + px, v = lvb[i]
    const t = Math.min(1, Math.max(0, (v - (lmax - opt.decades)) / opt.decades))
    const ci = Math.round(t * 255) * 3
    let r = cm[ci], g = cm[ci + 1], b = cm[ci + 2]
    const f = Math.floor(v)
    const edge = (px + 1 < W && Math.floor(lvb[i + 1]) !== f) || (py + 1 < H && Math.floor(lvb[i + W]) !== f)
    if (edge && v > lmax - opt.decades && v <= lmax) { r = r * 0.62 + 255 * 0.38; g = g * 0.62 + 255 * 0.38; b = b * 0.62 + 255 * 0.38 }
    const o = i * 4
    rgba[o] = r; rgba[o + 1] = g; rgba[o + 2] = b; rgba[o + 3] = 255
  }
  let tex = prev
  if (!tex || tex.image.data !== rgba) {
    tex?.dispose()
    tex = new THREE.DataTexture(rgba, W, L * H, THREE.RGBAFormat, THREE.UnsignedByteType)
    tex.colorSpace = THREE.SRGBColorSpace
    tex.magFilter = THREE.LinearFilter; tex.minFilter = THREE.LinearFilter
    tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping
    tex.generateMipmaps = false
  }
  tex.needsUpdate = true
  return tex
}

/* ----------------------------------------------------------------- dev/demo field (no engine) */
/** Illustrative diffusion-style fluence for QA screenshots when no engine capture exists. NEVER shown unless
 *  `?lfdemo=1` is in the URL of a dev build; the legend labels it as a demo. */
export function demoVolume(shape: LayeredBlockShape, objectId: string, mus: number[], mua: number[]): Volume {
  const [fx, fz] = shape.footprint_mm
  const L = shape.layers.length, S = 8
  const nx = Math.round(fx / 0.5), nz = Math.round(fz / 0.5), nd = L * S
  const xe = Array.from({ length: nx + 1 }, (_, i) => -fx / 2 + (fx * i) / nx)
  const ze = Array.from({ length: nz + 1 }, (_, i) => -fz / 2 + (fz * i) / nz)
  const data = new Float32Array(nx * nz * nd)
  const ent = [-fx * 0.2, 0, 0]                                   // beam enters left of centre, tilted +x (oblique)
  const dir = [Math.sin(0.62), -Math.cos(0.62), 0]
  const top = interfaceHeights(shape, ent[0], ent[2])[L]
  // ballistic path samples: position, accumulated optical depth, and which layer they are in
  const path: { x: number; y: number; z: number; w: number }[] = []
  let tau = 0
  for (let k = 0; k < 40; k++) {
    const u = k * 0.18, sx = ent[0] + dir[0] * u, sy = top + dir[1] * u
    const hh = interfaceHeights(shape, sx, ent[2]); let lay = sy > hh[L] ? L - 1 : 0
    for (let j = 0; j < L; j++) if (sy >= hh[j] && sy <= hh[j + 1]) { lay = j; break }
    tau += (mua[lay] + mus[lay] * 0.18) * 0.18
    path.push({ x: sx, y: sy, z: ent[2], w: Math.exp(-tau * 0.55) })
  }
  const meff = mua.map((a, l) => Math.sqrt(3 * a * (a + mus[l])))
  for (let ix = 0; ix < nx; ix++) for (let iz = 0; iz < nz; iz++) {
    const x = (xe[ix] + xe[ix + 1]) / 2, z = (ze[iz] + ze[iz + 1]) / 2
    const h = interfaceHeights(shape, x, z)
    for (let l = 0; l < L; l++) for (let s = 0; s < S; s++) {
      const y = h[l] + ((h[l + 1] - h[l]) * (s + 0.5)) / S
      let phi = 0
      for (const p of path) {   // sum of isotropic point sources along the ballistic path
        const r = Math.hypot(x - p.x, y - p.y, z - p.z) + 0.12
        phi += (p.w * Math.exp(-meff[l] * r)) / r
      }
      data[(ix * nz + iz) * nd + l * S + s] = phi
    }
  }
  const info: VolumeInfo = { object_id: objectId, kind: 'fluence', excitations_nm: [365], channels: [], dims: [nx, nz, nd], x_edges_mm: xe, z_edges_mm: ze, layer_names: (shape.layers as { name: string }[]).map((l) => l.name), sublayers: S, unit: 'a.u. (demo)' }
  return { info, data, dims: [nx, nz, nd], vmax: robustMax(data), demo: true }
}

/* ----------------------------------------------------------------- React hook */
export interface LightFieldResult { texture: THREE.Texture | null; infos: VolumeInfo[]; info: VolumeInfo | null }

/** Loads the volume for (active capture, block) per the light-field settings and returns the slice texture for the given cut.
 *  Updates `useLayerView.lfStatus` for the inspector. */
export function useLightField(objectId: string, shape: LayeredBlockShape | null, enabled: boolean, cut: { axis: 'x' | 'z'; position_mm: number } | null, demoMus?: { mus: number[]; mua: number[] }): LightFieldResult {
  const captureId = useStudio((s) => s.activeCaptureId)
  const meta = useStudio((s) => s.captures.find((c) => c.id === s.activeCaptureId))
  const engineKind = useStudio((s) => s.engine)
  const lf = useLayerView((s) => s.lf)
  const setStatus = useLayerView((s) => s.setLfStatus)
  const [infos, setInfosState] = useState<VolumeInfo[]>([])
  const [vol, setVol] = useState<Volume | null>(null)
  const texRef = useRef<THREE.DataTexture | null>(null)
  const [tex, setTex] = useState<THREE.Texture | null>(null)
  const demo = import.meta.env.DEV && typeof location !== 'undefined' && /[?&]lfdemo=1/.test(location.search)
  const metaVolumes = (meta as (typeof meta & { volumes?: VolumeInfo[] }) | undefined)?.volumes

  // 1. which volumes exist
  useEffect(() => {
    let dead = false
    const setInfos = (v: VolumeInfo[]) => { setInfosState(v); useLayerView.getState().setLfInfos(v) }
    if (!enabled || !shape) return
    if (demo && demoMus) { setInfos([demoVolume(shape, objectId, demoMus.mus, demoMus.mua).info]); return }
    if (!captureId) { setInfos([]); setStatus({ state: 'none', message: 'Run a capture first — the light field is a by-product of the 3D Monte Carlo.' }); return }
    if (metaVolumes) { setInfos(metaVolumes.filter((v) => v.object_id === objectId)); return }
    fetchVolumeInfos(captureId).then((v) => { if (!dead) setInfos(v.filter((x) => x.object_id === objectId)) })
      .catch((e: Error) => { if (!dead) { setInfos([]); setStatus({ state: 'error', message: e.message }) } })
    return () => { dead = true }
  }, [enabled, captureId, objectId, shape, engineKind, metaVolumes, demo, demoMus, setStatus])

  // 2. the volume itself
  const info = useMemo(() => infos.find((v) => v.kind === lf.kind) ?? null, [infos, lf.kind])
  useEffect(() => {
    let dead = false
    if (!enabled || !shape) { setVol(null); return }
    if (!info) { setVol(null); if (infos.length === 0 && useLayerView.getState().lfStatus.state === 'loading') setStatus({ state: 'none', message: 'No 3D volume stored for this block in the active capture.' }); return }
    if (demo && demoMus) { setVol(demoVolume(shape, objectId, demoMus.mus, demoMus.mua)); setStatus({ state: 'ready', message: 'Illustrative demo field (dev only)' }); return }
    const ex = lf.ex != null && info.excitations_nm.includes(lf.ex) ? lf.ex : info.excitations_nm[0] ?? null
    const channel = lf.kind === 'source' ? defaultChannel(info.channels, lf.channel) : null
    setStatus({ state: 'loading', message: 'Loading volume…' })
    fetchVolume(captureId!, info, lf.kind, ex, channel).then((v) => { if (!dead) { setVol(v); setStatus({ state: 'ready' }) } })
      .catch((e: Error) => { if (!dead) { setVol(null); setStatus({ state: 'error', message: e.message }) } })
    return () => { dead = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, info, lf.ex, lf.channel, lf.kind, captureId, demo])

  // 3. slice → texture
  useEffect(() => {
    if (!enabled || !vol || !cut) { setTex(null); return }
    texRef.current = sliceTexture(vol, { axis: cut.axis, position_mm: cut.position_mm, decades: lf.decades, cmap: lf.cmap }, texRef.current)
    setTex(texRef.current)
    useLayerView.getState().setLfLegend({ vmax: vol.vmax, vmin: vol.vmax * Math.pow(10, -lf.decades), unit: vol.info.unit, label: vol.info.kind, cmap: lf.cmap, demo: vol.demo })
  }, [enabled, vol, cut?.axis, cut?.position_mm, lf.decades, lf.cmap]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => () => { texRef.current?.dispose(); texRef.current = null }, [])
  useEffect(() => { if (!enabled) useLayerView.getState().setLfLegend(null) }, [enabled])

  return { texture: enabled ? tex : null, infos, info }
}
