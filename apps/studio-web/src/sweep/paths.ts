/** Enumerate the numeric parameters of an experiment (for the sweep picker) and read / write them by JSON pointer. */
import type { Experiment } from '../generated/experiment'

export interface ParamOption {
  path: string
  label: string
  group: string
  unit: string
  value: number
  /** layer name for layered-block thickness parameters */
  layer?: string
}

const UNITS: [RegExp, string, string][] = [
  [/_per_uM$/, 'µM⁻¹', ''], [/_per_mm$/, 'mm⁻¹', ''], [/_e_per_adu$/, 'e⁻/ADU', ''], [/_e_per_s$/, 'e⁻/s', ''], [/_e$/, 'e⁻', ''],
  [/_mm$/, 'mm', ''], [/_um$/, 'µm', ''], [/_nm$/, 'nm', ''], [/_uM$/, 'µM', ''], [/_mW$/, 'mW', ''], [/_ms$/, 'ms', ''], [/_pct$/, '%', ''],
  [/_deg$/, '°', ''], [/_px$/, 'px', ''], [/_s$/, 's', ''], [/_lux$/, 'lux', ''], [/_adu$/, 'ADU', ''], [/_od$/, 'OD', ''],
]
const WORDS: Record<string, string> = {
  mus: 'μs′', mua: 'μa', mw: 'mW', od: 'OD', qe: 'QE', ksv: 'Ksv', fwhm: 'FWHM', hsi: 'HSI', mc: 'MC', adu: 'ADU', prnu: 'PRNU', dsnu: 'DSNU', fret: 'FRET', px: 'px',
}

/** `end_thickness_mm` → { label: 'end thickness', unit: 'mm' } */
export function prettyKey(key: string): { label: string; unit: string } {
  let base = key, unit = ''
  for (const [re, u] of UNITS) if (re.test(base)) { unit = u; base = base.replace(re, ''); break }
  const label = base.split('_').filter(Boolean).map((w, i) => WORDS[w.toLowerCase()] ?? (i === 0 ? w : w)).join(' ')
  return { label, unit }
}

const SKIP_KEYS = new Set(['id', 'schema_version', 'seed', 'active_source_id', 'active_camera_id', 'bit_depth', 'axis'])

function numericLeaves(node: unknown, path: string, trail: string[], out: { path: string; trail: string[]; value: number }[], depth = 0): void {
  if (depth > 9 || node == null) return
  if (typeof node === 'number') { out.push({ path, trail, value: node }); return }
  if (Array.isArray(node)) {
    if (node.length && node.every((v) => typeof v === 'number') && node.length <= 3) {
      const axes = ['x', 'y', 'z']
      node.forEach((v, i) => out.push({ path: `${path}/${i}`, trail: [...trail, axes[i]], value: v as number }))
    } else if (node.length && node.every((v) => typeof v === 'object')) {
      node.forEach((v, i) => numericLeaves(v, `${path}/${i}`, [...trail, `#${i}`], out, depth + 1))
    }
    return
  }
  if (typeof node === 'object') {
    for (const [k, v] of Object.entries(node as Record<string, unknown>)) {
      if (SKIP_KEYS.has(k)) continue
      numericLeaves(v, `${path}/${k}`, [...trail, k], out, depth + 1)
    }
  }
}

export function enumerateParams(exp: Experiment): ParamOption[] {
  const opts: ParamOption[] = []
  const push = (path: string, label: string, group: string, unit: string, value: number, layer?: string) => opts.push({ path, label, group, unit, value, layer })
  const leaf = (key: string, trail: string[]) => {
    const p = prettyKey(key)
    return { label: [...trail, p.label].join(' › '), unit: p.unit }
  }

  exp.objects.forEach((o, oi) => {
    const sh = o.shape as unknown as Record<string, unknown>
    if (sh.kind === 'layered_block') {
      const layers = sh.layers as { name: string; material_id: string; thickness: { kind: string; thickness_mm: number; end_thickness_mm: number } }[]
      layers.forEach((l, li) => {
        const base = `/objects/${oi}/shape/layers/${li}/thickness`
        const lin = l.thickness.kind === 'linear'
        push(`${base}/thickness_mm`, `${o.name} › ${l.name} › ${lin ? 'start thickness' : 'thickness'}`, 'Layer thickness', 'mm', l.thickness.thickness_mm, l.name)
        if (lin) push(`${base}/end_thickness_mm`, `${o.name} › ${l.name} › end thickness`, 'Layer thickness', 'mm', l.thickness.end_thickness_mm, l.name)
      })
      const fp = sh.footprint_mm as number[]
      push(`/objects/${oi}/shape/footprint_mm/0`, `${o.name} › footprint x`, 'Objects', 'mm', fp[0])
      push(`/objects/${oi}/shape/footprint_mm/1`, `${o.name} › footprint z`, 'Objects', 'mm', fp[1])
    } else {
      const raw: { path: string; trail: string[]; value: number }[] = []
      numericLeaves(o.shape, `/objects/${oi}/shape`, [], raw)
      for (const r of raw) { const k = r.trail[r.trail.length - 1]; const l = leaf(k, [o.name, ...r.trail.slice(0, -1)]); push(r.path, l.label, 'Objects', l.unit, r.value) }
    }
    o.transform.position_mm.forEach((v, i) => push(`/objects/${oi}/transform/position_mm/${i}`, `${o.name} › position ${'xyz'[i]}`, 'Objects', 'mm', v))
    o.transform.rotation_deg.forEach((v, i) => push(`/objects/${oi}/transform/rotation_deg/${i}`, `${o.name} › rotation ${'xyz'[i]}`, 'Objects', '°', v))
  })

  exp.materials.forEach((m, mi) => {
    m.dyes.forEach((d, di) => push(`/materials/${mi}/dyes/${di}/concentration_uM`, `${m.name} › ${d.fluorophore} concentration`, 'Material dyes', 'µM', d.concentration_uM))
    const sc = m.scattering
    if (sc.model !== 'none') {
      push(`/materials/${mi}/scattering/mus_prime_500_per_mm`, `${m.name} › scattering μs′(500 nm)`, 'Material scattering', 'mm⁻¹', sc.mus_prime_500_per_mm)
      push(`/materials/${mi}/scattering/power_b`, `${m.name} › scattering power b`, 'Material scattering', '', sc.power_b)
      push(`/materials/${mi}/scattering/g`, `${m.name} › anisotropy g`, 'Material scattering', '', sc.g)
    }
    push(`/materials/${mi}/refractive_index`, `${m.name} › refractive index`, 'Material scattering', '', m.refractive_index)
    push(`/materials/${mi}/quenching_ksv_per_uM`, `${m.name} › quenching Ksv`, 'Material dyes', 'µM⁻¹', m.quenching_ksv_per_uM)
    push(`/materials/${mi}/surface_roughness`, `${m.name} › surface roughness`, 'Material scattering', '', m.surface_roughness)
  })

  exp.sources.forEach((s, si) => {
    s.sequence.forEach((st, k) => {
      const tag = `${s.name} › step ${k + 1} (${st.wavelength_nm} nm)`
      push(`/sources/${si}/sequence/${k}/power_mW`, `${tag} › power`, 'Source', 'mW', st.power_mW)
      push(`/sources/${si}/sequence/${k}/exposure_ms`, `${tag} › exposure`, 'Source', 'ms', st.exposure_ms)
      push(`/sources/${si}/sequence/${k}/wavelength_nm`, `${tag} › wavelength`, 'Source', 'nm', st.wavelength_nm)
    })
    push(`/sources/${si}/line_fwhm_nm`, `${s.name} › line FWHM`, 'Source', 'nm', s.line_fwhm_nm)
    push(`/sources/${si}/power_drift_pct`, `${s.name} › power drift`, 'Source', '%', s.power_drift_pct)
    s.position_mm.forEach((v, i) => push(`/sources/${si}/position_mm/${i}`, `${s.name} › position ${'xyz'[i]}`, 'Source', 'mm', v))
  })

  exp.cameras.forEach((c, ci) => {
    const raw: { path: string; trail: string[]; value: number }[] = []
    numericLeaves(c.lens, `/cameras/${ci}/lens`, ['lens'], raw)
    numericLeaves(c.sensor, `/cameras/${ci}/sensor`, ['sensor'], raw)
    numericLeaves(c.filter, `/cameras/${ci}/filter`, ['filter'], raw)
    for (const r of raw) { const l = leaf(r.trail[r.trail.length - 1], [c.name, ...r.trail.slice(0, -1)]); push(r.path, l.label, 'Camera', l.unit, r.value) }
  })

  const acq: { path: string; trail: string[]; value: number }[] = []
  numericLeaves({ mc_photons: exp.acquisition.mc_photons, preview_scale: exp.acquisition.preview_scale, volume_mc: exp.acquisition.volume_mc }, '/acquisition', [], acq)
  for (const r of acq) { const l = leaf(r.trail[r.trail.length - 1], r.trail.slice(0, -1).map((t) => (t === 'volume_mc' ? 'volume MC' : t))); push(r.path, l.label, 'Acquisition', l.unit, r.value) }

  if (exp.background.kind === 'plane') push('/background/thickness_mm', 'Background › thickness', 'Background', 'mm', exp.background.thickness_mm)
  push('/background/room_light/illuminance_lux', 'Room light › illuminance', 'Background', 'lux', exp.background.room_light.illuminance_lux)

  return opts
}

export const GROUP_ORDER = ['Layer thickness', 'Material dyes', 'Material scattering', 'Source', 'Camera', 'Objects', 'Acquisition', 'Background']

const seg = (p: string) => p.split('/').slice(1).map((s) => (/^\d+$/.test(s) ? Number(s) : s))

export function getByPointer(root: unknown, path: string): unknown {
  let cur: unknown = root
  for (const k of seg(path)) { if (cur == null) return undefined; cur = (cur as Record<string | number, unknown>)[k] }
  return cur
}

/** Immutable-style set: returns a deep copy of `root` with the pointed value replaced. */
export function setByPointer<T>(root: T, path: string, value: unknown): T {
  const copy = structuredClone(root) as unknown
  const keys = seg(path)
  let cur = copy as Record<string | number, unknown>
  for (let i = 0; i < keys.length - 1; i++) {
    const nxt = cur[keys[i]]
    if (nxt == null || typeof nxt !== 'object') throw new Error(`path ${path} does not exist in the experiment`)
    cur = nxt as Record<string | number, unknown>
  }
  const last = keys[keys.length - 1]
  if (!(last in cur)) throw new Error(`path ${path} does not exist in the experiment`)
  cur[last] = value
  return copy as T
}
