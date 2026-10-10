import type * as THREE from 'three'
import { useStudio, activeCamera, activeSource } from '../state/store'
import { viewApi } from '../scene/viewApi'
import { runPreview, runCapture } from '../state/actions'
import { getEngine } from '../api/client'
import { cameraIntrinsics } from '../lib/camera'
import { imagingGeometry } from '../three/hardware/CameraFrustum'
import { loadCube } from '../explorer/data'
import { useTour } from './tourStore'
import type { Experiment } from '../generated/experiment'
import type { CaptureMeta } from '../api/types'

type V3 = THREE.Vector3Tuple

export interface TourCtx {
  /** abort check: true once the user left this step */
  stale: () => boolean
  sleep: (ms: number) => Promise<void>
  caption: (text: string) => void
}
export interface TourStep {
  title: string
  /** caption text (may use the live experiment) */
  text: (e: Experiment) => string
  /** runs when the step is entered; resolves when its scripted action has finished */
  enter: (ctx: TourCtx) => Promise<void>
  /** seconds to dwell before auto-advancing */
  dwell: number
}

const KIND_TEXT: Record<string, string> = { tunable_lamp: 'a tunable xenon lamp with monochromator', led_array: 'an LED array', laser: 'a laser' }
const HOME: { pos: V3; target: V3 } = { pos: [800, 470, 1080], target: [-10, 200, 0] }

const v = {
  add: (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
  sub: (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
  mul: (a: V3, k: number): V3 => [a[0] * k, a[1] * k, a[2] * k],
  len: (a: V3) => Math.hypot(a[0], a[1], a[2]),
  norm: (a: V3): V3 => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l] },
  cross: (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]],
}
/** side-on viewpoint for an axis from `p` towards `q`: looks at `look`, from `dist` away, perpendicular to the axis and a little above */
function sideView(p: V3, q: V3, look: V3, dist: number, lift = 0.35, yaw = 0): { pos: V3; target: V3 } {
  const axis = v.norm(v.sub(q, p))
  let side = v.cross(axis, [0, 1, 0])
  side = v.len(side) < 0.15 ? v.norm([0.55, 0, 0.83]) : v.norm(side)   // vertical axis: view from the default front-right side
  const dir = v.norm(v.add(v.add(side, [0, lift, 0]), v.mul(axis, yaw)))
  return { pos: v.add(look, v.mul(dir, dist)), target: look }
}
function samplesCenter(e: Experiment): { c: V3; r: number } {
  const ps = e.objects.filter((o) => o.visible).map((o) => o.transform.position_mm as V3)
  if (!ps.length) return { c: [0, 10, 0], r: 60 }
  const c: V3 = [ps.reduce((a, p) => a + p[0], 0) / ps.length, 8, ps.reduce((a, p) => a + p[2], 0) / ps.length]
  return { c, r: Math.max(35, ...ps.map((p) => Math.hypot(p[0] - c[0], p[2] - c[2]) + 22)) }
}
const fly = (pos: V3, target: V3, ms = 2000) => viewApi.current?.flyTo(pos, target, ms) ?? Promise.resolve()
const st = () => useStudio.getState()

function nearest(e: Experiment, nm: number): number | null {
  const s = activeSource(e).sequence.filter((x) => x.enabled)
  return s.length ? s.reduce((b, x) => (Math.abs(x.wavelength_nm - nm) < Math.abs(b - nm) ? x.wavelength_nm : b), s[0].wavelength_nm) : null
}

/** the original experiment the tour started from, so the physics demo can be undone exactly */
export const tourBackup: { exp: Experiment | null; past: Experiment[]; dirty: boolean } = { exp: null, past: [], dirty: false }

export function resetUiForTour() {
  const s = st()
  s.setMode('design')
  s.setUi({ lookThrough: false, pip: true, showBeam: true, showFrustum: false, leftOpen: true, rightOpen: true })
  s.select(null)
}

const waitFor = async (cond: () => boolean, ctx: TourCtx, timeoutMs: number) => {
  const t0 = Date.now()
  while (!cond() && !ctx.stale() && Date.now() - t0 < timeoutMs) await ctx.sleep(120)
}

export const STEPS: TourStep[] = [
  {
    title: 'The bench',
    dwell: 6,
    text: (e) => `A virtual HSI bench: ${KIND_TEXT[activeSource(e).kind] ?? 'a light source'}, a ${activeCamera(e).name} and samples on an optical breadboard. Everything you see is a physical model that the engine ray-traces — one unit is one millimetre.`,
    enter: async () => {
      resetUiForTour()
      st().setUi({ showBeam: false })
      await fly(HOME.pos, HOME.target, 2200)
    },
  },
  {
    title: 'Camera optics',
    dwell: 7,
    text: (e) => {
      const c = activeCamera(e), g = imagingGeometry(c), i = cameraIntrinsics(c)
      return `Thin-lens model: f = ${c.lens.focal_length_mm} mm, f/${c.lens.f_number}, focused ${c.lens.focus_distance_mm} mm away. The green plane is the focus plane: ${(2 * g.s * g.tanX).toFixed(0)} × ${(2 * g.s * g.tanY).toFixed(0)} mm on the sample, vertical FOV ${i.fovYDeg.toFixed(1)}°. The amber slab is the depth of field (2-pixel blur circle).`
    },
    enter: async () => {
      const e = st().exp, c = activeCamera(e)
      st().setUi({ showFrustum: true, showBeam: false })
      st().select({ kind: 'camera', id: c.id })
      const mid = v.add(c.position_mm as V3, v.mul(v.norm(v.sub(c.target_mm as V3, c.position_mm as V3)), c.lens.focus_distance_mm * 0.5))
      const sv = sideView(c.position_mm as V3, c.target_mm as V3, v.add(mid, [0, 25, 0]), c.lens.focus_distance_mm * 2.2, 0.3)
      await fly(sv.pos, sv.target, 2400)
    },
  },
  {
    title: 'Source and beam',
    dwell: 7,
    text: (e) => {
      const s = activeSource(e), on = s.sequence.filter((x) => x.enabled)
      return `${s.name}: ${on.length} excitation lines from ${on[0]?.wavelength_nm} to ${on.at(-1)?.wavelength_nm} nm, ${s.line_fwhm_nm} nm linewidth. The ${s.beam.profile.replace('_', '-')} beam leaves a ${s.beam.aperture_diameter_mm} mm aperture and diverges ${s.beam.divergence_half_angle_deg}° — the engine integrates exactly this irradiance over every sample point.`
    },
    enter: async () => {
      const e = st().exp, s = activeSource(e)
      useStudio.setState({ previewNm: nearest(e, 365) })
      st().setUi({ showBeam: true, showFrustum: false })
      st().select({ kind: 'source', id: s.id })
      const p = s.position_mm as V3, t = s.target_mm as V3
      const look = v.add(p, v.mul(v.sub(t, p), 0.5))
      const sv = sideView(p, t, look, v.len(v.sub(t, p)) * 2.0, 0.4)
      await fly(sv.pos, sv.target, 2400)
    },
  },
  {
    title: 'Samples and materials',
    dwell: 7,
    text: (e) => {
      const names = e.objects.slice(0, 3).map((o) => e.materials.find((m) => m.id === o.material_id)?.name ?? o.material_id)
      return `${e.objects.length} samples (${names.join(', ')}…). Each material carries fluorophores with their own excitation/emission spectra, quantum yield, scattering and absorption coefficients. Glass vessels, liquids and turbid solids are modelled with their true refractive indices.`
    },
    enter: async () => {
      const e = st().exp
      st().setUi({ showFrustum: false, showBeam: true })
      const first = e.objects.find((o) => o.visible)
      if (first) st().select({ kind: 'object', id: first.id })
      const { c, r } = samplesCenter(e)
      await fly(v.add(c, v.mul(v.norm([0.5, 0.85, 0.75]), r * 3.3 + 80)), c, 2300)
    },
  },
  {
    title: 'Preview',
    dwell: 7,
    text: () => 'Preview renders one excitation at reduced resolution with the full physics engine in about a second. The Camera view on the right switches from the real-time GPU approximation to the engine’s own image of exactly what the HSI camera sees.',
    enter: async (ctx) => {
      const e = st().exp
      st().select(null)
      st().setUi({ pip: true, showBeam: true })
      useStudio.setState({ previewNm: nearest(e, 365) })
      const { c, r } = samplesCenter(e)
      await fly(v.add(c, v.mul(v.norm([0.6, 1.0, 0.8]), r * 3.6 + 120)), c, 1800)
      if (ctx.stale()) return
      await runPreview()
    },
  },
  {
    title: 'Capture',
    dwell: 3,
    text: (e) => `A capture sweeps all ${activeSource(e).sequence.filter((x) => x.enabled).length} excitation lines, each with its own tunable-filter band set, exposure and photobleaching history. The beam colour follows the line being simulated.`,
    enter: async (ctx) => {
      st().setMode('acquire')
      st().setUi({ showBeam: true, showFrustum: true })
      await ctx.sleep(350)
      void fly([700, 400, 940], [-10, 190, 0], 1800)
      const before = st().captures.length
      await runCapture('sample')
      await waitFor(() => st().captures.length > before || (st().job?.state === 'error'), ctx, 180_000)
      await ctx.sleep(900)
    },
  },
  {
    title: 'Explore a pixel',
    dwell: 9,
    text: () => 'Every pixel of the 4D cube (excitation × emission × y × x) can be decomposed: the stacked areas are the noise-free radiance of each physical component — dye fluorescence, re-emission, elastic leak, glints — and the table gives each one’s share of the total.',
    enter: async (ctx) => {
      const id = st().activeCaptureId ?? st().captures[0]?.id
      const meta = st().captures.find((c) => c.id === id)
      if (!id || !meta) { ctx.caption('No capture available — run Capture first.'); return }
      // brightest-but-robust pixel (99.5th percentile of total signal) at an excitation that fluoresces
      const exNm = meta.excitations_nm.reduce((b, x) => (Math.abs(x - 365) < Math.abs(b - 365) ? x : b), meta.excitations_nm[0])
      const cube = await loadCube(id, exNm)
      const N = cube.width * cube.height, B = cube.bands
      const sums = new Float64Array(N)
      for (let k = 0; k < N; k++) { let s = 0; for (let j = 0; j < B; j++) s += cube.data[k * B + j]; sums[k] = s }
      const sorted = Float64Array.from(sums).sort()
      const target = sorted[Math.floor(0.995 * (N - 1))]
      let best = 0, bd = Infinity
      for (let k = 0; k < N; k++) { const d = Math.abs(sums[k] - target); if (d < bd) { bd = d; best = k } }
      let bandBest = 0, bv = -Infinity
      for (let j = 0; j < B; j++) { const x = cube.data[best * B + j]; if (x > bv) { bv = x; bandBest = j } }
      if (ctx.stale()) return
      st().setMode('explore')
      await ctx.sleep(700)
      useStudio.setState({ pickRequest: { x: best % cube.width, y: Math.floor(best / cube.width), exNm, band: bandBest } })
    },
  },
  {
    title: 'Physics toggles',
    dwell: 8,
    text: () => 'Every physical effect can be switched off for teaching or ablation. Watch what happens to the image when light scattering inside the samples is turned off…',
    enter: async (ctx) => {
      const e0 = tourBackup.exp ?? st().exp
      st().setMode('design')
      st().setUi({ pip: true, showBeam: true, showFrustum: false })
      st().select({ kind: 'acquisition', id: 'acquisition' })
      await ctx.sleep(500)
      const { c, r } = samplesCenter(e0)
      const nm = nearest(e0, 365)
      useStudio.setState({ previewNm: nm })
      void fly(v.add(c, v.mul(v.norm([0.6, 1.0, 0.8]), r * 3.6 + 120)), c, 1600)
      const eng = getEngine()
      const shot = async (exp: Experiment, label: string) => {
        const meta = await eng!.preview({ experiment: exp, excitation_nm: nm, scale: 0.5 })
        useStudio.setState({ previewMeta: meta, previewExp: exp })
        return { label, url: eng!.kind === 'real' ? eng!.imageUrl(meta.id, 'true_color', meta.excitations_nm[0], undefined, 1.3) : '' }
      }
      if (!eng) return
      let metaA: CaptureMeta | null = null
      useTour.setState({ thumbs: [] })
      ctx.caption('Rendering the reference with all physics ON…')
      useStudio.setState({ previewing: true })
      try {
        const a = await shot(e0, 'Scattering ON')
        metaA = useStudio.getState().previewMeta
        if (ctx.stale()) return
        // switch scattering off (a real, undoable edit — the original is restored afterwards)
        st().edit((e) => { e.acquisition.physics.scattering = false })
        ctx.caption('Scattering OFF — rendering with the engine…')
        const b = await shot(st().exp, 'Scattering OFF')
        if (ctx.stale()) return
        useTour.setState({ thumbs: [a, b].filter((t) => t.url) })
        ctx.caption('Scattering OFF: the turbid lichens lose their diffuse glow, the depth-dependent colour shift and the soft edges — only surface reflection and direct fluorescence remain. Every toggle in the Acquisition panel works like this.')
        await ctx.sleep(6000)
      } finally {
        useStudio.setState({ previewing: false })
        // restore the original experiment exactly (also drops the tour's undo entry)
        if (tourBackup.exp) useStudio.setState({ exp: tourBackup.exp, past: tourBackup.past, future: [], dirty: tourBackup.dirty })
      }
      if (ctx.stale()) return
      if (metaA) useStudio.setState({ previewMeta: metaA, previewExp: tourBackup.exp })
      else await runPreview()
      ctx.caption(`Scattering is back ON${useTour.getState().thumbs.length ? ' (left: physical result, right: scattering off)' : ''}. The turbid lichens lose their diffuse glow and colour shift without it. Every toggle in the Acquisition panel works like this — try one, then Preview.`)
    },
  },
]

export const TOTAL = STEPS.length
export { useTour }
