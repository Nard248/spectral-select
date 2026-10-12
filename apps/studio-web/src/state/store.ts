import { create } from 'zustand'
import { produce } from 'immer'
import type { Experiment } from '../generated/experiment'
import type { CaptureMeta, PhysicsWarning, ValidationIssue } from '../api/types'
import { DEFAULT_PRESET, presetExperiment, clone, BUNDLE } from './presets'
import { uniqueId } from '../lib/ids'
import { LIBRARY_SHAPES, newCamera, newMaterial, newSource } from './library'

export type SelKind = 'object' | 'source' | 'camera' | 'background' | 'acquisition' | 'material' | 'experiment'
export interface Selection { kind: SelKind; id: string }
export type Mode = 'design' | 'acquire' | 'explore'
export type Gizmo = 'translate' | 'rotate'

export interface JobState {
  jobId: string | null
  kind: 'preview' | 'capture' | 'dark' | 'white_reference'
  fraction: number
  message: string
  /** wavelength currently exposing (drives beam colour), null if not yet known */
  activeNm: number | null
  /** readable stage label derived from the engine message */
  label?: string
  startedAt?: number
  state: 'running' | 'done' | 'error' | 'cancelled'
}

export interface UiState {
  leftOpen: boolean
  rightOpen: boolean
  gizmo: Gizmo
  snap: boolean
  lookThrough: boolean
  pip: boolean
  showFrustum: boolean
  showBeam: boolean
  hoverId: string | null
  quality: 'high' | 'low'
}

export interface StudioState {
  exp: Experiment
  past: Experiment[]
  future: Experiment[]
  dirty: boolean
  selection: Selection | null
  mode: Mode
  ui: UiState
  engine: 'unknown' | 'online' | 'mock'
  captures: CaptureMeta[]
  activeCaptureId: string | null
  compareCaptureId: string | null
  previewMeta: CaptureMeta | null
  /** the experiment the engine preview was rendered from (to flag a stale preview) */
  previewExp: Experiment | null
  previewing: boolean
  /** one-shot request for the Explorer to select a pixel (used by the guided tour) */
  pickRequest: { x: number; y: number; exNm?: number; band?: number } | null
  job: JobState | null
  errors: ValidationIssue[]
  warnings: PhysicsWarning[]
  /** wavelength shown in the viewport beam / PiP preview */
  previewNm: number | null

  /** Apply an immer recipe to the experiment as one undo step (coalesced per `coalesceKey` within 700 ms). */
  edit: (recipe: (e: Experiment) => void, coalesceKey?: string) => void
  setExperiment: (e: Experiment, opts?: { keepHistory?: boolean }) => void
  loadPreset: (key: string) => void
  undo: () => void
  redo: () => void
  select: (s: Selection | null) => void
  setMode: (m: Mode) => void
  setUi: (p: Partial<UiState>) => void
  addObject: (shapeKey: string, at?: [number, number, number]) => string
  addMaterial: (m?: Partial<Experiment['materials'][number]>) => string
  addSource: (kind?: Experiment['sources'][number]['kind']) => string
  addCamera: () => string
  deleteSelected: () => void
  duplicateSelected: () => void
  toggleVisible: (id: string) => void
  setCaptures: (c: CaptureMeta[]) => void
  upsertCapture: (c: CaptureMeta) => void
  setValidation: (e: ValidationIssue[], w: PhysicsWarning[]) => void
}

const HISTORY_LIMIT = 100
let lastKey: string | null = null
let lastTime = 0

const initial = (): Experiment => presetExperiment(DEFAULT_PRESET)

export const useStudio = create<StudioState>()((set, get) => ({
  exp: initial(),
  past: [], future: [], dirty: false,
  selection: null,
  mode: 'design',
  ui: { leftOpen: true, rightOpen: true, gizmo: 'translate', snap: false, lookThrough: false, pip: true, showFrustum: true, showBeam: true, hoverId: null, quality: 'high' },
  engine: 'unknown',
  captures: [], activeCaptureId: null, compareCaptureId: null,
  previewMeta: null, previewExp: null, previewing: false, pickRequest: null, job: null,
  errors: [], warnings: [], previewNm: null,

  edit: (recipe, coalesceKey) => {
    const { exp, past } = get()
    const next = produce(exp, recipe)
    if (next === exp) return
    const now = Date.now()
    const coalesce = coalesceKey != null && coalesceKey === lastKey && now - lastTime < 700
    lastKey = coalesceKey ?? null
    lastTime = now
    set({
      exp: next,
      past: coalesce ? past : [...past.slice(-(HISTORY_LIMIT - 1)), exp],
      future: [], dirty: true,
    })
  },

  setExperiment: (e, opts) => {
    lastKey = null
    set((s) => ({ exp: e, past: opts?.keepHistory ? [...s.past, s.exp] : [], future: [], selection: null, dirty: false }))
  },

  loadPreset: (key) => {
    lastKey = null
    set({ exp: presetExperiment(key), past: [], future: [], selection: null, dirty: false, previewMeta: null, previewExp: null })
  },

  undo: () => {
    const { past, future, exp } = get()
    if (!past.length) return
    lastKey = null
    set({ exp: past[past.length - 1], past: past.slice(0, -1), future: [exp, ...future], dirty: true })
    fixSelection()
  },
  redo: () => {
    const { past, future, exp } = get()
    if (!future.length) return
    lastKey = null
    set({ exp: future[0], future: future.slice(1), past: [...past, exp], dirty: true })
    fixSelection()
  },

  select: (selection) => set({ selection }),
  setMode: (mode) => set({ mode }),
  setUi: (p) => set((s) => ({ ui: { ...s.ui, ...p } })),

  addObject: (shapeKey, at) => {
    const lib = LIBRARY_SHAPES.find((l) => l.key === shapeKey) ?? LIBRARY_SHAPES[0]
    const { exp } = get()
    const id = uniqueId(shapeKey, exp.objects.map((o) => o.id))
    const matId = exp.materials.find((m) => m.dyes.length)?.id ?? exp.materials[0]?.id ?? 'water'
    const glass = exp.materials.find((m) => m.id === 'glass')?.id ?? null
    get().edit((e) => {
      for (const rid of lib.requires ?? []) {   // layered-block items bring their library materials along
        const m = BUNDLE.materials.find((x) => x.id === rid)
        if (m && !e.materials.some((x) => x.id === rid)) e.materials.push(clone(m))
      }
      const o = lib.make(id, matId, glass)
      if (at) { o.transform.position_mm[0] = at[0]; o.transform.position_mm[2] = at[2] }
      else { const k = e.objects.length; o.transform.position_mm[0] += ((k % 5) - 2) * 30; o.transform.position_mm[2] += 60 + Math.floor(k / 5) * 40 }
      e.objects.push(o)
    })
    set({ selection: { kind: 'object', id } })
    return id
  },

  addMaterial: (partial) => {
    const { exp } = get()
    const id = uniqueId(partial?.id ?? 'material', exp.materials.map((m) => m.id))
    get().edit((e) => { e.materials.push({ ...newMaterial(id, partial?.name ?? 'New material'), ...clone(partial ?? {}), id }) })
    return id
  },

  addSource: (kind) => {
    const { exp } = get()
    const id = uniqueId('src', exp.sources.map((s) => s.id))
    get().edit((e) => { e.sources.push(newSource(id, kind)); if (!e.active_source_id) e.active_source_id = id })
    set({ selection: { kind: 'source', id } })
    return id
  },

  addCamera: () => {
    const { exp } = get()
    const id = uniqueId('cam', exp.cameras.map((c) => c.id))
    get().edit((e) => {
      const base = e.cameras[0]
      if (!base) return
      const c = newCamera(id, base); c.position_mm = [60, 280, 120]; e.cameras.push(c)
    })
    set({ selection: { kind: 'camera', id } })
    return id
  },

  deleteSelected: () => {
    const sel = get().selection
    if (!sel) return
    if (sel.kind === 'object') get().edit((e) => { e.objects = e.objects.filter((o) => o.id !== sel.id) })
    else if (sel.kind === 'source') get().edit((e) => {
      if (e.sources.length <= 1) return
      e.sources = e.sources.filter((s) => s.id !== sel.id)
      if (e.active_source_id === sel.id) e.active_source_id = e.sources[0].id
    })
    else if (sel.kind === 'camera') get().edit((e) => {
      if (e.cameras.length <= 1) return
      e.cameras = e.cameras.filter((c) => c.id !== sel.id)
      if (e.active_camera_id === sel.id) e.active_camera_id = e.cameras[0].id
    })
    else if (sel.kind === 'material') get().edit((e) => {
      if (e.objects.some((o) => o.material_id === sel.id || o.vessel_material_id === sel.id) || e.background.material_id === sel.id) return
      e.materials = e.materials.filter((m) => m.id !== sel.id)
    })
    fixSelection()
  },

  duplicateSelected: () => {
    const sel = get().selection
    if (!sel) return
    const { exp } = get()
    if (sel.kind === 'object') {
      const src = exp.objects.find((o) => o.id === sel.id); if (!src) return
      const id = uniqueId(sel.id, exp.objects.map((o) => o.id))
      get().edit((e) => {
        const c = clone(src); c.id = id; c.name = `${src.name} copy`
        c.transform.position_mm = [src.transform.position_mm[0] + 25, src.transform.position_mm[1], src.transform.position_mm[2] + 25]
        e.objects.push(c)
      })
      set({ selection: { kind: 'object', id } })
    } else if (sel.kind === 'material') {
      const src = exp.materials.find((m) => m.id === sel.id); if (!src) return
      const id = uniqueId(sel.id, exp.materials.map((m) => m.id))
      get().edit((e) => { const c = clone(src); c.id = id; c.name = `${src.name} copy`; e.materials.push(c) })
      set({ selection: { kind: 'material', id } })
    } else if (sel.kind === 'source') {
      const src = exp.sources.find((s) => s.id === sel.id); if (!src) return
      const id = uniqueId(sel.id, exp.sources.map((s) => s.id))
      get().edit((e) => { const c = clone(src); c.id = id; c.name = `${src.name} copy`; c.position_mm[0] += 40; e.sources.push(c) })
      set({ selection: { kind: 'source', id } })
    } else if (sel.kind === 'camera') {
      const src = exp.cameras.find((c) => c.id === sel.id); if (!src) return
      const id = uniqueId(sel.id, exp.cameras.map((c) => c.id))
      get().edit((e) => { const c = clone(src); c.id = id; c.name = `${src.name} copy`; c.position_mm[0] += 40; e.cameras.push(c) })
      set({ selection: { kind: 'camera', id } })
    }
  },

  toggleVisible: (id) => get().edit((e) => { const o = e.objects.find((x) => x.id === id); if (o) o.visible = !o.visible }),

  setCaptures: (captures) => set({ captures }),
  upsertCapture: (c) => set((s) => ({
    captures: [c, ...s.captures.filter((x) => x.id !== c.id)],
    activeCaptureId: s.activeCaptureId ?? c.id,
  })),
  setValidation: (errors, warnings) => set({ errors, warnings }),
}))

/** Drop a selection that no longer exists (after delete / undo / load). */
function fixSelection() {
  const { selection: s, exp } = useStudio.getState()
  if (!s) return
  const ok =
    s.kind === 'object' ? exp.objects.some((o) => o.id === s.id)
    : s.kind === 'source' ? exp.sources.some((o) => o.id === s.id)
    : s.kind === 'camera' ? exp.cameras.some((o) => o.id === s.id)
    : s.kind === 'material' ? exp.materials.some((o) => o.id === s.id)
    : true
  if (!ok) useStudio.setState({ selection: null })
}

export const activeSource = (e: Experiment) => e.sources.find((s) => s.id === e.active_source_id) ?? e.sources[0]
export const activeCamera = (e: Experiment) => e.cameras.find((c) => c.id === e.active_camera_id) ?? e.cameras[0]
