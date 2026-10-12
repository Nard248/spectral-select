/** Analysis workspace state (ROIs, tools, normalisation, per-view settings). Independent of the main experiment store. */
import { create } from 'zustand'
import type { MetricSpec, NormSpec } from '../api/types'
import { KIND_LABEL, ROI_COLORS } from './roi'
import type { AnyRoi, NamedRoi, Stat } from './types'
import type { CmapName } from '../ui/colormaps'

export type Tool = 'pointer' | 'point' | 'disk' | 'rect' | 'ellipse' | 'polygon' | 'line'
export type AnalysisTab = 'pixel' | 'eem' | 'line' | 'response'
export type MidView = 'image' | 'lightfield' | 'sweep'
export type EemMode = 'single' | 'overlay' | 'diff' | 'ratio'

export interface EemSettings {
  mode: EemMode
  log: boolean
  contours: boolean
  cmap: CmapName
  /** diff / ratio: ROI A − ROI B (or A / B). `captureB` compares the same ROI across two captures. */
  roiB: string | null
  captureB: string | null
  sliceEx: number | null
  sliceEm: number | null
  showStd: boolean
  showComponents: boolean
}
export interface LineSettings { exNm: number | null; bandsNm: number[]; gt: string | null; mmAxis: boolean }
export interface ResponseSettings { gt: string | null; metric: MetricSpec | null; roiId: string | null; bins: number; scatter: boolean }

export const DEFAULT_NORM: NormSpec = { mode: 'adu', dark_capture_id: null, white_capture_id: null }

let seq = 0
export const newRoiId = () => `roi_${Date.now().toString(36)}_${(++seq).toString(36)}`

interface AnalysisState {
  tool: Tool
  tab: AnalysisTab
  mid: MidView
  wide: boolean
  rois: NamedRoi[]
  selectedId: string | null
  norm: NormSpec
  stat: Stat
  eem: EemSettings
  line: LineSettings
  response: ResponseSettings
  setTool: (t: Tool) => void
  setTab: (t: AnalysisTab) => void
  setMid: (m: MidView) => void
  setWide: (w: boolean) => void
  addRoi: (spec: AnyRoi, name?: string) => string
  updateRoi: (id: string, patch: Partial<Omit<NamedRoi, 'id'>>) => void
  setSpec: (id: string, spec: AnyRoi) => void
  removeRoi: (id: string) => void
  clearRois: () => void
  select: (id: string | null) => void
  setNorm: (n: Partial<NormSpec>) => void
  setStat: (s: Stat) => void
  setEem: (p: Partial<EemSettings>) => void
  setLine: (p: Partial<LineSettings>) => void
  setResponse: (p: Partial<ResponseSettings>) => void
}

export const useAnalysis = create<AnalysisState>()((set, get) => ({
  tool: 'pointer', tab: 'pixel', mid: 'image', wide: false,
  rois: [], selectedId: null,
  norm: DEFAULT_NORM, stat: 'mean',
  eem: { mode: 'single', log: true, contours: true, cmap: 'magma', roiB: null, captureB: null, sliceEx: null, sliceEm: null, showStd: true, showComponents: false },
  line: { exNm: null, bandsNm: [], gt: null, mmAxis: true },
  response: { gt: null, metric: null, roiId: null, bins: 24, scatter: true },
  setTool: (tool) => set({ tool }),
  setTab: (tab) => set({ tab }),
  setMid: (mid) => set({ mid }),
  setWide: (wide) => set({ wide }),
  addRoi: (spec, name) => {
    const { rois } = get()
    const id = newRoiId()
    const count = rois.filter((r) => r.spec.shape === spec.shape).length + 1
    const used = new Set(rois.map((r) => r.color))
    const color = ROI_COLORS.find((c) => !used.has(c)) ?? ROI_COLORS[rois.length % ROI_COLORS.length]
    set({ rois: [...rois, { id, name: name ?? `${KIND_LABEL[spec.shape]} ${count}`, color, visible: true, spec }], selectedId: id })
    return id
  },
  updateRoi: (id, patch) => set((s) => ({ rois: s.rois.map((r) => (r.id === id ? { ...r, ...patch } : r)) })),
  setSpec: (id, spec) => set((s) => ({ rois: s.rois.map((r) => (r.id === id ? { ...r, spec } : r)) })),
  removeRoi: (id) => set((s) => ({
    rois: s.rois.filter((r) => r.id !== id),
    selectedId: s.selectedId === id ? null : s.selectedId,
    eem: s.eem.roiB === id ? { ...s.eem, roiB: null } : s.eem,
    response: s.response.roiId === id ? { ...s.response, roiId: null } : s.response,
  })),
  clearRois: () => set({ rois: [], selectedId: null }),
  select: (selectedId) => set({ selectedId }),
  setNorm: (n) => set((s) => ({ norm: { ...s.norm, ...n } })),
  setStat: (stat) => set({ stat }),
  setEem: (p) => set((s) => ({ eem: { ...s.eem, ...p } })),
  setLine: (p) => set((s) => ({ line: { ...s.line, ...p } })),
  setResponse: (p) => set((s) => ({ response: { ...s.response, ...p } })),
}))

export const selectedRoi = (s: Pick<AnalysisState, 'rois' | 'selectedId'>): NamedRoi | null => s.rois.find((r) => r.id === s.selectedId) ?? null

if (import.meta.env.DEV && typeof window !== 'undefined') (window as unknown as { analysis: typeof useAnalysis }).analysis = useAnalysis
