/** View-only state of layered blocks (cutaway + light field). Deliberately separate from the experiment store: nothing
 *  here is part of the saved experiment or the undo history. */
import { create } from 'zustand'
import type { VolumeInfo } from '../api/types'

export interface SectionState {
  on: boolean
  axis: 'x' | 'z'
  /** local coordinate of the cut plane along `axis` (mm) */
  position_mm: number
  keep: 'low' | 'high'
  labels: boolean
}
export type LightKind = 'fluence' | 'absorbed' | 'source'
export interface LightFieldState {
  on: boolean
  kind: LightKind
  /** excitation (nm) shown; null = first available */
  ex: number | null
  channel: string | null
  /** dynamic range (decades below the maximum) */
  decades: number
  cmap: 'inferno' | 'magma' | 'turbo' | 'viridis'
}
export const DEFAULT_SECTION: SectionState = { on: false, axis: 'z', position_mm: 0, keep: 'low', labels: true }

interface LayerViewState {
  sections: Record<string, SectionState>
  /** inclusion highlighted in the 3D view, per block object id (view-only) */
  incSel: Record<string, number | null>
  setIncSel: (id: string, i: number | null) => void
  lf: LightFieldState
  /** a status line for the light-field loader (shown in the inspector) */
  lfStatus: { state: 'idle' | 'loading' | 'ready' | 'none' | 'error'; message?: string }
  /** colour-scale legend of the displayed light field (set by the loader) */
  lfLegend: { vmax: number; vmin: number; unit: string; label: string; cmap: string; demo?: boolean } | null
  setLfLegend: (l: LayerViewState['lfLegend']) => void
  /** volumes available for the displayed block in the active capture (set by the loader) */
  lfInfos: VolumeInfo[]
  setLfInfos: (v: VolumeInfo[]) => void
  section: (id: string) => SectionState
  setSection: (id: string, patch: Partial<SectionState>) => void
  setLf: (patch: Partial<LightFieldState>) => void
  setLfStatus: (s: LayerViewState['lfStatus']) => void
}

export const useLayerView = create<LayerViewState>()((set, get) => ({
  sections: {},
  incSel: {},
  setIncSel: (id, i) => set((s) => (s.incSel[id] === i ? s : { incSel: { ...s.incSel, [id]: i } })),
  lf: { on: false, kind: 'fluence', ex: null, channel: null, decades: 4, cmap: 'inferno' },
  lfStatus: { state: 'idle' },
  lfLegend: null,
  lfInfos: [],
  setLfInfos: (lfInfos) => set({ lfInfos }),
  setLfLegend: (lfLegend) => set({ lfLegend }),
  section: (id) => get().sections[id] ?? DEFAULT_SECTION,
  setSection: (id, patch) => set((s) => ({ sections: { ...s.sections, [id]: { ...(s.sections[id] ?? DEFAULT_SECTION), ...patch } } })),
  setLf: (patch) => set((s) => ({ lf: { ...s.lf, ...patch } })),
  setLfStatus: (lfStatus) => set({ lfStatus }),
}))

if (import.meta.env.DEV && typeof window !== 'undefined') (window as unknown as { layerView: typeof useLayerView }).layerView = useLayerView

/** The block the cutaway / light-field controls act on: the selected block, else the first visible one. */
export function targetBlockId(exp: { objects: { id: string; visible: boolean; shape: { kind: string } }[] }, selection: { kind: string; id: string } | null): string | null {
  if (selection?.kind === 'object') {
    const o = exp.objects.find((x) => x.id === selection.id)
    if (o?.shape.kind === 'layered_block') return o.id
  }
  return exp.objects.find((o) => o.visible && o.shape.kind === 'layered_block')?.id ?? null
}
