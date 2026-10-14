/** Engine-backed optical properties of a material (`POST /api/material_optics`), debounced; `null` while loading or offline
 *  (the inspector then falls back to the client-side tables of `lib/tissueSpectra.ts`). */
import { useEffect, useMemo, useRef, useState } from 'react'
import type { FluorophoreSpec, MaterialSpec } from '../generated/experiment'
import type { MaterialOptics } from '../api/researchTypes'
import { loadResearchApi } from './api'
import { getEngine } from '../api/client'

export interface OpticsState { data: MaterialOptics | null; source: 'engine' | 'client'; loading: boolean; error: string | null }

/** Linear interpolation of a tabulated curve (clamped at the ends). */
export function interp(nm: number[], y: number[], x: number): number {
  if (!nm.length) return NaN
  if (x <= nm[0]) return y[0]
  if (x >= nm[nm.length - 1]) return y[y.length - 1]
  let lo = 0, hi = nm.length - 1
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (nm[mid] <= x) lo = mid; else hi = mid }
  const t = (x - nm[lo]) / (nm[hi] - nm[lo])
  return y[lo] * (1 - t) + y[hi] * t
}

export function useMaterialOptics(mat: MaterialSpec | undefined, customDyes: FluorophoreSpec[], nmMin: number, nmMax: number, step: number, delayMs = 260): OpticsState {
  const [state, setState] = useState<OpticsState>({ data: null, source: 'client', loading: false, error: null })
  const seq = useRef(0)
  const used = useMemo(() => new Set(mat?.dyes.map((d) => d.fluorophore) ?? []), [mat])
  const extra = useMemo(() => customDyes.filter((f) => used.has(f.name)), [customDyes, used])
  const key = useMemo(() => JSON.stringify([mat, extra, nmMin, nmMax, step]), [mat, extra, nmMin, nmMax, step])
  useEffect(() => {
    if (!mat) return
    const my = ++seq.current
    const t = setTimeout(async () => {
      try {
        if (getEngine()?.kind !== 'real') { setState({ data: null, source: 'client', loading: false, error: null }); return }
        setState((s) => ({ ...s, loading: true }))
        const api = await loadResearchApi()
        if (api.kind !== 'real') { if (my === seq.current) setState({ data: null, source: 'client', loading: false, error: null }); return }
        const data = await api.materialOptics({ material: mat, fluorophores: extra, nm_min: nmMin, nm_max: nmMax, step })
        if (my === seq.current) setState({ data, source: 'engine', loading: false, error: null })
      } catch (e) { if (my === seq.current) setState({ data: null, source: 'client', loading: false, error: (e as Error).message.slice(0, 160) }) }
    }, delayMs)
    return () => clearTimeout(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key])
  return state
}
