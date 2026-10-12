/** Data hooks for the analysis endpoints: debounced, cancellable, cached by request key. */
import { useEffect, useMemo, useState } from 'react'
import { getEngine } from '../api/client'
import type { EemRequest, EemResult, LineProfile, LineRequest, NormSpec, ResponseCurve, ResponseRequest, RoiRequest, RoiStats, VolumeInfo } from '../api/types'
import type { Experiment } from '../generated/experiment'
import { normSpecClean } from './norm'
import type { Stat } from './types'

const cache = new Map<string, unknown>()
const CAP = 120
const remember = (k: string, v: unknown) => { cache.set(k, v); if (cache.size > CAP) cache.delete(cache.keys().next().value as string) }

export interface Remote<T> { data: T | null; loading: boolean; error: string | null }

/** Run `fn` when `key` changes (after `delay` ms), keep the previous data while loading, cache by key. */
export function useRemote<T>(key: string | null, fn: () => Promise<T>, delay = 140): Remote<T> {
  const [st, setSt] = useState<Remote<T>>({ data: null, loading: false, error: null })
  useEffect(() => {
    if (key == null) { setSt({ data: null, loading: false, error: null }); return }
    if (cache.has(key)) { setSt({ data: cache.get(key) as T, loading: false, error: null }); return }
    let dead = false
    setSt((s) => ({ data: s.data, loading: true, error: null }))
    const t = setTimeout(() => {
      fn().then((d) => { remember(key, d); if (!dead) setSt({ data: d, loading: false, error: null }) })
        .catch((e: Error) => { if (!dead) setSt({ data: null, loading: false, error: e.message }) })
    }, delay)
    return () => { dead = true; clearTimeout(t) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key])
  return st
}

const eng = () => { const e = getEngine(); if (!e) throw new Error('engine not ready'); return e }
const j = (o: unknown) => JSON.stringify(o)

export function useEem(captureId: string | null, roi: EemRequest['roi'] | null, norm: NormSpec, stat: Stat, enabled = true) {
  const req = useMemo<EemRequest | null>(() => (roi && enabled ? { roi, norm: normSpecClean(norm), stat } : null), [roi, norm, stat, enabled])
  return useRemote<EemResult>(captureId && req ? `eem|${captureId}|${j(req)}` : null, () => eng().eem(captureId!, req!))
}
export function useRoiStats(captureId: string | null, roi: RoiRequest['roi'] | null, norm: NormSpec, stat: Stat, components: boolean) {
  const req = useMemo<RoiRequest | null>(() => (roi ? { roi, norm: normSpecClean(norm), stat, components } : null), [roi, norm, stat, components])
  return useRemote<RoiStats>(captureId && req ? `roi|${captureId}|${j(req)}` : null, () => eng().roi(captureId!, req!))
}
export function useLine(captureId: string | null, req: LineRequest | null) {
  return useRemote<LineProfile>(captureId && req ? `line|${captureId}|${j(req)}` : null, () => eng().line(captureId!, req!))
}
export function useResponse(captureId: string | null, req: ResponseRequest | null) {
  return useRemote<ResponseCurve>(captureId && req ? `resp|${captureId}|${j(req)}` : null, () => eng().response(captureId!, req!), 220)
}

export function useVolumeInfos(captureId: string | null, hint?: VolumeInfo[]) {
  const r = useRemote<VolumeInfo[]>(captureId && !hint ? `vols|${captureId}` : null, () => eng().volumes(captureId!), 0)
  return hint ? { data: hint, loading: false, error: null } as Remote<VolumeInfo[]> : r
}

/** The experiment a capture came from (physical layer geometry). Falls back to null when the engine can't say. */
export function useCaptureExperiment(captureId: string | null) {
  return useRemote<Experiment | null>(captureId ? `exp|${captureId}` : null, () => eng().captureExperiment(captureId!), 0)
}

export function clearAnalysisCache() { cache.clear() }

export interface EemItem { key: string; captureId: string; roi: EemRequest['roi'] }
/** Several EEM requests at once (ROI overlay / A-B comparison). Results are keyed by `item.key`. */
export function useEemMulti(items: EemItem[], norm: NormSpec, stat: Stat) {
  const n = normSpecClean(norm)
  const sig = items.map((it) => `eem|${it.captureId}|${j({ roi: it.roi, norm: n, stat })}`).join('§')
  const [st, setSt] = useState<{ data: Record<string, EemResult>; loading: boolean; error: string | null }>({ data: {}, loading: false, error: null })
  useEffect(() => {
    if (!items.length) { setSt({ data: {}, loading: false, error: null }); return }
    let dead = false
    const keyed = items.map((it, i) => ({ it, ck: sig.split('§')[i] }))
    const have: Record<string, EemResult> = {}
    keyed.forEach(({ it, ck }) => { if (cache.has(ck)) have[it.key] = cache.get(ck) as EemResult })
    if (Object.keys(have).length === items.length) { setSt({ data: have, loading: false, error: null }); return }
    setSt((s) => ({ data: s.data, loading: true, error: null }))
    const t = setTimeout(() => {
      Promise.all(keyed.map(async ({ it, ck }) => {
        if (cache.has(ck)) return [it.key, cache.get(ck) as EemResult] as const
        const r = await eng().eem(it.captureId, { roi: it.roi, norm: n, stat })
        remember(ck, r)
        return [it.key, r] as const
      })).then((pairs) => { if (!dead) setSt({ data: Object.fromEntries(pairs), loading: false, error: null }) })
        .catch((e: Error) => { if (!dead) setSt({ data: {}, loading: false, error: e.message }) })
    }, 140)
    return () => { dead = true; clearTimeout(t) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sig])
  return st
}

const volCache = new Map<string, Promise<import('../api/client').VolumeData>>()
/** One 3D volume (float32 nx·nz·nd). Small LRU because a volume can be several MB. */
export function useVolume(captureId: string | null, q: import('../api/client').VolumeQuery | null) {
  const key = captureId && q ? `${captureId}|${q.object}|${q.kind}|${q.ex ?? ''}|${q.channel ?? ''}` : null
  const [st, setSt] = useState<Remote<import('../api/client').VolumeData>>({ data: null, loading: false, error: null })
  useEffect(() => {
    if (!key || !captureId || !q) { setSt({ data: null, loading: false, error: null }); return }
    let dead = false
    let p = volCache.get(key)
    if (!p) {
      p = eng().volume(captureId, q); volCache.set(key, p)
      p.catch(() => volCache.delete(key))
      if (volCache.size > 10) volCache.delete(volCache.keys().next().value as string)
    }
    setSt((s) => ({ data: s.data, loading: true, error: null }))
    p.then((d) => { if (!dead) setSt({ data: d, loading: false, error: null }) }).catch((e: Error) => { if (!dead) setSt({ data: null, loading: false, error: e.message }) })
    return () => { dead = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key])
  return st
}
