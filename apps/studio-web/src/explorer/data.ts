import { useEffect, useState } from 'react'
import { getEngine, type Cube, type Raster } from '../api/client'
import type { PixelExplain } from '../api/types'

const cubes = new Map<string, Promise<Cube>>()
const gts = new Map<string, Promise<Raster>>()

export function loadCube(id: string, ex: number): Promise<Cube> {
  const key = `${id}|${ex}`
  let p = cubes.get(key)
  if (!p) {
    const eng = getEngine(); if (!eng) return Promise.reject(new Error('engine not ready'))
    p = eng.cube(id, ex, 'adu'); cubes.set(key, p)
    p.catch(() => cubes.delete(key))
    if (cubes.size > 8) cubes.delete(cubes.keys().next().value as string)
  }
  return p
}
export function loadGt(id: string, name: string): Promise<Raster> {
  const key = `${id}|${name}`
  let p = gts.get(key)
  if (!p) { const eng = getEngine(); if (!eng) return Promise.reject(new Error('engine not ready')); p = eng.groundTruth(id, name); gts.set(key, p); p.catch(() => gts.delete(key)) }
  return p
}

function useAsync<T>(make: (() => Promise<T>) | null, deps: unknown[]) {
  const [state, setState] = useState<{ data: T | null; loading: boolean; error: string | null }>({ data: null, loading: !!make, error: null })
  useEffect(() => {
    if (!make) { setState({ data: null, loading: false, error: null }); return }
    let dead = false
    setState((s) => ({ data: s.data, loading: true, error: null }))
    make().then((d) => { if (!dead) setState({ data: d, loading: false, error: null }) })
      .catch((e: Error) => { if (!dead) setState({ data: null, loading: false, error: e.message }) })
    return () => { dead = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)
  return state
}
export const useCube = (id: string | null, ex: number | null) => useAsync(id != null && ex != null ? () => loadCube(id, ex) : null, [id, ex])
export const useGt = (id: string | null, name: string | null) => useAsync(id && name ? () => loadGt(id, name) : null, [id, name])
export const usePixel = (id: string | null, x: number | null, y: number | null) =>
  useAsync<PixelExplain>(id != null && x != null && y != null ? () => {
    const eng = getEngine(); if (!eng) return Promise.reject(new Error('engine not ready'))
    return eng.pixel(id, x, y)
  } : null, [id, x, y])
