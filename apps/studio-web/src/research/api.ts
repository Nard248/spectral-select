/** Client of the v1.2 research endpoints (studies, sensitivity, methods text, provenance, inverse problems, material optics).
 *  `researchApi()` returns the real HTTP implementation when the engine is online and the in-browser mock otherwise. */
import { ApiError, getEngine, json, parseLooseJson, post, nanify } from '../api/client'
import type { Raster } from '../api/client'
import type { JobEvent } from '../api/types'
import type {
  CalibratedFluorophore, DesignRequest, DesignResult, FisherRequest, FisherResult, FitMapResult, FitRequest, FitResult, MaterialOptics,
  MaterialOpticsRequest, MethodsRequest, MethodsText, Provenance, SensitivityResult, StudyEvent, StudyRuns, StudySpec, StudyStart,
  StudyStatus, SurrogateRequest, SurrogateSummary, UnmixRequest, UnmixResult,
} from '../api/researchTypes'

export interface ResearchApi {
  kind: 'real' | 'mock'
  startStudy(spec: StudySpec, keepCaptures: boolean): Promise<StudyStart>
  listStudies(): Promise<StudyStatus[]>
  studyStatus(id: string): Promise<StudyStatus>
  watchStudy(id: string, onEvent: (e: StudyEvent) => void): () => void
  cancelStudy(id: string): Promise<StudyStatus>
  studyRuns(id: string, offset: number, limit: number): Promise<StudyRuns>
  sensitivity(id: string, boot: number): Promise<SensitivityResult>
  methodsText(req: MethodsRequest): Promise<MethodsText>
  provenance(): Promise<Provenance>
  materialOptics(req: MaterialOpticsRequest): Promise<MaterialOptics>
  calibratedFluorophores(): Promise<CalibratedFluorophore[]>
  // inverse
  unmix(req: UnmixRequest): Promise<UnmixResult>
  unmixMap(unmixId: string, name: string): Promise<Raster>
  createSurrogate(req: SurrogateRequest): Promise<{ job_id: string; n_simulator_runs: number }>
  surrogate(id: string): Promise<SurrogateSummary>
  fit(req: FitRequest): Promise<FitResult>
  fitMap(req: FitRequest): Promise<FitMapResult>
  fisher(req: FisherRequest): Promise<FisherResult | { job_id: string }>
  design(req: DesignRequest): Promise<DesignResult | { job_id: string }>
  inverseResult<T>(id: string): Promise<T>
  /** Follow a job; the `done` event's `capture_id` is the inverse result id. */
  watchJob(jobId: string, onEvent: (e: JobEvent) => void): () => void
}

export const isJob = (x: unknown): x is { job_id: string } => !!x && typeof x === 'object' && 'job_id' in (x as object)

const enc = encodeURIComponent

export const realResearch: ResearchApi = {
  kind: 'real',
  startStudy: (spec, keep) => post<StudyStart>(`/api/studies${keep ? '?keep_captures=1' : ''}`, spec),
  listStudies: () => json<StudyStatus[]>('/api/studies'),
  studyStatus: (id) => json<StudyStatus>(`/api/studies/${enc(id)}`),
  watchStudy(id, onEvent) {
    let closed = false
    let poll: ReturnType<typeof setInterval> | null = null
    const startPolling = () => {
      if (poll || closed) return
      poll = setInterval(async () => {
        try {
          const s = await json<StudyStatus>(`/api/studies/${enc(id)}`)
          if (s.state === 'running' || s.state === 'queued') onEvent({ type: 'progress', fraction: s.fraction, message: s.message })
          else {
            if (poll) clearInterval(poll)
            closed = true
            if (s.state === 'done') onEvent({ type: 'done', study_id: id, n_failed: Number(s.manifest?.n_failed ?? 0) })
            else if (s.state === 'cancelled') onEvent({ type: 'cancelled' })
            else onEvent({ type: 'error', message: s.error ?? 'study failed' })
          }
        } catch { /* keep polling */ }
      }, 800)
    }
    let ws: WebSocket | null = null
    try {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws'
      ws = new WebSocket(`${proto}://${location.host}/api/studies/${enc(id)}/events`)
      ws.onmessage = (m) => {
        try {
          const ev = parseLooseJson<StudyEvent>(String(m.data))
          if (ev.type !== 'progress') closed = true
          onEvent(ev)
        } catch { /* ignore malformed frame */ }
      }
      ws.onerror = () => startPolling()
      ws.onclose = () => { if (!closed) startPolling() }
    } catch { startPolling() }
    return () => { closed = true; if (poll) clearInterval(poll); ws?.close() }
  },
  cancelStudy: (id) => post<StudyStatus>(`/api/studies/${enc(id)}/cancel`),
  studyRuns: async (id, offset, limit) => json<StudyRuns>(`/api/studies/${enc(id)}/runs?offset=${offset}&limit=${limit}`),
  sensitivity: (id, boot) => json<SensitivityResult>(`/api/studies/${enc(id)}/sensitivity?boot=${boot}`),
  methodsText: (req) => post<MethodsText>('/api/methods_text', req),
  provenance: () => json<Provenance>('/api/provenance'),
  materialOptics: (req) => post<MaterialOptics>('/api/material_optics', req),
  calibratedFluorophores: () => json<CalibratedFluorophore[]>('/api/calibrated_fluorophores'),
  unmix: async (req) => nanify(await post<UnmixResult>('/api/unmix', req)),
  async unmixMap(uid, name) {
    const r = await fetch(`/api/unmix/${enc(uid)}/map/${enc(name)}`)
    if (!r.ok) throw new ApiError(r.status, `GET unmix map → ${r.status}`)
    return { width: Number(r.headers.get('X-Width') ?? 0), height: Number(r.headers.get('X-Height') ?? 0), data: new Float32Array(await r.arrayBuffer()) }
  },
  createSurrogate: (req) => post('/api/surrogates', req),
  surrogate: (id) => json<SurrogateSummary>(`/api/surrogates/${enc(id)}`),
  fit: async (req) => nanify(await post<FitResult>('/api/fit', req)),
  fitMap: (req) => post<FitMapResult>('/api/fit', req),
  fisher: async (req) => { const r = await post<FisherResult | { job_id: string }>('/api/fisher', req); return isJob(r) ? r : nanify(r) },
  design: async (req) => { const r = await post<DesignResult | { job_id: string }>('/api/design', req); return isJob(r) ? r : nanify(r) },
  inverseResult: async <T,>(id: string) => nanify(await json<T>(`/api/inverse/results/${enc(id)}`)),
  watchJob(jobId, onEvent) {
    const eng = getEngine()
    return eng ? eng.watchJob(jobId, onEvent) : (() => {})
  },
}

let mock: Promise<ResearchApi> | null = null
/** Synchronous accessor once `loadResearchApi()` has resolved. */
let current: ResearchApi | null = null

/** Pick the implementation matching the active engine (real HTTP, or the offline mock). */
export async function loadResearchApi(): Promise<ResearchApi> {
  const eng = getEngine()
  if (eng?.kind === 'real') return (current = realResearch)
  mock ??= import('./mockResearch').then((m) => m.mockResearch)
  return (current = await mock)
}
export const researchApi = (): ResearchApi | null => current
