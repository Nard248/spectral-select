/** Turn the engine's terse job messages into a readable stage label + the excitation being simulated.
 *
 * Real engine messages (capture.py): "tracing rays", "radiance", "radiance k/n" (k-th ray block of n; every block
 * is evaluated for ALL excitations), "sensor k/n" (k-th excitation step is being exposed), "done".
 * Mock engine: "Exposing 405 nm (i/n)". Anything else is shown verbatim. */
export interface ProgressInfo { label: string; stage: 'trace' | 'transport' | 'sensor' | 'other'; activeNm: number | null; step?: [number, number] }

export function describeProgress(message: string, wls: number[]): ProgressInfo {
  const m = message.trim()
  const n = wls.length
  const named = /(\d+(?:\.\d+)?)\s*nm/.exec(m)
  const frac = /(\d+)\s*\/\s*(\d+)/.exec(m)
  const k = frac ? Number(frac[1]) : null
  const of = frac ? Number(frac[2]) : null
  if (/^trac/i.test(m)) return { label: 'Tracing rays through the scene', stage: 'trace', activeNm: wls[0] ?? null }
  if (/^radiance/i.test(m)) {
    // all excitations are solved per ray block: cycle the beam colour through them so the viewport shows what is being solved
    const idx = k != null && n ? (k - 1) % n : 0
    return { label: k != null && of != null ? `Fluorescence transport · ${n} excitations · block ${k}/${of}` : 'Fluorescence transport', stage: 'transport', activeNm: wls[idx] ?? null, step: k != null && of != null ? [k, of] : undefined }
  }
  if (/^stor/i.test(m)) return { label: 'Storing the capture', stage: 'other', activeNm: null }
  if (/^sensor/i.test(m)) {
    const nm = k != null ? wls[k - 1] ?? null : null
    return { label: nm != null ? `Exposing sensor · ${nm} nm excitation (${k}/${of})` : 'Exposing sensor', stage: 'sensor', activeNm: nm, step: k != null && of != null ? [k, of] : undefined }
  }
  if (named) {
    const nm = Number(named[1])
    return { label: k != null && of != null ? `Exposing ${nm} nm excitation (${k}/${of})` : `Exposing ${nm} nm excitation`, stage: 'sensor', activeNm: nm, step: k != null && of != null ? [k, of] : undefined }
  }
  return { label: m ? m.charAt(0).toUpperCase() + m.slice(1) : 'Working…', stage: 'other', activeNm: null }
}

/** Remaining seconds from a progress fraction; null while the estimate is meaningless. */
export function etaSeconds(fraction: number, startedAt: number | undefined, now = Date.now()): number | null {
  if (startedAt == null || fraction < 0.04 || fraction >= 1) return null
  const el = (now - startedAt) / 1000
  return Math.max(0, (el * (1 - fraction)) / fraction)
}

export const fmtEta = (s: number) => (s < 1.5 ? '<2 s left' : s < 90 ? `~${Math.round(s)} s left` : `~${Math.floor(s / 60)} min ${Math.round(s % 60)} s left`)
