import { useStudio, activeSource } from '../state/store'

/** Excitation (nm) currently driving beam colour / glow: the running job's step, else the preview pick, else the first enabled step. */
export function useActiveNm(): number | null {
  return useStudio((s) => {
    if (s.job && s.job.state === 'running' && s.job.activeNm != null) return s.job.activeNm
    if (s.previewNm != null) return s.previewNm
    return activeSource(s.exp)?.sequence.find((x) => x.enabled)?.wavelength_nm ?? null
  })
}
export const useAcquiring = () => useStudio((s) => (s.job?.state === 'running') || s.previewing)
