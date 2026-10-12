import { create } from 'zustand'

/** Guided-tour runtime state. `token` invalidates in-flight step scripts when the user navigates or exits. */
interface TourState {
  active: boolean
  step: number
  total: number
  auto: boolean
  busy: boolean
  /** caption override (a step can update its text as its script progresses) */
  caption: string | null
  token: number
  /** engine preview thumbnails shown in the caption card (e.g. scattering ON vs OFF) */
  thumbs: { label: string; url: string }[]
}
export const useTour = create<TourState>()(() => ({ active: false, step: 0, total: 0, auto: false, busy: false, caption: null, token: 0, thumbs: [] }))
