/** Imperative bridge between DOM (shortcuts, drop targets, PiP frame) and the R3F scene. */
import type * as THREE from 'three'

export interface ViewApi {
  frame: (target?: THREE.Vector3Tuple | null, radius?: number) => void
  /** world point on the table (y=0) under client coordinates, or null */
  /** eased camera fly-to (position, orbit target) over `ms`; resolves when the move ends (or is interrupted) */
  flyTo: (pos: THREE.Vector3Tuple, target: THREE.Vector3Tuple, ms?: number) => Promise<void>
  /** current orbit pose */
  pose: () => { position: THREE.Vector3Tuple; target: THREE.Vector3Tuple }
  pickTable: (clientX: number, clientY: number) => THREE.Vector3Tuple | null
}
export const viewApi: { current: ViewApi | null } = { current: null }
export const pipFrame: { el: HTMLElement | null } = { el: null }
export const viewportEl: { el: HTMLElement | null } = { el: null }
if (import.meta.env.DEV && typeof window !== 'undefined') Object.defineProperty(window, '__viewApi', { get: () => viewApi.current, configurable: true })
