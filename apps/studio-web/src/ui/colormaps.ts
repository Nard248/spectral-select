/** Perceptual colormaps as 9-stop control points, interpolated. */
const STOPS: Record<string, string[]> = {
  viridis: ['#440154', '#472d7b', '#3b528b', '#2c728e', '#21918c', '#28ae80', '#5ec962', '#addc30', '#fde725'],
  magma: ['#000004', '#180f3e', '#451077', '#721f81', '#9f2f7f', '#cd4071', '#f1605d', '#fd9567', '#fcfdbf'],
  inferno: ['#000004', '#1b0c42', '#4b0c6b', '#781c6d', '#a52c60', '#cf4446', '#ed6925', '#fb9b06', '#fcffa4'],
  gray: ['#000000', '#ffffff'],
  turbo: ['#30123b', '#4662d7', '#36aaf9', '#1ae4b6', '#72fe5e', '#c7ef34', '#faba39', '#f66b19', '#7a0403'],
}
export type CmapName = keyof typeof STOPS
export const CMAP_NAMES = Object.keys(STOPS) as CmapName[]

const hex = (h: string): [number, number, number] => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]
const luts = new Map<string, Uint8ClampedArray>()

/** 256×3 lookup table. */
export function lut(name: CmapName): Uint8ClampedArray {
  let l = luts.get(name)
  if (l) return l
  const st = STOPS[name].map(hex)
  l = new Uint8ClampedArray(256 * 3)
  for (let i = 0; i < 256; i++) {
    const p = (i / 255) * (st.length - 1)
    const k = Math.min(st.length - 2, Math.floor(p)), t = p - k
    for (let c = 0; c < 3; c++) l[i * 3 + c] = st[k][c] * (1 - t) + st[k + 1][c] * t
  }
  luts.set(name, l)
  return l
}
export const cssGradient = (name: CmapName) => `linear-gradient(90deg, ${STOPS[name].join(',')})`
export const colorAt = (name: CmapName, t: number): [number, number, number] => {
  const l = lut(name); const i = Math.round(Math.min(1, Math.max(0, t)) * 255) * 3
  return [l[i], l[i + 1], l[i + 2]]
}
