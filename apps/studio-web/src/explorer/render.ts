/** Pixel rendering of float rasters / cubes into RGBA with colormaps — shared by the Explorer and the PiP "Captured" view. */
import { lut, type CmapName } from '../ui/colormaps'
import { wavelengthToRGB } from '../lib/color'
import type { Cube } from '../api/client'

export function percentiles(data: ArrayLike<number>, lo: number, hi: number, stride = 7): [number, number] {
  const s: number[] = []
  for (let i = 0; i < data.length; i += stride) { const v = data[i]; if (Number.isFinite(v)) s.push(v) }
  if (!s.length) return [0, 1]
  const a = Float32Array.from(s).sort()
  return [a[Math.floor(lo * (a.length - 1))], a[Math.floor(hi * (a.length - 1))]]
}

export interface ScalarStyle { cmap: CmapName; lo: number; hi: number; log: boolean }

export function paintScalar(img: ImageData, band: ArrayLike<number>, st: ScalarStyle) {
  const L = lut(st.cmap)
  const span = st.hi - st.lo || 1
  for (let k = 0; k < band.length; k++) {
    let t = (band[k] - st.lo) / span
    t = Math.min(1, Math.max(0, t))
    if (st.log) t = Math.log10(1 + 99 * t) / 2
    const i = Math.round(t * 255) * 3, o = k * 4
    img.data[o] = L[i]; img.data[o + 1] = L[i + 1]; img.data[o + 2] = L[i + 2]; img.data[o + 3] = 255
  }
}

/** Slice band `j` of an H*W*B cube (band fastest). */
export function sliceBand(cube: Cube, j: number, out?: Float32Array): Float32Array {
  const N = cube.width * cube.height, B = cube.bands
  const o = out && out.length === N ? out : new Float32Array(N)
  for (let k = 0; k < N; k++) o[k] = cube.data[k * B + j]
  return o
}

/** Weighted RGB composite: each band's colour from its wavelength (true colour = what an eye would see through the long-pass). */
export function paintComposite(img: ImageData, cube: Cube, emission: number[], opts: { gain?: number; falseColor?: [number, number, number] } = {}) {
  const N = cube.width * cube.height, B = cube.bands
  const w: [number, number, number][] = emission.map((wl) => {
    if (opts.falseColor) return [0, 0, 0]
    return wavelengthToRGB(wl)
  })
  if (opts.falseColor) {
    // three chosen bands → R,G,B
    opts.falseColor.forEach((b, c) => { if (w[b]) w[b][c] = 1 })
  }
  // per-band black level: lowest 1 % of that band
  const bl = new Float32Array(B)
  for (let j = 0; j < B; j++) { const s: number[] = []; for (let k = 0; k < N; k += 31) s.push(cube.data[k * B + j]); s.sort((a, b) => a - b); bl[j] = s[Math.floor(s.length * 0.02)] ?? 0 }
  const rgb = new Float32Array(N * 3)
  let mx = 1e-9
  for (let k = 0; k < N; k++) {
    let r = 0, g = 0, b = 0
    for (let j = 0; j < B; j++) { const v = Math.max(0, cube.data[k * B + j] - bl[j]); r += v * w[j][0]; g += v * w[j][1]; b += v * w[j][2] }
    rgb[k * 3] = r; rgb[k * 3 + 1] = g; rgb[k * 3 + 2] = b
  }
  const lum = new Float32Array(Math.ceil(N / 13))
  for (let k = 0, i = 0; k < N; k += 13, i++) lum[i] = Math.max(rgb[k * 3], rgb[k * 3 + 1], rgb[k * 3 + 2])
  lum.sort(); mx = Math.max(lum[Math.floor(lum.length * 0.995)] || 1e-9, 1e-9) / (opts.gain ?? 1)
  for (let k = 0; k < N; k++) {
    for (let c = 0; c < 3; c++) img.data[k * 4 + c] = Math.pow(Math.min(1, rgb[k * 3 + c] / mx), 0.62) * 255
    img.data[k * 4 + 3] = 255
  }
}
