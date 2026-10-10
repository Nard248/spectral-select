import { describe, expect, it } from 'vitest'
import { sliceBand } from './render'

/** The engine sends H*W*B float32 in C order (row y, column x, band fastest): value(x, y, b) = data[(y*W + x)*B + b].
 *  The explorer's click -> /pixel?x=&y= mapping relies on this exact orientation (no row/column flip). */
describe('cube indexing', () => {
  it('sliceBand returns row-major H*W with x fastest', () => {
    const W = 5, H = 3, B = 4
    const data = new Float32Array(W * H * B)
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) for (let b = 0; b < B; b++) data[(y * W + x) * B + b] = 1000 * y + 10 * x + b
    const band2 = sliceBand({ width: W, height: H, bands: B, data }, 2)
    expect(band2.length).toBe(W * H)
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) expect(band2[y * W + x]).toBe(1000 * y + 10 * x + 2)
  })
})
