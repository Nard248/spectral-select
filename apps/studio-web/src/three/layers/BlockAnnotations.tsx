import { useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { Html, Line } from '@react-three/drei'
import { useFrame, useThree } from '@react-three/fiber'
import type { LayeredBlockShape } from '../../generated/experiment'
import type { BlockSection } from '../types'
import type { Region } from './blockGeometry'
import { interfaceHeights } from './profile'
import type { LayerLook } from './look'

const fmtT = (t: number) => (t >= 10 ? t.toFixed(1) : t >= 0.095 ? t.toFixed(2) : t === 0 ? '0' : (t * 1000).toFixed(0) + ' µm')

/** Spread label heights so that neighbours are at least `gap` apart (greedy, keeps order). */
export function spreadLabels(ys: number[], gap: number): number[] {
  const order = ys.map((y, i) => [y, i] as const).sort((a, b) => a[0] - b[0])
  const out = new Array<number>(ys.length)
  let prev = -Infinity
  for (const [y, i] of order) { const v = Math.max(y, prev + gap); out[i] = v; prev = v }
  // re-centre the cluster around its original mean so the stack does not drift upward
  const shift = ys.reduce((a, b) => a + b, 0) / ys.length - out.reduce((a, b) => a + b, 0) / ys.length
  return out.map((v) => v + shift)
}

/** Thickness callouts on the section face: layer name + thickness at one end, thickness at the other end (so a wedge reads
 *  e.g. "cover 2.00 mm … 0 mm"), leader lines to the layer mid-heights, and the footprint dimension line.
 *  Labels are HTML (constant pixel size); their stacking gap adapts to the camera distance. */
export function BlockAnnotations({ shape, section, region, looks }: { shape: LayeredBlockShape; section: BlockSection; region: Region; looks: LayerLook[] }) {
  const camera = useThree((s) => s.camera)
  const height = useThree((s) => s.size.height)
  const ref = useRef<THREE.Group>(null)
  const [mmPerPx, setMmPerPx] = useState(0.2)
  const tmp = useMemo(() => new THREE.Vector3(), [])
  useFrame(() => {
    if (!ref.current) return
    ref.current.getWorldPosition(tmp)
    const dist = camera.position.distanceTo(tmp)
    const fov = (camera as THREE.PerspectiveCamera).fov ?? 34
    const m = (2 * dist * Math.tan((fov * Math.PI) / 360)) / height
    const q = Math.pow(1.12, Math.round(Math.log(m) / Math.log(1.12)))   // bucketed so React does not re-render every frame
    if (Math.abs(q - mmPerPx) > 1e-9) setMmPerPx(q)
  })

  const fp = shape.footprint_mm
  const cut = section.axis === 'x' ? region[section.keep === 'low' ? 'x1' : 'x0'] : region[section.keep === 'low' ? 'z1' : 'z0']
  const w0 = section.axis === 'x' ? -fp[1] / 2 : -fp[0] / 2, w1 = -w0
  const nrm = section.keep === 'low' ? 1 : -1
  const at = (w: number, y: number, off = 0): [number, number, number] =>
    section.axis === 'x' ? [cut + nrm * off, y, w] : [w, y, cut + nrm * off]
  const hAt = (w: number) => (section.axis === 'x' ? interfaceHeights(shape, cut, w) : interfaceHeights(shape, w, cut))

  const gap = 21 * mmPerPx   // 21 px between label centres
  const pad = 9 * mmPerPx + 1.2
  const rows = useMemo(() => {
    const hl = hAt(w0), hr = hAt(w1)
    return shape.layers.map((l, i) => ({
      i, name: l.name, midL: (hl[i] + hl[i + 1]) / 2, midR: (hr[i] + hr[i + 1]) / 2,
      tL: hl[i + 1] - hl[i], tR: hr[i + 1] - hr[i],
    }))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [shape, section.axis, cut])
  const left = spreadLabels(rows.map((r) => r.midL), gap), right = spreadLabels(rows.map((r) => r.midR), gap)
  const showRight = rows.some((r) => Math.abs(r.tL - r.tR) > 1e-6)
  const dimY = -2.4 * Math.max(1, mmPerPx * 6)
  const sizeW = w1 - w0
  const colorOf = (i: number) => '#' + (looks[i]?.color.getHexString() ?? 'bdbdbd')

  return (
    <group ref={ref} renderOrder={80}>
      {rows.map((r) => r.tL > 1e-6 && (
        <group key={`l${r.i}`}>
          <Line points={[at(w0, r.midL, 0.05), at(w0 - pad * 0.65, left[r.i], 0.05)]} color={colorOf(r.i)} lineWidth={1.2} transparent opacity={0.9} toneMapped={false} depthTest={false} renderOrder={81} />
          <Html position={at(w0 - pad, left[r.i], 0.05)} zIndexRange={[20, 0]} style={{ pointerEvents: 'none' }}>
            <div className="blk-label l" style={{ ['--c' as string]: colorOf(r.i) }}><i />{r.name}<b>{fmtT(r.tL)}{r.tL >= 0.095 ? ' mm' : ''}</b></div>
          </Html>
        </group>
      ))}
      {showRight && rows.map((r) => r.tR > 1e-6 && Math.abs(r.tL - r.tR) > 1e-6 && (
        <group key={`r${r.i}`}>
          <Line points={[at(w1, r.midR, 0.05), at(w1 + pad * 0.65, right[r.i], 0.05)]} color={colorOf(r.i)} lineWidth={1.2} transparent opacity={0.9} toneMapped={false} depthTest={false} renderOrder={81} />
          <Html position={at(w1 + pad, right[r.i], 0.05)} zIndexRange={[20, 0]} style={{ pointerEvents: 'none' }}>
            <div className="blk-label r" style={{ ['--c' as string]: colorOf(r.i) }}><b>{fmtT(r.tR)}{r.tR >= 0.095 ? ' mm' : ''}</b></div>
          </Html>
        </group>
      ))}
      {showRight && rows.map((r) => r.tR <= 1e-6 && r.tL > 1e-6 && (
        <Html key={`z${r.i}`} position={at(w1 + pad, right[r.i], 0.05)} zIndexRange={[20, 0]} style={{ pointerEvents: 'none' }}>
          <div className="blk-label r zero" style={{ ['--c' as string]: colorOf(r.i) }}><b>0 mm</b></div>
        </Html>
      ))}
      {/* footprint dimension line */}
      <Line points={[at(w0, dimY), at(w1, dimY)]} color="#9fb3c8" lineWidth={1} transparent opacity={0.8} toneMapped={false} depthTest={false} renderOrder={81} />
      <Line points={[at(w0, dimY - 0.6), at(w0, dimY + 0.6)]} color="#9fb3c8" lineWidth={1} transparent opacity={0.8} toneMapped={false} depthTest={false} renderOrder={81} />
      <Line points={[at(w1, dimY - 0.6), at(w1, dimY + 0.6)]} color="#9fb3c8" lineWidth={1} transparent opacity={0.8} toneMapped={false} depthTest={false} renderOrder={81} />
      <Html position={at((w0 + w1) / 2, dimY - 0.2)} zIndexRange={[20, 0]} style={{ pointerEvents: 'none' }}>
        <div className="blk-label dim">{sizeW.toFixed(0)} mm</div>
      </Html>
    </group>
  )
}
