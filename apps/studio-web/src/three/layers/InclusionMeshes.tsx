import { useEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import { Html, Line } from '@react-three/drei'
import { useFrame, useThree } from '@react-three/fiber'
import type { Inclusion, LayeredBlockShape, MaterialSpec } from '../../generated/experiment'
import type { BlockSection } from '../types'
import { cylinderSegment, crossSection, describeSize, depthBelowTop } from '../../lib/inclusions'

export interface InclusionItem { inc: Inclusion; material?: MaterialSpec }

const colorOf = (m?: MaterialSpec) => new THREE.Color(m?.display_color ?? '#c8d0dc')

/** Local-space geometry (centred on the origin) of one inclusion; cylinders are clipped to the footprint along their axis. */
function incGeometry(shape: LayeredBlockShape, inc: Inclusion): { geom: THREE.BufferGeometry; position: [number, number, number] } | null {
  const [cx, cy, cz] = inc.center_mm
  switch (inc.kind) {
    case 'sphere': return { geom: new THREE.SphereGeometry(inc.radius_mm, 48, 32), position: [cx, cy, cz] }
    case 'ellipsoid': { const g = new THREE.SphereGeometry(1, 48, 32); g.scale(...inc.semi_axes_mm); return { geom: g, position: [cx, cy, cz] } }
    case 'box': return { geom: new THREE.BoxGeometry(...inc.size_mm), position: [cx, cy, cz] }
    case 'cylinder': {
      const seg = cylinderSegment(shape, inc); if (!seg) return null
      const g = new THREE.CylinderGeometry(inc.radius_mm, inc.radius_mm, seg[1] - seg[0], 40, 1, false)
      if (inc.axis === 'x') g.rotateZ(Math.PI / 2); else g.rotateX(Math.PI / 2)
      const mid = (seg[0] + seg[1]) / 2
      return { geom: g, position: inc.axis === 'x' ? [mid, cy, cz] : [cx, cy, mid] }
    }
  }
}

/** Translucent coloured inclusions inside a layered block (block-local frame), drawn after the layers as an x-ray overlay so they
 *  stay legible through turbid covers (the layer faces do not write depth). A cutaway clips it with a world-space clipping plane and caps it
 *  with the exact analytic cross-section (ellipse / rectangle) plus an outline. */
export function InclusionMeshes({ shape, items, section, cutAt, labels, highlight }: {
  shape: LayeredBlockShape; items: InclusionItem[]; section?: BlockSection | null; cutAt: number; labels?: boolean; highlight?: number | null
}) {
  const gl = useThree((s) => s.gl)
  useEffect(() => { gl.localClippingEnabled = true }, [gl])
  const group = useRef<THREE.Group>(null)
  const plane = useMemo(() => new THREE.Plane(), [])
  const planes = useMemo(() => [plane], [plane])
  // world-space plane from the local cut plane, refreshed every frame (the block can be moved / rotated)
  useFrame(() => {
    const g = group.current; if (!g || !section) return
    g.updateWorldMatrix(true, false)
    const n = new THREE.Vector3(section.axis === 'x' ? 1 : 0, 0, section.axis === 'z' ? 1 : 0).multiplyScalar(section.keep === 'low' ? -1 : 1)
    plane.setFromNormalAndCoplanarPoint(n, new THREE.Vector3(section.axis === 'x' ? cutAt : 0, 0, section.axis === 'z' ? cutAt : 0)).applyMatrix4(g.matrixWorld)
  })
  const built = useMemo(() => items.map((it) => incGeometry(shape, it.inc)), [items, shape])
  useEffect(() => () => built.forEach((b) => b?.geom.dispose()), [built])
  const halfW = section ? (section.axis === 'x' ? shape.footprint_mm[1] : shape.footprint_mm[0]) / 2 : 0
  const nrm = section ? (section.keep === 'low' ? 1 : -1) : 1

  return (
    <group ref={group}>
      {items.map((it, i) => {
        const b = built[i]; if (!b) return null
        const color = colorOf(it.material)
        const hl = highlight === i
        const dyed = !!it.material?.dyes.length
        const cs = section ? crossSection(it.inc, section.axis, cutAt, halfW) : null
        const label = labels && cs && section
        return (
          <group key={i}>
            <group renderOrder={36}><mesh geometry={b.geom} position={b.position} renderOrder={36}>
              <meshPhysicalMaterial color={color} transparent opacity={hl ? 0.9 : 0.62} depthWrite={false} roughness={0.35} metalness={0} clearcoat={0.5}
                emissive={color} emissiveIntensity={hl ? 0.55 : dyed ? 0.28 : 0.14} clippingPlanes={section ? planes : undefined} side={THREE.DoubleSide} />
            </mesh></group>
            {cs && section && (() => {
              const pos: [number, number, number] = section.axis === 'x' ? [cutAt + nrm * 0.03, cs.cy, cs.cw] : [cs.cw, cs.cy, cutAt + nrm * 0.03]
              const rot: [number, number, number] = section.axis === 'x' ? [0, Math.PI / 2, 0] : [0, 0, 0]
              const outline: [number, number, number][] = []
              const toW = (u: number, v: number): [number, number, number] => (section.axis === 'x' ? [pos[0] + nrm * 0.02, cs.cy + v, cs.cw + u] : [cs.cw + u, cs.cy + v, pos[2] + nrm * 0.02])
              if (cs.shape === 'ellipse') for (let k = 0; k <= 64; k++) outline.push(toW(cs.hw * Math.cos((k / 64) * 2 * Math.PI), cs.hy * Math.sin((k / 64) * 2 * Math.PI)))
              else [[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]].forEach(([a, c]) => outline.push(toW(a * cs.hw, c * cs.hy)))
              return (
                <group renderOrder={46}>
                  <mesh position={pos} rotation={rot} scale={cs.shape === 'ellipse' ? [cs.hw, cs.hy, 1] : [2 * cs.hw, 2 * cs.hy, 1]} renderOrder={46}>
                    {cs.shape === 'ellipse' ? <circleGeometry args={[1, 64]} /> : <planeGeometry args={[1, 1]} />}
                    <meshStandardMaterial color={color} roughness={0.5} emissive={color} emissiveIntensity={hl ? 0.7 : 0.4} side={THREE.DoubleSide} polygonOffset polygonOffsetFactor={-4} polygonOffsetUnits={-4} />
                  </mesh>
                  <Line points={outline} color={hl ? '#ffffff' : '#f2f7ff'} lineWidth={hl ? 2.4 : 1.6} transparent opacity={0.95} toneMapped={false} depthTest={false} renderOrder={62} />
                  {label && (
                    <Html position={[pos[0], pos[1] + cs.hy + 0.15, pos[2]]} zIndexRange={[20, 0]} style={{ pointerEvents: 'none' }}>
                      <div className="blk-label inc" style={{ ['--c' as string]: '#' + color.getHexString() }}><i />{it.inc.name}<b>{describeSize(it.inc)} · {depthBelowTop(shape, it.inc).toFixed(1)} mm deep</b></div>
                    </Html>
                  )}
                </group>
              )
            })()}
          </group>
        )
      })}
    </group>
  )
}
