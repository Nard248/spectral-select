import { useMemo } from 'react'
import * as THREE from 'three'
import type { StageProps } from './types'
import { breadboardMaps, radialBlob } from './util/textures'
import { Rubber } from './util/materials'

export { Post, Clamp, Joint, Base } from './util/parts'

const PITCH = 25            // mm, M6 hole pitch
const THICK_DEFAULT = 12    // mm board thickness when the experiment gives none

interface Look { color: string; metalness: number; roughness: number; holes: boolean; transmission?: number }
function lookFor(id: string | null | undefined, display?: string | null, rough?: number): Look {
  switch (id) {
    case 'black_anodized': return { color: '#1b1c21', metalness: 0.78, roughness: 0.5, holes: true }
    case 'matte_grey': return { color: '#8a8d93', metalness: 0.35, roughness: 0.68, holes: true }
    case 'white_paper': return { color: '#f3f5fb', metalness: 0, roughness: 0.92, holes: false }
    case 'spectralon': return { color: '#ffffff', metalness: 0, roughness: 1, holes: false }
    case 'glass': return { color: '#cfe6ff', metalness: 0, roughness: 0.04, holes: false, transmission: 0.92 }
    default: return { color: display ?? '#2a2c31', metalness: 0.4, roughness: THREE.MathUtils.clamp(0.3 + (rough ?? 0.5) * 0.6, 0.2, 1), holes: true }
  }
}

/** Optical breadboard: chamfered slab, M6 tapped holes at a 25 mm pitch (mipmapped, anisotropic texture), feet and a soft ground blob. */
function Breadboard({ extent, thick, look }: { extent: number; thick: number; look: Look }) {
  const THICK = thick
  const geo = useMemo(() => {
    const bevel = Math.min(1.8, thick / 3)
    const half = extent / 2 - bevel
    const shape = new THREE.Shape()
    const r = 3
    shape.moveTo(-half + r, -half); shape.lineTo(half - r, -half); shape.quadraticCurveTo(half, -half, half, -half + r)
    shape.lineTo(half, half - r); shape.quadraticCurveTo(half, half, half - r, half); shape.lineTo(-half + r, half)
    shape.quadraticCurveTo(-half, half, -half, half - r); shape.lineTo(-half, -half + r); shape.quadraticCurveTo(-half, -half, -half + r, -half)
    const depth = thick - 2 * bevel
    const g = new THREE.ExtrudeGeometry(shape, { depth, bevelEnabled: true, bevelThickness: bevel, bevelSize: bevel, bevelSegments: 2, curveSegments: 6 })
    g.rotateX(-Math.PI / 2)
    g.translate(0, -(depth + bevel), 0)
    return g
  }, [extent, thick])
  const maps = useMemo(() => {
    const m = breadboardMaps(look.holes)
    ;[m.map, m.rough, m.bump].forEach((t) => { t.repeat.set(1 / PITCH, 1 / PITCH); t.offset.set(0, 0) })
    return m
  }, [look.holes])
  const side = useMemo(() => new THREE.MeshPhysicalMaterial({ color: look.color, metalness: look.metalness, roughness: Math.min(look.roughness + 0.15, 1), envMapIntensity: 1 }), [look])
  const top = useMemo(() => new THREE.MeshPhysicalMaterial({
    color: look.color, metalness: look.metalness, roughness: 1, map: maps.map, roughnessMap: maps.rough, bumpMap: maps.bump, bumpScale: look.holes ? 3 : 0.2,
    ...(look.transmission ? { transmission: look.transmission, thickness: THICK, ior: 1.5 } : {}), envMapIntensity: 1,
  }), [look, maps])
  // roughnessMap multiplies `roughness`; scale so the map's mid-grey lands near the target
  top.roughness = Math.min(look.roughness * 1.5, 1)
  const feet = [[-1, -1], [1, -1], [-1, 1], [1, 1]]
  return (
    <group>
      <mesh geometry={geo} material={[top, side]} receiveShadow castShadow />
      {feet.map(([x, z], i) => (
        <mesh key={i} position={[x * (extent / 2 - 40), -THICK - 9, z * (extent / 2 - 40)]} castShadow>
          <cylinderGeometry args={[16, 18, 18, 32]} /><Rubber />
        </mesh>
      ))}
      <mesh position={[0, -THICK - 18.5, 0]} rotation={[-Math.PI / 2, 0, 0]} renderOrder={-5}>
        <planeGeometry args={[extent * 1.35, extent * 1.35]} />
        <meshBasicMaterial map={radialBlob()} transparent opacity={0.9} depthWrite={false} toneMapped={false} fog={false} />
      </mesh>
    </group>
  )
}

const GRID_VERT = 'varying vec3 vW; void main(){ vec4 w = modelMatrix*vec4(position,1.0); vW = w.xyz; gl_Position = projectionMatrix*viewMatrix*w; }'
const GRID_FRAG = /* glsl */ `
varying vec3 vW;
float grid(vec2 p, float s){ vec2 c = p/s; vec2 g = abs(fract(c-0.5)-0.5)/max(fwidth(c),vec2(1e-5)); return 1.0 - min(min(g.x,g.y),1.0); }
void main(){
  float r = length(vW.xz);
  float fine = grid(vW.xz, 10.0), coarse = grid(vW.xz, 100.0);
  float fade = 1.0 - smoothstep(250.0, 2400.0, r);
  float pool = exp(-r*r/(2.0*380.0*380.0));
  float a = (fine*0.14 + coarse*0.42) * fade;
  vec3 col = mix(vec3(0.20,0.34,0.52), vec3(0.45,0.7,1.0), coarse);
  col += pool*vec3(0.02,0.035,0.06);
  gl_FragColor = vec4(col*(0.7+0.6*pool), a + pool*0.08);
}`
/** `background.kind = 'none'`: subtle infinite grid in a void. */
function VoidGrid() {
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.4, 0]} renderOrder={-4}>
      <planeGeometry args={[9000, 9000]} />
      <shaderMaterial vertexShader={GRID_VERT} fragmentShader={GRID_FRAG} transparent depthWrite={false} toneMapped={false} />
    </mesh>
  )
}

/** The table: optical breadboard (kind 'plane', look follows the background material) or an infinite grid (kind 'none'). */
export function Stage({ background, material }: StageProps) {
  if (background.kind === 'none') return <VoidGrid />
  const look = lookFor(material?.id ?? background.material_id, material?.display_color, material?.surface_roughness)
  return <Breadboard extent={background.extent_mm} thick={background.thickness_mm ?? THICK_DEFAULT} look={look} />
}
