import { useMemo, type ReactElement } from 'react'
import * as THREE from 'three'
import type { MaterialSpec, SceneObject } from '../generated/experiment'
import type { SampleMeshProps } from './types'
import { Edges } from '@react-three/drei'
import { HullMat, hullColor } from './util/parts'
import { noiseBump } from './util/textures'
import { LayeredBlockMesh } from './LayeredBlockMesh'

/** thin bright glass edge lines so clear vessels read against dark backgrounds */
const GlassEdges = () => <Edges threshold={25} color="#d6ecff" transparent opacity={0.35} toneMapped={false} />

type MatFn = () => ReactElement
type Kind = 'glass' | 'liquid' | 'gel' | 'solid'

/** Visual class of a material from its optical spec. */
export function classifyMaterial(m: MaterialSpec): Kind {
  if (m.id === 'glass' || (m.scattering.model === 'none' && !m.aqueous && m.refractive_index > 1.4)) return 'glass'
  if (m.scattering.model === 'none' && m.aqueous) return 'liquid'
  if (m.scattering.model === 'none') return 'liquid'
  if (m.scattering.mus_prime_500_per_mm < 1 || m.aqueous) return 'gel'
  return 'solid'
}

const col = (m: MaterialSpec, fb = '#bdbdbd') => new THREE.Color(m.display_color ?? fb)

function glassMat(m: MaterialSpec, thickness: number): MatFn {
  const tint = col(m, '#dfeeff')
  return () => <meshPhysicalMaterial color={tint} transmission={1} thickness={thickness} ior={1.52} roughness={0.03} metalness={0}
    specularIntensity={1} envMapIntensity={1.6} attenuationColor={tint} attenuationDistance={400} clearcoat={0.3} />
}

function bodyMat(m: MaterialSpec, glow: THREE.Color | null, gi: number, thickness: number): MatFn {
  const kind = classifyMaterial(m)
  const c = col(m)
  const emissive = glow ?? new THREE.Color(0, 0, 0)
  const ei = glow ? 0.15 + gi * 1.8 : 0
  if (kind === 'glass') return glassMat(m, thickness)
  if (kind === 'liquid') {
    return () => <meshPhysicalMaterial color={c} transmission={0.9} thickness={Math.max(thickness, 1)} ior={m.refractive_index} roughness={0.02}
      attenuationColor={c} attenuationDistance={Math.max(thickness * 1.4, 2)} clearcoat={1} clearcoatRoughness={0.03} specularIntensity={1}
      emissive={emissive} emissiveIntensity={ei} envMapIntensity={1.6} />
  }
  if (kind === 'gel') {
    return () => <meshPhysicalMaterial color={c} transmission={0.55} thickness={Math.max(thickness, 1)} ior={m.refractive_index} roughness={0.12 + 0.5 * m.surface_roughness}
      attenuationColor={c} attenuationDistance={Math.max(thickness, 3)} sheen={0.5} sheenColor={c.clone().lerp(new THREE.Color('#ffffff'), 0.5)} sheenRoughness={0.5}
      clearcoat={0.5} clearcoatRoughness={0.2} emissive={emissive} emissiveIntensity={ei} />
  }
  const rough = THREE.MathUtils.clamp(0.28 + 0.72 * m.surface_roughness, 0.2, 1)
  const lowScatter = m.scattering.mus_prime_500_per_mm < 3
  return () => <meshPhysicalMaterial color={c} roughness={rough} metalness={0} sheen={0.7} sheenColor={c.clone().lerp(new THREE.Color('#ffffff'), 0.55)}
    sheenRoughness={0.55} transmission={lowScatter ? 0.12 : 0} thickness={Math.max(thickness, 1)} attenuationColor={c} attenuationDistance={4}
    bumpMap={noiseBump()} bumpScale={1.2 * m.surface_roughness} clearcoat={m.surface_roughness < 0.3 ? 0.4 : 0} clearcoatRoughness={0.4} emissive={emissive} emissiveIntensity={ei} envMapIntensity={0.9} />
}

/** Opaque liquid fill (kept non-transmissive so it appears behind the transmissive glass vessel). */
function fillMat(m: MaterialSpec, glow: THREE.Color | null, gi: number): MatFn {
  const kind = classifyMaterial(m)
  const c = col(m)
  const emissive = glow ?? c.clone().multiplyScalar(0.25)
  const ei = glow ? 0.3 + gi * 1.8 : 0.3
  if (kind === 'solid' || kind === 'gel') {
    const rough = THREE.MathUtils.clamp(0.3 + 0.7 * m.surface_roughness, 0.2, 1)
    return () => <meshPhysicalMaterial color={c} roughness={rough} sheen={0.6} sheenColor={c.clone().lerp(new THREE.Color('#fff'), 0.5)} sheenRoughness={0.5}
      emissive={glow ?? new THREE.Color(0, 0, 0)} emissiveIntensity={glow ? ei : 0} clearcoat={0.3} />
  }
  return () => <meshPhysicalMaterial color={c.clone().multiplyScalar(0.8)} roughness={0.06} metalness={0} clearcoat={1} clearcoatRoughness={0.02}
    emissive={emissive} emissiveIntensity={ei} ior={m.refractive_index} envMapIntensity={1.5} />
}

function hullFn(color: string): MatFn { return () => <HullMat color={color} /> }

function dropletProfile(a: number, h: number) {
  const Rs = (a * a + h * h) / (2 * h)
  const phiMax = Math.atan2(a, Rs - h)
  const n = 40
  const pts: THREE.Vector2[] = []
  for (let i = 0; i <= n; i++) {
    const phi = (i / n) * phiMax
    pts.push(new THREE.Vector2(Math.max(Rs * Math.sin(phi), 0.0001), h - Rs + Rs * Math.cos(phi)))
  }
  pts[pts.length - 1].y = 0
  return pts
}

/** Open cylinder wall (outer or inner surface) + helpers for the petri dish. */
function Petri({ s, body, fill, vessel }: { s: Extract<SceneObject['shape'], { kind: 'petri_dish' }>; body: MatFn; fill: MatFn; vessel: MatFn }) {
  const ro = s.inner_radius_mm + s.wall_mm, ri = s.inner_radius_mm, h = s.height_mm, b = s.base_mm
  const fillH = Math.min(s.fill_height_mm, h - b)   // engine clamps fill to the cavity depth
  void body
  return (
    <group>
      {/* base disc */}
      <mesh position={[0, b / 2, 0]} castShadow receiveShadow><cylinderGeometry args={[ro, ro, b, 96]} />{vessel()}<GlassEdges /></mesh>
      {/* wall as a tube: outer + inner skins + rim */}
      <mesh position={[0, (h + b) / 2, 0]} castShadow><cylinderGeometry args={[ro, ro, h - b, 96, 1, true]} />{vessel()}</mesh>
      <mesh position={[0, (h + b) / 2, 0]}><cylinderGeometry args={[ri, ri, h - b, 96, 1, true]} /><meshPhysicalMaterial transmission={1} thickness={s.wall_mm} ior={1.52} roughness={0.03} side={THREE.BackSide} color="#dfeeff" specularIntensity={1} /></mesh>
      <mesh position={[0, h, 0]} rotation={[-Math.PI / 2, 0, 0]}><ringGeometry args={[ri, ro, 96]} />{vessel()}<GlassEdges /></mesh>
      {/* fill */}
      {fillH > 0 && (
        <mesh position={[0, b + fillH / 2, 0]} castShadow receiveShadow><cylinderGeometry args={[ri - 0.02, ri - 0.02, fillH, 96]} />{fill()}</mesh>
      )}
    </group>
  )
}

function Cuvette({ s, fill, vessel }: { s: Extract<SceneObject['shape'], { kind: 'cuvette' }>; fill: MatFn; vessel: MatFn }) {
  const [ix, iy, iz] = s.inner_mm, w = s.wall_mm
  const ox = ix + 2 * w, oz = iz + 2 * w
  const fillH = Math.min(s.fill_height_mm, iy)   // engine clamps fill to the cavity height
  return (
    <group>
      <mesh position={[0, w / 2, 0]} castShadow receiveShadow><boxGeometry args={[ox, w, oz]} />{vessel()}<GlassEdges /></mesh>
      {[-1, 1].map((sx) => (
        <mesh key={`x${sx}`} position={[sx * (ix / 2 + w / 2), w + iy / 2, 0]} castShadow><boxGeometry args={[w, iy, oz]} />{vessel()}<GlassEdges /></mesh>
      ))}
      {[-1, 1].map((sz) => (
        <mesh key={`z${sz}`} position={[0, w + iy / 2, sz * (iz / 2 + w / 2)]} castShadow><boxGeometry args={[ix, iy, w]} />{vessel()}<GlassEdges /></mesh>
      ))}
      {fillH > 0 && (
        <mesh position={[0, w + fillH / 2, 0]} castShadow><boxGeometry args={[ix - 0.04, fillH, iz - 0.04]} />{fill()}</mesh>
      )}
    </group>
  )
}

function Shape({ obj, body, fill, vessel, block }: { obj: SceneObject; body: MatFn; fill: MatFn; vessel: MatFn; block?: ReactElement | null }) {
  const s = obj.shape
  const drop = useMemo(() => (s.kind === 'droplet' ? new THREE.LatheGeometry(dropletProfile(s.base_radius_mm, s.height_mm), 64) : null),
    [s.kind === 'droplet' ? s.base_radius_mm : 0, s.kind === 'droplet' ? s.height_mm : 0]) // eslint-disable-line react-hooks/exhaustive-deps
  switch (s.kind) {
    case 'box': return <mesh castShadow receiveShadow><boxGeometry args={s.size_mm} />{body()}</mesh>
    case 'sphere': return <mesh castShadow receiveShadow><sphereGeometry args={[s.radius_mm, 64, 40]} />{body()}</mesh>
    case 'cylinder': return <mesh castShadow receiveShadow><cylinderGeometry args={[s.radius_mm, s.radius_mm, s.height_mm, 64]} />{body()}</mesh>
    case 'droplet': return (
      <group>
        <mesh geometry={drop!} castShadow receiveShadow>{body()}</mesh>
        <mesh rotation={[Math.PI / 2, 0, 0]} position={[0, 0.001, 0]}><circleGeometry args={[s.base_radius_mm, 48]} />{body()}</mesh>
      </group>
    )
    case 'petri_dish': return <Petri s={s} body={body} fill={fill} vessel={vessel} />
    case 'cuvette': return <Cuvette s={s} fill={fill} vessel={vessel} />
    case 'layered_block': return block ?? null
  }
}

/** Sample geometry that matches the physics shapes exactly (mm). Glass vessels use a transmissive physical material
 *  (ior 1.52); liquids/gels/turbid solids get class-appropriate PBR from the material's optics; `glowColor` drives an
 *  emissive fluorescence glow (bloom picks it up). */
export function SampleMesh({ object, material, vesselMaterial, selected, hovered, glowColor, glowIntensity = 0.6, layerMaterials, layerGlows, section, sectionLabels, lightField }: SampleMeshProps) {
  const t = object.transform
  const glow = useMemo(() => (glowColor ? new THREE.Color(glowColor) : null), [glowColor])
  const s = object.shape
  const thick = s.kind === 'sphere' ? s.radius_mm * 2 : s.kind === 'droplet' ? s.height_mm : s.kind === 'box' ? Math.min(...s.size_mm) : s.kind === 'cylinder' ? s.radius_mm : 4
  const hull = hullColor(selected, hovered)
  const mats = useMemo(() => {
    const vm = vesselMaterial ?? ({ ...material, id: 'glass', display_color: '#dbeaff', scattering: { ...material.scattering, model: 'none' }, aqueous: false, refractive_index: 1.52 } as MaterialSpec)
    return {
      body: bodyMat(material, glow, glowIntensity, thick),
      fill: fillMat(material, glow, glowIntensity),
      vessel: glassMat(vm, 1.5),
    }
  }, [material, vesselMaterial, glow, glowIntensity, thick])
  if (!object.visible) return null
  const block = s.kind === 'layered_block' && layerMaterials?.length === s.layers.length
    ? <LayeredBlockMesh shape={s} materials={layerMaterials} glows={layerGlows} section={section} sectionLabels={sectionLabels} lightField={lightField} selected={selected} hovered={hovered} />
    : null
  const rot = t.rotation_deg.map((d) => (d * Math.PI) / 180) as [number, number, number]
  return (
    <group position={t.position_mm} rotation={rot}>
      <Shape obj={object} {...mats} block={block} />
      {hull && s.kind !== 'layered_block' && <group scale={1.04}><Shape obj={object} body={hullFn(hull)} fill={hullFn(hull)} vessel={hullFn(hull)} /></group>}
    </group>
  )
}
