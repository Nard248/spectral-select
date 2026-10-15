import { useEffect, useMemo } from 'react'
import * as THREE from 'three'
import { Line } from '@react-three/drei'
import type { LayeredBlockShape, MaterialSpec } from '../generated/experiment'
import type { BlockSection, LayerGlow } from './types'
import { buildBlockMesh, buildSectionStrips, cutRegion, cutSide, type Patch, type Side } from './layers/blockGeometry'
import { layerThicknesses } from './layers/profile'
import { layerAlpha, layerLook, type LayerLook } from './layers/look'
import { BlockAnnotations } from './layers/BlockAnnotations'
import { InclusionMeshes, type InclusionItem } from './layers/InclusionMeshes'
import { hullColor } from './util/parts'

export interface LayeredBlockMeshProps {
  shape: LayeredBlockShape
  /** one material per layer (bottom → top) */
  materials: MaterialSpec[]
  glows?: (LayerGlow | null)[]
  section?: BlockSection | null
  sectionLabels?: boolean
  /** colour-mapped light-field texture painted on the section face (rows stacked per layer, see lightfield.ts) */
  lightField?: THREE.Texture | null
  /** inclusions (sphere / ellipsoid / cylinder / box) with their materials, drawn translucent inside the block */
  inclusions?: InclusionItem[]
  /** index of the inclusion to emphasise (hover in the editor) */
  highlightInclusion?: number | null
  selected?: boolean
  hovered?: boolean
}

const SIDES: Side[] = ['-x', '+x', '-z', '+z']

/** Patch → BufferGeometry with per-vertex RGBA: the alpha is the physical see-through opacity of the layer slab at that
 *  vertex (a wedge fades smoothly to transparent as it thins), boosted where the layer fluoresces. */
function toGeometry(p: Patch, look: LayerLook, glow: number, wall: boolean, ghost = 1): THREE.BufferGeometry {
  const g = new THREE.BufferGeometry()
  const n = p.thickness.length
  const col = new Float32Array(n * 4)
  for (let i = 0; i < n; i++) {
    let a = layerAlpha(look.kappa, p.thickness[i])
    if (wall) a = 0.38 + 0.6 * a
    a = Math.min(0.97, Math.max(a, glow > 0 ? 0.28 + 0.5 * glow : 0.05)) * ghost
    col[4 * i] = look.color.r; col[4 * i + 1] = look.color.g; col[4 * i + 2] = look.color.b; col[4 * i + 3] = a
  }
  g.setAttribute('position', new THREE.BufferAttribute(p.positions, 3))
  g.setAttribute('normal', new THREE.BufferAttribute(p.normals, 3))
  g.setAttribute('color', new THREE.BufferAttribute(col, 4))
  g.setIndex(new THREE.BufferAttribute(p.indices, 1))
  g.computeBoundingSphere()
  return g
}
/** Fluorescence glow of layer `i` as an additive overlay on its top face. The emission is attenuated by the layers above it
 *  (T = exp(−Σ κ·t) per vertex), so a buried fluorophore glows brightly where the cover is thin and fades under the thick end. */
function glowGeometry(p: Patch, shape: LayeredBlockShape, i: number, looks: LayerLook[], glow: LayerGlow): THREE.BufferGeometry {
  const g = new THREE.BufferGeometry()
  const n = p.thickness.length, col = new Float32Array(n * 3)
  const c = new THREE.Color(glow.color ?? '#ffffff')
  const gain = 0.35 + 1.6 * glow.intensity
  for (let v = 0; v < n; v++) {
    const x = p.positions[3 * v], z = p.positions[3 * v + 2]
    const ts = layerThicknesses(shape, x, z)
    let tau = 0
    for (let j = i + 1; j < ts.length; j++) tau += (looks[j]?.kappa ?? 0.3) * ts[j]
    const k = gain * Math.exp(-tau)
    col[3 * v] = c.r * k; col[3 * v + 1] = c.g * k; col[3 * v + 2] = c.b * k
  }
  g.setAttribute('position', new THREE.BufferAttribute(p.positions, 3))
  g.setAttribute('color', new THREE.BufferAttribute(col, 3))
  g.setIndex(new THREE.BufferAttribute(p.indices, 1))
  return g
}
const patchGeometry = (p: Patch) => {
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.BufferAttribute(p.positions, 3))
  g.setAttribute('normal', new THREE.BufferAttribute(p.normals, 3))
  g.setIndex(new THREE.BufferAttribute(p.indices, 1))
  return g
}

/** The exact layered mesh of a `layered_block` (local frame: footprint centred on x/z, bottom on y = 0, every layer
 *  interface a plane). Layers are translucent slabs whose opacity follows their optical thickness; fluorescent layers glow
 *  under the active excitation; a cutaway shows the stack in cross-section with a capped, banded section face. */
export function LayeredBlockMesh({ shape, materials, glows, section, sectionLabels, lightField, inclusions, highlightInclusion, selected, hovered }: LayeredBlockMeshProps) {
  const region = useMemo(() => (section ? cutRegion(shape.footprint_mm, section.axis, section.position_mm, section.keep) : undefined), [shape.footprint_mm, section])
  const mesh = useMemo(() => buildBlockMesh(shape, region), [shape, region])
  const looks = useMemo(() => materials.map(layerLook), [materials])
  const cut: Side | null = section ? cutSide(section.axis, section.keep) : null
  const ghosted = !!(section && lightField)   // the light field is the star: the block itself turns to ghost glass

  const parts = useMemo(() => shape.layers.map((_, i) => {
    const look = looks[i] ?? layerLook(materials[0])
    const glow = glows?.[i]?.color ? glows[i]!.intensity : 0
    const lp = mesh.layers[i]
    return {
      look, glow,
      top: toGeometry(lp.top, look, 0, false, ghosted ? 0.3 : 1),
      glowTop: glows?.[i]?.color && glow > 0.002 ? glowGeometry(lp.top, shape, i, looks, glows[i]!) : null,
      bottom: lp.bottom ? toGeometry(lp.bottom, look, 0, false) : null,
      walls: SIDES.filter((s) => s !== cut && lp.sides[s]).map((s) => ({ side: s, geom: toGeometry(lp.sides[s]!, look, glow, true, ghosted ? 0.4 : 1) })),
      cut: cut && lp.sides[cut] ? patchGeometry(lp.sides[cut]!) : null,
    }
  }), [mesh, looks, glows, cut, shape, materials, ghosted])
  const proxy = useMemo(() => {
    // depth-only silhouette that casts the block's shadow (the translucent layers do not write depth)
    const g = new THREE.BufferGeometry(); const ps: Patch[] = []
    mesh.layers.forEach((l) => { ps.push(l.top); if (l.bottom) ps.push(l.bottom); for (const s of SIDES) if (l.sides[s]) ps.push(l.sides[s]!) })
    const pos: number[] = [], idx: number[] = []
    for (const p of ps) { const o = pos.length / 3; pos.push(...p.positions); for (const k of p.indices) idx.push(k + o) }
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); g.setIndex(idx)
    return g
  }, [mesh])
  const cutAt = section ? Math.min(Math.max(section.position_mm, -(section.axis === 'x' ? shape.footprint_mm[0] : shape.footprint_mm[1]) / 2), (section.axis === 'x' ? shape.footprint_mm[0] : shape.footprint_mm[1]) / 2) : 0
  const strips = useMemo(() => (section && lightField ? buildSectionStrips(shape, section.axis, cutAt, section.keep, 64, 10, 0.5 / 160) : null),
    [shape, section, cutAt, lightField])
  const stripGeoms = useMemo(() => strips?.map((s) => {
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(s.positions, 3))
    g.setAttribute('uv', new THREE.BufferAttribute(s.uvs, 2))
    g.setIndex(new THREE.BufferAttribute(s.indices, 1))
    const n = s.normal; g.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(Array.from({ length: s.positions.length / 3 }, () => n).flat()), 3))
    return g
  }) ?? null, [strips])
  const contourGeom = useMemo(() => { const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(mesh.contours, 3)); return g }, [mesh])
  const edgeGeom = useMemo(() => { const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(mesh.edges, 3)); return g }, [mesh])

  useEffect(() => () => { parts.forEach((p) => { p.top.dispose(); p.glowTop?.dispose(); p.bottom?.dispose(); p.walls.forEach((w) => w.geom.dispose()); p.cut?.dispose() }) }, [parts])
  useEffect(() => () => proxy.dispose(), [proxy])
  useEffect(() => () => { stripGeoms?.forEach((g) => g.dispose()) }, [stripGeoms])
  useEffect(() => () => edgeGeom.dispose(), [edgeGeom])
  useEffect(() => () => contourGeom.dispose(), [contourGeom])

  const hull = hullColor(selected, hovered)
  const hullPts = useMemo(() => {
    const out: [number, number, number][] = []
    for (let i = 0; i < mesh.hull.length; i += 3) out.push([mesh.hull[i], mesh.hull[i + 1], mesh.hull[i + 2]])
    return out
  }, [mesh])

  return (
    <group>
      {/* shadow caster (no colour, no depth) */}
      <mesh geometry={proxy} castShadow><meshBasicMaterial colorWrite={false} depthWrite={false} side={THREE.DoubleSide} /></mesh>

      {parts.map((p, i) => {
        const em = glows?.[i]?.color ? new THREE.Color(glows[i]!.color!) : null
        const ei = em ? 0.12 + (glows?.[i]?.intensity ?? 0) * 1.1 : 0
        const base = { vertexColors: true, transparent: true, depthWrite: false, roughness: p.look.roughness, metalness: 0, sheen: p.look.sheen * 0.3, sheenColor: p.look.color.clone().lerp(new THREE.Color('#ffffff'), 0.4), sheenRoughness: 0.6, clearcoat: p.look.clearcoat, clearcoatRoughness: 0.12, envMapIntensity: p.look.clear ? 1.1 : 0.32, ior: materials[i]?.refractive_index ?? 1.4, emissive: em ?? new THREE.Color(0, 0, 0), emissiveIntensity: ei } as const
        return (
          <group key={i} renderOrder={10 + i}>
            <mesh geometry={p.top} renderOrder={10 + i}><meshPhysicalMaterial {...base} emissive={new THREE.Color(0, 0, 0)} emissiveIntensity={0} /></mesh>
            {p.glowTop && <mesh geometry={p.glowTop} renderOrder={30 + i}><meshBasicMaterial vertexColors transparent blending={THREE.AdditiveBlending} depthWrite={false} toneMapped={false} polygonOffset polygonOffsetFactor={-1} polygonOffsetUnits={-1} /></mesh>}
            {p.bottom && <mesh geometry={p.bottom} renderOrder={10 + i}><meshPhysicalMaterial {...base} side={THREE.DoubleSide} /></mesh>}
            {p.walls.map((w) => <mesh key={w.side} geometry={w.geom} renderOrder={10 + i}><meshPhysicalMaterial {...base} /></mesh>)}
            {/* capped section face: opaque layer bands (or the light-field strips) */}
            {p.cut && !stripGeoms && (
              <mesh geometry={p.cut} renderOrder={40 + i}>
                <meshStandardMaterial color={p.look.color} roughness={0.55 - 0.2 * (p.look.clear ? 1 : 0)} metalness={0} emissive={em ?? p.look.color}
                  emissiveIntensity={em ? 0.16 + (glows?.[i]?.intensity ?? 0) * 0.45 : 0.12} polygonOffsetFactor={-1} polygonOffsetUnits={-1} />
              </mesh>
            )}
          </group>
        )
      })}
      {stripGeoms && lightField && stripGeoms.map((g, i) => (
        <mesh key={`lf${i}`} geometry={g} renderOrder={45}>
          <meshBasicMaterial map={lightField} toneMapped={false} polygonOffset polygonOffsetFactor={-2} polygonOffsetUnits={-2} />
        </mesh>
      ))}

      {inclusions && inclusions.length > 0 && <InclusionMeshes shape={shape} items={inclusions} section={section} cutAt={cutAt} labels={sectionLabels} highlight={highlightInclusion} />}

      {/* interface lines */}
      <lineSegments geometry={edgeGeom} renderOrder={60}>
        <lineBasicMaterial color={stripGeoms ? '#ffffff' : '#e9f6ff'} transparent opacity={stripGeoms ? 0.9 : 0.55} toneMapped={false} depthWrite={false} />
      </lineSegments>
      {mesh.contours.length > 0 && <lineSegments geometry={contourGeom} renderOrder={59}><lineBasicMaterial color="#6f86a0" transparent opacity={ghosted ? 0.25 : 0.55} toneMapped={false} depthWrite={false} /></lineSegments>}
      {hull && <Line points={hullPts} segments color={hull} lineWidth={2} transparent opacity={0.95} toneMapped={false} depthTest={false} renderOrder={70} />}

      {section && sectionLabels && <BlockAnnotations shape={shape} section={section} region={mesh.region} looks={looks} />}
    </group>
  )
}
