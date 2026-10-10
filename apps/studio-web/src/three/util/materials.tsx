import * as THREE from 'three'
import { brushedRoughness, knurlBump } from './textures'

/** Dark anodised aluminium (camera body, source heads). */
export function Anodized({ color = '#3a404c', rough = 0.36 }: { color?: string; rough?: number }) {
  return <meshPhysicalMaterial color={color} metalness={0.82} roughness={rough} roughnessMap={brushedRoughness()}
    clearcoat={0.25} clearcoatRoughness={0.35} envMapIntensity={1.15} />
}
/** Bright machined / bead-blasted aluminium. */
export function Alu({ color = '#c8ced6', rough = 0.38 }: { color?: string; rough?: number }) {
  return <meshPhysicalMaterial color={color} metalness={1} roughness={rough} roughnessMap={brushedRoughness()} envMapIntensity={1.3} />
}
export function Chrome({ color = '#f0f3f8' }: { color?: string }) {
  return <meshPhysicalMaterial color={color} metalness={1} roughness={0.12} envMapIntensity={1.6} />
}
export function Gold() {
  return <meshPhysicalMaterial color="#e2b24a" metalness={1} roughness={0.22} envMapIntensity={1.4} />
}
export function Rubber({ color = '#0c0d0f' }: { color?: string }) {
  return <meshStandardMaterial color={color} roughness={0.85} metalness={0} />
}
export function Plastic({ color = '#202329', rough = 0.5 }: { color?: string; rough?: number }) {
  return <meshPhysicalMaterial color={color} roughness={rough} metalness={0.05} clearcoat={0.4} clearcoatRoughness={0.4} />
}
/** Knurled ring material: bump stripes wrapping the circumference (set the repeat on the texture clone via `map`). */
export function KnurlMetal({ color = '#23262d', map }: { color?: string; map?: THREE.Texture }) {
  return <meshPhysicalMaterial color={map ? '#ffffff' : color} map={map} metalness={0.8} roughness={0.42} bumpMap={knurlBump()}
    bumpScale={1.6} envMapIntensity={1.1} />
}
/** Emissive that survives tone mapping unchanged enough to bloom. */
export function Glow({ color, intensity = 2 }: { color: THREE.ColorRepresentation; intensity?: number }) {
  return <meshBasicMaterial color={new THREE.Color(color).multiplyScalar(intensity)} toneMapped={false} />
}
