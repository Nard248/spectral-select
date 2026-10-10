import { useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import type { LightSourceSpec } from '../../generated/experiment'
import { wavelengthToRGB } from '../../lib/color'
import { lookAtQuaternion } from '../pose'

const VERT = /* glsl */ `
varying vec3 vN; varying vec3 vView; varying vec3 vWorld; varying float vT; varying float vR; varying float vAng;
uniform float uLen;
void main(){
  vec4 mv = modelViewMatrix * vec4(position,1.0);
  vN = normalize(normalMatrix * normal);
  vView = -mv.xyz;
  vWorld = (modelMatrix * vec4(position,1.0)).xyz;
  vT = clamp(-position.z / uLen, 0.0, 1.0);
  vR = length(position.xy);
  vAng = atan(position.y, position.x);
  gl_Position = projectionMatrix * mv;
}`
const FRAG = /* glsl */ `
varying vec3 vN; varying vec3 vView; varying vec3 vWorld; varying float vT; varying float vR; varying float vAng;
uniform vec3 uColor; uniform float uTime; uniform float uFlat; uniform float uUV; uniform float uGain; uniform float uR0;
void main(){
  if (vWorld.y < 0.0) discard;
  vec3 N = normalize(vN); if(!gl_FrontFacing) N = -N;
  float ndv = abs(dot(N, normalize(vView)));         // = chord fraction through the cone at this silhouette offset
  float s2 = 1.0 - ndv*ndv;                           // (normalised radial offset)^2 of the sight line
  float prof = uFlat > 0.5 ? (1.0 - smoothstep(0.80, 1.0, s2)) : exp(-2.4*s2);
  float a = prof * pow(ndv, 0.85);
  float dil = clamp(uR0 / max(vR, 0.001), 0.0, 1.0);  // irradiance dilution as the cone widens
  a *= mix(1.0, dil, 0.55);
  a *= exp(-1.1*vT) * smoothstep(0.0, 0.03, vT) * (1.0 - smoothstep(0.86, 1.0, vT));
  a *= smoothstep(0.0, 7.0, vWorld.y);                // melt into the table
  if (uUV > 0.5) {
    float sh = 0.74 + 0.26*sin(vT*46.0 - uTime*7.0 + sin(vAng*3.0 + uTime)*2.0);
    sh *= 0.88 + 0.12*sin(vAng*9.0 + uTime*1.7);
    a *= sh;
  }
  gl_FragColor = vec4(uColor * uGain, a * 0.62);
}`
const FP_FRAG = /* glsl */ `
varying vec2 vUv; uniform vec3 uColor; uniform float uFlat; uniform float uUV; uniform float uTime; uniform float uGain;
void main(){
  vec2 p = vUv*2.0-1.0; float r2 = dot(p,p);
  if (r2 > 1.0) discard;
  float prof = uFlat > 0.5 ? (1.0 - smoothstep(0.78, 1.0, r2)) : exp(-3.0*r2) * (1.0 - smoothstep(0.85, 1.0, r2));
  float ring = smoothstep(0.80, 0.93, r2) * (1.0 - smoothstep(0.93, 1.0, r2)) * (uFlat > 0.5 ? 0.55 : 0.25);
  float a = prof*0.42 + ring;
  if (uUV > 0.5) a *= 0.85 + 0.15*sin(r2*30.0 - uTime*5.0);
  gl_FragColor = vec4(uColor * uGain, a);
}`

export interface BeamVolumeProps {
  source: LightSourceSpec
  wavelengthNm: number
  /** brightness multiplier (e.g. scale with relative power); default 1 */
  intensity?: number
  /** distance (mm) the cone is extended beyond the target; default 35 % of the source→target distance */
  extension?: number
}

/** Volumetric beam: a real cone of radius D/2 + z·tan α (aperture_diameter_mm, divergence_half_angle_deg) from the
 *  source toward `target_mm`, drawn additively with a gaussian / flat-top radial profile, plus a glowing footprint
 *  ellipse where the cone meets the table (y = 0). Rendered in WORLD space (give it no parent transform). */
export function BeamVolume({ source, wavelengthNm, intensity = 1, extension }: BeamVolumeProps) {
  const matRef = useRef<THREE.ShaderMaterial>(null)
  const fpRef = useRef<THREE.ShaderMaterial>(null)
  const { position_mm: p, target_mm: t, roll_deg, beam } = source
  const r0 = beam.aperture_diameter_mm / 2
  const tanA = Math.tan(THREE.MathUtils.degToRad(Math.max(beam.divergence_half_angle_deg, 0)))
  const dist = Math.max(new THREE.Vector3(...t).distanceTo(new THREE.Vector3(...p)), 1)
  const len = dist + (extension ?? 0.35 * dist)
  const q = useMemo(() => lookAtQuaternion(p, t, roll_deg), [p, t, roll_deg])
  const geo = useMemo(() => {
    const g = new THREE.CylinderGeometry(r0 + len * tanA, r0, len, 64, 1, true)
    g.rotateX(-Math.PI / 2)               // +Y → −Z : small end at the source (z=0), large end at z=−len
    g.translate(0, 0, -len / 2)
    return g
  }, [r0, tanA, len])
  const col = useMemo(() => new THREE.Color(...wavelengthToRGB(wavelengthNm)), [wavelengthNm])
  const flat = beam.profile === 'flat_top' ? 1 : 0
  const uv = wavelengthNm < 400 ? 1 : 0

  const gBeam = 1.5 * intensity * Math.min(1.3, Math.max(0.45, Math.pow(6 / Math.max(r0, 3), 0.35)))
  const gFp = (0.7 + 1.6 * Math.exp(-r0 / 8)) * intensity
  const uniforms = useMemo(() => ({
    uColor: { value: col.clone() }, uTime: { value: 0 }, uLen: { value: len }, uFlat: { value: flat }, uUV: { value: uv },
    uGain: { value: gBeam }, uR0: { value: Math.max(r0, 0.5) },
  }), []) // eslint-disable-line react-hooks/exhaustive-deps
  const fpUniforms = useMemo(() => ({
    uColor: { value: col.clone() }, uFlat: { value: flat }, uUV: { value: uv }, uTime: { value: 0 }, uGain: { value: gFp },
  }), []) // eslint-disable-line react-hooks/exhaustive-deps

  // footprint: cone axis ∩ table
  const fp = useMemo(() => {
    const dir = new THREE.Vector3(0, 0, -1).applyQuaternion(q)
    if (dir.y > -0.02 || p[1] <= 0) return null
    const t0 = -p[1] / dir.y
    const r = r0 + t0 * tanA
    const cosT = Math.abs(dir.y)
    return { pos: [p[0] + dir.x * t0, 0.35, p[2] + dir.z * t0] as [number, number, number],
      az: Math.atan2(dir.z, dir.x), a: Math.min(r / Math.max(cosT, 0.12), r * 8), b: r }
  }, [q, p, r0, tanA])

  useFrame(({ clock }) => {
    const m = matRef.current, f = fpRef.current
    const tt = clock.elapsedTime
    if (m) {
      m.uniforms.uTime.value = tt; m.uniforms.uColor.value.copy(col); m.uniforms.uLen.value = len
      m.uniforms.uFlat.value = flat; m.uniforms.uUV.value = uv; m.uniforms.uGain.value = gBeam; m.uniforms.uR0.value = Math.max(r0, 0.5)
    }
    if (f) {
      f.uniforms.uTime.value = tt; f.uniforms.uColor.value.copy(col); f.uniforms.uFlat.value = flat; f.uniforms.uUV.value = uv
      f.uniforms.uGain.value = gFp
    }
  })

  return (
    <group>
      <mesh geometry={geo} position={p} quaternion={q} renderOrder={10} frustumCulled={false}>
        <shaderMaterial ref={matRef} uniforms={uniforms} vertexShader={VERT} fragmentShader={FRAG} transparent depthWrite={false}
          blending={THREE.AdditiveBlending} side={THREE.DoubleSide} toneMapped={false} />
      </mesh>
      {fp && (
        <group position={fp.pos} rotation={[0, -fp.az, 0]}>
          <mesh rotation={[-Math.PI / 2, 0, 0]} scale={[fp.a, fp.b, 1]} renderOrder={9}>
            <circleGeometry args={[1, 64]} />
            <shaderMaterial ref={fpRef} uniforms={fpUniforms} vertexShader={'varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0);} '}
              fragmentShader={FP_FRAG} transparent depthWrite={false} blending={THREE.AdditiveBlending} toneMapped={false}
              polygonOffset polygonOffsetFactor={-2} polygonOffsetUnits={-2} />
          </mesh>
        </group>
      )}
    </group>
  )
}
