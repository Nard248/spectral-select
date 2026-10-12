import { useEffect, type ReactElement } from 'react'
import { useThree } from '@react-three/fiber'
import { Environment, Lightformer, ContactShadows } from '@react-three/drei'
import { EffectComposer, Bloom, Vignette, SMAA, N8AO, ToneMapping } from '@react-three/postprocessing'
import { ToneMappingMode } from 'postprocessing'
import * as THREE from 'three'

export type Quality = 'high' | 'low'
export interface StudioEnvironmentProps {
  /** 'low' for weak projectors / integrated GPUs: smaller shadow maps, no contact-shadow refresh */
  quality?: Quality
  /** draw the gradient backdrop sphere (default true) */
  backdrop?: boolean
  /** half-extent of the area (mm) covered by shadows (default 450) */
  shadowExtent?: number
}

const BACK_VERT = 'varying vec3 vP; void main(){ vP = position; gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.0); }'
const BACK_FRAG = /* glsl */ `
varying vec3 vP;
void main(){
  vec3 n = normalize(vP);
  float h = n.y;
  vec3 top = vec3(0.010,0.016,0.032), hor = vec3(0.060,0.085,0.120), bot = vec3(0.008,0.010,0.016);
  float t = smoothstep(-0.55, 0.85, h);
  vec3 c = t < 0.4 ? mix(bot, hor, smoothstep(0.0, 0.4, t)) : mix(hor, top, smoothstep(0.4, 1.0, t));
  c += vec3(0.02,0.045,0.07) * exp(-pow((n.x*0.6 + n.z*0.8 + 0.2)*1.5, 2.0)) * smoothstep(-0.2, 0.3, h) * (1.0 - smoothstep(0.3, 0.9, h));
  gl_FragColor = vec4(c, 1.0);
}`

/** Studio lighting: Lightformer softboxes (key/fill/rim/top) for reflections, a shadow-casting key light, contact
 *  shadows, dark gradient backdrop and depth fog. Fully offline (no HDRI). Units are mm; Y up, table at y = 0. */
export function StudioEnvironment({ quality = 'high', backdrop = true, shadowExtent = 450 }: StudioEnvironmentProps) {
  const gl = useThree((s) => s.gl)
  useEffect(() => {
    gl.shadowMap.enabled = true
    gl.shadowMap.type = THREE.PCFShadowMap
    gl.shadowMap.needsUpdate = true
  }, [gl])
  const hi = quality === 'high'
  return (
    <>
      <color attach="background" args={['#06080c']} />
      <fog attach="fog" args={['#07090d', 1400, 5200]} />
      {backdrop && (
        <mesh renderOrder={-1000} frustumCulled={false}>
          <sphereGeometry args={[3000, 48, 24]} />
          <shaderMaterial vertexShader={BACK_VERT} fragmentShader={BACK_FRAG} side={THREE.BackSide} depthWrite={false} fog={false} toneMapped={false} />
        </mesh>
      )}
      <Environment resolution={hi ? 256 : 128} frames={1} environmentIntensity={1.5} far={2000}>
        {/* ambient dome so dark metals have something soft to reflect */}
        <mesh scale={1800}>
          <sphereGeometry args={[1, 24, 12]} />
          <meshBasicMaterial color={new THREE.Color('#3b4b66').multiplyScalar(0.9)} side={THREE.BackSide} toneMapped={false} />
        </mesh>
        {/* key softbox (warm), front-right high */}
        <Lightformer form="rect" intensity={7} color="#fff1e0" position={[420, 520, 380]} scale={[420, 280, 1]} target={[0, 0, 0]} />
        {/* fill (cool), left */}
        <Lightformer form="rect" intensity={2.4} color="#bcd4ff" position={[-620, 260, 260]} scale={[360, 300, 1]} target={[0, 60, 0]} />
        {/* rim strips behind */}
        <Lightformer form="rect" intensity={6} color="#ffffff" position={[0, 300, -700]} scale={[1000, 70, 1]} target={[0, 100, 0]} />
        <Lightformer form="rect" intensity={4} color="#9fd0ff" position={[-520, 180, -480]} scale={[60, 600, 1]} target={[0, 100, 0]} />
        <Lightformer form="rect" intensity={4} color="#ffd9b0" position={[560, 180, -420]} scale={[60, 600, 1]} target={[0, 100, 0]} />
        {/* overhead panel */}
        <Lightformer form="rect" intensity={1.6} color="#ffffff" position={[0, 900, 0]} rotation={[Math.PI / 2, 0, 0]} scale={[900, 900, 1]} />
        {/* dark floor bounce */}
        <Lightformer form="rect" intensity={0.25} color="#233044" position={[0, -500, 0]} rotation={[-Math.PI / 2, 0, 0]} scale={[1400, 1400, 1]} />
      </Environment>
      <hemisphereLight args={['#aac4ff', '#1a2030', 0.5]} />
      <directionalLight
        position={[360, 700, 320]} intensity={2.0} color="#fff2e2" castShadow
        shadow-mapSize={hi ? [2048, 2048] : [1024, 1024]} shadow-bias={-0.0004} shadow-normalBias={0.6} shadow-radius={hi ? 4 : 2}
        shadow-camera-left={-shadowExtent} shadow-camera-right={shadowExtent} shadow-camera-top={shadowExtent} shadow-camera-bottom={-shadowExtent}
        shadow-camera-near={50} shadow-camera-far={2200}
      />
      <directionalLight position={[-500, 300, -300]} intensity={1.1} color="#9ec4ff" />
      <ContactShadows position={[0, 0.15, 0]} scale={shadowExtent * 1.6} blur={hi ? 2.4 : 3.5} far={120} opacity={0.6} resolution={hi ? 1024 : 256}
        frames={hi ? Infinity : 1} color="#000000" />
    </>
  )
}

/** Post-processing: N8AO (high only), bloom for beams/glows, ACES filmic tone mapping, vignette, SMAA (high only). */
export function Effects({ quality = 'high', bloom = 1 }: { quality?: Quality; bloom?: number }) {
  const hi = quality === 'high'
  const fx: ReactElement[] = []
  if (hi) fx.push(<N8AO key="ao" aoRadius={14} distanceFalloff={0.5} intensity={2.4} quality="medium" color="black" />)
  fx.push(<Bloom key="bloom" mipmapBlur intensity={0.85 * bloom} luminanceThreshold={1.0} luminanceSmoothing={0.3} radius={hi ? 0.75 : 0.6} levels={hi ? 8 : 5} />)
  fx.push(<ToneMapping key="tm" mode={ToneMappingMode.ACES_FILMIC} />)
  fx.push(<Vignette key="vig" offset={0.32} darkness={0.45} />)
  if (hi) fx.push(<SMAA key="smaa" />)
  return <EffectComposer multisampling={0} enableNormalPass={false}>{fx}</EffectComposer>
}
