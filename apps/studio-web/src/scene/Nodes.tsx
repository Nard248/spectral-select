import { useMemo } from 'react'
import type { ThreeEvent } from '@react-three/fiber'
import { useStudio, activeSource, type Selection } from '../state/store'
import { fluorophoreMap, materialMap } from '../state/presets'
import { objectGlow } from '../lib/glow'
import { SampleMesh } from '../three/SampleMesh'
import { LightSourceModel } from '../three/hardware/LightSourceModel'
import { CameraModel } from '../three/hardware/CameraModel'
import { useActiveNm, useAcquiring } from './useExcitation'

/** Click that ignores orbit drags and look-through. */
export function useSelectHandlers(sel: Selection) {
  const select = useStudio((s) => s.select)
  const setUi = useStudio((s) => s.setUi)
  return {
    onClick: (e: ThreeEvent<MouseEvent>) => {
      if (useStudio.getState().ui.lookThrough || e.delta > 4) return
      e.stopPropagation(); select(sel)
    },
    onPointerOver: (e: ThreeEvent<PointerEvent>) => { e.stopPropagation(); setUi({ hoverId: sel.id }); document.body.style.cursor = 'pointer' },
    onPointerOut: () => { setUi({ hoverId: null }); document.body.style.cursor = '' },
  }
}

export function ObjectNodes() {
  const exp = useStudio((s) => s.exp)
  const nm = useActiveNm()
  const fl = useMemo(() => fluorophoreMap(exp), [exp])
  const mats = useMemo(() => materialMap(exp), [exp])
  const src = activeSource(exp)
  const power = src?.sequence.find((x) => x.enabled && x.wavelength_nm === nm)?.power_mW ?? 3
  return <>{exp.objects.filter((o) => o.visible).map((o) => {
    const g = objectGlow(exp, o, mats[o.material_id], src, fl, nm, power)
    return <ObjectNode key={o.id} id={o.id} glowColor={g.color} glowIntensity={g.intensity} />
  })}</>
}

function ObjectNode({ id, glowColor, glowIntensity }: { id: string; glowColor: string | null; glowIntensity: number }) {
  const exp = useStudio((s) => s.exp)
  const obj = exp.objects.find((o) => o.id === id)!
  const selected = useStudio((s) => s.selection?.kind === 'object' && s.selection.id === id)
  const hovered = useStudio((s) => s.ui.hoverId === id)
  const h = useSelectHandlers({ kind: 'object', id })
  const mats = materialMap(exp)
  const material = mats[obj.material_id]
  if (!material) return null
  return (
    <group {...h}>
      <SampleMesh object={obj} material={material} vesselMaterial={obj.vessel_material_id ? mats[obj.vessel_material_id] : undefined}
        selected={selected} hovered={hovered} glowColor={glowColor} glowIntensity={glowIntensity} />
    </group>
  )
}

export function SourceNodes() {
  const exp = useStudio((s) => s.exp)
  const nm = useActiveNm()
  const showBeam = useStudio((s) => s.ui.showBeam)
  return <>{exp.sources.map((s) => <SourceNode key={s.id} id={s.id} active={s.id === exp.active_source_id || exp.sources.length === 1} nm={nm} showBeam={showBeam} />)}</>
}
function SourceNode({ id, active, nm, showBeam }: { id: string; active: boolean; nm: number | null; showBeam: boolean }) {
  const src = useStudio((s) => s.exp.sources.find((x) => x.id === id))!
  const selected = useStudio((s) => s.selection?.kind === 'source' && s.selection.id === id)
  const hovered = useStudio((s) => s.ui.hoverId === id)
  const h = useSelectHandlers({ kind: 'source', id })
  return <group {...h}><LightSourceModel source={src} activeWavelengthNm={active ? nm : null} showBeam={showBeam && active} selected={selected} hovered={hovered} /></group>
}

export function CameraNodes() {
  const exp = useStudio((s) => s.exp)
  const showFrustum = useStudio((s) => s.ui.showFrustum)
  const acquiring = useAcquiring()
  const lookThrough = useStudio((s) => s.ui.lookThrough)
  return <>{exp.cameras.filter((c) => !(lookThrough && (c.id === exp.active_camera_id || exp.cameras.length === 1))).map((c) => <CameraNode key={c.id} id={c.id} active={c.id === exp.active_camera_id || exp.cameras.length === 1} showFrustum={showFrustum} acquiring={acquiring} />)}</>
}
function CameraNode({ id, active, showFrustum, acquiring }: { id: string; active: boolean; showFrustum: boolean; acquiring: boolean }) {
  const cam = useStudio((s) => s.exp.cameras.find((x) => x.id === id))!
  const selected = useStudio((s) => s.selection?.kind === 'camera' && s.selection.id === id)
  const hovered = useStudio((s) => s.ui.hoverId === id)
  const h = useSelectHandlers({ kind: 'camera', id })
  // the active camera's body & frustum would obstruct the HSI-eye views
  return <group {...h} userData={active ? { hideInPip: true } : undefined}><CameraModel camera={cam} showFrustum={showFrustum && (active || selected)} acquiring={active && acquiring} selected={selected} hovered={hovered} /></group>
}
