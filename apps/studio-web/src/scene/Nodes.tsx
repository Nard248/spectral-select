import { useMemo } from 'react'
import type { Experiment, Inclusion, MaterialSpec } from '../generated/experiment'
import type { LayerGlow } from '../three/types'
import { useLayerView } from '../state/layerView'
import { useLightField } from './lightfield'
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
    const layerGlows = o.shape.kind === 'layered_block' ? o.shape.layers.map((l) => objectGlow(exp, o, mats[l.material_id], src, fl, nm, power)) : undefined
    return <ObjectNode key={o.id} id={o.id} glowColor={g.color} glowIntensity={g.intensity} layerGlows={layerGlows} />
  })}</>
}

/** Stable identity for per-layer glows so the block geometry is not rebuilt on unrelated re-renders. */
function useStableGlows(glows: LayerGlow[] | undefined): LayerGlow[] | undefined {
  const key = glows ? glows.map((g) => `${g.color ?? ''}:${g.intensity.toFixed(3)}`).join('|') : ''
  // eslint-disable-next-line react-hooks/exhaustive-deps
  return useMemo(() => glows, [key])
}
function useLayerMaterials(exp: Experiment, obj: Experiment['objects'][number]): MaterialSpec[] | undefined {
  return useMemo(() => {
    if (obj.shape.kind !== 'layered_block') return undefined
    const m = materialMap(exp)
    const out = obj.shape.layers.map((l) => m[l.material_id])
    return out.every(Boolean) ? out : undefined
  }, [exp, obj])
}

function useInclusionItems(exp: Experiment, obj: Experiment['objects'][number]) {
  return useMemo(() => {
    if (obj.shape.kind !== 'layered_block') return undefined
    const m = materialMap(exp)
    return (obj.shape.inclusions as Inclusion[]).map((inc) => ({ inc, material: m[inc.material_id] }))
  }, [exp.materials, obj.shape])  // eslint-disable-line react-hooks/exhaustive-deps
}

function ObjectNode({ id, glowColor, glowIntensity, layerGlows }: { id: string; glowColor: string | null; glowIntensity: number; layerGlows?: LayerGlow[] }) {
  const exp = useStudio((s) => s.exp)
  const obj = exp.objects.find((o) => o.id === id)!
  const selected = useStudio((s) => s.selection?.kind === 'object' && s.selection.id === id)
  const hovered = useStudio((s) => s.ui.hoverId === id)
  const h = useSelectHandlers({ kind: 'object', id })
  const mats = materialMap(exp)
  const material = mats[obj.material_id]
  const layerMaterials = useLayerMaterials(exp, obj)
  const glows = useStableGlows(layerGlows)
  const inclusions = useInclusionItems(exp, obj)
  const highlightInclusion = useLayerView((st) => st.incSel[id] ?? null)
  const sectionState = useLayerView((s) => s.sections[id])
  const lfOn = useLayerView((s) => s.lf.on)
  const isBlock = obj.shape.kind === 'layered_block'
  const section = isBlock && sectionState?.on ? { axis: sectionState.axis, position_mm: sectionState.position_mm, keep: sectionState.keep } : null
  const cut = useMemo(() => (section ? { axis: section.axis, position_mm: section.position_mm } : null), [section?.axis, section?.position_mm]) // eslint-disable-line react-hooks/exhaustive-deps
  const demoMus = useMemo(() => (layerMaterials ? { mus: layerMaterials.map((m) => (m.scattering.model === 'none' ? 0 : m.scattering.mus_prime_500_per_mm)), mua: layerMaterials.map(() => 0.02) } : undefined), [layerMaterials])
  const lf = useLightField(id, obj.shape.kind === 'layered_block' ? obj.shape : null, isBlock && lfOn && !!section, cut, demoMus)
  // a block draws its own layer materials, so a dangling `material_id` (ignored by the engine for blocks) must not hide it
  if (!material && !(isBlock && layerMaterials)) return null
  return (
    <group {...h}>
      <SampleMesh object={obj} material={material ?? layerMaterials![0]} vesselMaterial={obj.vessel_material_id ? mats[obj.vessel_material_id] : undefined}
        selected={selected} hovered={hovered} glowColor={glowColor} glowIntensity={glowIntensity}
        layerMaterials={layerMaterials} layerGlows={glows} section={section} sectionLabels={!!sectionState?.labels} lightField={lf.texture} inclusions={inclusions} highlightInclusion={highlightInclusion} />
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
