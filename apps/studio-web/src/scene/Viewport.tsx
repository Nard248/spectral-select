import { Suspense, useEffect, useRef, useState } from 'react'
import { Canvas } from '@react-three/fiber'
import { Move3d, Rotate3d, Magnet, Focus, Video, PictureInPicture2, Sun, ScanLine, Gauge, Slice } from 'lucide-react'
import { useStudio, activeCamera } from '../state/store'
import { useLayerView, targetBlockId, DEFAULT_SECTION } from '../state/layerView'
import { SectionCard, LightFieldLegend } from './SectionCard'
import { setCutaway } from './blockView'
import { SceneRoot } from './SceneRoot'
import { Pip } from './Pip'
import { viewApi, viewportEl } from './viewApi'
import { fitRect } from './RenderLoop'
import { cameraIntrinsics } from '../lib/camera'
import { Tip } from '../ui/controls'

function ToolBtn({ on, onClick, tip, kbd, children }: { on?: boolean; onClick: () => void; tip: string; kbd?: string; children: React.ReactNode }) {
  return <Tip text={kbd ? `${tip}  [${kbd}]` : tip}><button className={`btn ghost icon${on ? ' on' : ''}`} onClick={onClick} aria-label={tip}>{children}</button></Tip>
}

export function Viewport() {
  const ui = useStudio((s) => s.ui)
  const mode = useStudio((s) => s.mode)
  const setUi = useStudio((s) => s.setUi)
  const select = useStudio((s) => s.select)
  const addObject = useStudio((s) => s.addObject)
  const cam = useStudio((s) => activeCamera(s.exp))
  const blockId = useStudio((s) => targetBlockId(s.exp, s.selection))
  const cutOn = useLayerView((s) => (blockId ? (s.sections[blockId] ?? DEFAULT_SECTION).on : false))
  const host = useRef<HTMLDivElement>(null)
  const [size, setSize] = useState({ w: 800, h: 500 })

  useEffect(() => {
    const el = host.current; if (!el) return
    viewportEl.el = el
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }))
    ro.observe(el); setSize({ w: el.clientWidth, h: el.clientHeight })
    return () => { ro.disconnect(); viewportEl.el = null }
  }, [])

  const intr = cameraIntrinsics(cam)
  const lr = fitRect(size.w, size.h, intr.aspect)

  return (
    <div className="viewport" ref={host}
      onDragOver={(e) => { if (e.dataTransfer.types.includes('application/x-sf-shape')) { e.preventDefault(); e.dataTransfer.dropEffect = 'copy' } }}
      onDrop={(e) => {
        const key = e.dataTransfer.getData('application/x-sf-shape'); if (!key) return
        e.preventDefault()
        const p = viewApi.current?.pickTable(e.clientX, e.clientY)
        addObject(key, p ? [p[0], 0, p[2]] : undefined)
      }}>
      <div style={ui.lookThrough ? { position: 'absolute', left: lr.x, top: lr.y, width: lr.w, height: lr.h } : { position: 'absolute', inset: 0 }}>
      <Canvas frameloop={mode === 'explore' || mode === 'research' ? 'never' : 'always'} dpr={[1, 2]} shadows="soft" camera={{ position: [780, 430, 1060], fov: 34, near: 1, far: 8000 }}
        gl={{ antialias: false, powerPreference: 'high-performance', preserveDrawingBuffer: true }}
        onPointerMissed={() => { if (!useStudio.getState().ui.lookThrough) select(null) }}>
        <Suspense fallback={null}><SceneRoot /></Suspense>
      </Canvas>
      </div>

      {ui.lookThrough && (
        <div className="letterbox">
          <div className="thirds" style={{ left: lr.x, top: lr.y, width: lr.w, height: lr.h }} />
          <div style={{ position: 'absolute', left: lr.x, top: lr.y, width: lr.w, height: lr.h, boxShadow: '0 0 0 9999px rgba(0,0,0,.0)', border: '1px solid rgba(255,255,255,.35)' }} />
          <div className="readout mono" style={{ position: 'absolute', left: lr.x + 10, top: lr.y + lr.h - 26, transform: 'none' }}>
            {cam.name} · {cam.sensor.width_px}×{cam.sensor.height_px} · f={cam.lens.focal_length_mm} mm f/{cam.lens.f_number} · vFOV {intr.fovYDeg.toFixed(1)}°
          </div>
        </div>
      )}
      {ui.lookThrough && <div className="lt-badge badge mock" style={{ cursor: 'default' }} ><i />Looking through HSI camera <span className="kbd">C</span> to exit</div>}

      <div className="vp-toolbar">
        <div className="group">
          <ToolBtn on={ui.gizmo === 'translate'} onClick={() => setUi({ gizmo: 'translate' })} tip="Move" kbd="G"><Move3d size={16} /></ToolBtn>
          <ToolBtn on={ui.gizmo === 'rotate'} onClick={() => setUi({ gizmo: 'rotate' })} tip="Rotate" kbd="R"><Rotate3d size={16} /></ToolBtn>
          <ToolBtn on={ui.snap} onClick={() => setUi({ snap: !ui.snap })} tip="Snap to 25 mm grid / 15°" kbd="S"><Magnet size={16} /></ToolBtn>
        </div>
        <div className="group">
          <ToolBtn onClick={() => viewApi.current?.frame()} tip="Frame selection / bench" kbd="F"><Focus size={16} /></ToolBtn>
          <ToolBtn on={ui.lookThrough} onClick={() => setUi({ lookThrough: !ui.lookThrough })} tip="Look through HSI camera" kbd="C"><Video size={16} /></ToolBtn>
          <ToolBtn on={ui.pip} onClick={() => setUi({ pip: !ui.pip })} tip="Camera view (picture-in-picture)"><PictureInPicture2 size={16} /></ToolBtn>
        </div>
        {blockId && <div className="group">
          <ToolBtn on={cutOn} onClick={() => { const o = useStudio.getState().exp.objects.find((x) => x.id === blockId); if (o) setCutaway(o, !cutOn) }} tip="Cutaway / section view of the layered block" kbd="X"><Slice size={16} /></ToolBtn>
        </div>}
        <div className="group">
          <ToolBtn on={ui.showBeam} onClick={() => setUi({ showBeam: !ui.showBeam })} tip="Show light beam"><Sun size={16} /></ToolBtn>
          <ToolBtn on={ui.quality === 'high'} onClick={() => setUi({ quality: ui.quality === 'high' ? 'low' : 'high' })} tip="Render quality: ambient occlusion + SMAA (turn off on slow GPUs)"><Gauge size={16} /></ToolBtn>
          <ToolBtn on={ui.showFrustum} onClick={() => setUi({ showFrustum: !ui.showFrustum })} tip="Show camera frustum, focus plane & depth of field"><ScanLine size={16} /></ToolBtn>
        </div>
      </div>

      <SectionCard />
      <LightFieldLegend />
      {ui.pip && <Pip />}
      <div className="vp-hint glass"><span>Drag orbit</span><span>Right-drag pan</span><span>Scroll zoom</span></div>
    </div>
  )
}
