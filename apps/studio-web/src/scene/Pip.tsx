import { useEffect, useRef, useState } from 'react'
import { Maximize2, X, Cpu, Radio } from 'lucide-react'
import { useStudio, activeCamera } from '../state/store'
import { cameraIntrinsics } from '../lib/camera'
import { getEngine } from '../api/client'
import { paintComposite } from '../explorer/render'
import { runPreview } from '../state/actions'
import { pipFrame } from './viewApi'
import { useActiveNm } from './useExcitation'
import { wavelengthToHex } from '../lib/color'

/** Picture-in-picture "Camera view".
 *  - Live: the HSI camera's GPU render (RenderLoop scissors it into `.frame`) with approximate fluorescence glow — "Preview (approx.)".
 *  - Engine: the physics engine's own true-colour preview image of the same camera, swapped in by "Run engine preview". */
export function Pip() {
  const cam = useStudio((s) => activeCamera(s.exp))
  const meta = useStudio((s) => s.previewMeta)
  const stale = useStudio((s) => s.previewExp !== null && s.previewExp !== s.exp)
  const previewing = useStudio((s) => s.previewing)
  const setUi = useStudio((s) => s.setUi)
  const lookThrough = useStudio((s) => s.ui.lookThrough)
  const nm = useActiveNm()
  const [showEngine, setShowEngine] = useState(false)
  const cv = useRef<HTMLCanvasElement>(null)
  const { aspect } = cameraIntrinsics(cam)
  const w = 300
  const eng = getEngine()
  const realImg = !!meta && eng?.kind === 'real'
  const exNm = meta?.excitations_nm[0]

  useEffect(() => { if (meta) setShowEngine(true) }, [meta])
  // mock engine has no PNG renderer: paint the composite from its cube instead
  useEffect(() => {
    if (!showEngine || !meta || !eng || realImg || !cv.current) return
    let dead = false
    const ex = meta.excitations_nm[0]
    eng.cube(meta.id, ex, 'adu').then((cube) => {
      if (dead || !cv.current) return
      const c = cv.current; c.width = cube.width; c.height = cube.height
      const g = c.getContext('2d')!; const img = g.createImageData(cube.width, cube.height)
      paintComposite(img, cube, meta.emission_nm[String(ex)])
      g.putImageData(img, 0, 0)
    }).catch(() => { /* leave the previous frame */ })
    return () => { dead = true }
  }, [showEngine, meta, eng, realImg])

  if (lookThrough) return null
  const engineShown = showEngine && !!meta
  return (
    <div className="pip glass" style={{ width: w }}>
      <header>
        <Maximize2 size={11} /> Camera view
        {nm != null && !engineShown && <span className="tag" style={{ color: wavelengthToHex(nm) }}>{nm} nm</span>}
        {engineShown && exNm != null && <span className="tag" style={{ color: wavelengthToHex(exNm) }}>{exNm} nm</span>}
        <span className={`pip-state ${engineShown ? (stale ? 'stale' : 'engine') : 'approx'}`}
          title={engineShown ? (stale ? 'The scene changed since this engine preview — re-run it' : 'Rendered by the physics engine (true colour of the simulated camera)') : 'Real-time GPU approximation of the HSI camera view (no scattering/transport physics)'}>
          {engineShown ? (stale ? 'Engine · stale' : 'Engine preview') : 'Preview (approx.)'}
        </span>
        <button className="btn ghost icon sm" style={{ width: 18, height: 18, padding: 2, marginLeft: 0 }} onClick={() => setUi({ pip: false })} aria-label="Close camera view"><X size={11} /></button>
      </header>
      <div className="frame" style={{ width: w, height: w / aspect }}>
        <div ref={(el) => { pipFrame.el = el }} style={{ position: 'absolute', inset: 0 }} />
        {engineShown && (realImg
          ? <img alt="engine preview" src={eng!.imageUrl(meta!.id, 'true_color', exNm!, undefined, 1.3)} style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', imageRendering: 'auto', objectFit: 'fill', background: '#000' }} />
          : <canvas ref={cv} style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', background: '#000' }} />)}
        <div className="pip-vignette" />
        <div className="thirds" style={{ inset: 0, opacity: 0.3, pointerEvents: 'none' }} />
      </div>
      <footer>
        <button className="btn sm" disabled={previewing} onClick={() => void runPreview()} title="Render this view with the physics engine (Space)">
          <Cpu size={12} />{previewing ? 'Rendering…' : meta && !stale ? 'Re-run engine preview' : 'Run engine preview'}
        </button>
        {meta && <button className="btn ghost sm" onClick={() => setShowEngine(!showEngine)} title="Toggle between the live approximation and the engine result"><Radio size={12} />{engineShown ? 'Live' : 'Engine'}</button>}
      </footer>
    </div>
  )
}
