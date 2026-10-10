import { useFrame, useThree } from '@react-three/fiber'
import { useMemo } from 'react'
import * as THREE from 'three'
import { useStudio, activeCamera } from '../state/store'
import { buildThreeCamera } from '../lib/camera'
import { pipFrame, viewportEl } from './viewApi'

/** Largest rect of the given aspect centred in (w,h). */
export function fitRect(w: number, h: number, aspect: number) {
  let rw = w, rh = w / aspect
  if (rh > h) { rh = h; rw = h * aspect }
  return { x: (w - rw) / 2, y: (h - rh) / 2, w: rw, h: rh }
}

/** Second render pass (priority 2, after the post-processing composer): the HSI camera's "Camera view"
 *  scissored into the PiP DOM frame. Helpers, gizmos and the active camera body are hidden for this pass and
 *  filmic tone mapping is applied directly (the composer's own ToneMapping effect does not run here). */
export function PipPass() {
  const gl = useThree((s) => s.gl)
  const cam = useStudio((s) => activeCamera(s.exp))
  const hsi = useMemo(() => buildThreeCamera(cam), [cam])

  useFrame(({ scene, size }) => {
    const st = useStudio.getState()
    const frame = pipFrame.el, host = viewportEl.el
    if (!st.ui.pip || st.ui.lookThrough || !frame || !host) return
    const fr = frame.getBoundingClientRect(), hr = host.getBoundingClientRect()
    if (fr.width < 4 || fr.height < 4) return
    const x = fr.left - hr.left, y = size.height - (fr.bottom - hr.top)
    const hidden: THREE.Object3D[] = []
    scene.traverse((o) => { if (o.userData.hideInPip && o.visible) { o.visible = false; hidden.push(o) } })
    const prevTm = gl.toneMapping, prevExp = gl.toneMappingExposure
    gl.toneMapping = THREE.ACESFilmicToneMapping; gl.toneMappingExposure = 1
    gl.setScissorTest(true)
    gl.setViewport(x, y, fr.width, fr.height)
    gl.setScissor(x, y, fr.width, fr.height)
    try { gl.render(scene, hsi) } finally {
      gl.setScissorTest(false)
      gl.setViewport(0, 0, size.width, size.height)
      gl.toneMapping = prevTm; gl.toneMappingExposure = prevExp
      hidden.forEach((o) => { o.visible = true })
    }
  }, 2)
  return null
}
