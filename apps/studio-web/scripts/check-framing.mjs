// "What you see is what you capture": projects each object's centre with the SAME pinhole model the viewport/PiP use
// (three PerspectiveCamera built from the HSI intrinsics) and compares it with the centroid of that object's pixels in the
// engine's ground-truth `object` layer. Requires the engine on :8765. Usage: node scripts/check-framing.mjs [preset]
import fs from 'fs'
import * as THREE from 'three'

const presets = JSON.parse(fs.readFileSync(new URL('../src/generated/presets.json', import.meta.url), 'utf8')).experiments
const keys = process.argv[2] ? [process.argv[2]] : Object.keys(presets)
const base = 'http://127.0.0.1:8765'
let worst = 0

function lookAtQ(pos, tgt, roll) {
  const m = new THREE.Matrix4(), eye = new THREE.Vector3(...pos), t = new THREE.Vector3(...tgt)
  const dir = t.clone().sub(eye).normalize()
  m.lookAt(eye, t, Math.abs(dir.y) > 0.999 ? new THREE.Vector3(0, 0, -1) : new THREE.Vector3(0, 1, 0))
  const q = new THREE.Quaternion().setFromRotationMatrix(m)
  if (roll) q.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), THREE.MathUtils.degToRad(roll)))
  return q
}

for (const key of keys) {
  const exp = presets[key]
  const cam = exp.cameras.find((c) => c.id === exp.active_camera_id) ?? exp.cameras[0]
  const f = cam.lens.focal_length_mm, s = cam.lens.focus_distance_mm
  const v = (f * s) / (s - f)
  const pitch = (cam.sensor.pixel_pitch_um * cam.sensor.binning) / 1000
  const sh = cam.sensor.height_px * pitch
  const fov = 2 * Math.atan(sh / (2 * v))
  const aspect = cam.sensor.width_px / cam.sensor.height_px
  const three = new THREE.PerspectiveCamera(THREE.MathUtils.radToDeg(fov), aspect, 1, 5000)
  three.position.set(...cam.position_mm); three.quaternion.copy(lookAtQ(cam.position_mm, cam.target_mm, cam.roll_deg)); three.updateMatrixWorld(true); three.updateProjectionMatrix()

  const meta = await (await fetch(`${base}/api/preview`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ experiment: exp, scale: 0.5 }) })).json()
  const g = await fetch(`${base}/api/captures/${meta.id}/gt/object`)
  const W = +g.headers.get('x-width'), H = +g.headers.get('x-height')
  const lab = new Float32Array(await g.arrayBuffer())
  console.log(`\n${key}: engine image ${W}x${H}, three aspect ${aspect.toFixed(3)} vFOV ${THREE.MathUtils.radToDeg(fov).toFixed(2)} deg`)
  exp.objects.forEach((o, idx) => {
    let n = 0, sx = 0, sy = 0, border = false
    for (let k = 0; k < lab.length; k++) if (lab[k] === idx) {
      const x = k % W, y = Math.floor(k / W)
      n++; sx += x; sy += y
      if (x === 0 || y === 0 || x === W - 1 || y === H - 1) border = true
    }
    if (n < 6) { console.log(`  ${o.id}: not visible in engine image (${n} px)`); return }
    if (border) { console.log(`  ${o.id}: cut by the image border - centroid not comparable, skipped`); return }
    if (o.shape.kind === 'cuvette') { console.log(`  ${o.id}: tall vessel - skipped`); return }
    // centroid of the visible top surface is the projection of the object's top-centre; use the shape's top height
    const sp = o.shape, top = sp.kind === 'box' ? sp.size_mm[1] / 2 : sp.kind === 'cylinder' ? sp.height_mm / 2 : sp.kind === 'sphere' ? 0 : sp.kind === 'droplet' ? sp.height_mm : sp.kind === 'petri_dish' ? sp.height_mm : sp.kind === 'cuvette' ? sp.inner_mm[1] + sp.wall_mm : 0
    const p = new THREE.Vector3(...o.transform.position_mm); p.y += top
    p.project(three)
    const px = ((p.x + 1) / 2) * W, py = ((1 - p.y) / 2) * H
    const ex = sx / n, ey = sy / n
    const err = Math.hypot(px - ex, py - ey)
    worst = Math.max(worst, err)
    console.log(`  ${o.id.padEnd(12)} engine centroid (${ex.toFixed(1)}, ${ey.toFixed(1)})  projected top-centre (${px.toFixed(1)}, ${py.toFixed(1)})  error ${err.toFixed(2)} px`)
  })
}
console.log(`\nworst error ${worst.toFixed(2)} px (image up = world -Z for top-down cameras; x right = world +X)`)
process.exit(worst > 3.5 ? 1 : 0)
