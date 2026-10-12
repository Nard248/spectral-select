import * as THREE from 'three'

/** Orientation for hardware posed by position + target + roll. The device's optical axis is its
 * local **-Z** (three.js camera convention), local +Y is "up" on the device body. */
export function lookAtQuaternion(position: [number, number, number], target: [number, number, number],
  rollDeg = 0): THREE.Quaternion {
  const m = new THREE.Matrix4()
  const eye = new THREE.Vector3(...position)
  const tgt = new THREE.Vector3(...target)
  const dir = tgt.clone().sub(eye).normalize()
  // world up, unless looking (almost) straight down/up — then use -Z as up (image top = -Z)
  const up = Math.abs(dir.y) > 0.999 ? new THREE.Vector3(0, 0, -1) : new THREE.Vector3(0, 1, 0)
  m.lookAt(eye, tgt, up)
  const q = new THREE.Quaternion().setFromRotationMatrix(m)
  if (rollDeg) q.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), THREE.MathUtils.degToRad(rollDeg)))
  return q
}
