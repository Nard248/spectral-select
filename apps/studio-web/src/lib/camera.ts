import * as THREE from 'three'
import type { CameraSpec } from '../generated/experiment'
import { verticalFovRad } from './spectral'
import { lookAtQuaternion } from '../three/pose'

/** Intrinsics of the HSI camera: vertical FOV (rad) and sensor aspect (see brief: v = f·s/(s−f)). */
export function cameraIntrinsics(c: CameraSpec) {
  const fov = verticalFovRad(c.lens.focal_length_mm, c.lens.focus_distance_mm, c.sensor.height_px, c.sensor.pixel_pitch_um, c.sensor.binning)
  return { fovY: fov, fovYDeg: THREE.MathUtils.radToDeg(fov), aspect: c.sensor.width_px / c.sensor.height_px }
}

/** A three.js PerspectiveCamera posed exactly like the HSI camera. */
export function buildThreeCamera(c: CameraSpec): THREE.PerspectiveCamera {
  const { fovYDeg, aspect } = cameraIntrinsics(c)
  const cam = new THREE.PerspectiveCamera(fovYDeg, aspect, 1, 5000)
  cam.position.set(...c.position_mm)
  cam.quaternion.copy(lookAtQuaternion(c.position_mm, c.target_mm, c.roll_deg))
  cam.updateMatrixWorld(true)
  cam.updateProjectionMatrix()
  return cam
}

export const distance = (a: readonly number[], b: readonly number[]) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])
