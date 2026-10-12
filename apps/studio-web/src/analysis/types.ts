/** Client-side types for the v1.1 analysis workspace (ROI model, capture metadata with block / volume info). */
import type { BlockInfo, CaptureMeta, NormSpec, RoiSpec, VolumeInfo } from '../api/types'

/** `CaptureMeta` as returned by a v1.1 engine: optional layered-block and light-field info. */
export type CaptureMetaX = CaptureMeta & { volumes?: VolumeInfo[]; blocks?: BlockInfo[] }

/** A line ROI (profile path). `width_px` is the averaging width perpendicular to the line. */
export interface LineSpec { shape: 'line'; p0: [number, number]; p1: [number, number]; width_px: number }
export type AnyRoi = RoiSpec | LineSpec

export interface NamedRoi {
  id: string
  name: string
  color: string
  visible: boolean
  spec: AnyRoi
}

export type RegionSpec = Exclude<AnyRoi, LineSpec>
export const isLine = (s: AnyRoi): s is LineSpec => s.shape === 'line'
export const isRegion = (s: AnyRoi): s is RegionSpec => s.shape !== 'line'

export type NormMode = NormSpec['mode']
export type Stat = 'mean' | 'median'
