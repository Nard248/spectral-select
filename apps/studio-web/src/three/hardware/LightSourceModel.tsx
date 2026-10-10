import { useMemo } from 'react'
import type { LightSourceModelProps } from '../types'
import { sourceQuat } from './sourceCommon'
import { TunableLampModel } from './TunableLampModel'
import { LedArrayModel } from './LedArrayModel'
import { LaserModel } from './LaserModel'
import { BeamVolume } from './BeamVolume'

export { BeamVolume } from './BeamVolume'

/** Per-kind hardware model: `tunable_lamp` (collimator head + fibre + Xe/monochromator unit), `led_array`, `laser`.
 *  The exit aperture is at `position_mm`, the beam travels along local −Z toward `target_mm`.
 *  When `showBeam` and `activeWavelengthNm != null` a volumetric BeamVolume is drawn (world space). */
export function LightSourceModel({ source, activeWavelengthNm, showBeam, selected, hovered }: LightSourceModelProps) {
  const q = useMemo(() => sourceQuat(source), [source.position_mm, source.target_mm, source.roll_deg]) // eslint-disable-line react-hooks/exhaustive-deps
  const common = { source, wavelengthNm: activeWavelengthNm, q, selected, hovered }
  return (
    <group>
      <group position={source.position_mm}>
        {source.kind === 'tunable_lamp' ? <TunableLampModel {...common} />
          : source.kind === 'led_array' ? <LedArrayModel {...common} />
          : <LaserModel {...common} />}
      </group>
      {showBeam && activeWavelengthNm != null && <BeamVolume source={source} wavelengthNm={activeWavelengthNm} />}
    </group>
  )
}
