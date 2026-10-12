import { Box, Circle, Cylinder, Droplet, FlaskConical, Disc3, Lightbulb, Camera, Layers, Zap, Atom, Sun, Square, TestTube, Wand2, Layers3 } from 'lucide-react'
import type { SceneObject } from '../generated/experiment'

export function ShapeIcon({ kind, size = 15 }: { kind: SceneObject['shape']['kind']; size?: number }) {
  switch (kind) {
    case 'box': return <Box size={size} />
    case 'sphere': return <Circle size={size} />
    case 'cylinder': return <Cylinder size={size} />
    case 'droplet': return <Droplet size={size} />
    case 'petri_dish': return <Disc3 size={size} />
    case 'cuvette': return <TestTube size={size} />
    case 'layered_block': return <Layers3 size={size} />
  }
}
export const SourceIcon = ({ size = 15, kind }: { size?: number; kind?: string }) => kind === 'laser' ? <Zap size={size} /> : kind === 'led_array' ? <Sun size={size} /> : <Lightbulb size={size} />
export const CameraIcon = ({ size = 15 }: { size?: number }) => <Camera size={size} />
export const BackgroundIcon = ({ size = 15 }: { size?: number }) => <Layers size={size} />
export { FlaskConical, Atom, Square, Wand2 }
