import { useStudio } from '../state/store'
import { ObjectInspector } from './inspector/ObjectInspector'
import { MaterialInspector } from './inspector/MaterialInspector'
import { SourceInspector } from './inspector/SourceInspector'
import { CameraInspector } from './inspector/CameraInspector'
import { AcquisitionInspector, BackgroundInspector, ExperimentInspector } from './inspector/SceneInspectors'

export function Inspector() {
  const sel = useStudio((s) => s.selection)
  return (
    <>
      <div className="ptitle">Inspector</div>
      <div className="scroll" key={sel ? `${sel.kind}:${sel.id}` : 'none'}>
        {!sel || sel.kind === 'experiment' ? <ExperimentInspector />
          : sel.kind === 'object' ? <ObjectInspector id={sel.id} />
          : sel.kind === 'material' ? <MaterialInspector id={sel.id} />
          : sel.kind === 'source' ? <SourceInspector id={sel.id} />
          : sel.kind === 'camera' ? <CameraInspector id={sel.id} />
          : sel.kind === 'background' ? <BackgroundInspector />
          : <AcquisitionInspector />}
      </div>
    </>
  )
}
