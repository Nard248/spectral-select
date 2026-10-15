import time, numpy as np
from spectraforge.studio import api, presets as P
from spectraforge.studio.schema import *
import sys
sys.path.insert(0, "publications/spectralforge_studio/experiments")

def build(photons=300_000, sensor=(64,48), exs=(340,365,385,420)):
    exp = P.mehsi_lab_replica()
    mats = {m.id: m for m in exp.materials}
    from spectraforge.studio.presets_tissue import tissue_materials
    for m in tissue_materials(): mats[m.id]=m
    incl=[Inclusion(name="nodule",material_id="dysplasia_nodule",kind="box",center_mm=(0,3.45,-2.3),size_mm=(30,0.5,2.2)),
          Inclusion(name="vessel",material_id="blood_oxygenated",kind="cylinder",center_mm=(0,3.45,0.0),radius_mm=0.45,axis="x",length_mm=1000),
          Inclusion(name="fat",material_id="subcutis_fat",kind="box",center_mm=(0,3.45,2.3),size_mm=(30,0.5,2.2))]
    block=LayeredBlockShape(footprint_mm=(7,7),layers=[P._layer("substrate","dermis_reticular",3.0),P._layer("tissue","mucosa_colon",0.8),P._layer("cover","collagen_dense",1.5,0.0,"x")],inclusions=incl)
    seq=[ExcitationStep(wavelength_nm=e,power_mW=3.0,exposure_ms=200) for e in exs]
    src=P.tunable_lamp(target=(0,2.0,0)).model_copy(update={"sequence":seq})
    cam=P.nuance_camera(target=(0,2.0,0))
    cam=cam.model_copy(update={"sensor":cam.sensor.model_copy(update={"width_px":sensor[0],"height_px":sensor[1],"binning":2})})
    acq=exp.acquisition.model_copy(update={"preview_scale":0.5,"volume_mc":exp.acquisition.volume_mc.model_copy(update={"excitation_photons":photons,"emission_photons":photons})})
    return exp.model_copy(update={"name":"E9","materials":list(mats.values()),"objects":[P._obj("block","block",block,"collagen_dense",(0,0,0))],"sources":[src],"cameras":[cam],"acquisition":acq})
if __name__=="__main__":
    exp=build()
    t=time.time(); r=api.capture(exp,mode="fast"); print("t",time.time()-t,r.timings_s, r.height,r.width,r.excitations_nm)
    print(sorted(r.ground_truth.keys()))
    import pickle
    for k in ["inclusion_mask:nodule","inclusion_mask:vessel","inclusion_mask:fat","layer_thickness:cover"]:
        g=np.asarray(r.ground_truth[k]); print(k,np.nanmin(g),np.nanmax(g),np.nanmean(g))
    print({e:(r.cubes[e].shape if hasattr(r,'cubes') else None) for e in r.excitations_nm} if hasattr(r,'cubes') else dir(r))
