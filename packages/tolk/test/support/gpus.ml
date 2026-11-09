let opened =
  [
    ("Metal", lazy (Rig.open_ (Rig_metal.gpu 0)));
    ("CUDA", lazy (Rig.open_ (Rig_cuda.gpu 0)));
    ("NV", lazy (Rig.open_ (Rig_nv_nvidia.gpu 0)));
    ("AMD", lazy (Rig.open_ (Rig_amd_amdgpu.gpu 0)));
  ]

let kinds = List.map fst opened

let lacks_compiler why =
  String.starts_with ~prefix:"failed to load library" why
  || String.starts_with ~prefix:"comgr not available" why

let gpu kind =
  match Lazy.force (List.assoc kind opened) with
  | Error why -> Windtrap.skip ~reason:why ()
  | Ok d -> (
      match Tolk_engine.renderer d with
      | _ -> d
      | exception Failure why when lacks_compiler why ->
          Windtrap.skip ~reason:why ())
