(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let openers () =
  let path count (gpu : int -> _ Rig_edge.opener) =
    let name = (gpu 0).name in
    let n = Result.value (count ()) ~default:0 in
    List.init n (fun i -> (name, fun () -> Rig.open_ (gpu i)))
  in
  List.concat
    [
      path Rig_metal.count Rig_metal.gpu;
      path Rig_cuda.count Rig_cuda.gpu;
      path Rig_nv_nvidia.count Rig_nv_nvidia.gpu;
      path Rig_amd_amdgpu.count Rig_amd_amdgpu.gpu;
      path Rig_nv_pci.count Rig_nv_pci.gpu;
      path Rig_amd_pci.count Rig_amd_pci.gpu;
    ]
