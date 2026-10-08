(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

(* A path whose RM is of release [release] and whose every function fails the
   test: [make] must refuse it before calling any. *)
let path release : unit Device_nv.path =
  let called what = fail (what ^ " was called") in
  let rm =
    {
      Device_nv.release;
      client = 0;
      alloc = (fun ~parent:_ _ _ -> called "rm.alloc");
      control = (fun _ _ _ -> called "rm.control");
      free = (fun ~parent:_ _ -> called "rm.free");
    }
  in
  {
    Device_nv.id = Type.Id.make ();
    rm;
    device = 0;
    subdevice = 0;
    vaspace = 0;
    gpu =
      {
        channel_class = 0;
        compute_class = 0;
        copy_class = 0;
        sm_version = 0;
        gpcs = 1;
        tpcs_per_gpc = 1;
        sms_per_tpc = 1;
        warps_per_sm = 1;
      };
    budget = 0;
    doorbell = 0n;
    alloc = (fun _ _ -> called "alloc");
    map_host = (fun _ _ -> called "map_host");
    map_peer = (fun _ -> called "map_peer");
    free = (fun _ -> called "free");
    register = (fun _ -> called "register");
    unregister = (fun _ -> called "unregister");
  }

let () =
  exit
  @@ run "device_nv"
       [
         group ~timeout:10. "paths"
           [
             test "an NVIDIA display controller is a GPU" (fun () ->
                 equal (list bool)
                   [ true; true; false; false ]
                   (List.map
                      (fun (vendor, class_) -> Device_nv.is_gpu ~vendor ~class_)
                      [
                        (0x10de, 0x030000);
                        (0x10de, 0x030200);
                        (0x10de, 0x040300);
                        (0x1002, 0x030000);
                      ]));
             test "make refuses an RM of an unknown release" (fun () ->
                 let e = require_error (Device_nv.make (path 999)) in
                 contains ~sub:"999" e);
           ];
       ]
