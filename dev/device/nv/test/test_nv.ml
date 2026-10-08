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
    Device_nv.key = Type.Id.make ();
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

(* A GPU of this machine, opened once for the suite; the tests that need one
   skip without it. *)
let gpu =
  lazy
    (match Device_nv_nvidia.open_ 0 with
    | Ok g -> g
    | Error why -> skip ~reason:why ())

let cubin =
  lazy
    (In_channel.with_open_bin "fixtures/simple_add_sm89.cubin"
       In_channel.input_all)

(* Regions freed and images unloaded from two domains: whatever the order, the
   first call returns and every later one raises. *)

type live = { mutable live : bool }

let once m =
  if not m.live then invalid_arg "freed";
  m.live <- false

let release f x = try f x with Invalid_argument _ -> ()
let free_system r = Device_nv.free (Lazy.force gpu) r
let unload_system i = Device_nv.unload (Lazy.force gpu) i
let region = abstract "r" ~release:(release free_system)
let image = abstract "i" ~release:(release unload_system)

let once_commands =
  [
    command "alloc"
      (Gen.unit @-> makes region)
      (fun () -> { live = true })
      (fun () -> require_some (Device_nv.alloc (Lazy.force gpu) `Device 4096));
    command "free" (region ^-> returns unit) once free_system;
    command "image"
      (Gen.unit @-> makes image)
      (fun () -> { live = true })
      (fun () ->
        fst (require_ok (Device_nv.image (Lazy.force gpu) (Lazy.force cubin))));
    command "unload" (image ^-> returns unit) once unload_system;
  ]

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
         group ~timeout:60. "domains"
           [
             stateful ~domains:2 ~count:30
               "a region freed or an image unloaded from two domains is so once"
               once_commands;
           ];
       ]
