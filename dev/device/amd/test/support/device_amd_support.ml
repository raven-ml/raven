(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

external lock : string -> bool = "device_amd_test_lock"
external read : nativeint -> int -> string = "device_amd_test_read"
external write : nativeint -> string -> unit = "device_amd_test_write"

let gpu_lock = "DEVICE_AMD_TEST_GPU_LOCK"

(* The lock is taken once and kept: [Some true] once taken. *)
let held = ref None

let take_lock () =
  match !held with
  | Some taken -> taken
  | None ->
      let taken =
        match Sys.getenv_opt gpu_lock with
        | None | Some "" -> skip ~reason:(gpu_lock ^ " names no lock file") ()
        | Some file -> lock file
      in
      held := Some taken;
      taken

(* The device gpu opened, until a test stops it. *)
let opened = ref None

let stop g =
  (match !opened with Some o when o == g -> opened := None | _ -> ());
  Device_amd.stop g

let gpu () =
  if Device_amd_amdgpu.count () = 0 then
    skip ~reason:"the machine has no AMD GPU" ();
  if not (take_lock ()) then
    skip ~reason:"another process holds the GPU lock" ();
  Option.iter (fun g -> ignore (stop g)) !opened;
  match Device_amd_amdgpu.open_ 0 with
  | Ok g ->
      opened := Some g;
      g
  | Error why -> failwith why

let with_gpu f =
  let g = gpu () in
  let stop_left () =
    match !opened with Some o when o == g -> ignore (stop g) | _ -> ()
  in
  Fun.protect ~finally:stop_left (fun () -> f g)

let wait g v =
  let rec loop () =
    let seen = Device_amd.signaled g in
    if seen < v then begin
      Device_amd.sleep g ~seen ~still_ms:200;
      loop ()
    end
  in
  loop ()
