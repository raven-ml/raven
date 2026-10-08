(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

let strf = Printf.sprintf

external read : int -> int -> string = "device_amd_test_read"
external write : int -> string -> unit = "device_amd_test_write"
external pages : int -> int = "device_amd_test_pages"
external free_pages : int -> int -> unit = "device_amd_test_free_pages"

external fill_arg :
  nativeint -> nativeint -> int array -> int -> int -> int -> nativeint
  = "device_amd_test_fill_arg_byte" "device_amd_test_fill_arg"

external fill_entry : unit -> nativeint = "device_amd_test_fill_entry"

(* The machine's GPU lock *)

external lock : string -> string -> int = "device_amd_test_lock"

let gpu_lock = "/tmp/raven-device-gpu.lock"

(* The longest wait for the lock, in seconds: the machine's suites, from every
   checkout and user, take it in turn. *)
let gpu_wait = 300

let holder () =
  match In_channel.with_open_bin gpu_lock In_channel.input_all with
  | note -> String.trim note
  | exception Sys_error _ -> "a process that left no note"

(* [lock] naps 100 ms each time it is refused. *)
let rec take refused =
  match lock gpu_lock Sys.executable_name with
  | 0 -> ()
  | -1 when refused < gpu_wait * 10 -> take (refused + 1)
  | -1 ->
      failwith
        (strf "%s: still held after %d s, by %s" gpu_lock gpu_wait (holder ()))
  | errno -> failwith (strf "%s: errno %d" gpu_lock errno)

let hold_gpu () = if Device_amd_amdgpu.count () > 0 then take 0

(* The device gpu opened, until a test stops it. *)
let opened = ref None

let stop g =
  (match !opened with Some o when o == g -> opened := None | _ -> ());
  Device_amd.stop g

let gpu () =
  if Device_amd_amdgpu.count () = 0 then
    skip ~reason:"the machine has no AMD GPU" ();
  hold_gpu ();
  Option.iter stop !opened;
  match Device_amd_amdgpu.open_ 0 with
  | Ok g ->
      opened := Some g;
      g
  | Error why -> failwith why

let with_gpu f =
  let g = gpu () in
  let stop_left () =
    match !opened with Some o when o == g -> stop g | _ -> ()
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

let still ?msg w x f ~ms =
  let t0 = Sys.time () in
  while Sys.time () -. t0 < Float.of_int ms /. 1000. do
    equal ?msg w x (f ())
  done

let fill ?(code = 0) ?(split = 0) (c : Device_amd.capability) ws ~bytes =
  (fill_entry (), fill_arg c.place c.segment ws split bytes code)

external room_c :
  nativeint -> nativeint -> int -> int -> bool -> int -> int -> int
  = "device_amd_test_room_byte" "device_amd_test_room"

let room ?(words = 0) ?(fill = false) ?(copy = 0) ?(after = -1) g ~queue =
  match room_c Device_amd.room_entry (Device_amd.self g) queue words fill copy after with
  | 0 -> `Fits
  | 1 -> `Later
  | _ -> `Never
