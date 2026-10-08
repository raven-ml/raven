(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

(* The machine's GPU lock *)

external lock : string -> string -> int = "rig_metal_test_lock"

let gpu_lock = "/tmp/raven-rig-gpu.lock"

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

let hold_gpu () = if Rig_metal.count () > 0 then take 0

(* A device's ring, by hand *)

type ring

external ring : int -> ring = "rig_metal_test_ring"
external commit : ring -> last:bool -> int = "rig_metal_test_commit"

external complete : ring -> int -> failed:bool -> unit
  = "rig_metal_test_complete"

external word : ring -> int = "rig_metal_test_word"
external times : ring -> int -> int * int = "rig_metal_test_times"
external failure : ring -> string option = "rig_metal_test_failure"
external sleep : ring -> string option = "rig_metal_test_sleep"
external stop : ring -> bool = "rig_metal_test_stop"

(* Host memory *)

external get8 : int -> int -> int = "rig_metal_test_get8"
external set8 : int -> int -> int -> unit = "rig_metal_test_set8"
external get32 : int -> int -> int = "rig_metal_test_get32"
external set32 : int -> int -> int -> unit = "rig_metal_test_set32"
external get64 : int -> int -> int64 = "rig_metal_test_get64"
external set64 : int -> int -> int64 -> unit = "rig_metal_test_set64"
external pages : int -> int = "rig_metal_test_pages"

(* Fills *)

type arg =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

external failing_arg : int -> arg = "rig_metal_test_failing"
external failing_fill : unit -> nativeint = "rig_metal_test_failing_fill"

external dispatch_arg : nativeint -> nativeint -> int -> int -> int -> arg
  = "rig_metal_test_dispatch"

external split_arg : arg -> nativeint -> int -> int -> unit
  = "rig_metal_test_split"

external dispatch_fill : unit -> nativeint = "rig_metal_test_dispatch_fill"

external execute_arg : nativeint -> int -> nativeint array -> arg
  = "rig_metal_test_execute"

external execute_fill : unit -> nativeint = "rig_metal_test_execute_fill"

type fill = { fn : nativeint; arg : arg }

let part f =
  let arg = Rig.Buffer.of_bigarray f.arg in
  {
    Rig.Submission.queue = "COMPUTE:0";
    after = [||];
    work = Fill { fill = f.fn; arg; ring_units = 0; segment_bytes = 0 };
  }

let failing code = { fn = failing_fill (); arg = failing_arg code }

let dispatch ~pipeline ?(offset = 0) args ~groups ~threads =
  let arg =
    dispatch_arg
      (Nativeint.of_int pipeline)
      (Rig_metal.handle args) offset groups threads
  in
  { fn = dispatch_fill (); arg }

let split f d k ~times =
  split_arg f.arg (Rig_metal.capability d).split k times

let execute (b : Rig_metal_abi.icb) ~pipelines =
  let n = Array.length b.commands in
  let arg = execute_arg b.handle n (Array.map Nativeint.of_int pipelines) in
  { fn = execute_fill (); arg }

external watching_arg : unit -> arg * nativeint = "rig_metal_test_watching"
external watching_fill : unit -> nativeint = "rig_metal_test_watching_fill"

let watching () =
  let arg, slot = watching_arg () in
  ({ fn = watching_fill (); arg }, slot)

external resize : nativeint -> groups:int -> threads:int -> unit
  = "rig_metal_test_resize"

(* Probes *)

external macos : unit -> bool = "rig_metal_test_macos"

let macos = macos ()

let fixture ~dir f =
  In_channel.with_open_bin
    (Filename.concat dir (f ^ ".metallib"))
    In_channel.input_all

external weak : nativeint -> nativeint = "rig_metal_test_weak"
external alive : nativeint -> bool = "rig_metal_test_alive"
external uptime : unit -> int = "rig_metal_test_uptime"

let wait d v =
  let host = Option.get (Rig_metal.host (Rig_metal.word d)) in
  while Int64.to_int (get64 host 0) < v do
    Domain.cpu_relax ()
  done
