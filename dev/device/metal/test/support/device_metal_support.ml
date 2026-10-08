(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A device's ring, by hand *)

type ring

external ring : int -> ring = "device_metal_test_ring"
external commit : ring -> last:bool -> int = "device_metal_test_commit"

external complete : ring -> int -> failed:bool -> unit
  = "device_metal_test_complete"

external word : ring -> int = "device_metal_test_word"
external times : ring -> int -> int * int = "device_metal_test_times"
external failure : ring -> string option = "device_metal_test_failure"
external sleep : ring -> string option = "device_metal_test_sleep"
external stop : ring -> bool = "device_metal_test_stop"

(* Host memory *)

external get8 : int -> int -> int = "device_metal_test_get8"
external set8 : int -> int -> int -> unit = "device_metal_test_set8"
external get32 : int -> int -> int = "device_metal_test_get32"
external set32 : int -> int -> int -> unit = "device_metal_test_set32"
external get64 : int -> int -> int64 = "device_metal_test_get64"
external set64 : int -> int -> int64 -> unit = "device_metal_test_set64"
external pages : int -> int = "device_metal_test_pages"

(* Fills *)

type arg =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

external failing_arg : int -> arg = "device_metal_test_failing"
external failing_fill : unit -> nativeint = "device_metal_test_failing_fill"

external dispatch_arg : nativeint -> nativeint -> int -> int -> int -> arg
  = "device_metal_test_dispatch"

external split_arg : arg -> nativeint -> int -> int -> unit
  = "device_metal_test_split"

external dispatch_fill : unit -> nativeint = "device_metal_test_dispatch_fill"

external execute_arg : nativeint -> int -> nativeint array -> arg
  = "device_metal_test_execute"

external execute_fill : unit -> nativeint = "device_metal_test_execute_fill"

type fill = { fn : nativeint; arg : arg }

let part f =
  let arg = Device_core.Buffer.of_bigarray f.arg in
  {
    Device_core.Submission.queue = "COMPUTE:0";
    after = [||];
    work = Fill { fill = f.fn; arg; ring_units = 0; segment_bytes = 0 };
  }

let failing code = { fn = failing_fill (); arg = failing_arg code }

let dispatch ~pipeline ?(offset = 0) args ~groups ~threads =
  let arg =
    dispatch_arg
      (Nativeint.of_int pipeline)
      (Device_metal.handle args) offset groups threads
  in
  { fn = dispatch_fill (); arg }

let split f d k ~times =
  split_arg f.arg (Device_metal.capability d).split k times

let execute (b : Device_metal_abi.icb) ~pipelines =
  let n = Array.length b.commands in
  let arg = execute_arg b.handle n (Array.map Nativeint.of_int pipelines) in
  { fn = execute_fill (); arg }

external watching_arg : unit -> arg * nativeint = "device_metal_test_watching"
external watching_fill : unit -> nativeint = "device_metal_test_watching_fill"

let watching () =
  let arg, slot = watching_arg () in
  ({ fn = watching_fill (); arg }, slot)

external resize : nativeint -> groups:int -> threads:int -> unit
  = "device_metal_test_resize"

(* Probes *)

external macos : unit -> bool = "device_metal_test_macos"

let macos = macos ()

let fixture ~dir f =
  In_channel.with_open_bin
    (Filename.concat dir (f ^ ".metallib"))
    In_channel.input_all

external weak : nativeint -> nativeint = "device_metal_test_weak"
external alive : nativeint -> bool = "device_metal_test_alive"
external uptime : unit -> int = "device_metal_test_uptime"

let wait d v =
  let host = Option.get (Device_metal.host (Device_metal.word d)) in
  while Int64.to_int (get64 host 0) < v do
    Domain.cpu_relax ()
  done
