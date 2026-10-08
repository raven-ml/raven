(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Wire = Rig_remote_proxy.Wire

external tune : Unix.file_descr -> unit = "rig_remote_bench_tune"

external ask : Unix.file_descr -> Rig_remote_abi.area -> int -> int -> int
  = "rig_remote_bench_ask"

external echo : Unix.file_descr -> Rig_remote_abi.area -> int -> int -> int
  = "rig_remote_bench_echo"

external stream_open : Unix.file_descr -> Unix.file_descr -> int -> nativeint
  = "rig_remote_bench_stream_open"

external stream_run : nativeint -> int = "rig_remote_bench_stream_run"
external stream_close : nativeint -> unit = "rig_remote_bench_stream_close"

external rail_run_c :
  nativeint -> nativeint -> Rig_remote_abi.area -> int -> int
  = "rig_remote_bench_rail_run"

let rail_run (s : Rig_remote_abi.end_) (r : Rig_remote_abi.end_) c =
  rail_run_c s.ready_fn s.ready_arg r.counts c

(* Requests *)

let alloc = Wire.Alloc { id = 1; device = 0; memory = `Device; bytes = 4096 }

(* A header of 9 bytes, the kind, id, device, memory and bytes; the answer's:
   the header, 0 and the [bool]. *)
let request_bytes = 9 + 1 + 8 + 8 + 1 + 8
let answer_bytes = 9 + 1 + 1
