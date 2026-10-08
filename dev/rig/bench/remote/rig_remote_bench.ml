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

external rail_run : Rig_remote_abi.area -> Rig_remote_abi.area -> int -> int
  = "rig_remote_bench_rail_run"

(* Requests *)

let alloc = Wire.Alloc { id = 1; device = 0; memory = `Device; bytes = 4096 }

(* A header of 9 bytes, the kind, id, device, memory and bytes; the answer's:
   the header, 0 and the [bool]. *)
let request_bytes = 9 + 1 + 8 + 8 + 1 + 8
let answer_bytes = 9 + 1 + 1
