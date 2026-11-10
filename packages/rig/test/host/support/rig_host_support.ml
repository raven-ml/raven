(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type words = (int64, Bigarray.int64_elt, Bigarray.c_layout) Bigarray.Array1.t

let words n =
  let b = Bigarray.Array1.create Bigarray.int64 Bigarray.c_layout (max n 1) in
  Bigarray.Array1.fill b 0L;
  b

external address : ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> int
  = "rig_host_test_address"

external host_machine : unit -> string = "rig_host_test_machine"
external executable : int -> bool = "rig_host_test_executable"

external empty_job : threads:int -> total:int -> chunks:int -> unit
  = "rig_host_test_empty_job"

external workers : unit -> int = "rig_host_test_workers"

let machine = host_machine ()

let fixture ~dir f =
  let file m = Filename.concat dir (f ^ "_" ^ m ^ ".o") in
  let windows = file "x86_64_windows" in
  let path =
    if Sys.win32 && machine = "x86_64" && Sys.file_exists windows then windows
    else file machine
  in
  In_channel.with_open_bin path In_channel.input_all

(* From 1: a fixture's padding is zero. *)
let made = Atomic.make 1

let fresh o =
  let b = Bytes.of_string o in
  Bytes.set_int32_le b 9 (Int32.of_int (Atomic.fetch_and_add made 1));
  Bytes.unsafe_to_string b
