(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The process's files *)

external files : unit -> int = "rig_nv_nvidia_test_files"
external limit : unit -> int = "rig_nv_nvidia_test_limit"
external set_limit : int -> unit = "rig_nv_nvidia_test_set_limit"
external limit_for : int -> int = "rig_nv_nvidia_test_limit_for"

let with_limit n f =
  let soft = limit () in
  set_limit n;
  Fun.protect ~finally:(fun () -> set_limit soft) f

(* The process's addresses *)

external occupy : int -> int -> bool = "rig_nv_nvidia_test_occupy"
external vacate : int -> int -> unit = "rig_nv_nvidia_test_vacate"

(* The GPU's mappings *)

(* The kernel path reserves this interval, as Rig_nv_nvidia specifies. The
   process's allocator and sanitizer can grow their own mappings after free. *)
let gpu_base = Int64.shift_left 384L 30
let gpu_limit = Int64.shift_left 1L 40

let mappings () =
  let belongs line =
    let in_range =
      Scanf.sscanf line "%Lx-%Lx" (fun first limit ->
          first < gpu_limit && limit > gpu_base)
    in
    let device =
      String.split_on_char ' ' line
      |> List.exists (fun field ->
             String.starts_with ~prefix:"nvidia" (Filename.basename field))
    in
    in_range || device
  in
  In_channel.with_open_text "/proc/self/maps" In_channel.input_lines
  |> List.filter belongs
