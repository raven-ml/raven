(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

external set_open_files : int -> int = "rig_disk_test_set_open_files"
external set_file_size : int -> int = "rig_disk_test_set_file_size"
external drop_pages : string -> int = "rig_disk_test_drop_pages"
external identified : string -> bool = "rig_disk_test_identified"
external fault_runs : bool -> float -> unit = "rig_disk_test_fault_reads"

let fault_reads ~wait = fault_runs true wait
let restore_reads () = fault_runs false 0.

external hold_pool : unit -> unit = "rig_disk_test_hold_pool"
external pool_held : unit -> bool = "rig_disk_test_pool_held"
external release_pool : unit -> unit = "rig_disk_test_release_pool"

(* Whether /dev/fd lists this process's descriptors: Windows has none, and
   FreeBSD's lists 0, 1 and 2 alone unless fdescfs is mounted on it. Read once,
   at start: it is the system's, and a lazy value forced from two domains at
   once raises. *)
let lists_descriptors =
  (not Sys.win32)
  &&
  let listed () = Array.length (Sys.readdir "/dev/fd") in
  let before = listed () in
  let fd = Unix.openfile "/dev/null" [ Unix.O_RDONLY; Unix.O_CLOEXEC ] 0 in
  Fun.protect
    ~finally:(fun () -> Unix.close fd)
    (fun () -> listed () = before + 1)

let descriptors () =
  if lists_descriptors then Some (Array.length (Sys.readdir "/dev/fd"))
  else None
