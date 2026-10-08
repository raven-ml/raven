(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

external set_open_files : int -> int = "rig_disk_test_set_open_files"
external open_files : unit -> int = "rig_disk_test_open_files"
external set_file_size : int -> int = "rig_disk_test_set_file_size"
external drop_pages : string -> int = "rig_disk_test_drop_pages"
external heap_bytes : unit -> int = "rig_disk_test_heap_bytes"

let heap_bytes () = match heap_bytes () with -1 -> None | n -> Some n

let descriptors () =
  if Sys.win32 then None else Some (Array.length (Sys.readdir "/dev/fd"))
