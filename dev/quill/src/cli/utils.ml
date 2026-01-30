(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let read_text_from_file filename =
  try In_channel.with_open_text filename In_channel.input_all
  with Sys_error msg -> failwith ("Failed to read from file: " ^ msg)

let write_text_to_file filename text =
  Out_channel.with_open_text filename (fun out_channel ->
      Out_channel.output_string out_channel text)
