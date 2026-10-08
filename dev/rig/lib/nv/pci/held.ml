(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let minor_line = "Device Minor:"

let minor info =
  let of_line l =
    if not (String.starts_with ~prefix:minor_line l) then None
    else
      let n = String.length minor_line in
      int_of_string_opt (String.trim (String.sub l n (String.length l - n)))
  in
  List.find_map of_line (String.split_on_char '\n' info)

let nodes ~root bus =
  let file =
    Filename.concat root ("proc/driver/nvidia/gpus/" ^ bus ^ "/information")
  in
  match In_channel.with_open_bin file In_channel.input_all with
  | exception Sys_error _ -> []
  | info -> (
      match minor info with
      | Some n -> [ "dev/nvidia" ^ string_of_int n ]
      | None -> [])
