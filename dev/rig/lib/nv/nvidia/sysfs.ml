(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The integer in the file [path], as Linux writes [vendor] and [class]:
   ["0x10de\n"]. *)
let read path =
  match In_channel.with_open_bin path In_channel.input_line with
  | Some line -> int_of_string_opt (String.trim line)
  | None -> None
  | exception Sys_error _ -> None

(* A bus address "dddd:bb:dd.f" as integers, for sorting: a domain may have more
   than four digits. *)
let key bus =
  match String.split_on_char ':' bus with
  | [ domain; b; df ] -> (
      match String.split_on_char '.' df with
      | [ d; f ] ->
          List.map (fun x -> int_of_string_opt ("0x" ^ x)) [ domain; b; d; f ]
      | _ -> [])
  | _ -> []

let gpus root =
  let dir = Filename.concat root "sys/bus/pci/devices" in
  match Sys.readdir dir with
  | exception Sys_error _ -> []
  | names ->
      let gpu bus =
        let file f = read (Filename.concat (Filename.concat dir bus) f) in
        match (file "vendor", file "class") with
        | Some vendor, Some class_ -> Rig_nv.is_gpu ~vendor ~class_
        | _ -> false
      in
      List.filter gpu (Array.to_list names)
      |> List.sort (fun a b -> compare (key a, a) (key b, b))
