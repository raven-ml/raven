(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The integer in the file [path], as Linux writes [vendor] and [class]:
   ["0x10de\n"]; [None] if the file cannot be read. *)
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

(* Linux spells a bus address in lowercase hexadecimal, the domain in four
   digits or more. *)
let hex n s =
  String.length s >= n
  && String.for_all (function '0' .. '9' | 'a' .. 'f' -> true | _ -> false) s

let spelled bus =
  match String.split_on_char ':' bus with
  | [ domain; b; df ] -> (
      match String.split_on_char '.' df with
      | [ d; f ] ->
          hex 4 domain
          && String.length b = 2
          && hex 2 b
          && String.length d = 2
          && hex 2 d
          && String.length f = 1
          && hex 1 f
      | _ -> false)
  | _ -> false

let known root bus =
  if not (spelled bus) then
    Error (Printf.sprintf "%S is no PCI bus address, expected DDDD:BB:DD.F" bus)
  else if
    not (Sys.file_exists (Filename.concat root ("sys/bus/pci/devices/" ^ bus)))
  then Error (Printf.sprintf "%s names no PCI function of the machine" bus)
  else Ok ()

(* A function whose files cannot be read, as while the kernel removes it, is
   left out, as rig.pci leaves it out, so that one card is listed by every
   path or by none. *)
let gpus root =
  let dir = Filename.concat root "sys/bus/pci/devices" in
  let gpu bus =
    let file f = read (Filename.concat (Filename.concat dir bus) f) in
    match (file "vendor", file "class") with
    | Some vendor, Some class_ -> Rig_nv.is_gpu ~vendor ~class_
    | _ -> false
  in
  if not (Sys.file_exists dir) then Ok []
  else
    match Sys.readdir dir with
    | exception Sys_error e -> Error e
    | names ->
        let gpus = List.filter gpu (Array.to_list names) in
        Ok (List.sort (fun a b -> compare (key a, a) (key b, b)) gpus)
