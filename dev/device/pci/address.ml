(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt
let v ~domain ~bus ~device ~fn = strf "%04x:%02x:%02x.%x" domain bus device fn

(* Each number is one to eight hexadecimal digits, so that nothing else, such as
   a path's "/" or ".", passes. *)
let numbers a =
  let num s =
    if
      s <> ""
      && String.length s <= 8
      && String.for_all Char.Ascii.is_hex_digit s
    then Some (int_of_string ("0x" ^ s))
    else None
  in
  match String.split_on_char ':' a with
  | [ domain; bus; df ] -> (
      match String.split_on_char '.' df with
      | [ device; fn ] -> (
          match (num domain, num bus, num device, num fn) with
          | Some d, Some b, Some v, Some f -> Some (d, b, v, f)
          | _ -> None)
      | _ -> None)
  | _ -> None

let numbers_exn a =
  match numbers a with
  | Some n -> n
  | None ->
      invalid_argf
        "Machine.compare_address: %S is no PCI bus address, expected \
         DDDD:BB:DD.F"
        a

let compare a b = Stdlib.compare (numbers_exn a) (numbers_exn b)
