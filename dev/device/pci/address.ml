(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Bus addresses, which Machine exports and Sysfs sorts by. *)

let v ~domain ~bus ~device ~fn =
  Printf.sprintf "%04x:%02x:%02x.%x" domain bus device fn

(* The numbers of the bus address [a], spelled "DDDD:BB:DD.F". *)
let numbers a =
  let invalid () = invalid_arg (Printf.sprintf "%S is no PCI bus address" a) in
  let num s =
    if
      s <> ""
      && String.length s <= 8
      && String.for_all Char.Ascii.is_hex_digit s
    then int_of_string ("0x" ^ s)
    else invalid ()
  in
  match String.split_on_char ':' a with
  | [ domain; bus; df ] -> (
      match String.split_on_char '.' df with
      | [ device; fn ] -> (num domain, num bus, num device, num fn)
      | _ -> invalid ())
  | _ -> invalid ()

let compare a b = Stdlib.compare (numbers a) (numbers b)
