(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Failures in data. Decoders raise [Fail] where a check fails, and each
   public function turns it into its [Error]. *)

exception Fail of string

let fail fmt = Printf.ksprintf (fun s -> raise (Fail s)) fmt
let ok_or_fail = function Ok x -> x | Error e -> raise (Fail e)

(* [catch f] is [Ok (f ())], or the [Error] of a failed check or of a read of a
   file that ended or vanished since it was opened. *)
let catch f =
  match f () with
  | x -> Ok x
  | exception Fail e -> Error e
  | exception Sys_error e -> Error e

(* Places *)

(* Where a failure is: a file's name, then the places inside it, in order, as
   in "f.fits: HDU 1 (SCI), card 7 (BUNIT)". *)
type place = { name : string; parts : string list }

let nowhere = { name = ""; parts = [] }
let file name = { name; parts = [] }
let sub p part = { p with parts = p.parts @ [ part ] }

let place_string { name; parts } =
  match (name, parts) with
  | "", ps -> String.concat ", " ps
  | n, [] -> n
  | n, ps -> n ^ ": " ^ String.concat ", " ps

let msg p what = match place_string p with "" -> what | s -> s ^ ": " ^ what
let fail_at p fmt = Printf.ksprintf (fun s -> raise (Fail (msg p s))) fmt

(* Checked arithmetic on sizes. *)
let mul a b =
  if a < 0 || b < 0 then fail "a negative size"
  else if a <> 0 && b > max_int / a then None
  else Some (a * b)

let add a b = if a > max_int - b then None else Some (a + b)
