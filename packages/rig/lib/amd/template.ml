(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Rig_amd_abi

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

type form = Packet | Launch | Program

let name = function
  | Packet -> "Rig_amd.make: a packet template"
  | Launch -> "Rig_amd.entry: a launch"
  | Program -> "Rig_amd.make: a counters program"

(* Each form's bounds in rig_amd_stubs.h, which C refuses past with a fixed
   message: its most words, its most holes and the arguments a hole reads. *)
external template_words : unit -> int = "caml_rig_amd_template_words"
[@@noalloc]

external template_holes : unit -> int = "caml_rig_amd_template_holes"
[@@noalloc]

external template_args : unit -> int = "caml_rig_amd_template_args"
[@@noalloc]

external launch_words : unit -> int = "caml_rig_amd_launch_words" [@@noalloc]
external launch_holes : unit -> int = "caml_rig_amd_launch_holes" [@@noalloc]
external launch_args : unit -> int = "caml_rig_amd_launch_args" [@@noalloc]
external program_words : unit -> int = "caml_rig_amd_program_words" [@@noalloc]
external program_holes : unit -> int = "caml_rig_amd_program_holes" [@@noalloc]

let most_words = function
  | Packet -> template_words ()
  | Launch -> launch_words ()
  | Program -> program_words ()

let most_holes = function
  | Packet -> template_holes ()
  | Launch -> launch_holes ()
  | Program -> program_holes ()

(* A program's holes read one argument, the address its records start at. *)
let most_args = function
  | Packet -> template_args ()
  | Launch -> launch_args ()
  | Program -> 1

let hole form nwords arg b (at, (w : _ Packet.word)) =
  let t, n =
    match w with
    | W32 t -> (t, 1)
    | W64 t -> (t, 2)
    | Dword _ -> assert false (* a hole holds a term *)
  in
  let refuse why = invalid_argf "%s's hole %s" (name form) why in
  let arg, offset = arg t.Packet.value in
  if arg < 0 || arg >= most_args form then refuse "reads an unknown argument";
  if at < 0 || at + n > nwords then refuse "lies outside its words";
  if t.shift < 0 || t.shift > 63 then refuse "shifts outside 0 to 63";
  let put v = Buffer.add_int64_le b v in
  List.iter put
    [
      Int64.of_int at;
      Int64.of_int n;
      Int64.of_int arg;
      Int64.add t.add offset;
      Int64.of_int t.shift;
      t.set;
    ]

let flatten form known arg p =
  let words, holes = Packet.template known p in
  if String.length words > 4 * most_words form then
    invalid_argf "%s exceeds %d words" (name form) (most_words form);
  if List.length holes > most_holes form then
    invalid_argf "%s has more than %d holes" (name form) (most_holes form);
  let b = Buffer.create 256 in
  List.iter (hole form (String.length words / 4) arg b) holes;
  (words, Buffer.contents b)
