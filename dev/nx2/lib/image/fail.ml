(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* [rename ~by m] raises [Invalid_argument] for the message [m] of another
   function, naming [by] in its place and keeping its reason. *)
let rename ~by m =
  match String.index_opt m ':' with
  | Some i -> invalid_arg (by ^ String.sub m i (String.length m - i))
  | None -> invalid_arg (by ^ ": " ^ m)

let reading ~by read f =
  let started = ref false in
  try
    read (fun a ->
        started := true;
        f a)
  with Invalid_argument m when not !started -> rename ~by m
