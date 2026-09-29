(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let remove_if_exists path = try Sys.remove path with Sys_error _ -> ()
let prng = Domain.DLS.new_key Random.State.make_self_init

(* A fresh empty file beside [path]. It is created with [Unix], so that a
   directory that cannot be written raises [Unix.Unix_error] as the other writes
   of this library do. *)
let sibling path =
  let dir = Filename.dirname path and base = Filename.basename path in
  let rec create attempts =
    let tag = Random.State.bits (Domain.DLS.get prng) land 0xffffff in
    let name = Filename.concat dir (Printf.sprintf "%s.%06x.tmp" base tag) in
    match Unix.openfile name [ O_WRONLY; O_CREAT; O_EXCL; O_CLOEXEC ] 0o600 with
    | fd ->
        Unix.close fd;
        name
    | exception Unix.Unix_error (EEXIST, _, _) when attempts > 1 ->
        create (attempts - 1)
  in
  create 1000

let mode = 0o640

let replace temp path =
  match
    Unix.chmod temp mode;
    Unix.rename temp path
  with
  | () -> ()
  | exception exn ->
      remove_if_exists temp;
      raise exn
