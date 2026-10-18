(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Def

(* Devices *)

type t = device

let host = Dev.host
let name (d : t) = d.name
let host_of = Dev.host_of
let arch d = d.arch
let computes d =
  match d.kind with Host | Driver _ -> true | Io _ -> false
let runs_on_host d = Dev.is_host d || d.memory_device
let alignment d = d.alignment
let reaches = Dev.reaches
let budget d = d.budget
let allocated = Memory.allocated
let set_budget = Memory.set_budget
let free_cache = Memory.free_cache

let capability (type a) d (k : a Type.Id.t) : a option =
  match d.capability with
  | Some (Rig_edge.Capability (k', c)) -> (
      match Type.Id.provably_equal k' k with
      | Some Type.Equal -> Some c
      | None -> None)
  | None -> None

module Local = struct
  type 'a key = 'a Dev.key

  let new_key = Dev.new_key
  let get = Dev.local
end

let equal = ( == )
let pp ppf (d : t) = Format.pp_print_string ppf d.name

(* Timeline *)

module Point = struct
  type t = int

  let value = Point.value

  let device p =
    let d = Dev.of_index (Point.index p) in
    if Dev.owns d p then Some d else None

  let wait p =
    Dev.check_held "Point.wait";
    Dev.wait_point 0 true p

  let reached = Dev.reached_point

  let pp ppf p =
    match device p with
    | Some d -> Format.fprintf ppf "%s:%d" d.name (value p)
    | None -> Format.fprintf ppf "ended:%d" (value p)
end

let submitted = Dev.submitted
let signaled = Dev.signaled

exception Lost = Dev.Lost

let lost = Dev.lost
let close = Dev.close
let fail = Dev.fail
let failure = Dev.failure

exception Out_of_memory = Dev.Out_of_memory

(* Memory and work *)

type access = Def.access = Read | Read_write

module Buffer = struct
  include Buffer

  let read = Claim.read_bytes
  let write = Claim.write_bytes
  let copy = Copy.copy
  let blit_from_string = Copy.blit_from_string
  let blit_to_bytes = Copy.blit_to_bytes
  let fill = Copy.fill
  let free = Claim.free
end

module Claim = Claim
module Submission = Submission

let submit = Submission.submit

module Image = Image
module Profile = Profile

let open_ ?machine (o : 'a Rig_edge.opener) =
  Dev.open_driver o.driver ~machine:(Dev.machine_of machine) ~name:o.name o.make

let open_host ~machine (o : 'a Rig_edge.opener) =
  Dev.open_driver o.driver ~machine:(Hosts machine) ~name:o.name o.make

let open_io ?machine (o : 'a Rig_edge.io_opener) =
  Dev.open_io o.io ~machine:(Dev.machine_of machine) ~name:o.name o.make

let memory_device = Memory_device.open_
