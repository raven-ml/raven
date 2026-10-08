(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Def

(* Devices *)

type t = device

let host = Dev.host
let name d = d.name
let host_of = Dev.host_of
let arch d = d.arch
let computes d = not (Dev.is_io d)
let runs_on_host d = Dev.is_host d || d.memory_device
let reaches = Dev.reaches
let shares_host_memory d = Dev.reaches Dev.host d && Dev.reaches d Dev.host
let budget d = d.budget
let set_budget = Memory.set_budget
let free_cache = Memory.free_cache

let capability (type a) d (k : a Type.Id.t) : a option =
  match d.capability with
  | Some (Capability (k', c)) -> (
      match Type.Id.provably_equal k' k with
      | Some Type.Equal -> Some c
      | None -> None)
  | None -> None

let equal = ( == )
let pp ppf d = Format.pp_print_string ppf d.name

(* Timeline *)

module Point = struct
  type t = int

  let device p = Dev.of_index (Point.index p)
  let value = Point.value
  let pp ppf p = Format.fprintf ppf "%s:%d" (device p).name (value p)
end

let submitted = Dev.submitted
let signaled = Dev.word
let wait = Dev.wait

exception Lost = Dev.Lost

let lost = Dev.lost
let close = Dev.close
let fail = Dev.fail
let failure = Dev.failure

exception Out_of_memory = Dev.Out_of_memory

(* Memory and work *)

module Buffer = struct
  include Buffer

  let copy = Copy.copy
end

module Claim = Claim
module Hold = Hold
module Submission = Submission

let submit = Submission.submit

module Image = Image
module Profile = Profile

(* Drivers *)

module type Driver = Sigs.Driver
module type Io = Sigs.Io

let open_ m ?machine ?host ~name make =
  Dev.open_driver m ?machine ?host ~name make

let open_io m ?machine ~name make = Dev.open_io m ?machine ~name make
let memory_device = Memory_device.open_
