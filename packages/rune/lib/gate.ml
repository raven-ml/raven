(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Tracing gate shared by the differentiation handlers: [no_grad] and [detach]
   turn interception off for a scope, in both reverse and forward mode. Handler
   callbacks run outside the continuation containing an inner [no_grad], so a
   private effect query from those callbacks cannot recover its tracing state.
   Scopes belong to a systhread within its domain; overlapping fibers on that
   same thread still require a separate scope mechanism. *)

module Threads = Map.Make (Int)

type state = { enabled : bool; transform_depth : int }

let default = { enabled = true; transform_depth = 0 }
let scopes = Domain.DLS.new_key (fun () -> Atomic.make Threads.empty)

let current () =
  let active = Atomic.get (Domain.DLS.get scopes) in
  if Threads.is_empty active then default
  else
    Option.value (Threads.find_opt (Thread.id (Thread.self ())) active)
      ~default

let with_state update f =
  let scopes = Domain.DLS.get scopes and thread = Thread.id (Thread.self ()) in
  let previous =
    Option.value (Threads.find_opt thread (Atomic.get scopes)) ~default
  in
  let rec install state =
    let before = Atomic.get scopes in
    let after =
      if state.enabled && state.transform_depth = 0 then Threads.remove thread before
      else Threads.add thread state before
    in
    if not (Atomic.compare_and_set scopes before after) then install state
  in
  install (update previous);
  Fun.protect f ~finally:(fun () -> install previous)

let enabled () = (current ()).enabled
let without_tracing f = with_state (fun state -> { state with enabled = false }) f

(* [Rune.jit] steps aside when a transformation is observing the operations: a
   compiled replay performs no effects, so running one under grad/vmap/debug
   would hide the computation from the enclosing handler. *)
let with_transform f =
  with_state (fun state -> { state with transform_depth = state.transform_depth + 1 }) f

let transforming () = (current ()).transform_depth > 0
