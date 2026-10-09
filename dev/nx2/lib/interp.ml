(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Value

let invalid_argf fmt = Format.kasprintf invalid_arg fmt

(* Each domain's next start and its count of live [Extent] interpretations. *)
type domain = { mutable next : int; extents : int Atomic.t }

let here = Domain.DLS.new_key (fun () -> { next = 0; extents = Atomic.make 0 })

(* Live [Extent] interpretations on every domain: with none, an operation reads
   this and not its domain's count, a domain-local read it saves. *)
let all_extents = Atomic.make 0

let quiet () =
  Atomic.get all_extents = 0 || Atomic.get (Domain.DLS.get here).extents = 0

(* The innermost [Extent] interpretation around the calling fiber that may
   receive an operation. *)
type _ Effect.t += Extent_of : interpretation option Effect.t

let handler i =
  {
    Effect.Deep.effc =
      (fun (type a) (e : a Effect.t) ->
        match e with
        | Extent_of when (not i.running) && i.domain = Domain.self () ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                Effect.Deep.continue k (Some i))
        | _ -> None);
  }

let interpret ~name reach
    (rule : 'r. interpretation -> by:string -> 'r prim -> 'r) f =
  let d = Domain.DLS.get here in
  let start = d.next in
  d.next <- start + 1;
  let i =
    {
      name;
      reach;
      rule;
      start;
      domain = Domain.self ();
      extents = d.extents;
      live = Atomic.make true;
      running = false;
    }
  in
  match reach with
  | Values ->
      Fun.protect ~finally:(fun () -> Atomic.set i.live false) (fun () -> f i)
  | Extent ->
      (* The count is the start domain's, decremented there whichever domain the
         fiber ends on. *)
      Atomic.incr all_extents;
      Atomic.incr i.extents;
      Fun.protect
        ~finally:(fun () ->
          Atomic.set i.live false;
          Atomic.decr i.extents;
          Atomic.decr all_extents)
        (fun () -> Effect.Deep.try_with f i (handler i))

let traced i form payload = Traced { form; owner = i; payload }

let payload (type v s d) i (x : (v, s, d) t) =
  match x with
  | Traced { owner; payload; _ } when owner == i -> Some payload
  | Traced _ | Array _ | Shards _ | Donated _ | Deferred _ -> None

let owner (type v s d) (x : (v, s, d) t) =
  match x with
  | Traced { owner; _ } -> Some owner
  | Array _ | Shards _ | Donated _ | Deferred _ -> None

let later a b =
  if a.domain <> b.domain then
    invalid_argf "Nx.Prim.later: %s and %s started on two domains" a.name b.name;
  a.start > b.start

let check ~by i =
  if not (Atomic.get i.live) then
    invalid_argf "%s: a value of %s, used after %s returned" by i.name i.name;
  if i.domain <> Domain.self () then
    invalid_argf "%s: a value of %s, used on another domain than its own" by
      i.name;
  if i.running then
    invalid_argf "%s: %s's rule applied an operation to its own value" by i.name

let innermost a b =
  match (a, b) with
  | None, x | x, None -> x
  | Some i, Some j -> if j.start > i.start then b else a

let receiver ~by op =
  let (Prim.Operands xs) = Prim.operands op in
  let best =
    List.fold_left
      (fun best (Any x) ->
        match x with
        | Traced { owner; _ } ->
            check ~by owner;
            innermost best (Some owner)
        | Array _ | Shards _ | Donated _ | Deferred _ -> best)
      None xs
  in
  if quiet () then best
  else
    match Effect.perform Extent_of with
    | e -> innermost best e
    | exception Effect.Unhandled Extent_of -> best

let apply i ~by op =
  let was = i.running in
  i.running <- true;
  Fun.protect ~finally:(fun () -> i.running <- was) (fun () -> i.rule i ~by op)

let expanding i f =
  let was = i.running in
  i.running <- false;
  Fun.protect ~finally:(fun () -> i.running <- was) f
