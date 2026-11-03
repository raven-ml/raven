(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A composite's extent on a domain, and its arguments' nodes, each keyed by
   the argument's physical identity. Its end may run on another domain,
   which cannot pop it here: [ended] marks it, and the domain drops it once
   it is on top. *)
type frame = {
  by : string;
  depth : int;
  ended : bool Atomic.t;
  mutable args : (Obj.t * node) list;
}

(* How a value derives: it is an argument, which may itself derive from an
   outer composite's, or an operation made it by [rule] from its operands,
   each with its shape where it derives from an argument. *)
and node =
  | Argument of {
      frame : frame;
      name : string;
      place : string;
      at : unit Devices.placement;
      origin : node option;
    }
  | Made of {
      rule : Route.rule;
      operands : (int iarray * node) option array;
    }

(* The open frames, innermost first, and each noted value's node, newest
   first, keyed by the value itself, which the key does not keep alive. *)
type state = {
  mutable frames : frame list;
  mutable notes : (Obj.t, node) Ephemeron.K1.t list;
}

let state = Domain.DLS.new_key (fun () -> { frames = []; notes = [] })

(* Drops the ended frames on top, and the notes once no frame is open. *)
let rec prune s =
  match s.frames with
  | f :: rest when Atomic.get f.ended ->
      s.frames <- rest;
      prune s
  | [] -> s.notes <- []
  | _ :: _ -> ()

let idle () =
  match (Domain.DLS.get state).frames with [] -> true | _ :: _ -> false

type value = Value : ('v, 's, 'd) Value.t -> value

(* The physical identity of a value. *)
let key (Value x) = Obj.repr x

(* The node keyed [k]: an open frame's argument's, innermost first, else
   the newest note's. An operation's operands are mostly arguments and
   recent results, which this order reaches first. Loops: a miss allocates
   nothing. *)
let rec in_args k = function
  | [] -> None
  | (k', n) :: rest -> if k' == k then Some n else in_args k rest

let rec in_frames k = function
  | [] -> None
  | f :: rest -> (
      match in_args k f.args with
      | Some _ as n -> n
      | None -> in_frames k rest)

let rec in_notes k = function
  | [] -> None
  | e :: rest -> (
      match Ephemeron.K1.query e k with
      | Some _ as n -> n
      | None -> in_notes k rest)

let find_key s k =
  match in_frames k s.frames with
  | Some _ as n -> n
  | None -> in_notes k s.notes

let find s v = find_key s (key v)

let call ~by args f =
  let s = Domain.DLS.get state in
  prune s;
  let frame =
    { by; depth = List.length s.frames; ended = Atomic.make false; args = [] }
  in
  frame.args <-
    List.map
      (fun (name, place, at, v) ->
        (key v, Argument { frame; name; place; at; origin = find s v }))
      args;
  s.frames <- frame :: s.frames;
  let finally () =
    Atomic.set frame.ended true;
    prune (Domain.DLS.get state)
  in
  Fun.protect ~finally f

let source x =
  let s = Domain.DLS.get state in
  prune s;
  match s.frames with
  | [] -> None
  | _ :: _ -> find_key s (Obj.repr x)

let derives x = Option.is_some (source x)
let made rule operands = Made { rule; operands }

let note y n =
  let s = Domain.DLS.get state in
  s.notes <- Ephemeron.K1.make (Obj.repr y) n :: s.notes

(* Walks back from [x]'s node, each operand's grid moved back through the
   rule that made its result, and keeps the argument of the outermost open
   frame, the first in operand order, whose grid is not where it lies. A
   node is visited once. *)
let named x q =
  let s = Domain.DLS.get state in
  match find s x with
  | None -> None
  | Some n ->
      let best = ref None and seen = ref [] in
      let rec walk n q =
        if not (List.memq n !seen) then begin
          seen := n :: !seen;
          match n with
          | Argument { frame; name; place; at; origin } ->
              let read = if Devices.partial at then q else Grid.closed q in
              let outer =
                match !best with
                | Some (d, _) -> frame.depth < d
                | None -> true
              in
              if
                outer
                && (not (Atomic.get frame.ended))
                && not (Grid.equal read (Devices.grid at))
              then begin
                let n = { Route.by = frame.by; name; place; at; read } in
                best := Some (frame.depth, n)
              end;
              Option.iter (fun o -> walk o q) origin
          | Made { rule; operands } ->
              Array.iteri
                (fun i -> function
                  | Some (shape, n) ->
                      walk n (Route.backward rule i (Iarray.to_array shape) q)
                  | None -> ())
                operands
        end
      in
      walk n q;
      Option.map snd !best
