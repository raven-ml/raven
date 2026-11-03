(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A composite's extent on a domain. Its end may run on another domain, which
   cannot pop it here: [ended] marks it, and the domain drops it once it is
   on top. *)
type frame = { by : string; depth : int; ended : bool Atomic.t }

(* How a value derives: it is an argument, which may itself derive from an
   outer composite's, or an operation made it by [rule] from its operands,
   each with its shape where it derives from an argument. *)
type node =
  | Argument of {
      frame : frame;
      name : string;
      at : unit Devices.placement;
      origin : node option;
    }
  | Made of { rule : Route.rule; operands : (int array * node) option array }

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

let find s v =
  let k = key v in
  List.find_map (fun e -> Ephemeron.K1.query e k) s.notes

let note s v n = s.notes <- Ephemeron.K1.make (key v) n :: s.notes

let call ~by args f =
  let s = Domain.DLS.get state in
  prune s;
  let frame = { by; depth = List.length s.frames; ended = Atomic.make false } in
  List.iter
    (fun (name, at, v) ->
      note s v (Argument { frame; name; at; origin = find s v }))
    args;
  s.frames <- frame :: s.frames;
  let finally () =
    Atomic.set frame.ended true;
    prune (Domain.DLS.get state)
  in
  Fun.protect ~finally f

let record rule operands results =
  let s = Domain.DLS.get state in
  prune s;
  match s.frames with
  | [] -> ()
  | _ :: _ ->
      let derived (v, shape) = Option.map (fun n -> (shape, n)) (find s v) in
      let operands = Array.map derived operands in
      if Array.exists Option.is_some operands then begin
        let n = Made { rule; operands } in
        List.iter (fun v -> note s v n) results
      end

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
          | Argument { frame; name; at; origin } ->
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
              then
                best :=
                  Some (frame.depth, { Route.by = frame.by; name; at; read });
              Option.iter (fun o -> walk o q) origin
          | Made { rule; operands } ->
              Array.iteri
                (fun i -> function
                  | Some (shape, n) -> walk n (Route.backward rule i shape q)
                  | None -> ())
                operands
        end
      in
      walk n q;
      Option.map snd !best
