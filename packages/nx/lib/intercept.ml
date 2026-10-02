(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The gate: how many interceptions are live, on any domain. While none is,
   nothing performs an effect to find one. The count is global because a
   suspended fiber may resume on another domain. *)
let intercepts = Atomic.make 0
let intercepting () = Atomic.get intercepts > 0

(* Interception

   [intercept i f] runs [f] with every operation its fiber performs and [i]
   claims delivered to [i.run], which runs outside [f]'s handlers: the
   operations [i.run] issues reach the enclosing interpretation. An operation
   [i] does not claim reaches the enclosing interpretation as performed. The
   gate is raised for the extent of [f], however it ends. *)

type interpreter = { run : 'r. 'r Op.t -> 'r; claims : 'r. 'r Op.t -> bool }
type _ Effect.t += E_op : 'r Op.t -> 'r Effect.t | E_intercepted : bool Effect.t

let intercept i f =
  Atomic.incr intercepts;
  Fun.protect ~finally:(fun () -> Atomic.decr intercepts) @@ fun () ->
  let effc : type c a.
      c Effect.t -> ((c, a) Effect.Deep.continuation -> a) option = function
    | E_op op -> (
        match i.claims op with
        | false -> None
        | true ->
            Some
              (fun k ->
                match i.run op with
                | v -> Effect.Deep.continue k v
                | exception e ->
                    let bt = Printexc.get_raw_backtrace () in
                    Effect.Deep.discontinue_with_backtrace k e bt)
        | exception e ->
            let bt = Printexc.get_raw_backtrace () in
            Some (fun k -> Effect.Deep.discontinue_with_backtrace k e bt))
    | E_intercepted -> Some (fun k -> Effect.Deep.continue k true)
    | _ -> None
  in
  Effect.Deep.match_with f () { retc = Fun.id; exnc = raise; effc }

(* Whether the calling fiber is inside an interception, outside its [run]. *)
let intercepted () =
  intercepting ()
  &&
  let e = E_intercepted in
  match Effect.perform e with
  | b -> b
  | exception Effect.Unhandled e' when Obj.repr e' == Obj.repr e -> false

(* [perform op] delivers [op] to the interception around the caller, and answers
   it directly when there is none: only an unhandled perform of this very effect
   falls back. *)
let perform : type r. r Op.t -> r =
 fun op ->
  let e = E_op op in
  match Effect.perform e with
  | v -> v
  | exception Effect.Unhandled e' when Obj.repr e' == Obj.repr e ->
      Dispatch.direct op

(* [eval op] is [op] in the current interpretation. *)
let eval op = if intercepting () then perform op else Dispatch.direct op
