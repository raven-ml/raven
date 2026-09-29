(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Pausing differentiation. [pause ()] performs [E_pause]; every reverse and
   forward interpreter around the calling fiber answers it by pausing itself,
   passing it on to those around it and answering with the function that
   resumes them all. While paused, an interpreter passes every operation on as
   it is. *)

type _ Effect.t += E_pause : (unit -> unit) Effect.t

let pause () =
  match Effect.perform E_pause with
  | resume -> resume
  | exception Effect.Unhandled E_pause -> Fun.id

(* [hold paused ()] is an interpreter's answer to [E_pause]: it counts itself
   in [paused] until the function it returns resumes it and those around it. *)
let hold paused () =
  incr paused;
  let resume = pause () in
  fun () ->
    decr paused;
    resume ()

(* [during f] is [f ()] with the differentiation around it paused. *)
let during f =
  let resume = pause () in
  Fun.protect ~finally:resume f
