(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Tracing gate shared by the differentiation handlers: [no_grad] and [detach]
   turn interception off for a scope, in both reverse and forward mode. *)

let enabled = ref true

let without_tracing f =
  let prev = !enabled in
  enabled := false;
  Fun.protect f ~finally:(fun () -> enabled := prev)

(* Is a transformation observing operations at this point?

   [Rune.jit] must step aside inside a transformation: a compiled replay
   performs no effects, so it would hide the computation from the enclosing
   handler. The question is asked as an effect rather than tracked in a global
   counter, because a counter cannot be maintained reliably: an exception raised
   in a handler body is re-raised in the handler's fiber, abandoning the fiber
   below it without unwinding it, so a [Fun.protect] around the installation
   site never runs its cleanup. Each transformation handler answers the probe
   from its own extent — the same idiom as [Scan.E_scan_probe]. *)

type _ Effect.t += E_transforming : bool Effect.t

let transforming () =
  match Effect.perform E_transforming with
  | b -> b
  | exception Effect.Unhandled _ -> false
