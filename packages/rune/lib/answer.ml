(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A handler answers the effect that asked: its case for an effect is a rule
   computing the answer, and [deliver run k] resumes the performer of [k] with
   the value of [run ()], or with the exception it raises. OCaml sends an
   exception raised in a handler's case to whoever installed the handler, past
   the performer's own handlers and finalisers; delivered, it raises where the
   effect was performed, as it would with no handler. Only [run] is guarded: an
   exception the performer raises once resumed leaves through [continue]
   untouched. A performer that falls back when its effect is unhandled matches
   [Effect.Unhandled] of its own effect: another's is a rule's error. *)
let deliver run k =
  match run () with
  | v -> Effect.Deep.continue k v
  | exception e -> Effect.Deep.discontinue k e
