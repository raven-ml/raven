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

(* Depth of installed transformation handlers. [Rune.jit] consults it to step
   aside when a transformation is observing the operations: a compiled replay
   performs no effects, so running one under grad/vmap/debug would hide the
   computation from the enclosing handler. *)

let transform_depth = ref 0

let with_transform f =
  incr transform_depth;
  Fun.protect f ~finally:(fun () -> decr transform_depth)

let transforming () = !transform_depth > 0
