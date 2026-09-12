(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Reading tangents: the query a forward-mode consumer asks its enclosing
   handler.

   Code that *consumes* tangents — a curvature collector, a debugger, a metric
   that wants directional derivatives of an intermediate — needs the tangent a
   differentiation maintains for a tensor it already holds, not the
   transformation itself. The query travels the same effect channel as every
   other operation, so the innermost forward handler answers it from the store
   it already keeps: [Forward] with the single tangent, [Forward_k] with the
   [k]-lane batch. With no forward mode installed the effect is unhandled and
   the query is [None], so calling code degrades to its no-tangent path without
   having to know whether a forward mode is running at all.

   The innermost handler owns the answer *and its shape convention*: a tensor
   the innermost store does not track is a constant of that differentiation —
   [None] — rather than a question handed outward to an enclosing store, whose
   tangent would be in that scope's convention (a single tangent where a batch
   is expected). Under [no_grad] the handlers stay silent for the same reason
   every other operation is untracked there.

   A consumer should rely on the shape of the answer rather than on which
   handler produced it: a single tangent carries the tracked tensor's shape, a
   [k]-lane batch carries [k] prepended to it. Code that needs the batch form
   can check the leading axis instead of asking which transformation is
   installed, and should treat anything else as a shape error rather than
   contracting mismatched axes. *)

type _ Effect.t +=
  | E_tangent : ('a, 'b) Nx_effect.t -> ('a, 'b) Nx_effect.t option Effect.t

(* [query x] is the tangent the innermost forward mode maintains for [x]:
   [None] when no forward mode is installed, when tracing is disabled, or when
   [x] is a constant of the differentiation. *)
let query (type a b) (x : (a, b) Nx_effect.t) : (a, b) Nx_effect.t option =
  match Effect.perform (E_tangent x) with
  | dy -> dy
  | exception Effect.Unhandled _ -> None
