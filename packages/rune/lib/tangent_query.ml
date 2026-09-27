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
   it already keeps, [Forward] with the tensor's tangent. With no forward mode
   installed the effect is unhandled and the query is [None], so calling code
   degrades to its no-tangent path without having to know whether a forward
   mode is running at all.

   The innermost handler owns the answer *and its shape convention*: a tensor
   the innermost store does not track is a constant of that differentiation —
   [None] — rather than a question handed outward to an enclosing store, whose
   answer would be in that scope's convention. Under [no_grad] the handlers
   stay silent for the same reason every other operation is untracked there.

   A consumer should rely on the shape of the answer rather than on which
   handler produced it. Under the batching composition — [Vmap] around [jvp] —
   the answer is the whole [k]-lane batch on a leading axis, but the lanes are
   visible only outside the map's extent: within it [Nx.shape] reports the
   map's virtual, lane-less shape, so an in-scope consumer cannot contract the
   lane axis. Carry the tensor out (for instance in an effect payload) and read
   it there. *)

type _ Effect.t +=
  | E_tangent : ('a, 'b) Nx_effect.t -> ('a, 'b) Nx_effect.t option Effect.t

(* [query x] is the tangent the innermost forward mode maintains for [x]:
   [None] when no forward mode is installed, when tracing is disabled, or when
   [x] is a constant of the differentiation. *)
let query (type a b) (x : (a, b) Nx_effect.t) : (a, b) Nx_effect.t option =
  match Effect.perform (E_tangent x) with
  | dy -> dy
  | exception Effect.Unhandled _ -> None
