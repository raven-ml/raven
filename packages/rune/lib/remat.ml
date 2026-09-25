(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Gradient checkpointing, communicated to the ambient handlers through effects.

   [run] performs [E_remat]. Every handler passes the call on to the enclosing
   context with [f] wrapped in itself, and none runs [f] outside itself: the
   tensors [f] captures then meet every transformation as they would without
   remat. Reverse mode wraps [f] over a scratch tape to learn whether the result
   depends on a tracked tensor and, if it does, records a tape entry that runs
   [f] again during the backward pass instead of taping it; it sets [residuals]:
   a backward pass will read the arguments. Forward mode passes on the remat of
   [f]'s jvp; vmap passes on [f] batched; jit materialises the arguments of a
   call with residuals and runs [f]. With no handler [f] runs.

   The recomputation reads its arguments through [barrier ~after:cts]: the
   identity everywhere but under jit, where it makes the arguments distinct
   graph nodes that are read only once the cotangents [cts] exist. Without it
   the recomputation traces to the nodes of the forward pass, which the graph
   shares, and the forward's intermediates stay live until the backward pass
   reads them. *)

type 'q call =
  | Call : {
      params_s : 'p Nx.Ptree.t;
      result_s : 'q Nx.Ptree.t;
      params : 'p;
      f : 'p -> 'q;
      residuals : bool;
    }
      -> 'q call

type barrier = { values : Nx.packed list; after : Nx.packed list }

type _ Effect.t +=
  | E_remat : 'q call -> 'q Effect.t
  | E_barrier : barrier -> Nx.packed list Effect.t

let run (Call { params; f; _ } as call) =
  match Effect.perform (E_remat call) with
  | y -> y
  | exception Effect.Unhandled _ -> f params

let barrier ~after values =
  match Effect.perform (E_barrier { values; after }) with
  | values -> values
  | exception Effect.Unhandled _ -> values
