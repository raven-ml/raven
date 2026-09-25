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
   [f] again during the backward pass instead of taping it. Forward mode passes
   on the remat of [f]'s jvp; vmap passes on [f] batched. With no handler [f]
   runs. *)

type 'q call =
  | Call : {
      params_s : 'p Nx.Ptree.t;
      result_s : 'q Nx.Ptree.t;
      params : 'p;
      f : 'p -> 'q;
    }
      -> 'q call

type _ Effect.t += E_remat : 'q call -> 'q Effect.t

let run (Call { params; f; _ } as call) =
  match Effect.perform (E_remat call) with
  | y -> y
  | exception Effect.Unhandled _ -> f params
