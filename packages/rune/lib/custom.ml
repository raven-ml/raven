(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* User-defined differentiation rules, communicated to the ambient handlers
   through effects.

   Extensible effect constructors require their payload type variables to be
   deducible from the result type, which the parameter structure and residual
   types are not. Packing the call in an ordinary GADT whose only exposed
   parameters are the output's makes the effects fully typed — no erasure.

   Dispatch is by handler stacking: the innermost transformation that
   understands the effect applies its treatment. A differentiation of the wrong
   mode raises (a custom vjp is not forward-differentiable, and vice versa);
   vmap batches the forward function, so only a differentiation inside the vmap
   applies the rule. When no handler intercepts — no transformation in scope —
   the plain forward function runs at the call site. *)

type 'q vjp_call =
  | Vjp_call : {
      params_s : 'p Nx.Ptree.t;
      result_s : 'q Nx.Ptree.t;
      params : 'p;
      fwd : 'p -> 'q * 'res;
      bwd : 'res -> 'q -> 'p;
    }
      -> 'q vjp_call

type 'q jvp_call =
  | Jvp_call : {
      params_s : 'p Nx.Ptree.t;
      result_s : 'q Nx.Ptree.t;
      params : 'p;
      f : 'p -> 'q;
      jvp : 'p -> 'p -> 'q * 'q;
    }
      -> 'q jvp_call

type _ Effect.t +=
  | E_custom_vjp : 'q vjp_call -> 'q Effect.t
  | E_custom_jvp : 'q jvp_call -> 'q Effect.t

let custom_vjp params_s result_s ~fwd ~bwd params =
  try
    Effect.perform
      (E_custom_vjp (Vjp_call { params_s; result_s; params; fwd; bwd }))
  with Effect.Unhandled _ -> fst (fwd params)

let custom_jvp params_s result_s ~f ~jvp params =
  try
    Effect.perform
      (E_custom_jvp (Jvp_call { params_s; result_s; params; f; jvp }))
  with Effect.Unhandled _ -> f params
