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
   understands the effect applies the rule. A transformation of the wrong mode
   raises (a custom vjp is not forward-differentiable, and vice versa). When no
   handler intercepts — no differentiation in scope, or only vmap — the plain
   forward function runs at the call site, so vmap batches it like any other
   code. *)

type ('c, 'd) vjp_call =
  | Vjp_call : {
      tree : (module Differentiable.S with type t = 'p);
      params : 'p;
      fwd : 'p -> ('c, 'd) Nx.t * 'res;
      bwd : 'res -> ('c, 'd) Nx.t -> 'p;
    }
      -> ('c, 'd) vjp_call

type ('c, 'd) jvp_call =
  | Jvp_call : {
      tree : (module Differentiable.S with type t = 'p);
      params : 'p;
      f : 'p -> ('c, 'd) Nx.t;
      jvp : 'p -> 'p -> ('c, 'd) Nx.t * ('c, 'd) Nx.t;
    }
      -> ('c, 'd) jvp_call

type _ Effect.t +=
  | E_custom_vjp : ('c, 'd) vjp_call -> ('c, 'd) Nx.t Effect.t
  | E_custom_jvp : ('c, 'd) jvp_call -> ('c, 'd) Nx.t Effect.t

let custom_vjp (type c d) (module P : Differentiable.S)
    ~(fwd : P.t -> (c, d) Nx.t * 'res) ~(bwd : 'res -> (c, d) Nx.t -> P.t)
    (params : P.t) : (c, d) Nx.t =
  try
    Effect.perform
      (E_custom_vjp (Vjp_call { tree = (module P); params; fwd; bwd }))
  with Effect.Unhandled _ -> fst (fwd params)

let custom_jvp (type c d) (module P : Differentiable.S)
    ~(f : P.t -> (c, d) Nx.t) ~(jvp : P.t -> P.t -> (c, d) Nx.t * (c, d) Nx.t)
    (params : P.t) : (c, d) Nx.t =
  try
    Effect.perform
      (E_custom_jvp (Jvp_call { tree = (module P); params; f; jvp }))
  with Effect.Unhandled _ -> f params
