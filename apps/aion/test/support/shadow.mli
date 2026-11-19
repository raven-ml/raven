(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Running error bounds for {!Aion.Spectrum}'s float32 preparation.

    Each function here repeats one step of the codec's preparation, with the
    same nx calls on the same values, and bounds each element's distance from
    the step's exact value on exact inputs, the spectrum's float32 data: a
    {e shadowed} value is the computed tensor [x] with a tensor [e] of its
    bounds. A step adds two terms to the bound of its input: the rounding nx.mli
    allows the step itself, at the operands it is given, and the input's bound
    carried through the step to first order, as running error analysis does
    (Higham, {e Accuracy and Stability of Numerical Algorithms}, 2nd ed., §3.3).

    The roundings are nx.mli's worst-case bounds: a sum within [γ(β(n)) Σ|x|],
    [sqrt] correctly rounded and taken within 2 ulps, [asinh] 2 ulps, [log10] 1,
    [pow] 4, an arithmetic operation 1/2. The bounds are computed in float32,
    each step's multiplied by [1 + 2{^-12}], more than the relative error of its
    own float32 arithmetic.

    The network after the preparation is checked by the rounding ensemble
    ([Spread]), since worst-case bounds through it grow without use. *)

type 'd v = { x : 'd Nx.float32_t; e : 'd Nx.float32_t }
(** The type for shadowed values: [x] and its bounds [e], of one shape. *)

val normalisation : (Nx.float32_elt, 'd) Aion.Spectrum.spectrum -> 'd v
(** [normalisation x] is {!Aion.Spectrum.normalisation}[ x], shadowed. *)

val scale : 'd Nx.float32_t -> 'd v
(** [scale v] is {!Aion.Spectrum.scale}[ v] of the exact [v], shadowed. *)

val prepare :
  (Nx.float32_elt, 'd) Aion.Spectrum.t ->
  (Nx.float32_elt, 'd) Aion.Spectrum.spectrum ->
  'd v * 'd Nx.int64_t
(** [prepare t x] is {!Aion.Spectrum.prepare}[ t x], shadowed, with the bounds
    of the normalisation's token's value at [0]: they hold where the token is
    the exact one.

    Raises [Failure] if its values are not {!Aion.Spectrum.prepare}'s. *)
