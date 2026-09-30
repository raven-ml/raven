(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Traced functions and the values their graphs compute.

    A function is traced under a scope of {!Rune_next.Lower}, as a compiled call
    traces it. The value of a traced result is its graph evaluated by tolk's
    reference interpreter ([Tensors]), each capture's storage holding the bytes
    the scope binds to it: what the lowering means, before any compiler sees it.
*)

open Rune_next

val host : Nx.Device.t -> Tolk_next.Renderer.t
(** [host d] is the host's renderer, whatever [d]: the programs of every device
    of a test are rendered for the host. *)

val scope :
  ?renderer:(Nx.Device.t -> Tolk_next.Renderer.t) -> unit -> Lower.scope
(** [scope ~renderer ()] is a new scope with [renderer] (defaults to {!host}).
*)

val within : Lower.scope -> (unit -> 'a) -> 'a
(** [within s f] is [f ()] traced under [s]. *)

val trace :
  ?renderer:(Nx.Device.t -> Tolk_next.Renderer.t) ->
  (unit -> 'a) ->
  Lower.scope * 'a
(** [trace ~renderer f] is [f ()] traced under a new scope, and that scope. *)

val argument : Lower.scope -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [argument s x] is the traced parameter that stands for the host value [x] in
    [s] ({!Lower.param}), in a slot of its own, which {!value} binds to [x]. *)

val value : Lower.scope -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [value s y] is the value of the traced [y] as a host value: its graph
    evaluated with the captures and the arguments of [s] bound, read from its
    first device.

    Raises [Invalid_argument] if [y] is not traced, and as [Tensors.eval] does.
*)

val exact : ?__POS__:Windtrap.pos -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> unit
(** [exact expected actual] asserts that [actual] has the dtype, the shape and
    the bits of each element of [expected]: [-0.] is not [0.], and every NaN
    equals every NaN, since arithmetic leaves a NaN's sign and payload
    unspecified. Compare bit patterns through a bitcast where they matter. *)

val ulps :
  ?__POS__:Windtrap.pos ->
  budget:int ->
  expected:(float, 'b) Nx.t ->
  (float, 'b) Nx.t array ->
  (float, 'b) Nx.t ->
  unit
(** [ulps ~budget ~expected inputs actual] asserts that each element of
    [actual], a float of 16, 32 or 64 bits, is within [budget] units in the last
    place of the element of [expected] at the same position, the correctly
    rounded result. NaNs and infinities must be equal. The failure names the
    worst element and the elements of [inputs] it was computed from. *)

val contents : Rune_next.Lower.scope -> (int * Tolk_next.Dtype.value array) list
(** [contents s] is the elements of each storage and each argument [s] binds, by
    slot. *)

val of_const : ('a, 'b) Nx_dtype.t -> Tolk_next.Dtype.const -> 'a
(** [of_const dt c] is the element [c] of [dt].

    Raises [Invalid_argument] if [c] is not of [dt]'s kind. *)
