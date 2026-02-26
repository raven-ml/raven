(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Per-call auxiliary data for layers.

    A {!type:t} carries read-only tensors (attention masks, position ids,
    encoder memory) that specific layers consume during a forward pass. Most
    layers ignore the context; transformer layers read from it by well-known key
    names.

    {[
      let ctx =
        Context.empty
        |> Context.set ~name:"attention_mask" (Ptree.P mask)
        |> Context.set ~name:"token_type_ids" (Ptree.P ids)
      in
      Layer.apply model vars ~training:false ~ctx input_ids
    ]} *)

(** {1:types Types} *)

type t
(** The type for forward-pass contexts. *)

(** {1:constructors Constructors} *)

val empty : t
(** [empty] is the empty context. *)

val set : name:string -> Ptree.tensor -> t -> t
(** [set ~name tensor ctx] is [ctx] with [name] bound to [tensor].

    Shadows any previous binding for [name]. *)

(** {1:lookup Lookup} *)

val find : t -> name:string -> Ptree.tensor option
(** [find ctx ~name] is the tensor bound to [name] in [ctx], if any. *)

val get_float_exn :
  ctx:string ->
  t ->
  name:string ->
  dtype:(float, 'l) Rune.dtype ->
  (float, 'l) Rune.t
(** [get_float_exn ~ctx t ~name ~dtype] is the float tensor bound to [name],
    cast-checked against [dtype].

    Raises [Invalid_argument] if [name] is missing or has a different dtype.
    [ctx] is used in error messages. *)

val get_int32_exn :
  ctx:string -> t -> name:string -> (int32, Bigarray.int32_elt) Rune.t
(** [get_int32_exn ~ctx t ~name] is the int32 tensor bound to [name].

    Raises [Invalid_argument] if [name] is missing or has a different dtype. *)

val get_bool_exn : ctx:string -> t -> name:string -> (bool, Nx.bool_elt) Rune.t
(** [get_bool_exn ~ctx t ~name] is the bool tensor bound to [name].

    Raises [Invalid_argument] if [name] is missing or has a different dtype. *)
