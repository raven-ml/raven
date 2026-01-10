(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Ptree = Ptree

val normalize_mask : ('a, 'layout) Rune.t -> Rune.bool_t
(** Convert any numeric/bool mask into a boolean tensor (non-zero => true). *)

module Multi_head : sig
  type config = {
    embed_dim : int;
    num_heads : int;
    num_kv_heads : int option;
    head_dim : int option;
    dropout : float;
    use_qk_norm : bool;
    attn_logits_soft_cap : float option;
    query_pre_attn_scalar : float option;
  }

  val make_config :
    embed_dim:int ->
    num_heads:int ->
    ?num_kv_heads:int ->
    ?head_dim:int ->
    ?dropout:float ->
    ?use_qk_norm:bool ->
    ?attn_logits_soft_cap:float ->
    ?query_pre_attn_scalar:float ->
    unit ->
    config

  type params = Ptree.t

  val init :
    config -> rngs:Rune.Rng.key -> dtype:(float, 'layout) Rune.dtype -> params

  val apply :
    ?rngs:Rune.Rng.key ->
    ?attention_mask:Rune.bool_t ->
    config ->
    params ->
    training:bool ->
    query:(float, 'layout) Rune.t ->
    key:(float, 'layout) Rune.t ->
    value:(float, 'layout) Rune.t ->
    (float, 'layout) Rune.t
end
