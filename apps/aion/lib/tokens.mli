(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Tokens: one modality's tokens for a batch of objects.

    A codec makes them from data ({!Codecs.encode}), the transformer reads them
    ({!Transformer.encode}), and random-order generation draws them
    ({!Roar}). *)

type 'd t = {
  modality : Modality.t;
  tokens : 'd Nx.int64_t;
      (** [[|b; n|]], [n] the modality's {!Modality.tokens}. *)
  present : 'd Nx.bool_t;
      (** [true] where a token is present, [[|b; n|]]. An absent token's value
          is not read. *)
}
(** The type for the tokens of one modality of each of [b] objects, on the
    device set ['d]. *)

val walk : ('a, 'b, 'd, 'e) Nx.Ptree.Walk.cursor -> 'd t -> 'e t
(** [walk c t] reports [t]'s modality ({!Modality.walk}) and walks its arrays
    as fixed ones: [Nx.Ptree.v walk] describes tokens to [Rune.jit]. *)
