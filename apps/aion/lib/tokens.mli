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
      (** [[|b; n|]], [n] the modality's {!Modality.tokens}, or [[|b|]] for a
          modality of one token. *)
  present : 'd Nx.bool_t option;
      (** [true] where a token is present, of [tokens]' shape; [None] where all
          are. An absent token's value is not read. *)
}
(** The type for the tokens of one modality of each of [b] objects, on the
    device set ['d]. *)
