(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Lookup-free quantization: tokens for vectors of signs.

    Lookup-free quantization (Yu et al., "Language Model Beats Diffusion:
    Tokenizer Is Key to Visual Generation", 2023) turns a vector of [d] numbers
    into a {e code} of [d] bits: number [c] sets bit [d - 1 - c] where it is
    positive, so that number [0] is the most significant bit, as in AION-1's
    spectrum tokens. A number of [0], [-0.] or NaN clears its bit. Decoding
    gives each bit's sign, [1] for a set bit and [-1] for a clear one.

    AION-1's spectra use vectors of 10 numbers, 1024 codes. Compiled by
    [Rune.jit], codes and vectors are the eager ones bit for bit: the
    computation is comparisons, casts, integer arithmetic and selections. *)

val encode : (float, 's, 'd) Nx.t -> 'd Nx.int64_t
(** [encode z] is the code of each vector along [z]'s last axis, of [z]'s shape
    without that axis.

    Raises [Invalid_argument] if [z] is 0-d or its last axis has more than 62
    entries. *)

val decode :
  (float, 's) Nx.dtype -> bits:int -> 'd Nx.int64_t -> (float, 's, 'd) Nx.t
(** [decode dt ~bits k] is the vector of each code of [k], its [bits] signs in
    [dt] along a last axis added to [k]'s shape.

    Raises [Invalid_argument] if [bits] is outside [\[0, 62\]], or if a code is
    outside [\[0, 2{^bits})]; on a set other than the host it raises for a code
    at the next wait ({!Nx.check}). *)
