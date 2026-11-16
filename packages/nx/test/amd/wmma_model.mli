(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** RDNA4's matrix instructions (WMMA, 16 x 16 outputs, waves of 32) as the
    gfx1201 model of AMD's rocjitsu emulator describes them
    ([emulation/rocjitsu/lib/arith/src/dot/dot.h], MIT), which the R9700 matches
    bit for bit: the AMD suite checks it on the GPU, then states nx.amd's bounds
    and bits through it. *)

(** {1:formats Formats} *)

(** The type for the floats the units multiply: [float16], [bfloat16] and OCP's
    float8 [e4m3fn] (no infinity, NaN [S.1111.111]) and [e5m2]. *)
type fmt = F16 | Bf16 | E4m3 | E5m2

val bits : fmt -> int
(** [bits f] is the width of [f]'s codes. *)

val value : fmt -> int -> float
(** [value f c] is the value of [f]'s code [c]. *)

(** {1:steps Steps} *)

val dot : fmt -> fmt -> group:int -> int array -> int array -> int -> int
(** [dot fa fb ~group a b c] is the float32 bits of one step: the codes [a] of
    [fa] times [b] of [fb], term by term, added to the float32 bits [c]. Each
    product is exact, on the grid [2{^ea + eb - 24}] of its factors' exponents
    ([e = max field 1 - bias]); the products, in runs of [group], are aligned to
    their run's largest grid; the runs' sums and [c] are aligned to
    [G = max (largest grid) (max (-126) (lead c) - 26)], every alignment
    truncating toward minus infinity; the sum rounds once to nearest even. A NaN
    factor gives [0xFFC00A3D], an invalid product or sum of infinities
    [0xFFC00000] ([0xFFC00A3D] of float8), a NaN [c] itself quieted. A sum that
    rounds to zero is [+0], a negative one included. *)

val wmma : fmt -> fmt -> int array -> int array -> int -> int
(** [wmma fa fb a b c] is one instruction's output of a row [a] and a column [b]
    of 16 codes, from the float32 bits [c]: of [F16] or [Bf16], four steps of
    runs of 2 over the k [0-3], [8-11], [4-7] then [12-15]; of float8, two of
    runs of 4 over [0-7] then [8-15]. *)

val ordered : fmt -> fmt -> int array -> int array -> int -> int
(** [ordered fa fb a b c] is {!wmma} of the 16 codes [a] and [b] taken in
    increasing k, as nx.amd's kernels place them: of [F16] or [Bf16], four steps
    over the k [0-3], [4-7], [8-11] then [12-15]; of float8 as {!wmma}. *)

val iwmma :
  bits:int -> sa:bool -> sb:bool -> int array -> int array -> int -> int
(** [iwmma ~bits ~sa ~sb a b c] is the int32 bits of [c] plus the products of
    the [bits]-bit integers [a] and [b], signed where [sa] and [sb], wrapping.
*)

(** {1:images Register images}

    An image is a wave's registers as 32-bit words, lane by lane: an A or B
    operand four words a lane, C and D eight. *)

val slot : k:int -> int -> int -> int * int
(** [slot ~k r kk] is the (lane, element) of A's row (or B's column) [r] at [kk]
    in an instruction of [k] terms: lane [r + 16 (kk / 8)], element [kk mod 8],
    and for [k] 32 lane [r + 16 ((kk / 8) mod 2)], element
    [kk mod 8 + 8 (kk / 16)]. *)

val image : bits:int -> k:int -> int array array -> int array
(** [image ~bits ~k m] is the image of the operand [m] of [bits]-bit codes
    ([m.(r).(kk)], A's rows or B's columns), 128 words: element [e] of a lane
    lies at bit [bits e] of its words, low first. *)

val c_image : int array array -> int array
(** [c_image c] is the image of the 16 x 16 matrix of 32-bit words [c]: row [i],
    column [j] at lane [j + 16 (i / 8)], word [i mod 8]. *)

val c_matrix : int array -> int array array
(** [c_matrix img] is the matrix of the image [img], inverse of {!c_image}. *)
