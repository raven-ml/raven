(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Quantised weights.

    A quantised weight is packed codes and their scales, in the byte layout a
    checkpoint stores, so a weight loaded from a mapped file is used without a
    copy. Its logical shape is [[| ...; n; k |]], [n] outputs by [k] inputs, as
    files store linear layers, with blocks of values running along [k] inside
    each row. {!dequant} is its meaning and {!apply} its one product.

    Each format has one constructor, taking the bytes as the file stores them. A
    GGUF file's block formats are read from the [uint8] tensors
    [Nx_io.load_gguf] gives, after matching on the tensor's stored type:
    [Nx_quant.q8_0 t] for a [Q8_0] tensor [t], whose shape is the weight's with
    its last axis in bytes.

    {!dequant} and {!apply} are compositions of nx's operations, so every
    transformation and compiled call sees them as it sees any other program: the
    values assembled from the code bytes with integer operations, a gather of
    the experts [ids] selects and one product. Their results live where their
    operands join ({!Nx.place}).

    With [ids], a product whose positions number fewer than twice the weight's
    experts, those of all its lanes, multiplies each position by its expert.
    With more, as when a prompt's tokens each choose a few of the experts, the
    positions are sorted by expert, each expert's run is padded to whole blocks
    of 2 positions, and each block is one product with its expert, whose matrix
    is then read once per block. Positions split over devices are grouped on
    each device.

    Run eagerly, the composition holds the matrices it multiplies, decoded at
    float32: {!dequant} holds the whole weight, and {!apply} with [ids] holds
    one matrix per position, or one per block when it groups them. A compiled
    call decodes them inside the product. Large weights are therefore for
    compiled calls.

    A quantised weight has no gradient: build it once and capture it.

    {[
    let gate_up =
      Nx_quant.mxfp4 ~scales
        (Nx.reshape [| experts; outputs; inputs / 2 |] blocks)
    in
    Nx_quant.apply ~ids gate_up x
    ]} *)

(** {1:weights Weights} *)

(** The type for quantised weights. Match on a weight to read its parts; only
    the constructors build one.

    A GGUF format stores each block of [k]'s values as one run of bytes, in the
    layout of ggml's [block_q8_0], [block_q4_K] and [block_q6_K]
    (ggml-common.h). Its float16 fields are little-endian. *)
type t = private
  | Mxfp4 of {
      codes : (int, Nx.uint8_elt) Nx.t;
          (** [[| ...; n; k / 2 |]]: two e2m1 codes per byte, the low nibble
              first. A code is a sign bit over a magnitude among 0, 0.5, 1, 1.5,
              2, 3, 4 and 6. *)
      scales : (int, Nx.uint8_elt) Nx.t;
          (** [[| ...; n; k / 32 |]]: one e8m0 scale per 32 values. *)
    }
      (** MXFP4 (OCP Microscaling Formats, 2023). A value is its code's
          magnitude, signed, times [2 ^ (s - 127)] for its group's scale byte
          [s]; the byte [255] makes its whole group NaN. *)
  | Q8_0 of {
      blocks : (int, Nx.uint8_elt) Nx.t;
          (** [[| ...; n; k / 32 * 34 |]]: per 32 values, a float16 scale [d]
              then 32 int8 quants. *)
    }  (** Q8_0 (GGUF). A value is [d * q] for its quant [q]. *)
  | Q4_K of {
      blocks : (int, Nx.uint8_elt) Nx.t;
          (** [[| ...; n; k / 256 * 144 |]]: per 256 values, float16 [d] and
              [dmin], 12 bytes packing a 6-bit scale and a 6-bit min for each of
              8 sub-blocks of 32 values, and 128 bytes of 4-bit quants. *)
    }
      (** Q4_K (GGUF). A value is [d * sc * q - dmin * m] for its quant [q] and
          its sub-block's scale [sc] and min [m]. *)
  | Q6_K of {
      blocks : (int, Nx.uint8_elt) Nx.t;
          (** [[| ...; n; k / 256 * 210 |]]: per 256 values, 128 bytes of the
              quants' low 4 bits, 64 of their high 2 bits, an int8 scale for
              each of 16 sub-blocks of 16 values, and a float16 [d]. *)
    }
      (** Q6_K (GGUF). A value is [d * sc * (q - 32)] for its 6-bit quant [q]
          and its sub-block's scale [sc]. *)

val mxfp4 : scales:(int, Nx.uint8_elt) Nx.t -> (int, Nx.uint8_elt) Nx.t -> t
(** [mxfp4 ~scales codes] is the MXFP4 weight with [codes] and [scales]. Only
    their shapes are read: no byte of either is.

    Raises [Invalid_argument] naming the part if [codes] does not have shape
    [[| ...; n; k / 2 |]] with [k] a multiple of 32, or if [scales] does not
    have shape [[| ...; n; k / 32 |]]. *)

val q8_0 : (int, Nx.uint8_elt) Nx.t -> t
(** [q8_0 blocks] is the Q8_0 weight stored as [blocks]. Only its shape is read.

    Raises [Invalid_argument] if [blocks] does not have shape
    [[| ...; n; k / 32 * 34 |]] with [k] a multiple of 32. *)

val q4_k : (int, Nx.uint8_elt) Nx.t -> t
(** [q4_k blocks] is the Q4_K weight stored as [blocks]. Only its shape is read.

    Raises [Invalid_argument] if [blocks] does not have shape
    [[| ...; n; k / 256 * 144 |]] with [k] a multiple of 256. *)

val q6_k : (int, Nx.uint8_elt) Nx.t -> t
(** [q6_k blocks] is the Q6_K weight stored as [blocks]. Only its shape is read.

    Raises [Invalid_argument] if [blocks] does not have shape
    [[| ...; n; k / 256 * 210 |]] with [k] a multiple of 256. *)

val shape : t -> int array
(** [shape w] is [w]'s logical shape, [[| ...; n; k |]]. *)

val place : Nx.Placement.t -> t -> t
(** [place p w] is [w] with every part placed with [p] ({!Nx.place}). A leading
    axis or [n] splits wherever {!Nx.place} can split it; [k] splits only
    between blocks, of 32 values for MXFP4 and Q8_0 and of 256 for Q4_K and
    Q6_K, so that each shard holds whole blocks and their scales.

    Raises [Invalid_argument] naming the axis if [p] splits [k] across a block,
    or as {!Nx.place} does for a part. *)

(** {1:products Products} *)

val dequant : (float, 'b) Nx.dtype -> t -> (float, 'b) Nx.t
(** [dequant dt w] is the values of [w] at [dt], of shape [shape w].

    Values are computed at float32, then rounded once to [dt].

    An MXFP4 value is exact at float32 barring overflow: codes of magnitude 4 or
    more at scale byte 253, and of 2 or more at 254, are infinite at every
    dtype. Float32, bfloat16 and float64 hold every other value exactly; float16
    rounds each value once.

    A GGUF value is computed as ggml's float32 dequantisation computes it, its
    products left to right, and equals it bit for bit: a Q8_0 value and a Q4_K
    product are exact, a Q4_K value rounds once at its subtraction and a Q6_K
    value once at its last product. A block whose float16 scale is infinite or
    NaN gives infinite or NaN values. *)

val apply : ?ids:Nx.int64_t -> t -> (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [apply ?ids w x] is
    [Nx.matmul x (Nx.matrix_transpose (dequant Nx.float32 w))] at [x]'s dtype,
    with {!Nx.matmul}'s shapes: [w] is [[| ...; n; k |]], [x] is
    [[| ...; m; k |]] or [[| k |]], and batch axes broadcast. Products
    accumulate at float32 and the result is rounded once to [x]'s dtype.

    With [ids], [w] is [[| b...; e; n; k |]], [e] experts behind [b] leading
    axes, and [ids] is [[| b...; s... |]], its [b] leading axes broadcasting
    against [w]'s. The product is then over [w'] of shape
    [[| b...; s...; n; k |]], whose matrix at [(b, s)] is [w]'s expert
    [ids.(b, s)] of lane [b]. A weight with no leading axes is gathered by [ids]
    whole: [apply ~ids w x] with [w] of shape [[| e; n; k |]], [ids] of shape
    [[| t; 4 |]] and [x] of shape [[| t; 1; 1; k |]] is [[| t; 4; 1; n |]].

    An id outside \[[0], [e]), [-1] included, selects no expert: its position of
    the result is exactly zero, whatever [x] holds there. Ids may repeat. An
    empty [ids] or [x] gives an empty result.

    Raises [Invalid_argument] if [x] is a scalar, if [x]'s last axis is not [k],
    if the batch axes do not broadcast, or, with [ids], if [w] has no expert
    axis or [ids] lacks [w]'s leading axes. *)

(** {1:structure Structure}

    A weight is a structure without a parameter ({!Nx.Ptree.S} with
    [type _ t = t]): its parts are tensors of a fixed type, so casts and
    {!Nx.Ptree.Payload} operations keep them. *)

val walk : ('a, 'b) Nx.Ptree.Walk.cursor -> t -> t
(** [walk c w] walks [w] at [c]'s path: it reports [w]'s case, ["mxfp4"],
    ["q8_0"], ["q4_k"] or ["q6_k"], then walks each part at its field's name,
    [codes] then [scales] or [blocks], with {!Nx.Ptree.Walk.tensor}. It checks
    the parts it rebuilds as the case's constructor does, and reads neither
    bytes nor placement, so it runs under every transformation and compiled
    trace. A model's own [walk] walks a quantised field with it:
    [field c "gate_up" Nx_quant.walk w].

    Raises [Invalid_argument] as the constructors do, naming [Nx_quant.walk], if
    a walk returns parts of other shapes. *)

val ptree : t Nx.Ptree.t
(** [ptree] is a weight as a structure at one type, walked by {!walk}: an MXFP4
    weight's visits are [the root: case "mxfp4"], [codes: a leaf] and
    [scales: a leaf], and a Q8_0 weight's [the root: case "q8_0"] and
    [blocks: a leaf]. Compiled programs therefore key on the format, and
    [Nx.Ptree.map ptree f w] maps its parts. *)
