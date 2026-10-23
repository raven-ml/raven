(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Quantised weights.

    A quantised weight is a float array of shape [[| …; n; k |]], [n] outputs by
    [k] inputs, stored in a block format: blocks of values along [k], each as
    codes and a shared scale, in the bytes a file holds. Each constructor takes
    those bytes and reads their shape alone, so a mapped file is used without a
    copy. {!dequant} is a weight's meaning, {!apply} its product and {!take} a
    gather of its rows. A weight's parts lie on one device set, and each
    function computes there.

    A GGUF file's block formats are read from the [uint8] tensors a GGUF reader
    gives, after matching on the tensor's stored type: [q8_0 t] for a [Q8_0]
    tensor [t], [mxfp4_blocks t] for an [MXFP4] one, whose shape is the weight's
    with its last axis in bytes.

    {b Placement.} A weight's parts may lie at any placement of their set; its
    values, and {!apply}'s bits except a NaN's sign and payload, do not depend
    on where they lie. Cut along [n]
    or a leading axis, each device decodes its own rows. Cut along [k] over [t]
    devices, a weight with values and [b] blocks along [k] decodes on its own
    devices where [t] divides [b]: {!dequant}'s result is cut along [k] as the
    parts are, and {!apply} sums the devices' partial products. Any other cut
    along [k] moves the weight's packed bytes between the devices of its set at
    each {!dequant} and {!apply}, never its decoded values; with [b] at least 2,
    each device gathers all of [k] and decodes it, and {!dequant}'s result lies
    whole on each. A layer cut along its inputs, as a row-parallel layer is,
    then moves its weight between devices at every step.

    {!take} gathers before decoding, so a caller reads a large table's rows
    without decoding the table. Routing positions to a stack of weights, as a
    mixture of experts does, is {!apply_experts}:

    {[
    let gate_up =
      Nx_quant.mxfp4 ~scales
        (Nx.reshape [| experts; outputs; inputs / 2 |] codes)
    in
    (* ids : [| tokens; k |], x : [| tokens; inputs |] *)
    Nx_quant.apply_experts ~ids gate_up (Nx.unsqueeze ~axes:[ -2 ] x)
    ]} *)

(** {1:weights Weights} *)

(** The type for quantised weights on the set ['d]. Match on a weight to read
    its parts; only the constructors and {!walk} build one.

    MXFP4 (OCP Microscaling Formats, 2023) decodes a value as its e2m1 code, a
    sign bit over a magnitude among 0, 0.5, 1, 1.5, 2, 3, 4 and 6, times
    [2 ^ (s - 127)] for its group's e8m0 scale byte [s] of 32 values; the scale
    byte [255] makes its whole group NaN. A GGUF format stores each block of
    [k]'s values as one run of bytes, in the layout of ggml's [block_mxfp4],
    [block_q8_0], [block_q4_K] and [block_q6_K] (ggml-common.h), its float16
    fields little-endian. In every format a block of zero bytes decodes to
    zeros, [-0.] in Q6_K. *)
type 'd t = private
  | Mxfp4 of {
      codes : 'd Nx.uint8_t;
          (** [[| …; n; k / 2 |]]: two codes per byte, value [2 i] in byte [i]'s
              low nibble and value [2 i + 1] in its high one, as safetensors
              checkpoints store them. *)
      scales : 'd Nx.uint8_t;  (** [[| …; n; k / 32 |]]: one per 32 values. *)
    }  (** MXFP4 with its scales apart. *)
  | Mxfp4_blocks of {
      blocks : 'd Nx.uint8_t;
          (** [[| …; n; k / 32 * 17 |]]: per 32 values, the scale byte, then 16
              bytes, byte [j] holding value [j] in its low nibble and value
              [j + 16] in its high one. *)
    }  (** MXFP4 in GGUF's blocks. *)
  | Q8_0 of {
      blocks : 'd Nx.uint8_t;
          (** [[| …; n; k / 32 * 34 |]]: per 32 values, a float16 scale [d] then
              32 int8 quants. *)
    }  (** Q8_0 (GGUF). A value is [d * q] for its quant [q]. *)
  | Q4_K of {
      blocks : 'd Nx.uint8_t;
          (** [[| …; n; k / 256 * 144 |]]: per 256 values, float16 [d] and
              [dmin], 12 bytes packing a 6-bit scale and a 6-bit min for each of
              8 sub-blocks of 32 values, and 128 bytes of 4-bit quants. *)
    }
      (** Q4_K (GGUF). A value is [d * sc * q - dmin * m] for its quant [q] and
          its sub-block's scale [sc] and min [m]. *)
  | Q6_K of {
      blocks : 'd Nx.uint8_t;
          (** [[| …; n; k / 256 * 210 |]]: per 256 values, 128 bytes of the
              quants' low 4 bits, 64 of their high 2 bits, an int8 scale for
              each of 16 sub-blocks of 16 values, and a float16 [d]. *)
    }
      (** Q6_K (GGUF). A value is [d * sc * (q - 32)] for its 6-bit quant [q]
          and its sub-block's scale [sc]. *)

val mxfp4 : scales:'d Nx.uint8_t -> 'd Nx.uint8_t -> 'd t
(** [mxfp4 ~scales codes] is the MXFP4 weight of [codes], two per byte, and
    [scales]: [Mxfp4 { codes; scales }].

    Raises [Invalid_argument] naming the part if [codes] does not have shape
    [[| …; n; k / 2 |]] with [k] a multiple of 32, or [scales] does not have
    shape [[| …; n; k / 32 |]]. *)

val mxfp4_blocks : 'd Nx.uint8_t -> 'd t
(** [mxfp4_blocks blocks] is the MXFP4 weight stored as GGUF's [blocks].

    Its values are the format's, {!type-t}'s: the scale byte [255] makes its
    group NaN where ggml reads it as [2 ^ 128], and the code of sign 1 and
    magnitude 0 is [-0.] where ggml gives [0.].

    Raises [Invalid_argument] if [blocks] does not have shape
    [[| …; n; k / 32 * 17 |]]. *)

val q8_0 : 'd Nx.uint8_t -> 'd t
(** [q8_0 blocks] is the Q8_0 weight stored as [blocks].

    Raises [Invalid_argument] if [blocks] does not have shape
    [[| …; n; k / 32 * 34 |]]. *)

val q4_k : 'd Nx.uint8_t -> 'd t
(** [q4_k blocks] is the Q4_K weight stored as [blocks].

    Raises [Invalid_argument] if [blocks] does not have shape
    [[| …; n; k / 256 * 144 |]]. *)

val q6_k : 'd Nx.uint8_t -> 'd t
(** [q6_k blocks] is the Q6_K weight stored as [blocks].

    Raises [Invalid_argument] if [blocks] does not have shape
    [[| …; n; k / 256 * 210 |]]. *)

val shape : 'd t -> int array
(** [shape w] is [w]'s logical shape, [[| …; n; k |]]. *)

(** {1:products Products} *)

val dequant : (float, 'b) Nx.dtype -> 'd t -> (float, 'b, 'd) Nx.t
(** [dequant dt w] is the values of [w] at [dt], of shape [shape w], each
    rounded once to [dt].

    A value of [w] is a float32 value. An MXFP4 value is exact barring overflow:
    codes of magnitude 4 or more at scale byte 253, and of 2 or more at 254, are
    infinite at every dtype. Float32, bfloat16 and float64 hold every other
    value exactly. A GGUF value is computed as ggml's float32 dequantisation
    computes it, its products left to right, and equals it bit for bit: a Q8_0
    value and a Q4_K product are exact, a Q4_K value rounds once at its
    subtraction and a Q6_K value once at its last product. A block whose float16
    scale is infinite or NaN gives infinite or NaN values. *)

val apply : 'd t -> (float, 'b, 'd) Nx.t -> (float, 'b, 'd) Nx.t
(** [apply w x] is [w] applied to [x] as a linear map from [k] inputs to [n]
    outputs:

    {[
    Nx.matmul x (Nx.matrix_transpose (dequant (Nx.dtype x) w))
    ]}

    where every value of [w]'s format is exact at [x]'s dtype: at float32 and
    float64, and MXFP4's at bfloat16. At another dtype the values are not
    rounded to it, and the product is at float32:

    {[
    Nx.cast (Nx.dtype x)
      (Nx.matmul (Nx.cast Nx.float32 x)
         (Nx.matrix_transpose (dequant Nx.float32 w)))
    ]}

    [w] is [[| …; n; k |]], [x] is [[| …; m; k |]] or [[| k |]], batch axes
    broadcast as {!Nx.matmul}'s, and the result is [[| …; m; n |]], or
    [[| …; n |]] for a vector.

    An [Mxfp4] weight applied to a float32, bfloat16 or float16 [x], where
    every device of each placement holds all of its value, is read where it
    lies: the product decodes each block as it sums it and stores no decoded
    weight. Any other product first decodes [w] as {!dequant} does, at [x]'s
    dtype or at float32. Read where it lies, the product needs kernels that
    compute a block-scaled contraction, as nx.cpu's do.

    Raises [Invalid_argument] if [w] is read where it lies on a set whose
    kernels decline that contraction, if [x] is 0-d, if [x]'s last axis is
    not [k], or if the batch axes do not broadcast. *)

val apply_experts :
  ids:'d Nx.int64_t -> 'd t -> (float, 'b, 'd) Nx.t -> (float, 'b, 'd) Nx.t
(** [apply_experts ~ids w x] is each row of [x] through the expert its id
    names: at each position [p] of [ids], {!apply} of [w]'s matrix [ids.{p}]
    to [p]'s row of [x], or zeros where [ids.{p}] is not in [\[0, e)],
    whatever that row holds. [w] is [[| e; n; k |]], [ids] [[| s… |]] and [x]
    [[| s'…; k |]], [s'…] broadcasting to [s…]; the result is [[| s…; n |]].

    Where {!apply} reads an [Mxfp4] [w] where it lies and [ids] lies whole on
    each device, every position is summed in one contraction grouped by
    expert, which needs kernels that compute a segmented block-scaled
    contraction, as nx.cpu's do. Any other product, a cut [w]'s among them, is

    {[
    Nx.map_segments ~segments:e ids
      (fun owners rows -> apply (take ~axis:0 owners w) rows)
      x
    ]}

    So its bits depend on whether [w], [ids] and [x] lie whole: the two sums
    may round differently, each within {!Nx.contract}'s bound.

    Raises [Invalid_argument] if the grouped sum lies on a set whose kernels
    decline it, if [w] is not 3-d, if [x] does not have one axis
    more than [ids], if [x]'s last axis is not [k], if [x]'s leading axes
    do not broadcast to [ids]', or as {!apply} raises in the map. *)

val take : axis:int -> 'd Nx.int64_t -> 'd t -> 'd t
(** [take ~axis p w] is the rows or matrices of [w] at the positions [p] along
    [axis], gathered from its packed parts with no value decoded, [p] read as
    {!Nx.take} reads it: at each position in range,
    [dequant dt (take ~axis p w)] is [Nx.take ~axis p (dequant dt w)]. A
    position out of range gathers zero bytes, which decode to zeros.

    Raises [Invalid_argument] if [axis] is [w]'s last axis, negative or not,
    since a gather along it would cut blocks, or if [axis] is not an axis of
    [w]. *)

(** {1:structure Structure} *)

val walk : ('a, 'b, 'd, 'e) Nx.Ptree.Walk.cursor -> 'd t -> 'e t
(** [walk c w] walks [w] at [c]'s path: its case, ["mxfp4"], ["mxfp4_blocks"],
    ["q8_0"], ["q4_k"] or ["q6_k"], then each part as a fixed array
    ({!Nx.Ptree.Walk.tensor}) at its field's name, [codes] then [scales], or
    [blocks]. It rebuilds the weight as the case's constructor does. A structure
    holding a weight walks it with [field c "w" Nx_quant.walk w].

    Raises [Invalid_argument] as the constructors do, naming [Nx_quant.walk],
    where the walk returns parts of other shapes. *)

val place : 'e Nx.Placement.t -> 'd t -> 'e t
(** [place p w] is [w] with each part at [p], as {!Nx.place}. *)
