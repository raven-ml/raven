(* Parity case: dequantised expert rows times a token, the decode MoE product.

   The ids of k = 2 experts gather MXFP4 blocks and scales, the codes decode
   by arithmetic to rows [tokens; k; outputs; inputs], the rows are made
   contiguous, and one token [tokens; inputs] multiplies their transpose.
   The contiguous rows are a buffer, so the product is a sum over a bare
   product of two loads: on Metal the matrix-vector heuristic takes it (group
   8, local 4, upcast 4), which it does not when the decode is fused into the
   product. 16 outputs and 64 inputs are the smallest widths the heuristic
   accepts. Four kernels: scale bytes, scale values, rows, product.

   Backends are limited to cpu and metal: kernel-name counters are shared
   across backends, so the reference must be generated with exactly the
   backends the OCaml side renders.

   Paired with main.py. Run `uv run main.py` to regenerate *.expected. *)

open Tolk_frontend
module D = Tolk_uop.Dtype

let backends =
  List.filter
    (fun (name, _) -> name = "cpu" || name = "metal")
    Helpers.all_backends

let ones shape = List.map (fun _ -> 1) shape

(* A scalar as the nx frontend hands it to a binary op: a constant broadcast to
   the shape of its peer. *)
let scalar value dtype shape =
  let const = Creation.full ~buffer:false ~dtype [] value in
  Movement.expand (Movement.reshape const (ones shape)) shape

(* table[codes], as [take] of a flat table. *)
let lookup table codes =
  let index = Movement.reshape (Dtype_ops.cast codes D.int32) [ -1 ] in
  Movement.reshape (Op.gather table ~dim:0 index) (Tensor.shape codes)

(* t[ids] along axis 0, as [take ~axis:0]. *)
let take_rows ids t =
  let n = Tensor.numel ids and rest = List.tl (Tensor.shape t) in
  let index =
    Movement.expand (Movement.reshape ids (n :: ones rest)) (n :: rest)
  in
  Movement.reshape (Op.gather t ~dim:0 index) (Tensor.shape ids @ rest)

(* Each byte holds two codes, low nibble first. *)
let nibbles blocks =
  let shape = Tensor.shape blocks in
  let low = Elementwise.bitwise_and blocks (scalar (Sint 15) D.uint8 shape) in
  let high = Elementwise.cdiv blocks (scalar (Sint 16) D.uint8 shape) in
  Op.cat ~dim:(-1) (Movement.unsqueeze low (-1)) [ Movement.unsqueeze high (-1) ]

(* The magnitudes 0, 0.5, 1, 1.5, 2, 3, 4 and 6 are m / 2 up to 4, m - 2 for 5
   and 6, then 6; bit 3 is the sign. *)
let code_values codes =
  let shape = Tensor.shape codes in
  let f v = scalar (Sfloat v) D.float32 shape in
  let m =
    Dtype_ops.cast
      (Elementwise.bitwise_and codes (scalar (Sint 7) D.uint8 shape))
      D.float32
  in
  let magnitude =
    Elementwise.where
      (Elementwise.lt m (f 5.0))
      (Elementwise.mul m (f 0.5))
      (Elementwise.where
         (Elementwise.lt m (f 7.0))
         (Elementwise.sub m (f 2.0))
         (f 6.0))
  in
  let sign =
    Dtype_ops.cast
      (Elementwise.cdiv codes (scalar (Sint 8) D.uint8 shape))
      D.float32
  in
  Elementwise.mul magnitude
    (Elementwise.sub (f 1.0) (Elementwise.mul sign (f 2.0)))

let experts = 4
let outputs = 16
let groups = 2
let tokens = 1
let k = 2
let inputs = groups * 32

let build () =
  let param idx ?dtype shape = Tensor.of_uop (Helpers.mk_param ~idx ?dtype shape) in
  let blocks = param 0 ~dtype:D.uint8 [ experts; outputs; groups; 16 ] in
  let scales = param 1 ~dtype:D.uint8 [ experts; outputs; groups ] in
  let ids = param 2 ~dtype:D.int32 [ tokens; k ] in
  let scale_table = param 3 [ 256 ] in
  let x = param 4 [ tokens; inputs ] in
  let values = code_values (nibbles (take_rows ids blocks)) in
  let scale =
    Movement.reshape
      (lookup scale_table (take_rows ids scales))
      [ tokens; k; outputs; groups; 1; 1 ]
  in
  let rows =
    Movement.reshape (Elementwise.mul values scale) [ tokens; k; outputs; inputs ]
  in
  let weight =
    Movement.transpose ~dim0:(-1) ~dim1:(-2) (Elementwise.contiguous rows)
  in
  Helpers.wrap_sink
    [
      Tensor.uop
        (Op.matmul (Movement.reshape x [ tokens; 1; 1; inputs ]) weight);
    ]

let () =
  Helpers.dump_tensor ~backends
    ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (build ())
